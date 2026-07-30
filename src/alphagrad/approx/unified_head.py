"""ONE approximation head, SIMPLIFIED. 16 outputs, one forward pass.

Was 64 outputs. The prime-exponent gates (32), the reduce-axis gates (9), the
reduce-fn softmax (5) and the dtype gate (1) are all gone -- each replaced by a
fixed rule, because the free parameter was not buying anything the search could
use.

LAYOUT (16 logits)
------------------
    [ 0: 1)  skip   Bernoulli
    [ 1: 4)  op     softmax {blockdiag, reduce, quant}
    [ 4:10)  i      softmax over 1..6
    [10:16)  j      softmax over 1..6

FIXED RULES (no longer sampled)
-------------------------------
* BLOCKDIAG. Block count is ``k = min(N_i, N_j)``, so:
    - SQUARE pair (N x N)  -> k = N  -> blocks are 1x1 -> a PURE DIAGONAL.
    - NON-SQUARE (2 x 8)   -> k = 2  -> blocks are 1x4, two of them.
  This is the whole factor decision. It removes the prime-exponent machinery,
  which sampled factors up to ~7e8 that OVERFLOWED int32 and were then clamped
  to a legal divisor anyway -- the clamp was doing the real work.
* REDUCE. Always ``axes=(0,)``, ``kind="mean"``: first axis, mean.
* QUANT. Always ``bfloat16``.

Every remaining field is still branch-masked in log-prob and entropy: only what
the chosen op CONSUMES contributes, so an unused head cannot collect gradient
from a reward it had no part in.

Because ``k`` is now a deterministic function of the operand shapes rather than
a sampled quantity, there is no exponent to score, and therefore no
Poisson-binomial term -- sample and evaluate agree trivially.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple

# op codes -- deliberately aligned with heads.py's OP_DIAG/OP_COMPRESS/OP_QUANT
OP_BLOCKDIAG, OP_REDUCE, OP_QUANT = 0, 1, 2
NUM_APPROX_OPS = 3

MAX_PAIR_IDX = 6

O_SKIP = 0
O_OP = 1
O_I = O_OP + NUM_APPROX_OPS          # 4
O_J = O_I + MAX_PAIR_IDX             # 10
HEAD_WIDTH = O_J + MAX_PAIR_IDX      # 16


class ApproxAction(NamedTuple):
    skip: jax.Array
    op: jax.Array
    i: jax.Array          # 1..6
    j: jax.Array          # 1..6
    log_prob: jax.Array
    entropy: jax.Array


def _cat_logp_ent(logits, mask, idx):
    logits = jnp.where(mask > 0.5, logits, -1e9)
    logp_all = jnn.log_softmax(logits)
    p = jnn.softmax(logits)
    ent = -jnp.sum(jnp.where(p > 0, p * logp_all, 0.0))
    return logp_all[idx], ent


def _bern_logp_ent(logit, x):
    lp1 = jnn.log_sigmoid(logit)
    lp0 = jnn.log_sigmoid(-logit)
    p = jnn.sigmoid(logit)
    ent = -(p * lp1 + (1.0 - p) * lp0)
    return jnp.where(x, lp1, lp0), ent


def block_count(n_i, n_j):
    """Blocks for a (n_i, n_j) pair: ``min(n_i, n_j)``.

    Square -> one block per row/col, i.e. a pure diagonal.
    Non-square -> the short side sets the count, so 2x8 gives two 1x4 blocks.
    """
    return jnp.maximum(jnp.minimum(n_i, n_j), 1).astype(jnp.int32)


class UnifiedApproxHead(eqx.Module):
    """embd_dim -> 16 logits -> skip / op / i / j."""

    proj: eqx.nn.MLP

    def __init__(self, embd_dim: int, *, hidden: int | None = None, key):
        self.proj = eqx.nn.MLP(embd_dim, HEAD_WIDTH, hidden or embd_dim,
                               depth=1, key=key)

    def logits(self, ctx):
        return self.proj(ctx)

    def score(self, z, skip, op, i_idx, j_idx, *, op_mask, i_mask, j_mask):
        """Branch-masked (log_prob, entropy) for a GIVEN assignment.

        Shared by sample and evaluate so the two can never score different
        variables -- the PPO ratio is only meaningful if they agree.
        """
        lp_skip, ent_skip = _bern_logp_ent(z[O_SKIP], skip)
        lp_op, ent_op = _cat_logp_ent(z[O_OP:O_I], op_mask, op)
        lp_i, ent_i = _cat_logp_ent(z[O_I:O_J], i_mask, i_idx)
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        lp_j, ent_j = _cat_logp_ent(z[O_J:HEAD_WIDTH], j_mask_eff, j_idx)

        act = (~skip).astype(jnp.float32)
        # i / j are consumed by BLOCKDIAG only; REDUCE and QUANT are fully
        # determined once chosen (first axis + mean, bfloat16).
        is_bd = act * (op == OP_BLOCKDIAG).astype(jnp.float32)

        lp = lp_skip + act * lp_op + is_bd * (lp_i + lp_j)
        ent = ent_skip + act * ent_op + is_bd * (ent_i + ent_j)
        return lp.astype(jnp.float32), ent.astype(jnp.float32)

    def sample_fields(self, ctx, key, *, op_mask=None, i_mask=None, j_mask=None):
        z = self.logits(ctx)
        k_skip, k_op, k_i, k_j = jrand.split(key, 4)
        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask

        skip = jrand.bernoulli(k_skip, jnn.sigmoid(z[O_SKIP]))
        op = jrand.categorical(k_op, jnp.where(op_mask > 0.5, z[O_OP:O_I], -1e9))
        i_idx = jrand.categorical(k_i, jnp.where(i_mask > 0.5, z[O_I:O_J], -1e9))
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        j_idx = jrand.categorical(
            k_j, jnp.where(j_mask_eff > 0.5, z[O_J:HEAD_WIDTH], -1e9))
        return z, skip, op, i_idx, j_idx

    def __call__(self, ctx, key, *, op_mask=None, i_mask=None, j_mask=None):
        z, skip, op, i_idx, j_idx = self.sample_fields(
            ctx, key, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask)
        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        lp, ent = self.score(z, skip, op, i_idx, j_idx,
                             op_mask=op_mask, i_mask=i_mask, j_mask=j_mask)
        return ApproxAction(
            skip=skip,
            op=op.astype(jnp.int32),
            i=(i_idx + 1).astype(jnp.int32),
            j=(j_idx + 1).astype(jnp.int32),
            log_prob=lp,
            entropy=ent,
        )
