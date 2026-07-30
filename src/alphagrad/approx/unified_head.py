"""ONE approximation head. 32 outputs, one forward pass, one sample.

The 64-output layout MINUS the 32 prime-exponent gates and the gcd /
force-pure-diag option. Everything else is unchanged.

LAYOUT (32 logits)
------------------
    [ 0: 1)  skip            Bernoulli
    [ 1: 5)  op              softmax {blockdiag, reduce, quant, none}
    [ 5:11)  blockdiag i     softmax over 1..6
    [11:17)  blockdiag j     softmax over 1..6
    [17:26)  reduce axes     9 independent gates (axes 0..8)
    [26:31)  reduce fn       softmax {mean, min, max, abs_min, abs_max}
    [31:32)  dtype           Bernoulli {float32, bfloat16}

WHY THE FACTOR IS NO LONGER SAMPLED
-----------------------------------
The prime gates spanned ~7e8, OVERFLOWED int32, and were then clamped to a
legal divisor of ``gcd(N_i, N_j)`` -- so the clamp, not the gates, decided the
factor. 32 of the head's 64 outputs were spending gradient on a quantity the
environment overrode. They are gone. Instead:

    factor = gcd(N_i, N_j)

which is the LARGEST legal factor, i.e. the SMALLEST blocks. A square pair
(N x N) therefore gives a PURE DIAGONAL; a 2 x 8 pair gives two 1 x 4 blocks.

and pairs whose sizes are COPRIME are MASKED OUT of the (i, j) choice: gcd = 1
admits only factor 1, which is a no-op block-diagonal. Those pairs used to
absorb probability mass for an action that could not change anything.

BRANCH MASKING
--------------
Only the fields the chosen op actually CONSUMES contribute to log-prob and
entropy. Otherwise unused sub-heads receive gradient from rewards they had no
part in and drift -- the cause of the old incomparable per-head entropies.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple

OP_BLOCKDIAG, OP_REDUCE, OP_QUANT, OP_NONE = 0, 1, 2, 3
NUM_APPROX_OPS = 4

MAX_PAIR_IDX = 6
NUM_REDUCE_AXES = 9
REDUCE_FNS = ("mean", "min", "max", "abs_min", "abs_max")
NUM_REDUCE_FNS = len(REDUCE_FNS)

O_SKIP = 0
O_OP = 1
O_I = O_OP + NUM_APPROX_OPS                 # 5
O_J = O_I + MAX_PAIR_IDX                    # 11
O_AXES = O_J + MAX_PAIR_IDX                 # 17
O_RFN = O_AXES + NUM_REDUCE_AXES            # 26
O_DTYPE = O_RFN + NUM_REDUCE_FNS            # 31
HEAD_WIDTH = O_DTYPE + 1                    # 32


class ApproxAction(NamedTuple):
    skip: jax.Array
    op: jax.Array
    i: jax.Array             # 1..6
    j: jax.Array             # 1..6
    axes: jax.Array          # (9,) bool
    reduce_fn: jax.Array
    dtype_idx: jax.Array
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


def j_mask_given_i(i_idx, j_mask, pair_ok=None):
    """Legal ``j`` once ``i`` is fixed.

    Base mask, minus ``i`` itself, minus every partner that is COPRIME with it
    (``pair_ok`` is the (6, 6) non-coprime table built by the adapter). Falls
    back to the unrestricted mask if that would leave nothing selectable, so
    the categorical is never fully -1e9.
    """
    m = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
    if pair_ok is not None:
        m = m * pair_ok[i_idx]
    return jnp.where(jnp.sum(m) > 0, m, j_mask)


class UnifiedApproxHead(eqx.Module):
    """embd_dim -> 32 logits -> one complete approximation decision."""

    proj: eqx.nn.MLP

    # Output-bias init. See module docstring: a zero-bias head sits at p=0.5 on
    # every gate, which approximates half the vertices and compresses ~4-5 of
    # the 9 axes per REDUCE -- measured to give ||J_approx|| == 0 on every plan,
    # i.e. a constant-zero cosine channel with no gradient anywhere.
    INIT_SKIP_BIAS = 1.5      # p(skip) ~ 0.82
    INIT_AXIS_BIAS = -2.0     # p(axis) ~ 0.12

    def __init__(self, embd_dim: int, *, hidden: int | None = None, key):
        self.proj = eqx.nn.MLP(embd_dim, HEAD_WIDTH, hidden or embd_dim,
                               depth=1, key=key)
        # Bias the START of the search toward EXACT. Weights are untouched, so
        # the head keeps its full range and full gradient; only the point it
        # starts from moves.
        layers = self.proj.layers
        last = max(i for i, l in enumerate(layers)
                   if isinstance(l, eqx.nn.Linear))
        b = layers[last].bias
        if b is not None:
            b = b.at[O_SKIP].add(self.INIT_SKIP_BIAS)
            b = b.at[O_AXES:O_RFN].add(self.INIT_AXIS_BIAS)
            self.proj = eqx.tree_at(
                lambda m: m.layers[last].bias, self.proj, b)

    def logits(self, ctx):
        return self.proj(ctx)

    # ------------------------------------------------------------------ score
    def score(self, z, skip, op, i_idx, j_idx, axes, rfn, dt,
              *, op_mask, i_mask, j_mask, axis_mask, pair_ok=None):
        """Branch-masked (log_prob, entropy) for a GIVEN field assignment.

        Shared by sampling and evaluation so the two can never disagree --
        the PPO ratio is only meaningful if both score the same variable.
        """
        lp_skip, ent_skip = _bern_logp_ent(z[O_SKIP], skip)
        lp_op, ent_op = _cat_logp_ent(z[O_OP:O_I], op_mask, op)
        lp_i, ent_i = _cat_logp_ent(z[O_I:O_J], i_mask, i_idx)
        lp_j, ent_j = _cat_logp_ent(z[O_J:O_AXES],
                                    j_mask_given_i(i_idx, j_mask, pair_ok),
                                    j_idx)

        lp_ax, ent_ax = _bern_logp_ent(z[O_AXES:O_RFN], axes)
        keep = (axis_mask > 0.5).astype(lp_ax.dtype)
        lp_ax = jnp.sum(lp_ax * keep)
        ent_ax = jnp.sum(ent_ax * keep)
        lp_rf, ent_rf = _cat_logp_ent(z[O_RFN:O_DTYPE],
                                      jnp.ones((NUM_REDUCE_FNS,)), rfn)
        lp_dt, ent_dt = _bern_logp_ent(z[O_DTYPE], dt)

        act = (~skip).astype(jnp.float32)
        is_bd = act * (op == OP_BLOCKDIAG).astype(jnp.float32)
        is_rd = act * (op == OP_REDUCE).astype(jnp.float32)
        is_qt = act * (op == OP_QUANT).astype(jnp.float32)

        lp = lp_skip + act * lp_op
        lp = lp + is_bd * (lp_i + lp_j)
        lp = lp + is_rd * (lp_ax + lp_rf)
        lp = lp + is_qt * lp_dt

        ent = ent_skip + act * ent_op
        ent = ent + is_bd * (ent_i + ent_j)
        ent = ent + is_rd * (ent_ax + ent_rf)
        ent = ent + is_qt * ent_dt
        return lp.astype(jnp.float32), ent.astype(jnp.float32)

    # ---------------------------------------------------------- sample fields
    def sample_fields(self, ctx, key, *, op_mask=None, i_mask=None,
                      j_mask=None, axis_mask=None, pair_ok=None):
        """Draw the raw fields WITHOUT scoring them.

        Scoring stays with the caller so the adapter can score exactly what it
        emits. (With the prime gates gone nothing is clamped any more, but the
        split is what made the old 10.9-nat sample/evaluate mismatch
        impossible to reintroduce, so it stays.)
        """
        z = self.logits(ctx)
        k_skip, k_op, k_i, k_j, k_ax, k_rf, k_dt = jrand.split(key, 7)
        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        axis_mask = jnp.ones((NUM_REDUCE_AXES,)) if axis_mask is None else axis_mask

        skip = jrand.bernoulli(k_skip, jnn.sigmoid(z[O_SKIP]))
        op = jrand.categorical(k_op, jnp.where(op_mask > 0.5, z[O_OP:O_I], -1e9))
        i_idx = jrand.categorical(k_i, jnp.where(i_mask > 0.5, z[O_I:O_J], -1e9))
        jm = j_mask_given_i(i_idx, j_mask, pair_ok)
        j_idx = jrand.categorical(
            k_j, jnp.where(jm > 0.5, z[O_J:O_AXES], -1e9))
        ax_logits = jnp.where(axis_mask > 0.5, z[O_AXES:O_RFN], -1e9)
        axes = jrand.bernoulli(k_ax, jnn.sigmoid(ax_logits)) & (axis_mask > 0.5)
        rfn = jrand.categorical(k_rf, z[O_RFN:O_DTYPE])
        dt = jrand.bernoulli(k_dt, jnn.sigmoid(z[O_DTYPE]))
        return z, skip, op, i_idx, j_idx, axes, rfn, dt

    # ----------------------------------------------------------------- sample
    def __call__(self, ctx, key, *, op_mask=None, i_mask=None, j_mask=None,
                 axis_mask=None, pair_ok=None):
        z, skip, op, i_idx, j_idx, axes, rfn, dt = self.sample_fields(
            ctx, key, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, pair_ok=pair_ok)
        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        axis_mask = jnp.ones((NUM_REDUCE_AXES,)) if axis_mask is None else axis_mask
        lp, ent = self.score(z, skip, op, i_idx, j_idx, axes, rfn, dt,
                             op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
                             axis_mask=axis_mask, pair_ok=pair_ok)
        return ApproxAction(
            skip=skip,
            op=jnp.where(skip, OP_NONE, op).astype(jnp.int32),
            i=(i_idx + 1).astype(jnp.int32),
            j=(j_idx + 1).astype(jnp.int32),
            axes=axes,
            reduce_fn=rfn.astype(jnp.int32),
            dtype_idx=dt.astype(jnp.int32),
            log_prob=lp,
            entropy=ent,
        )
