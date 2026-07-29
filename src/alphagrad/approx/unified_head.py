"""ONE approximation head. 64 outputs, one forward pass, one sample.

Replaces the autoregressive sub-episode (op -> i -> j -> exponent -> kind, each
its own head and its own step) with a single flat emission per vertex.

LAYOUT (64 logits)
------------------
    [ 0: 1)  skip            Bernoulli
    [ 1: 5)  op              softmax {blockdiag, reduce, quant, none}
    [ 5:11)  blockdiag i     softmax over 1..6
    [11:17)  blockdiag j     softmax over 1..6
    [17:49)  prime gates     13/8/5/3/1/1/1 for 2/3/5/7/11/13/17
    [49:58)  reduce axes     9 independent gates (axes 0..8)
    [58:63)  reduce fn       softmax {mean, min, max, abs_min, abs_max}
    [63:64)  dtype           Bernoulli {float32, bfloat16}

WHY THRESHOLD-SUM FOR THE FACTOR
--------------------------------
The factor is a product of prime powers, so a flat categorical would need one
class per admissible divisor -- large, unordered, and with no gradient relation
between neighbouring factors. Summing independent gates per prime gives

    factor = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g,
    a = sum(13 gates), b = sum(8), c = sum(5), d = sum(3), e,f,g in {0,1}

which is compact (32 params), ORDINAL (one more gate = one more factor of that
prime) and monotone, so the gradient on "slightly bigger factor" is meaningful.
Ranges cover 2^13 = 8192 and 3^8 = 6561, past any axis size in these graphs.

``force_pure_diag`` overrides the sampled factor with the largest LEGAL one for
the chosen (i, j) pair -- a pure diagonal is the highest-factor block-diagonal,
so this is the "collapse to true diagonal" switch rather than a separate op.

Every field is sampled unconditionally (cheap, and keeps shapes static for
jit), but LOG-PROB AND ENTROPY ARE MASKED BY THE CHOSEN BRANCH: a blockdiag
sample must not be credited or penalised for the reduce-axis gates it did not
use. Otherwise the unused heads receive gradient from unrelated rewards and
drift, which is what made the old per-head entropies incomparable.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple

# ---- op codes ------------------------------------------------------------
OP_BLOCKDIAG, OP_REDUCE, OP_QUANT, OP_NONE = 0, 1, 2, 3
NUM_APPROX_OPS = 4

# (prime, n_gates); n_gates caps that prime's exponent
PRIME_GATES = ((2, 13), (3, 8), (5, 5), (7, 3), (11, 1), (13, 1), (17, 1))
PRIMES = tuple(p for p, _ in PRIME_GATES)
N_PRIME_GATES = sum(n for _, n in PRIME_GATES)          # 32

MAX_PAIR_IDX = 6            # i, j drawn from 1..6
NUM_REDUCE_AXES = 9         # axes 0..8, independently gated
REDUCE_FNS = ("mean", "min", "max", "abs_min", "abs_max")
NUM_REDUCE_FNS = len(REDUCE_FNS)
QUANT_DTYPES = ("float32", "bfloat16")

# slice offsets
O_SKIP = 0
O_OP = 1
O_I = O_OP + NUM_APPROX_OPS                 # 5
O_J = O_I + MAX_PAIR_IDX                    # 11
O_PRIME = O_J + MAX_PAIR_IDX                # 17
O_AXES = O_PRIME + N_PRIME_GATES            # 49
O_RFN = O_AXES + NUM_REDUCE_AXES            # 58
O_DTYPE = O_RFN + NUM_REDUCE_FNS            # 63
HEAD_WIDTH = O_DTYPE + 1                    # 64


class ApproxAction(NamedTuple):
    skip: jax.Array          # () bool
    op: jax.Array            # () int32
    i: jax.Array             # () int32, 1..6
    j: jax.Array             # () int32, 1..6
    exps: jax.Array          # (7,) int32 prime exponents
    factor: jax.Array        # () int32
    axes: jax.Array          # (9,) bool
    reduce_fn: jax.Array     # () int32
    dtype_idx: jax.Array     # () int32 (0=f32, 1=bf16)
    sampled_factor: jax.Array   # () int32, before any force/clamp override
    factor_agrees: jax.Array    # () bool, sampled == applied factor
    log_prob: jax.Array      # () float32, branch-masked
    entropy: jax.Array       # () float32, branch-masked


def _cat_logp_ent(logits, mask, idx):
    """log p(idx) and entropy of a masked categorical."""
    logits = jnp.where(mask > 0.5, logits, -1e9)
    logp_all = jnn.log_softmax(logits)
    p = jnn.softmax(logits)
    ent = -jnp.sum(jnp.where(p > 0, p * logp_all, 0.0))
    return logp_all[idx], ent


def _bern_logp_ent(logit, x):
    """log p(x) and entropy of a Bernoulli given its logit."""
    lp1 = jnn.log_sigmoid(logit)
    lp0 = jnn.log_sigmoid(-logit)
    p = jnn.sigmoid(logit)
    ent = -(p * lp1 + (1.0 - p) * lp0)
    return jnp.where(x, lp1, lp0), ent


class UnifiedApproxHead(eqx.Module):
    """embd_dim -> 64 logits -> one complete approximation decision."""

    proj: eqx.nn.MLP
    force_pure_diag: bool = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, force_pure_diag: bool = False,
                 hidden: int | None = None, key):
        self.force_pure_diag = force_pure_diag
        self.proj = eqx.nn.MLP(
            embd_dim, HEAD_WIDTH, hidden or embd_dim, depth=1, key=key
        )

    def logits(self, ctx):
        return self.proj(ctx)

    def __call__(
        self,
        ctx,
        key,
        *,
        op_mask=None,        # (4,)  legal ops
        i_mask=None,         # (6,)  legal i indices
        j_mask=None,         # (6,)  legal j indices
        axis_mask=None,      # (9,)  reducible axes
        max_factor=None,     # ()    largest legal factor, for force_pure_diag
    ):
        z = self.logits(ctx)
        k_skip, k_op, k_i, k_j, k_pr, k_ax, k_rf, k_dt = jrand.split(key, 8)

        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        axis_mask = jnp.ones((NUM_REDUCE_AXES,)) if axis_mask is None else axis_mask

        # ---- skip -------------------------------------------------------
        skip_logit = z[O_SKIP]
        skip = jrand.bernoulli(k_skip, jnn.sigmoid(skip_logit))
        lp_skip, ent_skip = _bern_logp_ent(skip_logit, skip)

        # ---- op ---------------------------------------------------------
        op_logits = jnp.where(op_mask > 0.5, z[O_OP:O_I], -1e9)
        op = jrand.categorical(k_op, op_logits)
        lp_op, ent_op = _cat_logp_ent(z[O_OP:O_I], op_mask, op)

        # ---- blockdiag i / j --------------------------------------------
        i_idx = jrand.categorical(k_i, jnp.where(i_mask > 0.5, z[O_I:O_J], -1e9))
        lp_i, ent_i = _cat_logp_ent(z[O_I:O_J], i_mask, i_idx)
        # j != i: mask the chosen i out so the pair is always distinct
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        j_idx = jrand.categorical(
            k_j, jnp.where(j_mask_eff > 0.5, z[O_J:O_PRIME], -1e9))
        lp_j, ent_j = _cat_logp_ent(z[O_J:O_PRIME], j_mask_eff, j_idx)

        # ---- prime-exponent gates ---------------------------------------
        pr_logits = z[O_PRIME:O_AXES]
        gates = jrand.bernoulli(k_pr, jnn.sigmoid(pr_logits))
        lp_pr_all, ent_pr_all = _bern_logp_ent(pr_logits, gates)
        exps, off = [], 0
        for _, n in PRIME_GATES:
            exps.append(jnp.sum(gates[off:off + n].astype(jnp.int32)))
            off += n
        exps = jnp.stack(exps)
        factor = jnp.prod(
            jnp.asarray(PRIMES, dtype=jnp.int32) ** exps).astype(jnp.int32)
        factor = jnp.maximum(factor, 1)
        if max_factor is not None:
            # The gates span 2^13 * 3^8 * ... ~ 7e8, far past any axis size, so
            # an unclamped sample is almost always illegal. Clamp to the largest
            # LEGAL factor for this (i, j) pair; the env then snaps to a divisor.
            # Clamping (not rejecting) keeps the sample usable and the gradient
            # meaningful -- "too big" still reads as "wants the biggest".
            factor = jnp.minimum(factor, jnp.asarray(max_factor, jnp.int32))
        sampled_factor = factor
        if self.force_pure_diag and max_factor is not None:
            # A pure diagonal IS the highest-factor block-diagonal.
            factor = jnp.asarray(max_factor, jnp.int32)
        # AGREEMENT: the forced (gcd / max) factor and the independently
        # sampled prime-exponent factor can coincide. When they do, the gate
        # sample DID produce the applied factor, so it is credited exactly like
        # the unforced case -- the prime path is never masked out for having
        # been overridden, and both routes to the same factor are rewarded.
        factor_agrees = (sampled_factor == factor)

        # ---- reduce axes + aggregation ----------------------------------
        ax_logits = jnp.where(axis_mask > 0.5, z[O_AXES:O_RFN], -1e9)
        axes = jrand.bernoulli(k_ax, jnn.sigmoid(ax_logits)) & (axis_mask > 0.5)
        lp_ax_all, ent_ax_all = _bern_logp_ent(z[O_AXES:O_RFN], axes)
        lp_ax_all = lp_ax_all * (axis_mask > 0.5)
        ent_ax_all = ent_ax_all * (axis_mask > 0.5)
        rfn = jrand.categorical(k_rf, z[O_RFN:O_DTYPE])
        lp_rf, ent_rf = _cat_logp_ent(
            z[O_RFN:O_DTYPE], jnp.ones((NUM_REDUCE_FNS,)), rfn)

        # ---- dtype ------------------------------------------------------
        dt_logit = z[O_DTYPE]
        dt = jrand.bernoulli(k_dt, jnn.sigmoid(dt_logit))
        lp_dt, ent_dt = _bern_logp_ent(dt_logit, dt)

        # ---- branch masking ---------------------------------------------
        # Only the fields the chosen op actually CONSUMES contribute. Without
        # this the unused sub-heads get gradient from rewards they had no part
        # in, drift, and make per-head entropies incomparable.
        act = (~skip).astype(jnp.float32)          # skip => nothing applies
        is_bd = act * (op == OP_BLOCKDIAG).astype(jnp.float32)
        is_rd = act * (op == OP_REDUCE).astype(jnp.float32)
        is_qt = act * (op == OP_QUANT).astype(jnp.float32)

        lp = lp_skip + act * lp_op
        # Prime gates stay in the objective under blockdiag REGARDLESS of the
        # force_pure_diag override: when the sampled factor matches the forced
        # one the gates earned the outcome, and when it does not we still train
        # them rather than dropping the signal.
        lp = lp + is_bd * (lp_i + lp_j + jnp.sum(lp_pr_all))
        lp = lp + is_rd * (jnp.sum(lp_ax_all) + lp_rf)
        lp = lp + is_qt * lp_dt

        ent = ent_skip + act * ent_op
        ent = ent + is_bd * (ent_i + ent_j + jnp.sum(ent_pr_all))
        ent = ent + is_rd * (jnp.sum(ent_ax_all) + ent_rf)
        ent = ent + is_qt * ent_dt

        return ApproxAction(
            skip=skip,
            op=jnp.where(skip, OP_NONE, op).astype(jnp.int32),
            i=(i_idx + 1).astype(jnp.int32),
            j=(j_idx + 1).astype(jnp.int32),
            exps=exps.astype(jnp.int32),
            factor=factor,
            axes=axes,
            reduce_fn=rfn.astype(jnp.int32),
            dtype_idx=dt.astype(jnp.int32),
            sampled_factor=sampled_factor.astype(jnp.int32),
            factor_agrees=factor_agrees,
            log_prob=lp.astype(jnp.float32),
            entropy=ent.astype(jnp.float32),
        )
