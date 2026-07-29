"""ONE approximation head. 64 outputs, one forward pass, one sample.

Replaces the autoregressive sub-episode (op -> i -> j -> exponent -> kind, each
its own head and its own scan step) with a single flat emission per vertex.

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

THRESHOLD-SUM FACTORS
---------------------
``factor = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g`` with each exponent the
SUM of that prime's gates. Compact (32 params), ORDINAL and monotone -- one
more gate is one more factor of that prime -- so "slightly bigger factor" has a
meaningful gradient, unlike a flat categorical over every divisor.

WHY THE EXPONENT IS SCORED AS A POISSON-BINOMIAL
------------------------------------------------
The stored action keeps only the exponent SUMS, not the individual gate draws
(``MicroAction.exponents`` is per-prime). Many gate vectors give the same sum,
so ``sum(log p(gate_i))`` is NOT recoverable at evaluate time -- and PPO needs
sample-time and evaluate-time to score the SAME random variable or the
importance ratio is meaningless. So the exponent is treated as the sampled
quantity and scored exactly under the Poisson-binomial distribution of its
gates: ``log P(sum(Bernoulli(sigmoid(z_k))) == e)``, computed by an O(n^2) DP
(n <= 13). Exact, differentiable, and identical on both paths.

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

PRIME_GATES = ((2, 13), (3, 8), (5, 5), (7, 3), (11, 1), (13, 1), (17, 1))
PRIMES = tuple(p for p, _ in PRIME_GATES)
N_PRIMES = len(PRIME_GATES)
N_PRIME_GATES = sum(n for _, n in PRIME_GATES)          # 32
MAX_GATES = max(n for _, n in PRIME_GATES)              # 13

MAX_PAIR_IDX = 6
NUM_REDUCE_AXES = 9
REDUCE_FNS = ("mean", "min", "max", "abs_min", "abs_max")
NUM_REDUCE_FNS = len(REDUCE_FNS)

O_SKIP = 0
O_OP = 1
O_I = O_OP + NUM_APPROX_OPS                 # 5
O_J = O_I + MAX_PAIR_IDX                    # 11
O_PRIME = O_J + MAX_PAIR_IDX                # 17
O_AXES = O_PRIME + N_PRIME_GATES            # 49
O_RFN = O_AXES + NUM_REDUCE_AXES            # 58
O_DTYPE = O_RFN + NUM_REDUCE_FNS            # 63
HEAD_WIDTH = O_DTYPE + 1                    # 64

_PRIME_SLICES = []
_off = 0
for _p, _n in PRIME_GATES:
    _PRIME_SLICES.append((_off, _n))
    _off += _n


class ApproxAction(NamedTuple):
    skip: jax.Array
    op: jax.Array
    i: jax.Array             # 1..6
    j: jax.Array             # 1..6
    exps: jax.Array          # (7,) int32
    factor: jax.Array
    axes: jax.Array          # (9,) bool
    reduce_fn: jax.Array
    dtype_idx: jax.Array
    sampled_factor: jax.Array
    factor_agrees: jax.Array
    log_prob: jax.Array
    entropy: jax.Array


def _pb_probs(logits, n_pad):
    """Poisson-binomial pmf of ``sum_k Bernoulli(sigmoid(logits[k]))``.

    Returns ``(n_pad + 1,)`` probabilities. DP: fold one gate at a time,
    ``p_new[k] = p[k]*(1-q) + p[k-1]*q``. Static shape so it jits.
    """
    n = logits.shape[0]
    probs = jnp.zeros((n_pad + 1,)).at[0].set(1.0)

    def step(p, q):
        shifted = jnp.concatenate([jnp.zeros((1,)), p[:-1]])
        return p * (1.0 - q) + shifted * q, None

    qs = jnn.sigmoid(logits)
    probs, _ = jax.lax.scan(step, probs, qs)
    return probs


def _pb_logp_ent(logits, k, n_pad):
    p = _pb_probs(logits, n_pad)
    p = jnp.clip(p, 1e-12, 1.0)
    logp = jnp.log(p[k])
    ent = -jnp.sum(p * jnp.log(p))
    return logp, ent


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


class UnifiedApproxHead(eqx.Module):
    """embd_dim -> 64 logits -> one complete approximation decision."""

    proj: eqx.nn.MLP
    force_pure_diag: bool = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, force_pure_diag: bool = False,
                 hidden: int | None = None, key):
        self.force_pure_diag = force_pure_diag
        self.proj = eqx.nn.MLP(embd_dim, HEAD_WIDTH, hidden or embd_dim,
                               depth=1, key=key)

    def logits(self, ctx):
        return self.proj(ctx)

    # ------------------------------------------------------------------ score
    def score(self, z, skip, op, i_idx, j_idx, exps, axes, rfn, dt,
              *, op_mask, i_mask, j_mask, axis_mask):
        """Branch-masked (log_prob, entropy) for a GIVEN field assignment.

        Shared by sampling and evaluation so the two can never disagree --
        the PPO ratio is only meaningful if both score the same variable.
        """
        lp_skip, ent_skip = _bern_logp_ent(z[O_SKIP], skip)
        lp_op, ent_op = _cat_logp_ent(z[O_OP:O_I], op_mask, op)
        lp_i, ent_i = _cat_logp_ent(z[O_I:O_J], i_mask, i_idx)
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        lp_j, ent_j = _cat_logp_ent(z[O_J:O_PRIME], j_mask_eff, j_idx)

        lp_pr, ent_pr = 0.0, 0.0
        for k, (off, n) in enumerate(_PRIME_SLICES):
            lg = z[O_PRIME + off: O_PRIME + off + n]
            l, e = _pb_logp_ent(lg, jnp.clip(exps[k], 0, n), n)
            lp_pr = lp_pr + l
            ent_pr = ent_pr + e

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
        lp = lp + is_bd * (lp_i + lp_j + lp_pr)
        lp = lp + is_rd * (lp_ax + lp_rf)
        lp = lp + is_qt * lp_dt

        ent = ent_skip + act * ent_op
        ent = ent + is_bd * (ent_i + ent_j + ent_pr)
        ent = ent + is_rd * (ent_ax + ent_rf)
        ent = ent + is_qt * ent_dt
        return lp.astype(jnp.float32), ent.astype(jnp.float32)

    # ---------------------------------------------------------- sample fields
    def sample_fields(self, ctx, key, *, op_mask=None, i_mask=None,
                      j_mask=None, axis_mask=None):
        """Draw the raw fields WITHOUT scoring them.

        Scoring is deferred so the caller can first clamp a field to what the
        environment will actually accept (the exponents are clamped to
        ``tables.max_exps[g]``). Scoring the pre-clamp value and then emitting
        the post-clamp one makes sample and evaluate score DIFFERENT variables
        and silently breaks the PPO ratio -- measured at 10.9 nats.
        """
        z = self.logits(ctx)
        k_skip, k_op, k_i, k_j, k_pr, k_ax, k_rf, k_dt = jrand.split(key, 8)
        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        axis_mask = jnp.ones((NUM_REDUCE_AXES,)) if axis_mask is None else axis_mask

        skip = jrand.bernoulli(k_skip, jnn.sigmoid(z[O_SKIP]))
        op = jrand.categorical(k_op, jnp.where(op_mask > 0.5, z[O_OP:O_I], -1e9))
        i_idx = jrand.categorical(k_i, jnp.where(i_mask > 0.5, z[O_I:O_J], -1e9))
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        j_idx = jrand.categorical(
            k_j, jnp.where(j_mask_eff > 0.5, z[O_J:O_PRIME], -1e9))
        gates = jrand.bernoulli(k_pr, jnn.sigmoid(z[O_PRIME:O_AXES]))
        exps = jnp.stack([jnp.sum(gates[off:off + n].astype(jnp.int32))
                          for off, n in _PRIME_SLICES])
        ax_logits = jnp.where(axis_mask > 0.5, z[O_AXES:O_RFN], -1e9)
        axes = jrand.bernoulli(k_ax, jnn.sigmoid(ax_logits)) & (axis_mask > 0.5)
        rfn = jrand.categorical(k_rf, z[O_RFN:O_DTYPE])
        dt = jrand.bernoulli(k_dt, jnn.sigmoid(z[O_DTYPE]))
        return z, skip, op, i_idx, j_idx, exps, axes, rfn, dt

    # ----------------------------------------------------------------- sample
    def __call__(self, ctx, key, *, op_mask=None, i_mask=None, j_mask=None,
                 axis_mask=None, max_factor=None):
        z = self.logits(ctx)
        k_skip, k_op, k_i, k_j, k_pr, k_ax, k_rf, k_dt = jrand.split(key, 8)

        op_mask = jnp.ones((NUM_APPROX_OPS,)) if op_mask is None else op_mask
        i_mask = jnp.ones((MAX_PAIR_IDX,)) if i_mask is None else i_mask
        j_mask = jnp.ones((MAX_PAIR_IDX,)) if j_mask is None else j_mask
        axis_mask = jnp.ones((NUM_REDUCE_AXES,)) if axis_mask is None else axis_mask

        skip = jrand.bernoulli(k_skip, jnn.sigmoid(z[O_SKIP]))
        op = jrand.categorical(k_op, jnp.where(op_mask > 0.5, z[O_OP:O_I], -1e9))
        i_idx = jrand.categorical(k_i, jnp.where(i_mask > 0.5, z[O_I:O_J], -1e9))
        j_mask_eff = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
        j_mask_eff = jnp.where(jnp.sum(j_mask_eff) > 0, j_mask_eff, j_mask)
        j_idx = jrand.categorical(
            k_j, jnp.where(j_mask_eff > 0.5, z[O_J:O_PRIME], -1e9))

        gates = jrand.bernoulli(k_pr, jnn.sigmoid(z[O_PRIME:O_AXES]))
        exps = jnp.stack([
            jnp.sum(gates[off:off + n].astype(jnp.int32))
            for off, n in _PRIME_SLICES
        ])
        factor = jnp.maximum(
            jnp.prod(jnp.asarray(PRIMES, jnp.int32) ** exps).astype(jnp.int32), 1)
        sampled_factor = factor
        if max_factor is not None:
            # The gates span ~7e8 and OVERFLOW int32, so an unclamped sample is
            # almost always illegal. Clamp rather than reject: "too big" still
            # reads as "wants the biggest" and keeps the gradient usable.
            factor = jnp.minimum(factor, jnp.asarray(max_factor, jnp.int32))
        if self.force_pure_diag and max_factor is not None:
            factor = jnp.asarray(max_factor, jnp.int32)
        # When the forced (gcd) factor and the independently sampled prime
        # factor coincide the gates DID produce the applied outcome, so they
        # are credited exactly as in the unforced case -- both routes rewarded.
        factor_agrees = (sampled_factor == factor)

        ax_logits = jnp.where(axis_mask > 0.5, z[O_AXES:O_RFN], -1e9)
        axes = jrand.bernoulli(k_ax, jnn.sigmoid(ax_logits)) & (axis_mask > 0.5)
        rfn = jrand.categorical(k_rf, z[O_RFN:O_DTYPE])
        dt = jrand.bernoulli(k_dt, jnn.sigmoid(z[O_DTYPE]))

        lp, ent = self.score(z, skip, op, i_idx, j_idx, exps, axes, rfn, dt,
                             op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
                             axis_mask=axis_mask)
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
            log_prob=lp,
            entropy=ent,
        )
