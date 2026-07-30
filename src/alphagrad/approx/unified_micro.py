"""Simplified UnifiedApproxHead behind MicroActionPolicy's contract.

The env decoder, Trajectory and PPO loss all consume ``MicroAction``, so this
adapter keeps that contract byte-for-byte and only changes what gets put in it.

FIXED RULES (see unified_head for why)
  * BLOCKDIAG factor = ``min(N_i, N_j)`` -- square gives a pure diagonal,
    2x8 gives two 1x4 blocks. Then SNAPPED DOWN to a divisor of the pair's gcd,
    because graphax only accepts a factor that divides it; the snap is what the
    old prime-exponent clamp was really doing.
  * REDUCE   -> one COMPRESS sub-step on axis 0, kind "mean".
  * QUANT    -> bfloat16.

Only ``skip / op / i / j`` are sampled, so ``evaluate`` recovers all of them
exactly from the stored MicroAction: skip = "every sub-step is END" (NONE is not
in the op set, so that mapping is unambiguous), op/i/j from sub-step 0.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import NamedTuple

from alphagrad.approx.heads import (
    COMPRESS_KINDS, MAX_EXPONENT, MAX_PRIMES, MicroAction, NUM_COMPRESS_KINDS,
    NUM_OPS, NUM_QUANT_DTYPES, OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT,
    QUANT_DTYPES)
from alphagrad.approx.unified_head import (
    HEAD_WIDTH, MAX_PAIR_IDX, NUM_APPROX_OPS, O_I, O_J, O_OP, O_SKIP,
    UnifiedApproxHead, block_count)

_MEAN_KIND = COMPRESS_KINDS.index("mean")


def _bf16_slot():
    for i, d in enumerate(QUANT_DTYPES):
        if str(d) == "bfloat16" or getattr(d, "__name__", "") == "bfloat16":
            return i
    return min(1, NUM_QUANT_DTYPES - 1)


_BF16 = _bf16_slot()


class _Fields(NamedTuple):
    skip: jax.Array
    op: jax.Array
    i: jax.Array
    j: jax.Array


class UnifiedMicroPolicy(eqx.Module):
    """MicroActionPolicy-compatible facade over the simplified head."""

    head: UnifiedApproxHead
    max_substeps: int = eqx.field(static=True)

    def __init__(self, *, embd_dim, max_substeps, key, **_ignored):
        self.head = UnifiedApproxHead(embd_dim, key=key)
        self.max_substeps = max_substeps

    # ------------------------------------------------------------- helpers
    def _pair_sizes(self, features, i_idx, j_idx):
        n = features.size.shape[0]
        return (features.size[jnp.clip(i_idx, 0, n - 1)].astype(jnp.int32),
                features.size[jnp.clip(j_idx, 0, n - 1)].astype(jnp.int32))

    def _factor(self, tables, features, i0, j0):
        """min(N_i, N_j), snapped DOWN to a divisor of gcd(N_i, N_j)."""
        N_i, N_j = self._pair_sizes(features, i0, j0)
        S = tables.gcd.shape[0]
        g = tables.gcd[jnp.clip(N_i, 0, S - 1), jnp.clip(N_j, 0, S - 1)]
        want = block_count(N_i, N_j)
        want = jnp.minimum(want, jnp.maximum(g, 1))
        # largest d <= want with g % d == 0 (g <= max_axis_size, so this
        # candidate sweep is small and fully static)
        cand = jnp.arange(1, tables.gcd.shape[0] + 1, dtype=jnp.int32)
        okc = (jnp.mod(jnp.maximum(g, 1), cand) == 0) & (cand <= want)
        return jnp.max(jnp.where(okc, cand, 1)).astype(jnp.int32)

    def _masks(self, features, compress_valid, op_legality_override):
        n = features.size.shape[0]
        valid = features.valid_mask
        m = min(MAX_PAIR_IDX, n)
        i_mask = jnp.zeros((MAX_PAIR_IDX,)).at[:m].set(valid[:m])
        i_mask = jnp.where(jnp.sum(i_mask) > 0, i_mask, jnp.ones((MAX_PAIR_IDX,)))
        op_mask = jnp.ones((NUM_APPROX_OPS,))
        if op_legality_override is not None:
            ov = op_legality_override.astype(jnp.float32)
            op_mask = op_mask * ov[:NUM_APPROX_OPS]
        # REDUCE needs axis 0 to be compressible
        if compress_valid is not None:
            cv = compress_valid.astype(jnp.float32)
            op_mask = op_mask.at[1].set(op_mask[1] * jnp.where(cv[0] > 0.5, 1.0, 0.0))
        op_mask = jnp.where(jnp.sum(op_mask) > 0, op_mask,
                            jnp.zeros((NUM_APPROX_OPS,)).at[2].set(1.0))
        return op_mask, i_mask, i_mask

    def _emit(self, f, tables, features):
        S = self.max_substeps
        i0, j0 = f.i - 1, f.j - 1
        factor = self._factor(tables, features, i0, j0)
        slots = jnp.arange(S)
        first = slots == 0
        act = ~f.skip
        is_bd = (f.op == OP_DIAG) & act
        is_rd = (f.op == OP_COMPRESS) & act
        is_qt = (f.op == OP_QUANT) & act

        op_seq = jnp.where(first & is_bd, OP_DIAG,
                           jnp.where(first & is_rd, OP_COMPRESS,
                                     jnp.where(first & is_qt, OP_QUANT, OP_END)))
        # COMPRESS: axis 0 always. DIAG: the sampled pair.
        i_seq = jnp.where(first & is_bd, i0, 0)
        j_seq = jnp.where(first & is_bd, j0, 0)
        exp_seq = jnp.zeros((S, MAX_PRIMES), jnp.int32)
        fac_seq = jnp.where(first & is_bd, factor, 0)
        kind_seq = jnp.where(first & is_rd, _MEAN_KIND, 0)
        qdt_seq = jnp.where(first & is_qt, _BF16, 0)

        actions = MicroAction(
            op_type=op_seq.astype(jnp.int32),
            i=i_seq.astype(jnp.int32),
            j=j_seq.astype(jnp.int32),
            exponents=exp_seq,
            factor=fac_seq.astype(jnp.int32),
            compress_kind=kind_seq.astype(jnp.int32),
            quant_dtype=qdt_seq.astype(jnp.int32),
            quant_scale_sign=jnp.ones((S,), jnp.int32),
            quant_scale_frac=jnp.zeros((S,), jnp.float32),
        )
        arity = jnp.where(act, 1, 0).astype(jnp.int32)
        return actions, arity

    def _dists(self, z, features):
        S, n = self.max_substeps, features.size.shape[0]
        opp = jnn.softmax(z[O_OP:O_I])
        op_d = jnp.broadcast_to(
            jnp.zeros((NUM_OPS,)).at[:NUM_APPROX_OPS].set(opp)[None, :], (S, NUM_OPS))
        pad = max(0, n - MAX_PAIR_IDX)
        i_p = jnp.concatenate([jnn.softmax(z[O_I:O_J]), jnp.zeros((pad,))])[:n]
        j_p = jnp.concatenate([jnn.softmax(z[O_J:HEAD_WIDTH]), jnp.zeros((pad,))])[:n]
        i_d = jnp.broadcast_to(i_p[None, :], (S, n))
        j_d = jnp.broadcast_to(j_p[None, :], (S, n))
        exp_d = jnp.broadcast_to(
            jnp.zeros((MAX_PRIMES, MAX_EXPONENT + 1)).at[:, 0].set(1.0)[None, ...],
            (S, MAX_PRIMES, MAX_EXPONENT + 1))
        kind_d = jnp.broadcast_to(
            jnp.zeros((NUM_COMPRESS_KINDS,)).at[_MEAN_KIND].set(1.0)[None, :],
            (S, NUM_COMPRESS_KINDS))
        return op_d, i_d, j_d, exp_d, kind_d

    # -------------------------------------------------------------- sample
    def sample(self, vertex_context, init_features, tables, key,
               pair_valid=None, compress_valid=None,
               quant_legality_mask=None, op_legality_override=None):
        op_mask, i_mask, j_mask = self._masks(
            init_features, compress_valid, op_legality_override)
        z, skip, op, i_idx, j_idx = self.head.sample_fields(
            vertex_context, key, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask)
        lp, ent = self.head.score(z, skip, op, i_idx, j_idx,
                                  op_mask=op_mask, i_mask=i_mask, j_mask=j_mask)
        actions, arity = self._emit(
            _Fields(skip=skip, op=op, i=i_idx + 1, j=j_idx + 1),
            tables, init_features)
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features)
        return (actions, lp, ent, arity,
                op_d, i_d, j_d, exp_d, kind_d, jnp.asarray(0.0, jnp.float32))

    # ------------------------------------------------------------ evaluate
    def evaluate(self, vertex_context, init_features, tables, actions,
                 pair_valid=None, compress_valid=None,
                 quant_legality_mask=None, op_legality_override=None):
        op_mask, i_mask, j_mask = self._masks(
            init_features, compress_valid, op_legality_override)
        z = self.head.logits(vertex_context)
        ops = actions.op_type
        skip = jnp.all(ops == OP_END)
        op = jnp.clip(ops[0], 0, NUM_APPROX_OPS - 1)
        i_idx = jnp.clip(actions.i[0], 0, MAX_PAIR_IDX - 1)
        j_idx = jnp.clip(actions.j[0], 0, MAX_PAIR_IDX - 1)
        lp, ent = self.head.score(z, skip, op, i_idx, j_idx,
                                  op_mask=op_mask, i_mask=i_mask, j_mask=j_mask)
        arity = jnp.sum((ops != OP_END).astype(jnp.int32))
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features)
        return (lp, ent, arity, op_d, i_d, j_d, exp_d, kind_d,
                jnp.asarray(0.0, jnp.float32))
