"""``UnifiedApproxHead`` as a DROP-IN for :class:`MicroActionPolicy`.

Rather than refactoring the action pipeline (env decoder, Trajectory fields and
PPO loss all consume ``MicroAction``), this adapter reproduces
``MicroActionPolicy``'s exact ``sample`` / ``evaluate`` contract and emits the
same ``MicroAction`` sequence. Everything downstream is untouched.

TRANSLATION
-----------
* op codes already coincide: BLOCKDIAG/REDUCE/QUANT/NONE == DIAG/COMPRESS/
  QUANT/END == 0/1/2/3.
* ``i``/``j`` are 1..6 in the head, 0-based axis-token indices in MicroAction.
* EXPONENTS: the head samples against a FIXED global prime list
  (2,3,5,7,11,13,17); MicroAction's ``exponents`` are indexed against
  ``tables.primes[g]`` -- the prime list OF THIS PAIR'S GCD. We scatter by
  matching prime VALUES and clamp to ``tables.max_exps[g]``, so the emitted
  factor always divides g and is legal by construction (strictly better than
  the head's own scalar clamp).
* REDUCE with several axes becomes several COMPRESS sub-steps, one per selected
  axis -- which is what ``max_substeps`` was always for.
* ``compress_kind`` maps the head's 5 reductions onto graphax's 6
  (``median`` is simply never emitted).

WHY ``evaluate`` CAN RECONSTRUCT THE FIELDS
-------------------------------------------
Everything the head scores is recoverable from the stored MicroAction: ``skip``
= "every sub-step is END" (and when skip fires the op term is masked out at
BOTH ends, so nothing is lost), op/i/j from sub-step 0, the axis set from the
COMPRESS sub-steps, kind and dtype from their fields, and the exponents by
inverting the prime scatter. The one thing NOT recoverable is the individual
gate draws -- which is exactly why the head scores the exponent as a
Poisson-binomial over its gates rather than a product of per-gate Bernoullis.
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
    MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES, NUM_REDUCE_FNS, PRIMES,
    REDUCE_FNS, UnifiedApproxHead, _PRIME_SLICES, _pb_probs, O_AXES, O_DTYPE,
    O_I, O_J, O_OP, O_PRIME, O_RFN)

_KIND_MAP = jnp.asarray([COMPRESS_KINDS.index(k) for k in REDUCE_FNS],
                        dtype=jnp.int32)
_KIND_INV = jnp.asarray(
    [REDUCE_FNS.index(k) if k in REDUCE_FNS else 0 for k in COMPRESS_KINDS],
    dtype=jnp.int32)


def _dtype_slot(name, fallback):
    for i, d in enumerate(QUANT_DTYPES):
        if str(d) == name or getattr(d, "__name__", "") == name:
            return i
    return fallback


_F32_SLOT = _dtype_slot("float32", 0)
_BF16_SLOT = _dtype_slot("bfloat16", min(1, NUM_QUANT_DTYPES - 1))
_PRIMES_ARR = jnp.asarray(PRIMES, dtype=jnp.int32)


class _Fields(NamedTuple):
    """The subset of ApproxAction that ``_emit`` consumes."""
    skip: jax.Array
    op: jax.Array
    i: jax.Array
    j: jax.Array
    exps: jax.Array
    axes: jax.Array
    reduce_fn: jax.Array
    dtype_idx: jax.Array


class UnifiedMicroPolicy(eqx.Module):
    """MicroActionPolicy-compatible facade over a single UnifiedApproxHead."""

    head: UnifiedApproxHead
    max_substeps: int = eqx.field(static=True)

    def __init__(self, *, embd_dim, max_substeps, force_pure_diag=False, key):
        self.head = UnifiedApproxHead(
            embd_dim, force_pure_diag=force_pure_diag, key=key)
        self.max_substeps = max_substeps

    def _pair_sizes(self, features, i_idx, j_idx):
        n = features.size.shape[0]
        gi = jnp.clip(i_idx, 0, n - 1)
        gj = jnp.clip(j_idx, 0, n - 1)
        return (features.size[gi].astype(jnp.int32),
                features.size[gj].astype(jnp.int32))

    def _env_exponents(self, tables, g, exps):
        env_primes = tables.primes[g]
        env_max = tables.max_exps[g]
        match = (env_primes[:, None] == _PRIMES_ARR[None, :])
        e = jnp.sum(jnp.where(match, exps[None, :], 0), axis=1)
        e = jnp.clip(e, 0, jnp.maximum(env_max, 0))
        factor = jnp.prod(jnp.where(env_primes > 0, env_primes, 1) ** e)
        return e.astype(jnp.int32), jnp.maximum(factor.astype(jnp.int32), 1)

    def _masks(self, features, pair_valid, compress_valid, op_legality_override):
        n = features.size.shape[0]
        valid = features.valid_mask
        m = min(MAX_PAIR_IDX, n)
        i_mask = jnp.zeros((MAX_PAIR_IDX,)).at[:m].set(valid[:m])
        i_mask = jnp.where(jnp.sum(i_mask) > 0, i_mask, jnp.ones((MAX_PAIR_IDX,)))
        j_mask = i_mask
        if compress_valid is not None:
            cv = compress_valid.astype(jnp.float32)
            k = min(NUM_REDUCE_AXES, cv.shape[0])
            axis_mask = jnp.zeros((NUM_REDUCE_AXES,)).at[:k].set(cv[:k])
        else:
            k = min(NUM_REDUCE_AXES, n)
            axis_mask = jnp.zeros((NUM_REDUCE_AXES,)).at[:k].set(valid[:k])
        op_mask = jnp.ones((NUM_APPROX_OPS,))
        if op_legality_override is not None:
            op_mask = op_mask * op_legality_override.astype(jnp.float32)
        # NONE is MASKED OUT: it is redundant with the skip head and produces a
        # byte-identical all-END sequence, so evaluate could not tell the two
        # apart and the round-trip log-prob was off by ~10.9 nats. With NONE
        # unavailable, "every sub-step is END" means EXACTLY skip=True.
        op_mask = op_mask.at[3].set(0.0)
        op_mask = jnp.where(jnp.sum(op_mask) > 0, op_mask,
                            jnp.ones((NUM_APPROX_OPS,)).at[3].set(0.0))
        return op_mask, i_mask, j_mask, axis_mask

    def _emit(self, act, tables, features):
        S = self.max_substeps
        i0 = act.i - 1
        j0 = act.j - 1
        N_i, N_j = self._pair_sizes(features, i0, j0)
        g = tables.gcd[jnp.clip(N_i, 0, tables.gcd.shape[0] - 1),
                       jnp.clip(N_j, 0, tables.gcd.shape[1] - 1)]
        env_exps, env_factor = self._env_exponents(tables, g, act.exps)
        if self.head.force_pure_diag:
            env_exps = tables.max_exps[g].astype(jnp.int32)
            env_factor = jnp.maximum(
                jnp.prod(jnp.where(tables.primes[g] > 0, tables.primes[g], 1)
                         ** env_exps).astype(jnp.int32), 1)

        slots = jnp.arange(S)
        sel = act.axes
        order = jnp.argsort(jnp.where(sel, 0, 1).astype(jnp.int32))
        n_sel = jnp.sum(sel.astype(jnp.int32))
        axis_for_slot = order[jnp.clip(slots, 0, NUM_REDUCE_AXES - 1)]

        is_bd = (act.op == OP_DIAG) & (~act.skip)
        is_rd = (act.op == OP_COMPRESS) & (~act.skip)
        is_qt = (act.op == OP_QUANT) & (~act.skip)
        comp_here = is_rd & (slots < n_sel)
        first = slots == 0

        op_seq = jnp.where(comp_here, OP_COMPRESS,
                           jnp.where(first & is_bd, OP_DIAG,
                                     jnp.where(first & is_qt, OP_QUANT, OP_END)))
        i_seq = jnp.where(comp_here, axis_for_slot,
                          jnp.where(first & is_bd, i0, 0))
        j_seq = jnp.where(first & is_bd, j0, 0)
        exp_seq = jnp.where((first & is_bd)[:, None], env_exps[None, :], 0)
        fac_seq = jnp.where(first & is_bd, env_factor, 0)
        kind_seq = jnp.where(comp_here, _KIND_MAP[act.reduce_fn], 0)
        qdt_seq = jnp.where(first & is_qt,
                            jnp.where(act.dtype_idx > 0, _BF16_SLOT, _F32_SLOT), 0)

        actions = MicroAction(
            op_type=op_seq.astype(jnp.int32),
            i=i_seq.astype(jnp.int32),
            j=j_seq.astype(jnp.int32),
            exponents=exp_seq.astype(jnp.int32),
            factor=fac_seq.astype(jnp.int32),
            compress_kind=kind_seq.astype(jnp.int32),
            quant_dtype=qdt_seq.astype(jnp.int32),
            quant_scale_sign=jnp.ones((S,), jnp.int32),
            quant_scale_frac=jnp.zeros((S,), jnp.float32),
        )
        arity = jnp.where(is_rd, n_sel, jnp.where(is_bd | is_qt, 1, 0))
        return actions, arity.astype(jnp.int32)

    def _dists(self, z, features):
        S, n = self.max_substeps, features.size.shape[0]
        op_d = jnp.broadcast_to(jnn.softmax(z[O_OP:O_I])[None, :], (S, NUM_OPS))
        pad = max(0, n - MAX_PAIR_IDX)
        i_p = jnp.concatenate([jnn.softmax(z[O_I:O_J]), jnp.zeros((pad,))])[:n]
        j_p = jnp.concatenate([jnn.softmax(z[O_J:O_PRIME]), jnp.zeros((pad,))])[:n]
        i_d = jnp.broadcast_to(i_p[None, :], (S, n))
        j_d = jnp.broadcast_to(j_p[None, :], (S, n))
        rows = []
        for off, ng in _PRIME_SLICES:
            p = _pb_probs(z[O_PRIME + off: O_PRIME + off + ng], ng)
            rows.append(jnp.concatenate(
                [p, jnp.zeros((MAX_EXPONENT + 1 - p.shape[0],))]))
        for _ in range(MAX_PRIMES - len(_PRIME_SLICES)):
            rows.append(jnp.zeros((MAX_EXPONENT + 1,)).at[0].set(1.0))
        exp_d = jnp.broadcast_to(jnp.stack(rows)[None, ...],
                                 (S, MAX_PRIMES, MAX_EXPONENT + 1))
        kp = jnn.softmax(z[O_RFN:O_DTYPE])
        kind_d = jnp.broadcast_to(
            jnp.zeros((NUM_COMPRESS_KINDS,)).at[_KIND_MAP].set(kp)[None, :],
            (S, NUM_COMPRESS_KINDS))
        return op_d, i_d, j_d, exp_d, kind_d

    def sample(self, vertex_context, init_features, tables, key,
               pair_valid=None, compress_valid=None,
               quant_legality_mask=None, op_legality_override=None):
        z = self.head.logits(vertex_context)
        op_mask, i_mask, j_mask, axis_mask = self._masks(
            init_features, pair_valid, compress_valid, op_legality_override)
        (z, skip, op, i_idx, j_idx, raw_exps, axes, rfn, dt) = (
            self.head.sample_fields(vertex_context, key, op_mask=op_mask,
                                    i_mask=i_mask, j_mask=j_mask,
                                    axis_mask=axis_mask))
        # Clamp the exponents to what the env will accept, THEN score, so the
        # scored value is exactly the value that gets emitted and later
        # recovered by evaluate.
        N_i, N_j = self._pair_sizes(init_features, i_idx, j_idx)
        g = tables.gcd[jnp.clip(N_i, 0, tables.gcd.shape[0] - 1),
                       jnp.clip(N_j, 0, tables.gcd.shape[1] - 1)]
        env_exps, _ = self._env_exponents(tables, g, raw_exps)
        match = (tables.primes[g][:, None] == _PRIMES_ARR[None, :])
        exps = jnp.sum(jnp.where(match, env_exps[:, None], 0), axis=0)
        lp, ent = self.head.score(
            z, skip, op, i_idx, j_idx, exps, axes, rfn, dt,
            op_mask=op_mask, i_mask=i_mask, j_mask=j_mask, axis_mask=axis_mask)
        act = _Fields(skip=skip, op=op, i=i_idx + 1, j=j_idx + 1, exps=exps,
                      axes=axes, reduce_fn=rfn, dtype_idx=dt.astype(jnp.int32))
        actions, arity = self._emit(act, tables, init_features)
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features)
        return (actions, lp, ent, arity,
                op_d, i_d, j_d, exp_d, kind_d, jnp.asarray(0.0, jnp.float32))

    def evaluate(self, vertex_context, init_features, tables, actions,
                 pair_valid=None, compress_valid=None,
                 quant_legality_mask=None, op_legality_override=None):
        z = self.head.logits(vertex_context)
        op_mask, i_mask, j_mask, axis_mask = self._masks(
            init_features, pair_valid, compress_valid, op_legality_override)

        ops = actions.op_type
        skip = jnp.all(ops == OP_END)
        head_op = ops[0]
        i_idx = jnp.clip(actions.i[0], 0, MAX_PAIR_IDX - 1)
        j_idx = jnp.clip(actions.j[0], 0, MAX_PAIR_IDX - 1)

        N_i, N_j = self._pair_sizes(init_features, i_idx, j_idx)
        g = tables.gcd[jnp.clip(N_i, 0, tables.gcd.shape[0] - 1),
                       jnp.clip(N_j, 0, tables.gcd.shape[1] - 1)]
        env_primes = tables.primes[g]
        match = (env_primes[:, None] == _PRIMES_ARR[None, :])
        exps = jnp.sum(jnp.where(match, actions.exponents[0][:, None], 0), axis=0)

        is_c = (ops == OP_COMPRESS)
        axes = jnp.any(
            (jnp.arange(NUM_REDUCE_AXES)[:, None] == actions.i[None, :])
            & is_c[None, :], axis=1)
        rfn = _KIND_INV[jnp.clip(actions.compress_kind[0], 0,
                                 NUM_COMPRESS_KINDS - 1)]
        dt = (actions.quant_dtype[0] == _BF16_SLOT)

        lp, ent = self.head.score(
            z, skip, head_op, i_idx, j_idx, exps, axes, rfn, dt,
            op_mask=op_mask, i_mask=i_mask, j_mask=j_mask, axis_mask=axis_mask)
        arity = jnp.sum((ops != OP_END).astype(jnp.int32))
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features)
        return (lp, ent, arity, op_d, i_d, j_d, exp_d, kind_d,
                jnp.asarray(0.0, jnp.float32))
