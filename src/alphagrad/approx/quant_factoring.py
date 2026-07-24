"""Factored quantization: turn the flat dtype catalog into per-factor value
vocabularies so the policy picks a dtype semantically.

The catalog (``graphax.sparse.micro_actions.QUANT_DTYPES``) is decomposed by
``QUANT_DTYPE_ATTRS`` into ``(kind, bits, exp, mantissa, bias, finite,
unsigned_zero)``. Restricted to the dtypes this backend can actually use
(``quant_hardware_masks()[0]``), that tuple is a UNIQUE key, so the policy can:

* sample each factor in order, MASKING every factor to the values that still
  leave ≥1 real dtype reachable given the earlier picks (a small match-reduce
  over the usable rows), and
* RESOLVE the completed tuple back to exactly one catalog dtype.

Correlated factors collapse for free under the ≥2-options head-skipping rule:
an integer kind forces ``exp=mantissa=bias=0`` etc., and once ``bias`` is chosen
``finite``/``unsigned_zero`` are usually pinned — those heads are then skipped.

``scale_sign`` (±1) is deliberately NOT part of this tuple: it is a separate
2-way head kept for every kind (it folds into ``scalar_mult`` and rescales the
value, so it is a no-op for signed/float but the arm-selector for unsigned).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from graphax.sparse.micro_actions import (
    QA_BIAS, QA_BITS, QA_EXP, QA_FINITE, QA_KIND, QA_MANTISSA, QA_UZ,
    QUANT_DTYPE_ATTRS, QUANT_DTYPES, quant_hardware_masks,
)

# The order the head samples the dtype-defining factors in. ``sign`` is handled
# separately (it is not part of the dtype identity).
FACTOR_ORDER = (QA_KIND, QA_BITS, QA_EXP, QA_MANTISSA, QA_BIAS, QA_FINITE, QA_UZ)
NUM_FACTORS = len(FACTOR_ORDER)  # 7
NUM_SCALE_SIGNS = 2              # {+1, -1}


class QuantFactorTables(NamedTuple):
    """Static, hardware-derived tables the factored quant head selects over."""

    vocabs: tuple           # per-factor sorted distinct values (python ints)
    vocab_sizes: tuple      # per-factor vocab size (static ints)
    idx: jax.Array          # (M, 7) int32 — each usable dtype as vocab-indices
    to_global: jax.Array    # (M,)  int32 — usable row -> QUANT_DTYPES index


def build_quant_factor_tables(avail_mask=None) -> QuantFactorTables:
    """Build the per-factor vocabularies + index matrix for the USABLE dtypes.

    ``avail_mask`` defaults to the memoized hardware scan; pass an explicit
    ``(NUM_QUANT_DTYPES,)`` mask to build for a hypothetical catalog (tests).
    """
    if avail_mask is None:
        avail_mask = quant_hardware_masks()[0]
    avail = np.asarray(avail_mask) > 0.5
    attrs = np.asarray(QUANT_DTYPE_ATTRS, dtype=np.int64)     # (N, 7)
    rows = attrs[avail]                                       # (M, 7)
    to_global = np.nonzero(avail)[0].astype(np.int32)         # (M,)

    vocabs = tuple(
        tuple(sorted({int(v) for v in rows[:, f].tolist()}))
        for f in FACTOR_ORDER
    )
    vocab_sizes = tuple(len(v) for v in vocabs)

    idx = np.zeros((rows.shape[0], NUM_FACTORS), dtype=np.int32)
    for fi, f in enumerate(FACTOR_ORDER):
        lut = {val: i for i, val in enumerate(vocabs[fi])}
        idx[:, fi] = [lut[int(v)] for v in rows[:, f].tolist()]

    return QuantFactorTables(
        vocabs=vocabs,
        vocab_sizes=vocab_sizes,
        idx=jnp.asarray(idx),
        to_global=jnp.asarray(to_global),
    )


_CACHE: QuantFactorTables | None = None


def quant_factor_tables() -> QuantFactorTables:
    """Memoized :func:`build_quant_factor_tables` on the live hardware scan."""
    global _CACHE
    if _CACHE is None:
        _CACHE = build_quant_factor_tables()
    return _CACHE


def factor_legal_mask(tables: QuantFactorTables, picks: jax.Array, upto: int):
    """``(vocab_sizes[upto],)`` mask of legal vocab-indices for factor ``upto``.

    A value is legal iff at least one usable dtype has it at position ``upto``
    AND matches ``picks[:upto]`` on the earlier factors. ``upto`` is static
    (the sampling loop is unrolled), so the vocab size is a concrete shape.
    """
    idx = tables.idx                                  # (M, 7)
    if upto == 0:
        consistent = jnp.ones(idx.shape[0], dtype=jnp.bool_)
    else:
        consistent = jnp.all(idx[:, :upto] == picks[:upto][None, :], axis=1)
    vf = tables.vocab_sizes[upto]
    onehot = jax.nn.one_hot(idx[:, upto], vf, dtype=jnp.float32)  # (M, vf)
    legal = jnp.sum(onehot * consistent[:, None].astype(jnp.float32), axis=0)
    return (legal > 0.0).astype(jnp.float32)


def resolve_global(tables: QuantFactorTables, picks: jax.Array) -> jax.Array:
    """Completed ``(7,)`` vocab-index tuple -> QUANT_DTYPES global index.

    Exactly one usable row matches (the tuple is a unique key), so ``argmax``
    of the row-match is well-defined.
    """
    match = jnp.all(tables.idx == picks[None, :], axis=1)        # (M,)
    return tables.to_global[jnp.argmax(match)]


def picks_for_global(tables: QuantFactorTables, global_idx: jax.Array) -> jax.Array:
    """QUANT_DTYPES global index -> its ``(7,)`` vocab-index tuple.

    Used at evaluate-time to recover the factor picks from the stored dtype so
    the per-factor log-probs can be re-scored under the current policy.
    """
    match = tables.to_global == global_idx                      # (M,)
    return tables.idx[jnp.argmax(match)]
