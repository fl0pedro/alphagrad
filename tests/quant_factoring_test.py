"""The factored-quant vocabularies must be safe and complete.

* SAFETY: any chain of per-factor *legal* picks resolves to a real usable dtype
  — the head can never assemble an unreachable ``(kind,bits,exp,...)`` tuple.
* COMPLETENESS: every usable dtype is reachable and round-trips
  ``global -> picks -> global``.
"""
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.quant_factoring import (
    NUM_FACTORS, build_quant_factor_tables, factor_legal_mask, picks_for_global,
    resolve_global)
from graphax.sparse.micro_actions import QUANT_DTYPES


def _tables():
    return build_quant_factor_tables()


def test_every_usable_dtype_round_trips_picks_to_global():
    t = _tables()
    for gi in np.asarray(t.to_global).tolist():
        picks = picks_for_global(t, jnp.int32(gi))
        assert int(resolve_global(t, picks)) == gi, QUANT_DTYPES[gi]


def test_legal_pick_chains_always_reach_a_real_dtype():
    t = _tables()
    usable = {tuple(r) for r in np.asarray(t.idx).tolist()}
    reached = set()

    def walk(prefix):
        f = len(prefix)
        if f == NUM_FACTORS:
            tup = tuple(prefix)
            assert tup in usable, f"legal chain {tup} is not a real dtype"
            reached.add(tup)
            return
        picks = jnp.asarray(prefix + [0] * (NUM_FACTORS - f), dtype=jnp.int32)
        legal = np.asarray(factor_legal_mask(t, picks, f))
        assert legal.sum() > 0, f"dead-end at factor {f} after {prefix}"
        for v in np.nonzero(legal)[0].tolist():
            walk(prefix + [v])

    walk([])
    # COMPLETENESS: the legal-pick tree reaches exactly every usable dtype.
    assert reached == usable


def test_kind_factor_offers_every_available_kind():
    t = _tables()
    legal = np.asarray(
        factor_legal_mask(t, jnp.zeros(NUM_FACTORS, jnp.int32), 0))
    assert int(legal.sum()) == len(t.vocabs[0])  # all present kinds are legal


def test_int_kind_forces_the_float_only_factors():
    """Under an integer dtype, exp/mantissa/bias/finite/uz each have exactly one
    legal value (ints are 0/1 there) — so the ≥2-gate skips those heads."""
    t = _tables()
    if "int8" not in QUANT_DTYPES:
        return
    gi = QUANT_DTYPES.index("int8")
    if gi not in np.asarray(t.to_global).tolist():
        return
    picks = picks_for_global(t, jnp.int32(gi))  # int8's real vocab-index tuple
    for f in range(2, NUM_FACTORS):             # exp, mantissa, bias, finite, uz
        legal = np.asarray(factor_legal_mask(t, picks, f))
        assert legal.sum() == 1, f"int factor {f} not forced: {legal}"
