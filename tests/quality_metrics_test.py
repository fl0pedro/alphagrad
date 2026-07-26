"""Honest quality metrics (Phase A2): degenerate reads WORST, layouts align,
complex cosine is real, aggregation is a true winsorized mean, latency floor.

The old ``_quality_metrics`` returned (1.0, 0.0) — perfect — for empty /
mismatched / zero-size Jacobians, which as a reward was a degeneracy backdoor
(a shape-destroying plan out-scored every honest approximation). These pin the
fixed semantics.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import (
    _LAT_FLOOR_NS,
    _aggregate_samples,
    _align_jac,
    _quality_metrics,
    _winsorized_mean,
)


def test_degenerate_reads_worst_not_perfect():
    ex = [jnp.ones((3, 4))]
    # no leaves
    cos, frob = _quality_metrics(ex, [])
    assert float(cos) == 0.0 and float(frob) == 1.0
    # zero size
    cos, frob = _quality_metrics([jnp.ones((0,))], [jnp.ones((0,))])
    assert float(cos) == 0.0 and float(frob) == 1.0
    # mismatched (non-transposable) shape
    cos, frob = _quality_metrics(ex, [jnp.ones((5, 5))])
    assert float(cos) == 0.0 and float(frob) == 1.0


def test_exact_match_is_perfect():
    ex = [jnp.arange(12.0).reshape(3, 4) + 1.0]
    cos, frob = _quality_metrics(ex, [ex[0]])
    assert float(cos) == pytest.approx(1.0, abs=1e-6)
    assert float(frob) == pytest.approx(0.0, abs=1e-6)


def test_transposed_leaf_is_aligned_not_orthogonalised():
    """A (4,3) grad delivered as (3,4)^T must compare as identical, not as a
    near-orthogonal ravel."""
    w = jnp.arange(12.0).reshape(4, 3) + 1.0
    cos_t, frob_t = _quality_metrics([w], [w.T])
    assert float(cos_t) == pytest.approx(1.0, abs=1e-6)
    assert float(frob_t) == pytest.approx(0.0, abs=1e-6)
    # and _align_jac leaves genuinely-different-shape leaves alone
    out = _align_jac([jnp.ones((2, 5))], [jnp.ones((4, 3))])
    assert out[0].shape == (2, 5)


def test_complex_cosine_is_real():
    ex = [jnp.ones((4,), dtype=jnp.float32)]
    ap = [jnp.ones((4,), dtype=jnp.complex64) * (1.0 + 0.1j)]
    cos, frob = _quality_metrics(ex, ap)
    assert not jnp.iscomplexobj(cos), "cosine must be real-valued"


def test_winsorized_mean_kills_both_tails():
    vals = jnp.asarray([0.0, 10.0, 11.0, 12.0, 13.0, 1000.0], dtype=jnp.float32)
    wm = float(_winsorized_mean(vals))
    assert 9.0 < wm < 15.0, f"outliers must not dominate: {wm}"
    # aggregation path uses it for >=4 samples
    agg = float(_aggregate_samples(list(np.asarray(vals)), want_top_quartile=True))
    assert 9.0 < agg < 15.0


def test_latency_floor_constant_sane():
    assert _LAT_FLOOR_NS > 0.0
