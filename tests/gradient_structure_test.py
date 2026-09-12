"""Exact structure equality in the quality comparisons (ticket dsnn-3qm.62).

``_quality_metrics`` (grad-cosine) and ``_residual_scores`` (fidelity) RAISE
``GradientStructureMismatch`` on a leaf-count, logical-shape or layout
mismatch instead of the silent ``(0.0, 1.0)`` clamp that hid finding 60. A
dead path (None) is a zero gradient. A Reduce'd gradient (an implicit dim) is
compared analytically and equals the broadcast comparison. ``_align_jac`` is
the jac_cosine path only and raises when the trees cannot be mapped.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import (
    GradientStructureMismatch, _align_jac, _quality_metrics, _residual_scores,
    grad_oracle)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor


def _st(val, dims_sizes, implicit=()):
    """A gradient SparseTensor in parameter layout; ``implicit`` positions
    store one representative (axis None)."""
    dims = []
    ax = 0
    for pos, n in enumerate(dims_sizes):
        if pos in implicit:
            dims.append(DenseIndex(pos, n, axis=None))
        else:
            dims.append(DenseIndex(pos, n, axis=ax))
            ax += 1
    return SparseTensor((), tuple(dims), val, check_consistency=False)


def _cos(e, a):
    e = np.asarray(e, np.float64).ravel()
    a = np.asarray(a, np.float64).ravel()
    return float(e @ a / (np.linalg.norm(e) * np.linalg.norm(a)))


W = jnp.arange(12.0).reshape(4, 3) + 1.0
B = jnp.arange(3.0) + 1.0
# SQUARE and asymmetric. The (4, 3) W above cannot exercise the layout
# predicate on its own: a transposed (4, 3) leaf is caught by the downstream
# shape check whether the predicate works or not (measured -- see
# test_a_SQUARE_transposed_leaf_is_caught_by_the_LAYOUT_guard_alone).
S = jnp.arange(16.0).reshape(4, 4) + 1.0


def test_exact_match_is_perfect():
    cos, frob = _quality_metrics([W, B], [W, B])
    assert float(cos) == pytest.approx(1.0, abs=1e-6)
    assert float(frob) == pytest.approx(0.0, abs=1e-6)


def test_transposed_leaf_raises_instead_of_clamping():
    with pytest.raises(GradientStructureMismatch):
        _quality_metrics([W, B], [W.T, B])
    with pytest.raises(GradientStructureMismatch):
        _residual_scores([W, B], [W.T, B], has_aux=False)


def test_leaf_count_mismatch_raises():
    with pytest.raises(GradientStructureMismatch):
        _quality_metrics([W, B], [W])


def test_dead_path_is_a_zero_gradient_not_a_fault():
    cos, frob = _quality_metrics([W, B], [W, None])
    expect = _cos(np.concatenate([np.ravel(W), np.ravel(B)]),
                  np.concatenate([np.ravel(W), np.zeros(3)]))
    assert float(cos) == pytest.approx(expect, abs=1e-6)
    assert 0.0 < float(frob) < 1.0


def test_sparse_leaf_in_parameter_layout_compares_like_an_array():
    a = _st(W * 0.5, (4, 3))
    cos, frob = _quality_metrics([W, B], [a, B])
    assert float(cos) == pytest.approx(_cos(np.concatenate([np.ravel(W), B]),
                                            np.concatenate([np.ravel(W) * 0.5, B])),
                                       abs=1e-6)


def test_sparse_leaf_not_in_parameter_layout_raises():
    bad = SparseTensor((), (DenseIndex(0, 4, axis=1), DenseIndex(1, 3, axis=0)),
                       jnp.asarray(W.T), check_consistency=False)
    with pytest.raises(GradientStructureMismatch):
        _quality_metrics([W, B], [bad, B])


def test_a_SQUARE_transposed_leaf_is_caught_by_the_LAYOUT_guard_alone():
    """The case where ``is_parameter_layout`` is the ONLY guard.

    ``test_sparse_leaf_not_in_parameter_layout_raises`` above is
    over-determined: its leaf is logically (4, 3) stored (3, 4), so the
    downstream shape check raises even with the predicate stubbed to return
    ``True`` -- the test is green whether the predicate works or not.

    For a SQUARE leaf no shape check can fire: a transposed (4, 4) buffer has
    the same shape and the same logical extents. Measured 2026-09-12 (job
    65006, pgi15-cpu2) with the predicate stubbed: NO RAISE, cosine 0.8807947,
    i.e. finding 60's failure mode again -- silently wrong instead of silently
    clamped. Measured on graphax cdabc9e AND on wip/t71-nominal-20260910
    (31df3e8, ticket .71's "nominal dim order" assert): byte-identical, because
    .71 asserts on LOGICAL EXTENTS and a square transposition leaves those
    alone. So this is the only test that pins the predicate, and the match=
    pins WHICH guard raised.
    """
    bad = SparseTensor((), (DenseIndex(0, 4, axis=1), DenseIndex(1, 4, axis=0)),
                       jnp.asarray(S.T), check_consistency=False)
    assert bad.shape == (4, 4)          # no shape check CAN fire here
    with pytest.raises(GradientStructureMismatch,
                       match="not in parameter layout"):
        _quality_metrics([S, B], [bad, B])
    with pytest.raises(GradientStructureMismatch,
                       match="not in parameter layout"):
        _residual_scores([S, B], [bad, B], has_aux=False)


def test_the_square_fixture_in_parameter_layout_still_compares():
    """The negative control: the same square leaf stored correctly scores 1.0,
    so the test above is pinning the LAYOUT and not the squareness."""
    good = _st(jnp.asarray(S), (4, 4))
    cos, frob = _quality_metrics([S, B], [good, B])
    assert float(cos) == pytest.approx(1.0, abs=1e-6)
    assert float(frob) == pytest.approx(0.0, abs=1e-6)


def test_implicit_dim_is_compared_analytically_without_densify():
    """A Reduce(mean) over axis 0 stores one row; the logical gradient is that
    row broadcast over the 4 rows. The analytic accumulator must equal the
    comparison against the materialized broadcast."""
    rep = jnp.mean(W, axis=0)                        # (3,)
    a = _st(rep, (4, 3), implicit=(0,))
    cos, frob = _quality_metrics([W, B], [a, B])
    dense_a = jnp.broadcast_to(rep, (4, 3))
    cos_d, frob_d = _quality_metrics([W, B], [dense_a, B])
    assert float(cos) == pytest.approx(float(cos_d), abs=1e-5)
    assert float(frob) == pytest.approx(float(frob_d), abs=1e-5)
    rf, c = _residual_scores([W, B], [a, B], has_aux=False)
    assert c == pytest.approx(float(cos_d), abs=1e-5)


def test_logical_shape_mismatch_of_a_sparse_leaf_raises():
    a = _st(jnp.ones((3, 4)), (3, 4))
    with pytest.raises(GradientStructureMismatch):
        _quality_metrics([W, B], [a, B])


def test_jac_cosine_path_still_aligns_a_transposed_block():
    cos, frob = _quality_metrics([W], [W.T], align=True, site="jac_cosine")
    assert float(cos) == pytest.approx(1.0, abs=1e-6)
    assert float(frob) == pytest.approx(0.0, abs=1e-6)


def test_align_jac_raises_when_the_trees_cannot_be_mapped():
    with pytest.raises(GradientStructureMismatch):
        _align_jac([W], [W, B])


def test_grad_oracle_default_is_reference(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_GRAD_ORACLE", raising=False)
    assert grad_oracle() == "reference"
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE", "off")
    assert grad_oracle() == "off"
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE", "sometimes")
    with pytest.raises(ValueError):
        grad_oracle()


def test_the_comparison_NEVER_MATERIALIZES_either_side():
    """Owner ruling (c), 2026-09-12: "the comparison should remain lazy, should
    not need to materialize, and we definitely don't want to densify."

    Counted, not read: ``SparseTensor.dense`` is wrapped and must not be called
    once, for any structural kind that reaches the comparison -- a diagonal
    pair (what a surviving Diag stores), a uniform leaf (``val is None``), an
    implicit dim (a Reduce'd gradient) and a materialized leaf. Before this
    change the first two were ``a.dense()`` and a structured EXACT leaf was
    ``e.dense()``; on a 1024x1024 logical pair storing 1024 values that was a
    4 MiB allocation to produce four scalars, on the path that runs for EVERY
    terminal measurement.
    """
    calls = []
    real = SparseTensor.dense

    def spy(self, **kw):
        calls.append(tuple(int(d.logical_size) for d in self.dims))
        return real(self, **kw)

    pair_o = DiagonalIndex(0, 4, axis=0, other_id=1, block_size=1, block_axis=1)
    pair_i = DiagonalIndex(1, 4, axis=0, other_id=0, block_size=1, block_axis=2)
    diag = SparseTensor((pair_o,), (pair_i,),
                        jnp.arange(4.0).reshape(4, 1, 1) + 1.0,
                        check_consistency=False)
    uniform = SparseTensor((), (DenseIndex(0, 4, axis=None),
                               DenseIndex(1, 4, axis=None)),
                           None, scalar_mult=jnp.asarray(0.0), check_consistency=False)
    cases = [
        ([S, B], [diag, B]),
        ([S, B], [uniform, B]),
        ([W, B], [_st(jnp.mean(W, axis=0), (4, 3), implicit=(0,)), B]),
        ([W, B], [_st(W * 0.5, (4, 3)), B]),
        ([S, B], [None, B]),                       # the dead path
        ([_st(W * 1.0, (4, 3)), B], [W * 0.5, B]),  # a SPARSE exact leaf
    ]
    SparseTensor.dense = spy
    try:
        for exact, approx in cases:
            cos, frob = _quality_metrics(exact, approx)
            assert np.isfinite(float(cos)) and np.isfinite(float(frob))
            _residual_scores(exact, approx, has_aux=False)
    finally:
        SparseTensor.dense = real
    assert calls == [], f"the comparison densified: {calls}"


def test_a_diagonal_pair_compares_exactly_like_its_dense_form():
    """The lazy contraction is not merely cheap, it is the same number."""
    pair_o = DiagonalIndex(0, 3, axis=0, other_id=1, block_size=2, block_axis=1)
    pair_i = DiagonalIndex(1, 3, axis=0, other_id=0, block_size=2, block_axis=2)
    val = jnp.asarray(np.arange(12.0).reshape(3, 2, 2) - 5.0, jnp.float32)
    diag = SparseTensor((pair_o,), (pair_i,), val, scalar_mult=jnp.asarray(0.5),
                        check_consistency=False)
    exact = jnp.asarray(np.linspace(-1.0, 2.0, 36, dtype=np.float32).reshape(6, 6))
    lazy_cos, lazy_frob = _quality_metrics([exact], [diag])
    dense_cos, dense_frob = _quality_metrics([exact], [jnp.asarray(diag.dense())])
    assert float(lazy_cos) == pytest.approx(float(dense_cos), abs=1e-5)
    assert float(lazy_frob) == pytest.approx(float(dense_frob), abs=1e-5)


def test_two_incompatible_structures_raise_rather_than_densifying():
    """Neither side materialized and the structures differ: there is no common
    compact frame. The comparison must say so, not quietly densify one side."""
    pair_o = DiagonalIndex(0, 4, axis=0, other_id=1, block_size=1, block_axis=1)
    pair_i = DiagonalIndex(1, 4, axis=0, other_id=0, block_size=1, block_axis=2)
    diag = SparseTensor((pair_o,), (pair_i,), jnp.ones((4, 1, 1)),
                        check_consistency=False)
    implicit = _st(jnp.arange(4.0) + 1.0, (4, 4), implicit=(0,))
    with pytest.raises(GradientStructureMismatch, match="without materializing"):
        _quality_metrics([diag], [implicit])
