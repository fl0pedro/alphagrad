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
from graphax.sparse.indexes import DenseIndex
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
