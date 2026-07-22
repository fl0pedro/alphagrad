"""The per-edge micro-action masks must never admit an action graphax rejects.

graphax's typed transform API fails LOUDLY on a non-fitting micro-action and
tells callers to mask invalid actions up front. These tests pin that the mask
in ``alphagrad.approx.common.masks`` is the exact complement of that failure:

* SAFETY (the property that matters): ``mask[i, j]`` True => ``apply_*``
  succeeds. A false positive is a hard crash mid-episode.
* EXACTNESS on tensors whose dims are all physical: the mask is not merely
  safe but tight, so the policy is not needlessly starved of legal actions.
"""
import itertools

import jax.numpy as jnp
import pytest
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.micro_actions import (
    Compress, Diag, apply_compress, apply_diag)
from graphax.sparse.tensor import SparseTensor

from alphagrad.approx.common.masks import (
    compress_valid_mask, diag_pair_gcd, diag_valid_mask)

MAX_DIMS = 5


def _dense2x2():
    return SparseTensor([DenseIndex(0, 4, 0), DenseIndex(1, 6, 1)],
                        [DenseIndex(2, 4, 2), DenseIndex(3, 6, 3)],
                        jnp.ones((4, 6, 4, 6)))


def _cases():
    yield "dense2x2", _dense2x2()
    # out0 <-> primal0 already paired; the other two dims are free
    yield "sparse_pair", SparseTensor(
        [DiagonalIndex(0, 4, 0, 2), DenseIndex(1, 6, 1)],
        [DiagonalIndex(2, 4, 0, 0), DenseIndex(3, 6, 2)], jnp.ones((4, 6, 6)))
    # the Helmholtz-shaped edge: an out axis, no primal axis at all
    yield "out_only", SparseTensor([DenseIndex(0, 4, 0)], [], jnp.ones((4,)))
    # pure-structure Jacobian (e.g. a constant -1): nothing materialised
    yield "no_val", SparseTensor([DenseIndex(0, 4, None)],
                                 [DenseIndex(1, 4, None)], None)
    # THE composed case the per-face policy creates: a Compress earlier in the
    # sub-episode dropped a physical axis, so that dim is now implicit and can
    # no longer take part in a Diag.
    yield "post_compress", apply_compress(_dense2x2(),
                                          Compress(axes=(1,), kind="mean"))


CASES = list(_cases())
IDS = [c[0] for c in CASES]


def _diag_applies(st, i, j):
    g = diag_pair_gcd(st, i, j)
    try:
        apply_diag(st, Diag(i, j, g if g > 1 else 1))
        return True
    except Exception:
        return False


@pytest.mark.parametrize("name,st", CASES, ids=IDS)
def test_diag_mask_never_admits_a_rejected_action(name, st):
    mask = diag_valid_mask(st, MAX_DIMS)
    n = len(st.out_dims) + len(st.primal_dims)
    for i, j in itertools.product(range(MAX_DIMS), range(MAX_DIMS)):
        if not mask[i, j]:
            continue
        assert i < n and j < n and i != j, f"{name}: mask admits out-of-range ({i},{j})"
        assert _diag_applies(st, i, j), (
            f"{name}: mask admits Diag({i},{j}) but graphax rejects it")


@pytest.mark.parametrize("name,st", CASES, ids=IDS)
def test_diag_mask_is_tight_when_all_dims_are_physical(name, st):
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    if st.val is None or any(d.axis is None for d in dims):
        pytest.skip("tightness is only claimed for all-physical tensors")
    mask = diag_valid_mask(st, MAX_DIMS)
    for i, j in itertools.product(range(len(dims)), range(len(dims))):
        assert bool(mask[i, j]) == _diag_applies(st, i, j), (
            f"{name}: mask disagrees with graphax on Diag({i},{j})")


@pytest.mark.parametrize("name,st", CASES, ids=IDS)
def test_compress_mask_never_admits_a_rejected_axis(name, st):
    mask = compress_valid_mask(st, MAX_DIMS)
    for a in range(MAX_DIMS):
        if not mask[a]:
            continue
        apply_compress(st, Compress(axes=(a,), kind="mean"))  # must not raise


def test_compress_mask_is_exactly_the_physical_axes():
    st = _dense2x2()
    mask = compress_valid_mask(st, MAX_DIMS)
    assert list(mask) == [True, True, True, True, False], (
        "val.ndim - 1 is the highest legal axis, counting from 0")


def test_diag_requires_the_out_primal_split():
    """out<->out and primal<->primal are not diagonals and must be masked."""
    mask = diag_valid_mask(_dense2x2(), MAX_DIMS)
    assert not mask[0, 1] and not mask[1, 0], "out<->out must be masked"
    assert not mask[2, 3] and not mask[3, 2], "primal<->primal must be masked"
    assert mask[0, 2] and mask[2, 0], "out<->primal is the legal direction"


def test_sparse_dim_may_only_pair_with_its_partner():
    st = SparseTensor([DiagonalIndex(0, 4, 0, 2), DenseIndex(1, 6, 1)],
                      [DiagonalIndex(2, 4, 0, 0), DenseIndex(3, 6, 2)],
                      jnp.ones((4, 6, 6)))
    mask = diag_valid_mask(st, MAX_DIMS)
    assert mask[0, 2], "an already-paired dim keeps its own partner"
    assert not mask[0, 3], "a paired dim may not take a different partner"
    assert mask[1, 3], "two free dense dims may always be paired"


def test_compress_then_diag_drops_the_implicit_dim():
    """The composed lhs/rhs/res case: once Compress drops a physical axis, no
    Diag may touch that dim again."""
    st = apply_compress(_dense2x2(), Compress(axes=(1,), kind="mean"))
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    implicit = [k for k, d in enumerate(dims) if d.axis is None]
    assert implicit, "the Compress should have made a dim implicit"
    mask = diag_valid_mask(st, MAX_DIMS)
    for k in implicit:
        assert not mask[k].any() and not mask[:, k].any(), (
            f"dim {k} is implicit and must be masked out of every Diag pair")


def test_diag_pair_gcd_bounds_the_factor():
    st = _dense2x2()
    assert diag_pair_gcd(st, 0, 2) == 4      # 4 vs 4
    assert diag_pair_gcd(st, 0, 3) == 2      # gcd(4, 6)
    assert diag_pair_gcd(st, 0, 99) == 0     # out of range
