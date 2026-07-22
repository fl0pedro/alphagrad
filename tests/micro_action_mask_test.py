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


# ---------------------------------------------------------------------------
# Head-side consumption of the mask
# ---------------------------------------------------------------------------

from alphagrad.approx.heads import (  # noqa: E402
    AXIS_TAG_BITS, AxisTokenFeatures, _compute_axis_masks)


def _feats(n, valid, compressed=None, in_diag=None):
    tb = (jnp.zeros((n, AXIS_TAG_BITS))
          .at[:, 1].set(jnp.array(compressed or [0.0] * n))
          .at[:, 2].set(jnp.array(in_diag or [0.0] * n)))
    return AxisTokenFeatures(
        size=jnp.ones(n) * 4, log_size=jnp.log(jnp.ones(n) * 4), tag_bits=tb,
        group_id=jnp.zeros(n, dtype=jnp.int32), valid_mask=jnp.array(valid))


def test_an_axis_with_no_legal_partner_is_not_diag_selectable():
    """A lone eligible axis has no partner once ``i != j`` is applied. If it
    stayed selectable the j-head would receive an all -1e9 logit vector, whose
    softmax is UNIFORM over illegal axes -- so a legal-looking sample lands on
    an illegal pair instead of being rejected."""
    diag_i, _, j_mask, _ = _compute_axis_masks(_feats(4, [1., 0., 0., 0.]))
    assert float(j_mask[0].sum()) == 0.0, "the lone axis has no partner"
    assert not bool((diag_i > 0).any()), "so it must not be DIAG-selectable"


def test_two_free_axes_point_at_each_other():
    diag_i, _, j_mask, _ = _compute_axis_masks(_feats(4, [1., 1., 0., 0.]))
    assert diag_i.tolist() == [1., 1., 0., 0.]
    assert j_mask[0].tolist() == [0., 1., 0., 0.], "i != j, partner is axis 1"


def test_env_pair_mask_is_authoritative():
    """The tag-bit reconstruction cannot see the out/primal split, so when the
    env supplies the exact per-edge mask it must win -- including removing an
    axis's last partner, which then removes the axis itself."""
    feats = _feats(4, [1., 1., 0., 0.])
    pair_valid = jnp.ones((4, 4)).at[0, 1].set(0.0)
    diag_i, _, j_mask, _ = _compute_axis_masks(feats, pair_valid=pair_valid)
    assert float(j_mask[0].sum()) == 0.0, "axis 0 lost its only partner"
    assert diag_i.tolist() == [0., 1., 0., 0.], "and so is no longer selectable"


# ---------------------------------------------------------------------------
# Factor space: which `factor` values are legal for a given legal pair
# ---------------------------------------------------------------------------

from alphagrad.approx.common.masks import diag_pair_factor_space  # noqa: E402


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _factors_graphax_accepts(st, i, j, max_f=9):
    out = set()
    for f in range(1, max_f + 1):
        try:
            apply_diag(st, Diag(i, j, f))
            out.add(f)
        except Exception:
            pass
    return out


def _square():
    return SparseTensor([DenseIndex(0, 4, 0), DenseIndex(1, 4, 1)],
                        [DenseIndex(2, 4, 2), DenseIndex(3, 4, 3)],
                        jnp.ones((4, 4, 4, 4)))


def test_free_pair_factor_space_is_the_divisors_of_the_gcd():
    st = _square()
    base, span = diag_pair_factor_space(st, 0, 2)
    assert base == 1 and span == diag_pair_gcd(st, 0, 2)
    predicted = {base * d for d in _divisors(span)}
    assert predicted == _factors_graphax_accepts(st, 0, 2)


def test_coupled_pair_may_only_subdivide_never_coarsen():
    """A second DIAG on an already-coupled pair must be a MULTIPLE of the
    current meta count. diag_pair_gcd alone would still offer factor=1 here,
    which graphax rejects -- that is what diag_pair_factor_space exists for."""
    st = apply_diag(_square(), Diag(0, 2, 2))
    assert st.out_dims[0].size == 2, "meta count after the first Diag"

    base, span = diag_pair_factor_space(st, 0, 2)
    assert base == 2, "the current meta count is the floor, not 1"
    predicted = {base * d for d in _divisors(span)}
    accepted = _factors_graphax_accepts(st, 0, 2)
    assert predicted == accepted
    assert 1 not in accepted, "coarsening a coupled pair is rejected"
    assert 1 in _divisors(diag_pair_gcd(st, 0, 2)), (
        "and the plain gcd WOULD have offered it -- the reason this helper exists")


def test_untouched_pair_stays_free_after_a_diag_elsewhere():
    st = apply_diag(_square(), Diag(0, 2, 2))
    assert diag_pair_factor_space(st, 1, 3) == (1, 4)


def test_factor_space_rejects_out_of_range():
    assert diag_pair_factor_space(_square(), 0, 99) == (0, 0)


def test_pair_rules_are_unconditional():
    """The block-structure pair rules used to sit behind
    ALPHAGRAD_MICRO_PAIR_MASKS, defaulting OFF -- so the shipped default
    disagreed with the executor: coupled axes were barred from DIAG entirely
    (no re-diagonalisation, which graphax supports out of the box) and
    i_coupled was pinned to zero, so the j-head contributed log-prob for a
    choice it never really had. No env var now; the rules always apply.

    Note -1 is the ungrouped sentinel (env.py stamps it); group 0 is a REAL
    group, so free axes must carry -1, not 0.
    """
    n = 4
    tb = jnp.zeros((n, AXIS_TAG_BITS)).at[:, 2].set(jnp.array([1., 1., 0., 0.]))
    feats = AxisTokenFeatures(
        size=jnp.ones(n) * 4, log_size=jnp.log(jnp.ones(n) * 4), tag_bits=tb,
        group_id=jnp.array([0, 0, -1, -1], dtype=jnp.int32),
        valid_mask=jnp.ones(n))
    diag_i, _, j_mask, i_coupled = _compute_axis_masks(feats)

    assert i_coupled.tolist() == [1., 1., 0., 0.], "only the real group is coupled"
    assert diag_i.tolist() == [1., 1., 1., 1.], "coupled axes may re-diagonalise"
    assert j_mask[0].tolist() == [0., 1., 0., 0.], "a coupled i is forced to its partner"
    assert j_mask[2].tolist() == [0., 0., 0., 1.], "a free i may only take a free j"


def test_group_zero_is_a_real_group_not_the_ungrouped_sentinel():
    """Guards the trap: -1 means ungrouped, 0 is the FIRST allocated group
    (env.py derives it as max(gid) + 1 from the -1 default)."""
    n = 2
    tb = jnp.zeros((n, AXIS_TAG_BITS)).at[:, 2].set(jnp.ones(n))
    feats = AxisTokenFeatures(
        size=jnp.ones(n) * 4, log_size=jnp.log(jnp.ones(n) * 4), tag_bits=tb,
        group_id=jnp.zeros(n, dtype=jnp.int32), valid_mask=jnp.ones(n))
    _, _, _, i_coupled = _compute_axis_masks(feats)
    assert i_coupled.tolist() == [1., 1.], "two axes in group 0 ARE partners"
