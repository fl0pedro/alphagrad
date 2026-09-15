"""The two independent order constraints (owner ruling 2026-09-15).

SPATIAL (``--fixed-order``) is the order among the vertices of ONE time step;
TEMPORAL (``--fixed-temporal-order``) the order across the step copies. Each
can be free or pinned, all nine combinations are legal, and a pinned one is a
PARTIAL order the policy fills in. This file pins:

  * the step tag every vertex carries, read off the named scope graphax emits;
  * that every combination yields a LEGAL COMPLETE plan whose exact gradient is
    ``jax.grad`` (the gradient oracle, on ADALIF_SNN_SHD at window 3);
  * that temporal reverse eliminates the step copies latest first and forward
    earliest first, on the RECORDED order;
  * that the temporal flag raises on a target without time steps.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")

from types import SimpleNamespace as NS                            # noqa: E402

import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402
import pytest                                                      # noqa: E402
from graphax import jacve                                          # noqa: E402

from alphagrad.approx.common import masks as M                     # noqa: E402
from alphagrad.approx.common.order import FIXED_ORDER_CHOICES      # noqa: E402
from alphagrad.approx.common.temporal_order import (               # noqa: E402
    BASE_STEP, FIXED_TEMPORAL_ORDER_CHOICES, build_order_constraint,
    step_tags,
)
from alphagrad.approx.tools.landscape_map import (                 # noqa: E402
    build_env, make_argparser)

#: The window every case here is built at. 3 steps is the smallest number that
#: has a first, a middle and a last copy, so "latest first" and "earliest
#: first" are different orders and neither is the identity.
WINDOW = 3


@pytest.fixture(scope="module")
def adalif_shd():
    """``(env, closed_jaxpr, target_fn, xs, argnums)`` for ADALIF_SNN_SHD."""
    args = make_argparser().parse_args(
        ["--example", "ADALIF_SNN_SHD", "--dataset", "none",
         "--target-grad-window", str(WINDOW),
         "--out-dir", "/tmp/test_temporal_order", "--dry-run"])
    env, _, closed = build_env(args)
    return env, closed


@pytest.fixture(scope="module")
def helmholtz():
    args = make_argparser().parse_args(
        ["--example", "Helmholtz", "--dataset", "none",
         "--out-dir", "/tmp/test_temporal_order_h", "--dry-run"])
    env, _, closed = build_env(args)
    return env, closed


def _walk(env, closed, spatial, temporal):
    """The plan the mask admits: at each step take the LOWEST legal vertex.

    Returns ``(order, constraint)``. The choice among the legal vertices is
    arbitrary on purpose -- the point is that ANY choice the mask admits is a
    complete legal plan, so the least interesting rule is the right one here.
    """
    tbl, oc = build_order_constraint(spatial, temporal, env, "ADALIF_SNN_SHD")
    total_v = len(closed.jaxpr.eqns)
    valid = [int(v) for v in env.valid_vertices]
    static = M.build_vertex_valid_static(valid, total_v)
    chosen = np.zeros(len(valid), dtype=np.int32)
    for k in range(len(valid)):
        st = NS(order=jnp.asarray(chosen),
                step_count=jnp.asarray(k, jnp.int32))
        a = np.asarray(M.vertex_avail_at_step(
            st, static, total_v, len(valid), fixed_order=tbl,
            order_constraint=oc))
        assert a.sum() > 0, (
            f"spatial={spatial} temporal={temporal}: the mask left NO legal "
            f"vertex at step {k} of {len(valid)}")
        chosen[k] = int(np.argmax(a > 0)) + 1
    st = NS(order=jnp.asarray(chosen),
            step_count=jnp.asarray(len(valid), jnp.int32))
    a = np.asarray(M.vertex_avail_at_step(
        st, static, total_v, len(valid), fixed_order=tbl, order_constraint=oc))
    assert int(a.sum()) == 0, "the terminal mask is not all-zero"
    return chosen.tolist(), oc


# ---------------------------------------------------------------------------
# 1. the step tag
# ---------------------------------------------------------------------------

def test_every_equation_is_tagged_with_its_step_or_the_base(adalif_shd):
    _, closed = adalif_shd
    tags = step_tags(closed.jaxpr)
    assert len(tags) == len(closed.jaxpr.eqns)
    steps = sorted({int(t) for t in tags if t != BASE_STEP})
    assert steps == list(range(WINDOW)), steps
    # Every step copy is the SAME block, so all of them have the same size.
    sizes = {s: int((tags == s).sum()) for s in steps}
    assert len(set(sizes.values())) == 1, sizes
    # And the base is what is left: the readout that closes the loss.
    assert int((tags == BASE_STEP).sum()) >= 1


def test_the_graph_is_base_plus_window_times_per_step(adalif_shd):
    """The shape the gradient window buys: one base, N identical blocks."""
    _, closed = adalif_shd
    tags = step_tags(closed.jaxpr)
    per_step = int((tags == 0).sum())
    base = int((tags == BASE_STEP).sum())
    assert len(tags) == base + WINDOW * per_step


# ---------------------------------------------------------------------------
# 2. every combination is a legal complete plan, and its gradient is jax.grad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temporal", FIXED_TEMPORAL_ORDER_CHOICES)
@pytest.mark.parametrize("spatial", FIXED_ORDER_CHOICES)
def test_every_combination_is_a_complete_permutation(adalif_shd, spatial,
                                                     temporal):
    env, closed = adalif_shd
    order, _ = _walk(env, closed, spatial, temporal)
    assert sorted(order) == sorted(int(v) for v in env.valid_vertices)


@pytest.mark.parametrize("temporal", FIXED_TEMPORAL_ORDER_CHOICES)
@pytest.mark.parametrize("spatial", FIXED_ORDER_CHOICES)
def test_every_combination_reproduces_jax_grad(adalif_shd, spatial, temporal):
    """THE GRADIENT ORACLE. A constraint restricts WHICH order is searched; it
    can never change WHAT an exact elimination computes, and this is the check
    that says so for all nine combinations."""
    from alphagrad.approx.common.examples import get_args, get_fn, infer_argnums

    env, closed = adalif_shd
    order, _ = _walk(env, closed, spatial, temporal)
    xs = get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(0), dataset=None,
                  grad_window=WINDOW)
    loss = get_fn("ADALIF_SNN_SHD")
    argnums = infer_argnums("ADALIF_SNN_SHD")
    got = jacve(loss, order, argnums=argnums)(*xs)
    ref = jax.grad(loss, argnums=argnums)(*xs)
    got_leaves = jax.tree_util.tree_leaves(got)
    ref_leaves = jax.tree_util.tree_leaves(ref)
    assert len(got_leaves) == len(ref_leaves)
    for i, (g, r) in enumerate(zip(got_leaves, ref_leaves)):
        scale = float(jnp.max(jnp.abs(r))) or 1.0
        worst = float(jnp.max(jnp.abs(g - r))) / scale
        assert worst < 1e-5, (
            f"spatial={spatial} temporal={temporal} leaf {i}: relative "
            f"residual {worst:.3e} against jax.grad")


# ---------------------------------------------------------------------------
# 3. the temporal direction, read off the RECORDED order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spatial", FIXED_ORDER_CHOICES)
def test_temporal_reverse_eliminates_step_copies_latest_first(adalif_shd,
                                                              spatial):
    env, closed = adalif_shd
    order, oc = _walk(env, closed, spatial, "reverse")
    groups = [int(np.asarray(oc.group_of)[v - 1]) for v in order]
    # The base group is numbered one past the last step, so reverse takes it
    # first; after it the copies run WINDOW-1, ..., 0.
    assert groups == sorted(groups, reverse=True), groups
    assert groups[0] == WINDOW, "the base block is not eliminated first"
    assert groups[-1] == 0, "the earliest step copy is not eliminated last"


@pytest.mark.parametrize("spatial", FIXED_ORDER_CHOICES)
def test_temporal_forward_eliminates_step_copies_earliest_first(adalif_shd,
                                                                spatial):
    env, closed = adalif_shd
    order, oc = _walk(env, closed, spatial, "forward")
    groups = [int(np.asarray(oc.group_of)[v - 1]) for v in order]
    assert groups == sorted(groups), groups
    assert groups[0] == 0, "the earliest step copy is not eliminated first"
    assert groups[-1] == WINDOW, "the base block is not eliminated last"


def test_free_temporal_interleaves_the_copies_under_a_spatial_pin(adalif_shd):
    """The partial order the ruling describes: with the spatial order pinned
    and the temporal one free, each copy keeps its relative order but the
    policy chooses which copy advances -- so a walk is free to interleave."""
    env, closed = adalif_shd
    tbl, oc = build_order_constraint("markowitz", "free", env,
                                     "ADALIF_SNN_SHD")
    assert tbl is None and oc is not None
    total_v = len(closed.jaxpr.eqns)
    valid = [int(v) for v in env.valid_vertices]
    static = M.build_vertex_valid_static(valid, total_v)
    st = NS(order=jnp.zeros(len(valid), jnp.int32),
            step_count=jnp.asarray(0, jnp.int32))
    a = np.asarray(M.vertex_avail_at_step(st, static, total_v, len(valid),
                                          order_constraint=oc))
    # One head per group: WINDOW step copies plus the base.
    assert int(a.sum()) == WINDOW + 1, int(a.sum())


def test_both_pinned_is_a_total_order(adalif_shd):
    env, closed = adalif_shd
    tbl, oc = build_order_constraint("markowitz", "reverse", env,
                                     "ADALIF_SNN_SHD")
    total_v = len(closed.jaxpr.eqns)
    valid = [int(v) for v in env.valid_vertices]
    static = M.build_vertex_valid_static(valid, total_v)
    chosen = np.zeros(len(valid), dtype=np.int32)
    for k in range(len(valid)):
        st = NS(order=jnp.asarray(chosen),
                step_count=jnp.asarray(k, jnp.int32))
        a = np.asarray(M.vertex_avail_at_step(
            st, static, total_v, len(valid), order_constraint=oc))
        assert int(a.sum()) == 1, (k, int(a.sum()))
        chosen[k] = int(np.argmax(a)) + 1


def test_both_free_is_no_mask(adalif_shd):
    env, _ = adalif_shd
    tbl, oc = build_order_constraint("free", "free", env, "ADALIF_SNN_SHD")
    assert tbl is None and oc is None


# ---------------------------------------------------------------------------
# 4. the flags refuse a target that has no time steps
# ---------------------------------------------------------------------------

def test_temporal_order_raises_on_a_target_without_time_steps(helmholtz):
    env, _ = helmholtz
    for temporal in ("reverse", "forward"):
        with pytest.raises(ValueError) as ei:
            build_order_constraint("markowitz", temporal, env, "Helmholtz")
        assert "has NO time steps" in str(ei.value)


def test_a_non_temporal_target_keeps_the_ticket_64_table(helmholtz):
    """Nothing about the campaign arms moves: with one copy and a free
    temporal order the builder hands back the SAME static table
    ``common/order.py`` has produced since 2026-09-05."""
    from alphagrad.approx.common.order import fixed_order_for_env

    env, _ = helmholtz
    tbl, oc = build_order_constraint("markowitz", "free", env, "Helmholtz")
    assert oc is None
    assert tbl.tolist() == fixed_order_for_env("markowitz", env).tolist()
    tbl, oc = build_order_constraint("free", "free", env, "Helmholtz")
    assert tbl is None and oc is None


def test_the_two_pins_are_never_passed_together():
    static = M.build_vertex_valid_static([1, 2], 3)
    st = NS(order=jnp.zeros(2, jnp.int32), step_count=jnp.asarray(0, jnp.int32))
    with pytest.raises(ValueError) as ei:
        M.vertex_avail_at_step(st, static, 3, 2,
                               fixed_order=np.array([2, 1], np.int32),
                               order_constraint=object())
    assert "never both" in str(ei.value)


def test_the_choices_are_the_three_the_ruling_names():
    assert FIXED_TEMPORAL_ORDER_CHOICES == ("free", "reverse", "forward")
    assert "markowitz" not in FIXED_TEMPORAL_ORDER_CHOICES
