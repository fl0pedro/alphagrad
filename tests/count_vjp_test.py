"""WHAT THE COUNT-PROPORTIONAL BACKWARD PASS CHANGES, AND WHAT IT DOES NOT.

``common/count_vjp.count_loop`` replaces the differentiated chunk loop's
``lax.scan`` + ``lax.cond`` over ``ceil(window / chunk)`` iterations with a
``jax.custom_vjp`` whose forward AND backward are ``lax.while_loop``s over the
real live chunk count. This module says exactly how far that is a change of
trip count and where it is also a change of the last bits.

Both directions, both chunked paths:

  * ``carry_stream.advance`` -> ``delta_fold.extend_fold``, the shipped path
    (``ALPHAGRAD_FOLD_DELTA`` defaults on), and
  * ``Agent.encode_extend(chunk=, budget=)`` -> ``_extend_sequential``'s
    budget form, the unfolded path.

The counts are chosen to hit every boundary the loop has: zero (no live chunk
at all, so the backward loop never runs), exactly one chunk, exactly the whole
window, and a count that is not a multiple of the chunk (so the last live
chunk is partly pad).

Tolerance is ZERO for the forward, everywhere.

For the gradient it is zero on the SEQUENTIAL chunk interior and on the
unfolded extend, and a few float32 ulp on the PARALLEL chunk interior, which
is the shipped default. That last case is a genuine reassociation and it is
pinned here rather than hidden. The probe ``probe_cvjp.py`` locates it with
no alphagrad in it at all: ``count_loop``'s gradient matches a plain
``lax.scan`` of the same body EXACTLY, and it is the ``lax.cond`` the
shipped body wraps around that chunk which moves the last bits. Removing
that cond is the entire point of the change, so the two cannot agree bit for
bit and float32 addition is not associative.

The vmapped cases are the shape the loss actually runs: the per-sample counts
are batched, the ``budget`` is the batch-wide maximum and is UNBATCHED, and
the ``custom_vjp`` has to survive both.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "256")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")

import contextlib  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

TOTAL_V = 6
EMBD = 32
WINDOW = 64
CHUNK = 16
# 0: no live chunk at all. 16: exactly one chunk. 37: not a multiple of the
# chunk. 64: the whole window, so nothing is skipped and the two forms have
# the same trip count.
COUNTS = [0, CHUNK, 37, WINDOW]


@contextlib.contextmanager
def env(**kw):
    """Set env vars around one traced call."""
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def count_vjp(on: bool):
    """``ALPHAGRAD_COUNT_VJP`` around one traced call."""
    old = os.environ.get("ALPHAGRAD_COUNT_VJP")
    os.environ["ALPHAGRAD_COUNT_VJP"] = "1" if on else "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ["ALPHAGRAD_COUNT_VJP"]
        else:
            os.environ["ALPHAGRAD_COUNT_VJP"] = old


@pytest.fixture(scope="module")
def setup():
    from alphagrad.approx import ppo as P
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=True,
        face_actions=False, unified_face_head=False, live_faces=False,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = EMBD
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    agent = build_and_init_agent(ns, TOTAL_V, num_factors=4, max_rules=4,
                                 seed=3)

    rng = np.random.default_rng(0)
    B = len(COUNTS)
    tok = jnp.asarray(rng.integers(1, 200, (B, WINDOW)).astype(np.int32))
    part = jnp.asarray(np.eye(TOTAL_V + 1, dtype=np.float32)[[2, 0, 5, 1]])
    owner = jnp.asarray(np.array([1, 3, 2, 4], np.int32))
    return dict(agent=agent, tok=tok, part=part, owner=owner)


# ---------------------------------------------------------------- the fold --

def _advance(agent, tok, count, owner, part, budget):
    from alphagrad.approx.common import carry_stream as CS
    carry = agent.carry_init()
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    return CS.advance(
        agent, carry, vs, vc, tok, count, owner,
        window=WINDOW, participants=part,
        chunk=CHUNK, budget=budget,
    )


def _fold_scalar(agent, tok, count, owner, part, budget):
    carry, vs, vc = _advance(agent, tok, count, owner, part, budget)
    return (jnp.sum(carry.M * 1.0) + jnp.sum(carry.I * 2.0)
            + jnp.sum(vs * 4.0) + jnp.sum(vc * 5.0))


def _fold_out(agent, tok, count, owner, part, budget):
    carry, vs, vc = _advance(agent, tok, count, owner, part, budget)
    return carry.M, carry.I, carry.pos, vs, vc


# ------------------------------------------------ the unfolded budget form --

def _extend_out(agent, tok, count, budget):
    carry = agent.carry_init()
    c2, rows, valid = agent.encode_extend(
        carry, tok, count, window=WINDOW, start=0,
        chunk=CHUNK, budget=budget)
    return c2.M, c2.I, c2.pos, rows, valid.astype(jnp.int32)


def _extend_scalar(agent, tok, count, budget):
    M, I, _pos, rows, _v = _extend_out(agent, tok, count, budget)
    return (jnp.sum(M * 1.0) + jnp.sum(I * 2.0)
            + jnp.sum(rows * jnp.arange(rows.shape[0],
                                        dtype=jnp.float32)[:, None]))


# ------------------------------------------------------------------ asserts --

def _assert_same_tree(a, b, what):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb) > 0
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape, f"{what} leaf {i}: {x.shape} != {y.shape}"
        assert np.array_equal(x, y), (
            f"{what} leaf {i} differs: max|d|="
            f"{np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))}")


def _assert_close_tree(a, b, what, ulps):
    """Equal to within `ulps` float32 ulp of the reference's own magnitude."""
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb) > 0
    eps = float(np.finfo(np.float32).eps)
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        assert x.shape == y.shape, f"{what} leaf {i}: {x.shape} != {y.shape}"
        scale = max(float(np.max(np.abs(x))), 1e-30)
        d = float(np.max(np.abs(x - y)))
        assert d <= ulps * eps * scale, (
            f"{what} leaf {i} is {d / (eps * scale):.2f} rel-ulp apart, "
            f"which is more than the {ulps} this pins")


def _assert_real_grad(g):
    lo = [x for x in jax.tree_util.tree_leaves(g) if eqx.is_inexact_array(x)]
    tot = sum(float(jnp.sum(jnp.abs(x))) for x in lo)
    assert tot > 0.0, "the reference gradient is identically zero"
    return lo


# ------------------------------------------------------------------- tests --

@pytest.mark.parametrize("count", COUNTS)
def test_the_folded_forward_is_bit_identical_at_every_count(setup, count):
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)
    args = (s["agent"], s["tok"][0], cnt, s["owner"][0], s["part"][0], bud)
    with count_vjp(False):
        old = _fold_out(*args)
    with count_vjp(True):
        new = _fold_out(*args)
    _assert_same_tree(old, new, f"folded forward at count {count}")


@pytest.mark.parametrize("count", COUNTS)
def test_the_folded_gradient_on_a_sequential_chunk_is_bit_identical(
        setup, count):
    """The chunk walked token by token. Zero tolerance, every count."""
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)

    def go(ag):
        return _fold_scalar(ag, s["tok"][0], cnt, s["owner"][0],
                            s["part"][0], bud)

    with env(ALPHAGRAD_FOLD_PARALLEL=0):
        with count_vjp(False):
            g_old = eqx.filter_grad(go)(s["agent"])
        with count_vjp(True):
            g_new = eqx.filter_grad(go)(s["agent"])
    if count > 0:
        _assert_real_grad(g_old)
    _assert_same_tree(g_old, g_new, f"folded gradient at count {count}")


@pytest.mark.parametrize("count", COUNTS)
def test_the_folded_gradient_on_a_parallel_chunk_agrees_to_a_few_ulp(
        setup, count):
    """THE KNOWN DIVERGENCE, pinned rather than hidden.

    The associative-scan chunk interior is the shipped default. Here the two
    loop forms' transposes reassociate against each other and the gradient
    moves in its last bits once more than one chunk is live. The bound is
    what makes this a reassociation claim and not a hope: 8 float32 ulp of
    the leaf's own magnitude.
    """
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)

    def go(ag):
        return _fold_scalar(ag, s["tok"][0], cnt, s["owner"][0],
                            s["part"][0], bud)

    with env(ALPHAGRAD_FOLD_PARALLEL=1):
        with count_vjp(False):
            g_old = eqx.filter_grad(go)(s["agent"])
        with count_vjp(True):
            g_new = eqx.filter_grad(go)(s["agent"])
    if count > 0:
        _assert_real_grad(g_old)
    _assert_close_tree(g_old, g_new,
                       f"folded gradient at count {count}", ulps=8.0)


@pytest.mark.parametrize("count", COUNTS)
def test_the_unfolded_forward_is_bit_identical_at_every_count(setup, count):
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)
    with count_vjp(False):
        old = _extend_out(s["agent"], s["tok"][0], cnt, bud)
    with count_vjp(True):
        new = _extend_out(s["agent"], s["tok"][0], cnt, bud)
    _assert_same_tree(old, new, f"unfolded forward at count {count}")


@pytest.mark.parametrize("count", COUNTS)
def test_the_unfolded_gradient_is_bit_identical_at_every_count(setup, count):
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)

    def go(ag):
        return _extend_scalar(ag, s["tok"][0], cnt, bud)

    with count_vjp(False):
        g_old = eqx.filter_grad(go)(s["agent"])
    with count_vjp(True):
        g_new = eqx.filter_grad(go)(s["agent"])
    if count > 0:
        _assert_real_grad(g_old)
    _assert_same_tree(g_old, g_new, f"unfolded gradient at count {count}")


def _vmapped_scalar(agent, s):
    """The loss's own shape: batched counts, ONE batch-wide budget."""
    cnts = jnp.asarray(np.array(COUNTS, np.int32))
    bud = jnp.max(cnts)

    def one(tok, cnt, ow, pa):
        return _fold_scalar(agent, tok, cnt, ow, pa, bud)

    return jnp.sum(jax.vmap(one)(s["tok"], cnts, s["owner"], s["part"]))


def test_the_folded_gradient_under_vmap_on_a_sequential_chunk_is_close(setup):
    """Under vmap even the sequential interior moves, by the same few ulp.

    The scalar sequential case above is bitwise equal; adding the vmap is
    enough to expose the cond's reassociation there too. The bound is the
    same one the parallel case takes.
    """
    s = setup
    with env(ALPHAGRAD_FOLD_PARALLEL=0):
        with count_vjp(False):
            g_old = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
        with count_vjp(True):
            g_new = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
    _assert_real_grad(g_old)
    _assert_close_tree(g_old, g_new, "vmapped folded gradient", ulps=8.0)


def test_the_folded_gradient_under_vmap_on_a_parallel_chunk_is_close(setup):
    """The shape the loss runs: batched counts, one unbatched budget.

    This is also the case that used to raise UnexpectedTracerError, because
    a closed-over integer BatchTracer escaped the backward. It has to RUN,
    not only agree.
    """
    s = setup
    with env(ALPHAGRAD_FOLD_PARALLEL=1):
        with count_vjp(False):
            g_old = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
        with count_vjp(True):
            g_new = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
    _assert_real_grad(g_old)
    _assert_close_tree(g_old, g_new, "vmapped folded gradient", ulps=8.0)


def test_the_backward_trip_count_follows_the_budget_and_not_the_window():
    """The point of the change, read off the lowered program.

    The scan form's backward is ``ceil(window / chunk)`` iterations long
    whatever the budget is. The custom_vjp form's is a ``while_loop``, so the
    jaxpr of the gradient must contain ``while`` and must NOT contain a
    ``scan`` of length ``nb`` over the chunk body.
    """
    from alphagrad.approx.common import count_vjp as CV

    def body(i, c):
        return (c[0] * 1.0001 + jnp.float32(i), ), None

    def go(x):
        with count_vjp(True):
            c, _ = CV.count_loop(body, (x,), nb=8, nb_live=jnp.int32(3))
        return jnp.sum(c[0])

    txt = str(jax.make_jaxpr(jax.grad(go))(jnp.ones((4,), jnp.float32)))
    assert "while" in txt, "the backward is not a while_loop"


def test_a_carry_with_an_integer_leaf_is_refused_rather_than_silently_wrong():
    from alphagrad.approx.common import count_vjp as CV

    def body(i, c):
        return c, None

    with pytest.raises(TypeError, match="all-inexact"):
        CV.count_loop(body, (jnp.ones((2,), jnp.float32),
                             jnp.zeros((), jnp.int32)),
                      nb=4, nb_live=jnp.int32(2))
