"""WHAT THE COUNT-PROPORTIONAL BACKWARD PASS CHANGES, AND WHAT IT DOES NOT.

``common/count_vjp.count_loop`` replaced the differentiated chunk loop's
``lax.scan`` + ``lax.cond`` over ``ceil(window / chunk)`` iterations with a
``jax.custom_vjp`` whose forward AND backward are ``lax.while_loop``s over the
real live chunk count. This module says exactly how far that is a change of
trip count and where it is also a change of the last bits.

IT IS NOW THE ONLY LOOP IN THE SOURCE (owner ruling 2026-09-15). The old body
lives in ``count_vjp_oracle.py`` beside this file and nowhere else, and
``shipped_loop(False)`` installs it under both production call sites. That is
what every comparison below is against.

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
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import count_vjp_oracle as _oracle  # noqa: E402


@pytest.fixture(autouse=True)
def _exact_read():
    """THIS MODULE PINS A PROPERTY OF THE EXACT OPERATOR, so it forces
    ``ALPHAGRAD_PALIMPSA_READ_ROLLOUT/_LOSS=exact`` even though the shipped default is now
    ``fast``.

    The claim is BIT-IDENTITY between `count_loop` and the scan-plus-cond form
    at an arbitrary chunk size (16) and arbitrary counts (0, 16, 37, 64). Under
    the fast read a chunk of 16 misaligns the 32-token grid and is refused, and
    the property being compared is a property of the recurrence the loop walks,
    not of the loop.

    The fast read is pinned separately, on the properties it does have, in
    ``tests/fast_read_extend_parity_test.py``.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALPHAGRAD_PALIMPSA_READ_ROLLOUT", "exact")
        mp.setenv("ALPHAGRAD_PALIMPSA_READ_LOSS", "exact")
        yield



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
def shipped_loop(on: bool):
    """Which chunk loop the production call sites use, around one trace.

    ``on=True`` is the shipped one, ``count_vjp.count_loop``, and the block
    does nothing at all: there is no switch to set any more (owner ruling
    2026-09-15).

    ``on=False`` installs the OLD scan-and-cond body from
    ``count_vjp_oracle``. Both ``delta_fold`` and ``ppo`` reach the loop as an
    attribute of the ``count_vjp`` MODULE, so replacing that one attribute
    covers both call sites, and the old body is written down once, in the
    oracle module, and nowhere else.
    """
    from alphagrad.approx.common import count_vjp as CV
    if on:
        yield
        return
    prev = CV.count_loop
    CV.count_loop = _oracle.scan_cond_loop
    try:
        yield
    finally:
        CV.count_loop = prev


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
    # The episode stream: one row per sample, long enough for the fold's
    # PADDED length past the largest start (episode_stream.stream_tail is
    # what sizes this in production).
    stream = jnp.asarray(
        rng.integers(1, 200, (B, 4 * WINDOW)).astype(np.int32))
    offs = jnp.asarray(np.array([0, 13, WINDOW, 2 * WINDOW], np.int32))
    return dict(agent=agent, tok=tok, part=part, owner=owner,
                stream=stream, offs=offs)


# ---------------------------------------------------------------- the fold --

def _advance(agent, tok, count, owner, part, budget):
    from alphagrad.approx.common import carry_stream as CS
    carry = agent.carry_init()
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    return CS.advance(
        agent, carry, vs, vc, tok, count, owner,
        window=WINDOW, participants=part,
        chunk=CHUNK, budget=budget, path="rollout",
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
    with shipped_loop(False):
        old = _fold_out(*args)
    with shipped_loop(True):
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
        with shipped_loop(False):
            g_old = eqx.filter_grad(go)(s["agent"])
        with shipped_loop(True):
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
        with shipped_loop(False):
            g_old = eqx.filter_grad(go)(s["agent"])
        with shipped_loop(True):
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
    with shipped_loop(False):
        old = _extend_out(s["agent"], s["tok"][0], cnt, bud)
    with shipped_loop(True):
        new = _extend_out(s["agent"], s["tok"][0], cnt, bud)
    _assert_same_tree(old, new, f"unfolded forward at count {count}")


@pytest.mark.parametrize("count", COUNTS)
def test_the_unfolded_gradient_is_bit_identical_at_every_count(setup, count):
    s = setup
    cnt = jnp.asarray(count, jnp.int32)
    bud = jnp.asarray(count, jnp.int32)

    def go(ag):
        return _extend_scalar(ag, s["tok"][0], cnt, bud)

    with shipped_loop(False):
        g_old = eqx.filter_grad(go)(s["agent"])
    with shipped_loop(True):
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
        with shipped_loop(False):
            g_old = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
        with shipped_loop(True):
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
        with shipped_loop(False):
            g_old = eqx.filter_grad(
                lambda ag: _vmapped_scalar(ag, s))(s["agent"])
        with shipped_loop(True):
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


# ------------------------------------------- the shape the loss really runs --

def _stream_scalar(agent, s, budget):
    """A CHECKPOINTED advance reading the EPISODE STREAM by (row, start).

    This is `_dynamic_loss_fn`'s own call and nothing above imitates it: the
    tokens are a `dynamic_slice` out of an (E, L) stream instead of a
    standalone window, the whole advance sits inside `jax.checkpoint`
    (ALPHAGRAD_CARRY_HEADS_REMAT), and it all runs under a vmap over samples.
    The first build of this change passed every case above and still failed
    here, three frames away, with a `broadcast_in_dim` rank complaint -- so
    the case is now pinned rather than left to the smoke.
    """
    from alphagrad.approx.common import carry_stream as CS

    def one(off, cnt, ow, pa, row):
        carry = agent.carry_init()
        vs, vc = CS.zero_memory(TOTAL_V, EMBD)
        step = jax.checkpoint(
            lambda c, u, w: CS.advance(
                agent, c, u, w, s["stream"], cnt, ow,
                start=off, row=row, window=WINDOW, participants=pa,
                chunk=CHUNK, budget=budget, path="rollout"))
        carry, vs, vc = step(carry, vs, vc)
        return (jnp.sum(carry.M * 1.0) + jnp.sum(carry.I * 2.0)
                + jnp.sum(vs * 4.0) + jnp.sum(vc * 5.0))

    cnts = jnp.asarray(np.array(COUNTS, np.int32))
    rows = jnp.arange(len(COUNTS), dtype=jnp.int32)
    return jnp.sum(jax.vmap(one)(s["offs"], cnts, s["owner"],
                                 s["part"], rows))


def test_the_streamed_checkpointed_advance_under_vmap_runs_and_is_close(setup):
    s = setup
    bud = jnp.asarray(max(COUNTS), jnp.int32)
    with shipped_loop(False):
        g_old = eqx.filter_grad(
            lambda ag: _stream_scalar(ag, s, bud))(s["agent"])
    with shipped_loop(True):
        g_new = eqx.filter_grad(
            lambda ag: _stream_scalar(ag, s, bud))(s["agent"])
    _assert_real_grad(g_old)
    _assert_close_tree(g_old, g_new, "streamed checkpointed gradient",
                       ulps=8.0)


def test_the_switch_is_gone_and_the_old_value_does_not_bring_it_back(setup):
    """``ALPHAGRAD_COUNT_VJP=0`` used to select the scan-and-cond loop.

    The variable is deleted (owner ruling 2026-09-15). Setting it must now do
    NOTHING: a stale launcher or a stale shell that still exports the old
    default must not quietly hand the trainer a different gradient. The proof
    is that the fold's gradient with the variable set to 0 is bitwise the
    gradient without it, and that both differ from the oracle's in the way
    the parallel interior always does.
    """
    s = setup
    cnt = jnp.asarray(37, jnp.int32)
    bud = jnp.asarray(37, jnp.int32)

    def go(ag):
        return _fold_scalar(ag, s["tok"][0], cnt, s["owner"][0],
                            s["part"][0], bud)

    with env(ALPHAGRAD_FOLD_PARALLEL=1):
        g_plain = eqx.filter_grad(go)(s["agent"])
        with env(ALPHAGRAD_COUNT_VJP=0):
            g_zero = eqx.filter_grad(go)(s["agent"])
    _assert_same_tree(g_plain, g_zero,
                      "the deleted ALPHAGRAD_COUNT_VJP still steers the loop")
    from alphagrad.approx.common import count_vjp as CV
    assert not hasattr(CV, "enabled"), (
        "count_vjp.enabled() is back -- the switch has to stay deleted")


# ------------------------------------- the gate's own inputs, as an oracle --
#
# THE CHILD INTERPRETER, AND WHY. The policy gate pins ten ALPHAGRAD_*
# variables at module scope and they have to be set BEFORE alphagrad is
# imported (env.py reads MAX_FACES / MAX_DELTA_TOKENS at import). In a shared
# pytest process alphagrad has already been imported, so importing the gate
# here would be ten dead writes AND would leak into every module collected
# afterwards. `policy_regression_gate_test.py` solves it the same way.

_THIS = Path(__file__).resolve()


def _run_child(*argv):
    child_env = {k: v for k, v in os.environ.items()
                 if not k.startswith("ALPHAGRAD_")}
    child_env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.run([sys.executable, str(_THIS), *argv],
                          env=child_env, capture_output=True, text=True,
                          timeout=3600)


def test_the_two_bodies_on_the_policy_gates_own_rollout_inputs():
    """THE PROOF THE OWNER ASKED FOR, on the real fixture.

    The child replays the policy gate's own seeded rollout, records every
    ``carry_stream.advance`` the gate makes -- the real per-step token deltas,
    counts, owners and participation masks of the real graph -- and then runs
    those SAME inputs through the differentiated shape the loss uses: one
    ``advance`` per sample under ``vmap``, with the batch-wide count as the
    unbatched ``budget``. It does that twice, once with the shipped
    ``count_loop`` and once with the oracle's scan-and-cond body, and compares.

    The bar: the forward BYTE-IDENTICAL, the gradient within 1e-6 relative.
    The child also prints the observed maximum relative difference, which is
    what a tolerance of zero would have had to accept.
    """
    r = _run_child("--gate-inputs")
    assert r.returncode == 0, (
        f"the child failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "[cvjp-oracle] forward byte-identical: True" in r.stdout, r.stdout
    assert "[cvjp-oracle] max relative gradient difference" in r.stdout, \
        r.stdout
    print(r.stdout)


# ----------------------------------------------------------- the child body --

def _gate_inputs_main() -> int:
    """Run in a FRESH interpreter (see `_run_child`). Never under pytest."""
    import importlib.util

    here = _THIS.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    # FIRST, and by path: importing it is what sets the gate's pins, and they
    # only take before alphagrad is imported.
    spec = importlib.util.spec_from_file_location(
        "_cvjp_gate", here / "policy_regression_gate.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)

    from alphagrad.approx.common import carry_stream as CS
    from alphagrad.approx.common import count_vjp as CV

    case = G.build_case()

    # Record what the gate's own rollout hands `advance`, without touching
    # the gate: `run_trace` looks the function up on the module every step.
    seen = []
    _real = CS.advance

    def _spy(*a, **k):
        seen.append((a, k))
        return _real(*a, **k)

    CS.advance = _spy
    try:
        G.run_trace(case)
    finally:
        CS.advance = _real
    if not seen:
        raise RuntimeError("the gate made no advance call to record")

    agent = case["agent"]
    total_v = case["total_v"]
    window = int(seen[0][1]["window"])
    embd = int(G.EMBD)
    toks = jnp.stack([jnp.asarray(a[4]) for a, _ in seen])
    cnts = jnp.stack([jnp.asarray(a[5], jnp.int32) for a, _ in seen])
    owns = jnp.stack([jnp.asarray(a[6], jnp.int32) for a, _ in seen])
    parts = jnp.stack([jnp.asarray(k["participants"]) for _, k in seen])
    # THE BUDGET IS BATCH-WIDE AND UNBATCHED, which is the contract the loop
    # carries: a per-sample bound would make the while_loop run to the
    # batch-wide maximum with a select on every lane and save nothing.
    budget = jnp.max(cnts)
    print(f"[cvjp-oracle] {len(seen)} advance calls from the gate, "
          f"window={window}, counts={[int(c) for c in cnts]}, "
          f"budget={int(budget)}")

    def _one(ag, tok, cnt, ow, pa):
        carry = ag.carry_init()
        vs, vc = CS.zero_memory(total_v, embd)
        c2, vs2, vc2 = CS.advance(
            ag, carry, vs, vc, tok, cnt, ow, window=window,
            participants=pa, budget=budget, path="loss")
        return c2.M, c2.I, vs2, vc2

    def _out(ag):
        return jax.vmap(lambda t, c, o, p: _one(ag, t, c, o, p))(
            toks, cnts, owns, parts)

    def _scalar(ag):
        M, I, vs, vc = _out(ag)
        return (jnp.sum(M * 1.0) + jnp.sum(I * 2.0)
                + jnp.sum(vs * 4.0) + jnp.sum(vc * 5.0))

    def _with_oracle(fn):
        prev = CV.count_loop
        CV.count_loop = _oracle.scan_cond_loop
        try:
            return fn()
        finally:
            CV.count_loop = prev

    fwd_new = _out(agent)
    fwd_old = _with_oracle(lambda: _out(agent))
    same = all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes()
        for x, y in zip(jax.tree_util.tree_leaves(fwd_new),
                        jax.tree_util.tree_leaves(fwd_old)))
    print(f"[cvjp-oracle] forward byte-identical: {same}")

    g_new = eqx.filter_grad(_scalar)(agent)
    g_old = _with_oracle(lambda: eqx.filter_grad(_scalar)(agent))
    tot = sum(float(jnp.sum(jnp.abs(x)))
              for x in jax.tree_util.tree_leaves(eqx.filter(
                  g_old, eqx.is_inexact_array)))
    if not tot > 0.0:
        raise RuntimeError("the oracle gradient is identically zero")
    worst, leaf = _oracle.max_rel_diff(
        eqx.filter(g_new, eqx.is_inexact_array),
        eqx.filter(g_old, eqx.is_inexact_array))
    print(f"[cvjp-oracle] max relative gradient difference = {worst:.6e} "
          f"(leaf {leaf}); a tolerance of 0 accepts only {0.0:.1f}, so a zero "
          f"tolerance would {'pass' if worst == 0.0 else 'FAIL'} here")
    if not same:
        raise AssertionError("the forward is not byte-identical")
    if not worst <= 1e-6:
        raise AssertionError(
            f"the gradient is {worst:.3e} relative apart, over the 1e-6 bar")
    return 0


if __name__ == "__main__":
    if "--gate-inputs" in sys.argv[1:]:
        raise SystemExit(_gate_inputs_main())
    raise SystemExit("usage: python tests/count_vjp_test.py --gate-inputs")
