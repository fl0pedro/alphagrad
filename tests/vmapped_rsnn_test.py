"""The batched one-step recurrent target (bead dsnn-dfw.139).

``VmappedRSNN_SHD`` is ``RSNN_SHD`` over ``B`` recordings: the step body
vmapped with synaptax's in_axes pattern (frame, label, carried state and every
carried block mapped on axis 0, the weights shared), ``B`` recordings and ``B``
step positions drawn by the producers, and the carry built per recording in
the container the plan implies. ``RSNN_SHD`` itself does not move.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.core import _inline_call_primitives

from alphagrad.approx.common import datasets as ds
from alphagrad.approx.common import examples as ex
from alphagrad.approx.common import rsnn_shd as R
from alphagrad.approx.common.datasets import SHD_CHANNELS, SHD_CLASSES

B, H, T = 4, 16, 12
BATCHED = "VmappedRSNN_SHD"
SINGLE = "RSNN_SHD"
RULES = ("tbptt", "bptt", "rtrl")


@pytest.fixture(autouse=True)
def small(monkeypatch):
    monkeypatch.setattr(ds, "NN_VMAP_BATCH", B)
    monkeypatch.setattr(R, "RSNN_HIDDEN", H)
    monkeypatch.setattr(R, "SHD_TIME_BINS", T)


def _args(rule, seed=1, **kw):
    return ex.get_args(BATCHED, jax.random.PRNGKey(seed), dataset=None,
                       temporal_rule=rule, **kw)


def _rows(xs, rule):
    """The ``B`` one-recording tuples the batched tuple holds."""
    head, ws, cs, given = xs[:7], xs[7:10], xs[10:16], xs[16:]
    return [tuple(a[i] for a in head) + tuple(ws) + tuple(cs)
            + tuple(b[i] for b in given) for i in range(B)]


def _recordings(seed=1):
    """The recordings and weights ``_args(rule, seed)`` drew."""
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    seq, y, _ = R._draw_recordings(k[0], None, -1, B)
    return seq, y, R.rsnn_weights(k[1])


def _close(a, b, quant=False):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    # A quantized block is bfloat16; one ulp of it is 2^-8.
    rtol = 2e-2 if quant else 1e-4
    np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-6)


# ---------------------------------------------------------------------------
# 1. The tuple: mapped slots, shared slots
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_the_batched_tuple_maps_the_recording_and_shares_the_weights(rule):
    xs = _args(rule)
    pos = R.last_step_position()
    assert pos["batch"] == B and len(pos["t"]) == B
    assert all(1 <= t < T for t in pos["t"])
    ref = ex.get_args(SINGLE, jax.random.PRNGKey(1), dataset=None,
                      temporal_rule=rule)
    assert len(xs) == len(ref)
    for slot in range(7):
        assert xs[slot].shape == (B,) + ref[slot].shape, slot
    for slot in range(7, 16):
        assert xs[slot].shape == ref[slot].shape, slot
    assert xs[7].shape == (H, SHD_CHANNELS)
    assert xs[8].shape == (H, H)
    assert xs[9].shape == (SHD_CLASSES, H)
    for slot in range(16, len(xs)):
        assert xs[slot].shape == (B,) + ref[slot].shape, slot
        assert xs[slot].dtype == ref[slot].dtype
    assert ex.infer_argnums(BATCHED) == R.RSNN_ARGNUMS


def test_the_unbatched_target_is_untouched():
    assert R.rsnn_batch(SINGLE) is None
    assert R.rsnn_batch(BATCHED) == B
    xs = ex.get_args(SINGLE, jax.random.PRNGKey(1), dataset=None,
                     temporal_rule="rtrl", step_position=5)
    assert xs[0].shape == (SHD_CHANNELS,)
    pos = R.last_step_position()
    assert pos["t"] == 5 and "batch" not in pos
    assert ex.infer_argnums(SINGLE) == R.RSNN_ARGNUMS


def test_window2_is_refused_on_the_batched_target():
    with pytest.raises(ValueError, match="window2"):
        R.resolve_temporal_rule(BATCHED, "window2")
    with pytest.raises(ValueError, match="window2"):
        ex.get_args(BATCHED, jax.random.PRNGKey(1), dataset=None,
                    temporal_rule="window2")
    with pytest.raises(ValueError, match="not batched"):
        ex.get_raw_fn("VmappedRSNN_SHD_W2")
    # The two-graph run alternates two graphs of the batched target.
    assert R.resolve_temporal_rules(BATCHED, ["bptt", "rtrl"]) == ["bptt", "rtrl"]
    assert R.target_example(BATCHED, ["bptt", "rtrl"]) == BATCHED


# ---------------------------------------------------------------------------
# 2. PROOF 1: the batched loss and gradient are the mean over recordings,
#    with the carries built per sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_the_batched_loss_and_gradient_are_the_mean_over_recordings(rule):
    xs = _args(rule)
    fb_full = ex.get_fn(BATCHED)
    fs_full = ex.get_fn(SINGLE)
    fb = lambda *a: R.loss_of(fb_full(*a))
    fs = lambda *a: R.loss_of(fs_full(*a))
    rows = _rows(xs, rule)
    out_b = fb_full(*xs)
    if rule == "rtrl":
        # (loss, S, I, U, a, Uo): the B next states behind the mean loss
        # (owner ruling 2026-09-24, Q27b).
        assert isinstance(out_b, tuple) and len(out_b) == 6
        assert [tuple(s.shape) for s in out_b[1:]] == [
            (B, H), (B, H), (B, H), (B, H), (B, SHD_CLASSES)]
    loss_b = R.loss_of(out_b)
    assert loss_b.shape == ()
    losses = [fs(*r) for r in rows]
    _close(loss_b, jnp.mean(jnp.stack(losses)))
    argnums = ex.infer_argnums(BATCHED)
    g_b = jax.grad(fb, argnums=argnums)(*xs)
    g_rows = [jax.grad(fs, argnums=argnums)(*r) for r in rows]
    for k in range(len(argnums)):
        _close(g_b[k], jnp.mean(jnp.stack([g[k] for g in g_rows]), axis=0))


@pytest.mark.parametrize("container", ["exact", "diag", "reduce", "quant",
                                       "diag+quant"])
def test_the_past_jacobian_is_built_per_recording(container):
    """The batched scan of the empty plan's program gives every recording
    the carry the one-recording scan gives it (owner ruling 2026-09-24,
    Q29), and the exact container is the exact carry."""
    xs = _args("rtrl", carry_container=container)
    seq, y, W = _recordings()
    ts = R.last_step_position()["t"]
    prog = R.empty_plan_program()
    for i in range(B):
        assert np.array_equal(np.asarray(xs[0][i]), np.asarray(seq[i][ts[i]]))
        single = R.carry_from_program(seq[i], y[i], ts[i], W, prog, container)
        for k, block in enumerate(single):
            got = xs[16 + k][i]
            assert got.shape == block.shape and got.dtype == block.dtype
            _close(got, block, quant="quant" in container)
        if container == "exact":
            for got, block in zip(single, R.carried_jacobians(seq[i], ts[i], W)):
                _close(got, block)


@pytest.mark.parametrize("container", ["exact", "reduce", "quant", "diag"])
def test_the_future_adjoint_is_built_per_recording(container):
    xs = _args("bptt", carry_container=container)
    seq, y, W = _recordings()
    ts = R.last_step_position()["t"]
    for i in range(B):
        st = tuple(R.prefix_state(seq[i], ts[i], W)(*W))
        for k, s in enumerate(st):
            _close(xs[2 + k][i], s)
        lam = R.future_adjoints(seq[i], y[i], ts[i], W, st, container)
        for k, a in enumerate(lam):
            got = xs[16 + k][i]
            assert got.shape == a.shape and got.dtype == a.dtype
            _close(got, a, quant="quant" in container)


# ---------------------------------------------------------------------------
# 3. PROOF 2: jacve on the batched step body against jax.jacrev, five seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("rule", RULES)
def test_jacve_on_the_batched_step_body_matches_jacrev(rule, seed):
    xs = _args(rule, seed=seed)
    fn = ex.get_fn(BATCHED)
    argnums = ex.infer_argnums(BATCHED)
    got = jacve(fn, "rev", argnums=argnums)(*xs)
    ref = jax.jacrev(fn, argnums=argnums)(*xs)
    got_leaves = jax.tree_util.tree_leaves(got)
    ref_leaves = jax.tree_util.tree_leaves(ref)
    # under rtrl the loss row and the five state rows, per weight
    n_rows = 6 if rule == "rtrl" else 1
    assert len(got_leaves) == len(ref_leaves) == len(argnums) * n_rows
    for i, (g, r) in enumerate(zip(got_leaves, ref_leaves)):
        scale = float(jnp.max(jnp.abs(r))) or 1.0
        worst = float(jnp.max(jnp.abs(g - r))) / scale
        assert worst < 1e-5, (f"{rule} seed {seed} leaf {i}: relative "
                              f"residual {worst:.3e} against jax.jacrev")


@pytest.mark.parametrize("rule", RULES)
def test_the_graph_shape_does_not_move_with_the_step_position(rule):
    fn = ex.get_fn(BATCHED)
    seen = set()
    for t in (1, 5, T - 1):
        xs = _args(rule, step_position=t)
        cj = jax.make_jaxpr(fn)(*xs)
        jx, _ = _inline_call_primitives(cj.jaxpr, cj.literals)
        seen.add(len(jx.eqns))
    assert len(seen) == 1, f"the batched graph moved with t: {seen}"


# ---------------------------------------------------------------------------
# 4. The generator: B recordings, B step positions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_the_generator_draws_b_recordings_and_b_step_positions(rule):
    gen = ex.data_gen(BATCHED, key=jax.random.PRNGKey(1), temporal_rule=rule)
    xs = _args(rule)
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    data = gen(keys)
    assert len(data) == len(gen.data_slots)
    for slot, d in zip(gen.data_slots, data):
        assert d.shape == xs[slot].shape, slot
        assert d.dtype == xs[slot].dtype, slot
    m = gen.meta(keys)
    assert m["batch"] == B and len(m["t"]) == B and len(m["recording"]) == B
    assert all(1 <= t < T for t in m["t"])
    drawn = {tuple(gen.meta(jax.random.split(jax.random.PRNGKey(s), 5))["t"])
             for s in range(6)}
    assert len(drawn) > 1, "the step positions did not move with the key"
    for slot in gen.data_slots:
        if 7 <= slot <= 9:
            d = data[gen.data_slots.index(slot)]
            assert np.array_equal(np.asarray(d), np.asarray(xs[slot]))


@pytest.mark.parametrize("container", ["diag", "reduce+quant"])
def test_the_generator_draws_the_container_per_recording(container):
    gen = ex.data_gen(BATCHED, key=jax.random.PRNGKey(1), temporal_rule="rtrl",
                      carry_container=container)
    xs = _args("rtrl", carry_container=container)
    data = gen(jax.random.split(jax.random.PRNGKey(3), 5))
    for slot, d in zip(gen.data_slots, data):
        assert d.shape == xs[slot].shape and d.dtype == xs[slot].dtype, slot
    ref = gen.reference_draw(jax.random.split(jax.random.PRNGKey(3), 5))
    exact = _args("rtrl", carry_container="exact")
    for slot, d in zip(gen.data_slots, ref):
        assert d.shape == exact[slot].shape and d.dtype == exact[slot].dtype
