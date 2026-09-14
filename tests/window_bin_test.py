# -*- coding: utf-8 -*-
"""THE PER-STEP DELTA WINDOW BIN: what `MAX_DELTA_TOKENS` stopped being.

Design: `.scratch/trustworthy-approx-search/design-window-bin.md` (owner
ruling 2026-09-14: bin the per-step window with the same rule as the stream
bin, first bin 4096, the two chosen together).

`ALPHAGRAD_MAX_DELTA_TOKENS` used to bound three different quantities at
once. It now bounds two -- the host-to-device wire and the loud ceiling --
and a third, the WINDOW the rollout and the loss actually scan, is a
per-episode bin. Six things are pinned here.

1. THE ARITHMETIC. The floor, the cap, the first bin, and the rule that
   picks the next one.
2. THE FLOOR RAISES. A window under the fold chunk changes the CHUNK, which
   regroups the loss's float32 partial sums. It is refused, not clamped.
   This is the single most likely way to move the gate's numbers.
3. THE OVERFLOW. A step whose delta, or whose face chunks in total, pass the
   bin is a device flag; it moves the WINDOW bin and leaves the stream bin
   alone, and a stream overflow does the reverse.
4. THE EQUIVALENCE. At or above the floor, a smaller bin gives bit-identical
   rows, vertex memory and edge ids. That is the whole reason the gate does
   not move.
5. THE SHAPES. `EnvState.delta_tokens` follows the config, the wire does
   not, and an env that differs only in its bin has a different treedef.
6. THE SIZE. What the bin is for, as a table.

This module pins no scale, so it is safe to collect beside anything, and
every alphagrad import is LAZY for the reason `tests/episode_stream_test.py`
states: a module-scope import freezes `env.py`'s constants at COLLECTION
time and turns every later module's scale pin into a dead write.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402


class _LazyModule:
    """Import on first attribute access, not at collection."""

    def __init__(self, path):
        self._path = path
        self._mod = None

    def __getattr__(self, name):
        if self._mod is None:
            import importlib

            self._mod = importlib.import_module(self._path)
        return getattr(self._mod, name)


ES = _LazyModule("alphagrad.approx.common.episode_stream")
FOLD = _LazyModule("alphagrad.approx.common.delta_fold")
CS = _LazyModule("alphagrad.approx.common.carry_stream")

E = 4


class StubAgent:
    """A real carried recurrence with cheap rows (`delta_fold_test`'s stub).

    What is under test is the plumbing -- which tokens reach which chunk at
    which offset, and whether the window changes the grouping -- not the
    encoder. A carry every token moves is what makes a misplaced chunk
    boundary impossible to match by accident.
    """

    embd_dim = E

    def encode_extend(self, carry, toks, count, *, window, start=0,
                      chunk=None, budget=None):
        del start, chunk, budget
        from jax import lax

        idx = jnp.arange(window, dtype=jnp.int32)
        valid = idx < count

        def step(c, x):
            tok, ok = x
            c2 = jnp.where(ok, c + tok.astype(jnp.float32), c)
            row = jnp.where(ok, c2 * (1.0 + jnp.arange(E, dtype=jnp.float32)),
                            jnp.zeros((E,), jnp.float32))
            return c2, row

        return (*lax.scan(step, carry, (toks[:window], valid)), valid)


def _win_policy(initial=None, **kw):
    """A window `BinPolicy` with the window bin's own knobs."""
    floor = kw.pop("floor", ES.window_floor_log2())
    cap = kw.pop("cap", 15)
    if initial is None:
        initial = max(12, floor)
    return ES.BinPolicy(
        initial, cap=cap, floor=floor,
        history_env=ES.WIN_HISTORY_ENV, margin_env=ES.WIN_MARGIN_ENV,
        cap_env=ES.WIN_LOG2_MAX_ENV,
        history_default=ES.WIN_HISTORY_DEFAULT,
        margin_default=ES.WIN_MARGIN_DEFAULT,
        **kw)


# --------------------------------------------- 1. the bin and its arithmetic

def test_the_window_bin_is_the_smallest_power_of_two_that_holds_the_recent_maximum_times_the_margin():
    """The rule the `--delta-window-log2` help quotes.

    3000 tokens times a margin of 1.5 is 4500, and the smallest power of two
    holding 4500 is 8192. Not 4096: the margin is headroom above a length
    already seen, and 4096 would leave none.
    """
    p = _win_policy(initial=13, history=4, margin=1.5)
    for _ in range(4):
        p.record(3000)
    assert p.pick() == 13
    assert 1 << 13 == 8192
    # And one doubling smaller once the whole history fits under it.
    p2 = _win_policy(initial=13, history=2, margin=1.5)
    p2.record(1300)
    p2.record(1300)
    assert p2.pick() == 12


def test_a_delta_history_that_shrinks_makes_the_driver_pick_the_smaller_window_bin():
    """DOWN, one power of two per episode, and only on the WHOLE window."""
    p = _win_policy(initial=15, history=3, margin=1.0, floor=10)
    for _ in range(3):
        p.record(1024)
    walk = [p.pick() for _ in range(6)]
    # 15 -> 14 -> 13 -> 12 -> 11 -> 10, then it sits on the floor.
    assert walk == [14, 13, 12, 11, 10, 10]


def test_the_window_bin_is_never_chosen_below_the_fold_chunk_floor():
    """A history of tiny deltas leaves the bin ON the floor, not under it.

    Going under it would shrink `plan_chunks`'s C, which regroups the
    loss's float32 partial sums. The floor is an upward clamp in the
    SELECTION (a short history is ordinary); it is a configured bin below
    the floor that raises.
    """
    p = _win_policy(initial=12, history=2, margin=1.0, floor=10)
    p.record(1)
    p.record(1)
    assert p.pick() == 11
    p.record(1)
    p.record(1)
    assert p.pick() == 10
    for _ in range(6):
        p.record(1)
        assert p.pick() == 10


def test_the_window_bin_is_never_chosen_above_max_delta_tokens():
    """The window slices the wire, so it cannot be wider than the wire."""
    p = _win_policy(initial=14, history=2, margin=1.0, cap=15)
    p.record(1 << 16)
    p.record(1 << 16)
    with pytest.raises(ES.EpisodeStreamCapReached) as exc:
        p.pick()
    assert exc.value.cap == 15
    assert ES.WIN_LOG2_MAX_ENV in str(exc.value) or "cap" in str(exc.value)


def test_a_window_margin_below_one_is_refused_because_it_asks_for_a_bin_smaller_than_a_length_already_seen():
    with pytest.raises(ValueError) as exc:
        _win_policy(margin=0.9)
    assert ES.WIN_MARGIN_ENV in str(exc.value)


def test_the_first_window_bin_with_no_history_is_the_owners_four_thousand_and_ninety_six():
    """The owner's ruling of 2026-09-14, and its two clamps.

    The design proposed starting AT the cap; the owner chose 4096. It is
    raised to the floor when the fold chunk is larger, and lowered to the
    cap when `ALPHAGRAD_MAX_DELTA_TOKENS` is smaller -- a run at a 2048 cap
    cannot start at 4096, because 4096 does not fit the wire.
    """
    assert ES.WIN_LOG2_DEFAULT == 12
    assert ES.resolve_window_log2(32768) == 12
    # Smaller cap: the first bin follows it down.
    assert ES.resolve_window_log2(2048) == 11
    assert "4096" in ES.WIN_FIRST_BIN_RULE


def test_the_environment_variable_and_the_flag_both_override_the_first_window_bin(
        monkeypatch):
    monkeypatch.setenv(ES.WIN_LOG2_ENV, "14")
    assert ES.resolve_window_log2(32768) == 14
    # The flag wins over the variable, which is the stream bin's order too.
    assert ES.resolve_window_log2(32768, override=13) == 13


def test_the_window_bin_is_chosen_from_the_maximum_of_the_delta_length_and_the_face_total():
    """Q1 and Q3 share ONE bin, so the history holds `max(Q1, Q3)`.

    They are two views of the same elimination: the step's whole token
    delta, and the total of the per-face chunks the head read out of it.
    The window bounds both, so a face total that needs a bigger bin must
    move the bin even when the delta alone would not.
    """
    delta_only = _win_policy(initial=11, history=1, margin=1.0, floor=10)
    both = _win_policy(initial=11, history=1, margin=1.0, floor=10)
    win_used, face_used = 500, 3000       # Q1 fits 2^11, Q3 does not
    delta_only.record(win_used)
    both.record(max(win_used, face_used))
    assert delta_only.pick() == 10        # the delta alone walks DOWN
    assert both.pick() == 12              # the face total drives it UP


# ------------------------------------------------ 2. the floor RAISES

def test_the_fold_chunk_floor_is_the_larger_of_the_fold_chunk_and_the_loss_extend_chunk(
        monkeypatch):
    monkeypatch.delenv(ES.LOSS_EXTEND_CHUNK_ENV, raising=False)
    monkeypatch.delenv(ES.EXTEND_CHUNK_ENV, raising=False)
    monkeypatch.setenv(ES.FOLD_CHUNK_ENV, "1024")
    assert ES.window_floor_tokens() == 1024
    assert ES.window_floor_log2() == 10
    # The campaign's 128 extend chunk does not move it; a 4096 one does.
    monkeypatch.setenv(ES.EXTEND_CHUNK_ENV, "128")
    assert ES.window_floor_tokens() == 1024
    monkeypatch.setenv(ES.LOSS_EXTEND_CHUNK_ENV, "4096")
    assert ES.window_floor_tokens() == 4096
    assert ES.window_floor_log2() == 12


def test_a_window_bin_below_the_fold_chunk_floor_is_refused_and_the_message_names_the_floor(
        monkeypatch):
    """RAISE, NOT CLAMP. The message has to say what the floor is and why."""
    monkeypatch.setenv(ES.FOLD_CHUNK_ENV, "1024")
    monkeypatch.delenv(ES.WIN_LOG2_ENV, raising=False)
    with pytest.raises(ES.DeltaWindowFloor) as exc:
        ES.resolve_window_log2(32768, override=9)
    msg = str(exc.value)
    assert "1024" in msg
    assert ES.FOLD_CHUNK_ENV in msg
    assert "regroup" in msg
    monkeypatch.setenv(ES.WIN_LOG2_ENV, "8")
    with pytest.raises(ES.DeltaWindowFloor):
        ES.resolve_window_log2(32768)


def test_a_configured_window_floor_below_the_fold_chunk_is_itself_refused(
        monkeypatch):
    monkeypatch.setenv(ES.FOLD_CHUNK_ENV, "1024")
    monkeypatch.setenv(ES.WIN_LOG2_MIN_ENV, "8")
    with pytest.raises(ES.DeltaWindowFloor):
        ES.window_floor_log2()
    monkeypatch.setenv(ES.WIN_LOG2_MIN_ENV, "12")
    assert ES.window_floor_log2() == 12


def test_the_env_refuses_a_delta_window_below_the_floor_or_off_the_power_of_two_grid():
    ENV = __import__("alphagrad.approx.env", fromlist=["env"])
    with pytest.raises(ES.DeltaWindowFloor):
        ENV.validate_delta_window(256)
    with pytest.raises(ValueError):
        ENV.validate_delta_window(3000)
    with pytest.raises(ValueError):
        ENV.validate_delta_window(ENV.MAX_DELTA_TOKENS * 2)
    # 0 means the cap, which is what every untouched caller passes.
    assert ENV.validate_delta_window(0) == ENV.MAX_DELTA_TOKENS


def test_plan_chunks_returns_the_same_chunk_size_at_every_window_bin_at_or_above_the_floor():
    """THE REASON THE FLOOR IS WHERE IT IS.

    `plan_chunks` does `C = min(C, W)`. At or above the floor the chunk is
    untouched and only the SCAN LENGTH changes, so the fold adds or removes
    whole chunks whose contribution is exactly +0.0. Below it the chunk
    itself shrinks, and then the real tokens are grouped differently.
    """
    floor = 1024
    for w in (1024, 2048, 4096, 8192, 16384, 32768):
        C, nb, padded = FOLD.plan_chunks(w, floor)
        assert C == floor, (w, C)
        assert nb == w // floor
        assert padded == w
    # One doubling under the floor and the chunk moves. That is the change
    # the floor exists to forbid.
    assert FOLD.plan_chunks(512, floor)[0] == 512


# ----------------------------------------- 3. the overflow and the two bins

def test_a_step_whose_delta_exceeds_the_window_bin_sets_the_device_flag_and_the_host_reads_it_back():
    """The same shape as the stream overflow: a value, never a raise.

    The rollout carries `(length, step, kind)` and runs to the END. Nothing
    is matched against a runtime's message text and nothing has to survive
    a callback boundary.
    """
    W = 4096
    seen = (jnp.zeros((), jnp.int32),) * 3
    for t, count in enumerate([10, 20, 9000, 30, 12000]):
        c = jnp.asarray(count, jnp.int32)
        seen = ES.carry_window_overflow(
            *seen, c, c > W, t, ES.WINDOW_KIND_DELTA)
    length, step, kind = (int(np.asarray(x)) for x in seen)
    assert (length, step, kind) == (9000, 2, ES.WINDOW_KIND_DELTA)
    rec = ES.window_overflow_from([0, length], [0, step], 12, [0, kind])
    assert rec.env_index == 1 and rec.step == 2 and rec.length == 9000
    assert rec.kind == ES.WindowOverflow.DELTA


def test_a_step_whose_face_chunks_total_more_than_the_window_bin_is_reported_as_the_face_kind():
    """Q3, and it says so. The old behaviour was a silent `jnp.minimum`."""
    W = 2048
    seen = (jnp.zeros((), jnp.int32),) * 3
    total = jnp.asarray(3300, jnp.int32)
    seen = ES.carry_window_overflow(
        *seen, total, total > W, 7, ES.WINDOW_KIND_FACE)
    rec = ES.window_overflow_from(
        [int(np.asarray(seen[0]))], [int(np.asarray(seen[1]))], 11,
        [int(np.asarray(seen[2]))])
    assert rec.kind == ES.WindowOverflow.FACE
    assert rec.length == 3300 and rec.step == 7
    assert "face concatenation" in str(rec)


def test_an_episode_that_did_not_overflow_its_window_reports_nothing():
    assert ES.window_overflow_from([0, 0, 0], [0, 0, 0], 12) is None


def test_the_window_overflow_marker_is_not_mistaken_for_the_episode_stream_overflow_marker():
    """The two messages must not share a prefix, in either direction.

    They are two different records now rather than two exception types, so
    the driver dispatches on the CLASS; the text still has to be unambiguous
    for the operator reading one log line.
    """
    w = ES.WindowOverflow(env_index=0, step=3, length=9000, log2=12)
    s = ES.StreamOverflow(env_index=0, step=3, length=9000, log2=12)
    assert not str(w).startswith(str(s)[:20])
    assert not str(s).startswith(str(w)[:20])
    assert "delta window overflow" in str(w)
    assert "delta window overflow" not in str(s)
    assert not isinstance(w, ES.StreamOverflow)
    assert not isinstance(w, Exception)


def test_a_window_overflow_repeats_the_episode_one_window_bin_up_and_leaves_the_stream_bin_alone():
    stream = ES.BinPolicy(15, history=4, margin=2.0, cap=24)
    win = _win_policy(initial=11, history=4, margin=1.5)
    seen, lines = [], []

    def fn(n, w):
        seen.append((n, w))
        if w < 13:
            return "bad", ES.WindowOverflow(0, 4, 5000, w)
        return "good", None

    got = ES.run_episode(stream, "episode 0", fn, log=lines.append,
                         window_policy=win)
    assert got == "good"
    # The stream bin never moved; the window bin went straight to what the
    # overflowing length needed.
    assert [n for n, _ in seen] == [15, 15]
    assert [w for _, w in seen] == [11, 13]
    assert stream.log2 == 15
    assert win.log2 == 13
    assert len(lines) == 1
    assert lines[0].startswith("[delta-window] bin 2^11 -> 2^13")


def test_a_stream_overflow_repeats_the_episode_one_stream_bin_up_and_leaves_the_window_bin_alone():
    stream = ES.BinPolicy(8, history=4, margin=2.0, cap=24)
    win = _win_policy(initial=12, history=4, margin=1.5)
    seen, lines = [], []

    def fn(n, w):
        seen.append((n, w))
        if n < 9:
            return "bad", ES.StreamOverflow(1, 3, 301, n)
        return "good", None

    assert ES.run_episode(stream, "episode 0", fn, log=lines.append,
                          window_policy=win) == "good"
    assert [w for _, w in seen] == [12, 12]
    assert win.log2 == 12
    assert stream.log2 == 9
    assert len(lines) == 1
    assert lines[0].startswith("[episode-stream] bin 2^8 -> 2^9")


def test_an_episode_that_would_overflow_both_bins_repeats_twice_and_ends_with_both_bins_raised_once():
    """One raise wins, that bin moves, the episode repeats; the other trips
    on the repeat. At most two repeats, with no joint reasoning."""
    stream = ES.BinPolicy(8, history=4, margin=2.0, cap=24)
    win = _win_policy(initial=11, history=4, margin=1.5)
    seen, lines = [], []

    def fn(n, w):
        seen.append((n, w))
        # The WINDOW is checked first, exactly as the driver checks it.
        if w < 12:
            return "bad", ES.WindowOverflow(0, 4, 3000, w)
        if n < 9:
            return "bad", ES.StreamOverflow(0, 4, 301, n)
        return "good", None

    assert ES.run_episode(stream, "episode 0", fn, log=lines.append,
                          window_policy=win) == "good"
    assert seen == [(8, 11), (8, 12), (9, 12)]
    assert win.log2 == 12 and stream.log2 == 9
    assert len(lines) == 2
    assert lines[0].startswith("[delta-window]")
    assert lines[1].startswith("[episode-stream]")


def test_the_window_bin_raises_at_the_cap_rather_than_growing_past_max_delta_tokens():
    win = _win_policy(initial=15, cap=15)
    with pytest.raises(ES.EpisodeStreamCapReached):
        win.bump(15, 40000)


def test_a_window_overflow_without_a_window_policy_is_a_wiring_error_not_a_silent_loop():
    stream = ES.BinPolicy(8, history=4, margin=2.0, cap=24)

    def fn(n):
        return "bad", ES.WindowOverflow(0, 1, 5000, 12)

    with pytest.raises(ValueError) as exc:
        ES.run_episode(stream, "episode 0", fn, log=lambda _l: None)
    assert "window policy" in str(exc.value)


# ------------------------------------------------------- 4. the equivalence

def _fold_rows(window, count, chunk=1024):
    """`extend_fold`'s accumulated rows over one delta, at `window`."""
    rng = np.random.RandomState(11)
    toks = np.zeros((window,), np.uint8)
    toks[:count] = rng.randint(1, 250, size=count).astype(np.uint8)

    def fold(acc, rows, valid, off):
        del off
        w = jnp.asarray(valid, jnp.float32)
        return (acc[0] + jnp.sum(rows * w[:, None], axis=0),
                acc[1] + jnp.sum(w))

    carry, acc = FOLD.extend_fold(
        StubAgent(), jnp.zeros(()), jnp.asarray(toks),
        jnp.asarray(count, jnp.int32),
        window=window, chunk=chunk,
        init_acc=(jnp.zeros((E,), jnp.float32), jnp.zeros((), jnp.float32)),
        fold_fn=fold)
    return np.asarray(carry), np.asarray(acc[0]), np.asarray(acc[1])


@pytest.mark.parametrize("count", [0, 1, 700, 1024, 3890])
def test_a_chunked_extend_at_a_smaller_window_bin_returns_bit_identical_rows_to_the_full_window(
        count):
    """THE CLAIM THE GATE RESTS ON.

    A window at or above the chunk only adds or removes chunks whose every
    token is invalid. `_step` freezes the carry and emits a zero row there,
    and every consumer is a weighted sum with the weight from `valid`, so
    the removed chunks contributed exactly +0.0. Bit for bit, not close.
    """
    big = _fold_rows(32768, count)
    small = _fold_rows(4096, count)
    for a, b in zip(big, small):
        assert np.array_equal(a, b), (count, a, b)


def test_advance_at_a_smaller_window_bin_gives_bit_identical_vertex_memory():
    """The same claim one level up, through `carry_stream.advance` -- the
    call the loss makes per sample per K step."""
    total_v, count = 5, 2900
    body = np.random.RandomState(3).randint(
        1, 250, size=count).astype(np.uint8)

    def run(window):
        t = np.zeros((window,), np.uint8)
        t[:count] = body
        vs, vc = CS.zero_memory(total_v, E)
        part = np.zeros((vs.shape[0] - 1,), np.float32)
        part[2] = 1.0
        return CS.advance(
            StubAgent(), jnp.zeros(()), vs, vc,
            jnp.asarray(t), jnp.asarray(count, jnp.int32),
            jnp.asarray(2, jnp.int32),
            window=window, participants=jnp.asarray(part))

    a = [np.asarray(x) for x in run(32768)]
    b = [np.asarray(x) for x in run(4096)]
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_edge_write_ids_at_a_smaller_window_bin_agrees_with_the_full_window_prefix():
    """`_edge_write_ids` is O(window) and its output is a vector.

    Binning it takes it from 32768 int32 per step per sample to 4096. Every
    entry past the delta is -1, which is the scatter's trash segment, so the
    smaller vector is the larger one's prefix.
    """
    from alphagrad.approx.ppo import _edge_write_ids

    counts = jnp.asarray([300, 220, 0, 0], jnp.int32)
    heads = jnp.asarray([10, 12, 0, 0], jnp.int32)
    slots = jnp.asarray([3, 1, -1, -1], jnp.int32)
    n_faces = jnp.asarray(2, jnp.int32)
    n_delta = jnp.asarray(700, jnp.int32)
    big = np.asarray(_edge_write_ids(counts, heads, slots, n_faces,
                                     n_delta, 32768))
    small = np.asarray(_edge_write_ids(counts, heads, slots, n_faces,
                                       n_delta, 4096))
    assert big.shape == (32768,) and small.shape == (4096,)
    assert np.array_equal(big[:4096], small)
    assert np.all(big[4096:] == -1)


# ----------------------------------------------------- 5. shapes and wiring

def _small_env(delta_window=0):
    """A four-equation env on the delta-observation path."""
    import jax
    from alphagrad.approx.env import EnvConfig, VertexEliminationEnv

    def fn(x, y):
        return jnp.tanh(jnp.sin(x) @ y) + jnp.exp(jnp.sin(x) @ y)

    args = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
    cj = jax.make_jaxpr(fn)(*args)
    cfg = EnvConfig(
        jaxpr=cj.jaxpr, argnums=(0, 1), has_aux=False, sparse=False,
        cmp_type="flops", mem_type="peak_memory",
        terminal_rewards_only=True, delta_obs=True,
        delta_window=int(delta_window))
    return VertexEliminationEnv(cfg, args, list(cj.literals))


def test_the_env_state_delta_buffer_follows_the_config_window_and_not_the_module_constant():
    from alphagrad.approx.env import MAX_DELTA_TOKENS

    default = _small_env()
    assert default.delta_window == MAX_DELTA_TOKENS
    assert default.reset().delta_tokens.shape == (MAX_DELTA_TOKENS,)
    binned = _small_env(4096)
    assert binned.delta_window == 4096
    assert binned.reset().delta_tokens.shape == (4096,)


def test_the_callback_output_shape_follows_the_env_obs_width_and_the_wire_arity_does_not_change():
    """DECISION 0.2: the WIRE stays at the cap.

    The Ray measurement pool preallocates at `obs_width` once per run, so a
    bin that moved the wire would mean tearing the actor pool down on every
    bin change. The bin applies from `step`'s slice of the wire onwards.
    """
    from alphagrad.approx.env import DELTA_HEADER_SLOTS, MAX_DELTA_TOKENS

    wide = DELTA_HEADER_SLOTS + MAX_DELTA_TOKENS
    for w in (0, 4096):
        env = _small_env(w)
        assert env.obs_width == wide
        assert env._callback_shape[0].shape == (wide,)
        assert env.wire_arity == 2


def test_two_envs_that_differ_only_in_their_delta_window_have_different_treedefs():
    """That is the retrace. The config rides in the pytree's AUX data, so
    `eqx.filter_jit` re-traces the rollout without anything being told."""
    import jax

    a = jax.tree_util.tree_structure(_small_env(0))
    b = jax.tree_util.tree_structure(_small_env(4096))
    assert a != b


def test_with_delta_window_carries_the_remote_measurement_pool_across_the_copy():
    """Losing the pool here would move every measurement back into the
    driver, in silence."""
    env = _small_env(0)
    sentinel = object()
    import dataclasses

    env = type(env)(
        env.config, env.args, env.consts, env.valid_vertices, env.num_envs,
        env.eval_args_samples,
        axis_state_static=env.axis_state_static,
        axis_valid_static=env.axis_valid_static,
        remote_pool=sentinel, remote_timeout_s=17.5)
    del dataclasses
    copy = env.with_delta_window(4096)
    assert copy.delta_window == 4096
    assert copy._remote_pool is sentinel
    assert copy._remote_timeout_s == 17.5
    assert copy.valid_vertices == env.valid_vertices
    assert copy.num_envs == env.num_envs
    # And the round trip is the identity on everything else.
    from alphagrad.approx.env import MAX_DELTA_TOKENS
    back = copy.with_delta_window(MAX_DELTA_TOKENS)
    assert back.delta_window == MAX_DELTA_TOKENS
    assert back._remote_pool is sentinel


def test_the_episode_stream_row_tail_shrinks_with_the_window_bin():
    """A second, free saving that arrives with the same edit.

    The tail is one write window plus whatever the fold pads it up to, so
    passing the BIN instead of the cap takes it from 32768 to the bin.
    """
    assert ES.stream_tail(32768) == 32768
    assert ES.stream_tail(4096) == 4096
    # At the small bin the Helmholtz smoke runs, the row is nearly halved.
    assert ES.stream_length(15, 32768) == 65536
    assert ES.stream_length(15, 4096) == 36864
    # And the reader's window must match the row it was handed.
    assert ES.validate_window_against_row(4096, 36864) == 15
    with pytest.raises(ValueError) as exc:
        ES.validate_window_against_row(32768, 36864)
    assert "must be the same number" in str(exc.value)


def test_a_live_face_chunk_longer_than_the_window_bin_rides_out_with_its_raw_count():
    """THE SILENT CLAMP IS GONE (design risk 7.2).

    `ppo.py`'s `ct_eff = jnp.minimum(ct_f, W - off)` truncated a face's
    chunk when the concatenation buffer filled. At the 32768 cap it was
    effectively dead; at a binned window it is reachable, and a truncated
    chunk changes the TOKENS THE HEAD READS, hence the action. The host
    callback now cuts the BUFFER to the bin and leaves the COUNT raw, so
    the rollout sees `sum(counts) > bin` and the driver repeats the episode
    one window bin up.
    """
    from alphagrad.approx.common.face_driver import fit_chunk_to_window

    tok = (np.arange(32768, dtype=np.int64) % 249 + 1).astype(np.uint8)
    out, cnt = fit_chunk_to_window(tok, 9000, 4096)
    assert out.shape == (4096,)
    assert cnt == 9000, "the count must NOT be truncated with the buffer"
    assert np.array_equal(out, tok[:4096])
    # A chunk that fits is copied whole and the rest of the buffer is zero.
    out2, cnt2 = fit_chunk_to_window(tok, 300, 4096)
    assert cnt2 == 300
    assert np.array_equal(out2[:300], tok[:300])
    assert not np.any(out2[300:])
    # At the cap the buffer is the stream's own and is passed through.
    out3, cnt3 = fit_chunk_to_window(tok, 300, 32768)
    assert out3 is tok and cnt3 == 300


# --------------------------------------------------------------- 6. the size

def test_the_window_bin_at_four_thousand_and_ninety_six_pays_a_quarter_of_the_iterations_and_the_bytes_of_the_cap():
    """WHAT THE BIN IS FOR, as a table. At the shipped fold chunk (1024):

        W       outer fold iterations   rows bytes/sample/K (embd 128)
        32768   32                      16 777 216
        8192     8                       4 194 304
        4096     4                       2 097 152

    The measured transformer worst case is 3890 tokens, so at W = 4096 the
    LIVE chunk count is 4 and nothing is wasted at all.
    """
    chunk, embd = 1024, 128
    table = {}
    for w in (32768, 8192, 4096):
        C, nb, padded = FOLD.plan_chunks(w, chunk)
        table[w] = (nb, w * embd * 4)
        assert C == chunk and padded == w
    assert table[32768] == (32, 16777216)
    assert table[8192] == (8, 4194304)
    assert table[4096] == (4, 2097152)
    # A factor of eight in both, and the live chunks at the measured 3890
    # exactly fill the 4096 bin.
    assert table[32768][0] // table[4096][0] == 8
    assert table[32768][1] // table[4096][1] == 8
    assert -(-3890 // chunk) == 4
    # The row is smaller too, at every stream bin.
    assert (ES.stream_length(19, 32768) - ES.stream_length(19, 4096)
            == 32768 - 4096)
