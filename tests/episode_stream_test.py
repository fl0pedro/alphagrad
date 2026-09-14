# -*- coding: utf-8 -*-
"""THE EPISODE TOKEN STREAM: one uint8 row per environment per episode.

Design: `.scratch/trustworthy-approx-search/episode-stream-design.md`
(owner rulings 2026-09-13, Q1 binned / Q3 dynamic slice / Q4 per-step
transport).

Six things are pinned here.

1. THE BIN AND ITS ARITHMETIC. The first `n`, the override, the hard cap,
   the row length and the tail the write window needs.
2. THE HOST CHECK. A step that would pass `2^n` raises, and the message
   names the environment, the step, the length and the bin.
3. THE BIN THE DRIVER CHOOSES. The selection rule, a shrinking episode
   history picking the smaller bin again, the repeat one bin up on an
   overflow, and a raise instead of a runaway at the cap.
4. THE WRITE. A rollout of known deltas lands at the right offsets and the
   row IS their concatenation, byte for byte.
5. THE READ. The chunked read from the stream at an offset equals the read
   from the standalone window it replaced, value for value -- which is the
   whole claim that the loss computes the same numbers as before.
6. THE SIZE. What the change was for.

The face replay's half of (5) needs a real rollout through the live-faces
path, so it lives with that rollout harness in
`tests/one_stream_claim_test.py`.

This module pins no scale (`ALPHAGRAD_MAX_DELTA_TOKENS` / `MAX_FACES`), so
it is safe to collect beside anything -- the size case states the
transformer width as arithmetic instead of freezing the interpreter at it.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp                                           # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402

# EVERY ALPHAGRAD IMPORT HERE IS LAZY, ON PURPOSE. This module sorts FIRST
# in tests/, and importing `alphagrad.approx` at module scope freezes
# env.py's MAX_DELTA_TOKENS / MAX_FACES at COLLECTION time -- which turns
# every later module's scale pin into a dead write (the config guard says so
# out loud, and then that module measures under a width it did not choose).
# Nothing here needs a scale, so nothing here takes one, and nothing here
# imports before the first test body runs.
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
_FOLD = _LazyModule("alphagrad.approx.common.delta_fold")


def extend_fold(*a, **k):
    return _FOLD.extend_fold(*a, **k)


def plan_chunks(*a, **k):
    return _FOLD.plan_chunks(*a, **k)


def sum_reducer(*a, **k):
    return _FOLD.sum_reducer(*a, **k)


E = 4


class StubAgent:
    """`delta_fold_test`'s stub: a real carried recurrence, cheap rows.

    The read under test is the plumbing -- which tokens reach which chunk at
    which offset -- not the encoder, which is the shipped `encode_extend`
    either way. A carry that every token moves is what makes a
    misplaced chunk boundary impossible to match by accident.
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


# ------------------------------------------------------- 1. the bin itself

def test_the_first_bin_is_the_step_budget_times_the_episode_rounded_up_over_eight():
    """The rule the flag help quotes, at the transformer width.

    32768 tokens a step times 95 steps is 3 112 960; the next power of two
    is 2^22; divided by eight that is 2^19 -- the `n` the design note's size
    table is written at.
    """
    assert ES.default_log2(32768, 95) == 19
    # The rule is stated once and the constant is what the help prints.
    assert "rounded UP to the next power of two" in ES.FIRST_BIN_RULE
    assert "divided by 8" in ES.FIRST_BIN_RULE


@pytest.mark.parametrize("budget,steps,want", [
    (32768, 95, 19),          # the transformer width
    (32768, 1, 12),           # 2^15 rounded up, over 8
    (1024, 64, 13),           # exactly 2^16, over 8
    (1025, 64, 14),           # one token past 2^16 -> 2^17, over 8
])
def test_the_first_bin_rounds_the_worst_case_up_and_never_down(budget, steps,
                                                               want):
    assert ES.default_log2(budget, steps) == want


def test_the_environment_variable_overrides_the_measured_first_bin(monkeypatch):
    monkeypatch.setenv(ES.LOG2_ENV, "21")
    assert ES.resolve_log2(32768, 95) == 21
    monkeypatch.delenv(ES.LOG2_ENV)
    assert ES.resolve_log2(32768, 95) == 19


def test_a_configured_bin_above_the_hard_cap_is_refused_not_clamped(monkeypatch):
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "20")
    monkeypatch.setenv(ES.LOG2_ENV, "21")
    with pytest.raises(ValueError) as exc:
        ES.resolve_log2(32768, 95)
    assert ES.LOG2_MAX_ENV in str(exc.value)


def test_a_bin_that_is_not_an_integer_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setenv(ES.LOG2_ENV, "nineteen")
    with pytest.raises(ValueError):
        ES.resolve_log2(32768, 95)


def test_the_row_is_the_bin_plus_a_tail_a_full_window_write_fits_in():
    """The tail is why the write needs no branch on the count: a full
    `MAX_DELTA_TOKENS` window may start at any cursor below `2^n`."""
    n, W = 19, 32768
    tail = ES.stream_tail(W)
    assert tail >= W
    assert ES.stream_length(n, W) == (1 << n) + tail


def test_the_tail_also_covers_the_folds_padded_window_not_only_the_write():
    """`dynamic_slice` CLAMPS out of range instead of raising, so a row one
    chunk short would return SHIFTED tokens in silence. The tail is the
    larger of the write window and the fold's padded window."""
    W = 32768
    for chunk in (1024, 3000, 7, W):
        _C, _nb, padded = plan_chunks(W, chunk)
        assert ES.stream_tail(W, chunk) >= padded
        assert ES.stream_tail(W, chunk) >= W


def test_a_row_that_is_not_a_power_of_two_plus_the_tail_is_refused():
    W = 32768
    good = ES.stream_length(19, W)
    assert ES.log2_of_row(good, W) == 19
    with pytest.raises(ValueError):
        ES.log2_of_row(good + 1, W)
    with pytest.raises(ValueError):
        ES.log2_of_row(ES.stream_tail(W), W)


# --------------------------------------------- 2. the DEVICE overflow check

def test_a_step_that_stays_inside_the_bin_writes_at_its_own_cursor():
    off, end, over = ES.write_offset(100, 10, 8)          # bin 256
    assert int(off) == 100
    assert int(end) == 110
    assert not bool(over)


def test_a_step_that_would_pass_the_bin_raises_the_flag_and_clamps_itself():
    """The check is on the END of the write, not its start: a cursor that
    still fits but a delta that does not is exactly the overflow.

    Nothing raises. The offset is CLAMPED to the bin, so the write still
    lands inside the row -- the row has a full window of tail after the bin
    -- and the flag is what the driver acts on.
    """
    off, end, over = ES.write_offset(250, 10, 8)
    assert bool(over)
    assert int(end) == 260
    # The cursor itself is still inside the bin, so the write starts where it
    # always would; the row's tail is what makes it land in the row.
    assert int(off) == 250
    # A LATER step of the same doomed attempt starts past the bin, and THAT
    # is what the clamp is for: without it the write would run off the row
    # and `dynamic_update_slice` would shift it in silence.
    off, end, over = ES.write_offset(400, 10, 8)
    assert bool(over)
    assert int(off) == 256              # clamped to the bin, inside the row
    assert int(end) == 410
    # A write that lands EXACTLY on the bin is legal: the bin is a length.
    off, end, over = ES.write_offset(246, 10, 8)
    assert not bool(over)
    assert int(off) == 246 and int(end) == 256


def test_the_carry_keeps_the_first_overflow_of_the_episode_not_the_last():
    """The log line must name the step the episode actually went wrong at."""
    seen_len = jnp.zeros((), jnp.int32)
    seen_step = jnp.zeros((), jnp.int32)
    # Step 3 overflows at 300, step 7 would overflow at 400.
    seen_len, seen_step = ES.carry_overflow(
        seen_len, seen_step, jnp.asarray(300, jnp.int32),
        jnp.asarray(True), 3)
    seen_len, seen_step = ES.carry_overflow(
        seen_len, seen_step, jnp.asarray(400, jnp.int32),
        jnp.asarray(True), 7)
    assert int(seen_len) == 300
    assert int(seen_step) == 3


def test_no_overflow_leaves_the_carry_at_zero():
    """Zero length IS the "nothing yet" sentinel: a real overflowing length
    is above the bin, so it can never be zero."""
    out_len, out_step = ES.carry_overflow(
        jnp.zeros((), jnp.int32), jnp.zeros((), jnp.int32),
        jnp.asarray(110, jnp.int32), jnp.asarray(False), 4)
    assert int(out_len) == 0 and int(out_step) == 0


def test_the_host_reads_the_flags_back_as_one_record_naming_all_four():
    """The environment, the step, the length and the bin, from the two
    per-environment arrays the rollout returns."""
    over = ES.overflow_from(np.asarray([0, 0, 260], np.int32),
                            np.asarray([0, 0, 17], np.int32), 8)
    assert over is not None
    assert over.env_index == 2         # the row, which IS the environment
    assert over.step == 17
    assert over.length == 260
    assert over.log2 == 8
    text = str(over)
    for fragment in ("environment 2", "step 17", "260", "2^8", "256",
                     ES.LOG2_ENV):
        assert fragment in text, (fragment, text)


def test_an_episode_that_did_not_overflow_reports_nothing():
    assert ES.overflow_from(np.zeros((4,), np.int32),
                            np.zeros((4,), np.int32), 8) is None


def test_the_host_read_refuses_a_length_and_step_pair_of_different_widths():
    with pytest.raises(ValueError):
        ES.overflow_from(np.zeros((3,), np.int32), np.zeros((2,), np.int32),
                         8)


# ----------------------------------------- 3. the bin the driver CHOOSES

def test_the_bin_with_no_history_yet_is_the_first_bin():
    p = ES.BinPolicy(19, history=4, margin=2.0)
    assert p.pick() == 19
    assert p.pick() == 19


def test_the_bin_moves_up_at_once_when_the_history_no_longer_fits_it():
    """UP IS IMMEDIATE. One episode that does not fit is enough, and the
    move is one power of two."""
    p = ES.BinPolicy(17, history=4, margin=2.0)
    p.record(100_000)
    # 100 000 x 2 = 200 000, which does not fit 2^17 = 131 072.
    assert p.pick() == 18
    for fragment in ("UP", "DOWN", ES.HISTORY_ENV, ES.MARGIN_ENV):
        assert fragment in ES.SELECTION_RULE


def test_the_bin_does_not_move_down_on_a_partial_history_window():
    """THE HYSTERESIS (review finding 1). A run at the default first bin
    used to fall five doublings off ONE toy episode, and every move is a
    retrace and a recompile of the rollout, the loss and the optimiser.
    Down needs the WHOLE window to agree."""
    p = ES.BinPolicy(15, history=8, margin=2.0)
    p.record(40)                    # a five-step toy graph
    assert p.pick() == 15           # NOT 2^10
    for _ in range(6):
        p.record(40)
    assert len(p.recent) == 7
    assert p.pick() == 15           # still one episode short of the window
    p.record(40)
    assert p.pick() == 14           # the window is full: one step down


def test_an_episode_history_that_shrinks_makes_the_driver_pick_the_smaller_bin():
    """THE POINT OF THE RULING. The bin is chosen, not only grown: a run
    whose deltas shrink drifts back DOWN, and the program for the smaller
    bin is already compiled, so the switch costs nothing. It just takes one
    episode per power of two instead of all of them at once."""
    p = ES.BinPolicy(19, history=3, margin=2.0)
    for _ in range(3):
        p.record(300_000)
    big = p.pick()
    assert big == 20                       # 600 000 -> 2^20, one step up

    # Three shorter episodes push the long ones out of the window. The
    # target is 2^16 (40 000), and the walk down is one power of two per
    # episode with the window staying full.
    for _ in range(3):
        p.record(20_000)
    seen = [p.pick()]
    for _ in range(6):
        p.record(20_000)
        seen.append(p.pick())
    assert seen == [19, 18, 17, 16, 16, 16, 16]
    assert seen[-1] < big

    # And it comes back up the moment a long episode reappears.
    p.record(300_000)
    assert p.pick() == 17


def test_one_long_episode_inside_the_window_keeps_the_bin_up():
    """The choice is the MAXIMUM over the window, not the mean, so a single
    short episode between two long ones cannot shrink the bin under them."""
    p = ES.BinPolicy(20, history=4, margin=2.0)
    p.record(300_000)
    p.record(1_000)
    p.record(1_000)
    p.record(1_000)
    assert p.pick() == 20


def test_the_window_and_the_margin_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(ES.HISTORY_ENV, "2")
    monkeypatch.setenv(ES.MARGIN_ENV, "1.0")
    p = ES.BinPolicy(19)
    assert p.window == 2
    assert p.margin == 1.0
    p.record(300_000)
    assert p.pick() == 19                  # 300 000 -> 2^19, no margin
    p.record(1_000)
    p.record(1_000)
    # The long one fell out of the 2-episode window, so the target is 2^10,
    # and the walk down is one power of two per episode.
    assert [p.pick()] + [p.pick() for _ in range(9)] == [
        18, 17, 16, 15, 14, 13, 12, 11, 10, 10]


def test_a_margin_below_one_is_refused_because_it_asks_for_an_overflow(monkeypatch):
    monkeypatch.setenv(ES.MARGIN_ENV, "0.5")
    with pytest.raises(ValueError) as exc:
        ES.BinPolicy(19)
    assert ES.MARGIN_ENV in str(exc.value)


def test_a_history_window_below_one_episode_is_refused(monkeypatch):
    monkeypatch.setenv(ES.HISTORY_ENV, "0")
    with pytest.raises(ValueError):
        ES.BinPolicy(19)


def test_a_history_that_asks_for_more_than_the_cap_raises_it_does_not_clamp(
        monkeypatch):
    """A silent clamp at the cap (review finding 8) only moved the failure
    to the overflow that followed, and named the wrong cause in the log."""
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "18")
    p = ES.BinPolicy(17)
    p.record(10_000_000)
    with pytest.raises(ES.EpisodeStreamCapReached) as exc:
        p.pick()
    assert ES.LOG2_MAX_ENV in str(exc.value)


@pytest.mark.parametrize("length,want", [
    (0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (1024, 10), (1025, 11),
])
def test_the_smallest_power_of_two_holding_a_length(length, want):
    assert ES.log2_for_length(length) == want


def test_growing_the_bin_adds_exactly_one_power_of_two():
    assert ES.grow(19) == 20
    assert ES.grow(20) == 21


def test_growth_stops_at_the_hard_cap_with_a_raise(monkeypatch):
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "20")
    assert ES.grow(19) == 20
    with pytest.raises(ES.EpisodeStreamCapReached) as exc:
        ES.grow(20)
    assert ES.LOG2_MAX_ENV in str(exc.value)


def test_the_driver_repeats_an_overflowing_episode_one_bin_up_and_records_it():
    """The overflow half of the ruling: log one line, re-run THAT episode at
    the next larger bin, and record the length that overflowed so the next
    episode's choice already knows about it.

    The episode REPORTS the overflow, it does not raise it. That is review
    finding 2: a raise inside a `pure_callback` does not survive XLA, so the
    driver used to recognise it by a substring of a jaxlib message.
    """
    p = ES.BinPolicy(19, history=4, margin=2.0)
    seen, lines = [], []

    def episode(n):
        seen.append(n)
        if n < 21:
            return ("attempt at %d" % n,
                    ES.StreamOverflow(env_index=1, step=7,
                                      length=(1 << n) + 5, log2=n))
        return "done at %d" % n, None

    out = ES.run_episode(p, "episode 3", episode, log=lines.append)

    assert out == "done at 21"
    # The episode ran again at each bin, in order, and only went up.
    assert seen == [19, 20, 21]
    assert len(lines) == 2
    # ONE line, and it names the old bin, the new bin, the episode, the
    # environment and the length.
    for line in lines:
        assert "\n" not in line
    assert "2^19 -> 2^20" in lines[0] and "episode 3" in lines[0]
    assert "environment 1" in lines[0]
    assert str((1 << 19) + 5) in lines[0]
    assert "2^20 -> 2^21" in lines[1]
    # Both overflowing lengths are in the history, so the NEXT episode does
    # not start back at a bin that cannot hold them.
    assert list(p.recent) == [(1 << 19) + 5, (1 << 20) + 5]
    assert p.pick() == 22


def test_the_repeat_goes_straight_to_the_bin_the_overflowing_length_needs():
    """One repeat, not one repeat per doubling. The length that overflowed
    is a MEASUREMENT, so the bump uses it; only the per-episode drift is
    limited to one power of two."""
    p = ES.BinPolicy(10, history=4, margin=2.0)
    seen = []

    def episode(n):
        seen.append(n)
        if n < 16:
            return None, ES.StreamOverflow(env_index=0, step=1,
                                           length=40_000, log2=n)
        return "done", None

    assert ES.run_episode(p, "episode 0", episode,
                          log=lambda _l: None) == "done"
    # 40 000 needs 2^16, so the repeat goes there in ONE step.
    assert seen == [10, 16]


def test_the_driver_hands_the_discarded_attempt_to_the_caller():
    """Review finding 5: a discarded attempt advanced host-side counters,
    and the caller is the only one that can roll them back."""
    p = ES.BinPolicy(19, history=4, margin=2.0)
    discarded = []

    def episode(n):
        if n == 19:
            return "attempt", ES.StreamOverflow(env_index=0, step=1,
                                                length=(1 << 19) + 1,
                                                log2=n)
        return "done", None

    assert ES.run_episode(p, "episode 0", episode, log=lambda _l: None,
                          on_discard=discarded.append) == "done"
    assert discarded == ["attempt"]


def test_the_driver_logs_one_line_when_the_bin_moves_without_an_overflow():
    """A bin change is a recompile the first time it happens, and an
    operator reading the log has no other way to see the drift. Down gets a
    line exactly as up does."""
    p = ES.BinPolicy(19, history=2, margin=2.0)
    lines = []

    def episode(n):
        return n, None

    # First episode: no previous bin, so nothing to report.
    assert ES.run_episode(p, "episode 0", episode, log=lines.append) == 19
    assert lines == []

    # Two short episodes fill the window and the bin steps down, and that is
    # one line.
    p.record(1_000)
    p.record(1_000)
    assert ES.run_episode(p, "episode 1", episode, log=lines.append) == 18
    assert len(lines) == 1
    assert "2^19 -> 2^18" in lines[0] and "episode 1" in lines[0]

    # A bin that does not move says nothing. 1000 x 2 = 2000 needs 2^11, so
    # the walk down stops there; what is pinned here is that a step which
    # does not move the bin is silent.
    for _ in range(9):
        p.record(1_000)
        ES.run_episode(p, "episode n", episode, log=lines.append)
    assert p.log2 == 11
    assert len(lines) == 8              # 2^19 -> 2^18, then seven more
    p.record(1_000)
    assert ES.run_episode(p, "episode last", episode,
                          log=lines.append) == 11
    assert len(lines) == 8


def test_the_driver_does_not_catch_exceptions_at_all():
    """A repeat loop that swallowed the wrong exception would retry a real
    bug until the cap and then report the cap instead of the bug. There is
    now no `except` in the driver, so nothing can be swallowed."""
    p = ES.BinPolicy(19, history=4, margin=2.0)

    def episode(_n):
        raise ValueError("something else entirely")

    with pytest.raises(ValueError):
        ES.run_episode(p, "episode 1", episode, log=lambda _l: None)


def test_the_driver_does_not_swallow_the_cap(monkeypatch):
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "20")
    p = ES.BinPolicy(20, history=4, margin=2.0)

    def episode(n):
        return None, ES.StreamOverflow(env_index=0, step=0,
                                       length=(1 << n) + 1, log2=n)

    with pytest.raises(ES.EpisodeStreamCapReached):
        ES.run_episode(p, "episode 0", episode, log=lambda _l: None)


# ------------------------------------------------------------- 4. the write

def _known_deltas(counts, window, seed=0):
    """One `(W,)` uint8 window per step, zero-padded past its count."""
    rng = np.random.RandomState(seed)
    out = []
    for c in counts:
        w = np.zeros((window,), np.uint8)
        w[:c] = rng.randint(1, 250, size=c).astype(np.uint8)
        out.append(w)
    return out


def test_the_rollout_writes_each_delta_at_the_cursor_and_advances_by_its_count():
    """THE STREAM IS THE DELTAS CONCATENATED, byte for byte.

    The cursor after step t is the sum of the counts up to t, and the span
    `[offset_t, offset_t + count_t)` holds step t's delta exactly.
    """
    from alphagrad.approx.ppo import _ep_stream_write

    W, n = 64, 8
    counts = [10, 0, 33, 1, 20]
    windows = _known_deltas(counts, W, seed=3)

    stream = jnp.zeros((ES.stream_length(n, W),), jnp.uint8)
    cursor, offsets = 0, []
    for w, c in zip(windows, counts):
        offsets.append(cursor)
        stream = _ep_stream_write(stream, cursor, jnp.asarray(w), c)
        cursor += c

    row = np.asarray(stream)
    want = np.concatenate([w[:c] for w, c in zip(windows, counts)])
    assert cursor == int(np.sum(counts))
    assert np.array_equal(row[:cursor], want)
    # Every step's span is its own delta, and nothing else is in the row.
    for off, w, c in zip(offsets, windows, counts):
        assert np.array_equal(row[off:off + c], w[:c])
    assert not np.any(row[cursor:])


def test_the_write_masks_the_window_so_a_step_leaves_no_padding_behind():
    """A window carries `W - count` pad slots. Writing them unmasked would
    read the same (every reader is bounded by a count) but would not SAY the
    same, and the last step's tail would stay in the row."""
    from alphagrad.approx.ppo import _ep_stream_write

    W, n = 32, 6
    w = np.full((W,), 7, np.uint8)             # NOT zero-padded
    stream = jnp.zeros((ES.stream_length(n, W),), jnp.uint8)
    stream = _ep_stream_write(stream, 3, jnp.asarray(w), 5)
    row = np.asarray(stream)
    assert np.array_equal(row[3:8], np.full((5,), 7, np.uint8))
    assert not np.any(row[8:])
    assert not np.any(row[:3])


# ------------------------------ 4b. the overflow, through the real machinery

def _mini_rollout(counts, log2, window=64):
    """THE ROLLOUT'S EPISODE-STREAM MACHINERY, and nothing else.

    `ppo.rollout_fn` is a closure built inside `ppo.main` and cannot be
    called on its own (`tests/one_stream_claim_test.py` explains that at
    length). What IS reachable is the module-level helper its `step_fn`
    calls, `ppo._ep_stream_step`, and the whole overflow mechanism is that
    helper plus the scan carry. So this drives exactly that, under the same
    `jax.vmap` over environments and `lax.scan` over steps the rollout uses,
    with real counts. The end-to-end proof is the forced-overflow smoke on
    the cluster; this is the part a unit test can hold.

    Returns `(stream, cursor, overflow_length, overflow_step, offsets)`,
    each with a leading environment axis, exactly as the rollout returns
    them.
    """
    import jax
    from jax import lax

    from alphagrad.approx.ppo import _ep_stream_step

    cnt = np.asarray(counts, np.int32)                 # (E, T)
    n_env, n_step = cnt.shape
    rng = np.random.RandomState(5)
    toks = np.zeros((n_env, n_step, window), np.uint8)
    for e in range(n_env):
        for t in range(n_step):
            toks[e, t, :cnt[e, t]] = rng.randint(
                1, 250, size=int(cnt[e, t])).astype(np.uint8)
    length = ES.stream_length(log2, window)

    def one_env(tokens, counts_e):
        def step(carry, xs):
            stream, cur, ovl, ovs, t = carry
            tok, c = xs
            stream, off, cur, ovl, ovs = _ep_stream_step(
                stream, cur, tok, c, t, log2, ovl, ovs)
            return (stream, cur, ovl, ovs, t + 1), off

        init = (jnp.zeros((length,), jnp.uint8),) + (
            jnp.zeros((), jnp.int32),) * 4
        (stream, cur, ovl, ovs, _t), offs = lax.scan(
            step, init, (tokens, counts_e))
        return stream, cur, ovl, ovs, offs

    out = jax.vmap(one_env)(jnp.asarray(toks), jnp.asarray(cnt))
    return out + (toks, cnt)


def test_an_overflowing_rollout_reports_a_flag_and_the_driver_repeats_it():
    """THE REPLACEMENT FOR THE `__cause__` TEST (review finding 2).

    Nothing raises anywhere. The rollout runs to the END with a flag in its
    carry, the host reads the flag afterwards, and the driver records the
    length, logs ONE line and repeats the episode one bin up. The old test
    built a `__cause__` chain the runtime never produced.
    """
    W = 64
    # Environment 1 reaches 300 slots, past the 256-slot bin, at step 4.
    counts = [[40, 40, 40, 40, 40],
              [60, 60, 60, 60, 60]]
    policy = ES.BinPolicy(8, history=4, margin=2.0)
    seen, lines, discarded = [], [], []

    def attempt(n):
        seen.append(n)
        out = _mini_rollout(counts, n, window=W)
        return out, ES.overflow_from(out[2], out[3], n)

    result = ES.run_episode(policy, "episode 0", attempt, log=lines.append,
                            on_discard=lambda r: discarded.append(r))

    # THE REPEAT. 300 needs 2^9, so one repeat, not one per doubling.
    assert seen == [8, 9]
    assert len(discarded) == 1

    # THE ONE LINE, naming the old bin, the new bin, the episode, the
    # environment and the length.
    assert len(lines) == 1
    assert "\n" not in lines[0]
    for fragment in ("2^8 -> 2^9", "episode 0", "environment 1", "300"):
        assert fragment in lines[0], (fragment, lines[0])

    # THE RECORDED LENGTH is the one that overflowed, so the next episode's
    # choice already knows about it.
    assert list(policy.recent) == [300]

    # THE DISCARDED ATTEMPT stayed inside its row -- the write offset is
    # clamped to the bin, and the row has a full window of tail after it --
    # so nothing was corrupted outside the attempt that was thrown away.
    d_stream, d_cur, d_ovl, d_ovs = discarded[0][:4]
    assert d_stream.shape == (2, ES.stream_length(8, W))
    assert int(d_ovl[0]) == 0 and int(d_ovl[1]) == 300
    assert int(d_ovs[1]) == 4
    assert int(d_cur[1]) == 300

    # THE TRAJECTORY THAT IS ACTUALLY USED is the repeat's, rebuilt whole at
    # the larger bin, and it is the deltas concatenated byte for byte.
    stream, cursor, ovl, ovs, offs, toks, cnt = result
    assert stream.shape == (2, ES.stream_length(9, W))
    assert not np.any(np.asarray(ovl))
    assert not np.any(np.asarray(ovs))
    assert [int(x) for x in np.asarray(cursor)] == [200, 300]
    row = np.asarray(stream)
    for e in range(2):
        want = np.concatenate([toks[e, t, :cnt[e, t]]
                               for t in range(cnt.shape[1])])
        got = row[e, :int(cursor[e])]
        assert np.array_equal(got, want), e
        assert not np.any(row[e, int(cursor[e]):])
        for t in range(cnt.shape[1]):
            o = int(offs[e, t])
            assert np.array_equal(row[e, o:o + cnt[e, t]],
                                  toks[e, t, :cnt[e, t]])


def test_a_rollout_that_fits_reports_no_overflow_and_never_repeats():
    policy = ES.BinPolicy(8, history=4, margin=2.0)
    seen, lines = [], []

    def attempt(n):
        seen.append(n)
        out = _mini_rollout([[40, 40, 40]], n, window=64)
        return out, ES.overflow_from(out[2], out[3], n)

    ES.run_episode(policy, "episode 0", attempt, log=lines.append)
    assert seen == [8]
    assert lines == []


def test_a_discarded_attempt_leaves_no_plan_records_or_terminals_behind():
    """REVIEW FINDING 5. The device side of a discarded attempt vanishes on
    its own; the host side does not. An overflow at a late step used to
    DOUBLE-COUNT that episode's terminal plans, which is the artifact the
    trustworthiness claims are built on.

    After one overflow-and-repeat the plan log holds exactly ONE set of
    terminal records for that episode, and they are the repeat's.
    """
    from alphagrad.approx import env as ENV

    outer = ENV.episode_telemetry_snapshot()
    try:
        ENV._PLAN_RECORDS.clear()
        ENV._PLAN_LOG_TERMINALS[0] = 0
        ENV._TRUNCATED_PLANS[0] = 0

        policy = ES.BinPolicy(8, history=4, margin=2.0)
        held = {"attempt": 0}

        def attempt(n):
            # What the driver does at the top of every attempt.
            held["snapshot"] = ENV.episode_telemetry_snapshot()
            ENV.set_plan_log_attempt(held["attempt"])
            # What one episode's env callbacks do to this module.
            for e in range(3):
                ENV._record_plan({"env_index": e, "bin": n})
            ENV._PLAN_LOG_TERMINALS[0] += 3
            ENV._TRUNCATED_PLANS[0] += 1
            if n == 8:
                return "attempt", ES.StreamOverflow(env_index=1, step=4,
                                                    length=300, log2=n)
            return "done", None

        def discard(_result):
            ENV.episode_telemetry_restore(held["snapshot"])
            held["attempt"] += 1

        assert ES.run_episode(policy, "episode 0", attempt,
                              log=lambda _l: None,
                              on_discard=discard) == "done"

        drained = ENV.consume_plan_records()
        assert drained["terminals"] == 3
        assert len(drained["records"]) == 3
        # And they are the REPEAT's records, not the discarded attempt's.
        assert [r["bin"] for r in drained["records"]] == [9, 9, 9]
        # Every record says WHICH attempt it came from, so two attempts at
        # one episode are never two byte-identical records with nothing to
        # tell them apart (verification jobs 65410 and 65413).
        assert [r["attempt"] for r in drained["records"]] == [1, 1, 1]
        assert ENV.consume_truncated_plan_count() == 1
    finally:
        ENV.episode_telemetry_restore(outer)
        ENV.set_plan_log_attempt(0)


def test_a_plan_record_says_which_attempt_at_the_episode_it_came_from():
    """The stamp itself, without the driver around it."""
    from alphagrad.approx import env as ENV

    outer = ENV.episode_telemetry_snapshot()
    try:
        ENV._PLAN_RECORDS.clear()
        ENV.set_plan_log_attempt(0)
        ENV._record_plan({"what": "first try"})
        assert ENV.plan_log_attempt() == 0
        ENV.set_plan_log_attempt(2)
        ENV._record_plan({"what": "third try"})
        assert [r["attempt"] for r in ENV._PLAN_RECORDS] == [0, 2]
        # The stamp is NOT rolled back by a restore: it has to keep counting
        # across a discarded attempt, which is the one thing the rollback
        # must not undo.
        snap = ENV.episode_telemetry_snapshot()
        ENV.set_plan_log_attempt(5)
        ENV.episode_telemetry_restore(snap)
        assert ENV.plan_log_attempt() == 5
    finally:
        ENV.episode_telemetry_restore(outer)
        ENV.set_plan_log_attempt(0)


def test_the_snapshot_names_only_containers_this_module_still_defines():
    """A renamed accumulator that silently drops out of the snapshot is the
    quiet gap the snapshot exists to close, so env.py refuses to import
    with a stale name. This states the same thing from the outside."""
    from alphagrad.approx import env as ENV

    snap = ENV.episode_telemetry_snapshot()
    assert set(snap) == set(ENV._EPISODE_TELEMETRY_NAMES)
    assert "_PLAN_RECORDS" in snap and "_PLAN_LOG_TERMINALS" in snap
    with pytest.raises(ValueError):
        ENV.episode_telemetry_restore({"_PLAN_RECORDS": []})


# -------------------------------------------------------------- 5. the read

# THE CHUNKS ARE MULTIPLES OF 32. Under the shipped
# ALPHAGRAD_PALIMPSA_READ=fast a multi-block fold has to start every block on
# a multiple of 32 (see `delta_fold.plan_chunks`), and 48 is allowed because
# the window clamps it to a single block. The claim under test is about WHERE
# the reader takes its tokens from, not about the chunk width, and the counts
# still sweep 0, 1, 17, 37, 63 and the full window.
@pytest.mark.parametrize("window,count,chunk", [
    (64, 64, 32), (64, 0, 32), (64, 1, 32), (64, 63, 32), (64, 17, 32),
    (64, 64, 64), (64, 64, 128), (48, 37, 48),
])
def test_the_chunked_read_from_the_stream_equals_the_read_from_the_window(
        window, count, chunk):
    """THE BIT-IDENTITY CLAIM, at the level it is actually made.

    The loss reads chunk j of sample (e, t) as
    `dynamic_slice(ep_tokens[e], offset + j*C, C)` instead of reshaping a
    stored `(window,)` buffer. Same tokens, same order, same count bounding
    them -- so the carry and the fold accumulator must come out equal, not
    close.
    """
    agent = StubAgent()
    rng = np.random.RandomState(11)
    win = jnp.asarray(rng.randint(1, 9, size=window).astype(np.uint8))
    cnt = jnp.asarray(count, jnp.int32)
    init, fold = sum_reducer(E)

    c_ref, acc_ref = extend_fold(agent, 0.0, win, cnt, window=window,
                                 chunk=chunk, init_acc=init, fold_fn=fold)

    # The same window, written into a stream at a non-zero offset, inside a
    # two-environment stream so the row index has something to choose.
    n = 10
    L = ES.stream_length(n, window, chunk)
    off = 123
    row = np.zeros((2, L), np.uint8)
    row[1, off:off + window] = np.asarray(win)
    stream = jnp.asarray(row)

    c_str, acc_str = extend_fold(agent, 0.0, stream, cnt, window=window,
                                 chunk=chunk, init_acc=init, fold_fn=fold,
                                 start=off, row=1)

    assert np.array_equal(np.asarray(c_ref), np.asarray(c_str))
    for a, b in zip(acc_ref, acc_str):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_one_dimensional_stream_reads_the_same_span_as_a_row_of_a_batch():
    agent = StubAgent()
    window, chunk, count, off = 64, 32, 40, 77
    rng = np.random.RandomState(5)
    win = rng.randint(1, 9, size=window).astype(np.uint8)
    L = ES.stream_length(10, window, chunk)
    flat = np.zeros((L,), np.uint8)
    flat[off:off + window] = win
    init, fold = sum_reducer(E)

    c1, a1 = extend_fold(agent, 0.0, jnp.asarray(flat),
                         jnp.asarray(count, jnp.int32), window=window,
                         chunk=chunk, init_acc=init, fold_fn=fold, start=off)
    c2, a2 = extend_fold(agent, 0.0, jnp.asarray(flat[None, :]),
                         jnp.asarray(count, jnp.int32), window=window,
                         chunk=chunk, init_acc=init, fold_fn=fold,
                         start=off, row=0)
    assert np.array_equal(np.asarray(c1), np.asarray(c2))
    for a, b in zip(a1, a2):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_reader_refuses_a_stream_whose_rank_does_not_match_the_row():
    agent = StubAgent()
    init, fold = sum_reducer(E)
    win = jnp.zeros((64,), jnp.uint8)
    with pytest.raises(ValueError):
        extend_fold(agent, 0.0, win, jnp.asarray(8, jnp.int32), window=64,
                    chunk=32, init_acc=init, fold_fn=fold, start=0, row=0)
    with pytest.raises(ValueError):
        extend_fold(agent, 0.0, win[None, :], jnp.asarray(8, jnp.int32),
                    window=64, chunk=32, init_acc=init, fold_fn=fold, start=0)


def test_the_k_window_is_one_contiguous_span_of_the_stream():
    """What replaced the `--grad-window` gather.

    The gather materialised every step's window K times. The K deltas that
    end at step t are one contiguous span of the stream -- it starts at
    `delta_offset[t-K+1]` and runs `sum(count[t-K+1..t])` tokens -- so the K
    offsets and the K counts carry the same information, and the span holds
    exactly the K windows' live prefixes concatenated.
    """
    from alphagrad.approx.ppo import _ep_stream_write

    W, n, K = 64, 9, 3
    counts = [12, 5, 0, 31, 9, 17]
    windows = _known_deltas(counts, W, seed=8)

    stream = jnp.zeros((ES.stream_length(n, W),), jnp.uint8)
    cursor, offsets = 0, []
    for w, c in zip(windows, counts):
        offsets.append(cursor)
        stream = _ep_stream_write(stream, cursor, jnp.asarray(w), c)
        cursor += c
    row = np.asarray(stream)

    for t in range(len(counts)):
        idx = [max(0, min(len(counts) - 1, t - K + 1 + j)) for j in range(K)]
        live = [(t - K + 1 + j) >= 0 for j in range(K)]
        # The loss zeroes a clamped pre-episode entry's COUNT, which makes
        # its advance an exact no-op whatever its offset.
        k_counts = [counts[i] if lv else 0 for i, lv in zip(idx, live)]
        span_start = offsets[idx[0]]
        span_len = int(sum(k_counts))
        # The old gather, token for token: K windows truncated to K counts.
        old = np.concatenate(
            [windows[i][:c] for i, c in zip(idx, k_counts)]
        ) if span_len else np.zeros((0,), np.uint8)
        assert np.array_equal(row[span_start:span_start + span_len], old)
        # And each of the K reads is its own sub-span of that one span.
        at = span_start
        for i, c in zip(idx, k_counts):
            assert np.array_equal(row[at:at + c], windows[i][:c])
            at += c


# --------------------------------------------------------------- 6. the size

def test_the_token_storage_is_under_twenty_kilobytes_per_step_per_environment():
    """WHAT THE CHANGE WAS FOR, at the transformer width.

    MAX_FACES 1920, N 8, MAX_DELTA_TOKENS 32768, T 95 steps, n 19.

    BEFORE: two uint8 windows per STEP per environment, `delta_tokens` and
    `face_delta_tokens`, 32768 slots each -- 65 536 B a step, about ninety
    percent of it padding (a measured step emits ~3000 tokens).

    AFTER: three int32 spans a step (`delta_offset`, `face_offset`,
    `env_index`) plus two uint8 rows per EPISODE per environment, amortised
    over the episode's steps.
    """
    W, T, n = 32768, 95, 19
    before = 2 * W                                        # two uint8 windows
    row = ES.stream_length(n, W)
    per_step_spans = 3 * 4                                # three int32 leaves
    after = per_step_spans + (2 * row) / T

    assert before == 65536
    assert row == (1 << 19) + ES.stream_tail(W)
    assert after < 20 * 1024, after
    # The whole point: better than a factor of five, and it is the PADDING
    # that went, not any token.
    assert before / after > 5.0


def test_the_first_bin_holds_the_measured_episode_with_headroom_short_of_a_doubling():
    """MEASURED, not assumed. At ~3000 tokens a step a 95 step episode is
    285 000 slots against 2^19 = 524 288 -- a factor of 1.84, NOT the factor
    of 2 the design note's size section claims. A per-step length that
    actually doubled overflows and costs exactly one growth (2^20), which is
    what the growth path is for. Pinned here so the claim and the arithmetic
    cannot drift apart again."""
    n, T, measured = 19, 95, 3000
    used = measured * T
    assert used < (1 << n)
    assert 1.8 < (1 << n) / used < 1.9
    # A doubled per-step length does NOT fit, and the next bin does.
    assert 2 * used > (1 << n)
    assert 2 * used < (1 << (n + 1))


def test_the_stream_is_bytes_not_words():
    """One byte a slot is what makes the row affordable; the narrow-id
    branch is what made it legal (vocabulary 256)."""
    from alphagrad.approx.env import DELTA_TOKEN_DTYPE

    assert jnp.dtype(DELTA_TOKEN_DTYPE).itemsize == 1
    assert jnp.zeros((4,), DELTA_TOKEN_DTYPE).dtype == jnp.uint8
