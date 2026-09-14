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
3. THE GROWTH. One doubling per overflow, the same episode run again, and
   a raise instead of a runaway at the cap.
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


# ----------------------------------------------- 2. the host overflow check

def test_a_step_that_stays_inside_the_bin_returns_its_cursors_unchanged():
    cur = np.asarray([0, 100, 7], np.int32)
    cnt = np.asarray([10, 10, 10], np.int32)
    out = ES.check_cursors(cur, cnt, 3, 8)          # bin 256
    assert np.array_equal(np.asarray(out), cur)


def test_a_step_that_would_pass_the_bin_raises_before_the_write():
    """The check is on the END of the write, not its start: a cursor that
    still fits but a delta that does not is exactly the overflow."""
    with pytest.raises(ES.EpisodeStreamOverflow):
        ES.check_cursors(np.asarray([250], np.int32),
                         np.asarray([10], np.int32), 4, 8)
    # A write that lands EXACTLY on the bin is legal: the bin is a length.
    ES.check_cursors(np.asarray([246], np.int32),
                     np.asarray([10], np.int32), 4, 8)


def test_the_overflow_message_names_the_environment_the_step_the_length_and_the_bin():
    with pytest.raises(ES.EpisodeStreamOverflow) as exc:
        ES.check_cursors(np.asarray([0, 0, 250], np.int32),
                         np.asarray([1, 1, 10], np.int32), 17, 8)
    err = exc.value
    assert err.env_index == 2          # the row, which IS the environment
    assert err.step == 17
    assert err.length == 260
    assert err.log2 == 8
    text = str(err)
    for fragment in ("environment 2", "step 17", "260", "2^8", "256",
                     ES.LOG2_ENV):
        assert fragment in text, (fragment, text)


def test_the_check_refuses_a_cursor_and_count_pair_of_different_widths():
    with pytest.raises(ValueError):
        ES.check_cursors(np.zeros((3,), np.int32), np.zeros((2,), np.int32),
                         0, 8)


# ------------------------------------------------------------ 3. the growth

def test_growing_the_bin_adds_exactly_one_power_of_two():
    assert ES.grow(19) == 20
    assert ES.grow(20) == 21


def test_growth_stops_at_the_hard_cap_with_a_raise(monkeypatch):
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "20")
    assert ES.grow(19) == 20
    with pytest.raises(ES.EpisodeStreamCapReached) as exc:
        ES.grow(20)
    assert ES.LOG2_MAX_ENV in str(exc.value)


def test_the_driver_grows_the_bin_by_one_power_of_two_and_repeats_the_episode():
    """The owner's rule end to end: catch, log one line, grow by one,
    run the SAME episode again -- with the new bin."""
    state = [19]
    seen, lines = [], []

    def episode(n):
        seen.append(n)
        if n < 21:
            raise ES.EpisodeStreamOverflow(env_index=1, step=7,
                                           length=(1 << n) + 5, log2=n)
        return "done at %d" % n

    out = ES.run_with_growth(state, "episode 3", episode, log=lines.append)

    assert out == "done at 21"
    # The episode ran again at each bin, in order, and only grew.
    assert seen == [19, 20, 21]
    assert state[0] == 21
    assert len(lines) == 2
    assert "2^19 -> 2^20" in lines[0] and "episode 3" in lines[0]
    assert "2^20 -> 2^21" in lines[1]


def test_the_driver_recognises_the_overflow_after_the_runtime_wrapped_it():
    """The raise happens inside a `jax.pure_callback`, so it comes back out
    through XLA and the runtime may wrap it in its own error class. The
    growth path must not depend on that class surviving."""
    state = [19]
    lines = []

    def episode(n):
        if n == 19:
            try:
                raise ES.EpisodeStreamOverflow(env_index=0, step=2,
                                               length=(1 << n) + 3, log2=n)
            except ES.EpisodeStreamOverflow as inner:
                raise RuntimeError("XlaRuntimeError: callback failed") \
                    from inner
        return "done"

    assert ES.run_with_growth(state, "episode 1", episode,
                              log=lines.append) == "done"
    assert state[0] == 20
    assert "2^19 -> 2^20" in lines[0]


def test_the_driver_recognises_the_overflow_by_its_text_alone():
    """Last resort: neither the class nor the chain survived, only the
    message. The marker is what the report then joins on."""
    state = [19]
    lines = []
    calls = []

    def episode(n):
        calls.append(n)
        if n == 19:
            raise RuntimeError(
                "jaxlib error: " + ES.OVERFLOW_MARKER
                + ": environment 0 at step 2 would reach length 600000")
        return "done"

    assert ES.run_with_growth(state, "episode 1", episode,
                              log=lines.append) == "done"
    assert calls == [19, 20]
    assert state[0] == 20


def test_the_driver_re_raises_anything_that_is_not_an_overflow():
    """A growth loop that swallowed the wrong exception would retry a real
    bug until the cap and then report the cap instead of the bug."""
    state = [19]

    def episode(_n):
        raise ValueError("something else entirely")

    with pytest.raises(ValueError):
        ES.run_with_growth(state, "episode 1", episode, log=lambda _l: None)
    assert state[0] == 19


def test_the_driver_does_not_swallow_the_cap(monkeypatch):
    monkeypatch.setenv(ES.LOG2_MAX_ENV, "20")
    state = [20]

    def episode(n):
        raise ES.EpisodeStreamOverflow(env_index=0, step=0,
                                       length=(1 << n) + 1, log2=n)

    with pytest.raises(ES.EpisodeStreamCapReached):
        ES.run_with_growth(state, "episode 0", episode, log=lambda _l: None)


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


# -------------------------------------------------------------- 5. the read

@pytest.mark.parametrize("window,count,chunk", [
    (64, 64, 16), (64, 0, 16), (64, 1, 16), (64, 63, 16), (64, 17, 16),
    (64, 64, 64), (64, 64, 7), (48, 37, 32),
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
    window, chunk, count, off = 64, 16, 40, 77
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
                    chunk=16, init_acc=init, fold_fn=fold, start=0, row=0)
    with pytest.raises(ValueError):
        extend_fold(agent, 0.0, win[None, :], jnp.asarray(8, jnp.int32),
                    window=64, chunk=16, init_acc=init, fold_fn=fold, start=0)


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
