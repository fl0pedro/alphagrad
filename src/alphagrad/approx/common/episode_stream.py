"""THE EPISODE TOKEN STREAM: one uint8 row per environment per episode.

Design: `.scratch/trustworthy-approx-search/episode-stream-design.md`
(owner rulings 2026-09-13, Q1 binned / Q3 dynamic slice / Q4 per-step
transport).

WHAT THIS REPLACES. The rollout used to store one PADDED TOKEN WINDOW per
step and per environment -- `Trajectory.delta_tokens (MAX_DELTA_TOKENS,)`
and `Trajectory.face_delta_tokens (MAX_DELTA_TOKENS,)`. A step emits about
three thousand tokens into a window of 32768, so 90 percent of the storage
was padding, and the `--grad-window` K gather materialised every window K
times on top of that. Now each environment holds ONE row per episode, each
step writes its delta at a cursor, and the trajectory stores the `(offset,
count)` pair instead of the window.

THE BIN. The row is `2^n + TAIL` slots long. `2^n` is the BIN -- the static
episode budget, a power of two, a configuration value
(`ALPHAGRAD_EPISODE_TOKENS_LOG2`). `TAIL` is the write window: the rollout
writes a FULL `MAX_DELTA_TOKENS`-wide window at any cursor below `2^n` with
one `dynamic_update_slice`, so the row has to have that many slots after the
last legal cursor. That is what removes the branch on the count from the
write path. Tokens pad with 0 and NOTHING reads a padded slot: every read is
bounded by a stored count.

TAIL IS NOT EXACTLY `MAX_DELTA_TOKENS`, AND THAT IS DELIBERATE. The loss
reads the row in chunks of `delta_fold.plan_chunks`'s `C`, and that planner
pads the window UP to a whole number of chunks (`nb * C`). A read of the
last chunk of a delta written at the last legal cursor therefore touches
`2^n + nb*C` slots, not `2^n + MAX_DELTA_TOKENS`. `lax.dynamic_slice`
CLAMPS an out-of-range start instead of raising, so a row one chunk short
would silently return SHIFTED tokens for that read. With the shipped chunk
(1024) and the shipped budget (32768) the two numbers are equal and the
design's arithmetic is exact; `stream_tail()` keeps them equal for every
other pair.

THE BIN IS CHOSEN, NOT ONLY GROWN (owner clarification of 2026-09-14). The
bins are a SMALL SET OF COMPILED PROGRAMS, one per power of two, and the
driver picks one per episode. Before each rollout it takes the smallest bin
that holds the RECENT HISTORY: the longest episode stream of the last
`ALPHAGRAD_EPISODE_TOKENS_HISTORY` episodes, times
`ALPHAGRAD_EPISODE_TOKENS_MARGIN`. So a run drifts back DOWN to a smaller
bin when the deltas shrink, and moves up only when an episode needs it. The
compile per bin happens once, because jit keys on the static shape and the
persistent JAX compilation cache carries it across runs, so switching
between bins that are already compiled is free.

OVERFLOW. The host knows every environment's exact stream length after each
step, so the check is exact and it is one per step per environment rather
than one per token. A step that would pass `2^n` raises
:class:`EpisodeStreamOverflow` BEFORE the write. The driver catches it, logs
one line, RE-RUNS THAT EPISODE at the next larger bin, and records the length
that overflowed in the history, so the next episode's choice already knows
about it. `ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX` is a hard cap and a raise -- a
runaway stream cannot fill the device in silence.
"""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

# THE CONFIGURATION KNOBS, named once.
LOG2_ENV = "ALPHAGRAD_EPISODE_TOKENS_LOG2"
LOG2_MAX_ENV = "ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX"
HISTORY_ENV = "ALPHAGRAD_EPISODE_TOKENS_HISTORY"
MARGIN_ENV = "ALPHAGRAD_EPISODE_TOKENS_MARGIN"

# How many recent episodes the choice looks at, and how much room it leaves
# above the longest of them. Eight episodes is long enough that one short
# episode cannot shrink the bin under a run that is merely between two long
# ones, and short enough that a real downward drift is picked up within a
# few episodes. A margin of 2 is one doubling of headroom.
HISTORY_DEFAULT = 8
MARGIN_DEFAULT = 2.0

# The hard cap's default. 2^24 slots is 16 MiB per environment per stream,
# which is already four doublings past the measured transformer width; a run
# that needs more is a configuration error, not a bigger bin.
LOG2_MAX_DEFAULT = 24

# THE FIRST BIN'S RULE, STATED ONCE (the flag help quotes this constant).
FIRST_BIN_RULE = (
    "MAX_DELTA_TOKENS times the episode length, rounded UP to the next "
    "power of two, divided by 8"
)

# THE PER-EPISODE SELECTION RULE, STATED ONCE (the flag help quotes it too).
SELECTION_RULE = (
    "the smallest power of two that holds the longest episode stream of the "
    "last " + HISTORY_ENV + " episodes (default "
    + str(HISTORY_DEFAULT) + ") times " + MARGIN_ENV + " (default "
    + str(MARGIN_DEFAULT) + "); with no history yet it is the first bin"
)


class EpisodeStreamOverflow(Exception):
    """A step's delta would have been written past the bin.

    Raised on the HOST, before the write, by :func:`check_cursors`. The
    driver catches it, raises `n` by one and repeats the episode; nothing
    else may catch it, because a swallowed overflow is a silently corrupt
    trajectory (``dynamic_update_slice`` clamps, it does not raise).
    """

    def __init__(self, env_index, step, length, log2):
        self.env_index = int(env_index)
        self.step = int(step)
        self.length = int(length)
        self.log2 = int(log2)
        super().__init__(
            f"episode token stream overflow: environment {self.env_index} "
            f"at step {self.step} would reach length {self.length}, past the "
            f"bin 2^{self.log2} = {1 << self.log2} slots "
            f"({LOG2_ENV}={self.log2})"
        )


def _int_env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer number of bits, got {raw!r}"
        ) from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _float_env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{name} must be a number, got {raw!r}") from None


def log2_max() -> int:
    """The hard cap on `n` (`ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX`)."""
    return _int_env(LOG2_MAX_ENV, LOG2_MAX_DEFAULT)


def default_log2(max_delta_tokens: int, episode_length: int) -> int:
    """The FIRST bin when the env var is unset. See :data:`FIRST_BIN_RULE`.

    `MAX_DELTA_TOKENS` is the per-step transport width -- the worst case a
    single step may emit -- so `MAX_DELTA_TOKENS * episode_length` is the
    worst case for a whole episode. That bound is about an order of
    magnitude above what a real episode emits (a measured step is ~3000
    tokens against a 32768 budget), so the rule divides it by 8 and leaves
    the growth path to cover the rest.
    """
    if int(max_delta_tokens) <= 0:
        raise ValueError(
            f"max_delta_tokens must be positive, got {max_delta_tokens}")
    if int(episode_length) <= 0:
        raise ValueError(
            f"episode_length must be positive, got {episode_length}")
    worst = int(max_delta_tokens) * int(episode_length)
    # Round UP to the next power of two, then divide by 8 -- i.e. drop three
    # bits off the rounded exponent.
    bits = int(worst - 1).bit_length()
    return max(0, bits - 3)


def resolve_log2(max_delta_tokens: int, episode_length: int,
                 override=None) -> int:
    """`n` for this run.

    In order: the caller's `override` (the `--episode-tokens-log2` flag, 0
    or None meaning "not given"), then `ALPHAGRAD_EPISODE_TOKENS_LOG2`, then
    :func:`default_log2`. The cap applies to all three.
    """
    raw = os.environ.get(LOG2_ENV)
    if override:
        n = int(override)
        if n <= 0:
            raise ValueError(
                f"the episode stream bin must be a positive number of bits, "
                f"got {override}")
    elif raw is None or raw == "":
        n = default_log2(max_delta_tokens, episode_length)
    else:
        n = _int_env(LOG2_ENV, 0)
    cap = log2_max()
    if n > cap:
        raise ValueError(
            f"{LOG2_ENV}={n} is above the cap {LOG2_MAX_ENV}={cap}")
    return n


def stream_tail(max_delta_tokens: int, chunk=None) -> int:
    """Slots after the last legal cursor. See the module docstring.

    The write needs `MAX_DELTA_TOKENS`; the loss's chunked read needs the
    fold's PADDED window. The tail is the larger of the two, so neither a
    write nor a read can run off the row.
    """
    from alphagrad.approx.common import delta_fold as _fold

    _C, _nb, padded = _fold.plan_chunks(int(max_delta_tokens), chunk)
    return max(int(max_delta_tokens), int(padded))


def stream_length(log2: int, max_delta_tokens: int, chunk=None) -> int:
    """The static row length `2^n + TAIL`."""
    if int(log2) < 0:
        raise ValueError(f"log2 must be >= 0, got {log2}")
    return (1 << int(log2)) + stream_tail(max_delta_tokens, chunk)


def single_row(tokens, max_delta_tokens: int, chunk=None):
    """One standalone window as a ONE-ROW stream: ``(1, TAIL)``.

    For callers that have a window and want the stream reader -- the direct
    unit tests of ``_face_replay`` and ``advance``. The row is padded to
    :func:`stream_tail` for the reason that function exists: a read of the
    last chunk runs to the fold's PADDED window, and ``dynamic_slice``
    clamps rather than raising if the row is shorter.
    """
    import jax.numpy as jnp

    tail = stream_tail(max_delta_tokens, chunk)
    t = jnp.asarray(tokens).reshape(-1)
    if t.shape[0] > tail:
        raise ValueError(
            f"single_row: {t.shape[0]} tokens do not fit a {tail}-slot row")
    pad = tail - int(t.shape[0])
    if pad:
        t = jnp.concatenate([t, jnp.zeros((pad,), t.dtype)])
    return t[None, :]


def log2_of_row(row_length: int, max_delta_tokens: int, chunk=None) -> int:
    """`n` read back off a row this module built. Raises on any other row.

    The bin is a power of two BY CONTRACT (owner ruling Q1: grow by one
    doubling), so a row that is not `2^n + TAIL` is a caller error and not a
    smaller bin to be inferred.
    """
    tail = stream_tail(max_delta_tokens, chunk)
    bin_slots = int(row_length) - tail
    if bin_slots <= 0 or (bin_slots & (bin_slots - 1)) != 0:
        raise ValueError(
            f"episode stream row of {row_length} slots is not "
            f"2^n + {tail} for any n (the tail is "
            f"max(MAX_DELTA_TOKENS={int(max_delta_tokens)}, the fold's "
            f"padded window))"
        )
    return int(bin_slots).bit_length() - 1


def grow(log2: int) -> int:
    """One doubling, or a raise at the cap (owner ruling Q1)."""
    cap = log2_max()
    if int(log2) >= cap:
        raise EpisodeStreamCapReached(int(log2), cap)
    return int(log2) + 1


# THE MARKER THE RAISE IS RECOGNISED BY when its TYPE did not survive.
# `check_cursors` runs inside a `jax.pure_callback`, so its exception comes
# back out through XLA, and the runtime is free to wrap it (jaxlib raises
# its own error class with the original message attached). Matching on the
# class alone would make the growth path depend on a jaxlib detail, and a
# missed match is a CRASHED RUN instead of a grown bin.
OVERFLOW_MARKER = "episode token stream overflow"


def overflow_in(exc):
    """The overflow inside ``exc``, however the runtime wrapped it, or None.

    Looks for the typed exception anywhere in the `__cause__` /
    `__context__` chain first, and falls back to :data:`OVERFLOW_MARKER` in
    the text. Returns the exception to report, never a bare bool, so the
    caller logs the real message.
    """
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, EpisodeStreamOverflow):
            return cur
        cur = cur.__cause__ or cur.__context__
    if OVERFLOW_MARKER in str(exc):
        return exc
    return None


def log2_for_length(length) -> int:
    """The smallest `n` with `2^n >= length`. `n = 0` for an empty stream."""
    n = int(max(0, int(length)))
    if n <= 1:
        return 0
    return int(n - 1).bit_length()


class BinPolicy:
    """WHICH BIN THE NEXT EPISODE COMPILES FOR. The rule lives here only.

    The bins are a small set of compiled programs, one per power of two, and
    this picks one per episode by :data:`SELECTION_RULE`. It goes DOWN as
    readily as up: an episode history that shrinks picks the smaller bin
    again, and the program for it is already compiled, so the switch is
    free.

    `initial_log2` is what :meth:`pick` returns while the history is empty
    (`ALPHAGRAD_EPISODE_TOKENS_LOG2` when set, else the measured first bin).
    It is a STARTING POINT, not a floor: once real lengths are recorded, the
    measurement decides.
    """

    def __init__(self, initial_log2, history=None, margin=None, cap=None):
        self.initial = int(initial_log2)
        if self.initial < 0:
            raise ValueError(
                f"the initial bin must be >= 0, got {self.initial}")
        self.window = (_int_env(HISTORY_ENV, HISTORY_DEFAULT)
                       if history is None else int(history))
        if self.window < 1:
            raise ValueError(
                f"{HISTORY_ENV} must be at least 1 episode, got "
                f"{self.window}")
        self.margin = (_float_env(MARGIN_ENV, MARGIN_DEFAULT)
                       if margin is None else float(margin))
        if self.margin < 1.0:
            # A margin below 1 asks for a bin SMALLER than a length already
            # seen, i.e. an overflow on every episode by construction.
            raise ValueError(
                f"{MARGIN_ENV} must be at least 1.0, got {self.margin}")
        self.cap = log2_max() if cap is None else int(cap)
        if self.initial > self.cap:
            raise ValueError(
                f"the initial bin 2^{self.initial} is above the cap "
                f"{LOG2_MAX_ENV}={self.cap}")
        self.recent = deque(maxlen=self.window)
        self.log2 = self.initial

    def record(self, length) -> None:
        """Note one episode's LONGEST stream (both streams, all environments).

        Called on success with what the episode actually used, and on an
        overflow with the length that overflowed, so the next choice already
        knows about it.
        """
        self.recent.append(int(max(0, int(length))))

    def pick(self) -> int:
        """The bin for the next episode. See :data:`SELECTION_RULE`."""
        if not self.recent:
            self.log2 = self.initial
            return self.log2
        need = math.ceil(max(self.recent) * self.margin)
        self.log2 = min(log2_for_length(need), self.cap)
        return self.log2

    def bump(self, log2) -> int:
        """One doubling for the repeat of an episode that overflowed."""
        self.log2 = grow(log2)
        return self.log2


def run_episode(policy, what, fn, log=print):
    """Run ONE episode at the bin `policy` chose; repeat it a bin up on
    overflow.

    `fn(n)` runs the episode compiled for `2^n`. On
    :class:`EpisodeStreamOverflow`: record the length that overflowed, log
    ONE line with the old bin, the new bin and what overflowed, and run the
    SAME episode again one bin up. A new bin is a new stream shape, so the
    caller's jit retraces by itself; a bin it has already compiled costs
    nothing to go back to.

    Nothing of the failed attempt survives: the caller rebinds the agent,
    the optimiser state and the env states only on RETURN, so the repeat
    starts from exactly the state the first attempt started from. The host
    callbacks of the failed attempt did run, so their counters saw it.

    The caller records the SUCCESSFUL episode's length itself
    (`policy.record(...)`), because only the caller sees the cursors the
    rollout came back with.
    """
    n = policy.pick()
    while True:
        try:
            return fn(n)
        except Exception as exc:                       # noqa: BLE001
            inner = overflow_in(exc)
            if inner is None:
                raise
            length = getattr(inner, "length", None)
            if length:
                policy.record(length)
            old, n = n, policy.bump(n)
            log("[episode-stream] bin 2^%d -> 2^%d, repeating %s: %s"
                % (old, n, what, inner))


class EpisodeStreamCapReached(Exception):
    """The bin hit `ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX` and cannot grow."""

    def __init__(self, log2, cap):
        self.log2 = int(log2)
        self.cap = int(cap)
        super().__init__(
            f"episode token stream bin 2^{self.log2} overflowed and the cap "
            f"{LOG2_MAX_ENV}={self.cap} forbids growing it. Either the "
            f"deltas are far longer than the measurement said, or a stream "
            f"is not being reset per episode."
        )


def check_cursors(cursors, counts, step, log2):
    """HOST side: raise if any environment's next write passes the bin.

    `cursors` and `counts` carry a leading environment axis when the caller
    is inside the rollout's `vmap` (``pure_callback(vmap_method=
    "expand_dims")`` gives the host the whole batch, rows in env order, so
    the row index IS the environment). Returns the cursors UNCHANGED, and
    the caller must use the RETURNED value as the write offset: that data
    dependency is what puts the raise BEFORE the write.
    """
    cur = np.atleast_1d(np.asarray(cursors)).reshape(-1).astype(np.int64)
    cnt = np.atleast_1d(np.asarray(counts)).reshape(-1).astype(np.int64)
    if cur.shape != cnt.shape:
        raise ValueError(
            f"check_cursors: {cur.shape} cursors against {cnt.shape} counts")
    bin_slots = 1 << int(log2)
    ends = cur + cnt
    over = np.nonzero(ends > bin_slots)[0]
    if over.size:
        e = int(over[0])
        raise EpisodeStreamOverflow(
            env_index=e,
            step=int(np.asarray(step).reshape(-1)[0]),
            length=int(ends[e]),
            log2=int(log2),
        )
    return np.asarray(cursors)
