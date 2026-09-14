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

GROWTH. The host knows every environment's exact stream length after each
step, so the check is exact and it is one per step per environment rather
than one per token. A step that would pass `2^n` raises
:class:`EpisodeStreamOverflow` BEFORE the write. The driver catches it, logs
one line, raises `n` by one, recompiles and REPEATS the episode. Growth is
monotone within a run and stops at `ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX` with
a raise -- a runaway stream cannot fill the device in silence.
"""

from __future__ import annotations

import os

import numpy as np

# THE CONFIGURATION KNOBS, named once.
LOG2_ENV = "ALPHAGRAD_EPISODE_TOKENS_LOG2"
LOG2_MAX_ENV = "ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX"

# The hard cap's default. 2^24 slots is 16 MiB per environment per stream,
# which is already four doublings past the measured transformer width; a run
# that needs more is a configuration error, not a bigger bin.
LOG2_MAX_DEFAULT = 24

# THE FIRST BIN'S RULE, STATED ONCE (the flag help quotes this constant).
FIRST_BIN_RULE = (
    "MAX_DELTA_TOKENS times the episode length, rounded UP to the next "
    "power of two, divided by 8"
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


def resolve_log2(max_delta_tokens: int, episode_length: int) -> int:
    """`n` for this run: the env var if set, else :func:`default_log2`."""
    raw = os.environ.get(LOG2_ENV)
    if raw is None or raw == "":
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
