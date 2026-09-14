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
driver picks one per episode from the recent history. The rule is
:data:`SELECTION_RULE` and it is HYSTERETIC: up is immediate, down needs the
whole history window to agree. A bin change is a retrace and a recompile of
the rollout, the loss and the optimiser update, so a bin that flips on one
measurement costs minutes at transformer width; a bin that is one power of
two too large costs only memory.

THERE ARE TWO BINS, AND THEY ARE CHOSEN TOGETHER (owner ruling
2026-09-14). The one above is the STREAM bin: how many slots one
environment's whole episode may write. The second is the WINDOW bin: how
long ONE STEP's delta may be. They bound different quantities and they
drift independently -- the stream length is roughly `T * mean(delta)`, the
window is `max(delta)` -- so they are two :class:`BinPolicy` objects with
two histories, and only the COMPILE KEY is joint. `ALPHAGRAD_MAX_DELTA_
TOKENS` is no longer the window: it is the HARD CAP and the width of the
host-to-device wire, and the window bin is the slice `env.step` takes out
of that wire (`EnvConfig.delta_window`). The window bin is where the loss's
`ceil(W / C)` outer fold iterations are paid, which is the whole reason it
exists. It has a FLOOR, the fold chunk, and going under the floor RAISES
rather than clamping -- see :class:`DeltaWindowFloor`.

`stream_tail`, `stream_length` and `log2_of_row` take the WINDOW BIN, not
the cap: the tail is sized by what one write actually is. Their parameter
is still called `max_delta_tokens` because every historical caller passed
the cap and the two were the same number.

OVERFLOW IS A DEVICE VALUE, NOT A RAISE (review finding 2, 2026-09-14). The
check `cursor + count <= 2^n` is arithmetic the rollout already has on the
device, so it is done there. A step that would pass the bin sets an
OVERFLOW FLAG in the scan carry and clamps its own write offset to `2^n`, so
every write and every later read still lands inside the row and nothing is
corrupted outside the attempt. The rollout RUNS TO THE END and hands the
flag, the offending length and the step back beside the used length. The
DRIVER reads the flag on the host after the rollout, records the length in
the bin history, logs ONE line, THROWS THE WHOLE ATTEMPT AWAY and repeats
the episode one bin up. Nothing is matched against a jaxlib message and
nothing raises inside a callback. `ALPHAGRAD_EPISODE_TOKENS_LOG2_MAX` is a
hard cap and a raise -- a runaway stream cannot fill the device in silence.
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

# THE SECOND BIN: THE PER-STEP DELTA WINDOW (owner ruling 2026-09-14).
#
# `ALPHAGRAD_MAX_DELTA_TOKENS` is the HARD CAP and the width of the
# host-to-device wire. It is NOT the window the rollout and the loss scan.
# That window is a bin, chosen per episode by the same rule as the stream
# bin, and it is where `ceil(W / C)` outer fold iterations are paid --
# 32 of them at the cap against 4 at 4096, for a measured worst case of
# 3890 tokens.
WIN_LOG2_ENV = "ALPHAGRAD_DELTA_WINDOW_LOG2"
WIN_LOG2_MAX_ENV = "ALPHAGRAD_DELTA_WINDOW_LOG2_MAX"
WIN_LOG2_MIN_ENV = "ALPHAGRAD_DELTA_WINDOW_LOG2_MIN"
WIN_HISTORY_ENV = "ALPHAGRAD_DELTA_WINDOW_HISTORY"
WIN_MARGIN_ENV = "ALPHAGRAD_DELTA_WINDOW_MARGIN"

# The chunk env vars the FLOOR is derived from. They are read here and not
# imported, because `delta_fold` and `ppo` read them at two different
# moments and the floor has to agree with both.
FOLD_CHUNK_ENV = "ALPHAGRAD_FOLD_CHUNK"
LOSS_EXTEND_CHUNK_ENV = "ALPHAGRAD_LOSS_EXTEND_CHUNK"
EXTEND_CHUNK_ENV = "ALPHAGRAD_EXTEND_CHUNK"

# The window bin's own history and margin. The margin is 1.5 and not the
# stream bin's 2.0 because the two bins measure different things: the
# stream length is roughly `T * mean(delta)` and drifts slowly, while the
# window is `max(delta)` over one episode and is heavy-tailed. 1.5 over the
# measured 3890 asks for 5835, i.e. the 8192 bin, which is the design's
# recommendation; lowering it to 1.05 would pick 4096 with five percent of
# headroom over a length already seen.
WIN_HISTORY_DEFAULT = 8
WIN_MARGIN_DEFAULT = 1.5

# THE FIRST WINDOW BIN (owner ruling 2026-09-14: 4096). With no history the
# driver has no measurement, so it starts where the owner said and lets the
# selection rule walk from there. An overflow on the first episode costs one
# repeat, and the repeat goes straight to the bin the overflowing length
# needs (`BinPolicy.bump`), so a wrong first guess costs one episode and not
# one episode per doubling.
WIN_LOG2_DEFAULT = 12
WIN_FIRST_BIN_RULE = (
    "4096 tokens (2^" + str(WIN_LOG2_DEFAULT) + "), the owner's ruling of "
    "2026-09-14, raised to " + WIN_LOG2_MIN_ENV + " when the fold chunk "
    "floor is larger and lowered to log2(ALPHAGRAD_MAX_DELTA_TOKENS) when "
    "the hard cap is smaller"
)

# THE WINDOW BIN'S SELECTION RULE. The same hysteresis as the stream bin,
# with one addition: a FLOOR. See :func:`window_floor_tokens`.
WIN_SELECTION_RULE = (
    "the window bin moves UP at once when the longest single-step delta (or "
    "per-step face concatenation) of the last " + WIN_HISTORY_ENV
    + " episodes (default " + str(WIN_HISTORY_DEFAULT) + ") times "
    + WIN_MARGIN_ENV + " (default " + str(WIN_MARGIN_DEFAULT) + ") no "
    "longer fits it, and DOWN only when that whole window is full and ALL "
    "of it fits the next smaller bin; either way by at most one power of "
    "two per episode, except after an overflow, which goes straight to the "
    "bin the overflowing length needs; it is never chosen below the fold "
    "chunk floor (" + WIN_LOG2_MIN_ENV + ") nor above "
    + WIN_LOG2_MAX_ENV + "; with no history yet it is the first bin"
)

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
#
# WHY HYSTERESIS. A bin is a SHAPE, so every move retraces and recompiles the
# rollout, the loss and the optimiser update. Moving on one measurement is
# what produced the observed `2^15 -> 2^10` five-doubling jump off a single
# toy episode. Up is cheap to be wrong about (one power of two of unused
# memory); down is expensive to be wrong about (an overflow, a discarded
# episode and a repeat). So they are not symmetric.
SELECTION_RULE = (
    "the bin moves UP at once when the longest of the last " + HISTORY_ENV
    + " episodes (default " + str(HISTORY_DEFAULT) + ") times " + MARGIN_ENV
    + " (default " + str(MARGIN_DEFAULT) + ") no longer fits it, and DOWN "
    "only when that whole window is full and ALL of it fits the next "
    "smaller bin; either way by at most one power of two per episode, "
    "except after an overflow, which goes straight to the bin the "
    "overflowing length needs; with no history yet it is the first bin"
)


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


class StreamOverflow:
    """ONE episode's overflow, as a plain record. NOT an exception.

    The rollout detects the overflow on the DEVICE and reports it as two
    int32 arrays (the offending length and the step, per environment). This
    is what the host turns them into: the environment, the step, the length
    and the bin, so the driver's one log line can name all four. Nothing
    raises, so nothing has to survive a `pure_callback` boundary and nothing
    is recognised by matching a runtime's message text.
    """

    __slots__ = ("env_index", "step", "length", "log2")

    def __init__(self, env_index, step, length, log2):
        self.env_index = int(env_index)
        self.step = int(step)
        self.length = int(length)
        self.log2 = int(log2)

    def __repr__(self):
        return (f"StreamOverflow(env_index={self.env_index}, "
                f"step={self.step}, length={self.length}, "
                f"log2={self.log2})")

    def __str__(self):
        return (
            f"environment {self.env_index} at step {self.step} would reach "
            f"length {self.length}, past the bin 2^{self.log2} = "
            f"{1 << self.log2} slots ({LOG2_ENV}={self.log2})"
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


class DeltaWindowFloor(ValueError):
    """A window bin was asked for below the fold chunk. It RAISES.

    WHY THIS IS NOT A CLAMP. `delta_fold.plan_chunks` does `C = min(C, W)`,
    so a window below the chunk changes the CHUNK SIZE. That regroups the
    floating-point partial sums inside the fold, and float32 addition is not
    associative, so the last bits of the loss move. A bin at or above the
    chunk only adds or removes chunks whose contribution is exactly `+0.0`,
    which is bit-exact. Clamping here would hide a configuration error
    behind a silent numerical change; raising names it.
    """

    def __init__(self, want_log2, floor_log2, floor_tokens, why):
        self.want_log2 = int(want_log2)
        self.floor_log2 = int(floor_log2)
        self.floor_tokens = int(floor_tokens)
        super().__init__(
            f"the per-step delta window bin 2^{self.want_log2} = "
            f"{1 << self.want_log2} tokens is below the fold chunk floor "
            f"2^{self.floor_log2} = {1 << self.floor_log2} tokens "
            f"(the floor is {self.floor_tokens} rounded up to a power of "
            f"two; {why}). A window below the chunk changes the chunk size "
            f"itself and regroups the loss's float32 partial sums, so it is "
            f"refused rather than clamped. Lower {FOLD_CHUNK_ENV} / "
            f"{LOSS_EXTEND_CHUNK_ENV} / {EXTEND_CHUNK_ENV}, or raise "
            f"{WIN_LOG2_ENV}."
        )


def window_floor_tokens(chunk=None) -> int:
    """The smallest window the bin may take, IN TOKENS, before rounding.

    `max(ALPHAGRAD_FOLD_CHUNK, ALPHAGRAD_LOSS_EXTEND_CHUNK or
    ALPHAGRAD_EXTEND_CHUNK)`. The first is the chunk `delta_fold` clamps to
    the window; the second is the chunk `Agent._extend_sequential` clamps
    against (`C >= W` switches it to the flat scan, a different reduction
    shape). Both are a chunk SIZE, and the window must stay above both.
    """
    from alphagrad.approx.common import delta_fold as _fold

    fold = int(_fold.default_chunk() if chunk is None else chunk)
    ext = _int_env(LOSS_EXTEND_CHUNK_ENV, 0) or _int_env(EXTEND_CHUNK_ENV, 0)
    return max(fold, int(ext), 1)


def window_floor_log2(chunk=None) -> int:
    """:func:`window_floor_tokens` rounded UP to a power of two, as `n`.

    `ALPHAGRAD_DELTA_WINDOW_LOG2_MIN` overrides it upwards only -- a
    configured floor BELOW the chunk is the situation the floor exists to
    prevent, so it raises.
    """
    derived = log2_for_length(window_floor_tokens(chunk))
    raw = os.environ.get(WIN_LOG2_MIN_ENV)
    if raw is None or raw == "":
        return derived
    given = _int_env(WIN_LOG2_MIN_ENV, derived)
    if given < derived:
        raise DeltaWindowFloor(
            given, derived, window_floor_tokens(chunk),
            f"{WIN_LOG2_MIN_ENV}={given} was set below it")
    return given


def window_log2_max(max_delta_tokens: int) -> int:
    """The window bin's hard cap: `ALPHAGRAD_MAX_DELTA_TOKENS` by default.

    The wire stays at the cap (design decision 0.2), so a window above it
    would index past the buffer `env.step` slices. `WIN_LOG2_MAX_ENV` may
    only lower it.
    """
    ceiling = log2_for_length(int(max_delta_tokens))
    raw = os.environ.get(WIN_LOG2_MAX_ENV)
    if raw is None or raw == "":
        return ceiling
    given = _int_env(WIN_LOG2_MAX_ENV, ceiling)
    if given > ceiling:
        raise ValueError(
            f"{WIN_LOG2_MAX_ENV}={given} is above the hard cap "
            f"log2(ALPHAGRAD_MAX_DELTA_TOKENS={int(max_delta_tokens)}) = "
            f"{ceiling}. The window bin slices the transport wire, so it "
            f"cannot be wider than the wire.")
    return given


def resolve_window_log2(max_delta_tokens: int, override=None,
                        chunk=None) -> int:
    """The FIRST window bin for this run. See :data:`WIN_FIRST_BIN_RULE`.

    In order: the caller's `override` (the `--delta-window-log2` flag, 0 or
    None meaning "not given"), then `ALPHAGRAD_DELTA_WINDOW_LOG2`, then the
    owner's 4096. The floor and the cap apply to all three, and the floor
    RAISES rather than clamping.
    """
    floor = window_floor_log2(chunk)
    cap = window_log2_max(max_delta_tokens)
    if cap < floor:
        raise DeltaWindowFloor(
            cap, floor, window_floor_tokens(chunk),
            f"the hard cap ALPHAGRAD_MAX_DELTA_TOKENS="
            f"{int(max_delta_tokens)} is itself below the chunk")
    raw = os.environ.get(WIN_LOG2_ENV)
    if override:
        n = int(override)
        if n <= 0:
            raise ValueError(
                f"the delta window bin must be a positive number of bits, "
                f"got {override}")
    elif raw is None or raw == "":
        n = min(max(WIN_LOG2_DEFAULT, floor), cap)
    else:
        n = _int_env(WIN_LOG2_ENV, 0)
    if n < floor:
        raise DeltaWindowFloor(
            n, floor, window_floor_tokens(chunk),
            "it was asked for explicitly")
    if n > cap:
        raise ValueError(
            f"the delta window bin 2^{n} is above the cap "
            f"{WIN_LOG2_MAX_ENV}={cap}")
    return n


class WindowOverflow:
    """ONE step's per-step WINDOW overflow, as a plain record.

    The sibling of :class:`StreamOverflow`, reported the same way: the
    rollout detects it on the DEVICE and hands back two int32 arrays, and
    the host turns them into this. `kind` says which of the two quantities
    the window bounds actually went over -- Q1, the length of one step's
    token delta, or Q3, the total of one step's face concatenation. They
    share one bin, so they bump the same policy, but an operator reading the
    one log line needs to know which one moved.

    The two records' `str()` deliberately share NO leading text with
    :class:`StreamOverflow`'s, so a window overflow can never be read as a
    stream overflow and bump the wrong bin.
    """

    __slots__ = ("env_index", "step", "length", "log2", "kind")

    DELTA = "delta"
    FACE = "face concatenation"

    def __init__(self, env_index, step, length, log2, kind=DELTA):
        self.env_index = int(env_index)
        self.step = int(step)
        self.length = int(length)
        self.log2 = int(log2)
        self.kind = str(kind)

    def __repr__(self):
        return (f"WindowOverflow(env_index={self.env_index}, "
                f"step={self.step}, length={self.length}, "
                f"log2={self.log2}, kind={self.kind!r})")

    def __str__(self):
        return (
            f"delta window overflow: environment {self.env_index} at step "
            f"{self.step} emitted a {self.kind} of {self.length} tokens, "
            f"past the window bin 2^{self.log2} = {1 << self.log2} tokens "
            f"({WIN_LOG2_ENV}={self.log2})"
        )


WINDOW_KIND_DELTA = 0
WINDOW_KIND_FACE = 1


def carry_window_overflow(seen_length, seen_step, seen_kind,
                          length, over, step, kind):
    """DEVICE side: keep the FIRST window overflow of the episode.

    The sibling of :func:`carry_overflow`, with one more carried value: Q1
    (one step's token delta) and Q3 (one step's face concatenation) share
    ONE window bin, so they fold into one record, and `kind` is what lets
    the driver's log line say which of the two actually went over.
    """
    import jax.numpy as jnp

    first = jnp.logical_and(over, seen_length <= 0)
    return (jnp.where(first, jnp.asarray(length, jnp.int32), seen_length),
            jnp.where(first, jnp.asarray(step, jnp.int32), seen_step),
            jnp.where(first, jnp.asarray(kind, jnp.int32), seen_kind))


def window_overflow_from(over_length, over_step, log2, over_kind=None):
    """HOST side: the per-environment window flags as one
    :class:`WindowOverflow`.

    The same shape as :func:`overflow_from`, and read at the same moment --
    once per episode, after the rollout. Zero length means that environment
    did not overflow. Returns the FIRST offending environment, or None.
    """
    lengths = np.atleast_1d(np.asarray(over_length)).reshape(-1)
    steps = np.atleast_1d(np.asarray(over_step)).reshape(-1)
    if lengths.shape != steps.shape:
        raise ValueError(
            f"window_overflow_from: {lengths.shape} lengths against "
            f"{steps.shape} steps")
    hit = np.nonzero(lengths > 0)[0]
    if hit.size == 0:
        return None
    e = int(hit[0])
    kind = WindowOverflow.DELTA
    if over_kind is not None:
        kinds = np.atleast_1d(np.asarray(over_kind)).reshape(-1)
        if kinds.shape != lengths.shape:
            raise ValueError(
                f"window_overflow_from: {lengths.shape} lengths against "
                f"{kinds.shape} kinds")
        if int(kinds[e]) == WINDOW_KIND_FACE:
            kind = WindowOverflow.FACE
    return WindowOverflow(
        env_index=e, step=int(steps[e]), length=int(lengths[e]),
        log2=int(log2), kind=kind)


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

    `max_delta_tokens` is REQUIRED and it is the READER's width, not the
    window's own. The two happen to be equal at the shipped scale; sizing
    the row from the window instead would make them diverge silently at any
    other scale, which is the exact failure :func:`stream_tail` exists to
    prevent.
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


def validate_window_against_row(window, row_length, where="", chunk=None):
    """The reader's window and the row it reads MUST agree. Raises if not.

    The row is `2^n + TAIL(window)`. A reader whose window is larger than
    the one the row was built for runs its last chunk off the end, and
    `dynamic_slice` CLAMPS an out-of-range start instead of raising, so it
    would return SHIFTED tokens in silence. This is the one place the two
    numbers meet, so it is checked here.
    """
    try:
        return log2_of_row(int(row_length), int(window), chunk)
    except ValueError as exc:
        raise ValueError(
            f"{where or 'episode stream'}: a reader at the window bin "
            f"{int(window)} was handed a {int(row_length)}-slot row, which "
            f"is not 2^n + {stream_tail(int(window), chunk)} for any n. The "
            f"window bin the loss reads at and the window bin the row's "
            f"tail was sized from must be the same number."
        ) from exc


def grow(log2: int) -> int:
    """One doubling, or a raise at the cap (owner ruling Q1)."""
    cap = log2_max()
    if int(log2) >= cap:
        raise EpisodeStreamCapReached(int(log2), cap)
    return int(log2) + 1


def log2_for_length(length) -> int:
    """The smallest `n` with `2^n >= length`. `n = 0` for an empty stream."""
    n = int(max(0, int(length)))
    if n <= 1:
        return 0
    return int(n - 1).bit_length()


# ---------------------------------------------------------------------------
# THE OVERFLOW CHECK, ON THE DEVICE (review finding 2).
# ---------------------------------------------------------------------------

def write_offset(cursor, count, log2):
    """DEVICE side: `(offset, end, over)` for one step's write.

    * `end = cursor + count` is where the cursor lands after this write.
    * `over` is `end > 2^n`, the overflow: this delta does not fit the bin.
    * `offset` is `min(cursor, 2^n)` -- CLAMPED, so the write is inside the
      row whatever happened. The row is `2^n + TAIL` slots and TAIL is at
      least a full window, so a write of a whole window at `2^n` still fits
      and `dynamic_update_slice` never has to clamp it silently. The stream
      of an overflowing attempt holds the wrong bytes, and that is fine: the
      DRIVER throws the whole attempt away.

    No host round trip and no raise. The old form asked the host for this
    offset through a `jax.pure_callback` purely so that a Python `raise`
    could happen before the write; the raise then had to survive XLA, which
    it did not (it arrived as a `JaxRuntimeError` recognised by a substring
    of a jaxlib message). The arithmetic here is the same arithmetic and it
    is where the numbers already are.
    """
    import jax.numpy as jnp

    bin_slots = jnp.asarray(1 << int(log2), jnp.int32)
    cur = jnp.asarray(cursor, jnp.int32)
    cnt = jnp.asarray(count, jnp.int32)
    end = cur + cnt
    return jnp.minimum(cur, bin_slots), end, end > bin_slots


def carry_overflow(seen_length, seen_step, end, over, step):
    """DEVICE side: keep the FIRST overflow of the episode in the carry.

    `seen_length` is 0 while nothing has overflowed -- a real overflowing
    length is above `2^n` and so above 0 for every bin, so 0 is an
    unambiguous "no". Keeping the FIRST one is what makes the log line name
    the step the episode actually went wrong at rather than the last step.
    """
    import jax.numpy as jnp

    first = jnp.logical_and(over, seen_length <= 0)
    return (jnp.where(first, end.astype(jnp.int32), seen_length),
            jnp.where(first, jnp.asarray(step, jnp.int32), seen_step))


def overflow_from(over_length, over_step, log2):
    """HOST side: the per-environment flags as one :class:`StreamOverflow`.

    `over_length` and `over_step` are the `(num_envs,)` int32 arrays the
    rollout returns. Zero length means that environment did not overflow.
    Returns the FIRST offending environment, or None when nothing
    overflowed. Reading these blocks on the rollout, which is one
    synchronisation per EPISODE -- the old form paid one host round trip per
    STEP for the same information.
    """
    lengths = np.atleast_1d(np.asarray(over_length)).reshape(-1)
    steps = np.atleast_1d(np.asarray(over_step)).reshape(-1)
    if lengths.shape != steps.shape:
        raise ValueError(
            f"overflow_from: {lengths.shape} lengths against "
            f"{steps.shape} steps")
    hit = np.nonzero(lengths > 0)[0]
    if hit.size == 0:
        return None
    e = int(hit[0])
    return StreamOverflow(env_index=e, step=int(steps[e]),
                          length=int(lengths[e]), log2=int(log2))


class BinPolicy:
    """WHICH BIN THE NEXT EPISODE COMPILES FOR. The rule lives here only.

    The bins are a small set of compiled programs, one per power of two, and
    this picks one per episode by :data:`SELECTION_RULE`. It goes DOWN as
    well as up, but NOT symmetrically, because a bin change is a recompile
    of the rollout, the loss and the optimiser update:

    * UP is immediate. The first episode whose recent maximum times the
      margin no longer fits the current bin moves it, by one power of two.
    * An OVERFLOW moves it as far as the overflowing length needs, at once,
      because that length is a measured hard requirement and stepping up one
      power of two at a time would discard one episode per step.
    * DOWN needs the WHOLE history window: `ALPHAGRAD_EPISODE_TOKENS_HISTORY`
      episodes must all have been recorded, and all of them must fit the
      next smaller bin with the margin. Then it moves by exactly one power
      of two. So the bin cannot fall five doublings off one toy episode,
      which is what a run at the default first bin used to do on its second
      rollout.

    `initial_log2` is what :meth:`pick` returns while the history is empty
    (`ALPHAGRAD_EPISODE_TOKENS_LOG2` when set, else the measured first bin).
    It is a STARTING POINT, not a floor: once real lengths are recorded, the
    measurement decides.
    """

    def __init__(self, initial_log2, history=None, margin=None, cap=None,
                 floor=0, history_env=HISTORY_ENV, margin_env=MARGIN_ENV,
                 cap_env=LOG2_MAX_ENV,
                 history_default=HISTORY_DEFAULT,
                 margin_default=MARGIN_DEFAULT):
        self.initial = int(initial_log2)
        if self.initial < 0:
            raise ValueError(
                f"the initial bin must be >= 0, got {self.initial}")
        self.history_env = str(history_env)
        self.margin_env = str(margin_env)
        self.cap_env = str(cap_env)
        self.window = (_int_env(self.history_env, history_default)
                       if history is None else int(history))
        if self.window < 1:
            raise ValueError(
                f"{self.history_env} must be at least 1 episode, got "
                f"{self.window}")
        self.margin = (_float_env(self.margin_env, margin_default)
                       if margin is None else float(margin))
        if self.margin < 1.0:
            # A margin below 1 asks for a bin SMALLER than a length already
            # seen, i.e. an overflow on every episode by construction.
            raise ValueError(
                f"{self.margin_env} must be at least 1.0, got {self.margin}")
        self.cap = log2_max() if cap is None else int(cap)
        # THE FLOOR. 0 for the stream bin, which has none: a stream row
        # shorter than the chunk is padded, not regrouped. The WINDOW bin
        # has one, and it is the fold chunk -- see :class:`DeltaWindowFloor`
        # for why going under it is a numerical change and not a saving.
        self.floor = int(floor)
        if self.floor < 0:
            raise ValueError(f"the bin floor must be >= 0, got {self.floor}")
        if self.floor > self.cap:
            raise ValueError(
                f"the bin floor 2^{self.floor} is above the cap "
                f"{self.cap_env}={self.cap}")
        if self.initial > self.cap:
            raise ValueError(
                f"the initial bin 2^{self.initial} is above the cap "
                f"{self.cap_env}={self.cap}")
        if self.initial < self.floor:
            raise ValueError(
                f"the initial bin 2^{self.initial} is below the floor "
                f"2^{self.floor}")
        self.recent = deque(maxlen=self.window)
        self.log2 = self.initial
        # The bin the LAST episode ran at, or None before the first one.
        # Only used to decide whether a bin change is worth a log line.
        self.last_used = None
        # Did the last recorded episode OVERFLOW? Set by
        # :meth:`record_overflow`, cleared by :meth:`pick`. It is the one
        # exception to the one-power-of-two-per-episode limit.
        self.overflowed = False

    def record(self, length) -> None:
        """Note one SUCCESSFUL episode's longest stream (both streams, all
        environments)."""
        self.recent.append(int(max(0, int(length))))

    def record_overflow(self, length) -> None:
        """Note the length that OVERFLOWED, and that it overflowed.

        The next :meth:`pick` may then move up by more than one power of two
        -- that length is a measurement, not a trend, and stepping towards
        it one doubling per episode would discard one episode per step.
        """
        self.recent.append(int(max(0, int(length))))
        self.overflowed = True

    def _target(self) -> int:
        """The bin the recent history asks for, cap CHECKED not clamped.

        The FLOOR is applied here, upwards, and is not an error: a history
        that fits inside the fold chunk is perfectly ordinary, and the bin
        simply sits on the floor. It is a configured bin BELOW the floor
        that raises (:class:`DeltaWindowFloor`), because that one is a
        request the apparatus cannot honour without moving the numbers.
        """
        need = math.ceil(max(self.recent) * self.margin)
        want = log2_for_length(need)
        if want > self.cap:
            # A silent clamp here just moves the failure to the overflow
            # that follows, and names the wrong cause in the log.
            raise EpisodeStreamCapReached(want, self.cap)
        return max(want, self.floor)

    def pick(self) -> int:
        """The bin for the next episode. See :data:`SELECTION_RULE`."""
        if not self.recent:
            self.log2 = self.initial
            self.overflowed = False
            return self.log2
        current = int(self.log2)
        target = self._target()
        if target > current:
            # UP, immediately. One power of two per episode, unless the last
            # episode actually overflowed, in which case go where the
            # measured length says.
            chosen = target if self.overflowed else current + 1
        elif target < current:
            # DOWN, but only on the WHOLE window and only one step.
            if len(self.recent) < self.window:
                chosen = current
            else:
                chosen = current - 1
        else:
            chosen = current
        self.log2 = min(max(chosen, self.floor), self.cap)
        self.overflowed = False
        return self.log2

    def bump(self, log2, length=None) -> int:
        """The bin for the REPEAT of an episode that overflowed.

        At least one doubling, and at least what `length` needs, so a bin
        that was several doublings too small costs one repeat rather than
        one repeat per doubling.
        """
        n = grow(log2)
        if length:
            n = max(n, log2_for_length(length))
        n = max(n, self.floor)
        if n > self.cap:
            raise EpisodeStreamCapReached(n, self.cap)
        self.log2 = n
        return self.log2


def run_episode(policy, what, fn, log=print, on_discard=None,
                window_policy=None):
    """Run ONE episode at the bin(s) the policies chose; repeat it a bin up
    on overflow.

    THE PAIR (owner ruling 2026-09-14). With `window_policy` given, the
    episode compiles for a PAIR: `fn(n_stream, n_window)`. The two policies
    are INDEPENDENT in their arithmetic and joint only in the compile key --
    the stream length is roughly `T * mean(delta)` and the window is
    `max(delta)`, so a run whose deltas become more uniform wants the stream
    bin flat and the window bin down. Each is chosen by its own history with
    its own hysteresis, and an overflow bumps ONLY the bin it belongs to: a
    :class:`WindowOverflow` moves the window bin, a :class:`StreamOverflow`
    moves the stream bin. An episode that would trip both trips one, repeats,
    then trips the other and repeats again -- at most two repeats, with no
    joint reasoning anywhere.

    `fn(n)` (or `fn(n_stream, n_window)`) runs the episode compiled for
    `2^n` and returns `(result, overflow)`, where `overflow` is None, a
    :class:`StreamOverflow` or a :class:`WindowOverflow`. NOTHING is raised
    and
    nothing is caught here: the rollout runs to the end whatever happened,
    and the overflow arrives as an ordinary device value the host reads
    afterwards. That is what makes the path independent of whether a
    callback's Python exception survives XLA, which it does not, and of
    asynchronous dispatch, which the exception form was never verified
    under.

    On an overflow: record the length in the history, log ONE line naming
    the old bin, the new bin, what was running, the environment and the
    length, THROW THE WHOLE RESULT AWAY, call `on_discard(result)` so the
    caller can roll back the host-side counters the discarded attempt
    advanced, and run the SAME episode again one bin up. A new bin is a new
    stream shape, so the caller's jit retraces by itself; a bin it has
    already compiled costs nothing to go back to.

    Nothing of the failed attempt survives on the device either: the caller
    rebinds the agent, the optimiser state and the env states only on
    RETURN, so the repeat starts from exactly the state the first attempt
    started from.

    The caller records the SUCCESSFUL episode's length itself
    (`policy.record(...)`), because only the caller sees the cursors the
    rollout came back with.
    """
    previous = policy.last_used
    n = policy.pick()
    policy.last_used = n
    if previous is not None and n != previous:
        # THE BIN MOVED WITHOUT AN OVERFLOW. One line, because a bin change
        # is a recompile the first time and an operator reading the log has
        # no other way to see the drift. Down is as normal as up here.
        log("[episode-stream] bin 2^%d -> 2^%d for %s (recent max %d slots, "
            "margin %g)"
            % (previous, n, what, max(policy.recent), policy.margin))
    w = None
    if window_policy is not None:
        w_prev = window_policy.last_used
        w = window_policy.pick()
        window_policy.last_used = w
        if w_prev is not None and w != w_prev:
            log("[delta-window] bin 2^%d -> 2^%d for %s (recent max %d "
                "tokens, margin %g)"
                % (w_prev, w, what, max(window_policy.recent),
                   window_policy.margin))
    while True:
        result, overflow = fn(n) if window_policy is None else fn(n, w)
        if overflow is None:
            return result
        if isinstance(overflow, WindowOverflow):
            if window_policy is None:
                raise ValueError(
                    "run_episode was handed a window overflow but no "
                    "window policy to move; pass window_policy= or stop "
                    "reporting one.")
            window_policy.record_overflow(overflow.length)
            old, w = w, window_policy.bump(w, overflow.length)
            window_policy.last_used = w
            log("[delta-window] bin 2^%d -> 2^%d, repeating %s: %s"
                % (old, w, what, overflow))
        else:
            policy.record_overflow(overflow.length)
            old, n = n, policy.bump(n, overflow.length)
            policy.last_used = n
            log("[episode-stream] bin 2^%d -> 2^%d, repeating %s: %s"
                % (old, n, what, overflow))
        if on_discard is not None:
            on_discard(result)
        result = None
