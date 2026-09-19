from __future__ import annotations

import gc
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.random as jrand
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

import numpy as np

from alphagrad.approx.common import carry_plan as _carry
from alphagrad.approx.common.relations import compute_eqn_ids_from_tokens
from alphagrad.approx.common.token_vocab import (
    DELTA_HEADER_SLOTS,
    DELTA_TOKEN_DTYPE,
    DELTA_TOKEN_MAX,
    DELTA_TOKEN_PAD,
    INCR_TOKEN_VOCAB_DEFAULT,
    check_delta_ids as _check_delta_ids,
    encode_delta_header,
    incr_token_vocab,
)
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from graphax.jaxpr import get_vocab as _graphax_get_vocab
from graphax.sparse.micro_actions import (
    COMPRESS_KINDS, QUANT_DTYPES, Compress, Diag, Quant,
)
try:
    from jax_memory_monitor import ResourceMonitor as _RealResourceMonitor
except (ImportError, AttributeError):
    _RealResourceMonitor = None


class _NoopResourceMonitor:
    """Drop-in replacement for ``jax_memory_monitor.ResourceMonitor`` that
    does nothing — used to isolate the C++ ``MemoryTracker`` from the
    rest of the reward harness during memory-leak experiments.

    Each real ``ResourceMonitor`` constructs a fresh
    ``xla_mem_bridge.MemoryTracker`` (and a ``TimeTracker``) at
    ``__init__`` and tears them down at ``__exit__``. The C++ destructors
    are reachable, but if they don't release every allocation the tracker
    held during its lifetime, each io_callback-scoped instance leaks a
    little — ~18 MB / call empirically, × 192 callbacks/episode = ~3.4
    GB/ep. Activate this stub by setting
    ``ALPHAGRAD_DISABLE_RESOURCE_MONITOR=1`` to confirm or rule out
    that hypothesis; ``peak`` returns 0 and ``stats`` has all-zero
    entries, so the reward harness silently records ``peak_memory=0`` for
    the run — fine for a leak-hunt, not for production reward shaping.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    @property
    def peak(self) -> int:
        return 0

    @property
    def duration(self) -> float:
        return 0.0

    @property
    def stats(self) -> dict:
        return {"time": 0.0, "memory": 0.0}


ResourceMonitor = (
    _NoopResourceMonitor
    if (os.environ.get("ALPHAGRAD_DISABLE_RESOURCE_MONITOR", "0") == "1"
        or _RealResourceMonitor is None)
    else _RealResourceMonitor
)

import math as _math

# Cache the graphax vocabulary used by `compute_eqn_ids_from_tokens` — the
# tokenizer always uses the same digit_base, so the vocab is constant and
# rebuilding it on every callback is pure overhead.
_TOKEN_VOCAB, _, _ = _graphax_get_vocab()

# LEGACY full-stream observation width. NOT a token budget any more.
#
# THE TOTAL-STREAM CAP ``ALPHAGRAD_MAX_TOKENS`` IS GONE. palimpsa is a
# RECURRENT linear-attention encoder: its entire state is a fixed
# ``(L, H, d, d)`` carry -- 1536 floats at the flagship's (3, 2, 16, 16) --
# however long the stream is. The base stream is consumed ONCE into that carry
# (``VertexEliminationEnv.base_observation``) and every step after it is an
# EXTEND by that step's DELTA, so the full stream is never materialised and a
# cap on its total length buys nothing.
#
# It was worse than nothing. ``stream[:MAX_TOKENS]`` keeps the OLDEST tokens,
# so once the buffer saturates the observation stops advancing and the policy
# reads a frozen prefix for the rest of the episode -- the delta shrinks to
# zero while the graph keeps changing. JAX needs static shapes, so a bound
# survives only on the per-step DELTA buffer (``MAX_DELTA_TOKENS`` below);
# that is the only place a bound belongs.
#
# What survives here is the width of the LEGACY full-stream observation
# (``EnvConfig.delta_obs=False``: the ``extract_jaxpr`` re-tokenize path, and
# the non-delta drivers that still import this symbol -- gfn, gdpo, mu0,
# alpha0, az_gumbel, ppo_ray_worker, pretrain, cpu_approx_worker), plus the
# radix of the legacy scalar action encoding
# ``sp_type * MAX_TOKENS + target_vertex``. approx/ppo.py's live path does not
# read it at all. This is the split ppo.py already made for the absolute
# positional table (``ALPHAGRAD_POS_ENC_LEN``): a legacy buffer gets its own
# knob so no live component depends on a deleted budget.
LEGACY_STREAM_TOKENS = int(
    os.environ.get("ALPHAGRAD_LEGACY_STREAM_TOKENS", "4096"))
if LEGACY_STREAM_TOKENS < 256:
    raise ValueError("ALPHAGRAD_LEGACY_STREAM_TOKENS must be >= 256, got "
                     f"{LEGACY_STREAM_TOKENS}")
# The legacy drivers and the action radix still spell it ``MAX_TOKENS``.
MAX_TOKENS = LEGACY_STREAM_TOKENS
if os.environ.get("ALPHAGRAD_MAX_TOKENS") is not None:
    # HARD ERROR, not a silent ignore: three smoke scripts set this knob, and
    # a knob that no longer does what its name says is how a run gets
    # mis-read. Never silent.
    raise RuntimeError(
        "ALPHAGRAD_MAX_TOKENS is GONE -- there is no total-stream token "
        "budget any more. The encoder is a recurrence: the base stream is "
        "consumed once and extended by per-step deltas, so the full stream is "
        "never materialised. On the live delta path this knob did nothing; on "
        "the legacy path it clipped the OLDEST tokens. Use "
        "ALPHAGRAD_MAX_DELTA_TOKENS for the per-step delta buffer (the only "
        "bound that is real), or ALPHAGRAD_LEGACY_STREAM_TOKENS if you really "
        "are running the legacy full-stream observation.")

# Per-process tokenization-truncation telemetry. ``_callback`` writes
# here whenever the un-truncated jaxpr token sequence exceeds
# ``MAX_TOKENS`` (so the slice on the next line is lossy). The first
# occurrence inside a process emits a ``warnings.warn`` so the user
# notices in stderr; subsequent occurrences are silent but counted.
#
# Three quantities are tracked because each answers a different
# question:
#   * ``count``           — HOW OFTEN was the clip lossy this period?
#   * ``max_observed_len``— HOW BIG was the largest jaxpr, in tokens?
#   * ``overflow_sum``    — HOW MUCH info did we throw away this
#                           period? (= sum of ``raw_len - MAX_TOKENS``)
#
# ``consume_tokenization_truncation_stats`` returns these as a
# delta-since-last-poll and resets them. Drivers poll once per rollout
# so the wandb values land as PER-EPISODE numbers (not cumulative
# across the run — wandb itself does the time-series aggregation).
# Lists (not bare ints) because Python rebinding inside ``_callback``
# would shadow a module-level int.
_TOKENIZATION_TRUNCATION_COUNT: list[int] = [0]
_TOKENIZATION_TRUNCATION_MAX_LEN: list[int] = [0]
_TOKENIZATION_TRUNCATION_OVERFLOW_SUM: list[int] = [0]
_TOKENIZATION_TRUNCATION_WARNED: list[bool] = [False]


# Token-length telemetry for EVERY step, not just the truncating ones. The
# truncation counters answer "did we lose information?"; these answer "how big
# is one palimpsa call?", which is what sizes the batched kernel's static
# buffer. Kept as running sum/max/count so the per-episode mean is exact
# without holding every sample.
_TOKLEN_SUM: list[int] = [0]
_TOKLEN_MAX: list[int] = [0]
_TOKLEN_COUNT: list[int] = [0]
_DELTALEN_SUM: list[int] = [0]
_DELTALEN_MAX: list[int] = [0]
_DELTALEN_COUNT: list[int] = [0]


def _record_token_length(raw_len: int) -> None:
    """Record one full-stream tokenization length."""
    _TOKLEN_SUM[0] += int(raw_len)
    _TOKLEN_COUNT[0] += 1
    _dist_add("stream_len", raw_len)
    if raw_len > _TOKLEN_MAX[0]:
        _TOKLEN_MAX[0] = int(raw_len)


def _record_delta_length(delta_len: int) -> None:
    """Record one per-elimination DELTA length = one palimpsa call's width."""
    _DELTALEN_SUM[0] += int(delta_len)
    _DELTALEN_COUNT[0] += 1
    _dist_add("delta_len", delta_len)
    if delta_len > _DELTALEN_MAX[0]:
        _DELTALEN_MAX[0] = int(delta_len)


def consume_token_length_stats() -> dict:
    """Pop per-episode token-size telemetry (mean/max for stream and delta).

    ``delta_*`` is the quantity that should size ALPHAGRAD_MAX_DELTA_TOKENS;
    ``stream_*`` is what MAX_TOKENS must cover while the full-buffer path is
    still in use. Both reset on read so wandb sees per-episode values.
    """
    n, dn = _TOKLEN_COUNT[0], _DELTALEN_COUNT[0]
    out = {
        "stream_mean": (_TOKLEN_SUM[0] / n) if n else 0.0,
        "stream_max": _TOKLEN_MAX[0],
        "stream_count": n,
        "delta_mean": (_DELTALEN_SUM[0] / dn) if dn else 0.0,
        "delta_max": _DELTALEN_MAX[0],
        "delta_count": dn,
    }
    _TOKLEN_SUM[0] = _TOKLEN_MAX[0] = _TOKLEN_COUNT[0] = 0
    _DELTALEN_SUM[0] = _DELTALEN_MAX[0] = _DELTALEN_COUNT[0] = 0
    return out


def _record_tokenization_truncation(raw_len: int) -> None:
    """Bump the per-process truncation counter and emit a one-time
    ``warnings.warn`` on the first observation. Cheap: a counter
    increment + one branch. The warning carries the actual raw token
    length so the user can see how much headroom they need.
    """
    if raw_len <= MAX_TOKENS:
        return
    overflow = raw_len - MAX_TOKENS
    _TOKENIZATION_TRUNCATION_COUNT[0] += 1
    _TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0] += overflow
    if raw_len > _TOKENIZATION_TRUNCATION_MAX_LEN[0]:
        _TOKENIZATION_TRUNCATION_MAX_LEN[0] = raw_len
    if not _TOKENIZATION_TRUNCATION_WARNED[0]:
        import warnings
        warnings.warn(
            f"[alphagrad.approx.env] jaxpr tokenization truncated: "
            f"raw_len={raw_len} > MAX_TOKENS={MAX_TOKENS} "
            f"(overflow={overflow}). The policy will see a clipped "
            f"observation for this step. Subsequent truncations are "
            f"silent but counted "
            f"(see ``tokenization/{{truncated_count, "
            f"overflow_sum_this_ep}}`` in the wandb log).",
            stacklevel=2,
        )
        _TOKENIZATION_TRUNCATION_WARNED[0] = True


def consume_tokenization_truncation_stats() -> dict:
    """Pop the per-episode (= per-poll) truncation telemetry.

    Drivers call this once per rollout, so the returned numbers are
    "since the last rollout" — not cumulative across the run. ``warned``
    stays sticky so the warning never re-fires within the same process.

    Returns:
        Dict with:
          * ``count`` — number of truncations this period.
          * ``max_observed_len`` — largest raw token length this period.
          * ``overflow_sum`` — sum of ``(raw_len - MAX_TOKENS)`` across
            this period's truncations; per-episode information loss
            (in tokens).
    """
    count = _TOKENIZATION_TRUNCATION_COUNT[0]
    max_len = _TOKENIZATION_TRUNCATION_MAX_LEN[0]
    overflow_sum = _TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0]
    _TOKENIZATION_TRUNCATION_COUNT[0] = 0
    _TOKENIZATION_TRUNCATION_MAX_LEN[0] = 0
    _TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0] = 0
    return {
        "count": int(count),
        "max_observed_len": int(max_len),
        "overflow_sum": int(overflow_sum),
    }


# ---------------------------------------------------------------------------
# APPEND-ONLY tokenization (the palimpsa-native observation).
# ---------------------------------------------------------------------------
# The default path re-tokenizes the WHOLE jaxpr every step and hands the policy
# a fixed MAX_TOKENS buffer, so the encoder re-reads the entire stream each time
# and any graph longer than the budget is silently clipped. That throws away the
# reason palimpsa (linear attention) was chosen: its state is a RECURRENCE, so
# it can absorb only what is NEW.
#
# The append-only form: tokenize the base function ONCE, then emit just the
# tokens produced by each elimination (its local paths / faces and the
# approximations applied to them). `graphax.IncrementalPathTokenizer` is the
# producer (`base_tokens()` then `eliminate(v)`); `approx.incremental_encoder`
# (`init_state` / `extend` / `enc_x`) is the consumer that carries the palimpsa
# state — already proven to match a full re-encode to 1e-4 for arbitrary splits
# (tests/incremental_encoder_equivalence_test.py). This function is the bridge:
# it turns an elimination prefix into that step's token DELTA.
#
# Deltas are cached by order prefix, so sibling envs that share a prefix share
# the replay, and extending a prefix by one vertex is one `eliminate` call
# rather than a full re-tokenize.
# THE ONLY REAL BOUND IN THE OBSERVATION PATH, and the only one JAX's static
# shapes actually require. SIZED FROM THE MEASURED DISTRIBUTION -- and it is
# a MEMORY FLOOR, not just a padding bound: every reverse-differentiated
# `encode_extend` in the loss materialises a ``(window, E)`` row block per
# sample per K-step whatever the actual delta length is, which is why the
# ``ALPHAGRAD_EXTEND_CHUNK`` sweep found peak memory FLAT at 2051 MB across
# every chunk size. Blocking cannot shrink a fixed window; only the window
# can.
#
# 32768 came from a single 25,737-token worst case reported by
# ``campaign_scratch/decode3_data.py``. A later per-step measurement of the actual delta
# distribution on BOTH the 2- and 3-block TransformerLM (380 / 540 steps)
# does not reproduce it: median 0, mean 75-114, p95 537-642, p99 1001-1411,
# MAX 1173-2833 -- i.e. the observed worst case is 0.23-0.35% of a 32768
# window and the buffer was 8-16x oversized on the very target it was sized
# for. 4096 clears the largest delta ever measured here (2833) with ~45%
# headroom and is still the next power of two above it.
#
# This is safe to shrink because overflow is LOUD: it RAISES by default
# (``ALPHAGRAD_DELTA_OVERFLOW``), so a target whose deltas really do exceed
# 4096 stops rather than silently desyncing the recurrence -- the failure
# mode that made the old bound feel like it had to be generous. Raise the
# env var if a new target trips it; do NOT switch to clip to hide it.
# IT IS THE HARD CAP, NOT THE WINDOW (owner ruling 2026-09-14). This number
# now means exactly two things and no longer means a third.
#
#   * it is the WIDTH OF THE WIRE. The host-to-device buffer is
#     ``DELTA_HEADER_SLOTS + MAX_DELTA_TOKENS`` uint8 and stays that width,
#     because three things outside the driver preallocate at it and none of
#     them is under per-episode control: the Ray measurement pool
#     (``ppo.py`` -> ``common/measure_pool.py``), the actor's failure
#     sentinel (``cpu_approx_worker.py``) and the launcher pins in
#     ``tools/gen_fq_launchers.py``.
#   * it is the LOUD CEILING. ``_record_delta_truncation`` raises against
#     it, so a delta the apparatus cannot carry at all stops the run.
#
# It is NO LONGER the window the rollout and the loss scan. That window is
# the PER-EPISODE BIN ``EnvConfig.delta_window`` -- a power of two at or
# below this cap, chosen from the recent measured maxima by
# ``common.episode_stream`` and applied from ``env.step``'s slice of the
# wire onwards. Every cost the bin targets (``ceil(W / C)`` outer fold
# iterations, the ``(W, embd_dim)`` row block per sample per K step, the
# ``(W,)`` edge-id vector) lives past that slice, not on the wire.
MAX_DELTA_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_DELTA_TOKENS", "32768"))

# ---------------------------------------------------------------------------
# THE NARROW TOKEN WIRE (owner's decisions, 2026-09-13).
#
# TOKENS RIDE AS uint8, on the wire, in `EnvState`, and in every trajectory
# leaf that stores them. The tokenizer id space is 256
# (`common.token_vocab`), so a token is a byte by construction.
#
# THERE ARE NO EQUATION IDS. A parallel `(MAX_DELTA_TOKENS,)` int32 buffer of
# graphax's stream-global segment ids used to ride beside the tokens in every
# one of those places. It had exactly one consumer, the relational
# forget-gate modulation in the palimpsa mixers (`rel_gate`, zero-init, one
# scalar per token per head, no ablation), reproduced inside the recurrence as
# a `(MAX_EQNS,)` histogram carried per step. All of it was removed. The pad
# sentinel -1 belonged to that buffer; token padding is 0.
#
# THE COUNT HEADER. The exact host-side token count used to ride in slot 0 of
# BOTH id buffers (`t[0] = n`, `e[0] = n`). A count up to MAX_DELTA_TOKENS
# (32768 by default) does not fit in a byte, so the header had to leave the
# token buffer's value space. It is now its OWN uint32, written little-endian
# across the first DELTA_HEADER_SLOTS byte slots of the wire (uint32); the tokens start
# at index DELTA_HEADER_SLOTS.
#
# WHY FOUR BYTE SLOTS AND NOT A SEPARATE CALLBACK OUTPUT. The wire arity is
# shared with `CpuApproximationServer.evaluate`, `CpuApproxPool.evaluate` /
# `evaluate_batch`, the batched host shim in `VertexEliminationEnv.tokenize`,
# and the trainers that still run the LEGACY full-stream observation (gfn,
# gdpo, mu0, alpha0, az_gumbel, ppo_ray_worker), which keep their own
# equation-id buffer for the DENSE encoder's pairwise T5 bias -- a different
# mechanism, not the one that was removed. Adding an output would change that
# protocol for every one of them. A four-byte header inside a buffer that is
# already byte-addressed costs four bytes and changes no signature.
# `encode_delta_header` / `decode_delta_header` are the ONLY two places that
# know the layout, and `env.step` reads the count through the second of them.
#
# PADDING IS NOT A VALUE. Token padding is 0, and token id 0 is the literal
# '-' graphax emits for a negative number -- it occurs INTERIOR to real
# streams. Nothing may therefore recover a length by scanning for the pad: the
# count in the header is the tokenizer's own `len()`, and every reader keys on
# it. `tests/trajectory_layout_test.py` pins that invariant.
#
# The dtype, pad, header width and host-side codec live in
# ``common.token_vocab`` (imported above and re-exported here) so the OTHER
# producer of these tokens, ``live_faces.LiveFaceStream``, can name them
# without importing this module and pulling JAX into a deliberately JAX-free
# file. ``env.DELTA_TOKEN_DTYPE`` and friends therefore still resolve, and
# there is still exactly one definition of each.


def decode_delta_header(tokens):
    """``() uint32``: the count `encode_delta_header` wrote, off the wire.

    THE ONE READER of the header layout on the device side. ``env.step`` calls
    this and nothing else reconstructs the count. Unsigned, so the full
    32-bit range is a count (owner ruling 2026-09-14); the caller narrows to
    int32 for indexing, which is exact because the host refuses any count
    above MAX_DELTA_TOKENS before it encodes.
    """
    h = jnp.asarray(tokens[:DELTA_HEADER_SLOTS]).astype(jnp.uint32)
    acc = jnp.zeros((), jnp.uint32)
    for i in range(DELTA_HEADER_SLOTS):
        acc = acc + (h[i] << jnp.uint32(8 * i))
    return acc


def delta_wire_width(window=None) -> int:
    """Width of ONE delta wire buffer: the header plus the id budget.

    ``window`` is the per-episode delta window BIN. It defaults to the hard
    cap, which is what every caller passes today: under the owner's ruling
    of 2026-09-14 the WIRE stays at the cap and the bin applies from
    ``env.step``'s slice of the wire onwards (see ``MAX_DELTA_TOKENS``). The
    argument exists so that binning the wire too is one edit in
    ``obs_width`` and not a rewrite -- the Ray pool's preallocation is what
    blocks it, nothing here.
    """
    return DELTA_HEADER_SLOTS + int(window or MAX_DELTA_TOKENS)


def validate_delta_window(window) -> int:
    """Check a delta window bin and return it in TOKENS. 0 means the cap.

    A bin must be a power of two, at most ``MAX_DELTA_TOKENS`` (it slices
    the wire, so it cannot be wider than the wire) and at least the fold
    chunk floor. The floor RAISES rather than clamping: a window below the
    chunk changes the chunk size itself, which regroups the loss's float32
    partial sums. ``common.episode_stream.DeltaWindowFloor`` says the rest.
    """
    w = int(window or 0)
    if w == 0:
        return int(MAX_DELTA_TOKENS)
    if w < 0:
        raise ValueError(
            f"EnvConfig.delta_window must be >= 0 (0 meaning the cap), got "
            f"{w}")
    if w & (w - 1):
        raise ValueError(
            f"EnvConfig.delta_window must be a power of two, got {w}. The "
            f"bins are a small set of compiled programs, one per power of "
            f"two (owner ruling 2026-09-14).")
    if w > MAX_DELTA_TOKENS:
        raise ValueError(
            f"EnvConfig.delta_window={w} is wider than the transport wire "
            f"ALPHAGRAD_MAX_DELTA_TOKENS={MAX_DELTA_TOKENS}. The wire is "
            f"the hard cap and the window is a slice of it.")
    from alphagrad.approx.common import episode_stream as _epstream

    floor = 1 << _epstream.window_floor_log2()
    if w < floor:
        raise _epstream.DeltaWindowFloor(
            _epstream.log2_for_length(w), _epstream.window_floor_log2(),
            _epstream.window_floor_tokens(),
            f"EnvConfig.delta_window={w} was asked for")
    return w

# BASE-TOKEN budget for the delta-buffer observation path
# (ALPHAGRAD_DELTA_TOKENS=1 in ppo.py). The base tokenized jaxpr is encoded
# into the policy carry ONCE, at episode start, so it needs a buffer of its
# own instead of a MAX_TOKENS-wide window over the whole growing stream.
#
# SIZED FROM THE MEASURED DISTRIBUTION, not a guess. `len(base_tokens())` is
# order-independent (it depends only on the jaxpr), measured 2026-08-05 with
# ALPHAGRAD_INCR_TOKEN_VOCAB=248 over every resolvable example:
#
#   Helmholtz 60 | Lighthouse 69 | NeuralNetwork 317 | ADALIF_SNN 340 |
#   ConvNet 397 | VmappedNeuralNetwork(h=256,mnist) 419 | VmappedConvNet 559 |
#   LIF_SNN 678 | MoE 797 | RoeFlux_1d 1184 | RobotArm_6DOF 1295 |
#   Encoder 1323 | TransformerLM 1406 | ViT 1457 | EncoderDecoder 1627 |
#   RoeFlux_3d 1777 | BlackScholes_Jacobian 4219
#
# Those figures are at the OLD default vocab (248). The default is 512 now,
# which only SHRINKS a base (a wider name alphabet spells fewer names out by
# concatenation: nn256 425 -> 331), so every entry above is a conservative
# upper bound and the budget below still holds with room to spare.
#
# 8192 is ~1.9x the largest measured base (4219) and ~20x the flagship
# nn256's (419), and one base buffer per env is 32 KB -- headroom is free
# here in a way it is not for the per-step buffers. ppo.py ASSERTS on the
# device-side count rather than clipping: a clipped base would desync the
# encoder's recurrence from the stream for the whole episode.
MAX_BASE_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_BASE_TOKENS", "8192"))


# What a delta that does not fit ``MAX_DELTA_TOKENS`` does.
#
#   "raise" (DEFAULT) -- stop. A clipped delta DESYNCS the encoder's
#       recurrence from the token stream for the rest of the episode: the
#       dropped tail is never re-read (the cursor is relative), so the carry
#       and the graph diverge silently and every later observation is wrong
#       about a graph the policy is still acting on. With the budget sized
#       from the measured distribution (32768 vs a measured worst case of
#       25737) this fires only when the budget is genuinely too small.
#   "clip" -- keep the old behaviour, but LOUDLY: one stderr line per
#       occurrence, never a once-per-process warning that a driver's
#       stderr handling can swallow.
#
# There is no third option. Dropping tokens silently is what #81 was.
_DELTA_OVERFLOW = os.environ.get(
    "ALPHAGRAD_DELTA_OVERFLOW", "raise").strip().lower()
if _DELTA_OVERFLOW not in ("raise", "clip"):
    raise ValueError("ALPHAGRAD_DELTA_OVERFLOW must be 'raise' or 'clip', "
                     f"got {_DELTA_OVERFLOW!r}")


def _record_delta_truncation(raw_len: int) -> None:
    """Same sink as ``_record_tokenization_truncation``, for the DELTA buffer.

    Under ``delta_obs`` the observation is not clipped at the (deleted)
    total-stream cap -- it is clipped at MAX_DELTA_TOKENS, per step. The wandb
    keys (``tokenization/{truncated_count, overflow_sum_this_ep}``) keep their
    meaning ("how often, and by how much, was the observation clipped"); only
    the budget they refer to changes.

    NEVER SILENT. The counters here are PROCESS-LOCAL and the driver that
    reads them is usually a different process (#81), so "no clipping
    reported" was never evidence that nothing was clipped. The raise/print
    below is the evidence.
    """
    if raw_len <= MAX_DELTA_TOKENS:
        return
    _TOKENIZATION_TRUNCATION_COUNT[0] += 1
    _TOKENIZATION_TRUNCATION_OVERFLOW_SUM[0] += raw_len - MAX_DELTA_TOKENS
    if raw_len > _TOKENIZATION_TRUNCATION_MAX_LEN[0]:
        _TOKENIZATION_TRUNCATION_MAX_LEN[0] = raw_len
    msg = (
        f"token DELTA truncated: {raw_len} > "
        f"MAX_DELTA_TOKENS={MAX_DELTA_TOKENS} "
        f"(overflow={raw_len - MAX_DELTA_TOKENS}). Those tokens are DROPPED "
        f"-- they are NOT re-read at the next step (the cursor is relative), "
        f"so the encoder's recurrence desyncs from the stream for the rest of "
        f"the episode. Raise ALPHAGRAD_MAX_DELTA_TOKENS, or set "
        f"ALPHAGRAD_DELTA_OVERFLOW=clip to accept the loss."
    )
    if _DELTA_OVERFLOW == "raise":
        raise ValueError(f"[alphagrad.approx.env] {msg}")
    print(f"[alphagrad.approx.env] {msg}", file=sys.stderr, flush=True)
    _TOKENIZATION_TRUNCATION_WARNED[0] = True


def _delta_observation(stream, last_start):
    """Wire form of ONE step's token delta: ``(delta_wire_width(),) uint8``.

    ONE buffer. The parallel equation-id buffer is gone (2026-09-13): those
    ids fed the palimpsa relational forget-gate modulation and nothing else,
    and the modulation went with them.

    Slots ``[0, DELTA_HEADER_SLOTS)`` are the count header -- one
    little-endian uint32, see :func:`encode_delta_header`. The tokens start at
    ``DELTA_HEADER_SLOTS`` and are pad-filled with 0.

    The count is the TOKENIZER'S OWN length. Nothing scans a padded buffer
    for it, so token id 0 -- the literal '-' graphax emits for a negative
    value, which occurs INTERIOR to real streams -- cannot make it short.

    RAISES on a token the byte-wide buffer cannot carry. That means the
    tokenizer was built at a vocabulary wider than ``common.token_vocab``
    allows, and the cast would WRAP.
    """
    blk = stream[last_start:]
    n_raw = len(blk)
    _record_delta_length(n_raw)
    # Raises unless ALPHAGRAD_DELTA_OVERFLOW=clip; the clamp below is what
    # that opt-out buys, and it is announced on stderr every time.
    _record_delta_truncation(n_raw)
    n = min(n_raw, MAX_DELTA_TOKENS)
    W = delta_wire_width()
    t = np.zeros((W,), dtype=DELTA_TOKEN_DTYPE)
    t[:DELTA_HEADER_SLOTS] = encode_delta_header(n)
    if n:
        _tk = np.asarray(blk[:n], dtype=np.int64)
        _check_delta_ids(_tk)
        t[DELTA_HEADER_SLOTS:DELTA_HEADER_SLOTS + n] = _tk.astype(
            DELTA_TOKEN_DTYPE)
    return jnp.asarray(t)


_INCR_TOK_CACHE: dict = {}
_INCR_TOK_CACHE_CAP = 512


def incremental_token_delta(jaxpr, argnums, consts, args, order_prefix,
                            vocab_size: int | None = None):
    """Tokens ADDED by the last vertex of ``order_prefix`` (1-based ids).

    NOT AN OBSERVATION PRODUCER -- ``tests/append_only_tokens_test.py`` is its
    only caller, and it must stay that way unless it is fixed first. It
    eliminates BARE (no per-vertex rules, no per-face transform dict), so its
    delta describes the EXACT graph while the measurement builds the
    APPROXIMATED one; and its cache key is ``(jaxpr, argnums, order_prefix)``,
    which collapses two plans that differ only in their approximation
    decisions onto one entry. Both are the divergence c83a6a1 fixed on the
    LiveFaceStream side. The live observation path is
    ``_incremental_stream_tokens`` (rules + face wires applied, both in the
    cache key), and ``_delta_observation`` ships its last block.

    ``order_prefix`` == () returns the base-function tokens. Returns a python
    list of ints; the caller pads it to ``MAX_DELTA_TOKENS`` for the callback's
    static output shape. Raises if a delta exceeds that budget — silently
    clipping a delta would desync the encoder's recurrence from the stream,
    which is much worse than clipping a re-read buffer.
    """
    from graphax import IncrementalPathTokenizer

    # THE one resolver. `None` means "whatever the observation path uses",
    # which is the only setting at which this function's tokens and the env's
    # can be compared at all.
    vocab_size = incr_token_vocab(vocab_size)
    key = (id(jaxpr), tuple(argnums), tuple(int(v) for v in order_prefix))
    hit = _INCR_TOK_CACHE.get(key)
    if hit is not None:
        return hit[1]

    prefix = tuple(int(v) for v in order_prefix)
    if not prefix:
        tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts),
                                      list(args), vocab_size=vocab_size)
        delta = [int(t) for t in tk.base_tokens()]
    else:
        parent = _INCR_TOK_CACHE.get(
            (id(jaxpr), tuple(argnums), prefix[:-1]))
        if parent is None:
            # Cold prefix: replay from the base once, then extend.
            tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts),
                                          list(args), vocab_size=vocab_size)
            list(tk.base_tokens())
            for v in prefix[:-1]:
                list(tk.eliminate(int(v)))
        else:
            tk = parent[0].__class__.__new__(parent[0].__class__)
            # The tokenizer is stateful and has no cheap clone, so a cold
            # replay is the honest fallback rather than aliasing the parent
            # (which would corrupt a sibling branch's stream).
            tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts),
                                          list(args), vocab_size=vocab_size)
            list(tk.base_tokens())
            for v in prefix[:-1]:
                list(tk.eliminate(int(v)))
        delta = [int(t) for t in tk.eliminate(int(prefix[-1]))]

    if len(delta) > MAX_DELTA_TOKENS:
        raise ValueError(
            f"token delta for prefix {prefix} is {len(delta)} > "
            f"MAX_DELTA_TOKENS={MAX_DELTA_TOKENS}. Raise "
            f"ALPHAGRAD_MAX_DELTA_TOKENS — clipping a delta would desync the "
            f"incremental encoder's recurrence from the token stream."
        )
    _record_delta_length(len(delta))
    if len(_INCR_TOK_CACHE) > _INCR_TOK_CACHE_CAP:
        _INCR_TOK_CACHE.clear()
    _INCR_TOK_CACHE[key] = (tk, delta)
    return delta


_INCR_STREAM_CACHE: dict = {}
_INCR_STREAM_CACHE_CAP = 64
# Engagement counters (proof instrumentation, not behavior): how often the
# append-only stream cache hit the full key / EXTENDED a parent / went cold.
_INCR_STREAM_STATS = {"hit": 0, "ext": 0, "cold": 0, "nostore": 0}

if os.environ.get("ALPHAGRAD_TOKENS_MID_COMPRESS") is not None:
    raise RuntimeError(
        "ALPHAGRAD_TOKENS_MID_COMPRESS is GONE. It existed to mitigate the "
        "COMPRESS prefix-property violation by tokenizing an intermediate "
        "last vertex WITHOUT its COMPRESS -- which made the observation "
        "describe an exact contraction while the measurement applied a "
        "reduction. The `is_last` gate it worked around has been removed: "
        "COMPRESS is now honored at every position, so the stream is "
        "append-only and there is nothing to mitigate. Unset the variable.")
# Pop-and-extend prefix cache for `_face_transforms_for_order`
# (ALPHAGRAD_FACE_ENUM_CACHE=1): (IncrementalJaxpr, out) keyed by the full
# decision prefix — one elimination per env step instead of a fresh
# build+replay of the whole prefix. Same COMPRESS soundness bound and same
# pop-on-extend policy as _INCR_STREAM_CACHE. Entries hold a live trace, so
# the cap is smaller and tunable.
_FACE_ENUM_CACHE: dict = {}
_FACE_ENUM_CACHE_CAP = int(os.environ.get("ALPHAGRAD_FACE_ENUM_CACHE_CAP",
                                          "64"))
_FACE_ENUM_STATS = {"ext": 0, "cold": 0, "compress": 0, "elims": 0,
                    "calls": 0, "build": 0}

# One ResourceMonitor per device-set, reused for every measurement.
# Constructing a fresh monitor per call leaks its C++ MemoryTracker/
# TimeTracker (~18 MB/call, the documented landmine): at h=256 that is
# 20 lifecycles/step x 12 steps -> the 756 GB RSS that thrashed v10exact
# (job 55508) into a D-state stall at ep 86. The monitor re-baselines in
# __enter__ (start() resets the trackers), so reuse is the intended
# lifecycle; ALPHAGRAD_MONITOR_REUSE=0 restores per-call construction.
_MONITOR_CACHE: dict = {}


def _get_resource_monitor(unique_devices):
    if os.environ.get("ALPHAGRAD_MONITOR_REUSE", "1") == "0":
        return ResourceMonitor(devices=unique_devices)
    key = tuple(sorted(id(d) for d in unique_devices))
    mon = _MONITOR_CACHE.get(key)
    if mon is None:
        mon = ResourceMonitor(devices=unique_devices)
        _MONITOR_CACHE[key] = mon
    return mon


def _face_wire_keys(faces_np, skips_np, n, joins_np=None):
    """Per-vertex compact hashable identity of the face wire rows.

    The stream cache keys on the face history, and the history it was keyed
    on was the DENSE int tuple: ``MAX_FACES x FACE_SLOTS x 3`` Python ints per
    vertex, for every vertex in the prefix, rebuilt AND rehashed on every
    callback. With the flagship's derived bound (MAX_FACES=2538) that is
    22842 ints per vertex, i.e. O(T x 22842) per step and O(T^2 x 22842) per
    episode -- while the measured live-face occupancy is 1.24 faces per
    vertex, so >99.9% of it encodes padding.

    The wire format pads with -1, so ``(positions of the non -1 entries,
    their values)`` is a COMPLETE and injective description of the row for a
    fixed shape -- every entry not listed is exactly -1. Same for the skips
    against their 0 padding. One vectorised pass over the prefix replaces the
    Python int construction, and the resulting keys are a few bytes each, so
    hashing them (and the ancestor-cut slices, which hash the whole prefix
    key once per probe) stops being O(T x MAX_FACES).
    """
    if n <= 0:
        return ()
    f = np.ascontiguousarray(faces_np[:n]).reshape(n, -1)
    s = np.ascontiguousarray(skips_np[:n]).reshape(n, -1)
    # THE JOIN BIT IS PART OF THE WIRE'S IDENTITY. Two prefixes that differ
    # only in a per-face join bit describe DIFFERENT computations (one merge
    # reconciled into the fresh contraction's container, the other into the
    # union), so a key that ignored the bit would serve one plan's cached
    # stream -- and, through the same signature, one plan's cached
    # elimination -- for the other. Appended to each vertex's tuple, so a
    # configuration without the bit produces byte-identical keys to before.
    j = (None if joins_np is None
         else np.ascontiguousarray(joins_np[:n]).reshape(n, -1))
    fr, fc = np.nonzero(f != -1)
    sr, sc = np.nonzero(s != 0)
    fb = np.searchsorted(fr, np.arange(n + 1))
    sb = np.searchsorted(sr, np.arange(n + 1))
    fv = f[fr, fc].astype(np.int32)
    sv = s[sr, sc].astype(np.int32)
    fc = fc.astype(np.int32)
    sc = sc.astype(np.int32)
    return tuple(
        (fc[fb[k]:fb[k + 1]].tobytes(), fv[fb[k]:fb[k + 1]].tobytes(),
         sc[sb[k]:sb[k + 1]].tobytes(), sv[sb[k]:sb[k + 1]].tobytes())
        + (() if j is None else (j[k].astype(np.int32).tobytes(),))
        for k in range(n)
    )


def _incremental_stream_tokens(config, consts, args, o_list, specs_list,
                               tok_rules_by_v, ft_by_vertex=None,
                               face_key=None,
                               face_rows_list=None, face_skips_list=None,
                               face_joins_list=None):
    """Full append-only observation stream for the prefix ``o_list``
    (ALPHAGRAD_INCREMENTAL_TOKENS=1): base tokens + one block per elimination
    (path tokens + ``approx`` echoes), from graphax's IncrementalPathTokenizer
    driven with the SAME transforms the measurement applies — the stream
    describes the approximated graph, not the intent.

    The stream for a prefix is a byte-wise prefix of the stream for any
    extension, so the cache extends the episode's tokenizer by ONE elimination
    per env step instead of re-tracing the whole Jacobian (``extract_jaxpr``)
    every step.

    Returns ``(stream, seg_ids, ft, last_start)``; ``last_start`` indexes the
    first token of the LAST elimination's block, which is what ``delta_obs``
    ships as the observation.

    Extending MUTATES the tokenizer, so the parent entry is POPPED
    before extension — a sibling chain that misses takes the honest cold
    replay (same policy as ``incremental_token_delta``).
    """
    from graphax import IncrementalPathTokenizer

    # `vocab_size` is the TOTAL id space, resolved through THE one resolver
    # (`common.token_vocab.incr_token_vocab`): 223 reserved structural tokens
    # plus 10 digits, leaving 23 symbols for the NAME alphabet at the 256 the
    # owner chose. A name past the alphabet spells itself out by
    # concatenation, so the stream is longer and every id fits in a byte --
    # which is what lets `DELTA_TOKEN_DTYPE` be uint8.
    #
    # CONSISTENCY, not speed: `base_observation` below resolves through the
    # same function, or the base and the deltas would be tokenized at
    # different vocabularies and would not concatenate.
    vocab = incr_token_vocab()
    steps = []
    for v_idx, v in enumerate(o_list):
        rows = tuple(tuple(int(x) for x in row)
                     for row in np.asarray(specs_list[v_idx]).reshape(-1, 3))
        steps.append((int(v), rows))
    base_key = (id(config.jaxpr), tuple(config.argnums))
    # Face actions change the stream (approx/SKIP blocks + downstream path
    # structure) — they must be part of the cache identity.
    key = base_key + (tuple(steps), face_key)

    hit = _INCR_STREAM_CACHE.get(key)
    if hit is not None:
        _INCR_STREAM_STATS["hit"] += 1
        return hit[1], hit[2], (hit[3] if len(hit) > 3 and hit[3]
                                else ft_by_vertex), hit[4]

    tk, stream, seg_ids, done = None, None, None, 0
    # Index in `stream` where the LAST elimination's block begins --
    # i.e. exactly this step's DELTA. 0 for the empty prefix (the base
    # IS the block). Rides the cache so a hit reports it too.
    last_start = 0
    # ALPHAGRAD_UNIFIED_FACE_ENUM=1: the caller hands the raw wire rows and
    # this function builds the per-face dicts on the TOKENIZER's own
    # IncrementalJaxpr (tk.ij) right before each elimination — no second
    # replay. `ft_out` accumulates per prefix and rides the cache entries.
    _unified = face_rows_list is not None
    ft_out: dict = {}
    # ANCESTOR EXTENSION IS SOUND FOR EVERY STATE.
    #
    # It used not to be. `decode_vertex_rule_specs` used to emit COMPRESS only
    # when `is_last=True`, so vertex k-1 was tokenized WITH its Compress at
    # prefix length k and WITHOUT it at k+1 — the streams diverged
    # (tests/stream_prefix_property_test.py measured token 681 of 931), and
    # extending a cached parent would have handed the policy a different
    # observation than a cold replay. Two carve-outs lived here: a prefix-wide
    # COMPRESS scan (which left the cache dead, ext=1/431 at v40 — a random
    # policy plants a COMPRESS within a step or two) and then a refined
    # "store only COMPRESS-free-last states" bound.
    #
    # The `is_last` gate is GONE (see decode_vertex_rule_specs), so the decode
    # of a vertex no longer depends on where the prefix ends: the stream for a
    # prefix is a byte-prefix of the stream for any extension for EVERY rule
    # kind, COMPRESS included, and the test asserts it. Nothing to carve out —
    # every state is storable and every parent is extendable.
    for cut in range(len(steps) - 1, 0, -1):
        parent = _INCR_STREAM_CACHE.pop(
            base_key + (tuple(steps[:cut]),
                        face_key[:cut] if isinstance(face_key, tuple)
                        else None), None)
        if parent is not None:
            _INCR_STREAM_STATS["ext"] += 1
            tk, stream, seg_ids, done = (
                parent[0], list(parent[1]), list(parent[2]), cut)
            last_start = parent[4]
            if len(parent) > 3 and parent[3]:
                ft_out = dict(parent[3])
            break
    if tk is None:
        _INCR_STREAM_STATS["cold"] += 1
        tk = IncrementalPathTokenizer(
            config.jaxpr, tuple(config.argnums), list(consts), list(args),
            vocab_size=vocab,
        )
        stream = [int(t) for t in tk.base_tokens()]
        seg_ids = [int(g) for g in tk.last_eqn_ids()]
        guard = os.environ.get("ALPHAGRAD_VOCAB_SIZE")
        if guard is not None and tk.max_token_id() >= int(guard):
            raise ValueError(
                f"incremental token ids reach {tk.max_token_id()} but the "
                f"policy embedding has only {guard} rows — raise --vocab-size "
                f"or lower ALPHAGRAD_INCR_TOKEN_VOCAB. (JAX CLAMPS an "
                f"out-of-range gather, silently reading the wrong row.)"
            )
    for _ki in range(done, len(steps)):
        v, _rows = steps[_ki]
        if _unified:
            _pf_v = _face_dict_for_vertex(
                config, tk.ij, int(v), face_rows_list[_ki],
                face_skips_list[_ki],
                face_join=(None if face_joins_list is None
                           else face_joins_list[_ki]))
            if _pf_v:
                ft_out[int(v)] = _pf_v
        else:
            _pf_v = (ft_by_vertex or {}).get(int(v))
        last_start = len(stream)
        stream += [int(t) for t in tk.eliminate(
            int(v), tok_rules_by_v.get(int(v), ()), _pf_v)]
        seg_ids += [int(g) for g in tk.last_eqn_ids()]

    _ft_ret = ft_out if _unified else ft_by_vertex
    if len(_INCR_STREAM_CACHE) > _INCR_STREAM_CACHE_CAP:
        _INCR_STREAM_CACHE.clear()
    _INCR_STREAM_CACHE[key] = (tk, stream, seg_ids,
                               ft_out if _unified else None, last_start)
    return stream, seg_ids, _ft_ret, last_start


# ---------------------------------------------------------------------------
# Host-phase profiling (always accumulated — a perf_counter pair per phase is
# noise — printed per episode by ppo.py when ALPHAGRAD_PROFILE=1). Answers
# "where does the episode actually go": tokenizer replay vs face-key
# enumeration vs count pass vs XLA compile vs execution vs the policy-side
# oracle replays (ppo adds its own keys into the same sink).
# ---------------------------------------------------------------------------
_PROF: dict = {}


def _prof_add(key: str, dt: float) -> None:
    _PROF[key] = _PROF.get(key, 0.0) + float(dt)


def consume_profile() -> dict:
    """Pop the accumulated per-phase host seconds since the last call."""
    out = dict(_PROF)
    _PROF.clear()
    return out


# ---------------------------------------------------------------------------
# Phase-0 attribution sinks (see ppo.py `_pp_mark`).
#
# `_PROF` answers "how many seconds did phase X cost this episode" but not
# "how much did ONE decision cost", which is the number the optimisation plan
# is gated on. `_prof_sample` keeps the individual observations so the driver
# can report mean AND p95 per decision; `_dist_add` keeps the raw
# distributions (faces per vertex, delta length, ...) that size the Phase-1
# bucket grids. Both are opt-in -- the sample lists grow with the step count,
# and nothing downstream may depend on a profiling buffer.
# ---------------------------------------------------------------------------
_PROF_SAMPLES: dict = {}
_PROF_DIST: dict = {}
_PROFILE_DIST = os.environ.get("ALPHAGRAD_PROFILE_DIST", "0") == "1"


def _prof_sample(key: str, dt: float) -> None:
    """Record ONE observation of phase `key` (per-decision granularity)."""
    _PROF_SAMPLES.setdefault(key, []).append(float(dt))


def consume_profile_samples() -> dict:
    """Pop {key: [seconds, ...]} since the last call."""
    out = {k: list(v) for k, v in _PROF_SAMPLES.items()}
    _PROF_SAMPLES.clear()
    return out


# ---------------------------------------------------------------------------
# EVENT TRACE (ALPHAGRAD_PROFILE_TRACE=1).
#
# `_prof_add` totals and `_prof_sample` per-decision means both assume the
# phases they time are disjoint. They are not: the env callback's own
# `prof/measure_wait` is nested inside ppo's `prof/envstep` mark interval, and
# a running-clock mark ("everything since the previous mark") silently absorbs
# whatever it does not name. A flat (t, label) event log is the only way to
# establish who contains whom -- reconstruct the nesting from the timestamps
# instead of trusting the bucket names.
# ---------------------------------------------------------------------------
_PROF_TRACE: list = []
_PROFILE_TRACE = os.environ.get("ALPHAGRAD_PROFILE_TRACE", "0") == "1"


def _trace(label: str) -> None:
    """Append one timestamped event (ALPHAGRAD_PROFILE_TRACE=1, else no-op)."""
    if _PROFILE_TRACE:
        _PROF_TRACE.append((time.perf_counter(), label))


def consume_trace() -> list:
    """Pop [(perf_counter, label), ...] since the last call."""
    out = list(_PROF_TRACE)
    _PROF_TRACE.clear()
    return out


def _dist_add(key: str, value) -> None:
    """Record one sample of a size distribution (ALPHAGRAD_PROFILE_DIST=1)."""
    if not _PROFILE_DIST:
        return
    _PROF_DIST.setdefault(key, []).append(int(value))


def consume_distributions() -> dict:
    """Pop {key: [int, ...]} since the last call."""
    out = {k: list(v) for k, v in _PROF_DIST.items()}
    _PROF_DIST.clear()
    return out


# Skip the symbolic count pass entirely (see _callback). It feeds only the
# log-only muls_adds_fmas / max_io_sum channels and the op-count cap; the
# trained objective (latency, peak_memory, frob) never reads it, and it costs
# more than the compile it guards once plans do real work.
# READ PER CALL, not frozen at import. A module constant here is a
# COLLECTION-ORDER HAZARD: ~35 test modules set the variable with
# ``os.environ.setdefault`` at their own import time, so whichever of them
# pytest imports first decides the setting for every test that shares the
# process, and a test that sets the variable itself changes nothing. Measured
# 2026-09-13: tests/test_all_cost_channels.py asserts the counted channels are
# non-zero and fails whenever it is collected beside tests/plan_log_test.py,
# passing only because xdist usually puts them in different workers.
def skip_count_ops() -> bool:
    """Is the symbolic count pass off in THIS process right now?"""
    return os.environ.get("ALPHAGRAD_SKIP_COUNT_OPS", "0") == "1"

# EXACT-JACOBIAN REUSE ACROSS ENVS (ALPHAGRAD_CACHE_EXACT=1, default on).
#
# The exact Jacobian is the quality REFERENCE for cos/frob, and it is
# IDENTICAL for every env in an episode, for two reasons:
#   * vertex elimination computes the same Jacobian for ANY order — the order
#     changes the cost, not the value;
#   * ppo.train_episode calls generate_eval_samples ONCE per episode and shares
#     the result across all envs.
# So the 16 envs were each executing a bit-identical exact Jacobian:
# 16 x n_points executions per episode where n_points would do (cb.quality was
# 31.1s/episode). Keyed on a content digest of that sample's eval args, so an
# entry can only be reused for genuinely identical inputs; the whole cache is
# dropped as soon as a new episode's samples appear.
_EXACT_CACHE: dict = {}
_CACHE_EXACT = os.environ.get("ALPHAGRAD_CACHE_EXACT", "1") == "1"


def _eval_digest(eval_args_list) -> bytes:
    """Content digest of one sample's eval arguments."""
    import hashlib as _hl
    h = _hl.blake2b(digest_size=16)
    for a in eval_args_list:
        arr = np.asarray(a)
        h.update(repr(arr.shape).encode())
        h.update(repr(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.digest()


# ---------------------------------------------------------------------------
# PER-EPISODE PLAN DEDUPLICATION (owner ruling 2026-09-14).
#
# At the identity init all 16 terminal plans of an episode ARE THE SAME
# PROGRAM, and a near-identity policy keeps producing duplicates for many
# episodes after that. Measuring the same program sixteen times costs sixteen
# times a second and learns nothing: the reward vector is a function of the
# plan and of the episode's eval samples, and neither changes between the
# duplicates.
#
# So: content-hash the plan (order, vertex rules, face rows, face skips, face
# joins) together with the episode's eval-sample key, and inside ONE EPISODE
# serve a repeat from the first measurement. The QUALITY channel is reused
# too, because the gradient cosine depends on the plan alone.
#
# NEVER ACROSS EPISODES. The samples change, and a reading taken against last
# episode's samples is not this episode's reading. The episode key below is
# what enforces that: it carries both the published episode index and a
# content digest of the samples, so a stale entry cannot survive either a new
# episode or a resampling within one.
#
# PER MEASURE ACTOR. The actors are separate processes and share no memory,
# so each measures a duplicated plan once. That is nearly the whole win in
# WALL TIME anyway: the pool hands actor j the slots j, j+M, j+2M..., so with
# M actors and 16 identical plans every actor measures its first slot and
# serves the rest from its own cache, and the episode's measurement phase
# costs one measurement instead of ceil(16/M). The pool cannot route by hash
# to do better: `CpuApproxPool._evaluate_batch_impl` assigns slot i to actor
# (i - wave_start), and a duplicate group larger than the number of waves
# cannot be placed on one actor under that schedule.
_PLAN_DEDUPE: dict = {}
# The episode's measurement accounting, drained by `flush_measure_episode`.
_MEASURE_EPISODE: dict = {
    "key": None,          # the episode key these numbers belong to
    "label": None,        # what to print for it
    "n_plans": 0,         # terminal plans seen
    "n_measured": 0,      # of those, actually measured
    "secs": [],           # seconds per measured plan, both halves
    "cand_secs": [],      # the candidate's half
    "ref_secs": [],       # the reference's half
}


def measure_dedupe_enabled() -> bool:
    """Is the per-episode duplicate cache armed? Default yes.

    ``ALPHAGRAD_MEASURE_DEDUPE=0`` disarms it. Any tool that deliberately
    measures THE SAME PLAN more than once -- `landscape_map`'s ``--reps``,
    any drift or noise probe -- must disarm it, or every repeat after the
    first returns the first one's numbers and the spread reads zero.

    A SECOND PRECONDITION is enforced at the call site rather than here: a
    measurement with no eval samples has no episode key, so it never dedupes
    whatever this returns.
    """
    return os.environ.get("ALPHAGRAD_MEASURE_DEDUPE", "1") not in (
        "0", "", "false", "False", "no")


# The counts the most recent measurement in this process actually used.
# Written at the bottom of `_callback_measured`; read by tools that need to
# stamp a row with the protocol it was measured under, because under the time
# budget that protocol is a property of the PLAN, not of the run.
_LAST_MEASURE_COUNTS: dict = {}


def last_measure_counts() -> dict:
    """A copy of the counts the last measurement used, or an empty dict."""
    return dict(_LAST_MEASURE_COUNTS)


def _episode_measure_key(eval_samples) -> bytes:
    """THE EPISODE a measurement belongs to, as bytes.

    The published episode index and the attempt (both of which the trainer
    republishes into the actor before every measurement) PLUS a content
    digest of the episode's first eval sample. Either alone would be wrong:
    the index is 0 in any process nobody published one into, and the samples
    could in principle repeat.

    Only the FIRST sample is digested. The samples are drawn together, once
    per episode, so the first one identifies the draw, and digesting all of
    them would hash the whole calibration set on every terminal callback.
    """
    import hashlib as _hl
    h = _hl.blake2b(digest_size=16)
    h.update(str(walk_episode()).encode())
    h.update(b"/")
    h.update(str(plan_log_attempt()).encode())
    if eval_samples:
        try:
            h.update(_eval_digest([a[0] for a in eval_samples]))
        except Exception:
            # A digest we cannot take is a cache we must not use. Make the
            # key unique so nothing can ever hit against it.
            h.update(os.urandom(16))
    return h.digest()


def _plan_content_key(order, rule_specs, face_specs, face_skips,
                      face_joins) -> bytes:
    """Content hash of a PLAN: what graphax would be asked to build.

    The elimination order, the per-vertex rules, the per-face rows, the
    per-face skips and the per-face join bits -- exactly the five wires the
    plan log records, and exactly what decides the executable. Two plans with
    this hash in common produce the same program and therefore the same
    reward vector against the same samples.
    """
    import hashlib as _hl
    h = _hl.blake2b(digest_size=16)
    h.update(np.asarray(order, dtype=np.int64).tobytes())
    for part in (rule_specs, face_specs, face_skips, face_joins):
        h.update(b"|")
        if part is None:
            h.update(b"none")
            continue
        # int32 is the wire dtype of all four, so this is a view plus one
        # copy into the hash rather than a widening copy of a face array
        # that is 95 x MAX_FACES x FACE_SLOTS x 3 on the flagship.
        arr = np.asarray(part, dtype=np.int32)
        h.update(repr(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.digest()


def flush_measure_episode() -> None:
    """Print THE EPISODE LINE and reset the accounting. Idempotent.

    One line per episode per measure actor, naming the median seconds a plan
    cost to measure and the duplicate count. Called when the episode key
    changes and again when the trainer drains the plan records, so the last
    episode of a run is reported too.
    """
    st = _MEASURE_EPISODE
    if not st["n_plans"]:
        st["key"] = None
        st["label"] = None
        return
    _secs = sorted(st["secs"])
    _cand = sorted(st["cand_secs"])
    _ref = sorted(st["ref_secs"])

    def _med(xs):
        if not xs:
            return 0.0
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    # THE PID IS PART OF THE LINE. The accounting is per measure ACTOR, and
    # a campaign arm runs several, so three lines per episode is the expected
    # output and each one has to say whose it is.
    print(f"[measure] episode {st['label']} pid {os.getpid()}: median "
          f"{_med(_secs):.3f} s/plan (candidate {_med(_cand):.3f} s, "
          f"reference {_med(_ref):.3f} s) dedupe: {int(st['n_plans'])} "
          f"plans {int(st['n_measured'])} measured", flush=True)
    st["key"] = None
    st["label"] = None
    st["n_plans"] = 0
    st["n_measured"] = 0
    st["secs"] = []
    st["cand_secs"] = []
    st["ref_secs"] = []


def _roll_measure_episode(key: bytes) -> None:
    """Start a new episode's accounting when `key` changes.

    Flushes the previous episode's line and drops the duplicate cache, which
    is the one place the never-across-episodes rule is enforced.
    """
    if _MEASURE_EPISODE["key"] == key:
        return
    flush_measure_episode()
    _PLAN_DEDUPE.clear()
    _MEASURE_EPISODE["key"] = key
    _MEASURE_EPISODE["label"] = str(walk_episode())


_DEGENERATE_PLANS = [0]
# Plans TRUNCATED for engineering reasons (resource limits: op-count cap,
# OOM). Counted separately from zero-work plans because the two get opposite
# treatment — see `_truncated_reward` and the RESOURCE-LIMIT block below.
_TRUNCATED_PLANS = [0]
# Zero-work plans are NO LONGER refused; this is pure telemetry.
_ZERO_WORK_PLANS = [0]


def _record_degenerate_plan() -> None:
    _DEGENERATE_PLANS[0] += 1


def _record_truncated_plan() -> None:
    _TRUNCATED_PLANS[0] += 1
    _DEGENERATE_PLANS[0] += 1     # keep the legacy aggregate meaningful


def _record_zero_work_plan() -> None:
    _ZERO_WORK_PLANS[0] += 1


# ---------------------------------------------------------------------------
# REFUSED TERMINAL MEASUREMENTS, BY KIND.
# ---------------------------------------------------------------------------
# A refused measurement is MISSING DATA (AGENTS.md: "never score them as the
# worst result"). The trainer now excludes the whole environment from the
# update, which is correct and also INVISIBLE: a run whose refusal rate walks
# from 2 percent to 40 percent trains on fewer and fewer environments and
# every panel still looks healthy. So the rate is telemetry, per episode and
# per KIND, and it rides the counter drain every actor already answers
# (`consume_collapse_stats`), not the plan log -- the rate must be readable
# with `--plan-log` off.
#
# THE KIND is the reason's prefix: `oom`, `untraceable`, `muls-cap`,
# `no-target-fun`, `raised`. The detail after the colon stays on the plan
# record's `refused` field, which is not aggregated.
#
# `oracle` IS GONE (owner ruling 2026-09-18). It named the gradient oracle's
# own float64 compile failing inside the measurement, which was 3.9 percent of
# every plan measured and 15.6 percent of an oracle-due episode's. The oracle
# left the measurement path: it runs asynchronously on the CPU and cannot
# refuse a plan any more, so nothing produces that kind. `refusal_kind` still
# parses it, because archived plan logs carry it.
_REFUSED_KINDS: dict = {}


def refusal_kind(reason: str) -> str:
    """The telemetry KIND of a plan record's ``refused`` reason."""
    r = str(reason or "unknown")
    return r.split(":", 1)[0] or "unknown"


def _record_refusal(reason: str) -> None:
    """Count ONE refused terminal measurement. Never raises, never skips."""
    k = refusal_kind(reason)
    _REFUSED_KINDS[k] = int(_REFUSED_KINDS.get(k, 0)) + 1
    _REFUSED_KINDS["total"] = int(_REFUSED_KINDS.get("total", 0)) + 1


def consume_refused_counts() -> dict:
    """Pop this process's per-kind refusal counts (mirrors the other pollers)."""
    out = {k: int(v) for k, v in _REFUSED_KINDS.items()}
    _REFUSED_KINDS.clear()
    return out


# ---------------------------------------------------------------------------
# THE FIDELITY CHANNEL (reward slot 8) -- CLIPPED RELATIVE FROBENIUS.
# ---------------------------------------------------------------------------
# Owner's choice, 2026-08-28: "clipped relative-Frobenius TRAINED, cosine
# LOGGED, gradient coverage a hard GUARD." This block is the trained half; the
# cosine subsample is `cos_log_every` below. (The coverage guard was REMOVED
# 2026-09-03 by owner ruling, ticket dsnn-3qm.15: no guard, no reward channel,
# no value head; reward slot 7 is reserved and never populated.) The two
# remaining instruments read the same pair of Jacobians and are NOT
# interchangeable:
#
#   * fidelity is a MAGNITUDE. `1 - ||J_e - J_a||_F / ||J_e||_F`, clipped to
#     [-1, 1]. It is defined (and equals 0) when the approximated gradient is
#     identically zero, where the cosine is 0/0.
#   * the cosine is an ANGLE and ignores magnitude entirely: a plan that halves
#     every entry of the Jacobian scores cos == 1.0 and rel_frob == 0.5.
#
# COST. Both fidelity and the cosine need the EXACT Jacobian. Under the
# `loss_drop` quality metric the exact executable is otherwise never compiled or
# run -- that is where loss_drop's 0.22 s / 40 MB per plan (vs the cosine's
# 9.70 s / 4.24 GB) comes from. So fidelity is NOT free: unless the quality
# metric is grad_cosine/jac_cosine (which build the reference per point and
# hand the residual over for free), the channel pays one exact execution per
# terminal plan through `_exact_ref_scores`, and its price is published as
# ``fidelity/wall_amortised_s``.
_FIDELITY_STATS: dict = {
    "n": 0,                 # fidelity measurements taken
    "sum": 0.0,
    "min": 1.0,
    "max": -1.0,
    "n_clipped_low": 0,     # rel_frob >= 2: the clip floor actually bound
    "n_cos": 0,             # cosine samples logged (subsampled)
    "sum_cos": 0.0,
    "min_cos": 1.0,
    "wall_s": 0.0,          # wall spent inside the exact-reference block
    "exact_execs": 0,       # exact executions this block was responsible for
    "last": None,
}
# Terminal plans seen since import, for the cosine subsample stride.
_COS_LOG_SEEN = [0]


def fidelity_enabled() -> bool:
    """Is the clipped-relative-Frobenius channel (slot 8) measured?

    DEFAULT OFF AT THE LIBRARY LEVEL, because measurement happens in Ray
    measure actors that are
    separate processes and may be RESPAWNED inside a job launched before this
    commit. ``ppo.configure_fidelity`` exports the variables before ``ray.init``.
    """
    if os.environ.get("ALPHAGRAD_FIDELITY", "") not in ("", "0"):
        return True
    try:
        return float(os.environ.get("ALPHAGRAD_FIDELITY_WEIGHT", "0")) != 0.0
    except ValueError:
        return False


def cos_log_every() -> int:
    """Stride of the COSINE LOG subsample; 0 (default) = never.

    The cosine is a diagnostic, not a trained channel, and it costs an exact
    reference (9.70 s / 4.24 GB per plan) that the default quality metric does
    not otherwise build -- 44x loss_drop's 0.22 s / 40 MB. Computing it per plan
    would dominate the measurement budget, so it is sampled every Nth TERMINAL
    plan per process.

    NOTE the asymmetry that makes this flag mostly unnecessary in practice: when
    the exact reference is materialised anyway (the fidelity channel, or
    ALPHAGRAD_QUALITY_METRIC=cosine), the cosine is one extra
    per-leaf dot product on leaves that are already resident, so it is taken on
    EVERY such plan regardless of this stride. The stride only ever FORCES an
    exact reference that nothing else asked for.
    """
    try:
        return max(0, int(os.environ.get("ALPHAGRAD_COS_LOG_EVERY", "0") or 0))
    except ValueError:
        return 0


def _cos_log_due() -> bool:
    """Consume one terminal-plan tick; True on every ``cos_log_every()``-th."""
    every = cos_log_every()
    if every <= 0:
        return False
    _COS_LOG_SEEN[0] += 1
    return (_COS_LOG_SEEN[0] - 1) % every == 0


def clipped_rel_frob(rel_frob) -> float:
    """THE FIDELITY VALUE: ``clip(1 - rel_frob, -1, +1)``.

    See the slot-8 entry in the REWARD_NAMES block for the sign convention.
    A non-finite ``rel_frob`` (nan/inf from a blown-up plan) scores the FLOOR:
    it is a real statement about the plan, not apparatus failure.
    """
    x = float(rel_frob)
    if not np.isfinite(x):
        return -1.0
    return float(min(1.0, max(-1.0, 1.0 - x)))


def _record_fidelity(fid: float | None, rel_frob: float | None,
                     cos: float | None) -> None:
    """Record one plan's fidelity and/or logged cosine.

    ``fid`` is ``None`` when only the COSINE subsample fired (the channel is
    off but ``--cos-log-every`` asked for a reference), so the cosine counter
    and the fidelity counter move independently and the amortised-cost line
    stays honest about which one paid.
    """
    s = _FIDELITY_STATS
    if fid is not None:
        s["n"] += 1
        s["sum"] += float(fid)
        s["min"] = min(s["min"], float(fid))
        s["max"] = max(s["max"], float(fid))
        if float(fid) <= -1.0:
            s["n_clipped_low"] += 1
    s["last"] = {"fidelity": (None if fid is None else float(fid)),
                 "rel_frob": (None if rel_frob is None else float(rel_frob)),
                 "cos": (None if cos is None else float(cos))}
    if cos is not None:
        s["n_cos"] += 1
        s["sum_cos"] += float(cos)
        s["min_cos"] = min(s["min_cos"], float(cos))


def consume_fidelity_stats() -> dict:
    """Pop the per-period fidelity + logged-cosine aggregate.

    ``cos_wall_amortised_s`` is the honest per-plan price of the cosine LOG:
    the wall this block spent divided by the number of fidelity measurements it
    served, which is the number the subsample stride is meant to control.
    """
    s = _FIDELITY_STATS
    n = max(int(s["n"]), 0)
    ncos = max(int(s["n_cos"]), 0)
    out = {
        "count": n,
        "mean": (s["sum"] / n) if n else float("nan"),
        "min": s["min"] if n else float("nan"),
        "max": s["max"] if n else float("nan"),
        "clipped_low": int(s["n_clipped_low"]),
        "cos_count": ncos,
        "cos_mean": (s["sum_cos"] / ncos) if ncos else float("nan"),
        "cos_min": s["min_cos"] if ncos else float("nan"),
        "wall_s": s["wall_s"],
        "exact_execs": int(s["exact_execs"]),
        # Wall per PLAN SERVED by the exact-reference block, whichever consumer
        # asked for it. This is the number to quote as "what the fidelity
        # channel / the cosine log actually cost per plan".
        "wall_amortised_s": ((s["wall_s"] / max(n, ncos))
                             if max(n, ncos) else float("nan")),
    }
    s.update({"n": 0, "sum": 0.0, "min": 1.0, "max": -1.0, "n_clipped_low": 0,
              "n_cos": 0, "sum_cos": 0.0, "min_cos": 1.0, "wall_s": 0.0,
              "exact_execs": 0})
    return out


# ---------------------------------------------------------------------------
# SPARSITY -- reward slot 10. HOW MUCH LESS THE APPROXIMATED ELIMINATION
# ACTUALLY STORES THAN THE EXACT ONE ON THE SAME ORDER.
#
# THE DEFINITION, and why it is this one:
#
#     sparsity_ratio = stored_bytes(approx) / stored_bytes(exact)
#     sparsity       = clip(1 - sparsity_ratio, -1, +1)      <- the CHANNEL
#
# `stored_bytes` is the sum, over every accumulated-Jacobian edge the
# elimination WRITES BACK into the graph, of ``val.size * dtype.itemsize`` --
# the buffer that is really allocated, read at graphax's single edge-store site
# AFTER the squeeze, once per TRACE. Both arms are the SAME elimination order
# (`_do_compile_exact` differs from `_do_compile_approx` in nothing but the two
# approximation kwargs), so the ratio isolates the approximation and nothing
# else. The identity plan therefore scores EXACTLY 1.0 / channel 0.0 by
# construction, not by tolerance.
#
# WHY STORED BYTES AND NOT A DECLARED CLASS. A tensor can DECLARE a sparse dim
# and still materialise dense, and the reverse (`val is None`) declares nothing
# while storing nothing -- so a class-based count overstates sparsity in both
# directions. This is the project's own recorded rule ("compare STORED BYTES
# not declared classes"), and it is the same quantity the offline structure
# audit sums. WHAT WAS REJECTED:
#   * an NNZ ratio over the OUTPUT Jacobians -- the output is drained dense at
#     the boundary in both arms, so it is blind to everything the elimination
#     did in between (this is the same mechanism behind "approx cuts latency
#     but never memory");
#   * a STRUCTURAL count (fraction of edges carried as Diag/blockdiag/
#     compressed) -- that is precisely the declared-class count the rule above
#     forbids, and a diagonal that is stored dense would score as sparse;
#   * the XLA memory-compression ratio already logged by
#     `consume_memory_compression_stats` -- it is a property of the COMPILED
#     EXECUTABLE (temp+output+argument bytes, forward pass included), not of
#     the accumulated Jacobians, and the output boundary dominates it.
# `cells` (graphax's `_structural_val_size`, the on-structure cell count) is
# tallied beside `bytes` and logged as a second ratio: it is quant-blind, so
# `bytes` vs `cells` separates "narrower dtype" from "fewer cells".
#
# SIGN CONVENTION, "higher is better" like every other slot:
#   +1  <=>  ratio 0     <=>  the plan stores NOTHING (all-SKIP).
#    0  <=>  ratio 1     <=>  the plan stores exactly what exact stores.
#   -1  <=>  ratio >= 2  <=>  the plan stores twice the exact arm or worse,
#            i.e. it DENSIFIED. That is a real, observed outcome, not a
#            theoretical one, and the floor is what keeps the channel bounded
#            and therefore symlog-free and PopArt-friendly.
#
# ############ READ THIS BEFORE PUTTING A WEIGHT ON IT ############
# SPARSITY IS THE MOST HACKABLE CHANNEL ON THE BOARD, BY CONSTRUCTION.
# An all-SKIP plan is MAXIMALLY sparse and scores the ceiling +1. The
# confirmed TLM reward hack -- one skipped face that deletes 62-73% of the
# backward pass and freezes 11-15 of 16 parameter leaves, which the 200-step
# quality probe prices at 0.02% (0.9258 vs 0.9260) -- scores NEAR the ceiling
# too. This channel rewards deleting computation. The gradient-coverage guard
# that used to refuse such plans (and was a HARD PRECONDITION of a non-zero
# weight here) was REMOVED 2026-09-03 by owner ruling -- no guard, never a
# reward gate (ticket dsnn-3qm.15) -- so NOTHING refuses them now. So:
#   * a SENTINELLED plan takes the channel's FLOOR, not its ceiling, so a
#     sentinelled destroyer can never be crowned on sparsity;
#   * and the channel is DEFAULT OFF and default weight 0 -- logged, not
#     trained -- until somebody has looked at what it correlates with.
# #################################################################
#
# COST. The tally is filled while `jacve` is TRACED, so it is free on any
# trace that was going to happen: the approx executable is compiled for every
# plan, and the exact one is materialised whenever the fidelity channel or a
# cosine quality metric builds it (see `_exact_ref_scores`). When the compile
# cache serves an executable WITHOUT tracing, the tally is recovered with an
# abstract `jax.eval_shape` walk of the same elimination -- traced, never
# compiled, never executed. Those fallbacks are counted and their wall is
# published as `sparsity/wall_amortised_s`.
_SPARSITY_STATS: dict = {
    "n": 0,
    "sum": 0.0,            # of the CHANNEL value
    "min": 1.0,
    "max": -1.0,
    "sum_ratio": 0.0,      # of the stored-BYTE ratio
    "min_ratio": float("inf"),
    "max_ratio": 0.0,
    "sum_cells": 0.0,      # of the stored-CELL ratio (quant-blind)
    "n_undefined": 0,      # exact arm stored nothing: ratio undefined
    "n_failed": 0,         # apparatus failure; slot stays 0.0
    "n_fallback_traces": 0,
    "wall_s": 0.0,
    "last": None,
}
# Per-plan stored-byte tallies, keyed by the SAME digests the compile cache
# uses. Bounded; see `_read_store_tally`.
_APPROX_STORE_BYTES: dict = {}
_EXACT_STORE_BYTES: dict = {}


def sparsity_enabled() -> bool:
    """Is the stored-byte sparsity channel (slot 10) measured?

    DEFAULT OFF AT THE LIBRARY LEVEL, exactly like `fidelity_enabled`, and
    for the same reason: measurement happens in
    Ray measure actors that are separate processes and may be RESPAWNED inside
    a job launched before this commit. ``ppo.configure_sparsity`` exports the
    variables before ``ray.init``.
    """
    if os.environ.get("ALPHAGRAD_SPARSITY", "") not in ("", "0"):
        return True
    try:
        return float(os.environ.get("ALPHAGRAD_SPARSITY_WEIGHT", "0")) != 0.0
    except ValueError:
        return False


def sparsity_channel(ratio) -> float:
    """THE SPARSITY VALUE: ``clip(1 - stored_approx/stored_exact, -1, +1)``.

    A non-finite ratio means the EXACT arm stored nothing measurable -- an
    undefined comparison, i.e. apparatus, not a verdict on the plan -- and
    reads 0.0 ("not measured"), never the floor. Contrast `clipped_rel_frob`,
    where a non-finite residual IS a statement about the plan.
    """
    x = float(ratio)
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(-1.0, 1.0 - x)))


def _arm_store_tally(on: bool) -> None:
    """Arm graphax's accumulated-Jacobian byte tally for ONE trace."""
    if not on:
        return
    try:
        from graphax.core import arm_store_accounting, reset_store_accounting
    except ImportError:
        return
    reset_store_accounting()
    arm_store_accounting()


def _read_store_tally(on: bool, store: dict, key) -> None:
    """Disarm and memoise the tally under ``key``.

    A compile-cache HIT never traces, and a tally from a call that did not
    walk the elimination must NOT overwrite a real earlier one with zeros
    -- the same caveat `_do_compile_approx` records for the per-face
    counters, except here it is load-bearing for a reward and so is
    handled rather than documented.

    THE TEST IS ``walks``, NOT ``edges``. An all-SKIP plan walks the whole
    order and stores NOTHING, so `edges == 0` is a real measurement there
    -- the most important one the channel takes. Reading it as "no walk"
    made the destruction ceiling read 0.0 = "not measured".
    """
    if not on:
        return
    try:
        from graphax.core import (disarm_store_accounting,
                                  store_accounting_totals)
        disarm_store_accounting()
        t = store_accounting_totals()
    except Exception:
        return
    if int(t.get("walks", 0)) <= 0:
        return
    if len(store) >= 64:
        store.clear()
    store[bytes(key)] = t


def _record_sparsity(ratio: float, cells_ratio: float, channel: float,
                     approx: dict, exact: dict, fallbacks: int,
                     wall_s: float) -> None:
    s = _SPARSITY_STATS
    s["wall_s"] += float(wall_s)
    s["n_fallback_traces"] += int(fallbacks)
    if not np.isfinite(ratio):
        s["n_undefined"] += 1
        return
    s["n"] += 1
    s["sum"] += float(channel)
    s["min"] = min(s["min"], float(channel))
    s["max"] = max(s["max"], float(channel))
    s["sum_ratio"] += float(ratio)
    s["min_ratio"] = min(s["min_ratio"], float(ratio))
    s["max_ratio"] = max(s["max_ratio"], float(ratio))
    if np.isfinite(cells_ratio):
        s["sum_cells"] += float(cells_ratio)
    s["last"] = {"ratio": float(ratio), "cells_ratio": float(cells_ratio),
                 "channel": float(channel),
                 "approx_bytes": int(approx.get("bytes", 0)),
                 "exact_bytes": int(exact.get("bytes", 0)),
                 "approx_edges": int(approx.get("edges", 0)),
                 "exact_edges": int(exact.get("edges", 0))}


_SPARSITY_FAIL_WARNED: list = []


def _record_sparsity_failure(exc: BaseException) -> None:
    """FAIL SOFT, LOUDLY -- the same policy the exact reference uses. A tally
    we could not take is apparatus, and scoring the plan for it would be the
    measurement bias `_truncated_reward` warns about."""
    _SPARSITY_STATS["n_failed"] += 1
    if not _SPARSITY_FAIL_WARNED:
        _SPARSITY_FAIL_WARNED.append(1)
        print("[sparsity] WARNING: the stored-byte tally could not be taken; "
              "the channel reads 0.0 = not measured for every affected plan: "
              f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)


def consume_sparsity_stats() -> dict:
    """Pop the per-period sparsity aggregate (mirrors the other pollers)."""
    s = _SPARSITY_STATS
    n = max(int(s["n"]), 0)
    out = {
        "count": n,
        "mean": (s["sum"] / n) if n else float("nan"),
        "min": s["min"] if n else float("nan"),
        "max": s["max"] if n else float("nan"),
        "ratio_mean": (s["sum_ratio"] / n) if n else float("nan"),
        "ratio_min": s["min_ratio"] if n else float("nan"),
        "ratio_max": s["max_ratio"] if n else float("nan"),
        "cells_ratio_mean": (s["sum_cells"] / n) if n else float("nan"),
        "undefined": int(s["n_undefined"]),
        "failed": int(s["n_failed"]),
        "fallback_traces": int(s["n_fallback_traces"]),
        "wall_s": s["wall_s"],
        "wall_amortised_s": (s["wall_s"] / n) if n else float("nan"),
        "last": s["last"],
    }
    s.update({"n": 0, "sum": 0.0, "min": 1.0, "max": -1.0, "sum_ratio": 0.0,
              "min_ratio": float("inf"), "max_ratio": 0.0, "sum_cells": 0.0,
              "n_undefined": 0, "n_failed": 0, "n_fallback_traces": 0,
              "wall_s": 0.0})
    return out


def _residual_scores(exact_out, approx_out, has_aux: bool):
    """``(rel_frob, cos)`` of ``approx_out`` against ``exact_out``, STREAMED.

    The same accumulator as `_quality_metrics` (``_gradient_similarity``: the
    per-leaf <e,a>, ||e||^2, ||a||^2, ||e-a||^2 that exists precisely so a
    >=4 GB flat concatenate is never materialised, with the exact-structure
    check of ticket .62 -- a mismatch RAISES ``GradientStructureMismatch``,
    a dead path is zero, an implicit dim is compared analytically). This
    variant exists only because the fidelity path holds the two outputs at a
    different place in `_callback`. Returns ``(nan, nan)`` when the exact side
    has nothing to compare against.
    """
    e_out = exact_out[1] if has_aux else exact_out
    a_out = approx_out[1] if has_aux else approx_out
    leaves_e = _gradient_leaves(e_out)
    if not leaves_e or all(x is None for x in leaves_e):
        return float("nan"), float("nan")
    dot, ee, aa, rr, _total = _gradient_similarity(e_out, a_out, "fidelity")
    if _total == 0:
        return float("nan"), float("nan")
    exact_norm = jnp.sqrt(ee)
    approx_norm = jnp.sqrt(aa)
    rel_frob = jnp.sqrt(jnp.maximum(rr, 0.0)) / jnp.maximum(exact_norm, jnp.sqrt(1e-7))
    cos = jnp.real(dot / (jnp.maximum(exact_norm, jnp.sqrt(1e-7))
                          * jnp.maximum(approx_norm, jnp.sqrt(1e-7))))
    return float(rel_frob), float(cos)


def _exact_ref_scores(compile_fn, eval_args, has_aux: bool, approx_out,
                      want_cos: bool = False):
    """ONE exact execution serving the fidelity channel (and the cosine log).

    Returns ``(rel_frob, cos)``; ``cos`` is ``None`` unless ``want_cos``.

    The exact Jacobian is the expensive object: execute once, score
    everything, drop. (Until 2026-09-03 this call also fed the gradient-
    coverage guard's per-leaf norms and memoised them; the guard is gone.)
    """
    out = compile_fn()(*eval_args)
    _FIDELITY_STATS["exact_execs"] += 1
    try:
        rel_frob, cos = _residual_scores(out, approx_out, has_aux)
    finally:
        out = None
    return rel_frob, (cos if want_cos else None)


def consume_truncated_plan_count() -> int:
    n = _TRUNCATED_PLANS[0]
    _TRUNCATED_PLANS[0] = 0
    return n


# Plans graphax could not TRACE (see _is_graphax_trace_failure in _callback).
# Kept separate from the OOM count: OOM is a size problem and scales with the
# plan, a trace failure is a library gap and scales with nothing we control.
_UNTRACEABLE_PLANS = [0]
_UNTRACEABLE_SEEN: set = set()


def _record_untraceable_plan(exc: BaseException) -> None:
    _UNTRACEABLE_PLANS[0] += 1
    _TRUNCATED_PLANS[0] += 1
    _DEGENERATE_PLANS[0] += 1
    key = f"{type(exc).__name__}: {str(exc)[:120]}"
    if key not in _UNTRACEABLE_SEEN:
        _UNTRACEABLE_SEEN.add(key)
        print(f"[trunc] graphax cannot trace this plan (excluded from "
              f"gradient, distinct #{len(_UNTRACEABLE_SEEN)}): {key}",
              flush=True)


def consume_untraceable_plan_count() -> int:
    n = _UNTRACEABLE_PLANS[0]
    _UNTRACEABLE_PLANS[0] = 0
    return n


def consume_zero_work_plan_count() -> int:
    n = _ZERO_WORK_PLANS[0]
    _ZERO_WORK_PLANS[0] = 0
    return n


def consume_degenerate_plan_count() -> int:
    """Pop the count of terminal plans that computed nothing and were
    sentinelled (see the degenerate-plan guard in ``_callback``)."""
    n = _DEGENERATE_PLANS[0]
    _DEGENERATE_PLANS[0] = 0
    return n


# Per-face application telemetry: how much of the policy's intent actually
# survived per-face masking. ``applied`` / ``skipped`` / ``skipped_raised``.
_PER_FACE_STATS: dict = {}


def consume_per_face_stats() -> dict:
    """Pop the per-period per-face apply counts (mirrors the other pollers)."""
    out = dict(_PER_FACE_STATS)
    _PER_FACE_STATS.clear()
    # Only the TWO aggregate buckets count toward the denominator — the
    # per-class ``applied_<kind>`` / ``skipped_<kind>`` keys added for the
    # approximation histogram are a PARTITION of those two, so summing every
    # value (the old formula) double-counted and halved the fraction.
    total = (out.get("applied", 0) + out.get("skipped", 0)
             + out.get("skipped_raised", 0)) or 1
    out["applied_fraction"] = out.get("applied", 0) / total
    return out


# ---------------------------------------------------------------------------
# A6 -- THE PLAN LOG. Every TERMINAL plan, the LOSERS included.
# ---------------------------------------------------------------------------
# ``ppo._dump_pareto`` persists the FRONT. Nothing persisted the plans that
# lost, and X3 -- "diff the lowered graphs of the recorded losers against
# identity and attribute the regression" -- has no input without them. This is
# the producing half: the measurement callback appends ONE record per terminal
# plan to a process-local list, the trainer drains it (its own, plus every
# measure actor's, via ``measure_pool.merge_pool_plan_records``) once per
# episode and appends it to a JSONL. See ``common/plan_log.py`` for the schema
# and for why the replayable spec is the raw integer WIRE and not a rendered
# ``seq``.
#
# WHY THE RECORD IS BUILT HERE AND NOT IN THE TRAINER. One of the things
# the record must carry only exists in this process:
#   * the per-kind APPLIED / IDEMPOTENT-NO-OP counts. ``_PER_FACE_STATS`` is a
#     module global written by the per-face legality hook and consumed per LOG
#     STEP, which is why ppo.py's own comment says the applied counts "CANNOT
#     be attributed to a plan without changing env.py's telemetry". The plan
#     log is that change: a snapshot at callback entry differenced at the
#     return attributes exactly this plan's hook invocations, because one
#     callback measures one plan and the pollers cannot interleave with it (a
#     Ray actor runs one task at a time; the trainer's io_callback is
#     synchronous on the main thread). The delta is clamped at 0 so a poll
#     that did interleave under-reports rather than reporting a negative.
#   (Until 2026-09-03 the record also carried the per-leaf gradient-coverage
#   census and a ``sentinelled`` flag for plans the coverage guard refused;
#   both went with the guard, ticket dsnn-3qm.15.)
#
# COST. Everything written is already materialised host-side; nothing here
# adds a device operation, an executable, an exact reference or a compile.
# With the flag OFF the whole feature is two ``dict`` reads and a one-element
# list per callback.
_PLAN_LOG_ENV = "ALPHAGRAD_PLAN_LOG"
_PLAN_LOG_CAP_ENV = "ALPHAGRAD_PLAN_LOG_CAP"
_PLAN_LOG_MAX_FACES_ENV = "ALPHAGRAD_PLAN_LOG_MAX_FACES"
_PLAN_RECORDS: list = []
_PLAN_LOG_DROPPED = [0]
_PLAN_LOG_WARNED: list = []
# TERMINAL callbacks this process saw with the log ON. The DENOMINATOR for
# `records`: 0 records with 0 terminals means this process measured nothing,
# 0 records with N terminals means the recorder itself dropped them. Without
# it "the plan log is empty" is not a diagnosable statement.
_PLAN_LOG_TERMINALS = [0]
# THE ENV SLOT THIS CALLBACK IS SERVING (gate G5's join). `_batched_host`
# loops over the batch SERIALLY and in slot order, and `_callback` runs
# inside that loop, so the slot the loop is on IS the env row the trainer
# will store the returned reward vector in. The plan record stamps it as
# `env_index`, which gives gate_telemetry.match_records_to_envs an identity
# to join on instead of two float reward slots. -1 = unknown: an unbatched
# call (reset), or a measure actor, which never sees the trainer's slots.
_ENV_SLOT = [-1]


def current_env_slot() -> int:
    """The env row of the callback running right now, or -1."""
    return int(_ENV_SLOT[0])


# ---------------------------------------------------------------------------
# THE ROLLOUT SHARDS' MEASUREMENT RENDEZVOUS (--rollout-shards).
#
# With one shard per GPU, each shard's step callback would make its own pool
# call, and `CpuApproxPool._pick()` sentinels every slot it cannot get an
# actor for -- seven of eight shards would measure nothing. The rendezvous
# gathers the shards' rows into ONE call instead. It is a process-wide object
# because the callback closure is built at TRACE time, inside a jit, and
# `tokenize` has nowhere else to find it. `None` = no shards.
_SHARD_GATHER = [None]


def set_shard_gather(gather) -> None:
    """Install (or, with None, remove) the rollout shards' rendezvous."""
    _SHARD_GATHER[0] = gather


def shard_gather():
    """The installed rendezvous, or None."""
    return _SHARD_GATHER[0]


# ---------------------------------------------------------------------------
# THE PIPELINED TERMINAL MEASUREMENT (owner ruling 2026-09-14).
#
# The terminal rewards of episode e are read by exactly one thing, the PPO
# update of episode e. Nothing in the rollout of e needs them: `done` is 1 at
# the terminal step, so GAE masks the bootstrap value the terminal tokens feed,
# and the terminal delta is never written to the episode stream (a step writes
# the PREVIOUS elimination's delta, and there is no step after the last one).
# So the terminal step can SUBMIT its plans to the measure pool and return a
# placeholder, and the driver can put device work in front of the wait.
#
# A TICKET is an int the DRIVER opens before a rollout attempt and closes
# after it. The terminal callback submits under whichever ticket is open and
# records the environment rows it submitted for; the driver later collects the
# ticket (blocking) or drops it (a discarded attempt). The ticket is the join
# key beside `env_index`: one ticket is one attempt at one episode, and the
# rows under it are that attempt's environments.
#
# THERE IS AT MOST ONE OPEN TICKET AND AT MOST ONE SUBMISSION PER TICKET.
# Both are checked rather than assumed, because a second submission would
# silently discard the first attempt's measurement.
# ---------------------------------------------------------------------------
_MEASURE_TICKETS: dict = {}
_MEASURE_TICKET_SEQ = [0]
_MEASURE_TICKET_OPEN = [None]

# ---------------------------------------------------------------------------
# WHERE THE PER-STEP TOKENIZATION RUNS (--tokenize-where, ruling 2026-09-15).
#
# A non-terminal callback row measures NOTHING under `terminal_rewards_only`:
# it tokenizes the elimination prefix, decides face legality, and returns the
# delta observation. Routing it to the measurement actors cost a Ray round
# trip per step AND held the actors, so a terminal measurement left in flight
# blocked the next rollout step for step.
#
# `pool`       -- the historical route: the measure actors serve it.
# `local`      -- the trainer process serves it, as the non-pooled path
#                 always has.
# `cpu-actors` -- a SECOND pool of CPU-only actors serves it and the measure
#                 pool is left for terminals only.
#
# A MODULE GLOBAL and not a field on the env, because the env is a JAX pytree
# and the host callback only ever runs in the trainer process. `pool` is the
# import-time value, so anything that never calls the setter behaves exactly
# as it did before this existed.
# ---------------------------------------------------------------------------
_TOKENIZE_WHERE = ["pool"]
_TOKENIZE_POOL = [None]
# Does the terminal callback START its pool submission, or only package it for
# the driver to start? See `start_measurement`.
_MEASURE_DEFER = [False]
TOKENIZE_WHERE_CHOICES = ("pool", "local", "cpu-actors")


def set_tokenize_where(where: str, pool=None) -> None:
    """Install the per-step tokenization route. Raises on an unknown one."""
    where = str(where)
    if where not in TOKENIZE_WHERE_CHOICES:
        raise ValueError(
            f"--tokenize-where {where!r} is not one of "
            f"{list(TOKENIZE_WHERE_CHOICES)}")
    if where == "cpu-actors" and pool is None:
        raise ValueError(
            "--tokenize-where cpu-actors needs the CPU-only pool handed in: "
            "set_tokenize_where('cpu-actors', pool=<CpuApproxPool>)")
    _TOKENIZE_WHERE[0] = where
    _TOKENIZE_POOL[0] = pool


def tokenize_where() -> str:
    return _TOKENIZE_WHERE[0]


def tokenize_pool():
    return _TOKENIZE_POOL[0]


# THE TRAINER'S OWN COPY OF THE BOUND OPERANDS.
#
# `VertexEliminationEnv.step` hands the callback ZERO-LENGTH PLACEHOLDERS for
# `args`, `consts` and the eval samples whenever the pool already owns them as
# an ObjectRef: marshalling tens of megabytes device-to-host on every one of
# the ninety-five decisions, so a remote closure could throw them away, was
# pure waste. A row served IN THIS PROCESS needs the real ones, and until
# `--tokenize-where` there was no such row in that configuration -- the
# comment at that call site claimed `ALPHAGRAD_POOL_TERMINAL_LOCAL=1` still
# got the real thing, and it did not.
#
# So the driver installs the env's own concrete operands here, once, and the
# locally served rows read them from here. They are the same constants a
# measure actor tokenizes from (its own env's bound args), they do not change
# between steps, and nothing is marshalled per step.
_LOCAL_BOUND = [None]


def set_local_bound_operands(args, consts, eval_samples=None) -> None:
    """Install the concrete `(args, consts, eval_samples)` for local rows."""
    _LOCAL_BOUND[0] = (args, consts, eval_samples)


def local_bound_operands():
    return _LOCAL_BOUND[0]


class MeasureTicketError(RuntimeError):
    """Misuse of the pipelined-measurement ticket protocol."""


def open_measure_ticket() -> int:
    """Open a ticket for the rollout attempt that is about to run.

    Returns the ticket. Raises if one is already open -- two open tickets
    would mean two attempts in flight, and the pool serves one batch at a
    time.
    """
    if _MEASURE_TICKET_OPEN[0] is not None:
        raise MeasureTicketError(
            f"measurement ticket {_MEASURE_TICKET_OPEN[0]} is still open; "
            f"close it before opening another (one rollout attempt at a "
            f"time).")
    _MEASURE_TICKET_SEQ[0] += 1
    t = int(_MEASURE_TICKET_SEQ[0])
    _MEASURE_TICKET_OPEN[0] = t
    _MEASURE_TICKETS[t] = {"future": None, "deferred": None,
                           "env_index": [], "n_envs": 0, "step": -1}
    return t


def current_measure_ticket():
    """The ticket the terminal callback should submit under, or None."""
    return _MEASURE_TICKET_OPEN[0]


def close_measure_ticket():
    """Stop routing terminal plans to the open ticket. Returns it, or None.

    The ticket stays in the table until it is collected or dropped: closing
    it only says that the rollout that fills it has finished.
    """
    t, _MEASURE_TICKET_OPEN[0] = _MEASURE_TICKET_OPEN[0], None
    return t


def measure_ticket_rows(ticket) -> list:
    """The environment rows submitted under ``ticket``, in submission order."""
    return list(_MEASURE_TICKETS[int(ticket)]["env_index"])


def pending_measure_tickets() -> list:
    """Every ticket that has been opened and neither collected nor dropped."""
    return sorted(int(t) for t in _MEASURE_TICKETS)


def _record_measure_submission(ticket, future, env_index, n_envs, step,
                               deferred=None):
    """Remember one pool submission so the driver can collect or drop it.

    ``future`` is a submission that has already started. ``deferred`` is a
    zero-argument callable that STARTS one, and exactly one of the two is
    given. See :func:`start_measurement` for why the deep pipeline hands the
    start to the driver.
    """
    rec = _MEASURE_TICKETS.get(int(ticket))
    if rec is None:
        raise MeasureTicketError(
            f"measurement ticket {ticket} is not open; the driver must open "
            f"one before the rollout that submits under it.")
    if (future is None) == (deferred is None):
        raise MeasureTicketError(
            f"measurement ticket {ticket}: give exactly one of a started "
            f"future and a deferred starter.")
    if rec["future"] is not None or rec["deferred"] is not None:
        raise MeasureTicketError(
            f"measurement ticket {ticket} already carries a submission for "
            f"rows {rec['env_index']}; a rollout submits its terminal plans "
            f"exactly once.")
    rec["future"] = future
    rec["deferred"] = deferred
    rec["env_index"] = [int(i) for i in env_index]
    rec["n_envs"] = int(n_envs)
    rec["step"] = int(step)


def measure_defer_enabled() -> bool:
    """Does the terminal callback DEFER its submission to the driver?"""
    return bool(_MEASURE_DEFER[0])


def set_measure_defer(on: bool) -> None:
    """Arm or disarm the deferred submission (the deep pipeline)."""
    _MEASURE_DEFER[0] = bool(on)


def start_measurement(ticket) -> bool:
    """Start a DEFERRED submission. Returns False when there was none.

    THE DEEP PIPELINE OWES THE ACTORS AN EMPTY WINDOW. It overlaps episode
    e's measurement with the ROLLOUT of e+1, so at the end of that rollout
    e+1's terminal step wants to submit while e has not been collected. If it
    submitted there, e+1's measurement could start the moment e's returned --
    and it would then be writing plan records into the actors while the
    driver was still draining e's out of them.

    So under the deep pipeline the terminal step only PACKAGES its batch, and
    the driver starts it here: after it has collected e's rewards and drained
    e's records, and before it takes the next rollout. One batch is in flight
    at any time, and every record in an actor belongs to exactly one episode.
    """
    rec = _MEASURE_TICKETS.get(int(ticket))
    if rec is None:
        raise MeasureTicketError(
            f"measurement ticket {ticket} was never opened, or has already "
            f"been collected or dropped.")
    start = rec["deferred"]
    if start is None:
        return False
    rec["deferred"] = None
    rec["future"] = start()
    return True


def collect_measurement(ticket) -> dict:
    """BLOCK until ``ticket``'s terminal measurement is back; return it.

    ``{"rewards": (E, NUM_REWARDS) float32, "sentinel": (E,) bool,
    "env_index": [...], "step": int}`` -- the rewards in ENVIRONMENT ROW
    order, which is the order the trajectory stores them in.

    Every environment must have submitted exactly one terminal plan. A
    missing or repeated row would put one environment's measurement on
    another's trajectory, so it raises instead of filling a gap.
    """
    t = int(ticket)
    rec = _MEASURE_TICKETS.get(t)
    if rec is None:
        raise MeasureTicketError(
            f"measurement ticket {t} was never opened, or has already been "
            f"collected or dropped.")
    if rec["future"] is None and rec["deferred"] is not None:
        # THE TICKET SURVIVES THIS. Collecting before the start is the
        # driver's mistake and the measurement is still startable, so the
        # record stays in the table rather than being popped on the way out.
        raise MeasureTicketError(
            f"measurement ticket {t} was packaged but never started: the "
            f"driver must call start_measurement({t}) before collecting it.")
    rec = _MEASURE_TICKETS.pop(t)
    fut = rec["future"]
    if fut is None:
        raise MeasureTicketError(
            f"measurement ticket {t} carries no submission: the rollout "
            f"never reached a terminal step with the pipeline armed.")
    rows = list(rec["env_index"])
    n = int(rec["n_envs"])
    if sorted(rows) != list(range(n)):
        raise MeasureTicketError(
            f"measurement ticket {t} submitted rows {sorted(rows)} but the "
            f"rollout has {n} environments; every environment must submit "
            f"exactly one terminal plan.")
    _pb = fut.result()
    rewards = np.asarray(_pb[-2], dtype=np.float32)
    sentinel = np.asarray(_pb[-1]).astype(bool)
    if rewards.shape != (len(rows), NUM_REWARDS):
        raise MeasureTicketError(
            f"measurement ticket {t} came back with rewards of shape "
            f"{rewards.shape}, expected ({len(rows)}, {NUM_REWARDS}).")
    out = np.zeros((n, NUM_REWARDS), dtype=np.float32)
    smask = np.zeros((n,), dtype=bool)
    for k, i in enumerate(rows):
        out[i] = rewards[k]
        smask[i] = bool(sentinel[k])
    return {"rewards": out, "sentinel": smask, "env_index": rows,
            "step": int(rec["step"]), "ticket": t}


def drop_measurement(ticket) -> dict:
    """DRAIN ``ticket``'s measurement and throw it away (a discarded attempt).

    The attempt that submitted it is gone, so its rewards must not reach any
    trajectory. The submission is still DRAINED rather than abandoned: the
    actors are measuring it, Ray runs one task per actor, and the repeat's own
    tokenization queues behind whatever is still running. Abandoning the
    future would leave the actors busy with work nobody will ever read, and
    `ray.cancel` on an actor task needs `force=True`, which kills the actor
    and its plan records with it.

    A DEFERRED submission that was never started needs no drain at all: it
    never reached an actor, so nothing there is measuring it and nothing there
    holds a record of it. That is the one real simplification the deep
    pipeline buys the discard.

    Returns ``{"rows": [...], "drained": bool}``; never raises on a ticket
    that carries no submission (a rollout can overflow before its terminal
    step).
    """
    t = int(ticket)
    rec = _MEASURE_TICKETS.pop(t, None)
    if rec is None:
        return {"rows": [], "drained": False}
    fut = rec["future"]
    if fut is None and rec["deferred"] is not None:
        return {"rows": list(rec["env_index"]), "drained": False,
                "deferred": True}
    if fut is None:
        return {"rows": [], "drained": False}
    try:
        fut.result()
    except Exception as _exc:                                  # noqa: BLE001
        # A discarded attempt's measurement failing is not this run's
        # problem: the result is thrown away either way. Say so and move on.
        print(f"[measure-pipeline] discarded attempt's measurement (ticket "
              f"{t}) failed while draining: {type(_exc).__name__}: "
              f"{str(_exc)[:160]}", flush=True)
    return {"rows": list(rec["env_index"]), "drained": True}

# The four telemetry buckets the per-face hook maintains, per approximation
# class. ``other`` exists because ``masks._kind_of`` emits it for a rule that
# is none of the three -- dropping it here would make the record's totals
# disagree with the hook's.
_PLAN_FACE_KINDS = ("diag", "compress", "quant", "other")


def plan_log_enabled() -> bool:
    """Is the A6 plan log on in THIS process? (``ALPHAGRAD_PLAN_LOG``)"""
    return os.environ.get(_PLAN_LOG_ENV, "0") not in ("", "0")


def _plan_log_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _plan_log_cap() -> int:
    """Ring capacity between two drains. A record past it is DROPPED and
    COUNTED, never silently overwritten -- the count is logged and printed."""
    return _plan_log_int_env(_PLAN_LOG_CAP_ENV, 4096) or 4096


def _plan_log_max_faces() -> int:
    """0 = record every live face (the default, and the only replayable
    setting). See ``plan_log.encode_wires``."""
    return _plan_log_int_env(_PLAN_LOG_MAX_FACES_ENV, 0)


# WHICH ATTEMPT AT THIS EPISODE THE RECORD BELONGS TO. An episode whose
# token stream overflows its bin is DISCARDED and repeated one bin up, and
# both attempts stamp the same ``episode``. Verification jobs 65410 and
# 65413 found three record PAIRS that were byte-identical, with no field
# telling them apart, which makes the plan log unreadable exactly where it
# matters most. The driver rolls the discarded attempt's records back (see
# ``episode_telemetry_snapshot``), and this stamp says which attempt the
# surviving ones came from, so a reader never has to infer it.
#
# NOT part of the telemetry snapshot, on purpose: it has to keep counting
# ACROSS a rollback, which is the one thing a rollback must not undo.
_PLAN_LOG_ATTEMPT = [0]


def set_plan_log_attempt(attempt: int) -> None:
    """Say which attempt at the current episode is running (0 = the first).

    Called by the episode driver at the top of every attempt, including the
    repeats.
    """
    _PLAN_LOG_ATTEMPT[0] = int(attempt)


def plan_log_attempt() -> int:
    """The attempt index records are being stamped with right now."""
    return int(_PLAN_LOG_ATTEMPT[0])


#: THE CONTAINER THE PLAN JUST MEASURED IMPLIED (owner ruling 2026-09-16).
#: Written by `_callback_measured` at the container seam and read by
#: `_record_plan`, which is the one choke point every record passes through.
#: `None` on every target that carries no temporal edge.
_PLAN_CARRY: list = [None]


def _record_plan(rec: dict) -> None:
    if len(_PLAN_RECORDS) >= _plan_log_cap():
        _PLAN_LOG_DROPPED[0] += 1
        return
    rec["attempt"] = int(_PLAN_LOG_ATTEMPT[0])
    # THE STEP POSITION (owner ruling 2026-09-16). On the one-step recurrent
    # target the graph is the same at every step position and only the GIVEN
    # VALUES move, so a record that does not say which step it measured cannot
    # be read back against another. The builder is the only place that knows
    # it; this is the one choke point every record passes through. Empty for
    # every other target, and then nothing is written.
    # The PROBE BATCH's draw wins when there is one: under the default
    # grad_cosine quality metric the batch is what the quality channel was
    # actually scored on, it is redrawn per (environment, episode), and the
    # builder's own tuple is only the run's starting point. Falls back to the
    # builder for a run with no probe (quality metric `none`, or a target
    # whose generator declares no `meta`).
    from alphagrad.approx.common.rsnn_shd import last_step_position
    _pos = probe_meta() or last_step_position()
    if _pos:
        rec["step_position"] = _pos
    # WHICH CONTAINER THE CARRY ARRIVED IN (owner ruling 2026-09-16). The
    # given temporal value is the RULE run over the recording, and the
    # container it is stored in is what the memory channel sees, so a record
    # that does not name it cannot be read against another. It is a property
    # of the PLAN, not of the run: the plan's own classes on the carried face
    # chose it, and `_callback_measured` publishes the choice it acted on.
    if _PLAN_CARRY[0] is not None:
        rec["carry_container"] = str(_PLAN_CARRY[0])
    elif _pos and _pos.get("carry") is not None:
        rec["carry_container"] = str(_pos["carry"])
    _PLAN_RECORDS.append(rec)


def consume_plan_records() -> dict:
    """Pop this process's plan records (mirrors the other pollers)."""
    # THE EPISODE LINE (owner ruling 2026-09-14). The trainer drains this
    # process once per episode, so the drain is the one moment an actor is
    # certain an episode is over -- including the LAST one, which no
    # following episode would ever roll over. `flush_measure_episode` is
    # idempotent and silent when there is nothing to report.
    flush_measure_episode()
    _fb_n = int(_MEASURE_COMPILE_FALLBACKS["n"])
    out = {"records": list(_PLAN_RECORDS),
           "dropped": int(_PLAN_LOG_DROPPED[0]),
           "terminals": int(_PLAN_LOG_TERMINALS[0]),
           # MEASURE TOOLCHAIN TELEMETRY (finding 03 sec 6). Drained HERE
           # because the measure compiles happen in THIS process -- the
           # actor under --ray-measure -- and this dict is the one the
           # actor already hands to the trainer. Reading the trainer's own
           # globals is the mistake that made grad_cov/* read 0 (ticket 07).
           # `compile_fallbacks` is the delta since the previous drain;
           # `_total` is the process lifetime count (resets on respawn).
           "compile_fallbacks": max(
               0, _fb_n - int(_MEASURE_FALLBACKS_AT_LAST_DRAIN[0])),
           "compile_fallbacks_total": _fb_n,
           "toolchain_ok": bool(_MEASURE_TOOLCHAIN["ok"]),
           "toolchain_host": str(_MEASURE_TOOLCHAIN["host"]),
           # MEMORY PARITY (ticket .49): the watermark logged beside the
           # channel, per measurement, drained on the same trip for the
           # same reason as the toolchain counters above.
           "mem_parity": consume_mem_parity(),
           # THE PAIRED REFERENCE (ticket .9): one record per rev-exact
           # measurement, same trip, same reason (the trainer logs
           # ref/latency_ns and ref/temp_bytes from these; ticket .45).
           "paired_ref": consume_paired_refs(),
           "enabled": plan_log_enabled(),
           "pid": os.getpid()}
    _PLAN_RECORDS.clear()
    _PLAN_LOG_DROPPED[0] = 0
    _PLAN_LOG_TERMINALS[0] = 0
    _MEASURE_FALLBACKS_AT_LAST_DRAIN[0] = _fb_n
    return out


def _plan_face_delta(before: dict | None, after: dict | None) -> dict:
    """This plan's share of the per-face hook counters.

    A DELTA, because ``_PER_FACE_STATS`` is process-global. Clamped at 0:
    the only way a component can go negative is a poller draining the dict
    mid-callback, and under-reporting that is honest where a negative count
    is not.
    """
    before = before or {}
    after = after or {}

    def d(k):
        return max(0, int(after.get(k, 0)) - int(before.get(k, 0)))

    applied = {kd: d(f"applied_{kd}") for kd in _PLAN_FACE_KINDS}
    applied["total"] = d("applied")
    skipped = {kd: d(f"skipped_{kd}") for kd in _PLAN_FACE_KINDS}
    skipped["raised"] = d("skipped_raised")
    skipped["total"] = d("skipped") + skipped["raised"]
    return {
        "applied": applied,
        "skipped": skipped,
        # The IDEMPOTENT re-request: the rule was refused because the operand
        # ALREADY has that property (DIAG coupled at exactly this
        # granularity, COMPRESS on an implicit/extent-1 axis, QUANT already
        # this dtype). It is the denominator correction that separates "the
        # approximation did not apply" from "it was already there".
        "idempotent_noop": {kd: d(f"skipped_{kd}_noop")
                            for kd in _PLAN_FACE_KINDS},
        "repaired": {kd: d(f"repaired_{kd}") for kd in _PLAN_FACE_KINDS},
    }


def _encode_face_joins(face_joins, face_specs, face_skips):
    """The plan log's ``face_joins`` column: ``[[ [face, mode], ... ], ...]``.

    ONE entry per (vertex, face) whose join bit actually DECIDED something --
    a face that is not skipped and carries at least one live slot row, which is
    exactly the condition :func:`_face_dict_for_vertex` builds an entry under
    and therefore the only place the bit is read. Sparse for the same reason
    ``plan_log.encode_wires`` is: ``MAX_FACES`` is a provable bound (2538 on the
    flagship) against a measured ~1.24 live faces per vertex, so a dense row
    would be >99% padding.

    ``None`` under every ``--approx-add`` value that FIXES the join semantics;
    the record's ``approx_add`` column answers for the whole plan there.
    """
    if face_joins is None:
        return None
    fj = np.asarray(face_joins)
    fs = np.asarray(face_specs)
    fk = np.asarray(face_skips)
    out = []
    for k in range(fj.shape[0]):
        row = []
        for f in range(min(fj.shape[1], fs.shape[1])):
            if int(fk[k, f]) == 1:
                continue
            if not np.any(fs[k, f, :, 0] != -1):
                continue
            row.append([f, join_mode_of_bit(fj[k, f])])
        out.append(row)
    return out


def _record_terminal_plan(*, order, rule_specs, face_specs, face_skips,
                          reward_vec, face_before, face_after,
                          counts_from_trace: bool,
                          mem_parity: dict | None = None,
                          paired_ref: dict | None = None,
                          measure_counts: dict | None = None,
                          measured_from: int | None = None,
                          face_joins=None, refused: str | None = None) -> None:
    """Append ONE terminal plan to this process's plan log. Never raises.

    ``refused`` names the resource limit or fault that stopped the
    measurement (``"degenerate"``, ``"muls-cap"``, ``"no-target-fun"``,
    ``"untraceable"``, ``"oom"``). The record then carries ``refused`` and
    ``sentinelled = True``, and its reward vector is the sentinel the
    callback returned, not a measurement. A refused plan IS a record: the
    log's contract is every terminal plan, win or lose, and a refusal is
    the single most important thing it can report.

    A logging failure must not kill a measurement, but it must not be
    invisible either: the first one prints to stderr with its exception.
    """
    try:
        from alphagrad.approx.common import plan_log as _plog
        from alphagrad.approx.common.masks import (
            reduce_axis_space as _reduce_axis_space)
        rec = {
            "schema": _plog.SCHEMA,
            "pid": os.getpid(),
            "wall_time": time.time(),
            # False = the approx compile came from the cache, so graphax
            # never re-traced the elimination and the per-face hook was
            # never invoked: `applied`/`skipped`/`idempotent_noop` are all
            # 0 because NOTHING WAS COUNTED, not because nothing applied.
            # `requested` is exact either way (it is read off the wire).
            "counts_from_trace": bool(counts_from_trace),
            # HOW THE FACE ADD's TWO ADDENDS MET while this plan was measured
            # (ticket .56, finding 73): "lossy" (both forced into the
            # approximated contraction's container) or "lossless" (the union
            # of the two supports). Read from the same function the face
            # emitter reads, so the record cannot disagree with the
            # measurement. NOT poolable with the retired ``approx_old``
            # column of pre-2026-09-10 records: "same" and "exact" named a
            # different computation -- see env.approx_add's block comment.
            "approx_add": approx_add(),
            # THE PER-FACE JOIN BITS, under --approx-add choose only (None
            # otherwise, which is what `approx_add` above then answers for the
            # whole plan). Recorded because the container a merge reconciled
            # into is part of WHAT WAS MEASURED, and under `choose` the
            # configuration no longer says it -- the head does, per face. A
            # replay that read only `approx_add` would reproduce a different
            # computation. Per eliminated vertex, live faces only, as
            # "lossy"/"lossless" names rather than raw bits so the record does
            # not depend on JOIN_LOSSY's numeric value.
            "face_joins": _encode_face_joins(
                face_joins, face_specs, face_skips),
            # WHICH quantity reward slot 5 holds (ticket .49) and BOTH
            # memory numbers of this plan's timed executable, so a record
            # can be re-scored on the other channel without a re-measure.
            "mem_channel": mem_channel(),
            "mem_temp_bytes": (mem_parity or {}).get("static_temp_bytes"),
            "mem_watermark_bytes": (mem_parity or {}).get(
                "runtime_peak_bytes"),
            "mem_peak_source": (mem_parity or {}).get(
                "peak_source", "not_measured"),
            # WHICH coordinate space the Reduce axes were applied in
            # (ticket .20): "physical" (wire token -> physical val axis at
            # decode -> canonical slot at graphax) or "canonical" (the
            # pre-ticket read). A replay must convert the same way.
            "reduce_axis_space": _reduce_axis_space(),
            # HOW slots 2 and 5 are expressed (ticket .9) and, under
            # paired-log, the rev-exact reference this plan was paired
            # with, in POSITIVE units -- so a record can be re-scored in
            # absolute units (candidate_* fields) or against a different
            # reference without a re-measure. None under absolute.
            "cost_form": cost_form(),
            "ref_latency_ns": (paired_ref or {}).get("latency_ns"),
            "ref_temp_bytes": (paired_ref or {}).get("temp_bytes"),
            "ref_watermark_bytes": (paired_ref or {}).get("watermark_bytes"),
            "candidate_latency_ns": (paired_ref or {}).get(
                "candidate_latency_ns"),
            "candidate_memory_bytes": (paired_ref or {}).get(
                "candidate_memory_bytes"),
            "mem_log_floored": (paired_ref or {}).get("mem_floored"),
            # TICKET dsnn-dfw.44: q05 / median / q95 of this plan's paired
            # per-window log ratios, per objective, so the archive's bands can
            # be rebuilt from the log without a re-measure.
            "ratio_log": (paired_ref or {}).get("ratio_log"),
            # THE COUNTS THIS PLAN WAS ACTUALLY MEASURED WITH (owner ruling
            # 2026-09-14). Under the time budget the protocol is no longer a
            # constant of the run -- a slow plan earns fewer windows than a
            # fast one, and a 121 us reference runs 50 executions per window
            # where an 18 ms candidate runs 5 -- so a record that did not
            # carry them could not be re-read. `measure_secs` is the
            # execution time inside the timed windows of that half, which is
            # what --measure-budget-secs is a target for.
            "measure_inner": (measure_counts or {}).get("inner"),
            "measure_windows": (measure_counts or {}).get("windows"),
            "measure_secs": (measure_counts or {}).get("secs"),
            "ref_measure_inner": (measure_counts or {}).get("ref_inner"),
            "ref_measure_windows": (measure_counts or {}).get("ref_windows"),
            "ref_measure_secs": (measure_counts or {}).get("ref_secs"),
            # THE PLAN THIS ONE'S NUMBERS CAME FROM (owner ruling
            # 2026-09-14). None on every measured plan. An integer names the
            # index, within this episode and this measure actor, of the
            # identical plan that WAS measured; this record then carries no
            # timing fields of its own, because no timing happened.
            "measured_from": (None if measured_from is None
                              else int(measured_from)),
        }
        rec.update(_plog.encode_wires(
            order, rule_specs, face_specs, face_skips,
            max_faces_recorded=_plan_log_max_faces(),
            compress_sentinel=COMPRESS_SENTINEL,
            quant_sentinel=QUANT_SENTINEL))
        # THE CONTENT HASH OF THE FIVE WIRES (see `_plan_content_key`). It is
        # what the per-episode dedupe already keys on, and it is the JOIN a
        # LATE record needs: the plan log is append-only, so the asynchronous
        # gradient oracle's answer cannot be added to this line later. It is
        # written as its own `oracle_result` line naming this hash and this
        # episode. Recorded on EVERY terminal plan, dedupe on or off, because
        # a join key that exists only under a flag is not a join key.
        rec["plan_hash"] = _plan_content_key(
            order, rule_specs, face_specs, face_skips, face_joins).hex()
        rec["rewards"] = [float(x) for x in
                          np.asarray(reward_vec).reshape(-1).tolist()]
        rec["reward_names"] = list(REWARD_NAMES)
        # THE ENV ROW THIS PLAN WAS MEASURED FOR (gate G5's join,
        # gate_telemetry.ENV_INDEX_KEY). Omitted, not set to -1, when the
        # slot is unknown: the gate's identity join is all-or-nothing, and a
        # -1 would claim an identity this record does not have.
        _slot = current_env_slot()
        if _slot >= 0:
            rec["env_index"] = _slot
        rec.update(_plan_face_delta(face_before, face_after))
        # Degraded-fusion compiles taken WHILE THIS PLAN WAS MEASURED (a
        # delta since the previous record in this pid; measurements are
        # sequential per actor). Non-zero = this plan's latency came from a
        # less-fused executable and is not comparable with a plan at 0.
        # `toolchain_ok` False = the link toolchain of this node failed the
        # gate probe or a real measure compile (finding 03).
        _fb_n = int(_MEASURE_COMPILE_FALLBACKS["n"])
        rec["compile_fallbacks"] = max(
            0, _fb_n - int(_MEASURE_FALLBACKS_AT_LAST_RECORD[0]))
        rec["compile_fallbacks_total"] = _fb_n
        rec["toolchain_ok"] = bool(_MEASURE_TOOLCHAIN["ok"])
        _MEASURE_FALLBACKS_AT_LAST_RECORD[0] = _fb_n
        if refused is not None:
            rec["refused"] = str(refused)
            rec["sentinelled"] = True
            rec["replayable"] = False
        _record_plan(rec)
    except Exception as _exc:          # pragma: no cover - telemetry only
        if not _PLAN_LOG_WARNED:
            _PLAN_LOG_WARNED.append(1)
            print(f"[plan-log] WARNING: could not record a terminal plan; "
                  f"the log will be INCOMPLETE: {type(_exc).__name__}: "
                  f"{_exc}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# XLA-analysis side-channel. ONE memory channel exists — ``peak_memory`` — and
# the deterministic ``memory_analysis()`` estimate is substituted INTO it in
# place wherever the runtime high-water mark is unavailable (see
# ``_note_static_peak_fallback``). This side-channel therefore no longer
# exports a second memory NUMBER; it carries only the approx/exact memory
# COMPRESSION RATIO, which is a different quantity (dimensionless, and about
# the exact executable as much as the approximated one). It travels host-side
# like the tokenization stats: `_callback` records at each TERMINAL
# measurement, the driver polls once per episode via
# `consume_memory_compression_stats`.
# ---------------------------------------------------------------------------
_XLA_MEM_APPROX: list = []   # bytes per terminal measurement this period
_XLA_MEM_EXACT: list = []    # bytes; aligned with _XLA_MEM_APPROX where known


def _record_xla_memory(approx_bytes: float, exact_bytes: float | None) -> None:
    _XLA_MEM_APPROX.append(float(approx_bytes))
    _XLA_MEM_EXACT.append(float(exact_bytes) if exact_bytes is not None else 0.0)


def _memory_analysis_bytes(compiled) -> float | None:
    """Deterministic XLA peak estimate (temp + output + argument bytes) for a
    compiled executable; None when memory_analysis is unavailable."""
    try:
        ma = compiled.memory_analysis()
        if ma is None:
            return None
        return float(
            getattr(ma, "temp_size_in_bytes", 0)
            + getattr(ma, "output_size_in_bytes", 0)
            + getattr(ma, "argument_size_in_bytes", 0)
        )
    except Exception:
        return None


def consume_memory_compression_stats() -> dict:
    """Pop the per-period approx/exact memory COMPRESSION ratio.

    ``compression_ratio`` = mean(exact/approx) over measurements where both
    sides were analyzable — >1 means the approximated executable is smaller
    than the exact one (the observable sparsity / compression proxy under the
    dense measurement pipeline). It deliberately does NOT return an absolute
    memory number: ``peak_memory`` is the single memory channel.
    """
    if not _XLA_MEM_APPROX:
        return {"compression_ratio": 0.0, "count": 0}
    approx = np.asarray(_XLA_MEM_APPROX, dtype=np.float64)
    exact = np.asarray(_XLA_MEM_EXACT, dtype=np.float64)
    both = (approx > 0) & (exact > 0)
    ratio = float(np.mean(exact[both] / approx[both])) if both.any() else 0.0
    out = {"compression_ratio": ratio, "count": int(approx.size)}
    _XLA_MEM_APPROX.clear()
    _XLA_MEM_EXACT.clear()
    return out
# Upper bound on rule_specs rows per vertex. In dynamic-substeps mode this
# also bounds the number of typed micro-actions per vertex that survive
# :func:`micro_actions_to_rule_specs_jax` — set it to the same scale as
# the policy's ``max_substeps`` (≈ 2 × MAX_AXES_PER_VERTEX) so the
# translator doesn't silently truncate DIAG / COMPRESS rows the policy
# emitted. Memory cost is O(total_v × MAX_RULES_PER_VERTEX × 3) int32.
MAX_RULES_PER_VERTEX = 16

# P1 per-path action space: per chosen vertex, up to MAX_FACES local paths
# (|fan-in| x |fan-out| faces, padded), each with one skip gate and one spec
# row per slot (pre / post / new). Must match the oracle's
# ``face_masks(v, max_faces)`` budget — both enumerate faces in the SAME
# canonical visit order (``faces_of``).
# NOT a knob. The width is the provable per-graph bound, set by
# ``configure_max_faces`` at launch: faces(v) = |live preds| x |live succs|,
# elimination only contracts paths, so live neighbours are subsets of the
# ORIGINAL ancestors/descendants and faces(v) <= |anc(v)|*|desc(v)| for every
# order. No order can exceed it, so nothing can be truncated. The history
# that mandates this: 8 silently dropped faces 9..12 of the xent graph's
# vertex 9 -- they ran exact while every counter reported a healthy run.
# ALPHAGRAD_MAX_FACES survives only as an explicit experiment override.
MAX_FACES = int(os.environ.get("ALPHAGRAD_MAX_FACES", "0")) or 16
_FACE_CAP_STATS = {"max_seen": 0}


def derived_max_faces(jaxpr, argnums, consts, args) -> int:
    """max_v |ancestors(v)| * |descendants(v)| over the PRUNED Jacobian
    graph -- an upper bound on any vertex's face count under any order."""
    from graphax.incremental import IncrementalJaxpr
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=False)
    g = {k: set(v.keys()) for k, v in ij.graph.items()}
    tg = {k: set(v.keys()) for k, v in ij.tgraph.items()}
    for d, o in ((g, tg), (tg, g)):
        for n in o:
            d.setdefault(n, set())

    def closure(adj, start):
        seen, stack = set(), [start]
        while stack:
            for w in adj.get(stack.pop(), ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    best = 1
    for v in g:
        b = len(closure(tg, v)) * len(closure(g, v))
        if b > best:
            best = b
    return int(best)


def configure_max_faces(n: int) -> None:
    """Rebind the face width BEFORE any shape is built from it. The env
    var, if set, wins -- it is the explicit experiment override."""
    global MAX_FACES
    if os.environ.get("ALPHAGRAD_MAX_FACES"):
        return
    MAX_FACES = int(n)


def consume_face_cap_stats() -> dict:
    return dict(_FACE_CAP_STATS)


# THE FACE WIRE'S WIDTH ON THE *LIVE-FACE HOST CALLBACKS*, which is not the
# state's width and must never be confused with it.
#
# `MAX_FACES` is the PROVABLE bound above, and the state keeps every column of
# it. But the four per-step face callbacks take the whole elimination-prefix
# history as an operand -- `(N, MAX_FACES, FACE_SLOTS, 3)` int32, 6.6 megabytes
# per environment on the campaign graph -- and the rollout profile of
# 2026-09-15 measured what that costs: 59.4 gigabytes copied device to host per
# episode, 625 megabytes per step, four of the five callback instructions
# moving 117 megabytes each. The same profile measured the OCCUPANCY: median 1
# face per vertex, maximum 13 in two episodes, 0.069 percent of the cap.
#
# So the callbacks may be handed the first `face_wire_faces()` columns instead
# of all of them. This is NOT a lowered bound and NOTHING is allowed to fall
# off the end: `live_faces.n_faces` raises the moment a vertex has more faces
# than the wire carries, before that vertex's decisions are ever written, and
# `live_faces._decided` raises if a prefix row is narrower than the face list
# it is being indexed by. 0 means "the full width", which is the historical
# wire byte for byte.
_FACE_WIRE_FACES = [0]


def configure_face_wire_faces(n: int) -> None:
    """Set the live-face callbacks' wire width (0 = the full MAX_FACES)."""
    n = int(n)
    if n < 0:
        raise ValueError(
            f"--face-wire-faces must be 0 (the full width) or positive, "
            f"got {n}")
    _FACE_WIRE_FACES[0] = n


def face_wire_faces() -> int:
    """The number of face columns the live-face callbacks are handed."""
    n = _FACE_WIRE_FACES[0]
    return MAX_FACES if n <= 0 else min(int(n), MAX_FACES)


# ---------------------------------------------------------------------------
# THE ELIMINATION-PREFIX FACE HISTORY, KEPT ON THE HOST (owner ruling
# 2026-09-15, item 2).
#
# Narrowing the wire to `face_wire_faces()` columns took the rollout's
# device-to-host traffic from 59.4 gigabytes per episode to about 2.4. What is
# left is still the WHOLE PREFIX: every face callback, and the env step
# callback, is handed `(N, W, FACE_SLOTS, 3)` int32 plus `(N, W)` skips, for a
# 95-step episode at sixteen environments, on every one of about four calls per
# step. The rows below `step_count` are the same bytes every time.
#
# So the device sends ONE ROW -- the last row of the call's own prefix -- and
# this module keeps the prefix. Three structural facts make that sound, and the
# guard below checks the first two instead of assuming them:
#
#   1. `step_count` rises by exactly one per rollout step, so a call at prefix
#      n either finds n rows already here (this step's siblings, and the row it
#      carries must MATCH the one stored) or n-1 (it is the first call of the
#      step, and its row is appended).
#   2. A REPEAT of a discarded episode restarts at `step_count == 0`, which
#      empties the prefix, and its first step then arrives at prefix 1, whose
#      whole history IS the row in hand. `face_prefix_reset` empties it
#      explicitly too, and ppo.py calls that at the top of every attempt.
#   3. Rows below `step_count` never move: `env.step` shift-and-inserts at
#      `idx = step_count` only. That is the same fact the pop-extend in
#      `live_faces._tokenizer_at` and the key chain in `live_faces.hist_key`
#      already rest on.
#
# ANYTHING ELSE RAISES. There is no truncation and no silent resynchronisation:
# without the history arrays there is no way to rebuild the prefix, so a
# disagreement between what this module holds and what the device says the step
# index is has to stop the run. That is the trade the ruling made, and this is
# where it is paid.
#
# ONE BATCHED BUFFER, NOT ONE PER ENVIRONMENT. Every callback here is dispatched
# with `vmap_method="broadcast_all"` and walks the batch in index order, so the
# loop index IS the environment identity (the same fact `EdgeSlotTable` and the
# per-env key chain already rest on). Holding the batch in one array means the
# arrays handed on are the buffers themselves -- the per-env reader takes a
# view, the batched reader takes the whole thing -- and nothing is stacked or
# copied per call.
#
# ALPHAGRAD_FACE_ROW_WIRE=0 restores the full-history operand, byte for byte,
# for an A/B of the change.
# ---------------------------------------------------------------------------
_FACE_PREFIX: dict = {}
_FACE_PREFIX_STATS = {"append": 0, "verify": 0, "reset": 0, "alloc": 0}


def face_row_wire() -> bool:
    """True when the callbacks ride with ONE row instead of the prefix."""
    return os.environ.get("ALPHAGRAD_FACE_ROW_WIRE", "1") == "1"


def face_prefix_reset() -> None:
    """Empty every environment's prefix, keeping the buffers.

    Called at the top of EVERY episode attempt (ppo.py `_ep_begin_attempt`)
    and by :func:`episode_telemetry_reset`, which is this module's "a fresh
    episode starts here" hook. A discarded attempt's rows are therefore gone
    before the repeat writes its own, and the repeat's own first call at
    `step_count == 0` would empty it again anyway.
    """
    n = _FACE_PREFIX.get("n")
    if n is not None and any(n):
        _FACE_PREFIX_STATS["reset"] += 1
    if n is not None:
        for i in range(len(n)):
            n[i] = 0


def face_prefix_stats() -> dict:
    """The counters, and reset. `append` + `verify` is the number of calls
    served; a nonzero `reset` past one per attempt means somebody is asking
    for prefixes out of order."""
    out = dict(_FACE_PREFIX_STATS)
    for k in _FACE_PREFIX_STATS:
        _FACE_PREFIX_STATS[k] = 0
    return out


def face_prefix_step(step_counts, rows, skips, steps, joins=None):
    """Extend the host prefix by one row per environment; return the history.

    ``step_counts`` is the PREFIX LENGTH each environment's call is about --
    `state.step_count` for a face callback, `step_count + 1` for the env step
    callback, which has just committed its own row. ``rows[i]`` / ``skips[i]``
    (and ``joins[i]``) is the row at index ``step_counts[i] - 1``: the last row
    of that call's own prefix, and the only row the device has to send.

    Returns ``(rows_hist, skips_hist, joins_hist)`` shaped
    ``(B, steps) + row.shape[1:]`` -- the buffers themselves, exactly what the
    full-history operand used to be. ``joins_hist`` is None until some call
    passes ``joins``.

    RAISES on any step-index disagreement. See the block comment.
    """
    rows = np.asarray(rows, np.int32)
    skips = np.asarray(skips, np.int32)
    joins = None if joins is None else np.asarray(joins, np.int32)
    B = int(rows.shape[0])
    T = int(steps)
    sc = np.asarray(step_counts).reshape(-1)
    if sc.size == 1 and B > 1:
        sc = np.repeat(sc, B)
    if sc.size != B:
        raise RuntimeError(
            f"face_prefix_step: {sc.size} step counts for {B} environments")

    want = (B, T, tuple(rows.shape[1:]), tuple(skips.shape[1:]))
    if _FACE_PREFIX.get("shape") != want:
        # A DIFFERENT SHAPE IS A DIFFERENT GRAPH, not a lost prefix: B is the
        # environment count, T the episode length and the row shape the face
        # wire, and none of the three moves inside one episode. Reallocating is
        # therefore not a resynchronisation, and it is counted.
        _FACE_PREFIX.clear()
        _FACE_PREFIX.update(
            shape=want,
            n=[0] * B,
            rows=np.zeros((B, T) + tuple(rows.shape[1:]), np.int32),
            skips=np.zeros((B, T) + tuple(skips.shape[1:]), np.int32),
            joins=None)
        _FACE_PREFIX_STATS["alloc"] += 1
    if joins is not None and _FACE_PREFIX["joins"] is None:
        _FACE_PREFIX["joins"] = np.zeros(
            (B, T) + tuple(joins.shape[1:]), np.int32)

    have = _FACE_PREFIX["n"]
    R, K, J = (_FACE_PREFIX["rows"], _FACE_PREFIX["skips"],
               _FACE_PREFIX["joins"])
    for i in range(B):
        n = int(sc[i])
        if n <= 0:
            have[i] = 0
            continue
        if n > T:
            raise RuntimeError(
                f"face_prefix_step: env {i} is at prefix {n} and the history "
                f"holds {T} steps")
        if n == 1 or have[i] == n - 1:
            # PREFIX 1 IS THE ROW IN HAND, whatever the store held before.
            # Its history is rows [0, 1), which IS the row the device just
            # sent, so nothing is inferred and nothing is resynchronised --
            # and this is the only thing that tells the store an episode
            # started. `env.reset` runs no step callback, so the env step
            # callback's first call of an episode is at prefix 1 and never at
            # 0; without this rule a second episode on the same env object
            # would look like a gap (measured: 20 test modules that run two
            # plans through one env).
            R[i, n - 1] = rows[i]
            K[i, n - 1] = skips[i]
            if joins is not None:
                J[i, n - 1] = joins[i]
            have[i] = n
            _FACE_PREFIX_STATS["append"] += 1
        elif have[i] == n:
            # A SIBLING CALL OF THE SAME STEP. Its row must be the row already
            # stored, or the device has decided something this prefix does not
            # know about -- never patched over, because a patched row would
            # make the tokenizer replay a graph the plan was not measured on.
            if not (np.array_equal(R[i, n - 1], rows[i])
                    and np.array_equal(K[i, n - 1], skips[i])):
                raise RuntimeError(
                    f"face_prefix_step: env {i} at prefix {n} carries a face "
                    f"row the host prefix does not hold. The host prefix is "
                    f"extended one row per step and rows below step_count "
                    f"never move (env.py, THE ELIMINATION-PREFIX FACE HISTORY); "
                    f"a row that changed under it means an attempt was "
                    f"repeated without face_prefix_reset.")
            if joins is not None:
                J[i, n - 1] = joins[i]
            _FACE_PREFIX_STATS["verify"] += 1
        else:
            raise RuntimeError(
                f"face_prefix_step: the host face prefix for env {i} holds "
                f"{have[i]} rows and the device is at step {n}. The device "
                f"sends one row per step and the host keeps the rest, so the "
                f"two indices cannot differ by more than one "
                f"(ALPHAGRAD_FACE_ROW_WIRE=0 restores the full-history "
                f"operand).")
    return R, K, J


def _face_prefix_host(fn, steps):
    """Wrap a host callback so its face operands are ROWS, not histories.

    ``fn`` is the env step callback's host function, which wants
    ``(args, consts, order, specs, face_specs, face_skips, face_joins, stop,
    *eval)`` with the two history arrays. The wrapper takes the same
    positions carrying ONE ROW each -- this step's, which ``env.step`` already
    has in hand -- extends the host prefix to ``stop`` rows and hands ``fn``
    the history buffers. ``fn`` sees exactly what it always saw.

    The batch is read the way :func:`_batched_host` reads it: ``order`` is
    ``(B, N)`` under ``vmap`` (``_env_callback`` dispatches with
    ``vmap_method="expand_dims"``) and ``(N,)`` outside one.
    """
    def _wrapped(args, consts, order, specs, face_row, skip_row, join_row,
                 stop, *eval_samples):
        _o = np.asarray(order)
        _b = _o.ndim >= 2
        _r = np.asarray(face_row)
        _k = np.asarray(skip_row)
        _j = None if join_row is None else np.asarray(join_row)
        if not _b:
            _r, _k = _r[None], _k[None]
            _j = None if _j is None else _j[None]
        _B = int(_o.shape[0]) if _b else 1
        # An operand `vmap` did not map arrives with a leading 1 (expand_dims);
        # the face row is mapped on every real rollout, so this only fires for
        # a caller that broadcasts one row across the batch.
        if _r.shape[0] == 1 and _B > 1:
            _r = np.repeat(_r, _B, axis=0)
            _k = np.repeat(_k, _B, axis=0)
            _j = None if _j is None else np.repeat(_j, _B, axis=0)
        R, K, J = face_prefix_step(
            np.asarray(stop).reshape(-1), _r, _k, steps, _j)
        if not _b:
            R, K = R[0], K[0]
            J = None if J is None else J[0]
        return fn(args, consts, order, specs, R, K,
                  (None if join_row is None else J), stop, *eval_samples)
    return _wrapped
FACE_SLOTS = 3  # pre (lhs), post (rhs), new (res)
NUM_AXIS_PAIRS = 4

# Per-vertex axis-state observation surface. The policy's dynamic action
# space (DIAG / COMPRESS / END) emits indices into a per-vertex axis set;
# this is the static observation that feeds heads.py's `AxisSetEncoder`.
# `MAX_AXES_PER_VERTEX` is the JAX-static upper bound on axes any vertex
# can have — most graphax ops have ≤ 4-6 axes (out_ndim + min_in_ndim);
# 8 leaves headroom without bloating state. `AXIS_FEATURE_DIM`'s four
# fields are [size, is_output, is_compressed, group_id]: only the first
# two are populated today (the others are placeholders for the future
# DIAG/COMPRESS state updates that micro_actions wiring will fill in).
MAX_AXES_PER_VERTEX = 8
AXIS_FEATURE_DIM = 4
_AXIS_FEAT_SIZE = 0
_AXIS_FEAT_IS_OUTPUT = 1
_AXIS_FEAT_IS_COMPRESSED = 2
_AXIS_FEAT_GROUP_ID = 3

# Canonical 8-component reward vector layout. The env reports raw reward values
# in the convention "higher is better": every cost component is stored *negated*
# (so r = -cost), `cosine_sim` is in [0, 1] (1 = identical Jacobian), and
# `frob_residual` is stored as `-||J_e - J_a||_F / ||J_e||_F` so larger residuals
# correspond to lower reward. Downstream code can therefore treat all 8 entries
# uniformly as "reward to maximize".
#
# Compute family (indices 0..5):
#   0 muls_adds_fmas   — graphax `adds + muls + fmas` op count from VE.
#   1 flops            — XLA cost-analysis FLOPs of the compiled approx fn.
#   2 latency_ns       — wall-clock latency in ns (only populated when
#                        `EnvConfig.measure_latency` is True; else 0).
#   3 max_io_sum       — graphax `mem` accumulator = sum over Jacobian
#                        accumulations of `max(in_size, out_size, edge_out_size)
#                        * itemsize`. (This is the "sum of max-input/max-output
#                        sizes per Jacobian accumulation" metric in the spec.)
#   4 bytes_accessed   — XLA cost-analysis bytes-accessed of the approx fn.
#   5 peak_memory      — peak HBM bytes during a single execution of the approx
#                        fn, captured via `ResourceMonitor`.
# Quality family (indices 6..7):
#   6 quality          — THE quality channel. Which QUANTITY sits in it is
#                        selected by ``ALPHAGRAD_QUALITY_METRIC`` (see
#                        ``quality_metric()`` below):
#                          "grad_cosine" (the default for any scalar-loss
#                            target, i.e. every trainable example; owner
#                            ruling 2026-09-02) -- the cosine between THIS
#                            PLAN's gradient and the rev-exact gradient on the
#                            same probe batch of real data, at init.
#                          "loss_drop" (by name only) -- the relative loss
#                            drop of a 200-step Adam walk driven by THIS
#                            PLAN's gradient, probed on a fixed batch of 512
#                            real MNIST images. Pearson 0.922 against final
#                            downstream test accuracy vs 0.610 for the
#                            Jacobian cosine, at 0.22 s / 40 MB per plan vs
#                            9.70 s / 4.24 GB; but finding 51 shows it can
#                            score 0.885 while the gradient points elsewhere.
#                          "cosine" (legacy) — cosine similarity between the
#                            flattened approximated and exact Jacobians,
#                            aggregated over the calibration samples.
#                        The slot was called ``cosine_sim`` until 2026-08-07;
#                        ``REWARD_INDEX["cosine_sim"]`` still resolves to 6 so
#                        every historical call site keeps working, but the
#                        NAME is now metric-agnostic because the wandb keys
#                        (mean_/measure_/popart_ are all built from
#                        REWARD_NAMES) must not claim "cosine" for a number
#                        that is not one. The concrete metric is published as
#                        ``quality/metric``.
#   7 grad_coverage    — RESERVED, AND THIS ENV NEVER POPULATES IT. It held
#                        per-leaf gradient coverage from 2026-08-27 until the
#                        owner ruling of 2026-09-03 (ticket dsnn-3qm.15) removed
#                        the guard, the channel and the value head; before that
#                        it was the dead ``frob_residual`` slot. The NAME and
#                        the INDEX stay so archived plan logs and the
#                        index-keyed reward_scaling mirror keep their meaning
#                        (the same convention slot 9 uses). The alias below
#                        keeps index 7 addressable as ``frob_residual`` for the
#                        pool's sentinel wire. This env emits 0.0 here, always.
#   8 fidelity         — CLIPPED RELATIVE FROBENIUS (workstream A2, owner's
#                        choice 2026-08-28):
#
#                            fidelity = clip(1 - ||J_e - J_a||_F / ||J_e||_F,
#                                            -1, +1)
#
#                        SIGN CONVENTION, deliberately "higher is better" like
#                        every other slot:
#                          +1  <=>  rel_frob == 0   <=>  J_a == J_e exactly.
#                           0  <=>  rel_frob == 1   <=>  the error is exactly
#                                   the size of the gradient. An all-zero
#                                   Jacobian lands here EXACTLY (||J_e-0||=||J_e||),
#                                   which is the whole reason this is preferable
#                                   to the cosine: cos(0, J_e) is 0/0, undefined,
#                                   and the eps-clamped formula reports it as 0
#                                   only by accident of the epsilon.
#                          -1  <=>  rel_frob >= 2 (clipped). rel_frob is
#                                   unbounded above -- a wrong-signed Jacobian of
#                                   the same magnitude gives 2, a blown-up one
#                                   gives arbitrarily more -- so the FLOOR is
#                                   what makes the channel bounded and therefore
#                                   symlog-free and PopArt-friendly.
#                        The slot is 0.0 whenever fidelity was not measured
#                        (non-terminal steps, and every step when the channel is
#                        off), matching the sparse-terminal convention slot 6
#                        already uses.
#                        MEASURED ONLY WHEN ASKED FOR: `fidelity_enabled()`
#                        (ppo's --fidelity-weight / ALPHAGRAD_FIDELITY). It needs
#                        the EXACT Jacobian, which under the default loss_drop
#                        quality metric is otherwise never built -- see
#                        `_exact_ref_scores`.
#   9 bkstep_acc     — RESERVED, AND THIS ENV NEVER POPULATES IT. The
#                        slot belongs to the DEPRECATED Ray line, whose
#                        table (`common.reward_scaling.REWARD_NAMES`)
#                        has held `bkstep_acc` at index 9 since long
#                        before this one reached nine channels, and
#                        `BKSTEP_ACC_IDX` / `NO_SYMLOG_REWARD_INDICES` /
#                        mu0's documented "10-channel layout" bridge all
#                        pin it there. Persisted PopArt / calibration
#                        state is keyed by INDEX, so index 9 cannot be
#                        reused and the sparsity channel is APPENDED at
#                        10 in BOTH tables instead. The reservation is
#                        what keeps the two tables IDENTICAL rather than
#                        merely one being a prefix of the other -- the
#                        property `tests/fidelity_channel_test.py` pins,
#                        and the property that stops a weight built from
#                        one table landing on a different channel of the
#                        other. This env emits 0.0 here, always.
#  10 sparsity       — STORED-BYTE SPARSITY RATIO, expressed against the
#                        exact plan on the SAME order:
#
#                            sparsity = clip(1 - stored_bytes(approx)
#                                              / stored_bytes(exact),
#                                            -1, +1)
#
#                        +1 = stores nothing (all-SKIP), 0 = stores what
#                        exact stores (the identity plan, EXACTLY),
#                        -1 = stores twice as much or worse (densified).
#                        0.0 also means "not measured" -- non-terminal
#                        steps, the channel off, an undefined denominator
#                        -- matching the sparse-terminal convention slots
#                        6/7/8 already use.
#                        MEASURED ONLY WHEN ASKED FOR: `sparsity_enabled()`
#                        (ppo's --sparsity-weight / --sparsity-log /
#                        ALPHAGRAD_SPARSITY). READ THE HACKABILITY WARNING
#                        on `_SPARSITY_STATS` before weighting it.
NUM_REWARDS = 11
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "quality",
    # RESERVED (held gradient coverage until 2026-09-03); never populated here.
    "grad_coverage",
    "fidelity",
    # RESERVED for the deprecated Ray line; never populated here.
    "bkstep_acc",
    "sparsity",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
# BACK-COMPAT ALIAS for slot 7. The slot was ``frob_residual`` until
# 2026-08-27 and had been DEAD since the quality channel absorbed it (see the
# REWARD_NAMES comment above: "the env still emits the frob_residual slot ...
# but nothing reads it"), carried GRADIENT COVERAGE from 2026-08-27 and is
# RESERVED since 2026-09-03. The alias keeps every historical call site
# (cpu_approx_pool's sentinel writer, alpha0's --lambda-frob, az_gumbel) on
# index 7. Exactly the precedent slot 6 set when cosine_sim became quality.
REWARD_INDEX["frob_residual"] = REWARD_INDEX["grad_coverage"]
# BACK-COMPAT ALIAS. 269 call sites (and the persisted PopArt/calibration
# state, which is keyed by INDEX) address slot 6 as "cosine_sim". The slot did
# not move; only its display name changed, so the alias is exact.
REWARD_INDEX["cosine_sim"] = REWARD_INDEX["quality"]
COMPUTE_REWARD_INDICES = tuple(range(0, 6))  # cost components
# quality (loss_drop / cosine), the reserved slot 7, clipped relative
# Frobenius, the reserved bkstep slot and stored-byte sparsity. These are
# the "higher is better" channels: they are NOT stored negated and must
# not be treated as costs. Mirrors reward_scaling._QUALITY_REWARD_INDICES.
QUALITY_REWARD_INDICES = (6, 7, 8, 9, 10)

# Sentinel reward returned when a per-vertex transform sequence matches an
# entry in the in-file blacklist (used during exploration to penalise
# pathological configurations). The blacklist is no longer wired up after
# the typed-transform migration; the array is kept for potential reuse.
# Worst-possible reward: every cost channel at the sentinel and the quality
# channels at their floor. Derived from REWARD_INDEX so adding a channel can't
# leave a stale hand-written row behind.
SENTINEL_COST = -1e10
# The BOUNDED channels take their own floor, not SENTINEL_COST: slot 7 and
# slot 8 both live in [-1, 1], and stamping -1e10 into a bounded channel would
# make its PopArt sigma meaningless for every real plan measured afterwards.
_SENTINEL_BOUNDED_FLOOR = {
    # RESERVED slot 7: was the coverage floor ("all leaves frozen"). Kept
    # at -1.0 so the sentinel vector is byte-identical to every archived
    # one; nothing reads the slot.
    REWARD_INDEX["frob_residual"]: -1.0,
    REWARD_INDEX["fidelity"]: -1.0,        # rel_frob >= 2, the clip floor
    # SPARSITY TAKES ITS FLOOR ON A SENTINELLED PLAN, NOT ITS CEILING.
    # This is the single most important line in the channel. A sentinelled
    # plan is, overwhelmingly, a plan that DELETED
    # computation -- which is the sparsity ceiling. Leaving the slot at
    # 0.0 (or letting it inherit SENTINEL_COST) would let the destroyer
    # top the sparsity ranking it was sentinelled for.
    REWARD_INDEX["sparsity"]: -1.0,
    # RESERVED slot 9: this env never populates bkstep_acc, so its
    # sentinel value must be the same 0.0 it carries on every real plan.
    # Stamping -1e10 into a bounded [0,1] channel would make its PopArt
    # sigma meaningless for anything that ever does populate it.
    REWARD_INDEX["bkstep_acc"]: 0.0,
}
_SENTINEL_BAD_REWARD = jnp.array(
    [
        0.0 if i == REWARD_INDEX["cosine_sim"]
        else _SENTINEL_BOUNDED_FLOOR.get(i, SENTINEL_COST)
        for i in range(NUM_REWARDS)
    ],
    dtype=jnp.float32,
)

# Axis pair index -> (base_idx1, base_idx2). base_idx1 picks output axis 0/1; base_idx2 picks input axis 0/1.
axis_pair_idx_to_base = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

# Sentinel used in `sparsity_specs[v, slot, 0]` to flag a COMPRESS sub-step
# (vs the regular `bi1 >= 0` DIAG payload or `bi1 == -1` end-of-sequence
# marker). The value is encoded as ``-2`` so existing `bi1 < 0` guards still
# recognise the slot as "not a DIAG", and `_callback` dispatches on the
# specific sentinel value.
#
# Caveat — COMPRESS through the vertex elimination DAG is only partial:
# `_callback` emits the correct `graphax.sparse.micro_actions.Compress`,
# but graphax's `_eliminate_vertex` assumes every edge keeps its nominal
# `(out_dims, primal_dims)` shape. Compress is lossy and reduces
# `val.ndim`, so a Compress applied to a vertex whose edge feeds into a
# subsequent elimination step trips the shape-preservation assertion at
# `core.py:417`. Practical implications:
#   * COMPRESS works end-to-end when it lands on the LAST vertex of the
#     elimination order (no downstream edge to matmul against).
#   * Earlier vertices in the order will assert. Hold off on
#     ``--allow-compress`` unless you've ordered the agent to only emit
#     COMPRESS on the final vertex, or are prepared to do the graphax
#     pre_transforms / shape-bookkeeping work.
# The reverse direction — graphax silently dropping a transform whose
# axes don't fit `val.ndim` at all (e.g. axis 1 on a 1-D val) — has been
# fixed (graphax commit `fa0a088`).
COMPRESS_SENTINEL = -2

# Sentinel used in `sparsity_specs[v, slot, 0]` to flag a QUANT sub-step
# (val.astype to a chosen JAX dtype). Row layout for a QUANT slot is
# ``[QUANT_SENTINEL, dtype_idx, 0]`` where ``dtype_idx`` indexes
# :data:`graphax.sparse.micro_actions.QUANT_DTYPES`. Sequential semantics:
# multiple QUANT slots on the same vertex chain through ``val.astype(...)``
# in order, last one wins (the SparseTensor's val dtype reflects the final
# Quant in the slot sequence). QUANT only mutates ``val.dtype`` — axis
# state stays untouched, so per-vertex `axis_state` updates ignore the
# sentinel.
QUANT_SENTINEL = -3


class EnvState(NamedTuple):
    order: Array
    # (N, MAX_RULES_PER_VERTEX, 3) int32. Per-slot row layout depends on
    # the leading column:
    #   row[0] >= 0:                 DIAG with `(bi1=row[0], bi2=row[1],
    #                                factor=row[2])`. bi1/bi2 are
    #                                base-axis positions (output-side /
    #                                primal-side, respectively).
    #   row[0] == COMPRESS_SENTINEL: COMPRESS with axis `row[1]` (physical
    #                                index into the SparseTensor edge:
    #                                out axes 0..out_len-1, then primal
    #                                axes out_len.. ) and `row[2]` indexes
    #                                :data:`COMPRESS_KINDS`.
    #                                Ticket .20: under --face-slot-frames
    #                                slot and --reduce-axis-space physical
    #                                `row[1]` is the LOGICAL dim of the
    #                                slot's tensor; masks.reduce_axis_spaces
    #                                resolves it to the physical val axis
    #                                at decode.
    #   row[0] == QUANT_SENTINEL:    QUANT with `row[1]` indexing
    #                                :data:`QUANT_DTYPES`. `row[2]` is
    #                                unused (kept at 0).
    #   row[0] == -1:                end-of-sequence sentinel; every slot
    #                                past it is treated as unused.
    sparsity_specs: Array
    # P1 per-path action history, index-aligned with ``order`` like
    # ``sparsity_specs``. ``face_specs`` rows use the SAME wire format as
    # sparsity_specs rows; row[0] == -1 ⇒ no approximation for that slot.
    # ``face_skips[k, f] == 1`` ⇒ face f of the k-th eliminated vertex is
    # SKIPPED (graphax.SKIP_FACE — the path's contraction never happens).
    face_specs: Array  # (N, MAX_FACES, wire_slots(), 3) int32
    face_skips: Array  # (N, MAX_FACES) int32
    # LEGACY full-stream observation. Under ``EnvConfig.delta_obs`` these are
    # degenerate ``(1,)`` sentinels and ``delta_*`` below carries the
    # observation instead; every non-delta consumer (alpha0 / gdpo / gfn /
    # mu0 / the ray workers) keeps the ``(MAX_TOKENS,)`` growing buffer.
    tokens: Array
    eqn_ids: Array  # (MAX_TOKENS,) int32; per-token equation ID, -1 for non-eqn tokens
    # DELTA observation (``EnvConfig.delta_obs``): the tokens THIS step's
    # elimination emitted, as a standalone buffer read from 0, plus their
    # exact count. Degenerate ``(1,)`` / 0 when delta_obs is off.
    delta_tokens: Array   # (env.delta_window,) uint8  (DELTA_TOKEN_DTYPE)
    delta_count: Array    # () int32 -- the header, decoded off the wire
    # Per-vertex axis state — observation surface for the dynamic action
    # space. `axis_state` is a packed int32 array of (size, is_output,
    # is_compressed, group_id) per axis slot; `axis_valid_mask` flags
    # which slots carry a real axis (vs. padding up to MAX_AXES_PER_VERTEX).
    # After `step()` the row for the just-eliminated vertex is updated to
    # reflect DIAG group_ids / shrunk sizes and COMPRESS marks
    # (see `_apply_rules_to_axis_state`). Downstream propagation across
    # the jaxpr DAG (where vertex `v`'s output axes feed into vertex
    # `v'`'s input axes later in the order) is intentionally not
    # implemented — the agent never revisits an eliminated vertex, and
    # the policy carries its own per-substep axis state through the
    # heads.py scan, so the missing signal is "useful debug metadata"
    # not "training signal".
    axis_state: Array          # (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM) int32
    axis_valid_mask: Array     # (total_v, MAX_AXES_PER_VERTEX) float32
    step_count: Array
    max_steps: int
    reward: Array  # (NUM_REWARDS,) float32; see REWARD_NAMES for layout
    terminated: bool
    # --approx-add choose: the per-face JOIN bit history, index-aligned with
    # ``order`` exactly as ``face_skips`` is. ``None`` under every value that
    # FIXES the join semantics -- not zeros, because 0 means ``lossy`` and a
    # zero-filled channel would silently measure every merge under a container
    # the policy never picked. A ``None`` leaf is an empty pytree node, so the
    # jit carry is shape-stable per configuration and the flag-off state is
    # byte-identical to the pre-flag one.
    face_joins: Array | None = None  # (N, MAX_FACES) int32


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class StepAction(NamedTuple):
    target_vertex: Array  # scalar int32
    rule_specs: Array  # (MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 marks unused
    # P1 per-path actions. None ⇒ per-vertex mode, byte-identical to before.
    # The SLOT axis is :func:`wire_slots` wide -- the head's --approx-add width
    # (3 / 3 / 3 / 4 / 5), NOT ``FACE_SLOTS`` (always the 3 contraction slots).
    face_rows: Array | None = None  # (MAX_FACES, wire_slots(), 3) int32 rows
    face_skip: Array | None = None  # (MAX_FACES,) int32; 1 ⇒ SKIP_FACE
    # --approx-add choose: ONE bit per face, 0 = lossy, 1 = lossless
    # (:func:`join_mode_of_bit`). ``None`` at every width that FIXES the join
    # semantics, which is what makes :func:`resolve_join_mode` answer from the
    # configuration instead. A SEPARATE channel from ``face_skip`` on purpose:
    # "drop this face's contraction" and "which container the ADD uses" are
    # unrelated decisions, and packing them into one field would make every
    # reader of a field called "skip" wrong.
    face_join: Array | None = None  # (MAX_FACES,) int32


class EnvConfig(NamedTuple):
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool
    # cmp_type / mem_type used to gate which compute/memory metric was measured.
    # The env now reports the full 8-component reward vector every step, so they
    # are kept only as *primary-metric hints* for legacy CLI/host-side reporting.
    # New callers should ignore them and pick the desired component from the
    # reward vector explicitly via REWARD_INDEX.
    cmp_type: Literal["graphax", "flops", "latency"]
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False
    # Latency requires running the compiled fn 10x per step, which roughly 10xs
    # rollout-to-reward time. Off by default; flip on when the latency component
    # of the reward is actually being weighted.
    measure_latency: bool = False
    # Skip the expensive jacve-compile/exec branch on every step EXCEPT the
    # terminal one. Tokens/eqn_ids are still produced (the agent needs them as
    # the next observation), but the reward vector is zero on intermediate
    # steps and fully populated only when the order is complete. This is the
    # paper-native form for AlphaZero / GDPO / GFlowNet and works fine for PPO
    # / MuZero (just yields a sparse reward signal).
    terminal_rewards_only: bool = False
    # Measurement protocol (spec): `num_data_points` distinct eval samples x
    # `reps_per_point` repetitions each = the sample budget per measurement,
    # reduced by a median. Defaults 5 x 4 = 20. Reps only matter for
    # timing noise, so when latency isn't measured we collapse to 1 rep.
    num_data_points: int = 5
    reps_per_point: int = 4
    # THE PAIRED REFERENCE'S OWN BUDGET (owner ruling 2026-09-14).
    #
    # The rev-exact reference used to be measured with the CANDIDATE's
    # `num_data_points` x `reps_per_point`. That is the wrong budget for it,
    # because the two halves of the pair are not the same size. Measured on
    # the transformer arm (job 65468, 128 repeats of one plan): the candidate
    # takes 18.2 ms per execution and the reference 0.121 ms, so the shared
    # 5 x 4 budget buys the candidate 18.3 s of integration and the reference
    # 0.12 s. The candidate reading then has a coefficient of variation of
    # 0.56 percent and the reference 4.53 percent, the two are independent
    # (the reference's lag-1 autocorrelation is 0.03) and so they add in
    # quadrature: the paired log ratio scatters by 0.0446 nats, of which
    # essentially all is the reference.
    #
    # 5 x 32 gives the reference 8005 executions, about 0.97 s, which is five
    # percent of the plan's measurement time. It takes the reference's CV to
    # about 1.6 percent and the paired ratio to about 2.0 percent -- half of
    # today's noise. The INNER reps are NOT decoupled: `latency_inner_reps`
    # stays shared between the two halves (owner ruling), because it is the
    # one number of the protocol that was actually measured.
    ref_num_data_points: int = 5
    ref_reps_per_point: int = 32
    # Spec's accumulation loop: each timed rep executes the compiled fn this
    # many times inside ONE ResourceMonitor window and divides the elapsed
    # time, amortizing dispatch/timer overhead (spec default 50; kept at 1
    # here so existing campaigns measure identically until a launcher opts in
    # via --latency-inner-reps).
    latency_inner_reps: int = 1
    # THE CANDIDATE'S TIME BUDGET (owner ruling 2026-09-14).
    #
    # Until this ruling the candidate ran a FIXED 5 points x 4 reps x 50 inner
    # = 1005 executions of the plan, whatever the plan cost. On the
    # transformer arm one execution is 18.2 ms, so a plan cost 18.3 s to
    # measure and an episode of 16 plans cost 293 s -- 73 percent of the
    # episode -- for twenty samples of a reading whose coefficient of
    # variation is 0.56 percent. The fixed counts were a specification
    # (commit 61e7027e, "the spec's 20"), never a noise measurement.
    #
    # The counts are now derived from ONE warm-up execution's measured time
    # `t`:
    #     inner   = clamp(ceil(measure_window_secs / t), 5, 50)
    #     windows = clamp(round(measure_budget_secs / (inner * t)),
    #                     1, num_data_points * reps_per_point)
    # so `num_data_points` and `reps_per_point` are CAPS on the window count,
    # not the count itself, and the windows are spread ROUND-ROBIN over the
    # data points (a plan with few windows still sees several samples).
    #
    # `measure_window_secs` is what keeps a timed window off the dispatch
    # floor: at inner 5 the identity plan reads 20.7 percent high against
    # inner 50 (docs/UNBIASED_PARETO_AND_MEASUREMENT.md), while 20 executions
    # per window already read within 3 percent of 50. A 50 ms window buys a
    # 121 us program the full 50 and an 18 ms program the floor of 5, which
    # is the regime each of them needs.
    #
    # THE REFERENCE keeps its own WINDOW COUNT (`ref_num_data_points` x
    # `ref_reps_per_point`, unchanged by this ruling) and takes only its INNER
    # from the same window rule, applied to its own execution time.
    #
    # A SLOW PLAN GETS FEWER WINDOWS, by construction. The owner's ruling on
    # that is explicit: slow runs do not matter, they are too large anyway.
    measure_budget_secs: float = 1.0
    measure_window_secs: float = 0.05
    # PER-FACE application. Off: the vertex's rule list is handed to graphax
    # literally and applied uniformly to EVERY face (so a rule must fit all of
    # them or it raises / is masked away). On: the rules are wrapped in a
    # per-face callable that applies each one only where it is legal on THAT
    # face's live operand — the per-path granularity the spec asks for, and
    # the per-path SKIP falls out of it (a face where nothing is legal is
    # left exact).
    per_face: bool = False
    # DELTA OBSERVATION (ppo.py; every other trainer leaves this False).
    #
    # False -- LEGACY: the callback returns the WHOLE append-only stream in a
    # ``(MAX_TOKENS,)`` buffer and the policy re-reads it at an ABSOLUTE
    # cursor. That buffer is the observation for alpha0 / gdpo / gfn / mu0 and
    # the ray workers, which have no incremental encoder to hand a delta to.
    #
    # True -- the callback returns ONLY the tokens the LAST elimination
    # emitted, in a ``(MAX_DELTA_TOKENS,)`` buffer with its EXACT host-side
    # length, and the base stream is a host-side constant
    # (``base_observation()``). Nothing downstream ever re-reads an earlier
    # token, so the growing buffer -- and the id-0 length hazard that came
    # with counting non-zeros in it -- is simply gone.
    delta_obs: bool = False
    # THE TRACED TARGET IS A SCALAR LOSS, so the plan's output IS a gradient.
    # A FACT READ OFF THE JAXPR, not a flag: ``from_jaxpr`` sets this from the
    # target's output avals. Recorded on the CONFIG (not just validated in
    # from_jaxpr) because ``_callback`` sees only the config, and
    # ``quality_metric()``'s ``auto`` default needs it to decide between the
    # gradient cosine (defined only for a scalar loss) and the legacy Jacobian
    # cosine.
    scalar_target: bool = False
    # DEPRECATED AND UNREAD. ``--measure-grad`` used to decide what was traced.
    # The traced target is now unconditionally the registered target
    # (``common.examples.get_fn`` = model + loss), so this field selects
    # nothing; it is kept accepted because callers still forward it.
    measure_grad: bool = False
    # WARMUP executions before the timed window. Passed by every caller
    # (cpu_approx_worker, az_gumbel) since the actor-args dict was written, but
    # until now it had NO READER in this module -- it was swallowed by
    # ``from_jaxpr(**_compat)``. The elimrl/POMO worker always runs exactly one
    # untimed warmup call before its timing loop; this stack ran none, so its
    # FIRST timed rep paid first-touch/allocator cost that the reference
    # protocol does not. Default 0 keeps every existing campaign bit-identical;
    # set it to 1 to match the reference protocol.
    latency_warmup: int = 0
    # THE PER-STEP DELTA WINDOW BIN (owner ruling 2026-09-14). A power of
    # two at or below MAX_DELTA_TOKENS, or 0 meaning "the cap".
    #
    # IT IS A SHAPE, not a budget: it sizes ``EnvState.delta_tokens`` and it
    # is the slice ``step`` takes out of the wire. The wire itself stays at
    # the cap (see MAX_DELTA_TOKENS), so this field changes nothing the Ray
    # measurement pool or the actors preallocate.
    #
    # It rides on the CONFIG, which rides in the env's pytree AUX data, so
    # two envs that differ only here have different treedefs and every
    # ``eqx.filter_jit`` above them retraces by itself. That is the same
    # retrace mechanism the episode-stream bin gets from its row shape.
    #
    # 0 is the default and it means "unchanged": an untouched caller -- the
    # policy regression gate, every legacy trainer, every test that does not
    # ask for a bin -- keeps exactly today's shapes and today's numbers.
    delta_window: int = 0
    # Episode cadence for checking the exact gradient against jax.grad (oracle A).
    # Default 50: runs on episode 0 and every 50 episodes. Set 1 to check on every episode.
    grad_oracle_cadence: int = 50


def _get_partials(order, sparsity_specs, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order
    partial_specs = (
        sparsity_specs[:v_stop] if v_stop < len(sparsity_specs) else sparsity_specs
    )
    return partial_order, partial_specs


def _apply_rules_to_axis_state(axis_state_v: Array, rule_specs: Array) -> Array:
    """Mutate one vertex's axis_state to reflect the rules just applied.

    Single-vertex update — the downstream / symbolic-shape propagation
    across the jaxpr DAG (where compressed / paired axes of vertex `v`
    show up as input axes of vertex `v'` later in the order) is the
    "deeper" piece called out in the env's roadmap and is **not**
    implemented here. The reason is that nothing in this scope actually
    needs the downstream signal: once a vertex is eliminated the agent
    never revisits it, and the policy carries its own per-substep axis
    state through `_features_after_diag` / `_features_after_compress` in
    `heads.py`. What this function buys is:

    * Honest top-N / replay output — `EnvState.axis_state[v]` after
      `step()` shows what the rules did to vertex `v`, which is helpful
      for debugging.
    * A baseline for the future cross-vertex propagation: when that
      lands, it will read the per-vertex mutations from here and
      forward them along the data-flow edges.

    Per rule:

    * DIAG ``(bi1, bi2, factor)`` — the output axis at relative position
      ``bi1`` and the primal axis at relative position ``bi2`` are paired
      under a fresh `group_id`. Their sizes shrink by `factor` (so an
      original (4, 4) pair with factor=2 becomes (2, 2)).
    * COMPRESS ``(SENTINEL, axis, _)`` — the axis at the recorded token
      position is marked `is_compressed = 1` and its size collapses to 1.
    * Unused / END sentinel rows leave the state unchanged.

    Implementation is JAX-traceable so callers inside the jitted
    `step()` can use it. ``rule_specs`` is iterated with
    ``lax.fori_loop`` and per-slot updates are gated by ``jnp.where``.
    """
    is_output = axis_state_v[:, _AXIS_FEAT_IS_OUTPUT]
    n_out = jnp.sum(is_output).astype(jnp.int32)

    # The fresh group_id starts after the largest existing one — that
    # way we don't overwrite groups recorded by earlier steps on this
    # same vertex (if any) and the per-vertex group sequence stays
    # monotonic. `_AXIS_FEAT_GROUP_ID` defaults to -1 (ungrouped), so
    # max(-1, ...) + 1 = 0 on the first DIAG.
    init_gid = jnp.max(axis_state_v[:, _AXIS_FEAT_GROUP_ID]) + 1

    def _body(slot, carry):
        state, gid = carry
        row = rule_specs[slot]
        bi1 = row[0]
        bi2 = row[1]
        factor = row[2]

        is_diag = bi1 >= 0
        is_compress = bi1 == COMPRESS_SENTINEL

        # DIAG: pair the (bi1, n_out + bi2) axes under `gid` and shrink
        # both sizes by `factor`. Clamp factor to >= 1 so the dummy
        # path (`factor == 0` from the unused row) leaves sizes alone.
        diag_out_tok = jnp.clip(bi1, 0, axis_state_v.shape[0] - 1)
        diag_prim_tok = jnp.clip(n_out + bi2, 0, axis_state_v.shape[0] - 1)
        safe_factor = jnp.maximum(factor, 1)

        def _apply_diag(s):
            s = s.at[diag_out_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_prim_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_out_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_out_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            s = s.at[diag_prim_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_prim_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            return s

        # COMPRESS: mark axis is_compressed, collapse size to 1.
        comp_tok = jnp.clip(bi2, 0, axis_state_v.shape[0] - 1)

        def _apply_compress(s):
            s = s.at[comp_tok, _AXIS_FEAT_IS_COMPRESSED].set(1)
            s = s.at[comp_tok, _AXIS_FEAT_SIZE].set(1)
            return s

        state = jax.lax.cond(is_diag, _apply_diag, lambda s: s, state)
        state = jax.lax.cond(is_compress, _apply_compress, lambda s: s, state)
        new_gid = jnp.where(is_diag, gid + 1, gid)
        return state, new_gid

    final_state, _ = jax.lax.fori_loop(
        0, MAX_RULES_PER_VERTEX, _body, (axis_state_v, init_gid)
    )
    return final_state


def compute_static_axis_state(jaxpr, total_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex axis features extracted statically from the jaxpr.

    Each vertex's axes are concatenated as ``(out_dims..., primal_dims...)``
    into a fixed-size slot of ``MAX_AXES_PER_VERTEX``. The primal proxy
    is the first non-literal input variable (matching the convention used
    by ``vertex_axis_dims`` in common/masks.py). Returns:

    * ``axis_state`` — ``(total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
      int32. Field layout: ``[size, is_output, is_compressed, group_id]``.
      ``is_compressed`` and ``group_id`` are placeholders (0 / -1) until
      the heads.py wiring lands and the env starts mutating them per
      sub-step.
    * ``axis_valid_mask`` — ``(total_v, MAX_AXES_PER_VERTEX)`` float32.
      ``1.0`` for slots carrying a real axis.

    Vertices with no shape info (literal-only inputs, etc.) get an
    all-zero / all-invalid row — same convention as
    ``vertex_axis_dims``.
    """
    axis_state = np.zeros(
        (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), dtype=np.int32,
    )
    axis_state[..., _AXIS_FEAT_GROUP_ID] = -1  # ungrouped sentinel
    axis_valid = np.zeros((total_v, MAX_AXES_PER_VERTEX), dtype=np.float32)

    for v_idx, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_shape = eqn.outvars[0].aval.shape
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        primal_shape = invars[0].aval.shape if invars else ()

        slot = 0
        for size in out_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 1
            axis_valid[v_idx, slot] = 1.0
            slot += 1
        for size in primal_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 0
            axis_valid[v_idx, slot] = 1.0
            slot += 1

    return axis_state, axis_valid


# ---------------------------------------------------------------------------
# Typed MicroAction -> EnvState.sparsity_specs row translator
# ---------------------------------------------------------------------------
#
# The heads.py policy emits a typed `MicroAction(op_type, i, j, exponents,
# factor)` sequence per vertex. The env's _callback turns the stored
# specs (one (base_idx1, base_idx2, factor) row per slot) into typed
# graphax.sparse.micro_actions.Diag entries before handing them to
# graphax's `transforms` API. This translator converts the policy's
# typed action sequence into the 3-tuple row format the EnvState carries
# in `sparsity_specs` — the rest of the env then dispatches normally.
# COMPRESS micro-actions are silently dropped today since the graphax
# `transforms` API only handles Diag end-to-end through the env's reward
# path (a Compress callable would need to be threaded all the way to
# graphax's per-vertex transform list — straightforward but not yet wired).


def diag_row_to_pair(jaxpr, vertex: int, bi1: int, bi2: int) -> tuple[int, int]:
    """``(i, j)`` the ``[bi1, bi2, factor]`` DIAG row addresses on ``vertex``.

    The single place the wire format's out-relative / primal-relative split is
    resolved back to the CONCATENATED ``out_dims + primal_dims`` numbering
    :class:`graphax.sparse.micro_actions.Diag` uses. Shared by the rule-spec
    translation and :class:`~alphagrad.approx.common.masks.LiveVertexMaskOracle`
    (which decides whether that transform is legal), so the mask can never be
    computed for a different pair than the one the env goes on to apply.
    """
    eqn = jaxpr.eqns[int(vertex) - 1]
    out_len = len(eqn.outvars[0].aval.shape)
    return int(bi1), out_len + int(bi2)


def micro_actions_to_rule_specs(
    op_types,
    i_indices,
    j_indices,
    factors,
    *,
    axis_state_for_vertex,
    compress_kinds=None,
    quant_dtypes=None,
):
    """Translate a sub-episode's typed micro-actions into legacy rule_specs.

    Args:
        op_types: (S,) int32 — per-sub-step op type (heads.py OP_DIAG /
            OP_COMPRESS / OP_QUANT / OP_END).
        i_indices: (S,) int32 — axis-token index for `i` (DIAG and COMPRESS).
        j_indices: (S,) int32 — axis-token index for `j` (DIAG only).
        factors: (S,) int32 — explicit positive factor (DIAG only),
            already collapsed from the prime-exponent head.
        axis_state_for_vertex: ``(MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
            int32 — used to map axis-token indices to the legacy
            ``base_idx1 / base_idx2`` (out_axis_position /
            primal_axis_position) representation. ``is_output`` of each
            axis token (column ``_AXIS_FEAT_IS_OUTPUT``) determines which
            side of the pair it lands on; the relative position is the
            running count of output-or-primal axes encountered before it.
        compress_kinds: optional (S,) int32 — index into
            :data:`COMPRESS_KINDS` per sub-step (only meaningful for
            COMPRESS rows; zeros default to ``"mean"``).
        quant_dtypes: optional (S,) int32 — index into
            :data:`QUANT_DTYPES` per sub-step (only meaningful for QUANT
            rows; zeros default to the first catalog entry).

    Returns:
        rule_specs: ``(MAX_RULES_PER_VERTEX, 3)`` int32 — same layout
        the env consumes. DIAG rows are ``[bi1, bi2, factor]``;
        COMPRESS rows are ``[COMPRESS_SENTINEL, physical_axis, kind_idx]``;
        QUANT rows are ``[QUANT_SENTINEL, dtype_idx, 0]`` where
        ``dtype_idx`` indexes :data:`QUANT_DTYPES`.
        Slots past the first ``OP_END`` (or past ``MAX_RULES_PER_VERTEX``,
        whichever comes first) are filled with the unused sentinel
        ``[-1, -1, 0]``.
    """
    # Lazy import to avoid circular dependency at module import time —
    # heads.py imports nothing from env.py but env.py only needs the
    # heads.py constants when this translator is actually invoked.
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT

    op_types_arr = np.asarray(op_types)
    i_arr = np.asarray(i_indices)
    j_arr = np.asarray(j_indices)
    f_arr = np.asarray(factors)
    if compress_kinds is None:
        k_arr = np.zeros_like(op_types_arr)
    else:
        k_arr = np.asarray(compress_kinds)
    if quant_dtypes is None:
        q_arr = np.zeros_like(op_types_arr)
    else:
        q_arr = np.asarray(quant_dtypes)
    axis_state_np = np.asarray(axis_state_for_vertex)

    n_out = int(np.sum(axis_state_np[:, _AXIS_FEAT_IS_OUTPUT]))
    # Build a per-token-index → (is_output, relative_position) map matching
    # `compute_static_axis_state`'s layout: out axes come first in slots
    # 0..n_out-1, then primal axes in slots n_out..n_out+n_primal-1.
    def _to_base(token_idx: int) -> tuple[int, int]:
        token_idx = int(token_idx)
        is_out = int(axis_state_np[token_idx, _AXIS_FEAT_IS_OUTPUT])
        rel = token_idx if is_out else token_idx - n_out
        return (rel, is_out)

    specs = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, 2] = 0  # factor=0 default for unused slots (matches reset path)

    slot = 0
    for s_idx, op in enumerate(op_types_arr.tolist()):
        if op == OP_END:
            break
        if op == OP_COMPRESS:
            # COMPRESS encodes a single axis to reduce. The env stores it
            # as `(COMPRESS_SENTINEL, physical_axis, kind_idx)` in the
            # sparsity specs row; `_callback` recognises the sentinel and
            # emits `Compress(axes=(physical_axis,), kind=COMPRESS_KINDS[kind_idx])`.
            # The axis-token index equals the physical position because
            # tokens are arranged as (outs..., primals...) matching the
            # SparseTensor edge layout.
            if slot >= MAX_RULES_PER_VERTEX:
                break
            physical_axis = int(i_arr[s_idx])
            specs[slot, 0] = COMPRESS_SENTINEL
            specs[slot, 1] = physical_axis
            specs[slot, 2] = int(k_arr[s_idx])
            slot += 1
            continue
        if op == OP_QUANT:
            # QUANT casts SparseTensor.val to QUANT_DTYPES[dtype_idx]. Stored
            # as `(QUANT_SENTINEL, dtype_idx, 0)`; `_callback` emits
            # `Quant(dtype=QUANT_DTYPES[dtype_idx])`. The third column is
            # reserved (kept at 0) — Quant carries no axis or factor state.
            if slot >= MAX_RULES_PER_VERTEX:
                break
            specs[slot, 0] = QUANT_SENTINEL
            specs[slot, 1] = int(q_arr[s_idx])
            specs[slot, 2] = 0
            slot += 1
            continue
        if op != OP_DIAG:
            raise ValueError(f"Unknown op_type {op!r} at sub-step {s_idx}.")
        if slot >= MAX_RULES_PER_VERTEX:
            break
        rel_i, is_out_i = _to_base(i_arr[s_idx])
        rel_j, is_out_j = _to_base(j_arr[s_idx])
        # Legacy rule_specs layout: row [base_idx1, base_idx2, factor]
        # where base_idx1 indexes the output axis and base_idx2 indexes
        # the primal axis. If both i and j are on the same side, the
        # mapping isn't lossless — log + fall through to OP_END so the
        # rest of the sub-episode doesn't poison the spec. This case
        # will go away when the typed action becomes the canonical form.
        if is_out_i == is_out_j:
            # Both axes on the same side (both output or both primal). The
            # env's sparsity_specs row format strictly pairs one output axis
            # with one primal axis; no representation for this. Terminate
            # the sub-episode here — the policy is expected to mask these
            # out before sampling, but a stray pair shouldn't crash the env.
            break
        # bi1 = output-side axis position, bi2 = primal-side axis position.
        if is_out_i:
            bi1, bi2 = rel_i, rel_j
        else:
            bi1, bi2 = rel_j, rel_i
        specs[slot, 0] = bi1
        specs[slot, 1] = bi2
        specs[slot, 2] = int(f_arr[s_idx])
        slot += 1

    return specs


def micro_actions_to_rule_specs_jax(
    op_types,
    i_indices,
    j_indices,
    factors,
    axis_state_for_vertex,
    compress_kinds=None,
    quant_dtypes=None,
    quant_scale_signs=None,   # (S,) int32 ±1 — negate head (arm select)
    quant_scale_fracs=None,   # (S,) float32 [0,1] — scale head; <0 ⇒ legacy
):
    """JAX-traceable MicroAction → rule_specs (DIAG, COMPRESS, and QUANT).

    Differs from :func:`micro_actions_to_rule_specs` in that the entire
    transform is JAX-tracer-friendly — no Python loops over sub-steps,
    no exceptions. It is *intended* for the rollout's JIT-compiled
    sample-then-step path; the Python translator stays for host-side
    code paths (e.g. tests, debugging, top-N replay).

    Semantics:

    * Each DIAG sub-step's `(i, j, factor)` becomes one rule_specs row
      `[bi1, bi2, factor]` where bi1/bi2 are out-side / primal-side
      relative positions.
    * Each COMPRESS sub-step writes a `[COMPRESS_SENTINEL, axis, kind_idx]`
      row where ``axis`` is the policy's i-index (which equals the
      physical axis position in the SparseTensor edge) and ``kind_idx``
      indexes :data:`graphax.sparse.micro_actions.COMPRESS_KINDS`. The
      env's `_callback` recognises the sentinel and emits a graphax
      `Compress(axes=(axis,), kind=COMPRESS_KINDS[kind_idx])`.
    * Each QUANT sub-step writes a `[QUANT_SENTINEL, dtype_idx, 0]` row
      where ``dtype_idx`` indexes
      :data:`graphax.sparse.micro_actions.QUANT_DTYPES`. The env's
      ``_callback`` emits a graphax
      ``Quant(dtype=QUANT_DTYPES[dtype_idx])``.
    * `op_type == OP_END` and every sub-step after the first END are
      marked unused.
    * The output is truncated to ``MAX_RULES_PER_VERTEX`` rows; trailing
      sub-steps beyond the legacy capacity are dropped. The policy's
      ``max_substeps`` should be ≤ ``MAX_RULES_PER_VERTEX`` to avoid
      silent truncation, or the trainer should accept the truncation
      (the dropped DIAGs / COMPRESSes / QUANTs become no-ops from the
      env's perspective).

    Args:
        op_types: (max_substeps,) int32 — heads.py OP_* values.
        i_indices, j_indices: (max_substeps,) int32 — axis-token indices
            into axis_state_for_vertex.
        factors: (max_substeps,) int32 — the integer factor produced
            by the prime-exponent head (already collapsed from exponents).
        axis_state_for_vertex: ``(MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
            int32 — per-vertex axis features from EnvState.
        compress_kinds: optional (max_substeps,) int32 — index into
            :data:`COMPRESS_KINDS` per sub-step (only used for COMPRESS).
        quant_dtypes: optional (max_substeps,) int32 — index into
            :data:`QUANT_DTYPES` per sub-step (only used for QUANT).

    Returns:
        rule_specs: ``(MAX_RULES_PER_VERTEX, 3)`` int32 in the legacy
        env layout ``[base_idx1, base_idx2, factor]``. Unused rows are
        ``[-1, -1, 0]``.
    """
    # Lazy import — heads.py imports nothing from env.py, but env.py
    # only needs the heads.py constants when this translator runs.
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT

    is_output = axis_state_for_vertex[:, _AXIS_FEAT_IS_OUTPUT].astype(jnp.int32)
    n_out = jnp.sum(is_output)

    if compress_kinds is None:
        compress_kinds = jnp.zeros_like(op_types)
    if quant_dtypes is None:
        quant_dtypes = jnp.zeros_like(op_types)
    if quant_scale_signs is None:
        quant_scale_signs = jnp.ones_like(op_types)
    if quant_scale_fracs is None:
        quant_scale_fracs = jnp.full(op_types.shape, -1.0, jnp.float32)

    is_end_per = (op_types == OP_END)
    prior_ends = (
        jnp.cumsum(is_end_per.astype(jnp.int32)) - is_end_per.astype(jnp.int32)
    )
    active = (prior_ends == 0)
    is_diag = (op_types == OP_DIAG)
    is_compress = (op_types == OP_COMPRESS)
    is_quant = (op_types == OP_QUANT)

    def _row(s_idx):
        i = i_indices[s_idx]
        j = j_indices[s_idx]
        is_out_i = is_output[i] > 0
        is_out_j = is_output[j] > 0
        # Relative position within out vs primal axes mirrors the
        # compute_static_axis_state layout: out axes come first.
        rel_i = jnp.where(is_out_i, i, i - n_out)
        rel_j = jnp.where(is_out_j, j, j - n_out)
        # DIAG spec: base_idx1 = output side, base_idx2 = primal side.
        # If both axes are on the same side, the row format can't
        # express the pair → mark unused.
        same_side = is_out_i == is_out_j
        diag_bi1 = jnp.where(is_out_i, rel_i, rel_j)
        diag_bi2 = jnp.where(is_out_i, rel_j, rel_i)
        diag_used = active[s_idx] & is_diag[s_idx] & (~same_side)

        # COMPRESS spec: bi1 = COMPRESS_SENTINEL (-2), bi2 = physical axis
        # position in the SparseTensor edge. Tokens are arranged as
        # (out axes..., primal axes...) which matches the edge's physical
        # layout, so token index `i` IS the physical axis index. bi2 is
        # plumbed through `_callback`, which double-checks the axis exists
        # in every invar's edge before emitting Compress.
        compress_used = active[s_idx] & is_compress[s_idx]
        compress_bi1 = jnp.asarray(COMPRESS_SENTINEL, dtype=jnp.int32)
        compress_bi2 = i.astype(jnp.int32)

        # QUANT spec: bi1 = QUANT_SENTINEL (-3), bi2 = dtype index into
        # :data:`QUANT_DTYPES`. The third column carries the negate + scale
        # heads: ``sign * (round(u*1e6) + 1)`` (0 = legacy auto-scale; see
        # decode_vertex_rule_specs). ``_callback`` emits
        # ``Quant(dtype, scale_sign, scale_frac)``.
        quant_used = active[s_idx] & is_quant[s_idx]
        quant_bi1 = jnp.asarray(QUANT_SENTINEL, dtype=jnp.int32)
        quant_bi2 = quant_dtypes[s_idx].astype(jnp.int32)
        quant_enc = (
            quant_scale_signs[s_idx].astype(jnp.int32)
            * (
                jnp.round(
                    jnp.clip(quant_scale_fracs[s_idx], 0.0, 1.0) * 1e6
                ).astype(jnp.int32)
                + 1
            )
        )
        # frac < 0 = "no scale head" sentinel → legacy row (enc 0).
        quant_enc = jnp.where(
            quant_scale_fracs[s_idx] >= 0.0, quant_enc, 0
        ).astype(jnp.int32)

        # Compose the row. Priority: QUANT > COMPRESS > DIAG > unused —
        # the *_used flags are mutually exclusive because they each gate
        # on the same op_type slot, so order is just for readability.
        # Third column: DIAG → factor; COMPRESS → kind index; QUANT → 0.
        bi1 = jnp.where(
            quant_used, quant_bi1,
            jnp.where(
                compress_used, compress_bi1,
                jnp.where(diag_used, diag_bi1, -1),
            ),
        ).astype(jnp.int32)
        bi2 = jnp.where(
            quant_used, quant_bi2,
            jnp.where(
                compress_used, compress_bi2,
                jnp.where(diag_used, diag_bi2, -1),
            ),
        ).astype(jnp.int32)
        f = jnp.where(
            quant_used, quant_enc,
            jnp.where(
                compress_used, compress_kinds[s_idx],
                jnp.where(diag_used, factors[s_idx], 0),
            ),
        ).astype(jnp.int32)
        return jnp.stack([bi1, bi2, f])

    rows = jax.vmap(_row)(jnp.arange(op_types.shape[0]))

    # Truncate to MAX_RULES_PER_VERTEX. If max_substeps < MAX_RULES we
    # pad the trailing rows with [-1, -1, 0].
    rows_truncated = rows[:MAX_RULES_PER_VERTEX]
    pad_needed = MAX_RULES_PER_VERTEX - rows_truncated.shape[0]
    if pad_needed > 0:
        pad = jnp.tile(
            jnp.array([-1, -1, 0], dtype=jnp.int32), (pad_needed, 1),
        )
        rows_truncated = jnp.concatenate([rows_truncated, pad], axis=0)
    return rows_truncated


# Lookup row used to convert a legacy scalar sp_type ∈ {0..4} into a single-rule (MAX_RULES, 3) spec.
_LEGACY_SP_TO_RULE_ROW = jnp.array(
    [
        [-1, -1, 0],   # sp 0: unused
        [0, 0, -1],    # sp 1 -> (0,0)
        [0, 1, -1],    # sp 2 -> (0,1)
        [1, 0, -1],    # sp 3 -> (1,0)
        [1, 1, -1],    # sp 4 -> (1,1)
    ],
    dtype=jnp.int32,
)


def _legacy_sp_to_specs(sp_type: Array) -> Array:
    """Convert a scalar legacy sp_type ∈ {0..4} into (MAX_RULES_PER_VERTEX, 3) rule specs."""
    first = _LEGACY_SP_TO_RULE_ROW[sp_type]  # (3,)
    pad = jnp.tile(jnp.array([-1, -1, 0], dtype=jnp.int32), (MAX_RULES_PER_VERTEX - 1, 1))
    return jnp.concatenate([first[None, :], pad], axis=0)


@jax.jit
def cossim(target, preds):
    target = target / jnp.maximum(
        jnp.linalg.norm(target, keepdims=True), jnp.sqrt(1e-7)
    )
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, keepdims=True), jnp.sqrt(1e-7))
    return jnp.sum(target * preds)


sp_type_to_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}

# things to try:
# error = MSE, cossim, Frobenius Norm
# other = {log, no log} x {div, no div}


def _flatten_jacobians(jac):
    """Concatenate all leaves of a (possibly nested) jacobian pytree to a flat 1-d array."""
    leaves = jax.tree_util.tree_leaves(jac)
    if not leaves:
        return None
    flats = [jnp.ravel(l) for l in leaves]
    return jnp.concatenate(flats)


_JAC_LAZY_NOTED: list = []


class GradientStructureMismatch(RuntimeError):
    """Two gradient pytrees that must share one structure do not (ticket
    dsnn-3qm.62). Raised by the grad-cosine and fidelity comparisons instead
    of the silent 0.0 clamp that hid finding 60; the measure actors re-raise
    it and the run stops. The message names the site and the leaf."""


def _align_jac(jac_approx, jac_exact, site: str = "jac_cosine"):
    """Align each approx-Jacobian leaf to its exact leaf's LAYOUT before the
    ANALYTIC JACOBIAN-COSINE comparison (the benchmarks whose target is a full
    Jacobian: Helmholtz, RoeFlux, ...). graphax ``jacve`` returns some
    Jacobian blocks in the transposed layout for certain elimination orders;
    flattening them as-is makes a (256,784) vs (784,256) ravel near-orthogonal.
    Transpose a 2-D leaf back when its shape is the exact leaf's reverse.

    Ticket .62: this is NOT on the grad-cosine or fidelity path any more (a
    parameter gradient has ONE layout, graphax's output contract), and a
    pytree that cannot be mapped against its reference RAISES instead of
    being returned unaligned (the blanket ``except`` that swallowed finding
    60's ``tree_map`` error is gone)."""
    def _al(a, e):
        # A dead path (None) and a SparseTensor pass through: the comparison
        # scores the first as zero and checks the second's layout itself.
        if a is None or e is None or _is_sparse_tensor(a) or _is_sparse_tensor(e):
            return a
        if getattr(a, "shape", None) == getattr(e, "shape", None):
            return a
        if getattr(a, "ndim", 0) == 2 and a.shape == e.shape[::-1]:
            print(f"[{site}] aligning a transposed Jacobian leaf "
                  f"{a.shape} -> {e.shape}", flush=True)
            return a.T
        return a
    try:
        return jax.tree_util.tree_map(
            _al, jac_approx, jac_exact,
            is_leaf=lambda x: x is None or _is_sparse_tensor(x))
    except Exception as exc:
        raise GradientStructureMismatch(
            f"[{site}] the approximated Jacobian pytree cannot be mapped "
            f"against the exact one: {type(exc).__name__}: "
            f"{' '.join(str(exc).split())[:600]}") from exc


def _gradient_leaves(tree):
    """The gradient leaves of a ``jacve`` output IN ORDER, with a dead path
    (``None``, a face SKIP deleted every path to that parameter) kept as a
    ``None`` leaf so the two sides pair up positionally."""
    return jax.tree_util.tree_leaves(
        tree, is_leaf=lambda x: x is None or _is_sparse_tensor(x))


def _is_sparse_tensor(x) -> bool:
    from graphax.sparse.tensor import SparseTensor
    return isinstance(x, SparseTensor)


def _leaf_shape(x):
    if x is None:
        return None
    if _is_sparse_tensor(x):
        return tuple(int(d.logical_size) for d in x.dims)
    return tuple(int(v) for v in getattr(x, "shape", ()))


def _gradient_similarity(jac_exact, jac_approx, site: str):
    """``(dot, ||e||^2, ||a||^2, ||e - a||^2, n_exact_elements)`` accumulated
    PER LEAF over the two gradient pytrees, with EXACT structure equality
    (ticket dsnn-3qm.62) and NEITHER SIDE MATERIALIZED (owner ruling (c),
    2026-09-12: "the comparison should remain lazy, should not need to
    materialize, and we definitely don't want to densify").

    All four accumulators are bilinear forms, so they contract from the two
    STORAGE forms; ``graphax.sparse.ops.bilinear`` does that arithmetic and
    that module's docstring carries the algebra and the measurements behind it.
    Nothing here calls ``SparseTensor.dense()`` any more -- on either side.

    Per pair of leaves (e, a), in ``jacve`` output order:

      * ``a is None`` (dead path after a SKIP): the plan's gradient for that
        parameter IS zero -- dot 0, ||a||^2 0, residual ||e||^2. Not a
        structure fault: graphax's dense path returns zeros there too.
      * ``a`` a plain array: its shape must equal ``e``'s.
      * ``a`` a SparseTensor: its LOGICAL shape must equal ``e``'s, and a
        fully materialized one must be in PARAMETER LAYOUT (graphax's output
        contract: the val axes in dim order). A compressed index in a returned
        gradient RAISES.
      * an IMPLICIT dim (``axis is None``, the storage of a Reduce'd gradient)
        is contracted ANALYTICALLY -- ``<e, bcast(a)> = <sum_implicit(e), a>``,
        ``||bcast(a)||^2 = N_implicit ||a||^2`` -- and a DIAGONAL PAIR is
        contracted at its structure positions only, by a gather the size of
        the stored ``val``. A uniform tensor (``val is None``) never builds a
        buffer at all, so the degenerate dead leaf (only implicit dims,
        ``scalar_mult == 0``) costs nothing.
      * a leaf count, logical shape or layout mismatch RAISES
        ``GradientStructureMismatch``.

    The EXACT side is no longer materialized either. A structured exact leaf
    (a full Jacobian target: the analytic AD benchmarks, or a Diag that
    survived to the boundary) used to be ``e.dense()``d purely so the sum could
    be taken; it is now contracted in place.

    When the two storage forms share no compact frame -- neither side fully
    materialized AND their structures differ -- this RAISES rather than
    densifying one of them. Densifying there would be the same defect with a
    longer stack trace, so the fix is to extend
    ``graphax.sparse.ops.bilinear``, and the message says so.
    """
    from graphax.sparse.ops.bilinear import (
        LazyContractionUnsupported, bilinear_accumulators, squared_norm)

    leaves_e = _gradient_leaves(jac_exact)
    leaves_a = _gradient_leaves(jac_approx)
    if len(leaves_e) != len(leaves_a):
        raise GradientStructureMismatch(
            f"[{site}] {len(leaves_a)} gradient leaves against "
            f"{len(leaves_e)} exact leaves: approx shapes "
            f"{[_leaf_shape(x) for x in leaves_a]} vs exact "
            f"{[_leaf_shape(x) for x in leaves_e]}")
    dot = ee = aa = rr = None
    total = 0
    for i, (e, a) in enumerate(zip(leaves_e, leaves_a)):
        if e is None:
            raise GradientStructureMismatch(
                f"[{site}] exact leaf {i} is a dead path (None); the exact "
                f"reference must carry every parameter gradient")
        e_shape = _leaf_shape(e)
        if _is_sparse_tensor(e):
            if e.val is not None and all(d.axis is not None and not d.is_sparse
                                         for d in e.dims):
                from graphax.sparse.ops.output_layout import is_parameter_layout
                if not is_parameter_layout(e):
                    raise GradientStructureMismatch(
                        f"[{site}] exact leaf {i} is not in parameter layout: "
                        f"dims {e.dims}, val {e.val.shape}")
            elif not _JAC_LAZY_NOTED:
                _JAC_LAZY_NOTED.append(1)
                print(f"[{site}] exact leaf {i} is a structured Jacobian "
                      f"(dims {e.dims}); it is contracted LAZILY -- neither "
                      "side is materialized (ticket dsnn-3qm.62 ruling (c))",
                      flush=True)
        total += int(np.prod(e_shape)) if e_shape else 1
        # graphax's compute-dtype rule, not ``jnp.promote_types``: a Quant'd
        # leaf can be float8 / int4, which JAX refuses to promote implicitly
        # (measured 2026-09-13: an all-slots float8 plan raised
        # TypePromotionError here, after the engine had contracted it fine).
        from graphax.sparse.dtype_compute import _compute_dtype
        _cdt = _compute_dtype(getattr(e, "dtype", jnp.float32), jnp.float32)
        if a is not None:
            _cdt = _compute_dtype(
                _cdt, getattr(a, "dtype", jnp.float32), jnp.float32)
        if a is None:
            _e2 = squared_norm(e, _cdt)
            _d = jnp.zeros((), _cdt)
            _a2 = jnp.zeros((), _cdt)
            _r2 = _e2
        else:
            a_shape = _leaf_shape(a)
            if a_shape != e_shape:
                raise GradientStructureMismatch(
                    f"[{site}] leaf {i}: logical shape {a_shape} vs exact "
                    f"{e_shape}"
                    + (f"; dims {a.dims}" if _is_sparse_tensor(a) else ""))
            if _is_sparse_tensor(a):
                if any(getattr(d, "is_compressed", False) for d in a.dims):
                    raise GradientStructureMismatch(
                        f"[{site}] leaf {i} carries a compressed index in a "
                        f"returned gradient: dims {a.dims}")
                if a.val is not None and not any(d.is_sparse for d in a.dims):
                    from graphax.sparse.ops.output_layout import (
                        is_parameter_layout)
                    if not is_parameter_layout(a):
                        raise GradientStructureMismatch(
                            f"[{site}] leaf {i} is not in parameter layout: "
                            f"dims {a.dims}, val {a.val.shape} (graphax "
                            f"output contract, ticket .62)")
            try:
                _d, _e2, _a2 = bilinear_accumulators(e, a, dtype=_cdt)
            except LazyContractionUnsupported as exc:
                raise GradientStructureMismatch(
                    f"[{site}] leaf {i}: the exact and approximated leaves "
                    f"cannot be contracted without materializing one of them, "
                    f"and materializing is what ticket dsnn-3qm.62 ruling (c) "
                    f"forbids. {exc}") from exc
            # The residual from the other three, which is what the analytic
            # branch has done since .62 landed: it agrees with the direct
            # sum of squares to float32 rounding (measured 2026-09-12, worst
            # relative disagreement 1.1e-7 over the pair fixtures).
            _r2 = _e2 - 2.0 * _d + _a2
        dot = _d if dot is None else dot + _d
        ee = _e2 if ee is None else ee + _e2
        aa = _a2 if aa is None else aa + _a2
        rr = _r2 if rr is None else rr + _r2
    return dot, ee, aa, rr, total


# ---------------------------------------------------------------------------
# THE PAIRED REFERENCE (ticket dsnn-3qm.9). Under ``--cost-form paired-log``
# every terminal measurement also measures REV-EXACT -- the reverse order,
# every face None, the jax.grad-equivalent -- in the same callback,
# INTERLEAVED with the candidate window by window since 2026-09-14 (it used
# to run as a second block right after it), through the same executable path
# and the same instrument (`_time_one_rep`, same eval args, same warmup, same
# median). Its POINTS x REPS are its own (`EnvConfig.ref_num_data_points`)
# and so is its INNER, which the window rule derives from its own execution
# time (`EnvConfig.measure_budget_secs`). The cost
# channels then carry the LOG-DIFFERENCE ``Delta_c = log cost_c(candidate)
# - log cost_c(rev-exact)`` (stored negated like every cost slot), so
# rev-exact scores 0 by construction and a GPU-state drift of 18-20 % between
# processes cancels instead of masquerading as a win. One record per
# reference measurement is kept here and drained with the plan records
# (`consume_plan_records` -> ``"paired_ref"``), so the trainer can log the
# reference in positive units (ref/latency_ns, ref/temp_bytes; ticket .45).
#
# Until 2026-09-04 this block held the additive quality gate
# (ALPHAGRAD_QUALITY_GATE_MIN, `_apply_quality_gate`, the per-order floor
# and the global rev reference). The gate clamped a destroyed plan's costs to
# a CACHED exact reference -- not paired, and the floor was measured with a
# different instrument for the whole v57-v66 campaign (see `_time_one_rep`).
# The quality floor that replaces it is a reward channel option in ppo.py
# (--quality-floor), not a clamp on the cost channels.
_PAIRED_REF: list = []
_PAIRED_REF_DROPPED = [0]
_PAIRED_REF_CAP = 65536

# THE FLOOR UNDER log(temp). ``memory_analysis().temp_size_in_bytes`` is an
# exact integer count of bytes and a plan whose gradient graph dead-code
# elimination removed entirely reports EXACTLY 0 (job 63632: every sampled
# NeuralNetwork plan with 5-11 skips), so log(temp) needs a floor. One byte
# is the smallest non-zero value the instrument can report, so the floor
# replaces only an exact 0 and never a measured value: it is not a cap
# (Delta_mem stays unbounded below in the size of the reference, which the
# owner ruled for -- no cap by default, ticket .9 Q37), and log(1 B) = 0
# makes a zero-temp plan's Delta_mem read directly as -log(temp_ref in
# bytes) -- "how many nats of temporaries the reference has". The
# alternative, flooring at the reference's own temp, would zero the memory
# channel for every zero-temp plan and HIDE the absorber's prize instead of
# leaving it to the quality floor to price (finding 53: that pricing is
# tau's and lambda's job, not the cost channel's). Latency needs no floor:
# a measured latency is already clamped up to `_LAT_FLOOR_NS`, and 0.0
# means "not measured" and stays 0.0 (see `paired_log_costs`). Every
# floored reading is counted in the reference record (``mem_floored``).
_MEM_LOG_FLOOR_BYTES = 1.0

#: The paired-cost floor POLICY (``--paired-cost-floor``, ticket .9,
#: owner decision 2026-09-13 on finding 63).
#:
#: ``"byte"`` is the behaviour above: memory floors at one byte, latency at
#: ``_LAT_FLOOR_NS``, and a plan that allocates nothing earns
#: ``log(ref_temp / 1 B)`` nats. Measured on TLM: 17.3 nats on the memory
#: channel alone, 29.7 nats over both channels on the Markowitz order
#: (finding 63, absorber job 65308). Pricing that with the quality floor
#: alone needs ``lambda_q > 32``.
#:
#: ``"reference"`` floors BOTH channels at the reference's own cost, so no
#: plan earns credit for being cheaper than the exact reference it is paired
#: against. The absorber's memory prize becomes exactly 0 instead of -17.3
#: nats, and ``lambda_q = 16`` then clears the contrast gate. Honest plans are
#: untouched wherever they cost MORE than the reference, which on the
#: Markowitz order is every one of them (their temp is 32x to 56x rev-exact).
#: A plan that really is cheaper than the reference loses that part of its
#: credit: on the reverse order the best float8 quant sits at 0.86x rev-exact,
#: so it forfeits 0.15 nats. Every floored reading is counted in
#: ``mem_floored`` on the reference record.
#:
#: This SUPERSEDES the ruling recorded in the comment above (ticket .9 Q37,
#: "no cap by default, leave the absorber's prize to tau and lambda"). That
#: ruling was taken before the absorber was measured. Set
#: ``--paired-cost-floor byte`` to reproduce a run made under it.
#: Transported by ENV VAR, like `cost_form` and `mem_channel` and for the
#: same reason: :func:`paired_log_costs` runs inside the MEASURE ACTOR, a
#: separate process, so a module-level setting in the trainer would not
#: reach it. ppo.py's ``--paired-cost-floor`` is the only writer.
_PAIRED_COST_FLOOR_ENV = "ALPHAGRAD_PAIRED_COST_FLOOR"
PAIRED_COST_FLOOR_CHOICES = ("byte", "reference")


def paired_cost_floor() -> str:
    """``"reference"`` (the default) or ``"byte"``.

    Anything else raises: a typo must not silently restore the unpriced
    absorber.
    """
    want = os.environ.get(_PAIRED_COST_FLOOR_ENV, "reference").strip().lower()
    if want not in PAIRED_COST_FLOOR_CHOICES:
        raise ValueError(
            f"{_PAIRED_COST_FLOOR_ENV} must be one of "
            f"{PAIRED_COST_FLOOR_CHOICES} (set by ppo.py from "
            f"--paired-cost-floor), got {want!r}")
    return want


def _time_one_rep(ex, eval_args, unique_devices, inner):
    """ONE timing repetition of `ex` under THE campaign protocol.

    Returns ``(latency_ns, peak_bytes, peak_src, last_output)``. The
    output is handed back rather than dropped so the campaign loop can
    keep scoring the very execution it timed (byte-identical to the
    inline loop this replaced).

    THIS IS THE SINGLE INSTRUMENT. Both the campaign measurement loop
    (the number a plan is scored on) and the paired rev-exact reference
    (the number it is log-differenced against, ticket dsnn-3qm.9; until
    2026-09-04 the quality gate's exact-cost floor) call it, with the
    same parameters, so the pair compares like with like.

    It did not use to. Until 2026-08-26 the floor was timed by
    ``_measure_exec_cost`` -- median of 3 laps of 20 back-to-back
    executions, after 4 untimed warmups -- while the plan was timed by
    this loop at ``--latency-inner-reps 5`` with no warmup. Twenty
    back-to-back executions amortise per-call dispatch far better than
    five, so the floor read 13-15% BELOW what an honest exact plan was
    charged in the same run (clamp prints: 133.6/137.5 us against
    155-160 us campaign means). The gate exists to make destruction
    cost-neutral; with two instruments it instead paid destruction a
    guaranteed -13% latency bonus, in every run of the v57-v66
    campaign. Do not re-fork this function.
    """
    if os.environ.get("ALPHAGRAD_BYPASS_RESOURCE_MONITOR", "0") == "1":
        return 0.0, 0.0, "bypassed", ex(*eval_args)
    if os.environ.get("ALPHAGRAD_DIRECT_MEASURE", "0") == "1":
        # Spec-native primitives, no jax_memory_monitor object at all
        # (its per-call C++ trackers are the leak that killed v10):
        # clear_memory_stats() resets the high-water mark, the inner
        # loop is timed with perf_counter around a drained queue, and
        # peak_bytes_in_use is read per device afterwards.
        #
        # ABOVE-BASELINE DELTA, not the absolute high-water mark.
        # peak_bytes_in_use is a DEVICE-WIDE ABSOLUTE counter, and
        # clear_memory_stats() resets the COUNTER but not the resident
        # baseline -- so an absolute reading is (resident baseline +
        # this call transient). CPU backends EXPOSE clear_memory_stats
        # but raise UNIMPLEMENTED when called, so the peak falls back
        # to the compiled executable's memory_analysis() static peak;
        # latency stays the real perf_counter timing either way.
        _have_stats = True
        # Drain FIRST: clear_memory_stats() resets the high-water
        # counter, but work still in flight from the previous rep lands
        # after the reset and is charged to THIS rep. Barrier -> clear
        # -> barrier makes the window tight.
        jax.effects_barrier()
        for _d in unique_devices:
            try:
                _d.clear_memory_stats()
            except Exception as _cexc:
                _have_stats = False
                if not _MEM_FALLBACK_WARNED:
                    _MEM_FALLBACK_WARNED.append(1)
                    print(
                        "[measure] WARNING peak_memory channel "
                        "switched to the STATIC memory_analysis() "
                        "estimate: clear_memory_stats() failed on "
                        f"{_d}: {type(_cexc).__name__}: {_cexc}. "
                        "This is a DIFFERENT quantity from the "
                        "measured peak_bytes_in_use delta.",
                        flush=True)
                break
        jax.effects_barrier()
        _base = 0.0
        if _have_stats:
            for _d in unique_devices:
                _bstats = _d.memory_stats() or {}
                _base += float(_bstats.get("bytes_in_use", 0.0))
        _t0 = time.perf_counter()
        for _k in range(inner):
            out = ex(*eval_args)
        jax.block_until_ready(out)
        _t1 = time.perf_counter()
        if _have_stats:
            _peak_abs = 0.0
            for _d in unique_devices:
                _stats = _d.memory_stats() or {}
                _peak_abs += float(_stats.get("peak_bytes_in_use", 0.0))
            _peak = max(0.0, _peak_abs - _base)
        else:
            # SUBSTITUTE IN PLACE -- one memory channel, so the static
            # estimate lands in peak_memory rather than travelling as a
            # second variable. Announced + counted.
            _peak = _memory_analysis_bytes(ex) or 0.0
            _note_static_peak_fallback(
                "this backend does not expose allocator statistics")
        return ((_t1 - _t0) / inner * 1e9, _peak,
                "runtime_delta" if _have_stats else "static_fallback", out)
    with _get_resource_monitor(unique_devices) as monitor:
        # Accumulation loop (spec, default 50 when opted in): the
        # executions queue back-to-back inside one monitor window and
        # the BLOCK BELOW drains them, so time/inner is a per-execution
        # latency with dispatch + timer overhead amortized. Peak memory
        # is unaffected (same executable, same buffers each pass).
        for _k in range(inner):
            out = ex(*eval_args)
        # DRAIN THE DEVICE QUEUE INSIDE THE TIMED WINDOW. JAX dispatch is
        # ASYNCHRONOUS: ``ex(*eval_args)`` returns as soon as the work is
        # ENQUEUED, so without this the timer stops when the host finishes
        # DISPATCHING, not when the device finishes EXECUTING.
        #
        # ``ResourceMonitor.__exit__`` is NOT a drain, whatever its name
        # suggests: it runs ``jax.effects_barrier()``, which waits only for
        # computations carrying ORDERED EFFECTS. A pure jitted Jacobian has
        # none, so the barrier returns immediately and the window closes
        # over an empty device queue.
        #
        # This read CORRECTLY for the whole fixed-count era by ACCIDENT.
        # At ``inner`` 50 the host cannot enqueue 50 executions of a big
        # program without blocking -- the PJRT queue saturates and each
        # execution's output buffer (780 MB on the transformer arm) makes
        # the allocator wait for an earlier one to be freed -- so the
        # enqueue loop itself became the barrier and the reading landed
        # near the truth. The budget rule of 2026-09-14 gives an 18 ms
        # program ``inner`` 5, five executions fit in the queue, the host
        # never blocks, and the SAME code reported DISPATCH TIME.
        #
        # MEASURED (canary job 65720, the campaign arm on the merged tip):
        # the 18.22 ms candidate read 119 us -- BELOW its own 135 us
        # rev-exact reference, which still saturated at inner 50 -- so
        # ``paired_log_costs`` floored the candidate at the reference and
        # the paired latency reward was EXACTLY 0 on every plan. The memory
        # channel was untouched (-3.1357 nats throughout) because
        # ``MemoryTracker`` polls the allocator from a native thread and
        # sees the real 781 MB watermark either way. Memory right, latency
        # wrong, is the signature of a missing drain.
        #
        # Blocking on the LAST output is sufficient and is what the
        # ``ALPHAGRAD_DIRECT_MEASURE`` branch above has always done:
        # executions of one executable on one device issue in order, so
        # waiting for the last one waits for all of them.
        jax.block_until_ready(out)
    # Key by name instead of unpacking ``.values()`` so this stays
    # robust to dict-order / API tweaks in jax_memory_monitor.
    _lat_s = float(monitor.stats.get("time", 0.0)) / inner
    _peak_bytes = float(monitor.stats.get("memory", 0.0))
    if _peak_bytes <= 0.0:
        # ResourceMonitor's DEVICE peak is structurally 0 on a CPU
        # backend, so without this the memory channel is a flat zero
        # and --mem-type peak_memory trains on nothing. Same in-place
        # substitution as the _direct branch above: one channel,
        # announced and counted.
        _peak_bytes = _memory_analysis_bytes(ex) or 0.0
        _note_static_peak_fallback(
            "ResourceMonitor reported a zero device peak "
            "(structural on CPU backends)")
        return _lat_s * 1e9, _peak_bytes, "static_fallback", out
    return _lat_s * 1e9, _peak_bytes, "runtime_delta", out


def _campaign_measure_cost(ex, eval_args_list, unique_devices,
                           inner, warmup, n_reps):
    """(latency_ns, peak_bytes) of `ex` under the FULL campaign
    instrument: ``len(eval_args_list)`` data points x `n_reps` timing
    repetitions of `inner` back-to-back executions each, `warmup`
    untimed executions per point, reduced by ``_aggregate_samples``
    (the same median the plan's own channels get).

    NOT ON THE MEASUREMENT PATH SINCE 2026-09-14, and say so plainly: the
    terminal callback measures the candidate and the paired rev-exact
    reference in ONE INTERLEAVED LOOP now, window by window, so that the
    two halves of the ratio occupy the same seconds (owner ruling; see
    `interleave_windows`). It carried the reference from ticket
    dsnn-3qm.9 until then, and the quality gate's exact floor until
    2026-09-04.

    It is kept because it is the FIXED-COUNT form of the instrument
    written down in one place: `len(eval_args_list)` points x `n_reps`
    windows, no budget, no window rule. The tests compare the callback's
    reading against it, which is the property that matters -- one
    instrument, whichever loop drives it.
    """
    _lat: list[float] = []
    _peak: list[float] = []
    for _eval_args in eval_args_list:
        for _w in range(max(0, int(warmup))):
            jax.block_until_ready(ex(*_eval_args))
        for _r in range(max(1, int(n_reps))):
            _l, _p, _s, _o = _time_one_rep(
                ex, _eval_args, unique_devices, inner)
            del _o
            _lat.append(_l)
            _peak.append(_p)
    return (
        float(_aggregate_samples(_lat, want_top_quartile=True)) if _lat else 0.0,
        float(_aggregate_samples(_peak, want_top_quartile=True)) if _peak else 0.0,
    )


def _resolve_warmup(config) -> int:
    """Untimed executions to run before the first TIMED one.

    ``config.latency_warmup`` when set; otherwise 1 unless
    ``ALPHAGRAD_MEASURE_WARMUP=0``. See the call site in the measurement
    loop for why the default is 1: without it the first timed sample of
    a plan is the first execution of a freshly compiled executable.
    """
    _w = max(0, int(getattr(config, "latency_warmup", 0) or 0))
    if _w == 0 and os.environ.get("ALPHAGRAD_MEASURE_WARMUP", "1") != "0":
        return 1
    return _w


# ---------------------------------------------------------------------------
# THE TIME BUDGET (owner ruling 2026-09-14). See `EnvConfig.measure_budget_secs`
# for the measurement that motivates it.
# ---------------------------------------------------------------------------
# The floor and the ceiling of the inner-rep count. NOT configurable: 50 is the
# one number of the protocol that was ever measured (inner 5 reads 20.7 percent
# high on the identity plan, inner 20 within 3 percent of inner 50), and 5 is
# the value every pre-2026-08 campaign ran, so nothing below it is a new
# regime. What the window rule chooses is where BETWEEN them a given program
# lands.
MEASURE_INNER_MIN = 5
MEASURE_INNER_MAX = 50


def resolve_measure_inner(t_exec_s: float, window_s: float,
                          hi: int = MEASURE_INNER_MAX) -> int:
    """Executions per timed window for a program that takes `t_exec_s`.

    ``clamp(ceil(window_s / t_exec_s), 5, hi)`` where `hi` is
    ``--latency-inner-reps``. At the campaign's 50 that is the owner's rule
    verbatim: a 121 us reference gets 50 (a 50 ms window would hold 413) and
    an 18.2 ms candidate gets the floor of 5 (it would hold 3).

    THE FLAG IS THE CEILING, never exceeded and never raised. Every caller
    that configures a SMALLER inner than 5 -- the legacy default of 1, and
    `landscape_map`'s 5 -- therefore measures exactly as it did before this
    ruling: the window rule can only choose BETWEEN 5 and the flag, and a
    flag below 5 collapses the interval onto itself.

    A non-positive or non-finite `t_exec_s` is a broken probe, not an
    infinitely fast program, and takes the ceiling -- the conservative end.
    """
    hi = max(1, int(hi))
    lo = min(MEASURE_INNER_MIN, hi)
    if not math.isfinite(t_exec_s) or t_exec_s <= 0.0:
        return hi
    want = math.ceil(float(window_s) / float(t_exec_s))
    return int(min(hi, max(lo, want)))


def resolve_measure_windows(t_exec_s: float, inner: int, budget_s: float,
                            cap: int) -> int:
    """Timed windows that fit `budget_s` seconds of executions, capped.

    ``clamp(round(budget_s / (inner * t_exec_s)), 1, cap)``. `cap` is
    ``num_data_points * reps_per_point`` for the candidate: under the ruling
    those two flags are the CEILING on the sample count, never the count.

    A broken probe (non-finite or non-positive `t_exec_s`) yields 1 window:
    one honest sample beats an unbounded loop.
    """
    cap = max(1, int(cap))
    if not math.isfinite(t_exec_s) or t_exec_s <= 0.0:
        return 1
    per_window = float(inner) * float(t_exec_s)
    if per_window <= 0.0:
        return 1
    want = int(round(float(budget_s) / per_window))
    return int(min(cap, max(1, want)))


def interleave_windows(n_a: int, n_b: int) -> list:
    """The A/B schedule of `n_a` candidate windows and `n_b` reference ones.

    Returns a list of 0 (candidate) and 1 (reference) of length ``n_a + n_b``
    in which the two streams are spread PROPORTIONALLY -- ``[0, 1, 0, 1, ...]``
    when the counts are equal, and one candidate window every ``n_b / n_a``
    reference windows when they are not.

    WHY, and why not two blocks: the paired ratio's job is to cancel GPU
    drift, and it can only cancel drift that is common to both halves. Two
    back-to-back blocks put the whole reference measurement AFTER the whole
    candidate measurement, so any clock or thermal excursion during the plan
    lands on one half only. Interleaving puts the two halves in the same
    seconds. The ratio is still computed from the two medians; only the ORDER
    of the windows changes.
    """
    n_a = max(0, int(n_a))
    n_b = max(0, int(n_b))
    out: list = []
    ia = ib = 0
    total = n_a + n_b
    for _ in range(total):
        # Take from whichever stream is furthest behind its share.
        if ib >= n_b or (ia < n_a and (ia + 0.5) * n_b <= (ib + 0.5) * n_a):
            out.append(0)
            ia += 1
        else:
            out.append(1)
            ib += 1
    return out


def _paired_log_delta(candidate: float, reference: float,
                      floor: float) -> float:
    """``log(max(candidate, floor)) - log(max(reference, floor))``."""
    return (math.log(max(float(candidate), floor))
            - math.log(max(float(reference), floor)))


def paired_log_costs(latency_ns: float, peak_memory: float,
                     ref_latency_ns: float, ref_memory: float
                     ) -> tuple[float, float, int]:
    """The two cost channels as PAIRED LOG-DIFFERENCES against rev-exact.

    ``(Delta_lat, Delta_mem, n_floored)``. ``Delta_c = log cost_c(candidate)
    - log cost_c(rev-exact)``: negative = the candidate is cheaper. The
    caller stores both negated, like every cost slot, so a cheaper plan
    scores above 0 and rev-exact scores exactly 0.

    Latency ``0.0`` means NOT MEASURED (``config.measure_latency`` off) and
    passes through as 0.0 -- for the pair, since candidate and reference
    are measured under one config. A measured latency is never 0: the
    campaign path clamps it up to `_LAT_FLOOR_NS` first.

    BOTH channels are floored on BOTH sides, at the policy
    :func:`paired_cost_floor` names (``--paired-cost-floor``): the
    reference's own cost by default, the one-byte / 100-ns pair under
    ``byte``. ``n_floored`` says how many of the two MEMORY readings the
    floor replaced.
    """
    _ref_floor = paired_cost_floor() == "reference"
    lat_floor = (max(_LAT_FLOOR_NS, float(ref_latency_ns)) if _ref_floor
                 else _LAT_FLOOR_NS)
    mem_floor = (max(_MEM_LOG_FLOOR_BYTES, float(ref_memory)) if _ref_floor
                 else _MEM_LOG_FLOOR_BYTES)
    if latency_ns > 0.0 and ref_latency_ns > 0.0:
        d_lat = _paired_log_delta(latency_ns, ref_latency_ns, lat_floor)
    else:
        d_lat = 0.0
    n_floored = int(float(peak_memory) < mem_floor) + int(
        float(ref_memory) < mem_floor)
    d_mem = _paired_log_delta(peak_memory, ref_memory, mem_floor)
    return float(d_lat), float(d_mem), n_floored


# TICKET dsnn-dfw.44, second design. ONE PAIRED LOG RATIO PER CANDIDATE
# WINDOW, in nats. The pair partner is the MEDIAN of that measurement's
# reference windows, not the interleaved neighbour, for three reasons: it is
# the same aggregate `_aggregate_samples` gives the reference in the paired
# cost, so the plan's median window ratio IS its cost channel; the median of
# 160 reference windows carries about one eighth of a single candidate
# window's standard error, so holding it fixed loses almost nothing; and a
# neighbour would fold one reference window's noise into every ratio and
# widen the band without saying anything more about the CANDIDATE, which is
# the thing being compared. The floor is PHYSICAL only (100 ns, 1 byte),
# never the reference floor `paired_log_costs` applies: the archive's
# coordinate must keep the half of the axis where a plan is FASTER than
# rev-exact, which the reward discards.
def paired_window_log_ratios(cand_samples, ref_samples, floor: float,
                             fallback: float) -> np.ndarray:
    c = np.asarray(list(cand_samples), dtype=np.float64)
    r = np.asarray(list(ref_samples), dtype=np.float64)
    if c.size == 0 or r.size == 0:
        # A channel with no timed window (the static temp) is one reading.
        return np.array([float(fallback)], dtype=np.float64)
    ref = float(np.median(np.maximum(r, float(floor))))
    return np.log(np.maximum(c, float(floor))) - math.log(ref)


def window_ratio_record(windows) -> dict:
    """The plan-log form of one objective's windows: the ratios themselves,
    so any rule can be recomputed offline, and the band the archive fits."""
    from alphagrad.approx.common.pareto_archive import median_band
    w = np.asarray(windows, dtype=np.float64).reshape(-1)
    med, lo, hi = median_band(w)
    return {"windows": [float(x) for x in w], "median": med,
            "lo": lo, "hi": hi, "n": int(w.size)}


def _static_temp_bytes(compiled) -> float | None:
    """``compiled.memory_analysis().temp_size_in_bytes`` -- the quantity
    reward slot 5 holds under ``--mem-channel temp`` (ticket .49) -- or
    None when the executable exposes no analysis."""
    try:
        ma = compiled.memory_analysis()
    except Exception:
        return None
    if ma is None:
        return None
    return float(getattr(ma, "temp_size_in_bytes", 0) or 0.0)


def _record_paired_ref(rec: dict) -> None:
    if len(_PAIRED_REF) >= _PAIRED_REF_CAP:
        _PAIRED_REF_DROPPED[0] += 1
        return
    _PAIRED_REF.append(rec)


def consume_paired_refs() -> dict:
    """Pop this process's paired-reference records (see `_PAIRED_REF`).

    ``{"records": [...], "dropped": int}``. Rides `consume_plan_records`,
    like the memory-parity drain, because the reference is measured in the
    process that measured the candidate -- the Ray measure actor under
    --ray-measure -- and the trainer's own module globals never see it.
    """
    out = {"records": list(_PAIRED_REF), "dropped": int(_PAIRED_REF_DROPPED[0])}
    _PAIRED_REF.clear()
    _PAIRED_REF_DROPPED[0] = 0
    return out


def paired_ref_summary(records) -> dict:
    """Per-period reference numbers for the log dict, in POSITIVE units
    (ticket .45): the mean rev-exact latency in ns and static temp bytes
    (and watermark bytes), how many references were taken, and how many
    memory readings (candidate or reference) the log floor replaced."""
    _lat = [float(r["latency_ns"]) for r in records
            if r.get("latency_ns") is not None and r["latency_ns"] > 0.0]
    _tmp = [float(r["temp_bytes"]) for r in records
            if r.get("temp_bytes") is not None]
    _wm = [float(r["watermark_bytes"]) for r in records
           if r.get("watermark_bytes") is not None]
    return {
        "n": int(len(records)),
        "latency_ns": float(np.mean(_lat)) if _lat else float("nan"),
        "temp_bytes": float(np.mean(_tmp)) if _tmp else float("nan"),
        "watermark_bytes": float(np.mean(_wm)) if _wm else float("nan"),
        "mem_floored": int(sum(int(r.get("mem_floored", 0)) for r in records)),
    }


def _dense_cosine(jac_exact, jac_approx):
    """``(cos, rel_frob)`` of two gradient pytrees, leaf by leaf, DENSE.

    The plain accumulator, for the one case ``_quality_metrics`` cannot serve:
    the reference is ``jax.grad``'s pytree and the candidate is jacve's, so the
    two do not carry the same container and the exact-structure check of
    ticket .62 would refuse them. Shapes still have to match leaf for leaf --
    a mismatch here is a real defect and RAISES, exactly as it does there.
    This is the same comparison ``_grad_oracle_check`` makes.
    """
    leaves_a = _gradient_leaves(jac_approx)
    leaves_e = jax.tree_util.tree_leaves(jac_exact)
    if len(leaves_a) != len(leaves_e):
        raise GradientStructureMismatch(
            f"[grad_cosine] the oracle reference has {len(leaves_e)} leaves "
            f"and the plan's gradient {len(leaves_a)}")
    dot = ee = aa = rr = 0.0
    for i, (e, a) in enumerate(zip(leaves_e, leaves_a)):
        e_np = np.asarray(e, dtype=np.float64)
        if a is None:
            # A DEAD PATH IS A ZERO GRADIENT, not a broken pytree. A face
            # SKIP can delete every path to one parameter, and then the
            # plan's gradient for it IS zero: graphax's dense path returns
            # zeros there and `_gradient_similarity` has always scored it as
            # zero. This accumulator raised instead, so the first real policy
            # plan that skipped its way to a dead parameter took the whole
            # measurement down (job 66105, a one-episode run on the rtrl
            # arm). Scored the same way here.
            ee += float(np.sum(e_np ** 2))
            rr += float(np.sum(e_np ** 2))
            continue
        a_arr = a.dense() if _is_sparse_tensor(a) else a
        a_np = np.asarray(a_arr, dtype=np.float64)
        if a_np.shape != e_np.shape and a_np.shape == e_np.shape[::-1]:
            a_np = a_np.T
        if a_np.shape != e_np.shape:
            raise GradientStructureMismatch(
                f"[grad_cosine] leaf {i}: the oracle reference has shape "
                f"{e_np.shape} and the plan's gradient {a_np.shape}")
        dot += float(np.sum(e_np * a_np))
        ee += float(np.sum(e_np ** 2))
        aa += float(np.sum(a_np ** 2))
        rr += float(np.sum((e_np - a_np) ** 2))
    if ee <= 0.0 or aa <= 0.0:
        return 0.0, 1.0
    return dot / math.sqrt(ee * aa), math.sqrt(rr) / math.sqrt(ee)


def _quality_metrics(jac_exact, jac_approx, *, align: bool = False,
                     site: str = "grad_cosine"):
    """`(cosine_sim, relative_frobenius)` of `jac_approx` against `jac_exact`.

    Ticket dsnn-3qm.62: the two gradient pytrees are compared with EXACT
    structure equality by ``_gradient_similarity`` -- a leaf count, logical
    shape or layout mismatch RAISES ``GradientStructureMismatch``; a dead
    path (None) is a zero gradient; an implicit (Reduce'd) dim is compared
    analytically, never densified. The old silent ``(0.0, 1.0)`` clamp on a
    shape mismatch is what hid finding 60 (an engine-induced axis
    permutation read as quality 0.0 on every applied plan of the .41 sweep).

    ``align=True`` is the ANALYTIC JACOBIAN-COSINE path only (``jac_cosine``,
    the full-Jacobian benchmarks): ``_align_jac`` transposes a 2-D block
    back first and raises when the trees cannot be mapped.

    Returns the WORST score `(0.0, 1.0)` only when the exact side has no
    leaves or zero size (nothing to compare against; a broken reference, not
    a broken plan).

    ``ALPHAGRAD_DEBUG_QUALITY=1`` enables a one-line diagnostic print.
    """
    if align:
        jac_approx = _align_jac(jac_approx, jac_exact, site=site)
    leaves_e = _gradient_leaves(jac_exact)
    if not leaves_e or all(x is None for x in leaves_e):
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    # Measure GPUs rotate but the exact reference is cached, so the two can
    # land on different devices -> jitted ops raise "Received incompatible
    # devices". Co-locate onto the reference's device (read-only, one transfer
    # only when they differ).
    try:
        _ed = next(iter(next(x for x in leaves_e if x is not None
                             and not _is_sparse_tensor(x)).devices()))
        jac_approx = jax.tree_util.tree_map(
            lambda a: (jax.device_put(a, _ed)
                       if hasattr(a, "devices") and next(iter(a.devices())) is not _ed
                       else a),
            jac_approx)
    except Exception:
        pass

    dot, ee, aa, rr, _total = _gradient_similarity(jac_exact, jac_approx, site)
    if _total == 0:
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    exact_norm = jnp.sqrt(ee)
    approx_norm = jnp.sqrt(aa)
    # THE DENOMINATOR IS GUARDED AGAINST ZERO, NOT AGAINST SMALL (defect found
    # 2026-09-16 on the recurrent SHD family). It used to be
    # ``max(||.||, sqrt(1e-7))`` on both factors, which is a FLOOR at a
    # gradient norm of 3.16e-4: below it the cosine of a plan that reproduced
    # the reference EXACTLY reads ``||g||^2 / 1e-7`` instead of 1.0, and it
    # reads it silently. Measured on the two-copy window arm at a sampled step
    # whose gradient norm is 2.19e-4: two bit-identical Jacobians scored
    # 0.3247. One step of a sparse spiking network is exactly the regime where
    # that happens, so the floor would have priced the step position instead
    # of the plan on every SNN row of the matrix.
    #
    # A norm of EXACTLY zero is a different case and is still scored 0.0 here:
    # the cosine is undefined there, and `_grad_cosine_quality` DROPS such a
    # batch rather than counting it (the zero-reference guard of 2026-09-16).
    _denom = exact_norm * approx_norm
    _safe = jnp.where(_denom > 0, _denom, 1.0)
    cos = jnp.where(_denom > 0, dot / _safe, 0.0)
    # Certain quant/compress combos yield complex-valued Jacobian leaves,
    # making the accumulated dot complex. Use the real part — matches the
    # reward path's existing real cast.
    cos = jnp.real(cos)
    _en = jnp.where(exact_norm > 0, exact_norm, 1.0)
    rel_frob = jnp.where(
        exact_norm > 0, jnp.sqrt(jnp.maximum(rr, 0.0)) / _en, 1.0)

    if os.environ.get("ALPHAGRAD_DEBUG_QUALITY", "0") == "1":
        print(
            f"[quality-debug] cos={float(cos):+.4f} frob={float(rel_frob):+.4f} "
            f"||exact||={float(exact_norm):.3g} ||approx||={float(approx_norm):.3g} "
            f"size={_total}",
            flush=True,
        )
    return cos, rel_frob


def _warn_cosine_is_now_grad_cosine() -> None:
    """One-shot, loud: ``cosine`` no longer means the Jacobian cosine."""
    if _COSINE_RENAME_WARNED:
        return
    _COSINE_RENAME_WARNED.append(1)
    print(
        "[measure] NOTE ALPHAGRAD_QUALITY_METRIC=cosine is DEPRECATED and its "
        "MEANING HAS CHANGED: it now resolves to 'grad_cosine' (the gradient "
        "cosine at init over ALPHAGRAD_GRAD_COSINE_K probe batches) for any "
        "scalar-loss target, and to 'jac_cosine' (the legacy Jacobian cosine "
        "at the calibration samples) only for the analytic AD benchmarks. "
        "Ask for 'jac_cosine' explicitly to reproduce a pre-2026-08-28 run.",
        flush=True,
    )


def _grad_cosine_k(config=None) -> int:
    """How many probe batches the gradient cosine averages over.

    K=1 where it has always been 1: it keeps the channel at EXACTLY ONE exact
    execution per terminal plan -- the same reference count the Jacobian
    cosine paid -- and the bake-off found K>1 buys essentially no extra
    correlation on the dense targets.

    A GENERATOR MAY ASK FOR MORE, AND ONE HAS TO. On a sparse spiking target a
    probe batch IS a step position, and at a step where nothing fired the exact
    gradient is identically zero, the cosine is undefined and the measurement
    is refused. 43 of the 99 legal step positions of the recording the RSNN_SHD
    campaign drew are silent (probe 66655), so K=1 refuses 43 percent of every
    measurement on that target. Such a generator declares `probe_batches` and
    that number is the default here. ALPHAGRAD_GRAD_COSINE_K overrides both.
    """
    _default = 1
    if config is not None:
        try:
            _default = max(1, int(getattr(getattr(config, "data_gen", None),
                                          "probe_batches", 1) or 1))
        except (TypeError, ValueError):
            _default = 1
    try:
        return max(1, int(os.environ.get("ALPHAGRAD_GRAD_COSINE_K",
                                         str(_default))))
    except ValueError:
        return _default



# ---------------------------------------------------------------------------
# THE QUALITY CHANNEL (reward slot 6).
#
# Until 2026-08-07 slot 6 held the cosine similarity between the plan's
# Jacobian and the exact Jacobian. Measured against the quantity we actually
# care about — the FINAL DOWNSTREAM TEST ACCURACY of a network trained with
# the plan's gradient (139 archived plans x 3-5 seeds x 100k MNIST steps):
#
#                             Pearson  Spearman  cost/plan  peak mem
#   Jacobian cosine            0.610    0.858     9.70 s     4.24 GB
#   gradient cosine at init    0.737    0.857     0.33 s     ~40 MB
#   loss drop, 200 Adam steps  0.922    0.854     0.22 s     40 MB
#
# THE THREE COST NUMBERS ABOVE ARE STALE. They predate 744fc3d, which made the
# traced target an unconditional SCALAR LOSS: what used to be a per-class
# Jacobian is now the gradient, so nothing materialises a Jacobian any more and
# the cosine got ~3000x cheaper. RE-MEASURED 2026-08-28 on one Blackwell GPU,
# VmappedNeuralNetwork/MNIST, 31 plans spanning cos 0.00 (frozen gradient, the
# k24/f0 analogue) to 1.00, ground truth = final downstream test accuracy at
# 20k steps x 3 seeds through downstream_train.py -- the SAME definition the
# table above used, over a NEWLY GENERATED plan population, because the 139
# archived plans no longer replay (they were recorded against a graph whose
# edges were 4-D; apply_compress rejects them now):
#
#                                    Pearson  Spearman  s/plan   peak MB
#   gradient cosine at init, K=1      0.874     0.805    0.003     295
#   gradient cosine at init, K=4      0.861     0.800    0.012     295
#   gradient cosine at init, K=8      0.856     0.807    0.024     295
#   mean cosine over 200 steps        0.815     0.797    1.191     295
#   last-step cosine, 200 steps       0.761     0.800    1.191     295
#   cosine of summed gradients        0.685     0.748    1.191     295
#   clipped relative Frobenius        0.350     0.413    0.003     295
#   loss drop, 200 Adam steps         0.885     0.765    0.606     295
#   Jacobian cosine (legacy)          0.397     0.498    0.003     443
#
# (peak MB is process peak; ~295 MB of it is the resident model+data baseline,
# so the Jacobian cosine's MARGINAL cost is the ~148 MB of Jacobian it builds
# and every gradient-space metric's marginal cost is ~0.)
#
# WHAT THIS SETTLED, and it answers the owner's 2026-08-07 question directly:
#   * the gradient cosine is best AT INIT and K=1 -- averaging over more probe
#     batches makes it slightly WORSE (0.874 -> 0.856), so K=1 is both the most
#     predictive and the cheapest, and it keeps the channel at EXACTLY ONE
#     exact execution per terminal plan;
#   * NONE of the trajectory formulations pay: aggregated-gradient cosine
#     0.685, last-step 0.761, mean-over-steps 0.815 -- all below the K=1 init
#     cosine and ~400x more expensive;
#   * the legacy JACOBIAN cosine is the worst cosine by a wide margin
#     (0.397/0.498). Replacing it with the gradient cosine is a 2.2x gain in
#     Pearson at IDENTICAL wall cost and 33% less peak memory, which is why
#     "cosine" now resolves to grad_cosine;
#   * CLIPPED RELATIVE FROBENIUS -- the A2 channel whose correlation had never
#     been measured -- is WEAK: 0.350 Pearson / 0.413 Spearman, worse than
#     every cosine variant including the legacy one. It should stay LOGGED and
#     should NOT be given a trained slot on this evidence.
#   * loss_drop remained the `auto` default until 2026-09-02: its Pearson
#     0.885 is within noise of the K=1 gradient cosine's 0.874, but note it is
#     WORSE on Spearman (0.765 vs 0.805) while costing 200x more. The owner
#     made the call on 2026-09-02 (ticket dsnn-3qm.39, finding 51: a plan can
#     score 0.885 on loss_drop while its gradient points elsewhere): `auto` IS
#     the gradient cosine for every scalar-loss target, and loss_drop is
#     selectable by name only.
#
# Under ALPHAGRAD_QUALITY_METRIC=loss_drop slot 6 holds the LOSS DROP OF A
# SHORT ADAM WALK DRIVEN BY THE PLAN'S OWN GRADIENT:
#
#     L0      = loss(W0, probe)
#     W_{t+1} = adam(W_t, plan_gradient(W_t, batch))     t = 0 .. T-1
#     quality = (L0 - loss(W_T, probe)) / |L0|
#
# ``plan_gradient`` is the gradient the PLAN BEING EVALUATED produces — its
# elimination order AND its approximations — while ``loss`` is the TRUE scalar
# loss. The channel therefore answers "does training with this approximate
# gradient actually reduce the real loss", which is the question the thesis is
# asking, rather than "does this approximate Jacobian point the same way".
#
# Rejected in the owner's sweep, do not re-litigate: 30 probe batches instead
# of 5 (+0.00 Pearson), weight noise +/-10..100% (+0.00), cosine at trained
# weights (-0.11), chained displacement over 20/200/1000 steps (-0.24),
# product of per-step cosines (-0.39), fraction of decreasing steps
# (Spearman -0.21).
_QUALITY_METRIC_ENV = "ALPHAGRAD_QUALITY_METRIC"

_COSINE_RENAME_WARNED: list[int] = []


def _warn_cosine_is_now_grad_cosine() -> None:
    """One-shot, loud: ``cosine`` no longer means the Jacobian cosine."""
    if _COSINE_RENAME_WARNED:
        return
    _COSINE_RENAME_WARNED.append(1)
    print(
        "[measure] NOTE ALPHAGRAD_QUALITY_METRIC=cosine is DEPRECATED and its "
        "MEANING HAS CHANGED: it now resolves to 'grad_cosine' (the gradient "
        "cosine at init over ALPHAGRAD_GRAD_COSINE_K probe batches) for any "
        "scalar-loss target, and to 'jac_cosine' (the legacy Jacobian cosine "
        "at the calibration samples) only for the analytic AD benchmarks. "
        "Ask for 'jac_cosine' explicitly to reproduce a pre-2026-08-28 run.",
        flush=True,
    )


def _grad_cosine_k(config=None) -> int:
    """How many probe batches the gradient cosine averages over.

    K=1 where it has always been 1: it keeps the channel at EXACTLY ONE exact
    execution per terminal plan -- the same reference count the Jacobian
    cosine paid -- and the bake-off found K>1 buys essentially no extra
    correlation on the dense targets.

    A GENERATOR MAY ASK FOR MORE, AND ONE HAS TO. On a sparse spiking target a
    probe batch IS a step position, and at a step where nothing fired the exact
    gradient is identically zero, the cosine is undefined and the measurement
    is refused. 43 of the 99 legal step positions of the recording the RSNN_SHD
    campaign drew are silent (probe 66655), so K=1 refuses 43 percent of every
    measurement on that target. Such a generator declares `probe_batches` and
    that number is the default here. ALPHAGRAD_GRAD_COSINE_K overrides both.
    """
    _default = 1
    if config is not None:
        try:
            _default = max(1, int(getattr(getattr(config, "data_gen", None),
                                          "probe_batches", 1) or 1))
        except (TypeError, ValueError):
            _default = 1
    try:
        return max(1, int(os.environ.get("ALPHAGRAD_GRAD_COSINE_K",
                                         str(_default))))
    except ValueError:
        return _default



_GRAD_ORACLE_ENV = "ALPHAGRAD_GRAD_ORACLE"
_GRAD_ORACLE_TOL_ENV = "ALPHAGRAD_GRAD_ORACLE_TOL"
_GRAD_ORACLE_CADENCE_ENV = "ALPHAGRAD_GRAD_ORACLE_CADENCE"
_GRAD_ORACLE_TIMEOUT_ENV = "ALPHAGRAD_GRAD_ORACLE_TIMEOUT"
# "Once per process and order" now lives with the worker that schedules the
# checks (`common.grad_oracle_async.AsyncGradOracle`), because the trainer has
# to know which orders it is still waiting for and a module-global set does not
# say. This file keeps only what the check itself needs.
_GRAD_ORACLE_STATS = {"checks": 0, "rel_l2_max": 0.0}

# THE BAR THE ORACLE ENFORCES. One constant, read by the check and by the
# ``--grad-oracle`` help of ppo.py, so the help cannot drift from the code
# again (it said 1e-4 until 2026-09-16, ticket dsnn-df8).
_GRAD_ORACLE_TOL = 1e-3

# THE MATMUL PRECISION BOTH SIDES OF THE ORACLE RUN AT (ticket dsnn-df8).
# The default float32 dot on this hardware is TF32: 10 mantissa bits, about
# 5e-4 relative error per product. Measured on pgi15-gpu17 on TransformerLM,
# ``jax.grad`` in float32 at the default precision sits 1.054e-3 from the
# float64 truth -- FURTHER from the truth than the 1e-3 gate it is supposed
# to guard -- while the same call at "highest" sits 9.4e-7 from it. Running
# both sides at "highest" therefore removes the mechanism behind 47 of the 48
# refusals seen on the free-order TLM run, instead of widening the gate.
_GRAD_ORACLE_PRECISION = "highest"

# The reference (``jax.grad`` of the target on the probe batch) DOES NOT
# DEPEND ON THE ELIMINATION ORDER, so it is computed once per process,
# device and probe batch and reused by every order. See
# :func:`_grad_oracle_reference` for the key.
_GRAD_ORACLE_REF: dict = {}
_GRAD_ORACLE_REF_MAX = 8
_GRAD_ORACLE_REF_STATS = {"hits": 0, "misses": 0}

# What the oracle SAW as the live matmul precision on each side, recorded by
# the check so a test can prove the "highest" context was active for the
# elimination AND for jax.grad.
_GRAD_ORACLE_LAST_PRECISION: dict = {"plan": None, "reference": None}

# What the oracle SAW as the live float64 setting, and what the PROCESS saw
# outside the oracle. A test reads both to prove the float64 scope is the
# oracle's alone.
_GRAD_ORACLE_LAST_X64: dict = {"inside": None, "outside": None}


def _x64_scope():
    """A float64 scope for the GRADIENT ORACLE ALONE.

    THE FLAG MUST NOT BE GLOBAL. ``jax.config.update("jax_enable_x64", True)``
    changes the whole process: every later trace in that process, on any
    thread, sees float64 weak types and float64 literals. In a measurement
    actor that process is also the one that compiles and times the plan, so a
    global toggle makes the measured program a different program from the one
    measured at cadence 0. It also moves what
    ``alphagrad.approx.common.plan_log`` writes, because the dtype names it
    records are read off ``jax.config.jax_enable_x64``.

    This helper returns a context manager that sets the SAME setting on the
    CURRENT THREAD only and restores it on exit. The oracle's own trace and
    compile see float64; nothing else in the process ever does.

    THREE NAMES, ONE THING. ``jax.experimental.enable_x64`` is the public
    spelling and is gone from this build (jax 0.10.2, measured, job 66209);
    the thing it wrapped is the ``enable_x64`` config STATE, whose ``__call__``
    is the thread-local context manager. This module already imports
    ``jax._src.core``, so reaching into ``jax._src.config`` for the state is
    the same dependency, not a new one.

    No silent fallback: if none of the three names is there, this raises.
    Running the oracle through the global flag again is not an option the
    apparatus may take on its own.
    """
    try:
        from jax.experimental import enable_x64 as _exp_enable_x64
    except ImportError:
        _exp_enable_x64 = None
    if callable(_exp_enable_x64):
        return _exp_enable_x64()
    _state = getattr(jax.config, "enable_x64", None)
    if callable(_state):
        return _state(True)
    try:
        from jax._src import config as _jax_src_config
    except ImportError:
        _jax_src_config = None
    _state = getattr(_jax_src_config, "enable_x64", None)
    if callable(_state):
        return _state(True)
    raise RuntimeError(
        "the gradient oracle needs a THREAD-LOCAL float64 scope and this JAX "
        f"({getattr(jax, '__version__', '?')}) exposes none of "
        "jax.experimental.enable_x64, jax.config.enable_x64 and "
        "jax._src.config.enable_x64; the global "
        "jax.config.update('jax_enable_x64', True) is refused because it "
        "would change the measured program (see _x64_scope)")


def grad_oracle_cadence() -> int:
    """Episode cadence for checking the exact gradient against jax.grad.
    Default 50 (checked on episode 0 and every 50 episodes). Set 1 to
    check on every episode."""
    try:
        return max(1, int(os.environ.get(_GRAD_ORACLE_CADENCE_ENV, "50")))
    except ValueError:
        return 50


def grad_oracle_timeout() -> float:
    """How long one asynchronous check may be in flight before the trainer
    counts it as a ``timeout`` and stops waiting for it. Default 600 s,
    ``--grad-oracle-timeout`` on ppo.py, published as
    ``ALPHAGRAD_GRAD_ORACLE_TIMEOUT``. A timeout is MISSING DATA: it is counted
    and written and it does not stop the run."""
    try:
        return max(1.0, float(os.environ.get(_GRAD_ORACLE_TIMEOUT_ENV, "600")))
    except ValueError:
        return 600.0


class GradientOracleFailure(RuntimeError):
    """The exact gradient of an elimination order disagrees with ``jax.grad``
    (oracle A, ticket dsnn-3qm.62).

    IT NO LONGER REFUSES A PLAN (owner ruling 2026-09-18). The oracle is a
    sanity check and not part of the scoring: it runs on the CPU, in float64,
    on the trainer's oracle thread, after the plan has already been measured
    and scored. So this exception never reaches a measurement and never puts a
    sentinel reward anywhere. It is raised by the check and recorded as a
    ``fail``, and the TRAINER re-raises at the next episode boundary, after
    that episode's checkpoint has been written.

    THE RUN STOPS because a disagreement in float64 on the CPU is a real
    graphax defect: every order of the campaign's targets sits at 1e-14 there
    (agent-df8 report, 2026-09-16), so there is no noise band it could be.
    The checkpoint is what makes stopping cheap -- the run resumes from it once
    the defect is fixed."""


def grad_oracle() -> str:
    """``"reference"`` (default) or ``"off"``: whether the exact gradient of
    every elimination order is checked ONCE per process against ``jax.grad``
    (oracle A, owner Q6/Q21; ``--grad-oracle`` on ppo.py, published as
    ``ALPHAGRAD_GRAD_ORACLE``).

    THE CHECK IS RETROACTIVE (owner ruling 2026-09-18). It runs on the
    trainer's oracle thread, on the CPU, AFTER the order has been measured and
    scored -- so "reference" no longer names anything the reward path waits
    for. The measure actors read the same value and act on none of it."""
    want = os.environ.get(_GRAD_ORACLE_ENV, "reference").strip().lower()
    if want not in ("reference", "off"):
        raise ValueError(
            f"{_GRAD_ORACLE_ENV}={want!r}: expected 'reference' or 'off'")
    return want


def grad_oracle_tol() -> float:
    """The relative L2 bar a checked order must meet. ``_GRAD_ORACLE_TOL``
    (1e-3), overridable with ``ALPHAGRAD_GRAD_ORACLE_TOL``. ppo.py builds its
    ``--grad-oracle`` help from this function, so the two cannot disagree."""
    return float(os.environ.get(_GRAD_ORACLE_TOL_ENV, repr(_GRAD_ORACLE_TOL)))


def _matmul_precision() -> str:
    """The default matmul precision a ``dot_general`` traced RIGHT NOW would
    carry. The oracle reads it on both sides and records what it read."""
    return str(jax.config.jax_default_matmul_precision)


def _is_cpu_device(device) -> bool:
    """True when ``device`` is a CPU device. The asynchronous oracle runs on
    one, and two decisions in this file turn on it: which compile path the
    elimination takes, and which compiler faults are reachable at all."""
    return str(getattr(device, "platform", "")).lower() == "cpu"


def _grad_oracle_exact(config, order, args, device=None):
    """The plan's exact vertex-elimination gradient, TRACED AND COMPILED HERE.

    NOT the caller's ``compiled_exact``. Matmul precision is baked into the
    HLO at TRACE time, so an already-compiled executable cannot be re-run at
    another precision. ``compiled_exact`` is also the paired latency reference
    and the quality reference, and its precision must not move -- so the
    oracle builds its own copy of the SAME elimination (same order, same
    argnums, same has_aux, same sparse representation, no approximation
    kwargs) inside the precision context. That is the extra compile per
    process and order the oracle now pays, and caching the reference pays for
    most of it.

    ``_compile_measure`` and not a plain ``.compile()`` ON A GPU DEVICE:
    outside that path the XLA:GPU bitcast-hoisting pass fails a CHECK and
    dumps core on the TransformerLM plans (dsnn-df8, jobs 66074/66075).

    ON THE CPU DEVICE -- where the asynchronous oracle runs -- the plain
    compile is the right one. Every option ``_compile_measure`` sets and every
    fault it retries around is XLA:GPU's (ptxas, Triton, shared-memory tiles,
    the nvlink toolchain); none of them exists on the CPU backend, and handing
    XLA:CPU a set of ``xla_gpu_*`` options is at best a no-op and at worst an
    INVALID_ARGUMENT. This is also why the oracle's own compile can no longer
    refuse anything: the Blackwell shared-memory limit that refused 15.6
    percent of oracle-due plans is not reachable from here.
    """
    fn = jacve(config.target_fun, list(order), argnums=config.argnums,
               has_aux=config.has_aux,
               sparse_representation=bool(getattr(config, "sparse", False)))
    lowered = jax.jit(fn, keep_unused=True).lower(*args)
    exe = (lowered.compile() if _is_cpu_device(device)
           else _compile_measure(lowered))
    out = exe(*args)
    return _gradient_leaves(out[1] if config.has_aux else out)


def _grad_oracle_reference(config, args, device, probe_seed):
    """``jax.grad`` of the target on the probe batch, ONCE per process.

    THE REFERENCE DOES NOT DEPEND ON THE ELIMINATION ORDER. Before this it was
    recomputed for every order, which is where the budget for the extra
    ``highest``-precision elimination compile comes from (dsnn-df8 section 7
    item 4).

    THE KEY is ``(target identity, argnums, has_aux, device, PROBE SEED, data
    generator identity, arg shapes and dtypes)``. The probe seed is
    ``_walk_seed(role, episode)``, exactly the number ``_probe_batch`` keys its
    own cache on, so a NEW PROBE BATCH -- a new episode under
    ``--walk-rotate``, a new probe seed, another role -- is a different key and
    a miss. The target's weights are drawn once per process and never updated
    (the policy searches elimination orders, not weights), so they are not in
    the key; their shapes and dtypes are, so a changed argument still misses.
    The cached entry holds the target and the data generator, which pins the
    two ``id()`` values in the key against object reuse.
    """
    key = (id(config.target_fun),
           getattr(config.target_fun, "__qualname__", ""),
           tuple(int(i) for i in config.argnums),
           bool(config.has_aux),
           str(device),
           int(probe_seed),
           id(config.data_gen),
           tuple((tuple(getattr(x, "shape", ())), str(getattr(x, "dtype", "")))
                 for x in args))
    hit = _GRAD_ORACLE_REF.get(key)
    if hit is not None:
        _GRAD_ORACLE_REF_STATS["hits"] += 1
        return hit[2]
    _GRAD_ORACLE_REF_STATS["misses"] += 1
    ref = jax.grad(config.target_fun, argnums=config.argnums,
                   has_aux=config.has_aux)(*args)
    ref = ref[0] if config.has_aux else ref
    leaves = jax.tree_util.tree_leaves(ref)
    if len(_GRAD_ORACLE_REF) >= _GRAD_ORACLE_REF_MAX:
        _GRAD_ORACLE_REF.clear()
    _GRAD_ORACLE_REF[key] = (config.target_fun, config.data_gen, leaves)
    return leaves


def _grad_oracle_rel_l2(config, order, a, device, probe_seed):
    """THE NUMBER THE ORACLE IS ABOUT: ``(rel_l2, n_leaves)``.

    The plan's exact gradient of ``order`` against ``jax.grad`` of the target,
    both in float64 under :func:`_x64_scope`, both at
    ``_GRAD_ORACLE_PRECISION``, on ``device``, on the probe batch already
    substituted into ``a``.

    IT RETURNS A NUMBER AND DOES NOT JUDGE IT. The tolerance belongs to the
    caller: :func:`grad_oracle_cpu_check` records a ``fail`` above it and the
    trainer stops the run at the next episode boundary, after that episode's
    checkpoint. A STRUCTURAL disagreement -- a different number
    of leaves, a dead path, a shape that does not match -- still raises here,
    because there is no number to return for it.
    """
    # THE FLOAT64 SCOPE IS THE ORACLE'S ALONE (see _x64_scope). The global
    # flag is read here only to record what the process saw OUTSIDE the
    # scope, so a test can prove the scope did not move it. The scope is
    # thread-local, which is what lets the oracle's worker thread hold it open
    # while the trainer traces its own float32 work on another thread.
    _GRAD_ORACLE_LAST_X64["outside"] = bool(jax.config.jax_enable_x64)
    with _x64_scope():
        _GRAD_ORACLE_LAST_X64["inside"] = bool(jax.config.jax_enable_x64)
        a_f64 = []
        for x in a:
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
                a_f64.append(jnp.asarray(x, dtype=jnp.float64))
            elif isinstance(x, (tuple, list)):
                a_f64.append(jax.tree_util.tree_map(
                    lambda v: jnp.asarray(v, dtype=jnp.float64)
                    if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating)
                    else v, x))
            else:
                a_f64.append(x)
        with jax.default_matmul_precision(_GRAD_ORACLE_PRECISION):
            _GRAD_ORACLE_LAST_PRECISION["plan"] = _matmul_precision()
            leaves = _grad_oracle_exact(
                config, [int(v) for v in order], a_f64, device)
            _GRAD_ORACLE_LAST_PRECISION["reference"] = _matmul_precision()
            ref_leaves = _grad_oracle_reference(config, a_f64, device, probe_seed)
            # Densify and convert to numpy while x64 is active so JAX does not
            # warn or truncate float64 tensors when x64 is restored.
            leaves_np = [
                np.asarray(e.dense() if _is_sparse_tensor(e) else e, dtype=np.float64)
                if e is not None else None
                for e in leaves
            ]
            ref_leaves_np = [np.asarray(r, dtype=np.float64) for r in ref_leaves]
    _o6 = tuple(int(v) for v in order)[:6]
    if len(leaves_np) != len(ref_leaves_np):
        raise GradientOracleFailure(
            f"[grad-oracle] order {_o6}...: {len(leaves_np)} exact "
            f"gradient leaves against {len(ref_leaves_np)} from jax.grad")
    num = den = 0.0
    for i, (e_np, r_np) in enumerate(zip(leaves_np, ref_leaves_np)):
        if e_np is None:
            raise GradientOracleFailure(
                f"[grad-oracle] order {_o6}...: exact leaf {i} is a "
                f"dead path")
        if e_np.shape != r_np.shape:
            raise GradientOracleFailure(
                f"[grad-oracle] order {_o6}...: exact leaf {i} has "
                f"shape {e_np.shape}, jax.grad {r_np.shape}")
        num += float(np.sum((e_np - r_np) ** 2))
        den += float(np.sum(r_np ** 2))
    return math.sqrt(num) / max(math.sqrt(den), 1e-30), len(leaves_np)


# ---------------------------------------------------------------------------
# THE ASYNCHRONOUS ORACLE'S TWO HALVES (owner ruling 2026-09-18).
#
# The oracle is a SANITY CHECK and no longer part of the scoring. It runs on
# the CPU device, in the trainer process, on one worker thread that nothing
# waits for -- see `alphagrad.approx.common.grad_oracle_async`. This file owns
# the two halves that need env's own state:
#
#   `grad_oracle_submission`  runs on the TRAINER thread and freezes what the
#                             check will need: the probe seed of the episode
#                             and the arguments, on the host as numpy.
#   `grad_oracle_cpu_check`   runs on the WORKER thread and answers.
#
# THE SPLIT IS NOT COSMETIC. `_probe_batch` and `_PROBE_BATCH` are process
# globals with no lock, and `base_args` are device arrays the trainer is using;
# reading either from the worker would be a race. Everything the worker touches
# is therefore host memory it was handed, plus the CPU device.
# ---------------------------------------------------------------------------
def grad_oracle_cpu_device():
    """The CPU device the asynchronous oracle runs on.

    ``jax.devices("cpu")[0]``, ALWAYS EXPLICIT. The trainer's default device is
    the GPU, and the whole point of the ruling is that this check does not
    touch it: a float64 elimination of a TransformerLM plan is the compile that
    exhausted Blackwell's shared memory, and it has no business on the device
    the campaign is timing. Raises if the backend has no CPU device, because
    silently falling back to the GPU is the behaviour this replaces."""
    devs = jax.devices("cpu")
    if not devs:
        raise RuntimeError(
            "the asynchronous gradient oracle runs on jax.devices('cpu')[0] "
            "and this backend exposes no CPU device")
    return devs[0]


def grad_oracle_submission(config, base_args, episode):
    """What the TRAINER freezes for one oracle-due episode.

    Returns ``(probe_seed, args_np)`` or ``None`` when this configuration has
    nothing the oracle can check (no target, no scalar loss, no data
    generator -- the same three guards the synchronous check had).

    ``args_np`` is HOST memory: the arguments with the episode's probe batch
    already in slots 0 and 1, pulled off the device here, on the trainer's own
    thread, at the moment the episode ends. The worker thread is then
    independent of every device array the trainer goes on to use, and of the
    probe-batch cache.
    """
    if grad_oracle() == "off":
        return None
    if config.target_fun is None or not getattr(config, "scalar_target", False):
        return None
    # THE EPISODE IS READ ONCE, not twice: `_probe_batch` would fold
    # `walk_episode()` in itself, and the reference cache has to key on the
    # SAME number the batch was drawn at. The episode is a PARAMETER here and
    # not the published rotation index, because `host_log` submits one episode
    # late under --measure-pipeline and must freeze that episode's batch.
    ep = int(episode) if walk_rotate_enabled() else 0
    probe_seed = _walk_seed("train", ep)
    data = _probe_batch(config, base_args, role="train", index=0, episode=ep)
    if data is None:
        return None
    a = list(jax.device_get(list(base_args)))
    # THE DECLARED SLOTS, not the first two: a generator whose draw is a
    # carried state puts arrays further along the tuple.
    for slot, d in zip(_data_slots(config, data), data):
        a[slot] = np.asarray(d)
    return int(probe_seed), a


def grad_oracle_cpu_check(config, args_np, order, probe_seed):
    """ONE check, on the CPU device, in float64. Returns ``(status, rel_l2)``.

    ``status`` is ``"pass"`` when the relative L2 distance is at or below
    :func:`grad_oracle_tol` and ``"fail"`` above it. A FAIL HERE IS A REAL
    DEFECT: in float64 on the CPU every order of this target sits at 1e-14
    (agent-df8 report, 2026-09-16), so there is no noise band to hide in and
    nothing to widen. The caller stops the run.

    ``jax.default_device`` is held over the whole call so that the arguments,
    the elimination's own trace and compile, and ``jax.grad``'s all land on the
    CPU; the probe batch and the arguments are ALSO device_put explicitly, so
    the placement is stated and not inferred. Both that context and
    :func:`_x64_scope` are thread-local, which is what makes this safe to run
    beside the trainer.
    """
    dev = grad_oracle_cpu_device()
    with jax.default_device(dev):
        a = []
        for x in args_np:
            if hasattr(x, "dtype") or isinstance(x, (tuple, list)):
                a.append(jax.device_put(x, dev))
            else:
                a.append(x)
        rel, n_leaves = _grad_oracle_rel_l2(
            config, [int(v) for v in order], a, dev, int(probe_seed))
    tol = grad_oracle_tol()
    _GRAD_ORACLE_STATS["checks"] += 1
    _GRAD_ORACLE_STATS["rel_l2_max"] = max(_GRAD_ORACLE_STATS["rel_l2_max"], rel)
    status = "pass" if rel <= tol else "fail"
    print(f"[grad-oracle] order {tuple(int(v) for v in order)[:6]}... on "
          f"{dev}: exact gradient vs jax.grad rel_l2={rel:.3e} "
          f"({n_leaves} leaves, float64, matmul precision "
          f"{_GRAD_ORACLE_PRECISION}) -> {status}", flush=True)
    return status, rel


def quality_metric(config=None) -> str:
    """``"grad_cosine"``, ``"jac_cosine"``, ``"loss_drop"`` or ``"none"`` —
    WHICH quantity reward slot 6 holds.

    ``ALPHAGRAD_QUALITY_METRIC`` selects it; the default ``auto`` resolves to
    ``grad_cosine`` (owner ruling 2026-09-02, ticket dsnn-3qm.39: the cosine
    between the plan's gradient and the rev-exact gradient on the same probe
    batch) whenever the measured graph IS a scalar loss, which is a FACT READ
    OFF THE TRACED JAXPR (``EnvConfig.scalar_target``, set by ``from_jaxpr``
    from the target's output avals) rather than a flag. Every trainable
    example is model + loss, so ``auto`` is ``grad_cosine`` for all of them;
    it falls back to the legacy ``jac_cosine`` only for the analytic AD
    benchmarks (Helmholtz / RoeFlux / Lighthouse / RobotArm / BlackScholes /
    Simple), whose target is a full Jacobian and which have no loss and no
    data generator. ``loss_drop`` (the 200-step Adam walk) stays selectable
    BY NAME and is never the default: finding 51 shows a plan scoring 0.885
    on loss_drop while its gradient points elsewhere. Read identically by the
    trainer and by every CpuApproximationActor — both run THIS function inside
    THIS module's ``_callback``, so the two paths cannot disagree.
    """
    want = os.environ.get(_QUALITY_METRIC_ENV, "auto").strip().lower()
    if want in ("loss_drop", "lossdrop", "walk"):
        # NAME RESOLUTION ONLY. Whether `loss_drop` is DEFINED for the running
        # target is a different question and is answered in
        # `_loss_drop_quality`, which is the thing that needs a scalar loss;
        # `tests/quality_metric_names_test.py` pins this function as a pure
        # name -> metric map for BOTH target kinds, and conflating the two
        # would break that contract to say something it does not claim.
        return "loss_drop"
    if want in ("grad_cosine", "gradcos", "grad_cos"):
        return "grad_cosine"
    if want in ("jac_cosine", "jacobian_cosine", "jaccos"):
        return "jac_cosine"
    # DEPRECATED NAME (2026-08-28). "cosine" used to mean the JACOBIAN cosine.
    # Post-744fc3d the traced target of every trainable example IS a scalar
    # loss, so the leaves jacve returns are already gradient-shaped and the
    # "Jacobian cosine" label had stopped being true; and the owner's own
    # 2026-08-07 sweep measured the GRADIENT cosine as the better predictor of
    # downstream accuracy (0.737 vs 0.610 Pearson). So "cosine" now resolves to
    # the gradient formulation wherever one is defined, and to the legacy
    # Jacobian cosine only for the analytic AD benchmarks, which have neither a
    # loss nor a data generator. Say so once, loudly -- this MOVES the number
    # every "cosine" run reports.
    if want in ("cosine", "cos", "cosine_sim"):
        _warn_cosine_is_now_grad_cosine()
        return ("grad_cosine"
                if bool(getattr(config, "scalar_target", False))
                else "jac_cosine")
    # "none" -- NO QUALITY CHANNEL AT ALL: neither the exact executable nor the
    # 200-step loss-drop walk is built, and reward slot 6 stays 0.0.
    #
    # For an ORDER-ONLY / exact arm the channel is a measured CONSTANT (TLM
    # seq32/dm128/vocab1024, --no-approx-head: 0.8853 on every plan of every
    # arm) because the plan cannot approximate anything -- the gradient it
    # returns is the exact gradient whatever the elimination order, so the walk
    # re-derives the same loss drop every time. It therefore contributes
    # exactly zero gradient while costing the single largest share of the
    # measurement budget (200 extra executions of the plan + ~1 s of host
    # overhead, vs 100 executions for the whole latency channel).
    #
    # The COST channels are untouched by this: with no quality sample
    # ``cosines`` stays empty and nothing downstream reads it into the cost
    # slots -- latency_ns and peak_memory are bit-identical to a run that
    # computed the walk and threw the number away.
    if want in ("none", "off", "skip"):
        return "none"
    if want not in ("auto", ""):
        raise ValueError(
            f"{_QUALITY_METRIC_ENV} must be one of "
            f"auto/loss_drop/grad_cosine/jac_cosine/cosine/none, got {want!r}"
        )
    return ("grad_cosine" if bool(getattr(config, "scalar_target", False))
            else "jac_cosine")


# Walk hyper-parameters. The defaults ARE the measured configuration above;
# changing them invalidates the correlation numbers, so they are env-tunable
# but never silently different between the trainer and the measure actors
# (both read this module in the same process tree / the same sbatch env).
def _walk_steps() -> int:
    return int(os.environ.get("ALPHAGRAD_WALK_STEPS", "200"))


def _walk_lr() -> float:
    return float(os.environ.get("ALPHAGRAD_WALK_LR", "1e-3"))


def _walk_probe_seed() -> int:
    """BASE seed of the PROBE BATCH.

    Scores are only comparable across plans measured on the SAME batch. Until
    2026-08-28 that requirement was implemented as "one batch for the entire
    process, for the entire run": the cache key carried no episode term and the
    dict was cleared on every miss. Two consequences, both measured:

    * the channel was bit-deterministic (48 readings of one plan, sd exactly
      0), so a plan's score carried no sampling variance at all; and
    * what it measured was single-batch OVERFITTING, which is why a plan that
      SKIPS one face -- freezing the gradient of most parameters -- scored
      0.9258 against an exact plan's 0.9260, and why the gap SHRANK 50-200x at
      longer walks.

    Comparability is only needed WITHIN one comparison set, i.e. within an
    episode. ``ALPHAGRAD_WALK_ROTATE`` therefore folds the episode index into
    this seed (see :func:`_walk_seed`), and ``ALPHAGRAD_WALK_HELDOUT`` draws a
    SECOND batch the walk never trains on and scores there instead.
    """
    return int(os.environ.get("ALPHAGRAD_WALK_PROBE_SEED", "20260807"))


# ---- (A3) held-out scoring + per-episode rotation -------------------------
#
# TWO INDEPENDENT SWITCHES, BOTH DEFAULT OFF, so a run launched without them
# reproduces the pre-A3 number bit-for-bit (pinned by
# ``tests/walk_heldout_test.py``: the default probe batch is asserted equal to
# ``data_gen(split(PRNGKey(20260807), 5))`` byte for byte, and a published
# episode is asserted to change neither the seed nor the score):
#
#   ALPHAGRAD_WALK_HELDOUT=1  the walk TRAINS on batch A and both endpoints of
#                             the loss drop are SCORED on a held-out batch B
#                             that the walk never saw.
#   ALPHAGRAD_WALK_ROTATE=1   A and B are re-drawn every EPISODE.
#
# WHY THE EPISODE TERM TRAVELS AS AN ENVIRONMENT VARIABLE. The trainer and each
# Ray measure actor are SEPARATE PROCESSES with separate ``_PROBE_BATCH``
# dicts, so the rotation index must be readable by both without new plumbing --
# the same "one env var, read by one module, by both sides" discipline the rest
# of the measurement configuration uses (``ppo.configure_fidelity``,
# ``masks.set_diag_per_face``).
#
# CAVEAT, RAISED BY A3 AND CLOSED BY A2: Ray workers inherit the driver's
# environment as it stood at ``ray.init`` time, so a value republished mid-run
# reaches the TRAINER -- which is where every TERMINAL row, the only rows
# quality is computed on, is measured under ``--exec-on-gpu`` -- but NOT a
# long-lived measure actor, which would keep whatever episode was current when
# it was spawned. That hole is now plugged at the WIRE rather than worked
# around: ``CpuApproxPool.evaluate`` / ``evaluate_batch`` carry an ``episode=``
# field, the two call sites in this file fill it from ``walk_episode()``
# whenever rotation is on, and ``CpuApproximationServer.evaluate`` republishes
# it into the ACTOR's own environment before measuring -- so a pooled
# measurement rotates in step with the trainer instead of pinning its spawn
# episode. ``_loss_drop_quality``'s explicit ``episode=`` argument remains for
# direct callers, and the fingerprint line still prints the episode and the
# batch digests actually used, so any residual divergence shows up in the log
# instead of silently de-pairing the comparison.
_WALK_EPISODE_ENV = "ALPHAGRAD_WALK_EPISODE"


def set_walk_episode(episode: int) -> None:
    """Publish the CURRENT episode index for the probe-batch rotation.

    Called once per episode by the trainer; a no-op for the score itself
    unless ``ALPHAGRAD_WALK_ROTATE=1``.
    """
    os.environ[_WALK_EPISODE_ENV] = str(int(episode))


def walk_episode() -> int:
    """The published episode index, 0 when nobody published one."""
    try:
        return int(os.environ.get(_WALK_EPISODE_ENV, "0"))
    except ValueError:
        return 0


def walk_rotate_enabled() -> bool:
    return os.environ.get("ALPHAGRAD_WALK_ROTATE", "0") not in (
        "0", "", "false", "False", "no")


def walk_heldout_enabled() -> bool:
    return os.environ.get("ALPHAGRAD_WALK_HELDOUT", "0") not in (
        "0", "", "false", "False", "no")


# Stride between consecutive episodes' probe seeds, and the offset from the
# TRAIN batch's seed to the HELD-OUT batch's seed. Large coprime constants, so
# no two (episode, role) pairs can land on the same PRNGKey and no episode's
# held-out batch can be another episode's training batch.
_WALK_EPISODE_STRIDE = 1000003
_WALK_HELDOUT_OFFSET = 7919


def _walk_seed(role: str = "train", episode: int | None = None) -> int:
    """PRNGKey seed of the ``role`` (``"train"`` / ``"eval"``) probe batch.

    With both switches off this returns exactly ``_walk_probe_seed()`` for the
    only role that is ever requested (``"train"``) -- that identity is what
    makes flag-off bit-identical, and it is asserted by the test.
    """
    base = _walk_probe_seed()
    if walk_rotate_enabled():
        ep = walk_episode() if episode is None else int(episode)
        base += _WALK_EPISODE_STRIDE * int(ep)
    if role == "eval":
        base += _WALK_HELDOUT_OFFSET
    return int(base)


def _walk_noise_std() -> float:
    """OPTIONAL, DEFAULT OFF. Resampling N(0, 0.3) pixel noise on the walk
    batch each step lifts Pearson 0.922 -> 0.950, but it changes the measured
    configuration, so the headline numbers stay reproducible only at 0.0."""
    return float(os.environ.get("ALPHAGRAD_WALK_NOISE_STD", "0.0"))


# One probe batch per (process, data-generator, shape, ROLE, EPISODE) — built
# once per key, kept on the host, device_put per measurement onto whichever
# device the plan was compiled for.
#
# BOUNDED rather than cleared-on-miss: one episode needs at most two entries
# (train + held-out) and the previous episode's pair is worth keeping while the
# pool drains. With both A3 switches off exactly ONE key is ever produced, so
# the dict never reaches the cap and the behaviour is the old "one batch per
# process, by construction". 512 MNIST images is ~1.6 MB, so four is free.
_PROBE_BATCH: dict = {}
_PROBE_BATCH_MAX = 4

# WHAT THE LAST PROBE BATCH WAS DRAWN AT, for the plan record. A generator
# that declares a ``meta`` callable answers "which draw is this?" -- on the
# one-step recurrent SHD target that is the STEP POSITION, and a record that
# does not say which step it measured cannot be read back against another
# (owner ruling 2026-09-16). Empty for every generator that declares none.
_PROBE_META: dict = {}

# Strides folded into the probe seed when a generator asks to be redrawn per
# (environment, episode). Large and coprime with `_WALK_EPISODE_STRIDE`, so no
# two (env row, episode) pairs can land on one PRNGKey.
_PROBE_ENV_STRIDE = 15485863
_PROBE_EPISODE_STRIDE = 32452843


def probe_meta() -> dict:
    """What the last probe batch of this process was drawn at."""
    return dict(_PROBE_META)


def _data_slots(config, data) -> tuple:
    """WHICH ARGUMENT SLOTS a probe batch fills.

    The generator's own statement (``data_slots``), or the contiguous
    ``0 .. len(data) - 1`` when it makes none -- which is what every image and
    token generator fills and is byte-identical to the positional assumption
    this replaces. A generator whose draw is a carried state and a block of
    given values further along the tuple cannot be read positionally.
    """
    slots = getattr(config.data_gen, "data_slots", None)
    if slots is None:
        return tuple(range(len(data)))
    slots = tuple(int(i) for i in slots)
    if len(slots) != len(data):
        raise ValueError(
            f"the data generator declares {len(slots)} slots {slots} and "
            f"returned {len(data)} arrays. A generator's `data_slots` is the "
            f"contract every refresher reads; a mismatch would put one of its "
            f"arrays in the wrong argument.")
    return slots


def _probe_seed(config, role: str = "train", episode: int | None = None,
                index: int = 0) -> int:
    """THE PRNG SEED OF ONE PROBE BATCH, and the only definition of it.

    Two places need this number: :func:`_probe_batch`, which keys the batch
    itself on it, and :func:`_grad_cosine_quality`, which keys the cosine's
    EXACT REFERENCE on it. They used to compute it separately and the second
    copy was short by the two terms below, so on a generator that redraws per
    environment and per episode every row after the first was scored against
    the FIRST row's exact gradient -- a gradient at another step of the
    recording, which is an unrelated vector. Job 66642 read a quality median of
    +0.0197 over [-0.2417, +1] on plans that were all exact; probe 66655
    reproduced it on that job's own arguments, four rows in one process:
    +1.000, -0.240, -0.174, -0.221 (dsnn-dfw.52).

    On every generator that does NOT declare ``resample_per_env_episode`` this
    is exactly ``_walk_seed(role, episode) + 104729 * index``, which is what
    both call sites computed before, so nothing else moves by one bit.
    """
    seed = _walk_seed(role, episode) + 104729 * int(index)
    if getattr(config.data_gen, "resample_per_env_episode", False):
        _ep = walk_episode() if episode is None else int(episode)
        seed += (_PROBE_EPISODE_STRIDE * int(_ep)
                 + _PROBE_ENV_STRIDE * (current_env_slot() + 1))
    return int(seed)


def _probe_batch(config, base_args, role: str = "train",
                 episode: int | None = None, index: int = 0, draw=None):
    """The probe batch: real data from ``config.data_gen`` at ``_walk_seed``.

    ``role`` is ``"train"`` (the batch the Adam walk steps on) or ``"eval"``
    (the held-out batch it is scored on); ``episode`` overrides the published
    rotation index. With both A3 switches off there is exactly one role, one
    seed and one entry -- the pre-A3 behaviour.

    SYNTHETIC DATA IS NOT AN OPTION for the loss-drop metric. Walking on noise
    and probing real MNIST gives Spearman 0.08 (gaussian) / 0.13 (uniform) and
    catches 2 of 7 degenerate plans, vs 6 of 7 with real images; self-probing
    on noise scores a healthy-looking 0.72-0.78 and is MISLEADING, because
    random labels are memorisable by any descending gradient. 8 tiled images
    give Pearson 0.261, one image 0.015. 60000 -> 512 distinct images costs
    +0.003, which is why 512 (one resident batch, no data pipeline) is enough.

    Returns ``None`` when the example has no data generator, which is the
    signal to fall back to the legacy cosine channel rather than to invent
    data.
    """
    if config.data_gen is None:
        return None
    # ``draw`` is an ALTERNATIVE producer for the same (role, episode, index)
    # SEED -- the generator's own `reference_draw`, which answers the same
    # step position in the exact container. Same seed, different tuple, its
    # own cache entry.
    draw = config.data_gen if draw is None else draw
    # THE KEY CARRIES THE EPISODE. ``_walk_seed`` already folds (role,
    # episode) into the seed, and the seed is in the key, so a new episode
    # cannot be served a stale batch -- which is exactly the bug that made the
    # pre-A3 cache serve one batch for the whole run.
    # ``index`` draws a DIFFERENT batch for each K of the gradient
    # cosine. index=0 is bit-identical to the pre-index behaviour, so
    # the loss-drop walk's batch does not move.
    # PER ENVIRONMENT AND PER EPISODE, when the generator asks for it (owner
    # ruling 2026-09-16). `_walk_seed` rotates per episode only behind
    # ALPHAGRAD_WALK_ROTATE, and never per environment, so a generator whose
    # DRAW is the quantity under study -- the recurrent SHD step position --
    # says so with `resample_per_env_episode`. `_probe_seed` folds both in,
    # and it is the ONE place that does: the cosine's reference is keyed on
    # the same number.
    _seed = _probe_seed(config, role, episode, index)
    _key = (id(config.data_gen), id(draw), _seed,
            tuple(getattr(a, "shape", ()) for a in base_args[:2]))
    hit = _PROBE_BATCH.get(_key)
    if hit is not None:
        _PROBE_META.clear()
        _PROBE_META.update(hit[1])
        return hit[0]
    k = jrand.PRNGKey(_seed)
    keys = jrand.split(k, 5)
    data = draw(keys)
    data = tuple(jax.device_get(d) for d in data)
    _meta_fn = getattr(config.data_gen, "meta", None)
    meta = dict(_meta_fn(keys)) if _meta_fn is not None else {}
    if len(_PROBE_BATCH) >= _PROBE_BATCH_MAX:
        _PROBE_BATCH.clear()
    _PROBE_BATCH[_key] = (data, meta)
    _PROBE_META.clear()
    _PROBE_META.update(meta)
    return data


def _walk_argnums(config, base_args) -> tuple[int, ...]:
    """The differentiated slots the walk is allowed to UPDATE.

    0-d argnums are excluded: under ``--seed-vertices`` the last differentiated
    argument is the tangent seed ``t``, whose gradient is a directional
    derivative and not a weight update — stepping it moves every weight by
    ``t*ones`` and saturates the net (the same reason
    ``generate_eval_samples`` leaves 0-d argnums at their injected value)."""
    return tuple(
        i for i in config.argnums
        if i < len(base_args) and getattr(base_args[i], "ndim", 0) > 0
    )


def _sanitise_grad(g, like):
    """Plan gradients arrive with two documented irregularities: some weight
    leaves come back TRANSPOSED for certain elimination orders (the reason
    ``_align_jac`` exists), and quant/compress plans can produce complex or
    non-finite leaves. Fix the layout, take the real part, and replace
    non-finite entries with 0 BEFORE the Adam step — a NaN that reaches the
    optimiser poisons every subsequent step and the walk would report a
    non-finite loss for a plan that is merely bad on a few entries."""
    def _fix(a, e):
        if getattr(a, "shape", None) != getattr(e, "shape", None):
            if getattr(a, "ndim", 0) == 2 and a.shape == e.shape[::-1]:
                a = a.T
            else:
                return None
        a = jnp.real(a) if jnp.iscomplexobj(a) else a
        return jnp.nan_to_num(a.astype(e.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    return [_fix(a, e) for a, e in zip(g, like)]


@partial(jax.jit, static_argnums=())
def _adam_step(w, g, m, v, t, lr, b1, b2, eps):
    m = [b1 * mi + (1.0 - b1) * gi for mi, gi in zip(m, g)]
    v = [b2 * vi + (1.0 - b2) * gi * gi for vi, gi in zip(v, g)]
    mh = [mi / (1.0 - b1 ** t) for mi in m]
    vh = [vi / (1.0 - b2 ** t) for vi in v]
    w = [wi - lr * mhi / (jnp.sqrt(vhi) + eps)
         for wi, mhi, vhi in zip(w, mh, vh)]
    return w, m, v



# THE QUALITY COSINE'S EXACT REFERENCE (owner ruling 2026-09-18). The exact
# gradient does not depend on the elimination order -- two exact orders agree
# to 1e-6 in float32 (reports-2026-09-16/agent-oracle-report.md) -- so the
# cosine reads the REV-EXACT reference the paired cost channel already
# compiles and caches, executed ONCE PER PROBE BATCH instead of once per plan.
# Before this every plan compiled its own same-order exact program for the
# cosine alone: two compiles per plan under a free order, where the reference
# is one executable for the whole process.
_COSINE_REF: dict = {}
# ONE ENTRY PER (environment row, episode, K). A generator that redraws per
# environment gives 16 rows x K batches distinct keys in one process, and a cap
# of 8 made every row after the eighth a miss that evicted the whole table. The
# entry is one gradient (about 1 MB on RSNN_SHD), so a cap that holds an
# episode's rows costs a few tens of megabytes and saves one reference
# execution per plan.
_COSINE_REF_MAX = 128
_COSINE_REF_STATS = {"hits": 0, "misses": 0}


def _cosine_reference(ref_ex, ref_key, args, device, probe_seed):
    """The reference gradient on ONE probe batch, computed once and reused.

    THE KEY is ``(the reference executable's compile key, device, PROBE SEED,
    arg shapes and dtypes)``. The compile key already carries the reverse
    order, the sparse flag, the argument shapes and the device, so two
    different reference executables cannot share an entry. The probe seed is
    the number ``_probe_batch`` keys its own cache on, so a new probe batch --
    a new episode under ``--walk-rotate``, a new K -- is a miss and the entry
    dies with the batch. The target's weights are not in the key, for the
    reason ``_grad_oracle_reference`` gives: they are drawn once per process
    and the policy searches elimination orders, not weights.
    """
    key = (bytes(ref_key), str(device), int(probe_seed),
           tuple((tuple(getattr(x, "shape", ())), str(getattr(x, "dtype", "")))
                 for x in args))
    hit = _COSINE_REF.get(key)
    if hit is not None:
        _COSINE_REF_STATS["hits"] += 1
        return hit[1]
    _COSINE_REF_STATS["misses"] += 1
    out = ref_ex(*args)
    if len(_COSINE_REF) >= _COSINE_REF_MAX:
        _COSINE_REF.clear()
    _COSINE_REF[key] = (ref_ex, out)
    return out


# THE CONFIGURATION HAS NO QUALITY CHANNEL AT ALL: no rev-exact reference and
# no data generator, so no probe batch exists and no plan of this run can ever
# be scored. This is NOT the refusable case. `None` means the apparatus DID
# draw a probe batch and the exact gradient on it was identically zero, which
# is one plan's missing datum (dsnn-dfw.51). Refusing on the structural case
# refuses EVERY terminal of the run, which leaves the episode with no
# measurement at all and turns the analytic AD benchmarks -- Helmholtz,
# RoeFlux, Lighthouse, RobotArm, BlackScholes, Simple, and every toy env in
# the test suite -- into a run that measures nothing.
_QUALITY_NO_CHANNEL = object()


def _grad_cosine_quality(config, compiled_approx, ref_ex, ref_key, base_args,
                         device, k_batches: int = 1):
    """THE GRADIENT COSINE: cos(g_approx, g_exact) at the INITIAL weights,
    averaged over ``k_batches`` fixed probe batches of REAL data.

    ``ref_ex`` is the REV-EXACT reference of the paired cost channel, keyed by
    ``ref_key``; its gradient is the ``g_exact`` of the cosine and it is
    executed once per probe batch, not once per plan (see ``_cosine_reference``
    and the owner's ruling of 2026-09-18).

    Returns ``(quality, rel_frobs, cosines)``; ``None`` when the probe batch
    was drawn and the exact gradient on it is identically zero, which is a
    REFUSED measurement; and ``_QUALITY_NO_CHANNEL`` when this configuration
    has no channel to measure at all.

    This differs from the legacy Jacobian cosine in WHERE it is evaluated, not
    only in WHAT is compared: the legacy channel scored at the calibration eval
    samples, which are synthetic argument draws, while this scores on the same
    real-data probe batches the loss-drop walk uses.  On a scalar-loss target
    the compared leaves are gradient-shaped either way.
    """
    if ref_ex is None:
        return _QUALITY_NO_CHANNEL
    cos_all: list[float] = []
    frob_all: list[float] = []
    _degenerate = 0
    for k in range(max(1, int(k_batches))):
        data = _probe_batch(config, base_args, role="train", index=k)
        if data is None:
            return _QUALITY_NO_CHANNEL
        a = list(base_args)
        # WHICH SLOTS THE BATCH FILLS is the generator's statement
        # (`data_slots`), not this function's guess. It used to be
        # `range(min(2, len(data)))` -- true of every image / token generator,
        # and identical to the line below for them, but false for a generator
        # whose draw is a carried state and a block of given values further
        # along the tuple.
        _slots = _data_slots(config, data)
        for slot, d in zip(_slots, data):
            a[slot] = jax.device_put(jnp.asarray(d), device)
        # THE REFERENCE, when the DRAW ITSELF is approximated. The in-band
        # cosine scores the plan against the rev-exact plan ON THE SAME
        # ARGUMENTS, so an approximation that lives in an ARGUMENT -- the
        # temporal carry a rule accumulated over a whole recording -- is
        # invisible to it: both sides read the same approximated value and the
        # cosine reads 1.0 whatever the rule did. A generator that draws an
        # approximated value publishes `reference_draw`, the EXACT draw at the
        # SAME step position, and the reference comes from `jax.grad` of the
        # target on that instead. Then reward slot 6 holds the error the rule
        # ACCUMULATED over the recording (owner ruling 2026-09-16).
        # A SKIP ON THE CARRIED FACE CHANGES THE TARGET, NOT ONLY THE DRAW.
        # Skip means NO CARRY, so the measured program is the truncated one,
        # and its own exact gradient is the truncated gradient -- which would
        # score 1.0 and hide that the plan threw the whole prefix away. The
        # reference has to come from the rule the ARM runs, on the arm's own
        # target and argument tuple. `reference_oracle` is how a generator
        # says that, and `reference_draw` stays the same statement for the
        # ordinary case where only the draw is approximated.
        _ref_oracle = getattr(config.data_gen, "reference_oracle", None)
        _ref_draw = getattr(config.data_gen, "reference_draw", None)
        if _ref_oracle is not None:
            _ref_draw = _ref_oracle["draw"]
        _oracle_ref = (_ref_draw is not None and config.target_fun is not None
                       and bool(getattr(config, "scalar_target", False)))
        # THE SAME SEED `_probe_batch` DREW THIS BATCH AT (dsnn-dfw.52). The
        # reference is cached on it, and a seed that does not carry the
        # environment row serves row 0's exact gradient to every other row.
        _seed = _probe_seed(config, "train", None, k)
        try:
            out_a = compiled_approx(*a)
            if _oracle_ref:
                r_data = _probe_batch(config, base_args, role="train",
                                      index=k, draw=_ref_draw)
                if _ref_oracle is None:
                    ar = list(base_args)
                    _r_slots = _data_slots(config, r_data)
                    _r_target = config.target_fun
                    _r_argnums = config.argnums
                else:
                    ar = list(_ref_oracle["args"])
                    _r_slots = tuple(_ref_oracle["slots"])
                    _r_target = _ref_oracle["target"]
                    _r_argnums = tuple(_ref_oracle["argnums"])
                for slot, d in zip(_r_slots, r_data):
                    ar[slot] = jax.device_put(jnp.asarray(d), device)
                jac_e = jax.grad(_r_target, argnums=_r_argnums,
                                 has_aux=config.has_aux)(*ar)
                jac_e = jac_e[0] if config.has_aux else jac_e
                out_e = None
            else:
                # The CACHED REV-EXACT REFERENCE, not a same-order exact
                # program: one executable for the process, one execution per
                # probe batch (agent/ref16, owner ruling 2026-09-18).
                out_e = _cosine_reference(ref_ex, ref_key, a, device, _seed)
                jac_e = out_e[1] if config.has_aux else out_e
        except Exception:
            return None
        jac_a = out_a[1] if config.has_aux else out_a
        if _oracle_ref:
            cos, rel = _dense_cosine(jac_e, jac_a)
        else:
            cos, rel = _quality_metrics(jac_e, jac_a)
        cos = float(cos)
        if cos == 0.0 and not _oracle_ref:
            # A ZERO REFERENCE IS NOT A BAD PLAN. The cosine is undefined when
            # the EXACT gradient is identically zero, and the formula above
            # then returns 0.0 -- the worst possible score, handed to a plan
            # that reproduced the reference perfectly. It happens on a sparse
            # spiking target: at a step where no hidden unit fired and the
            # input frame is empty, every weight gradient is exactly zero.
            # Drop that batch instead of scoring it.
            _d, _ee, _aa, _rr, _tot = _gradient_similarity(
                jac_e, jac_e, "grad_cosine")
            if float(_ee) <= 0.0:
                _degenerate += 1
                out_a = out_e = jac_a = jac_e = None
                continue
        cos_all.append(cos)
        frob_all.append(float(rel))
        out_a = jac_a = jac_e = None
    if not cos_all:
        if _degenerate and not _ZERO_REFERENCE_WARNED:
            _ZERO_REFERENCE_WARNED.append(1)
            print(
                f"[measure] WARNING quality channel: the EXACT gradient is "
                f"identically zero on all {_degenerate} probe batch(es), so "
                f"the gradient cosine is undefined and the measurement is "
                f"REFUSED (missing data, counted, excluded from the update -- "
                f"never scored 0.0, dsnn-dfw.51). On a spiking target this "
                f"means the sampled steps fired nothing; widen the probe "
                f"(ALPHAGRAD_GRAD_COSINE_K) or raise the target's firing "
                f"rate.", flush=True)
        return None
    return float(np.mean(cos_all)), frob_all, cos_all

def _loss_drop_quality(config, compiled_approx, base_args, device=None,
                       episode: int | None = None):
    """Reward slot 6 under ``ALPHAGRAD_QUALITY_METRIC=loss_drop``.

    ``compiled_approx`` is the AOT-compiled DENSE executable of the plan under
    evaluation, so the gradient it returns carries the plan's elimination order
    AND its approximations. ``config.target_fun`` is the TRUE scalar loss (this
    metric is only selected when the measured graph is a scalar loss), so the
    probe is never contaminated by the approximation.

    Returns ``None`` when the walk cannot be defined (no data generator, no
    updatable weight slots, shape mismatch), which the caller treats as "fall
    back to the legacy cosine" rather than as a score.

    HELD-OUT SCORING (``ALPHAGRAD_WALK_HELDOUT=1``, default off). The walk
    trains on batch A; BOTH endpoints of the loss drop are then evaluated on a
    second, independent batch B that the walk never saw:

        quality = (loss(W0, B) - loss(W_T, B)) / |loss(W0, B)|

    WHY L0 IS MEASURED ON B, NOT ON A. Mixing endpoints -- ``L0`` on A and
    ``L1`` on B -- would make the numerator ``(train loss at W0) - (test loss
    at W_T)``, i.e. a loss drop CONTAMINATED by the train/test gap of the two
    different draws. At W0 the weights have seen neither batch, so ``E[L0_A] =
    E[L0_B]`` and that contamination has zero mean but non-zero variance: it
    adds noise and no signal. Scoring both endpoints on B is the textbook
    definition of "how much did training reduce the loss on data it never
    saw", it keeps the ratio's numerator and denominator on the same
    distribution, and it is the lower-variance estimator. So: BOTH ON B.
    """
    probe = _probe_batch(config, base_args, "train", episode)
    if probe is None or config.target_fun is None:
        return None
    wnums = _walk_argnums(config, base_args)
    if not wnums:
        return None

    heldout = walk_heldout_enabled()
    probe_ev = (_probe_batch(config, base_args, "eval", episode)
                if heldout else probe)
    if probe_ev is None:
        return None

    full = list(base_args)
    for i, d in enumerate(probe):
        if i < len(full):
            full[i] = jnp.asarray(d)
    if device is not None:
        full = [jax.device_put(a, device) for a in full]
    # The walk batch IS the probe batch: one batch resident on device, no data
    # pipeline (spec). 512 images at batch 512; batch 128 costs 0.02 Pearson
    # and batch 32 costs 0.08, and memory is flat in batch size anyway
    # (33.6 MB XLA temp + 2.5-4.1 MB args).
    x0 = full[0]

    # THE SCORING ARGS. Identical to ``full`` (same object) unless held-out
    # scoring is on, in which case the data slots carry batch B instead. Same
    # shapes and dtypes either way, so ``_jit_loss`` does not re-trace and the
    # extra cost is one ``data_gen`` draw per (episode, role) plus one
    # ``device_put`` per measurement.
    if heldout:
        full_ev = list(base_args)
        for i, d in enumerate(probe_ev):
            if i < len(full_ev):
                full_ev[i] = jnp.asarray(d)
        if device is not None:
            full_ev = [jax.device_put(a, device) for a in full_ev]
    else:
        full_ev = full

    loss_fn = _jit_loss(config)
    w = [full[i] for i in wnums]

    def _call_loss(weights):
        # Scored on ``full_ev`` (batch B under --walk-heldout); the walk's own
        # gradient steps read ``full`` (batch A) in ``_call_grad`` below.
        a = list(full_ev)
        for i, wi in zip(wnums, weights):
            a[i] = wi
        return loss_fn(*a)

    # The plan's output has ONE leaf per entry of ``config.argnums``, in that
    # order. Index by POSITION IN argnums rather than assuming the updatable
    # slots are a prefix, so the --seed-vertices layout (weights..., tangent
    # seed t) picks the weight leaves and drops d/dt no matter where it sits.
    _grad_pos = [list(config.argnums).index(i) for i in wnums]

    def _call_grad(weights, xb):
        a = list(full)
        a[0] = xb
        for i, wi in zip(wnums, weights):
            a[i] = wi
        out = compiled_approx(*a)
        out = out[1] if config.has_aux else out
        leaves = jax.tree_util.tree_leaves(out)
        if len(leaves) <= max(_grad_pos):
            return None
        return _sanitise_grad([leaves[p] for p in _grad_pos], weights)

    L0 = float(_call_loss(w))
    if not np.isfinite(L0) or abs(L0) < 1e-12:
        return None

    # ONE-TIME FINGERPRINT of the walk's starting point, per process.
    #
    # W0 is ``env.args`` — the initial weights, which ppo.py never refreshes
    # (only ``eval_args_samples`` is re-drawn per episode), so W0 and the probe
    # batch are both constants of the run and plan scores are comparable.
    # CAVEAT, deliberately logged rather than "fixed": ppo.py derives its arg
    # key with ``jrand.split(key)`` while cpu_approx_worker uses
    # ``jrand.split(key, 3)``, so an ACTOR's W0 differs from the TRAINER's. It
    # does not bite in the GPU campaigns because ``--exec-on-gpu`` keeps every
    # TERMINAL row (the only rows quality is computed on) in the trainer
    # process. Printing the fingerprint makes a future divergence visible
    # instead of silent — compare the line across processes.
    #
    # PROVENANCE (A3). Readings now VARY, so the log has to say which batch a
    # reading came from. The line fires once per (probe seed, held-out) pair --
    # exactly once for a flag-off run whatever episodes get published, matching
    # the old one-shot behaviour -- and carries the episode, both probe seeds,
    # and a separate digest for the train batch, the held-out batch and W0, so
    # a trainer/actor divergence (see the env-var caveat above) is visible by
    # comparing lines rather than by inference.
    _ep_used = walk_episode() if episode is None else int(episode)
    _seed_tr = _walk_seed("train", episode)
    # LATCHED ON THE SEED, not on the episode. With rotation OFF every episode
    # resolves to the same seed, so a 250-episode run prints exactly ONE line
    # -- the pre-A3 behaviour. With rotation ON the seed changes every episode,
    # so there is exactly one line per batch actually used.
    _fp_key = (_seed_tr, heldout)
    if _fp_key not in _WALK_FINGERPRINT and len(_WALK_FINGERPRINT) < 1024:
        _WALK_FINGERPRINT.append(_fp_key)
        import hashlib as _hl

        def _dig(arrs):
            _h = _hl.blake2b(digest_size=8)
            for _a in arrs:
                _h.update(np.asarray(jax.device_get(_a)).tobytes())
            return _h.hexdigest()

        _fp_tr = _dig((x0,) + tuple(full[1:2]))
        _fp_w0 = _dig(tuple(w))
        # Back-compatible composite: the pre-A3 line hashed probe+W0 in one
        # stream, and that combined digest is what old logs can be diffed
        # against, so keep emitting it under the same name.
        _fp_all = _dig((x0,) + tuple(full[1:2]) + tuple(w))
        print(
            f"[measure] loss-drop walk armed: probe batch "
            f"{tuple(np.asarray(x0).shape)} (seed {_seed_tr}), "
            f"{_walk_steps()} Adam steps @ lr {_walk_lr():g}, "
            f"noise std {_walk_noise_std():g}, L0={L0:.6g}, "
            f"fingerprint(probe+W0)={_fp_all}"
            f" | episode={_ep_used} rotate={int(walk_rotate_enabled())}"
            f" heldout={int(heldout)}"
            f" train_batch={_fp_tr} W0={_fp_w0}"
            + (f" eval_seed={_walk_seed('eval', episode)}"
               f" eval_batch={_dig((full_ev[0],) + tuple(full_ev[1:2]))}"
               if heldout else " eval_batch=<same as train>"),
            flush=True)

    T = _walk_steps()
    lr = _walk_lr()
    noise = _walk_noise_std()
    nkey = jrand.PRNGKey(_walk_seed("train", episode) + 1)
    m = [jnp.zeros_like(wi) for wi in w]
    v = [jnp.zeros_like(wi) for wi in w]
    for t in range(1, T + 1):
        xb = x0
        if noise > 0.0:
            nkey, sk = jrand.split(nkey)
            xb = x0 + noise * jrand.normal(sk, x0.shape, x0.dtype)
        g = _call_grad(w, xb)
        if g is None or any(gi is None for gi in g):
            return None
        w, m, v = _adam_step(w, g, m, v, float(t), lr, 0.9, 0.999, 1e-8)
    L1 = float(_call_loss(w))
    if not np.isfinite(L1):
        # The walk DIVERGED. That is the worst outcome a plan can have on this
        # channel, and it must not read as "no progress" (0.0) — otherwise a
        # gradient that blows the network up ties with a gradient that is
        # identically zero.
        return -1.0
    drop = (L0 - L1) / abs(L0)
    # Clamped to [-1, 1]. Upper: a full loss wipe-out is 1.0, matching the old
    # cosine's ceiling so PopArt's per-channel sigma floor
    # (ALPHAGRAD_POPART_SIGMA_MIN_QUALITY, 0.2) and the head warm start keep
    # the scale they were tuned for. Lower: an unbounded blow-up would let one
    # catastrophic plan own the channel's PopArt sigma and crush the signal
    # for every other plan in the episode.
    return float(np.clip(drop, -1.0, 1.0))


_LOSS_JIT: dict = {}


def _jit_loss(config):
    key = id(config.target_fun)
    fn = _LOSS_JIT.get(key)
    if fn is None:
        fn = jax.jit(config.target_fun)
        _LOSS_JIT.clear()
        _LOSS_JIT[key] = fn
    return fn


# Smallest latency reading we accept as a real measurement. Sub-100ns for a
# compiled grad executable is physically implausible (a fake-fast artifact of
# timer bypass / zero-work executables); clamp UP so a degenerate plan can't
# report an unbeatable latency. 0.0 (= "not measured") passes through.
_LAT_FLOOR_NS = 100.0


_MEASURE_TURN = itertools.count()


# --------------------------------------------------------------------------
# Batched host callback (ALPHAGRAD_BATCHED_CALLBACK=1). See module docstring
# in the patch that introduced this for the full rationale.
# --------------------------------------------------------------------------
_BATCHED_CALLBACK = os.environ.get("ALPHAGRAD_BATCHED_CALLBACK", "0") == "1"

# Set by ppo.py's --ray-measure actors. Marks this process as a DEDICATED
# measurement process: no trainer shares it, so every visible GPU is a
# measurement GPU and one is sufficient.
_MEASURE_ACTOR = os.environ.get("ALPHAGRAD_MEASURE_ACTOR", "0") == "1"
# One-shot latch so the static-estimate fallback warning is printed once per
# process instead of once per measurement.
_MEM_FALLBACK_WARNED: list = []
# One-shot warning flag for an undefinable loss-drop walk (see _callback).
_WALK_UNDEFINED_WARNED: list = []

# One-shot flag for the zero-exact-gradient case in `_grad_cosine_quality`.
_ZERO_REFERENCE_WARNED: list = []
# One-shot warning when the EXACT reference gradient cannot be built (see the
# fail-soft branch of the fidelity block in `_callback`).
_FID_REF_WARNED: list = []
# One-shot flag for the walk's starting-point fingerprint line.
_WALK_FINGERPRINT: list = []
# ...and a COUNT, because the latch alone means a run can silently change what
# ``peak_memory`` MEANS mid-flight: the runtime high-water mark and the static
# memory_analysis() estimate are different quantities, and the substitution is
# in place. Polled per period by the trainers.
_STATIC_PEAK_FALLBACKS: list = [0]


def _note_static_peak_fallback(reason: str) -> None:
    """Record (and announce once) that ``peak_memory`` is a STATIC estimate."""
    _STATIC_PEAK_FALLBACKS[0] += 1
    if not _MEM_FALLBACK_WARNED:
        _MEM_FALLBACK_WARNED.append(1)
        print("[measure] NOTE peak_memory is being SUBSTITUTED IN PLACE with "
              "the deterministic memory_analysis() estimate (arguments + "
              f"outputs + temps) because {reason}. That is a DIFFERENT "
              "quantity from the runtime peak_bytes_in_use high-water mark, "
              "so readings from before and after this line are not "
              "comparable. Expected on CPU backends, which do not expose "
              "allocator statistics.", flush=True)


def consume_static_peak_fallbacks() -> int:
    """Pop the per-period count of static-estimate substitutions."""
    n = _STATIC_PEAK_FALLBACKS[0]
    _STATIC_PEAK_FALLBACKS[0] = 0
    return n


# ---------------------------------------------------------------------------
# MEMORY PARITY (always on since ticket dsnn-3qm.49; the ALPHAGRAD_MEM_PARITY
# switch is gone -- the record is what the watermark is LOGGED through).
#
# Two different quantities have been called "the" memory cost of a plan:
#
#   STATIC   ``compiled.memory_analysis()`` temp + output bytes -- deterministic
#            (CV exactly 0 by construction) and what the elimrl/POMO stack
#            optimises (``mem_total_bytes``);
#   RUNTIME  the ``peak_bytes_in_use`` delta across one execution window --
#            what ``peak_memory`` holds here whenever the backend exposes
#            allocator statistics.
#
# They are NOT incomparable: on the same plan they track each other to roughly
# a constant factor. Recording BOTH for EVERY measurement -- together with
# WHICH one actually landed in the reward -- is the only way to verify that
# factor across many plans instead of one, and it turns the static fallback
# from a silent in-place substitution into an explicit field of the record.
# ---------------------------------------------------------------------------
_MEM_PARITY: list = []
# Measurements this process completed since the last drain, and records
# the ring refused past the cap. ``records + dropped == measured`` is the
# invariant `check_mem_parity_complete` asserts in every measuring process
# (ticket dsnn-3qm.49, folded from .32): a measured plan without a parity
# record means the watermark was never logged beside the channel.
_MEM_PARITY_MEASURED = [0]
_MEM_PARITY_DROPPED = [0]
_MEM_PARITY_CAP = 65536

# THE MEMORY CHANNEL (ticket dsnn-3qm.49, ruling .29). What reward slot 5
# (``peak_memory``, stored negated) HOLDS:
#   temp       the plan's own compiled executable's XLA static temp bytes
#              (``compiled.memory_analysis().temp_size_in_bytes``) -- the
#              default. Finding 41: on TLM the runtime watermark is
#              temp + 2.104 MB with R^2 = 1.000000, so temp carries the
#              same ranking with 8x the relative dynamic range and no
#              allocator quantum; finding 49: the watermark channel read
#              sigma = 0 over 16,192 wave-1 plans.
#   watermark  the runtime ``peak_bytes_in_use`` delta over the timed
#              window (the pre-.49 channel; on a backend without allocator
#              statistics the in-place static substitution of
#              ``_note_static_peak_fallback``). Kept for the flag-off
#              bit-identity gate (ALPHAGRAD_EQ_DUMP).
# Under both the OTHER quantity is recorded beside it by `_record_mem_parity`
# and drained through `consume_mem_parity`. Published by ppo.py from
# --mem-channel before ray.init, exactly like ALPHAGRAD_APPROX_ADD: the
# flag is the only user surface, the variable is the transport to the Ray
# measure actors, and this function is the one reader.
_MEM_CHANNEL_ENV = "ALPHAGRAD_MEM_CHANNEL"
MEM_CHANNEL_CHOICES = ("temp", "watermark")


def mem_channel() -> str:
    """``"temp"`` or ``"watermark"`` -- WHICH quantity reward slot 5 holds.

    Absent (a process not started through ppo.py) means the declared
    default ``temp``. Anything else is a hand edit, not a fallback.
    """
    want = os.environ.get(_MEM_CHANNEL_ENV, "temp").strip().lower()
    if want not in MEM_CHANNEL_CHOICES:
        raise ValueError(
            f"{_MEM_CHANNEL_ENV} must be one of {MEM_CHANNEL_CHOICES} (set "
            f"by ppo.py from --mem-channel), got {want!r}")
    return want


# THE COST FORM (ticket dsnn-3qm.9). HOW reward slots 2 (latency_ns) and 5
# (peak_memory) are expressed:
#   absolute    the measured number, stored negated -- the pre-.9 form, kept
#               for the flag-off bit-identity gate (ALPHAGRAD_EQ_DUMP).
#   paired-log  the log-difference against rev-exact measured PAIRED in the
#               same callback (see `_PAIRED_REF`): ``-(log cost(candidate) -
#               log cost(rev-exact))``, so rev-exact scores 0 and a cheaper
#               plan scores above 0. The campaign form (ppo.py --cost-form
#               defaults to it).
# Same transport as --mem-channel: ppo.py publishes the flag before ray.init,
# the Ray measure actors read it through this one function. ABSENT means
# ``absolute``, unlike `mem_channel`: the paired form costs a second compile
# and a second measurement per terminal plan and changes what two slots MEAN,
# so a process that was not started through ppo.py (a test module, the other
# drivers, landscape_map -- which takes its own ratios from absolute numbers)
# never gets it implicitly.
_COST_FORM_ENV = "ALPHAGRAD_COST_FORM"
COST_FORM_CHOICES = ("absolute", "paired-log")


def cost_form() -> str:
    """``"absolute"`` or ``"paired-log"`` -- HOW slots 2 and 5 are expressed.

    Absent means ``absolute`` (see the block above for why this default
    differs from `mem_channel`'s). Anything else is a hand edit.
    """
    want = os.environ.get(_COST_FORM_ENV, "absolute").strip().lower()
    if want not in COST_FORM_CHOICES:
        raise ValueError(
            f"{_COST_FORM_ENV} must be one of {COST_FORM_CHOICES} (set by "
            f"ppo.py from --cost-form), got {want!r}")
    return want


def check_mem_parity_complete(mp: dict, where: str) -> None:
    """Assert every measured plan left a parity record (records + dropped
    == measured). Called in the process that measured -- the actor under
    --ray-measure -- and again by the trainer on the merged totals."""
    _n = len(mp.get("records", ()))
    _d = int(mp.get("dropped", 0))
    _m = int(mp.get("measured", 0))
    if _n + _d != _m:
        raise MemChannelFault(
            f"memory parity {where}: {_m} plans were measured but "
            f"{_n} parity records (+{_d} dropped at the cap) exist -- the "
            f"runtime watermark was not logged beside the channel for "
            f"{_m - _n - _d} plan(s)")


def _record_mem_parity(compiled, runtime_peak, source: str,
                       is_terminal: bool) -> dict | None:
    """One (static, runtime, source) triple per measurement.

    Returns the record (the channel reads ``static_temp_bytes`` off it),
    or None when there is no executable to analyse.
    """
    if compiled is None:
        return None
    _t0 = time.perf_counter()
    try:
        ma = compiled.memory_analysis()
    except Exception:
        ma = None
    if ma is None:
        _st = _so = _sa = None
    else:
        _st = float(getattr(ma, "temp_size_in_bytes", 0) or 0.0)
        _so = float(getattr(ma, "output_size_in_bytes", 0) or 0.0)
        _sa = float(getattr(ma, "argument_size_in_bytes", 0) or 0.0)
    rec = {
        "static_temp_bytes": _st,
        "static_output_bytes": _so,
        "static_argument_bytes": _sa,
        "static_total_bytes": None if _st is None else _st + _so,
        "runtime_peak_bytes": (None if runtime_peak is None
                               else float(runtime_peak)),
        # WHICH of the two the reward slot took (see `mem_channel`).
        "channel": mem_channel(),
        # "runtime_delta"   the reward's peak_memory IS the measured delta;
        # "static_fallback" allocator stats were unavailable and the STATIC
        #                   estimate was substituted in place (a different
        #                   quantity -- this is the field that used to be a
        #                   one-shot printed warning and nothing else);
        # "bypassed"        ALPHAGRAD_BYPASS_RESOURCE_MONITOR=1, no reading.
        "peak_source": source,
        "terminal": bool(is_terminal),
    }
    if len(_MEM_PARITY) >= _MEM_PARITY_CAP:
        _MEM_PARITY_DROPPED[0] += 1
    else:
        _MEM_PARITY.append(rec)
    _prof_add("cb.mem_parity", time.perf_counter() - _t0)
    if os.environ.get("ALPHAGRAD_DEBUG_MEASURE", "0") == "1":
        _s, _r = rec["static_total_bytes"], rec["runtime_peak_bytes"]
        _f = f"{_r / _s:.3f}" if (_s and _r) else "n/a"
        print(f"[mem-parity] static(temp+out)={_s} runtime_peak={_r} "
              f"runtime/static={_f} source={source} "
              f"terminal={bool(is_terminal)}", flush=True)
    return rec


def consume_mem_parity() -> dict:
    """Pop the per-period memory-parity records (see ``_record_mem_parity``).

    ``{"records": [...], "measured": int, "dropped": int}`` -- the counts
    are what `check_mem_parity_complete` compares. Drained in the process
    that measured: it rides `consume_plan_records`, which the Ray measure
    actors already hand to the trainer (the ticket-07 trap: a counter the
    actor wrote is invisible in the trainer's own module globals).
    """
    out = {"records": list(_MEM_PARITY),
           "measured": int(_MEM_PARITY_MEASURED[0]),
           "dropped": int(_MEM_PARITY_DROPPED[0])}
    _MEM_PARITY.clear()
    _MEM_PARITY_MEASURED[0] = 0
    _MEM_PARITY_DROPPED[0] = 0
    return out


def mem_parity_summary(records) -> dict:
    """Per-period numbers for the log dict (ticket .45 names them): the mean
    and the worst (largest and smallest) ``watermark - temp`` in bytes over
    the records that carry both, the two means, and how many records took
    the static substitution instead of a runtime watermark."""
    _gaps = []
    _temps = []
    _marks = []
    _fallbacks = 0
    for _r in records:
        _t = _r.get("static_temp_bytes")
        _w = _r.get("runtime_peak_bytes")
        if _r.get("peak_source") == "static_fallback":
            _fallbacks += 1
        if _t is not None:
            _temps.append(float(_t))
        if _w is not None:
            _marks.append(float(_w))
        if _t is not None and _w is not None:
            _gaps.append(float(_w) - float(_t))
    _g = np.asarray(_gaps, dtype=np.float64)
    return {
        "n": int(len(records)),
        "n_paired": int(_g.size),
        "gap_mean_bytes": float(_g.mean()) if _g.size else float("nan"),
        "gap_max_bytes": float(_g.max()) if _g.size else float("nan"),
        "gap_min_bytes": float(_g.min()) if _g.size else float("nan"),
        "temp_mean_bytes": (float(np.mean(_temps)) if _temps
                            else float("nan")),
        "watermark_mean_bytes": (float(np.mean(_marks)) if _marks
                                 else float("nan")),
        "static_fallbacks": int(_fallbacks),
    }


def _cb_slot(x, i, E):
    """Row ``i`` of a possibly-unbatched pytree.

    Under ``vmap_method="expand_dims"`` a per-env argument has leading dim E
    and a closed-over constant has leading dim 1, so dispatch on that.
    """
    def _one(v):
        shp = getattr(v, "shape", None)
        if shp is None or len(shp) < 1:
            return v
        # Return a JAX array, not numpy: the count pass hands these straight
        # to graphax, which reads `.aval` off them. np.asarray() here made the
        # batched path diverge from the per-env one with an AttributeError.
        if shp[0] == E:
            return jnp.asarray(v[i])
        if shp[0] == 1:
            return jnp.asarray(v[0])
        return v
    return jax.tree_util.tree_map(_one, x)


def _batched_host(fn, n_out: int = 3):
    """Per-env host fn -> one that takes the whole batch and stacks results.

    The loop is SERIAL and in slot order, so this is equivalent to the per-env
    callback it replaces. It moves the call boundary; it does not change what
    happens inside it.

    ``n_out`` is the callback's OUTPUT ARITY and the caller must pass the
    env's (``VertexEliminationEnv.wire_arity``): 2 under ``delta_obs``
    (tokens, reward), 3 on the legacy full-stream path, which still carries an
    equation-id buffer.
    """
    def _wrapped(*a):
        # Batch width = largest leading dim among the arguments. The per-env
        # arguments (order / specs / step) carry it; closed-over constants
        # arrive as size-1 and are broadcast by _cb_slot.
        E = 1
        for leaf in jax.tree_util.tree_leaves(a):
            v = np.asarray(leaf)
            if v.ndim >= 1:
                E = max(E, int(v.shape[0]))
        if os.environ.get("ALPHAGRAD_BATCHED_DEBUG", "0") == "1":
            print("[batched] E=", E, "shapes=",
                  [tuple(np.asarray(l).shape)
                   for l in jax.tree_util.tree_leaves(a)][:12], flush=True)
        outs = [[] for _ in range(n_out)]
        for i in range(E):
            # The slot is the env row (see `_ENV_SLOT`). Cleared in a
            # `finally` so a raising plan cannot stamp the next one.
            _ENV_SLOT[0] = i
            try:
                r = fn(*[_cb_slot(x, i, E) for x in a])
            finally:
                _ENV_SLOT[0] = -1
            if len(r) != n_out:
                raise ValueError(
                    f"batched host callback returned {len(r)} arrays, "
                    f"expected {n_out} -- the env's wire arity and this "
                    f"shim's must be the same object (env.wire_arity).")
            for k in range(n_out):
                outs[k].append(np.asarray(r[k]))
        return tuple(np.stack(o, axis=0) for o in outs)
    return _wrapped


def _pool_owns_bound_operands(pool):
    """True iff the STEP callback can skip shipping args/consts/samples.

    Three conditions, all necessary: a pool is attached (so the remote
    closure, which ignores args/consts, is the one that runs), the pool
    already holds an eval-samples ObjectRef (so it will not fall back to the
    per-call tuple), and no row is served in-process
    (ALPHAGRAD_POOL_TERMINAL_LOCAL, which calls `_callback` directly and needs
    the real arrays). Fails closed on anything unexpected.

    MODULE-LEVEL ON PURPOSE. As a `-> bool` method on VertexEliminationEnv
    this came back as a traced `bool[]` under jit -- the class's annotated
    methods are wrapped -- and `x if flag else y` then raised
    TracerBoolConversionError. The flag has to be a plain Python bool: it
    selects which operands are TRACED, so it cannot be a traced value.
    """
    if os.environ.get("ALPHAGRAD_CB_OMIT_BOUND", "1") != "1":
        return False
    if pool is None:
        return False
    if os.environ.get("ALPHAGRAD_POOL_TERMINAL_LOCAL", "0") == "1":
        return False
    return getattr(pool, "_eval_samples_ref", None) is not None


def _env_callback(fn, shapes, *args, batched: bool = False):
    """Dispatch the env callback, batched or per-env.

    Batched needs `pure_callback` because `io_callback` has no `vmap_method`.
    """
    if batched and _BATCHED_CALLBACK:
        return jax.pure_callback(fn, shapes, *args,
                                 vmap_method="expand_dims")
    return io_callback(fn, shapes, *args)


def _next_measure_device(devices):
    """Round-robin over the measurement GPUs (everything but the trainer's).

    A plain global counter is correct here: the callback is invoked serially
    from the host inside `jax.pure_callback(vmap_method="sequential")`, so
    there is no concurrent reader to race with.
    """
    devices = list(devices)
    return devices[next(_MEASURE_TURN) % len(devices)]


def _aggregate_samples(values, want_top_quartile: bool):
    """Reduce a list of per-sample scalars to a single jnp scalar.

    The central tendency is the MEDIAN. The winsorized mean it replaces still
    averaged, so it still moved with every reading inside the interquartile
    band; the median moves only with the middle one, which is what "a robust
    summary of a noisy latency sample" actually means. Identical to the mean
    whenever the readings are constant, and (unlike winsorizing) it needs no
    quantile parameters and is correct at every sample count.

    ``want_top_quartile`` is kept as the call-site's "this channel is noisy,
    summarise it robustly" switch. Handles the empty-list case by returning
    ``0.0``.
    """
    if not values:
        return jnp.array(0.0, dtype=jnp.float32)
    stack = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in values])
    if want_top_quartile and stack.shape[0] >= 2:
        return jnp.median(stack)
    return stack.mean()


# Compile cache: the original in-process LRU thrashed (2-11% hit rate)
# because Ray's round-robin dispatch sent the same (order, specs) tuple
# to different actors. Sticky routing was tried next and lifted hit rate
# to ~15%, but the per-actor cache memory offset the savings — wall-time
# was ~20% faster, leak rate basically unchanged.
#
# The current strategy lives in ``alphagrad.approx.common.compile_cache``:
# a single Ray named-actor (``CompileCacheCoordinator``) owns a
# ``key -> ObjectRef`` table. Any actor that compiles serialises via
# ``jax.experimental.serialize_executable`` and ``ray.put``s the blob;
# subsequent calls (from ANY actor) fetch the blob and
# ``deserialize_and_load`` locally. Hit rate becomes cluster-wide
# rather than per-actor, multiplying effective coverage.


def decode_vertex_rule_specs(jaxpr, vertex, spec_rows) -> tuple:
    """Decode ONE vertex's wire-format ``(MAX_RULES_PER_VERTEX, 3)`` rows into
    graphax transforms ``(Diag | Compress | Quant, ...)``.

    THE single source of truth for the wire→transform translation: extracted
    from ``_callback`` so the mask-oracle replay applies EXACTLY the rules the
    measurement applies (``LiveVertexMaskOracle.advance`` demands "the
    transforms the env ACTUALLY applied"; a structural rules=() replay
    desyncs the masks after the first landed approximation).

    Row layout (see :class:`EnvState.sparsity_specs`):
      * ``row[0] == -1``               end-of-sequence sentinel.
      * ``row[0] == QUANT_SENTINEL``   QUANT, ``row[1]`` → QUANT_DTYPES.
      * ``row[0] == COMPRESS_SENTINEL`` COMPRESS, physical axis ``row[1]``,
        kind ``row[2]`` — honored at EVERY position of the plan.

        There used to be an ``is_last`` gate here that emitted COMPRESS only
        for the NEWEST vertex of the prefix, on the stated grounds that a
        ``val.ndim`` reduction upstream of a later elimination trips graphax's
        shape-preservation assertion. It is gone, because the premise is false
        and the cost was severe:

        * graphax's stored-edge shape check (``core._set_inner``, ungated
          since dsnn-3qm.71) reads the LOGICAL shape, and ``apply_compress``
          drops the axis POINTER (``Index.axis = None``), not the logical
          size, so a compressed edge still contracts and still passes it.
          The over-conservative structural guard the note referred to was
          removed from ``apply_compress`` on 2026-07-15.
        * MEASURED (2026-08-15, ``compress_probe``/``compress_probe2``, CPU):
          the same decoded COMPRESS applied at all 23 positions of the
          NeuralNetwork plan and at 11 sampled positions of the TransformerLM
          plan, RAW and ``make_live_masked_hook``-wrapped, raised NOTHING
          (0/46 exceptions) and genuinely changed the Jacobian (cos vs exact
          AD 0.55–0.99). So the gate was not preventing a failure, it was
          silently DISCARDING every COMPRESS the policy placed anywhere but
          the terminal vertex: the terminal measurement of a mid-plan COMPRESS
          came back bit-identical to exact AD (cos 1.000000) while the honest
          application scores cos 0.653 (NN) / 0.9957 (TLM).
        * It is also what made the append-only stream non-prefix-stable: the
          same vertex tokenized WITH its COMPRESS at prefix length k and
          WITHOUT it at k+1 (``tests/stream_prefix_property_test.py``), which
          forced the COMPRESS carve-outs in both prefix caches
          (v40: ext=1/431).
      * ``row[0] >= 0``                DIAG ``(bi1, bi2, factor)`` with the
        legacy -1 (joint gcd) factor sentinel; 0/1 factors are dropped.

    A rule must fit EVERY non-literal invar of the eqn (graphax applies each
    per-vertex transform to every incoming edge). Non-fitting / axis-reusing
    rows are silently skipped — same best-effort semantics the measurement
    has always had.
    """
    frame = vertex_frame(jaxpr, vertex)
    if frame is None:
        return ()
    return decode_rule_specs_in_frame(frame[0], frame[1], spec_rows)


def vertex_frame(jaxpr, vertex):
    """``(out_shape, primal_shapes)`` of the eliminated vertex's OWN equation.

    The frame the per-vertex wire rows are decoded in -- and, of a face's
    three slots, the frame of ``lhs`` only (d central / d in_edge). ``None``
    when the vertex has no decodable output or no non-literal input.
    """
    eqn = jaxpr.eqns[vertex - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return None
    out_shape = eqn.outvars[0].aval.shape
    primal_shapes = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    if not primal_shapes:
        return None  # no non-literal inputs → no edges to transform
    return out_shape, primal_shapes


def slot_frame(st):
    """``(out_shape, [primal_shape])`` of ONE live face-slot tensor, in
    LOGICAL sizes (``Index.logical_size``, the numbering ``Diag(i, j)``,
    ``dim_logical_sizes`` and ``rule_is_legal`` share). Ticket .18, D2: the
    frame a slot's wire row is decoded in is the tensor the slot is handed,
    not the vertex's equation."""
    return (tuple(int(d.logical_size) for d in st.out_dims),
            [tuple(int(d.logical_size) for d in st.primal_dims)])


def decode_rule_specs_in_frame(out_shape, primal_shapes, spec_rows) -> tuple:
    """The wire -> transform decode of :func:`decode_vertex_rule_specs` in an
    EXPLICIT frame: ``Diag.j = len(out_shape) + bi2`` and a Reduce axis must
    lie in ``out_shape ++ primal_shape`` of THIS frame. One decoder, two
    frames: :func:`decode_vertex_rule_specs` calls it on the vertex frame,
    :func:`make_slot_frame_hook` on each slot's own (:func:`slot_frame`)."""
    out_len = len(out_shape)

    rules: list = []  # mixed list[Diag | Compress | Quant]
    used_axes: set[int] = set()
    for slot in range(MAX_RULES_PER_VERTEX):
        row = spec_rows[slot]
        bi1 = int(row[0])
        bi2 = int(row[1])
        factor = int(row[2])
        if bi1 == -1:
            break  # end-of-sequence sentinel
        if bi1 == QUANT_SENTINEL:
            # QUANT: row[1] indexes QUANT_DTYPES; row[2] carries the NEGATE +
            # SCALE heads' choices as ``sign * (round(u * 1e6) + 1)``:
            #   0        → legacy (sign +1, data-dependent absmax auto-scale)
            #   ±(k+1)   → scale_sign = sign(row[2]), scale_frac = k / 1e6
            # (the +1 keeps u=0 distinguishable from the legacy 0; the sign
            # head's arm-select was previously TRAINED BUT DROPPED here —
            # row[2] was written as 0 — so the engine always quantized the
            # +1 arm regardless of the policy's pick.)
            dtype_idx = bi2
            if not (0 <= dtype_idx < len(QUANT_DTYPES)):
                dtype_idx = 0
            enc = factor
            if enc == 0:
                rules.append(Quant(dtype=QUANT_DTYPES[dtype_idx]))
            else:
                _sign = 1 if enc > 0 else -1
                _u = min(max((abs(enc) - 1) / 1e6, 0.0), 1.0)
                rules.append(Quant(dtype=QUANT_DTYPES[dtype_idx],
                                   scale_sign=_sign, scale_frac=_u))
            continue
        if bi1 == COMPRESS_SENTINEL:
            # COMPRESS: physical axis row[1], kind row[2]. Honored wherever it
            # sits in the plan — see the docstring for why the last-vertex
            # gate is gone.
            axis_idx = bi2
            kind_idx = factor  # row[2] reused as kind index for COMPRESS
            fits_all = True
            if axis_idx < 0:
                fits_all = False
            elif axis_idx < out_len:
                pass  # output-side axis, always present
            else:
                primal_pos = axis_idx - out_len
                fits_all = all(primal_pos < len(ps) for ps in primal_shapes)
            if not fits_all:
                continue
            if axis_idx in used_axes:
                continue
            if not (0 <= kind_idx < len(COMPRESS_KINDS)):
                # Unknown kind — fall back to "mean" rather than dropping the
                # row; the axis-removal effect is the dominant signal.
                kind_idx = 0
            used_axes.add(axis_idx)
            rules.append(Compress(axes=(axis_idx,), kind=COMPRESS_KINDS[kind_idx]))
            continue
        if bi1 < 0:
            # Reserved future sentinels: skip without aborting the sequence.
            continue
        idx1 = bi1            # logical output-side axis
        idx2 = out_len + bi2  # logical primal-side axis
        if idx1 in used_axes or idx2 in used_axes or idx1 == idx2:
            continue
        if not (0 <= bi1 < out_len):
            continue
        n1 = int(out_shape[bi1])
        # The rule must fit every primal edge of this vertex.
        n2_list: list[int] = []
        fits_all = True
        for ps in primal_shapes:
            if not (0 <= bi2 < len(ps)):
                fits_all = False
                break
            n2_list.append(int(ps[bi2]))
        if not fits_all:
            continue
        if factor == 0 or factor == 1:
            continue  # drop-axes and no-op have no equivalent in the new API
        if factor == -1:
            # Joint gcd across the out axis and every primal axis.
            from functools import reduce as _reduce
            factor = _reduce(_math.gcd, [n1] + n2_list)
        # apply_diag requires factor | gcd(n_i, n_j) on every edge it touches.
        if (
            factor <= 0
            or n1 % factor != 0
            or any(n2 % factor != 0 for n2 in n2_list)
        ):
            continue
        used_axes.add(idx1)
        used_axes.add(idx2)
        rules.append(Diag(i=idx1, j=idx2, factor=factor))
    return tuple(rules)


def slot_rules_for_row(st, one_row) -> tuple:
    """The rules :func:`make_slot_frame_hook` WOULD apply to ``st`` for the
    wire row ``one_row`` -- the decode in this tensor's own frame
    (:func:`slot_frame`) plus ticket .20's logical-dim -> physical-axis
    conversion for COMPRESS.

    Module-level and used by BOTH the hook and ``masks.slot_legality``, so the
    mask asks the row the same question the engine will: one decoder, one
    conversion, per tensor. The mask needs it per TENSOR because a slot's hook
    is installed at several graphax sites (:func:`face_slot_sites`) and the
    same row decodes differently in each site's frame.
    """
    from alphagrad.approx.common.masks import (
        compress_rules_to_physical, reduce_axes_physical)
    from alphagrad.approx.common.plan_log import kind_of_slot

    row = tuple(int(x) for x in one_row)
    spec_rows = [list(row)] + [[-1, -1, 0]] * (MAX_RULES_PER_VERTEX - 1)
    out_shape, primal_shapes = slot_frame(st)
    rules = decode_rule_specs_in_frame(out_shape, primal_shapes, spec_rows)
    physical = (kind_of_slot(row[0], COMPRESS_SENTINEL, QUANT_SENTINEL)
                == "compress" and reduce_axes_physical())
    if physical and rules:
        rules = compress_rules_to_physical(st, rules)
    return rules


def make_slot_frame_hook(one_row, *, stats: dict | None = None,
                         gated: bool = False):
    """ONE face slot's CHOOSER under ``--face-slot-frames slot`` (ticket .18,
    D2).

    Decodes the slot's wire row ``(b0, b1, b2)`` at APPLY time, in the frame
    of the live tensor graphax hands the slot (:func:`slot_frame`), then
    DECIDES through :func:`~alphagrad.approx.common.masks.make_live_masked_chooser`
    -- same legality, same projection, same ``applied`` / ``skipped`` counters.
    A row that names no dim of this slot (a Diag on a slot with no out side, a
    Reduce axis past the slot's rank) is counted ``skipped_<kind>`` like any
    other miss: ``requested`` is read off the wire, so leaving it uncounted
    would inflate ``applied_fraction``.

    IT RETURNS THE ACTION, NOT THE TENSOR, and that is what puts the decision
    in the token stream. graphax writes an ``approx`` block only from
    ``core._record_micro``, which runs for a literal micro-action or for a
    chooser that RETURNS one; a callable that returns a tensor is applied and
    recorded by nobody (graphax says so in ``_apply_face_transform``). This
    hook used to be that third kind, so every DIAG / COMPRESS / QUANT the face
    head placed was applied to the Jacobian and left no marker at all -- only
    SKIP, which is a wire bit graphax records itself, ever reached the stream
    (finding 64). Returning the action makes graphax the ONE apply site for a
    face rule, and the block it records is byte-identical to the one a literal
    action records.

    ``None`` means DECLINE (graphax returns the operand untouched): the row
    decoded to nothing in this frame, the mask refused it, or the wire slot is
    empty. A decline emits no block, which is right -- nothing was applied.

    The decode is memoised per frame, so the tokenizer's replay of the same
    hook on the same shapes decodes once. ``hook.rules_for(st)`` exposes the
    decoded rules for a tensor (tests, .59's apply-rate audit).
    ``hook.chosen_applied(action, applied)`` is graphax's outcome callback; it
    is what keeps ``applied`` and ``skipped_<kind>_noop`` reading the same
    post-apply identity test that decides whether a block is emitted at all.
    """
    from alphagrad.approx.common.masks import (
        face_counts_armed, make_live_masked_chooser, reduce_axes_physical,
        reduce_axis_spaces)
    from alphagrad.approx.common.plan_log import kind_of_slot

    row = tuple(int(x) for x in one_row)
    kind = kind_of_slot(row[0], COMPRESS_SENTINEL, QUANT_SENTINEL)
    cache: dict = {}
    # The inner chooser of the LAST decode, so graphax's outcome callback
    # reaches the object that holds this frame's counters. graphax calls it
    # immediately after the call that returned the action, on the same thread
    # and before any other slot runs.
    last: dict = {}

    def _decoded(st):
        out_shape, primal_shapes = slot_frame(st)
        # --reduce-axis-space physical (ticket .20): the wire token is a
        # LOGICAL dim of this slot's tensor; the rule handed to the hook
        # names the PHYSICAL axis that dim is stored in, so the memo key
        # carries the storage layout, not only the logical frame.
        physical = kind == "compress" and reduce_axes_physical()
        key = (out_shape, primal_shapes[0],
               reduce_axis_spaces(st).phys_of_dim if physical else None)
        hit = cache.get(key)
        if hit is None:
            rules = slot_rules_for_row(st, row)
            inner = (make_live_masked_chooser(rules, stats=stats, gated=gated)
                     if rules else None)
            hit = cache[key] = (rules, inner)
        return hit

    def _hook(st):
        _rules, inner = _decoded(st)
        last["inner"] = inner
        if inner is None:
            if (kind is not None and stats is not None
                    and (not gated or face_counts_armed())):
                stats["skipped"] = stats.get("skipped", 0) + 1
                stats[f"skipped_{kind}"] = stats.get(f"skipped_{kind}", 0) + 1
            return None
        return inner(st)

    def _chosen_applied(action, applied):
        inner = last.get("inner")
        if inner is not None:
            inner.chosen_applied(action, applied)

    _hook.rules_for = lambda st: _decoded(st)[0]
    _hook.chosen_applied = _chosen_applied
    return _hook


# #73 / ticket .56 -- THE ADD. A face accumulation multiplies lhs by rhs into
# `new` and, when the predecessor-to-successor edge ALREADY exists, adds `new`
# onto it. Two addends: the FRESH contraction and the OLD EDGE.
#
# The flag is ``--approx-add``, not ``--approx-old``: an approximation only
# ever lands on the CONTRACTION side (that is what the three face slots are),
# and what this flag actually governs is how the two addends of the ADD are
# made to meet. Naming it after the old edge said the old edge was the subject;
# it is the object.
#
#   lossy    -- force the two addends into ONE container, the one the
#               approximated `new` slot landed on. The old edge is projected
#               onto that support, so the add costs what the head's choice
#               costs. Replaces ``same``.
#   lossless -- keep the most information possible: the sum's support is the
#               UNION of the two addends' supports, so no non-zero of either
#               addend is dropped. Replaces ``exact`` (which was neither: it
#               put the `new` slot's rule on the POST-JOIN SUM, approximating
#               the merge instead of the contraction).
#   choose   -- the HEAD picks one of the two PER FACE, one bit, from its own
#               logit (``unified_face_head``'s ``layout.choose_index``). The bit
#               rides the wire as ``face_join[f]``: 0 = lossy, 1 = lossless
#               (``JOIN_LOSSY`` / ``JOIN_LOSSLESS``). There is NO legality mask
#               on it, because both arms are always formable -- see
#               ``face_entry_from_slots``.
#   learned1 -- the head gets a FOURTH slot, which approximates the OLD EDGE
#               itself (graphax ``res:jr``). NO choose bit: the model picks the
#               old edge's approximation directly, and that pick is what answers
#               the container question. The ADD then reconciles with the UNION.
#   learned2 -- a FIFTH slot on top of learned1's, approximating the ADD OUTPUT
#               (graphax ``res:jres``). Again no bit, again union at the merge;
#               the learned output approximation compresses the sum, which is
#               where the loss belongs and is in the plan's action record.
#
# EACH VALUE IS ITS OWN HEAD WIDTH -- 94, 94, 95, 125, 156 logits -- so a field
# a value does not use does not exist rather than being gated off. See the
# ``unified_face_head`` module docstring for the arithmetic and ``wire_slots``
# for the matching wire width.
#
# WHY ``same`` AND ``exact`` ARE GONE RATHER THAN ALIASED. Both old names map
# onto a DIFFERENT object than their replacement measures:
#   * ``same`` installed the SAME hook object at graphax's `res:new` AND `jr`,
#     so one wire row ran on two tensors with two different index structures
#     and one legality mask answered for one of them (finding 72, fault 1).
#     ``lossy`` installs the row at `res:new` ONLY and reconciles the addends
#     structurally, which is a different computation, not a rename.
#   * ``exact`` moved the row to `res:jres`, the post-join sum. ``lossless``
#     puts it back on the fresh contraction and leaves the add alone.
# A launcher that still names a retired value believes it chose a semantics
# that no longer exists, so both the CLI flag and the hand-off variable RAISE
# and name the replacement (the repo's error taxonomy).
#
# ppo.py publishes the flag as ALPHAGRAD_APPROX_ADD before ray.init -- one
# hand-off variable, one reader, the same shape as ALPHAGRAD_QUALITY_METRIC --
# so the trainer and every measure actor resolve the SAME configuration and
# the plan log records what actually ran. Read at call time, never at import.
_APPROX_ADD_ENV = "ALPHAGRAD_APPROX_ADD"
_APPROX_OLD_ENV = "ALPHAGRAD_APPROX_OLD"          # RETIRED, raises
#: THE WIRE'S SLOT BANDS (#73, rewidthed 2026-09-11). ``FACE_SLOTS`` keeps
#: meaning "the CONTRACTION slots" -- lhs, rhs, new -- so every existing
#: ``range(FACE_SLOTS)`` loop stays correct without being read. The JOIN slots
#: come after them, AND HOW MANY THERE ARE IS THE RUNNING VALUE'S ANSWER
#: (:func:`wire_slots`):
#:
#:     0..2  contraction: lhs, rhs, new          every value
#:     3     learned1, the OLD EDGE    (``jr``)   learned1, learned2
#:     4     learned2, the SUMMED EDGE (``jres``) learned2
#:
#: So the wire is 3 slots wide under ``lossy`` / ``lossless`` / ``choose``, 4
#: under ``learned1`` and 5 under ``learned2`` -- the same widths the head is
#: built at, from the same table
#: (``unified_face_head._LAYOUT_SPEC``). ONE source of truth, so a wire row
#: cannot describe a slot the head has no logits for.
#:
#: ROUTE CHOSEN, and why: the join rows ride the EXISTING ``face_specs`` array
#: widened from 3 slots, not a second array beside it. The wire's transport
#: is already built -- ``face_specs`` is carried wholesale through the env state,
#: the callback, the plan record, the rollout and the replay (32 sites in env.py
#: and about 25 in ppo.py) -- so widening changes shape-bearing CONSTRUCTIONS and
#: no transport, while a second array would have to duplicate all of it. It also
#: keeps the head's ``FaceFields`` un-split: splitting it at the policy boundary
#: and rejoining it in env is exactly where a row could be mis-slotted.

#: Every valid ``--approx-add``. The two learned values were added on
#: 2026-09-11 with the width ruling: they are not "lossy/lossless plus an extra
#: slot", they are their own widths, and under them the model's own pick
#: answers the container question instead of a choose bit.
APPROX_ADD_CHOICES = ("lossy", "lossless", "choose", "learned1", "learned2")

#: The two CONTAINER semantics a single merge can be reconciled under. This is
#: the range of :func:`resolve_join_mode` and of :func:`join_mode_of_bit`; it is
#: NOT the list of ``--approx-add`` values, which is why the two are separate
#: names now that ``learned1`` / ``learned2`` exist and both reconcile under the
#: union.
JOIN_SEMANTICS = ("lossy", "lossless")

#: ``--approx-add`` value -> the join semantics it FIXES, or ``None`` when the
#: value decides per face.
#:
#: ``learned1`` and ``learned2`` map to ``lossless`` -- THE UNION -- by owner
#: ruling 2026-09-11, and the reason is worth stating where a reader would ask
#: it: under those values both addends have ALREADY been shaped by the model's
#: own picks (slot 3 on the old edge, slot 4 on the sum), so compressing them
#: further at the merge would silently override a decision the model made, and
#: the union is exact. Under ``learned2`` the learned output approximation then
#: compresses the sum -- which is the model's decision, applied at the place the
#: loss belongs, and recorded in the plan's action log rather than happening
#: inside the add.
_JOIN_SEMANTICS_OF = {
    "lossy": "lossy",
    "lossless": "lossless",
    "choose": None,
    "learned1": "lossless",
    "learned2": "lossless",
}

#: The values whose join semantics is FIXED for the whole run. ``choose`` is
#: not one of them: it is decided per FACE by the head's bit.
APPROX_ADD_FIXED = tuple(k for k in APPROX_ADD_CHOICES
                         if _JOIN_SEMANTICS_OF[k] is not None)

#: The values ``ppo.py`` and ``landscape_map`` OFFER on the command line.
#:
#: ALL FIVE, since 2026-09-11 (ticket dsnn-3qm.56): the trainer wire that kept
#: the other three off this tuple now exists. What it was missing, and where it
#: came from:
#:
#: * ``choose`` needed a JOIN field on the action record, carried from the
#:   head's draw to the env's measurement. The record is declared once in
#:   ``alphagrad.approx.face_action`` and every use derives from it, so the bit
#:   reaches ``FaceAction.join`` -> the stored trajectory leaf -> the loss
#:   replay -> ``StepAction.face_join`` -> ``EnvState.face_joins`` ->
#:   :func:`_face_dict_for_vertex`. ``sample`` and ``evaluate`` score the same
#:   variable at width 95 (``tests/test_face_head94.py``).
#: * ``learned1`` / ``learned2`` needed the per-slot SHAPES to follow the head's
#:   width instead of ``FACE_SLOTS``. They do: ``UnifiedFacePolicy`` sizes every
#:   per-slot array from ``head_layout(approx_add).n_slots``, the rollout wire
#:   and the env state are :func:`wire_slots` wide, and
#:   ``face_driver.make_face_slot_legality_callback`` hands the head one mask
#:   row per slot instead of narrowing to three.
#:
#: ``FACE_SLOTS`` still means the three CONTRACTION slots, so every
#: ``range(FACE_SLOTS)`` loop is still right; :func:`wire_slots_of_rows` and
#: ``face_action.check`` are what make a mis-slotted row RAISE.
APPROX_ADD_CLI = APPROX_ADD_CHOICES

#: The values a PLAN REPLAY tool can offer, which is NOT the trainer's list.
#:
#: A replay (``tools/landscape_map``, and anything else that reconstructs wires
#: from a stored plan rather than drawing them) has no head, so:
#:
#: * ``choose`` is unreachable -- the join is a PER-FACE decision the head made,
#:   and a replay that filled the channel with zeros would measure every merge
#:   under ``lossy`` while claiming to replay the plan. That is the silent
#:   action/reward mismatch, so the value is not offered rather than defaulted;
#: * ``learned1`` / ``learned2`` are unreachable while a tool's ``face_specs``
#:   are ``FACE_SLOTS`` wide -- :func:`wire_slots_of_rows` refuses those rows,
#:   which is correct, and widening the tool is what would make the values
#:   available.
#:
#: ``tools/landscape_map`` restates this literal (it builds its argparser before
#: importing env on purpose) and ``tests/approx_add_test.py`` asserts the two
#: still agree, so a drift is a test failure rather than a silently different
#: choice list.
APPROX_ADD_CLI_REPLAY = ("lossy", "lossless")
#: The declared default, ``lossless``: OWNER RULING 2026-09-10 (`90e0ab85` on
#: hostperf-caches). It overrides the trade-off the implementation measured, and
#: the measurement is kept here because it says what the ruling costs rather
#: than being an argument against it.
#:
#: MEASURED (TLM seq 16 / dmodel 64 / vocab 256, one graph, min-Markowitz,
#: 3 seeds, finding 73): the summed edge's ``val`` is 17.8 MB under the union
#: against 8.8 MB under ``lossy`` with the `new` slot armed alone, and 22.3
#: against 9.1 MB with all three armed -- 2.03x and 2.46x. So under this default
#: an approximated merge face costs roughly twice the storage it would under
#: ``lossy``, and a cost channel reading stored bytes will see that as a price
#: the head pays for approximating.
#:
#: The ruling is nonetheless the safer default for a SEARCH: ``lossless`` drops
#: no non-zero of either addend (max relative error 5.112e-08, float noise), so
#: the only information a plan loses is what its own slots asked to lose. Under
#: ``lossy`` the ADD silently discards part of the old edge as well, which is
#: not in the plan's action record. Paying storage to keep the measured object
#: equal to the chosen object is the conservative trade.
#:
#: NOTE the flop channel cannot see any of this: ``_ew_op_count`` counts logical
#: extent, not stored cells, and reports the two values as identical
#: (ticket dsnn-3qm.76).
APPROX_ADD_DEFAULT = "lossless"
_RETIRED_APPROX_OLD = {"same": "lossy", "exact": "lossless"}


def approx_add() -> str:
    """One of :data:`APPROX_ADD_CHOICES` -- how the two addends of a face ADD
    are made to meet.

    The user surface is ``ppo.py --approx-add``; unset means
    :data:`APPROX_ADD_DEFAULT`. Anything else is a programming error, not a
    fallback -- including the retired ``same`` / ``exact``, which named a
    different computation (see the block comment above).
    """
    if _APPROX_OLD_ENV in os.environ:
        raise ValueError(
            f"{_APPROX_OLD_ENV} is RETIRED and is still set to "
            f"{os.environ[_APPROX_OLD_ENV]!r}. The flag is now "
            f"--approx-add / {_APPROX_ADD_ENV} with values "
            f"{APPROX_ADD_CHOICES}. The old values are not aliases: `same` "
            f"installed one wire row on two differently-structured tensors "
            f"and `exact` approximated the post-join SUM, so neither names "
            f"the object its replacement measures. Unset it and choose "
            f"explicitly.")
    want = os.environ.get(_APPROX_ADD_ENV, APPROX_ADD_DEFAULT).strip().lower()
    if want in _RETIRED_APPROX_OLD:
        raise ValueError(
            f"{_APPROX_ADD_ENV}={want!r} names a RETIRED --approx-old value. "
            f"The nearest replacement is {_RETIRED_APPROX_OLD[want]!r}, but it "
            f"is NOT the same computation -- see alphagrad.approx.env for why. "
            f"Choose one of {APPROX_ADD_CHOICES} deliberately.")
    if want not in APPROX_ADD_CHOICES:
        raise ValueError(
            f"{_APPROX_ADD_ENV} must be one of {APPROX_ADD_CHOICES} (set by "
            f"ppo.py from --approx-add), got {want!r}")
    return want


def resolve_join_mode(mode=None) -> str:
    """The join semantics for ONE face: ``"lossy"`` or ``"lossless"``
    (:data:`JOIN_SEMANTICS`).

    ``mode`` is the per-FACE override the ``choose`` bit decodes to. ``None``
    means "take it from the configuration", which is only answerable when the
    configuration FIXES it:

    * a fixed value (:data:`APPROX_ADD_FIXED`) answers for every face, through
      :data:`_JOIN_SEMANTICS_OF`. For ``lossy`` / ``lossless`` that is the value
      itself; for ``learned1`` / ``learned2`` it is ``lossless``, THE UNION --
      see :data:`_JOIN_SEMANTICS_OF` for why, because this is the one place a
      reader asks "the learned values have no choose bit, so which container
      does the add use?";
    * ``choose`` does NOT -- the decision is the head's, one bit per face, so a
      caller that reaches here with ``mode=None`` under ``choose`` has lost the
      bit somewhere between the wire and the entry builder. That raises. It must
      not silently become ``lossy``: the plan would then be measured under a
      semantics the policy did not choose, and the log-prob the trainer stored
      would score a decision that never ran.
    """
    cfg = approx_add()
    fixed = _JOIN_SEMANTICS_OF[cfg]
    if mode is None:
        if fixed is not None:
            return fixed
        raise ValueError(
            f"--approx-add {cfg!r} decides the join PER FACE, so "
            f"face_entry_from_slots needs an explicit mode for this face and "
            f"got None. The bit rides the wire as face_join[f]; a caller that "
            f"drops it has lost the head's decision, and defaulting here would "
            f"measure a plan under a semantics the policy did not choose.")
    if mode not in JOIN_SEMANTICS:
        raise ValueError(
            f"per-face join mode must be one of {JOIN_SEMANTICS}, got "
            f"{mode!r}.")
    if fixed is not None and mode != fixed:
        raise ValueError(
            f"--approx-add {cfg!r} FIXES the join semantics at {fixed!r}, but "
            f"this face was handed mode={mode!r}. A per-face override is only "
            f"meaningful under 'choose'; accepting it here would let a wire "
            f"silently override the flag. (Under the learned values the model "
            f"picks the approximations and the ADD reconciles with the union; "
            f"there is no per-face container decision to make.)")
    return mode


def wire_slots(mode: str | None = None) -> int:
    """How many slots one face's wire row carries under ``--approx-add``.

    3 under ``lossy`` / ``lossless`` / ``choose``, 4 under ``learned1``, 5 under
    ``learned2`` -- DERIVED from the head's own layout table, not restated here,
    so the wire width and the head width are one number. ``mode`` defaults to
    :func:`approx_add`.

    This is what replaced the fixed ``N_WIRE_SLOTS = 5``: a wire five slots wide
    under ``lossless`` would carry two rows the head has no logits for, and the
    engine would apply them.
    """
    from alphagrad.approx.unified_face_head import head_layout
    return head_layout(approx_add() if mode is None else mode).n_slots


def join_mode_of_bit(bit) -> str:
    """``face_join[f]`` -> the join semantics it names.

    ONE decoder for the bit, so the head's encoding
    (:data:`~alphagrad.approx.unified_face_head.JOIN_LOSSY` = 0 = ``lossy``)
    and the engine's reading of it cannot drift.
    """
    return "lossless" if int(bit) else "lossy"


def _join_outcome_sink():
    """The telemetry sink handed to a join policy, or ``None`` when unarmed.

    A reconciliation is NOT a micro-action: it applies no rule off the wire and
    must never land in ``applied_<kind>`` / ``skipped_<kind>``, which are the
    apply-rate numerators. It gets its own counters in the same dict:

    * ``join_<mode>``              -- merges reconciled under that mode;
    * ``join_matched_target``      -- of those, the ones whose common container
      IS the one the policy aimed at. For ``lossless`` that is the union and
      always true; for ``lossy`` it is the fresh contraction's container, and
      ``join_wider_than_target`` is the honest count of merges whose add came
      out WIDER -- and therefore more expensive -- than the head asked for.

    Gated on ``face_counts_armed`` exactly as ``make_slot_frame_hook``'s
    counters are, so the tokenizer's and the face-enum walk's replays of the
    same hook objects do not inflate the measurement.
    """
    from alphagrad.approx.common.masks import face_counts_armed

    def _sink(outcome):
        if not face_counts_armed():
            return

        def _bump(key):
            _PER_FACE_STATS[key] = _PER_FACE_STATS.get(key, 0) + 1

        _bump(f"join_{outcome.mode}")
        _bump("join_matched_target" if outcome.matched_target
              else "join_wider_than_target")
        if outcome.rules:
            _bump("join_projected")
    return _sink


def face_entry_from_slots(slots, at_site=None, at_join=None, mode=None,
                          with_policy=True):
    """ONE face's ``face_transforms`` entry from its decoded per-slot hooks
    ``(lhs, rhs, new)`` -- the ONLY place a face wire becomes a graphax entry
    (ticket .17, D1). ``_face_dict_for_vertex`` (the measurement),
    ``live_faces.LiveFaceStream._decided`` (the head's tokens),
    ``plan_tokens.PlanTokenizer.face_transforms`` (the AZ tokens) and
    ``masks.LiveVertexMaskOracle._face_ft`` (the mask replay) all go through
    here, so one wire has one transform semantics and :func:`approx_add` has
    one reader.

    #73. EVERY APPROXIMATION LANDS ON THE CONTRACTION SIDE. The three slots
    are the two contraction operands and its result, so the entry is always
    the TWO-OP form with ``new`` at graphax's ``res:new`` -- the fresh
    contraction, BEFORE any join. The `new` slot's wire row therefore meets
    exactly ONE tensor, which is the tensor :func:`slot_rules_for_row` and
    ``masks.slot_legality`` are computed on: "the mask admits it" and "the hook
    applies it" are again one statement.

    That is the #73 change. Under the retired ``--approx-old same`` the SAME
    hook object was also installed at ``jr``, the pre-existing old edge, so one
    row ran on two tensors whose logical dims agreed and whose STORAGE did not
    -- a legal block subdivision on one, an idempotent no-op on the other
    (finding 72, fault 1). The ADD is no longer expressed as "apply the same
    rule twice"; it is expressed as a JOIN POLICY.

    THE JOIN POLICY sits at the ``jr`` position and is handed BOTH addends by
    graphax (``core._eliminate_vertex``; see
    :mod:`graphax.sparse.ops.join`). It returns both, in ONE container:

    * ``lossy``    -> ``MatchFreshJoin``: the common container is the one the
      approximated ``new`` slot landed on, with the old edge projected onto it.
    * ``lossless`` -> NO POLICY AT ALL. graphax's sparse ``+`` already builds
      the UNION container for a union op -- meta ``gcd``, block ``lcm``
      (``elementwise._pair_metric``) -- which IS "the largest of the two
      structures along each dim". Installing ``UnionJoin`` would compute the
      same container twice and move no value, so ``lossless`` emits the join
      triple ALL-None and lets the add do it. Measured exact: max relative
      error 0 over every merge face of the TLM census.

    ``jl`` and ``jres`` stay None here. ``jl`` would hit the tensor ``new``
    has already transformed, with no intervening op, so it can express nothing
    ``new`` cannot; ``jres`` is the post-join SUM, which is not a decision the
    contraction-side head makes. Both are the positions the LEARNED variants
    will occupy and are deliberately left free.

    A merge-free face is unaffected: graphax never reaches the join position,
    so the policy never runs and there is nothing to reconcile.

    ``at_site(site, hook)`` -- OPTIONAL per-SITE adapter. A slot's hook object
    could be installed at more than one graphax site, and a caller that has to
    tell those invocations apart -- the legality probe, which needs one mask
    per tensor the hook will meet -- cannot do it from the entry, because the
    object would be identical at every position. ``at_site`` is called once per
    placement with the site name and returns the object to place there.
    Default: the hook itself, which is what the measurement installs. The site
    names are graphax's own (:func:`graphax.core._unpack_face_slots`), prefixed
    ``res:`` for the four result sites.

    SITE NAMES ARE NOT DECORATION. ``_probe_faces`` builds its recording entry
    through THIS function precisely so the set of sites it records can never
    drift from the set the measurement installs -- the drift that made the
    ``new`` slot's mask clear Diags the engine refused on the old edge
    (finding 72, ticket .59 fault 1).

    ``at_join(policy)`` -- a SEPARATE adapter for the join policy, and separate
    on purpose: the policy is not a hook (it takes two tensors, not one), it
    applies no wire row, and there is no mask for it to answer for, so routing
    it through ``at_site`` would hand a per-SITE callback an object that has no
    site. It returns the object to place at ``jr``; the default is the policy
    itself.

    ``with_policy=False`` -- place the slot HOOKS but no join POLICY, and do not
    consult the arm. The legality probe uses it, and it is SAFE rather than
    convenient: the probe needs the TENSORS each slot hook will meet, which the
    hooks alone deliver, while the reconciliation only changes values and is
    undone by the probe's snapshot anyway. Leaving the live policy in would make
    every probe pay the reconciliation's arithmetic for a result nothing reads
    -- and under ``choose`` the probe holds no bit, so asking for the arm would
    raise.

    AS MANY SLOTS AS THE RUNNING VALUE HAS, EXACTLY. ``len(slots)`` must equal
    :func:`wire_slots` -- 3 under ``lossy`` / ``lossless`` / ``choose``, 4 under
    ``learned1``, 5 under ``learned2``:

        0..2  contraction: lhs, rhs, new
        3     learned1 -- the OLD EDGE,    graphax ``jr``
        4     learned2 -- the SUMMED EDGE, graphax ``jres``

    Any other length raises, and the check is EQUALITY rather than "3 or 5"
    (2026-09-11). The width is not a caller's choice any more, it is the
    configuration's, and it is the same number the HEAD is built at
    (``unified_face_head._LAYOUT_SPEC``). A wire four slots wide under
    ``lossless`` would carry a row the head has no logits for and the engine
    would apply it; a wire three wide under ``learned1`` would drop a row the
    head drew and scored. Both are silent, so both raise.

    learned1's hook lands on the old edge as a PLAIN ``jr`` hook, and learned2's
    on the summed edge at ``jres``. Under the 2026-09-11 widths that is the only
    route either of them takes: ``learned1`` / ``learned2`` reconcile with the
    UNION (``_JOIN_SEMANTICS_OF``), ``lossless`` installs no policy, so there is
    no policy for a learned hook to ride. The ``pre=`` wiring below is kept
    because it is the CORRECT route if a future value ever pairs a ``lossy``
    container with a learned old-edge slot -- under such a value the hook must
    run BEFORE the reconciliation, so the reconciliation still has the last word
    and the two addends still come out structurally identical -- but no value in
    :data:`APPROX_ADD_CHOICES` reaches it today, because every value whose join
    is ``lossy`` has exactly three slots and therefore no learned1 hook to pass.
    It is also not yet a route a DECISION can survive: ``FaceJoinPolicy.pre`` is
    called directly rather than through ``core._apply_face_transform``, so a
    chooser there would be applied and never recorded, and the branch RAISES
    rather than losing the marker.
    """
    def _at(site, hook):
        if hook is None or at_site is None:
            return hook
        return at_site(site, hook)

    _want = wire_slots()
    if len(slots) != _want:
        raise ValueError(
            f"--approx-add {approx_add()!r} has {_want} face slots "
            f"({FACE_SLOTS} contraction"
            + ("" if _want == FACE_SLOTS else
               " + learned1 on the old edge" if _want == FACE_SLOTS + 1 else
               " + learned1 on the old edge + learned2 on the summed edge")
            + f"), and face_entry_from_slots got {len(slots)}. The WIDTH is "
            f"the configuration's, not the caller's, and it is the width the "
            f"head is built at: a row the head has no logits for must not be "
            f"applied, and a row the head drew must not be dropped.")
    _new_hook = slots[2]
    _l1 = slots[FACE_SLOTS] if len(slots) > FACE_SLOTS else None
    _l2 = slots[FACE_SLOTS + 1] if len(slots) > FACE_SLOTS + 1 else None
    core3 = (_at("lhs", slots[0]), _at("rhs", slots[1]),
             _at("res:new", _new_hook))
    jr_hook = _at("res:jr", _l1)
    jres_hook = _at("res:jres", _l2)
    if not with_policy:
        return (core3, (None, jr_hook, jres_hook))
    mode = resolve_join_mode(mode)
    # AN UNARMED FACE STAYS EXACT, and the gate lives HERE.
    #
    # ``--approx-add`` says how the two addends of the ADD meet, and that is
    # only a question when one of them was approximated. A face whose three
    # slots are all None computes an EXACT contraction, and the old edge it
    # merges into may legitimately be WIDER than that contraction (it
    # accumulated approximated contributions at earlier steps). Installing a
    # ``lossy`` policy there would project that old edge down to a container
    # nobody asked to approximate -- information lost on a face the plan marked
    # exact. It would also falsify the documented property that a SKIP-only or
    # exact plan makes this flag INERT (``tools/gen_fq_launchers.py``).
    #
    # The three production callers (``_face_dict_for_vertex``,
    # ``live_faces._decided``, ``masks.LiveVertexMaskOracle._face_ft``) each
    # already skip an all-None face before calling here, so this gate changes
    # nothing today. It lives here anyway because this function is "the ONLY
    # place a face wire becomes a graphax entry": a guard held in three copies
    # at the call sites is exactly the duplication that let the probe's site
    # list drift from the measurement's (finding 72).
    _armed = any(h is not None
                 for h in (slots[0], slots[1], _new_hook, _l1, _l2))
    if mode == "lossy" and _armed:
        from graphax.sparse.ops.join import MatchFreshJoin
        # `pre` is the position a learned OLD-EDGE hook would occupy under a
        # lossy container: graphax applies it before the reconciliation, so the
        # reconciliation still has the last word and the two addends still come
        # out structurally identical. Under the current value table `jr_hook` is
        # always None here -- every lossy value has three slots (see the
        # docstring).
        #
        # AND IT MUST STAY NONE. Every other hook position goes through
        # `graphax.core._apply_face_transform`, which applies a chooser's action
        # AND RECORDS it as an `approx` block; `FaceJoinPolicy.reconcile` calls
        # `pre` DIRECTLY and takes back a tensor, so a slot hook there would be
        # a decision applied to the Jacobian with no marker in the stream --
        # exactly the defect finding 64 named. A wire that asks for it is
        # refused rather than applied unrecorded.
        if jr_hook is not None:
            raise NotImplementedError(
                "face_entry_from_slots: --approx-add lossy would put slot "
                f"{FACE_SLOTS}'s hook in FaceJoinPolicy.pre, which graphax "
                "applies directly instead of through _apply_face_transform, so "
                "the rule would be applied and left out of the token stream. "
                "Give the join slot its own graphax hook position before "
                "wiring it here.")
        policy = MatchFreshJoin(pre=jr_hook,
                                on_outcome=_join_outcome_sink())
        if at_join is not None:
            policy = at_join(policy)
        return (core3, (None, policy, jres_hook))
    if mode in ("lossy", "lossless"):
        # No policy: learned1 is a PLAIN hook at `jr`. Same tensor, same site.
        return (core3, (None, jr_hook, jres_hook))
    raise NotImplementedError(
        f"face_entry_from_slots: --approx-add {mode!r} is accepted by "
        f"approx_add() but has no entry form here. A value that cannot be "
        f"built must not silently fall back to another one.")


def face_slot_sites() -> tuple[tuple[str, ...], ...]:
    """Per slot ``(lhs, rhs, new)``, the graphax SITES
    :func:`face_entry_from_slots` places that slot's single hook at, under the
    CURRENT :func:`approx_add`. The FIRST entry is the site the per-slot mask
    is named for; the rest are the extra tensors the same wire row also lands
    on, which the mask has to be intersected over.

    DERIVED by calling the entry builder itself with tagging probes rather
    than restated, so the two cannot drift.

    DERIVED WITH ``with_policy=False``, and that is the whole reason this function
    can answer under EVERY ``--approx-add`` value including ``choose``. The join
    position holds a POLICY or nothing; a policy is not a slot hook, applies no
    wire row and answers to no mask, so it contributes no site and the slot
    topology cannot depend on which arm runs. Asking for an arm here would be
    asking a question this function does not need the answer to -- and under
    ``choose`` the arm is a per-face bit that the mask, which runs BEFORE the
    draw, does not have.

    If a future value ever put a SLOT HOOK at a join site, it would appear here
    automatically, because the site list is whatever the entry builder tags --
    never a restatement.
    """
    n = wire_slots()
    got: list[list[str]] = [[] for _ in range(n)]

    def _tag(site, hook):
        got[int(hook)].append(site)
        return hook

    face_entry_from_slots(tuple(range(n)), at_site=_tag, with_policy=False)
    return tuple(tuple(g) for g in got)


def wire_slots_of_rows(rows) -> int:
    """The slot count of a ``(F, S, 3)`` wire-row array, CHECKED.

    ONE place the rows' width meets :func:`wire_slots`, so a caller that loops
    over a row array cannot decide for itself how many bands are present. A row
    array narrower than the configuration would drop a slot the head drew and
    scored; a wider one would apply a row the head has no logits for. Both are
    silent, so both raise here -- including the trainer path under the learned
    values, whose rows still carry the contraction band only (finding 73 section
    9b: the policy's per-slot features and the rollout wire are not widened
    yet).
    """
    n = int(np.asarray(rows).shape[-2])
    want = wire_slots()
    if n != want:
        raise ValueError(
            f"a face wire row array carries {n} slots but --approx-add "
            f"{approx_add()!r} has {want} "
            f"(FACE_SLOTS={FACE_SLOTS} contraction + "
            f"{want - FACE_SLOTS} learned join). The rows and the head are "
            f"built at the same width by construction; a mismatch means the "
            f"producer of these rows has not been widened -- the trainer wire "
            f"for the learned join slots is not built (ticket dsnn-3qm.56).")
    return n


def _face_dict_for_vertex(config, ij, v, face_row, face_skip,
                          face_join=None, *, keys=None, upto=None):
    """ONE vertex's ``{face_key: slots|SKIP_FACE}`` from its wire rows,
    enumerated on ``ij``'s CURRENT graph — call BEFORE eliminating ``v``.

    THE ONE BUILDER. The measurement, the live-face stream's prefix replay
    (``live_faces.LiveFaceStream._decided``), the AZ plan tokenizer
    (``plan_tokens.PlanTokenizer.face_transforms``) and the mask oracle's wire
    replay (``masks.LiveVertexMaskOracle._face_ft``) all build their entries
    here, so one wire row decodes to one transform: in the SLOT TENSOR's frame
    at apply time (:func:`make_slot_frame_hook`), never in the vertex's nominal
    frame. The head-side decoders used to decode in the vertex frame
    (``decode_vertex_rule_specs``); a COMPRESS axis or a DIAG pair then landed
    on different dims in the stream's prefix than in the measured graph, the
    stream's operand drifted, and 7 of 839 rows the decide-time mask cleared
    were idempotent no-ops on the real operand (probe 65266/65270 on the
    dsnn-3qm.59 stage-2 smoke: with this builder the mask refuses all 7).

    ``keys`` -- the face keys to index the rows by; ``None`` enumerates them
    on ``ij``. A deciding tokenizer passes its own list so a later, shrunk
    enumeration cannot apply row ``f`` to a different face. ``upto`` -- build
    faces ``0..upto-1`` only (the stream reads face ``f`` on the prefix of the
    faces before it).

    ``face_join`` -- the ``(MAX_FACES,)`` int32 per-face JOIN bit of
    ``--approx-add choose`` (0 = lossy, 1 = lossless). ``None`` under a value
    that fixes the join semantics for the whole run. It is a SEPARATE channel
    from ``face_skip`` on purpose: ``face_skip`` means "drop this face's
    contraction" and overloading its bits with an unrelated decision would make
    every reader of a field called "skip" wrong.
    Single source of truth for `_face_transforms_for_order` (standalone
    replay) and the unified tokenizer path (ALPHAGRAD_UNIFIED_FACE_ENUM=1),
    which rides the tokenizer's own IncrementalJaxpr instead of replaying a
    second, byte-identical elimination."""
    from graphax import SKIP_FACE, faces_of
    if keys is None:
        keys = faces_of(ij.graph, ij.tgraph, int(v), config.jaxpr)
    else:
        keys = list(keys)
    if len(keys) > _FACE_CAP_STATS["max_seen"]:
        _FACE_CAP_STATS["max_seen"] = len(keys)
    if len(keys) > MAX_FACES:
        # The width is the PROVABLE bound, so this cannot fire unless
        # the ancestors x descendants argument is wrong -- in which case
        # a silent slice would shrink the action space while reporting
        # a healthy run. Die loudly instead.
        raise RuntimeError(
            f"vertex {v}: {len(keys)} faces exceed the derived bound "
            f"{MAX_FACES} -- the subset argument is violated")
    per_face: dict = {}
    face_row = np.asarray(face_row)
    face_skip = np.asarray(face_skip).reshape(-1)
    # THE WIRE MUST BE AT LEAST AS WIDE AS THE ENUMERATION. The loop below
    # BREAKS when the row runs out, which is the right answer for a caller
    # that genuinely has fewer rows and the WRONG one for a narrowed wire
    # (`--face-wire-faces`): the faces past the end would run exact while
    # every counter reported a healthy run. That is the fault class the face
    # width exists for -- eight silently dropped faces of the xent graph's
    # vertex 9 -- so it raises here instead of running the tail.
    _w = min(int(face_row.shape[0]), int(face_skip.shape[0]))
    if len(keys) > _w:
        raise RuntimeError(
            f"vertex {v}: {len(keys)} faces and a face wire {_w} columns "
            f"wide. Raise --face-wire-faces: the faces past {_w} carry no "
            f"decision and would run exact in silence.")
    for f, key in enumerate(keys[:MAX_FACES]):
        if upto is not None and f >= int(upto):
            break
        if f >= face_skip.shape[0] or f >= face_row.shape[0]:
            break
        if int(face_skip[f]) == 1:
            per_face[key] = SKIP_FACE
            continue
        slots = []
        # AS MANY SLOTS AS THE CONFIGURATION HAS, and the rows must already
        # carry exactly that many: `wire_slots_of_rows` is the one place the two
        # meet. `face_row` is (F, S, 3).
        _n_slots = wire_slots_of_rows(face_row)
        for s in range(_n_slots):
            # The decoder walks all MAX_RULES slots — pad the single face
            # row with end-sentinels.
            # `face_row` may be a numpy view (the caller no longer pays for a
            # dense `.tolist()`), so pull the 3 wire ints out explicitly.
            one_row = [[int(x) for x in face_row[f][s]]] + [
                [-1, -1, 0]
            ] * (MAX_RULES_PER_VERTEX - 1)
            # Ticket .18, D2. A vertex frame would describe the lhs tensor
            # only; rhs and new carry other out/primal dims. So the row is
            # decoded when the slot's LIVE tensor is in hand
            # (make_slot_frame_hook), through the same per-face sink.
            # THE per-FACE sink: under --live-faces the per-vertex rows are
            # all-exact END rows, so the per-vertex sink is never constructed
            # and this is the ONLY place approx_applied/* can come from.
            # ``gated`` because these same objects are replayed by the
            # face-enum walk and by the tokenizer -- only the armed scope
            # inside ``_do_compile_approx`` is the measurement.
            slots.append(
                make_slot_frame_hook(one_row[0], stats=_PER_FACE_STATS,
                                     gated=True)
                if one_row[0][0] != -1 else None)
        if any(sl is not None for sl in slots):
            # THE PER-FACE JOIN BIT (#73, --approx-add choose). `face_join` is
            # the wire's (F,) int32 channel: 0 = lossy, 1 = lossless. Under a
            # FIXED value it is absent and `resolve_join_mode` answers from the
            # configuration; under `choose` it must be present, and
            # `resolve_join_mode` raises rather than guessing if it is not.
            _mode = (None if face_join is None
                     else join_mode_of_bit(face_join[f]))
            per_face[key] = face_entry_from_slots(slots, mode=_mode)
    return per_face


def _measure_compiler_options():
    """Per-executable XLA options for MEASURE compiles only
    (ALPHAGRAD_MEASURE_COMPILER_OPTS=0 to disable). GEMM autotuning +
    Triton fusion work dominated cold variant compiles (measured 3.55s ->
    0.09s, 39x, with ZERO latency change on the bandwidth-bound nn256
    target — the 3.5s recurs per new GEMM config, it is not process
    warmup). Values must be typed int/bool: the string "false" is rejected
    with INVALID_ARGUMENT. Scoped per executable so the TRAINER jit keeps
    full optimization once the global XLA_FLAGS are dropped. Re-validate
    once on compute-bound targets (TLM): autotune-off can change kernel
    choice there."""
    if os.environ.get("ALPHAGRAD_MEASURE_COMPILER_OPTS", "1") == "0":
        return None
    try:
        if jax.default_backend() == "cpu":
            return None
    except Exception:
        pass
    return {
        "xla_gpu_autotune_level": 0,
        "xla_gpu_enable_triton_gemm": False,
        # Parallel LLVM-module compilation. OFF by default in this jax build
        # (gated behind a persistent-cache setting nobody enables), so every
        # run to date compiled serially. Isolated benchmark 61445 (12 random
        # TLM plans, paired on the SAME lowered object, arm order alternated):
        # median compile 14.46 -> 12.71 s (-12%), heaviest plans -29..-35%
        # (45.6 -> 32.7 s), no plan slower. Passed per-executable here rather
        # than via JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES so it does not
        # depend on cache state and never touches trainer compiles.
        "xla_gpu_enable_llvm_module_compilation_parallelism": True,
    }


# ---------------------------------------------------------------------------
# MEASURE TOOLCHAIN GATE (finding 03, ticket dsnn-3qm.21).
#
# `xla_gpu_enable_llvm_module_compilation_parallelism` splits the module, and
# splitting means LINKING: XLA shells out to the first `nvlink` it finds. On
# pgi15-gpu16/17 that was /usr/local/cuda/bin/nvlink -- a SYMLINK pointing at
# cuda-12.8 -- while the venv's ptxas (nvidia_cuda_nvcc_cu12 12.9.86) emits
# 12.9 cubins, so nvlink refused every one:
#     nvlink fatal : Input file '...cubin' newer than toolkit (129 vs 128)
# `_compile_measure` matched the "INTERNAL" signature and silently swapped in
# the degraded-fusion executable. That contaminated 51% of wave-1 arm w1b and
# 97% of w1c and was found only in post-hoc log analysis. gpu15 and gpu16 run
# the SAME driver (580.178.04): it was never a driver mismatch.
#
# The gate compiles a tiny throwaway module with the LIVE measure options at
# the first measure compile of every measuring process (trainer, every
# CpuApproximationActor, every respawn), which is the one site no call path
# can bypass. Measured cost 0.07 s; a 4-op probe trips the fault as reliably
# as a 40-op one (job 62796).
#
# THE PROBE MUST NOT BE CACHEABLE. Measured (job 62809): a persistent-cache
# entry written on a healthy node is reused verbatim on the broken one and
# the fault never fires, so a cacheable probe is a false-negative generator.
# See `_toolchain_probe_compile` for how the cache is bypassed and why the
# obvious way (jax_enable_compilation_cache=False) is wrong.
#
# Default = abort. A silent repair that swaps the toolkit would be the same
# class of invisible change as the bug. A SKIP IS A FAILURE.
# ---------------------------------------------------------------------------

# Published by ppo.py from --measure-toolchain-gate BEFORE ray.init, exactly
# like ALPHAGRAD_QUALITY_METRIC: the measure actors run this module's
# `_compile_measure` in their own processes, and one env var read by ONE
# function is what makes trainer and actors unable to disagree. This is the
# transport, not a knob -- the flag is the only control.
_MEASURE_TOOLCHAIN_GATE_ENV = "ALPHAGRAD_MEASURE_TOOLCHAIN_GATE"
_MEASURE_TOOLCHAIN_GATE_MODES = ("abort", "warn", "off")

# Process-global gate state. `ok` goes False on a probe failure OR a link
# fault in a real measure compile; every plan record and every
# `consume_plan_records` drain carries it.
_MEASURE_TOOLCHAIN = {"checked": False, "ok": True, "detail": "",
                      "mode": "", "host": "", "link_faults": 0}

# The XLA error text of a link-toolchain fault. Either line alone is
# sufficient (finding 03 sec 1).
_LINK_FAULT_SIGNATURES = ("nvlink", "The CUDA linking API did not work")


class MeasureToolchainFault(RuntimeError):
    """The measure compile toolchain on THIS node is broken.

    Deliberately its own class: the measurement sentinel machinery
    (``cpu_approx_worker.evaluate``, ``CpuApproxPool.evaluate_batch``)
    absorbs every other exception into a sentinel row and lets the run
    continue, which for THIS fault would mean a run that measures nothing
    or measures degraded executables while exiting 0. Both re-raise it.
    """


class MemChannelFault(MeasureToolchainFault):
    """The memory channel could not be read for a measured plan, or a
    measured plan left no parity record (ticket dsnn-3qm.49).

    A subclass of :class:`MeasureToolchainFault` ON PURPOSE: the sentinel
    machinery re-raises that class instead of absorbing it into a sentinel
    row, and a plan scored on a memory channel that read nothing is exactly
    the run that must not continue exiting 0 (a skip is a failure).
    """


def measure_toolchain_gate_mode() -> str:
    """``abort`` | ``warn`` | ``off`` -- THE one reader of the gate mode.

    Absent (a process not started through ppo.py, e.g. a test or a tool)
    means ``abort``. A value outside the three modes raises: argparse
    already validated the flag, so garbage here is a hand edit.
    """
    want = os.environ.get(_MEASURE_TOOLCHAIN_GATE_ENV, "abort").strip().lower()
    if want not in _MEASURE_TOOLCHAIN_GATE_MODES:
        raise ValueError(
            f"{_MEASURE_TOOLCHAIN_GATE_ENV}={want!r} is not one of "
            f"{_MEASURE_TOOLCHAIN_GATE_MODES}; it is published from "
            f"--measure-toolchain-gate, not set by hand")
    return want


def _is_link_toolchain_fault(msg: str) -> bool:
    return any(_sig in msg for _sig in _LINK_FAULT_SIGNATURES)


def _tool_version(name: str, path: str | None = None) -> str:
    """``'12.9.86 at /usr/local/cuda-12.9/bin/ptxas'`` for the ``ptxas`` /
    ``nvlink`` FIRST ON PATH -- the one XLA shells out to -- or a reason.
    With ``path`` given, that binary instead."""
    import re
    import shutil
    import subprocess
    p = path if path is not None else shutil.which(name)
    if p is None:
        return "not on PATH"
    if not os.path.exists(p):
        return f"absent ({p})"
    try:
        txt = subprocess.run([p, "--version"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception as _exc:
        return f"unreadable at {p} ({type(_exc).__name__})"
    m = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", txt)
    return f"{m.group(2)} at {p}" if m else f"unparsed version at {p}"


def _venv_ptxas_version() -> str:
    """The ptxas the venv ships (nvidia_cuda_nvcc_cu12): with none on PATH
    this is the one XLA compiles PTX with, and the release its cubins carry
    -- the '129' in 'newer than toolkit (129 vs 128)'."""
    import sysconfig
    p = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cuda_nvcc",
                     "bin", "ptxas")
    return _tool_version("ptxas", p)


def _toolchain_probe_compile():
    """Compile a tiny throwaway executable with the LIVE measure options,
    BYPASSING the persistent compile cache. Returns the lowered object.

    Two mechanisms, both needed:

    * UNIQUE KEY. A fresh random constant is baked into the module, so its
      cache key has never been written by any process on any node and the
      lookup cannot hit. The input is a ``ShapeDtypeStruct``: a concrete
      ``jnp.ones`` would run eager helper ops OUTSIDE this function's
      control and those DO get cached (measured locally).
    * NO WRITE. The compile-time threshold is raised for this one compile
      (thread-local jax state), so the probe never leaves an entry behind
      even under JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.

    NOT ``jax_enable_compilation_cache=False``: ``compilation_cache.
    is_cache_used`` latches its answer on the first compile of the process,
    and this gate IS the first measure compile in every actor -- it would
    switch the persistent cache off for the whole process.
    """
    import secrets
    from jax._src import config as _jcfg
    _c = 1.0 + secrets.randbits(52) / float(1 << 52)

    def _probe(x):
        y = jnp.sin(x * _c)
        y = jnp.tanh(y) + 0.5
        y = y @ jnp.eye(y.shape[-1], dtype=y.dtype)
        return jnp.log1p(jnp.abs(y))

    with _jcfg.persistent_cache_min_compile_time_secs(1e9):
        lowered = jax.jit(_probe).lower(
            jax.ShapeDtypeStruct((16, 16), jnp.float32))
        lowered.compile(compiler_options=_measure_compiler_options())
    return lowered


def _toolchain_fault_message(where: str, detail: str) -> str:
    import socket
    host = socket.gethostname()
    detail = " ".join(str(detail).split())[:400]
    return (
        f"[measure] TOOLCHAIN FAULT on {host} ({where}): the live measure "
        f"compile options do not produce a working executable, so every "
        f"measurement on this node would silently be a degraded-fusion "
        f"FALLBACK and not comparable with the other nodes.\n"
        f"  fault:  {detail}\n"
        f"  ptxas:  {_tool_version('ptxas')} (PATH); "
        f"{_venv_ptxas_version()} (venv)\n"
        f"  nvlink: {_tool_version('nvlink')} (PATH)\n"
        f"  /usr/local/cuda -> {os.path.realpath('/usr/local/cuda')}\n"
        f"  FIX: put a toolkit whose ptxas and nvlink MATCH first on PATH in "
        f"the job script (tools/gen_fq_launchers.py does this; finding 03 "
        f"sec 4a). Not xla_gpu_cuda_data_dir: it silences the error for any "
        f"path. --measure-toolchain-gate warn runs anyway and tags every "
        f"plan; off skips the probe.")


def _measure_toolchain_check() -> dict:
    """One-time per-process gate on the measure compile toolchain."""
    if _MEASURE_TOOLCHAIN["checked"]:
        return _MEASURE_TOOLCHAIN
    import socket
    mode = measure_toolchain_gate_mode()
    host = socket.gethostname()
    _MEASURE_TOOLCHAIN["checked"] = True
    _MEASURE_TOOLCHAIN["mode"] = mode
    _MEASURE_TOOLCHAIN["host"] = host
    if mode == "off":
        print(f"[measure] toolchain gate OFF on {host} "
              f"(--measure-toolchain-gate off): a broken link toolchain on "
              f"this node is NOT probed", flush=True)
        return _MEASURE_TOOLCHAIN
    t0 = time.perf_counter()
    try:
        _toolchain_probe_compile()
    except Exception as _exc:
        detail = f"{type(_exc).__name__}: {_exc}"
    else:
        print(f"[measure] toolchain gate OK on {host} "
              f"({time.perf_counter() - t0:.2f}s; ptxas "
              f"{_tool_version('ptxas')} (PATH), {_venv_ptxas_version()} "
              f"(venv); nvlink {_tool_version('nvlink')} (PATH))",
              flush=True)
        return _MEASURE_TOOLCHAIN
    _MEASURE_TOOLCHAIN["ok"] = False
    _MEASURE_TOOLCHAIN["detail"] = " ".join(detail.split())[:400]
    msg = _toolchain_fault_message("gate probe", detail)
    if mode == "warn":
        print(msg, file=sys.stderr, flush=True)
        return _MEASURE_TOOLCHAIN
    raise MeasureToolchainFault(msg)


_MEASURE_COMPILE_FALLBACKS = {"n": 0}
# Fallback count at the previous plan record / drain in THIS process, so a
# per-plan and a per-episode DELTA can be attributed (measurements are
# sequential per actor: pure_callback(vmap_method="sequential")).
_MEASURE_FALLBACKS_AT_LAST_RECORD = [0]
_MEASURE_FALLBACKS_AT_LAST_DRAIN = [0]


def _compile_measure(lowered):
    """Compile a MEASURE executable; on an INTERNAL GPU-compiler failure
    (ptxas exit-139 segfault / Triton fusion compile error -- both observed
    on Blackwell TLM plans, v53 jobs 59279/59306, across actors and order
    families) retry ONCE with a degraded-fusion option set instead of
    sentineling the whole plan. Option names are PROBED-VALID on this
    jax/XLA build (an unknown name raises INVALID_ARGUMENT and would
    defeat the fallback). The fallback executable is less fused, so its
    latency reads conservatively -- a real measurement, not a sentinel;
    every use is printed and counted so the bias stays visible.
    ALPHAGRAD_MEASURE_COMPILE_FALLBACK=0 disables.

    A LINK-toolchain fault (nvlink refusing a newer ptxas cubin) is NOT
    fallbackable: see the MEASURE TOOLCHAIN GATE block above."""
    _measure_toolchain_check()
    try:
        return lowered.compile(compiler_options=_measure_compiler_options())
    except Exception as _e:
        _m = str(_e)
        # A LINK-toolchain fault is an ENVIRONMENT fault, not a plan-specific
        # compiler bug: it recurs on every plan for the whole job and
        # silently degrades every latency on this node. Absorbing it into
        # the per-plan retry is how it hid inside wave 1 (finding 03). The
        # gate above should have caught it; reaching here means the
        # environment changed mid-run. Only --measure-toolchain-gate warn
        # takes the degraded set, and then the plan is TAGGED
        # (compile_fallbacks / toolchain_ok in its record).
        _link_fault = _is_link_toolchain_fault(_m)
        if _link_fault:
            _MEASURE_TOOLCHAIN["ok"] = False
            _MEASURE_TOOLCHAIN["detail"] = " ".join(_m.split())[:400]
            _MEASURE_TOOLCHAIN["link_faults"] += 1
            if (measure_toolchain_gate_mode() != "warn"
                    or os.environ.get("ALPHAGRAD_MEASURE_COMPILE_FALLBACK",
                                      "1") == "0"):
                raise MeasureToolchainFault(_toolchain_fault_message(
                    "measure compile, mid-run", _m)) from _e
        if os.environ.get("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", "1") == "0":
            raise
        # "Shared memory size limit exceeded" is labelled
        # RESOURCE_EXHAUSTED but is a compiler KERNEL-CONFIG failure
        # (XLA chose a tile above the SM's shared-mem budget -- observed
        # on Blackwell per-order exact compiles: requested 131072,
        # available 101376), not a real allocation OOM -- degraded
        # fusion legitimately avoids it. True OOMs still re-raise.
        # "A cycle is detected" (FAILED_PRECONDITION) is an XLA
        # fusion-pass graph bug -- observed on an exact TLM plan
        # (fusion.113, Blackwell); by construction a degraded-fusion
        # retry can avoid the offending fusion.
        if not _link_fault and not any(_sig in _m for _sig in (
                "ptxas exited", "Triton kernel", "INTERNAL",
                "Shared memory size limit", "A cycle is detected")):
            raise
        _MEASURE_COMPILE_FALLBACKS["n"] += 1
        print(
            f"[measure] compile FALLBACK #{_MEASURE_COMPILE_FALLBACKS['n']} "
            f"(degraded fusion) after: {_m[:200]}", flush=True)
        return lowered.compile(compiler_options={
            "xla_gpu_autotune_level": 0,
            "xla_gpu_enable_triton_gemm": False,
            "xla_gpu_enable_dynamic_slice_fusion": False,
            "xla_gpu_use_runtime_fusion": False,
            # The observed Blackwell failures are kCustom __triton
            # fusions with block_level_fusion_config -- produced by the
            # BLOCK-LEVEL rewriter, which the four knobs above do not
            # touch (observed: fallback #1 fired and still died on
            # fusion.205). Both names probed-valid on this build.
            "xla_gpu_experimental_enable_fusion_block_level_rewriter":
                False,
            "xla_gpu_experimental_enable_triton_heroless_priority_fusion":
                False,
        })


# ---------------------------------------------------------------------------
# LIVE ELIMINATION STATE
#
# `_face_transforms_for_order` is a PURE FUNCTION of the whole plan prefix, so
# every env step rebuilt an `IncrementalJaxpr` and replayed all k eliminations
# again: O(k) per step, O(T^2) per episode. `_FACE_ENUM_CACHE` tried to buy
# that back with a dict, but in a LINEAR ROLLOUT no prefix is ever visited
# twice, so that dict could never hit AS A CACHE -- the only path that could
# fire was pop-and-extend, i.e. "keep the live object", written as a lookup
# with an LRU cap that can evict the one entry the next step needs. And the
# lookup rebuilt its key from the DENSE wires (MAX_FACES x FACE_SLOTS x 3 ints
# per vertex, for every vertex of the prefix, every step -- 45936 ints per
# vertex at the 3-block TLM's derived bound of 5104), so the O(T^2) term the
# cache existed to remove was still being paid, inside the cache.
#
# This OWNS the state instead. A small pool of live chains -- one per
# concurrent env, since `pure_callback(vmap_method="sequential")` walks the
# batch one host call at a time -- each holding its `IncrementalJaxpr`, the
# face-transform dicts it has produced, and the exact prefix it has consumed.
# A step whose prefix extends a chain's ADVANCES that chain by the one new
# elimination. Work per step is constant, so O(T) per episode is STRUCTURAL:
# a mid-episode rebuild is a counted anomaly (`restart`), never a silent
# return to O(T^2).
#
# This is only possible because a vertex's decode no longer depends on where
# the prefix ends (COMPRESS is honored at every position -- the `is_last` gate
# is gone). Under that gate the last vertex of a prefix was decoded
# differently from the same vertex one step later, so a live builder would
# have had to speculate and roll back; that is what the deleted COMPRESS
# carve-outs were, and why v40 measured the cache essentially dead (ext=1/431)
# whenever COMPRESS was in the action space. The elimination stream is now
# append-only, and so is this.
#
# Identity is EXACT, never hashed: a chain matches on byte-equality of the
# order and spec prefixes plus tuple-equality of `_face_wire_keys` -- the
# sparse injective wire encoding `_INCR_STREAM_CACHE` already keys on,
# computed ONCE per callback and shared with it, so the check is free rather
# than a second O(T x MAX_FACES) pass.
#
# ALPHAGRAD_FACE_LIVE_STATE=0 restores the stateless rebuild, and with it
# ALPHAGRAD_FACE_ENUM_CACHE -- which is kept for BRANCHING search (GAZ/MCTS
# really do revisit prefixes, and there a cache is a cache).
# ---------------------------------------------------------------------------
_FACE_LIVE_STATE = os.environ.get("ALPHAGRAD_FACE_LIVE_STATE", "1") != "0"
_LIVE_CHAIN_CAP = int(os.environ.get("ALPHAGRAD_FACE_LIVE_CHAINS", "8"))
_LIVE_CHAINS: list = []
# `elims` is the asymptotic quantity: T per env per episode is O(T),
# T(T+1)/2 per env is O(T^2). `restart` counts mid-episode cold rebuilds --
# the failure mode this design exists to make impossible, so a nonzero value
# is a bug report, not noise.
_LIVE_CHAIN_STATS = {"step": 0, "elims": 0, "cold": 0, "restart": 0,
                     "evict": 0, "poison": 0}


class _ElimChain:
    """One env's live elimination state: `ij` has consumed `n` vertices and
    `out` holds their `{face_key: slots|SKIP_FACE}` dicts."""

    __slots__ = ("base", "ij", "out", "n", "okey", "skey", "fsig")

    def __init__(self, base, ij):
        self.base = base
        self.ij = ij
        self.out: dict[int, dict] = {}
        self.n = 0
        self.okey = b""
        self.skey = b""
        self.fsig: tuple = ()


def consume_live_chain_stats() -> dict:
    out = dict(_LIVE_CHAIN_STATS)
    for k in _LIVE_CHAIN_STATS:
        _LIVE_CHAIN_STATS[k] = 0
    return out


def _live_face_transforms(config, consts, args, o_list, specs_list,
                          face_rows_list, face_skips_list, wire_sig,
                          face_joins_list=None):
    """`_face_transforms_for_order` served from live state. Same result."""
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    K = len(o_list)
    base = (id(config.jaxpr), tuple(config.argnums))
    if wire_sig is None:
        wire_sig = _face_wire_keys(
            np.asarray(face_rows_list), np.asarray(face_skips_list), K,
            None if face_joins_list is None else np.asarray(face_joins_list))
    _ord = np.ascontiguousarray(np.asarray(o_list, dtype=np.int64))
    _sp = np.ascontiguousarray(np.asarray(specs_list, dtype=np.int64))
    okey, skey = _ord.tobytes(), _sp.tobytes()
    _ob, _sb = _ord.itemsize, int(_sp[0].nbytes)

    # LONGEST chain whose consumed prefix is a prefix of this request. A fixed
    # per-step stride makes a BYTE prefix exactly an ARRAY prefix.
    ch = None
    for c in _LIVE_CHAINS:
        if (c.base == base and c.n <= K
                and okey.startswith(c.okey) and skey.startswith(c.skey)
                and c.fsig == wire_sig[:c.n]
                and (ch is None or c.n > ch.n)):
            ch = c
    _LIVE_CHAIN_STATS["step"] += 1
    if ch is None:
        _LIVE_CHAIN_STATS["cold"] += 1
        if K > 1:
            _LIVE_CHAIN_STATS["restart"] += 1
        ch = _ElimChain(base, IncrementalJaxpr(
            config.jaxpr, tuple(config.argnums), list(consts), list(args),
            track_faces=False))
        _LIVE_CHAINS.append(ch)
        while len(_LIVE_CHAINS) > _LIVE_CHAIN_CAP:
            _LIVE_CHAINS.pop(0)
            _LIVE_CHAIN_STATS["evict"] += 1
    else:
        _LIVE_CHAINS.remove(ch)          # LRU: most recently used last
        _LIVE_CHAINS.append(ch)

    try:
        while ch.n < K:
            k = ch.n
            v = int(o_list[k])
            per_face = _face_dict_for_vertex(
                config, ch.ij, v, face_rows_list[k], face_skips_list[k],
                face_join=(None if face_joins_list is None
                           else face_joins_list[k]))
            if per_face:
                ch.out[v] = per_face
            vertex_rules = decode_vertex_rule_specs(
                config.jaxpr, v, specs_list[k])
            ch.ij.eliminate(
                v,
                (make_live_masked_hook(tuple(vertex_rules)),)
                if vertex_rules else (),
                ch.out.get(v))
            ch.n += 1
            ch.okey = okey[:ch.n * _ob]
            ch.skey = skey[:ch.n * _sb]
            ch.fsig = wire_sig[:ch.n]
            _LIVE_CHAIN_STATS["elims"] += 1
    except BaseException:
        # A half-applied elimination leaves the trace inconsistent; drop the
        # chain so the next step rebuilds instead of extending the damage.
        if ch in _LIVE_CHAINS:
            _LIVE_CHAINS.remove(ch)
        _LIVE_CHAIN_STATS["poison"] += 1
        raise
    # the chain keeps growing -- hand back a snapshot, like the cache does
    return dict(ch.out)


def _face_transforms_for_order(config, consts, args, o_list, specs_list,
                               face_rows_list, face_skips_list,
                               wire_sig=None, face_joins_list=None):
    """Per-vertex ``face_transforms`` dicts for graphax, from the wire arrays.

    Face KEYS are graph-state dependent, so enumerate with ``faces_of`` on a
    structural replay that applies the SAME per-vertex rules and face
    transforms the measurement will — the k-th vertex's keys are only valid on
    the graph produced by the first k-1 (transformed) eliminations. Slots wrap
    their decoded rules in ``make_live_masked_hook`` (a rule illegal on ITS
    operand is skipped per-slot, never raises). The hooks DO carry the
    ``_PER_FACE_STATS`` sink, but gated: this replay invokes them on a graph
    nothing is measured on, so it must not count — only the armed scope in
    ``_do_compile_approx`` does. ``face_skips`` rows become
    ``graphax.SKIP_FACE``.
    """
    from graphax import SKIP_FACE, faces_of
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    if _FACE_LIVE_STATE and len(o_list):
        return _live_face_transforms(
            config, consts, args, o_list, specs_list, face_rows_list,
            face_skips_list, wire_sig, face_joins_list)

    ij = None
    out: dict[int, dict] = {}
    _start = 0
    _cache_key = None
    if os.environ.get("ALPHAGRAD_FACE_ENUM_CACHE", "0") == "1":
        # Pop-and-extend prefix cache: step k's replay is step k-1's replay
        # plus ONE elimination, so advance the cached builder instead of
        # rebuilding it from scratch every env step — O(T) eliminations per
        # episode instead of O(T^2). Sound for EVERY prefix now that the
        # COMPRESS `is_last` gate is gone and a vertex's decode no longer
        # depends on where the prefix ends — same reason as
        # _INCR_STREAM_CACHE. Extension MUTATES the builder, so the parent
        # entry is POPPED; a sibling chain that misses takes the honest
        # cold replay.
        _t_key = time.perf_counter()
        _sigs = tuple(
            (int(o_list[k]),
             np.asarray(specs_list[k], dtype=np.int64).tobytes(),
             np.asarray(face_rows_list[k], dtype=np.int64).tobytes(),
             np.asarray(face_skips_list[k], dtype=np.int64).tobytes())
            for k in range(len(o_list)))
        _prof_add("cb.face_enum_key", time.perf_counter() - _t_key)
        _FACE_ENUM_STATS["calls"] += 1
        _base = (id(config.jaxpr), tuple(config.argnums))
        # No COMPRESS carve-out: the decode is position-independent, so every
        # prefix is a legal parent and every result is storable.
        _cache_key = _base + (_sigs,)
        for cut in range(len(_sigs) - 1, 0, -1):
            parent = _FACE_ENUM_CACHE.pop(_base + (_sigs[:cut],), None)
            if parent is not None:
                ij, out, _start = parent[0], parent[1], cut
                _FACE_ENUM_STATS["ext"] += 1
                break
        if ij is None:
            _FACE_ENUM_STATS["cold"] += 1
    if ij is None:
        _FACE_ENUM_STATS["build"] += 1
        ij = IncrementalJaxpr(config.jaxpr, tuple(config.argnums),
                              list(consts), list(args), track_faces=False)
    _FACE_ENUM_STATS["elims"] += len(o_list) - _start
    for k in range(_start, len(o_list)):
        v = int(o_list[k])
        per_face = _face_dict_for_vertex(
            config, ij, v, face_rows_list[k], face_skips_list[k],
            face_join=(None if face_joins_list is None
                       else face_joins_list[k]))
        if per_face:
            out[v] = per_face
        vertex_rules = decode_vertex_rule_specs(
            config.jaxpr, v, specs_list[k])
        # Hook-wrap like the measurement/tokenizer paths do under per_face:
        # a raw rule that doesn't fit one face's operand would hit the strict
        # TRANSFORM-DID-NOT-FIT guard here — DURING KEY ENUMERATION — and
        # kill the callback before measurement even starts (3b smoke: a
        # policy-proposed Compress legal by the logical-axis oracle but
        # unappliable on an implicit-dim edge's 1-D val).
        ij.eliminate(
            v,
            (make_live_masked_hook(tuple(vertex_rules)),) if vertex_rules else (),
            out.get(v),
        )
    if _cache_key is not None:
        # FIFO eviction (dict preserves insertion order): a clear-all here
        # would cost one full cold replay PER CONCURRENT ENV CHAIN on the
        # next step; evicting the oldest entries only sheds finished chains.
        while len(_FACE_ENUM_CACHE) >= _FACE_ENUM_CACHE_CAP:
            _FACE_ENUM_CACHE.pop(next(iter(_FACE_ENUM_CACHE)))
        _FACE_ENUM_CACHE[_cache_key] = (ij, out)
        # the cached dict keeps growing on extension — hand back a snapshot
        return dict(out)
    return out


def _decode_vertex_transforms(config, o_list, specs_list):
    """``(transforms, tok_rules_by_v)`` for one order and its per-vertex rows.

    Each row in ``sparsity_specs`` is ``[base_idx1, base_idx2, factor]``; the
    decode resolves the logical axis indices and the legacy -1 (gcd) / 0
    (drop) / 1 (no-op) sentinels into explicit ``Diag(i, j, factor)`` entries
    with a strictly positive integer factor, which is the only form graphax's
    ``apply_diag`` accepts. Slots with factor 0 (legacy drop-axes) or 1
    (legacy no-op) are skipped: drop has no replacement under the new API and
    no-op is dead weight. A rule must also fit EVERY non-literal invar of the
    equation, because graphax's ``_eliminate_vertex`` applies each transform
    to every incoming edge and ``apply_diag`` raises when the primal axis
    index is out of range for any of them (a division by a scalar denominator
    has one ``(n,)`` edge and one ``()`` edge, and a Diag with ``j=1`` only
    fits the first).

    MODULE LEVEL because it is called TWICE: once on the policy's graph, and
    once on the program a plan's carry container implies, which is a
    different jaxpr with a different vertex numbering.
    """
    transforms: list[tuple[int, tuple]] = []
    tok_rules_by_v: dict[int, tuple] = {}
    for v_idx, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            config.jaxpr, int(v), specs_list[v_idx])
        # TOKENIZER-side rules are the SAME rules: the decode no longer
        # depends on the vertex's position in the prefix, so the observation
        # and the measured graph cannot disagree about a COMPRESS.
        tok_rules = rules
        if rules or tok_rules:
            if getattr(config, "per_face", False):
                # graphax invokes a CALLABLE transform once per face, handing
                # it that face's live operand, so this is where per-path
                # legality is decided. Rules that do not fit a given face are
                # skipped for that face only, not for the whole vertex.
                from alphagrad.approx.common.masks import make_live_masked_hook
                _face_stats = _PER_FACE_STATS
                if rules:
                    transforms.append(
                        (int(v),
                         (make_live_masked_hook(rules, stats=_face_stats,
                                                gated=True),))
                    )
                # The tokenizer eliminates its OWN graph copy with equivalent
                # hooks but no stats sink: the measured graph's hooks own the
                # applied and skipped counters.
                if tok_rules:
                    tok_rules_by_v[int(v)] = (
                        make_live_masked_hook(tok_rules),)
            else:
                if rules:
                    transforms.append((int(v), tuple(rules)))
                if tok_rules:
                    tok_rules_by_v[int(v)] = tuple(tok_rules)
    return transforms, tok_rules_by_v


def _callback(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_specs,
    face_specs,
    face_skips,
    stop,
    *eval_samples,
    init: bool = False,
    face_joins=None,
):
    """THE PLAN LOG'S LAST GUARANTEE: a counted terminal is a record.

    :func:`_callback_measured` counts the terminal at the top and writes its
    record at the bottom, and its four deliberate refusals each write one on
    the way out. An EXCEPTION between the two writes nothing, so the trainer
    reads `pool_terminals=N ... wrote=0` and cannot tell a crashed plan from
    a plan that never ran. Measured 2026-09-13 (job 65339): a TLM arm whose
    observation delta overflowed `MAX_DELTA_TOKENS` raised inside the
    tokenizer, and four counted terminals left one record.

    This wrapper records whatever the failed call counted and RE-RAISES. It
    never swallows: the raise is the apparatus telling the operator to fix
    the configuration, and that ruling stands.
    """
    _t0 = int(_PLAN_LOG_TERMINALS[0])
    _r0 = len(_PLAN_RECORDS)
    try:
        return _callback_measured(
            config, args, consts, order, sparsity_specs, face_specs,
            face_skips, stop, *eval_samples, init=init, face_joins=face_joins)
    except BaseException as _exc:
        # THE RATE IS TELEMETRY, WHATEVER THE PLAN LOG IS DOING. A raised
        # terminal is a refused measurement; the trainer excludes the whole
        # environment from the update, so without this counter the exclusion
        # would be silent.
        #
        # EVERY RAISE FROM IN HERE IS THE PLAN'S NOW. It used to be necessary
        # to separate the gradient oracle's own float64 compile failing from
        # the plan failing, because the oracle ran inside this callback. It
        # does not any more (owner ruling 2026-09-18), so there is one source
        # again and `raised:` names it.
        _reason = f"raised:{type(_exc).__name__}"
        if int(_PLAN_LOG_TERMINALS[0]) > _t0:
            _record_refusal(_reason)
        # Only when THIS call counted a terminal and wrote nothing for it.
        if int(_PLAN_LOG_TERMINALS[0]) > _t0 and len(_PLAN_RECORDS) == _r0:
            try:
                _n = int(np.asarray(order).reshape(-1).shape[0]
                         if stop is None else int(stop))
                _record_terminal_plan(
                    order=[int(x) for x in
                           np.asarray(order).reshape(-1)[:_n].tolist()],
                    rule_specs=np.asarray(sparsity_specs)[:_n],
                    face_specs=np.asarray(face_specs)[:_n],
                    face_skips=np.asarray(face_skips)[:_n],
                    face_joins=(None if face_joins is None
                                else np.asarray(face_joins)[:_n]),
                    reward_vec=_SENTINEL_BAD_REWARD,
                    face_before=None, face_after=_PER_FACE_STATS,
                    counts_from_trace=False,
                    refused=_reason)
            except Exception:
                pass          # a logging failure must not mask the real one
        raise


def _callback_measured(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_specs,
    face_specs,
    face_skips,
    stop,
    *eval_samples,
    init: bool = False,
    face_joins=None,
):
    """Stage A reward harness: returns `(tokens, rewards)` where `rewards` is
    the canonical `(NUM_REWARDS,)` float32 vector documented at the top of this
    file. Every component is computed every (non-init) call, except `latency`
    which is gated behind `config.measure_latency`. When
    `config.terminal_rewards_only` is on, intermediate steps return tokens
    only — every reward component is zeroed so the heavy jacve compile/exec
    is skipped entirely until the elimination order is complete.
    """
    _pf_last = [time.perf_counter()]

    def _pf(key):
        now = time.perf_counter()
        _prof_add(key, now - _pf_last[0])
        _pf_last[0] = now

    partial_order, partial_specs = _get_partials(order, sparsity_specs, stop)
    is_terminal = int(stop) >= len(order)
    # SPARSITY (reward slot 10). Read ONCE per callback, like `_fid_on`
    # -- but bound HERE, not beside it, because
    # `_do_compile_approx` closes over it and is CALLED long before that
    # point. Terminal only: mid-rollout the elimination is a prefix, and
    # slots 6/7/8 already establish that convention.
    _sparsity_on = bool(is_terminal) and sparsity_enabled()
    # A6 PLAN LOG (``ALPHAGRAD_PLAN_LOG``), terminal only for the same reason
    # slots 6/7/8 are. A snapshot of ``_PER_FACE_STATS`` is taken HERE and
    # differenced at the return, because it is process-global and only a
    # per-callback delta can be attributed to this plan. `_plan_traced` is set by
    # `_do_compile_approx`, which runs only on a compile-cache MISS -- the
    # exact condition under which the per-face counters saw this plan.
    _plan_log_on = bool(is_terminal) and plan_log_enabled()
    if _plan_log_on:
        _PLAN_LOG_TERMINALS[0] += 1
    # THE COST FORM (ticket .9): under ``paired-log`` the terminal step
    # measures rev-exact beside the candidate and the two cost slots become
    # log-differences (see `_PAIRED_REF`). Terminal only, like slots 6/8/10:
    # a partial order's cost against rev-exact is not a statement about a
    # plan, so non-terminal steps carry 0.0 in slots 2 and 5 under this form
    # (training uses terminal rewards strictly; ruling 2026-09-01).
    _paired = bool(is_terminal) and cost_form() == "paired-log"
    _paired_ref_rec = None
    _plan_pf0 = dict(_PER_FACE_STATS) if _plan_log_on else None
    _plan_traced = [False]

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()  # list of MAX_RULES x 3 lists
    _pf("cb.wire2py")

    # P1 per-path actions: build graphax's {vertex: {face_key: slots|SKIP}}
    # only when any face action is present in the prefix (all -1 / all 0 is
    # the per-vertex mode and must stay byte-identical to it).
    _faces_np = np.asarray(face_specs)[: len(o_list)]
    _skips_np = np.asarray(face_skips)[: len(o_list)]
    # --approx-add choose only; None under every fixed value (EnvState).
    _joins_np = (None if face_joins is None
                 else np.asarray(face_joins)[: len(o_list)])

    def _log_refused(reason: str, reward_vec):
        """Record a terminal plan the callback REFUSES to measure.

        Every ``return`` between the terminal counter above and the record
        at the bottom of this function used to drop the record while the
        counter had already fired, so the trainer saw
        ``pool_terminals=16 ... wrote=0`` and the log of a whole campaign was
        empty (canary job 65319, 2026-09-13). A refusal is a plan-log record
        like any other, marked ``refused`` and ``sentinelled``.

        THE COUNT COMES FIRST and is independent of the plan log: the
        refusal RATE has to be readable from a run with ``--plan-log`` off,
        because it is the rate at which the trainer is now dropping whole
        environments from the update. It is taken on the SAME predicate the
        trainer excludes on -- every cost channel at the sentinel -- so the
        two can never disagree about what was refused. ``no-target-fun``
        returns a partial reward vector rather than the sentinel, the
        trainer keeps that environment, and it is therefore not counted."""
        if bool(np.all(
                np.asarray(reward_vec)[list(COMPUTE_REWARD_INDICES)]
                <= SENTINEL_COST * 0.99)):
            _record_refusal(reason)
        if not _plan_log_on:
            return
        _record_terminal_plan(
            order=o_list, rule_specs=partial_specs,
            face_specs=_faces_np, face_skips=_skips_np,
            face_joins=_joins_np,
            reward_vec=reward_vec,
            face_before=_plan_pf0, face_after=_PER_FACE_STATS,
            counts_from_trace=False, refused=reason)
    ft_by_vertex = None
    _have_face_actions = bool(
        len(o_list) and (np.any(_skips_np == 1)
                         or np.any(_faces_np[..., 0] >= 0)
                         or np.any(_faces_np[..., 0] == COMPRESS_SENTINEL)
                         or np.any(_faces_np[..., 0] == QUANT_SENTINEL)))
    # ALPHAGRAD_UNIFIED_FACE_ENUM=1 (+ incremental tokens): face keys are
    # enumerated on the TOKENIZER's IncrementalJaxpr inside
    # _incremental_stream_tokens — the standalone replay below is skipped
    # and this phase's time moves into cb.tokenize.
    _unified_fe = (os.environ.get("ALPHAGRAD_UNIFIED_FACE_ENUM", "0") == "1"
                   and os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
                   == "1")
    # ONE sparse per-vertex encoding of the face wires per callback,
    # shared by the live elimination state below and by the stream
    # cache's `face_key` further down -- they used to build one each.
    _wire_sig = None
    if _have_face_actions:
        _wire_sig = _face_wire_keys(_faces_np, _skips_np, len(o_list),
                                    _joins_np)
    _pf("cb.face_wire_sig")
    if _have_face_actions and not _unified_fe:
        # NUMPY, not `.tolist()`: `_face_dict_for_vertex` pulls the three
        # wire ints out of `face_row[f][s]` explicitly and reads
        # `face_skip[f]` scalar-wise, so it never needed Python lists --
        # and materialising T x MAX_FACES x FACE_SLOTS x 3 Python ints
        # per callback is O(T x MAX_FACES) per step and
        # O(T^2 x MAX_FACES) per episode, >99% of it -1 padding (measured
        # live occupancy is ~1.24 faces per vertex). Same argument, same
        # fix as the `_fe_inline` wires below already use.
        ft_by_vertex = _face_transforms_for_order(
            config, consts, args, o_list, specs_list,
            _faces_np, _skips_np, wire_sig=_wire_sig,
            face_joins_list=_joins_np,
        )
    _pf("cb.face_enum")

    # Build the per-vertex `transforms` sequence consumed by graphax's
    # typed-transform API. Each row in `sparsity_specs` is
    # ``[base_idx1, base_idx2, factor]``; we resolve the logical axis
    # indices and the legacy -1 (gcd) / 0 (drop) / 1 (no-op) sentinels
    # into explicit Diag(i, j, factor) entries with a strictly positive
    # integer factor — the only form graphax's apply_diag accepts. Slots
    # with factor=0 (legacy drop-axes) or factor=1 (legacy no-op) are
    # silently skipped: drop has no replacement under the new API, and
    # no-op is dead weight. The rule must also fit **every** non-literal
    # invar of the eqn: graphax's `_eliminate_vertex` applies each
    # transform to every incoming edge, and apply_diag raises if the
    # primal axis index is out of range for any of them (e.g. a div by
    # a scalar denominator has one (n,) edge and one () edge — a Diag
    # with j=1 only fits the first).
    transforms, tok_rules_by_v = _decode_vertex_transforms(
        config, o_list, specs_list)

    _pf("cb.decode")
    if os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "1") == "1":
        # Append-only observation (spec): the stream grows by one block per
        # elimination and the whole extract_jaxpr re-trace of the Jacobian is
        # skipped. eqn_ids come from the tokenizer's own segment record —
        # one stream-global id per contraction/approx group, -1 elsewhere —
        # which is exactly the relational-gate contract (same/earlier/later
        # comparisons, no embedding-table bound).
        _fe_inline = _unified_fe and _have_face_actions
        stream, seg_ids, _ft_ret, _last_start = _incremental_stream_tokens(
            config, consts, args, o_list, specs_list, tok_rules_by_v,
            ft_by_vertex=ft_by_vertex,
            # NUMPY, not `.tolist()`: only the ~1.24 LIVE faces of the
            # CURRENT vertex are ever indexed out of these, so materialising
            # T x MAX_FACES x FACE_SLOTS x 3 Python ints per step was pure
            # padding cost (O(T^2) per episode).
            face_rows_list=_faces_np if _fe_inline else None,
            face_skips_list=_skips_np if _fe_inline else None,
            face_joins_list=_joins_np if _fe_inline else None,
            # PER-STEP signatures (not one whole-prefix blob) so the stream
            # cache can find the parent at every ancestor cut under face
            # actions instead of replaying the whole prefix cold each step.
            face_key=_wire_sig
            if (ft_by_vertex is not None or _fe_inline) else None,
        )
        if _fe_inline:
            # measurement sites below read the same dict the eliminations
            # actually applied
            ft_by_vertex = _ft_ret or None
        _record_token_length(len(stream))
        if config.delta_obs:
            if init:
                raise ValueError(
                    "delta_obs=True: the BASE stream is delivered by "
                    "VertexEliminationEnv.base_observation() (host-side, "
                    "exact length, order-independent), never through the "
                    "reset callback -- clipping the base to "
                    "MAX_DELTA_TOKENS would desync the encoder's recurrence "
                    "for the whole episode."
                )
            tokens = _delta_observation(stream, _last_start)
            # DELTA WIRE: no equation ids exist on this path at all, so the
            # callback's output tuple is one element shorter. `_wire` below
            # is what every `return` in this function goes through.
            eqn_ids = None
        else:
            _record_tokenization_truncation(len(stream))
            tokens = jnp.asarray(stream[:MAX_TOKENS], dtype=jnp.int32)
            tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))
            _ids = np.full((MAX_TOKENS,), -1, dtype=np.int32)
            _n_ids = min(len(seg_ids), MAX_TOKENS)
            _ids[:_n_ids] = np.asarray(seg_ids[:_n_ids], dtype=np.int32)
            eqn_ids = jnp.asarray(_ids)
    else:
        if config.delta_obs:
            raise ValueError(
                "delta_obs=True needs ALPHAGRAD_INCREMENTAL_TOKENS=1: the "
                "per-step DELTA only exists under the append-only "
                "tokenizer; the extract_jaxpr path re-tokenizes the whole "
                "Jacobian and has no notion of a delta."
            )
        ve = extract_jaxpr(
            config.jaxpr,
            config.argnums,
            o_list,
            config.sparse,
            args,
            consts,
            transforms=transforms,
        )
        # Measure the raw token length before slicing so we can detect
        # truncation. ``_record_tokenization_truncation`` is a no-op for
        # short sequences (the common case) and is cheap otherwise.
        raw_tokens = ve.tokenized()
        _record_token_length(int(raw_tokens.shape[0]))
        _record_tokenization_truncation(int(raw_tokens.shape[0]))
        tokens = raw_tokens[:MAX_TOKENS]
        tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

        # Compute per-token equation IDs once for the relational-bias encoder
        # (Stage B.1). Cheap (single Python scan over a length-≤4096 numpy
        # array) and adds (MAX_TOKENS,) int32 to EnvState.
        tokens_np = np.asarray(tokens)
        eqn_ids_np = compute_eqn_ids_from_tokens(tokens_np, _TOKEN_VOCAB)
        eqn_ids = jnp.asarray(eqn_ids_np, dtype=jnp.int32)

    def _wire(_t, _e, _r):
        """THE CALLBACK'S OUTPUT TUPLE, in one place.

        ``(tokens, reward)`` under ``delta_obs`` -- there are no equation ids
        on that path -- and ``(tokens, eqn_ids, reward)`` on the LEGACY
        full-stream path, whose ids still feed the dense encoder's pairwise
        T5 relational bias. ``VertexEliminationEnv._callback_shape`` declares
        the matching arity, and `io_callback` enforces it.
        """
        if config.delta_obs:
            if _e is not None:
                raise RuntimeError(
                    "delta_obs produced equation ids; they were removed on "
                    "2026-09-13 and nothing may reintroduce them silently.")
            return _t, _r
        return _t, _e, _r

    _pf("cb.tokenize")
    if init:
        return _wire(tokens, eqn_ids,
                     jnp.zeros(NUM_REWARDS, dtype=jnp.float32))

    # Terminal-only fast path: every reward component is sparse — only the
    # final step (when the elimination order is complete) gets a non-zero
    # signal, so we skip the expensive jacve compile/exec on every prior
    # step. Cumsum-style returns (alpha0/mu0) and per-rollout aggregations
    # (gdpo) collapse to the terminal reward; gfn already reads only the
    # last step. PPO sees a sparse-reward MDP, which GAE handles natively.
    if config.terminal_rewards_only and not is_terminal:
        _pf("cb.nonterm_tail")
        return _wire(tokens, eqn_ids,
                     jnp.zeros(NUM_REWARDS, dtype=jnp.float32))

    # ------------------------------------------------------------------
    # PER-EPISODE DEDUPLICATION (owner ruling 2026-09-14). See _PLAN_DEDUPE.
    # ------------------------------------------------------------------
    # Placed HERE, after the tokens exist and before anything is compiled or
    # executed. The tokens are the next observation and must be produced for
    # every env whatever happens; everything below this point is measurement,
    # and for a plan this episode has already measured it would produce the
    # same numbers a second time.
    # The episode accounting runs whether or not the cache is armed, because
    # it is also what prints the per-episode seconds-per-plan line.
    _dedupe_key = None
    _plan_index = -1
    if is_terminal:
        _roll_measure_episode(_episode_measure_key(eval_samples))
        _plan_index = int(_MEASURE_EPISODE["n_plans"])
        _MEASURE_EPISODE["n_plans"] = _plan_index + 1
    # NO EVAL SAMPLES, NO CACHE. The ruling keys the cache on the plan AND on
    # the episode's eval-sample key, and the samples are the only thing that
    # tells one episode's measurement from another's when nobody publishes an
    # episode index. A configuration that measures on the fixed `args` (every
    # probe and every direct caller of `_callback`) therefore never dedupes:
    # a cache that could not be bounded to an episode would live for the whole
    # process and turn a deliberate re-measurement into a replay.
    if is_terminal and measure_dedupe_enabled() and eval_samples:
        _dedupe_key = _plan_content_key(
            o_list, partial_specs, _faces_np, _skips_np, _joins_np)
        _hit = _PLAN_DEDUPE.get(_dedupe_key)
        if _hit is not None:
            _from_idx, _hit_slots = _hit
            if _plan_log_on:
                _record_terminal_plan(
                    order=o_list, rule_specs=partial_specs,
                    face_specs=_faces_np, face_skips=_skips_np,
                    face_joins=_joins_np,
                    reward_vec=_hit_slots,
                    face_before=_plan_pf0, face_after=_PER_FACE_STATS,
                    counts_from_trace=False,
                    measured_from=_from_idx)
            _pf("cb.dedupe_hit")
            return _wire(tokens, eqn_ids,
                         jnp.array(_hit_slots, dtype=jnp.float32))

    # ------------------------------------------------------------------
    # THE CARRY CONTAINER, PER PLAN (owner rulings 2026-09-16, A and B).
    # ------------------------------------------------------------------
    # Placed HERE, after the tokens and before anything is compiled, because
    # that is exactly the seam the rulings describe: the POLICY sees the step
    # body with the dense carry edge (the tokens above are built from
    # `config`, always), and the MEASUREMENT compiles the consistent
    # recursion -- the plan applied at every step of the prefix or the suffix
    # -- with the carry in the container that choice implies.
    #
    # Everything below this point runs on the swapped program. The plan
    # RECORD keeps the policy's own order and wires, which is what the reader
    # of the log needs: the plan the policy emitted, plus the container it
    # implied.
    # A PLAN TRAVELS BY POSITION, NOT BY FACE KEY. A face key is a pair of
    # stable var indices on the LIVE graph, and every elimination rewires it,
    # so the keys a body vertex shows depend on what the carry block left
    # behind -- which is exactly what the container changes. The wire arrays
    # move to the transported order's positions and the faces are enumerated
    # again on the variant's own replay, the way the policy's graph enumerates
    # them.
    _rec_order = o_list
    _carry_container = None
    if is_terminal and _carry.armed():
        _carry_container = _carry.container_for_plan(
            config, o_list, _faces_np, _skips_np, partial_specs)
        _variant = _carry.measurement_env(_carry_container)
        if _variant is not None:
            (o_list, _m_specs, _m_faces, _m_skips, _m_joins) = \
                _carry.transport_wires(
                    o_list, _variant, partial_specs, _faces_np, _skips_np,
                    _joins_np)
            config = _variant["config"]
            args = _variant["args"]
            consts = _variant["consts"]
            specs_list = _m_specs.tolist()
            transforms, _ = _decode_vertex_transforms(
                config, o_list, specs_list)
            _m_have = bool(
                len(o_list) and (np.any(_m_skips == 1)
                                 or np.any(_m_faces[..., 0] >= 0)
                                 or np.any(_m_faces[..., 0] == COMPRESS_SENTINEL)
                                 or np.any(_m_faces[..., 0] == QUANT_SENTINEL)))
            ft_by_vertex = (
                _face_transforms_for_order(
                    config, consts, args, o_list, specs_list,
                    _m_faces, _m_skips,
                    wire_sig=_face_wire_keys(_m_faces, _m_skips, len(o_list),
                                             _m_joins),
                    face_joins_list=_m_joins)
                if _m_have else None)
            # THE VARIANT'S OWN EVAL SAMPLES. They cannot be the base ones --
            # the shapes of the given values move with the container -- so
            # they are drawn from a DIGEST of the base draw, which every
            # process that measures this plan computes the same way and which
            # moves per episode exactly as the base draw does.
            if eval_samples:
                eval_samples = tuple(
                    _carry.eval_samples_for(_carry_container, eval_samples))
    _PLAN_CARRY[0] = _carry_container
    _pf("cb.carry_container")

    # ------------------------------------------------------------------
    # Compute family — graphax counters (always) → muls_adds_fmas, max_io_sum.
    # ------------------------------------------------------------------
    # Phase breadcrumb (ALPHAGRAD_DEBUG_MEASURE=1): printed BEFORE the two
    # heavy host phases (count pass here, XLA compile below) so a wedged
    # process's log names the phase AND the plan. v15 post-mortem (job
    # 55814): 234 GB RSS, GPUs idle, zero log output — undiagnosable.
    _dbg_measure = os.environ.get("ALPHAGRAD_DEBUG_MEASURE", "0") == "1"
    if _dbg_measure:
        print(f"[measure] count-pass step={int(stop)} order={o_list}",
              flush=True)
    # THE COUNT PASS IS OPTIONAL (ALPHAGRAD_SKIP_COUNT_OPS=1).
    #
    # `vertex_elimination_jaxpr(count_ops=True)` symbolically REPLAYS the whole
    # elimination just to total up adds/muls/fmas and max-IO. Its cost scales
    # with how much the plan actually computes, so it is free while the policy
    # is degenerate and ruinous once it is not: measured 0.2s/episode in v17
    # (zero-work plans) versus 259s/episode in v18 (2.4e12 muls) — 73% of the
    # entire episode, against a 16.6s XLA compile.
    #
    # It buys exactly two LOG-ONLY reward channels (muls_adds_fmas, max_io_sum)
    # plus the op-count cap. The trained objective is latency + peak_memory +
    # frob, none of which touch it, so with this flag we simply do not pay for
    # it. The elimination still gets traced once by the compile below — this
    # was the second, redundant walk of the same graph.
    #
    # CONSEQUENCE, stated plainly: the MULS cap goes inert (there is nothing to
    # compare), so the pre-XLA guard against compile-monster plans is gone. The
    # OOM handler still catches device exhaustion, but NOT the v15-style
    # compile hang. Re-enable the count pass if that reappears.
    if skip_count_ops():
        muls_adds_fmas = 0.0
        max_io_sum = 0.0
    else:
        _, aux = vertex_elimination_jaxpr(
            config.jaxpr,
            o_list,
            consts,
            *args,
            argnums=config.argnums,
            count_ops=True,
            sparse_representation=config.sparse,
            transforms=transforms,
            face_transforms=ft_by_vertex,
        )
        muls_adds_fmas = float(aux["adds"] + aux["muls"] + aux["fmas"])
        max_io_sum = float(aux["mem"])
    _pf("cb.count_pass")

    # SOFT SENTINEL (v16 post-mortem, ep-39 cliff). The hard ±1e10 sentinel
    # plus the trainer's "degenerate steps are NEUTRAL (advantage 0)" rule
    # created an absorbing basin: once the value baseline rises, a measured-
    # but-mediocre plan earns a NEGATIVE advantage while a degenerate plan
    # earns exactly 0 — degeneracy becomes the safe haven and the policy
    # ratchets into zero-work compress/quant spam (observed: all-16 plans
    # no_work, frob=1, ~78 quants + ~14 compresses, WITHOUT PopArt, so not
    # a normaliser artifact). A refused plan now reports FINITE, realistic-
    # worst channel values — strictly worse than any real plan (100 ms
    # latency, 2 GB peak, frob 1) yet only a bounded symlog step below the
    # measured population, so it takes an ORDINARY negative advantage and
    # trains the policy AWAY instead of sheltering it. The magnitudes stay
    # far under the trainer's sentinel-detection threshold (|5e9|), so the
    # advantage-neutralisation branch never fires for them.
    # ALPHAGRAD_DEGEN_SOFT=0 restores the hard sentinel.
    def _truncated_reward():
        """Reward for a plan TRUNCATED BY A RESOURCE LIMIT (op-count cap, OOM).

        This is the "Time Limits in Reinforcement Learning" (Pardo et al.,
        2018) distinction, applied to resource limits instead of clocks.

        An OOM or an op-count refusal is NOT an outcome of the MDP — it is the
        experimental apparatus giving up. The plan might have been excellent;
        we simply failed to evaluate it. Scoring it (well OR badly) teaches the
        policy something we did not measure:

          * score it BADLY  -> the policy learns to avoid a region for reasons
            that have nothing to do with the objective. v16 did this: the cap
            sat BELOW the cost of every exact plan, so "do the computation
            correctly" was ranked the single worst action available.
          * score it WELL   -> a free lunch: refuse to compute, collect reward.
          * score it CONSTANT -> what I shipped yesterday; once every env is
            refused they all score identically, the advantage is uniformly
            zero, and the run freezes (v17, 22 episodes).

        The correct treatment is to BOOTSTRAP: emit the exact sentinel vector,
        which `train_episode`'s `_is_degen` recognises (all six cost channels
        at SENTINEL_COST) and which causes the transition to be EXCLUDED from
        the gradient — advantage forced to 0 and the value target replaced by
        the critic's own prediction, so neither the actor nor the critic
        trains on a number we never measured. That is partial-episode
        bootstrapping: "we stopped here for our own reasons, assume the
        critic's estimate."
        """
        return _SENTINEL_BAD_REWARD

    # WEDGE GUARD (v15 post-mortem): with the cos-sentinel gone, genuinely
    # huge approximated graphs go all the way to XLA — one episode-6 plan
    # wedged the host at 234 GB RSS with all GPUs idle (multi-thread compile
    # spin, the v10-style stall). The count pass has already run here, so a
    # symbolic-op ceiling refuses the monster BEFORE the compile. Healthy
    # v15 plans measured ~4e12 muls; the default cap only fires on true
    # blowups.
    _muls_cap = float(os.environ.get("ALPHAGRAD_MULS_SENTINEL_CAP", "5e13"))
    if not skip_count_ops() and muls_adds_fmas > _muls_cap:
        _record_truncated_plan()
        if _dbg_measure or os.environ.get("ALPHAGRAD_DEBUG_DEGEN", "0") == "1":
            print(f"[trunc] MULS-CAP muls={muls_adds_fmas:.3g} > "
                  f"{_muls_cap:.3g} step={int(stop)} order={o_list} "
                  f"(excluded from gradient)", flush=True)
        _tr = _truncated_reward()
        _log_refused("muls-cap", _tr)
        return _wire(tokens, eqn_ids, _tr)

    # If no `target_fun` is supplied, we can't compile/execute. Skip every
    # execution-derived metric and return a partial reward vector.
    if config.target_fun is None:
        # cosine_sim stays 0.0, NOT 1.0. Without a target function nothing is
        # compiled or executed, so there is no fidelity measurement to report;
        # 1.0 handed a plan that computed nothing a PERFECT score on the only
        # quality channel. (tests/test_all_cost_channels.py has asserted this
        # since 2026-05-24; the constant below had drifted back to 1.0.)
        rewards = jnp.zeros(NUM_REWARDS, dtype=jnp.float32)
        rewards = rewards.at[REWARD_INDEX["muls_adds_fmas"]].set(-muls_adds_fmas)
        rewards = rewards.at[REWARD_INDEX["max_io_sum"]].set(-max_io_sum)
        _log_refused("no-target-fun", rewards)
        return _wire(tokens, eqn_ids, rewards)

    # ------------------------------------------------------------------
    # Compile both the approximated and exact jacobian functions once.
    # ------------------------------------------------------------------
    if _dbg_measure:
        print(f"[measure] compile step={int(stop)} "
              f"muls={muls_adds_fmas:.3g}", flush=True)
    # ``--exec-on-gpu`` pins the reward harness to a GPU distinct from the
    # one the trainer (the main process) is loaded on — otherwise the
    # callback's compile/exec would deadlock against the main program's
    # outstanding work on gpu[0]. Pick the *last* available GPU so we
    # stay as far from the trainer as possible; requires ≥ 2 GPUs.
    callback_device = None
    if config.exec_on_gpu and _MEASURE_ACTOR:
        # Dedicated measure process (see module note): every visible GPU is a
        # measurement GPU and there is no trainer to reserve one for. The
        # actor is pinned to a single device, so this is it.
        _gd = jax.devices("gpu")
        if not _gd:
            raise RuntimeError(
                "ALPHAGRAD_MEASURE_ACTOR=1 with --exec-on-gpu but no GPU is "
                "visible to this process; check the actor's "
                "CUDA_VISIBLE_DEVICES pin."
            )
        callback_device = _next_measure_device(_gd)
    elif config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) < 2:
            raise RuntimeError(
                "--exec-on-gpu requires at least two GPUs (one for the "
                f"trainer, one for the env callback); got {len(gpu_devices)}."
            )
        # GPU 0 is the TRAINER's. Every other GPU is a MEASUREMENT device,
        # and consecutive measurements rotate through them.
        #
        # The rotation is not about throughput — the callback is
        # `vmap_method="sequential"`, so measurements are serial either way.
        # It is about the reading being CLEAN. Peak memory and latency are
        # both contaminated by whatever the previous plan left in the
        # allocator, and by the trainer's own outstanding work. Rotating over
        # k devices gives each measurement k-1 measurements' worth of time for
        # its device to drain before it is read again, and keeps all of it off
        # the device that is running the policy update.
        callback_device = _next_measure_device(gpu_devices[1:])

    args_for_lower = (
        jax.device_put(args, callback_device)
        if callback_device is not None
        else args
    )

    # Per-call jit + lower + compile, wrapped by the cluster-wide
    # cache in ``alphagrad.approx.common.compile_cache``. The
    # coordinator named-actor holds ``key -> ObjectRef`` to a
    # serialised executable (StableHLO + pytree metadata, pickled).
    # On a hit, the calling actor ``ray.get``\\s the blob and
    # ``deserialize_and_load``\\s it locally — no fresh compile, no
    # XLA Executable allocation, no JAX-tracing residue. On a miss,
    # the actor compiles, ``ray.put``\\s the serialised form, and
    # registers the ref so the next actor that needs it skips
    # compilation entirely.
    #
    # The (approx, exact) pair share a compile target almost always
    # (same o_list, same args), so we cache both as a 2-tuple under
    # a single key. The shape signature is part of the key because
    # ``jacve`` produces a different traced function per shape.
    from alphagrad.approx.common.compile_cache import cached_compile
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(order, dtype=np.int32).tobytes())
    h.update(np.asarray(sparsity_specs, dtype=np.int32).tobytes())
    # Face actions produce a different executable for the same (order, specs)
    # — they MUST be in the key or a per-vertex compile would be replayed for
    # a face-actioned plan (and vice versa).
    h.update(np.asarray(face_specs, dtype=np.int32).tobytes())
    h.update(np.asarray(face_skips, dtype=np.int32).tobytes())
    # SO IS THE JOIN BIT, for exactly the same reason: under --approx-add
    # choose two plans can share (order, specs, face rows, skips) and differ
    # only in which container a merge reconciles into, which is a different
    # executable. Absent under every fixed value, so the key is byte-identical
    # to before there.
    if face_joins is not None:
        h.update(np.asarray(face_joins, dtype=np.int32).tobytes())
    h.update(int(stop).to_bytes(4, "little", signed=False))
    h.update(b"sparse" if bool(config.sparse) else b"dense")
    # Include the shape signature of args_for_lower so we don't
    # collide across rollouts that share (order, specs) but differ
    # in batch shape.
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            h.update(repr(a.shape).encode())
            h.update(repr(a.dtype).encode())
    # The device is part of the key: `.lower(*args).compile()` bakes the
    # device assignment into the executable, so an entry compiled for GPU 1
    # cannot be replayed on GPU 2.
    if callback_device is not None:
        h.update(repr(callback_device).encode())
    cache_key = h.digest()

    def _jacve_fn(approx: bool):
        """THE elimination, built once. `approx=False` is the exact
        reference: same order, same argnums, same has_aux, same sparse
        representation, and NOTHING but the two approximation kwargs
        dropped -- which is what makes any approx/exact ratio taken over
        this pair a statement about the approximation alone. Both the
        compiles below and the sparsity tally's abstract fallback walk
        go through here so a change to one cannot miss the other."""
        _kw = ({"transforms": transforms,
                "face_transforms": ft_by_vertex} if approx else {})
        return jacve(
            config.target_fun,
            list(o_list),
            argnums=config.argnums,
            has_aux=config.has_aux,
            sparse_representation=config.sparse,
            # ONE JAXPR FOR BOTH PATHS (dsnn-dfw.24). The order and the face
            # keys are numbered on `config.jaxpr`; a fresh trace inside
            # `.lower()` is a different equation list for the same function
            # (measured on window2: 90 equations against 72, every
            # `convert_element_type` moved), and then the plan addresses
            # vertices that are not there.
            jaxpr=config.jaxpr,
            consts=list(consts),
            **_kw,
        )

    def _do_compile_approx():
        # THE MEASURED ELIMINATION. graphax invokes every per-vertex/per-face
        # hook once per face while ``.lower()`` traces, and this is the only
        # elimination whose applied/skipped counts describe what was actually
        # measured -- so this is the ONE scope the per-face counters are armed
        # in. NOT armed: the face-enum replay, the tokenizer replay, the count
        # pass, and the sparse-boundary cost re-trace below (a second trace of
        # the same plan). CAVEAT: a compile-cache HIT skips the trace, so the
        # counters describe distinct measured plans, not repeats of one.
        from alphagrad.approx.common.masks import (
            arm_face_counts, disarm_face_counts)
        arm_face_counts()
        # A6: this closure is what `cached_compile` calls on a MISS, so
        # reaching it is exactly "the per-face hook got a chance to count
        # this plan". Recorded as `counts_from_trace`.
        _plan_traced[0] = True
        # SAME SCOPE, SAME REASON as the per-face counters: this is the
        # ONE trace that walks the elimination that is actually measured.
        _arm_store_tally(_sparsity_on)
        try:
            return _compile_measure(
                jax.jit(_jacve_fn(approx=True), keep_unused=True)
                .lower(*args_for_lower)
            )
        finally:
            disarm_face_counts()
            _read_store_tally(_sparsity_on, _APPROX_STORE_BYTES,
                              cache_key)

    def _do_compile_exact():
        # Wrapped, not duplicated: every consumer of the exact
        # executable (the cosine quality metric, the fidelity channel's
        # `_exact_ref_scores`) reaches it
        # through this one closure, so arming here is what makes the
        # sparsity denominator FREE on the reference somebody else is
        # already paying for.
        _arm_store_tally(_sparsity_on)
        try:
            return _compile_measure(
                jax.jit(_jacve_fn(approx=False), keep_unused=True)
                .lower(*args_for_lower)
            )
        finally:
            _read_store_tally(_sparsity_on, _EXACT_STORE_BYTES,
                              exact_cache_key)

    def _store_totals(store: dict, key, approx: bool):
        """``(totals, did_fallback_trace)`` for one arm.

        The tally is filled by whichever `.lower()` actually traced. When
        the compile cache served the executable without tracing, walk the
        SAME elimination abstractly: `jax.eval_shape` runs
        `_eliminate_vertex` in Python and emits no HLO, so it costs a
        trace and neither a compile nor an execution."""
        _t = store.get(bytes(key))
        if _t is not None:
            return _t, False
        _arm_store_tally(True)
        try:
            jax.eval_shape(_jacve_fn(approx=approx), *args_for_lower)
        finally:
            _read_store_tally(True, store, key)
        _t = store.get(bytes(key))
        if _t is None:
            raise RuntimeError(
                "the abstract elimination walk did not run; the sparsity "
                "tally cannot be taken for this plan")
        return _t, True

    # The EXACT compile ignores `transforms` / `face_transforms` entirely (see
    # _do_compile_exact above — it only takes o_list + config + arg shapes), so
    # keying it on the approximation specs makes two envs that picked the SAME
    # ORDER with different approximations compile a byte-identical executable
    # twice. Narrow the exact key to what the exact executable actually depends
    # on. Bit-identical result, strictly more cache-friendly.
    h_ex = hashlib.blake2b(digest_size=16)
    h_ex.update(np.asarray(partial_order, dtype=np.int32).tobytes())
    h_ex.update(b"sparse" if bool(config.sparse) else b"dense")
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            h_ex.update(repr(a.shape).encode())
            h_ex.update(repr(a.dtype).encode())
    if callback_device is not None:
        h_ex.update(repr(callback_device).encode())
    exact_cache_key = h_ex.digest()

    # THE PAIRED REFERENCE (ticket .9): rev-exact = the reverse order over
    # the SAME vertex set the candidate eliminated, no rule on any vertex,
    # no action on any face. Built through the same ``jacve`` call shape an
    # identity candidate gets from `_jacve_fn(approx=True)` -- ``transforms``
    # is the empty list and ``face_transforms`` None whenever a plan has no
    # rule and no face action -- so an identity plan and its reference are
    # the SAME executable and land on the same lowering path (ticket .24's
    # armed-vs-unarmed question is thereby moot for the pair; its GPU
    # landing test still stands). Descending vertex ids IS graphax's
    # ``"rev"`` over these vertices (core._checkify_order). The COMPILE is
    # cached like every other executable (order, arg shapes, device); the
    # MEASUREMENT is taken anew in every terminal callback, right after the
    # candidate's -- that is what makes it paired.
    _rev_order = sorted(o_list, reverse=True)
    h_rf = hashlib.blake2b(digest_size=16)
    h_rf.update(np.asarray(_rev_order, dtype=np.int32).tobytes())
    h_rf.update(b"sparse" if bool(config.sparse) else b"dense")
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            h_rf.update(repr(a.shape).encode())
            h_rf.update(repr(a.dtype).encode())
    if callback_device is not None:
        h_rf.update(repr(callback_device).encode())
    paired_ref_key = h_rf.digest()

    def _do_compile_paired_ref():
        return _compile_measure(
            jax.jit(
                jacve(
                    config.target_fun,
                    list(_rev_order),
                    argnums=config.argnums,
                    has_aux=config.has_aux,
                    sparse_representation=config.sparse,
                    transforms=[],
                    face_transforms=None,
                    # The paired reference walks the SAME graph as the
                    # candidate, or the ratio is not about the plan.
                    jaxpr=config.jaxpr,
                    consts=list(consts),
                ),
                keep_unused=True,
            ).lower(*args_for_lower)
        )

    # RESOURCE-LIMIT TRUNCATION (OOM) — "Time Limits in RL" applied to memory.
    #
    # A big approximated graph can exhaust device memory during compile or
    # execution. Before this, a RESOURCE_EXHAUSTED propagated out of the
    # io_callback and killed the whole run — losing every episode already
    # collected. But an OOM says nothing about the PLAN's quality: it is the
    # apparatus failing, exactly like a time-limit truncation, so the correct
    # response is to truncate this transition and EXCLUDE it from the gradient
    # (see `_truncated_reward`) rather than to score it.
    #
    # Caught here (compile) and around execution below. `_is_oom` matches on
    # the XLA error text because jaxlib raises a generic XlaRuntimeError for
    # RESOURCE_EXHAUSTED rather than a dedicated class.
    def _is_oom(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        txt = f"{type(exc).__name__}: {exc}".upper()
        return any(k in txt for k in (
            "RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OUT_OF_MEMORY",
            "OOM WHEN ALLOCATING", "CUDA_ERROR_OUT_OF_MEMORY",
        ))

    def _is_graphax_trace_failure(exc: BaseException) -> bool:
        """True when the exception was RAISED INSIDE graphax.

        Deliberately keyed on traceback origin rather than on the message, so
        an alphagrad-side bug with a similar message still propagates and kills
        the run. The known instance is
        ``_normalize_inputs`` refusing to add a tensor to its own transpose
        ((16,10,784,256) vs (16,10,256,784)) -- the open canonical-output-order
        gap -- but the whole family belongs here: the plan is well-defined and
        the library cannot build it, which is apparatus failure, not plan
        quality.
        """
        tb = exc.__traceback__
        while tb is not None:
            fn = tb.tb_frame.f_code.co_filename
            if f"{os.sep}graphax{os.sep}" in fn:
                return True
            tb = tb.tb_next
        return False

    def _trace_truncate(where: str, exc: BaseException):
        _record_untraceable_plan(exc)
        _tr = _truncated_reward()
        _log_refused(f"untraceable:{where}", _tr)
        return _wire(tokens, eqn_ids, _tr)

    def _oom_truncate(where: str, exc: BaseException):
        _record_truncated_plan()
        # Free whatever the failed attempt is still holding before returning,
        # or the next callback inherits a poisoned allocator.
        try:
            jax.clear_caches()
            gc.collect()
        except Exception:
            pass
        print(f"[trunc] OOM during {where} step={int(stop)} order={o_list} "
              f"(excluded from gradient): {type(exc).__name__}: "
              f"{str(exc)[:160]}", flush=True)
        _tr = _truncated_reward()
        _log_refused(f"oom:{where}", _tr)
        return _wire(tokens, eqn_ids, _tr)

    try:
        compiled_approx = cached_compile(
            b"approx:" + cache_key, _do_compile_approx)
    except Exception as _exc:
        if _is_graphax_trace_failure(_exc):
            return _trace_truncate("approx compile", _exc)
        if not _is_oom(_exc):
            raise
        return _oom_truncate("approx compile", _exc)
    # SPARSE-BOUNDARY COST MEASUREMENT (ALPHAGRAD_MEASURE_SPARSE=1). The
    # dense executable drains every output to the full nominal Jacobian, so
    # diag/compress plans measure byte-identical latency+peak to exact --
    # the boundary write dominates both channels (nn256@512: 4.17GB output
    # = 2.6ms at HBM rate, temps 0-3MB). That is the mechanism behind
    # "approx cuts latency ~-8% but NEVER memory". Under the flag the COST
    # channels (latency, peak) time a second executable compiled with
    # sparse_representation=True -- same values, compact output buffers
    # (measured: diag2 0.51x lat / -50% peak, compress ax1 0.12x / -89%) --
    # while the dense executable stays the ONLY source of quality outputs:
    # a compact output would shape-mismatch _quality_metrics into the
    # worst score, and the cosine keeps its dense comparability.
    compiled_cost = compiled_approx
    if os.environ.get("ALPHAGRAD_MEASURE_SPARSE", "0") == "1":
        def _do_compile_approx_sparse():
            # #46: factored outputs for the COST executable only — the trace
            # happens inside .lower(), so scoping the env var here keeps the
            # dense/quality/exact executables, tokenizer and replays
            # byte-untouched. DEFAULT OFF: a respawned measure actor imports
            # this file fresh, and a mid-campaign default flip would make its
            # measurements incomparable with its siblings' — opt in per
            # campaign with ALPHAGRAD_FACTORED_OUTPUTS=1 in the sbatch.
            _fo = os.environ.get("ALPHAGRAD_FACTORED_OUTPUTS", "0") == "1"
            _prev = os.environ.get("GRAPHAX_FACTORED_OUTPUTS")
            if _fo:
                os.environ["GRAPHAX_FACTORED_OUTPUTS"] = "1"
            try:
                return _compile_measure(
                    jax.jit(
                        jacve(
                            config.target_fun,
                            list(o_list),
                            argnums=config.argnums,
                            has_aux=config.has_aux,
                            sparse_representation=True,
                            transforms=transforms,
                            face_transforms=ft_by_vertex,
                            jaxpr=config.jaxpr,
                            consts=list(consts),
                        ),
                        keep_unused=True,
                    )
                    .lower(*args_for_lower)
                )
            finally:
                if _fo:
                    if _prev is None:
                        os.environ.pop("GRAPHAX_FACTORED_OUTPUTS", None)
                    else:
                        os.environ["GRAPHAX_FACTORED_OUTPUTS"] = _prev
        try:
            compiled_cost = cached_compile(
                b"approx-sparse:" + cache_key, _do_compile_approx_sparse)
        except Exception as _exc:
            if _is_graphax_trace_failure(_exc):
                return _trace_truncate("approx-sparse compile", _exc)
            if not _is_oom(_exc):
                raise
            return _oom_truncate("approx-sparse compile", _exc)
    # ``compiled_exact`` is ONLY needed for the quality metrics
    # (cosine_sim, frob_residual). Those are meaningful only when the
    # elimination order is complete — graphax's ``jacve`` returns a
    # zero-norm Jacobian for any partial order, so comparing approx vs
    # exact mid-rollout yields ``(cos=0, frob=0)`` regardless. Skip
    # the compile + execute when the step is non-terminal; the cache
    # entry would never be re-used productively anyway.
    #
    # LOSS-DROP QUALITY (2026-08-07): the walk never looks at the exact
    # Jacobian, so under ``ALPHAGRAD_QUALITY_METRIC=loss_drop`` the exact
    # executable is not compiled and not executed at all. That is where the
    # 9.70 s -> 0.22 s and 4.24 GB -> 40 MB per-plan saving comes from.
    # THE SAME-ORDER EXACT PROGRAM IS NO LONGER BUILT FOR THE GRADIENT COSINE
    # (owner ruling 2026-09-18). Under a free order ``exact_cache_key`` is a
    # new key for every plan, so that compile was a SECOND compile per plan
    # for a quantity the cached rev-exact reference already carries: the exact
    # gradient is order-independent to 1e-6 in float32. The cosine now reads
    # the reference (see ``_cosine_reference``); ``jac_cosine``, which scores
    # the plan's own Jacobian at the calibration samples and not a gradient on
    # the probe batch, still needs the same-order program and still builds it.
    _qmetric = quality_metric(config)
    if is_terminal and _qmetric == "jac_cosine":
        try:
            compiled_exact = cached_compile(
                b"exact:" + exact_cache_key, _do_compile_exact)
        except Exception as _exc:
            if _is_graphax_trace_failure(_exc):
                return _trace_truncate("exact compile", _exc)
            if not _is_oom(_exc):
                raise
            return _oom_truncate("exact compile", _exc)
    else:
        compiled_exact = None
    # THE REV-EXACT REFERENCE, compiled here because BOTH consumers are here:
    # the paired cost channel's timing windows below and, since 2026-09-18,
    # the gradient cosine. One executable per (vertex set, shapes, device) and
    # therefore one compile per process, cached on ``paired_ref_key``.
    _ref_ex = None
    if _paired or (is_terminal and _qmetric == "grad_cosine"):
        try:
            _ref_ex = cached_compile(
                b"paired-ref:" + paired_ref_key, _do_compile_paired_ref)
        except Exception as _exc:
            if _is_graphax_trace_failure(_exc):
                return _trace_truncate("paired-ref compile", _exc)
            if not _is_oom(_exc):
                raise
            return _oom_truncate("paired-ref compile", _exc)
    _pf("cb.xla_compile")
    # ORACLE A (ticket .62) IS NOT HERE ANY MORE (owner ruling 2026-09-18).
    #
    # It used to run right at this point: the same-order exact gradient against
    # jax.grad, on the measure actor's GPU, in float64, BEFORE the measurement
    # it guards. That put a sanity check inside the scoring path, and on
    # Blackwell the check's own float64 compile exhausted the SM's shared
    # memory and REFUSED the plan -- 15.6 percent of the plans of an
    # oracle-due episode, which is 3.9 percent of every plan measured
    # (agent-sentinel report, 2026-09-18, section 4). The apparatus was
    # scoring the apparatus.
    #
    # The oracle is now ASYNCHRONOUS, RETROACTIVE and on the CPU: one worker
    # thread in the TRAINER process, fed the distinct orders of an oracle-due
    # episode after that episode is over, writing its answers back as late
    # `oracle_result` records. See `grad_oracle_cpu_check` in this file and
    # `alphagrad.approx.common.grad_oracle_async`. Nothing in this callback
    # waits for it, and the `refused/oracle` telemetry kind is therefore gone:
    # the oracle can no longer refuse anything.
    #
    # The async oracle is still fed the CANDIDATE's own orders, while the
    # cosine now reads the reverse order's reference (dsnn-dfw.41).

    # XLA cost analysis — flops + bytes accessed. Falls back to 0 when the
    # backend doesn't expose them (CPU sometimes returns an empty dict).
    # ``ALPHAGRAD_SKIP_COST_ANALYSIS=1`` skips the call entirely as a
    # memory-leak probe: ``Compiled.cost_analysis()`` triggers an
    # ``HloCostAnalysis`` C++ pass, and per-call C++ state held there may
    # not be released even when the Python wrapper returns. Empirical
    # cost of one analysis ≈ 9 MB × 384 io_callbacks/ep ≈ 3.5 GB/ep —
    # the exact magnitude of our remaining leak after the disk-cache fix.
    # When skipped, ``flops`` and ``bytes_accessed`` reward channels read
    # zero; ``muls_adds_fmas`` (the actual compute target) is unaffected
    # since it's computed by graphax's symbolic counter above.
    if os.environ.get("ALPHAGRAD_SKIP_COST_ANALYSIS", "0") == "1":
        flops = 0.0
        bytes_accessed = 0.0
    else:
        cost_analysis = compiled_approx.cost_analysis() or {}
        flops = float(cost_analysis.get("flops", 0))
        bytes_accessed = float(cost_analysis.get("bytes accessed", 0))

    # ------------------------------------------------------------------
    # Execution loop — runs once for peak_memory + quality, or 10x when
    # `measure_latency` is on (the latency reading is noisy enough that the
    # median smoothing from the original code is worth keeping).
    # ------------------------------------------------------------------
    # Measurement budget: `num_data_points` distinct eval samples x
    # `reps_per_point` timing repetitions is now the CAP on the number of
    # timed windows, not the number itself (owner ruling 2026-09-14; see
    # `EnvConfig.measure_budget_secs`). The windows a plan actually gets are
    # derived from one warm-up execution's measured time so that the plan
    # costs about `measure_budget_secs` of executions however fast or slow it
    # is, and they are spread ROUND-ROBIN over the data points. Reps exist
    # only to average timer noise, so without --measure-latency we take 1 rep
    # per point. Quality is deterministic in the input, so it is computed ONCE
    # PER POINT (not per rep) and medianed across points.
    n_points = max(1, int(getattr(config, "num_data_points", 5)))
    if eval_samples:
        n_points = min(n_points, len(eval_samples[0]))
    n_reps = max(1, int(getattr(config, "reps_per_point", 4))) if config.measure_latency else 1
    _budget_s = float(getattr(config, "measure_budget_secs", 1.0) or 0.0)
    _window_s = float(getattr(config, "measure_window_secs", 0.05) or 0.0)
    if _budget_s <= 0.0 or _window_s <= 0.0:
        raise ValueError(
            "measure_budget_secs and measure_window_secs must both be > 0 "
            f"(got {_budget_s!r} and {_window_s!r}); they set the per-plan "
            "execution budget and the target timed-window duration, and a "
            "zero would mean 'measure nothing'")
    # THE PAIRED REFERENCE'S OWN BUDGET, decoupled from the candidate's
    # (owner ruling 2026-09-14; see EnvConfig.ref_num_data_points for the
    # measurement that motivates it). The reference is 150x cheaper per
    # execution than the candidate on the Markowitz order, so the candidate's
    # counts leave it with a tenth of a second of integration and it carries
    # essentially all of the paired ratio's noise. The inner reps and the
    # warmup stay SHARED; only the points and the reps fork here.
    n_ref_points = max(1, int(getattr(config, "ref_num_data_points", 5)))
    if eval_samples:
        n_ref_points = min(n_ref_points, len(eval_samples[0]))
    n_ref_reps = (max(1, int(getattr(config, "ref_reps_per_point", 32)))
                  if config.measure_latency else 1)

    # Match the monitor to whichever device the compiled JIT actually runs
    # on. Without --exec-on-gpu the args arrive as CpuDevice JAX arrays
    # from the io_callback and the JIT lands on CPU; with --exec-on-gpu we
    # explicitly device_put both `args_for_lower` and `eval_args_i` onto
    # the pinned callback device above, so the JIT lands there. Reading
    # the device off `args_for_lower` covers both branches without
    # forcing the user to opt into GPU monitoring.
    monitoring_devices: list = []
    for x in jax.tree_util.tree_leaves(args_for_lower):
        if hasattr(x, "devices"):
            monitoring_devices.extend(list(x.devices()))
    unique_devices = list({id(d): d for d in monitoring_devices}.values())
    if not unique_devices:
        unique_devices = jax.local_devices()

    # STREAMED quality (#54, 2026-08-04): each data point is scored inside
    # the measure loop and its Jacobian pair dropped immediately. The old
    # out_approxs/out_exacts lists held n_points x 2 full Jacobians
    # (~41.7GB at batch 512) before scoring — an OOM-truncation source that
    # said nothing about the plan.
    # Samples of THE QUALITY CHANNEL (reward slot 6). Historical name: under
    # ALPHAGRAD_QUALITY_METRIC=jac_cosine it holds one Jacobian cosine per
    # calibration point; under the default grad_cosine (and under loss_drop)
    # it holds exactly ONE entry per plan.
    cosines: list = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []
    # FIDELITY (reward slot 8) + the COSINE LOG subsample. Both need the exact
    # Jacobian, and under ALPHAGRAD_QUALITY_METRIC=cosine one is already being
    # built and scored PER POINT below -- in that case take the residual from
    # there (free) instead of asking for a second exact execution.
    _fid_from_quality = _qmetric in ("jac_cosine", "grad_cosine")
    _fid_on = bool(is_terminal) and fidelity_enabled()
    # `_sparsity_on` (slot 10) is bound up at `is_terminal`, not here:
    # the compile closures that arm the tally run before this line.
    # `_cos_log_due` CONSUMES a tick, so it must be evaluated exactly once per
    # terminal callback and only when the subsample could actually be taken.
    _cos_due = bool(is_terminal) and (not _fid_from_quality) and _cos_log_due()
    # Does anything need an exact reference that the quality metric is not
    # already producing? Keeping the approx output of point 0 alive costs one
    # resident Jacobian, so only do it when a residual is actually wanted.
    _fid_needs_ref = (_fid_on or _cos_due) and not _fid_from_quality
    _fid_approx_out = None
    _fid_eval_args = None
    _rel_frobs: list = []
    _cos_logged: list = []
    # WHICH quantity the peak samples hold (see _record_mem_parity): the
    # measured runtime delta, the substituted static estimate, or nothing.
    _peak_src = "not_measured"

    # OOM during EXECUTION is truncation too (see _oom_truncate): the
    # measurement allocates the full approximated Jacobian, so a graph that
    # compiled fine can still exhaust the device here. Excluded from the
    # gradient rather than scored.
    try:
        # THE MEASUREMENT INPUTS, materialised ONCE. The paired rev-exact
        # reference (ticket .9) measures its executable over the SAME DATA
        # with the SAME instrument (see _time_one_rep) -- but since
        # 2026-09-14 over its OWN number of points, so the list is built to
        # whichever of the two is longer and each half takes its prefix.
        # The candidate's windows walk `eval_args_all[:n_points]` round
        # robin and the reference's walk `eval_args_all[:n_ref_points]`; at
        # the campaign defaults the two are both 5 and the list is exactly
        # what it always was. A half with fewer windows than points simply
        # does not reach the later ones, which is what the budget means.
        # Built
        # inside the try so a device_put OOM still truncates rather than
        # escaping.
        eval_args_all: list = []
        for i in range(max(n_points, n_ref_points)):
            if eval_samples:
                _ea = [arg[i] for arg in eval_samples]
            else:
                _ea = list(args)
            if callback_device is not None:
                _ea = [jax.device_put(d, callback_device) for d in _ea]
            eval_args_all.append(_ea)
        # WARMUP (config.latency_warmup): untimed executions before the
        # first timed rep, matching the elimrl/POMO worker's single warmup
        # call. Runs OUTSIDE every timing and memory window, so it can only
        # remove first-touch bias, never add to it.
        #
        # DEFAULT ON (2026-08-26, ALPHAGRAD_MEASURE_WARMUP=0 to restore the
        # old behaviour). Without it the FIRST timed sample of a plan is the
        # FIRST EXECUTION of a freshly compiled executable, so first-touch
        # (buffer setup, lazy scratch allocation, cold caches/clocks) lands
        # inside a timed window. The 5x4 median absorbs one cold sample out
        # of twenty almost completely -- so this is not the source of the
        # campaign's latency numbers -- but a paired harness with few
        # samples is fully exposed to it (identity-vs-itself measured
        # 0.51 +/- 0.59 without a warmup, 1.00 +/- 0.14 with one), and the
        # gate floor is a FRESH compile every time it is measured. One
        # untimed execution costs one execution and removes the whole class.
        _warmup = _resolve_warmup(config)
        # THE CONFIGURED INNER-REP CEILING. Since the owner's ruling of
        # 2026-09-14 `--latency-inner-reps` is the CEILING of the window
        # rule, not the inner itself (see `resolve_measure_inner`): the
        # campaign's 50 gives the ruling's clamp(., 5, 50) verbatim, and any
        # caller that configures fewer than 5 measures exactly as before.
        _cfg_inner = max(1, int(getattr(config, "latency_inner_reps", 1)))

        # THE PAIRED REFERENCE'S EXECUTABLE (ticket .9) is `_ref_ex`, built
        # beside the candidate's compile above. It has to exist before the
        # timing loop, because since 2026-09-14 its windows are INTERLEAVED
        # with the candidate's instead of forming a second block after them.

        def _probe_one(ex, eval_args) -> float:
            """Seconds of ONE execution of `ex`, measured on the WARM-UP.

            THE LAST warm-up execution, and there are at least TWO. A single
            warm-up is a COLD reading, and a cold reading here does not just
            add noise to the counts, it can change the SCALE of the result:
            the inner-rep count is ``ceil(window / t)``, so a `t` that reads
            HIGH gives a SMALLER inner, and a small inner on a microsecond
            program reads high against the dispatch floor (section 1.3 of
            docs/UNBIASED_PARETO_AND_MEASUREMENT.md).

            MEASURED, on the transformer arm (job 65666, one measure actor,
            two episodes, one warm-up): the rev-exact reference settled at
            138 us, but some plans' cold probe read about 2.5 ms and took
            inner 20 instead of 50. Those plans then read up to 220 us, and
            the reference's coefficient of variation over 112 plans was 9.6
            percent against 2.1 percent under the fixed protocol. The second
            warm-up costs one execution of the reference, 0.14 ms, and the
            docs size the cold read as gone by the second reading (the
            identity plan's first cold reading is 7.2 percent off and is
            back inside 2 percent by reading two).

            With ``ALPHAGRAD_MEASURE_WARMUP=0`` the budget still has to size
            itself from something, so that configuration now pays exactly
            TWO untimed executions per half per plan where it used to pay
            none. They buy the counts.
            """
            _t = 0.0
            for _w in range(max(2, _warmup)):
                _p0 = time.perf_counter()
                jax.block_until_ready(ex(*eval_args))
                _t = time.perf_counter() - _p0
            return _t

        # ---- THE CANDIDATE'S BUDGET (owner ruling 2026-09-14) -----------
        # One second of executions per plan, whatever the plan costs, instead
        # of a fixed 5 x 4 x 50 = 1005 executions that cost 18.3 s on the
        # transformer arm and 0.12 s on the reference. See
        # `EnvConfig.measure_budget_secs`.
        _warmed_cand: set = set()
        _warmed_ref: set = set()
        if config.measure_latency:
            _t_cand = _probe_one(compiled_cost, eval_args_all[0])
            _inner = resolve_measure_inner(_t_cand, _window_s, _cfg_inner)
            _n_windows = resolve_measure_windows(
                _t_cand, _inner, _budget_s, n_points * n_reps)
        else:
            # THE BUDGET IS A TIMING INSTRUMENT. With --measure-latency off
            # there is no timer noise to integrate, `n_reps` is already 1,
            # and this path stays exactly what it was before the ruling.
            _t_cand = 0.0
            _inner = _cfg_inner
            _n_windows = n_points * n_reps
            for _w in range(_warmup):
                jax.block_until_ready(compiled_cost(*eval_args_all[0]))
        _warmed_cand.add(0)

        # ---- THE REFERENCE'S BUDGET -------------------------------------
        # It KEEPS ITS OWN WINDOW COUNT, `ref_num_data_points` x
        # `ref_reps_per_point` (owner rulings of 2026-09-14, in that order):
        # the reference is 150x cheaper per execution than the candidate on
        # this order, so 160 windows of it cost about a second and it is the
        # half that carries the paired ratio's noise. Only its INNER comes
        # from the window rule, applied to ITS OWN execution time -- a 121 us
        # program fills a 50 ms window 413 times over and takes the ceiling
        # of 50, where the candidate takes the floor of 5.
        _ref_inner = 0
        _ref_windows = 0
        if _paired:
            if config.measure_latency:
                _t_ref = _probe_one(_ref_ex, eval_args_all[0])
                _ref_inner = resolve_measure_inner(
                    _t_ref, _window_s, _cfg_inner)
            else:
                _t_ref = 0.0
                _ref_inner = _cfg_inner
                for _w in range(_warmup):
                    jax.block_until_ready(_ref_ex(*eval_args_all[0]))
            _ref_windows = n_ref_points * n_ref_reps
            _warmed_ref.add(0)

        # ---- THE INTERLEAVED TIMING LOOP --------------------------------
        # A B A B ... (see `interleave_windows`), not two blocks: the paired
        # ratio can only cancel drift that both halves saw, so the two halves
        # have to occupy the same seconds. The windows of each half are
        # spread ROUND-ROBIN over that half's data points, so a plan that
        # earns only three windows still sees three different samples rather
        # than three repetitions of sample 0.
        #
        # THE DRAIN LIVES IN ``_time_one_rep``, inside the timed window
        # (see the block there). It used to say here that ResourceMonitor's
        # ``jax.effects_barrier()`` drained the queue for us; it does not --
        # that barrier waits only for ORDERED EFFECTS, and a jitted Jacobian
        # has none. Job 65720 measured what the belief cost.
        #
        # ``ALPHAGRAD_BYPASS_RESOURCE_MONITOR=1`` skips the construction +
        # context manager entirely (vs. the lighter
        # ``ALPHAGRAD_DISABLE_RESOURCE_MONITOR`` which only swapped the
        # class to a no-op). peak_memory + latency_ns are zero for the run.
        #
        # ONE instrument (see `_time_one_rep`): both halves call the same
        # function, with the same warmup and the same window rule, so the
        # reference is exactly what the campaign path would have printed for
        # the rev-exact plan -- not a throughput timing that reads 13% low.
        _ref_lat_samples: list[float] = []
        _ref_peak_samples: list[float] = []
        _ia = 0
        _ib = 0
        for _who in interleave_windows(_n_windows, _ref_windows):
            if _who == 0:
                _p = _ia % n_points
                if _p not in _warmed_cand:
                    for _w in range(_warmup):
                        jax.block_until_ready(
                            compiled_cost(*eval_args_all[_p]))
                    _warmed_cand.add(_p)
                _lat_ns, _peak_b, _peak_src, _out = _time_one_rep(
                    compiled_cost, eval_args_all[_p], unique_devices, _inner)
                # DROPPED IMMEDIATELY. The old loop kept the last timed
                # output alive because the per-point quality work read it;
                # that work now runs in its own loop below, so nothing needs
                # it here and holding a full Jacobian across the timing loop
                # is pure OOM headroom spent for nothing.
                del _out
                latency_samples.append(_lat_ns)
                peak_mem_samples.append(_peak_b)
                _ia += 1
            else:
                _p = _ib % n_ref_points
                if _p not in _warmed_ref:
                    for _w in range(_warmup):
                        jax.block_until_ready(_ref_ex(*eval_args_all[_p]))
                    _warmed_ref.add(_p)
                _l, _pk, _s, _o = _time_one_rep(
                    _ref_ex, eval_args_all[_p], unique_devices, _ref_inner)
                del _o, _s
                _ref_lat_samples.append(_l)
                _ref_peak_samples.append(_pk)
                _ib += 1
        # SECONDS OF EXECUTION actually spent inside timed windows, per half.
        # Window w of a half took ``latency_ns[w] * inner`` nanoseconds, which
        # is the quantity the budget is a target for. Warm-ups, probes and the
        # quality walk are outside it, by the same rule that keeps them
        # outside the timing windows themselves.
        _meas_secs = float(sum(latency_samples)) * _inner / 1e9
        _ref_secs = float(sum(_ref_lat_samples)) * _ref_inner / 1e9
        _pf("cb.exec_measure")

        # ---- PER-POINT QUALITY, OUTSIDE every timed window --------------
        # Until 2026-09-14 the per-point approximated Jacobian was taken off
        # the LAST timed window of that point, which was free. The windows
        # are now spread round-robin and a plan may earn FEWER windows than
        # there are data points, so no point is guaranteed a timed execution
        # of its own and the output has to be produced here.
        #
        # It costs ONE untimed execution per scored point, and only under the
        # consumers that need one: `jac_cosine`, which scores per point, and
        # the fidelity / cosine-log subsample, which needs point 0 alone. The
        # campaign runs `grad_cosine`, which scores once per PLAN below, so on
        # every campaign arm this loop executes nothing at all.
        _quality_points = 0
        if compiled_exact is not None and _qmetric == "jac_cosine":
            _quality_points = n_points
        elif _fid_needs_ref and is_terminal:
            _quality_points = 1
        for i in range(_quality_points):
            eval_args_i = eval_args_all[i]
            # DENSE by construction: `compiled_approx`, never the
            # sparse-boundary `compiled_cost`, because the residual and the
            # cosine both need leaf parity with the exact reference.
            out_approx = compiled_approx(*eval_args_i)

            # FIDELITY, approx half. Point 0 only, terminal only.
            if _fid_needs_ref and is_terminal and i == 0:
                _fid_eval_args = eval_args_i
                # THE ONE EXTRA RESIDENT JACOBIAN the fidelity channel
                # costs. Held from point 0 until the exact-reference block,
                # which runs AFTER this loop -- so the expensive half (the
                # exact execution and the per-leaf reductions) is never
                # inside a timing or peak-memory window. Since 2026-09-14 it
                # is no longer held across the timing loop either, because
                # this loop runs after it.
                _fid_approx_out = out_approx

            # ``compiled_exact`` is only executed at the terminal step (see
            # the ``is_terminal`` guard around its compile, above).
            if compiled_exact is not None and _qmetric == "jac_cosine":
                _ex_key = _eval_digest(eval_args_i) if _CACHE_EXACT else None
                _hit = _EXACT_CACHE.get(_ex_key) if _ex_key else None
                if _hit is not None:
                    out_exact = _hit
                else:
                    out_exact = compiled_exact(*eval_args_i)
                    if _ex_key is not None:
                        # Bounded: one episode's worth of samples. A new
                        # episode changes every digest, so the old entries are
                        # dead weight and get dropped wholesale.
                        if len(_EXACT_CACHE) >= max(n_points, 1):
                            _EXACT_CACHE.clear()
                        _EXACT_CACHE[_ex_key] = out_exact
                # Score THIS point now and let the pair go out of scope - cos
                # is the trained quality channel under this metric, and the
                # residual is NO LONGER DISCARDED (A2): it is the fidelity
                # channel, and here it is genuinely free because the exact
                # reference is already resident for the cosine.
                _jac_a = out_approx[1] if config.has_aux else out_approx
                _jac_e = out_exact[1] if config.has_aux else out_exact
                _cos, _rf = _quality_metrics(_jac_e, _jac_a, align=True,
                                             site="jac_cosine")
                cosines.append(_cos)
                if _fid_on:
                    _rel_frobs.append(float(_rf))
                    _cos_logged.append(float(_cos))

        # ---- THE PAIRED REFERENCE'S READING (ticket .9) -----------------
        # Reduced by the SAME median the candidate's channels get
        # (`_aggregate_samples`), off windows taken in the same seconds as
        # the candidate's. The static temp is read off the reference
        # executable the same way `_record_mem_parity` reads the candidate's.
        _ref_lat_ns = 0.0
        _ref_peak = 0.0
        _ref_temp = None
        if _paired:
            _ref_lat_ns = (
                float(_aggregate_samples(_ref_lat_samples,
                                         want_top_quartile=True))
                if _ref_lat_samples else 0.0)
            _ref_peak = (
                float(_aggregate_samples(_ref_peak_samples,
                                         want_top_quartile=True))
                if _ref_peak_samples else 0.0)
            _ref_temp = _static_temp_bytes(_ref_ex)
            _pf("cb.paired_ref")

        # ---- LOSS-DROP QUALITY ------------------------------------------
        # ONE walk per PLAN (not per data point): the probe batch is fixed
        # across plans by construction, so repeating the walk over the
        # calibration samples would re-measure the same number. Runs after
        # the cost loop so the timing/peak windows above never contain it.
        _pf("cb.exec_measure")
        # ---- GRADIENT COSINE --------------------------------------------
        # ONE scoring per PLAN, like the walk: the probe batches are fixed
        # across plans, so repeating over the calibration samples would
        # re-measure the same number. Runs after the cost loop so the
        # timing/peak windows never contain it.
        if is_terminal and _qmetric == "grad_cosine":
            _gc = _grad_cosine_quality(
                config, compiled_approx, _ref_ex, paired_ref_key, list(args),
                callback_device, _grad_cosine_k(config))
            if _gc is _QUALITY_NO_CHANNEL:
                # NO CHANNEL IS NOT A REFUSAL. This configuration has no data
                # generator and no rev-exact reference, so there is no probe
                # batch, no plan of this run can be scored, and refusing would
                # refuse every terminal of every episode -- the run would
                # measure nothing at all. The channel reads 0.0 and says so
                # once, which is what it did before dsnn-dfw.51 and what the
                # analytic AD benchmarks and the toy envs depend on.
                if not _WALK_UNDEFINED_WARNED:
                    _WALK_UNDEFINED_WARNED.append(1)
                    print(
                        "[measure] WARNING quality channel: the GRADIENT "
                        "COSINE HAS NO CHANNEL for this configuration (no "
                        "data_gen, or the rev-exact reference failed to build "
                        "or to execute on the probe batch). The channel reads "
                        "0.0 for every plan of this run; ask for "
                        "ALPHAGRAD_QUALITY_METRIC=jac_cosine to score at the "
                        "calibration samples instead, or =none to drop the "
                        "channel.", flush=True)
                cosines.append(0.0)
            elif _gc is None:
                # AN UNDEFINED COSINE IS A REFUSED MEASUREMENT, NOT A SCORE
                # (owner ruling 2026-09-19, dsnn-dfw.51). It used to read 0.0,
                # which under `--reward-mode lagrangian --quality-floor 0.90`
                # is a full constraint violation: on RSNN_SHD bptt (job 66633)
                # every measure actor printed the warning below and more than
                # half the order-only plans were penalised for a step position
                # that fired nothing. The taxonomy this file already applies to
                # an OOM and to an untraceable plan applies here: the
                # apparatus could not measure the plan, so the measurement is
                # MISSING DATA -- counted, recorded, and excluded from the
                # update -- and never the worst possible number.
                if not _WALK_UNDEFINED_WARNED:
                    _WALK_UNDEFINED_WARNED.append(1)
                    print(
                        "[measure] WARNING quality channel: the GRADIENT "
                        "COSINE is UNDEFINED for this measurement (the exact "
                        "gradient is identically zero on every probe batch "
                        "this plan was scored on). "
                        "Every affected measurement is REFUSED and excluded "
                        "from the update; watch `refused/quality-undefined`. "
                        "On a spiking target this means the sampled steps "
                        "fired nothing: widen the probe "
                        "(ALPHAGRAD_GRAD_COSINE_K) or raise the target's "
                        "firing rate.", flush=True)
                _tr = _truncated_reward()
                _log_refused("quality-undefined:grad_cosine", _tr)
                return _wire(tokens, eqn_ids, _tr)
            else:
                _gc_q, _gc_frobs, _gc_cos = _gc
                cosines.append(_gc_q)
                if _fid_on:
                    _rel_frobs.extend(float(x) for x in _gc_frobs)
                    _cos_logged.extend(float(x) for x in _gc_cos)
        if is_terminal and _qmetric == "loss_drop":
            _ld = _loss_drop_quality(
                config, compiled_approx, list(args), callback_device)
            if _ld is None:
                # The walk is undefined for this env (no data generator, no
                # updatable weight slot, or a plan whose output does not even
                # have the weights' shapes). Say so LOUDLY once — silently
                # scoring 0 would look like "this plan does not train".
                if not _WALK_UNDEFINED_WARNED:
                    _WALK_UNDEFINED_WARNED.append(1)
                    print(
                        "[measure] WARNING quality channel: the loss-drop "
                        "walk is UNDEFINED for this configuration (no "
                        "data_gen / no non-scalar differentiated arg / "
                        "gradient shape mismatch). The channel reads 0.0 "
                        "for every affected plan; set "
                        "ALPHAGRAD_QUALITY_METRIC=cosine to train on the "
                        "legacy Jacobian cosine instead.", flush=True)
                cosines.append(0.0)
            else:
                cosines.append(_ld)
        # The walk is the single most expensive phase of a measurement on an
        # approximation arm (200 executions of the plan + ~1 s of host time,
        # vs 100 executions for the entire latency channel), so it gets its
        # own bucket instead of hiding inside cb.exec_measure.
        _pf("cb.quality_walk")

    except Exception as _exc:
        if _is_graphax_trace_failure(_exc):
            return _trace_truncate("measurement", _exc)
        if not _is_oom(_exc):
            raise
        return _oom_truncate('measurement', _exc)
    latency_ns = (
        float(_aggregate_samples(latency_samples, want_top_quartile=True))
        if config.measure_latency
        else 0.0
    )
    # Fake-fast guard: a positive-but-implausibly-small reading is clamped UP
    # to the floor rather than trusted — a degenerate (zero-work) plan must
    # not report an unbeatable latency. 0.0 stays 0.0 (= "not measured").
    if 0.0 < latency_ns < _LAT_FLOOR_NS:
        latency_ns = _LAT_FLOOR_NS
    # Medianed like latency (spec: one robust summary over the 5x4 budget); a
    # plain max let a single outlier reading own the channel. Identical to max
    # whenever the readings are constant — the common case today.
    peak_memory = (
        float(_aggregate_samples(peak_mem_samples, want_top_quartile=True))
        if peak_mem_samples
        else 0.0
    )
    # BOTH memory numbers, per measurement, with the source of the one that
    # trains made explicit (see _record_mem_parity).
    _MEM_PARITY_MEASURED[0] += 1
    _mp = _record_mem_parity(compiled_cost,
                             peak_memory if peak_mem_samples else None,
                             _peak_src, is_terminal)
    if mem_channel() == "temp":
        # THE MEMORY CHANNEL (ticket .49, ruling .29): slot 5 is the static
        # temp bytes of the executable that was just timed; the watermark
        # it replaced stays in the parity record. No executable or no
        # memory_analysis() is a fault, not a zero.
        if _mp is None or _mp["static_temp_bytes"] is None:
            raise MemChannelFault(
                "memory channel: memory_analysis() returned nothing for the "
                "timed executable, so the static temp bytes of this plan "
                "cannot be read (peak_source=%s)" % _peak_src)
        peak_memory = float(_mp["static_temp_bytes"])

    # ------------------------------------------------------------------
    # Quality family — reward slot 6 (``quality``) + frob_residual.
    # ------------------------------------------------------------------
    # WHICH quantity lands in slot 6 is ``quality_metric(config)``:
    # ``grad_cosine`` (the default for any scalar-loss target) = the cosine
    # between this plan's gradient and the rev-exact gradient; ``loss_drop``
    # (by name) = the loss drop of a 200-step Adam walk driven by this plan's
    # gradient; ``jac_cosine`` = the legacy Jacobian cosine. All are "higher
    # is better", all are ~[0, 1]
    # (loss_drop can reach -1 when the walk diverges), so every downstream
    # consumer — PopArt's per-channel sigma floor, --lambda-acc, the symlog
    # bypass, the mult gate — keeps its calibration.
    #
    # Sparse-terminal channels: only computed on the terminal step
    # of the rollout. Partial elimination orders produce
    # ``||jac||=0`` for both ``compiled_approx`` and ``compiled_exact``
    # (verified empirically against graphax.jacve), so neither the
    # comparison nor the walk is meaningful mid-rollout. Skip the work
    # entirely — the cost channels above (muls/io/flops/peak_memory) still
    # compute per step, only quality is sparse.
    if is_terminal and cosines:
        # loss_drop appends exactly one entry, so the aggregation is the
        # identity there; it still runs so the cosine path is untouched.
        cosine_sim = float(_aggregate_samples(cosines, want_top_quartile=True))
        # The RAW relative residual, telemetry only: what TRAINS is the
        # clipped `fidelity` in slot 8, written by the exact-reference block
        # below (which may overwrite this). 0.0 here means "not measured",
        # which is what it meant for every real plan before A2.
        frob_residual = 0.0
        # XLA-analysis side-channel (log-only: the exact/approx compression
        # RATIO; the absolute static estimate is not exported — it only ever
        # substitutes into peak_memory). Nothing trains on it, so it rides the
        # same skip flag as the count pass.
        # memory_analysis() walks the compiled HLO, which is not free on the
        # big graphs a working policy produces.
        if not skip_count_ops():
            _approx_bytes = _memory_analysis_bytes(compiled_approx)
            _exact_bytes = (
                _memory_analysis_bytes(compiled_exact)
                if compiled_exact is not None
                else None
            )
            if _approx_bytes is not None:
                _record_xla_memory(_approx_bytes, _exact_bytes)
    else:
        cosine_sim = 0.0
        frob_residual = 0.0

    # ---- THE COST FORM (ticket .9): see `_PAIRED_REF` and
    # `paired_log_costs`. Under ``absolute`` the two slots below carry the
    # measured numbers exactly as before (the flag-off bit-identity arm).
    if cost_form() == "paired-log":
        if _paired:
            # The reference's memory is the SAME quantity as the candidate's
            # (ticket .49): static temp under --mem-channel temp, the
            # runtime watermark (or its static substitution) otherwise.
            if mem_channel() == "temp":
                if _ref_temp is None:
                    raise MemChannelFault(
                        "paired reference: memory_analysis() returned "
                        "nothing for the rev-exact executable, so its static "
                        "temp bytes cannot be read")
                _ref_mem = float(_ref_temp)
            else:
                _ref_mem = float(_ref_peak)
            # Same fake-fast clamp the candidate's latency got above, so a
            # zero-work reference cannot make the pair implausible either.
            if 0.0 < _ref_lat_ns < _LAT_FLOOR_NS:
                _ref_lat_ns = _LAT_FLOOR_NS
            if not config.measure_latency:
                _ref_lat_ns = 0.0
            _abs_lat, _abs_mem = latency_ns, peak_memory
            latency_ns, peak_memory, _n_floored = paired_log_costs(
                _abs_lat, _abs_mem, _ref_lat_ns, _ref_mem)
            # TICKET dsnn-dfw.44: the band this plan is one measurement of.
            # The memory half has per-window samples only under the WATERMARK
            # channel; the static temp is one reading per executable.
            _lat_fb = (
                math.log(max(_abs_lat, _LAT_FLOOR_NS))
                - math.log(max(_ref_lat_ns, _LAT_FLOOR_NS))
                if (_abs_lat > 0.0 and _ref_lat_ns > 0.0) else 0.0)
            _mem_fb = (math.log(max(_abs_mem, _MEM_LOG_FLOOR_BYTES))
                       - math.log(max(_ref_mem, _MEM_LOG_FLOOR_BYTES)))
            _mem_c, _mem_r = ((), ()) if mem_channel() == "temp" else (
                peak_mem_samples, _ref_peak_samples)
            _ratio_log = {
                "latency": window_ratio_record(paired_window_log_ratios(
                    latency_samples if config.measure_latency else (),
                    _ref_lat_samples if config.measure_latency else (),
                    _LAT_FLOOR_NS, _lat_fb)),
                "memory": window_ratio_record(paired_window_log_ratios(
                    _mem_c, _mem_r, _MEM_LOG_FLOOR_BYTES, _mem_fb)),
            }
            _paired_ref_rec = {
                "ratio_log": _ratio_log,
                # POSITIVE units (ticket .45 logs these as ref/*).
                "latency_ns": float(_ref_lat_ns),
                "temp_bytes": _ref_temp,
                "watermark_bytes": float(_ref_peak),
                "memory_bytes": float(_ref_mem),
                "mem_channel": mem_channel(),
                "candidate_latency_ns": float(_abs_lat),
                "candidate_memory_bytes": float(_abs_mem),
                "delta_latency": float(latency_ns),
                "delta_memory": float(peak_memory),
                "mem_floored": int(_n_floored),
                "order": list(_rev_order),
            }
            _record_paired_ref(_paired_ref_rec)
            if os.environ.get("ALPHAGRAD_DEBUG_MEASURE", "0") == "1":
                print(f"[paired-ref] rev-exact lat={_ref_lat_ns/1e3:.1f}us "
                      f"mem={_ref_mem:.0f}B | candidate "
                      f"lat={_abs_lat/1e3:.1f}us mem={_abs_mem:.0f}B | "
                      f"Delta_lat={latency_ns:+.4f} Delta_mem={peak_memory:+.4f}"
                      f" floored={_n_floored}", flush=True)
        else:
            latency_ns, peak_memory = 0.0, 0.0
    _pf("cb.quality")

    # ------------------------------------------------------------------
    # FIDELITY -- reward slot 8. See the block comment above `fidelity_enabled`.
    # ------------------------------------------------------------------
    # Runs AFTER the cost measurement (so nothing it does is inside a timing
    # or peak-memory window; until 2026-09-04 also after the quality gate,
    # so a clamped plan was still scored). Until 2026-09-03 this block also ran the gradient-
    # coverage census and the frozen-gradient HARD GUARD on the same exact
    # reference; both were removed by owner ruling (ticket dsnn-3qm.15).
    fidelity = 0.0
    _fid_measured = False
    # (A) Under ALPHAGRAD_QUALITY_METRIC=cosine the residual was scored PER
    # POINT inside the measure loop off the exact reference the cosine already
    # needed, so there is nothing left to execute. Aggregate the FIDELITY
    # values (not the residuals) with the same summary the cosine gets: both
    # are "higher is better", so the two channels are summarised identically.
    if _fid_on and _fid_from_quality and _rel_frobs:
        _fids = [clipped_rel_frob(x) for x in _rel_frobs]
        fidelity = float(_aggregate_samples(_fids, want_top_quartile=True))
        frob_residual = float(np.median(
            np.asarray(_rel_frobs, dtype=np.float64)))
        _fid_measured = True
        _record_fidelity(
            fidelity, frob_residual,
            float(np.median(np.asarray(_cos_logged, dtype=np.float64)))
            if _cos_logged else None)
    if _fid_needs_ref and _fid_eval_args is not None:
        _fid_t0 = time.perf_counter()
        _rf_m = None
        _cos_m = None
        try:
            _rf_m, _cos_m = _exact_ref_scores(
                lambda: cached_compile(b"exact:" + exact_cache_key,
                                       _do_compile_exact),
                _fid_eval_args,
                config.has_aux,
                approx_out=_fid_approx_out,
                want_cos=True,
            )
        except Exception as _exc:
            _fid_approx_out = None
            # FAIL SOFT, LOUDLY. The exact reference is apparatus, not plan
            # quality: an exact reference we could not build is apparatus
            # failure, and scoring the plan -1.0 for it would be the very
            # bias `_truncated_reward` warns about. The slot stays 0.0 =
            # "not measured".
            if not _FID_REF_WARNED:
                _FID_REF_WARNED.append(1)
                print("[fidelity] WARNING: the EXACT reference gradient could "
                      "not be built for this order; fidelity is NOT MEASURED "
                      "for every affected plan: "
                      f"{type(_exc).__name__}: {str(_exc)[:160]}", flush=True)
        else:
            _fid_approx_out = None
            if _rf_m is not None:
                # A non-finite residual is a BROKEN COMPARISON (leaf-count or
                # shape mismatch -- `_residual_scores` returns nan for exactly
                # the cases `_quality_metrics` scores worst). That is evidence
                # about the PLAN, not the apparatus, so it takes the floor,
                # matching `_quality_metrics`'s documented rule.
                if np.isfinite(_rf_m):
                    frob_residual = float(_rf_m)
                    fidelity = clipped_rel_frob(_rf_m)
                else:
                    frob_residual = float("inf")
                    fidelity = -1.0
                _fid_measured = True
                _record_fidelity(fidelity, frob_residual, _cos_m)
            elif _cos_m is not None:
                # Cosine subsample with the channel off: record the cosine and
                # DO NOT touch the fidelity counter, so the amortised-cost line
                # attributes the exact reference to the right consumer.
                # (Branch (A) has already recorded its own measurement and
                # must not be double-counted here.)
                _record_fidelity(None, None, _cos_m)
        _FIDELITY_STATS["wall_s"] += time.perf_counter() - _fid_t0
    _pf("cb.fidelity")
    # ------------------------------------------------------------------
    # SPARSITY -- reward slot 10. See the block comment on
    # `_SPARSITY_STATS` for the definition and the hackability warning.
    # ------------------------------------------------------------------
    sparsity = 0.0
    if _sparsity_on:
        _sp_t0 = time.perf_counter()
        try:
            _ap, _ap_fb = _store_totals(_APPROX_STORE_BYTES, cache_key,
                                        True)
            _ex, _ex_fb = _store_totals(_EXACT_STORE_BYTES,
                                        exact_cache_key, False)
            _ex_b = float(_ex.get("bytes", 0))
            _ex_c = float(_ex.get("cells", 0))
            _ratio = (float(_ap.get("bytes", 0)) / _ex_b
                      if _ex_b > 0.0 else float("nan"))
            _cratio = (float(_ap.get("cells", 0)) / _ex_c
                       if _ex_c > 0.0 else float("nan"))
            sparsity = sparsity_channel(_ratio)
            _record_sparsity(_ratio, _cratio, sparsity, _ap, _ex,
                             int(_ap_fb) + int(_ex_fb),
                             time.perf_counter() - _sp_t0)
        except Exception as _exc:
            sparsity = 0.0
            _SPARSITY_STATS["wall_s"] += time.perf_counter() - _sp_t0
            _record_sparsity_failure(_exc)
    _pf("cb.sparsity")
    # A6: the slot list is NAMED before it becomes a device array. The plan
    # log records THESE host-side floats, so recording a terminal plan costs
    # no device->host transfer and no sync; `jnp.array` of this same list,
    # below, is bit-for-bit what this expression always built.
    _reward_slots = (
        [
            -muls_adds_fmas,
            -flops,
            -latency_ns,
            -max_io_sum,
            -bytes_accessed,
            -peak_memory,
            cosine_sim,
            # Slot 7: RESERVED (gradient coverage until 2026-09-03, the dead
            # frob slot before that). A literal 0.0, always -- byte-identical
            # to what the slot carried on every plan with the census off.
            0.0,
            # Slot 8: FIDELITY = clip(1 - rel_frob, -1, 1). 0.0 whenever it was
            # not measured (non-terminal, channel off, or the exact reference
            # failed), matching the sparse-terminal convention of slot 6.
            fidelity,
            # Slot 9: RESERVED (`bkstep_acc`, the deprecated Ray line's).
            # This env does not measure it and emits a literal 0.0 so the
            # two channel tables stay index-identical -- see the slot-9
            # entry in the REWARD_NAMES block for why it cannot be reused.
            0.0,
            # Slot 10: SPARSITY = clip(1 - stored_approx/stored_exact,
            # -1, 1). 0.0 whenever it was not measured, same convention.
            sparsity,
        ]
    )
    rewards = jnp.array(_reward_slots, dtype=jnp.float32)

    # ZERO-WORK PLANS ARE KEPT (user-directed 2026-07-28).
    #
    # A plan that computes nothing reports every COST channel at its best
    # (fake-fast latency, tiny peak memory) — historically it was sentinelled
    # because that is the classic reward hack. It is no longer refused:
    #   * `frob_residual` is EXACTLY 1.0 for an all-zero Jacobian, i.e. the
    #     `fidelity` channel reads EXACTLY 0.0 there (A2) -- not the -1.0 floor,
    #     which is reserved for a Jacobian that is actively wrong rather than
    #     absent. A zero-work plan therefore scores strictly below every plan
    #     that computes anything correct and strictly above one that computes
    #     something backwards, which is the ordering we want;
    #   * under PopArt each channel is normalised by its own running sigma, so
    #     frob's [0, 1] range is rescaled to compete on equal terms with
    #     nanoseconds and bytes — which is precisely the commensurability the
    #     symlog+lambda scheme lacked (there, destroying the Jacobian paid 8.8x
    #     better than computing it);
    #   * and a refusal is a FLAT signal, which is what froze v17.
    # So we let it measure, let the QUALITY CHANNEL punish it, and keep the
    # gradient. Under loss_drop that punishment is direct and no longer relies
    # on frob: a plan that computes nothing produces an all-zero gradient, the
    # 200-step Adam walk therefore never moves the weights, and the loss drop
    # is exactly 0.0 -- the floor for any plan that does not actively diverge.
    # This is telemetry only.
    if (not skip_count_ops() and is_terminal
            and (muls_adds_fmas <= 0.0) and (flops <= 0.0)):
        _record_zero_work_plan()
        if os.environ.get("ALPHAGRAD_DEBUG_DEGEN", "0") == "1":
            _n_skips = int(np.sum(_skips_np == 1)) if len(o_list) else 0
            _spec_rows = np.asarray(partial_specs)
            _n_quant = int(np.sum(_spec_rows[..., 0] == QUANT_SENTINEL))
            _n_comp = int(np.sum(_spec_rows[..., 0] == COMPRESS_SENTINEL))
            _n_diag = int(np.sum(_spec_rows[..., 0] >= 0))
            print(
                f"[zero-work] KEPT (the quality channel punishes): muls=0 "
                f"lat={latency_ns:.3g} peak={peak_memory:.3g} "
                f"{_qmetric}={cosine_sim:.3e} frob={frob_residual:.3e} "
                f"fid={fidelity:+.4f} "
                f"rules(d/c/q)={_n_diag}/{_n_comp}/{_n_quant} "
                f"skips={_n_skips} order={o_list}",
                flush=True,
            )

    # ---- THE COUNTS THIS PLAN WAS MEASURED WITH ----------------------
    # (owner ruling 2026-09-14). Recorded rather than assumed: under the time
    # budget the protocol is per-plan, not per-run.
    _measure_counts = {
        "inner": int(_inner),
        "windows": int(_n_windows),
        "secs": float(_meas_secs),
        "ref_inner": int(_ref_inner),
        "ref_windows": int(_ref_windows),
        "ref_secs": float(_ref_secs),
    }
    _LAST_MEASURE_COUNTS.clear()
    _LAST_MEASURE_COUNTS.update(_measure_counts)
    if is_terminal:
        _MEASURE_EPISODE["n_measured"] += 1
        _MEASURE_EPISODE["secs"].append(float(_meas_secs) + float(_ref_secs))
        _MEASURE_EPISODE["cand_secs"].append(float(_meas_secs))
        _MEASURE_EPISODE["ref_secs"].append(float(_ref_secs))
    # ---- THE DUPLICATE CACHE: this episode, this actor ----------------
    # Stored AFTER the whole reward vector exists, so a plan that raised or
    # was truncated on the way here leaves nothing behind for a duplicate to
    # inherit. The stored value is the host-side slot list, which is what the
    # duplicate's own `jnp.array` is built from and what its plan record
    # carries -- no device array is retained.
    if _dedupe_key is not None:
        _PLAN_DEDUPE[_dedupe_key] = (int(_plan_index), list(_reward_slots))

    # ---- A6 PLAN LOG: this plan, win or lose -------------------------
    if _plan_log_on:
        # THE POLICY'S OWN ORDER, not the transported one: the record is what
        # the policy emitted, and the container it implied is a field beside
        # it (`_PLAN_CARRY`, read in `_record_plan`).
        _record_terminal_plan(
            order=_rec_order, rule_specs=partial_specs,
            face_specs=_faces_np, face_skips=_skips_np,
            face_joins=_joins_np,
            reward_vec=_reward_slots,
            face_before=_plan_pf0, face_after=_PER_FACE_STATS,
            counts_from_trace=bool(_plan_traced[0]),
            mem_parity=_mp, paired_ref=_paired_ref_rec,
            measure_counts=_measure_counts)

    return _wire(tokens, eqn_ids, rewards)


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    config: EnvConfig
    args: tuple
    consts: tuple
    valid_vertices: tuple
    # Static per-vertex axis state derived from `config.jaxpr` at __init__
    # time. Stored as a JAX array so it travels through reset/step without
    # recomputation; values are constant across all episodes for a given
    # env. The pair `(axis_state_static, axis_valid_static)` matches the
    # shape contract documented on EnvState's `axis_state` / `axis_valid_mask`
    # fields.
    axis_state_static: Array | None = None
    axis_valid_static: Array | None = None
    num_envs: int | None = None
    eval_args_samples: tuple | None = None
    # Optional remote-evaluation pool. When set (typically by
    # ``mu0_ray_worker.SPMDServerWorker.init_server`` after spawning a
    # bank of ``CPUApproximationActor``s), ``tokenize()`` returns a
    # closure that dispatches ``(order, specs, step)`` requests to the
    # pool via ``ray.get(future, timeout=_remote_timeout_s)`` instead
    # of running ``jax.jit(jacve(...)).lower().compile()`` inline. The
    # pool field is an opaque Python object (a
    # :class:`alphagrad.approx.cpu_approx_pool.CpuApproxPool`), so it
    # rides in ``tree_flatten``'s ``aux_data`` rather than as a JAX
    # pytree child.
    _remote_pool: Any = None
    _remote_timeout_s: float = 60.0
    # WHICH ROLLOUT SHARD THIS ENV BELONGS TO (--rollout-shards). Data
    # parallelism over environments gives every GPU its own block of them and
    # every shard its own copy of this env, on its own device. The pair rides
    # in ``tree_flatten``'s aux data, so two shards are two different static
    # arguments and each gets its own trace -- which is what lets the step
    # callback of shard `i` be the one the rendezvous knows as `i`.
    # ``(0, 1)`` is a run without shards and wraps nothing at all.
    rollout_shard: int = 0
    rollout_shards: int = 1

    def __init__(
        self,
        config: EnvConfig,
        args: Sequence,
        consts: Sequence,
        valid_vertices: tuple | None = None,
        num_envs: int | None = None,
        eval_args_samples: tuple | None = None,
        axis_state_static: Array | None = None,
        axis_valid_static: Array | None = None,
        remote_pool: Any = None,
        remote_timeout_s: float = 60.0,
        rollout_shard: int = 0,
        rollout_shards: int = 1,
    ):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "eval_args_samples", eval_args_samples)
        object.__setattr__(self, "_remote_pool", remote_pool)
        object.__setattr__(self, "_remote_timeout_s", float(remote_timeout_s))
        if not (0 <= int(rollout_shard) < int(rollout_shards)):
            raise ValueError(
                f"rollout_shard {rollout_shard} is not in "
                f"0..{int(rollout_shards) - 1}")
        object.__setattr__(self, "rollout_shard", int(rollout_shard))
        object.__setattr__(self, "rollout_shards", int(rollout_shards))

        if num_envs is None:
            num_envs = jax.local_device_count()
        object.__setattr__(self, "num_envs", num_envs)

        if valid_vertices is None:
            _, _, _, vo_vertices = _build_graph(
                config.jaxpr, args, consts, config.argnums
            )
            valid = []
            for i, eqn in enumerate(config.jaxpr.eqns, 1):
                if eqn.outvars[0] not in config.jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        object.__setattr__(self, "valid_vertices", valid_vertices)

        if axis_state_static is None or axis_valid_static is None:
            total_v = len(config.jaxpr.eqns)
            axis_state_np, axis_valid_np = compute_static_axis_state(
                config.jaxpr, total_v,
            )
            axis_state_static = jnp.asarray(axis_state_np, dtype=jnp.int32)
            axis_valid_static = jnp.asarray(axis_valid_np, dtype=jnp.float32)
        object.__setattr__(self, "axis_state_static", axis_state_static)
        object.__setattr__(self, "axis_valid_static", axis_valid_static)
        # THE WINDOW BIN IS CHECKED WHERE THE ENV IS BUILT. It is one int
        # comparison, and this runs on every `tree_unflatten` too, so a bin
        # that is not a power of two or is under the fold chunk cannot reach
        # a trace at all.
        validate_delta_window(getattr(config, "delta_window", 0))

    @property
    def delta_window(self) -> int:
        """The per-step delta window BIN, in tokens. The cap when unset.

        This is the width of ``EnvState.delta_tokens`` and the slice
        ``step`` takes out of the wire. It is NOT ``obs_width``: the wire
        stays at ``DELTA_HEADER_SLOTS + MAX_DELTA_TOKENS``.
        """
        return validate_delta_window(
            getattr(self.config, "delta_window", 0))

    def with_delta_window(self, window):
        """A copy of this env at a different window bin.

        Everything bound on the env travels: the measurement pool and its
        timeout, the eval samples, the static axis state, the valid-vertex
        tuple. Losing the pool here would silently move every measurement
        from the actors back into the driver, which is why this is a method
        and not a `dataclasses.replace` at each call site.
        """
        w = validate_delta_window(window)
        if int(getattr(self.config, "delta_window", 0) or 0) == int(
                0 if w == MAX_DELTA_TOKENS else w):
            return self
        cfg = self.config._replace(
            delta_window=int(0 if w == MAX_DELTA_TOKENS else w))
        return type(self)(
            cfg,
            self.args,
            self.consts,
            self.valid_vertices,
            self.num_envs,
            self.eval_args_samples,
            axis_state_static=self.axis_state_static,
            axis_valid_static=self.axis_valid_static,
            remote_pool=self._remote_pool,
            remote_timeout_s=self._remote_timeout_s,
            rollout_shard=self.rollout_shard,
            rollout_shards=self.rollout_shards,
        )

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: core.ClosedJaxpr,
        argnums=None,
        args=None,
        has_aux=False,
        sparse=True,
        num_envs=None,
        data_gen: Callable | None = None,
        target_fun: Callable | None = None,
        cmp_type: str = "flops",
        mem_type: str = "peak_memory",
        exec_on_gpu: bool = False,
        measure_latency: bool = False,
        terminal_rewards_only: bool = False,
        latency_samples: int = 1,
        num_data_points: int = 5,
        reps_per_point: int = 4,
        # The paired rev-exact reference's OWN budget (owner ruling
        # 2026-09-14). Named explicitly rather than left to ``**_compat``,
        # which would swallow them silently.
        ref_num_data_points: int = 5,
        ref_reps_per_point: int = 32,
        measure_budget_secs: float = 1.0,
        measure_window_secs: float = 0.05,
        latency_inner_reps: int = 1,
        latency_warmup: int = 0,
        per_face: bool = False,
        measure_grad: bool = False,
        scalar_target: bool = False,
        delta_obs: bool = False,
        # THE PER-STEP DELTA WINDOW BIN. 0 means the hard cap, which is what
        # every caller that does not bin passes and what keeps their shapes
        # and their numbers exactly as they are. See EnvConfig.delta_window.
        delta_window: int = 0,
        quality_rewarded=None,
        grad_oracle_cadence: int | None = None,
        **_compat,
    ):
        # ``num_data_points`` / ``reps_per_point`` ARE wired (see EnvConfig and
        # the execution loop in ``_callback``). ``latency_samples`` is the
        # legacy spelling: when a caller passes it explicitly we honour it as
        # the total budget so old scripts keep working.
        if latency_samples and latency_samples > 1:
            reps_per_point = max(1, int(latency_samples) // max(1, num_data_points))
        # Is the traced target scalar? A FACT about the jaxpr, computed
        # unconditionally so ``EnvConfig.scalar_target`` records what was
        # actually traced (``quality_metric``'s ``auto`` default reads it) and
        # so the contract check below has one definition of "scalar", not two.
        _outs0 = getattr(jaxpr, "out_avals", None) or [
            getattr(v, "aval", None) for v in jaxpr.jaxpr.outvars
        ]
        _is_scalar = not [
            a for a in _outs0
            if a is not None and getattr(a, "shape", ()) not in ((), (1,))
        ]
        if measure_grad or scalar_target:
            # THE SCALAR-OUTPUT CONTRACT: the traced function is a SCALAR
            # loss, so differentiating its jaxpr already yields gradients —
            # which is what the spec asks to measure ("instead of returning
            # the Jacobian we return the gradients from the Jacobian"). It is
            # therefore a CONTRACT on the caller's target_fun, not extra work
            # for the env, and the previous hard NotImplementedError made
            # az_gumbel unimportable rather than protecting anything. Verify
            # the contract and continue.
            #
            # ``scalar_target`` is what ARMS the check on the PRODUCTION
            # paths, and they pass ``examples.has_scalar_loss(example)``: True
            # for every trainable family (whose registered target IS model +
            # loss) and False for the analytic AD benchmarks, which have no
            # training loss and are measured as full Jacobians on purpose.
            # ``--measure-grad`` does not arm anything any more.
            # It stays default-False so the callers that legitimately trace a
            # NON-scalar target (elimrl, bare-jaxpr tests) are unaffected.
            _outs = getattr(jaxpr, "out_avals", None) or [
                getattr(v, "aval", None) for v in jaxpr.jaxpr.outvars
            ]
            _bad = [
                a for a in _outs
                if a is not None and getattr(a, "shape", ()) not in ((), (1,))
            ]
            if _bad:
                raise ValueError(
                    "a SCALAR-output target is required (so jacve of it "
                    "yields gradients); got output avals "
                    f"{[getattr(a, 'shape', a) for a in _outs]}. The "
                    "registered target must BE model + loss (see "
                    "common.examples.get_fn), or drop scalar_target."
                )
        if argnums is not None and args is None:
            raise ValueError("argnums requires args: pass both or neither")
        config = EnvConfig(
            jaxpr=jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.invars)))
            if argnums is None
            else tuple(argnums),
            has_aux=has_aux,
            sparse=sparse,
            cmp_type=cmp_type,
            mem_type=mem_type,
            num_data_points=int(num_data_points),
            reps_per_point=int(reps_per_point),
            ref_num_data_points=int(ref_num_data_points),
            ref_reps_per_point=int(ref_reps_per_point),
            measure_budget_secs=float(measure_budget_secs),
            measure_window_secs=float(measure_window_secs),
            latency_inner_reps=int(latency_inner_reps),
            latency_warmup=int(latency_warmup),
            per_face=bool(per_face),
            target_fun=target_fun,
            data_gen=data_gen,
            exec_on_gpu=exec_on_gpu,
            measure_latency=measure_latency,
            terminal_rewards_only=terminal_rewards_only,
            delta_obs=bool(delta_obs),
            measure_grad=bool(measure_grad),
            scalar_target=bool(_is_scalar),
            delta_window=int(delta_window or 0),
            grad_oracle_cadence=int(grad_oracle_cadence) if grad_oracle_cadence is not None else 50,
        )
        return cls(
            config,
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            num_envs=num_envs,
        )

    def tree_flatten(self):
        children = (
            self.args, self.consts, self.eval_args_samples,
            self.axis_state_static, self.axis_valid_static,
        )
        # ``_remote_pool`` is an opaque Python object (a CpuApproxPool
        # instance, which itself holds Ray actor handles) — it can't
        # round-trip through JAX's pytree machinery as a child. Stash
        # it in aux_data alongside the other static fields. JAX will
        # call ``tree_unflatten`` whenever the env is reconstructed
        # inside a jit/vmap trace; we want the same pool handle to
        # come out the other side.
        aux_data = (
            self.config, self.valid_vertices, self.num_envs,
            self._remote_pool, self._remote_timeout_s,
            self.rollout_shard, self.rollout_shards,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts, eval_args_samples, axis_state_static, axis_valid_static = (
            children
        )
        # Back-compat: aux_data tuples produced before the remote-pool
        # fields were added are 3-tuples; the remote-pool ones are 5-tuples;
        # the rollout-shard ones are 7.
        rollout_shard, rollout_shards = 0, 1
        if len(aux_data) == 3:
            config, valid_vertices, num_envs = aux_data
            remote_pool = None
            remote_timeout_s = 60.0
        elif len(aux_data) == 5:
            (
                config, valid_vertices, num_envs,
                remote_pool, remote_timeout_s,
            ) = aux_data
        else:
            (
                config, valid_vertices, num_envs,
                remote_pool, remote_timeout_s,
                rollout_shard, rollout_shards,
            ) = aux_data
        return cls(
            config, args, consts, valid_vertices, num_envs, eval_args_samples,
            axis_state_static=axis_state_static,
            axis_valid_static=axis_valid_static,
            remote_pool=remote_pool,
            remote_timeout_s=remote_timeout_s,
            rollout_shard=rollout_shard,
            rollout_shards=rollout_shards,
        )

    def with_rollout_shard(self, shard: int, n_shards: int):
        """A copy of this env as rollout shard `shard` of `n_shards`.

        Everything bound on the env travels, exactly as in
        :meth:`with_delta_window`: losing the measurement pool here would move
        every measurement back into the driver in silence.
        """
        if (int(shard) == int(self.rollout_shard)
                and int(n_shards) == int(self.rollout_shards)):
            return self
        return type(self)(
            self.config,
            self.args,
            self.consts,
            self.valid_vertices,
            self.num_envs,
            self.eval_args_samples,
            axis_state_static=self.axis_state_static,
            axis_valid_static=self.axis_valid_static,
            remote_pool=self._remote_pool,
            remote_timeout_s=self._remote_timeout_s,
            rollout_shard=int(shard),
            rollout_shards=int(n_shards),
        )

    def _shard_wrap(self, fn):
        """The BATCHED step callback as rollout shard `self.rollout_shard`.

        Without shards (`rollout_shards == 1`) this is the identity and the
        trace is the one a run without the flag produced. With shards it is
        the rendezvous of :mod:`common.rollout_shards`: the N shards deposit
        their `E` environment rows, ONE of them calls `fn` once over all
        `N*E` rows in global environment order, and each takes its own slice
        back. The pool therefore still sees exactly one `evaluate_batch` per
        step, and the terminal step still makes one submission under one
        ticket.
        """
        if int(self.rollout_shards) <= 1:
            return fn
        g = shard_gather()
        if g is None:
            raise RuntimeError(
                f"this env is rollout shard {self.rollout_shard} of "
                f"{self.rollout_shards} but no measurement rendezvous is "
                f"installed. `ppo.main` calls `env.set_shard_gather(...)` "
                f"before it traces a shard; a caller that shards the rollout "
                f"itself has to do the same.")
        if int(g.n) != int(self.rollout_shards):
            raise RuntimeError(
                f"the installed measurement rendezvous is for {g.n} shards "
                f"and this env is shard {self.rollout_shard} of "
                f"{self.rollout_shards}.")
        return g.wrap(int(self.rollout_shard), fn)

    def tokenize(self, init: bool = False, batched: bool = False,
                 bound_dropped: bool = False):
        """Build the host-side function passed into ``io_callback``.

        If ``self._remote_pool`` is set, return a closure that
        dispatches each call to the pool via ``ray.get(timeout=...)``;
        on timeout / actor death the closure returns the standard
        sentinel ``(zeros, -1e10 reward)`` tuple so the rollout
        proceeds. Otherwise fall back to the inline ``_callback``
        path (single-process, no Ray) for backward compatibility
        with ``ppo.py`` / non-Ray callers.

        ``bound_dropped`` says the caller handed placeholders instead of the
        real ``args`` / ``consts`` / eval samples, because the pool already
        owns them. A row served in THIS process then reads them from
        :func:`local_bound_operands` instead.
        """
        if batched and int(self.rollout_shards) > 1 and not _BATCHED_CALLBACK:
            # THE RENDEZVOUS ONLY EXISTS FOR THE BATCHED CALLBACK. Without
            # ALPHAGRAD_BATCHED_CALLBACK the step callback runs once per
            # environment, there is nothing to gather, and every shard would
            # make its own measurement call -- which the pool answers by
            # sentinelling the rows it could not place an actor for. Refuse
            # here rather than in the driver alone, so a second caller that
            # shards a rollout cannot reach that state at all.
            raise RuntimeError(
                f"this env is rollout shard {self.rollout_shard} of "
                f"{self.rollout_shards} and ALPHAGRAD_BATCHED_CALLBACK is "
                f"off, so its step callback runs one environment at a time "
                f"and the shards have nothing to gather into one measurement "
                f"call.")
        if self._remote_pool is None:
            # THE JOIN CHANNEL IS KEYWORD-ONLY ON `_callback`, ON PURPOSE.
            # `io_callback` / `pure_callback` pass their operands POSITIONALLY,
            # and this env's own plumbing sends `face_joins` between
            # `face_skips` and `stop` -- but `_callback` is called directly from
            # a dozen other places (the measure actors, landscape_map, az_gumbel,
            # six test modules), and adding a positional there silently shifted
            # `stop` into `face_joins` for every one of them. Measured: it turned
            # `tests/delta_obs_emission_test.py` from 8 passed to 4 failed.
            # So the adapter absorbs the positional here and `_callback`'s
            # signature is UNCHANGED for everybody else.
            def _fn(args, consts, order, specs, face_specs, face_skips,
                    face_joins, stop, *eval_samples):
                return _callback(
                    self.config, args, consts, order, specs, face_specs,
                    face_skips, stop, *eval_samples, init=init,
                    face_joins=face_joins)
            # Only the STEP callback runs under vmap; reset() is called once,
            # unbatched, and must not be wrapped.
            if batched and _BATCHED_CALLBACK:
                return self._shard_wrap(
                    _batched_host(_fn, n_out=self.wire_arity))
            return _fn

        # The pool's ``evaluate`` signature is
        # ``(order, specs, step, eval_samples, *, init)`` — but
        # ``io_callback`` passes ``(args, consts, order, specs, step,
        # *eval_samples)`` as positional args (see ``env.step`` line
        # 1360 and ``env.reset`` line 1297). Build an adapter that
        # drops ``args``/``consts`` (the actor's own env has its own
        # bound args) and re-packages ``eval_samples`` as a tuple.
        pool = self._remote_pool
        _obs_w = self.obs_width
        # The wire the pool and this shim must BOTH produce. Read off the env
        # so the three descriptions of one buffer (here, the pool's
        # preallocation, `_callback_shape`) cannot disagree. ``_eqn`` is False
        # under ``delta_obs``: the equation-id buffer does not exist there.
        _tok_dt = self.wire_token_dtype
        _eqn = not self.config.delta_obs
        _eqn_dt = self.wire_eqn_dtype if _eqn else None

        if batched:
            def _remote_callback_batched(args, consts, order, specs,
                                         face_specs, face_skips, face_joins,
                                         step, *eval_samples):
                # P3: the pool stack CARRIES face wires now (worker builds
                # empty ones only when handed None), so face-action rows
                # ship through instead of raising. TERMINAL rows measure on
                # the trainer's own device when exec_on_gpu — the pool's
                # CPU actors must not time a GPU campaign — while
                # non-terminal rows (tokens-only under
                # terminal_rewards_only) shard to the pool.
                _trace("cb_batched.enter")
                # prof/env_cb_host: the WHOLE host callback, so the episode
                # table closes without an event trace. `prof/measure_wait` is
                # nested inside this, and this is nested inside ppo's
                # `prof/envcb` mark interval (which adds only the mark's own
                # dispatch on either side).
                _cb0 = time.perf_counter()
                _o = np.asarray(order)
                E = int(_o.shape[0])
                _st = np.asarray(step).reshape(-1)
                _sti = [int(_st[i] if _st.size > 1 else _st[0])
                        for i in range(E)]
                _ev = (tuple(_cb_slot(x, 0, E) for x in eval_samples)
                       if eval_samples else None)
                ro = [np.asarray(_cb_slot(order, i, E)) for i in range(E)]
                rs = [np.asarray(_cb_slot(specs, i, E)) for i in range(E)]
                # LIVE PREFIX ONLY. `_callback` reads `face_specs[:stop]` and
                # nothing else, but the wire buffer is (N, MAX_FACES,
                # FACE_SLOTS, 3) int32 -- 8.7 MB at the flagship's N=95 /
                # MAX_FACES=2538 -- and every byte of it was being pickled
                # and shipped to the actor on every step, including the
                # all-(-1) rows for eliminations that have not happened yet.
                # Slicing here is exactly what the callee would have sliced.
                rf = [np.ascontiguousarray(
                          np.asarray(_cb_slot(face_specs, i, E))[:_sti[i]])
                      for i in range(E)]
                rk = [np.ascontiguousarray(
                          np.asarray(_cb_slot(face_skips, i, E))[:_sti[i]])
                      for i in range(E)]
                rj = ([None] * E if face_joins is None else
                      [np.ascontiguousarray(
                          np.asarray(_cb_slot(face_joins, i, E))[:_sti[i]])
                       for i in range(E)])
                _any_faces = any(
                    (f[..., 0] >= 0).any() or (k == 1).any()
                    or (f[..., 0] == COMPRESS_SENTINEL).any()
                    or (f[..., 0] == QUANT_SENTINEL).any()
                    for f, k in zip(rf, rk))
                # Terminal rows go to the pool BY DEFAULT — measure actors
                # are GPU-pinned (#23) and their idle device is a quieter
                # timer than the busy trainer GPU. Opt into trainer-local
                # terminals (CPU-only actor pools) with
                # ALPHAGRAD_POOL_TERMINAL_LOCAL=1.
                _term_local = os.environ.get(
                    "ALPHAGRAD_POOL_TERMINAL_LOCAL", "0") == "1"
                _is_term = [_sti[i] >= int(ro[i].shape[0]) for i in range(E)]
                # WHO RUNS THE PER-STEP TOKENIZATION (owner ruling
                # 2026-09-15). A NON-TERMINAL row is tokenization and nothing
                # else: under `terminal_rewards_only` `_callback_measured`
                # returns right after the delta observation and never
                # compiles or times anything. So it does not need a
                # measurement actor -- and while it rides one, a terminal
                # measurement left in flight blocks the next rollout step for
                # step, because Ray runs one task per actor.
                #
                # `--tokenize-where` says where that work runs. `pool` is the
                # historical route (the measure actors). `local` keeps it in
                # the trainer process, which is what the non-pooled path has
                # always done. `cpu-actors` sends it to a SECOND pool of
                # CPU-only actors that measures nothing.
                #
                # THE TOKENS DO NOT DEPEND ON THE CHOICE. The stream for a
                # prefix is a pure function of (jaxpr, argnums, consts, args,
                # prefix, decoded rules, face wires); the incremental caches
                # are caches, and a cold process replays the prefix to the
                # same bytes. Only locality and IPC differ.
                _twhere = tokenize_where()
                _tokpool = tokenize_pool() if _twhere == "cpu-actors" else None
                if _twhere == "cpu-actors" and _tokpool is None:
                    raise RuntimeError(
                        "--tokenize-where cpu-actors, but no tokenize pool "
                        "was installed: env.set_tokenize_where('cpu-actors', "
                        "pool=...) must be called with the CPU-only pool "
                        "before the first rollout.")
                _local = set(
                    i for i in range(E)
                    if (_term_local and _is_term[i])
                    or (_twhere == "local" and not _is_term[i]))
                _tokrows = [i for i in range(E)
                            if _twhere == "cpu-actors" and not _is_term[i]
                            and i not in _local]
                _tokset = set(_tokrows)
                _remote = [i for i in range(E)
                           if i not in _local and i not in _tokset]
                # THE PIPELINED TERMINAL ROWS (owner ruling 2026-09-14). With
                # a ticket open, a terminal row is SUBMITTED and left to the
                # actors; its tokens and its reward come back as zeros and the
                # driver fills the reward into the trajectory when it collects
                # the ticket. Non-terminal rows are unaffected: they carry the
                # tokenization the next step's encoder reads, so they are
                # still served synchronously here, wherever they are served.
                _ticket = current_measure_ticket()
                _pipe = []
                if _ticket is not None:
                    _pipe = [i for i in _remote if _is_term[i]]
                    if _pipe:
                        _pipe_set = set(_pipe)
                        _remote = [i for i in _remote
                                   if i not in _pipe_set]
                # THE POOL PROTOCOL DOES NOT CARRY THE JOIN BIT, AND SAYS SO.
                # `CpuApproxPool.evaluate_batch` takes face_specs / face_skips
                # and nothing else, so a remote row would be measured under
                # the configuration's default container while the trainer
                # stored the log-prob of the bit the head actually drew --
                # finding 72's action/reward mismatch, one process boundary
                # along. Rows served IN-PROCESS below DO carry it, so this is
                # a pool-protocol gap and not a semantics gap; widening
                # `cpu_approx_pool` / `cpu_approx_actors` / `cpu_approx_worker`
                # by one keyword is the remaining work (ticket dsnn-3qm.56).
                #
                # THE CHECK STANDS AFTER THE CLASSIFICATION NOW. It used to
                # stand above it and read `_remote`, a name Python binds
                # locally further down, so a pooled `choose` run raised
                # UnboundLocalError instead of this message. Reading the
                # classified lists is also the honest test: under
                # `--tokenize-where local` the non-terminal rows carry the
                # join bit again and only the terminal ones leave the process.
                if face_joins is not None and (_remote or _pipe or _tokrows):
                    raise NotImplementedError(
                        "--approx-add choose decides the face join PER FACE, "
                        "and the Ray measurement pool's protocol carries only "
                        "face_specs / face_skips. The actors would measure "
                        f"every merge under {approx_add()!r}'s default "
                        "container while the stored log-prob scored the bit "
                        "the head drew. Run choose without a measure pool, or "
                        "with ALPHAGRAD_POOL_TERMINAL_LOCAL=1 and no "
                        "non-terminal remote rows.")
                tk = np.zeros((E, _obs_w), _tok_dt)
                ei = np.zeros((E, _obs_w), _eqn_dt) if _eqn else None
                rw = np.zeros((E, NUM_REWARDS), np.float32)

                def _wire_row(a, want_dt, name):
                    """One env's wire row, shape- and range-checked.

                    The pool's buffers and this shim's are two descriptions of
                    one wire. A width mismatch used to broadcast or truncate
                    inside the assignment below; a value outside the narrow
                    dtype used to wrap at the cast. Both raise now.
                    """
                    a = np.asarray(a)
                    if a.shape != (_obs_w,):
                        raise ValueError(
                            f"measurement wire {name} has shape {a.shape}, "
                            f"expected ({_obs_w},) -- the Ray pool and the "
                            f"batched host shim must agree on the wire "
                            f"arity (env.obs_width).")
                    if a.dtype != want_dt:
                        _info = np.iinfo(want_dt)
                        _a64 = a.astype(np.int64)
                        if _a64.size and (int(_a64.min()) < _info.min
                                          or int(_a64.max()) > _info.max):
                            raise ValueError(
                                f"measurement wire {name} carries values "
                                f"outside {want_dt.__name__} "
                                f"[{_info.min}, {_info.max}]; casting would "
                                f"WRAP them.")
                    return a.astype(want_dt)
                if _remote:
                    # prof/measure_wait: the host BLOCKS here until the
                    # measurement actors return. Timed separately from the
                    # cb.* phases because it is not trainer compute at all --
                    # it is idle time the rollout pays per (terminal) step.
                    _trace("measure_wait.enter")
                    _mw0 = time.perf_counter()
                    try:
                        _pb = pool.evaluate_batch(
                            [ro[i] for i in _remote],
                            [rs[i] for i in _remote],
                            [_sti[i] for i in _remote],
                            eval_samples=_ev,
                            init=init,
                            face_specs_batch=(
                                [rf[i] for i in _remote] if _any_faces
                                else None),
                            face_skips_batch=(
                                [rk[i] for i in _remote] if _any_faces
                                else None),
                            # A3 rotation, pooled path: the TRAINER's current
                            # episode travels with the request because the
                            # actor's inherited environment is frozen at
                            # ray.init. `None` unless rotation is on, so the
                            # flag-off wire is unchanged.
                            episode=(walk_episode() if walk_rotate_enabled()
                                     else None),
                        )
                    finally:
                        _mwdt = time.perf_counter() - _mw0
                        _prof_add("prof/measure_wait", _mwdt)
                        _prof_sample("prof/measure_wait", _mwdt)
                        _trace("measure_wait.exit")
                    # `(tokens, rewards, sentinel)` under delta_obs,
                    # `(tokens, eqn_ids, rewards, sentinel)` on the legacy
                    # path -- the pool serves the arity the env declares.
                    tokens, rewards = _pb[0], _pb[-2]
                    eqn_ids = _pb[1] if _eqn else None
                    for k2, i in enumerate(_remote):
                        tk[i] = _wire_row(np.asarray(tokens)[k2], _tok_dt,
                                          "tokens")
                        if _eqn:
                            ei[i] = _wire_row(np.asarray(eqn_ids)[k2],
                                              _eqn_dt, "eqn_ids")
                        rw[i] = np.asarray(rewards)[k2]
                if _tokrows:
                    # --tokenize-where cpu-actors. The SAME wire, a DIFFERENT
                    # pool: these actors hold no GPU and measure nothing, so
                    # the measure pool stays free for the terminal plan that
                    # is in flight behind them. `prof/tokenize_wait` is the
                    # per-step host cost of this arm, the number
                    # `prof/measure_wait` reports for `--tokenize-where pool`.
                    _trace("tokenize_wait.enter")
                    _tw0 = time.perf_counter()
                    try:
                        _tb = _tokpool.evaluate_batch(
                            [ro[i] for i in _tokrows],
                            [rs[i] for i in _tokrows],
                            [_sti[i] for i in _tokrows],
                            eval_samples=_ev,
                            init=init,
                            face_specs_batch=(
                                [rf[i] for i in _tokrows] if _any_faces
                                else None),
                            face_skips_batch=(
                                [rk[i] for i in _tokrows] if _any_faces
                                else None),
                            episode=(walk_episode() if walk_rotate_enabled()
                                     else None),
                        )
                    finally:
                        _twdt = time.perf_counter() - _tw0
                        _prof_add("prof/tokenize_wait", _twdt)
                        _prof_sample("prof/tokenize_wait", _twdt)
                        _trace("tokenize_wait.exit")
                    _t_tok, _r_tok = _tb[0], _tb[-2]
                    _e_tok = _tb[1] if _eqn else None
                    for k2, i in enumerate(_tokrows):
                        tk[i] = _wire_row(np.asarray(_t_tok)[k2], _tok_dt,
                                          "tokens")
                        if _eqn:
                            ei[i] = _wire_row(np.asarray(_e_tok)[k2],
                                              _eqn_dt, "eqn_ids")
                        rw[i] = np.asarray(_r_tok)[k2]
                if _local:
                    # `prof/tokenize_local` is the TRAINER-PROCESS half of the
                    # per-step cost: under `--tokenize-where local` this loop
                    # is the whole tokenization of the step, and under
                    # ALPHAGRAD_POOL_TERMINAL_LOCAL=1 it also holds the
                    # terminal measurement. One key, and the arm the operator
                    # ran says which of the two it is.
                    _tl0 = time.perf_counter()
                    # THE REAL BOUND OPERANDS. `step` handed this callback
                    # zero-length placeholders when the pool owns them, so a
                    # local row has to read the trainer's own copy instead --
                    # the tokenizer builds its graph from `args` and `consts`
                    # and a placeholder makes graphax refuse the jaxpr.
                    if bound_dropped:
                        _lb = local_bound_operands()
                        if _lb is None:
                            raise RuntimeError(
                                "a row is served in the trainer process, but "
                                "the pool owns the bound operands and none "
                                "were installed. The driver must call "
                                "env.set_local_bound_operands(env.args, "
                                "env.consts, env.eval_args_samples) before "
                                "the first rollout.")
                        _l_args, _l_consts, _l_ev = _lb
                        _l_ev = tuple(_l_ev or ())
                    else:
                        _l_args, _l_consts, _l_ev = None, None, None
                for i in _local:
                    _out = _callback(
                        self.config,
                        (_l_args if bound_dropped
                         else _cb_slot(args, i, E)),
                        (_l_consts if bound_dropped
                         else _cb_slot(consts, i, E)),
                        ro[i], rs[i], rf[i], rk[i], _sti[i],
                        *(_l_ev if bound_dropped
                          else [_cb_slot(x, i, E) for x in eval_samples]),
                        init=init, face_joins=rj[i],
                    )
                    tk[i] = _wire_row(_out[0], _tok_dt, "tokens")
                    if _eqn:
                        ei[i] = _wire_row(_out[1], _eqn_dt, "eqn_ids")
                    rw[i] = np.asarray(_out[-1])
                if _local:
                    _tldt = time.perf_counter() - _tl0
                    _prof_add("prof/tokenize_local", _tldt)
                    _prof_sample("prof/tokenize_local", _tldt)
                if _pipe:
                    # SUBMIT AND RETURN. `tk` and `rw` are already zeros for
                    # these rows and they stay that way: a zero delta header
                    # decodes to count 0, so the terminal step contributes an
                    # empty delta to the bootstrap carry (masked by done=1 in
                    # GAE) and nothing to either bin, and the zero reward
                    # vector is overwritten on the host when the driver
                    # collects the ticket.
                    _trace("measure_submit.enter")
                    _ms0 = time.perf_counter()
                    _pipe_kw = dict(
                        eval_samples=_ev,
                        init=init,
                        face_specs_batch=(
                            [rf[i] for i in _pipe] if _any_faces
                            else None),
                        face_skips_batch=(
                            [rk[i] for i in _pipe] if _any_faces
                            else None),
                        episode=(walk_episode() if walk_rotate_enabled()
                                 else None),
                        # THE ENVIRONMENT ROWS of these slots. `_ENV_SLOT` is
                        # a fact of the in-process batch loop and does not
                        # cross the Ray hop, so the row travels in the
                        # request; a probe batch redrawn per environment
                        # (owner ruling 2026-09-16) reads it back through
                        # `current_env_slot`.
                        env_rows=list(_pipe),
                    )
                    _pipe_pos = ([ro[i] for i in _pipe],
                                 [rs[i] for i in _pipe],
                                 [_sti[i] for i in _pipe])
                    try:
                        if measure_defer_enabled():
                            # PACKAGE ONLY. The driver starts it once the
                            # previous episode is collected and drained; see
                            # `start_measurement`.
                            _fut = None
                            _defer = (lambda _p=_pipe_pos, _k=_pipe_kw:
                                      pool.submit_batch(*_p, **_k))
                        else:
                            _defer = None
                            _fut = pool.submit_batch(*_pipe_pos, **_pipe_kw)
                    finally:
                        _msdt = time.perf_counter() - _ms0
                        _prof_add("prof/measure_submit", _msdt)
                        _prof_sample("prof/measure_submit", _msdt)
                        _trace("measure_submit.exit")
                    _record_measure_submission(
                        _ticket, _fut, _pipe, E,
                        int(_sti[_pipe[0]]), deferred=_defer)
                _cbdt = time.perf_counter() - _cb0
                _prof_add("prof/env_cb_host", _cbdt)
                _prof_sample("prof/env_cb_host", _cbdt)
                _trace("cb_batched.exit")
                return (tk, ei, rw) if _eqn else (tk, rw)

            return self._shard_wrap(_remote_callback_batched)

        def _remote_callback(args, consts, order, specs, face_specs,
                             face_skips, face_joins, step, *eval_samples):
            # The Ray pool path predates face actions (DEPRECATED line) —
            # they are dropped here; the pool's own env measures per-vertex.
            if face_joins is not None:
                raise NotImplementedError(
                    "--approx-add choose needs the per-face join bit at the "
                    "measurement, and this UNBATCHED Ray pool closure drops "
                    "the face wires entirely (its actors measure per-vertex). "
                    "It would measure every merge under the configuration's "
                    "default while the trainer stored the log-prob of the bit "
                    "the head drew. Use the batched pool path "
                    "(ALPHAGRAD_BATCHED_CALLBACK) or no pool.")
            eval_samples_t = tuple(eval_samples) if eval_samples else None
            _mw0 = time.perf_counter()
            try:
                _pout = pool.evaluate(
                    order, specs, int(step),
                    eval_samples=eval_samples_t,
                    init=init,
                    episode=(walk_episode() if walk_rotate_enabled()
                             else None),
                )
            finally:
                _mwdt = time.perf_counter() - _mw0
                _prof_add("prof/measure_wait", _mwdt)
                _prof_sample("prof/measure_wait", _mwdt)
            # Same wire contract as the batched shim above: the pool may hand
            # back its own preallocation dtype, and `io_callback` demands the
            # dtype `_callback_shape` declares. The pool returns two arrays
            # plus the reward on the legacy path and one plus the reward under
            # ``delta_obs`` (no equation-id buffer exists there).
            _t = np.asarray(_pout[0])
            _rw = _pout[-1]
            if _t.shape != (_obs_w,):
                raise ValueError(
                    f"measurement wire tokens have shape {_t.shape}, expected "
                    f"({_obs_w},) -- the Ray pool and the env must agree on "
                    f"the wire arity (env.obs_width).")
            if not _eqn:
                return _t.astype(_tok_dt), _rw
            _e = np.asarray(_pout[1])
            if _e.shape != (_obs_w,):
                raise ValueError(
                    f"measurement wire eqn_ids have shape {_e.shape}, "
                    f"expected ({_obs_w},).")
            return _t.astype(_tok_dt), _e.astype(_eqn_dt), _rw

        return _remote_callback

    @property
    def obs_width(self) -> int:
        """Width of the callback's token output (and, on the legacy path, of
        its equation-id output).

        ``DELTA_HEADER_SLOTS + MAX_DELTA_TOKENS`` under ``delta_obs`` (count
        header + delta), ``MAX_TOKENS`` for the legacy full stream. The Ray
        measurement pool preallocates its buffers at this width too
        (``CpuApproxPool(max_tokens=...)``), so it must be read from the env,
        not assumed.
        """
        return (delta_wire_width() if self.config.delta_obs else MAX_TOKENS)

    @property
    def wire_token_dtype(self):
        """numpy dtype of the callback's TOKEN output.

        uint8 under ``delta_obs`` (the tokenizer id space is 256, see
        ``common.token_vocab``), int32 on the legacy full-stream path, whose
        buffer is not narrowed. Read by the Ray pool and by the batched host
        shim so the three of them cannot name different dtypes.
        """
        return DELTA_TOKEN_DTYPE if self.config.delta_obs else np.int32

    @property
    def wire_eqn_dtype(self):
        """numpy dtype of the LEGACY full-stream equation-id output.

        Meaningless under ``delta_obs``, which emits no such buffer at all --
        ``_callback_shape`` is two entries long there. Kept for the legacy
        drivers (gfn, gdpo, mu0, alpha0, az_gumbel, ppo_ray_worker) whose
        dense encoder still consumes the ids as a pairwise T5 bias.
        """
        if self.config.delta_obs:
            raise RuntimeError(
                "delta_obs carries no equation-id buffer; there is no wire "
                "dtype to report. The ids were removed on 2026-09-13.")
        return np.int32

    @property
    def _callback_shape(self):
        """What the host callback returns, exactly.

        TWO entries under ``delta_obs`` -- ``(tokens uint8, reward)`` -- and
        THREE on the legacy full-stream path, which still carries an
        equation-id buffer for the dense encoder's pairwise T5 bias. See
        ``_callback._wire``.
        """
        _w = self.obs_width
        _tok = jax.ShapeDtypeStruct((_w,), jnp.dtype(self.wire_token_dtype))
        _rw = jax.ShapeDtypeStruct((NUM_REWARDS,), jnp.float32)
        if self.config.delta_obs:
            return (_tok, _rw)
        return (
            _tok,
            jax.ShapeDtypeStruct((_w,), jnp.dtype(self.wire_eqn_dtype)),
            _rw,
        )

    @property
    def wire_arity(self) -> int:
        """Number of arrays the callback returns; 2 or 3. Read by the batched
        host shim (``_batched_host``) and by the Ray pool, which must stack
        exactly as many."""
        return len(self._callback_shape)

    def base_observation(self):
        """The BASE token stream as a standalone constant buffer.

        ``len(base_tokens())`` depends only on the jaxpr, not on the
        elimination order, so the base is the SAME array for every env and
        every episode: compute it once, on the host, and share it. There is
        no callback, no per-env copy, and the length is the tokenizer's own
        -- not a device-side scan over a padded buffer, which is where the
        id-0 undercount used to come from.

        Returns ``(tokens, count)`` -- ``(MAX_BASE_TOKENS,) uint8`` and a
        python int. The equation-id buffer that used to ride beside it is
        gone (2026-09-13); ``base_owners`` is the per-token attribution the
        base scatter actually keys on, and always was.
        """
        from graphax import IncrementalPathTokenizer

        vocab = incr_token_vocab()
        tk = IncrementalPathTokenizer(
            self.config.jaxpr, tuple(self.config.argnums),
            list(self.consts), list(self.args), vocab_size=vocab,
        )
        toks = [int(t) for t in tk.base_tokens()]
        guard = os.environ.get("ALPHAGRAD_VOCAB_SIZE")
        if guard is not None and tk.max_token_id() >= int(guard):
            raise ValueError(
                f"incremental token ids reach {tk.max_token_id()} but the "
                f"policy embedding has only {guard} rows -- raise "
                f"--vocab-size or lower ALPHAGRAD_INCR_TOKEN_VOCAB. (JAX "
                f"CLAMPS an out-of-range gather, silently reading the wrong "
                f"row.)"
            )
        n = len(toks)
        if n > MAX_BASE_TOKENS:
            raise ValueError(
                f"base token stream is {n} tokens > MAX_BASE_TOKENS="
                f"{MAX_BASE_TOKENS}. Raise ALPHAGRAD_MAX_BASE_TOKENS -- a "
                f"CLIPPED base desyncs the encoder's recurrence from the "
                f"stream for the whole episode."
            )
        _record_token_length(n)
        t = np.zeros((MAX_BASE_TOKENS,), dtype=DELTA_TOKEN_DTYPE)
        if n:
            _tk = np.asarray(toks, dtype=np.int64)
            # SAME raise as the delta path: a base token past the byte means
            # the tokenizer was built at a vocabulary the wire cannot carry.
            _check_delta_ids(_tk, where="base_observation")
            t[:n] = _tk.astype(DELTA_TOKEN_DTYPE)
        return jnp.asarray(t), n

    def base_owners(self):
        """Per-token OWNING VERTEX for the base stream, ``(MAX_BASE_TOKENS,)``.

        1-based vertex, 0 = no owner (the ``inputs`` header, shape
        declarations, and anything not emitted from an equation).

        This is what the per-vertex memory must be keyed on. The stream's
        ``eqn_ids`` CANNOT serve: they are stream-global SEGMENT ids -- the
        whole base block is one ``_emit_eqns`` call, so every base equation
        token shares id 0, and reading them as vertex indices credited the
        entire base stream to vertex 1 (#92).

        Returns an all-zero array (i.e. everything to the global slot, the
        safe #92 behaviour) against a graphax without ``last_owner_ids``.
        """
        from graphax import IncrementalPathTokenizer

        vocab = incr_token_vocab()
        tk = IncrementalPathTokenizer(
            self.config.jaxpr, tuple(self.config.argnums),
            list(self.consts), list(self.args), vocab_size=vocab,
        )
        toks = tk.base_tokens()
        own_fn = getattr(tk, "last_owner_ids", None)
        owners = [int(v) for v in own_fn()] if own_fn is not None else []
        o = np.zeros((MAX_BASE_TOKENS,), dtype=np.int32)
        k = min(len(owners), len(toks), MAX_BASE_TOKENS)
        if k:
            o[:k] = np.asarray(owners[:k], dtype=np.int32)
        return jnp.asarray(o)

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        initial_specs = jnp.full(
            (initial_order.shape[0], MAX_RULES_PER_VERTEX, 3),
            -1,
            dtype=jnp.int32,
        )
        initial_specs = initial_specs.at[..., 2].set(0)  # factor=0 default for unused rows
        initial_face_specs = jnp.full(
            (initial_order.shape[0], MAX_FACES, wire_slots(), 3), -1,
            dtype=jnp.int32,
        )
        initial_face_skips = jnp.zeros(
            (initial_order.shape[0], MAX_FACES), dtype=jnp.int32
        )
        # THE JOIN CHANNEL EXISTS IFF THE WIDTH HAS THE BIT. `None` under every
        # fixed value, so `resolve_join_mode` answers from the configuration
        # and a flag-off state carries no extra leaf at all.
        initial_face_joins = (
            jnp.zeros((initial_order.shape[0], MAX_FACES), dtype=jnp.int32)
            if approx_add() not in APPROX_ADD_FIXED else None)

        if self.config.delta_obs:
            # The base stream is a host-side CONSTANT (base_observation()),
            # not part of the state, so reset makes NO host callback: at
            # step 0 nothing has been eliminated and the delta is empty.
            tokens = jnp.zeros((1,), dtype=jnp.int32)
            eqn_ids = jnp.zeros((1,), dtype=jnp.int32)
            # THE BIN, not the cap: this buffer is carried through the whole
            # rollout scan and read by the loss, so it is the shape the
            # window bin exists to shrink.
            delta_tokens = jnp.zeros((self.delta_window,),
                                     dtype=DELTA_TOKEN_DTYPE)
            delta_count = jnp.zeros((), dtype=jnp.int32)
        else:
            tokens, eqn_ids, _ = _env_callback(
                self.tokenize(init=True),
                self._callback_shape,
                self.args,
                self.consts,
                initial_order,
                initial_specs,
                initial_face_specs,
                initial_face_skips,
                initial_face_joins,
                0,
                *(self.eval_args_samples
                  if self.eval_args_samples is not None else ()),
            )
            delta_tokens = jnp.zeros((1,), dtype=DELTA_TOKEN_DTYPE)
            delta_count = jnp.zeros((), dtype=jnp.int32)

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.zeros(NUM_REWARDS, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            sparsity_specs=initial_specs,
            face_specs=initial_face_specs,
            face_skips=initial_face_skips,
            face_joins=initial_face_joins,
            tokens=tokens,
            eqn_ids=eqn_ids,
            delta_tokens=delta_tokens,
            delta_count=delta_count,
            axis_state=self.axis_state_static,
            axis_valid_mask=self.axis_valid_static,
            step_count=step_count,
            max_steps=max_steps,
            reward=reward,
            terminated=terminated,
        )

        if num_envs is not None and num_envs > 0:
            state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)), state
            )

        return state

    @jit
    def step(self, state: EnvState, action) -> EnvOut:
        # Action may be either a `StepAction` (multi-rule) or a legacy scalar int
        # encoded as `sp_type * MAX_TOKENS + target_vertex`.
        if isinstance(action, StepAction):
            target_vertex = jnp.asarray(action.target_vertex, dtype=jnp.int32)
            rule_specs = jnp.asarray(action.rule_specs, dtype=jnp.int32)
            face_join = None
            if action.face_rows is not None:
                face_rows = jnp.asarray(action.face_rows, dtype=jnp.int32)
                face_skip = jnp.asarray(action.face_skip, dtype=jnp.int32)
                if action.face_join is not None:
                    face_join = jnp.asarray(action.face_join, dtype=jnp.int32)
            else:
                face_rows = jnp.full(
                    (MAX_FACES, wire_slots(), 3), -1, dtype=jnp.int32
                )
                face_skip = jnp.zeros((MAX_FACES,), dtype=jnp.int32)
        else:
            action = jnp.asarray(action, dtype=jnp.int32)
            sp_type = action // MAX_TOKENS
            target_vertex = action % MAX_TOKENS
            rule_specs = _legacy_sp_to_specs(sp_type)
            face_rows = jnp.full((MAX_FACES, wire_slots(), 3), -1, dtype=jnp.int32)
            face_skip = jnp.zeros((MAX_FACES,), dtype=jnp.int32)
            face_join = None
        # THE BIT AND THE CHANNEL MUST BOTH EXIST OR NEITHER. A bit with no
        # channel to store it in would be dropped between `sample` and the
        # measurement; a channel with no bit would be measured as all-lossy.
        # Both are the action/reward mismatch of finding 72, so both raise.
        if (face_join is None) != (state.face_joins is None):
            raise ValueError(
                f"--approx-add {approx_add()!r}: the action "
                f"{'carries' if face_join is not None else 'carries no'} "
                f"per-face join bit while the env state "
                f"{'has' if state.face_joins is None else 'has no'} nowhere to "
                f"keep it. Under 'choose' both exist; under every fixed value "
                f"neither does and `resolve_join_mode` answers from the "
                f"configuration.")

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_specs = state.sparsity_specs

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_specs = curr_specs[shifted.astype(jnp.int32)].at[idx].set(rule_specs)
        new_face_joins = (
            None if face_join is None else
            state.face_joins[shifted.astype(jnp.int32)].at[idx].set(face_join))
        new_face_specs = (
            state.face_specs[shifted.astype(jnp.int32)].at[idx].set(face_rows)
        )
        new_face_skips = (
            state.face_skips[shifted.astype(jnp.int32)].at[idx].set(face_skip)
        )

        # BOUND OPERANDS: only the in-process callback reads them.
        #
        # `args` / `consts` / `eval_args_samples` are episode constants that
        # the MEASURE ACTOR already owns a copy of -- the remote closure
        # drops args/consts outright, and the pool serves eval_samples from
        # its own ObjectRef. Passing them as callback operands anyway forces
        # JAX to marshal every one of them device->host on EVERY decision
        # (tens of MB and ~150 separate arrays for TransformerLM) so the
        # remote closure can throw them away. Hand the pooled path
        # zero-length placeholders instead; the in-process path (no pool) and
        # the trainer-local terminal rows (ALPHAGRAD_POOL_TERMINAL_LOCAL=1)
        # still get the real thing.
        _drop_bound = _pool_owns_bound_operands(self._remote_pool)
        _z = jnp.zeros((1,), jnp.int32)
        # THE NARROWED FACE WIRE (`--face-wire-faces`). The state keeps every
        # column of the provable bound; this callback is handed the first
        # `face_wire_faces()` of them. Measured on pgi15-gpu17, one episode of
        # the campaign arm under an XLA trace: this instruction alone copies
        # 11.1 gigabytes device to host per episode, 117 megabytes per step,
        # and the profile of 2026-09-15 measured the occupancy at a median of
        # one face per vertex against a cap of 1920. Nothing falls off the end
        # in silence -- `_face_dict_for_vertex` raises when the wire is
        # narrower than the enumeration, and the live-face count raises one
        # step earlier, before the decisions are even written. The default is
        # the full width, and then these slices are the identity.
        _fw = face_wire_faces()
        # AND ONLY THE ROW THIS STEP DECIDED (owner ruling 2026-09-15, item 2).
        # The narrowed history above is still the WHOLE PREFIX -- `(N, W,
        # FACE_SLOTS, 3)` per environment, the same rows on every step. The
        # host keeps it (`face_prefix_step`), so the device sends `face_rows` /
        # `face_skip` / `face_join`, which are this step's row and are already
        # in hand: `new_face_specs[idx]` IS `face_rows`. The host extends its
        # prefix to `new_step` rows and hands `_callback` the very arrays it
        # used to be handed. ALPHAGRAD_FACE_ROW_WIRE=0 restores the operand.
        # AND ONLY WHERE THE HOST CAN TELL THE ENVIRONMENTS APART.
        # `_env_callback` dispatches through `pure_callback` with
        # `vmap_method="expand_dims"` under ALPHAGRAD_BATCHED_CALLBACK, and the
        # host then sees the WHOLE BATCH in index order, which is what makes
        # the loop index the environment identity -- the same fact
        # `EdgeSlotTable` and the per-env key chain rest on. Without it the
        # dispatch is `io_callback`, which `vmap` runs once per environment
        # with no batch axis and nothing on the wire saying which environment
        # it is, so a host prefix would mix every environment's rows into slot
        # 0. Measured: `--num-envs 2` under the sequential dispatch raised the
        # step-index guard at step 2 (tests/popart_seed_init_test.py). The
        # face callbacks are not affected -- they dispatch with
        # `vmap_method="broadcast_all"` unconditionally.
        _row_wire = face_row_wire() and _BATCHED_CALLBACK
        _fn = self.tokenize(batched=True, bound_dropped=_drop_bound)
        _cbout = _env_callback(
            _face_prefix_host(_fn, int(new_order.shape[-1]))
            if _row_wire else _fn,
            self._callback_shape,
            _z if _drop_bound else self.args,
            _z if _drop_bound else self.consts,
            new_order,
            new_specs,
            (face_rows[:_fw] if _row_wire else new_face_specs[:, :_fw]),
            (face_skip[:_fw] if _row_wire else new_face_skips[:, :_fw]),
            (None if new_face_joins is None else
             (face_join[:_fw] if _row_wire else new_face_joins[:, :_fw])),
            new_step,
            *(() if _drop_bound
              else (self.eval_args_samples
                    if self.eval_args_samples is not None else ())),
            batched=True,
        )

        # `(tokens, reward)` under delta_obs, `(tokens, eqn_ids, reward)` on
        # the legacy full-stream path (`_callback._wire` / `_callback_shape`).
        reward = _cbout[-1]
        if self.config.delta_obs:
            # The first DELTA_HEADER_SLOTS byte slots of the wire are the
            # exact host-side count as one little-endian uint32 (see
            # `_delta_observation`); the tokens start after them.
            # `decode_delta_header` is the only reader of that layout.
            # uint32 on the wire, int32 for indexing: exact, the host bounds
            # the count by MAX_DELTA_TOKENS before it encodes.
            delta_count = decode_delta_header(_cbout[0]).astype(jnp.int32)
            # THE BIN'S SLICE OF THE WIRE. The wire is the cap wide; the
            # state buffer is the window bin wide. `delta_count` is the
            # EXACT host-side length and is NOT clipped here: a count past
            # the bin is a window overflow, which the rollout carries out as
            # a device flag and the driver turns into a repeat one bin up
            # (`common.episode_stream`). Clipping it here instead would
            # change the action in silence.
            delta_tokens = _cbout[0][
                DELTA_HEADER_SLOTS:DELTA_HEADER_SLOTS + self.delta_window]
            tokens = jnp.zeros((1,), dtype=jnp.int32)
            eqn_ids = jnp.zeros((1,), dtype=jnp.int32)
        else:
            tokens, eqn_ids = _cbout[0], _cbout[1]
            delta_tokens = jnp.zeros((1,), dtype=DELTA_TOKEN_DTYPE)
            delta_count = jnp.zeros((), dtype=jnp.int32)

        terminated = new_step >= state.max_steps

        # Single-vertex axis_state mutation: the just-acted-on vertex's
        # row records the DIAG pairings / COMPRESS markings the agent
        # committed to. See `_apply_rules_to_axis_state` for the per-rule
        # semantics and for why downstream propagation is deliberately
        # deferred. `target_vertex` is 1-indexed (matches the env's
        # vertex IDs); axis_state is 0-indexed by equation, so subtract 1.
        v_idx = target_vertex - jnp.int32(1)
        updated_axis_v = _apply_rules_to_axis_state(
            state.axis_state[v_idx], rule_specs,
        )
        new_axis_state = state.axis_state.at[v_idx].set(updated_axis_v)

        new_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            face_specs=new_face_specs,
            face_skips=new_face_skips,
            face_joins=new_face_joins,
            tokens=tokens,
            eqn_ids=eqn_ids,
            delta_tokens=delta_tokens,
            delta_count=delta_count,
            axis_state=new_axis_state,
            axis_valid_mask=state.axis_valid_mask,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(
                state,
                jnp.zeros(NUM_REWARDS, jnp.float32),
                jnp.array(True, dtype=jnp.bool_),
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)

    # ---------------------------------------------------------------------
    # External-tokenizer entry points
    # ---------------------------------------------------------------------
    # The pair below splits `step()` (and `reset()`) into the JIT-only
    # half (`*_external_jax_part`) and a host-side stitcher
    # (`assemble_*_result`). The point is to take `io_callback` out of
    # the hot path: today every `step()` does a host roundtrip + GIL-
    # bound Python pass + unbounded `cost_analysis()` C++ allocation
    # (see `_callback` and the comment at env.py:1182). With the split,
    # the driver runs the JIT-side over a batch of envs once, ships the
    # `(order, specs, step)` triples to a pool of CPU workers (see
    # `cpu_approx_worker.CpuApproximationServer`), and stitches the
    # tokenizer outputs back into the state in numpy.
    #
    # The existing `reset()` / `step()` are unchanged — single-process
    # callers (legacy ppo, mu0, tests) keep working as-is. The new
    # methods are opt-in and have **no behavioural difference** from the
    # io_callback path for the same inputs: the only thing that moves is
    # where the tokenizer work runs.

    def step_external_jax_part(self, state: EnvState, action):
        """JIT-only half of `step()` — everything except the tokenizer.

        Returns the partial new state with placeholders for the three
        tokenizer-dependent fields (`tokens`, `eqn_ids`, `reward`), plus
        the `(order, specs, step)` triple the caller hands to the CPU
        worker. The caller then calls `assemble_step_result(...)` with
        the worker's output to recover the full :class:`EnvOut`.

        The action-handling and order-update logic is byte-for-byte
        identical to `step()` — this method is the JIT-friendly subset
        of the same function. Keep them in sync if either changes.

        Notes:
            * Not `@jit`-decorated so callers can compose with their
              policy's act() into a single JIT step.
            * The terminated-state guard (the `state.terminated` branch
              from `step()`) lives on the assemble side; if the env was
              already terminated, `assemble_step_result` returns
              the input state unchanged and the tokenizer output is
              discarded.
        """
        if self.config.delta_obs:
            raise NotImplementedError(
                "the external (Ray-split) tokenizer path predates the DELTA "
                "observation and still stitches a full stream back into "
                "`tokens`/`eqn_ids` -- it would leave `delta_*` empty and "
                "the encoder would never advance. Build the env with "
                "delta_obs=False for this path (ppo_ray_worker does), or "
                "port assemble_step_result to split the delta header."
            )
        if isinstance(action, StepAction):
            target_vertex = jnp.asarray(action.target_vertex, dtype=jnp.int32)
            rule_specs = jnp.asarray(action.rule_specs, dtype=jnp.int32)
        else:
            action = jnp.asarray(action, dtype=jnp.int32)
            sp_type = action // MAX_TOKENS
            target_vertex = action % MAX_TOKENS
            rule_specs = _legacy_sp_to_specs(sp_type)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_specs = state.sparsity_specs

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_specs = curr_specs[shifted.astype(jnp.int32)].at[idx].set(rule_specs)

        terminated = new_step >= state.max_steps

        v_idx = target_vertex - jnp.int32(1)
        updated_axis_v = _apply_rules_to_axis_state(
            state.axis_state[v_idx], rule_specs,
        )
        new_axis_state = state.axis_state.at[v_idx].set(updated_axis_v)

        partial_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            # External (Ray) split predates face actions: carry the histories
            # through the same reorder so the state stays well-formed.
            face_specs=state.face_specs[shifted.astype(jnp.int32)],
            face_skips=state.face_skips[shifted.astype(jnp.int32)],
            face_joins=(None if state.face_joins is None else
                        state.face_joins[shifted.astype(jnp.int32)]),
            tokens=jnp.zeros_like(state.tokens),
            eqn_ids=jnp.zeros_like(state.eqn_ids),
            # Legacy sentinels, byte-identical to what `step()` writes on
            # the non-delta path (pad is -1 for eqn ids, not 0) -- this
            # state is compared against `step()`'s field by field.
            delta_tokens=jnp.zeros((1,), dtype=DELTA_TOKEN_DTYPE),
            delta_count=jnp.zeros((), dtype=jnp.int32),
            axis_state=new_axis_state,
            axis_valid_mask=state.axis_valid_mask,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=jnp.zeros(NUM_REWARDS, dtype=jnp.float32),
            terminated=terminated,
        )
        return partial_state, new_order, new_specs, new_step

    def assemble_step_result(
        self,
        state_before: EnvState,
        partial_state: EnvState,
        tokens,
        eqn_ids,
        reward,
    ) -> EnvOut:
        """Stitch tokenizer outputs into the partial state.

        Mirrors the `state.terminated` branch from `step()`: if the env
        was already terminated, returns the input `state_before`
        unchanged (with zero reward, terminated=True). Otherwise the
        new EnvState gets the tokenizer's `(tokens, eqn_ids, reward)`
        plus everything from `partial_state`.

        Inputs may be numpy or jnp arrays — they're coerced to the env's
        canonical dtypes before being stored.
        """
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        eqn_ids = jnp.asarray(eqn_ids, dtype=jnp.int32)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        new_state = partial_state._replace(
            tokens=tokens,
            eqn_ids=eqn_ids,
            reward=reward,
        )

        def _process(_):
            return EnvOut(new_state, reward, partial_state.terminated)

        def _done(_):
            return EnvOut(
                state_before,
                jnp.zeros(NUM_REWARDS, jnp.float32),
                jnp.array(True, dtype=jnp.bool_),
            )

        return jax.lax.cond(state_before.terminated, _done, _process, None)

    def reset_external_jax_part(self):
        """JIT-only half of `reset()`. Returns ``(partial_state, order,
        specs, step=0)``.

        Unlike `reset()`, this **does not** broadcast across
        ``num_envs`` — the caller is responsible for vmapping (or
        building a batch by hand in a Python loop). Broadcasting is
        skipped because the typical caller wants to ship per-env
        `(order, specs)` tensors to the worker pool, which is easier
        when each shard owns its own (un-broadcast) state.
        """
        if self.config.delta_obs:
            raise NotImplementedError(
                "the external (Ray-split) tokenizer path predates the DELTA "
                "observation and still stitches a full stream back into "
                "`tokens`/`eqn_ids` -- it would leave `delta_*` empty and "
                "the encoder would never advance. Build the env with "
                "delta_obs=False for this path (ppo_ray_worker does), or "
                "port assemble_step_result to split the delta header."
            )
        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        initial_specs = jnp.full(
            (initial_order.shape[0], MAX_RULES_PER_VERTEX, 3),
            -1,
            dtype=jnp.int32,
        )
        initial_specs = initial_specs.at[..., 2].set(0)

        partial_state = EnvState(
            order=initial_order,
            sparsity_specs=initial_specs,
            face_specs=jnp.full(
                (initial_order.shape[0], MAX_FACES, wire_slots(), 3), -1,
                dtype=jnp.int32,
            ),
            face_skips=jnp.zeros(
                (initial_order.shape[0], MAX_FACES), dtype=jnp.int32
            ),
            tokens=jnp.zeros((self.obs_width,), dtype=jnp.int32),
            eqn_ids=jnp.zeros((self.obs_width,), dtype=jnp.int32),
            delta_tokens=jnp.zeros((1,), dtype=DELTA_TOKEN_DTYPE),
            delta_count=jnp.zeros((), dtype=jnp.int32),
            axis_state=self.axis_state_static,
            axis_valid_mask=self.axis_valid_static,
            step_count=jnp.array(0, dtype=jnp.int32),
            max_steps=initial_order.shape[0],
            reward=jnp.zeros(NUM_REWARDS, dtype=jnp.float32),
            terminated=jnp.array(False, dtype=jnp.bool_),
        )
        return partial_state, initial_order, initial_specs, jnp.int32(0)

    def assemble_reset_result(self, partial_state: EnvState, tokens, eqn_ids) -> EnvState:
        """Fold the reset-time tokenizer output into the partial state."""
        return partial_state._replace(
            tokens=jnp.asarray(tokens, dtype=jnp.int32),
            eqn_ids=jnp.asarray(eqn_ids, dtype=jnp.int32),
        )


# ---------------------------------------------------------------------------
# THE EPISODE TELEMETRY SNAPSHOT (review finding 5, 2026-09-14).
#
# An episode whose token stream overflows its bin is DISCARDED and repeated
# one bin up (`common.episode_stream.run_episode`). The device side of the
# discarded attempt vanishes on its own -- the driver rebinds the agent, the
# optimiser state and the env states only on RETURN. The HOST side does not:
# every step of the failed attempt already ran its env callback, so the plan
# log, the truncation counters and every other per-episode accumulator in
# this module saw it. An overflow at a late step therefore DOUBLE-COUNTED
# that episode's terminal plans, which is the very artifact the
# trustworthiness claims are built on.
#
# These two functions bound it. The driver takes a snapshot before an
# attempt and restores it when the attempt is thrown away, so a discarded
# attempt leaves exactly nothing behind in this module. They cover the
# containers that a per-episode `consume_*` drain empties, which is the same
# set the trainer logs to wandb once per episode.
#
# NOT RESTORED, on purpose: `_TOKENIZATION_TRUNCATION_WARNED` and
# `_UNTRACEABLE_SEEN` are "have we already printed this warning" latches, not
# measurements. Rolling them back would print the same warning twice.
# ---------------------------------------------------------------------------
_EPISODE_TELEMETRY_NAMES = (
    # tokenization and delta lengths
    "_TOKENIZATION_TRUNCATION_COUNT",
    "_TOKENIZATION_TRUNCATION_MAX_LEN",
    "_TOKENIZATION_TRUNCATION_OVERFLOW_SUM",
    "_TOKLEN_SUM", "_TOKLEN_MAX", "_TOKLEN_COUNT",
    "_DELTALEN_SUM", "_DELTALEN_MAX", "_DELTALEN_COUNT",
    # host phase profiling
    "_PROF", "_PROF_SAMPLES", "_PROF_DIST", "_PROF_TRACE",
    # plan health counters
    "_DEGENERATE_PLANS", "_TRUNCATED_PLANS", "_ZERO_WORK_PLANS",
    "_UNTRACEABLE_PLANS", "_REFUSED_KINDS",
    # reward-channel telemetry
    "_FIDELITY_STATS", "_COS_LOG_SEEN",
    "_SPARSITY_STATS", "_APPROX_STORE_BYTES", "_EXACT_STORE_BYTES",
    "_PER_FACE_STATS", "_FACE_CAP_STATS",
    "_XLA_MEM_APPROX", "_XLA_MEM_EXACT",
    # the A6 plan log and the two drains that ride with it
    "_PLAN_RECORDS", "_PLAN_LOG_DROPPED", "_PLAN_LOG_TERMINALS",
    "_PAIRED_REF", "_PAIRED_REF_DROPPED",
    "_MEM_PARITY", "_MEM_PARITY_MEASURED", "_MEM_PARITY_DROPPED",
)

# Fail at IMPORT, not at the first discarded episode: a renamed container
# that silently drops out of the snapshot is exactly the kind of quiet gap
# this exists to close.
for _name in _EPISODE_TELEMETRY_NAMES:
    if _name not in globals():
        raise RuntimeError(
            f"_EPISODE_TELEMETRY_NAMES lists {_name}, which env.py does not "
            f"define. Rename it there too, or drop it from the list.")
    if not isinstance(globals()[_name], (list, dict)):
        raise RuntimeError(
            f"_EPISODE_TELEMETRY_NAMES lists {_name}, which is a "
            f"{type(globals()[_name]).__name__}. The snapshot restores IN "
            f"PLACE, so every entry has to be a list or a dict.")
del _name


def _telemetry_copy(value):
    """Copy the list/dict SHELLS and share everything else.

    Not `copy.deepcopy`: a plan record's values are numpy and jax arrays,
    and deep-copying those is both expensive and, for a device array, a
    different object than the one the drain expects. Nothing rebinds a
    value inside a record once it is appended, so sharing the leaves is
    exact; what has to be copied is every container the callbacks APPEND to
    or COUNT in, which is precisely the shells.
    """
    if isinstance(value, list):
        return [_telemetry_copy(v) for v in value]
    if isinstance(value, dict):
        return {k: _telemetry_copy(v) for k, v in value.items()}
    return value


def episode_telemetry_snapshot() -> dict:
    """Copy every per-episode host accumulator. See the block comment."""
    g = globals()
    return {n: _telemetry_copy(g[n]) for n in _EPISODE_TELEMETRY_NAMES}


def episode_telemetry_restore(snapshot: dict) -> None:
    """Put the accumulators back as :func:`episode_telemetry_snapshot` found
    them. IN PLACE, because `_callback` closes over the containers."""
    g = globals()
    unknown = set(snapshot) - set(_EPISODE_TELEMETRY_NAMES)
    if unknown:
        raise ValueError(
            f"episode_telemetry_restore: {sorted(unknown)} is not a "
            f"per-episode accumulator of this module")
    missing = set(_EPISODE_TELEMETRY_NAMES) - set(snapshot)
    if missing:
        raise ValueError(
            f"episode_telemetry_restore: {sorted(missing)} is missing from "
            f"the snapshot; pass the whole dict back, not a slice of it")
    for name, value in snapshot.items():
        container = g[name]
        if isinstance(container, list):
            container[:] = value
        else:
            container.clear()
            container.update(value)


# THE FRESH-PROCESS STATE, captured at IMPORT, before any callback has run.
# This is what "zero" means for these containers, and it is captured rather
# than fabricated because the two kinds here do not have the same zero.
#
# NINE OF THEM ARE ONE-ELEMENT COUNTERS. `_PLAN_LOG_TERMINALS`,
# `_PLAN_LOG_DROPPED`, `_TOKLEN_SUM` and the rest are `[0]`, and every reader
# indexes element 0. Emptying such a list is not zero, it is a MISSING
# ELEMENT, and the next read raises IndexError. Canary job 65715 died exactly
# there: a driver that reset the accumulators by emptying every list ran two
# window-bin repeats of episode 0, emptied `_PLAN_LOG_TERMINALS` on the
# successful attempt's collect, and `consume_plan_records` raised on the next
# drain. The others are COLLECTIONS and their zero is empty. Restoring a copy
# of the import-time state gets both right and cannot drift, because it is
# literally what a fresh process has.
_EPISODE_TELEMETRY_FRESH = {
    _n: _telemetry_copy(globals()[_n]) for _n in _EPISODE_TELEMETRY_NAMES}


def episode_telemetry_reset() -> None:
    """Put every per-episode accumulator back to its FRESH-PROCESS state.

    For a driver that PARKS an episode's counters and then wants the next
    episode to count from zero. A COPY of the fresh state is restored, so a
    later append cannot reach into the template and change what "fresh" means.
    """
    episode_telemetry_restore(
        {_n: _telemetry_copy(_v)
         for _n, _v in _EPISODE_TELEMETRY_FRESH.items()})
    # THE HOST FACE PREFIX IS PER-EPISODE HOST STATE TOO (owner ruling
    # 2026-09-15, item 2). It is not in `_EPISODE_TELEMETRY_NAMES` because it
    # is not an accumulator a reader drains: it is a buffer the callbacks WRITE
    # IN PLACE, so `_telemetry_copy`'s share-the-leaves rule -- which is right
    # for the plan records -- would hand the snapshot the very arrays the next
    # attempt overwrites. Its zero is "no rows", and this is where a fresh
    # episode says so. ppo.py also calls it at the top of every attempt, which
    # is the case that matters: a discarded attempt's rows must be gone before
    # the repeat writes its own.
    face_prefix_reset()


def episode_telemetry_fixed_counters() -> tuple:
    """The accumulators that are FIXED-SIZE counters, not collections.

    A list here is a number every reader indexes. Emptying one is the bug
    above. Derived from the import-time state rather than listed by hand, so
    a new counter cannot be forgotten.
    """
    return tuple(
        _n for _n, _v in _EPISODE_TELEMETRY_FRESH.items()
        if isinstance(_v, list) and len(_v) > 0)
