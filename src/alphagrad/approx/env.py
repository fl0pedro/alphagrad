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

from alphagrad.approx.common.relations import compute_eqn_ids_from_tokens
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from graphax.jaxpr import get_vocab as _graphax_get_vocab
from graphax.sparse.micro_actions import (
    COMPRESS_KINDS, QUANT_DTYPES, Compress, Diag, Quant,
)
from jax_memory_monitor import ResourceMonitor as _RealResourceMonitor


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
    if os.environ.get("ALPHAGRAD_DISABLE_RESOURCE_MONITOR", "0") == "1"
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
# shapes actually require. SIZED FROM THE MEASURED DISTRIBUTION:
# ``decode3_data.py`` over 192 TLM trajectories measured the largest SINGLE
# delta at 25,737 tokens (mean whole-stream length 43,678, max 122,910), so
# 32768 is the next power of two with headroom. The previous 1024 dropped
# ~96% of the flagship's worst delta EVERY step, and the drop was silent
# (issue #81: the ``tokenization/*`` counters are process-blind -- they read 0
# from the driver while the callback process clips).
MAX_DELTA_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_DELTA_TOKENS", "32768"))

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


def _delta_observation(stream, seg_ids, last_start):
    """Wire form of ONE step's token delta: ``(1 + MAX_DELTA_TOKENS,)`` pair.

    Slot 0 is a HEADER carrying the exact host-side token count; slots 1..
    are the delta itself, pad-filled (0 for tokens, -1 for eqn ids). The
    header rides in-band because the callback's output arity is shared with
    the Ray measurement pool and the batched host shim -- one extra slot
    changes no signature anywhere, and ``env.step`` splits it straight back
    out into ``EnvState.delta_count`` / ``delta_tokens`` / ``delta_eqns``.

    The count is the TOKENIZER'S OWN length. Nothing scans a padded buffer
    for it, so token id 0 -- the literal '-' graphax emits for a negative
    value, which occurs INTERIOR to real streams -- cannot make it short.
    """
    blk = stream[last_start:]
    ids = seg_ids[last_start:]
    n_raw = len(blk)
    _record_delta_length(n_raw)
    # Raises unless ALPHAGRAD_DELTA_OVERFLOW=clip; the clamp below is what
    # that opt-out buys, and it is announced on stderr every time.
    _record_delta_truncation(n_raw)
    n = min(n_raw, MAX_DELTA_TOKENS)
    t = np.zeros((1 + MAX_DELTA_TOKENS,), dtype=np.int32)
    e = np.full((1 + MAX_DELTA_TOKENS,), -1, dtype=np.int32)
    t[0] = n
    e[0] = n
    if n:
        t[1:1 + n] = np.asarray(blk[:n], dtype=np.int32)
        e[1:1 + n] = np.asarray(ids[:n], dtype=np.int32)
    return jnp.asarray(t), jnp.asarray(e)


_INCR_TOK_CACHE: dict = {}
_INCR_TOK_CACHE_CAP = 512


def incremental_token_delta(jaxpr, argnums, consts, args, order_prefix,
                            vocab_size: int = 512):
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


def _face_wire_keys(faces_np, skips_np, n):
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
        for k in range(n)
    )


def _incremental_stream_tokens(config, consts, args, o_list, specs_list,
                               tok_rules_by_v, ft_by_vertex=None,
                               face_key=None,
                               face_rows_list=None, face_skips_list=None):
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

    # `vocab_size` is the TOTAL id space: 230 reserved structural tokens + 10
    # digits, leaving `vocab - 240` symbols for the NAME alphabet (the
    # tokenizer needs >= 2); a name past the alphabet spells itself out by
    # concatenation. 512 == the launchers' --vocab-size == the policy
    # embedding's row count, so the tokenizer and the embedding table name
    # the SAME id space (272 name symbols, max_token_id 511 < 512).
    #
    # It was 248, sized for a 256-row embedding constraint that no longer
    # applies, which left an 8-symbol alphabet (9 before graphax added the
    # `^` slot separator) and paid for it in concatenated names: measured on
    # nn256 (ALPHAGRAD_NN_HIDDEN=256, reverse order), base 425 -> 331 tokens,
    # full stream 13114 -> 10453 (1.25x), max single-step delta 2385 -> 2063.
    # CONSISTENCY, not speed: `base_observation` below must resolve the same
    # default or the base and the deltas are tokenized at different vocabs
    # and do not concatenate.
    vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))
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
                face_skips_list[_ki])
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
_SKIP_COUNT_OPS = os.environ.get("ALPHAGRAD_SKIP_COUNT_OPS", "0") == "1"

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
#                          "loss_drop" (default under --measure-grad) — the
#                            relative loss drop of a 200-step Adam walk driven
#                            by THIS PLAN's gradient, probed on a fixed batch
#                            of 512 real MNIST images. Pearson 0.922 against
#                            final downstream test accuracy vs 0.610 for the
#                            Jacobian cosine, at 0.22 s / 40 MB per plan vs
#                            9.70 s / 4.24 GB.
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
#   7 frob_residual    — relative Frobenius residual ||J_e - J_a||_F / ||J_e||_F.
NUM_REWARDS = 8
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "quality",
    "frob_residual",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
# BACK-COMPAT ALIAS. 269 call sites (and the persisted PopArt/calibration
# state, which is keyed by INDEX) address slot 6 as "cosine_sim". The slot did
# not move; only its display name changed, so the alias is exact.
REWARD_INDEX["cosine_sim"] = REWARD_INDEX["quality"]
COMPUTE_REWARD_INDICES = tuple(range(0, 6))  # cost components
QUALITY_REWARD_INDICES = (6, 7)             # cosine, frobenius

# Sentinel reward returned when a per-vertex transform sequence matches an
# entry in the in-file blacklist (used during exploration to penalise
# pathological configurations). The blacklist is no longer wired up after
# the typed-transform migration; the array is kept for potential reuse.
# Worst-possible reward: every cost channel at the sentinel and the quality
# channels at their floor. Derived from REWARD_INDEX so adding a channel can't
# leave a stale hand-written row behind.
SENTINEL_COST = -1e10
_SENTINEL_BAD_REWARD = jnp.array(
    [
        0.0 if i == REWARD_INDEX["cosine_sim"]
        else (-1.0 if i == REWARD_INDEX["frob_residual"] else SENTINEL_COST)
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
    face_specs: Array  # (N, MAX_FACES, FACE_SLOTS, 3) int32
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
    delta_tokens: Array   # (MAX_DELTA_TOKENS,) int32
    delta_eqns: Array     # (MAX_DELTA_TOKENS,) int32
    delta_count: Array    # () int32
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


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class StepAction(NamedTuple):
    target_vertex: Array  # scalar int32
    rule_specs: Array  # (MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 marks unused
    # P1 per-path actions. None ⇒ per-vertex mode, byte-identical to before.
    face_rows: Array | None = None  # (MAX_FACES, FACE_SLOTS, 3) int32 spec rows
    face_skip: Array | None = None  # (MAX_FACES,) int32; 1 ⇒ SKIP_FACE


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
    # Spec's accumulation loop: each timed rep executes the compiled fn this
    # many times inside ONE ResourceMonitor window and divides the elapsed
    # time, amortizing dispatch/timer overhead (spec default 50; kept at 1
    # here so existing campaigns measure identically until a launcher opts in
    # via --latency-inner-reps).
    latency_inner_reps: int = 1
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
    # --measure-grad: the traced target IS a scalar loss and the plan's output
    # IS a gradient. Recorded on the CONFIG (not just validated in from_jaxpr)
    # because ``_callback`` sees only the config, and ``quality_metric()``'s
    # ``auto`` default needs it to decide between the loss-drop walk (defined
    # only for a scalar loss) and the legacy Jacobian cosine.
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


def _align_jac(jac_approx, jac_exact):
    """Align each approx-Jacobian/grad leaf to its exact leaf's LAYOUT before
    comparison. graphax ``jacve`` returns some weight grads in the transposed
    (dL/dW^T) layout for certain elimination orders; flattening them as-is makes
    a (256,784) vs (784,256) ravel near-orthogonal, so cosine/frob become a
    layout ARTIFACT that badly underestimates true gradient quality. Transpose a
    2-D leaf back when its shape is the exact leaf's reverse; leave other
    mismatches for the size/shape guard downstream. (Ported from the fat-line
    stash — the fix that lifted Spearman-vs-trainability 0.67 -> 0.79.)"""
    def _al(a, e):
        if getattr(a, "shape", None) == getattr(e, "shape", None):
            return a
        if getattr(a, "ndim", 0) == 2 and a.shape == e.shape[::-1]:
            return a.T
        return a
    try:
        return jax.tree_util.tree_map(_al, jac_approx, jac_exact)
    except Exception:
        return jac_approx


_COST_REF: dict = {}
_QUALITY_GATE_STATS = {"clamps": 0}


def _measure_exec_cost(ex, base_args):
    """(latency_ns, peak_bytes) of a compiled executable, measured with
    the gate's own light protocol (3x20 perf_counter, allocator-delta
    peak with static fallback). Shared by the global rev reference and
    the per-order floor so the clamp compares like with like."""
    out = ex(*base_args)
    jax.block_until_ready(out)
    _devs = set()
    for _l in jax.tree_util.tree_leaves(out):
        try:
            _devs |= set(_l.devices())
        except Exception:
            pass
    _have = bool(_devs) and all(
        (d.memory_stats() or {}).get("peak_bytes_in_use") is not None
        for d in _devs)
    for _ in range(3):
        jax.block_until_ready(ex(*base_args))
    _base = 0.0
    if _have:
        _base = sum(float((d.memory_stats() or {}).get(
            "bytes_in_use", 0.0)) for d in _devs)
        for d in _devs:
            try:
                d.client.clear_memory_stats()
            except Exception:
                pass
    _laps = []
    for _ in range(3):
        _t0 = time.perf_counter()
        for _ in range(20):
            out = ex(*base_args)
        jax.block_until_ready(out)
        _laps.append((time.perf_counter() - _t0) / 20)
    _lat = float(np.median(_laps) * 1e9)
    if _have:
        _peak = max(0.0, sum(float((d.memory_stats() or {}).get(
            "peak_bytes_in_use", 0.0)) for d in _devs) - _base)
    else:
        _peak = float(_memory_analysis_bytes(ex) or 0.0)
    return _lat, _peak


def _exact_cost_reference(config, base_args):
    """(latency_ns, peak_bytes) of the EXACT 'rev' plan -- the additive
    quality gate's floor. Measured ONCE per process through the same
    compile path (_compile_measure) so the clamp compares like with
    like. Returns None (gate fails OPEN, with one warning) if the
    reference cannot be built."""
    if "ref" in _COST_REF:
        return _COST_REF["ref"]
    try:
        from graphax import jacve as _jacve
        ex = _compile_measure(
            jax.jit(
                _jacve(config.target_fun, "rev",
                       argnums=config.argnums, has_aux=config.has_aux,
                       sparse_representation=config.sparse),
                keep_unused=True,
            ).lower(*base_args))
        _lat, _peak = _measure_exec_cost(ex, base_args)
        _COST_REF["ref"] = (_lat, _peak)
        print(f"[measure] quality gate armed: exact-rev reference "
              f"latency={_lat/1e3:.1f}us peak={_peak/1e6:.1f}MB "
              f"(qmin={os.environ.get('ALPHAGRAD_QUALITY_GATE_MIN')})",
              flush=True)
    except Exception as _exc:
        print(f"[measure] WARNING quality gate: exact-rev reference "
              f"failed ({type(_exc).__name__}: {str(_exc)[:120]}) -- "
              f"gate fails OPEN", flush=True)
        _COST_REF["ref"] = None
    return _COST_REF["ref"]


def _apply_quality_gate(latency_ns, peak_memory, quality, is_terminal,
                        has_quality, config, base_args,
                        order_floor_fn=None):
    """ADDITIVE quality gate: below ALPHAGRAD_QUALITY_GATE_MIN the cost
    channels are FLOORED at the exact-reverse reference -- destruction
    pays what exact computation pays, so it gains nothing, while every
    channel stays a plain additive term. Cost channels are PENALTIES:
    scaling them toward zero would reward destruction, hence the clamp
    form. No-op unless the env var is set, quality was actually
    measured this step, and it fell below the threshold."""
    try:
        _qmin = float(os.environ.get(
            "ALPHAGRAD_QUALITY_GATE_MIN", "0") or 0.0)
    except ValueError:
        _qmin = 0.0
    if (_qmin <= 0.0 or not is_terminal or not has_quality
            or float(quality) >= _qmin):
        return latency_ns, peak_memory
    # Prefer the SAME-ORDER exact floor: a destroyed plan pays what its
    # OWN order would cost done exactly, so destruction is strictly
    # dominated at every fixed order while order search stays rewarded
    # (the global rev reference under-floored: random orders measure
    # ~500x above rev, so a clamped-to-rev SKIP still "won" latency).
    _ref = None
    _oom_floor_bytes = 0.0
    if order_floor_fn is not None:
        try:
            _ref = order_floor_fn()
        except Exception as _exc:
            _m = str(_exc)
            # A genuine allocation OOM on the per-order exact floor means
            # this ORDER's honest cost is AT LEAST the failed allocation.
            # Falling back to the tiny rev floor would mint a second-order
            # cliff (destroy + pick an order whose exact plan cannot even
            # compile -> cheapest floor in the pool); carry the requested
            # bytes into the MEMORY floor instead.
            import re as _re
            _g = _re.search(
                r"allocate ([0-9.]+)\s*([KMGT])iB", _m)
            if _g:
                _oom_floor_bytes = float(_g.group(1)) * {
                    "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40,
                }[_g.group(2)]
            print(f"[measure] quality gate: per-order floor failed "
                  f"({type(_exc).__name__}: {_m[:120]}) -- "
                  f"falling back to the rev reference"
                  + (f" + OOM mem floor "
                     f"{_oom_floor_bytes/2**30:.1f}GiB"
                     if _oom_floor_bytes else ""), flush=True)
    if _ref is None:
        _ref = _exact_cost_reference(config, base_args)
        if _ref is not None and _oom_floor_bytes > 0.0:
            _ref = (_ref[0], max(_ref[1], _oom_floor_bytes))
    if _ref is None:
        return latency_ns, peak_memory
    _rl, _rm = _ref
    _QUALITY_GATE_STATS["clamps"] += 1
    if _QUALITY_GATE_STATS["clamps"] == 1 \
            or _QUALITY_GATE_STATS["clamps"] % 50 == 0:
        print(f"[measure] quality gate CLAMP "
              f"#{_QUALITY_GATE_STATS['clamps']}: q={float(quality):.4f}"
              f" < {_qmin}; lat {latency_ns/1e3:.1f}->"
              f"{max(latency_ns, _rl)/1e3:.1f}us", flush=True)
    return max(latency_ns, _rl), max(peak_memory, _rm)


def _quality_metrics(jac_exact, jac_approx):
    """`(cosine_sim, relative_frobenius)` of `jac_approx` against `jac_exact`.

    Returns the WORST score `(0.0, 1.0)` when either side has no leaves,
    mismatched shapes, or zero size. The old fallback returned (1.0, 0.0) —
    "perfect" — which as a REWARD is a degeneracy backdoor: a plan that
    destroys the Jacobian's shape (e.g. a terminal COMPRESS) out-scored every
    honest approximation on all quality channels. A broken comparison is
    evidence of a broken plan, so it must score as such.

    ``ALPHAGRAD_DEBUG_QUALITY=1`` enables a one-line diagnostic print
    when cosine_sim collapses to ~0 with non-zero norms — used to
    investigate the persistent ``reward_mean/cosine_sim=0`` we see
    on the PPO dynamic-substeps path. The print fires only when the
    formula would have produced a meaningful value but didn't.
    """
    # Layout-align the approx leaves to the exact layout (transpose-back) so
    # cosine/frob compare the SAME entries, not a transposed-layout artifact.
    jac_approx = _align_jac(jac_approx, jac_exact)
    # PER-LEAF accumulation (2026-08-04). The old flatten+concatenate built a
    # >=4GB flat copy of EACH side, and XLA's concatenate kernel faults with
    # CUDA_ERROR_ILLEGAL_ADDRESS on >=2^31-byte operands once allocations sit
    # high enough in the address space (residency-dependent: the identical
    # concatenate passes in an empty process; reproduced with PLAIN EXACT
    # leaves and fresh elementwise copies at nn256 batch 512 — the v45b
    # measure-actor crash). Accumulating <e,a>, ||e||², ||a||², ||e-a||² per
    # leaf is mathematically identical (same eps semantics as ``cossim``:
    # each side clamped at sqrt(1e-7)) and never materializes the flats.
    leaves_e = jax.tree_util.tree_leaves(jac_exact)
    leaves_a = jax.tree_util.tree_leaves(jac_approx)
    if not leaves_e or not leaves_a or len(leaves_e) != len(leaves_a):
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    if any(getattr(a, "shape", None) != getattr(e, "shape", None)
           for a, e in zip(leaves_a, leaves_e)):
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    _total = sum(int(getattr(e, "size", 0)) for e in leaves_e)
    if _total == 0:
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)

    # Measure GPUs rotate but the exact reference is cached, so the two can
    # land on different devices -> jitted ops raise "Received incompatible
    # devices". Co-locate onto the reference's device (read-only, one transfer
    # only when they differ).
    try:
        _ed = next(iter(leaves_e[0].devices()))
        leaves_a = [
            jax.device_put(a, _ed)
            if next(iter(a.devices())) is not _ed else a
            for a in leaves_a
        ]
    except Exception:
        pass

    dot = ee = aa = rr = None
    for e, a in zip(leaves_e, leaves_a):
        # Promote per pair to at least f32 (bf16-Quant'd leaves square
        # horribly in bf16); complex leaves promote to their complex type —
        # the plain product (no conjugate) matches the old ``cossim``.
        _cdt = jnp.promote_types(jnp.promote_types(e.dtype, a.dtype),
                                 jnp.float32)
        ef = jnp.ravel(e).astype(_cdt)
        af = jnp.ravel(a).astype(_cdt)
        _d = jnp.sum(ef * af)
        _e2 = jnp.sum(jnp.abs(ef) ** 2)
        _a2 = jnp.sum(jnp.abs(af) ** 2)
        _r2 = jnp.sum(jnp.abs(ef - af) ** 2)
        dot = _d if dot is None else dot + _d
        ee = _e2 if ee is None else ee + _e2
        aa = _a2 if aa is None else aa + _a2
        rr = _r2 if rr is None else rr + _r2
    exact_norm = jnp.sqrt(ee)
    approx_norm = jnp.sqrt(aa)
    cos = dot / (jnp.maximum(exact_norm, jnp.sqrt(1e-7))
                 * jnp.maximum(approx_norm, jnp.sqrt(1e-7)))
    # Certain quant/compress combos yield complex-valued Jacobian leaves,
    # making the accumulated dot complex. Use the real part — matches the
    # reward path's existing real cast.
    cos = jnp.real(cos)
    rel_frob = jnp.sqrt(rr) / jnp.maximum(exact_norm, jnp.sqrt(1e-7))

    if os.environ.get("ALPHAGRAD_DEBUG_QUALITY", "0") == "1":
        approx_norm = float(approx_norm)
        e_norm = float(exact_norm)
        # Now that ``_callback`` only invokes ``_quality_metrics`` on
        # the terminal step (partial-order zero-Jacobian case is
        # short-circuited upstream), every line here represents a
        # real terminal evaluation. ``flush=True`` because Ray actor
        # stdout is line-buffered.
        print(
            f"[quality-debug] cos={float(cos):+.4f} frob={float(rel_frob):+.4f} "
            f"||exact||={e_norm:.3g} ||approx||={approx_norm:.3g} "
            f"size={_total}",
            flush=True,
        )
    return cos, rel_frob


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
# So slot 6 now holds the LOSS DROP OF A SHORT ADAM WALK DRIVEN BY THE PLAN'S
# OWN GRADIENT:
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


def quality_metric(config=None) -> str:
    """``"loss_drop"`` or ``"cosine"`` — WHICH quantity reward slot 6 holds.

    ``ALPHAGRAD_QUALITY_METRIC`` selects it; the default ``auto`` resolves to
    ``loss_drop`` whenever the measured graph IS a scalar loss (i.e. under
    ``--measure-grad``, where the plan's output is a gradient and the walk is
    defined) and to the legacy ``cosine`` otherwise. Read identically by the
    trainer and by every CpuApproximationActor — both run THIS function inside
    THIS module's ``_callback``, so the two paths cannot disagree.
    """
    want = os.environ.get(_QUALITY_METRIC_ENV, "auto").strip().lower()
    if want in ("loss_drop", "lossdrop", "walk"):
        return "loss_drop"
    if want in ("cosine", "cos", "cosine_sim"):
        return "cosine"
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
    # ``cosines`` stays empty, so ``_apply_quality_gate`` is handed
    # has_quality=False and returns its inputs unchanged -- latency_ns and
    # peak_memory are bit-identical to a run that computed the walk and threw
    # the number away.
    if want in ("none", "off", "skip"):
        return "none"
    if want not in ("auto", ""):
        raise ValueError(
            f"{_QUALITY_METRIC_ENV} must be one of "
            f"auto/loss_drop/cosine/none, got {want!r}"
        )
    return "loss_drop" if bool(getattr(config, "measure_grad", False)) else "cosine"


# Walk hyper-parameters. The defaults ARE the measured configuration above;
# changing them invalidates the correlation numbers, so they are env-tunable
# but never silently different between the trainer and the measure actors
# (both read this module in the same process tree / the same sbatch env).
def _walk_steps() -> int:
    return int(os.environ.get("ALPHAGRAD_WALK_STEPS", "200"))


def _walk_lr() -> float:
    return float(os.environ.get("ALPHAGRAD_WALK_LR", "1e-3"))


def _walk_probe_seed() -> int:
    """Seed of the PROBE BATCH. The batch must be IDENTICAL for every plan in a
    run or the scores are not comparable, so it is a fixed constant rather than
    anything derived from the episode / env / actor."""
    return int(os.environ.get("ALPHAGRAD_WALK_PROBE_SEED", "20260807"))


def _walk_noise_std() -> float:
    """OPTIONAL, DEFAULT OFF. Resampling N(0, 0.3) pixel noise on the walk
    batch each step lifts Pearson 0.922 -> 0.950, but it changes the measured
    configuration, so the headline numbers stay reproducible only at 0.0."""
    return float(os.environ.get("ALPHAGRAD_WALK_NOISE_STD", "0.0"))


# One probe batch per (process, data-generator, shape) — built once, kept on
# the host, device_put per measurement onto whichever device the plan was
# compiled for.
_PROBE_BATCH: dict = {}


def _probe_batch(config, base_args):
    """The FIXED probe batch: real data from ``config.data_gen`` at a fixed key.

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
    _key = (id(config.data_gen), _walk_probe_seed(),
            tuple(getattr(a, "shape", ()) for a in base_args[:2]))
    hit = _PROBE_BATCH.get(_key)
    if hit is not None:
        return hit
    k = jrand.PRNGKey(_walk_probe_seed())
    data = config.data_gen(jrand.split(k, 5))
    data = tuple(jax.device_get(d) for d in data)
    _PROBE_BATCH.clear()          # one batch per process, by construction
    _PROBE_BATCH[_key] = data
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


def _loss_drop_quality(config, compiled_approx, base_args, device=None):
    """Reward slot 6 under ``ALPHAGRAD_QUALITY_METRIC=loss_drop``.

    ``compiled_approx`` is the AOT-compiled DENSE executable of the plan under
    evaluation, so the gradient it returns carries the plan's elimination order
    AND its approximations. ``config.target_fun`` is the TRUE scalar loss (this
    metric is only selected when the measured graph is a scalar loss), so the
    probe is never contaminated by the approximation.

    Returns ``None`` when the walk cannot be defined (no data generator, no
    updatable weight slots, shape mismatch), which the caller treats as "fall
    back to the legacy cosine" rather than as a score.
    """
    probe = _probe_batch(config, base_args)
    if probe is None or config.target_fun is None:
        return None
    wnums = _walk_argnums(config, base_args)
    if not wnums:
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

    loss_fn = _jit_loss(config)
    w = [full[i] for i in wnums]

    def _call_loss(weights):
        a = list(full)
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
    if not _WALK_FINGERPRINT:
        _WALK_FINGERPRINT.append(1)
        import hashlib as _hl
        _h = _hl.blake2b(digest_size=8)
        for _a in (x0,) + tuple(full[1:2]) + tuple(w):
            _h.update(np.asarray(jax.device_get(_a)).tobytes())
        print(
            f"[measure] loss-drop walk armed: probe batch "
            f"{tuple(np.asarray(x0).shape)} (seed {_walk_probe_seed()}), "
            f"{_walk_steps()} Adam steps @ lr {_walk_lr():g}, "
            f"noise std {_walk_noise_std():g}, L0={L0:.6g}, "
            f"fingerprint(probe+W0)={_h.hexdigest()}",
            flush=True)

    T = _walk_steps()
    lr = _walk_lr()
    noise = _walk_noise_std()
    nkey = jrand.PRNGKey(_walk_probe_seed() + 1)
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
# MEMORY PARITY (ALPHAGRAD_MEM_PARITY=1, default on; 0 disables).
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
_MEM_PARITY_ON = os.environ.get("ALPHAGRAD_MEM_PARITY", "1") != "0"


def _record_mem_parity(compiled, runtime_peak, source: str,
                       is_terminal: bool) -> None:
    """One (static, runtime, source) triple per measurement."""
    if not _MEM_PARITY_ON or compiled is None:
        return
    _t0 = time.perf_counter()
    try:
        ma = compiled.memory_analysis()
    except Exception:
        ma = None
    if ma is None:
        _st = _so = None
    else:
        _st = float(getattr(ma, "temp_size_in_bytes", 0) or 0.0)
        _so = float(getattr(ma, "output_size_in_bytes", 0) or 0.0)
    rec = {
        "static_temp_bytes": _st,
        "static_output_bytes": _so,
        "static_total_bytes": None if _st is None else _st + _so,
        "runtime_peak_bytes": (None if runtime_peak is None
                               else float(runtime_peak)),
        # "runtime_delta"   the reward's peak_memory IS the measured delta;
        # "static_fallback" allocator stats were unavailable and the STATIC
        #                   estimate was substituted in place (a different
        #                   quantity -- this is the field that used to be a
        #                   one-shot printed warning and nothing else);
        # "bypassed"        ALPHAGRAD_BYPASS_RESOURCE_MONITOR=1, no reading.
        "peak_source": source,
        "terminal": bool(is_terminal),
    }
    _MEM_PARITY.append(rec)
    _prof_add("cb.mem_parity", time.perf_counter() - _t0)
    if os.environ.get("ALPHAGRAD_DEBUG_MEASURE", "0") == "1":
        _s, _r = rec["static_total_bytes"], rec["runtime_peak_bytes"]
        _f = f"{_r / _s:.3f}" if (_s and _r) else "n/a"
        print(f"[mem-parity] static(temp+out)={_s} runtime_peak={_r} "
              f"runtime/static={_f} source={source} "
              f"terminal={bool(is_terminal)}", flush=True)


def consume_mem_parity() -> list:
    """Pop the per-period memory-parity records (see ``_record_mem_parity``)."""
    out = list(_MEM_PARITY)
    _MEM_PARITY.clear()
    return out


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
            r = fn(*[_cb_slot(x, i, E) for x in a])
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

        * graphax's nominal-shape asserts are EXACT-AD only — core.py gates
          them on ``not _perpath and not _is_approx_cfg and not approx_active()``
          — and ``apply_compress`` drops the axis POINTER (``Index.axis =
          None``), not the logical size, so a compressed edge still contracts.
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
    eqn = jaxpr.eqns[vertex - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return ()
    out_shape = eqn.outvars[0].aval.shape
    out_len = len(out_shape)
    primal_shapes = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    if not primal_shapes:
        return ()  # no non-literal inputs → no edges to transform

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


# #72: apply a face's `new`-slot approximation to the EXISTING EDGE at the
# join as well as to the contraction result, so compute is saved on both
# operands of the add. Set 0 to restore the contraction-only behaviour --
# the two are NOT comparable, since this changes the measured object.
_NEW_SLOT_JOIN = os.environ.get("ALPHAGRAD_NEW_SLOT_JOIN", "1") != "0"


def _face_dict_for_vertex(config, ij, v, face_row, face_skip):
    """ONE vertex's ``{face_key: slots|SKIP_FACE}`` from its wire rows,
    enumerated on ``ij``'s CURRENT graph — call BEFORE eliminating ``v``.
    Single source of truth for `_face_transforms_for_order` (standalone
    replay) and the unified tokenizer path (ALPHAGRAD_UNIFIED_FACE_ENUM=1),
    which rides the tokenizer's own IncrementalJaxpr instead of replaying a
    second, byte-identical elimination."""
    from graphax import SKIP_FACE, faces_of
    from alphagrad.approx.common.masks import make_live_masked_hook

    keys = faces_of(ij.graph, ij.tgraph, int(v), config.jaxpr)
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
    for f, key in enumerate(keys[:MAX_FACES]):
        if int(face_skip[f]) == 1:
            per_face[key] = SKIP_FACE
            continue
        slots = []
        for s in range(FACE_SLOTS):
            # The decoder walks all MAX_RULES slots — pad the single face
            # row with end-sentinels.
            # `face_row` may be a numpy view (the caller no longer pays for a
            # dense `.tolist()`), so pull the 3 wire ints out explicitly.
            one_row = [[int(x) for x in face_row[f][s]]] + [
                [-1, -1, 0]
            ] * (MAX_RULES_PER_VERTEX - 1)
            rules = decode_vertex_rule_specs(config.jaxpr, int(v), one_row)
            # THE per-FACE sink. Under --live-faces the per-vertex rows are
            # all-exact END rows, so the per-vertex sink below is never
            # constructed and this is the ONLY place approx_applied/* can come
            # from. ``gated`` because these same objects are replayed by the
            # face-enum walk right below and by the tokenizer -- only the
            # armed scope inside ``_do_compile_approx`` is the measurement.
            slots.append(
                make_live_masked_hook(tuple(rules), stats=_PER_FACE_STATS,
                                      gated=True) if rules else None)
        if any(sl is not None for sl in slots):
            # #72. A bare 3-tuple means CONTRACTION ONLY to graphax
            # (_normalize_perpath, core.py:797): pre/post hit the two
            # contraction operands and `new` the contraction RESULT, while
            # the join hooks stay None. The requested semantics is that a
            # `new`-slot approximation ALSO applies to the existing edge
            # this result is added to, so compute is saved on both
            # operands of the join.
            #
            # graphax already wires that: `_h_rhs` is applied to `_edge`
            # -- the existing edge -- immediately before the add
            # (core.py:1679). So emit the TWO-OP form and put the new-slot
            # hook in `rhs`.
            #
            # `lhs` stays None on purpose: `new` has already transformed
            # the contraction result at core.py:1657, and `lhs` hits that
            # SAME tensor at 1678, so setting it would apply the
            # approximation twice. `res` (the summed edge) is a decision
            # the head does not make.
            #
            # Faces with no existing edge are unaffected -- graphax simply
            # never reaches the join hooks for them.
            _new_hook = slots[2] if len(slots) > 2 else None
            if _new_hook is not None and _NEW_SLOT_JOIN:
                per_face[key] = (tuple(slots), (None, _new_hook, None))
            else:
                per_face[key] = tuple(slots)
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
    return {
        "xla_gpu_autotune_level": 0,
        "xla_gpu_enable_triton_gemm": False,
    }


_MEASURE_COMPILE_FALLBACKS = {"n": 0}


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
    ALPHAGRAD_MEASURE_COMPILE_FALLBACK=0 disables."""
    try:
        return lowered.compile(compiler_options=_measure_compiler_options())
    except Exception as _e:
        if os.environ.get("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", "1") == "0":
            raise
        _m = str(_e)
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
        if not any(_sig in _m for _sig in (
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
                          face_rows_list, face_skips_list, wire_sig):
    """`_face_transforms_for_order` served from live state. Same result."""
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    K = len(o_list)
    base = (id(config.jaxpr), tuple(config.argnums))
    if wire_sig is None:
        wire_sig = _face_wire_keys(np.asarray(face_rows_list),
                                   np.asarray(face_skips_list), K)
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
                config, ch.ij, v, face_rows_list[k], face_skips_list[k])
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
                               wire_sig=None):
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
            face_skips_list, wire_sig)

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
            config, ij, v, face_rows_list[k], face_skips_list[k])
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

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()  # list of MAX_RULES x 3 lists
    _pf("cb.wire2py")

    # P1 per-path actions: build graphax's {vertex: {face_key: slots|SKIP}}
    # only when any face action is present in the prefix (all -1 / all 0 is
    # the per-vertex mode and must stay byte-identical to it).
    _faces_np = np.asarray(face_specs)[: len(o_list)]
    _skips_np = np.asarray(face_skips)[: len(o_list)]
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
                   and os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "0")
                   == "1")
    # ONE sparse per-vertex encoding of the face wires per callback,
    # shared by the live elimination state below and by the stream
    # cache's `face_key` further down -- they used to build one each.
    _wire_sig = None
    if _have_face_actions:
        _wire_sig = _face_wire_keys(_faces_np, _skips_np, len(o_list))
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
                # it that face's live operand — so this is where per-path
                # legality is decided. Rules that don't fit a given face are
                # skipped for that face only (not for the whole vertex).
                from alphagrad.approx.common.masks import make_live_masked_hook
                _face_stats = _PER_FACE_STATS
                if rules:
                    transforms.append(
                        (int(v),
                         (make_live_masked_hook(rules, stats=_face_stats,
                                                gated=True),))
                    )
                # The tokenizer eliminates its OWN graph copy with equivalent
                # hooks but no stats sink — the measured graph's hooks own the
                # applied/skipped counters.
                if tok_rules:
                    tok_rules_by_v[int(v)] = (
                        make_live_masked_hook(tok_rules),)
            else:
                if rules:
                    transforms.append((int(v), tuple(rules)))
                if tok_rules:
                    tok_rules_by_v[int(v)] = tuple(tok_rules)

    _pf("cb.decode")
    if os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "0") == "1":
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
            tokens, eqn_ids = _delta_observation(stream, seg_ids, _last_start)
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

    _pf("cb.tokenize")
    if init:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # Terminal-only fast path: every reward component is sparse — only the
    # final step (when the elimination order is complete) gets a non-zero
    # signal, so we skip the expensive jacve compile/exec on every prior
    # step. Cumsum-style returns (alpha0/mu0) and per-rollout aggregations
    # (gdpo) collapse to the terminal reward; gfn already reads only the
    # last step. PPO sees a sparse-reward MDP, which GAE handles natively.
    if config.terminal_rewards_only and not is_terminal:
        _pf("cb.nonterm_tail")
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

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
    if _SKIP_COUNT_OPS:
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
    if not _SKIP_COUNT_OPS and muls_adds_fmas > _muls_cap:
        _record_truncated_plan()
        if _dbg_measure or os.environ.get("ALPHAGRAD_DEBUG_DEGEN", "0") == "1":
            print(f"[trunc] MULS-CAP muls={muls_adds_fmas:.3g} > "
                  f"{_muls_cap:.3g} step={int(stop)} order={o_list} "
                  f"(excluded from gradient)", flush=True)
        return tokens, eqn_ids, _truncated_reward()

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
        return tokens, eqn_ids, rewards

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
    h.update(int(stop).to_bytes(4, "little", signed=False))
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
        try:
            return _compile_measure(
                jax.jit(
                    jacve(
                        config.target_fun,
                        list(o_list),
                        argnums=config.argnums,
                        has_aux=config.has_aux,
                        sparse_representation=config.sparse,
                        transforms=transforms,
                        face_transforms=ft_by_vertex,
                    ),
                    keep_unused=True,
                )
                .lower(*args_for_lower)
            )
        finally:
            disarm_face_counts()

    def _do_compile_exact():
        return _compile_measure(
            jax.jit(
                jacve(
                    config.target_fun,
                    list(o_list),
                    argnums=config.argnums,
                    has_aux=config.has_aux,
                    sparse_representation=config.sparse,
                ),
                keep_unused=True,
            )
            .lower(*args_for_lower)
        )

    # The EXACT compile ignores `transforms` / `face_transforms` entirely (see
    # _do_compile_exact above — it only takes o_list + config + arg shapes), so
    # keying it on the approximation specs makes two envs that picked the SAME
    # ORDER with different approximations compile a byte-identical executable
    # twice. Narrow the exact key to what the exact executable actually depends
    # on. Bit-identical result, strictly more cache-friendly.
    h_ex = hashlib.blake2b(digest_size=16)
    h_ex.update(np.asarray(partial_order, dtype=np.int32).tobytes())
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            h_ex.update(repr(a.shape).encode())
            h_ex.update(repr(a.dtype).encode())
    if callback_device is not None:
        h_ex.update(repr(callback_device).encode())
    exact_cache_key = h_ex.digest()

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
        return tokens, eqn_ids, _truncated_reward()

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
        return tokens, eqn_ids, _truncated_reward()

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
    _qmetric = quality_metric(config)
    if is_terminal and _qmetric == "cosine":
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
    _pf("cb.xla_compile")

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
    # `reps_per_point` timing repetitions. Reps exist only to average timer
    # noise, so without --measure-latency we take 1 rep per point. Quality is
    # deterministic in the input, so it is computed ONCE PER POINT (not per
    # rep) and medianed across points — the old code used point 0 only.
    n_points = max(1, int(getattr(config, "num_data_points", 5)))
    if eval_samples:
        n_points = min(n_points, len(eval_samples[0]))
    n_reps = max(1, int(getattr(config, "reps_per_point", 4))) if config.measure_latency else 1

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
    # ALPHAGRAD_QUALITY_METRIC=cosine it holds one Jacobian cosine per
    # calibration point; under the default loss_drop it holds exactly ONE
    # entry, the plan's 200-step Adam-walk loss drop.
    cosines: list = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []
    # WHICH quantity the peak samples hold (see _record_mem_parity): the
    # measured runtime delta, the substituted static estimate, or nothing.
    _peak_src = "not_measured"

    # OOM during EXECUTION is truncation too (see _oom_truncate): the
    # measurement allocates the full approximated Jacobian, so a graph that
    # compiled fine can still exhaust the device here. Excluded from the
    # gradient rather than scored.
    try:
        for i in range(n_points):
            if eval_samples:
                eval_args_i = [arg[i] for arg in eval_samples]
            else:
                eval_args_i = list(args)
            if callback_device is not None:
                eval_args_i = [jax.device_put(d, callback_device) for d in eval_args_i]

            # ResourceMonitor already runs ``jax.effects_barrier()`` in
            # ``__enter__`` / ``__exit__``, so we don't need an extra
            # ``block_until_ready`` on the result — the barriers drain the
            # device queue both for the timer and the memory tracker.
            #
            # ``ALPHAGRAD_BYPASS_RESOURCE_MONITOR=1`` skips the construction +
            # context manager entirely (vs. the lighter
            # ``ALPHAGRAD_DISABLE_RESOURCE_MONITOR`` which only swapped the
            # class to a no-op). This is the strongest cut available short
            # of patching the import: no monitor object is created, no
            # ``__enter__`` / ``__exit__`` runs, no ``stats`` dict is read.
            # Used to isolate whether the per-call Python lifecycle around
            # ``ResourceMonitor`` (not its C++ tracker) leaks. peak_memory +
            # latency_ns are zero for the run.
            inner = max(1, int(getattr(config, "latency_inner_reps", 1)))
            # WARMUP (config.latency_warmup, default 0 = unchanged): untimed
            # executions before the first timed rep, matching the elimrl/POMO
            # worker's single warmup call. Runs OUTSIDE every timing and memory
            # window, so it can only remove first-touch bias, never add to it.
            for _w in range(max(0, int(getattr(config, "latency_warmup", 0)))):
                jax.block_until_ready(compiled_cost(*eval_args_i))
            _direct = os.environ.get("ALPHAGRAD_DIRECT_MEASURE", "0") == "1"
            for _rep in range(n_reps):
                if os.environ.get("ALPHAGRAD_BYPASS_RESOURCE_MONITOR", "0") == "1":
                    out_approx = compiled_cost(*eval_args_i)
                    latency_samples.append(0.0)
                    peak_mem_samples.append(0.0)
                    _peak_src = "bypassed"
                elif _direct:
                    # Spec-native primitives, no jax_memory_monitor object at all
                    # (its per-call C++ trackers are the leak that killed v10):
                    # clear_memory_stats() resets the high-water mark, the inner
                    # loop is timed with perf_counter around a drained queue, and
                    # peak_bytes_in_use is read per device afterwards.
                    #
                    # ABOVE-BASELINE DELTA, not the absolute high-water mark.
                    # peak_bytes_in_use is a DEVICE-WIDE ABSOLUTE counter, and
                    # clear_memory_stats() resets the COUNTER but not the
                    # resident baseline -- so an absolute reading is
                    # (resident baseline + this call transient). Measured: a
                    # dirty allocator made four very different approximations
                    # all read ~281 MB when the true per-call cost was 1.6 MB.
                    # A 9-method comparison ranked this delta first among
                    # runtime methods: CV 0.0000% over 200 reps, byte-identical
                    # across separate processes, 0 drift after a 3 GB
                    # alloc/free or a real OOM. It is exactly what
                    # ResourceMonitor computes internally.
                    # CAVEAT: device-wide, so a co-resident actor allocating on
                    # the same GPU inflates it (CV 49.7% under a noisy
                    # neighbour) -- keep one measure process per device.
                    # CPU backends EXPOSE clear_memory_stats but raise
                    # UNIMPLEMENTED when called (hasattr passes, the call
                    # dies) -- measured killing both v40 arms at episode 0.
                    # Without allocator stats the peak falls back to the
                    # compiled executable's memory_analysis(): args + outputs
                    # + temps, the deterministic static peak the AZ stack
                    # validated as its memory cost (CV exactly 0 by
                    # construction; latency stays the real perf_counter
                    # timing either way).
                    _have_stats = True
                    # Drain FIRST: clear_memory_stats() resets the high-water
                    # counter, but work still in flight from the previous rep
                    # lands after the reset and is charged to THIS rep. Barrier
                    # -> clear -> barrier makes the window tight.
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
                        out_approx = compiled_cost(*eval_args_i)
                    jax.block_until_ready(out_approx)
                    _t1 = time.perf_counter()
                    if _have_stats:
                        _peak_abs = 0.0
                        for _d in unique_devices:
                            _stats = _d.memory_stats() or {}
                            _peak_abs += float(
                                _stats.get("peak_bytes_in_use", 0.0))
                        _peak = max(0.0, _peak_abs - _base)
                    else:
                        # SUBSTITUTE IN PLACE — one memory channel, so the
                        # static estimate lands in peak_memory rather than
                        # travelling as a second variable. Announced + counted.
                        _peak = _memory_analysis_bytes(compiled_cost) or 0.0
                        _note_static_peak_fallback(
                            "this backend does not expose allocator statistics")
                    _peak_src = ("runtime_delta" if _have_stats
                                 else "static_fallback")
                    latency_samples.append((_t1 - _t0) / inner * 1e9)  # → ns
                    peak_mem_samples.append(_peak)
                else:
                    with _get_resource_monitor(unique_devices) as monitor:
                        # Accumulation loop (spec, default 50 when opted in): the
                        # executions queue back-to-back inside one monitor window
                        # and the exit barrier drains them all, so time/inner is a
                        # per-execution latency with dispatch + timer overhead
                        # amortized. Peak memory is unaffected (same executable,
                        # same buffers each pass).
                        for _k in range(inner):
                            out_approx = compiled_cost(*eval_args_i)
                    # Key by name instead of unpacking ``.values()`` so this
                    # stays robust to dict-order / API tweaks in
                    # jax_memory_monitor.
                    latency_s = float(monitor.stats.get("time", 0.0)) / inner
                    peak_bytes = float(monitor.stats.get("memory", 0.0))
                    if peak_bytes <= 0.0:
                        # ResourceMonitor's DEVICE peak is structurally 0 on a
                        # CPU backend, so without this the memory channel is a
                        # flat zero and --mem-type peak_memory trains on
                        # nothing. Same in-place substitution as the _direct
                        # branch above: one channel, announced and counted.
                        peak_bytes = _memory_analysis_bytes(compiled_cost) or 0.0
                        _note_static_peak_fallback(
                            "ResourceMonitor reported a zero device peak "
                            "(structural on CPU backends)")
                        _peak_src = "static_fallback"
                    else:
                        _peak_src = "runtime_delta"
                    latency_samples.append(latency_s * 1e9)  # → ns
                    peak_mem_samples.append(peak_bytes)

            if compiled_cost is not compiled_approx and compiled_exact is not None:
                # The timed run above used the sparse-boundary executable;
                # quality must compare DENSE outputs (shape parity with the
                # exact reference). One untimed dense call, terminal only.
                out_approx = compiled_approx(*eval_args_i)
            # ``compiled_exact`` is only executed at the terminal step
            # (see the ``is_terminal`` guard around its compile, above).
            # For non-terminal steps we still loop n_samples times for
            # ``compiled_approx`` (peak_memory + latency need it), but
            # we skip the gold-standard execution that the quality
            # comparison would otherwise consume.
            if compiled_exact is not None:
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
                # Score THIS point now and let the pair go out of scope —
                # cos is the trained quality channel; the residual is
                # discarded (frob was dropped as a channel).
                _jac_a = out_approx[1] if config.has_aux else out_approx
                _jac_e = out_exact[1] if config.has_aux else out_exact
                _cos, _rf = _quality_metrics(_jac_e, _jac_a)
                cosines.append(_cos)

        # ---- LOSS-DROP QUALITY ------------------------------------------
        # ONE walk per PLAN (not per data point): the probe batch is fixed
        # across plans by construction, so repeating the walk over the
        # calibration samples would re-measure the same number. Runs after
        # the cost loop so the timing/peak windows above never contain it.
        _pf("cb.exec_measure")
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
    _record_mem_parity(compiled_cost,
                       peak_memory if peak_mem_samples else None,
                       _peak_src, is_terminal)

    # ------------------------------------------------------------------
    # Quality family — reward slot 6 (``quality``) + frob_residual.
    # ------------------------------------------------------------------
    # WHICH quantity lands in slot 6 is ``quality_metric(config)``:
    # ``loss_drop`` (default under --measure-grad) = the relative loss drop of
    # a 200-step Adam walk driven by this plan's gradient; ``cosine`` = the
    # legacy Jacobian cosine. Both are "higher is better", both are ~[0, 1]
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
        # frob is no longer a channel anything reads. The slot stays 0.0 for
        # real plans; the SENTINEL writers still stamp it, and the Ray pool's
        # sentinel test keys on that, so the wire format is unchanged.
        frob_residual = 0.0
        # XLA-analysis side-channel (log-only: the exact/approx compression
        # RATIO; the absolute static estimate is not exported — it only ever
        # substitutes into peak_memory). Nothing trains on it, so it rides the
        # same skip flag as the count pass.
        # memory_analysis() walks the compiled HLO, which is not free on the
        # big graphs a working policy produces.
        if not _SKIP_COUNT_OPS:
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

    # ---- ADDITIVE QUALITY GATE (owner 2026-08-09): see
    # _apply_quality_gate. Fires only when quality was MEASURED this
    # step (terminal + cosines non-empty) and fell below the env-var
    # threshold; clamps the two campaign cost channels to the
    # exact-reverse reference so destruction has no cost advantage.
    def _gate_order_floor():
        # LAZY: only clamped plans pay this compile+measure. Reuses the
        # order-keyed exact cache, so a repeat offender order is free.
        _gex = cached_compile(b"exact:" + exact_cache_key,
                              _do_compile_exact)
        return _measure_exec_cost(_gex, list(args_for_lower))
    latency_ns, peak_memory = _apply_quality_gate(
        latency_ns, peak_memory, cosine_sim, is_terminal,
        bool(cosines), config, list(args),
        order_floor_fn=_gate_order_floor)
    _pf("cb.quality")
    rewards = jnp.array(
        [
            -muls_adds_fmas,
            -flops,
            -latency_ns,
            -max_io_sum,
            -bytes_accessed,
            -peak_memory,
            cosine_sim,
            -frob_residual,
        ],
        dtype=jnp.float32,
    )

    # ZERO-WORK PLANS ARE KEPT (user-directed 2026-07-28).
    #
    # A plan that computes nothing reports every COST channel at its best
    # (fake-fast latency, tiny peak memory) — historically it was sentinelled
    # because that is the classic reward hack. It is no longer refused:
    #   * `frob_residual` is EXACTLY 1.0 for an all-zero Jacobian (the worst
    #     attainable value), so the quality channel already punishes it;
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
    if (not _SKIP_COUNT_OPS and is_terminal
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
                f"rules(d/c/q)={_n_diag}/{_n_comp}/{_n_quant} "
                f"skips={_n_skips} order={o_list}",
                flush=True,
            )

    return tokens, eqn_ids, rewards


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
    ):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "eval_args_samples", eval_args_samples)
        object.__setattr__(self, "_remote_pool", remote_pool)
        object.__setattr__(self, "_remote_timeout_s", float(remote_timeout_s))

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

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: core.ClosedJaxpr,
        argnums=None,
        args=None,
        has_aux=False,
        sparse=False,
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
        latency_inner_reps: int = 1,
        latency_warmup: int = 0,
        per_face: bool = False,
        measure_grad: bool = False,
        delta_obs: bool = False,
        quality_rewarded=None,
        **_compat,
    ):
        # ``num_data_points`` / ``reps_per_point`` ARE wired (see EnvConfig and
        # the execution loop in ``_callback``). ``latency_samples`` is the
        # legacy spelling: when a caller passes it explicitly we honour it as
        # the total budget so old scripts keep working.
        if latency_samples and latency_samples > 1:
            reps_per_point = max(1, int(latency_samples) // max(1, num_data_points))
        if measure_grad:
            # ``measure_grad`` asserts that the traced function is a SCALAR
            # loss, so differentiating its jaxpr already yields gradients —
            # which is what the spec asks to measure ("instead of returning
            # the Jacobian we return the gradients from the Jacobian"). It is
            # therefore a CONTRACT on the caller's target_fun, not extra work
            # for the env, and the previous hard NotImplementedError made
            # az_gumbel unimportable rather than protecting anything. Verify
            # the contract and continue.
            _outs = getattr(jaxpr, "out_avals", None) or [
                getattr(v, "aval", None) for v in jaxpr.jaxpr.outvars
            ]
            _bad = [
                a for a in _outs
                if a is not None and getattr(a, "shape", ()) not in ((), (1,))
            ]
            if _bad:
                raise ValueError(
                    "measure_grad=True requires a SCALAR-output target "
                    "(so jacve of it yields gradients); got output avals "
                    f"{[getattr(a, 'shape', a) for a in _outs]}. Wrap the "
                    "model in a scalar loss (see common.examples."
                    "scalar_loss_fn) or pass measure_grad=False."
                )
        assert (argnums is None and args is None) or not (args is None or args is None)
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
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts, eval_args_samples, axis_state_static, axis_valid_static = (
            children
        )
        # Back-compat: aux_data tuples produced before the remote-pool
        # fields were added are 3-tuples; new ones are 5-tuples.
        if len(aux_data) == 3:
            config, valid_vertices, num_envs = aux_data
            remote_pool = None
            remote_timeout_s = 60.0
        else:
            (
                config, valid_vertices, num_envs,
                remote_pool, remote_timeout_s,
            ) = aux_data
        return cls(
            config, args, consts, valid_vertices, num_envs, eval_args_samples,
            axis_state_static=axis_state_static,
            axis_valid_static=axis_valid_static,
            remote_pool=remote_pool,
            remote_timeout_s=remote_timeout_s,
        )

    def tokenize(self, init: bool = False, batched: bool = False):
        """Build the host-side function passed into ``io_callback``.

        If ``self._remote_pool`` is set, return a closure that
        dispatches each call to the pool via ``ray.get(timeout=...)``;
        on timeout / actor death the closure returns the standard
        sentinel ``(zeros, -1e10 reward)`` tuple so the rollout
        proceeds. Otherwise fall back to the inline ``_callback``
        path (single-process, no Ray) for backward compatibility
        with ``ppo.py`` / non-Ray callers.
        """
        if self._remote_pool is None:
            _fn = partial(_callback, self.config, init=init)
            # Only the STEP callback runs under vmap; reset() is called once,
            # unbatched, and must not be wrapped.
            return _batched_host(_fn) if (batched and _BATCHED_CALLBACK) else _fn

        # The pool's ``evaluate`` signature is
        # ``(order, specs, step, eval_samples, *, init)`` — but
        # ``io_callback`` passes ``(args, consts, order, specs, step,
        # *eval_samples)`` as positional args (see ``env.step`` line
        # 1360 and ``env.reset`` line 1297). Build an adapter that
        # drops ``args``/``consts`` (the actor's own env has its own
        # bound args) and re-packages ``eval_samples`` as a tuple.
        pool = self._remote_pool
        _obs_w = self.obs_width

        if batched:
            def _remote_callback_batched(args, consts, order, specs,
                                         face_specs, face_skips, step,
                                         *eval_samples):
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
                _local = set(
                    i for i in range(E)
                    if _term_local and _sti[i] >= int(ro[i].shape[0]))
                _remote = [i for i in range(E) if i not in _local]
                tk = np.zeros((E, _obs_w), np.int32)
                ei = np.zeros((E, _obs_w), np.int32)
                rw = np.zeros((E, NUM_REWARDS), np.float32)
                if _remote:
                    # prof/measure_wait: the host BLOCKS here until the
                    # measurement actors return. Timed separately from the
                    # cb.* phases because it is not trainer compute at all --
                    # it is idle time the rollout pays per (terminal) step.
                    _trace("measure_wait.enter")
                    _mw0 = time.perf_counter()
                    try:
                        (tokens, eqn_ids, rewards,
                         _sent) = pool.evaluate_batch(
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
                        )
                    finally:
                        _mwdt = time.perf_counter() - _mw0
                        _prof_add("prof/measure_wait", _mwdt)
                        _prof_sample("prof/measure_wait", _mwdt)
                        _trace("measure_wait.exit")
                    for k2, i in enumerate(_remote):
                        tk[i] = np.asarray(tokens)[k2]
                        ei[i] = np.asarray(eqn_ids)[k2]
                        rw[i] = np.asarray(rewards)[k2]
                for i in _local:
                    t_i, e_i, r_i = _callback(
                        self.config,
                        _cb_slot(args, i, E),
                        _cb_slot(consts, i, E),
                        ro[i], rs[i], rf[i], rk[i], _sti[i],
                        *[_cb_slot(x, i, E) for x in eval_samples],
                        init=init,
                    )
                    tk[i] = np.asarray(t_i)
                    ei[i] = np.asarray(e_i)
                    rw[i] = np.asarray(r_i)
                _cbdt = time.perf_counter() - _cb0
                _prof_add("prof/env_cb_host", _cbdt)
                _prof_sample("prof/env_cb_host", _cbdt)
                _trace("cb_batched.exit")
                return tk, ei, rw

            return _remote_callback_batched

        def _remote_callback(args, consts, order, specs, face_specs,
                             face_skips, step, *eval_samples):
            # The Ray pool path predates face actions (DEPRECATED line) —
            # they are dropped here; the pool's own env measures per-vertex.
            eval_samples_t = tuple(eval_samples) if eval_samples else None
            _mw0 = time.perf_counter()
            try:
                tokens, eqn_ids, reward = pool.evaluate(
                    order, specs, int(step),
                    eval_samples=eval_samples_t,
                    init=init,
                )
            finally:
                _mwdt = time.perf_counter() - _mw0
                _prof_add("prof/measure_wait", _mwdt)
                _prof_sample("prof/measure_wait", _mwdt)
            return tokens, eqn_ids, reward

        return _remote_callback

    @property
    def obs_width(self) -> int:
        """Width of the callback's token/eqn_id outputs.

        ``1 + MAX_DELTA_TOKENS`` under ``delta_obs`` (header slot + delta),
        ``MAX_TOKENS`` for the legacy full stream. The Ray measurement pool
        preallocates its buffers at this width too (``CpuApproxPool(
        max_tokens=...)``), so it must be read from the env, not assumed.
        """
        return (1 + MAX_DELTA_TOKENS if self.config.delta_obs else MAX_TOKENS)

    @property
    def _callback_shape(self):
        _w = self.obs_width
        return (
            jax.ShapeDtypeStruct((_w,), jnp.int32),
            jax.ShapeDtypeStruct((_w,), jnp.int32),
            jax.ShapeDtypeStruct((NUM_REWARDS,), jnp.float32),
        )

    def base_observation(self):
        """The BASE token stream as a standalone constant buffer.

        ``len(base_tokens())`` depends only on the jaxpr, not on the
        elimination order, so the base is the SAME array for every env and
        every episode: compute it once, on the host, and share it. There is
        no callback, no per-env copy, and the length is the tokenizer's own
        -- not a device-side scan over a padded buffer, which is where the
        id-0 undercount used to come from.

        Returns ``(tokens, eqn_ids, count)`` with both buffers
        ``(MAX_BASE_TOKENS,) int32`` and ``count`` a python int.
        """
        from graphax import IncrementalPathTokenizer

        vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))
        tk = IncrementalPathTokenizer(
            self.config.jaxpr, tuple(self.config.argnums),
            list(self.consts), list(self.args), vocab_size=vocab,
        )
        toks = [int(t) for t in tk.base_tokens()]
        ids = [int(g) for g in tk.last_eqn_ids()]
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
        t = np.zeros((MAX_BASE_TOKENS,), dtype=np.int32)
        e = np.full((MAX_BASE_TOKENS,), -1, dtype=np.int32)
        if n:
            t[:n] = np.asarray(toks, dtype=np.int32)
            e[:len(ids)] = np.asarray(ids, dtype=np.int32)
        return jnp.asarray(t), jnp.asarray(e), n

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

        vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))
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
            (initial_order.shape[0], MAX_FACES, FACE_SLOTS, 3), -1,
            dtype=jnp.int32,
        )
        initial_face_skips = jnp.zeros(
            (initial_order.shape[0], MAX_FACES), dtype=jnp.int32
        )

        if self.config.delta_obs:
            # The base stream is a host-side CONSTANT (base_observation()),
            # not part of the state, so reset makes NO host callback: at
            # step 0 nothing has been eliminated and the delta is empty.
            tokens = jnp.zeros((1,), dtype=jnp.int32)
            eqn_ids = jnp.zeros((1,), dtype=jnp.int32)
            delta_tokens = jnp.zeros((MAX_DELTA_TOKENS,), dtype=jnp.int32)
            delta_eqns = jnp.full((MAX_DELTA_TOKENS,), -1, dtype=jnp.int32)
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
                0,
                *(self.eval_args_samples
                  if self.eval_args_samples is not None else ()),
            )
            delta_tokens = jnp.zeros((1,), dtype=jnp.int32)
            delta_eqns = -jnp.ones((1,), dtype=jnp.int32)
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
            tokens=tokens,
            eqn_ids=eqn_ids,
            delta_tokens=delta_tokens,
            delta_eqns=delta_eqns,
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
            if action.face_rows is not None:
                face_rows = jnp.asarray(action.face_rows, dtype=jnp.int32)
                face_skip = jnp.asarray(action.face_skip, dtype=jnp.int32)
            else:
                face_rows = jnp.full(
                    (MAX_FACES, FACE_SLOTS, 3), -1, dtype=jnp.int32
                )
                face_skip = jnp.zeros((MAX_FACES,), dtype=jnp.int32)
        else:
            action = jnp.asarray(action, dtype=jnp.int32)
            sp_type = action // MAX_TOKENS
            target_vertex = action % MAX_TOKENS
            rule_specs = _legacy_sp_to_specs(sp_type)
            face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, dtype=jnp.int32)
            face_skip = jnp.zeros((MAX_FACES,), dtype=jnp.int32)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_specs = state.sparsity_specs

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_specs = curr_specs[shifted.astype(jnp.int32)].at[idx].set(rule_specs)
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
        tokens, eqn_ids, reward = _env_callback(
            self.tokenize(batched=True),
            self._callback_shape,
            _z if _drop_bound else self.args,
            _z if _drop_bound else self.consts,
            new_order,
            new_specs,
            new_face_specs,
            new_face_skips,
            new_step,
            *(() if _drop_bound
              else (self.eval_args_samples
                    if self.eval_args_samples is not None else ())),
            batched=True,
        )

        if self.config.delta_obs:
            # Slot 0 is the exact host-side token count (see
            # `_delta_observation`); slots 1.. are this step's delta.
            delta_count = tokens[0].astype(jnp.int32)
            delta_tokens = tokens[1:]
            delta_eqns = eqn_ids[1:]
            tokens = jnp.zeros((1,), dtype=jnp.int32)
            eqn_ids = jnp.zeros((1,), dtype=jnp.int32)
        else:
            delta_tokens = jnp.zeros((1,), dtype=jnp.int32)
            delta_eqns = -jnp.ones((1,), dtype=jnp.int32)
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
            tokens=tokens,
            eqn_ids=eqn_ids,
            delta_tokens=delta_tokens,
            delta_eqns=delta_eqns,
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
            tokens=jnp.zeros_like(state.tokens),
            eqn_ids=jnp.zeros_like(state.eqn_ids),
            # Legacy sentinels, byte-identical to what `step()` writes on
            # the non-delta path (pad is -1 for eqn ids, not 0) -- this
            # state is compared against `step()`'s field by field.
            delta_tokens=jnp.zeros((1,), dtype=jnp.int32),
            delta_eqns=-jnp.ones((1,), dtype=jnp.int32),
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
                (initial_order.shape[0], MAX_FACES, FACE_SLOTS, 3), -1,
                dtype=jnp.int32,
            ),
            face_skips=jnp.zeros(
                (initial_order.shape[0], MAX_FACES), dtype=jnp.int32
            ),
            tokens=jnp.zeros((self.obs_width,), dtype=jnp.int32),
            eqn_ids=jnp.zeros((self.obs_width,), dtype=jnp.int32),
            delta_tokens=jnp.zeros((1,), dtype=jnp.int32),
            delta_eqns=-jnp.ones((1,), dtype=jnp.int32),
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
