from __future__ import annotations

import gc
import itertools
import math
import os
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

# Observation token budget. The jaxpr token stream is clipped to this and
# zero-padded, so a graph whose stream is LONGER is only partially visible to
# the policy — nn256 emits ~4657 tokens, so the historical 4096 silently hid
# the tail of every observation. Settable via ALPHAGRAD_MAX_TOKENS (it sizes
# the io_callback's static output shape, so it must be fixed before the env is
# built, not per-call). Raise it until tokenization/truncated_count logs 0.
MAX_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_TOKENS", "4096"))
if MAX_TOKENS < 256:
    raise ValueError(f"ALPHAGRAD_MAX_TOKENS must be >= 256, got {MAX_TOKENS}")

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
    if raw_len > _TOKLEN_MAX[0]:
        _TOKLEN_MAX[0] = int(raw_len)


def _record_delta_length(delta_len: int) -> None:
    """Record one per-elimination DELTA length = one palimpsa call's width."""
    _DELTALEN_SUM[0] += int(delta_len)
    _DELTALEN_COUNT[0] += 1
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
MAX_DELTA_TOKENS = int(os.environ.get("ALPHAGRAD_MAX_DELTA_TOKENS", "1024"))

_INCR_TOK_CACHE: dict = {}
_INCR_TOK_CACHE_CAP = 512


def incremental_token_delta(jaxpr, argnums, consts, args, order_prefix,
                            vocab_size: int = 512):
    """Tokens ADDED by the last vertex of ``order_prefix`` (1-based ids).

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


def _incremental_stream_tokens(config, consts, args, o_list, specs_list,
                               tok_rules_by_v, ft_by_vertex=None,
                               face_key=None):
    """Full append-only observation stream for the prefix ``o_list``
    (ALPHAGRAD_INCREMENTAL_TOKENS=1): base tokens + one block per elimination
    (path tokens + ``approx`` echoes), from graphax's IncrementalPathTokenizer
    driven with the SAME transforms the measurement applies — the stream
    describes the approximated graph, not the intent.

    The stream for a prefix is a byte-wise prefix of the stream for any
    extension, so the cache extends the episode's tokenizer by ONE elimination
    per env step instead of re-tracing the whole Jacobian (``extract_jaxpr``)
    every step. Extending MUTATES the tokenizer, so the parent entry is POPPED
    before extension — a sibling chain that misses takes the honest cold
    replay (same policy as ``incremental_token_delta``).
    """
    from graphax import IncrementalPathTokenizer

    # 229 reserved tokens + 10 digits leave `vocab - 239` symbols for the name
    # alphabet (the tokenizer needs >= 2). 248 fits under the default 256-row
    # policy embedding with a 9-symbol alphabet.
    vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "248"))
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
        return hit[1], hit[2]

    tk, stream, seg_ids, done = None, None, None, 0
    # ANCESTOR EXTENSION IS ONLY SOUND WITHOUT COMPRESS IN THE PREFIX.
    #
    # Measured (tests/stream_prefix_property_test.py): the stream for a
    # length-k prefix is NOT a byte-prefix of the length-k+1 stream when the
    # prefix carries COMPRESS. `decode_vertex_rule_specs` emits Compress only
    # when `is_last=True`, so vertex k-1 is tokenized WITH its Compress at
    # length k and WITHOUT it at length k+1 — the streams diverge (token 681
    # of 931 on the test graph). Extending a cached parent would then hand the
    # policy a different observation than a cold replay, i.e. the observation
    # would depend on cache state.
    #
    # The `face_key is None` clause was already here (its comment gave a
    # different, weaker reason); the COMPRESS check is the one that actually
    # makes this correct, and it also protects the per-vertex path that the
    # original guard left exposed. Plans without COMPRESS still take the fast
    # path. Remove both conditions only after the COMPRESS last-vertex
    # restriction is lifted.
    _prefix_has_compress = any(
        any(int(r[0]) == COMPRESS_SENTINEL for r in rows) for _v, rows in steps
    )
    if face_key is None and not _prefix_has_compress:
        for cut in range(len(steps) - 1, 0, -1):
            parent = _INCR_STREAM_CACHE.pop(
                base_key + (tuple(steps[:cut]), None), None)
            if parent is not None:
                tk, stream, seg_ids, done = (
                    parent[0], list(parent[1]), list(parent[2]), cut)
                break
    if tk is None:
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
    for v, _rows in steps[done:]:
        stream += [int(t) for t in tk.eliminate(
            int(v), tok_rules_by_v.get(int(v), ()),
            (ft_by_vertex or {}).get(int(v)))]
        seg_ids += [int(g) for g in tk.last_eqn_ids()]

    if len(_INCR_STREAM_CACHE) > _INCR_STREAM_CACHE_CAP:
        _INCR_STREAM_CACHE.clear()
    _INCR_STREAM_CACHE[key] = (tk, stream, seg_ids)
    return stream, seg_ids


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
    total = sum(out.values()) or 1
    out["applied_fraction"] = out.get("applied", 0) / total
    return out


# ---------------------------------------------------------------------------
# XLA-analysis side-channel. The reward vector's shape is baked into the jit
# (NUM_REWARDS), so the extra diagnostics the logging spec asks for —
# xla_peak_memory (deterministic memory_analysis estimate) and the
# approx/exact memory COMPRESSION ratio — travel host-side like the
# tokenization stats: `_callback` records at each TERMINAL measurement, the
# driver polls once per episode via `consume_xla_memory_stats`.
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


def consume_xla_memory_stats() -> dict:
    """Pop the per-period XLA-memory telemetry (mirrors the truncation poll).

    Returns ``xla_peak_memory`` (mean approx bytes over the period's terminal
    measurements), and ``compression_ratio`` = mean(exact/approx) over
    measurements where both sides were analyzable — >1 means the approximated
    executable is smaller than the exact one (the observable sparsity /
    compression proxy under the dense measurement pipeline).
    """
    if not _XLA_MEM_APPROX:
        return {"xla_peak_memory": 0.0, "compression_ratio": 0.0, "count": 0}
    approx = np.asarray(_XLA_MEM_APPROX, dtype=np.float64)
    exact = np.asarray(_XLA_MEM_EXACT, dtype=np.float64)
    both = (approx > 0) & (exact > 0)
    ratio = float(np.mean(exact[both] / approx[both])) if both.any() else 0.0
    out = {
        "xla_peak_memory": float(approx.mean()),
        "compression_ratio": ratio,
        "count": int(approx.size),
    }
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
# 16, not 8. MEASURED on the nn256 cross-entropy graph over 82 elimination
# orders: the per-vertex face count peaks at 12 (vertex 9, `add`, 3 preds x 4
# succs after fill-in), and 8 silently dropped faces 9..12 -- they ran exact
# while every counter reported a healthy run. There is no cheap hard bound
# (faces = |preds| x |succs| and fill-in grows both), so the cap stays, but
# `_face_transforms_for_order` now COUNTS what it drops instead of slicing
# quietly. Raise it if `faces/over_cap` is ever non-zero.
MAX_FACES = int(os.environ.get("ALPHAGRAD_MAX_FACES", "16"))
_FACE_CAP_STATS = {"over_cap": 0, "max_seen": 0}


def consume_face_cap_stats() -> dict:
    out = dict(_FACE_CAP_STATS)
    _FACE_CAP_STATS["over_cap"] = 0
    return out
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
#   6 cosine_sim       — cosine similarity between flattened approximated and
#                        exact Jacobians, averaged over the calibration samples.
#   7 frob_residual    — relative Frobenius residual ||J_e - J_a||_F / ||J_e||_F.
NUM_REWARDS = 8
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
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
    tokens: Array
    eqn_ids: Array  # (MAX_TOKENS,) int32; per-token equation ID, -1 for non-eqn tokens
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
    flat_exact = _flatten_jacobians(jac_exact)
    flat_approx = _flatten_jacobians(jac_approx)
    if flat_exact is None or flat_approx is None:
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    if flat_approx.shape != flat_exact.shape or flat_approx.size == 0:
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)

    # Measure GPUs rotate but the exact reference is cached, so the two can
    # land on different devices -> jitted cossim raises "Received incompatible
    # devices". Co-locate onto the reference's device (read-only, one transfer
    # only when they differ). Also fixes the later flat_exact - flat_approx.
    try:
        _ed = next(iter(flat_exact.devices()))
        if next(iter(flat_approx.devices())) is not _ed:
            flat_approx = jax.device_put(flat_approx, _ed)
    except Exception:
        pass

    cos = cossim(flat_exact, flat_approx)
    # Certain quant/compress combos yield a complex-valued flattened Jacobian,
    # making cossim complex. Use the real part — matches the reward path's
    # existing real cast and keeps downstream float emission from crashing.
    cos = jnp.real(cos)
    exact_norm = jnp.linalg.norm(flat_exact)
    resid_norm = jnp.linalg.norm(flat_exact - flat_approx)
    rel_frob = resid_norm / jnp.maximum(exact_norm, jnp.sqrt(1e-7))

    if os.environ.get("ALPHAGRAD_DEBUG_QUALITY", "0") == "1":
        approx_norm = float(jnp.linalg.norm(flat_approx))
        e_norm = float(exact_norm)
        # Now that ``_callback`` only invokes ``_quality_metrics`` on
        # the terminal step (partial-order zero-Jacobian case is
        # short-circuited upstream), every line here represents a
        # real terminal evaluation. ``flush=True`` because Ray actor
        # stdout is line-buffered.
        print(
            f"[quality-debug] cos={float(cos):+.4f} frob={float(rel_frob):+.4f} "
            f"||exact||={e_norm:.3g} ||approx||={approx_norm:.3g} "
            f"size={flat_exact.size}",
            flush=True,
        )
    return cos, rel_frob


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


def decode_vertex_rule_specs(jaxpr, vertex, spec_rows, is_last: bool) -> tuple:
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
        kind ``row[2]`` — honored only when ``is_last`` (COMPRESS reduces
        ``val.ndim``, which trips graphax's shape-preservation assertion when
        the compressed edge feeds a later elimination).
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
            # COMPRESS: physical axis row[1], kind row[2]. Only honored on the
            # LAST vertex of the partial order (val.ndim reduction upstream of
            # a later elimination trips graphax's shape assertion).
            if not is_last:
                continue
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


def _face_transforms_for_order(config, consts, args, o_list, specs_list,
                               face_rows_list, face_skips_list):
    """Per-vertex ``face_transforms`` dicts for graphax, from the wire arrays.

    Face KEYS are graph-state dependent, so enumerate with ``faces_of`` on a
    structural replay that applies the SAME per-vertex rules and face
    transforms the measurement will — the k-th vertex's keys are only valid on
    the graph produced by the first k-1 (transformed) eliminations. Slots wrap
    their decoded rules in ``make_live_masked_hook`` (no stats sink here — a
    rule illegal on ITS operand is skipped per-slot, never raises);
    ``face_skips`` rows become ``graphax.SKIP_FACE``.
    """
    from graphax import SKIP_FACE, faces_of
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    ij = IncrementalJaxpr(config.jaxpr, tuple(config.argnums), list(consts),
                          list(args), track_faces=False)
    out: dict[int, dict] = {}
    last = len(o_list) - 1
    for k, v in enumerate(o_list):
        v = int(v)
        keys = faces_of(ij.graph, ij.tgraph, v, config.jaxpr)
        per_face: dict = {}
        rows_f = face_rows_list[k]
        skips_f = face_skips_list[k]
        if len(keys) > _FACE_CAP_STATS["max_seen"]:
            _FACE_CAP_STATS["max_seen"] = len(keys)
        if len(keys) > MAX_FACES:
            # NEVER a silent slice: the dropped faces run exact, which is a
            # smaller action space reported as if it were the full one.
            _FACE_CAP_STATS["over_cap"] += len(keys) - MAX_FACES
        for f, key in enumerate(keys[:MAX_FACES]):
            if int(skips_f[f]) == 1:
                per_face[key] = SKIP_FACE
                continue
            slots = []
            for s in range(FACE_SLOTS):
                # The decoder walks all MAX_RULES slots — pad the single face
                # row with end-sentinels.
                one_row = [list(rows_f[f][s])] + [
                    [-1, -1, 0]
                ] * (MAX_RULES_PER_VERTEX - 1)
                rules = decode_vertex_rule_specs(
                    config.jaxpr, v, one_row,
                    is_last=(k == last),
                )
                slots.append(
                    make_live_masked_hook(tuple(rules)) if rules else None
                )
            if any(sl is not None for sl in slots):
                per_face[key] = tuple(slots)
        if per_face:
            out[v] = per_face
        vertex_rules = decode_vertex_rule_specs(
            config.jaxpr, v, specs_list[k], is_last=(k == last)
        )
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

    # P1 per-path actions: build graphax's {vertex: {face_key: slots|SKIP}}
    # only when any face action is present in the prefix (all -1 / all 0 is
    # the per-vertex mode and must stay byte-identical to it).
    _faces_np = np.asarray(face_specs)[: len(o_list)]
    _skips_np = np.asarray(face_skips)[: len(o_list)]
    ft_by_vertex = None
    if len(o_list) and (np.any(_skips_np == 1) or np.any(_faces_np[..., 0] >= 0)
                        or np.any(_faces_np[..., 0] == COMPRESS_SENTINEL)
                        or np.any(_faces_np[..., 0] == QUANT_SENTINEL)):
        ft_by_vertex = _face_transforms_for_order(
            config, consts, args, o_list, specs_list,
            _faces_np.tolist(), _skips_np.tolist(),
        )

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
    last_v_idx = len(o_list) - 1
    for v_idx, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            config.jaxpr, int(v), specs_list[v_idx],
            is_last=(v_idx == last_v_idx),
        )
        if rules:
            if getattr(config, "per_face", False):
                # graphax invokes a CALLABLE transform once per face, handing
                # it that face's live operand — so this is where per-path
                # legality is decided. Rules that don't fit a given face are
                # skipped for that face only (not for the whole vertex).
                from alphagrad.approx.common.masks import make_live_masked_hook
                _face_stats = _PER_FACE_STATS
                transforms.append(
                    (int(v), (make_live_masked_hook(rules, stats=_face_stats),))
                )
                # The tokenizer eliminates its OWN graph copy with equivalent
                # hooks but no stats sink — the measured graph's hooks own the
                # applied/skipped counters.
                tok_rules_by_v[int(v)] = (make_live_masked_hook(rules),)
            else:
                transforms.append((int(v), tuple(rules)))
                tok_rules_by_v[int(v)] = tuple(rules)

    _pf("cb.decode+face_enum")
    if os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "0") == "1":
        # Append-only observation (spec): the stream grows by one block per
        # elimination and the whole extract_jaxpr re-trace of the Jacobian is
        # skipped. eqn_ids come from the tokenizer's own segment record —
        # one stream-global id per contraction/approx group, -1 elsewhere —
        # which is exactly the relational-gate contract (same/earlier/later
        # comparisons, no embedding-table bound).
        stream, seg_ids = _incremental_stream_tokens(
            config, consts, args, o_list, specs_list, tok_rules_by_v,
            ft_by_vertex=ft_by_vertex,
            face_key=(_faces_np.tobytes(), _skips_np.tobytes())
            if ft_by_vertex else None,
        )
        _record_token_length(len(stream))
        _record_tokenization_truncation(len(stream))
        tokens = jnp.asarray(stream[:MAX_TOKENS], dtype=jnp.int32)
        tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))
        _ids = np.full((MAX_TOKENS,), -1, dtype=np.int32)
        _n_ids = min(len(seg_ids), MAX_TOKENS)
        _ids[:_n_ids] = np.asarray(seg_ids[:_n_ids], dtype=np.int32)
        eqn_ids = jnp.asarray(_ids)
    else:
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
        return (
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
            .compile()
        )

    def _do_compile_exact():
        return (
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
            .compile()
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
    # ``compiled_exact`` is ONLY needed for the quality metrics
    # (cosine_sim, frob_residual). Those are meaningful only when the
    # elimination order is complete — graphax's ``jacve`` returns a
    # zero-norm Jacobian for any partial order, so comparing approx vs
    # exact mid-rollout yields ``(cos=0, frob=0)`` regardless. Skip
    # the compile + execute when the step is non-terminal; the cache
    # entry would never be re-used productively anyway.
    if is_terminal:
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

    out_approxs: list = []
    out_exacts: list = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []

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
            _direct = os.environ.get("ALPHAGRAD_DIRECT_MEASURE", "0") == "1"
            for _rep in range(n_reps):
                if os.environ.get("ALPHAGRAD_BYPASS_RESOURCE_MONITOR", "0") == "1":
                    out_approx = compiled_approx(*eval_args_i)
                    latency_samples.append(0.0)
                    peak_mem_samples.append(0.0)
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
                    for _d in unique_devices:
                        if hasattr(_d, "clear_memory_stats"):
                            _d.clear_memory_stats()
                    jax.effects_barrier()
                    _base = 0.0
                    for _d in unique_devices:
                        _bstats = _d.memory_stats() or {}
                        _base += float(_bstats.get("bytes_in_use", 0.0))
                    _t0 = time.perf_counter()
                    for _k in range(inner):
                        out_approx = compiled_approx(*eval_args_i)
                    jax.block_until_ready(out_approx)
                    _t1 = time.perf_counter()
                    _peak_abs = 0.0
                    for _d in unique_devices:
                        _stats = _d.memory_stats() or {}
                        _peak_abs += float(_stats.get("peak_bytes_in_use", 0.0))
                    _peak = max(0.0, _peak_abs - _base)
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
                            out_approx = compiled_approx(*eval_args_i)
                    # Key by name instead of unpacking ``.values()`` so this
                    # stays robust to dict-order / API tweaks in
                    # jax_memory_monitor.
                    latency_s = float(monitor.stats.get("time", 0.0)) / inner
                    peak_bytes = float(monitor.stats.get("memory", 0.0))
                    latency_samples.append(latency_s * 1e9)  # → ns
                    peak_mem_samples.append(peak_bytes)

            out_approxs.append(out_approx)
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
                    out_exacts.append(_hit)
                else:
                    out_exact = compiled_exact(*eval_args_i)
                    out_exacts.append(out_exact)
                    if _ex_key is not None:
                        # Bounded: one episode's worth of samples. A new
                        # episode changes every digest, so the old entries are
                        # dead weight and get dropped wholesale.
                        if len(_EXACT_CACHE) >= max(n_points, 1):
                            _EXACT_CACHE.clear()
                        _EXACT_CACHE[_ex_key] = out_exact

    except Exception as _exc:
        if _is_graphax_trace_failure(_exc):
            return _trace_truncate("measurement", _exc)
        if not _is_oom(_exc):
            raise
        return _oom_truncate('measurement', _exc)
    _pf("cb.exec_measure")
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

    # ------------------------------------------------------------------
    # Quality family — cosine similarity + relative Frobenius residual.
    # ------------------------------------------------------------------
    # Sparse-terminal channels: only computed on the terminal step
    # of the rollout. Partial elimination orders produce
    # ``||jac||=0`` for both ``compiled_approx`` and ``compiled_exact``
    # (verified empirically against graphax.jacve), so the
    # comparison is meaningless mid-rollout. Skip the work entirely
    # — the cost channels above (muls/io/flops/peak_memory) still
    # compute per step, only quality is sparse.
    if is_terminal and out_exacts:
        cosines: list = []
        frobs: list = []
        for out_approx, out_exact in zip(out_approxs, out_exacts):
            jac_approx = out_approx[1] if config.has_aux else out_approx
            jac_exact = out_exact[1] if config.has_aux else out_exact
            # cos is the trained quality channel; the residual it returns
            # alongside is discarded (frob was dropped as a channel).
            cos, _rel_frob = _quality_metrics(jac_exact, jac_approx)
            cosines.append(cos)
        cosine_sim = float(_aggregate_samples(cosines, want_top_quartile=True))
        # frob is no longer a channel anything reads. The slot stays 0.0 for
        # real plans; the SENTINEL writers still stamp it, and the Ray pool's
        # sentinel test keys on that, so the wire format is unchanged.
        frob_residual = 0.0
        # XLA-analysis side-channel (log-only: xla_peak_memory + the
        # exact/approx compression ratio). Nothing trains on it — the memory
        # objective is the MEASURED peak_bytes_in_use, not this static
        # estimate — so it rides the same skip flag as the count pass.
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
    # So we let it measure, let frob punish it, and keep the gradient.
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
                f"[zero-work] KEPT (frob punishes): muls=0 "
                f"lat={latency_ns:.3g} peak={peak_memory:.3g} "
                f"cos={cosine_sim:.3e} frob={frob_residual:.3e} "
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
        per_face: bool = False,
        measure_grad: bool = False,
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
            per_face=bool(per_face),
            target_fun=target_fun,
            data_gen=data_gen,
            exec_on_gpu=exec_on_gpu,
            measure_latency=measure_latency,
            terminal_rewards_only=terminal_rewards_only,
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

        if batched:
            def _remote_callback_batched(args, consts, order, specs,
                                         face_specs, face_skips, step,
                                         *eval_samples):
                # Face actions must never reach this path -- it cannot carry
                # them, and silently dropping them would measure a plan the
                # policy did not choose. ppo.py refuses the combination up
                # front; this is the backstop.
                _fs = np.asarray(face_specs)
                _sk = np.asarray(face_skips)
                if _fs.size and (_fs[..., 0] >= 0).any() or (_sk == 1).any():
                    raise RuntimeError(
                        "Ray measurement pool cannot carry face actions "
                        "(face_specs/face_skips are dropped by the pool's "
                        "per-vertex env). Run without --face-actions."
                    )
                _o = np.asarray(order)
                E = int(_o.shape[0])
                _sp = np.asarray(specs)
                _st = np.asarray(step).reshape(-1)
                _ev = (tuple(_cb_slot(x, 0, E) for x in eval_samples)
                       if eval_samples else None)
                tokens, eqn_ids, rewards, _sent = pool.evaluate_batch(
                    [_o[i] for i in range(E)],
                    [_sp[i] for i in range(E)],
                    [int(_st[i] if _st.size > 1 else _st[0]) for i in range(E)],
                    eval_samples=_ev,
                    init=init,
                )
                return (np.asarray(tokens), np.asarray(eqn_ids),
                        np.asarray(rewards))

            return _remote_callback_batched

        def _remote_callback(args, consts, order, specs, face_specs,
                             face_skips, step, *eval_samples):
            # The Ray pool path predates face actions (DEPRECATED line) —
            # they are dropped here; the pool's own env measures per-vertex.
            eval_samples_t = tuple(eval_samples) if eval_samples else None
            tokens, eqn_ids, reward = pool.evaluate(
                order, specs, int(step),
                eval_samples=eval_samples_t,
                init=init,
            )
            return tokens, eqn_ids, reward

        return _remote_callback

    @property
    def _callback_shape(self):
        return (
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((NUM_REWARDS,), jnp.float32),
        )

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
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
        )

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

        tokens, eqn_ids, reward = _env_callback(
            self.tokenize(batched=True),
            self._callback_shape,
            self.args,
            self.consts,
            new_order,
            new_specs,
            new_face_specs,
            new_face_skips,
            new_step,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
            batched=True,
        )

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
            tokens=jnp.zeros((MAX_TOKENS,), dtype=jnp.int32),
            eqn_ids=jnp.zeros((MAX_TOKENS,), dtype=jnp.int32),
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
