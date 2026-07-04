from __future__ import annotations

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


class _SafeResourceMonitor:
    """Wrap the real ``ResourceMonitor`` so a tracker failure degrades to a
    no-op reading instead of escalating to a SENTINEL (zeroed reward).

    A genuinely-degenerate edge can densify to a ``val=None`` SparseTensor
    (the uniform-grid canonical form, graphax 69e55fb): a tensor that is
    everywhere ``scalar_mult`` with no stored ``val`` array. When such a
    tensor (or any other uniform/empty-stat case) reaches the C++
    ``MemoryTracker.start()``, the native code subscripts a per-device
    stats entry that comes back ``None`` and raises
    ``TypeError: 'NoneType' object is not subscriptable`` -- which the
    callback turns into a SENTINEL, zeroing the whole reward (the
    51034/51040 collapse path). The COMPRESS cap upstream now makes the
    full-reduction-to-uniform edge unreachable, but a ``val=None`` tensor
    can arise from anywhere, so make the measure path itself robust: if the
    tracker raises on enter, fall back to NO memory tracking for THIS
    measurement (peak/stats report 0) and let the perf_counter timer still
    produce a valid latency -- the call measures instead of crashing.
    """

    def __init__(self, *args, **kwargs):
        self._inner = None
        self._failed = False
        try:
            self._inner = _RealResourceMonitor(*args, **kwargs)
        except Exception:
            self._failed = True

    def __enter__(self):
        if self._inner is not None:
            try:
                self._inner.__enter__()
            except Exception:
                # Tracker start failed (e.g. val=None / uniform tensor with
                # None device stats). Drop memory tracking for this measure
                # rather than aborting -> SENTINEL.
                self._failed = True
                try:
                    self._inner.__exit__(None, None, None)
                except Exception:
                    pass
                self._inner = None
        return self

    def __exit__(self, *a):
        if self._inner is not None:
            try:
                return self._inner.__exit__(*a)
            except Exception:
                self._failed = True
                self._inner = None
        return False

    @property
    def peak(self) -> int:
        if self._inner is None:
            return 0
        try:
            return self._inner.peak
        except Exception:
            return 0

    @property
    def duration(self) -> float:
        if self._inner is None:
            return 0.0
        try:
            return self._inner.duration
        except Exception:
            return 0.0

    @property
    def stats(self) -> dict:
        if self._inner is None:
            return {"time": 0.0, "memory": 0.0}
        try:
            return self._inner.stats
        except Exception:
            return {"time": 0.0, "memory": 0.0}


ResourceMonitor = (
    _NoopResourceMonitor
    if os.environ.get("ALPHAGRAD_DISABLE_RESOURCE_MONITOR", "0") == "1"
    else _SafeResourceMonitor
)

import math as _math

# ---------------------------------------------------------------------------
# Phase-2 memory-mitigation probe #4: in-process compiled-executable LRU.
# ``ALPHAGRAD_MEASURE_INPROC_LRU=M`` (M>0) wraps the two per-measure
# ``cached_compile`` results in a module-level ``OrderedDict`` bounded to M
# entries keyed on the callback cache_key. On overflow the oldest entry is
# popped and explicitly ``del``'d so the deserialized ``jax.stages.Compiled``
# (and, we hope, the XLA/PJRT executable + device buffers it references) can be
# released back to the measure GPU under the platform allocator. Default OFF
# (M=0) is byte-identical to the legacy uncached-in-proc path. NOTE: whether the
# ``del`` actually frees DEVICE memory must be confirmed empirically via
# nvidia-smi — if XLA retains the executable internally the ceiling won't move.
from collections import OrderedDict as _OrderedDict

try:
    _MEASURE_INPROC_LRU_MAX = int(os.environ.get("ALPHAGRAD_MEASURE_INPROC_LRU", "0") or "0")
except ValueError:
    _MEASURE_INPROC_LRU_MAX = 0
_MEASURE_INPROC_LRU: "_OrderedDict" = _OrderedDict()


def _inproc_lru_cached_compile(key: bytes, compile_fn):
    """LRU wrapper around the shared ``cached_compile``. When the in-proc LRU
    is enabled (M>0), retain the compiled executable in a bounded OrderedDict
    and evict+``del`` the oldest on overflow. When disabled, delegate straight
    through (no retention)."""
    from alphagrad.approx.common.compile_cache import cached_compile as _cc
    if _MEASURE_INPROC_LRU_MAX <= 0:
        return _cc(key, compile_fn)
    hit = _MEASURE_INPROC_LRU.get(key)
    if hit is not None:
        _MEASURE_INPROC_LRU.move_to_end(key)
        return hit
    compiled = _cc(key, compile_fn)
    _MEASURE_INPROC_LRU[key] = compiled
    _MEASURE_INPROC_LRU.move_to_end(key)
    while len(_MEASURE_INPROC_LRU) > _MEASURE_INPROC_LRU_MAX:
        _old_key, _old_val = _MEASURE_INPROC_LRU.popitem(last=False)
        del _old_val
    return compiled


# Cache the graphax vocabulary used by `compute_eqn_ids_from_tokens` — the
# tokenizer always uses the same digit_base, so the vocab is constant and
# rebuilding it on every callback is pure overhead.
_TOKEN_VOCAB, _, _ = _graphax_get_vocab()

MAX_TOKENS = 16384  # deep-COMPRESS 256-NN grad states reach ~13356 tok; 16k headroom

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

# Always-on raw_len telemetry — tracks EVERY tokenization (not just the ones
# exceeding MAX_TOKENS), so the per-episode mean/max/min jaxpr token length is
# visible in wandb regardless of truncation. This is what reveals how much the
# dynamic-substep expansion inflates the sequence and how low --max-substeps
# must go. Reset each poll by ``consume_tokenization_truncation_stats``.
_RAW_LEN_SUM: list[int] = [0]
_RAW_LEN_COUNT: list[int] = [0]
_RAW_LEN_MAX: list[int] = [0]
_RAW_LEN_MIN: list[int] = [0]   # 0 = unset (first sample initializes it)


def _record_tokenization_truncation(raw_len: int) -> None:
    """Bump the per-process truncation counter and emit a one-time
    ``warnings.warn`` on the first observation. Cheap: a counter
    increment + one branch. The warning carries the actual raw token
    length so the user can see how much headroom they need.
    """
    # Always-on raw_len stats (every tokenization, truncated or not).
    _RAW_LEN_SUM[0] += raw_len
    _RAW_LEN_COUNT[0] += 1
    if raw_len > _RAW_LEN_MAX[0]:
        _RAW_LEN_MAX[0] = raw_len
    if _RAW_LEN_MIN[0] == 0 or raw_len < _RAW_LEN_MIN[0]:
        _RAW_LEN_MIN[0] = raw_len
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
    rl_sum, rl_cnt = _RAW_LEN_SUM[0], _RAW_LEN_COUNT[0]
    rl_max, rl_min = _RAW_LEN_MAX[0], _RAW_LEN_MIN[0]
    _RAW_LEN_SUM[0] = 0
    _RAW_LEN_COUNT[0] = 0
    _RAW_LEN_MAX[0] = 0
    _RAW_LEN_MIN[0] = 0
    return {
        "count": int(count),
        "max_observed_len": int(max_len),
        "overflow_sum": int(overflow_sum),
        # Always-on raw_len (all tokenizations this period).
        "raw_len_sum": int(rl_sum),
        "raw_len_count": int(rl_cnt),
        "raw_len_max": int(rl_max),
        "raw_len_min": int(rl_min),
    }
# Upper bound on rule_specs rows per vertex. In dynamic-substeps mode this
# also bounds the number of typed micro-actions per vertex that survive
# :func:`micro_actions_to_rule_specs_jax` — set it to the same scale as
# the policy's ``max_substeps`` (≈ 2 × MAX_AXES_PER_VERTEX) so the
# translator doesn't silently truncate DIAG / COMPRESS rows the policy
# emitted. Memory cost is O(total_v × MAX_RULES_PER_VERTEX × 3) int32.
MAX_RULES_PER_VERTEX = 16
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
NUM_REWARDS = 10
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
    # index 8: DETERMINISTIC XLA peak (memory_analysis temp+output+args).
    # ``peak_memory`` (index 5) is the REAL measured peak (exact on GPU, sampled
    # & unreliable for sub-ms execs on CPU); this channel is the compile-time
    # XLA estimate — reliable + order-discriminating on CPU. Use peak_memory for
    # GPU-measured runs, xla_peak_memory for CPU-measured runs.
    "xla_peak_memory",
    # index 9: B_kstep closed-loop TRAINABILITY accuracy in [0, 1]. Runs a short
    # real training (K Adam steps, S seeds) with the rule's OWN approximate
    # gradient (from the ``measure_grad`` compiled fn) on real MNIST batches,
    # then reads the resulting MNIST test accuracy. Directly answers "would
    # training with this learning rule work" — the trainability proxy from the
    # qsig_B_kstep bake-off. Only populated (terminal step, measure_grad) when
    # ``ALPHAGRAD_BKSTEP=1``; else stays 0.0. Higher = better, so it is a
    # QUALITY channel (not stored negated).
    "bkstep_acc",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
QUALITY_REWARD_INDICES = (
    REWARD_INDEX["cosine_sim"], REWARD_INDEX["frob_residual"],
    REWARD_INDEX["bkstep_acc"],
)
# Cost = every non-quality channel — DERIVED (not hardcoded) so adding a channel
# (e.g. xla_peak_memory at idx 8) is picked up automatically, like
# reward_scaling.COST_REWARD_INDICES.
COMPUTE_REWARD_INDICES = tuple(
    i for i in range(NUM_REWARDS) if i not in QUALITY_REWARD_INDICES
)

# Sentinel reward returned when a per-vertex transform sequence matches an
# entry in the in-file blacklist (used during exploration to penalise
# pathological configurations). The blacklist is no longer wired up after
# the typed-transform migration; the array is kept for potential reuse.
# Worst-case sentinel reward (blacklist path). NUM_REWARDS-aware so it can't
# rot out of sync when channels are added: every channel is -1e10 except
# cosine_sim, whose worst value is -1.0.
_SENTINEL_BAD_REWARD = jnp.array(
    [
        -1.0 if i == REWARD_INDEX["cosine_sim"]
        else 0.0 if i == REWARD_INDEX["bkstep_acc"]  # worst trainability = 0 acc
        else -1e10
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
# COMPRESS through the vertex elimination DAG is now supported on ANY
# vertex (not just the last). graphax core-v2 owns the implicit-
# (compressed-)dim algebra: `apply_compress` drops the physical `val`
# axis and marks the logical dim `axis=None`, and produce_compress.py
# defines how that implicit dim CONTRACTS / combines downstream WITHOUT
# materializing the dropped axis (closed in {Dense, Block-diag}). A
# Compress whose reduced edge feeds a subsequent `_eliminate_vertex` is
# therefore handled, not asserted. The two genuine validity constraints
# remain enforced below:
#   * the FULL-REDUCTION CAP — never drop the last remaining physical
#     axis (that folds the edge to `val=None` / uniform grid, which both
#     cossim-collapses the reward and trips the measure device tracker);
#   * the axis-fit / kind-range checks (apply_compress would ValueError
#     on a mis-fitting axis otherwise).
# Any residual per-edge geometry miss is caught by graphax core.py's
# per-edge try/except (apply_compress raises ValueError -> that edge's
# transform is skipped, the edge densified back to nominal form), so a
# non-terminal COMPRESS degrades gracefully instead of crashing.
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
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory", "xla_peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False
    # Latency requires running the compiled fn N times per step (1× for a
    # single noisy sample; >=8 for the top-quartile-mean smoothing). With
    # ``--rewards cmp`` we always compute latency now (to populate the
    # full 6-cost-channel vector), so the default went from N=10 to N=1 —
    # per-call latency is noisier but the per-episode mean across ~192
    # calls/ep is dominated by mean(noise)=0 anyway. Use
    # ``latency_samples`` to crank back up if per-step denoised latency
    # matters for your policy.
    measure_latency: bool = False
    latency_samples: int = 1
    # Robust noisy-channel aggregation: when `num_data_points >= 1` and
    # `reps_per_point >= 1`, the reward loop runs `num_data_points *
    # reps_per_point` total measurements (5 × 4 = 20 by default — five
    # data points sampled per episode, each replayed four times). Noisy
    # channels (latency_ns, peak_memory, frob_residual, cosine_sim) are
    # aggregated by the **P-`percentile_keep`** percentile across the
    # full pool: e.g. `percentile_keep=0.60` returns the 60th percentile
    # of the pool, which is "slowest 60% latency / highest 60% memory /
    # worst 60% frob" in the user's notation. Deterministic channels
    # (muls_adds_fmas, max_io_sum, flops, bytes_accessed) are computed
    # exactly once and reused.
    num_data_points: int = 5
    reps_per_point: int = 4
    percentile_keep: float = 0.60
    # Per-exec slow-order cutoff (seconds). During early training the
    # policy samples catastrophic elimination orders whose approx-
    # Jacobian execution is ~10-40× a good order's (the 500× FLOP
    # blowup, profiled 2026-06-05); running all `num_data_points *
    # reps_per_point` measurements on one costs ~15 min/terminal-step.
    # We don't need 20 reps to learn an order is slow — one execution
    # does. If any single measured exec exceeds this threshold, the
    # order is pathological: we keep the samples gathered so far and
    # stop. A good order's exec (~4.5s) never trips it and gets the full
    # 5×4 pool; a bad order (~44s) trips on the first exec and is capped
    # at one. 0 disables the cutoff (measure everything). Chosen just
    # above the good-order exec regime (~4.5s) and well below the
    # moderate/bad regimes (~12s / ~44-82s) so anything non-cheap caps
    # at its first sample while good orders keep the full 5×4 pool.
    slow_exec_cutoff_seconds: float = 8.0
    # FLOP-gate: XLA cost_analysis gives each order's FLOP count for FREE
    # (no execution). Pathological elimination orders (the 500× blowup
    # common early in training) have huge FLOP counts AND a ~50s/exec
    # latency on CPU — and that 50s exec can't be interrupted once
    # started, so the slow-exec cutoff can't save it. When flops exceed
    # this threshold we SKIP the expensive measurement entirely and
    # assign FLOP/bytes-derived cost surrogates + a worst-case quality
    # penalty, so the policy is pushed away from the order without ever
    # paying the exec. 0 disables. Profiled 2026-06-06.
    flop_gate_threshold: float = 0.0
    # Skip the expensive jacve-compile/exec branch on every step EXCEPT the
    # terminal one. Tokens/eqn_ids are still produced (the agent needs them as
    # the next observation), but the reward vector is zero on intermediate
    # steps and fully populated only when the order is complete. This is the
    # paper-native form for AlphaZero / GDPO / GFlowNet and works fine for PPO
    # / MuZero (just yields a sparse reward signal).
    terminal_rewards_only: bool = False
    # --- Latency-measurement noise control (2026-06-10) ---------------------
    # Timing is done with ``time.perf_counter`` around a tight inner loop of
    # ``latency_inner_reps`` back-to-back executions with a single
    # ``block_until_ready`` barrier, divided by the rep count. This amortizes
    # per-call dispatch/barrier overhead (the dominant noise for sub-ms
    # kernels) and replaces the old ResourceMonitor wall-timer, whose
    # ``stop()`` fired BEFORE the closing effects-barrier drained the async
    # device queue (systematic under-measure + a fake "0 ns" reading on
    # failure). ``latency_warmup`` discards the first K executions per data
    # point (first-touch / cache warm-up). ResourceMonitor is still used for
    # the peak-memory channel only. See the latency-noise investigation.
    latency_inner_reps: int = 1
    latency_warmup: int = 0
    # Aggregation of the latency pool: ``latency_winsor > 0`` uses a symmetric
    # winsorized mean (clamp the lowest/highest ``frac`` of samples, then
    # average) — empirically the most reproducible + discriminative estimator
    # (winsor-20% ≈ +80% discriminability vs the P60 percentile). 0.0 keeps
    # the legacy ``percentile_keep`` percentile path. Non-positive latency
    # readings (failed measurements) are dropped before aggregation; if none
    # survive the channel is marked sentinel so it is filtered downstream.
    latency_winsor: float = 0.0
    # Measurement target. ``False`` (default): measure the full JACOBIAN of
    # ``target_fun`` (approx via ``graphax.jacve`` with the policy's order +
    # micro-action transforms, exact via ``jax.jacrev``). ``True``: measure the
    # GRADIENT of a SCALAR-output ``target_fun`` (a training loss) — approx via
    # ``graphax.value_and_grad`` (rides jacve's has_aux path, returns
    # ``(value, grads)``), exact via ``jax.value_and_grad`` — and the quality
    # channels compare the approx gradient vs the exact gradient. This matches
    # how the elimination plan would actually be used in a training step (the
    # gradient that hits the optimizer), so it's the more faithful accuracy
    # signal. The caller MUST pass a scalar-output ``target_fun`` + matching
    # scalar ``jaxpr`` when this is set (see the workers' grad-mode wrapping).
    measure_grad: bool = False
    # Latency timer. ``"perf_counter"`` (default): time a tight inner loop with
    # ``time.perf_counter`` + one closing ``block_until_ready`` (peak memory via
    # a separate ResourceMonitor pass). ``"rm"``: time via the (fixed)
    # ``ResourceMonitor`` — ``effects_barrier`` now drains BEFORE ``stop()`` and
    # we ``block_until_ready`` INSIDE the context, so the duration is the true
    # device time, and peak memory comes from the SAME pass (one execution for
    # both channels instead of two). Validated RM≈perf_counter (ratio 0.95).
    latency_timer: str = "perf_counter"
    # Search-space simplification: when True, only the FIRST Quant emitted in an
    # episode takes effect (later Quant ops dropped) — one global quantization
    # choice (a single dtype, or none) instead of per-vertex repeated quant.
    quant_once: bool = False


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
        # :data:`QUANT_DTYPES`. The third column is unused for QUANT (kept
        # at 0). ``_callback`` emits ``Quant(dtype=QUANT_DTYPES[bi2])`` and
        # graphax's apply_quant casts ``val`` only.
        quant_used = active[s_idx] & is_quant[s_idx]
        quant_bi1 = jnp.asarray(QUANT_SENTINEL, dtype=jnp.int32)
        quant_bi2 = quant_dtypes[s_idx].astype(jnp.int32)

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
            quant_used, jnp.asarray(0, dtype=jnp.int32),
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
    comparison. graphax ``value_and_grad``/``jacve`` returns some weight grads in
    the transposed (∂L/∂Wᵀ) layout for certain elimination orders; flattening
    them as-is makes a (256,784) vs (784,256) ravel near-orthogonal, so the
    cosine/frob become a layout ARTIFACT that badly underestimates true gradient
    quality (empirically lifts Spearman-vs-trainability 0.67→0.79). Transpose a
    2-D leaf back when its shape is the exact leaf's reverse; leave other
    mismatches for the size/shape guard downstream."""
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

    Returns the trivial `(1.0, 0.0)` (perfect agreement) when either side has
    no leaves, mismatched shapes, or zero size, mirroring the original `error`
    fallback so a degenerate plan can't poison downstream normalisation.

    ``ALPHAGRAD_DEBUG_QUALITY=1`` enables a one-line diagnostic print
    when cosine_sim collapses to ~0 with non-zero norms — used to
    investigate the persistent ``reward_mean/cosine_sim=0`` we see
    on the PPO dynamic-substeps path. The print fires only when the
    formula would have produced a meaningful value but didn't.
    """
    # Layout-align the approx leaves to the exact layout (transpose-back) so the
    # cosine/frob compare the SAME entries, not a transposed-layout artifact.
    jac_approx = _align_jac(jac_approx, jac_exact)
    flat_exact = _flatten_jacobians(jac_exact)
    flat_approx = _flatten_jacobians(jac_approx)
    # A degenerate / incomparable approx Jacobian (no leaves, mismatched shape,
    # or zero size) is a FAILED approximation, NOT a perfect one. Returning
    # (cos=1, frob=0) "perfect" lets an over-compressed plan whose Jacobian
    # collapsed to a different shape (compresses drop axes) score perfect
    # quality and — combined with its ~0 latency/memory — Pareto-dominate every
    # real solution, emptying the archive. Score it worst-case (cos=0, frob=1).
    if flat_exact is None or flat_approx is None:
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    if flat_approx.shape != flat_exact.shape or flat_approx.size == 0:
        if os.environ.get("ALPHAGRAD_DEBUG_QUALITY", "0") == "1":
            _ea = jax.tree_util.tree_leaves(jac_exact)
            _aa = jax.tree_util.tree_leaves(jac_approx)
            print(
                "[quality-debug] SHAPE-MISMATCH -> cos=0 | "
                f"exact_leaves={[tuple(jnp.shape(x)) for x in _ea]} "
                f"approx_leaves(aligned)={[tuple(jnp.shape(x)) for x in _aa]} "
                f"flat_exact={None if flat_exact is None else flat_exact.shape} "
                f"flat_approx={None if flat_approx is None else flat_approx.shape}",
                flush=True,
            )
        return jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)

    cos = cossim(flat_exact, flat_approx)
    # Some approximations (certain quant/compress combos) yield a complex-valued
    # flattened Jacobian, making cossim complex. Use the real part — it matches the
    # reward path's existing real cast (the source of the ComplexWarning) and unblocks
    # the per-point float emission in raw_sink that otherwise crashes on complex.
    cos = jnp.real(cos)
    exact_norm = jnp.linalg.norm(flat_exact)
    resid_norm = jnp.linalg.norm(flat_exact - flat_approx)
    rel_frob = resid_norm / jnp.maximum(exact_norm, jnp.sqrt(1e-7))
    # A NaN/inf-producing approximation (e.g. quant overflow -> non-finite
    # Jacobian) is a FAILED approximation, not a missing measurement. Clamp to
    # worst-case FINITE quality (cos=0, frob=1) so a single bad config can't
    # poison reward-normalisation (symlog), GAE, or the Lagrangian dual-ascent
    # downstream (which has no NaN guard and would otherwise latch lambda=NaN).
    cos = jnp.where(jnp.isfinite(cos), cos, jnp.float32(0.0))
    rel_frob = jnp.where(jnp.isfinite(rel_frob), rel_frob, jnp.float32(1.0))

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


# ---------------------------------------------------------------------------
# Signal B_kstep — closed-loop TRAINABILITY probe (the "acc" reward channel
# when ``ALPHAGRAD_ACC_PROXY=bkstep``). Runs a short REAL MNIST training using
# the rule's OWN approximate gradient (the ``measure_grad`` compiled fn), then
# reads the resulting MNIST test accuracy. Ported from the qsig_B_kstep
# bake-off (sigB) so it runs inline inside ``_callback`` — self-contained here
# (no ``qsig_common`` import, which would be circular: qsig_common imports env).
# ---------------------------------------------------------------------------
_BKSTEP_MNIST: dict = {}
# Monotonic per-process probe-call counter. Each _bkstep_probe call is one
# terminal-step trainability measurement (~one episode's acc reward); mixing
# this ordinal into the DATA rng rotates the minibatches + eval subset per
# episode (fixing the old fixed-seed bug where every episode saw identical
# data), while the init-seed averaging stays for variance reduction. List so
# it can be rebound inside the function.
_BKSTEP_EP_COUNTER: list = [0]


def _bkstep_mnist():
    """Lazily load + cache MNIST (train/test) tensors for the B_kstep probe.

    Cheap after first call (module-level cache). Kept in float32 device arrays.
    """
    if not _BKSTEP_MNIST:
        from alphagrad.approx.common.datasets import load_dataset
        xtr, ytr = load_dataset("mnist", None, "train")
        xte, yte = load_dataset("mnist", None, "test")
        _BKSTEP_MNIST.update(
            xtr=jnp.asarray(xtr), ytr=jnp.asarray(ytr),
            xte=jnp.asarray(xte), yte=jnp.asarray(yte),
        )
    return (
        _BKSTEP_MNIST["xtr"], _BKSTEP_MNIST["ytr"],
        _BKSTEP_MNIST["xte"], _BKSTEP_MNIST["yte"],
    )


@jax.jit
def _bkstep_predict(x, W1, b1, W2, b2):
    # Mirror examples._neural_network's forward pass (tanh-tanh MLP).
    a1 = jnp.tanh(x @ W1.T + b1)
    return jnp.tanh(a1 @ W2.T + b2)


def _bkstep_accuracy(W, x, y):
    preds = []
    for i in range(0, x.shape[0], 2000):
        preds.append(jnp.argmax(_bkstep_predict(x[i:i + 2000], *W), -1))
    return float(jnp.mean(jnp.concatenate(preds) == jnp.argmax(y, -1)))


def _bkstep_align_grads(ga, ge):
    """Layout-align each approx grad leaf to its exact-weight leaf.

    graphax's reverse pass can emit a W-grad transposed vs the weight layout
    (the known W1/W2 grad-transpose artifact). Transpose back so the Adam
    update lands correctly; zero-out an un-alignable leaf so a broken shape
    can't crash the training loop (it just contributes no update that step).
    """
    out = []
    for a, e in zip(ga, ge):
        a = jnp.asarray(a)
        if a.shape == e.shape:
            out.append(a)
        elif a.ndim == 2 and a.shape == e.shape[::-1]:
            out.append(a.T)
        else:
            out.append(jnp.zeros_like(e))
    return tuple(out)


@jax.jit
def _bkstep_eval_mse(x, y, W1, b1, W2, b2):
    """True mean-squared-error eval loss with the exact tanh-tanh forward.

    Mirrors examples._neural_network's per-element 0.5*(pred-y)**2, meaned —
    the honest trainability loss, computed on a held-out eval batch with the
    real (non-approx) forward so the signal reflects what Adam actually did to
    the weights, not the noisy per-batch approx-path value.
    """
    a1 = jnp.tanh(x @ W1.T + b1)
    pred = jnp.tanh(a1 @ W2.T + b2)
    return jnp.mean(0.5 * (pred - y) ** 2)


def _bkstep_eval_loss(W, x, y):
    """Batched eval MSE over a (possibly large) eval subset."""
    tot, n = 0.0, 0
    for i in range(0, x.shape[0], 2000):
        xb, yb = x[i:i + 2000], y[i:i + 2000]
        tot += float(_bkstep_eval_mse(xb, yb, *W)) * xb.shape[0]
        n += xb.shape[0]
    return tot / max(n, 1)


def _bkstep_probe(approx_grad_fn, weights, argnums, n_steps=40, seeds=(0, 1),
                  ep=None):
    """K-step closed-loop MNIST trainability signal in [0, 1], higher=better.

    ``approx_grad_fn(x, y, *weights) -> (loss_value, approx_grads)`` is the
    policy's compiled ``measure_grad`` fn. For each seed: re-init the 2-layer
    MLP weights, run ``n_steps`` Adam updates on random MNIST minibatches using
    the APPROX gradient, then read a CONTINUOUS trainability signal derived from
    the true (exact-forward) eval MSE-loss trajectory. Returns the mean over
    seeds. Only the weight args (``argnums``) are updated; x/y are fed per step.

    Continuous signal (edge-resolved, replaces the old thresholded argmax-acc so
    the cliff at cos≈0.02-0.25 has a usable gradient). Selected by
    ``ALPHAGRAD_BKSTEP_SIGNAL``:

      * ``fracred`` (default): 1 - L_final/L_init, clamped [0,1] — fractional
        eval-loss reduction. Degenerate (no descent / divergence) -> ~0, full
        descent -> ~1. Smooth across the cliff because eval-loss keeps moving
        even where argmax-accuracy is pinned.
      * ``auc``: mean over the recorded trajectory of (1 - L_t/L_init), clamped
        [0,1] — rewards fast AND sustained descent (integral of the loss-drop
        curve), sharper early-step resolution.
      * ``expneg``: exp(-L_final / L_init) mapped so no-descent->~1/e, blowup->0
        (relative, unit-free).
      * ``invloss``: 1/(1 + L_final) — absolute, saturates once loss is small.

    The training minibatches AND the eval subset rotate per episode (``ep``,
    default = monotonic probe-call counter); the init-seed loop re-inits weights
    from ``seeds`` (variance reduction). Reproducible given ``(ep, seed)``.
    """
    import optax
    if ep is None:
        _BKSTEP_EP_COUNTER[0] += 1
        ep = _BKSTEP_EP_COUNTER[0]
    ep = int(ep)
    signal = os.environ.get("ALPHAGRAD_BKSTEP_SIGNAL", "fracred").strip().lower()
    xtr, ytr, xte, yte = _bkstep_mnist()
    # Weight leaves in argnums order = the approx-grad pytree order.
    w0 = [jnp.asarray(weights[a]) for a in argnums]
    n_tr = int(xtr.shape[0])
    n_te = int(xte.shape[0])
    # Per-episode eval subset (rotates with ``ep``).
    eval_rng = np.random.default_rng([ep, 0xE7A1])
    n_eval = min(5000, n_te)
    eval_idx = eval_rng.choice(n_te, size=n_eval, replace=False)
    xte_eval, yte_eval = xte[eval_idx], yte[eval_idx]
    # How often to sample the eval-loss trajectory for AUC (init + every k).
    _rec_every = max(1, int(os.environ.get("ALPHAGRAD_BKSTEP_REC_EVERY", "4")))
    _eps = 1e-8
    sigs = []
    for s in seeds:
        keys = jax.random.split(jax.random.PRNGKey(int(s)), len(w0))
        W = []
        for k, w in zip(keys, w0):
            if w.ndim == 2:
                fan_in = w.shape[1]
                W.append(jax.random.normal(k, w.shape) / jnp.sqrt(fan_in))
            else:
                W.append(jnp.zeros_like(w))
        W = tuple(W)
        opt = optax.adam(1e-3)
        ost = opt.init(W)
        rng = np.random.default_rng([ep, int(s)])
        # Init eval loss (before any update) — the trajectory baseline.
        L0 = _bkstep_eval_loss(W, xte_eval, yte_eval)
        traj = [L0]  # eval loss recorded at init + every _rec_every steps
        for _t in range(n_steps):
            idx = rng.integers(0, n_tr, 16)
            xb, yb = xtr[idx], ytr[idx]
            _val, ga = approx_grad_fn(xb, yb, *W)
            ga = _bkstep_align_grads(ga, W)
            upd, ost = opt.update(ga, ost, W)
            W = optax.apply_updates(W, upd)
            if (_t + 1) % _rec_every == 0:
                traj.append(_bkstep_eval_loss(W, xte_eval, yte_eval))
        Lf = traj[-1]
        L0s = max(L0, _eps)
        if signal == "auc":
            # Mean fractional loss-drop over the recorded trajectory (skip the
            # init point which is 0 by construction) -> integral of descent.
            drops = [1.0 - (Lt / L0s) for Lt in traj[1:]] or [0.0]
            sig = float(np.clip(np.mean(drops), 0.0, 1.0))
        elif signal == "expneg":
            sig = float(np.exp(-(Lf / L0s)))
        elif signal == "invloss":
            sig = float(1.0 / (1.0 + Lf))
        else:  # fracred (default)
            sig = float(np.clip(1.0 - (Lf / L0s), 0.0, 1.0))
        sigs.append(sig)
    return float(np.mean(sigs))


def _aggregate_samples(values, want_top_quartile: bool):
    """Reduce a list of per-sample scalars to a single jnp scalar.

    With ≥8 samples and `want_top_quartile`, takes the top-quartile mean
    (matching legacy behaviour for latency); otherwise falls back to a plain
    mean. Handles the empty-list case by returning `0.0`.
    """
    if not values:
        return jnp.array(0.0, dtype=jnp.float32)
    stack = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in values])
    if want_top_quartile and stack.shape[0] >= 8:
        return stack.sort()[6:8].mean()
    return stack.mean()


def _percentile_pool(values, q: float) -> float:
    """Return the `q`-percentile of a pool of measurements (q in [0, 1]).

    Used for the noisy-channel aggregation under the 5×4 design:
    `q=0.60` → the slowest 60% latency / highest 60% memory / worst
    60% frob, in the user's notation. Falls back to 0.0 on an empty
    pool.
    """
    if not values:
        return 0.0
    arr = jnp.asarray(values, dtype=jnp.float32)
    return float(jnp.percentile(arr, float(q) * 100.0))


def _winsorized_mean(values, frac: float) -> float:
    """Symmetric winsorized mean: clamp the lowest/highest ``frac`` fraction
    of samples to the corresponding quantiles, then average.

    Empirically the most reproducible + discriminative latency aggregator on
    this measurement harness (≈ +80% discriminability vs the P60 percentile,
    and far better than the noisy minimum). ``frac`` in [0, 0.5). Falls back
    to the plain mean for tiny pools where trimming would remove everything.
    """
    if not values:
        return 0.0
    a = np.sort(np.asarray(values, dtype=np.float64))
    n = a.size
    k = int(n * float(frac))
    if k > 0 and n - 2 * k >= 1:
        a = a.copy()
        a[:k] = a[k]
        a[n - k:] = a[n - k - 1]
    return float(a.mean())


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


def _callback(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_specs,
    stop,
    *eval_samples,
    init: bool = False,
    point_idx: int = -1,
    raw_sink: dict | None = None,
):
    """Stage A reward harness: returns `(tokens, rewards)` where `rewards` is
    the canonical `(NUM_REWARDS,)` float32 vector documented at the top of this
    file. Every component is computed every (non-init) call, except `latency`
    which is gated behind `config.measure_latency`. When
    `config.terminal_rewards_only` is on, intermediate steps return tokens
    only — every reward component is zeroed so the heavy jacve compile/exec
    is skipped entirely until the elimination order is complete.
    """
    partial_order, partial_specs = _get_partials(order, sparsity_specs, stop)
    is_terminal = int(stop) >= len(order)

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()  # list of MAX_RULES x 3 lists

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
    last_v_idx = len(o_list) - 1
    # quant-once: only the FIRST Quant across the whole episode (vertex x slot
    # order) survives; later Quant rows are dropped → one global quantization
    # choice (a dtype, or none) instead of per-vertex repeated quant.
    _quant_once = bool(getattr(config, "quant_once", False))
    _quant_used = False
    for v_idx, v in enumerate(o_list):
        eqn = config.jaxpr.eqns[v - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_shape = eqn.outvars[0].aval.shape
        out_len = len(out_shape)
        primal_shapes = [
            iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")
        ]
        if not primal_shapes:
            continue  # no non-literal inputs → no edges to transform

        rules: list = []  # mixed list[Diag | Compress | Quant]
        used_axes: set[int] = set()
        # FULL-REDUCTION COMPRESS CAP. A COMPRESS removes one physical
        # axis from the edge val (ndim == out_len + primal_dims). When the
        # LAST physical axis is dropped graphax canonicalizes the edge to
        # val=None (uniform grid, see micro_actions.apply_compress 69e55fb)
        # -- a genuinely-degenerate ~zero-mean tensor that both
        # cossim-collapses the reward AND crashes the measure path device
        # tracker (NoneType not subscriptable). Forbid the final-axis
        # COMPRESS so >=1 physical axis always remains; partial COMPRESS
        # (leaving >=1 axis) stays allowed.
        _edge_phys_axes = out_len + max(
            (len(ps) for ps in primal_shapes), default=0
        )
        _n_compressed = 0
        for slot in range(MAX_RULES_PER_VERTEX):
            row = specs_list[v_idx][slot]
            bi1 = int(row[0])
            bi2 = int(row[1])
            factor = int(row[2])
            if bi1 == -1:
                break  # end-of-sequence sentinel
            if bi1 == QUANT_SENTINEL:
                # QUANT slot: row[1] is the dtype index into QUANT_DTYPES,
                # row[2] is unused. Unlike Compress, Quant doesn't touch
                # axes or `val.ndim`, so it's safe on any vertex (no
                # shape-preservation issue with downstream eliminations).
                # Out-of-range dtype indices silently fall back to the
                # first catalog entry rather than crashing the callback.
                if _quant_once and _quant_used:
                    continue  # quant-once: a quantization was already chosen
                dtype_idx = bi2
                if not (0 <= dtype_idx < len(QUANT_DTYPES)):
                    dtype_idx = 0
                rules.append(Quant(dtype=QUANT_DTYPES[dtype_idx]))
                _quant_used = True
                continue
            if bi1 == COMPRESS_SENTINEL:
                # COMPRESS slot: row[1] is the *physical* axis index in the
                # SparseTensor edge (same layout as Diag's idx: out axes 0..
                # out_len-1, primal axes out_len..out_len+primal_dims-1).
                # row[2] is the kind index into COMPRESS_KINDS. Skip the
                # slot if the axis index doesn't fit every invar's edge —
                # graphax's apply_compress will validate too, but raising
                # would crash the io_callback.
                #
                # NON-TERMINAL COMPRESS IS ALLOWED. graphax core-v2's
                # implicit-dim algebra (produce_compress.py) propagates a
                # compressed (axis=None) edge through downstream
                # ``_eliminate_vertex`` steps without materializing the
                # dropped axis, and core.py's per-edge try/except skips any
                # genuinely-mis-fitting edge (densifying it back to nominal
                # form). The FULL-REDUCTION CAP below is still enforced.
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
                # CAP: never drop the last remaining physical axis -- that
                # canonicalizes the edge to val=None (uniform/degenerate).
                if _n_compressed + 1 >= _edge_phys_axes:
                    continue
                if not (0 <= kind_idx < len(COMPRESS_KINDS)):
                    # Unknown kind — fall back to the default "mean" rather
                    # than dropping the row, since the axis-removal effect is
                    # the dominant signal.
                    kind_idx = 0
                used_axes.add(axis_idx)
                _n_compressed += 1
                rules.append(
                    Compress(axes=(axis_idx,), kind=COMPRESS_KINDS[kind_idx])
                )
                continue
            if bi1 < 0:
                # Any other negative bi1 is reserved for future sentinels;
                # skip without aborting the sequence so a new sentinel
                # introduced upstream doesn't silently break older specs.
                continue
            idx1 = bi1            # logical output-side axis
            idx2 = out_len + bi2  # logical primal-side axis
            # Skip rules that would reuse an axis (graphax expected bipartite
            # disjoint pairs; the legacy translator filtered them, do it here
            # so apply_diag's stricter checks don't crash).
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
            # apply_diag requires factor | gcd(n_i, n_j) on every edge it
            # touches. Silently skip mismatches rather than crash.
            if (
                factor <= 0
                or n1 % factor != 0
                or any(n2 % factor != 0 for n2 in n2_list)
            ):
                continue
            used_axes.add(idx1)
            used_axes.add(idx2)
            rules.append(Diag(i=idx1, j=idx2, factor=factor))
        if rules:
            transforms.append((int(v), tuple(rules)))

    # ------------------------------------------------------------------
    # PREVALIDATE-BEFORE-MEASURE (opt-in via ALPHAGRAD_PREVALIDATE_MEASURE=1)
    # ------------------------------------------------------------------
    # graphax's typed micro-action transforms (DIAG/COMPRESS) can produce
    # logically-misaligned sparse edges that only fail deep inside the host
    # shape algebra (``_eliminate_vertex`` -> sparse matmul / ``_normalize_
    # approx_edge``): an edge-shape AssertionError, a matmul "Contraction
    # size mismatch" ValueError, a broadcast/transpose TypeError. For ViT
    # (seed-free Jacobian) EVERY terminal order in a rollout tripped one of
    # these, so the reward was all-[SENTINEL] and PPO learned nothing.
    #
    # The fix runs a HOST-SIDE dry-run BEFORE committing to the order's
    # measurement, reusing graphax's OWN shape algebra as the oracle (no
    # re-implementation, drift-proof): ``vertex_elimination_jaxpr(...,
    # count_ops=True)`` walks the identical ``_eliminate_vertex`` ->
    # matmul-topology -> ``_normalize_approx_edge`` path on the SparseTensor
    # dim/logical_size metadata and raises the EXACT same error classes — it
    # does NOT need ``extract_jaxpr``'s output (they are sibling calls that
    # both build their graph from ``config.jaxpr``/``args``/``consts``), so
    # it is a faithful, cheap probe we can run first.
    #
    # On a clean dry-run we proceed with the sampled transforms unchanged.
    # On failure we apply a SAFE-SUBSET prune ladder, re-dry-running after
    # each step until it traces clean, and measure with the FIRST clean
    # subset:
    #   (a) drop all COMPRESS rows,
    #   (b) then also drop DIAG rows on every non-terminal vertex,
    #   (c) then transforms=[] (plain exact elimination — a provably valid
    #       floor that always passes).
    #
    # Flag OFF (unset / != "1"): this whole block is skipped, so the path
    # is byte-identical to the prior behaviour.
    _prevalidate = os.environ.get("ALPHAGRAD_PREVALIDATE_MEASURE", "0") == "1"
    # The REALIZED per-vertex specs after pruning, mirroring the input
    # ``specs_list`` row layout (-1 in col 0 = unused slot). Off-policy
    # bookkeeping consumes this; see the worker return-path note. When the
    # flag is off (or no pruning happens) it stays None (no behaviour change).
    realized_specs = None
    # GATE: prevalidate runs the costly count_ops dry-run + prune-ladder
    # ONLY on the terminal step. Non-terminal steps return early (below)
    # without ever measuring, so the dry-run there was 100% waste
    # (~1.9s/call on the GPU host). is_terminal is defined above (~L1179).
    if _prevalidate and transforms and is_terminal:
        # Terminal vertex of THIS partial order — the only vertex with no
        # downstream elimination within this callback (COMPRESS is already
        # restricted to it above; DIAG on it is the safest to keep).
        _terminal_vid = int(o_list[-1]) if o_list else None

        def _probe(_t):
            # Faithful oracle: the count_ops shape-pass raises the SAME
            # AssertionError / ValueError / TypeError classes that would
            # otherwise surface inside ``extract_jaxpr`` at measure time.
            vertex_elimination_jaxpr(
                config.jaxpr,
                o_list,
                consts,
                *args,
                argnums=config.argnums,
                count_ops=True,
                sparse_representation=config.sparse,
                transforms=_t,
            )

        def _drop_compress(_t):
            out = []
            for _v, _rs in _t:
                _kept = tuple(r for r in _rs if not isinstance(r, Compress))
                if _kept:
                    out.append((_v, _kept))
            return out

        def _terminal_only(_t):
            # Keep transforms ONLY on the terminal vertex (drops every
            # non-terminal DIAG/COMPRESS row).
            return [
                (_v, _rs) for _v, _rs in _t if _v == _terminal_vid
            ]

        _ladder = [
            ("compress", _drop_compress(transforms)),
            ("diag", _terminal_only(_drop_compress(transforms))),
            ("empty", []),
        ]
        _pruned_level = None
        try:
            _probe(transforms)
        except (AssertionError, ValueError, TypeError):
            _n_before = sum(len(_rs) for _, _rs in transforms)
            for _lvl, _cand in _ladder:
                try:
                    if _cand:
                        _probe(_cand)
                    transforms = _cand
                    _pruned_level = _lvl
                    break
                except (AssertionError, ValueError, TypeError):
                    continue
            else:
                transforms = []
                _pruned_level = "empty"
            _n_after = sum(len(_rs) for _, _rs in transforms)
            print(
                f"[PREVALIDATE] pruned {_n_before - _n_after} transforms "
                f"(lvl={_pruned_level})",
                flush=True,
            )
            # --- Off-policy bookkeeping: rebuild the REALIZED specs ---------
            # The executed action (pruned transforms) != the sampled action.
            # Reconstruct the per-vertex specs that correspond to the pruned
            # transforms so the reward can be attributed to the action the
            # env actually measured. NOTE: the Ray io_callback return
            # signature is fixed to (tokens, eqn_ids, reward); threading
            # ``realized_specs`` back into the PPO rollout buffer is a
            # separate, bounded follow-up (see report). We compute it here so
            # the plumbing has a single, correct source of truth to consume.
            _kept_vids = {int(_v) for _v, _ in transforms}
            realized_specs = np.array(specs_list, dtype=np.int32).copy()
            for _vi, _vid in enumerate(o_list):
                if int(_vid) not in _kept_vids:
                    realized_specs[_vi, :, 0] = -1  # mark all slots unused

    # STATE-TOKENIZER integration (GRAPHAX_STATE_TOKENS=1, palimpsapprox-
    # statetok branch): graphax's gated state-tokenizer path keys off the
    # ``o_list`` (elimination order) + ``transforms`` (per-vertex DIAG/
    # COMPRESS/QUANT micro-actions) we already pass here, emitting an
    # append-only, ~order-of-magnitude-shorter token stream
    #     <original-graph tokens> | <order ; per-vertex micro-actions>
    # instead of re-tracing the fused Jacobian every step. That stream is a
    # lossless sufficient statistic for the APPROXIMATED policy STATE (it is
    # only the policy's state encoding -- the measured computation below still
    # runs the untouched jacve/AD path). When the env var is unset this call
    # is byte-identical to the legacy re-traced tokenizer. No change to the
    # call itself is needed: the gate lives in graphax.extract_jaxpr.
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
    _record_tokenization_truncation(int(raw_tokens.shape[0]))
    tokens = raw_tokens[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    # Compute per-token equation IDs once for the relational-bias encoder
    # (Stage B.1). Cheap (single Python scan over a length-≤4096 numpy array)
    # and adds (MAX_TOKENS,) int32 to EnvState.
    tokens_np = np.asarray(tokens)
    eqn_ids_np = compute_eqn_ids_from_tokens(tokens_np, _TOKEN_VOCAB)
    eqn_ids = jnp.asarray(eqn_ids_np, dtype=jnp.int32)

    if init:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # Terminal-only fast path: every reward component is sparse — only the
    # final step (when the elimination order is complete) gets a non-zero
    # signal, so we skip the expensive jacve compile/exec on every prior
    # step. Cumsum-style returns (alpha0/mu0) and per-rollout aggregations
    # (gdpo) collapse to the terminal reward; gfn already reads only the
    # last step. PPO sees a sparse-reward MDP, which GAE handles natively.
    if config.terminal_rewards_only and not is_terminal:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # GATE 1 (pre-compile size gate): if the eliminated jaxpr is too big, skip
    # the compile+exec entirely — they are the OOM/host-leak/timeout-prone part
    # that was OOM-killing the measure actor. raw_tokens.shape[0] (= the
    # tokenized jacve graph length) is the graph-size proxy. We RAISE so the
    # worker's existing handler turns it into a logged [SENTINEL] (excluded from
    # the loss) instead of attempting a measurement that could kill the actor.
    _max_measure_tokens = int(os.environ.get("ALPHAGRAD_MAX_MEASURE_TOKENS", "0") or 0)
    if _max_measure_tokens > 0 and int(raw_tokens.shape[0]) > _max_measure_tokens:
        raise RuntimeError(
            f"size-gate: jaxpr raw_len={int(raw_tokens.shape[0])} > "
            f"ALPHAGRAD_MAX_MEASURE_TOKENS={_max_measure_tokens} — skip measure"
        )

    # ------------------------------------------------------------------
    # Compute family — graphax counters (always) → muls_adds_fmas, max_io_sum.
    # ------------------------------------------------------------------
    _, aux = vertex_elimination_jaxpr(
        config.jaxpr,
        o_list,
        consts,
        *args,
        argnums=config.argnums,
        count_ops=True,
        sparse_representation=config.sparse,
        transforms=transforms,
    )
    muls_adds_fmas = float(aux["adds"] + aux["muls"] + aux["fmas"])
    max_io_sum = float(aux["mem"])

    # If no `target_fun` is supplied, we can't compile/execute. Skip every
    # execution-derived metric and return a partial reward vector.
    #
    # cossim=0.0 (NOT 1.0 as in earlier revisions): the early-return path
    # signifies "this channel is unmeasured", not "perfect fidelity". A 1.0
    # value would (a) tank the scalar reward through the cossim weight,
    # (b) falsely trigger anti-degeneracy ``cossim<=1-δ`` constraints
    # whose entire point is to prevent the policy collapsing to cossim=1.
    # 0.0 leaves the policy under no quality pressure during the cheap
    # phase, which is what 2-phase schedules (--cost-pipeline-schedule
    # cheap_first) want — measurement comes online at phase cutover.
    if config.target_fun is None:
        rewards = jnp.array(
            [
                -muls_adds_fmas, 0.0, 0.0, -max_io_sum,
                0.0, 0.0, 0.0, 0.0,
                0.0,  # xla_peak_memory (no compiled fn on the cheap path)
                0.0,  # bkstep_acc (no compiled fn on the cheap path)
            ],
            dtype=jnp.float32,
        )
        return tokens, eqn_ids, rewards

    # ------------------------------------------------------------------
    # Compile both the approximated and exact jacobian functions once.
    # ------------------------------------------------------------------
    # ``--exec-on-gpu`` pins the reward harness to a GPU distinct from the
    # one the trainer (the main process) is loaded on — otherwise the
    # callback's compile/exec would deadlock against the main program's
    # outstanding work on gpu[0]. Pick the *last* available GPU so we
    # stay as far from the trainer as possible; requires ≥ 2 GPUs.
    callback_device = None
    if config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) < 1:
            raise RuntimeError(
                "--exec-on-gpu needs a GPU visible to the measurement process; "
                f"got {len(gpu_devices)}."
            )
        # The measurement runs on the LAST visible GPU. In the legacy
        # single-process layout that was a 2nd GPU distinct from the trainer's;
        # in the separate-measure-actor layout (cpu_actor_num_gpus=1) the actor
        # owns its own 1 GPU (Ray gives it a device disjoint from the trainer),
        # so a single visible GPU is correct.
        callback_device = gpu_devices[-1]

    # --- Data-parallel sharding of the on-GPU measurement (opt-in) -----------
    # ``ALPHAGRAD_SHARD_MEASURE=1`` shards the batched (vmap) axis of the
    # measurement's grad/jacobian exec across ALL of the measure actor's
    # visible GPUs, so a model whose single-device peak exceeds one GPU (e.g.
    # the 16-substep VmappedConvNet) splits ~1/N memory per GPU instead of
    # OOM-ing GPU_0 while the others sit idle. The Vmapped* models are vmapped
    # over the data batch with ``in_axes=(0, 0, None, ...)`` — the DATA args
    # (the non-``argnums`` positional leaves x, y) carry the batch axis on
    # axis 0; the WEIGHTS (``argnums``) are replicated. We build a 1-D mesh
    # over ``gpu_devices[:N]``, ``device_put`` the batched data leaves with
    # ``NamedSharding(P('batch'))`` and the rest replicated, and let XLA's
    # GSPMD partitioner shard the computation (in/out shardings are taken from
    # the committed input arrays). OFF (default) is byte-for-byte the legacy
    # single-device path. Falls back to single-device when <2 GPUs, batch size
    # is unknown, or the batch is not divisible by N (so a wrong/uneven shard
    # can never silently corrupt the reward).
    _shard_measure = (
        os.environ.get("ALPHAGRAD_SHARD_MEASURE", "0") == "1"
        and config.exec_on_gpu
        and callback_device is not None
        and len(gpu_devices) > 1
    )
    _shard_mesh = None
    _shard_devices = None
    if _shard_measure:
        import numpy as _np_shard
        from jax.sharding import (
            Mesh as _Mesh,
            NamedSharding as _NS,
            PartitionSpec as _PSpec,
        )
        _shard_devices = list(gpu_devices)
        _N = len(_shard_devices)
        # Batch size = leading dim of the batched DATA leaves (the positional
        # args NOT in argnums that are >=1-D). All such leaves must share the
        # same leading dim and be divisible by N; otherwise fall back.
        _argnums_set = set(config.argnums or ())
        _batch_dims = {
            int(a.shape[0])
            for i, a in enumerate(args)
            if i not in _argnums_set and getattr(a, "ndim", 0) >= 1
        }
        if len(_batch_dims) == 1 and next(iter(_batch_dims)) % _N == 0:
            _shard_mesh = _Mesh(_np_shard.array(_shard_devices), axis_names=("batch",))
        else:
            # Indivisible / ambiguous batch — stay single-device.
            _shard_measure = False
            _shard_mesh = None

    def _put_measure(leaves):
        """device_put a positional-arg list onto the measure device(s).

        When sharding is active: batched data leaves (non-argnums, >=1-D) get
        ``P('batch')`` (split on axis 0), everything else is replicated
        (``P()``). When off: the legacy single ``callback_device`` put."""
        if _shard_mesh is not None:
            from jax.sharding import NamedSharding as _NS, PartitionSpec as _PSpec
            _argnums_set = set(config.argnums or ())
            out = []
            for i, a in enumerate(leaves):
                if i not in _argnums_set and getattr(a, "ndim", 0) >= 1:
                    sh = _NS(_shard_mesh, _PSpec("batch"))
                else:
                    sh = _NS(_shard_mesh, _PSpec())
                out.append(jax.device_put(a, sh))
            return out
        if callback_device is not None:
            return [jax.device_put(a, callback_device) for a in leaves]
        return list(leaves)

    args_for_lower = (
        _put_measure(list(args))
        if (callback_device is not None or _shard_mesh is not None)
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
    h.update(int(stop).to_bytes(4, "little", signed=False))
    # Include the shape signature of args_for_lower so we don't
    # collide across rollouts that share (order, specs) but differ
    # in batch shape.
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            h.update(repr(a.shape).encode())
            h.update(repr(a.dtype).encode())
    if _shard_mesh is not None:
        # Sharded vs single-device produce DIFFERENT executables (different
        # in/out shardings + GSPMD partition); keep their cache entries
        # distinct so a single-device blob is never loaded for a sharded run.
        h.update(b"shard:" + str(len(_shard_devices)).encode())
    cache_key = h.digest()

    # Exact-Jacobian cache key — SHAPE-ONLY (no order/specs/stop). The
    # exact reference is order-invariant and computed via jax.jacrev, so
    # one compiled executable per (arg-shape, has_aux) is reused across
    # every terminal step of every episode — a permanent cache hit after
    # the first compile, instead of a per-episode recompile.
    he = hashlib.blake2b(digest_size=16)
    he.update(b"jacrev-exact")
    he.update(b"aux1" if config.has_aux else b"aux0")
    for a in args_for_lower:
        if hasattr(a, "shape") and hasattr(a, "dtype"):
            he.update(repr(a.shape).encode())
            he.update(repr(a.dtype).encode())
    exact_cache_key = he.digest()

    def _do_compile_approx():
        if config.measure_grad:
            # Gradient mode: measure ``graphax.value_and_grad`` of the scalar
            # loss along the policy's order + micro-action transforms. Returns
            # ``(value, grads)``; the order/transforms ride jacve internally.
            from graphax import value_and_grad as _gx_value_and_grad
            fn = _gx_value_and_grad(
                config.target_fun,
                list(o_list),
                argnums=config.argnums,
                transforms=transforms,
            )
        else:
            fn = jacve(
                config.target_fun,
                list(o_list),
                argnums=config.argnums,
                has_aux=config.has_aux,
                sparse_representation=config.sparse,
                transforms=transforms,
            )
        return jax.jit(fn, keep_unused=True).lower(*args_for_lower).compile()

    def _do_compile_exact():
        # The exact reference Jacobian is ORDER-INVARIANT: vertex
        # elimination yields the same Jacobian for any order, only the
        # FLOP cost differs. The previous implementation compiled
        # ``jacve(..., list(o_list))`` with the *policy's* terminal
        # order — and run dense (no transforms) that order is often a
        # catastrophic high-FLOP path during early training, costing
        # ~125s per terminal step (95% of episode wall-clock, profiled
        # via ALPHAGRAD_DBG_TIMING). ``jax.jacrev`` computes the
        # identical Jacobian (verified rel-err 1e-7 vs jacve) via
        # native reverse-mode AD — order-free and XLA-optimised, ~0s.
        # See env-timing investigation 2026-06-05.
        if config.measure_grad:
            # Gradient mode: exact reference is native ``jax.value_and_grad``
            # (order-free reverse-mode AD) → ``(value, grads)``; the quality
            # block reads ``out_exact[1]`` (grads), mirroring the approx path.
            fn = jax.value_and_grad(config.target_fun, argnums=config.argnums)
        elif config.has_aux:
            # jacve(has_aux) returns ``(primal, jac)`` and the caller
            # reads ``out_exact[1]``. Match that ordering so the
            # quality-metric indexing stays correct.
            def _exact_with_aux(*a):
                jac, aux = jax.jacrev(
                    config.target_fun, argnums=config.argnums, has_aux=True,
                )(*a)
                return (aux, jac)
            fn = _exact_with_aux
        else:
            fn = jax.jacrev(config.target_fun, argnums=config.argnums)
        return jax.jit(fn, keep_unused=True).lower(*args_for_lower).compile()

    _dbg_t = os.environ.get("ALPHAGRAD_DBG_TIMING", "0") == "1"
    if _dbg_t:
        import time as _time
        _t0 = _time.time()
    compiled_approx = _inproc_lru_cached_compile(b"approx:" + cache_key, _do_compile_approx)
    if _dbg_t:
        print(f"[DBG-env] term={is_terminal} approx_compile={_time.time()-_t0:.1f}s", flush=True)
        _t0 = _time.time()
    # DETERMINISTIC peak memory from the compiled executable's XLA memory
    # analysis (no execution, no polling). The ResourceMonitor's CPU peak is a
    # SAMPLED high-water mark polled at ~1ms — for a sub-ms gradient exec the
    # transient allocation is freed between polls, so RM reads ~0 (or only the
    # ~constant output once). ``memory_analysis`` gives the true working set:
    # temp (XLA scratch — strongly ORDER-dependent, e.g. fwd 0.97MB vs rev
    # 13MB) + output (the gradient) + arguments. This is the reliable,
    # order-discriminating peak_memory signal; RM-sampled peak stays as a
    # fallback only when memory_analysis is unavailable.
    _det_peak = None
    try:
        _ma = compiled_approx.memory_analysis()
        _det_peak = float(
            int(getattr(_ma, "temp_size_in_bytes", 0) or 0)
            + int(getattr(_ma, "output_size_in_bytes", 0) or 0)
            + int(getattr(_ma, "argument_size_in_bytes", 0) or 0)
        )
        if _det_peak <= 0.0:
            _det_peak = None
    except Exception:
        _det_peak = None

    # GATE 2 (post-compile per-device memory gate): compiled, but BEFORE we
    # EXECUTE, decide whether the exec can fit on the measure device(s); if not,
    # skip it (RAISE -> logged ``[SENTINEL] mem-gate``, excluded from loss).
    # This is the ONLY reliable guard against the measure-GPU OOM: under
    # ``ALPHAGRAD_SHARD_MEASURE`` an OOM happens INSIDE the NCCL collective ->
    # ``rendezvous ... waiting`` -> an UNCATCHABLE hang (the job deadlocks), so
    # it MUST be caught pre-exec.
    #
    # WHY A PER-ORDER PEAK ESTIMATE ALONE IS INSUFFICIENT (diagnosed on ConvNet
    # seed-free Jacobian, 2-GPU sharded, gpu16): every single order's
    # memory_analysis peak is tiny (<=0.7GiB/device) yet the actor still OOMs —
    # the GPU's live use climbs (resident executables + replicated weights) and
    # the XLA exec / conv-autotuner WORKSPACE (~2.5GiB transient, NOT in
    # memory_analysis) is what trips the BFC allocator. So the gate combines a
    # per-order estimate with LIVE free-memory checks, the decisive one being an
    # absolute FLOOR on the remaining free: once free drops below the floor,
    # EVERY further exec is skipped -> the collective is never entered without
    # headroom for the (unpredicted) workspace -> no OOM, no hang.
    #
    # DETERMINISM: the sharded measure runs in ONE process that dispatches a
    # single GSPMD executable across both GPUs, so this gate is evaluated ONCE
    # per (order, specs) before the collective — there is no cross-process race.
    # Even read as independent ranks, every input is identical across ranks: the
    # estimate comes only from ``compiled_approx`` (byte-identical per cache key)
    # and the live readings are the SAME physical per-device values regardless
    # of which rank queries them (we take the MIN over the shared
    # ``_shard_devices`` list). So all ranks reach the SAME skip decision and the
    # collective is never half-skipped.
    #
    # KNOBS (all per-device):
    #   ALPHAGRAD_MAX_MEASURE_MEM_GIB  static cap; 0/"auto" -> FRAC*bytes_limit;
    #                                  negative -> gate fully disabled.
    #   ALPHAGRAD_MEASURE_MEM_FRAC     (0.85) auto static-cap fraction of limit.
    #   ALPHAGRAD_MEASURE_MEM_SAFETY   (2.0)  multiplier on the per-order estimate.
    #   ALPHAGRAD_MEASURE_MEM_HEADROOM (0.90) estimate must fit 90% of live free.
    #   ALPHAGRAD_MEASURE_MEM_FLOOR_GIB(1.0)  ESTIMATE-BASED gate: the real
    #                                  protection is est*safety > headroom*free
    #                                  (above); genuine big configs still gate.
    #                                  This floor is now only a small ABSOLUTE
    #                                  minimum so we never measure into a
    #                                  near-empty device; it must NOT dominate.
    #                                  Was 8.0G (fixed) which spuriously skipped
    #                                  a 0.32G diag measure whenever free<8G even
    #                                  with tens of GiB truly free -> sentinels/
    #                                  failed_transitions under moderate load
    #                                  (51557 ep235/236 + ep362->363 spike).
    #                                  0 disables the floor.
    _gate_raw = (os.environ.get("ALPHAGRAD_MAX_MEASURE_MEM_GIB", "0") or "").strip().lower()
    _gate_disabled = False
    _gate_auto = _gate_raw in ("", "0", "0.0", "auto")
    _max_measure_mem_gib = 0.0
    if not _gate_auto:
        try:
            _max_measure_mem_gib = float(_gate_raw)
        except ValueError:
            _max_measure_mem_gib = 0.0
        if _max_measure_mem_gib < 0:
            _gate_disabled = True
    _mem_safety = float(os.environ.get("ALPHAGRAD_MEASURE_MEM_SAFETY", "2.0") or 2.0)
    _mem_frac = float(os.environ.get("ALPHAGRAD_MEASURE_MEM_FRAC", "0.85") or 0.85)
    _mem_headroom = float(os.environ.get("ALPHAGRAD_MEASURE_MEM_HEADROOM", "0.90") or 0.90)
    _mem_floor = float(os.environ.get("ALPHAGRAD_MEASURE_MEM_FLOOR_GIB", "1.0") or 1.0) * (1024 ** 3)
    # ALPHAGRAD_MEMGATE_USE_CURRENT (default ON): compute _live_free from CURRENT
    # occupancy (bytes_in_use / bytes_reserved) and DROP the latched high-water
    # peaks (peak_bytes_in_use / peak_bytes_reserved). CONFIRMED BUG: under
    # XLA_PYTHON_CLIENT_PREALLOCATE=false the BFC pool GROWS on a transient peak
    # (e.g. a heavy COMPRESS densification) and peak_bytes_in_use LATCHES that
    # high-water mark forever, so _live_free = bytes_limit - peak reads ~0 for
    # the rest of the run even though the memory was freed and nvidia-smi shows
    # 23-31G actually free -> every subsequent 0.32G grad-measure spuriously
    # gates ("free 0.0G < floor 8.0G") -> sentinels / failed_transitions (the
    # basin signal). bytes_limit is the process's STATIC XLA fraction ceiling
    # (~71G of a 96G Blackwell, verified), NOT the grown pool, so subtracting
    # CURRENT usage yields true per-process headroom. Set to 0 to restore the
    # old latched (peak-inclusive) behaviour.
    _memgate_use_current = os.environ.get("ALPHAGRAD_MEMGATE_USE_CURRENT", "1") != "0"

    # Measure device(s) — the SAME list on every rank.
    _gate_devices = []
    if _shard_devices:
        _gate_devices = list(_shard_devices)
    elif callback_device is not None:
        _gate_devices = [callback_device]

    if (not _gate_disabled) and _gate_devices:
        # Per-order conservative estimate (only when memory_analysis worked;
        # else estimate stays None and only the live-free floor can fire).
        _gate_est = None
        if _det_peak is not None:
            _ca_bytes = 0.0
            if os.environ.get("ALPHAGRAD_SKIP_COST_ANALYSIS", "0") != "1":
                try:
                    _ca_probe = compiled_approx.cost_analysis() or {}
                    if isinstance(_ca_probe, (list, tuple)):
                        _ca_probe = _ca_probe[0] if _ca_probe else {}
                    _ca_bytes = float(_ca_probe.get("bytes accessed", 0) or 0)
                except Exception:
                    _ca_bytes = 0.0
            _gate_est = max(_det_peak, _ca_bytes) * _mem_safety

        _static_budget = None
        _live_free = None
        try:
            _stats0 = _gate_devices[0].memory_stats() or {}
            _blim = int(_stats0.get("bytes_limit", 0) or 0)
            if _max_measure_mem_gib > 0:
                _static_budget = _max_measure_mem_gib * (1024 ** 3)
            elif _blim > 0:
                _static_budget = _mem_frac * _blim
            _frees = []
            for _d in _gate_devices:
                _st = _d.memory_stats() or {}
                _lim = int(_st.get("bytes_limit", 0) or 0)
                # Conservative "used": the BFC allocator can hold large
                # FREED-but-RESERVED regions (fragmentation) that a fresh exec
                # allocation cannot reuse, so bytes_in_use ALONE under-counts
                # OOM risk. Take the max of the live, peak, and any
                # reserved/pool field exposed by memory_stats so the gate sees
                # the true pressure on the device.
                # NB do NOT include pool_bytes/peak_pool_bytes here — those
                # report the BFC POOL SIZE (== bytes_limit once the pool has
                # grown), not occupancy, so they would force free->0 and skip
                # EVERY order.
                #
                # DEFAULT (ALPHAGRAD_MEMGATE_USE_CURRENT=1): use CURRENT
                # occupancy only. The peak_* fields LATCH a transient
                # high-water mark that never decays under a growable BFC pool,
                # so once any heavy order peaks, _live_free = limit - peak reads
                # ~0 for the whole run and gates every later 0.32G measure even
                # though 23-31G is truly free. bytes_in_use/bytes_reserved
                # reflect the memory the next allocation actually has to fit
                # around, which is the correct headroom signal.
                if _memgate_use_current:
                    _use = max(
                        int(_st.get("bytes_in_use", 0) or 0),
                        int(_st.get("bytes_reserved", 0) or 0),
                    )
                else:
                    # Legacy latched behaviour (peak-inclusive fragmentation
                    # proxy) — revertible via env for A/B comparison.
                    _use = max(
                        int(_st.get("bytes_in_use", 0) or 0),
                        int(_st.get("peak_bytes_in_use", 0) or 0),
                        int(_st.get("bytes_reserved", 0) or 0),
                        int(_st.get("peak_bytes_reserved", 0) or 0),
                    )
                if _lim > 0:
                    _frees.append(_lim - _use)
            if _frees:
                _live_free = min(_frees)
        except Exception:
            pass

        _why = []
        if _gate_est is not None and _static_budget is not None and _gate_est > _static_budget:
            _why.append(f"est {_gate_est / 1024 ** 3:.2f}G>static {_static_budget / 1024 ** 3:.1f}G")
        if _gate_est is not None and _live_free is not None and _gate_est > _mem_headroom * _live_free:
            _why.append(
                f"est {_gate_est / 1024 ** 3:.2f}G>{_mem_headroom:g}*free {_live_free / 1024 ** 3:.1f}G"
            )
        if _mem_floor > 0 and _live_free is not None and _live_free < _mem_floor:
            _why.append(f"free {_live_free / 1024 ** 3:.1f}G<floor {_mem_floor / 1024 ** 3:.1f}G")
        if _why:
            print(
                f"[SENTINEL] mem-gate: per-device "
                f"({'; '.join(_why)}) sharded={_shard_mesh is not None} — skip exec",
                flush=True,
            )
            raise RuntimeError(
                f"mem-gate: per-device limit reached ({'; '.join(_why)}) — skip exec"
            )
    # ``compiled_exact`` is ONLY needed for the quality metrics
    # (cosine_sim, frob_residual). Those are meaningful only when the
    # elimination order is complete — graphax's ``jacve`` returns a
    # zero-norm Jacobian for any partial order, so comparing approx vs
    # exact mid-rollout yields ``(cos=0, frob=0)`` regardless. Skip
    # the compile + execute when the step is non-terminal; the cache
    # entry would never be re-used productively anyway.
    if is_terminal:
        compiled_exact = _inproc_lru_cached_compile(b"exact:" + exact_cache_key, _do_compile_exact)
        if _dbg_t:
            print(f"[DBG-env] exact_compile={_time.time()-_t0:.1f}s", flush=True)
            _t0 = _time.time()
    else:
        compiled_exact = None

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
    if _dbg_t:
        print(
            f"[DBG-env] cost_analysis={_time.time()-_t0:.1f}s "
            f"flops={flops:.3g} bytes={bytes_accessed:.3g} "
            f"muls_adds_fmas={muls_adds_fmas:.3g} max_io={max_io_sum:.3g}",
            flush=True,
        )
        _t0 = _time.time()

    # ------------------------------------------------------------------
    # FLOP-gate (see EnvConfig.flop_gate_threshold). A pathological order
    # would cost ~50s/exec to measure; its FLOP count (free, above) flags
    # it first. Short-circuit the whole measurement with FLOP/bytes-
    # derived cost surrogates + a worst-case quality penalty so the
    # policy still gets a strong "avoid this" gradient at ~zero cost.
    # Only meaningful on the terminal step (where the full path runs).
    _flop_gate = float(getattr(config, "flop_gate_threshold", 0.0) or 0.0)
    if _flop_gate > 0.0 and flops > _flop_gate:
        if _dbg_t:
            print(
                f"[DBG-env] FLOP-GATE: flops={flops:.3g} > {_flop_gate:.3g} "
                "— skipping measurement, using cost surrogates",
                flush=True,
            )
        # Cost surrogates: the cmp/mem channels read the (free) symbolic
        # counts directly. latency_ns ← flops, peak_memory ← bytes — same
        # "lower is better" direction, and after the per-channel
        # symlog+EMA norm downstream these land in the bad tail. Quality
        # is set to worst-case (cossim 0, frob 1.0 = 100% error) since we
        # did not measure it and the order is being rejected.
        rewards = jnp.array(
            [
                -muls_adds_fmas,
                -flops,
                -float(flops),          # latency_ns surrogate
                -max_io_sum,
                -bytes_accessed,
                -float(bytes_accessed),  # peak_memory surrogate
                0.0,                     # cosine_sim (worst)
                -1.0,                    # frob_residual = 1.0 (100% error)
                -float(bytes_accessed),  # xla_peak_memory surrogate (rejected)
                0.0,                     # bkstep_acc (worst trainability = 0)
            ],
            dtype=jnp.float32,
        )
        return tokens, eqn_ids, rewards

    # ------------------------------------------------------------------
    # Execution loop — runs once for peak_memory + quality, or 10x when
    # `measure_latency` is on (the latency reading is noisy enough that the
    # top-quartile-mean smoothing from the original code is worth keeping).
    # ------------------------------------------------------------------
    # Per-step latency sample count. The legacy default was hard-coded
    # 10 (and 1 when measure_latency=False). Now driven by
    # ``EnvConfig.latency_samples``; preserves the "1 sample, latency=0"
    # behaviour when measure_latency is off so non-latency runs keep
    # the same speed as before.
    #
    # IMPORTANT: ``compiled_approx`` (and ``compiled_exact``) were
    # populated once above via ``cached_compile`` — the cluster-wide
    # ObjectRef cache means only ONE actor in the pool compiles per
    # unique (order, specs, shape) key, and all other actors fetch +
    # ``deserialize_and_load`` from the shared blob. The for-loop below
    # invokes the SAME compiled object n_samples times — there is no
    # per-iteration recompile. So raising latency_samples is linear in
    # exec cost only (no compile blow-up); the cache makes the per-step
    # compile cost essentially zero after the first call per unique
    # (order, transforms).
    # Noisy-channel measurement plan: ``num_data_points`` distinct
    # rollout-sampled args × ``reps_per_point`` reruns each. When
    # ``measure_latency`` is off we collapse to a single run (deterministic
    # channels are all we'd be measuring; reps and extra points are wasted
    # compute). When the eval_samples bank is smaller than ``num_data_points``
    # we cap to whatever's available.
    n_points = max(int(getattr(config, "num_data_points", 1)), 1)
    reps = max(int(getattr(config, "reps_per_point", 1)), 1)
    if not config.measure_latency:
        n_points = 1
        reps = 1
    if eval_samples:
        bank_size = int(eval_samples[0].shape[0])
        n_points = min(n_points, max(bank_size, 1))
    # Per-point dispatch (global measurement queue): when point_idx >= 0,
    # this call measures EXACTLY ONE data point (R reps) instead of the
    # full n_points sweep. The driver fans 16 envs × n_points such calls
    # across the actor pool (ray.util.ActorPool) so an env's points run
    # in parallel and gated/cheap tasks free workers for stragglers —
    # full core utilisation. The driver then P60-aggregates each env's
    # per-point reward vectors. See ppo_ray_worker._fan_out_terminal_queue.
    if point_idx >= 0 and eval_samples:
        _pi = point_idx if point_idx < int(eval_samples[0].shape[0]) else 0
        _point_iter = [_pi]
        n_samples = reps
    else:
        _point_iter = list(range(n_points))
        n_samples = n_points * reps

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

    # Iteration order: data-point outer, rep inner. Same compiled fn
    # invoked `reps` times on each of `n_points` distinct args sets.
    #
    # Two DISTINCT sample populations come out of this loop:
    #   * latency_ns / peak_memory  — one entry per (data-point, rep),
    #     i.e. ``n_points * reps`` noisy timing/memory samples. Reps
    #     exist to denoise these.
    #   * quality (cosine_sim / frob_residual) — the Jacobian VALUES are
    #     deterministic given a data point, so reps would only duplicate
    #     identical (approx, exact) pairs. We therefore collect the
    #     quality pair ONCE per data point (first rep). This keeps the
    #     expensive exact-Jacobian execution + ~600 MB flatten/compare
    #     at ``n_points`` invocations instead of ``n_points * reps`` —
    #     the difference between a 5× and a 20× quality cost at the
    #     default 5×4 design.
    # Deterministic channels (muls_adds_fmas, max_io_sum, flops,
    # bytes_accessed) were already computed once above.
    #
    # Slow-order cutoff (see EnvConfig.slow_exec_cutoff_seconds): if any
    # single approx exec exceeds the cutoff, the elimination order is
    # pathologically expensive — keep the samples gathered so far and
    # stop, instead of repeating a 40s exec 20×.
    import time as _measure_time
    _slow_cutoff = float(getattr(config, "slow_exec_cutoff_seconds", 0.0) or 0.0)
    _inner = max(int(getattr(config, "latency_inner_reps", 1) or 1), 1)
    _warmup = max(int(getattr(config, "latency_warmup", 0) or 0), 0)
    if not config.measure_latency:
        # Latency is discarded (hard-set to 0.0 below) — don't pay the
        # inner-loop / warmup executions for a reading nobody reads.
        _inner = 1
        _warmup = 0
    _bypass_rm = os.environ.get("ALPHAGRAD_BYPASS_RESOURCE_MONITOR", "0") == "1"
    # Time via the (fixed) ResourceMonitor instead of perf_counter when asked —
    # it gives latency AND peak memory from a single execution pass.
    _use_rm_timer = (
        str(getattr(config, "latency_timer", "perf_counter")) == "rm"
        and not _bypass_rm
    )
    _budget_hit = False
    # Fix 2(b): guard the whole exec loop against a GPU OOM / RESOURCE_
    # EXHAUSTED (a too-big COMPRESS densification exceeding the mem-gate's
    # headroom). Convert it to a RuntimeError so the pool's existing
    # handler turns it into a clean [SENTINEL] (bounded, excluded from the
    # loss by Fix 1) instead of crashing / hanging the measure actor.
    try:
        for d in _point_iter:
            if _budget_hit:
                break
            if eval_samples:
                eval_args_i = [arg[d] for arg in eval_samples]
            else:
                eval_args_i = list(args)
            if callback_device is not None or _shard_mesh is not None:
                eval_args_i = _put_measure(eval_args_i)
            # Per-data-point warmup: discard the first ``_warmup`` executions so
            # first-touch / cache / allocation effects don't pollute the timed
            # readings (see EnvConfig.latency_warmup). Each warmup exec is itself
            # checked against the slow-order cutoff so a pathological order bails
            # after one ~cutoff-second exec instead of running the full warmup +
            # inner loop first. The LAST warmup exec doubles as the peak-memory
            # read (it's discarded for timing anyway) — avoids a separate extra
            # execution per data point when warmup is enabled. ``None`` ⇒ not yet
            # captured, fall back to the dedicated r==0 monitor below.
            _peak_captured = None
            for _w in range(_warmup):
                _ws = _measure_time.perf_counter()
                if (not _bypass_rm) and _w == _warmup - 1:
                    with ResourceMonitor(devices=unique_devices) as _wmon:
                        _wout = compiled_approx(*eval_args_i)
                    jax.block_until_ready(_wout)
                    _peak_captured = float(_wmon.stats.get("memory", 0.0))
                else:
                    jax.block_until_ready(compiled_approx(*eval_args_i))
                if _slow_cutoff > 0.0 and (
                    _measure_time.perf_counter() - _ws
                ) > _slow_cutoff:
                    _budget_hit = True
                    break
            for r in range(reps):
                if _budget_hit:
                    break
                # Latency: time a tight inner loop of ``_inner`` back-to-back
                # executions with a SINGLE closing ``block_until_ready`` barrier,
                # using ``perf_counter``, then divide by the rep count. This
                # amortizes per-call dispatch/barrier overhead (the dominant noise
                # for sub-ms kernels) and — unlike the old ResourceMonitor wall
                # timer, whose ``stop()`` fired before its barrier drained the
                # async queue — measures the true end-to-end latency with a stable
                # absolute value. ResourceMonitor is now used for the peak-memory
                # channel ONLY (one call per data point; peak is deterministic).
                _exec_wall0 = _measure_time.time()
                if _use_rm_timer:
                    # RM times the inner loop with ``block_until_ready`` INSIDE the
                    # context (duration = true device time, validated ≈ perf_counter)
                    # and reads peak memory from the SAME pass on the first rep — one
                    # execution serves both the latency and peak-memory channels.
                    _want_peak = (r == 0 and _peak_captured is None)
                    with ResourceMonitor(
                        devices=unique_devices, time=True, peak=_want_peak,
                    ) as _tmon:
                        for _ in range(_inner):
                            out_approx = compiled_approx(*eval_args_i)
                        jax.block_until_ready(out_approx)
                    _lat_ns = _tmon.duration / _inner * 1e9
                    if r == 0:
                        peak_mem_samples.append(
                            _peak_captured if _peak_captured is not None
                            else float(_tmon.stats.get("memory", 0.0))
                        )
                else:
                    # perf_counter inner loop + a separate ResourceMonitor pass for
                    # peak memory (once per data point, first rep).
                    _t0 = _measure_time.perf_counter()
                    for _ in range(_inner):
                        out_approx = compiled_approx(*eval_args_i)
                    jax.block_until_ready(out_approx)
                    _lat_ns = (_measure_time.perf_counter() - _t0) / _inner * 1e9
                    if r == 0:
                        if _peak_captured is not None:
                            peak_mem_samples.append(_peak_captured)
                        elif _bypass_rm:
                            peak_mem_samples.append(0.0)
                        else:
                            with ResourceMonitor(devices=unique_devices) as monitor:
                                _mout = compiled_approx(*eval_args_i)
                            jax.block_until_ready(_mout)
                            peak_mem_samples.append(
                                float(monitor.stats.get("memory", 0.0))
                            )
                latency_samples.append(_lat_ns)
                _exec_wall = _measure_time.time() - _exec_wall0
                # Per-EXEC time for the cutoff (the inner loop runs _inner execs;
                # _exec_wall spans the whole loop + the r==0 peak-memory exec, so
                # comparing it directly to the per-exec cutoff would trip on
                # healthy orders once _inner>1). Use the amortized per-exec time.
                _per_exec_s = _lat_ns / 1e9

                # Quality pair: collect once per data point (first rep only).
                # ``compiled_exact`` is non-None only at the terminal step.
                # Force-materialize the exact result HERE so its execution
                # cost is attributed to this point's measurement (and to the
                # quality phase intent) rather than leaking, lazily, into the
                # NEXT iteration's ResourceMonitor barrier — which previously
                # mis-attributed exact-compute time to the approx latency
                # reading and hid it from the slow-order cutoff.
                if compiled_exact is not None and r == 0:
                    out_approxs.append(out_approx)
                    _oe = compiled_exact(*eval_args_i)
                    jax.block_until_ready(_oe)
                    out_exacts.append(_oe)

                # Slow-order cutoff: if a single exec exceeded the cutoff, the
                # order is pathologically expensive — one sample is enough to know
                # it's slow. Keep what we have and stop.
                if _slow_cutoff > 0.0 and _per_exec_s > _slow_cutoff:
                    _budget_hit = True
                    if _dbg_t:
                        print(
                            f"[DBG-env] slow-order cutoff at d={d + 1}/{n_points} "
                            f"r={r + 1}/{reps}: per_exec={_per_exec_s:.1f}s > "
                            f"{_slow_cutoff:.0f}s — capping samples",
                            flush=True,
                        )
                    break

    except (MemoryError, RuntimeError) as _oom_e:
        _m = str(_oom_e)
        if 'mem-gate' in _m:
            raise
        print(f'[SENTINEL] measure-oom during exec: {_m[:160]}', flush=True)
        raise RuntimeError(f'measure-oom: {_m[:200]}')
    except Exception as _xla_e:
        # jaxlib XlaRuntimeError (RESOURCE_EXHAUSTED / OOM) is not a
        # subclass of the above; catch by name so a device OOM never
        # crashes the actor — it becomes a bounded sentinel.
        _n = type(_xla_e).__name__
        _m = str(_xla_e)
        if ('XlaRuntimeError' in _n or 'RESOURCE_EXHAUSTED' in _m
                or 'out of memory' in _m.lower() or 'RESOURCE_EXHAUSTED' in _n):
            print(f'[SENTINEL] measure-oom (xla) during exec: {_m[:160]}', flush=True)
            raise RuntimeError(f'measure-oom: {_m[:200]}')
        raise
    # Noisy-channel aggregation: P-``percentile_keep`` over the full
    # pool of ``n_points * reps`` measurements (default P60 of 20).
    # Replaces the legacy top-quartile-mean (latency) and max
    # (peak_memory). Higher percentile → more conservative / worse-case
    # estimate. See EnvConfig.percentile_keep.
    if _dbg_t:
        print(f"[DBG-env] exec_loop(n={n_samples})={_time.time()-_t0:.1f}s", flush=True)
        _t0 = _time.time()
    pk = float(getattr(config, "percentile_keep", 0.60))
    _winsor = float(getattr(config, "latency_winsor", 0.0) or 0.0)
    if not config.measure_latency:
        latency_ns = 0.0
    else:
        # Drop non-positive readings (failed measurements). With perf_counter
        # timing a genuine reading is always > 0; a 0 only appears on failure.
        _lat_valid = [x for x in latency_samples if x > 0.0 and np.isfinite(x)]
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        if not _lat_valid:
            # No usable latency reading -> genuine failed-measure sentinel.
            latency_ns = -SENTINEL_REWARD_VALUE
        else:
            if _winsor > 0.0:
                latency_ns = _winsorized_mean(_lat_valid, _winsor)
            else:
                latency_ns = _percentile_pool(_lat_valid, pk)
    # RM-sampled peak: exact on GPU (clear_memory_stats + peak_bytes_in_use),
    # a sampled high-water mark on CPU (misses sub-ms grad allocs → unreliable).
    _rm_peak = float(_percentile_pool(peak_mem_samples, pk))
    # XLA-analysis peak (temp+output+args); always emitted as its own channel.
    # Falls back to the RM peak only if memory_analysis was unavailable.
    xla_peak_memory = _det_peak if _det_peak is not None else _rm_peak
    # ``peak_memory`` (idx 5) ALWAYS carries the ResourceMonitor measurement —
    # exact on GPU, a sampled high-water mark on CPU — so the RM signal is
    # recorded on every run regardless of device. The deterministic XLA
    # estimate lives in its own ``xla_peak_memory`` channel (idx 8). To use the
    # XLA peak as the CPU memory REWARD, select ``--mem-type xla_peak_memory``
    # (routes the mem weight to idx 8 in build_reward_weights) — that keeps the
    # reward deterministic while still measuring/logging the RM peak at idx 5.
    peak_memory = _rm_peak

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
        _t_approx = _t_exact = _t_metric = 0.0
        # Grad mode: ``value_and_grad`` returns ``(value, grads)`` for both the
        # approx and exact paths, so the gradient pytree is at ``[1]`` — same
        # slot as jacve's has_aux ``(primal, jac)``. Compare the GRADIENTS.
        _take_second = config.has_aux or config.measure_grad
        for out_approx, out_exact in zip(out_approxs, out_exacts):
            jac_approx = out_approx[1] if _take_second else out_approx
            jac_exact = out_exact[1] if _take_second else out_exact
            if _dbg_t:
                _tt = _time.time()
                jax.block_until_ready(jac_approx); _t_approx += _time.time() - _tt
                _tt = _time.time()
                jax.block_until_ready(jac_exact); _t_exact += _time.time() - _tt
                _tt = _time.time()
            cos, rel_frob = _quality_metrics(jac_exact, jac_approx)
            if _dbg_t:
                jax.block_until_ready((cos, rel_frob)); _t_metric += _time.time() - _tt
            cosines.append(cos)
            frobs.append(rel_frob)
        if _dbg_t:
            print(f"[DBG-env] quality_split approx_block={_t_approx:.1f}s exact_block={_t_exact:.1f}s metric={_t_metric:.1f}s npairs={len(out_exacts)}", flush=True)
        # cosine_sim is "measure but don't reward" per the current
        # scalarization design — kept for diagnostic logging, weight
        # stays 0 in build_reward_weights for --rewards cmp/mem. Use
        # the same P-`percentile_keep` aggregation as the cost channels
        # for consistency; flip direction below.
        # frob_residual: P60 = worst-60% of the pool (higher = worse).
        cosine_sim = _percentile_pool(cosines, pk)
        frob_residual = _percentile_pool(frobs, pk)
    else:
        cosine_sim = 0.0
        frob_residual = 0.0
    if _dbg_t and is_terminal:
        print(f"[DBG-env] quality={_time.time()-_t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # B_kstep — closed-loop trainability accuracy (the "acc" reward channel
    # under ALPHAGRAD_ACC_PROXY=bkstep). Terminal + measure_grad only: it needs
    # the policy's approx-GRADIENT fn (compiled_approx returns (value, grads)).
    # Runs K Adam steps × S seeds of REAL MNIST training with the approx grad,
    # then reads test accuracy. Gated behind ALPHAGRAD_BKSTEP=1 so cosine-only
    # runs pay nothing. Any failure -> 0.0 (worst trainability), never a crash.
    # ------------------------------------------------------------------
    bkstep_acc = 0.0
    _bkstep_on = os.environ.get("ALPHAGRAD_BKSTEP", "0") == "1"
    if _bkstep_on and is_terminal and config.measure_grad:
        try:
            _bk_k = int(os.environ.get("ALPHAGRAD_BKSTEP_K", "40") or "40")
            _bk_ns = int(os.environ.get("ALPHAGRAD_BKSTEP_SEEDS", "2") or "2")
            _bk_seeds = tuple(range(max(_bk_ns, 1)))
            if _dbg_t:
                _tbk = _time.time()
            bkstep_acc = _bkstep_probe(
                compiled_approx, args, config.argnums,
                n_steps=max(_bk_k, 1), seeds=_bk_seeds,
            )
            if _dbg_t:
                print(
                    f"[DBG-env] bkstep acc={bkstep_acc:.4f} K={_bk_k} "
                    f"seeds={_bk_ns} t={_time.time()-_tbk:.1f}s", flush=True,
                )
        except Exception as _bk_e:  # pragma: no cover - defensive
            print(f"[bkstep] probe failed -> acc=0.0 | {_bk_e}", flush=True)
            bkstep_acc = 0.0

    # Raw-measurement sink: when a caller passes ``raw_sink={}`` it gets the
    # per-sample/per-point distributions (the "10x8" the sampler records)
    # instead of only the percentile-aggregated reward vector. Deterministic
    # channels are scalars; latency/peak are per-rep sample lists; cosine/frob
    # are per-point lists (terminal only). The ternary guards mean cosines/
    # frobs are only referenced when they were actually computed.
    if raw_sink is not None:
        raw_sink["muls_adds_fmas"] = float(muls_adds_fmas)
        raw_sink["max_io_sum"] = float(max_io_sum)
        raw_sink["flops"] = float(flops)
        raw_sink["bytes_accessed"] = float(bytes_accessed)
        raw_sink["latency_ns_samples"] = (
            [float(x) for x in latency_samples] if config.measure_latency else []
        )
        raw_sink["peak_memory_samples"] = [float(x) for x in peak_mem_samples]
        # Deterministic XLA-analysis peak (temp+output+args) — the reliable
        # peak channel on CPU where the RM sampled peak misses sub-ms allocs.
        raw_sink["xla_peak_memory"] = float(xla_peak_memory)
        raw_sink["cosine_sim_per_point"] = (
            [float(x) for x in cosines] if (is_terminal and out_exacts) else []
        )
        raw_sink["frob_residual_per_point"] = (
            [float(x) for x in frobs] if (is_terminal and out_exacts) else []
        )
        raw_sink["bkstep_acc"] = float(bkstep_acc)

    # ------------------------------------------------------------------
    # Capped-cossim GUIDE (anti flat-zero-basin). When
    # ``ALPHAGRAD_COSSIM_GUIDE_CAP`` is set to a float C, the cosine_sim
    # channel emitted to the scalar reward is capped at C via ``min`` (NOT
    # clip-at-0): so it stays a MONOTONIC climb signal even from NEGATIVE
    # cossim up to C. Below the trainability edge (where B_kstep(fracred)==0
    # has no gradient) this lets a small cosine weight (--lambda-cossim-guide)
    # pull the untrained policy up toward the edge; above C the term is a
    # constant so B_kstep(fracred) resolves the compute/mem/fidelity tradeoff.
    # The raw (uncapped) cosine stays in raw_sink for honest logging.
    # RAW full-range cossim threading (bridge-cse). When
    # ALPHAGRAD_RAW_COSSIM_THREAD=1 the env emits the RAW (uncapped,
    # full-range) cosine_sim in idx6 and the guide cap is deferred to the
    # PPO worker (applied into the GAE buffer only), so the worker can log
    # ``reward/cosine_sim_raw`` and feed the dynamic-sentinel EMA with the
    # honest full-range value. When 0 (legacy) the cap is applied here.
    _raw_thread = os.environ.get("ALPHAGRAD_RAW_COSSIM_THREAD", "1") == "1"
    _guide_cap_raw = os.environ.get("ALPHAGRAD_COSSIM_GUIDE_CAP", "").strip()
    cosine_channel = cosine_sim
    if _guide_cap_raw and not _raw_thread:
        try:
            _C = float(_guide_cap_raw)
            cosine_channel = float(min(cosine_sim, _C))
        except ValueError:
            cosine_channel = cosine_sim

    rewards = jnp.array(
        [
            -muls_adds_fmas,
            -flops,
            -latency_ns,
            -max_io_sum,
            -bytes_accessed,
            -peak_memory,
            cosine_channel,  # capped-cossim guide when GUIDE_CAP set, else raw
            -frob_residual,
            -xla_peak_memory,
            bkstep_acc,  # positive quality channel (trainability accuracy)
        ],
        dtype=jnp.float32,
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
            # GRAD MODE (--measure-grad): graphax value_and_grad accumulates
            # the gradient via a FULL reverse elimination pass that MUST include
            # the output / loss-reduction vertices — the cotangent flows from the
            # scalar loss back to the weights THROUGH them. Excluding them (the
            # Jacobian-path rule below) leaves the policy emitting an order over a
            # SUBSET of vertices; graphax then never propagates past the missing
            # output vertices and the gradient pytree is STRUCTURALLY ZERO (primal
            # value still correct) -> cossim==0 for every rule. So in grad mode
            # EVERY equation is an eliminable vertex. The Jacobian path (default)
            # keeps the original rule: output vertices are not eliminated (their
            # edges become the Jacobian), which tolerates a partial order.
            if bool(getattr(config, "measure_grad", False)):
                valid_vertices = tuple(range(1, len(config.jaxpr.eqns) + 1))
            else:
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
        latency_samples: int = 1,
        num_data_points: int = 5,
        reps_per_point: int = 4,
        percentile_keep: float = 0.60,
        # Match the EnvConfig field default (8.0) so the cutoff behaviour is
        # the same whether the env is built via from_jaxpr or constructed
        # directly (tests) — they previously diverged 15.0 vs 8.0.
        slow_exec_cutoff_seconds: float = 8.0,
        flop_gate_threshold: float = 0.0,
        terminal_rewards_only: bool = False,
        latency_inner_reps: int = 1,
        latency_warmup: int = 0,
        latency_winsor: float = 0.0,
        measure_grad: bool = False,
        latency_timer: str = "perf_counter",
        quant_once: bool = False,
    ):
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
            target_fun=target_fun,
            data_gen=data_gen,
            exec_on_gpu=exec_on_gpu,
            measure_latency=measure_latency,
            latency_samples=latency_samples,
            num_data_points=num_data_points,
            reps_per_point=reps_per_point,
            percentile_keep=percentile_keep,
            slow_exec_cutoff_seconds=slow_exec_cutoff_seconds,
            flop_gate_threshold=flop_gate_threshold,
            terminal_rewards_only=terminal_rewards_only,
            latency_inner_reps=latency_inner_reps,
            latency_warmup=latency_warmup,
            latency_winsor=latency_winsor,
            measure_grad=measure_grad,
            latency_timer=latency_timer,
            quant_once=quant_once,
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

    def tokenize(self, init: bool = False):
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
            return partial(_callback, self.config, init=init)

        # The pool's ``evaluate`` signature is
        # ``(order, specs, step, eval_samples, *, init)`` — but
        # ``io_callback`` passes ``(args, consts, order, specs, step,
        # *eval_samples)`` as positional args (see ``env.step`` line
        # 1360 and ``env.reset`` line 1297). Build an adapter that
        # drops ``args``/``consts`` (the actor's own env has its own
        # bound args) and re-packages ``eval_samples`` as a tuple.
        pool = self._remote_pool

        def _remote_callback(args, consts, order, specs, step, *eval_samples):
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

        tokens, eqn_ids, _ = io_callback(
            self.tokenize(init=True),
            self._callback_shape,
            self.args,
            self.consts,
            initial_order,
            initial_specs,
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

        tokens, eqn_ids, reward = io_callback(
            self.tokenize(),
            self._callback_shape,
            self.args,
            self.consts,
            new_order,
            new_specs,
            new_step,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
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
