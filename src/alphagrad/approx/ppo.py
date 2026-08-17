"""PPO trainer for the vertex-elimination env.

The agent factors as `Agent = encoder + VertexPolicy + RulePolicy + value_head`.
Two independent CLI flags swap each policy axis:

* vertex selection (`--no-ptr` toggles)
    * default          : `PointerVertexPolicy` — learned per-vertex queries
                          cross-attend to the encoded tokens.
    * `--no-ptr`       : `MLPVertexPolicy` — masked MLP over the encoded summary.
* rule emission (`--not-autoreg` toggles)
    * default          : `AutoregRulePolicy` — `RuleDecoder` scan emitting up to
                          `--max-rules` `(axis_pair, factor)` rules per vertex.
    * `--not-autoreg`  : `SingleRulePolicy` — single per-vertex sp head, one rule
                          per vertex with factor fixed to -1.

All four combinations are valid; they share the same trajectory layout, GAE
machinery, and PPO loss. Every hyperparameter (network widths, optimisation,
PPO knobs, factor table, etc.) is behind a CLI argument.
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys

# tqdm allocates a multiprocessing.RLock on first use (`TqdmDefaultWriteLock`)
# for cross-process bar coordination. The RLock is backed by a named POSIX
# semaphore on macOS / Linux; if the process is signal-killed (SIGTERM from
# `timeout`, Ctrl-C, OOM killer) before tqdm's atexit cleanup runs, the
# semaphore leaks and `multiprocessing.resource_tracker` prints:
#   "There appear to be N leaked semaphore objects to clean up at shutdown"
# at the next Python shutdown. We only ever drive tqdm from the main thread
# of a single process, so a threading.RLock is sufficient and never touches
# `multiprocessing`.
import threading as _threading
from functools import partial
from typing import NamedTuple

import distrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
from tqdm import tqdm

import wandb

tqdm.set_lock(_threading.RLock())

from alphagrad.approx.common import (
    NUM_VERTEX_FEATURES,
    OP_TYPE_VOCAB_SIZE,
    build_pair_valid_mask,
    build_vertex_valid_static,
    data_gen,
    generate_eval_samples,
    get_advantages,
    make_get_advantages,
    get_args,
    get_fn,
    get_num_clipping_triggers,
    infer_argnums,
    init_linear_weights,
    inverse_reward_normalization_fn,
    reward_normalization_fn,
    scale_module_weight,
    shuffle_and_batch,
    shuffle_and_batch_by_trajectory,
    vertex_avail_at_step,
)
from alphagrad.approx.common.schedules import cosine_warmup_exp_decay_lr
from alphagrad.approx.env import (
    quality_metric as _env_quality_metric,
    _AXIS_FEAT_GROUP_ID,
    consume_degenerate_plan_count,
    consume_truncated_plan_count,
    consume_untraceable_plan_count,
    consume_zero_work_plan_count,
    consume_per_face_stats,
    consume_tokenization_truncation_stats,
    consume_memory_compression_stats,
    consume_static_peak_fallbacks,
    _AXIS_FEAT_IS_COMPRESSED,
    _AXIS_FEAT_IS_OUTPUT,
    _AXIS_FEAT_SIZE,
    FACE_SLOTS,
    MAX_AXES_PER_VERTEX,
    MAX_FACES as ENV_MAX_FACES,
    MAX_RULES_PER_VERTEX,
    MAX_DELTA_TOKENS,
    consume_token_length_stats,
    COMPUTE_REWARD_INDICES,
    NUM_AXIS_PAIRS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    SENTINEL_COST,
    StepAction,
    VertexEliminationEnv,
    micro_actions_to_rule_specs_jax,
)
from alphagrad.approx.common import carry_stream as _carry_stream
from alphagrad.approx.common.face_driver import (
    bind_step_callbacks,
    build_live_face_stream,
    make_face_callbacks,
)
from alphagrad.approx.unified_face_policy import UnifiedFacePolicy
from alphagrad.approx.heads import (
    COMPRESS_KINDS,
    MAX_EXPONENT,
    MAX_PRIMES,
    NUM_COMPRESS_KINDS,
    NUM_OPS,
    NUM_QUANT_DTYPES,
    OP_COMPRESS,
    OP_DIAG,
    OP_END,
    OP_QUANT,
    QUANT_DTYPES,
    AxisTokenFeatures,
    FaceAction,
    FacePathPolicy,
    FactorTables,
    MicroAction,
    MicroActionPolicy,
    precompute_factor_tables,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder, make_encoder
from alphagrad.approx import vertex_memory as _vmem
from alphagrad.transformer.encoder import RelationalMultiheadAttention
from alphagrad.transformer.palimpsa_encoder import (
    palimpsa_beta as _pal_beta, palimpsa_qk_norm as _pal_qk_norm)
from alphagrad.approx.common import delta_fold as _fold
from alphagrad.approx.common import feature_probe as _fprobe

# Same switch carry_stream reads, so one flag turns the whole fold on or
# off rather than leaving half the consumers folded and half not.
_FOLD_DELTA = os.environ.get("ALPHAGRAD_FOLD_DELTA", "1") != "0"
from alphagrad.utils import entropy, explained_variance

# ---------------------------------------------------------------------------
# PHASE-0 POLICY-PATH ATTRIBUTION  (ALPHAGRAD_PROFILE_POLICY=1)
# ---------------------------------------------------------------------------
# env.py's `_PROF` covers the HOST phases only (`cb.*`, `faces.live_chunk`,
# `oracle.*`). Everything else in an episode was a single undifferentiated
# residual, and the per-decision policy cost was INFERRED from it by
# subtraction rather than measured. This attributes it.
#
# Why callbacks and not a wall timer: `train_episode` is ONE
# `eqx.filter_jit` region -- the rollout scan AND the PPO update epochs are
# inside it -- so a host timer around the dispatch can only ever see the
# episode total. The only way to get a boundary inside is to drop a host
# timestamp there:
#
#   * an `io_callback` whose OPERAND is a cheap scalar reduction of the
#     phase's output, so it cannot be scheduled before the phase finished
#     and (being effectful) cannot be dead-code-eliminated -- a
#     `pure_callback` here IS eliminated, because the only consumer of its
#     result is an `optimization_barrier` output we drop, and JAX's DCE rule
#     for that primitive drops the matching operand with it. Measured: 0
#     host calls with `pure_callback`, 2 per mark with `io_callback`.
#   * plus an `optimization_barrier` across the phase's outputs, so XLA
#     cannot fuse work across the boundary and blur the two segments, and
#   * the callback's RESULT -- an opaque runtime EXACT ZERO -- added into
#     every array the phase produced. That add is a numeric identity but XLA
#     cannot prove it (the value comes from the host), so every consumer of
#     the phase's output is forced to wait for the mark. WITHOUT this the
#     barrier alone is not enough: its token output is unused, JAX's DCE rule
#     for `optimization_barrier` then drops the matching OPERAND, and the
#     ordering constraint disappears -- the first version of this measured
#     0.4 ms for an encode whose cost had been hoisted into the neighbouring
#     segment.
#
# A mark is a REAL serialization of the device stream: it costs
# a device->host->device round trip and it removes the overlap XLA would
# otherwise get between adjacent phases. It is therefore strictly a
# measurement mode, OFF BY DEFAULT, and the gate is read at TRACE time --
# with it off not one extra HLO op is emitted and the compiled graph is
# byte-identical to the uninstrumented one.
#
# Convention follows env.py's `_pf`: a mark CLOSES the segment since the
# previous mark and attributes it to the mark's key. `key=None` only resets
# the clock (used once per episode so the first segment isn't charged with
# whatever preceded the episode).
_PROFILE_POLICY = os.environ.get("ALPHAGRAD_PROFILE_POLICY", "0") == "1"
_PP_LAST: list = [None]
_PP_SINK: list = [None, None, None]


def _pp_sinks():
    if _PP_SINK[0] is None:
        from alphagrad.approx.env import _prof_add, _prof_sample, _trace
        _PP_SINK[0], _PP_SINK[1], _PP_SINK[2] = _prof_add, _prof_sample, _trace
    return _PP_SINK


def _pp_mark_host(key, x):
    import time as _t
    now = _t.perf_counter()
    prev = _PP_LAST[0]
    _PP_LAST[0] = now
    if _PP_SINK[0] is None:
        _pp_sinks()
    if _PP_SINK[2] is not None:
        _PP_SINK[2](f"mark:{key}")
    if prev is not None and key is not None:
        dt = now - prev
        _add, _samp = _pp_sinks()[:2]
        _add(key, dt)
        _samp(key, dt)
    a = np.asarray(x)
    # io_callback's batching rule calls the host ONCE PER BATCH ELEMENT, so
    # with num_envs > 1 the marks of the E envs interleave and the running
    # clock splits one phase across E entries. The per-decision numbers are
    # therefore only literal at --num-envs 1 (the flagship config); at E > 1
    # read the per-key TOTALS, not the per-sample mean.
    return np.float32(0.0) if a.ndim == 0 else np.zeros(a.shape[:1], np.float32)


def _pp_mark(key, x):
    """Timestamp the instant `x` is ready; return `x`, barriered.

    No-op (and zero HLO) unless ALPHAGRAD_PROFILE_POLICY=1.
    """
    if not _PROFILE_POLICY:
        return x
    flat, treedef = jax.tree_util.tree_flatten(x)
    arrs = [l for l in flat if hasattr(l, "shape") or hasattr(l, "dtype")]
    if not arrs:
        return x
    # Anchor: a scalar the phase's output must be computed to produce.
    #
    # EVERY leaf, not the first four. With a partial anchor the mark is free
    # to fire before the leaves it skipped are ready, and XLA takes that
    # freedom: `prof/envstep` anchored on EnvState's leading (device-only)
    # leaves fired while the host callback that fills the token leaves was
    # still running, so the callback's cost landed in the NEXT mark and the
    # buckets stopped being nestable at all (measured: `prof/measure_wait`
    # totalled MORE than the `prof/envstep` it was supposed to sit inside).
    # A sum over all leaves is a reduction over a few MB -- microseconds on
    # device, and far below any phase being timed.
    acc = jnp.zeros((), jnp.float32)
    for leaf in arrs:
        acc = acc + jnp.sum(jnp.asarray(leaf).astype(jnp.float32))
    from jax.experimental import io_callback as _io_callback
    tok = _io_callback(
        partial(_pp_mark_host, key),
        jax.ShapeDtypeStruct((), jnp.float32),
        acc,
    )
    out = list(lax.optimization_barrier(tuple(flat) + (tok,)))
    tok2, leaves = out[-1], out[:-1]
    gated = []
    for leaf in leaves:
        a = jnp.asarray(leaf) if hasattr(leaf, "dtype") else leaf
        if hasattr(a, "dtype") and (jnp.issubdtype(a.dtype, jnp.floating)
                                    or jnp.issubdtype(a.dtype, jnp.integer)):
            gated.append(a + tok2.astype(a.dtype))
        else:
            gated.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, gated)


def _pp_summary(samples: dict) -> str:
    """`key n=.. mean=..ms p95=..ms tot=..s` for each timed phase."""
    if not samples:
        return ""
    rows = []
    for k, v in sorted(samples.items(), key=lambda kv: -sum(kv[1])):
        a = np.asarray(v, np.float64)
        rows.append(
            f"{k} n={a.size} mean={a.mean() * 1e3:.1f}ms "
            f"p95={np.percentile(a, 95) * 1e3:.1f}ms "
            f"max={a.max() * 1e3:.1f}ms tot={a.sum():.1f}s")
    return "\n  ".join(rows)


def _pp_dist_summary(dists: dict, caps: dict) -> str:
    """median / mean / p95 / max / occupancy for each measured distribution."""
    if not dists:
        return ""
    rows = []
    for k, v in sorted(dists.items()):
        a = np.asarray(v, np.float64)
        cap = caps.get(k)
        occ = f" occ={a.mean() / cap * 100:.3f}%(cap={cap})" if cap else ""
        # Coarse decile histogram against the cap (or the observed max).
        top = float(cap) if cap else max(a.max(), 1.0)
        hist, _ = np.histogram(a, bins=10, range=(0.0, top))
        rows.append(
            f"{k} n={a.size} median={np.median(a):.0f} mean={a.mean():.1f} "
            f"p95={np.percentile(a, 95):.0f} p99={np.percentile(a, 99):.0f} "
            f"max={a.max():.0f}{occ}\n      hist[0..{top:.0f}]="
            + ",".join(str(int(h)) for h in hist))
    return "\n  ".join(rows)


# ---------------------------------------------------------------------------
# Constants shared by all three agent variants
# ---------------------------------------------------------------------------

# NUM_AXIS_PAIRS axis pairs + 1 STOP token marking end of a rule sequence.
NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS

# Three-head value/advantage configuration. The value head emits one scalar
# per training reward — (V_latency, V_mem, V_cos) — and the per-episode
# preference vector `w` (stored on Trajectory.preference) weights these
# advantages when scalarizing for the PPO loss. The mapping into the env's
# 8-component reward vector is fixed:
#   head 0  latency_ns     (REWARD_INDEX["latency_ns"])
#   head 1  peak_memory    (REWARD_INDEX["peak_memory"])
#   head 2  cosine_sim     (REWARD_INDEX["cosine_sim"])
# HISTORY: this was 3 heads (flops, mem, frob) with the frob head LABELED
# "acc", so cosine similarity — the accuracy metric everyone watches — never
# entered the training signal at all, and zeroing the computation maximized 2
# of 3 trained channels (run 55403). It then ran as 4 heads with cosine and
# frob side by side. It is 3 again, but now the QUALITY head is cosine and
# frob is gone entirely: the two measured the same thing on the same pair of
# Jacobians, and a plan could trade one against the other. The env still emits
# the frob_residual slot (the sentinel wire format keys on it) but nothing
# reads it.
# The remaining env-reward components are still emitted for host-side
# logging / top-N heaps but do not enter the value head or advantage path.
HEAD_REWARD_INDICES: tuple[int, ...] = (
    # Spec: the reward is measured latency + peak memory + cossim. flops
    # (XLA cost analysis) used to sit in this slot as a latency proxy — the
    # policy then optimized analyzed flops while measured latency drifted
    # free. Requires --measure-latency (the channel is 0 without it).
    REWARD_INDEX["latency_ns"],
    REWARD_INDEX["peak_memory"],
    REWARD_INDEX["cosine_sim"],
)
NUM_VALUE_HEADS = len(HEAD_REWARD_INDICES)
# Resolved from --quality-metric in `main`; names the quantity reward slot 6
# actually holds, for every human-readable log line and the wandb
# ``quality/metric`` key.
_QUALITY_METRIC: str = "quality"
HEAD_NAMES: tuple[str, ...] = ("latency", "mem", "quality")
# Slot 2 was named "cos" until 2026-08-07; it is the value head for reward
# slot 6, which now holds whichever quality metric env.quality_metric()
# selects (the 200-step Adam-walk loss drop by default under --measure-grad,
# the legacy Jacobian cosine otherwise) -- so the wandb keys built from this
# tuple (popart/mu_*, popart/sigma_*, weighted_mean_*) no longer claim
# "cosine" for a number that is not one. The eqx MODULE ATTRIBUTE keeps its
# historical spelling ``value_head_cos``: renaming it would change the
# pytree structure and invalidate every saved checkpoint for a cosmetic win.

# Print every loss component the moment the total goes non-finite. Off by
# default because it forces a host callback inside the jitted update.
_DEBUG_NAN = os.environ.get("ALPHAGRAD_DEBUG_NAN", "0") == "1"
# --- ORDER-SEARCH DIAGNOSTICS (task: PPO/GAZ vs POMO on the order space) ---
# Both default OFF, so the trained path is byte-identical unless asked for.
#   ALPHAGRAD_ADV_DIAG=1        -> one host callback per episode capturing the
#     spread of the RAW and NORMALISED advantages, the GAE return targets and
#     the critic's own predictions. The prior TLM order run reported explained
#     variance -1.105 (critic worse than the mean), which is a statement about
#     exactly these arrays, and none of them were observable.
#   ALPHAGRAD_UPDATE_JSONL=<f>  -> append one JSON record per episode with the
#     complete log_dict (scalars), the per-env elimination ORDERS and the raw
#     terminal reward vectors, so a best-so-far-vs-unique-measurements curve
#     and an order-diversity count can be built offline. wandb is not a
#     substitute: --lean-logging drops every measure/<channel>/*_ep key.
_ADV_DIAG = os.environ.get("ALPHAGRAD_ADV_DIAG", "0") == "1"
_ADV_STATS = {}
_UPD_JSONL = os.environ.get("ALPHAGRAD_UPDATE_JSONL", "")


def _spread(prefix, x):
    """min/max/mean/std/absmax of a finite-masked array, as a flat dict."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f = np.isfinite(x)
    out = {prefix + "/n": int(x.size),
           prefix + "/n_nonfinite": int(x.size - int(f.sum()))}
    if not f.any():
        return out
    y = x[f]
    out.update({prefix + "/mean": float(y.mean()),
                prefix + "/std": float(y.std()),
                prefix + "/min": float(y.min()),
                prefix + "/max": float(y.max()),
                prefix + "/absmax": float(np.abs(y).max()),
                prefix + "/n_zero": int((y == 0.0).sum())})
    return out
# How many leading episodes print the stdout health line (see its use site).
_HEALTH_N = [0]

# Per-step vertex-pick trace (see the jax.debug.print below). Off by default:
# it forces a host callback inside the jitted rollout scan.
_DEBUG_ORDER = os.environ.get("ALPHAGRAD_DEBUG_ORDER", "0") == "1"

# PopArt decodes the value head itself (value * sigma + mu), so GAE must
# NOT symexp on top of that. See the call site for why this only bites on
# the second update.
_GAE_POPART = make_get_advantages(use_symlog=False)
_HEAD_REWARD_INDICES_ARR = jnp.asarray(HEAD_REWARD_INDICES, dtype=jnp.int32)

# Cross-channel scale handling. Reward channels span ~10¹⁰ in flops, ~10⁹
# in peak_memory, ~1 in cosine_sim. Without
# per-channel normalization the flops gradient (1e10) drowns out cosine_sim
# (1) by ten orders of magnitude, and the policy learns to ignore quality.
# Fix is two-fold:
#   1) symlog every monotone channel before the per-step weighted sum;
#      compresses the dynamic range to ~25 / ~21 / ~5 / ~1 across channels.
#   2) calibration measures mean |symlog(r_i)| over K rollouts of the
#      un-trained agent and rescales reward_weights[i] by 1/mean_abs_i so
#      each weighted channel contributes on a comparable scale.
# cosine_sim used to be exempted from (1) on the grounds that it is already
# bounded to [0, 1]. It is no longer exempt. With frob gone cosine is the ONLY
# quality channel, and PopArt is seeded and updated per channel from the same
# sampled returns — an exempt channel is seeded in one space and normalised in
# another, which is exactly the raw-vs-symlog seeding bug that cost ~500
# episodes of EMA correction. Symlog on [0, 1] is a near-identity anyway
# (symlog(1) = 0.693), so nothing about the CLI lambda's units really moves.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = ()
_NO_SYMLOG_MASK: "jax.Array" = (
    jnp.zeros((NUM_REWARDS,), dtype=jnp.bool_)
    .at[jnp.asarray(_NO_SYMLOG_REWARD_INDICES, dtype=jnp.int32)]
    .set(True)
)  # currently all-False: every channel is normalised the same way
_NO_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=np.bool_)
if _NO_SYMLOG_REWARD_INDICES:
    _NO_SYMLOG_MASK_NP[list(_NO_SYMLOG_REWARD_INDICES)] = True



def _traced_inlined(target_fn, xs):
    """``jax.make_jaxpr(target_fn)(*xs)``, numbered on the form that is actually
    eliminated.

    jacve and the AOJ splice jit/pjit bodies into the parent jaxpr before
    eliminating, which ADDS equations. Numbering vertices from the raw trace
    therefore addresses a different graph -- the order misses every spliced-in
    vertex, and the elimination refuses ("the elimination order left N
    intermediate vertices with live edges un-eliminated") rather than quietly
    returning a Jacobian with those paths dropped.
    """
    import jax
    from graphax import inline_call_primitives

    cj = jax.make_jaxpr(target_fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj                      # nothing to inline -- keep the original
    try:                                   # jax >= 0.4.31
        from jax.extend.core import ClosedJaxpr
    except ImportError:                    # older / internal layout
        from jax._src.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


# Set once from --no-symlog BEFORE any jit tracing, so the traced graphs
# capture the decision as a constant. A one-element list rather than a bare
# global so the setter does not need `global` in every scope.
_NO_SYMLOG_ALL: list = [False]


def _symlog_rewards(reward_vec: "jax.Array") -> "jax.Array":
    """Apply symlog elementwise on the last axis (see ``_NO_SYMLOG_MASK``).

    ``reward_vec`` has trailing dim ``NUM_REWARDS``. The mask is broadcast
    against any leading batch / time dims so the call is shape-agnostic. The
    mask is presently empty (cosine is normalised like everything else); the
    machinery stays so a channel can be exempted again without a refactor.

    Under ``--no-symlog`` this is the IDENTITY: PopArt already normalises each
    channel by its own running sigma, which is the same job symlog was doing,
    and stacking the two pushes the per-channel spread under ``sigma_min``
    (measured: mem sigma 0.00694 vs a 0.1 floor) so the channel gets shrunk
    instead of scaled.
    """
    if _NO_SYMLOG_ALL[0]:
        return reward_vec
    return jnp.where(
        _NO_SYMLOG_MASK,
        reward_vec,
        reward_normalization_fn(reward_vec),
    )


def _value_target(x: "jax.Array") -> "jax.Array":
    """The value head's regression target.

    Under PopArt the returns are already built from whatever space the rewards
    live in, so applying ``reward_normalization_fn`` here symlogs them A SECOND
    TIME. Identity under --no-symlog; unchanged otherwise so the legacy path is
    bit-identical.
    """
    return x if _NO_SYMLOG_ALL[0] else reward_normalization_fn(x)


def _apply_mult_gate(
    rewards: "jax.Array",
    cost_weights: "jax.Array",
    gate_tau: float,
    gate_w: float,
    anti_degen_penalty: float,
    anti_degen_tau: float,
    gate_fidelity: str = "cos",
) -> "jax.Array":
    """Multiplicative cosine-gate reward (``--reward-mode mult``).

    Ported from the ray worker's ``_apply_mult_reward_gate`` (the structural
    anti-collapse design), jnp-ified for the in-jit reward path::

        g(cos)    = clip((cos - tau) / (1 - tau), 0, 1)        # fidelity gate
        cheapness = max(0, W - sum_c w_c * symlog(cost_c))     # >0 == cheap
        reward    = g(cos) * cheapness

    ``rewards`` is ``(E, T, NUM_REWARDS)`` raw env emission (costs stored
    negated, cosine in [0, 1]). The gated scalar lands in the cosine channel
    and every other channel is zeroed — with a one-hot-cosine preference the
    scalarization recovers it exactly. cosine→0 ⇒ reward→0 regardless of
    cheapness, which structurally kills the cost→0/cos→0 hack.

    ANTI-DEGENERACY: a flat-0 plateau below tau would let the policy drift
    into the basin and never climb out, so degenerate TERMINAL transitions
    (cos < ``anti_degen_tau``) get a strictly negative SHAPED penalty with a
    positive slope in cos (−P at cos=0 ramping toward −P·(1−tau_d)), giving a
    gradient pointing out of the basin while staying below every valid gated
    reward (which is ≥ 0). Non-terminal steps legitimately carry cos=0
    (sparse-terminal quality) and stay at the gated 0.
    """
    # Fidelity source is cosine similarity. frob is gone: it was the same
    # comparison of the same two Jacobians, and carrying both let a plan trade
    # one against the other. Only the TERMINAL step carries a real value
    # (sparse-terminal quality), so fidelity is masked to the terminal step —
    # a mid-rollout cos=0 must not read as a degenerate plan.
    terminal = jnp.zeros(rewards.shape[:2], dtype=bool).at[:, -1].set(True)
    fid_raw = rewards[..., REWARD_INDEX["cosine_sim"]]
    fid = jnp.where(terminal, jnp.clip(fid_raw, 0.0, 1.0), 0.0)  # (E, T)
    denom = jnp.maximum(1.0 - gate_tau, 1e-6)
    g = jnp.clip((fid - gate_tau) / denom, 0.0, 1.0)

    cost_mag = -rewards                                          # (E, T, R)
    cost_sl = jnp.sign(cost_mag) * jnp.log1p(jnp.abs(cost_mag))
    weighted_cost = jnp.sum(cost_sl * cost_weights, axis=-1)     # (E, T)
    cheapness = jnp.maximum(0.0, gate_w - weighted_cost)
    gated = g * cheapness

    # Shaped anti-degeneracy penalty on the TERMINAL step only, keyed on the
    # same fidelity as the gate, so the basin has a continuous slope and a
    # marginally-less-destroyed Jacobian scores strictly better — which is what
    # lets plans train OUT of all-zeros.
    degen = (fid < anti_degen_tau) & terminal
    fid_basin = jnp.clip(fid, 0.0, anti_degen_tau)
    shaped = -(anti_degen_penalty - fid_basin * anti_degen_penalty)
    gated = jnp.where(degen, shaped, gated)

    out = jnp.zeros_like(rewards)
    return out.at[..., REWARD_INDEX["cosine_sim"]].set(gated)


def _popart_derive(m1, m2, w, sigma_min, sigma_max):
    """Debiased (mu, sigma) from the raw EMA accumulators.

    The state carried across episodes is the RAW accumulators (m1, m2, w);
    debiasing happens only at point of use. Returning the debiased value and
    feeding it back as the accumulator double-counts and the scale explodes
    (mu ran 1e7 -> 5e8 in two updates before this split).
    """
    wc = jnp.maximum(w, 1e-8)
    mu = m1 / wc
    var = m2 / wc - jnp.square(mu)
    sigma = jnp.clip(jnp.sqrt(jnp.maximum(var, 1e-12)), sigma_min, sigma_max)
    # Before the first update (w == 0) fall back to the identity transform so
    # de/re-normalisation is a no-op rather than a divide-by-noise.
    warm = w > 1e-8
    return jnp.where(warm, mu, 0.0), jnp.where(warm, sigma, 1.0)


def _popart_update(m1, m2, w, returns, beta, sigma_min, sigma_max, winsor_k):
    """One debiased-EMA PopArt step on the RAW accumulators, jax-native so it
    runs inside the jit.

    ``returns`` is ``(E, T, K)`` raw per-channel value targets. Mirrors
    ``common.popart.PopArtStats`` (numpy/host-side, hence unusable inside
    ``train_episode``): winsorize each channel to ``mu +/- winsor_k*sigma``
    so one extreme cost outlier can't spike a channel's sigma and crush the
    others' relative advantage; debias with the Adam-style ``w`` accumulator
    so the FIRST update adopts the batch stats exactly instead of crawling
    away from the (0, 1) init.

    Returns the new ``(m1, m2, w)``.
    """
    mu, sigma = _popart_derive(m1, m2, w, sigma_min, sigma_max)
    flat = returns.reshape(-1, returns.shape[-1])                 # (B, K)
    # Winsorize against the current stats — but only once they mean something.
    # On the first update the stats are the arbitrary (0, 1) init, so clipping
    # would crush a 1e7-scale channel to +/-5 and debiasing could never
    # recover the true scale.
    warm = w > 1e-8
    lo, hi = mu - winsor_k * sigma, mu + winsor_k * sigma
    flat = jnp.where(warm, jnp.clip(flat, lo, hi), flat)
    batch_m1 = jnp.mean(flat, axis=0)
    batch_m2 = jnp.mean(jnp.square(flat), axis=0)
    new_m1 = m1 * (1.0 - beta) + batch_m1 * beta
    new_m2 = m2 * (1.0 - beta) + batch_m2 * beta
    new_w = w + beta * (1.0 - w)
    return new_m1, new_m2, new_w


def _popart_rescale_heads(agent, old_mu, old_sigma, new_mu, new_sigma):
    """Output-preserving rescale of the three single-output value heads.

    ``sigma'*head'(x) + mu' == sigma*head(x) + mu`` for every x, so shifting
    the normalisation does not perturb the critic's predictions (the "ART" in
    PopArt). ``common.popart.popart_rescale_mlp_head`` assumes ONE head with K
    output rows; ours are K separate 1-row MLPs, so apply it per head.
    """
    heads = ("value_head_flops", "value_head_mem", "value_head_cos")
    for k, name in enumerate(heads):
        mlp = getattr(agent, name)
        seq = mlp.layers.layers
        li = max(i for i, l in enumerate(seq) if isinstance(l, eqx.nn.Linear))
        lin = seq[li]
        ratio = (old_sigma[k] / new_sigma[k]).astype(lin.weight.dtype)
        new_w = lin.weight * ratio
        new_b = ((old_sigma[k] * lin.bias + old_mu[k] - new_mu[k])
                 / new_sigma[k]).astype(lin.bias.dtype)
        mlp = eqx.tree_at(
            lambda m, _li=li: (m.layers.layers[_li].weight,
                               m.layers.layers[_li].bias),
            mlp, (new_w, new_b),
        )
        agent = eqx.tree_at(lambda a, _n=name: getattr(a, _n), agent, mlp)
    return agent

# Mapping from pair index 0..NUM_AXIS_PAIRS-1 -> (base_idx1, base_idx2). STOP is unused.
_PAIR_TO_BASE = jnp.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [-1, -1],  # STOP sentinel
    ],
    dtype=jnp.int32,
)

# In the legacy 5-action policy, sp_type ∈ {0..4} where 0 = no-rule (dense).
# Conversion to pair-index space (0..3 = real pairs, PAIR_STOP = no-rule):
#   sp_type 0 -> PAIR_STOP, sp_type 1..4 -> pair 0..3
_SP_TYPE_TO_PAIR = jnp.array([PAIR_STOP, 0, 1, 2, 3], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Trajectory / TrainBatch — same shape across all agent variants. The simpler
# (non-autoregressive) agents fill in degenerate distributions for the slots
# they don't use, so the rollout / loss / GAE code stays uniform.
# ---------------------------------------------------------------------------


def _zero_micro_action(max_substeps):
    """Canonical inactive MicroAction: every sub-step END, nothing applied."""
    S = max_substeps
    z = jnp.zeros((S,), jnp.int32)
    return MicroAction(
        op_type=jnp.full((S,), OP_END, dtype=jnp.int32),
        i=z, j=z,
        exponents=jnp.zeros((S, MAX_PRIMES), jnp.int32),
        factor=z, compress_kind=z, quant_dtype=z,
        quant_scale_sign=jnp.ones((S,), jnp.int32),
        quant_scale_frac=jnp.zeros((S,), jnp.float32),
    )


def _zero_micro_dists(max_substeps, n_axes):
    """Point-mass dists matching the inactive action, so every KL term is 0."""
    S = max_substeps
    op_d = jnp.broadcast_to(
        jnp.zeros((NUM_OPS,)).at[OP_END].set(1.0)[None, :], (S, NUM_OPS))
    ij_d = jnp.broadcast_to(
        jnp.zeros((n_axes,)).at[0].set(1.0)[None, :], (S, n_axes))
    exp_d = jnp.broadcast_to(
        jnp.zeros((MAX_PRIMES, MAX_EXPONENT + 1)).at[:, 0].set(1.0)[None, ...],
        (S, MAX_PRIMES, MAX_EXPONENT + 1))
    kind_d = jnp.broadcast_to(
        jnp.zeros((NUM_COMPRESS_KINDS,)).at[0].set(1.0)[None, :],
        (S, NUM_COMPRESS_KINDS))
    return op_d, ij_d, ij_d, exp_d, kind_d


def _zero_face_action():
    """Canonical inactive FaceAction (padding faces: no skip, END slots)."""
    F, S = ENV_MAX_FACES, FACE_SLOTS
    z2 = jnp.zeros((F, S), jnp.int32)
    return FaceAction(
        skip=jnp.zeros((F,), jnp.int32),
        op_type=jnp.full((F, S), OP_END, dtype=jnp.int32),
        i=z2, j=z2,
        exponents=jnp.zeros((F, S, MAX_PRIMES), jnp.int32),
        factor=z2, compress_kind=z2, quant_dtype=z2,
        quant_scale_sign=jnp.ones((F, S), jnp.int32),
        quant_scale_frac=jnp.zeros((F, S), jnp.float32),
    )


# ---------------------------------------------------------------------------
# Phase 3b — incremental autoregressive encode (--incremental-encode).
#
# The palimpsa recurrence is causal, so the encoder state after consuming the
# append-only stream up to step t is a small carry: per-layer (M, I) plus the
# causal relational-gate histogram. The rollout extends the carry by each
# step's DELTA tokens only (O(delta) instead of O(S) re-encode), folds the new
# rows into the per-vertex memory (vertex_memory.py), and the pointer/value
# heads read from that memory. The trajectory stores each step's PRE-step
# carry so the loss re-derives the encoding by the SAME delta extension —
# the stored-context ratio-1 pattern (gradient truncates at the stored
# carry, i.e. flows through the last delta only; accepted by design).
#
# MAX_EQNS bounds the causal relational-gate histogram (eqn ids at or above
# it share the top bucket — an approximation only in overflow, identical on
# both rollout and loss so ratio-1 is unaffected).
# ---------------------------------------------------------------------------
MAX_EQNS = int(os.environ.get("ALPHAGRAD_MAX_EQNS", "4096"))


class EncCarry(NamedTuple):
    M: jax.Array        # (L, H, d, d) float32 — per-layer palimpsa numerator
    I: jax.Array        # (L, H, d, d) float32 — per-layer palimpsa precision
    cumhist: jax.Array  # (MAX_EQNS,) float32 — cumulative eqn-id counts (<= id)
    nvalid: jax.Array   # () float32 — valid (eqn_id >= 0) tokens consumed
    pos: jax.Array      # () int32 — stream position consumed so far


_FACE_MASK_FAILS = [0]


# ---------------------------------------------------------------------------
# DELTA BUFFERS (stage 2: the ONLY path).
#
# Nothing in the pipeline ever re-read earlier tokens -- the rollout consumed
# `stream[pos : pos + delta]` and the loss re-derived the SAME window from the
# stored per-step carry -- so the growing (MAX_TOKENS,) buffer was pure
# artefact of the cursor being ABSOLUTE. The env now emits each step's delta
# DIRECTLY, with the tokenizer's own length (`EnvConfig.delta_obs`), and the
# base stream is a host-side constant (`env.base_observation()`). The vertex
# stream therefore has exactly the shape the FACE stream always had: a
# standalone buffer read from 0 with an explicit count.
#
# `_stream_end` / `_window_copy` below are no longer on any live path. They
# are the REFERENCE DEFINITION of the window the old absolute cursor read,
# and tests/delta_buffer_equivalence_test.py drives them against real
# tokenizer streams to keep that description pinned to the buffers the env
# now emits. Deleting them would delete the proof, not dead code.
# ---------------------------------------------------------------------------


def _stream_end(tokens):
    """Index one past the LAST non-zero token of an append-only buffer.

    ``_stream_len`` COUNTS non-zeros, so an interior id 0 -- the literal '-'
    graphax emits for any negative value -- makes it undercount and the delta
    window comes up short (the hazard documented on ``_stream_len``). An
    append-only stream is contiguous with a zero-padded TAIL, so "one past the
    last non-zero" is the same number when there is no interior zero and the
    RIGHT number when there is. Same cost, one reduction.
    """
    nz = tokens != 0
    n = tokens.shape[-1]
    last = n - 1 - jnp.argmax(nz[::-1])
    return jnp.where(jnp.any(nz), last + 1, 0).astype(jnp.int32)


def _window_copy(tokens_buf, eqn_ids_buf, pos, count, width):
    """``stream[pos : pos + width]`` as a STANDALONE ``(width,)`` buffer pair
    with a RELATIVE cursor (read it back with ``encode_extend(..., start=0)``).

    Entries at or past ``count`` are written as pad (0 / -1), so the buffer
    holds exactly the delta and then padding. That is bitwise what the
    absolute cursor read out of the growing stream, because an append-only
    stream past its own length is already pad -- which is the token-for-token
    equality tests/delta_buffer_equivalence_test.py asserts.
    """
    pos = jnp.asarray(pos, jnp.int32)
    count = jnp.clip(jnp.asarray(count, jnp.int32), 0, width)
    ar = jnp.arange(width, dtype=jnp.int32)
    idx = pos + ar
    toks = jnp.take(tokens_buf, idx, mode="fill",
                    fill_value=0).astype(jnp.int32)
    eqns = jnp.take(eqn_ids_buf, idx, mode="fill",
                    fill_value=-1).astype(jnp.int32)
    keep = ar < count
    return jnp.where(keep, toks, 0), jnp.where(keep, eqns, -1)


class Trajectory(NamedTuple):
    preference: jax.Array  # (NUM_VALUE_HEADS,) — weights V_latency/V_mem/V_cos
    vertex_idx: jax.Array
    # Legacy rule-head action — zero-filled in --dynamic-substeps mode.
    pair_seq: jax.Array
    factor_seq: jax.Array
    # Typed micro-action sequence — zero-filled in legacy mode. Shapes
    # depend on args.max_substeps + heads.py constants
    # (NUM_OPS, MAX_PRIMES, MAX_EXPONENT) and env's MAX_AXES_PER_VERTEX.
    micro_op_seq: jax.Array  # (max_substeps,) int32
    micro_i_seq: jax.Array  # (max_substeps,) int32
    micro_j_seq: jax.Array  # (max_substeps,) int32
    micro_exp_seq: jax.Array  # (max_substeps, MAX_PRIMES) int32
    micro_factor_seq: jax.Array  # (max_substeps,) int32
    micro_compress_kind_seq: jax.Array  # (max_substeps,) int32
    micro_quant_dtype_seq: jax.Array  # (max_substeps,) int32
    micro_quant_scale_sign_seq: jax.Array  # (max_substeps,) int32 — ±1
    micro_quant_scale_frac_seq: jax.Array  # (max_substeps,) float32 — scale head u
    reward: jax.Array  # (NUM_REWARDS,) — full env emission, kept for host logging
    done: jax.Array
    value: jax.Array  # (NUM_VALUE_HEADS,) per-head value prediction
    next_value: jax.Array  # (NUM_VALUE_HEADS,) per-head bootstrap value
    vertex_dist: jax.Array
    pair_dists: jax.Array
    factor_dists: jax.Array
    # Dynamic-substeps per-component distributions — zero-filled in legacy mode.
    micro_op_dists: jax.Array  # (max_substeps, NUM_OPS) float32
    micro_i_dists: jax.Array  # (max_substeps, MAX_AXES_PER_VERTEX) float32
    micro_j_dists: jax.Array  # (max_substeps, MAX_AXES_PER_VERTEX) float32
    micro_exp_dists: jax.Array  # (max_substeps, MAX_PRIMES, MAX_EXPONENT+1) float32
    micro_kind_dists: jax.Array  # (max_substeps, NUM_COMPRESS_KINDS) float32
    micro_quant_logp: jax.Array  # (max_substeps,) float32 — RAW factored-quant log-prob
    # Live oracle DIAG/COMPRESS masks for the CHOSEN vertex, stored so the loss
    # re-masks identically to the rollout (keeps the PPO ratio 1 at epoch 0).
    micro_pair_valid: jax.Array  # (MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX) float32
    micro_compress_valid: jax.Array  # (MAX_AXES_PER_VERTEX,) float32
    # Live axis features at sample time (pre-step state). The loss must
    # evaluate against these, not env.axis_state_static: once a DIAG/COMPRESS
    # lands, the static copy diverges from what the rollout sampled under and
    # the PPO ratio silently leaves 1 at epoch 0.
    axis_state: jax.Array  # (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM) int32
    axis_valid_mask: jax.Array  # (total_v, MAX_AXES_PER_VERTEX)
    # P1c per-path decisions (--face-actions; zero-filled otherwise). The
    # stored face masks are the loss's re-masking source (ratio-1), and
    # face_old_logp is the behaviour policy's joint face log-prob (skip gates
    # + slot heads) captured at sample time.
    face_skip: jax.Array          # (MAX_FACES,) int32
    face_op_type: jax.Array       # (MAX_FACES, FACE_SLOTS) int32
    face_i: jax.Array             # (MAX_FACES, FACE_SLOTS) int32
    face_j: jax.Array             # (MAX_FACES, FACE_SLOTS) int32
    face_exponents: jax.Array     # (MAX_FACES, FACE_SLOTS, MAX_PRIMES) int32
    face_factor: jax.Array        # (MAX_FACES, FACE_SLOTS) int32
    face_compress_kind: jax.Array # (MAX_FACES, FACE_SLOTS) int32
    face_quant_dtype: jax.Array   # (MAX_FACES, FACE_SLOTS) int32
    face_quant_scale_sign: jax.Array  # (MAX_FACES, FACE_SLOTS) int32
    face_quant_scale_frac: jax.Array  # (MAX_FACES, FACE_SLOTS) float32
    face_pair_valid: jax.Array    # (MAX_FACES, N, N) float32
    face_comp_valid: jax.Array    # (MAX_FACES, N) float32
    face_valid: jax.Array         # (MAX_FACES,) float32
    # The face's IDENTITY: its (in_edge, out_edge) endpoint vertices, 1-based
    # with 0 = "no vertex" (a jaxpr input). Stored for the same reason the
    # masks are: the loss must gather the SAME two endpoint contexts the
    # behaviour policy read, or the ratio is not 1 at epoch 0.
    face_endpoints: jax.Array     # (MAX_FACES, 2) int32
    face_old_logp: jax.Array      # () float32
    # Per-face chunk LENGTHS -- the only face-stream data the loss needs.
    # The chunks concatenate to exactly the step delta (pinned property), so
    # boundaries are the counts' cumsum and the loss pools the delta rows it
    # already computes in `_carry_heads`. The token windows that used to sit
    # here were the storage that made any face width expensive.
    face_counts: jax.Array        # (MAX_FACES,) int32
    # THIS step's emission (the current vertex's contractions + approx
    # echoes) -- the NEXT state's delta, i.e. what the chosen elimination
    # produced. Not the same buffer as `delta_tokens` below (that is the
    # PREVIOUS step's delta, the one this step's carry consumes), and it is
    # what the face chunks concatenate to -- the loss scans it once from the
    # stored carry's continuation and pools between chunk boundaries.
    face_delta_tokens: jax.Array  # (MAX_DELTA_TOKENS,) int32
    face_delta_eqns: jax.Array    # (MAX_DELTA_TOKENS,) int32
    # Phase 3b incremental encode (mandatory since stage 2). The PRE-step
    # encoder carry + vertex memory, and the delta's owner vertex —
    # everything the loss needs to re-derive this step's encoding by
    # extending with the delta tokens only.
    enc_M: jax.Array        # (L, H, d, d)
    enc_I: jax.Array        # (L, H, d, d)
    enc_cumhist: jax.Array  # (MAX_EQNS,)
    enc_nvalid: jax.Array   # ()
    enc_pos: jax.Array      # () int32
    vmem_sums: jax.Array    # (V+1, E)
    vmem_counts: jax.Array  # (V+1,)
    # THIS step's delta, straight off the env, as a standalone buffer read
    # from 0 plus its EXACT length -- the same shape the face stream stores.
    # This IS the observation: there is no episode-wide token buffer any
    # more, and no absolute cursor with which to index one.
    delta_tokens: jax.Array   # (MAX_DELTA_TOKENS,) int32
    delta_eqns: jax.Array     # (MAX_DELTA_TOKENS,) int32
    delta_count: jax.Array    # () int32
    delta_owner: jax.Array  # () int32 — vertex whose elimination emitted the delta
    # The slots that delta TOUCHES (vertex + its faces' endpoints), 0/1 over
    # (total_v + 1). Stored because the loss re-runs the same `advance` and
    # must credit the same slots the rollout did.
    delta_participants: jax.Array  # (total_v + 1,) float32
    discount: jax.Array
    vertex_avail_mask: jax.Array
    # FEATURE PROBE (ALPHAGRAD_FEATURE_PROBE=1, default OFF). ``None`` when the
    # probe is off, and None is not a pytree LEAF -- so the default path stores
    # no extra array, changes no shape, and adds nothing to the scan carry.
    # P = ALPHAGRAD_FEATURE_PROBE_FACES (<= the face bound): the face count is
    # ~1.7 against a bound of thousands, so the probe pays for a prefix rather
    # than for the padding.
    probe_targets: jax.Array = None    # (P, NFT) float32 host oracle targets
    probe_extents: jax.Array = None    # (P, MAX_AXES) float32 log2 axis sizes
    probe_valid: jax.Array = None      # (P,) float32
    probe_step: jax.Array = None       # () int32 elimination step index


class TrainBatch(NamedTuple):
    preference: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    micro_op_seq: jax.Array
    micro_i_seq: jax.Array
    micro_j_seq: jax.Array
    micro_exp_seq: jax.Array
    micro_factor_seq: jax.Array
    micro_compress_kind_seq: jax.Array
    micro_quant_dtype_seq: jax.Array
    micro_quant_scale_sign_seq: jax.Array
    micro_quant_scale_frac_seq: jax.Array
    old_vertex_dist: jax.Array
    old_pair_dists: jax.Array
    old_factor_dists: jax.Array
    old_micro_op_dists: jax.Array
    old_micro_i_dists: jax.Array
    old_micro_j_dists: jax.Array
    old_micro_exp_dists: jax.Array
    old_micro_kind_dists: jax.Array
    old_micro_quant_logp: jax.Array  # (max_substeps,) RAW factored-quant log-prob
    micro_pair_valid: jax.Array  # (MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX)
    micro_compress_valid: jax.Array  # (MAX_AXES_PER_VERTEX,)
    axis_state: jax.Array  # (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM) int32
    axis_valid_mask: jax.Array  # (total_v, MAX_AXES_PER_VERTEX)
    face_skip: jax.Array
    face_op_type: jax.Array
    face_i: jax.Array
    face_j: jax.Array
    face_exponents: jax.Array
    face_factor: jax.Array
    face_compress_kind: jax.Array
    face_quant_dtype: jax.Array
    face_quant_scale_sign: jax.Array
    face_quant_scale_frac: jax.Array
    face_pair_valid: jax.Array
    face_comp_valid: jax.Array
    face_valid: jax.Array
    face_endpoints: jax.Array
    face_old_logp: jax.Array
    face_counts: jax.Array
    face_delta_tokens: jax.Array
    face_delta_eqns: jax.Array
    enc_M: jax.Array
    enc_I: jax.Array
    enc_cumhist: jax.Array
    enc_nvalid: jax.Array
    enc_pos: jax.Array
    vmem_sums: jax.Array
    vmem_counts: jax.Array
    # WINDOWED (leading K axis, --grad-window; K=1 is the historical single
    # delta): the last K step deltas ending at THIS step, oldest first. The
    # enc_*/vmem_* above are the carry at the OLDEST of them.
    delta_tokens: jax.Array        # (K, MAX_DELTA_TOKENS)
    delta_eqns: jax.Array          # (K, MAX_DELTA_TOKENS)
    delta_count: jax.Array         # (K,)
    delta_owner: jax.Array         # (K,)
    delta_participants: jax.Array  # (K, total_v + 1)
    estim_returns: jax.Array
    norm_adv: jax.Array
    vertex_avail_mask: jax.Array
    # Feature-probe targets, threaded exactly like `delta_participants`.
    probe_targets: jax.Array = None
    probe_extents: jax.Array = None
    probe_valid: jax.Array = None
    probe_step: jax.Array = None


# ---------------------------------------------------------------------------
# Helpers used by every variant
# ---------------------------------------------------------------------------


def old_micro_log_prob_for_action(
    vertex_idx,
    op_seq,
    i_seq,
    j_seq,
    exp_seq,
    kind_seq,
    quant_seq,
    vertex_dist,
    op_dists,
    i_dists,
    j_dists,
    exp_dists,
    kind_dists,
    quant_logp,
):
    """Joint log-prob of a typed micro-action sequence under stored dists.

    Dynamic-substeps analog of :func:`old_log_prob_for_action`. The vertex
    log-prob plus the per-sub-step (op_type, i, j, prime-exponents,
    compress_kind, quant_dtype) contributions are summed, with the
    per-component activity gating matching :meth:`MicroActionHead.log_prob_step`:

    * i active for DIAG / COMPRESS.
    * j and prime-exponent active for DIAG only.
    * compress_kind active for COMPRESS only.
    * quant_dtype active for QUANT only.
    * Every component zeros out for sub-steps past the first OP_END
      (sticky termination — mirrors the scan's post-END mask).

    For padded primes in the exponent head, the stored distribution
    places ~1.0 mass on exponent=0 (the per-prime mask enforces it at
    sample time), so ``log(1.0 + 1e-8) ≈ 1e-8`` — the contribution from
    padded primes is negligible and we don't need to store a separate
    prime mask in the trajectory.
    """
    # Lazy import to keep the heads.py dependency one-way (heads → ppo via
    # the loss path, never ppo → heads at module load).
    log_p_v = jnp.log(vertex_dist[vertex_idx] + 1e-8)

    is_end = op_seq == OP_END
    prior_ends = jnp.cumsum(is_end.astype(jnp.int32)) - is_end.astype(jnp.int32)
    active = (prior_ends == 0).astype(jnp.float32)

    is_diag = (op_seq == OP_DIAG).astype(jnp.float32)
    is_compress = (op_seq == OP_COMPRESS).astype(jnp.float32)
    is_quant = (op_seq == OP_QUANT).astype(jnp.float32)
    is_diag_or_compress = ((op_seq == OP_DIAG) | (op_seq == OP_COMPRESS)).astype(
        jnp.float32
    )

    S = op_seq.shape[0]
    arange_s = jnp.arange(S)

    log_p_op = jnp.log(op_dists[arange_s, op_seq] + 1e-8) * active
    log_p_i = jnp.log(i_dists[arange_s, i_seq] + 1e-8) * active * is_diag_or_compress
    log_p_j = jnp.log(j_dists[arange_s, j_seq] + 1e-8) * active * is_diag

    # Per-prime gather: exp_dists has shape (S, MAX_PRIMES, MAX_EXPONENT+1).
    # take_along_axis gives (S, MAX_PRIMES, 1) → squeeze last dim →
    # (S, MAX_PRIMES). Sum across primes; padded primes contribute
    # log(~1.0) ≈ 0 because the head's mask forces exp=0 with prob 1.
    log_p_per_prime = jnp.log(
        jnp.take_along_axis(
            exp_dists,
            exp_seq[..., None],
            axis=-1,
        ).squeeze(-1)
        + 1e-8
    )
    log_p_exp = jnp.sum(log_p_per_prime, axis=-1) * active * is_diag
    log_p_kind = jnp.log(kind_dists[arange_s, kind_seq] + 1e-8) * active * is_compress
    # The factored quant head has no flat dtype dist to gather from; its RAW
    # per-step log-prob (dtype factors + scale_sign) was stored at rollout.
    log_p_quant = quant_logp * active * is_quant

    return log_p_v + jnp.sum(
        log_p_op + log_p_i + log_p_j + log_p_exp + log_p_kind + log_p_quant
    )


class PointerVertexPolicy(eqx.Module):
    """Vertex selection via cross-attention from learned per-vertex queries."""

    vertex_embedding: eqx.nn.Embedding
    cross_attn: eqx.nn.MultiheadAttention
    pointer_proj: eqx.nn.Linear

    num_vertices: int = eqx.field(static=True)

    def __init__(self, *, num_vertices, embd_dim, num_heads, key):
        keys = jrand.split(key, 3)
        self.num_vertices = num_vertices
        self.vertex_embedding = eqx.nn.Embedding(num_vertices, embd_dim, key=keys[0])
        self.cross_attn = eqx.nn.MultiheadAttention(num_heads, embd_dim, key=keys[1])
        self.pointer_proj = eqx.nn.Linear(embd_dim, 1, key=keys[2])

    def __call__(self, enc_x, token_mask):
        v_q = jax.vmap(self.vertex_embedding)(jnp.arange(self.num_vertices))
        attn_mask = jnp.broadcast_to(
            token_mask[None, :], (self.num_vertices, enc_x.shape[0])
        )
        vertex_reprs = self.cross_attn(v_q, enc_x, enc_x, mask=attn_mask)
        vertex_logits = jax.vmap(self.pointer_proj)(vertex_reprs).squeeze(-1)
        return vertex_logits, vertex_reprs

    def from_vertex_memory(self, vmem, vmask):
        """Same head, cross-attending over a per-vertex MEMORY (V+1, E).

        Identical weights and identical math to ``__call__`` — only the
        key/value set changes, from S token rows to V+1 pooled slots. That
        drops the pointer head's cost from O(V*S) to O(V^2) (V=13 on nn256,
        S~7000) and, more importantly, removes the last reason to keep the
        raw (S, E) sequence around once the positional encoding is gone.

        Queries stay the V real vertices, so the trailing global slot
        (structural tokens) can be ATTENDED but never SELECTED.
        """
        v_q = jax.vmap(self.vertex_embedding)(jnp.arange(self.num_vertices))
        attn_mask = jnp.broadcast_to(
            vmask[None, :], (self.num_vertices, vmem.shape[0])
        )
        vertex_reprs = self.cross_attn(v_q, vmem, vmem, mask=attn_mask)
        vertex_logits = jax.vmap(self.pointer_proj)(vertex_reprs).squeeze(-1)
        return vertex_logits, vertex_reprs


# `VertexIdentityPool` IS GONE (2026-08-15), and with it the whole idea of a
# separate identity CHANNEL.
#
# It was a learned attention pool over each vertex's own span of the BASE token
# stream, concatenated beside the dynamic slot as [identity || dynamic]. Two
# things were wrong with it and only one was the module's fault.
#
#   1. THE POOL TRAINED; THE ENCODE THAT FED IT DID NOT. Its input rows were
#      `carry_stream.base_identity_stream`'s output, computed ONCE PER EPISODE
#      OUTSIDE the loss (ppo.py:8205 as it then stood). Inside `filter_grad`
#      those rows are a CONSTANT, so palimpsa's base encode -- the thing that
#      actually produces vertex identity -- received exactly zero cotangent.
#      Measured, on the pre-change build: ||d loss / d(base-encode params)||
#      = 0.000000e+00, against 1.18e+03 available the moment the same encode
#      is moved inside. The encoder was being trained on the ONE step delta
#      that `--grad-window` K=1 admits, and on nothing else.
#   2. It was a SECOND mechanism for something the memory already does. A
#      vertex's identity is the rows palimpsa emits for that vertex's own
#      equation. Those rows can simply be SCATTERED into that vertex's slot,
#      by the same parameter-free `_vmem.scatter` a step delta goes through.
#      One scatter, one address, slots E wide instead of 2E, and `ctx_proj`
#      (which existed only to bring 2E contexts back to E) deletes with it.
#
# The replacement is `carry_stream.init_carry` folding owned base rows into
# their owner's slot, plus `base_memory` re-deriving that fold INSIDE the loss.


# STAGE B.4 `residual_state` IS GONE (2026-08-14).
#
# `ResidualStateUpdate` blended the eliminated vertex's context into a
# per-vertex state that `encode` / `heads_from_memory` ADDED to the vertex
# contexts. It was identically ZERO for every run this repo ever did:
# `init_linear_weights` zeroes every Linear bias and `_scale_output_heads`
# zeroed `event_proj.weight`, so the "update" was 0*ctx + 0 = 0 forever, and
# `residual_to_summary` (the B.4.next projection) had no reader at all. Zero
# is exactly what the additive path contributed, which is why deleting it is
# BIT-IDENTICAL on tests/policy_regression_gate.py.
#
# The dynamic per-vertex state the spec wanted is not this: it is the
# PARTICIPATION channel (every vertex a delta touches, concatenated beside a
# static identity), which lands in its own stage.


# ATTENTION-ENTROPY DIAGNOSTIC (entropy/palimpsa).
#
# NOT A POLICY ENTROPY. entropy/ve_head and entropy/approx_head are entropies
# of ACTION distributions (what the policy might do). This is the mean entropy
# of the ENCODER's attention rows -- how spread each query's attention is over
# the valid keys. It answers "has the representation collapsed", and its units
# are not comparable to the other two: the alphabets are slots/axes, not
# actions. Read it alone, never against the other two curves.
#
# Two sources are averaged, whichever exist on the agent:
#   * SetPointerVertexPolicy's Set-Transformer blocks over the (V+1) pooled
#     vertex slots  -- the vertex-side encoder attention;
#   * AxisSetEncoder's self-attention blocks over one vertex's axis tokens --
#     the approximation-side encoder attention.
# The palimpsa token mixer itself is LINEAR attention with no softmax, so it
# has no attention distribution to take an entropy of; these two softmax
# attentions are the only distributional objects in the encoder stack.
#
# COST: one extra encoder forward per EPISODE on ONE state (the rollout's first
# step), outside the gradient. The scores are recomputed rather than plumbed
# out of the hot path, so nothing in the vmapped loss changes. Default ON;
# ALPHAGRAD_ATTN_ENTROPY=0 disables it and the key is simply not logged.
_ATTN_ENTROPY_ON = os.environ.get("ALPHAGRAD_ATTN_ENTROPY", "1") == "1"


@eqx.filter_jit
def attention_entropy_diagnostic(agent, tokens, eqn_ids=None,
                                 axis_state=None, axis_valid=None):
    """Mean encoder attention-row entropy, or NaN when nothing applies."""
    from alphagrad.approx.set_pointer import SetPointerVertexPolicy
    parts = []
    pol = getattr(agent, "vertex_policy", None)
    if isinstance(pol, SetPointerVertexPolicy):
        token_mask = tokens != 0
        x = jax.vmap(agent.embedding)(tokens)
        if agent.pos_enc is not None:
            x = agent.pos_enc(x)
        enc_mask = None if agent.pos_enc is not None else token_mask
        enc_x = agent.encoder(x, eqn_ids=eqn_ids, mask=enc_mask,
                              key=jrand.PRNGKey(0))
        # Same segment pooling SetPointerVertexPolicy.__call__ does, so the
        # slots this scores are the slots the pointer actually sees.
        n_slots = pol.num_vertices + 1
        w = (token_mask > 0.5).astype(enc_x.dtype)
        if eqn_ids is None:
            pooled = jnp.broadcast_to(
                jnp.mean(enc_x, axis=0), (n_slots, enc_x.shape[-1]))
            vmask = jnp.ones((n_slots,), enc_x.dtype)
        else:
            ids = jnp.clip(eqn_ids, 0, n_slots - 1)
            sums = jax.ops.segment_sum(enc_x * w[:, None], ids,
                                       num_segments=n_slots)
            cnts = jax.ops.segment_sum(w, ids, num_segments=n_slots)
            pooled = sums / jnp.maximum(cnts, 1.0)[:, None]
            vmask = (cnts > 0).astype(enc_x.dtype)
        # The slots are E wide -- one address, no [identity || dynamic] half
        # to zero-pad (see the VertexIdentityPool obituary above).
        parts.append(pol.attention_entropy(pooled, vmask))
        if axis_state is not None and axis_valid is not None:
            # Whichever approximation head exists owns the AxisSetEncoder:
            # UnifiedFacePolicy (--live-faces) or MicroActionPolicy.
            _ax = None
            for _owner in (getattr(agent, "face_path_policy", None),
                           getattr(agent, "micro_action_policy", None)):
                _cand = getattr(_owner, "encoder", None)
                if _cand is not None and hasattr(_cand, "attention_entropy"):
                    _ax = _cand
                    break
            if _ax is not None:
                _logits, _ctx = pol.from_vertex_memory(pooled, vmask)
                feats = _axis_features_from_state(axis_state[0], axis_valid[0])
                parts.append(_ax.attention_entropy(feats, _ctx[0]))
    if not parts:
        return jnp.asarray(float("nan"), jnp.float32)
    # NaN-aware: a source whose attention has no real alphabet (e.g. the
    # raw-stream pointer fallback, where every vertex slot is the SAME
    # broadcast mean and only one slot is occupied) reports NaN and is
    # skipped rather than dragging the average to 0.
    from alphagrad.approx.set_pointer import _nanmean
    return _nanmean(jnp.stack(parts)).astype(jnp.float32)


def _axis_features_from_state(
    axis_state_v: jax.Array,
    axis_valid_v: jax.Array,
) -> AxisTokenFeatures:
    """Convert one row of `EnvState.axis_state` to :class:`AxisTokenFeatures`.

    `axis_state_v` is shape `(MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)` int32
    with field layout `[size, is_output, is_compressed, group_id]`. The
    heads.py tag_bits layout is `(is_logical, is_compressed, in_diag_group)`,
    so we derive `in_diag_group` from `group_id >= 0` and treat every
    valid axis as logical.
    """
    size = axis_state_v[:, _AXIS_FEAT_SIZE]
    log_size = jnp.log(jnp.maximum(size.astype(jnp.float32), 1.0))
    is_compressed = axis_state_v[:, _AXIS_FEAT_IS_COMPRESSED].astype(jnp.float32)
    group_id = axis_state_v[:, _AXIS_FEAT_GROUP_ID]
    in_diag_group = (group_id >= 0).astype(jnp.float32)
    # is_logical: every valid axis is "logical" in the dynamic-action sense
    # (the env-side static state doesn't distinguish physical vs logical;
    # COMPRESS sees the same set today).
    is_logical = axis_valid_v.astype(jnp.float32)
    tag_bits = jnp.stack([is_logical, is_compressed, in_diag_group], axis=-1)
    return AxisTokenFeatures(
        size=size,
        log_size=log_size,
        tag_bits=tag_bits,
        group_id=group_id,
        valid_mask=axis_valid_v.astype(jnp.float32),
    )


class Agent(eqx.Module):
    """Encoder + composable (vertex policy, rule policy) + three value heads.

    The value head is split into three single-output MLPs, one per training
    reward: ``value_head_flops`` (latency), ``value_head_mem``,
    ``value_head_cos``. Their concatenation is the (NUM_VALUE_HEADS,) = (3,)
    value vector the trainer consumes; the per-head split keeps gradient
    scales sane across the qualitatively different reward families and
    matches the per-head GAE / preference-scalarization in ``train_episode``.

    Stage B.2.A adds a data-dependent path: when `vertex_features` are
    supplied, the agent embeds the per-vertex op-type id and projects the
    remaining `NUM_VERTEX_FEATURES - 1` continuous moments + static features
    into the encoder's `embd_dim`, then *adds* the result to the per-vertex
    contexts that feed both the rule policy and the value head. The op
    embedding + projection are zero-initialised on the output side so the
    initial behaviour matches the pre-B.2 agent and gradient signal can flow
    in once features start mattering.
    """

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    vertex_policy: eqx.Module
    # The autoregressive micro-action policy: one sub-episode of typed
    # approximation actions (DIAG / COMPRESS / QUANT / END) per eliminated
    # vertex. Optional only so a bare Agent can be constructed in tests.
    micro_action_policy: MicroActionPolicy | None
    # Needed to shape the inactive action when the approximation head is
    # REMOVED (--no-approx-head): there is then no policy to ask for it.
    max_substeps: int = eqx.field(static=True, default=16)
    # P1c: per-path decisions (--face-actions). None ⇒ per-vertex mode.
    face_path_policy: FacePathPolicy | None
    value_head_flops: MLP
    value_head_mem: MLP
    value_head_cos: MLP
    op_embedding: eqx.nn.Embedding
    # NO identity_pool and NO ctx_proj. A vertex's identity is palimpsa's rows
    # for its own equation, scattered into its own slot by `carry_stream`;
    # the slots are E wide, so there is nothing to project back down.
    # F: preference-vector → embd_dim projection. Adds the per-episode
    # preference w ∈ Δ^7 into the policy's per-vertex contexts and the
    # value-head summary so a single net covers the whole Pareto front.
    # Zero-initialised so the conditioned and unconditioned paths agree at
    # step 0; gradient learns the conditioning from there.
    pref_proj: eqx.nn.Linear

    num_vertices: int = eqx.field(static=True)
    num_value_heads: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    op_embd_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        embedding,
        pos_enc,
        encoder,
        vertex_policy,
        value_head_flops,
        value_head_mem,
        value_head_cos,
        op_embedding,
        pref_proj,
        num_vertices,
        num_value_heads,
        max_rules,
        num_pair_choices,
        num_factors,
        embd_dim,
        op_embd_dim,
        micro_action_policy=None,
        max_substeps=16,
        face_path_policy=None,
    ):
        self.embedding = embedding
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.vertex_policy = vertex_policy
        self.micro_action_policy = micro_action_policy
        self.max_substeps = int(max_substeps)
        self.face_path_policy = face_path_policy
        self.value_head_flops = value_head_flops
        self.value_head_mem = value_head_mem
        self.value_head_cos = value_head_cos
        self.op_embedding = op_embedding
        self.pref_proj = pref_proj
        self.num_vertices = num_vertices
        self.num_value_heads = num_value_heads
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.embd_dim = embd_dim
        self.op_embd_dim = op_embd_dim

    def encode(
        self,
        tokens,
        eqn_ids=None,
        preference=None,
        key=None,
    ):
        token_mask = tokens != 0
        mask = token_mask[..., None]
        x = jax.vmap(self.embedding)(tokens)
        # pos_enc is None under a recurrent backbone (see _build_agent): the
        # palimpsa carry already encodes relative position via its decay.
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        # Stage B.1: when `eqn_ids` is provided, the encoder layers add
        # learned per-relation biases derived from it. When None, the encoder
        # falls back to vanilla self-attention so the path is preserved for
        # callers that haven't been wired yet.
        # Pad-mask threading (same as ppo_ray_worker.encode_tokens): the
        # palimpsa recurrence must not accumulate the ~16k pad positions —
        # harmless for the causal variant only because trailing pads sit
        # after every real row; a hard bug under palimpsa_bi. pos_enc is
        # None exactly for the recurrent backbones (see _build_agent); the
        # transformer keeps mask=None so its path stays byte-identical.
        enc_mask = None if self.pos_enc is not None else token_mask
        enc_x = self.encoder(x, eqn_ids=eqn_ids, mask=enc_mask, key=enc_key)

        vertex_logits, vertex_contexts = self.vertex_policy(enc_x, token_mask)

        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        # Stage F: add the per-episode preference projection to both the
        # per-vertex contexts (so the rule head sees w) and the summary
        # (so the value heads see w). With pref_proj zero-initialised the
        # path is a no-op at step 0 and matches the unconditioned baseline.
        if preference is not None:
            pref_emb = self.pref_proj(preference)
            vertex_contexts = vertex_contexts + pref_emb[None, :]
            summary = summary + pref_emb
        v_flops = self.value_head_flops(summary)
        v_mem = self.value_head_mem(summary)
        v_cos = self.value_head_cos(summary)
        value = jnp.concatenate([v_flops, v_mem, v_cos], axis=-1)
        return vertex_logits, vertex_contexts, value

    def value_for(
        self,
        tokens,
        eqn_ids=None,
        preference=None,
        key=None,
    ):
        _, _, value = self.encode(
            tokens,
            eqn_ids=eqn_ids,
            preference=preference,
            key=key,
        )
        return value

    # -- Phase 3b: incremental autoregressive encode ------------------------

    def carry_init(self):
        """Empty :class:`EncCarry` (M0 = 0, I0 = broadcast Ip — the palimpsa
        recurrence's t=0 state, matching ``palimpsa_ref``)."""
        layers = self.encoder.layers
        L = len(layers)
        H = layers[0].attn_layer.num_heads
        d = layers[0].attn_layer.head_dim
        M0 = jnp.zeros((L, H, d, d), jnp.float32)
        I0 = jnp.stack([
            jnp.broadcast_to(
                jnn.softplus(l.attn_layer.Ip_raw)[:, None, None], (H, d, d)
            ).astype(jnp.float32)
            for l in layers
        ])
        return EncCarry(
            M=M0, I=I0,
            cumhist=jnp.zeros((MAX_EQNS,), jnp.float32),
            nvalid=jnp.zeros((), jnp.float32),
            pos=jnp.zeros((), jnp.int32),
        )

    def encode_extend(self, carry, tokens_buf, eqn_ids_buf, count, *, window,
                      start=None, chunk=None, budget=None):
        """Extend the palimpsa carry by ``count`` tokens read from
        ``tokens_buf`` at ``carry.pos`` (fixed static ``window``; pad steps
        freeze the carry, so the valid prefix is bitwise-independent of the
        window size). Returns ``(new_carry, rows, valid, eqn_window)`` with
        ``rows`` (window, E) zeroed on invalid steps.

        ``chunk`` bounds how many of those pad steps are actually SCANNED --
        see :meth:`_extend_sequential`. ``None`` reads
        ``ALPHAGRAD_EXTEND_CHUNK`` (0 = scan the whole window, the original
        behaviour).

        ``budget`` is what makes the chunked path usable under reverse-mode
        AD: an UNBATCHED upper bound on ``count`` valid for every sample of
        the surrounding ``vmap``. With it the trip count comes from a
        ``lax.scan`` + ``lax.cond`` pair instead of a ``lax.while_loop``
        (which has no transpose rule). CONTRACT: the caller must derive it
        from the same array the counts come from, e.g.
        ``jnp.max(batch.delta_count)`` computed OUTSIDE the vmap -- a budget
        below some sample's ``count`` silently drops that sample's tail.

        Byte-identical math to the PalimpsaMixer/EncoderLayer stack with ONE
        deliberate exception: the relational forget-gate features are CAUSAL
        prefix counts from the carried histogram — the full path's whole-
        stream counts are acausal and cannot ride a recurrence. Rollout and
        loss share this exact computation, which is what ratio-1 needs.
        """
        if self.pos_enc is not None:
            raise RuntimeError(
                "encode_extend requires the palimpsa backbone "
                "(ALPHAGRAD_POLICY=palimpsa; pos_enc must be None)."
            )
        layers = self.encoder.layers
        count = jnp.clip(count, 0, window).astype(jnp.int32)
        # `start` reads a STANDALONE buffer from 0 instead of slicing the
        # episode stream at the carry's position -- the per-face chunks are
        # their own arrays, not a window into `state.tokens`.
        base = carry.pos if start is None else jnp.asarray(start, jnp.int32)
        idx = base + jnp.arange(window, dtype=jnp.int32)
        toks = jnp.take(tokens_buf, idx, mode="fill", fill_value=0).astype(jnp.int32)
        eqns = jnp.take(eqn_ids_buf, idx, mode="fill", fill_value=-1).astype(jnp.int32)
        valid = jnp.arange(window, dtype=jnp.int32) < count

        _mode = os.environ.get("ALPHAGRAD_CHUNKED_EXTEND", "0")
        if _mode == "1" or os.environ.get(
                "ALPHAGRAD_CHUNKED_SELFTEST", "0") == "1":
            par = self._extend_parallel(carry, toks, eqns, valid, count)
            if os.environ.get("ALPHAGRAD_CHUNKED_SELFTEST", "0") == "1":
                seq = self._extend_sequential(carry, toks, eqns, valid, count)
                jax.debug.print(
                    "[chunked-selftest] max|rows|={r:.3e} max|M|={m:.3e} "
                    "max|cumhist|={c:.3e}",
                    r=jnp.max(jnp.abs(par[1] - seq[1])),
                    m=jnp.max(jnp.abs(par[0].M - seq[0].M)),
                    c=jnp.max(jnp.abs(par[0].cumhist - seq[0].cumhist)))
                if _mode != "1":
                    return seq
            return par
        return self._extend_sequential(carry, toks, eqns, valid, count,
                                       chunk=chunk, budget=budget)

    def _extend_parallel(self, carry, toks, eqns, valid, count):
        """Blocked parallel extend: scan across fixed-size blocks, one
        associative_scan within each.

        A single whole-window associative_scan materializes the per-token
        states (T, H, d, n) -- at T=2048, E=256 that is ~67 MB per tensor per
        LAYER, and the loss vmaps 108 minibatch samples: the very first
        train_episode asked the shared 24 GB 4090 for one 15.64 GiB buffer.
        Blocking bounds the working set at (block, H, d, n) per sample --
        ALPHAGRAD_CHUNK_BLOCK=128 is ~4 MB -- while keeping T/block fewer
        sequential steps than the per-token scan. Same affine composition,
        so the selftest still holds to float-reassociation noise.
        """
        Bk = int(os.environ.get("ALPHAGRAD_CHUNK_BLOCK", "128"))
        T = toks.shape[0]
        nb = -(-T // Bk)
        pad = nb * Bk - T

        def _pad(a, fill):
            if pad == 0:
                return a
            return jnp.concatenate(
                [a, jnp.full((pad,) + a.shape[1:], fill, a.dtype)])

        b_toks = _pad(toks, 0).reshape(nb, Bk)
        b_eqns = _pad(eqns, -1).reshape(nb, Bk)
        b_ok = _pad(valid, False).reshape(nb, Bk)
        layers = self.encoder.layers

        def _affine(l, r):
            return (r[0] * l[0], r[0] * l[1] + r[1])

        def _block(c, blk):
            M, I, ch, nv0 = c
            btoks, beqns, ok = blk
            tv = (ok & (beqns >= 0)).astype(jnp.float32)
            e = jnp.clip(beqns, 0, MAX_EQNS - 1)
            # Causal relational feats: carried histogram supplies the prefix,
            # one (Bk, Bk) masked comparison supplies the within-block part.
            causal = jnp.tril(jnp.ones((Bk, Bk), jnp.float32))
            w = causal * tv[None, :]
            le = (e[None, :] <= e[:, None]).astype(jnp.float32)
            lt = (e[None, :] < e[:, None]).astype(jnp.float32)
            at = ch[e] + jnp.sum(w * le, axis=1)
            below = (jnp.where(e > 0, ch[jnp.maximum(e - 1, 0)], 0.0)
                     + jnp.sum(w * lt, axis=1))
            nval = nv0 + jnp.cumsum(tv)
            denom = jnp.maximum(nval, 1.0)
            feats = (jnp.stack([at - below, below, nval - at], axis=-1)
                     / denom[:, None] * tv[:, None])

            x = jax.vmap(self.embedding)(btoks)
            newM, newI = [], []
            for li, layer in enumerate(layers):
                mixer = layer.attn_layer
                H, d = mixer.num_heads, mixer.head_dim
                y = jax.vmap(layer.attn_norm)(x)
                q = _pal_qk_norm(
                    jax.vmap(mixer.query_proj)(y).reshape(Bk, H, d))
                kk = _pal_qk_norm(
                    jax.vmap(mixer.key_proj)(y).reshape(Bk, H, d))
                v = jax.vmap(mixer.value_proj)(y).reshape(Bk, H, d)
                b = _pal_beta(
                    jax.vmap(mixer.bias_proj)(y).reshape(Bk, H, d),
                    mixer.b_scale_raw)
                gt = jnn.softplus(jax.vmap(mixer.gate_proj)(y)
                                  + feats @ mixer.rel_gate.weight.T
                                  + mixer.rel_gate.bias)
                g = jnn.softplus(mixer.g_raw)
                Ip = jnn.softplus(mixer.Ip_raw)
                a = jnp.exp(-gt[:, :, None, None]
                            * g[None, :, None, None])
                okb = ok[:, None, None, None]
                a = jnp.where(okb, a, 1.0)
                B_M = jnp.where(
                    okb, v[:, :, :, None] * kk[:, :, None, :], 0.0)
                B_I = jnp.where(
                    okb,
                    b[:, :, :, None] * (kk[:, :, None, :] ** 2)
                    + (1.0 - a) * Ip[None, :, None, None], 0.0)
                A_M, S_M = lax.associative_scan(_affine, (a, B_M))
                _, S_I = lax.associative_scan(_affine, (a, B_I))
                M_t = A_M * M[li][None] + S_M
                I_t = A_M * I[li][None] + S_I
                mu = M_t / I_t
                out = jnp.einsum("thdn,thn->thd", mu,
                                 q * (d ** -0.5)).reshape(Bk, H * d)
                x = x + jax.vmap(mixer.output_proj)(out)
                y2 = jax.vmap(layer.mlp_norm)(x)
                x = x + jax.vmap(layer.mlp)(y2)
                newM.append(M_t[-1])
                newI.append(I_t[-1])

            rows_b = jnp.where(ok[:, None], x, 0.0)
            cnt = jnp.zeros((MAX_EQNS,), jnp.float32).at[e].add(tv)
            c2 = (jnp.stack(newM), jnp.stack(newI),
                  ch + jnp.cumsum(cnt), nv0 + jnp.sum(tv))
            return c2, rows_b

        # CHECKPOINTED blocks: without remat, reverse-mode AD saves every
        # block's per-token M_t/I_t -- O(T,H,d,n) residuals PER SAMPLE no
        # matter the block size, which is why blocking alone moved the OOM
        # from 15.6 GiB to 31.4 GiB instead of fixing it. With remat the
        # backward stores only the block-boundary carries (~13 MB/sample at
        # these shapes) and recomputes each block's forward -- the classic
        # sqrt-storage trade, paying one extra forward per block.
        (M2, I2, ch2, nv2), rows_b = lax.scan(
            jax.checkpoint(_block),
            (carry.M, carry.I, carry.cumhist, carry.nvalid),
            (b_toks, b_eqns, b_ok))
        rows = rows_b.reshape(nb * Bk, -1)[:T]
        new_carry = EncCarry(M=M2, I=I2, cumhist=ch2, nvalid=nv2,
                             pos=carry.pos + count)
        return new_carry, rows, valid, eqns

    def _extend_sequential(self, carry, toks, eqns, valid, count,
                           chunk=None, budget=None):
        layers = self.encoder.layers
        eq_arange = jnp.arange(MAX_EQNS, dtype=jnp.int32)

        def _step(c, tev):
            M, I, cumhist, nvalid = c
            tok, eid, ok = tev
            tokvalid = ok & (eid >= 0)
            tvf = tokvalid.astype(jnp.float32)
            e = jnp.clip(eid, 0, MAX_EQNS - 1)
            # Causal same/earlier/later counts incl. this token (the full
            # path's counts include self too), normalized by the running
            # valid-token count. Structural/pad tokens contribute zero
            # features but still see rel_gate's bias — matching the full
            # path, which zeroes feats yet applies the (learned) bias.
            cumhist2 = cumhist + tvf * (eq_arange >= e).astype(jnp.float32)
            nvalid2 = nvalid + tvf
            at = cumhist2[e]
            below = jnp.where(e > 0, cumhist2[jnp.maximum(e - 1, 0)], 0.0)
            denom = jnp.maximum(nvalid2, 1.0)
            feats = jnp.stack([at - below, below, nvalid2 - at]) / denom * tvf

            x = self.embedding(tok)
            new_M, new_I = [], []
            for li, layer in enumerate(layers):
                mixer = layer.attn_layer
                H = mixer.num_heads
                d = mixer.head_dim
                y = layer.attn_norm(x)
                q = _pal_qk_norm(mixer.query_proj(y).reshape(H, d))
                kk = _pal_qk_norm(mixer.key_proj(y).reshape(H, d))
                v = mixer.value_proj(y).reshape(H, d)
                b = _pal_beta(mixer.bias_proj(y).reshape(H, d),
                              mixer.b_scale_raw)
                gt = jnn.softplus(mixer.gate_proj(y) + mixer.rel_gate(feats))
                g = jnn.softplus(mixer.g_raw)
                Ip = jnn.softplus(mixer.Ip_raw)
                decay = jnp.exp(-gt[:, None, None] * g[:, None, None])
                M_l = v[:, :, None] * kk[:, None, :] + decay * M[li]
                I_l = (b[:, :, None] * (kk[:, None, :] ** 2)
                       + (1.0 - decay) * Ip[:, None, None] + decay * I[li])
                mu = M_l / I_l
                out = jnp.einsum("hdn,hn->hd", mu, q * (d ** -0.5)).reshape(H * d)
                x = x + mixer.output_proj(out)
                y2 = layer.mlp_norm(x)
                x = x + layer.mlp(y2)
                new_M.append(jnp.where(ok, M_l, M[li]))
                new_I.append(jnp.where(ok, I_l, I[li]))
            row = jnp.where(ok, x, jnp.zeros_like(x))
            return (jnp.stack(new_M), jnp.stack(new_I), cumhist2, nvalid2), row

        # unroll: the per-token body is a handful of small matvecs — a
        # sequential 16k-iteration scan of those is GPU launch-latency bound
        # (v15e profile: jit compute ~85% of the episode). Unrolling batches
        # K bodies per loop iteration into one fused kernel; values are
        # step-identical (unrolling never reassociates), so rollout/loss
        # parity is untouched.
        unroll = int(os.environ.get("ALPHAGRAD_EXTEND_UNROLL", "8"))
        c0 = (carry.M, carry.I, carry.cumhist, carry.nvalid)
        W = toks.shape[0]
        C = int(os.environ.get("ALPHAGRAD_EXTEND_CHUNK", "0")
                if chunk is None else chunk)
        if budget is not None:
            # The differentiated form pays nb_max = ceil(W/C) OUTER scan
            # iterations whatever the budget is -- the trip count is a
            # predicate inside the body, not a loop bound -- and on GPU a
            # scan iteration carrying this much state costs far more than the
            # ~C token steps it guards. Its optimum therefore sits at a much
            # LARGER chunk than the rollout while_loop's, which really does
            # stop after ceil(count/C) iterations and so wants C small. One
            # knob each; unset falls back to ALPHAGRAD_EXTEND_CHUNK, so the
            # old behaviour is one env var away.
            C = int(os.environ.get("ALPHAGRAD_LOSS_EXTEND_CHUNK", C))

        if C <= 0 or C >= W:
            (M2, I2, ch2, nv2), rows = lax.scan(
                _step, c0, (toks, eqns, valid), unroll=unroll)
        else:
            # DYNAMIC TRIP COUNT. The scan above is `window` long no matter
            # what `count` is -- on the TLM flagship a median 78-token delta
            # paid for 16384 steps, and `prof/encode` was dead constant at
            # 191.0/191.0/192.1 ms (mean/p95/max) because of it: 21% + 12% of
            # the episode spent scanning zeros.
            #
            # Why a while_loop of chunk-sized scans and NOT a `lax.switch`
            # over a bucket grid: the rollout that owns this cost is
            # `@jax.vmap`-decorated (rollout_fn), and vmap lowers a switch on
            # a BATCHED index to `select_n` over every branch -- measured:
            # all 7 grid scans, 16384 included, survive into the jaxpr, so
            # bucketing by switch would have cost MORE, not less. A while_loop
            # keeps one compiled body (no retrace per bucket at all) and runs
            # ceil(count/C) of them.
            #
            # Bit-identical, not approximately: `_step` FREEZES the whole
            # carry on an invalid step (`where(ok, M_l, M[li])`, `cumhist +
            # 0*`, `nvalid + 0`, `row = 0`), so the steps this skips are
            # provably no-ops, and `rows` is written into the SAME full-width
            # zero buffer the scan produced, so every downstream reduction
            # (segment_sum, cumsum, masked mean) sees the identical array.
            # NEVER truncates: `count` is already clipped to `window` by
            # `encode_extend`, so an over-long delta still takes the existing
            # truncation path and its telemetry, untouched.
            #
            # Reverse-mode AD does not work through `lax.while_loop`, so a
            # differentiated caller passes `budget` instead and gets the
            # `lax.scan` + `lax.cond` form below -- same skipping, but every
            # primitive on the path has a transpose rule. A differentiated
            # caller that passes NEITHER still lands on the flat scan.
            nb_max = -(-W // C)
            pad = nb_max * C - W
            if pad:
                toks_p = jnp.concatenate(
                    [toks, jnp.zeros((pad,), toks.dtype)])
                eqns_p = jnp.concatenate(
                    [eqns, jnp.full((pad,), -1, eqns.dtype)])
                valid_p = jnp.concatenate(
                    [valid, jnp.zeros((pad,), valid.dtype)])
            else:
                toks_p, eqns_p, valid_p = toks, eqns, valid
            _trip = count if budget is None else jnp.asarray(
                budget, jnp.int32)
            nb = jnp.minimum(
                (jnp.maximum(_trip, 0) + C - 1) // C, nb_max).astype(jnp.int32)

            if budget is not None:
                # DIFFERENTIABLE dynamic trip count. `nb` here is a batch-wide
                # bound, so the predicate `i < nb` is UNBATCHED under the
                # loss's vmap -- vmap then keeps a real `cond` instead of
                # lowering it to `select_n` over both branches, which is the
                # whole point (a per-sample predicate would compute the
                # skipped chunk anyway and save nothing).
                #
                # Bit-identical for the same reason the while_loop is: `_step`
                # freezes the entire carry and emits a zero row on an invalid
                # step, so a skipped chunk is the identity map -- and its
                # gradient is exactly zero, because every param path out of a
                # padded step goes through `jnp.where(ok, ., <carry>)` /
                # `jnp.where(ok, x, 0)` whose cotangent on the frozen side is
                # zero. The zero rows are still materialised at full width, so
                # every downstream reduction sees the identical array.
                b_toks = toks_p.reshape(nb_max, C)
                b_eqns = eqns_p.reshape(nb_max, C)
                b_valid = valid_p.reshape(nb_max, C)

                def _chunk_d(c, xs):
                    i, bt, be, bv = xs

                    def _run(c):
                        return lax.scan(_step, c, (bt, be, bv), unroll=unroll)

                    def _skip(c):
                        return c, jnp.zeros((C, self.embd_dim), jnp.float32)

                    return lax.cond(i < nb, _run, _skip, c)

                # REMAT the chunk body. Without it the scan stores every
                # step's per-layer activations for the backward -- and the
                # SKIPPED chunks store a zero block of exactly the same
                # shape, because `cond`'s partial-eval joins both branches'
                # residuals. That residual traffic is O(window) no matter how
                # few chunks actually run, which is why shrinking the chunk
                # count alone left a floor. With remat each chunk stores only
                # its boundary carry and recomputes its forward, so a skipped
                # chunk costs a predicate.
                # ON by default: measured 4.8s -> 2.1s of prof/update on the
                # TLM flagship and 4.1s -> 3.2s on CPU, with a bitwise
                # identical forward. ALPHAGRAD_LOSS_EXTEND_REMAT=0 restores
                # the stored-residual form.
                _body = (jax.checkpoint(_chunk_d)
                         if os.environ.get(
                             "ALPHAGRAD_LOSS_EXTEND_REMAT", "1") != "0"
                         else _chunk_d)
                (M2, I2, ch2, nv2), rows_b = lax.scan(
                    _body, c0,
                    (jnp.arange(nb_max, dtype=jnp.int32),
                     b_toks, b_eqns, b_valid))
                rows = rows_b.reshape(nb_max * C, -1)[:W]
                new_carry = EncCarry(M=M2, I=I2, cumhist=ch2, nvalid=nv2,
                                     pos=carry.pos + count)
                return new_carry, rows, valid, eqns

            def _chunk(st):
                i, M, I, ch, nv, rows = st
                off = i * C

                def _sl(a):
                    return lax.dynamic_slice(a, (off,), (C,))

                (M2, I2, ch2, nv2), r = lax.scan(
                    _step, (M, I, ch, nv),
                    (_sl(toks_p), _sl(eqns_p), _sl(valid_p)), unroll=unroll)
                return (i + 1, M2, I2, ch2, nv2,
                        lax.dynamic_update_slice(rows, r, (off, 0)))

            _i, M2, I2, ch2, nv2, rows = lax.while_loop(
                lambda st: st[0] < nb, _chunk,
                (jnp.zeros((), jnp.int32),) + c0
                + (jnp.zeros((nb_max * C, self.embd_dim), jnp.float32),))
            rows = rows[:W]

        new_carry = EncCarry(M=M2, I=I2, cumhist=ch2, nvalid=nv2,
                             pos=carry.pos + count)
        return new_carry, rows, valid, eqns

    def heads_from_memory(
        self,
        vmem_sums,
        vmem_counts,
        base_mem=None,
        preference=None,
    ):
        """The ``encode()`` head block, fed from the per-vertex memory instead
        of the raw (S, E) sequence: pointer via ``from_vertex_memory`` (same
        weights, keys/values = the V+1 pooled slots), value summary via the
        token-count-weighted slot mean (identical to the full path's masked
        token mean by associativity). Returns the same
        ``(vertex_logits, vertex_contexts, value)`` triple.

        ``base_mem`` is ``carry_stream.base_memory``'s ``(sums, counts)`` --
        the BASE stream's own contribution to the same slots, kept as a
        separate argument so its caller can recompute it INSIDE the gradient.
        Adding two (sum, count) memories before the mean IS pooling the union
        of their rows, so the split costs nothing and buys the entire point of
        this design: palimpsa's base encode has a cotangent.

        There is ONE slot per vertex and it is E wide. The
        ``[identity || dynamic]`` concatenation is gone: identity is not a
        second half, it is the vertex's own base rows sitting in the same
        slot, having arrived through the same scatter.
        """
        if base_mem is not None:
            vmem_sums = vmem_sums + base_mem[0]
            vmem_counts = vmem_counts + base_mem[1]
        vmem_rows = _vmem.read(vmem_sums, vmem_counts)
        vmask = _vmem.occupancy(vmem_counts)
        vertex_logits, vertex_contexts = (
            self.vertex_policy.from_vertex_memory(vmem_rows, vmask))
        if _DEBUG_ORDER:
            jax.debug.print(
                "[vmem] occupied={o}/{n} total_tokens={t} global_slot={g} "
                "logit_max={lm}",
                o=jnp.sum(vmask.astype(jnp.int32)),
                n=vmem_counts.shape[0],
                t=jnp.sum(vmem_counts),
                g=vmem_counts[-1],
                lm=jnp.max(vertex_logits),
            )
        # The SUMMARY slot, not a re-add of the per-slot sums: under
        # participation crediting a row lands in every slot it touches, so
        # the sums no longer total the tokens (see carry_stream.advance).
        summary = _vmem.summary(vmem_sums, vmem_counts,
                                summary_slot=vmem_sums.shape[0] - 1)
        if preference is not None:
            pref_emb = self.pref_proj(preference)
            vertex_contexts = vertex_contexts + pref_emb[None, :]
            summary = summary + pref_emb
        v_flops = self.value_head_flops(summary)
        v_mem = self.value_head_mem(summary)
        v_cos = self.value_head_cos(summary)
        value = jnp.concatenate([v_flops, v_mem, v_cos], axis=-1)
        return vertex_logits, vertex_contexts, value

    def sample_action_dynamic(
        self,
        tokens,
        vertex_avail_mask,
        axis_state,  # (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM) int32
        axis_valid_mask,  # (total_v, MAX_AXES_PER_VERTEX) float32
        factor_tables: FactorTables,
        op_legality_override,  # (NUM_OPS,) float32 — multiplied into op_legal (e.g. zeros COMPRESS)
        key,
        eqn_ids=None,
        cached_encoding=None,
        preference=None,
        vertex_temperature=None,
        oracle_pair_all=None,     # (total_v+1, N, N) live per-vertex DIAG mask
        oracle_comp_all=None,     # (total_v+1, N) live per-vertex COMPRESS mask
        face_masks_all=None,      # P1c: (fpair (V+1,F,N,N), fcomp (V+1,F,N), fvalid (V+1,F))
        precomputed=None,         # 3b: (vertex_logits, vertex_contexts, value) from the carry path
        oracle_fn=None,           # perf: called with the SAMPLED vertex, returns that vertex's masks only
        face_chunk_fn=None,       # (f, vertex_specs, rows, skips) -> that face's token chunk
        face_count_fn=None,       # vertex -> ACTUAL face count (while_loop trip count)
        enc_carry=None,           # step carry the per-face SIDE carry branches from
    ):
        """Same as :meth:`sample_action` but routes the rule head through
        :class:`MicroActionPolicy`. ``axis_state`` and ``axis_valid_mask``
        come straight from :class:`EnvState`; the per-vertex slice for
        the chosen vertex is converted to :class:`AxisTokenFeatures` and
        scanned by the policy. ``op_legality_override`` is a (NUM_OPS,) = (4,) mask
        multiplied into the per-step op legality — used by
        ``--allow-compress=False`` to keep the policy from emitting
        COMPRESS micro-actions until the graphax wiring lands.
        """
        net_key, vertex_key, micro_key = jrand.split(key, 3)
        if precomputed is not None:
            # 3b incremental-encode path: the caller already ran the carry
            # extension + vertex-memory heads; everything downstream (micro
            # + face policies) conditions on that carry via vertex_contexts.
            vertex_logits, vertex_contexts, value = precomputed
        else:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                preference=preference,
                key=net_key,
            )

        masked_v_logits = _mask_vertex_logits(vertex_logits, vertex_avail_mask)
        if vertex_temperature is not None:
            masked_v_logits = masked_v_logits / vertex_temperature
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        vertex_idx = distrax.Categorical(probs=vertex_dist).sample(seed=vertex_key)
        if _DEBUG_ORDER:
            _legal = vertex_avail_mask > 0.5
            jax.debug.print(
                "[pick] idx={v} p_pick={p} avail={a} psum={ps} "
                "pmass_illegal={pi} log_max={mx} log_minlegal={mn} "
                "nonfinite={nf}",
                v=vertex_idx,
                p=vertex_dist[vertex_idx],
                a=vertex_avail_mask[vertex_idx],
                ps=jnp.sum(vertex_dist),
                pi=jnp.sum(jnp.where(_legal, 0.0, vertex_dist)),
                mx=jnp.max(masked_v_logits),
                mn=jnp.min(jnp.where(_legal, masked_v_logits, jnp.inf)),
                nf=jnp.sum((~jnp.isfinite(vertex_logits)).astype(jnp.int32)),
            )

        v_context = vertex_contexts[vertex_idx]
        features = _axis_features_from_state(
            axis_state[vertex_idx],
            axis_valid_mask[vertex_idx],
        )

        if self.micro_action_policy is None and (
                self.face_path_policy is None or face_chunk_fn is None):
            # --no-approx-head: there IS no approximation head. Emit the
            # canonical inactive action; nothing is sampled or scored, and
            # face_out stays None so no face policy is consulted either.
            # Under --live-faces the micro head is ALSO None but the face
            # policy is live -- fall through so the faces still sample.
            _n = features.size.shape[0]
            _od, _id_, _jd, _ed, _kd = _zero_micro_dists(self.max_substeps, _n)
            return (
                vertex_idx,
                _zero_micro_action(self.max_substeps),
                vertex_dist,
                _od, _id_, _jd, _ed, _kd,
                jnp.asarray(0.0, jnp.float32),
                jnp.zeros((_n, _n), jnp.float32),
                jnp.zeros((_n,), jnp.float32),
                None,
                value,
                v_context,
            )

        # Live per-vertex DIAG/COMPRESS masks for the CHOSEN vertex (oracle rows
        # are 1-based; vertex_idx is 0-based). These are the authoritative masks
        # that keep the micro DIAG head from proposing a per-face-invalid pair;
        # stored (v_pair/v_comp) so the loss re-masks identically -> ratio 1.
        # Per-vertex legality masks. `oracle_fn` (perf path) probes ONLY the
        # vertex just sampled and returns its rows directly; the all-vertex
        # arrays are the equivalent legacy path (row `vertex_idx + 1` — the
        # oracle is 1-based). Values are identical either way.
        _face_from_fn = None
        if oracle_fn is not None:
            _o = oracle_fn(vertex_idx)
            v_pair, v_comp = _o[0], _o[1]
            _sample_pair, _sample_comp = v_pair, v_comp
            _face_from_fn = (_o[2], _o[3], _o[4])
        elif oracle_pair_all is not None:
            v_pair = oracle_pair_all[vertex_idx + 1]
            v_comp = oracle_comp_all[vertex_idx + 1]
            _sample_pair, _sample_comp = v_pair, v_comp
        else:
            _N = MAX_AXES_PER_VERTEX
            v_pair = jnp.zeros((_N, _N), jnp.float32)
            v_comp = jnp.zeros((_N,), jnp.float32)
            _sample_pair, _sample_comp = None, None  # tag-bit fallback

        # The variant/curriculum override is masked into the op distribution
        # itself (heads._compute_op_legality): a disallowed op is
        # unrepresentable rather than sampled-then-rewritten-to-END, so under
        # --exact the op head is a forced single-option head (no log-prob, no
        # entropy, no gradient) and the recorded log-prob IS the behaviour
        # policy's. evaluate_action_dynamic must receive the SAME override.
        if self.micro_action_policy is None:
            # --live-faces: no per-vertex head. Canonical inactive action,
            # zero log-prob/entropy, point-mass dists -- identical values to
            # the early return above, so the loss's reconstruction stays 0.
            _n0 = features.size.shape[0]
            actions = _zero_micro_action(self.max_substeps)
            (op_dists, i_dists, j_dists, exp_dists,
             kind_dists) = _zero_micro_dists(self.max_substeps, _n0)
            joint_logp = jnp.asarray(0.0, jnp.float32)
            joint_ent = jnp.asarray(0.0, jnp.float32)
            sub_episode_length = jnp.asarray(0, jnp.int32)
            quant_logp = jnp.asarray(0.0, jnp.float32)
        else:
            (
                actions,
                joint_logp,
                joint_ent,
                sub_episode_length,
                op_dists,
                i_dists,
                j_dists,
                exp_dists,
                kind_dists,
                quant_logp,
            # NOTE: quant_legality_mask defaults to None here — the
            # scan-driven hardware mask inside the policy.
            ) = self.micro_action_policy.sample(
                v_context,
                features,
                factor_tables,
                micro_key,
                pair_valid=_sample_pair,
                compress_valid=_sample_comp,
                op_legality_override=op_legality_override,
            )

        # P1c: per-path decisions for the chosen vertex, from the SAME
        # v_context/features the micro path used (no extra full encode).
        face_out = None
        _have_faces = ((_face_from_fn is not None)
                       or (face_masks_all is not None)
                       or (face_chunk_fn is not None))
        if _have_faces and self.face_path_policy is not None:
            _n_faces = None
            if _face_from_fn is not None:
                f_pair, f_comp, f_valid = _face_from_fn
            elif face_masks_all is not None:
                _fp_all, _fc_all, _fv_all = face_masks_all
                f_pair = _fp_all[vertex_idx + 1]
                f_comp = _fc_all[vertex_idx + 1]
                f_valid = _fv_all[vertex_idx + 1]
            else:
                # STATIC sampling masks: axis validity only. DIAG pair
                # divisibility is already the head's own pair_ok (gcd > 1)
                # gate; everything finer is per-face legality, which the
                # application hooks decide on the live operand -- an illegal
                # draw becomes a no-op there, it is never scored as applied.
                _F = self.face_path_policy.max_faces
                _av = axis_valid_mask[vertex_idx].astype(jnp.float32)
                _pair = (_av[:, None] * _av[None, :]
                         * (1.0 - jnp.eye(_av.shape[0])))
                f_pair = jnp.broadcast_to(
                    _pair, (_F,) + _pair.shape)
                f_comp = jnp.broadcast_to(_av, (_F,) + _av.shape)
                _n_faces = face_count_fn(vertex_idx)
                f_valid = (jnp.arange(_F) < _n_faces).astype(jnp.float32)
            face_key = jrand.fold_in(micro_key, 7)
            if face_chunk_fn is None:
                fa, face_logp, face_ent, _far, _skp, _fod, _fql = (
                    self.face_path_policy.sample(
                        v_context, features, factor_tables, face_key,
                        f_pair, f_comp, f_valid,
                        op_legality_override=op_legality_override,
                    )
                )
                _F = self.face_path_policy.max_faces
                f_cnt = jnp.zeros((_F,), jnp.int32)
                # No live stream -> no endpoints; 0 is "no vertex", which
                # gathers zero contexts, and the blind `sample` path above
                # already used the vertex context for both slots.
                f_ends = jnp.zeros((_F, 2), jnp.int32)
                f_dt = jnp.zeros((MAX_DELTA_TOKENS,), jnp.int32)
                f_de = -jnp.ones((MAX_DELTA_TOKENS,), jnp.int32)
            else:
                if self.micro_action_policy is None:
                    # No per-vertex head: the vertex rules are ALWAYS the
                    # exact END rows -- approximation is purely per-face.
                    _vspecs = -jnp.ones(
                        (MAX_RULES_PER_VERTEX, 3), jnp.int32)
                else:
                    # The per-vertex micro action applies to every face, so
                    # the contraction the head reads has to carry it.
                    _vspecs = micro_actions_to_rule_specs_jax(
                        actions.op_type, actions.i, actions.j,
                        actions.factor, axis_state[vertex_idx],
                        compress_kinds=actions.compress_kind,
                        quant_dtypes=actions.quant_dtype,
                        quant_scale_signs=actions.quant_scale_sign,
                        quant_scale_fracs=actions.quant_scale_frac,
                    ).astype(jnp.int32)
                (fa, face_logp, face_ent, f_cnt, f_dt,
                 f_de, f_ends) = self._face_loop(
                    features, factor_tables, face_key,
                    f_pair, f_comp, f_valid, enc_carry, face_chunk_fn,
                    vertex_idx, _vspecs, axis_state[vertex_idx],
                    op_legality_override,
                    (_n_faces if _n_faces is not None
                     else face_count_fn(vertex_idx)),
                )
            face_out = (fa, face_logp, face_ent, f_pair, f_comp, f_valid,
                        f_cnt, f_dt, f_de, f_ends)

        return (
            vertex_idx,
            actions,
            vertex_dist,
            op_dists,
            i_dists,
            j_dists,
            exp_dists,
            kind_dists,
            quant_logp,
            v_pair,
            v_comp,
            face_out,
            value,
            v_context,
        )

    # ------------------------------------------------------------------
    # The per-face pipeline. `_face_loop` samples; `_face_replay` scores the
    # stored decisions off the STORED chunks. They must stay gate for gate
    # identical or the ratio is not 1 at epoch 0.
    # ------------------------------------------------------------------
    def _face_encode(self, carry, tokens, eqns, count):
        """Extend the side carry by one face's chunk; ``(carry, summary)``.

        The scan is SKIPPED when the chunk is empty. That is not a micro-
        optimisation: the loop is a static ``range(MAX_FACES)`` because jit
        needs a fixed trip count, but a vertex has 1-12 faces on this graph
        and 4 have none at all, so most iterations carry nothing -- measured
        84 real faces against 23x16 = 368 iterations, i.e. 77% of the
        per-face encoder work was a full window scan over
        padding. An empty chunk must also leave the carry EXACTLY where it
        was, which the skip guarantees and the all-invalid scan only
        approximates.
        """
        def _run(c):
            c2, rows, valid, _e = self.encode_extend(
                c, tokens, eqns, count, window=MAX_DELTA_TOKENS, start=0)
            # THE SAME SCATTER, KEYED BY FACE. One chunk is one segment, so
            # this is `_vmem.scatter` with a single key -- the identical
            # primitive the vertex slots are built from, and it has no
            # parameters. `_face_replay` runs the many-key form of exactly
            # this over the whole stored emission window.
            return c2, _vmem.scatter_mean(
                rows, jnp.zeros((rows.shape[0],), jnp.int32), valid, 1)[0]

        def _skip(c):
            return c, jnp.zeros((self.embd_dim,), jnp.float32)

        return lax.cond(count > 0, _run, _skip, carry)

    def _face_row_specs(self, row, axis_state_v):
        """One face's per-slot wire row -> the env's ``[bi1, bi2, factor]``
        spec rows, via the SAME translator the env action uses."""
        def _one(op, i, j, factor, kind, dtype, qsign, qfrac):
            return micro_actions_to_rule_specs_jax(
                op[None], i[None], j[None], factor[None], axis_state_v,
                compress_kinds=kind[None], quant_dtypes=dtype[None],
                quant_scale_signs=qsign[None], quant_scale_fracs=qfrac[None],
            )[0]

        return jax.vmap(_one)(
            row["op_type"], row["i"], row["j"], row["factor"],
            row["compress_kind"], row["quant_dtype"],
            row["quant_scale_sign"], row["quant_scale_frac"],
        ).astype(jnp.int32)

    _WIRE_KEYS = ("op_type", "i", "j", "exponents", "factor",
                  "compress_kind", "quant_dtype", "quant_scale_sign",
                  "quant_scale_frac")

    @staticmethod
    def participation_mask(total_v, owner, face_ends, face_valid):
        """``(total_v + 1,)`` 0/1 mask: every slot this step's delta TOUCHES.

        ``{v} u {endpoints of v's faces}``. The face endpoints are the ones
        the head already reads its per-face contexts from, so participation
        costs one scatter and no new host traffic. The trailing entry is the
        GLOBAL slot, used when a delta touches nothing (the pre-scan
        bootstrap, whose owner is -1).
        """
        m = jnp.zeros((total_v + 1,), jnp.float32)
        m = m.at[jnp.where(owner >= 0, owner, total_v)].add(1.0)
        live = jnp.asarray(face_valid, jnp.float32) > 0.5
        ids = jnp.where(live[:, None], jnp.asarray(face_ends, jnp.int32),
                        0).reshape(-1) - 1
        # mode="drop": an endpoint of 0 ("no vertex": a jaxpr input) and a
        # padding face have no slot, and must not be folded into one.
        m = m.at[ids].add(jnp.where(ids >= 0, 1.0, 0.0), mode="drop")
        return (m > 0).astype(jnp.float32)

    # `_endpoint_ctx` IS GONE (2026-08-15). The face head no longer reads
    # `[ctx_i || ctx_j || face_latent]`; its input is the face's OWN palimpsa
    # latent and nothing else (`UnifiedFacePolicy._repr`). The face
    # ENDPOINTS are still read -- by `participation_mask`, which is data
    # routing (which slots a delta touched), not a head input.

    def _face_loop(self, features, factor_tables, key,
                   f_pair, f_comp, f_valid, enc_carry, face_chunk_fn,
                   vertex_idx, vertex_specs, axis_state_v,
                   op_legality_override, n_faces):
        """Read face f's chunk, decide face f, repeat -- for the ACTUAL face
        count, as a while_loop.

        The width F is the provable per-graph bound (196 on nn256-xent) and
        a python loop unrolled F copies of the encoder scan into the
        program: the F=16 smokes were OOM-killed at 64G during COMPILE.
        The body compiles once and runs ``n_faces`` times (the rollout is
        forward-only, so a dynamic trip count is legal). Padding faces
        never run -- wire rows stay END/skip-0 and counts stay 0, which the
        loss's gated evaluate scores as exactly zero, the same contract the
        unrolled loop's padding iterations had."""
        pol = self.face_path_policy
        F = pol.max_faces
        S = FACE_SLOTS
        n = jnp.minimum(jnp.asarray(n_faces, jnp.int32), F)
        wire0 = (
            jnp.full((F, S), OP_END, dtype=jnp.int32),     # op_type
            jnp.zeros((F, S), jnp.int32),                  # i
            jnp.zeros((F, S), jnp.int32),                  # j
            jnp.zeros((F, S, MAX_PRIMES), jnp.int32),      # exponents
            jnp.zeros((F, S), jnp.int32),                  # factor
            jnp.zeros((F, S), jnp.int32),                  # compress_kind
            jnp.zeros((F, S), jnp.int32),                  # quant_dtype
            jnp.ones((F, S), jnp.int32),                   # quant_scale_sign
            jnp.zeros((F, S), jnp.float32),                # quant_scale_frac
        )
        W = MAX_DELTA_TOKENS
        st0 = (jnp.asarray(0, jnp.int32), enc_carry, jnp.array(0.0),
               jnp.array(0.0), jnp.zeros((F,), jnp.int32),
               jnp.zeros((F,), jnp.int32),
               -jnp.ones((F, S, 3), jnp.int32), wire0,
               jnp.zeros((W,), jnp.int32), -jnp.ones((W,), jnp.int32),
               jnp.asarray(0, jnp.int32), jnp.zeros((F, 2), jnp.int32))

        def _body(st):
            (f, carry, logp, ent, skips, cnts, rs, wa, ftok, feqn, off,
             fends) = st
            # The chunk callback hands back the face's ENDPOINT VERTICES with
            # its tokens: the face enumeration that produced the chunk keyed
            # the face by exactly that pair, so it is free.
            tk_f, eq_f, ct_f, ends_f = face_chunk_fn(
                f, vertex_idx, vertex_specs, rs, skips)
            # Concatenate this chunk into the step's face stream -- the
            # EXACT tokens the head reads. The final emission is NOT a
            # substitute: chunk f's contraction is deliberately unhooked
            # (the face is undecided when read), while the real stream
            # emits it WITH its approximation -- a counterfactual the loss
            # can only reproduce from what was actually read (emission-
            # window replay measured ratio/max_log 778 at epoch 0).
            # ct_eff clamps to the remaining buffer AND feeds the encoder,
            # so sampling and the loss truncate identically if a step's
            # chunks ever exceed the window.
            ct_eff = jnp.minimum(jnp.asarray(ct_f, jnp.int32), W - off)
            _ar_w = jnp.arange(W, dtype=jnp.int32)
            _m = _ar_w < ct_eff
            ftok = ftok.at[off + _ar_w].set(
                jnp.where(_m, tk_f, 0), mode="drop")
            feqn = feqn.at[off + _ar_w].set(
                jnp.where(_m, eq_f, -1), mode="drop")
            carry, summ = self._face_encode(carry, tk_f, eq_f, ct_eff)
            sk, row, lp, e, _ar, _sp, _od = pol.sample_face(
                features, factor_tables, jrand.fold_in(key, f),
                f, f_pair[f], f_comp[f], f_valid[f], face_context=summ,
                op_legality_override=op_legality_override)
            rs = rs.at[f].set(self._face_row_specs(row, axis_state_v))
            skips = skips.at[f].set(sk.astype(jnp.int32))
            cnts = cnts.at[f].set(ct_eff)
            wa = tuple(w.at[f].set(row[k])
                       for w, k in zip(wa, self._WIRE_KEYS))
            return (f + 1, carry, logp + lp, ent + e, skips, cnts, rs, wa,
                    ftok, feqn, off + ct_eff, fends.at[f].set(ends_f))

        (_f, _c, logp, ent, skips, cnts, _rs, wa, ftok, feqn,
         _off, fends) = lax.while_loop(lambda st: st[0] < n, _body, st0)
        fa = FaceAction(skip=skips, **dict(zip(self._WIRE_KEYS, wa)))
        return fa, logp, ent, cnts, ftok, feqn, fends

    def _face_replay(self, features, factor_tables, fa,
                     f_pair, f_comp, f_valid, enc_carry, face_chunks,
                     op_legality_override, face_bound=None,
                     face_win_budget=None):
        """Score the stored FaceAction against contexts pooled from ONE scan
        of the stored emission window.

        The chunks the head read at sampling concatenate to a prefix of this
        step's emission (the tail is the last face's approximation, which no
        chunk contains), and the side carry branched from exactly
        ``enc_carry`` -- so one ``encode_extend`` over the stored window
        reproduces every chunk row byte-for-byte (the recurrence is causal),
        and face f's context is the mean of rows [s_f, e_f) with boundaries
        the counts' cumsum. One window scan per sample, against the old
        design's F scans over F stored windows. Gradient reaches palimpsa
        through this scan; truncation at the stored carry, as everywhere.

        ``face_chunks`` is ``(counts (F,), tokens (W,), eqns (W,))``.

        ``face_bound`` / ``face_win_budget`` are the batch-wide (UNBATCHED,
        so the predicates stay real ``cond``s under the loss's vmap) bounds on
        the live-face index and on the emission length. The sampling side is a
        ``while_loop`` over the ACTUAL face count; without these the replay was
        the only place still paying the padded width -- F = 2538 slots and a
        16384-token window against a measured 1.2 faces x ~80 tokens.
        """
        pol = self.face_path_policy
        F = pol.max_faces
        f_cnt, f_toks, f_eqns = face_chunks
        total = jnp.sum(f_cnt.astype(jnp.int32))
        # This is the LOSS side and it is reverse-differentiated (gradient
        # reaches palimpsa through exactly this scan), so the trip count comes
        # from `budget` (scan/cond) and never from a while_loop.
        if _FOLD_DELTA:
            # FOLDED. This is the reverse-differentiated face path, so its
            # (window, E) rows are STORED for the backward -- unlike the
            # rollout's `_face_encode`, whose rows are transient. Folding
            # here is the one that moves loss-side peak.
            #
            # `scatter_mean` is `scatter` then `s / max(c, 1)`, so
            # accumulating (s, c) per chunk and dividing ONCE at the end is
            # exact, and calling `_vmem.scatter` itself rather than
            # reimplementing it keeps the spill and out-of-range handling
            # identical.
            #
            # THE OFFSET IS LOAD-BEARING. The key is
            # `searchsorted(ends, position)`, so a chunk starting at `off`
            # must key on `off + arange(C)`. Dropping it sends every chunk
            # after the first to face 0 -- a plausible wrong answer, not a
            # crash, which is why delta_fold pins this in a dedicated test.
            _ends = jnp.cumsum(f_cnt.astype(jnp.int32))
            _C, _nb, _pad_len = _fold.plan_chunks(MAX_DELTA_TOKENS)

            def _face_fold(acc, rows_c, valid_c, _eqns_c, off):
                s_acc, c_acc = acc
                pos = off + jnp.arange(rows_c.shape[0], dtype=jnp.int32)
                fid = jnp.searchsorted(_ends, pos, side="right").astype(
                    jnp.int32)
                live = jnp.asarray(valid_c, jnp.float32) * (pos < total)
                s_c, c_c = _vmem.scatter(rows_c, fid, live, F)
                return (s_acc + s_c, c_acc + c_c)

            _, (_fs, _fc) = _fold.extend_fold(
                self, enc_carry, f_toks, f_eqns, total,
                window=MAX_DELTA_TOKENS,
                init_acc=(jnp.zeros((F, self.embd_dim), jnp.float32),
                          jnp.zeros((F,), jnp.float32)),
                fold_fn=_face_fold, budget=face_win_budget)
            face_latents = _fs / jnp.maximum(_fc, 1.0)[:, None]
        else:
            _, rows, _valid, _e = self.encode_extend(
                enc_carry, f_toks, f_eqns, total,
                window=MAX_DELTA_TOKENS, start=0,
                chunk=(None if face_win_budget is not None else 0),
                budget=face_win_budget)
        # ONE SCATTER, KEYED BY FACE. The chunks concatenate in face order,
        # so token t belongs to the face whose exclusive-prefix interval
        # contains t -- a `searchsorted` against the counts' cumsum. That key
        # plus `_vmem.scatter` gives every face's latent in a single pass:
        # the same parameter-free primitive the vertex slots use, with the
        # face as the key instead of the vertex. An empty chunk owns no
        # token, so its segment is empty and its latent is exactly zero --
        # matching the rollout's `_face_encode` skip.
            ends = jnp.cumsum(f_cnt.astype(jnp.int32))
            _pos = jnp.arange(rows.shape[0], dtype=jnp.int32)
            _fid = jnp.searchsorted(ends, _pos, side="right").astype(jnp.int32)
            _live = jnp.asarray(_valid, jnp.float32) * (_pos < total)
            face_latents = _vmem.scatter_mean(rows, _fid, _live, F)  # (F, E)

        # scan, not a python loop: F is the provable bound (196 here), and
        # unrolling it multiplied the program by F. scan keeps reverse-mode
        # AD (while_loop would not).
        def _one_face(acc, f, gate=None):
            logp, ent, arity = acc
            summ = face_latents[f]
            lp, e, ar, _sp, _od = pol.evaluate_face(
                features, factor_tables, fa, f,
                f_pair[f], f_comp[f], f_valid[f], face_context=summ,
                op_legality_override=op_legality_override)
            if gate is not None:
                _z = jnp.zeros((), jnp.float32)
                lp = jnp.where(gate, lp, _z)
                e = jnp.where(gate, e, _z)
                ar = jnp.where(gate, ar, _z)
            return (logp + lp, ent + e, arity + ar)

        _acc0 = (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        K = int(os.environ.get("ALPHAGRAD_FACE_REPLAY_BLOCK", "64"))
        if face_bound is None or K <= 0 or K >= F:
            (logp, ent, arity), _ = lax.scan(
                lambda a, f: (_one_face(a, f), None), _acc0, jnp.arange(F))
            # `face_latents` is the 4th return ONLY so the feature probe can
            # read the live head's own input -- `face_context=summ` above IS
            # `face_latents[f]`. Discarded (and DCE'd) on the default path.
            return logp, ent, arity, face_latents

        # BLOCKED gate. Skipping f >= face_bound is EXACT, not an
        # approximation: the head's gates SELECT rather than multiply, so a
        # slot with `face_valid == 0` returns (0, 0, 0) and contributes a zero
        # cotangent to every parameter, and `face_bound` is one past the last
        # index any sample in the minibatch marks valid.
        #
        # The gate is per BLOCK, not per face, and that is the whole point: a
        # `lax.cond` per face turned all F=2538 iterations into device-side
        # branches, which measured SLOWER than just running the (tiny) head --
        # 2538 predicates cost more than 2538 small MLPs. One predicate per
        # 64-face block leaves ceil(F/K) = 40 of them and one live block, and
        # the faces inside a live block run as a plain scan.
        #
        # Faces are still visited in increasing f, and the padded tail of the
        # last block is gated to an exact 0 before it is added, so the
        # accumulation order -- and therefore the float result -- is the same
        # as the unblocked scan's.
        nblk = -(-F // K)
        nlive = (jnp.maximum(jnp.asarray(face_bound, jnp.int32), 0)
                 + K - 1) // K

        def _blk(acc, b):
            def _run(acc):
                def _inner(a, k):
                    fi = b * K + k
                    return _one_face(a, jnp.minimum(fi, F - 1),
                                     gate=fi < F), None

                return lax.scan(_inner, acc, jnp.arange(K, dtype=jnp.int32))[0]

            return lax.cond(b < nlive, _run, lambda a: a, acc), None

        (logp, ent, arity), _ = lax.scan(
            _blk, _acc0, jnp.arange(nblk, dtype=jnp.int32))
        return logp, ent, arity, face_latents

    def evaluate_action_dynamic(
        self,
        tokens,
        vertex_idx,
        actions: MicroAction,
        vertex_avail_mask,
        axis_state,
        axis_valid_mask,
        factor_tables: FactorTables,
        key,
        eqn_ids=None,
        cached_encoding=None,
        preference=None,
        pair_valid=None,       # stored live DIAG mask for the chosen vertex
        compress_valid=None,   # stored live COMPRESS mask
        op_legality_override=None,  # (NUM_OPS,) variant mask — MUST match sample
        face_action: FaceAction | None = None,  # P1c stored per-path actions
        face_pair_valid=None,  # stored (F,N,N) sampling mask
        face_comp_valid=None,  # stored (F,N)
        face_valid=None,       # stored (F,)
        # UNUSED by the head since 2026-08-15 (the face input is the face's
        # own latent, not its endpoints' contexts). Still accepted because
        # the trajectory stores it for `participation_mask` on the rollout
        # side, and dropping it from the batch is a separate edit.
        face_ends=None,        # stored (F, 2) endpoint vertex ids (1-based)
        precomputed=None,      # 3b: (vertex_logits, vertex_contexts, value) from the carry path
        face_chunks=None,      # (counts, emission tokens, emission eqns)
        face_carry=None,       # carry2: where the sampling side carry branched
        face_bound=None,       # batch-wide live-face bound (unbatched)
        face_win_budget=None,  # batch-wide emission-length bound (unbatched)
    ):
        """Joint log-prob / entropy for a stored typed action sequence.

        Mirrors :meth:`evaluate_action` but uses
        :class:`MicroActionPolicy.evaluate` for the rule path. Returns
        ``(total_log_p, total_entropy, value, vertex_dist,
        op_dists, i_dists, j_dists, exp_dists, sub_episode_length)``.
        """
        if self.micro_action_policy is None and face_action is None:
            # --no-approx-head: mirror sample() exactly -- zero micro log-prob
            # and entropy, point-mass dists, so the PPO ratio is 1 on the
            # approximation factor and every micro KL term is 0.
            _feat = _axis_features_from_state(
                axis_state[vertex_idx], axis_valid_mask[vertex_idx])
            _n = _feat.size.shape[0]
            _od, _id_, _jd, _ed, _kd = _zero_micro_dists(self.max_substeps, _n)
            if precomputed is not None:
                _vl, _vc, _val = precomputed
            else:
                _vl, _vc, _val = self.encode(
                    tokens, eqn_ids=eqn_ids,
                    preference=preference, key=key)
            _vd = jnn.softmax(
                _mask_vertex_logits(_vl, vertex_avail_mask), axis=-1)
            return (
                jnp.log(_vd[vertex_idx] + 1e-8),
                entropy(_vd),
                _val,
                _vd,
                jnp.asarray(0, jnp.int32),
                _od, _id_, _jd, _ed, _kd,
                jnp.asarray(0.0, jnp.float32),
                # slot 11: face-head entropy. This branch is taken only when
                # there is no face action at all, so it is exactly 0.
                jnp.asarray(0.0, jnp.float32),
                # slot 12: no face path here, so no probe representation.
                ((None, _vc[vertex_idx]) if _fprobe.PROBE_ON else None),
            )
        if precomputed is not None:
            # 3b: same carry-derived triple the rollout sampled under (the
            # loss re-derives it from the stored pre-step carry + delta).
            vertex_logits, vertex_contexts, value = precomputed
        else:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                preference=preference,
                key=key,
            )

        masked_v_logits = _mask_vertex_logits(vertex_logits, vertex_avail_mask)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        log_p_vertex = jnp.log(vertex_dist[vertex_idx] + 1e-8)
        vertex_ent = entropy(vertex_dist)

        v_context = vertex_contexts[vertex_idx]
        features = _axis_features_from_state(
            axis_state[vertex_idx],
            axis_valid_mask[vertex_idx],
        )

        if self.micro_action_policy is None:
            # --live-faces: mirror the sampling side's zero micro terms.
            _n0 = features.size.shape[0]
            (new_op_dists, new_i_dists, new_j_dists, new_exp_dists,
             new_kind_dists) = _zero_micro_dists(self.max_substeps, _n0)
            log_p_sub = jnp.asarray(0.0, jnp.float32)
            ent_sub = jnp.asarray(0.0, jnp.float32)
            sub_episode_length = jnp.asarray(0, jnp.int32)
            new_quant_logp = jnp.asarray(0.0, jnp.float32)
        else:
            (
                log_p_sub,
                ent_sub,
                sub_episode_length,
                new_op_dists,
                new_i_dists,
                new_j_dists,
                new_exp_dists,
                new_kind_dists,
                new_quant_logp,
            # Masks MUST match sample_action_dynamic (the stored live oracle
            # masks; all-dtypes QUANT) or the PPO ratio is not 1 at epoch 0.
            ) = self.micro_action_policy.evaluate(
                v_context,
                features,
                factor_tables,
                actions,
                pair_valid=pair_valid,
                compress_valid=compress_valid,
                op_legality_override=op_legality_override,
            )

        total_log_p = log_p_vertex + log_p_sub
        # PER-HEAD ENTROPY NORMALISATION.
        #
        # The bonus used to be (vertex_ent + micro_ent + face_ent) / micro_arity
        # — three policies' entropy over ONE policy's action count. As QUANT
        # spam lengthened sub-episodes (1.76 -> 8.4 actions in v17) the shared
        # denominator grew, so the effective entropy weight FELL 0.028 -> 0.006:
        # emitting more junk actions bought the policy LESS exploration
        # pressure, which is exactly backwards and let DIAG underflow
        # unopposed. Each policy is now divided by its own arity, so one head
        # becoming chatty cannot silence the others. The vertex head emits
        # exactly one action per step, hence arity 1.
        total_entropy = vertex_ent + ent_sub / jnp.maximum(sub_episode_length, 1.0)
        # APPROXIMATION-HEAD ENTROPY, kept SEPARATELY from ``total_entropy``.
        # Under --live-faces the per-vertex MicroActionPolicy is not
        # constructed at all (see the agent factory), so ``ent_sub`` is a
        # constant 0 and the per-sub-step entropy slots are point masses: the
        # only live approximation entropy is the per-FACE head's. It is
        # returned as its own component so the logger can report it instead of
        # a dead 0.
        face_entropy = jnp.asarray(0.0, jnp.float32)
        # Feature probe: the LEAN arm's input is the face-keyed scatter, which
        # only the replay path builds. None elsewhere.
        _probe_face_lat = None
        # P1c: fold the face decisions' log-prob/entropy into the totals so
        # the PPO ratio covers them (evaluated with the STORED masks, same
        # gates as sampling — see FacePathPolicy).
        if face_action is not None and self.face_path_policy is not None:
            if face_chunks is None:
                f_logp, f_ent, f_arity, _sp, _od, _ql = (
                    self.face_path_policy.evaluate(
                        v_context, features, factor_tables, face_action,
                        face_pair_valid, face_comp_valid, face_valid,
                        op_legality_override=op_legality_override,
                    )
                )
            else:
                f_logp, f_ent, f_arity, _probe_face_lat = self._face_replay(
                    features, factor_tables, face_action,
                    face_pair_valid, face_comp_valid, face_valid,
                    face_carry, face_chunks, op_legality_override,
                    face_bound=face_bound,
                    face_win_budget=face_win_budget,
                )
            total_log_p = total_log_p + f_logp
            # Arity-normalised per the PER-HEAD ENTROPY NORMALISATION note
            # above: divided by the FACE head's own action count so a chatty
            # face head cannot silence the vertex head (and vice versa).
            face_entropy = f_ent / jnp.maximum(f_arity, 1.0)
            total_entropy = total_entropy + face_entropy
        # Per-step dists are forwarded for KL tracking against the rollout-time
        # old-policy snapshots; ``new_quant_logp`` is the factored-quant log-prob
        # (no flat dist), forwarded for parity with the trajectory schema.
        return (
            total_log_p,
            total_entropy,
            value,
            vertex_dist,
            sub_episode_length,
            new_op_dists,
            new_i_dists,
            new_j_dists,
            new_exp_dists,
            new_kind_dists,
            new_quant_logp,
            # slot 11: the FACE head's arity-normalised entropy, already folded
            # into ``total_entropy`` above and returned separately so the PPO
            # metrics can log it as its own channel (entropy/approx_head).
            face_entropy,
            # slot 12: the FEATURE PROBE's read-only view of the representation
            # -- (face latents (F, E), this step's vertex slot row (E,)). None
            # unless ALPHAGRAD_FEATURE_PROBE=1, and None is not a pytree leaf,
            # so nothing is computed, stored or transferred when it is off.
            ((_probe_face_lat, v_context) if _fprobe.PROBE_ON else None),
        )

    def to_env_action_dynamic(
        self,
        vertex_idx,
        actions: MicroAction,
        axis_state,
        face_action: FaceAction | None = None,
    ):
        """Convert a sampled :class:`MicroAction` sequence into a legacy
        :class:`StepAction` the env can consume.

        Uses :func:`micro_actions_to_rule_specs_jax` to translate the
        typed sequence into ``(MAX_RULES_PER_VERTEX, 3)`` rule_specs.
        DIAG / COMPRESS / QUANT micro-actions all map to typed rows; END
        sub-steps are dropped. The translator is JAX-traceable so the
        whole rollout step stays inside jit.

        ``actions.factor`` (stored on each MicroAction by ``sample_step``)
        is the integer factor consumed by the legacy spec — no re-
        derivation from exponents needed here. ``actions.compress_kind``
        and ``actions.quant_dtype`` carry the per-step kind / dtype
        indices for COMPRESS / QUANT rows.
        """
        axis_state_v = axis_state[vertex_idx]
        rule_specs = micro_actions_to_rule_specs_jax(
            actions.op_type,
            actions.i,
            actions.j,
            actions.factor,
            axis_state_v,
            compress_kinds=actions.compress_kind,
            quant_dtypes=actions.quant_dtype,
            quant_scale_signs=actions.quant_scale_sign,
            quant_scale_fracs=actions.quant_scale_frac,
        )
        if face_action is None:
            return StepAction(
                target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
                rule_specs=rule_specs,
            )

        # P1c: one spec row per (face, slot) — the same translator, run on
        # length-1 sequences; END translates to the all-(-1) unused row.
        def _one(op, i, j, factor, kind, dtype, qsign, qfrac):
            rows = micro_actions_to_rule_specs_jax(
                op[None], i[None], j[None], factor[None], axis_state_v,
                compress_kinds=kind[None], quant_dtypes=dtype[None],
                quant_scale_signs=qsign[None], quant_scale_fracs=qfrac[None],
            )
            return rows[0]

        face_rows = jax.vmap(jax.vmap(_one))(
            face_action.op_type, face_action.i, face_action.j,
            face_action.factor, face_action.compress_kind,
            face_action.quant_dtype, face_action.quant_scale_sign,
            face_action.quant_scale_frac,
        )
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=rule_specs,
            face_rows=face_rows.astype(jnp.int32),
            face_skip=face_action.skip.astype(jnp.int32),
        )


# ---------------------------------------------------------------------------
# Argparse and helpers used during main()
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PPO trainer for vertex-elimination with sparsification."
    )

    # Run / logging
    p.add_argument("--name", type=str, default="approx-ppo")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument(
        "--wandb",
        type=str,
        default="offline",
        choices=["disabled", "offline", "online"],
    )
    p.add_argument(
        "--wandb-project", type=str, default="dsnn-vertex",
        help="wandb project name.",
    )
    p.add_argument(
        "--wandb-entity", type=str, default="",
        help="wandb entity (team / user namespace). Empty = personal default. "
             "Set to 'dll-streetview' to land in that team's project.",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--grad-window", type=int, default=1,
        help="How many CONSECUTIVE step deltas the loss re-runs under the "
             "current parameters before scoring a step (K). The stored "
             "vertex memory is read as a CONSTANT, so K=1 -- the default and "
             "an exact reproduction of the previous behaviour -- gives "
             "palimpsa gradient from ONE delta and none from the elimination "
             "history. K>1 anchors the replay K-1 steps earlier and folds "
             "the intervening deltas in with gradient, at K times the stored "
             "delta buffers and K times the loss-side extend. "
             "K=0 selects the FULL-HORIZON path instead of a window: one "
             "`lax.scan` over the WHOLE episode from the base carry, so the "
             "gradient horizon is T rather than K, the per-step anchors and "
             "the (T, K) delta gather are dead, and the compile stops "
             "scaling with K (one scan body, not K unrolled ones). The "
             "minibatch axis moves from steps to SEQUENCES, because a scan "
             "needs a trajectory's steps in order.")
    p.add_argument("--no-jit", action="store_true")
    p.add_argument(
        "--exec-on-gpu",
        action="store_true",
        help="Pin training to GPU 0 and the env eval callback to GPU 1.",
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help="Bypass all micro actions (Vertex Elimination only)",
    )

    # Environment / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument("--disable-sparsification", action="store_true")
    p.add_argument(
        "--cmp-type", type=str, default="flops", choices=["graphax", "flops", "latency"]
    )
    p.add_argument(
        "--mem-type",
        type=str,
        default="peak_memory",
        choices=["graphax", "bytes_accessed", "peak_memory"],
    )
    # Reward selection / weighting. The env now reports the full 8-component
    # vector `REWARD_NAMES` every step; these flags determine how those
    # components are combined into the scalar advantage used by PPO. The
    # defaults reproduce the legacy `[-cmp, error, -mem]` behaviour with
    # `cmp` and `mem` mapped to whichever components `--cmp-type` / `--mem-type`
    # select.
    p.add_argument(
        "--rewards",
        nargs="+",
        type=str,
        default=["cmp", "mem", "acc"],
        choices=["cmp", "mem", "acc"],
    )
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    p.add_argument(
        "--lambda-acc",
        type=float,
        default=2.0,
        help="Weight on the cosine-similarity value head (trained signal). "
        "Default 2.0 so quality balances the two cost heads' combined vote — "
        "at 1.0 the z-scored advantage vote is 2-vs-1 for compute-zeroing.",
    )
    p.add_argument(
        "--lambda-frob",
        type=float,
        default=0.0,
        help="NO-OP, accepted for launcher compatibility. The Frobenius "
             "residual is no longer a trained channel and has no value head.",
    )
    p.add_argument("--gate-tau", type=float, default=0.5,
                   help="mult mode: cosine gate threshold tau (g=0 below it).")
    p.add_argument("--gate-w", type=float, default=40.0,
                   help="mult mode: cheapness budget W in symlog-cost units "
                   "(cheapness = max(0, W - sum w_c*symlog(cost_c))).")
    p.add_argument("--anti-degen-penalty", type=float, default=2.0,
                   help="mult mode: penalty floor P for degenerate terminals "
                   "(shaped ramp -P -> -P*(1-tau_d) over cos in [0, tau_d]).")
    p.add_argument("--anti-degen-tau", type=float, default=0.05,
                   help="mult mode: cosine threshold below which a TERMINAL "
                   "transition counts as degenerate.")
    p.add_argument(
        "--gate-fidelity",
        type=str,
        default="cos",
        choices=["cos"],
        help="mult mode's fidelity source. Only cosine similarity remains; "
             "the frob arm was removed with the channel.",
    )
    p.add_argument(
        "--reward-mode",
        type=str,
        default="additive",
        choices=["additive", "mult"],
        help="additive: per-head advantages scalarized by the preference "
        "weights (default). mult: multiplicative cosine gate — the scalar "
        "reward is g(cos)·max(0, W − Σ w_c·symlog(cost_c)) with an "
        "anti-degeneracy penalty, written into the cosine head with a "
        "one-hot preference (ported from the ray worker's "
        "ALPHAGRAD_REWARD_MODE=mult; the structural anti-collapse option).",
    )
    p.add_argument(
        "--per-face", action="store_true",
        help="Apply each vertex's approximation rules PER FACE (per local "
        "path) instead of uniformly to every face: a rule lands only where "
        "it is legal on that face's live operand, and a face where nothing "
        "is legal is left exact (the per-path skip). This is the O(|E|^2) "
        "action granularity the spec asks for.",
    )
    p.add_argument(
        "--face-actions", action="store_true",
        help="P1 per-path DECISIONS: the FacePathPolicy chooses, per face of "
        "the eliminated vertex, one SKIP gate (drops that path's contraction "
        "via graphax.SKIP_FACE) and one approximation per pre/post/new slot. "
        "Face log-probs enter the PPO loss; masks come from the oracle's "
        "face_masks and are stored for ratio-1. Subsumes --per-face's "
        "projection of a single per-vertex rule list.",
    )
    p.add_argument(
        "--incremental-encode", action="store_true",  # mandatory (stage 2)
        help="Phase 3b: autoregressive O(delta) encoding. The palimpsa carry "
        "rides the rollout scan (base stream consumed once, each step extends "
        "by its delta tokens only); pointer/value heads read from the "
        "per-vertex memory; the loss extends each sample's stored PRE-step "
        "carry (ratio-1 by construction, gradient truncates at the carry). "
        "Requires ALPHAGRAD_POLICY=palimpsa (causal, no pos-enc), "
        "ALPHAGRAD_INCREMENTAL_TOKENS=1 (append-only stream) and "
        "--dynamic-substeps.",
    )
    p.add_argument(
        "--advantage-norm", type=str, default="popart",
        choices=["popart", "zscore", "none"],
        help="popart (default): per-channel debiased-EMA normalisation of "
        "value targets + sigma-scaled advantages, with an output-preserving "
        "head rescale. zscore: the legacy per-batch z-score, which has a "
        "collapse ratchet (a uniformly-degenerate batch drives std->0 so the "
        "opposing channel vanishes). none: NO adaptive normalisation — "
        "advantages stay in raw symlog units and the CLI --lambda-* weights "
        "are the ONLY scaling (manual-weight mode; reward semantics are "
        "stationary across the whole run).",
    )
    p.add_argument(
        "--no-symlog", action="store_true",
        help="Disable the symlog reward transform and let PopArt do the "
             "per-channel scaling alone. Symlog and PopArt address the SAME "
             "cross-channel dynamic range; stacking them squashes the "
             "per-channel spread under --popart-sigma-min (measured: mem "
             "sigma 0.00694 against a 0.1 floor) so the channel is shrunk "
             "~14x instead of normalised. Only meaningful with "
             "--advantage-norm popart.")
    p.add_argument(
        "--lean-logging", action="store_true",
        help="Log only aggregates (means, entropy, KL, collapse counts, "
             "losses). Drops per-channel best/median/worst + all-time stats, "
             "the three per-episode Pareto scatter tables, and the "
             "elimination-order table. The all-time block also grew a python "
             "list by one entry per episode per channel and re-medianed it "
             "every episode, so its cost rises with episode count.")
    p.add_argument(
        "--unified-head", action="store_true",
        help="Replace the autoregressive approximation sub-episode with ONE "
             "32-output head per vertex (skip / op / i / j / reduce axes+fn / "
             "dtype). The block-diagonal factor is NOT sampled: it is "
             "gcd(N_i, N_j), the largest legal factor = the smallest blocks "
             "(square pair -> pure diagonal), and coprime pairs are masked "
             "out. Exposed through MicroActionPolicy's contract, so the env "
             "and loss are unchanged.")
    p.add_argument(
        "--ray-measure", type=int, default=0, metavar="N",
        help="Fan the env measurement callback out over N Ray actors "
             "(0 = off, in-process serial). Requires "
             "ALPHAGRAD_BATCHED_CALLBACK=1. With --exec-on-gpu each actor is "
             "pinned to its OWN gpu (num_gpus=1) so no two TIMED executions "
             "ever share a device -- co-residency measured CV 0.0000%% -> "
             "49.7%%. Ray rather than threads because the per-measure XLA "
             "executable leak is only freed by process teardown. Incompatible "
             "with --face-actions (the pool's env is per-vertex and would "
             "silently drop the per-face decisions).")
    p.add_argument(
        "--ray-measure-timeout", type=float, default=600.0,
        help="Per-call timeout for a --ray-measure actor, seconds.")
    p.add_argument(
        "--pareto-dump-every", type=int, default=50, metavar="N",
        help="Write the Pareto FRONT (objectives + the sequences that "
             "achieved them) to the wandb run dir every N episodes, plus once "
             "at the end. 0 disables. Without this only pareto/hypervolume "
             "and pareto/archive_size are recorded — scalars describing a "
             "front whose sequences are then discarded at process exit.")
    p.add_argument(
        "--no-approx-head", action="store_true",
        help="REMOVE the approximation heads instead of masking them. "
             "--variant ve_only only multiplies the op categorical by "
             "[0,0,0,1]; both heads still exist, still hold parameters, and a "
             "head the mask does not reach can still act -- the face SKIP gate "
             "did exactly that, deleting Jacobian paths under a variant that "
             "asked for none. With this flag neither micro_action_policy nor "
             "face_path_policy is constructed: the pure "
             "vertex-elimination-order control.")
    p.add_argument(
        "--force-pure-diag", action="store_true",
        help="NO-OP, accepted for launcher compatibility. --unified-head now "
             "ALWAYS uses the largest legal factor gcd(N_i, N_j), so a square "
             "pair is already a pure diagonal.")
    p.add_argument(
        "--set-pointer", action="store_true",
        help="Use SetPointerVertexPolicy: a CONTENT-based pointer over the "
             "segment-pooled vertex memory with permutation-equivariant "
             "Set-Transformer blocks, instead of PointerVertexPolicy whose "
             "queries came from a fixed Embedding(num_vertices, embd_dim) "
             "table indexed by vertex id. No parameter depends on V, so the "
             "same weights transfer across graph sizes.")
    p.add_argument(
        "--set-pointer-blocks", type=int, default=2,
        help="Number of Set-Transformer blocks used by --set-pointer.")
    p.add_argument(
        # Default 3, not 0: every launcher passed 3 explicitly, so an unflagged
        # run was silently unseeded and defined its scale from whatever the
        # untrained policy happened to produce. az_gumbel's flag of the same
        # name has the same default.
        "--popart-init-episodes", type=int, default=3,
        help="Warm-start PopArt (mu, sigma) from this many rollouts of RANDOM "
             "but VALID plans before training. 0 disables. The rollouts use "
             "the real legality masks and the real measurement path, so the "
             "seeded scale is the true measurement scale; without this the "
             "first gradient step defines the scale from whatever the "
             "untrained policy produced.")
    p.add_argument(
        "--popart-init-temperature", type=float, default=10.0,
        help="Softmax temperature applied to the VERTEX pointer during the "
             "PopArt warm-start rollouts. Large => ~uniform over the legal "
             "vertices. The micro heads are already uniform at init.")
    p.add_argument("--popart-beta", type=float, default=1e-2,
                   help="PopArt EMA rate per update.")
    p.add_argument("--popart-sigma-min", type=float, default=0.1,
                   help="Per-channel sigma floor (stops advantage blow-up "
                   "when a channel goes uniform).")
    p.add_argument(
        "--num-data-points", type=int, default=5,
        help="Measurement protocol: distinct eval samples measured per "
        "reward (spec default 5). Quality is computed once per point and "
        "winsorized across points.",
    )
    p.add_argument(
        "--reps-per-point", type=int, default=4,
        help="Measurement protocol: timing repetitions per data point "
        "(spec default 4, so 5x4=20). Only used with --measure-latency; "
        "quality/memory don't need repeats.",
    )
    p.add_argument(
        "--latency-inner-reps", type=int, default=1,
        help="Measurement protocol: executions per timed rep inside one "
        "monitor window; elapsed time is divided by this, amortizing "
        "dispatch/timer overhead (spec default 50; CLI default 1 keeps "
        "existing campaigns' readings comparable).",
    )
    p.add_argument(
        "--measure-grad",
        action="store_true",
        help="Measure the GRADIENT pipeline instead of the raw Jacobian: the "
        "example is wrapped in scalar_loss_fn (mean -> MSE for the NN "
        "examples) BEFORE tracing, so the policy graph, mask oracle, and "
        "measured executable all live on the same scalar-loss graph and "
        "jacve of it yields the gradients the spec asks to time.",
    )
    p.add_argument(
        "--quality-metric",
        choices=["auto", "loss_drop", "cosine", "none"],
        default="auto",
        help="WHICH quantity reward slot 6 (the --lambda-acc channel) holds. "
        "loss_drop = the relative loss drop of a 200-step Adam walk driven by "
        "the PLAN's own gradient, probed on a fixed batch of 512 real MNIST "
        "images (Pearson 0.922 against final downstream test accuracy, 0.22 s "
        "and 40 MB per plan). cosine = the legacy Jacobian cosine (Pearson "
        "0.610, 9.70 s, 4.24 GB). auto = loss_drop under --measure-grad "
        "(where the plan's output IS a gradient and the walk is defined), "
        "cosine otherwise. Published as ALPHAGRAD_QUALITY_METRIC so the Ray "
        "measure actors resolve the SAME metric as the trainer.",
    )
    p.add_argument(
        "--walk-steps", type=int, default=200,
        help="Adam steps in the loss-drop walk (measured configuration: 200).",
    )
    p.add_argument(
        "--walk-lr", type=float, default=1e-3,
        help="Adam lr for the loss-drop walk (measured configuration: 1e-3, "
        "b1 0.9, b2 0.999, eps 1e-8).",
    )
    p.add_argument(
        "--walk-probe-seed", type=int, default=20260807,
        help="Seed of the loss-drop PROBE BATCH. Fixed across plans within a "
        "run by construction -- plan scores are only comparable on one batch.",
    )
    p.add_argument(
        "--walk-noise-std", type=float, default=0.0,
        help="OPTIONAL, default OFF. Resample N(0, std) pixel noise on the "
        "walk batch each step (std 0.3 lifts Pearson 0.922 -> 0.950). Leave "
        "at 0.0 to reproduce the headline numbers.",
    )
    p.add_argument(
        "--seed-vertices",
        action="store_true",
        help="With --measure-grad, use seed_loss_fn instead of scalar_loss_fn: "
        "the tangent seed and the adjoint contraction enter the graph as "
        "ORDINARY ELIMINABLE VERTICES, so the elimination/action space stays "
        "the Jacobian graph (plus seed nodes) while the MEASURED object is the "
        "gradient. The policy then learns WHEN to apply the seed — seeding "
        "early is one VJP (gradient-cost), seeding late builds the full "
        "Jacobian — i.e. forward/reverse/cross-country becomes part of the "
        "search. Requires graphax >= 5c56105 (seed-vertex sentinel fix).",
    )
    p.add_argument(
        "--measure-latency",
        action="store_true",
        help="Run the compiled approx fn 10x per env step to populate the latency reward "
        "component. Significantly slower; turn on only when latency is being weighted.",
    )
    p.add_argument(
        "--terminal-rewards-only",
        action="store_true",
        help="Compute the env's reward vector only at the final elimination step; "
        "intermediate steps return zeros. Skips per-step jacve compile/exec — the "
        "dominant rollout cost. PPO+GAE handles sparse rewards natively.",
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "wikitext2", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # Agent variant — the two flags are independent. Combinations:
    #   (default)                : pointer + autoregressive RuleDecoder
    #   --not-autoreg            : pointer + single-rule sp head
    #   --no-ptr                 : masked MLP vertex head + autoregressive RuleDecoder
    #   --no-ptr --not-autoreg   : masked MLP vertex head + single-rule sp head (the
    #                              pre-pointer-net baseline)

    # Dynamic-substeps (heads.py) mode is the default. The MicroActionPolicy
    # emits a typed (op_type, i, j, prime_exponents) sequence per vertex,
    # scanned to `--max-substeps` with END termination (and a forced END at
    # the per-vertex `2 × num_axes` hard cap to bound rollout length). The
    # FactorTables (precomputed prime / gcd lookup) are built from
    # `--max-axis-size`. COMPRESS legality is gated by `--allow-compress`
    # (off by default — graphax's vertex_elimination_jaxpr doesn't yet
    # consume real mean-compression, so any emitted COMPRESS is silently
    # dropped by the legacy translator). Pass `--no-dynamic-substeps` to
    # fall back to the legacy AutoregRulePolicy (pair + categorical-factor
    # over a static factor table).
    p.add_argument(
        "--dynamic-substeps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the heads.py MicroActionPolicy (default). Pass "
        "--no-dynamic-substeps to use the legacy "
        "AutoregRulePolicy with the static `--factors` table.",
    )
    p.add_argument(
        "--max-substeps",
        type=int,
        default=2 * MAX_AXES_PER_VERTEX,
        help="Hard upper bound on sub-episode length per vertex. "
        "At runtime each vertex's cap is `2 × #active axes`; "
        "this flag is the static JAX-shape ceiling. Defaults "
        "to 2 × MAX_AXES_PER_VERTEX so the cap is never "
        "truncated by the static bound.",
    )
    p.add_argument(
        "--max-axis-size",
        type=int,
        default=1024,
        help="Max bound on logical axis sizes for the precomputed "
        "FactorTables (gcd / prime / exp lookup). Must be ≥ "
        "the largest dim in the env's jaxpr. Memory cost is "
        "O(max_axis_size²) for the gcd table; 1024 ≈ 4 MiB.",
    )
    p.add_argument(
        "--allow-compress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable COMPRESS legality in the MicroActionPolicy. "
        "On by default now that the typed-COMPRESS path is wired through "
        "the env translator and graphax's apply_compress (with a "
        "selectable reduction kind). Pass --no-allow-compress to "
        "force every COMPRESS sub-step to END instead — useful for "
        "isolating the elimination-order / DIAG signal.",
    )
    p.add_argument(
        "--axis-group-embedding",
        action="store_true",
        help="Add a learned per-DIAG-group embedding to each axis "
        "token in the AxisSetEncoder. Off by default: "
        "tag_bits[in_diag_group] already exposes group "
        "membership, and the embedding bloats proj_in by "
        "embd_dim plus a (max_groups+1, embd_dim) table. "
        "Turn on when you want the encoder to learn per-block "
        "representations beyond bare membership.",
    )

    # Multi-rule / autoregressive head config (ignored when --no-ptr or --not-autoreg)
    p.add_argument(
        "--max-rules",
        type=int,
        default=MAX_RULES_PER_VERTEX,
        help="Max number of (axis_pair, factor) rules per chosen vertex (autoregressive head only). "
        "Pending the heads.py rewrite this is a static upper bound; the future dynamic head will "
        "let sub-episodes terminate via END at any sub-step.",
    )
    p.add_argument(
        "--factors",
        type=str,
        default="-1,1,2,4",
        help="Comma-separated factor choices for the per-rule factor head. "
        "Legacy sentinels: -1 = gcd-collapse, 0 = drop-axes (NOT semantically COMPRESS — "
        "real mean-COMPRESS lives in graphax.sparse.micro_actions.apply_compress and "
        "will be exposed as a distinct op once the heads.py rewrite lands). "
        "Going forward, prefer explicit positive divisors; the gcd is just one such value.",
    )

    # Comparison-study variants. `custom` honours whatever was passed on the
    # CLI verbatim. The other values pre-set --factors / --max-rules /
    # --pin-rules-to-exact to specific comparison points (see VARIANT_PRESETS
    # below). Explicit later flags still override the preset.
    p.add_argument(
        "--variant",
        type=str,
        default="custom",
        choices=[
            "custom",
            "ve_only",
            "diag_gcd",
            "diag_factor",
            "compress",
            "compress_scalar",
            "quantize",
            "quant_smallest_float",
            "full",
        ],
        help=(
            "Pre-canned configuration mapping to --factors / --max-rules / "
            "--pin-rules-to-exact for the architecture comparison study. "
            "`custom` (default) honours your explicit flags. `ve_only` "
            "freezes the rule head (no DIAG/COMPRESS, pointer + vertex-order "
            "only). `diag_gcd` allows a single gcd-collapse DIAG per vertex. "
            "`diag_factor` allows a single DIAG with factor choice. "
            "`compress` is currently unwired (depends on the heads.py / "
            "atomic-compress rewrite — see graphax.sparse.micro_actions). "
            "`full` enables the existing multi-rule DIAG path."
        ),
    )

    # Network architecture
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument(
        "--op-embd-dim",
        type=int,
        default=8,
        help="Per-vertex op-type embedding dimension (Stage B.2.A).",
    )
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument(
        "--unified-face-head", action="store_true",
        help="Per-face approximation head with 94 outputs (one shared skip "
             "Bernoulli + 3 x 31 slot fields) drawn from ONE MLP forward, "
             "instead of FacePathPolicy's 32 encoder + 24 head calls per "
             "vertex. Requires --face-actions.")
    p.add_argument(
        "--live-faces", action="store_true",
        help="Approximate each face AFTER reading its contraction. Per face: "
             "the host emits that face's token chunk (the previous face's "
             "approximation equations followed by this face's contraction), "
             "palimpsa extends a side carry with only those tokens, and the "
             "approximation head decides from it — so face f+1's contraction "
             "reflects face f's approximation. Without it every face of a "
             "vertex is decided from one pre-elimination summary and they are "
             "indistinguishable to the head. Requires --face-actions and "
             "--incremental-encode (the loss re-runs the same recurrence from "
             "the stored step carry).")
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument(
        "--value-dims",
        type=str,
        default="64,32",
        help="MLP value head hidden widths (comma-separated).",
    )

    # Optimisation
    p.add_argument(
        "--num-envs",
        type=int,
        default=-1,
        help="Parallel rollout envs. -1 = os.cpu_count() (or 16 for Vmapped examples).",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-clip-eps", type=float, default=0.2)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-b1", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    p.add_argument(
        "--lr-decay-min-mult",
        type=float,
        default=0.1,
        help="Cosine decay floor as a multiple of the initial learning rate.",
    )
    p.add_argument(
        "--head-init-scale",
        type=float,
        default=0.1,
        help="Multiplier applied to output-head weights at startup for a near-uniform initial policy.",
    )
    p.add_argument(
        "--pin-rules-to-exact",
        action="store_true",
        help="Stage C: pin the axis-pair / factor heads to exact-AD "
        "(every slot = STOP, factor 0). The rule policy is skipped "
        "entirely so gradient only flows through the vertex head. "
        "Use this when isolating elimination-order learning from "
        "approximation choice.",
    )
    p.add_argument(
        "--pin-factor",
        type=int,
        default=None,
        help="Stage D: pin every emitted factor to this VALUE from "
        "--factors (e.g. -1 for the gcd-collapse / strict-diagonal "
        "default). The axis-pair head still trains; the factor "
        "head's contribution is deterministic so its log-prob "
        "and entropy are zero. Mutually exclusive with "
        "--pin-rules-to-exact.",
    )
    p.add_argument(
        "--axis-warmup-steps",
        type=int,
        default=0,
        help="Stage D head LR warmup (§3.2): linearly ramp the "
        "axis-pair head's LR multiplier from 1/3 → 1 over the "
        "first N optimizer steps after which point it stays at 1. "
        "0 = ramp disabled, full LR from step 0. The vertex and "
        "shared params always run at full base LR.",
    )
    p.add_argument(
        "--factor-warmup-steps",
        type=int,
        default=0,
        help="Stage E head LR warmup: same ramp, applied to the "
        "factor head. 0 = no ramp.",
    )
    p.add_argument(
        "--loss-mode",
        type=str,
        default="multi_head",
        choices=["multi_head", "scalar"],
        help="``multi_head`` (default): keep the 3-value-head architecture, "
        "compute per-head GAE on (latency, peak_memory, cosine_sim), and "
        "scalarize advantages with either ``--preference-conditioned`` "
        "Dirichlet samples or the static ``--lambda-*`` weights. "
        "``scalar``: collapse to the single-channel vertex_ppo.py-style PPO — "
        "build one scalar reward per step as "
        "``sum_i(reward_weights[i] * symlog(reward_vec[i]))`` (symlog absorbs "
        "the wide magnitude range across the 8 components), run single-channel "
        "GAE, and use only the flops value head for the value loss. The other "
        "two value heads stay frozen in this mode.",
    )
    p.add_argument(
        "--preference-conditioned",
        action="store_true",
        help="Stage F: train a single preference-conditioned policy "
        "πθ(a | s, w) where w ∈ Δ^7 over the 8-vec reward simplex. "
        "w is sampled per episode from Dirichlet(α) with α set by "
        "--dirichlet-alpha; the same w drives advantage weighting. "
        "When off, the static --lambda-* CLI weights are used.",
    )
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Concentration parameter for the corner-emphasis "
        "component of the Stage F preference mixture. Values "
        "< 1 emphasise corners and edges of the simplex so "
        "the policy sees pure-FLOP / pure-quality regimes.",
    )
    p.add_argument(
        "--dirichlet-alpha-uniform",
        type=float,
        default=1.0,
        help="Concentration parameter for the uniform-coverage "
        "component of the mixture. Spec calls for both "
        "corners and interior coverage; this is the interior "
        "side. Default α=1.0 yields a uniform Dirichlet.",
    )
    p.add_argument(
        "--dirichlet-mix-ratio",
        type=float,
        default=0.5,
        help="Probability of drawing each per-env preference from "
        "the corner Dirichlet (the rest go through the "
        "uniform component). 0.5 ≈ even mixture (spec "
        "recommendation); 1.0 = corners only; 0.0 = uniform "
        "only.",
    )
    p.add_argument(
        "--set-transformer-agg",
        action="store_true",
        help="Stage B.3: aggregate per-vertex features across the "
        "calibration samples with a learned Set Transformer "
        "instead of a simple mean. Lets the policy distinguish "
        "stable behaviour from sample-specific accidents.",
    )

    # Reporting
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument(
        "--capture-perfect-grads",
        action="store_true",
        help="Allow the top-N quality heap to keep trajectories whose quality "
             "channel reads exactly 1.0 (a perfect Jacobian cosine, or a walk "
             "that wiped the loss out entirely).",
    )
    p.add_argument(
        "--print-top-every",
        type=int,
        default=0,
        help="Print the running top-N Total Reward every N training episodes "
        "(in addition to the post-training + post-calibration dumps). 0 disables. "
        "Useful for short smoke runs where waiting for the end-of-training dump "
        "isn't ergonomic.",
    )

    return p


# ---------------------------------------------------------------------------
# Setup helpers (each builds one chunk of state used by main())
# ---------------------------------------------------------------------------


def _resolve_num_envs(arg_value: int, example: str) -> int:
    if arg_value > 0:
        return arg_value
    if "Vmapped" in example:
        return 16
    return os.cpu_count() or 64


def _resolve_main_device(args):
    if not args.exec_on_gpu:
        return None
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if len(gpus) < 2:
        raise RuntimeError(
            f"--exec-on-gpu requested but only {len(gpus)} GPU(s) found. "
            "Check your --gpus argument and CUDA_VISIBLE_DEVICES."
        )
    return gpus[0]


def _build_factor_table(args):
    factors_py = tuple(_parse_int_list(args.factors))
    if not factors_py:
        raise ValueError("--factors must contain at least one factor value")

    if args.max_rules > MAX_RULES_PER_VERTEX:
        raise ValueError(
            f"--max-rules ({args.max_rules}) exceeds env-side "
            f"MAX_RULES_PER_VERTEX ({MAX_RULES_PER_VERTEX})"
        )
    max_rules = args.max_rules
    factor_table = jnp.array(factors_py, dtype=jnp.int32)
    return factor_table, factors_py, factor_table.shape[0], max_rules


# Comparison-study presets — applied by `_apply_variant_preset` when
# `--variant` is anything other than "custom". Each value is `(factors,
# max_rules, pin_rules_to_exact)`; None means "don't override". Explicit
# CLI flags later in argv still win because argparse picks the last
# occurrence — see `_apply_variant_preset` for the merge order.
VARIANT_PRESETS: dict[str, dict] = {
    "custom": {},
    "ve_only": {"pin_rules_to_exact": True},
    "diag_gcd": {"factors": "-1", "max_rules": 1, "pin_rules_to_exact": False},
    "diag_factor": {
        "factors": "2,3,4,8,16",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    # COMPRESS is wired end-to-end via the typed MicroActionPolicy + the env's
    # COMPRESS_SENTINEL rule_specs encoding + graphax.sparse.apply_compress.
    # The preset's legacy fields (factors / max_rules) only matter when
    # --no-dynamic-substeps is used — the dynamic path consults
    # _op_legality_for_variant("compress") at sample time instead and masks
    # DIAG out, leaving COMPRESS + END.
    "compress": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "full": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": MAX_RULES_PER_VERTEX,
        "pin_rules_to_exact": False,
    },
}


def _apply_variant_preset(args, variant: str | None = None):
    """In-place apply a `--variant` preset to ``args``.

    `variant` overrides ``args.variant`` if given. Raises if the preset is
    not yet wired (currently ``compress``, which depends on the
    atomic-COMPRESS action — see graphax.sparse.micro_actions and the
    heads.py rewrite).
    """
    if getattr(args, "exact", False):
        variant = "ve_only"
        
    name = variant if variant is not None else getattr(args, "variant", "custom")
    if name not in VARIANT_PRESETS:
        raise ValueError(f"Unknown --variant '{name}'. Valid: {list(VARIANT_PRESETS)}.")
    preset = VARIANT_PRESETS[name]
    if preset is None:
        raise NotImplementedError(
            f"--variant '{name}' is not yet wired through the trainer. "
            "It needs the atomic-COMPRESS action emitted by the autoregressive "
            "sub-episode head (see graphax.sparse.micro_actions.apply_compress "
            "and the pending heads.py rewrite). For now, use --variant=full "
            "with --factors=0 in the factor table for a coarse approximation."
        )
    for k, v in preset.items():
        setattr(args, k, v)
    # Make the resolved preset name stick to args.variant. The dynamic-substeps
    # op_legality_override is built from args.variant (not this local `name`), so
    # without this --exact set a local ve_only + pin_rules_to_exact=True but left
    # args.variant at its default — the micro-action policy then ran fully
    # unrestricted (int4 quant + compress) despite --exact, making the "exact"
    # baseline secretly approximate. Keep them in sync.
    args.variant = name


def _op_legality_for_variant(
    variant: str,
    allow_compress: bool,
    allow_quant: bool = True,
) -> jax.Array:
    """Per-variant op-type legality mask for the dynamic action space.

    Maps a comparison-study variant to the ``(NUM_OPS,) = (DIAG,
    COMPRESS, QUANT, END)`` float32 mask consumed by
    :meth:`Agent.sample_action_dynamic`. END is always legal so the
    sub-episode can terminate. ``allow_compress`` / ``allow_quant``
    override the COMPRESS / QUANT slots to 0 for any variant — useful
    for ablations or while the graphax-side wiring is pending.

    * ``custom`` / ``full``: every op legal (gated by the allow flags).
    * ``ve_only``: force END at every sub-step (the dynamic-mode
      analogue of ``--pin-rules-to-exact``).
    * ``diag_gcd`` / ``diag_factor``: DIAG and END only.
    * ``compress``: COMPRESS and END only (requires allow_compress).
    * ``quant``: QUANT and END only (requires allow_quant).
    """
    diag = 1.0
    compress = 1.0 if allow_compress else 0.0
    quant = 1.0 if allow_quant else 0.0
    end = 1.0
    if variant == "ve_only":
        return jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    if variant in ("diag_gcd", "diag_factor"):
        return jnp.array([diag, 0.0, 0.0, end], dtype=jnp.float32)
    if variant == "compress":
        if not allow_compress:
            raise ValueError(
                "Variant 'compress' requires --allow-compress (and the "
                "graphax vertex_elimination_jaxpr rewrite to actually "
                "consume COMPRESS micro-actions through the env)."
            )
        return jnp.array([0.0, compress, 0.0, end], dtype=jnp.float32)
    if variant == "quant":
        if not allow_quant:
            raise ValueError(
                "Variant 'quant' requires allow_quant=True."
            )
        return jnp.array([0.0, 0.0, quant, end], dtype=jnp.float32)
    # `custom` and `full` (and anything else) get the unrestricted mask
    # gated by the allow flags.
    return jnp.array([diag, compress, quant, end], dtype=jnp.float32)


def _build_agent(
    args,
    total_v: int,
    num_factors: int,
    max_rules: int,
    key,
):
    encoder_keys = jrand.split(key, 15)
    embedding = eqx.nn.Embedding(args.vocab_size, args.embd_dim, key=encoder_keys[0])
    # Flag-gated token-mixer for the policy backbone. Resolved FIRST because
    # whether the model needs a positional encoding follows from it.
    _policy = os.environ.get("ALPHAGRAD_POLICY", "transformer").strip().lower()
    if _policy not in ("transformer", "palimpsa", "palimpsa_bi"):
        raise ValueError(
            "ALPHAGRAD_POLICY must be 'transformer', 'palimpsa' or "
            f"'palimpsa_bi', got {_policy!r}"
        )
    # POSITIONAL ENCODING IS A TRANSFORMER REQUIREMENT, NOT A PALIMPSA ONE.
    #
    # Self-attention is permutation-equivariant, so the transformer backbone
    # cannot see order without an explicit signal. Palimpsa can: its carry is a
    # GATED EXPONENTIAL-DECAY accumulation (M = outer + decay*M_prev, decay =
    # exp(-softplus(gate(y_t)) * softplus(g))), so a token k steps back is
    # attenuated by prod(decay) — learned, relative, input-dependent position
    # information. Same reason RWKV / RetNet / Mamba carry no absolute PE.
    #
    # Under append-only it is worse than redundant. `pe` is FIXED SINUSOIDAL
    # ABSOLUTE position, and absolute position in an append-only stream is
    # arbitrary: whether a path lands at token 3000 or 3500 depends on how many
    # tokens earlier eliminations happened to emit, which is a function of the
    # elimination order, not of the content. The sinusoid then gives identical
    # local content different representations for no reason.
    #
    # It is also the ONLY thing that indexes an absolute position into a fixed
    # table, i.e. the only hard MAX_TOKENS bound in the model itself.
    #
    # ALPHAGRAD_POS_ENC=1 forces it back on for an A/B.
    _force_pe = os.environ.get("ALPHAGRAD_POS_ENC", "auto").strip().lower()
    _use_pe = (_policy == "transformer") if _force_pe == "auto" else (
        _force_pe in ("1", "true", "yes"))
    # LEGACY absolute-PE table, transformer backbone only. Unreachable on the
    # live path (incremental encode is mandatory, requires palimpsa, and
    # rejects ALPHAGRAD_POS_ENC=1); sized by its own knob so no part of the
    # model depends on the deleted full-stream budget.
    _pe_len = int(os.environ.get("ALPHAGRAD_POS_ENC_LEN", "8192"))
    pos_enc = PositionalEncoder(args.embd_dim, _pe_len) if _use_pe else None
    print(f"[alphagrad] positional encoding: {'ON' if _use_pe else 'OFF'} "
          f"(backbone={_policy})", flush=True)
    if _policy == "palimpsa":
        print("[alphagrad] policy backbone: PALIMPSA (unidirectional/causal) "
              "linear-attention encoder", flush=True)
    elif _policy == "palimpsa_bi":
        print("[alphagrad] policy backbone: PALIMPSA_BI (bidirectional + "
              "relational-gate) linear-attention encoder", flush=True)
    encoder = make_encoder(
        _policy,
        args.num_layers,
        args.num_heads,
        args.embd_dim,
        args.hidden_dim,
        key=encoder_keys[1],
    )
    if getattr(args, "set_pointer", False):
        from alphagrad.approx.set_pointer import SetPointerVertexPolicy
        # embd_dim, not 2*embd_dim: one slot per vertex, E wide. The
        # [identity || dynamic] concatenation is gone -- a vertex's identity
        # is its own base rows, scattered into the same slot.
        vertex_policy = SetPointerVertexPolicy(
            num_vertices=total_v,
            embd_dim=args.embd_dim,
            num_heads=args.num_heads,
            num_blocks=int(getattr(args, "set_pointer_blocks", 2)),
            key=encoder_keys[2],
        )
    else:
        vertex_policy = PointerVertexPolicy(
            num_vertices=total_v,
            embd_dim=args.embd_dim,
            num_heads=args.num_heads,
            key=encoder_keys[2],
        )
    # One single-output MLP per training reward (latency / peak_memory /
    # cosine_sim). Per-head split keeps gradient scales sane
    # across the qualitatively different reward families and matches the
    # per-head GAE and preference-vector scalarization in `train_episode`.
    value_dims = _parse_int_list(args.value_dims)
    value_head_flops = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[4])
    value_head_mem = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[5])
    value_head_cos = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[12])
    op_embedding = eqx.nn.Embedding(
        OP_TYPE_VOCAB_SIZE,
        args.op_embd_dim,
        key=encoder_keys[6],
    )
    # keys[7] (vertex_feature_proj, later ctx_proj), [8] (the identity pool),
    # [9] (the B.4 residual module) and [10] (the Set-Transformer
    # aggregator) are DEAD SLOTS. They are left
    # unused rather than re-packed: every later key is positional, so
    # re-indexing would move pref_proj and the approximation heads onto
    # different randomness and change every seeded run for no reason.
    pref_proj = eqx.nn.Linear(NUM_VALUE_HEADS, args.embd_dim, key=encoder_keys[11])
    # Dynamic-substeps head: only constructed when the flag is on so the
    # default agent stays leaner (one extra encoder + MicroActionHead is
    # non-trivial parameter cost).
    if getattr(args, "no_approx_head", False) or getattr(
            args, "live_faces", False):
        # REMOVED, not masked: no approximation head is constructed, so the
        # pytree holds no approximation parameters at all. Under --live-faces
        # approximation is purely PER-FACE (the 94-head): the per-vertex
        # rules are always the exact END rows, so a per-vertex head would be
        # dead weight with a live gradient path.
        micro_action_policy = None
    elif getattr(args, "dynamic_substeps", False) and getattr(args, "unified_head", False):
        # ONE flat head per vertex instead of the autoregressive sub-episode.
        # Presented through MicroActionPolicy's sample/evaluate contract so the
        # env decoder, Trajectory and PPO loss are untouched.
        from alphagrad.approx.unified_micro import UnifiedMicroPolicy
        micro_action_policy = UnifiedMicroPolicy(
            embd_dim=args.embd_dim,
            max_substeps=args.max_substeps,
            key=encoder_keys[13],
        )
    elif getattr(args, "dynamic_substeps", False):
        micro_action_policy = MicroActionPolicy(
            embd_dim=args.embd_dim,
            num_heads=args.num_heads,
            max_substeps=args.max_substeps,
            num_encoder_layers=1,
            max_groups=max(args.max_substeps, 16),
            key=encoder_keys[13],
            use_group_embedding=getattr(args, "axis_group_embedding", False),
        )
    else:
        micro_action_policy = None
    if getattr(args, "face_actions", False) and not getattr(
            args, "no_approx_head", False) and getattr(
            args, "unified_face_head", False):
        # 94 outputs = 32*3 - 2: ONE skip Bernoulli for the whole face (it
        # deletes the contraction, so it is a property of the face, not of an
        # operand slot) plus 31 fields for each of the pre/post/new slots,
        # from a single MLP forward. The reduce axis is a SOFTMAX over 9, so
        # one slot IS one rule row -- that is what makes max_substeps=1
        # structural and deletes _emit.
        face_path_policy = UnifiedFacePolicy(
            embd_dim=args.embd_dim,
            num_heads=args.num_heads,
            max_faces=ENV_MAX_FACES,
            num_encoder_layers=1,
            max_groups=max(args.max_substeps, 16),
            key=encoder_keys[14],
            use_group_embedding=getattr(args, "axis_group_embedding", False),
        )
    elif getattr(args, "face_actions", False) and not getattr(
            args, "no_approx_head", False):
        # FacePathPolicy is retired: 32 encoder + 24 head calls per vertex,
        # python-unrolled -- it cannot compile at the derived face width, and
        # its per-slot embeddings are exactly the label-not-content design
        # the 94-head replaced.
        raise ValueError(
            "--face-actions now requires --unified-face-head.")
    else:
        face_path_policy = None
    return Agent(
        embedding=embedding,
        pos_enc=pos_enc,
        encoder=encoder,
        vertex_policy=vertex_policy,
        value_head_flops=value_head_flops,
        value_head_mem=value_head_mem,
        value_head_cos=value_head_cos,
        op_embedding=op_embedding,
        pref_proj=pref_proj,
        num_vertices=total_v,
        num_value_heads=NUM_VALUE_HEADS,
        max_rules=max_rules,
        num_pair_choices=NUM_PAIR_CHOICES,
        num_factors=num_factors,
        embd_dim=args.embd_dim,
        op_embd_dim=args.op_embd_dim,
        micro_action_policy=micro_action_policy,
        max_substeps=int(getattr(args, "max_substeps", 16)),
        face_path_policy=face_path_policy,
    )


def _scale_output_heads(agent, scale: float):
    """Scale policy-head weights so the initial action distribution is near-uniform."""
    # Vertex-head logit magnitude. PointerVertexPolicy scores through
    # pointer_proj; SetPointerVertexPolicy scores through
    # (k_proj(h) . q_proj(summary))/sqrt(E), so scaling k_proj scales
    # the logits the same way and gives the same near-uniform init.
    if hasattr(agent.vertex_policy, "pointer_proj"):
        agent = scale_module_weight(
            agent, lambda a: a.vertex_policy.pointer_proj.weight, scale
        )
    else:
        agent = scale_module_weight(
            agent, lambda a: a.vertex_policy.k_proj.weight, scale
        )
    # There is no identity pool to leave un-zeroed any more, and no need for
    # one: the v31 uniform-pick failure was a slot that stayed EMPTY until
    # something touched it, and a vertex's slot now holds its own base rows
    # from step 0. The discrimination is content, not initialisation.
    # F: zero the preference projection so the conditioned and
    # unconditioned paths produce identical step-0 policies on the same seed.
    agent = scale_module_weight(
        agent,
        lambda a: a.pref_proj.weight,
        0.0,
    )
    # Dynamic-substeps heads — scale the four output projections so the
    # initial typed-action distribution is also near-uniform. The base
    # encoder and AxisSetEncoder still contribute non-zero magnitude
    # signal, so "near uniform" rather than "exactly uniform"; this is
    # the same trade-off the legacy rule-head scaling makes.
    if agent.micro_action_policy is not None:
        if hasattr(agent.micro_action_policy.head, "op_head"):
            agent = scale_module_weight(
                agent,
                lambda a: a.micro_action_policy.head.op_head.proj.weight,
                scale,
            )
            agent = scale_module_weight(
                agent,
                lambda a: a.micro_action_policy.head.axis_i_head.key_proj.weight,
                scale,
            )
            agent = scale_module_weight(
                agent,
                lambda a: a.micro_action_policy.head.axis_j_head.key_proj.weight,
                scale,
            )
            agent = scale_module_weight(
                agent,
                lambda a: a.micro_action_policy.head.factor_head.head_proj.weight,
                scale,
            )
        else:
            # UnifiedApproxHead: every field shares ONE output projection, so
            # scaling its final Linear scales all 64 logits together and gives
            # the same near-uniform initial distribution the four separate
            # sub-head scalings gave the autoregressive head.
            agent = scale_module_weight(
                agent,
                lambda a: a.micro_action_policy.head.proj.layers[-1].weight,
                scale,
            )
    return agent


def _mask_vertex_logits(vertex_logits, vertex_avail_mask):
    """Availability mask that holds for ANY finite logits.

    `-inf` rather than `-1e9`: an absolute sentinel is only a mask while the
    legal logits sit above it, and the set pointer emits exactly -1e9 for its
    own unoccupied slots. Two sentinels of equal magnitude make the vector
    constant, and softmax(constant) is UNIFORM -- the mask silently inverts
    into "sample anything". With -inf the illegal entries are exactly 0 after
    softmax no matter what the legal ones are.

    If nothing is legal, fall back to a flat vector over everything: a
    softmax of all -inf is NaN, which would propagate into the sampler and
    the log-prob rather than failing loudly.
    """
    legal = vertex_avail_mask > 0.5
    any_legal = jnp.any(legal)
    return jnp.where(
        any_legal,
        jnp.where(legal, vertex_logits, -jnp.inf),
        jnp.zeros_like(vertex_logits),
    )


def _action_to_pylist_dynamic(
    vertex_seq,
    op_seq,
    i_seq,
    j_seq,
    factor_seq,
    kind_seq,
    quant_seq,
    max_substeps,
):
    """Decode typed micro-action sequences into copy-pastable per-vertex lists.

    Each entry is ``(vertex, [<call>, ...])`` where ``<call>`` is one of:

    * ``diag(i, j, factor)`` for an OP_DIAG sub-step.
    * ``compress("kind", axis)`` for an OP_COMPRESS sub-step (the kind
      is the string from :data:`COMPRESS_KINDS` at the sampled index).
    * ``quant("dtype")`` for an OP_QUANT sub-step (the dtype is the
      string from :data:`QUANT_DTYPES` at the sampled index).

    OP_END (and every sub-step past it) is dropped from the output, so a
    vertex whose sub-episode is just OP_END renders as ``(v, [])`` —
    matching how the user reads the no-approximation case.
    """
    out = []
    for v_idx, op_row, i_row, j_row, f_row, k_row, q_row in zip(
        vertex_seq,
        op_seq,
        i_seq,
        j_seq,
        factor_seq,
        kind_seq,
        quant_seq,
    ):
        steps: list[str] = []
        for slot in range(max_substeps):
            op = int(op_row[slot])
            if op == OP_END:
                break
            if op == OP_DIAG:
                steps.append(
                    f"diag({int(i_row[slot])}, {int(j_row[slot])}, {int(f_row[slot])})"
                )
            elif op == OP_COMPRESS:
                k = int(k_row[slot])
                if 0 <= k < len(COMPRESS_KINDS):
                    kind_name = COMPRESS_KINDS[k]
                else:
                    kind_name = f"kind{k}"
                steps.append(f"compress({kind_name!r}, {int(i_row[slot])})")
            elif op == OP_QUANT:
                q = int(q_row[slot])
                if 0 <= q < len(QUANT_DTYPES):
                    dtype_name = QUANT_DTYPES[q]
                else:
                    dtype_name = f"dtype{q}"
                steps.append(f"quant({dtype_name!r})")
            else:
                steps.append(f"op{op}({int(i_row[slot])}, {int(j_row[slot])})")
        out.append((int(v_idx) + 1, steps))
    return out


# ---------------------------------------------------------------------------
# Reward-weight helpers (Stage A: backward-compatible mapping from the legacy
# `--cmp-type` / `--mem-type` / `--rewards` CLI surface onto the canonical
# 8-component reward vector). Stage F will replace this with full preference
# conditioning over the simplex.
# ---------------------------------------------------------------------------


_CMP_TYPE_TO_REWARD = {
    "graphax": "muls_adds_fmas",
    "flops": "flops",
    "latency": "latency_ns",
}
_MEM_TYPE_TO_REWARD = {
    "graphax": "max_io_sum",
    "bytes_accessed": "bytes_accessed",
    "peak_memory": "peak_memory",
    # DEPRECATED ALIAS (see common.reward_scaling): one memory channel.
    "xla_peak_memory": "peak_memory",
}


def _cmp_reward_index(cmp_type: str) -> int:
    return REWARD_INDEX[_CMP_TYPE_TO_REWARD[cmp_type]]


def _mem_reward_index(mem_type: str) -> int:
    from alphagrad.approx.common.reward_scaling import warn_deprecated_mem_type
    return REWARD_INDEX[_MEM_TYPE_TO_REWARD[warn_deprecated_mem_type(mem_type)]]



def _dump_pareto(archive, args, ep, *, final=False):
    """Persist the front + a replayable best_sequences.json. Never raises."""
    if archive is None or not getattr(archive, "pts", None):
        return
    try:
        import json as _json
        import os as _os
        try:
            _dir = wandb.run.dir if wandb.run is not None else "."
        except Exception:
            _dir = "."
        _os.makedirs(_dir, exist_ok=True)
        archive.dump_front(
            _os.path.join(_dir, "pareto_front.json"),
            extra={"episode": int(ep), "final": bool(final),
                   "run_name": getattr(args, "name", None)},
        )
        # Replayable form. Objective 0 is the compute channel and objective 1
        # memory (see ParetoArchive construction); lower is better in the
        # archive's minimisation convention, so rank by objective 0.
        _pts = [list(map(float, p)) for p in archive.pts]
        _order = sorted(range(len(_pts)), key=lambda i: _pts[i][0])
        _doc = {
            "best_overall": {"seq": archive.seqs[_order[0]],
                             "obj": _pts[_order[0]]},
            "best_per_channel": {
                f"rank{r}": {"seq": archive.seqs[i], "obj": _pts[i]}
                for r, i in enumerate(_order)
            },
            "_provenance": {"source": "ParetoArchive.dump", "episode": int(ep),
                            "num_points": len(_pts)},
        }
        with open(_os.path.join(_dir, "best_sequences.json"), "w") as _fh:
            _json.dump(_doc, _fh, indent=2)
    except Exception as _exc:
        try:
            tqdm.write(f"[pareto-dump] failed at ep={ep}: {_exc!r}",
                       file=sys.stderr)
        except Exception:
            pass


def _build_reward_weights(args) -> np.ndarray:
    """Map legacy CLI flags to a (NUM_REWARDS,) advantage-weight vector.

    `--rewards` selects which families contribute; within a family the weight
    lands on the canonical component picked by `--cmp-type` / `--mem-type`.
    Quality terms: cosine gets weight 1.0 (matching legacy behaviour) when
    "acc" is in `--rewards`.

    Returned as an 8-vec for *host-side display* only (top-N heaps, mean
    return printout). Training-side weighting uses the 3-vec from
    :func:`_build_head_weights`.
    """
    weights = np.zeros(NUM_REWARDS, dtype=np.float32)
    if "cmp" in args.rewards:
        weights[_cmp_reward_index(args.cmp_type)] = args.lambda_cmp
    if "mem" in args.rewards:
        weights[_mem_reward_index(args.mem_type)] = args.lambda_mem
    if "acc" in args.rewards:
        weights[REWARD_INDEX["cosine_sim"]] = 1.0
    return weights


def _build_head_weights(args) -> np.ndarray:
    """Build the (NUM_VALUE_HEADS,) = (3,) static preference vector.

    Indexes the three training rewards (latency / peak_memory / cosine_sim) —
    the value head and advantage path operate on exactly these. ``"acc" in
    --rewards`` weights the COSINE head (this used to silently weight frob
    while cosine never trained — the root cause of the zero-compute collapse).
    The `--cmp-type` and `--mem-type` flags only affect host-side display.
    """
    weights = np.zeros(NUM_VALUE_HEADS, dtype=np.float32)
    if "cmp" in args.rewards:
        weights[0] = args.lambda_cmp
    if "mem" in args.rewards:
        weights[1] = args.lambda_mem
    if "acc" in args.rewards:
        weights[2] = args.lambda_acc
    return weights


# ---------------------------------------------------------------------------
# Stage D: per-head learning-rate ramp
# ---------------------------------------------------------------------------


# Single source of truth for the per-head parameter taxonomy. Each entry maps
# a head label to the substrings that identify that head's parameters in the
# agent pytree's path strings. ``defstructure``-style: add a label here once
# and every downstream consumer (LR ramp, freeze mask, calibration mask) sees
# it. Keep the entries narrow — the smallest set of params whose updates are
# *exclusively* driven by that head's gradient signal.
_HEAD_PATH_MARKERS: dict[str, tuple[str, ...]] = {
    "axis": (
        "micro_action_policy.head.i_head",
        "micro_action_policy.head.j_head",
    ),
    "factor": (
        "micro_action_policy.head.exp_head",
        "micro_action_policy.head.factor",
    ),
    # The identity pool it named is deleted; the key stays so `_path_in`'s
    # callers keep a stable alphabet, and it now matches nothing.
    "aggregator": (),
}


def _path_in(path: str, head: str) -> bool:
    return any(marker in path for marker in _HEAD_PATH_MARKERS[head])


def _build_param_mask(agent, predicate) -> "jax.Array":
    """Build a bool pytree aligned with ``eqx.filter(agent, eqx.is_inexact_array)``
    where each leaf is True iff its path matches ``predicate(path_str)``."""
    params = eqx.filter(agent, eqx.is_inexact_array)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    return treedef.unflatten(
        [
            jnp.full_like(leaf, predicate(jax.tree_util.keystr(path)), dtype=jnp.bool_)
            for path, leaf in leaves_with_path
        ]
    )


def _build_head_masks(agent):
    """Return ``(axis_mask, factor_mask, vertex_mask, micro_mask)`` — bool trees for the LR ramp.

    `vertex_mask` selects every parameter belonging to the base pointer
    net (``vertex_policy.*``). `micro_mask` selects every parameter of
    the dynamic-substeps head (``micro_action_policy.*``) — both the
    AxisSetEncoder and the three sub-heads (op_type / axis pointers /
    prime-exponent). They are used by :func:`_scale_grads` to apply
    per-head LR multipliers (the axis / factor warm-up ramp; the vertex
    and micro heads run at the base LR).

    For agents without a dynamic head (``--dynamic-substeps`` off,
    ``micro_action_policy is None``), the micro_mask is empty since
    None children don't appear in the pytree.
    """
    return (
        _build_param_mask(agent, lambda p: _path_in(p, "axis")),
        _build_param_mask(agent, lambda p: _path_in(p, "factor")),
        _build_param_mask(agent, lambda p: "vertex_policy" in p),
        _build_param_mask(agent, lambda p: "micro_action_policy" in p),
    )


def _head_lr_mult(step: "jax.Array", warmup_steps: int) -> "jax.Array":
    """Linear warm-up multiplier from 1/3 → 1 over `warmup_steps` steps.

    Matches §3.2 of the architecture spec with ``T_h = 0`` (head introduced
    at the start of this run). Returns 1.0 when ``warmup_steps <= 0``.
    """
    if warmup_steps <= 0:
        return jnp.array(1.0, dtype=jnp.float32)
    frac = jnp.minimum(step.astype(jnp.float32) / float(warmup_steps), 1.0)
    return (1.0 / 3.0) + (2.0 / 3.0) * frac


def _scale_grads(
    grads,
    axis_mask,
    factor_mask,
    vertex_mask,
    micro_mask,
    freeze_mask,
    axis_mult,
    factor_mult,
    vertex_mult,
    micro_mult,
):
    """Single fused per-leaf gradient scaling.

    Combines the Stage D/E head-LR ramp (per-head multiplier on axis / factor
    params), the vertex_policy / micro_action_policy head multipliers,
    and the Stage G freeze mask (zero gradient
    for non-trainable params during calibration) into one pass through
    the pytree. Mask precedence (first match wins):

        freeze → axis → factor → vertex → micro → 1.0
    """
    return jax.tree_util.tree_map(
        lambda g, am, fm, vm, mm, fz: jnp.where(
            fz,
            g
            * jnp.where(
                am,
                axis_mult,
                jnp.where(
                    fm,
                    factor_mult,
                    jnp.where(
                        vm,
                        vertex_mult,
                        jnp.where(mm, micro_mult, 1.0),
                    ),
                ),
            ),
            jnp.zeros_like(g),
        ),
        grads,
        axis_mask,
        factor_mask,
        vertex_mask,
        micro_mask,
        freeze_mask,
    )




# `_episode_vertex_features` lived here: the per-episode hand-written
# per-vertex feature matrix (op-type id + calibration-derived scalars), fed to
# `Agent._data_embedding`. Both are gone -- a vertex is described by the
# palimpsa rows of its OWN base tokens, scattered into its own slot by
# `carry_stream.init_carry`, which is content rather than a feature list and
# needs no calibration samples.
# `compute_vertex_features` / `compute_per_sample_vertex_features` remain in
# heads.py with no caller here; `--set-transformer-agg` is accepted and inert.


# ---------------------------------------------------------------------------
# Stage C: Markowitz behaviour-clone warm-start
# ---------------------------------------------------------------------------


def _repo_commits() -> dict:
    """Short SHAs of the repos this process is actually running from.

    Resolved from each module's own file location (not a hardcoded ~/dsnn),
    so a run launched from a worktree logs that worktree's HEAD rather than
    some other checkout's — run provenance that can't silently lie.
    """
    import subprocess
    from pathlib import Path

    out = {}
    try:
        import graphax as _gx
        roots = {
            "alphagrad": Path(__file__).resolve(),
            "graphax": Path(_gx.__file__).resolve(),
        }
    except Exception:
        roots = {"alphagrad": Path(__file__).resolve()}
    for name, path in roots.items():
        try:
            sha = subprocess.run(
                ["git", "-C", str(path.parent), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            out[f"commit/{name}"] = sha or "unknown"
        except Exception:
            out[f"commit/{name}"] = "unknown"
    return out


def _setup_jax_compile_cache() -> None:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.compile_cache.setup_jax_compile_cache`.

    The default now points at a per-SLURM-job, per-node `/tmp/dsnn-jax-cache-...`
    directory (previously: shared NFS ``~/.cache/jax-compilation-cache/<host>``).
    Sbatch scripts that need cross-job reuse can set
    ``DSNN_JAX_CACHE_REUSE=1`` before invocation.
    """
    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache
    setup_jax_compile_cache()


def main():
    args = make_argparser().parse_args()

    # ``ALPHAGRAD_TRACEMALLOC=1`` — start the Python allocator tracker
    # before any model code runs. Per-episode snapshots are diffed
    # against the previous one (see host_log) to identify Python lines
    # that allocate the most bytes / episode. Won't see C++ leaks
    # (XLA, jax_memory_monitor's MemoryTracker, glibc malloc pool growth)
    # but closes the loop on whether the leak has a Python ref-keeping
    # side. Enabled here, before any imports finish, so the tracker is
    # active for all subsequent allocations. ~5-10% overhead.
    if os.environ.get("ALPHAGRAD_TRACEMALLOC", "0") == "1":
        import tracemalloc

        tracemalloc.start()
        print(
            "[experiment] tracemalloc started "
            "(ALPHAGRAD_TRACEMALLOC=1)",
            flush=True,
        )

    # Apply the --variant preset onto args before anything else looks at
    # args.factors / args.max_rules / args.pin_rules_to_exact. `custom` is a
    # no-op; other variants overwrite those three flags. Explicit CLI values
    # passed alongside --variant are clobbered — pick `custom` if you want
    # to mix-and-match.
    _apply_variant_preset(args)
    if args.variant != "custom":
        print(
            f"--variant={args.variant} applied: factors={args.factors!r}, "
            f"max_rules={args.max_rules}, "
            f"pin_rules_to_exact={args.pin_rules_to_exact}"
        )

    variant_label = "pointer+micro-actions"

    # 3b mode preconditions — fail fast, not 20 minutes into a compile. The
    # carry is a CAUSAL recurrence: it needs the unidirectional palimpsa
    # backbone (no pos-enc, no bidirectional second pass) and the append-only
    # token stream (a re-traced stream has no prefix property to extend).
    #
    # STAGE 2: no longer optional. The full-stream observation, and with it
    # the non-incremental rollout and loss, is gone -- the env emits one
    # DELTA per step and only the carry can consume that. The CLI flag stays
    # for launcher compatibility; the preconditions are now requirements.
    args.incremental_encode = True
    _pol = os.environ.get("ALPHAGRAD_POLICY", "transformer").strip().lower()
    if _pol != "palimpsa":
        raise ValueError(
            "the delta observation requires ALPHAGRAD_POLICY=palimpsa "
            f"(causal); got {_pol!r} (palimpsa_bi's reverse pass cannot "
            "ride a causal carry, and the transformer needs absolute "
            "positions)."
        )
    if os.environ.get("ALPHAGRAD_POS_ENC", "auto").strip().lower() in (
        "1", "true", "yes"
    ):
        raise ValueError(
            "the delta observation is incompatible with "
            "ALPHAGRAD_POS_ENC=1 (there is no absolute position to encode: "
            "the policy never sees the whole stream)."
        )
    if os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "0") != "1":
        raise ValueError(
            "the delta observation requires "
            "ALPHAGRAD_INCREMENTAL_TOKENS=1 (a per-step DELTA only exists "
            "under the append-only tokenizer; the extract_jaxpr path "
            "re-tokenizes the whole Jacobian every step)."
        )
    if not args.dynamic_substeps:
        raise ValueError(
            "the delta observation is only wired for --dynamic-substeps."
        )

    main_device = _resolve_main_device(args)
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    # Persistent JIT disk cache. Off-by-env when investigating memory leaks:
    # the cache loader may retain in-memory references to every loaded
    # ``Executable``, masquerading as an XLA C++ leak in profiling. Set
    # ``ALPHAGRAD_DISABLE_JIT_DISK_CACHE=1`` to skip wiring up
    # ``compilation_cache.set_cache_dir(...)`` for that experiment.
    if os.environ.get("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "0") != "1":
        _setup_jax_compile_cache()
    else:
        print(
            "[experiment] JIT disk cache disabled "
            "(ALPHAGRAD_DISABLE_JIT_DISK_CACHE=1)",
            flush=True,
        )

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    # Resolve example, build env, derive masks.
    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and (
        args.example.endswith("NeuralNetwork")
        or args.example.startswith("TransformerLM"))
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    if args.measure_grad:
        # Wrap BEFORE tracing: closed_jaxpr, the mask oracle, and the env's
        # measured executable must all address the SAME scalar-loss graph —
        # wrapping only at measurement time is the graph-mismatch that
        # produced the old stack's zero-gradient bug.
        pass
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(
        args.example, dataset=dataset_for_call, dataset_size=args.dataset_size
    )
    # GRAD-TARGET SETUP — routed through the shared builder so the trainer and
    # every measure-actor construct the IDENTICAL graph (jaxpr / vertex+action
    # space / argnums). Three modes:
    #
    #   neither flag        -> raw Jacobian target (the historical default).
    #   --measure-grad      -> scalar_loss_fn: mean BEFORE tracing. Measures the
    #                          gradient, but the traced graph IS the loss graph,
    #                          so the ACTION SPACE CHANGES (measured: 15 vertices
    #                          vs 13 for the Jacobian graph).
    #   + --seed-vertices   -> seed_loss_fn: the tangent seed `t` and the
    #                          <ones/N, .> adjoint contraction become ORDINARY
    #                          ELIMINABLE VERTICES. The graph stays the
    #                          Jacobian-elimination graph (plus the seed nodes),
    #                          and the policy chooses WHEN to apply the seed —
    #                          i.e. forward / reverse / cross-country seeding is
    #                          part of the search rather than hardcoded. Seeding
    #                          early costs one VJP (gradient-like); seeding late
    #                          builds the Jacobian. This is the "act in Jacobian
    #                          space, measure in grad space" configuration.
    #
    # `xs` gains the appended tangent seed and `argnums` shifts accordingly, so
    # both must come back from the builder rather than being recomputed.
    from alphagrad.approx.common import grad_target_setup as _grad_target_setup
    target_fn, xs, argnums = _grad_target_setup(args, target_fn, xs, args.example)
    closed_jaxpr = _traced_inlined(target_fn, xs)
    # Always pass target_fun so flops/bytes_accessed/latency_ns/peak_memory
    # populate every step (see cpu_approx_worker.py for the full rationale).
    env_target_fun = target_fn

    # Latency is the only optional component of the reward harness; auto-enable
    # measurement when the user has selected it as their primary compute metric
    # so the reward isn't silently zeroed out.
    measure_latency = args.measure_latency or args.cmp_type == "latency"
    # Under ALPHAGRAD_INCREMENTAL_TOKENS=1 the env's tokenizer guards its id
    # space against this embedding size (an out-of-range gather CLAMPS
    # silently); publish it where the host callback can see it.
    os.environ["ALPHAGRAD_VOCAB_SIZE"] = str(int(args.vocab_size))
    # QUALITY CHANNEL — published to the ENVIRONMENT, not passed as an
    # argument, because the Ray measure actors run env._callback in their own
    # processes and this codebase's dominant bug class is "two paths that must
    # agree". One env var read by one function (env.quality_metric) in one
    # module makes disagreement impossible. Set BEFORE ray.init so every actor
    # inherits it.
    # ORDER-ONLY / EXACT ARM: with --no-approx-head no plan can approximate
    # anything, so every plan returns the EXACT gradient and the quality
    # channel is a CONSTANT (measured on TLM: 0.88532-0.88533 on every plan of
    # every arm, 503/503 progress samples of job 59311). A constant channel
    # contributes exactly zero gradient while the loss-drop walk that produces
    # it costs 200 executions of the plan -- twice the entire latency budget.
    # Resolve "auto" to "none" there and say so; an explicit --quality-metric
    # is always honoured.
    _qm = str(args.quality_metric)
    if _qm == "auto" and bool(getattr(args, "no_approx_head", False)):
        _qm = "none"
        print("[alphagrad] ORDER-ONLY arm (--no-approx-head): the quality "
              "channel is constant by construction, so it is NOT computed "
              "(--quality-metric none). Pass --quality-metric loss_drop to "
              "force it.", flush=True)
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = _qm
    os.environ["ALPHAGRAD_WALK_STEPS"] = str(int(args.walk_steps))
    os.environ["ALPHAGRAD_WALK_LR"] = repr(float(args.walk_lr))
    os.environ["ALPHAGRAD_WALK_PROBE_SEED"] = str(int(args.walk_probe_seed))
    os.environ["ALPHAGRAD_WALK_NOISE_STD"] = repr(float(args.walk_noise_std))
    global _QUALITY_METRIC
    from types import SimpleNamespace as _NS
    _QUALITY_METRIC = _env_quality_metric(
        _NS(measure_grad=bool(args.measure_grad)))
    print(
        f"[alphagrad] quality channel (reward slot 6, --lambda-acc) = "
        f"{_QUALITY_METRIC}"
        + (f" (walk: {int(args.walk_steps)} Adam steps, lr {args.walk_lr:g}, "
           f"probe seed {int(args.walk_probe_seed)}, noise std "
           f"{args.walk_noise_std:g})" if _QUALITY_METRIC == "loss_drop"
           else (" (NOT COMPUTED -- reward slot 6 stays 0.0; the cost "
                 "channels are unaffected)" if _QUALITY_METRIC == "none"
                 else " (Jacobian cosine vs the exact reference)")),
        flush=True)
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=xs,
        argnums=argnums,
        num_envs=0,
        data_gen=gen,
        target_fun=env_target_fun,
        cmp_type=args.cmp_type,
        mem_type=args.mem_type,
        exec_on_gpu=args.exec_on_gpu,
        measure_latency=measure_latency,
        num_data_points=int(args.num_data_points),
        reps_per_point=int(args.reps_per_point),
        latency_inner_reps=int(args.latency_inner_reps),
        # --face-actions IMPLIES per-face legality masking for the per-vertex
        # micro rules too: face slots are already live-mask-hooked, but a raw
        # per-vertex rule that doesn't fit one face's operand would hit the
        # strict TRANSFORM-DID-NOT-FIT guard and kill the episode's
        # tokenization (found by the 3b smoke: Compress on an implicit-dim
        # edge, legal by the logical-axis oracle, unappliable on the 1-D val).
        per_face=bool(args.per_face or args.face_actions),
        measure_grad=bool(args.measure_grad),
        terminal_rewards_only=args.terminal_rewards_only,
        # STAGE 2: emit the per-step token DELTA, not the growing stream.
        delta_obs=True,
    )

    # THE BASE STREAM, once, on the host. `len(base_tokens())` depends only on
    # the jaxpr -- not on the elimination order -- so this is a constant every
    # env and every episode shares, and its length is the tokenizer's own
    # rather than a device-side non-zero count over a padded buffer (which is
    # what token id 0, the literal '-', used to corrupt). Sliced to its exact
    # length, so the base encode scan is exactly as long as the base is.
    _BASE_TOK, _BASE_EQN, _BASE_N = env.base_observation()
    _BASE_W = max(int(_BASE_N), 1)
    _BASE_TOK = _BASE_TOK[:_BASE_W]
    _BASE_EQN = _BASE_EQN[:_BASE_W]
    # Per-token owning VERTEX for the base stream (1-based, 0 = none). It is
    # the KEY of the base scatter -- which vertex slot each base row lands in
    # -- and it is read by the rollout, the loss and AZ, so it is resolved
    # ONCE here.
    try:
        _BASE_OWN = env.base_owners()
    except Exception:
        _BASE_OWN = None
    print(f"[alphagrad] base token stream: {int(_BASE_N)} tokens "
          f"(per-step delta budget {MAX_DELTA_TOKENS})", flush=True)
    # THE BUDGET IS FREE ONLY IF THE SCAN IS PREFIX-PROPORTIONAL, and by
    # default it is NOT: `_extend_sequential` falls back to a FLAT
    # `lax.scan` over the whole window when ALPHAGRAD_EXTEND_CHUNK is 0
    # (the shipped default), so the cost follows `window`, not `count`.
    # Measured on one Blackwell GPU, E=32/L=3/H=2, one `encode_extend`:
    #   chunk=0    window  1024, count    78 -> fwd  18.2 ms / grad   84 ms
    #   chunk=0    window 32768, count    78 -> fwd 529.1 ms / grad 2966 ms
    #   chunk=0    window 32768, count 25737 -> fwd 528.7 ms / grad 2965 ms
    #   chunk=256  window  1024, count    78 -> fwd   4.1 ms / grad   27 ms
    #   chunk=256  window 32768, count    78 -> fwd   4.1 ms / grad   67 ms
    # i.e. flat in `count` and linear in `window` at chunk=0 (29x for the
    # 1024 -> 32768 raise), and flat in `window` at chunk=256. Say so.
    if int(os.environ.get("ALPHAGRAD_EXTEND_CHUNK", "0")) <= 0 \
            and MAX_DELTA_TOKENS > 2048:
        print(f"[alphagrad] WARNING: ALPHAGRAD_EXTEND_CHUNK is 0, so every "
              f"encode_extend scans all {MAX_DELTA_TOKENS} window steps "
              f"whatever the delta's real length is. Set "
              f"ALPHAGRAD_EXTEND_CHUNK (256 measured well) to make the "
              f"scan prefix-proportional; otherwise the delta budget is "
              f"paid in full every step.", flush=True)

    # ---- --ray-measure: fan the measurement callback out over Ray actors ----
    if int(getattr(args, "ray_measure", 0) or 0) > 0:
        _n_actors = int(args.ray_measure)
        # P3: the pool stack carries face_specs/face_skips end-to-end now
        # (cpu_approx_{worker,actors,pool} pass-through; env's remote
        # closure ships them and keeps exec_on_gpu TERMINAL rows local),
        # so --face-actions no longer needs to be refused here.
        if os.environ.get("ALPHAGRAD_BATCHED_CALLBACK", "0") != "1":
            raise ValueError(
                "--ray-measure needs ALPHAGRAD_BATCHED_CALLBACK=1; without it "
                "the callback is invoked once per env and the pool would add "
                "Ray IPC with no parallelism."
            )
        import ray as _ray
        from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
        from alphagrad.approx.cpu_approx_pool import CpuApproxPool

        if not _ray.is_initialized():
            # PER-JOB Ray session dir. Two Ray jobs on ONE node collided:
            # each sbatch prologue did  (the stale-cluster
            # fix) and so deleted the other job's live session, producing
            # "The current node timed out during startup". RAY_TMPDIR gives
            # each job its own directory, so no prologue wipe is needed.
            _rt = os.environ.get("RAY_TMPDIR") or None
            # #77 MITIGATION. The raylet blocks on the dashboard/metrics
            # AGENT publishing its port file, and that agent fails to
            # import its deps on these nodes -- `include_dashboard=False`
            # does not stop it, because the UI and the agent are separate
            # processes. Three launches have been lost to this.
            # Ray maps RAY_<system_config> env vars onto its internal
            # config, so this asks it not to start metrics collection.
            # Set as an ENV VAR, not ray.init(_system_config=...), on
            # purpose: an unrecognised _system_config KEY raises at init
            # and would turn an intermittent failure into a certain one,
            # while an unrecognised env var is ignored.
            # UNVERIFIED by construction -- the failure is intermittent
            # (attempt 1 succeeded while 2 and 3 failed on the same node
            # in the same window). If startup fails again, read
            # raylet.err under $RAY_TMPDIR and confirm the agent is still
            # the blocker before crediting this. The 3-attempt retry
            # below remains the real safety net.
            os.environ.setdefault("RAY_enable_metrics_collection", "0")
            _kw = dict(ignore_reinit_error=True, include_dashboard=False)
            if _rt:
                os.makedirs(_rt, exist_ok=True)
                _kw["_temp_dir"] = _rt
            # Ray sizes its worker pool from the MACHINE's cpu count, not the
            # SLURM allocation. On the 128-CPU node it therefore tried to
            # start ~128 workers inside a 32-CPU allocation and the raylet
            # missed its startup deadline on every retry, while the same code
            # started fine on the 64-CPU nodes. Match the allocation.
            _ncpu = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
                "SLURM_JOB_CPUS_PER_NODE")
            try:
                if _ncpu:
                    _kw["num_cpus"] = max(2, int(str(_ncpu).split("(")[0]))
            except Exception:
                pass
            # NOTE ray.init() does NOT accept port / node_manager_port /
            # object_manager_port / min_worker_port / max_worker_port -- those
            # are `ray start` (RayParams) options and raise "Unknown keyword
            # argument(s)" here. So port windows cannot be set from Python.
            #
            # CORRECTION (2026-08-05). An earlier version of this comment
            # claimed the gpu19 startup failures were two Ray heads colliding
            # on default ports and prescribed "one Ray job per node". THAT WAS
            # WRONG, and the counter-evidence was already available: 58362 and
            # 58364 were ALONE on an idle gpu19 and failed anyway, and gpu20
            # (the other dual-socket 8-GPU node) co-hosts Ray fine.
            #
            # What actually holds: gpu19 is not congenitally broken -- Ray
            # started on it six times on 2026-07-31 (56860/56861/56863/56864/
            # 56866/56875). EVERY failure falls in one ~6h window on
            # 2026-08-05 (00:49-05:03 CEST), beginning at the instant job
            # 58274 -- which uses no Ray at all -- died on gpu19 in a GPU-OOM
            # meltdown. The node entered a bad state and stayed there;
            # leading hypothesis is raylet startup blowing Ray's hard 30s
            # deadline while accelerator autodetection stalls on a wedged GPU,
            # or orphaned processes outside the cgroup.
            #
            # Two dead ends, recorded so they are not retried: the
            # RAY_raylet_start_wait_time_s the error message suggests is never
            # read by ray 2.55.1 (raylet_start_wait_time_s = 30 is hard-coded
            # at _private/node.py:411); and /dev/shm, memory and fd exhaustion
            # were all excluded from live node metrics.
            #
            # If startup fails again, READ THE SESSION LOGS -- raylet.err /
            # gcs_server.err under $RAY_TMPDIR carry the real error. The
            # client-side "current node timed out during startup" is generic
            # and says nothing about the cause.
            print(f"[ray] node={os.environ.get('SLURMD_NODENAME', '?')} "
                  f"job={os.environ.get('SLURM_JOB_ID', '?')} "
                  f"cpus={_kw.get('num_cpus', 'default')}", flush=True)
            # RETRY: raylet/GCS startup on these nodes intermittently exceeds
            # Ray's internal timeout ("The current node timed out during
            # startup") even with the node to ourselves, and the whole run
            # dies before episode 0. Three attempts with backoff turns a
            # multi-hour loss into a 40s delay.
            import time as _time
            _last = None
            for _try in range(3):
                try:
                    _ray.init(**_kw)
                    _last = None
                    break
                except Exception as _rexc:
                    _last = _rexc
                    print(f"[ray] init attempt {_try + 1}/3 failed: "
                          f"{type(_rexc).__name__}: {str(_rexc)[:120]}",
                          flush=True)
                    try:
                        _ray.shutdown()
                    except Exception:
                        pass
                    _time.sleep(20.0 * (_try + 1))
            if _last is not None:
                raise _last
        # One actor per MEASUREMENT device. Under --exec-on-gpu that is
        # num_gpus=1 each, so Ray hands every actor a disjoint
        # CUDA_VISIBLE_DEVICES and the timed execs cannot collide.
        # Ray must launch workers with the SAME interpreter as the driver.
        # Under `uv run` the raylet otherwise picks a python without ray
        # installed and every worker dies with ModuleNotFoundError. (The
        # `.ray_*venv` paths hardcoded in ppo/ray_vertex_ppo.py no longer
        # exist; sys.executable is correct and self-maintaining.)
        import sys as _sys
        _gpu = bool(getattr(args, "exec_on_gpu", False))

        def _actor_opts(idx: int) -> dict:
            """Pin actor ``idx`` to its OWN measurement GPU.

            Letting Ray allocate (num_gpus=1) handed the FIRST actor GPU 0 --
            the trainer's own device. Two processes then contend on it, which
            both hung the pool and broke the isolation the timing depends on.
            So: num_gpus=0 (Ray does not allocate) plus an explicit
            CUDA_VISIBLE_DEVICES, mirroring ray_vertex_ppo.py. Device 0 is
            reserved for the trainer; actors take 1..N in order, so no two
            timed executions can ever share a device.
            """
            rt = {"py_executable": _sys.executable}
            if _gpu:
                # num_gpus=0 makes Ray MASK the GPUs (it sets
                # CUDA_VISIBLE_DEVICES="" for workers that request none),
                # which overrode our pin and dropped the actor to CPU. The
                # NOSET flag tells Ray to leave CUDA_VISIBLE_DEVICES alone so
                # our explicit pin stands; it must also be exported in the
                # DRIVER environment so it reaches Ray's worker startup.
                rt["env_vars"] = {
                    "CUDA_VISIBLE_DEVICES": str(idx + 1),
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    # Dedicated measure process: no trainer shares this
                    # actor, so its single pinned GPU IS the measure device
                    # and env.py must not reserve one for a trainer.
                    # Measurement semantics must not depend on the TRAINER process's
                    # allocator: an inherited XLA_PYTHON_CLIENT_ALLOCATOR=platform puts
                    # raw cudaMalloc/cudaFree in the timed region (154us -> 379us, 2.46x
                    # flat, probe jobs 59598/59599) and breaks clear_memory_stats() so
                    # peak_memory silently becomes the STATIC estimate. Pin the default
                    # (BFC) allocator in every measure actor.
                    "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
                    "ALPHAGRAD_MEASURE_ACTOR": "1",
                }
            return {"runtime_env": rt, "num_gpus": 0}
        # The actor slices CPU cores by ``num_cpu_workers`` BEFORE importing
        # jax, so XLA sizes its Eigen pool to that slice. ppo.py never set it,
        # so every actor defaulted to n_workers=1 and claimed ALL cores --
        # "N actors each defaulting to all 64 cores oversubscribe
        # catastrophically (~48s/exec vs 0.45s)" (cpu_approx_actors.py).
        # Observed: only 1 of 3 actors finished __init__ before the pool
        # timed out. Copy rather than mutate the parsed args.
        _args_dict = dict(vars(args))
        _args_dict["num_cpu_workers"] = _n_actors
        _next_id = [0]

        def _spawn(slot: int | None = None):
            _next_id[0] += 1
            _slot = _next_id[0] - 1 if slot is None else int(slot)
            # Wrap around the available measurement devices so a RESPAWN
            # lands back on a real device instead of drifting past the last.
            _slot = _slot % max(_n_actors, 1)
            return CpuApproximationActor.options(**_actor_opts(_slot)).remote(
                _args_dict, variant=None, actor_id=_next_id[0],
            )

        _actors = [_spawn(i) for i in range(_n_actors)]
        _pool = CpuApproxPool(
            _actors,
            timeout_s=float(args.ray_measure_timeout),
            initial_timeout_s=float(args.ray_measure_timeout) * 4.0,
            warm_after=3,
            respawn_factory=_spawn,
            max_tokens=int(env.obs_width),
            num_rewards=int(NUM_REWARDS),
            cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
            frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        )
        object.__setattr__(env, "_remote_pool", _pool)
        object.__setattr__(env, "_remote_timeout_s",
                           float(args.ray_measure_timeout))
        print(f"[ray-measure] {_n_actors} actors on gpus "
              f"{[i + 1 for i in range(_n_actors)] if _gpu else 'cpu'} "
              f"(trainer keeps gpu 0), timeout={args.ray_measure_timeout}s",
              flush=True)

    # DIAG per-face masking. The dynamic policy's DIAG head must be masked by the
    # LIVE per-vertex pair / compress validity — the nominal tag-bit mask admits
    # per-face-invalid DIAGs and graphax then throws "TRANSFORM DID NOT FIT".
    # ppo.py's rollout is jitted + vmapped, so we bridge to the host-side
    # LiveVertexMaskOracle with a `pure_callback` that replays the elimination
    # prefix each step WITH THE ACTUALLY-APPLIED RULES (decoded from the env's
    # own sparsity_specs history via decode_vertex_rule_specs — the same
    # translation _callback uses for the measurement). advance()'s contract
    # demands the applied transforms; the old structural rules=() replay went
    # stale after the first landed approximation. Cost is O(steps) replay per
    # call; fine for the small vertex graphs.
    from alphagrad.approx.common.masks import LiveVertexMaskOracle as _LVMO
    from alphagrad.approx.env import decode_vertex_rule_specs as _decode_specs
    _oracle_jaxpr = closed_jaxpr.jaxpr
    _oracle_consts = list(closed_jaxpr.literals)
    _oracle_args = list(xs)
    _oracle_argnums = tuple(int(a) for a in argnums)
    _oracle_N = MAX_AXES_PER_VERTEX
    _oracle_total_v = len(_oracle_jaxpr.eqns)

    # Host-phase profiling: the oracle replays are prime slow-suspects (a
    # full LVMO elimination replay per CALL, per step). Accumulated into the
    # env module's shared sink; printed per episode under ALPHAGRAD_PROFILE=1.
    from alphagrad.approx.env import _prof_add as _env_prof_add
    from alphagrad.approx.env import consume_profile as _consume_profile
    import time as _prof_time

    def _oracle_masks_host(order, spec_hist, step_count):
        _pt0 = _prof_time.perf_counter()
        try:
            return _oracle_masks_host_inner(order, spec_hist, step_count)
        finally:
            _env_prof_add("oracle.vertex_masks", _prof_time.perf_counter() - _pt0)

    def _oracle_masks_host_inner(order, spec_hist, step_count):
        eo = np.asarray(order).reshape(-1)
        specs = np.asarray(spec_hist)
        n = int(np.asarray(step_count))
        o = _LVMO(_oracle_jaxpr, _oracle_consts, _oracle_args, _oracle_argnums,
                  max_axes=_oracle_N)
        for k in range(n):
            v = int(eo[k])  # env order is 1-based already
            try:
                rules = _decode_specs(
                    _oracle_jaxpr, v, specs[k]
                )
            except Exception:
                rules = ()
            try:
                o.advance(v, rules=rules)
            except Exception:
                break
        pair, comp = o.masks()
        return np.asarray(pair, np.float32), np.asarray(comp, np.float32)

    def _oracle_masks(order, spec_hist, step_count):
        """(pair (total_v+1, N, N), comp (total_v+1, N)) for the current graph."""
        return jax.pure_callback(
            _oracle_masks_host,
            (jax.ShapeDtypeStruct((_oracle_total_v + 1, _oracle_N, _oracle_N),
                                  jnp.float32),
             jax.ShapeDtypeStruct((_oracle_total_v + 1, _oracle_N), jnp.float32)),
            order, spec_hist, step_count, vmap_method="sequential",
        )

    # P1c: per-FACE masks for every candidate vertex (1-based rows like the
    # per-vertex masks). Only used under --face-actions; the extra probing
    # (face_masks per candidate) roughly doubles the oracle's host cost.
    if getattr(args, "face_actions", False):
        # Face width = the provable per-graph bound, derived BEFORE anything
        # builds a shape from it (env wire arrays, the 94-head's max_faces,
        # the trajectory zero-fills all read it downstream of here).
        from alphagrad.approx import env as _env_mod
        _B = _env_mod.derived_max_faces(
            closed_jaxpr.jaxpr, argnums, closed_jaxpr.literals, xs)
        _env_mod.configure_max_faces(_B)
        global ENV_MAX_FACES
        ENV_MAX_FACES = _env_mod.MAX_FACES
        print(f"face width: derived bound {_B} "
              f"(max_v |anc|x|desc|; in force: {ENV_MAX_FACES})")

    _F_FACES = ENV_MAX_FACES

    def _oracle_face_masks_host(order, spec_hist, step_count):
        _pt0 = _prof_time.perf_counter()
        try:
            return _oracle_face_masks_host_inner(order, spec_hist, step_count)
        finally:
            _env_prof_add("oracle.face_masks", _prof_time.perf_counter() - _pt0)

    def _oracle_face_masks_host_inner(order, spec_hist, step_count):
        eo = np.asarray(order).reshape(-1)
        specs = np.asarray(spec_hist)
        n = int(np.asarray(step_count))
        o = _LVMO(_oracle_jaxpr, _oracle_consts, _oracle_args, _oracle_argnums,
                  max_axes=_oracle_N)
        for k in range(n):
            v = int(eo[k])
            try:
                rules = _decode_specs(
                    _oracle_jaxpr, v, specs[k]
                )
            except Exception:
                rules = ()
            try:
                o.advance(v, rules=rules)
            except Exception:
                break
        pair, comp = o.masks()
        V, F, N = _oracle_total_v, _F_FACES, _oracle_N
        fpair = np.zeros((V + 1, F, N, N), np.float32)
        fcomp = np.zeros((V + 1, F, N), np.float32)
        fvalid = np.zeros((V + 1, F), np.float32)
        for v in range(1, V + 1):
            try:
                fp, fc, nf = o.face_masks(v, F)
            except Exception:
                continue
            fpair[v] = np.asarray(fp, np.float32)
            fcomp[v] = np.asarray(fc, np.float32)
            fvalid[v, : int(nf)] = 1.0
        return (np.asarray(pair, np.float32), np.asarray(comp, np.float32),
                fpair, fcomp, fvalid)

    def _oracle_face_masks(order, spec_hist, step_count):
        V, F, N = _oracle_total_v, _F_FACES, _oracle_N
        return jax.pure_callback(
            _oracle_face_masks_host,
            (jax.ShapeDtypeStruct((V + 1, N, N), jnp.float32),
             jax.ShapeDtypeStruct((V + 1, N), jnp.float32),
             jax.ShapeDtypeStruct((V + 1, F, N, N), jnp.float32),
             jax.ShapeDtypeStruct((V + 1, F, N), jnp.float32),
             jax.ShapeDtypeStruct((V + 1, F), jnp.float32)),
            order, spec_hist, step_count, vmap_method="sequential",
        )

    # ---- single-vertex oracle (ALPHAGRAD_ORACLE_ONE_VERTEX=1, default) ----
    # The all-vertex variants above probe every one of the ~13 vertices —
    # 4 tracing probes each — but the policy reads exactly ONE row
    # (`oracle_*_all[vertex_idx + 1]`). The vertex distribution provably does
    # NOT depend on the oracle (it is softmax(vertex_logits) masked by
    # vertex_avail_mask only), so the callback can run AFTER the vertex is
    # sampled and probe just that vertex: 4x(13-k) probes/step -> 4.
    # Values are bit-identical (same LVMO, same prefix replay, same
    # face_masks(chosen_v)); only the discarded rows disappear. The stored
    # masks the loss re-reads are unchanged, so ratio-1 is untouched.
    def _oracle_replay(eo, specs, n):
        o = _LVMO(_oracle_jaxpr, _oracle_consts, _oracle_args, _oracle_argnums,
                  max_axes=_oracle_N)
        for k in range(n):
            v = int(eo[k])
            try:
                rules = _decode_specs(
                    _oracle_jaxpr, v, specs[k]
                )
            except Exception:
                rules = ()
            try:
                o.advance(v, rules=rules)
            except Exception:
                break
        return o

    # Bounded memo for the prefix replay. Keyed on exactly the inputs the
    # result depends on: the eliminated prefix, its specs, and the probed
    # vertex. Bounded so a long run cannot grow it without limit; FIFO-evicted
    # because the useful entries are the recent prefixes.
    _ORACLE_MEMO: dict = {}
    _ORACLE_MEMO_MAX = int(os.environ.get("ALPHAGRAD_ORACLE_MEMO", "8192"))
    _ORACLE_STATS = [0, 0]        # [hits, misses]

    def _oracle_one_host(order, spec_hist, step_count, vertex_idx):
        _pt0 = _prof_time.perf_counter()
        try:
            eo = np.asarray(order).reshape(-1)
            specs = np.asarray(spec_hist)
            n = int(np.asarray(step_count))
            v = int(np.asarray(vertex_idx)) + 1   # policy 0-based -> oracle 1-based
            F, N = _F_FACES, _oracle_N
            if _ORACLE_MEMO_MAX > 0:
                _k = (eo[:n].tobytes(), specs[:n].tobytes(), v)
                _hit = _ORACLE_MEMO.get(_k)
                if _hit is not None:
                    _ORACLE_STATS[0] += 1
                    return _hit
                _ORACLE_STATS[1] += 1
            else:
                _k = None
            o = _oracle_replay(eo, specs, n)
            pair, comp = o.masks(candidates=[v])
            fp_arr = np.zeros((F, N, N), np.float32)
            fc_arr = np.zeros((F, N), np.float32)
            fv_arr = np.zeros((F,), np.float32)
            try:
                fp, fc, nf = o.face_masks(v, F)
                fp_arr[:] = np.asarray(fp, np.float32)
                fc_arr[:] = np.asarray(fc, np.float32)
                fv_arr[: int(nf)] = 1.0
            except Exception as _fmexc:
                # DO NOT SWALLOW (2026-08-05). fv_arr stays ALL-ZERO here,
                # which marks every face invalid and disables the whole
                # per-face action space: the head multiplies its skip draw by
                # face_valid (so skip becomes identically 0), and the
                # all-zero pair/compress masks leave only END legal (so every
                # slot draws "none"). Measured before this was visible:
                # skip=0.0000, none=99.2%, and not one rule ever applied.
                _FACE_MASK_FAILS[0] += 1
                _n = _FACE_MASK_FAILS[0]
                if _n <= 3 or _n % 500 == 0:
                    print("[oracle] face_masks FAILED for vertex "
                          f"{int(v)} (count={_n}): "
                          f"{type(_fmexc).__name__}: {str(_fmexc)[:200]} -- "
                          "ALL FACES MARKED INVALID, per-face approximation "
                          "disabled for this vertex", flush=True)
                    if _n == 1:
                        import traceback as _tb
                        _tb.print_exc()
            _res = (np.asarray(pair[v], np.float32),
                    np.asarray(comp[v], np.float32), fp_arr, fc_arr, fv_arr)
            if _k is not None:
                if len(_ORACLE_MEMO) >= _ORACLE_MEMO_MAX:
                    # FIFO evict a chunk rather than one-at-a-time, so the
                    # eviction cost is amortised.
                    for _dk in list(_ORACLE_MEMO)[: _ORACLE_MEMO_MAX // 8]:
                        _ORACLE_MEMO.pop(_dk, None)
                _ORACLE_MEMO[_k] = _res
            return _res
        finally:
            _env_prof_add("oracle.one_vertex",
                          _prof_time.perf_counter() - _pt0)

    def _oracle_one(order, spec_hist, step_count, vertex_idx):
        F, N = _F_FACES, _oracle_N
        return jax.pure_callback(
            _oracle_one_host,
            (jax.ShapeDtypeStruct((N, N), jnp.float32),
             jax.ShapeDtypeStruct((N,), jnp.float32),
             jax.ShapeDtypeStruct((F, N, N), jnp.float32),
             jax.ShapeDtypeStruct((F, N), jnp.float32),
             jax.ShapeDtypeStruct((F,), jnp.float32)),
            order, spec_hist, step_count, vertex_idx,
            vmap_method="sequential",
        )

    # ---- FEATURE PROBE targets (ALPHAGRAD_FEATURE_PROBE=1) ------------
    # Modelled on `_oracle_one`: the SAME `_LVMO` prefix replay, the SAME
    # deferred-until-the-vertex-is-known shape, one extra `probe_faces` on top
    # of the `face_masks` the oracle already runs (see face_targets_host).
    # DEFAULT OFF -- the whole block is dead, no callback and no field, when
    # `ALPHAGRAD_FEATURE_PROBE` is unset.
    _PROBE_FACES = min(
        int(os.environ.get("ALPHAGRAD_FEATURE_PROBE_FACES", "32")), _F_FACES)
    # The probe measures the LIVE face pipeline, so it needs ALL of it.
    # --no-approx-head DELETES the face head (the agent factory sets
    # face_path_policy = None), which leaves face_out = None in the rollout:
    # zero endpoints, zero latents, and a probe that decodes zeros. Job 61427
    # (NN256, GPU) ran exactly that flag combination for an hour -- every
    # control target was 0 and every R2 column read 0.000. Refuse loudly.
    _PROBE_ON = bool(_fprobe.PROBE_ON) and all((
        bool(getattr(args, "dynamic_substeps", False)),
        bool(getattr(args, "live_faces", False)),
        bool(getattr(args, "face_actions", False)),
        bool(getattr(args, "unified_face_head", False)),
        not getattr(args, "no_approx_head", False),
    ))
    if _fprobe.PROBE_ON and not _PROBE_ON:
        print("[probe] ALPHAGRAD_FEATURE_PROBE=1 IGNORED: the probe reads "
              "the per-face scatter, which exists only under "
              "--dynamic-substeps --live-faces --face-actions "
              "--unified-face-head and WITHOUT --no-approx-head "
              "(--no-approx-head removes the face head entirely, so the "
              "probe would decode zeros -- job 61427).", flush=True)
    # [ppo-level target-build failures, first-callback census done,
    #  first nonzero-face census done]
    _PROBE_FAILS = [0, 0, 0]
    # Per-episode n_faces histogram over probe callbacks; printed and reset
    # by the per-episode probe census line in host_log.
    _PROBE_NF_HIST = np.zeros(_PROBE_FACES + 1, np.int64)
    if _PROBE_ON:
        # THE STATIC CONTROLS' pre-image: log2 numel of an endpoint vertex's
        # output var, keyed by the 1-based vertex id `face_endpoints` stores
        # (0 = a jaxpr input, which has no equation and so scores 0). This is
        # the same quantity decode2_face_data.py's `ln_of_vidx` holds, keyed by
        # vertex instead of by var index.
        _PROBE_LNV = np.zeros((_oracle_total_v + 2,), np.float32)
        for _pi, _peq in enumerate(_oracle_jaxpr.eqns, start=1):
            _pn = 1
            if _peq.outvars and hasattr(_peq.outvars[0], "aval"):
                for _ps in _peq.outvars[0].aval.shape:
                    _pn *= int(_ps)
            _PROBE_LNV[_pi] = float(np.log2(max(_pn, 1)))
        print("[probe] feature probe ON: arm=%s width=%d faces=%d/%d "
              "(targets %s)"
              % (_fprobe.PROBE_ARM, _fprobe.PROBE_WIDTH, _PROBE_FACES,
                 _F_FACES, ",".join(_fprobe.FACE_NAMES)), flush=True)

        def _probe_targets_host(order, spec_hist, step_count, vertex_idx,
                                ends):
            _pt0 = _prof_time.perf_counter()
            try:
                eo = np.asarray(order).reshape(-1)
                specs = np.asarray(spec_hist)
                n = int(np.asarray(step_count))
                v = int(np.asarray(vertex_idx)) + 1
                P, N = _PROBE_FACES, _oracle_N
                tgt = np.zeros((P, _fprobe.NFT), np.float32)
                ext = np.zeros((P, N), np.float32)
                val = np.zeros((P,), np.float32)
                try:
                    o = _oracle_replay(eo, specs, n)
                    _t, _e, _nf = _fprobe.face_targets_host(
                        o, _oracle_jaxpr, v, P, N,
                        ln_of_vidx=_PROBE_LNV,
                        endpoints=np.asarray(ends, np.int32))
                except Exception as _pexc:
                    _PROBE_FAILS[0] += 1
                    _PROBE_NF_HIST[0] += 1
                    if _PROBE_FAILS[0] <= 3:
                        print("[probe census] target build FAILED for vertex "
                              f"{v}: {type(_pexc).__name__}: "
                              f"{str(_pexc)[:200]}", flush=True)
                    return tgt, ext, val
                _nf = min(int(_nf), P)
                tgt[:_nf] = _t[:_nf]
                ext[:_nf] = _e[:_nf]
                val[:_nf] = 1.0
                _PROBE_NF_HIST[_nf] += 1
                # TARGET CENSUS. A within-step R2 of 0 has two very different
                # causes -- "the representation cannot decode it" and "the
                # target is constant, so there is nothing to decode" -- and
                # within_step_r2 reports 0 for BOTH by design. Printed
                # UNCONDITIONALLY on the first callback (and again on the
                # first vertex that yields faces, if the first had none), so
                # the two are distinguishable from any log. The marker is
                # "probe census" -- job 61427's launcher grepped for exactly
                # that and the old "[probe] first targets" line vanished.
                if _PROBE_FAILS[1] == 0 or (_PROBE_FAILS[2] == 0
                                            and _nf > 0):
                    _PROBE_FAILS[1] = 1
                    if _nf > 0:
                        _PROBE_FAILS[2] = 1
                    print("[probe census] first targets: vertex=%d "
                          "n_faces=%d endpoints=%s"
                          % (v, _nf, np.asarray(ends)[:_nf].tolist()),
                          flush=True)
                    for _c, _nm in enumerate(_fprobe.FACE_NAMES):
                        _col = tgt[:max(_nf, 1), _c]
                        print("[probe census]   %-10s min=%.3f max=%.3f "
                              "std=%.3f"
                              % (_nm, float(_col.min()), float(_col.max()),
                                 float(_col.std())), flush=True)
                return tgt, ext, val
            finally:
                _env_prof_add("oracle.feature_probe",
                              _prof_time.perf_counter() - _pt0)

        def _probe_targets(order, spec_hist, step_count, vertex_idx, ends):
            P, N = _PROBE_FACES, _oracle_N
            return jax.pure_callback(
                _probe_targets_host,
                (jax.ShapeDtypeStruct((P, _fprobe.NFT), jnp.float32),
                 jax.ShapeDtypeStruct((P, N), jnp.float32),
                 jax.ShapeDtypeStruct((P,), jnp.float32)),
                order, spec_hist, step_count, vertex_idx, ends,
                vmap_method="sequential",
            )
    else:
        _probe_targets = None

    # ---- per-FACE token chunks (--live-faces) -------------------------
    # One callback per face. It replays the elimination prefix (cached) and
    # re-eliminates the CURRENT vertex with faces 0..f-1 carrying their
    # decided approximations, returning the tokens emitted between the
    # previous decision and this one. graphax has no resumable elimination,
    # so re-running is the only way to reach face f's contraction with face
    # f-1's approximation in place; the prefix is replayed once per distinct
    # prefix, so the cost is n_faces eliminations per env step.
    _LIVE_FACES = None
    _live_face = _live_face_count = None
    if getattr(args, "live_faces", False):
        _LIVE_FACES = build_live_face_stream(
            _oracle_jaxpr, _oracle_argnums, _oracle_consts, _oracle_args,
            vocab=int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512")),
            max_faces=_F_FACES, max_axes=_oracle_N,
            # A chunk is a slice of the step delta, so the delta cap is the
            # one honest window: truncation becomes impossible whenever the
            # delta itself fits, and the stored counts stay exact for the
            # loss's cumsum boundaries.
            window=MAX_DELTA_TOKENS,
            # The prefix key now carries the prefix's FACE wires, so two envs
            # on the same elimination order no longer share a tokenizer. The
            # live working set is therefore one prefix PER ENV (all faces of
            # one vertex share it; nothing across envs or steps does). Sized
            # 4x num_envs so the FIFO never evicts an entry the next face
            # substep of the same batch still needs -- a capacity below the
            # env count would turn every substep into a cold replay.
            # (AZ, with no envs, passes its own capacity;
            # ALPHAGRAD_FACE_PREFIX_CACHE still overrides.)
            cache=max(64, 4 * _resolve_num_envs(
                args.num_envs, args.example)),
        )
        # The `pure_callback` wrappers themselves now live in
        # common/face_driver.py so AZ drives the SAME stream, not a copy.
        _live_face, _live_face_count = make_face_callbacks(
            _LIVE_FACES, window=MAX_DELTA_TOKENS, prof_sink=_env_prof_add)

    # Live elimination chains: one per concurrent env, plus the previous
    # episode's, which the LRU only sheds once the new ones exist. Sized like
    # the face-prefix cache above and for the same reason -- a capacity below
    # the concurrent chain count turns every step into a cold O(T) rebuild,
    # i.e. straight back to O(T^2) per episode. That is not silent (the
    # `restart` counter reports it in the [prof] line), but it should not be
    # possible by default. ALPHAGRAD_FACE_LIVE_CHAINS still overrides.
    if not os.environ.get("ALPHAGRAD_FACE_LIVE_CHAINS"):
        from alphagrad.approx import env as _env_chain_mod
        _env_chain_mod._LIVE_CHAIN_CAP = max(
            8, 4 * _resolve_num_envs(args.num_envs, args.example))

    _ORACLE_ONE_VERTEX = os.environ.get(
        "ALPHAGRAD_ORACLE_ONE_VERTEX", "1") == "1"
    # No approximation head => no legality to compute. See _NO_ORACLE use.
    # --live-faces: sampling masks are STATIC (axis validity); per-face
    # legality is enforced once, at application, by make_live_masked_hook --
    # so the live probe (the single largest host cost) leaves the rollout.
    _NO_ORACLE = bool(getattr(args, "no_approx_head", False)
                      or getattr(args, "live_faces", False))

    if getattr(args, "live_faces", False):
        # Both are load-bearing, not stylistic. Without --face-actions there
        # is no per-face decision to condition. Without --incremental-encode
        # the loss has no step carry to branch the per-face side carry from,
        # so it could only re-score against a context the rollout never used
        # and the PPO ratio would silently stop being 1 at epoch 0.
        if not args.face_actions:
            raise ValueError("--live-faces requires --face-actions.")
        if not args.incremental_encode:
            raise ValueError(
                "--live-faces requires --incremental-encode (the loss "
                "re-runs the per-face recurrence from the stored step carry)."
            )

    total_v = len(closed_jaxpr.jaxpr.eqns)
    num_valid = len(env.valid_vertices)
    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

    vertex_valid_static = build_vertex_valid_static(env.valid_vertices, total_v)
    pair_valid_mask = build_pair_valid_mask(
        closed_jaxpr.jaxpr,
        total_v,
        num_pair_choices=NUM_PAIR_CHOICES,
        pair_stop_idx=PAIR_STOP,
        disable_sparsification=args.disable_sparsification,
    )

    # Hyperparameters / agent.
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args)
    factor_table_np = np.array(factors_py, dtype=np.int32)

    # Resolve --pin-factor to an index in the factor_table. Mutually exclusive
    # with --pin-rules-to-exact (which pins both axis and factor).
    pin_factor_idx: int | None = None
    if args.pin_factor is not None:
        if args.pin_rules_to_exact:
            raise ValueError(
                "--pin-factor and --pin-rules-to-exact are mutually exclusive."
            )
        matches = np.where(factor_table_np == args.pin_factor)[0]
        if matches.size == 0:
            raise ValueError(
                f"--pin-factor={args.pin_factor} not in --factors {factors_py}; "
                f"available indices: {list(zip(range(num_factors), factors_py))}"
            )
        pin_factor_idx = int(matches[0])
        print(
            f"Stage D: pinning factor to {args.pin_factor} (index {pin_factor_idx} "
            f"in factor_table). Axis head trains; factor head deterministic."
        )
    num_envs = _resolve_num_envs(args.num_envs, args.example)
    # 8-component reward vector is still emitted by the env and used for
    # host-side display (top-N heaps, per-component means). Training-side
    # value / advantage path operates on the 3-vec (latency / peak_memory /
    # cosine_sim); see HEAD_REWARD_INDICES and
    # `_build_head_weights`.
    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
    head_reward_weights_np = _build_head_weights(args)
    head_reward_weights = jnp.asarray(head_reward_weights_np, dtype=jnp.float32)
    cmp_idx = _cmp_reward_index(args.cmp_type)
    mem_idx = _mem_reward_index(args.mem_type)
    # Reward slot 6 -- THE quality channel. ``REWARD_INDEX["cosine_sim"]`` is
    # the back-compat alias for ``REWARD_INDEX["quality"]``; the local name
    # stays ``cosine_idx`` only because ~30 references in this file use it.
    cosine_idx = REWARD_INDEX["quality"]
    # ``--reward-mode mult``: cost weights for the cheapness term = the display
    # weights with the quality channels zeroed (the gate multiplies fidelity
    # back in); the preference collapses to one-hot on the cosine head so the
    # scalarization recovers the gated scalar exactly.
    mult_cost_weights_np = reward_weights_np.copy()
    mult_cost_weights_np[cosine_idx] = 0.0
    mult_cost_weights = jnp.asarray(mult_cost_weights_np, dtype=jnp.float32)
    if args.reward_mode == "mult":
        head_reward_weights_np = np.zeros(NUM_VALUE_HEADS, dtype=np.float32)
        head_reward_weights_np[HEAD_NAMES.index("quality")] = 1.0
        head_reward_weights = jnp.asarray(head_reward_weights_np, dtype=jnp.float32)

    # Per-(vertex, pair, factor) validity mask. The legacy mask filtered
    # out factors that didn't divide the relevant axis sizes. With
    # graphax's typed-transform API (apply_diag) silently skipping
    # non-dividing factors at apply time, the pre-mask is redundant —
    # the policy can emit any factor index and the env-side translator
    # drops the invalid slot. Pass an all-ones tensor in the shape the
    # autoreg factor head expects.
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors),
        dtype=jnp.float32,
    )

    # Dynamic-substeps state: only constructed when the flag is on, but
    # always referenced by the rollout closure so it must be defined.
    # FactorTables is an env-static FactorTables NamedTuple (pytree of
    # int32 / float32 lookup arrays); op_legality_override is a (NUM_OPS,)=(4,)
    # float32 mask gating COMPRESS via --allow-compress.
    if args.dynamic_substeps:
        factor_tables = precompute_factor_tables(args.max_axis_size)
        # The variant's op-type legality must reach the dynamic policy via
        # ``op_legality_override``. The legacy ``pin_rules_to_exact`` only
        # gates the legacy rule head, so without this branch
        # ``--variant ve_only`` (or diag_gcd / diag_factor / compress)
        # silently behaved like ``custom`` in dynamic-substeps mode — the
        # MicroActionPolicy was free to emit DIAG / COMPRESS micro-actions
        # that the env then applied, breaking the variant-comparison
        # intent.
        op_legality_override = _op_legality_for_variant(
            args.variant, args.allow_compress
        )
        print(
            f"dynamic-substeps: max_substeps={args.max_substeps}, "
            f"max_axis_size={args.max_axis_size}, "
            f"allow_compress={args.allow_compress}, "
            f"op_legality={op_legality_override.tolist()}"
        )
    else:
        # Placeholder values so the rollout closure can reference these
        # names unconditionally. With dynamic_substeps off the rollout
        # branch never touches them.
        factor_tables = FactorTables(
            gcd=jnp.zeros((1, 1), dtype=jnp.int32),
            primes=jnp.zeros((1, MAX_PRIMES), dtype=jnp.int32),
            max_exps=jnp.zeros((1, MAX_PRIMES), dtype=jnp.int32),
            prime_mask=jnp.zeros((1, MAX_PRIMES), dtype=jnp.float32),
        )
        op_legality_override = jnp.ones((NUM_OPS,), dtype=jnp.float32)

    # In dynamic-substeps mode the static `--factors` table is unused (the
    # prime-exponent head emits factors on the fly), so don't pollute the
    # banner with it.
    if args.dynamic_substeps:
        print(
            f"variant={variant_label}, num_envs={num_envs}, max_rules={max_rules}, "
            f"rollout_length={num_valid}, minibatches={args.minibatches}"
        )
    else:
        print(
            f"variant={variant_label}, num_envs={num_envs}, max_rules={max_rules}, "
            f"factors={factors_py}, rollout_length={num_valid}, "
            f"minibatches={args.minibatches}"
        )
    # Reward-vector column order: top-N / progress-bar dumps print eight
    # floats per episode without per-column labels, so name the order
    # once here.
    # PRIOR SCAN: probe every dtype and every contraction pair on THIS
    # hardware, in BOTH operand orders, before a single episode runs — so
    # the catalog is visible up front instead of being discovered by a
    # TypePromotionError 200 episodes in.
    try:
        from graphax.sparse.micro_actions import report_hardware_scan
        report_hardware_scan()
    except Exception as _e:
        print(f"[quant-scan] unavailable: {_e!r}")
    print(f"reward order: {', '.join(REWARD_NAMES)}")
    # Catch the "20-samples-into-32-minibatches → 0-elem minibatch → silent NaN
    # loss" pitfall as early as possible. `shuffle_and_batch` does integer
    # division (num_envs * rollout_length // minibatches); when the result is
    # zero, the PPO loss is `jnp.mean(<empty>) = NaN`, training is a no-op,
    # and the only surface signal is `ent:nan` in the progress bar.
    _mb_size = (num_envs * num_valid) // args.minibatches
    if _mb_size == 0:
        raise ValueError(
            f"--minibatches={args.minibatches} > num_envs * rollout "
            f"({num_envs} * {num_valid} = {num_envs * num_valid}). "
            "Each minibatch would be empty, so the PPO loss becomes NaN "
            "and no learning happens. Lower --minibatches or raise "
            "--num-envs."
        )
    # FULL-HORIZON SCAN (--grad-window 0) minibatches over SEQUENCES, not
    # steps: a scan needs a trajectory's steps in order, so the unit of a
    # minibatch is a whole env. `shuffle_and_batch_by_trajectory` floor-divides
    # num_envs by minibatches and returns EMPTY minibatches (NaN loss, silent
    # no-op training) when that is 0, which is the same trap the check above
    # catches for the per-step path -- so clamp loudly instead.
    if int(getattr(args, "grad_window", 1)) == 0 and args.minibatches > num_envs:
        print(
            f"[grad-window 0] --minibatches={args.minibatches} > "
            f"--num-envs={num_envs}: the full-horizon path batches whole "
            f"TRAJECTORIES, so minibatches is clamped to {num_envs}. "
            "Updates per epoch are the minibatch count, i.e. this run does "
            f"{num_envs} instead of {args.minibatches}."
        )
        args.minibatches = int(num_envs)
    nonzero_w = ", ".join(
        f"{REWARD_NAMES[i]}={float(reward_weights_np[i]):+.3g}"
        for i in range(NUM_REWARDS)
        if reward_weights_np[i] != 0.0
    )
    print(f"reward weights (display): {nonzero_w or '<all zero — debug only>'}")
    head_w_str = ", ".join(
        f"{HEAD_NAMES[i]}={float(head_reward_weights_np[i]):+.3g}"
        for i in range(NUM_VALUE_HEADS)
    )
    print(f"head weights (training): {head_w_str}")

    agent_key, init_key, key = jrand.split(key, 3)
    agent = _build_agent(args, total_v, num_factors, max_rules, agent_key)
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale)
    # Identity-init parity with the factory path (az): ppo.main predates
    # build_and_init_agent and does not route through it.
    from alphagrad.approx.common.agent_factory import apply_face_none_bias
    agent = apply_face_none_bias(agent)
    # Stage D: build per-head boolean parameter masks once. Used inside
    # train_minibatch to scale gradients by the head-specific LR multiplier
    # (warm-up ramp from §3.2). Masks are pytree leaves aligned with the
    # filtered (inexact-array) agent params.
    axis_mask, factor_mask, vertex_mask, micro_mask = _build_head_masks(agent)

    # Stage G default freeze mask: all-True (no freezing) — the same
    # train_episode path serves both regular training and calibration. The
    # all-True mask makes the gradient unchanged so there's no JIT recompile
    # when we swap to the cal_mask in the calibration phase.
    default_freeze_mask = jax.tree_util.tree_map(
        lambda x: jnp.ones_like(x, dtype=jnp.bool_),
        eqx.filter(agent, eqx.is_inexact_array),
    )

    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent,
        )

    # Optimiser: single cosine decay across the whole run.
    #
    # CLAMPED TO >= 1. The product is zero whenever any factor is zero, and
    # optax then raises "cosine_decay_schedule requires positive decay_steps"
    # from deep inside the optimiser build -- an opaque failure a long way
    # from the flag that caused it. --ppo-epochs 0 is a LEGITIMATE mode: it
    # runs the rollout and skips the update entirely, which is how the
    # rollout's memory footprint is isolated from the loss forward and
    # backward (they share one XLA program and no flag separates them).
    # A run that takes zero optimiser steps never evaluates the schedule, so
    # the clamped value is unobservable -- it exists only to keep the
    # construction total.
    _decay_steps = max(
        1, int(args.episodes) * int(args.ppo_epochs) * int(args.minibatches))
    schedule = optax.cosine_decay_schedule(
        args.lr,
        _decay_steps,
        args.lr_decay_min_mult,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(schedule, b1=args.adam_b1, eps=args.adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    # FEATURE PROBES: A SEPARATE PARAMETER TREE WITH A SEPARATE OPTIMISER.
    # `opt_state` above is initialised from `agent` ALONE, so the PPO chain
    # provably cannot touch a probe weight, and `_probe_loss` below is a
    # function of `probes` alone, so the probe gradient provably cannot touch
    # palimpsa. Neither guarantee rests on a coefficient being small.
    if _PROBE_ON:
        probes = _fprobe.FeatureProbes(
            embd_dim=int(args.embd_dim),
            max_axes=int(MAX_AXES_PER_VERTEX),
            # The vertex probe decodes the same seven columns, aggregated over
            # the vertex's faces -- see `_probe_loss`.
            n_vertex_out=_fprobe.NFT,
            width=_fprobe.PROBE_WIDTH,
            key=jrand.PRNGKey(
                int(os.environ.get("ALPHAGRAD_FEATURE_PROBE_SEED", "0"))),
            arm=_fprobe.PROBE_ARM,
        )
        probe_optimizer = optax.adam(
            float(os.environ.get("ALPHAGRAD_FEATURE_PROBE_LR", "1e-3")))
        probe_opt_state = probe_optimizer.init(
            eqx.filter(probes, eqx.is_inexact_array))
    else:
        probes = probe_opt_state = probe_optimizer = None

    # Rollout / loss / training step factories.
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    @eqx.filter_jit
    # +1 entry for vertex_temperature (broadcast, not mapped): the PopArt
    # warm-start flattens the vertex pointer to ~uniform over legal vertices.
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, 0, None, None, None))
    def rollout_fn(
        agent,
        env_obj,
        rollout_length,
        env_state,
        key,
        base_mem,
        preference,
        op_legality_override,
        pin_rules_to_exact_jax,
        vertex_temperature=None,
    ):
        keys = jrand.split(key, rollout_length)

        encode_key, scan_key = jrand.split(keys[0], 2)

        # Shapes for the unused trajectory branch. The Trajectory NamedTuple
        # carries both legacy and dynamic action fields so the rollout's
        # output structure is identical across the two modes (lax.scan
        # outputs need uniform leaf shapes).
        _legacy_zero_pair_seq = jnp.zeros((args.max_rules,), dtype=jnp.int32)
        _legacy_zero_factor_seq = jnp.zeros((args.max_rules,), dtype=jnp.int32)
        _legacy_zero_pair_dists = jnp.zeros(
            (args.max_rules, NUM_PAIR_CHOICES),
            dtype=jnp.float32,
        )
        _legacy_zero_factor_dists = jnp.zeros(
            (args.max_rules, num_factors),
            dtype=jnp.float32,
        )
        _dyn_zero_op_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_i_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_j_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_exp_seq = jnp.zeros(
            (args.max_substeps, MAX_PRIMES),
            dtype=jnp.int32,
        )
        _dyn_zero_factor_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_op_dists = jnp.zeros(
            (args.max_substeps, NUM_OPS),
            dtype=jnp.float32,
        )
        _dyn_zero_i_dists = jnp.zeros(
            (args.max_substeps, MAX_AXES_PER_VERTEX),
            dtype=jnp.float32,
        )
        _dyn_zero_j_dists = jnp.zeros(
            (args.max_substeps, MAX_AXES_PER_VERTEX),
            dtype=jnp.float32,
        )
        _dyn_zero_exp_dists = jnp.zeros(
            (args.max_substeps, MAX_PRIMES, MAX_EXPONENT + 1),
            dtype=jnp.float32,
        )
        _dyn_zero_kind_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_kind_dists = jnp.zeros(
            (args.max_substeps, NUM_COMPRESS_KINDS),
            dtype=jnp.float32,
        )
        _dyn_zero_quant_seq = jnp.zeros((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_quant_sign = jnp.ones((args.max_substeps,), dtype=jnp.int32)
        _dyn_zero_quant_logp = jnp.zeros((args.max_substeps,), dtype=jnp.float32)

        # Consume the base stream ONCE (a host-side constant of exactly
        # _BASE_W tokens -- no window, no cursor, no padded scan) to seed the
        # palimpsa carry. Its per-vertex SCATTER is `base_mem`, passed in
        # from `train_episode` and NOT accumulated into the dynamic memory:
        # the dynamic half is what the trajectory stores, and the base half
        # is re-derived inside the loss so palimpsa's base encode is
        # differentiated. Keeping them added together here would put an
        # encoder output into stored state, which is the defect this design
        # exists to remove.
        # `_BASE_OWN` (the per-token owning vertex) is resolved once with the
        # base stream itself; without it every base row lands in the global
        # slot and no vertex has any content of its own.
        _enc_base = _carry_stream.init_carry(
            agent, _BASE_TOK, _BASE_EQN, _BASE_N,
            window=_BASE_W, total_v=total_v, embd_dim=args.embd_dim,
            base_owners=_BASE_OWN,
        )[0]
        _init_pre = (_enc_base,) + _carry_stream.zero_memory(
            total_v, args.embd_dim)
        # The pre-scan delta was emitted before any elimination this rollout
        # made, so it touches nothing: an all-zero participation mask, which
        # `advance` sends to the global slot -- authorship's destination for
        # an owner of -1.
        _init_part = jnp.zeros((total_v + 1,), jnp.float32)
        # The scan carries the encoder state at TWO points, PRE and POST this
        # step's delta, because they are the same two things every iteration
        # already computed -- twice. See `step_fn`'s bootstrap block. POST is
        # seeded here with what iteration 0's extend used to do: step 0's
        # delta is whatever `env_state` carries, with owner -1 (no previous
        # elimination emitted it).
        _init_post = _carry_stream.advance(
            agent, *_init_pre,
            env_state.delta_tokens, env_state.delta_eqns,
            env_state.delta_count,
            # `delta_owner` as step_fn computes it on iteration 0, spelled
            # out: `elim_order` is all zeros at scan entry, so its lookup
            # arm is 0 and the step_count == 0 arm is -1.
            jnp.where(env_state.step_count > 0,
                      jnp.zeros((), jnp.int32),
                      jnp.array(-1, jnp.int32)).astype(jnp.int32),
            window=MAX_DELTA_TOKENS,
            participants=_init_part,
        )
        init_enc_state = _init_pre + _init_post

        def step_fn(carry, k):
            state, elim_order, enc_state, prev_part = carry
            sample_key, next_net_key = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid
            )

            # PRE (synced through the PREVIOUS delta) and POST (synced through
            # THIS step's delta). POST is not recomputed here: the previous
            # iteration's value-bootstrap already extended the carry by
            # exactly this delta, under exactly this owner, and threading it
            # forward is what makes that extend cost once instead of twice.
            # See the bootstrap block below for the proof of equality.
            enc_carry, vmem_s, vmem_c, enc_carry2, vmem_s2, vmem_c2 = enc_state
            # THIS step's delta was emitted by the PREVIOUS step's elimination
            # (empty at step 0 — the base stream is already consumed). It
            # arrives from the env as its own buffer with its own exact
            # count; nothing is sliced out of a growing stream.
            delta_owner = jnp.where(
                state.step_count > 0,
                elim_order[jnp.maximum(state.step_count - 1, 0)],
                jnp.array(-1, jnp.int32),
            ).astype(jnp.int32)
            delta_tok = state.delta_tokens
            delta_eqn = state.delta_eqns
            delta_count = state.delta_count
            # prof/envcb: scan glue (avail mask, key split, the delta unpack
            # above). Under the partial mark anchor this key used to absorb
            # the WHOLE env callback -- `prof/envstep` fired on EnvState's
            # device-only leading leaves while the callback was still in
            # flight, and the first mark that had to wait for it was this one
            # (its anchor, `state.delta_tokens`, IS the callback's output).
            # With the full anchor the callback is back inside `prof/envstep`
            # and this reads ~0; `prof/env_cb_host` is its host span.
            delta_tok = _pp_mark("prof/envcb", delta_tok)
            # Kept as a mark so the key still reports: `prof/encode` is now
            # the cost of NOT extending here, i.e. ~0. The extend it used to
            # time is `prof/encode_bootstrap`, run once.
            enc_carry2, vmem_s2, vmem_c2 = _pp_mark(
                "prof/encode", (enc_carry2, vmem_s2, vmem_c2))
            precomputed = _carry_stream.heads(
                agent, vmem_s2, vmem_c2,
                base_mem=base_mem,
                preference=(
                    preference if args.preference_conditioned else None
                ),
            )
            precomputed = _pp_mark("prof/heads", precomputed)

            face_chunk_fn = None
            face_count_fn = None
            if _LIVE_FACES is not None:
                # The FULL per-face history rides along: `_rows`/`_skips` are
                # the CURRENT vertex's in-flight decisions (the face loop's
                # carry), while `_fh`/`_kh` are every decision already
                # committed to the prefix. The prefix replay needs the latter
                # or it rebuilds an exact graph the measurement never builds.
                face_chunk_fn, face_count_fn = bind_step_callbacks(
                    _live_face, _live_face_count,
                    state.order, state.sparsity_specs, state.step_count,
                    state.face_specs, state.face_skips,
                )

            if args.dynamic_substeps:
                # Live per-vertex DIAG/COMPRESS masks for the current graph
                # (replayed from the elimination prefix so far). Under
                # --face-actions the same host replay also returns the
                # PER-FACE masks for every candidate vertex.
                oracle_one_fn = None
                if _NO_ORACLE:
                    # --no-approx-head: nothing to mask, so every probe is
                    # pure waste (48% of host time when it does run).
                    oracle_pair_all = oracle_comp_all = None
                    face_masks_all = None
                elif _ORACLE_ONE_VERTEX:
                    # Perf path: defer the probe until the vertex is known,
                    # then probe ONLY that vertex (see _oracle_one).
                    _st_o, _st_s, _st_k = (
                        state.order, state.sparsity_specs, state.step_count)

                    def oracle_one_fn(_v, _o=_st_o, _s=_st_s, _k=_st_k):
                        return _oracle_one(_o, _s, _k, _v)

                    oracle_pair_all = oracle_comp_all = None
                    face_masks_all = None
                elif args.face_actions:
                    (oracle_pair_all, oracle_comp_all, _fp_all, _fc_all,
                     _fv_all) = _oracle_face_masks(
                        state.order, state.sparsity_specs, state.step_count)
                    face_masks_all = (_fp_all, _fc_all, _fv_all)
                else:
                    oracle_pair_all, oracle_comp_all = _oracle_masks(
                        state.order, state.sparsity_specs, state.step_count)
                    face_masks_all = None
                (
                    vertex_idx,
                    micro_actions,
                    vertex_dist,
                    micro_op_dists,
                    micro_i_dists,
                    micro_j_dists,
                    micro_exp_dists,
                    micro_kind_dists,
                    micro_quant_logp,
                    micro_pair_valid,
                    micro_compress_valid,
                    face_out,
                    value,
                    v_context,
                ) = agent.sample_action_dynamic(
                    # No token argument: `precomputed` below IS the encoding,
                    # derived from the carry plus this step's delta.
                    None,
                    vertex_avail_mask,
                    state.axis_state,
                    state.axis_valid_mask,
                    factor_tables,
                    op_legality_override,
                    sample_key,
                    eqn_ids=None,
                    preference=preference if args.preference_conditioned else None,
                    oracle_pair_all=oracle_pair_all,
                    oracle_comp_all=oracle_comp_all,
                    face_masks_all=face_masks_all,
                    vertex_temperature=vertex_temperature,
                    precomputed=precomputed,
                    oracle_fn=oracle_one_fn,
                    face_chunk_fn=face_chunk_fn,
                    face_count_fn=face_count_fn,
                    enc_carry=enc_carry2,
                )
                # prof/action: the vertex pointer sample + the micro/face
                # head loop (INCLUDES the faces.live_chunk host callbacks,
                # which env.py counts separately -- the two overlap by
                # construction).
                vertex_idx, value, v_context = _pp_mark(
                    "prof/action", (vertex_idx, value, v_context))
                # Record this vertex in the elimination prefix for the next
                # step's oracle replay.
                elim_order = elim_order.at[state.step_count].set(
                    vertex_idx.astype(elim_order.dtype))
                if face_out is not None:
                    (face_action, face_old_logp, _face_ent, face_pair_v,
                     face_comp_v, face_valid_v, face_cnt_v, face_dt_v,
                     face_de_v, face_ends_v) = face_out
                else:
                    face_action = _zero_face_action()
                    face_old_logp = jnp.array(0.0)
                    face_pair_v = jnp.zeros(
                        (ENV_MAX_FACES, MAX_AXES_PER_VERTEX,
                         MAX_AXES_PER_VERTEX), jnp.float32)
                    face_comp_v = jnp.zeros(
                        (ENV_MAX_FACES, MAX_AXES_PER_VERTEX), jnp.float32)
                    face_valid_v = jnp.zeros((ENV_MAX_FACES,), jnp.float32)
                    face_ends_v = jnp.zeros((ENV_MAX_FACES, 2), jnp.int32)
                    face_cnt_v = jnp.zeros((ENV_MAX_FACES,), jnp.int32)
                    face_dt_v = jnp.zeros((MAX_DELTA_TOKENS,), jnp.int32)
                    face_de_v = -jnp.ones((MAX_DELTA_TOKENS,), jnp.int32)
                if _DEBUG_ORDER:
                    # avail = how many vertices are still selectable; picked =
                    # the 0-based index chosen; was_avail = 1.0 iff that pick
                    # was legal. was_avail == 0 is the duplicate-pick bug.
                    jax.debug.print(
                        "[order] step={s} n_avail={a} pick={v} was_avail={w} "
                        "order={o}",
                        s=state.step_count,
                        a=jnp.sum(vertex_avail_mask),
                        v=vertex_idx,
                        w=vertex_avail_mask[vertex_idx],
                        o=state.order,
                    )
                env_action = agent.to_env_action_dynamic(
                    vertex_idx,
                    micro_actions,
                    state.axis_state,
                    face_action=face_action if face_out is not None else None,
                )
                # Legacy fields zero-filled; dynamic fields populated.
                pair_seq = _legacy_zero_pair_seq
                factor_seq = _legacy_zero_factor_seq
                pair_dists = _legacy_zero_pair_dists
                factor_dists = _legacy_zero_factor_dists
                micro_op_seq = micro_actions.op_type
                micro_i_seq = micro_actions.i
                micro_j_seq = micro_actions.j
                micro_exp_seq = micro_actions.exponents
                micro_factor_seq = micro_actions.factor
                micro_compress_kind_seq = micro_actions.compress_kind
                micro_quant_dtype_seq = micro_actions.quant_dtype
                micro_quant_scale_sign_seq = micro_actions.quant_scale_sign
                micro_quant_scale_frac_seq = micro_actions.quant_scale_frac
            env_out = env_obj.step(state, env_action)
            next_state = env_out.state
            raw_rewards = env_out.reward
            # prof/envstep: the device-side env step (order/spec/face-wire
            # shift-and-insert, axis-state update) PLUS the host callback.
            # Subtract `prof/env_cb_host` for the device-only half.
            next_state, raw_rewards = _pp_mark(
                "prof/envstep", (next_state, raw_rewards))
            rewards = raw_rewards
            done = env_out.terminated.astype(jnp.float32)
            # Stage C potential-based shaping. Bootstraps a denser per-step
            # learning signal from the critic; provably preserves the
            # optimum (Ng et al. 1999). The shaping uses raw (un-symlog'd)
            # value space and is stop-gradient'd so the value head only
            # trains against the original returns. Disabled when the
            # coefficient is zero.

            pref_arg = preference if args.preference_conditioned else None
            # Bootstrap value at next_state: extend the post-decision carry by
            # the delta the JUST-CHOSEN elimination emitted -- which is
            # exactly what next_state carries.
            #
            # This used to be thrown away and recomputed by the next scan
            # iteration, on the reasoning that "deltas are O(hundreds) of
            # tokens, so the duplicate extend is cheap". The deltas ARE
            # O(hundreds) (TLM mean 94) but the extend is O(WINDOW), and the
            # duplicate measured 30.5 s/episode = 20.9%. So keep the carry
            # and thread it: it IS the next iteration's post-delta state.
            #
            # Equality, term by term, at iteration t -> t+1:
            #   the scan carries next_state, so state_{t+1}.delta_* is the
            #     next_state.delta_* fed here;
            #   owner_{t+1} = elim_order[step_count_{t+1} - 1] =
            #     elim_order[t] = vertex_idx_t, which is what is passed here
            #     (elim_order was written at index step_count = t just above);
            #   the base carry is enc_carry2/vmem2 in both cases.
            # Same function, same inputs, so the same bits -- this is a
            # deletion of recomputation, not a change of semantics.
            # PARTICIPATION of the delta the just-chosen elimination emits:
            # the vertex itself plus the endpoints of every face it contracted
            # through. `face_ends_v` is the face loop's own enumeration, so no
            # second host probe is needed.
            step_part = agent.participation_mask(
                total_v, vertex_idx.astype(jnp.int32),
                face_ends_v, face_valid_v)
            nxt_carry, nv_s_raw, nv_c_raw = _carry_stream.advance(
                agent, enc_carry2, vmem_s2, vmem_c2,
                next_state.delta_tokens, next_state.delta_eqns,
                next_state.delta_count, vertex_idx.astype(jnp.int32),
                window=MAX_DELTA_TOKENS, participants=step_part,
            )
            # The UNMARKED triple is what gets threaded (see next_enc_state):
            # `_pp_mark` adds a host-produced 0.0 to every numeric leaf, so
            # threading the marked one would put two marks on the value the
            # next iteration re-marks as `prof/encode`, where today there is
            # exactly one. Timing is unaffected -- the mark still forces the
            # extend to be complete before its callback fires.
            nv_s, nv_c = _pp_mark(
                "prof/encode_bootstrap", (nv_s_raw, nv_c_raw))
            _, _, next_value = _carry_stream.heads(
                agent, nv_s, nv_c,
                base_mem=base_mem,
                preference=pref_arg,
            )
            next_value = _pp_mark("prof/heads_bootstrap", next_value)

            # PRE-step snapshots (§7b): the carry/memory synced to the
            # PREVIOUS step's delta — the loss re-derives this step's
            # encoding by the same delta extension.
            _enc_fields = dict(
                enc_M=enc_carry.M, enc_I=enc_carry.I,
                enc_cumhist=enc_carry.cumhist,
                enc_nvalid=enc_carry.nvalid, enc_pos=enc_carry.pos,
                vmem_sums=vmem_s, vmem_counts=vmem_c,
                delta_owner=delta_owner,
                delta_participants=prev_part,
                delta_tokens=delta_tok, delta_eqns=delta_eqn,
                delta_count=jnp.asarray(delta_count, jnp.int32),
            )

            # FEATURE PROBE targets for the vertex this step actually
            # eliminated -- the same prefix + same chosen vertex the oracle
            # callback above is keyed on, so the targets describe the very
            # faces the head decided over.
            if _PROBE_ON:
                _pb_t, _pb_e, _pb_v = _probe_targets(
                    state.order, state.sparsity_specs, state.step_count,
                    vertex_idx, face_ends_v)
                _probe_fields = dict(
                    probe_targets=_pb_t, probe_extents=_pb_e,
                    probe_valid=_pb_v,
                    probe_step=jnp.asarray(state.step_count, jnp.int32),
                )
            else:
                _probe_fields = {}
            transition = Trajectory(
                preference=preference.astype(jnp.float32),
                vertex_idx=jnp.asarray(vertex_idx, dtype=jnp.int32),
                pair_seq=jnp.asarray(pair_seq, dtype=jnp.int32),
                factor_seq=jnp.asarray(factor_seq, dtype=jnp.int32),
                micro_op_seq=micro_op_seq,
                micro_i_seq=micro_i_seq,
                micro_j_seq=micro_j_seq,
                micro_exp_seq=micro_exp_seq,
                micro_factor_seq=micro_factor_seq,
                micro_compress_kind_seq=micro_compress_kind_seq,
                micro_quant_dtype_seq=micro_quant_dtype_seq,
                micro_quant_scale_sign_seq=micro_quant_scale_sign_seq,
                micro_quant_scale_frac_seq=micro_quant_scale_frac_seq,
                reward=jnp.atleast_1d(rewards),
                done=jnp.array(done, dtype=jnp.float32),
                value=jnp.atleast_1d(value),
                next_value=jnp.atleast_1d(next_value),
                vertex_dist=vertex_dist,
                pair_dists=pair_dists,
                factor_dists=factor_dists,
                micro_op_dists=micro_op_dists,
                micro_i_dists=micro_i_dists,
                micro_j_dists=micro_j_dists,
                micro_exp_dists=micro_exp_dists,
                micro_kind_dists=micro_kind_dists,
                micro_quant_logp=micro_quant_logp,
                micro_pair_valid=micro_pair_valid,
                micro_compress_valid=micro_compress_valid,
                axis_state=state.axis_state,
                axis_valid_mask=state.axis_valid_mask,
                face_skip=face_action.skip,
                face_op_type=face_action.op_type,
                face_i=face_action.i,
                face_j=face_action.j,
                face_exponents=face_action.exponents,
                face_factor=face_action.factor,
                face_compress_kind=face_action.compress_kind,
                face_quant_dtype=face_action.quant_dtype,
                face_quant_scale_sign=face_action.quant_scale_sign,
                face_quant_scale_frac=face_action.quant_scale_frac,
                face_pair_valid=face_pair_v,
                face_comp_valid=face_comp_v,
                face_valid=face_valid_v,
                face_endpoints=face_ends_v,
                face_old_logp=jnp.asarray(face_old_logp, jnp.float32),
                face_counts=face_cnt_v,
                face_delta_tokens=face_dt_v,
                face_delta_eqns=face_de_v,
                **_enc_fields,
                **_probe_fields,
                discount=jnp.array(args.discount),
                vertex_avail_mask=vertex_avail_mask,
            )
            # (PRE, POST) for the next iteration: this step's POST becomes its
            # PRE, and the bootstrap above is already its POST.
            next_enc_state = (enc_carry2, vmem_s2, vmem_c2,
                              nxt_carry, nv_s_raw, nv_c_raw)
            return (
                (next_state, elim_order, next_enc_state, step_part),
                (transition, raw_rewards),
            )

        (final_state, _, _, _), (traj, all_raw_rewards) = lax.scan(
            step_fn,
            (env_state, jnp.zeros((total_v,), dtype=jnp.int32),
             init_enc_state, _init_part),
            keys,
        )
        return final_state, traj, all_raw_rewards[-1]

    def loss_fn(
        agent,
        batch: TrainBatch,
        key,
        pin_rules_to_exact_jax,
        op_legality_override,
    ):
        # Dynamic-substeps path branches off here so the legacy path
        # stays exactly as written. `_dynamic_loss_fn` lives below and
        # mirrors the same return shape — total_loss + 10-tuple of
        # metrics — so the train_episode plumbing doesn't care which
        # path was taken. (The dynamic path doesn't use pin_rules_to_exact;
        # the JAX-traced arg is ignored there.)
        if args.dynamic_substeps:
            return _dynamic_loss_fn(
                agent, batch, key, op_legality_override
            )
    def _dynamic_loss_fn(
        agent, batch: TrainBatch, key, op_legality_override
    ):
        """Dynamic-substeps loss: routes through MicroActionPolicy.evaluate.

        Mirrors :func:`loss_fn`'s legacy structure (same cached/no-cache
        batching regimes, same return shape) but evaluates the typed
        micro-action sequence stored in ``batch.micro_*_seq`` instead of
        the legacy ``pair_seq`` / ``factor_seq``. PPO ratio is computed
        from the joint log-prob; entropy is normalized by the per-sample
        sub-episode length returned by ``MicroActionPolicy.evaluate``.

        KL tracking is currently a scalar zero placeholder — per-component
        KL across op_type / i / j / prime-exponents is a clean follow-up
        once the legacy/dynamic split has stabilised.
        """
        pref_or_none = (
            (lambda p: p) if args.preference_conditioned else (lambda _: None)
        )

        # --grad-window 0: THE FULL-HORIZON PATH. The minibatch arrives as
        # whole TRAJECTORIES, `(envs_per_mb, T, ...)`, because a scan needs a
        # trajectory's steps in order. Everything downstream of the carry --
        # the action evaluation, the ratio, the advantage, every mean -- is
        # written against a batch that is FLAT in samples, so flatten (env,
        # step) here and keep the sequence-shaped view for the scan alone.
        # Row-major `reshape(-1)` maps (e, t) -> e*T + t, which is exactly the
        # order `jax.vmap(scan)` produces its outputs in, so the two sides
        # line up without an index.
        _FULL_SCAN = int(getattr(args, "grad_window", 1)) == 0
        _ep_batch = batch if _FULL_SCAN else None
        if _FULL_SCAN:
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), batch)

        keys = jrand.split(key, batch.vertex_idx.shape[0])

        actions = MicroAction(
            op_type=batch.micro_op_seq,
            i=batch.micro_i_seq,
            j=batch.micro_j_seq,
            exponents=batch.micro_exp_seq,
            factor=batch.micro_factor_seq,
            compress_kind=batch.micro_compress_kind_seq,
            quant_dtype=batch.micro_quant_dtype_seq,
            quant_scale_sign=batch.micro_quant_scale_sign_seq,
            quant_scale_frac=batch.micro_quant_scale_frac_seq,
        )
        # P1c: the stored per-path decisions, as one vmapped pytree. None when
        # --face-actions is off (static) — evaluate then skips the face pass.
        face_actions_b = (
            FaceAction(
                skip=batch.face_skip,
                op_type=batch.face_op_type,
                i=batch.face_i,
                j=batch.face_j,
                exponents=batch.face_exponents,
                factor=batch.face_factor,
                compress_kind=batch.face_compress_kind,
                quant_dtype=batch.face_quant_dtype,
                quant_scale_sign=batch.face_quant_scale_sign,
                quant_scale_frac=batch.face_quant_scale_frac,
            )
            if args.face_actions
            else None
        )

        def _eval_dyn(pref, vidx, action, vmask, ax_st, ax_vm,
                      k, pv, cv, fa=None, fpv=None, fcv=None, fv=None,
                      fen=None, pc3=None, fch=None, fcy=None, fb=None,
                      fwb=None):
            # No token arguments: `pc3` (the carry-derived heads) IS the
            # encoding, exactly as on the rollout side.
            return agent.evaluate_action_dynamic(
                None,
                vidx,
                action,
                vmask,
                ax_st,
                ax_vm,
                factor_tables,
                k,
                eqn_ids=None,
                cached_encoding=None,
                preference=pref_or_none(pref),
                pair_valid=pv,
                compress_valid=cv,
                # Static per-call variant mask, identical for every sample in
                # the batch (closed over, not vmapped) — must match sampling.
                op_legality_override=op_legality_override,
                face_action=fa,
                face_pair_valid=fpv,
                face_comp_valid=fcv,
                face_valid=fv,
                face_ends=fen,
                precomputed=pc3,
                face_chunks=fch,
                face_carry=fcy,
                face_bound=fb,
                face_win_budget=fwb,
            )

        # Re-derive each sample's encoding by extending its stored
        # PRE-step carry with the STORED delta buffer + count, fold into
        # the stored vertex memory, and run the heads — the exact
        # computation the rollout sampled under, through CURRENT params
        # (ratio 1 at epoch 0; gradient flows through the delta + heads,
        # truncating at the stored carry by design).

        # BATCH-WIDE window bounds, computed once OUTSIDE the vmaps below and
        # closed over (a closed-over tracer is unbatched inside vmap, which is
        # what keeps the chunk predicate a real `cond`). The loss scans
        # ceil(bound / chunk) chunks instead of the full 16384-step window.
        _delta_budget = jnp.max(batch.delta_count.astype(jnp.int32))
        if args.face_actions and _LIVE_FACES is not None:
            # One past the LAST valid face index, not the count: the sampling
            # masks come from the live oracle and are not contractually
            # packed, so a sum would be wrong the day one is not a prefix.
            _fv = batch.face_valid
            _F_ax = jnp.arange(_fv.shape[-1], dtype=jnp.int32)
            _face_bound = jnp.max(
                jnp.max(jnp.where(_fv > 0, _F_ax, -1), axis=-1)) + 1
            _face_win_budget = jnp.max(
                jnp.sum(batch.face_counts.astype(jnp.int32), axis=-1))
        else:
            _face_bound = None
            _face_win_budget = None

        # THE BASE ENCODE, RE-RUN INSIDE THE LOSS. This is the whole point of
        # the design: `base_memory` is palimpsa's pass over the WHOLE base
        # token stream, under the CURRENT parameters, inside `filter_grad`.
        # Before this existed the base rows were computed once per episode
        # outside the loss and the encoder that produced them received a
        # cotangent of exactly 0.0 (measured); the only palimpsa gradient was
        # the `--grad-window` K deltas. Now every vertex slot's content is
        # differentiated end to end.
        #
        # Computed ONCE per loss call and OUTSIDE the vmaps below, because the
        # base stream is a constant of the graph: every sample in the
        # minibatch shares it, and a closed-over UNBATCHED tracer is exactly
        # what vmap wants (the same reason `_delta_budget` is computed here).
        # It costs one base-length scan per loss call, not one per sample.
        # `base_memory` IS `init_carry(...)[1:]`; taking the whole triple costs
        # nothing extra (one call, and the unused carry is DCE'd on the K
        # path) and gives the full-horizon scan its starting carry -- the
        # palimpsa state after the base stream, recomputed under the current
        # parameters rather than read back from the trajectory.
        _init0 = _carry_stream.init_carry(
            agent, _BASE_TOK, _BASE_EQN, _BASE_N,
            window=_BASE_W, total_v=total_v, embd_dim=args.embd_dim,
            base_owners=_BASE_OWN,
        )
        _base_carry = _init0[0]
        _base_mem = _init0[1:]

        def _advance_k(carry2, vs2, vc2, dtok_k, deqn_k, dcnt_k, own_k,
                       part_k):
            return _carry_stream.advance(
                agent, carry2, vs2, vc2,
                dtok_k, deqn_k, dcnt_k, own_k,
                window=MAX_DELTA_TOKENS, participants=part_k,
                # The loss is reverse-differentiated through this extend,
                # so it cannot use the rollout's while_loop -- it passes
                # the batch-wide `budget` instead and gets the scan/cond
                # form, which has a transpose rule and skips exactly the
                # same pad steps.
                chunk=None, budget=_delta_budget,
            )

        # REMAT THE K-LOOP BODY. `advance` is a whole `encode_extend` over
        # the delta window, and the K of them were an UNROLLED PYTHON LOOP:
        # reverse-mode AD stored every step's rows -- a (window, E) array per
        # K per sample -- so peak memory tracked K almost linearly (measured
        # on the TLM: 8971 MiB at K=1, 9037 at 2, 13197 at 4, 13401 at 8,
        # 21785 at 16). With remat each of the K steps stores only its
        # boundary `(carry, vmem_sums, vmem_counts)` and recomputes its own
        # forward when the cotangent arrives -- the same trade `_block` and
        # `_chunk_d` already make one level down, and the reason nesting is
        # correct rather than doubly wasteful: the inner remat bounds what a
        # single recomputed step costs.
        #
        # NOTHING COMPUTED CHANGES, only where the activations live: the
        # recomputation replays the identical jaxpr on the identical inputs,
        # so the forward is bitwise identical and the cotangents are the
        # cotangents of the same function (proved in
        # tests/carry_heads_remat_equiv_test.py: outputs AND gradients
        # bit-identical at K=1..4, remat on vs off).
        # ALPHAGRAD_CARRY_HEADS_REMAT=0 restores the stored-residual form.
        _advance_step = (
            jax.checkpoint(_advance_k)
            if os.environ.get("ALPHAGRAD_CARRY_HEADS_REMAT", "1") != "0"
            else _advance_k
        )

        def _carry_heads(M, I, ch, nv, pos, owner, part, vs, vc,
                         pref, dtok, deqn, dcnt):
            carry2 = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
            vs2, vc2 = vs, vc
            # The deltas are STORED, with their lengths. The loss re-derives
            # no window from anything, so there is no length to get wrong.
            # PYTHON loop, not a scan: K is static, and at K=1 this is
            # literally the single call it replaced -- same ops, same order,
            # bit-identical.
            for _k in range(dtok.shape[0]):
                carry2, vs2, vc2 = _advance_step(
                    carry2, vs2, vc2,
                    dtok[_k], deqn[_k], dcnt[_k], owner[_k], part[_k],
                )
            # carry2 is where the sampling side carry branched: the
            # stored pre-step carry advanced past the PREVIOUS delta.
            # The face replay continues from it over the stored emission
            # window. (The rows above are the previous delta's -- the
            # WRONG tokens for face contexts; see face_delta_tokens.)
            return _carry_stream.heads(
                agent, vs2, vc2,
                base_mem=_base_mem,
                preference=pref_or_none(pref),
            ) + (carry2,)

        def _episode_heads(dtok, deqn, dcnt, own, part, pref):
            """ONE episode, ONE `lax.scan`: gradient horizon T, not K.

            THE RECURRENCE IS ALREADY IN THE TRAJECTORY. The rollout carries
            the encoder state at two points, PRE and POST this step's delta,
            and threads POST forward as the next step's PRE (`next_enc_state`
            in `step_fn`); PRE at step 0 is `init_carry`'s carry over the base
            stream with a ZERO dynamic memory. So

                POST_t = advance(POST_{t-1}, delta_t),  POST_{-1} = base carry

            is exactly a scan, and the per-step anchors the K path gathers are
            that scan's intermediate carries -- STORED, i.e. constants, which
            is precisely what truncated the gradient at K steps. Running the
            recurrence here instead makes every earlier delta an ancestor of
            step t's logits, so one cotangent reaches all T encodes.

            The body is `_advance_k` UNCHANGED -- the same reverse-
            differentiable `chunk=None, budget=` scan/cond form the K loop
            uses -- so this is a re-association of the SAME per-step function,
            not a different one. At K=1 the two paths do the same NUMBER of
            advances (one per (env, step)); what changes is that they are
            chained rather than independent.

            `jax.checkpoint` on the whole body is what keeps that affordable:
            a length-T scan otherwise stores every step's encode residuals for
            the backward pass. With it each step stores only its boundary
            `(carry, vmem_sums, vmem_counts)` and replays its own forward when
            the cotangent arrives -- the same trade the K loop makes, one
            level up. ALPHAGRAD_CARRY_HEADS_REMAT=0 restores stored residuals.
            """
            def _body(state, x):
                c2, s2, n2 = state
                _dt, _de, _dc, _ow, _pa, _pr = x
                c2, s2, n2 = _advance_k(c2, s2, n2, _dt, _de, _dc, _ow, _pa)
                # The heads run INSIDE the scan, off this step's POST memory:
                # the K path's `heads` call, once per step, unchanged.
                out = _carry_stream.heads(
                    agent, s2, n2,
                    base_mem=_base_mem,
                    preference=pref_or_none(_pr),
                ) + (c2,)
                return (c2, s2, n2), out

            _bd = (
                jax.checkpoint(_body)
                if os.environ.get("ALPHAGRAD_CARRY_HEADS_REMAT", "1") != "0"
                else _body
            )
            _vs0, _vc0 = _carry_stream.zero_memory(total_v, args.embd_dim)
            _, ys = lax.scan(
                _bd, (_base_carry, _vs0, _vc0),
                (dtok, deqn, dcnt, own, part, pref),
            )
            return ys

        if _FULL_SCAN:
            # STATE THE HORIZON, once per compile. "Full T" is a claim about
            # the scan length, and the scan length is a shape -- so print the
            # shape rather than leave the reader to infer it from the flag.
            # Trace time, so one line per compile and nothing per step.
            print("[grad-window 0] full-horizon scan: T=%d steps x %d "
                  "envs/minibatch, delta window %d"
                  % (_ep_batch.delta_count.shape[1],
                     _ep_batch.delta_count.shape[0],
                     _ep_batch.delta_tokens.shape[-1]), flush=True)
            # vmap over ENVS (the sequence axis is the scan's), then flatten
            # (env, step) so everything downstream sees the flat batch it
            # always has. The stored anchors -- enc_M / enc_I / enc_cumhist /
            # enc_nvalid / enc_pos / vmem_sums / vmem_counts -- are NOT read
            # here: that is the whole point, and `full_batch` carries them as
            # dead placeholders on this path.
            pc_logits, pc_ctx, pc_value, pc_carry = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                jax.vmap(_episode_heads)(
                    _ep_batch.delta_tokens, _ep_batch.delta_eqns,
                    _ep_batch.delta_count, _ep_batch.delta_owner,
                    _ep_batch.delta_participants, _ep_batch.preference,
                ),
            )
        else:
            pc_logits, pc_ctx, pc_value, pc_carry = jax.vmap(_carry_heads)(
                batch.enc_M, batch.enc_I, batch.enc_cumhist,
                batch.enc_nvalid, batch.enc_pos, batch.delta_owner,
                batch.delta_participants,
                batch.vmem_sums, batch.vmem_counts,
                batch.preference,
                batch.delta_tokens, batch.delta_eqns, batch.delta_count,
            )
        (
            log_probs,
            entropies,
            values,
            new_vertex_dist,
            sub_lengths,
            new_op_dists,
            new_i_dists,
            new_j_dists,
            new_exp_dists,
            new_kind_dists,
            new_quant_logp,
            face_ents,
            probe_reprs,
        ) = (
            jax.vmap(
                lambda pref, vidx, action, vmask, ax_st,
                ax_vm, k, pv, cv, fa, fpv, fcv, fv, fen, pl, pc, pvl,
                fct, fdt, fde, cy:
                _eval_dyn(
                    pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv, fa, fpv, fcv, fv, fen,
                    pc3=(pl, pc, pvl),
                    # Chunk lengths + THIS step's emission window + the
                    # branch-point carry: everything the behaviour
                    # policy's face contexts were built from. Anything
                    # else and the ratio is not 1.
                    fch=((fct, fdt, fde) if _LIVE_FACES is not None
                         else None),
                    fcy=(cy if _LIVE_FACES is not None else None),
                    fb=_face_bound,
                    fwb=_face_win_budget,
                )
            )(
                batch.preference,
                batch.vertex_idx,
                actions,
                batch.vertex_avail_mask,
                batch.axis_state,
                batch.axis_valid_mask,
                keys,
                batch.micro_pair_valid,
                batch.micro_compress_valid,
                face_actions_b,
                batch.face_pair_valid,
                batch.face_comp_valid,
                batch.face_valid,
                batch.face_endpoints,
                pc_logits, pc_ctx, pc_value,
                batch.face_counts,
                batch.face_delta_tokens, batch.face_delta_eqns,
                pc_carry,
            )
            if args.face_actions
            else jax.vmap(
                lambda pref, vidx, action, vmask, ax_st,
                ax_vm, k, pv, cv, pl, pc, pvl:
                _eval_dyn(
                    pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv, pc3=(pl, pc, pvl)
                )
            )(
                batch.preference,
                batch.vertex_idx,
                actions,
                batch.vertex_avail_mask,
                batch.axis_state,
                batch.axis_valid_mask,
                keys,
                batch.micro_pair_valid,
                batch.micro_compress_valid,
                pc_logits, pc_ctx, pc_value,
            )
        )

        old_log_probs = jax.vmap(old_micro_log_prob_for_action)(
            batch.vertex_idx,
            batch.micro_op_seq,
            batch.micro_i_seq,
            batch.micro_j_seq,
            batch.micro_exp_seq,
            batch.micro_compress_kind_seq,
            batch.micro_quant_dtype_seq,
            batch.old_vertex_dist,
            batch.old_micro_op_dists,
            batch.old_micro_i_dists,
            batch.old_micro_j_dists,
            batch.old_micro_exp_dists,
            batch.old_micro_kind_dists,
            batch.old_micro_quant_logp,
        )
        if getattr(args, "unified_head", False):
            # The unified head's joint log-prob is NOT rebuildable from the
            # per-sub-step dists: there is no slot for the skip Bernoulli, the
            # nine axis gates, the reduce-fn or the dtype bit. Reconstructing
            # it anyway made the two sides of the ratio different formulas --
            # measured median 2.37, max 2.3e23, at epoch 0 with identical
            # weights, which is the 1e5-1e10 PPO loss seen from episode 0.
            # The adapter stores the real joint log-prob in the quant slot and
            # emits point-mass sub-step dists (so the reconstruction's other
            # terms are log(1) = 0); take it directly, ungated, because the
            # in-function quant term only fires for OP_QUANT.
            _vd = jnp.clip(batch.vertex_idx.astype(jnp.int32), 0,
                           batch.old_vertex_dist.shape[-1] - 1)
            _v_old = jnp.log(
                jnp.take_along_axis(batch.old_vertex_dist, _vd[:, None],
                                    axis=-1).squeeze(-1) + 1e-8)
            old_log_probs = _v_old + batch.old_micro_quant_logp
        if args.face_actions:
            # The behaviour policy's face log-prob was captured as a scalar at
            # sample time (FacePathPolicy sample == evaluate parity is unit-
            # pinned); the new side lives inside `log_probs` via
            # evaluate_action_dynamic's face pass.
            old_log_probs = old_log_probs + batch.face_old_logp

        ratio = jnp.exp(log_probs - old_log_probs)
        # T2 diagnostics. max|log-ratio| rather than the mean: the unified-head
        # ratio bug sat in the TAIL (median 2.374, max 2.32e23), which a
        # batch-averaged KL hides. At epoch 0 this must be ~0 by construction.
        _log_ratio = log_probs - old_log_probs
        _max_log_ratio = jnp.max(jnp.abs(_log_ratio))
        # KL(old || new) on the JOINT log-prob -- the quantity the ratio uses,
        # and the only live KL for this head now that the per-sub-step dists
        # are point masses.
        # Schulman's low-variance, non-negative KL estimator:
        #   k3 = (r - 1) - log r,  with r = exp(new - old)
        _kl_approx = jnp.mean((ratio - 1.0) - _log_ratio)
        num_triggers = get_num_clipping_triggers(ratio, args.ppo_clip_eps)
        trigger_ratio = num_triggers / len(ratio)

        clipping_objective = jnp.minimum(
            ratio * batch.norm_adv,
            jnp.clip(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
            * batch.norm_adv,
        )
        ppo_loss = jnp.mean(-clipping_objective)

        # Entropy normalized by per-sample sub-episode length (returned by
        # MicroActionPolicy.evaluate). Clamp to ≥ 1.0 to avoid divide-by-
        # zero on samples where the sub-episode was forced END at step 0.
        # Already per-head normalised inside evaluate_action_dynamic (each
        # policy divided by its OWN arity), so this is a plain mean. Dividing
        # again by sub_lengths here would re-introduce the shared-denominator
        # bug from the other side.
        entropy_loss = jnp.mean(entropies)

        # See the legacy loss path's value-mode switch for the rationale;
        # in scalar mode only slot 0 of (values, estim_returns) is alive.
        if args.loss_mode == "scalar":
            value_loss = jnp.mean(
                (values[..., 0] - _value_target(batch.estim_returns[..., 0]))
                ** 2
            )
            explained_var = explained_variance(
                values[..., 0], batch.estim_returns[..., 0]
            )
        else:
            value_loss = jnp.mean(
                jnp.sum(
                    (values - _value_target(batch.estim_returns)) ** 2,
                    axis=-1,
                )
            )
            explained_var = explained_variance(
                jnp.sum(values, axis=-1),
                jnp.sum(batch.estim_returns, axis=-1),
            )

        # Per-component KL on the dynamic path: vertex + op_type + i + j +
        # prime-exponents. Mean over the batch dim; for the per-sub-step
        # dists we also sum over the max_substeps axis after the per-step
        # KL, then divide by the per-sample sub-episode length so the
        # contribution is normalized the same way as the entropy bonus.
        kl_vertex = jnp.mean(
            optax.kl_divergence(
                jnp.log(new_vertex_dist + 1e-7),
                batch.old_vertex_dist,
            )
        )

        # Active-sub-step gating mirrors old_micro_log_prob_for_action:
        # entries past the first OP_END contribute 0.
        is_end_seq = batch.micro_op_seq == OP_END
        prior_ends = jnp.cumsum(
            is_end_seq.astype(jnp.int32), axis=-1
        ) - is_end_seq.astype(jnp.int32)
        active_steps = (prior_ends == 0).astype(jnp.float32)  # (B, S)
        is_diag_step = (batch.micro_op_seq == OP_DIAG).astype(jnp.float32)
        is_compress_step = (batch.micro_op_seq == OP_COMPRESS).astype(jnp.float32)
        is_quant_step = (batch.micro_op_seq == OP_QUANT).astype(jnp.float32)
        is_diag_or_compress = (
            (batch.micro_op_seq == OP_DIAG) | (batch.micro_op_seq == OP_COMPRESS)
        ).astype(jnp.float32)
        denom = jnp.maximum(sub_lengths, 1.0)  # (B,)

        def _per_step_kl(new_d, old_d, gate):
            """KL[new || old] per (batch, sub-step) gated by `gate`, then
            sum-over-substeps and batch-mean-with-sub-episode-length norm."""
            kl = optax.kl_divergence(jnp.log(new_d + 1e-7), old_d)  # (B, S, ...)
            # Collapse any trailing component axes (e.g. NUM_OPS, MAX_AXES)
            # into the scalar per (B, S).
            while kl.ndim > 2:
                kl = jnp.sum(kl, axis=-1)
            kl = kl * gate  # (B, S)
            per_sample = jnp.sum(kl, axis=-1) / denom  # (B,)
            return jnp.mean(per_sample)

        kl_op = _per_step_kl(new_op_dists, batch.old_micro_op_dists, active_steps)
        kl_i = _per_step_kl(
            new_i_dists,
            batch.old_micro_i_dists,
            active_steps * is_diag_or_compress,
        )
        kl_j = _per_step_kl(
            new_j_dists,
            batch.old_micro_j_dists,
            active_steps * is_diag_step,
        )
        # exp_dists shape (B, S, MAX_PRIMES, MAX_EXPONENT+1). KL collapses
        # last two axes; gate by DIAG-active.
        kl_exp = _per_step_kl(
            new_exp_dists,
            batch.old_micro_exp_dists,
            active_steps * is_diag_step,
        )
        kl_kind = _per_step_kl(
            new_kind_dists,
            batch.old_micro_kind_dists,
            active_steps * is_compress_step,
        )
        # The factored quant head has no single dtype dist, so its per-component
        # KL diagnostic is dropped (the joint log-prob still drives the ratio;
        # the joint entropy still carries its exploration signal). TODO: surface
        # the factored quant entropy/KL separately for logging.
        kl_quant = jnp.zeros_like(kl_kind)
        kl_div = kl_vertex + kl_op + kl_i + kl_j + kl_exp + kl_kind + kl_quant
        # Stash per-component KLs so they can be logged separately — they're
        # the most useful single signal for debugging the dynamic head
        # (factor head and END decision are where collapse starts per the
        # design spec). The 5-slot layout is preserved for back-compat; the
        # kind and quant KLs are folded into the exponent slot since both
        # gate on their respective op-type (COMPRESS / QUANT) and the legacy
        # consumer reads slot 4 as "non-vertex / non-op_type / non-axis"
        # collateral.
        # slots 5/6 are T2: joint-ratio KL and max|log-ratio|. The per-sub-step
        # slots above read 0 for the unified head (point-mass dists), so these
        # are the live diagnostics there.
        _kl_components = (kl_vertex, kl_op, kl_i, kl_j,
                          kl_exp + kl_kind + kl_quant,
                          jnp.asarray(_kl_approx, jnp.float32),
                          jnp.asarray(_max_log_ratio, jnp.float32))

        # Per-component entropy under the current policy, using the same
        # active-substep / DIAG gating as the KL split. Pairs with the
        # per-component KL for diagnosing which head is collapsing vs which
        # is exploring. The trailing component axes (NUM_OPS / MAX_AXES /
        # MAX_PRIMES × MAX_EXPONENT+1) are collapsed via the standard
        # -sum(p log p) entropy.
        def _step_entropy(d):
            return -jnp.sum(d * jnp.log(d + 1e-8), axis=-1)

        op_ent_per = _step_entropy(new_op_dists)  # (B, S)
        i_ent_per = _step_entropy(new_i_dists)
        j_ent_per = _step_entropy(new_j_dists)
        # exp_dists: (B, S, MAX_PRIMES, MAX_EXPONENT+1) — entropy over the
        # exponent axis, then sum over the (padded) prime axis.
        exp_ent_per = jnp.sum(_step_entropy(new_exp_dists), axis=-1)  # (B, S)
        kind_ent_per = _step_entropy(new_kind_dists)  # (B, S)
        quant_ent_per = jnp.zeros_like(kind_ent_per)  # factored quant: see kl_quant note

        ent_vertex = jnp.mean(_step_entropy(new_vertex_dist))
        ent_op = jnp.mean(jnp.sum(op_ent_per * active_steps, axis=-1) / denom)
        ent_i = jnp.mean(
            jnp.sum(i_ent_per * active_steps * is_diag_or_compress, axis=-1) / denom
        )
        ent_j = jnp.mean(
            jnp.sum(j_ent_per * active_steps * is_diag_step, axis=-1) / denom
        )
        ent_exp = jnp.mean(
            jnp.sum(exp_ent_per * active_steps * is_diag_step, axis=-1) / denom
        )
        ent_kind = jnp.mean(
            jnp.sum(kind_ent_per * active_steps * is_compress_step, axis=-1) / denom
        )
        ent_quant = jnp.mean(
            jnp.sum(quant_ent_per * active_steps * is_quant_step, axis=-1) / denom
        )
        # Fold the kind + quant entropies into the exp slot to keep the
        # first 5 slots identical to the legacy loss path's layout. Slot 5 is
        # NEW: the per-FACE approximation head's arity-normalised entropy,
        # which is the ONLY live approximation entropy under --live-faces.
        _entropy_components = (
            ent_vertex, ent_op, ent_i, ent_j, ent_exp + ent_kind + ent_quant,
            jnp.mean(face_ents),
        )

        total_loss = (
            ppo_loss
            + args.value_weight * value_loss
            - args.entropy_weight * entropy_loss
        )

        # ---------------------------------------------------- FEATURE PROBE
        # The pack is returned as `has_aux` DATA, never as a term of
        # `total_loss`: `eqx.filter_grad` does not differentiate aux, and the
        # representation is `stop_gradient`-ed here as well as inside
        # `Probe.__call__`. Two independent reasons the probe cannot train what
        # it measures, either of which alone is sufficient.
        _probe_pack = None
        if _PROBE_ON:
            _lat, _vctx = probe_reprs
            if _lat is None:
                # No face latents on the loss path. Decoding zeros is not a
                # measurement (job 61427); the setup gate should have refused
                # this configuration already, so reaching here is a bug.
                raise RuntimeError(
                    "feature probe: the loss path produced no face latents; "
                    "the probe needs the --live-faces replay "
                    "(face_path_policy present and face_chunks stored).")
            _lat = _lat[:, :_PROBE_FACES]
            _lat = jax.lax.stop_gradient(_lat)
            _vctx = jax.lax.stop_gradient(_vctx)
            if _fprobe.PROBE_ARM == _fprobe.FaceProbeArm.ENDPOINTS:
                # The reference arm every prior face probe was measured on:
                # the two ENDPOINT slot rows, gathered from the same pointer
                # memory. `face_endpoints` is 1-based with 0 = jaxpr input, so
                # index e-1 and zero the input rows.
                _ends = batch.face_endpoints[:, :_PROBE_FACES]
                _g = jax.vmap(lambda c, e: c[e])(
                    jax.lax.stop_gradient(pc_ctx),
                    jnp.clip(_ends - 1, 0, pc_ctx.shape[1] - 1))
                _g = _g * (_ends > 0).astype(jnp.float32)[..., None]
                _cxi, _cxj = _g[:, :, 0], _g[:, :, 1]
            else:
                _cxi = _cxj = None
            _probe_pack = (_lat, _vctx, batch.probe_targets,
                           batch.probe_extents, batch.probe_valid,
                           batch.probe_step, _cxi, _cxj)

        # NaN LOCALIZER (ALPHAGRAD_DEBUG_NAN=1).
        # `ent:nan` in the progress bar means the params are ALREADY NaN, which
        # is one update too late to say why. This prints each loss component
        # the moment any of them goes non-finite, which splits the two cases
        # that need completely different fixes:
        #   * a component is NaN  -> the data/masking is bad (0/0, all-masked
        #     head, empty minibatch)
        #   * all finite but the param update still NaNs -> the GRADIENT is
        #     the problem (sqrt/norm/abs evaluated at exactly 0)
        if _DEBUG_NAN:
            _bad = jnp.logical_not(jnp.isfinite(total_loss))
            jax.lax.cond(
                _bad,
                lambda: jax.debug.print(
                    "[nan] ppo={p} value={v} entropy={e} kl={k} expvar={x} "
                    "adv_nan={an} adv_absmax={a} ret_nan={rn} ret_absmax={r}",
                    p=ppo_loss, v=value_loss, e=entropy_loss, k=kl_div,
                    x=explained_var,
                    an=jnp.sum(jnp.logical_not(jnp.isfinite(batch.norm_adv))),
                    a=jnp.max(jnp.abs(jnp.nan_to_num(batch.norm_adv))),
                    rn=jnp.sum(
                        jnp.logical_not(jnp.isfinite(batch.estim_returns))),
                    r=jnp.max(jnp.abs(jnp.nan_to_num(batch.estim_returns))),
                ),
                lambda: None,
            )
        _metrics = (
            kl_div,
            entropy_loss,
            0.0,
            explained_var,
            ppo_loss,
            args.value_weight * value_loss,
            args.entropy_weight * entropy_loss,
            total_loss,
            trigger_ratio,
            # Per-component KLs for dynamic-mode debugging; legacy loss_fn
            # returns the same 5-slot suffix with zeros so the metrics
            # tuple shape is uniform across modes (lax.scan needs that).
            jnp.stack(_kl_components),
            # Per-component entropies (the same 5-slot layout in slots 0-4);
            # legacy returns zeros for the dynamic-only slots and the vertex
            # entropy in slot 0 if available. Slot 5 = face/approximation head.
            jnp.stack(_entropy_components),
        )
        if _PROBE_ON:
            return total_loss, (_metrics, _probe_pack)
        return total_loss, _metrics

    # n_steps for the WITHIN-STEP R2. Elimination step indices run 0..T-1 with
    # T <= total_v, so total_v + 1 is a safe static bound for the one-hot.
    _PROBE_NSTEPS = int(total_v) + 1

    def _probe_loss(probes, pack):
        """MSE of the probe decode, as a function of the PROBE ALONE.

        `pack` arrives as `has_aux` data from the PPO loss -- already
        `stop_gradient`-ed there and again inside `Probe.__call__` -- and the
        agent is not an argument, so this function has no derivative with
        respect to any policy parameter. That is the invariant the whole probe
        exists to preserve: it must not train the thing it measures.
        """
        lat, vctx, tgt, ext, val, sid, cxi, cxj = pack
        n_p = val.shape[-1]
        if _fprobe.PROBE_ARM == _fprobe.FaceProbeArm.LEAN:
            pred = jax.vmap(jax.vmap(probes.face_predict))(lat)
        elif _fprobe.PROBE_ARM == _fprobe.FaceProbeArm.EXTENTS:
            pred = jax.vmap(jax.vmap(
                lambda l, e: probes.face_predict(l, extents=e)))(lat, ext)
        else:
            pred = jax.vmap(jax.vmap(
                lambda l, a, b: probes.face_predict(
                    l, ctx_i=a, ctx_j=b)))(lat, cxi, cxj)
        f_pred = pred.reshape(-1, _fprobe.NFT)
        f_tgt = tgt.reshape(-1, _fprobe.NFT)
        f_val = val.reshape(-1)
        l_face = _fprobe.masked_mse(f_pred, f_tgt, f_val).sum()

        # THE VERTEX PROBE reads ONE pointer slot row (E wide, the lean
        # collapse) and decodes the same seven columns AGGREGATED over that
        # vertex's valid faces -- the vertex-level form of the identical
        # quantity, so the face and vertex numbers sit on one axis instead of
        # measuring two unrelated things.
        w = val[..., None]
        v_tgt = (tgt * w).sum(1) / jnp.maximum(val.sum(1), 1.0)[:, None]
        v_val = (val.sum(1) > 0).astype(jnp.float32)
        v_pred = jax.vmap(probes.vertex_predict)(vctx)
        l_vertex = _fprobe.masked_mse(v_pred, v_tgt, v_val).sum()

        # NO R2 HERE. Computed per minibatch, a within-step group is only
        # (rows at that step in the minibatch) x faces; at --minibatches >=
        # --num-envs the groups degenerate to single vertices, centring
        # annihilates them and the statistic is noise regardless of the
        # representation. The predictions and targets ride out through the
        # aux instead, and train_episode computes ONE R2 per episode over
        # the full batch -- the largest groups obtainable.
        _sid = sid.astype(jnp.int32)
        aux = (f_pred, f_tgt, f_val, jnp.repeat(_sid, n_p),
               v_pred, v_tgt, v_val, _sid, l_face, l_vertex)
        return l_face + l_vertex, aux

    def train_episode(
        agent,
        opt_state,
        env_states,
        env_obj,
        base_mem,
        preferences_per_env,
        global_step,
        key,
        freeze_mask,
        op_legality_override_arg,
        vertex_mult_arg,
        pin_rules_to_exact_arg,
        micro_mult_arg,
        popart_m1,
        popart_m2,
        popart_w,
        probes,
        probe_opt_state,
    ):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, num_envs)
        # Phase-0: start the attribution clock at the top of the episode.
        env_states = _pp_mark(None, env_states)

        # entropy/palimpsa, once per episode on the rollout's FIRST state of
        # env 0 (the root elimination state -- the same input every episode, so
        # the curve isolates the ENCODER's drift rather than state drift).
        # See attention_entropy_diagnostic: representation diagnostic, not a
        # policy entropy.
        env_states, traj, total_rewards_full = rollout_fn(
            agent,
            env_obj,
            num_valid,
            env_states,
            rollout_keys,
            base_mem,
            preferences_per_env,
            op_legality_override_arg,
            pin_rules_to_exact_arg,
            # vertex_temperature: None in training (the vmap in_axes tuple is
            # positional, so this must be passed explicitly).
            None,
        )

        # GAE on the (E, T, NUM_VALUE_HEADS) reward tensor.
        #
        # ``multi_head`` (the default): use the three training-reward indices
        #   (flops, peak_memory, frob_residual) as separate channels; downstream
        #   the per-head advantages get normalized and scalarized by the
        #   (Dirichlet or static-lambda) preference vector.
        #
        # ``scalar`` (vertex_ppo.py-style): collapse the reward vector to a
        #   single scalar per step via ``sum_i(reward_weights[i] * r_i)`` —
        #   raw weighted sum, *without* a per-component pre-normalization. The
        #   cross-component magnitude gap (flops ~1e10 vs cosine_sim ~1) is
        #   handled the same way ``vertex_ppo.py`` handles it: the value head
        #   learns a ``symlog``-normalized target via
        #   ``reward_normalization_fn(estim_returns)`` in the value loss, and
        #   the standard PPO advantage normalization handles the rest. The
        #   scalar is written into channel 0; channels 1/2 stay at zero so the
        #   loss path's value-head shape contract is unchanged. The value loss
        #   below masks out channels 1/2 in scalar mode so the (un-trained)
        #   mem / acc heads don't drift on synthetic zero targets.
        # Project the raw reward vector through symlog (cosine_sim is
        # passed through unchanged — see ``_NO_SYMLOG_MASK``). This
        # compresses the ~10¹⁰ dynamic range across flops / peak_memory /
        # frob_residual / … so the per-channel weighted sum is no longer
        # dominated by flops by 10 orders of magnitude. (NOTE: the
        # "pre-training calibration" older comments referenced never
        # existed — cross-channel balance comes from symlog + the
        # per-channel advantage z-score + the preference weights.)
        # ``--reward-mode mult`` replaces the additive channels entirely
        # with the cosine-gated scalar (see _apply_mult_gate) BEFORE
        # symlog: the gate output lives in the cosine channel, which
        # symlog passes through raw.
        traj_reward = traj.reward
        if args.reward_mode == "mult":
            traj_reward = _apply_mult_gate(
                traj_reward,
                mult_cost_weights,
                args.gate_tau,
                args.gate_w,
                args.anti_degen_penalty,
                args.anti_degen_tau,
                gate_fidelity=args.gate_fidelity,
            )
        sl_reward = _symlog_rewards(traj_reward)  # (E, T, NUM_REWARDS)
        if args.loss_mode == "scalar":
            scalar_reward = jnp.sum(sl_reward * reward_weights, axis=-1)  # (E, T)
            zeros = jnp.zeros_like(scalar_reward)
            head_rewards = jnp.stack(
                [scalar_reward] + [zeros] * (NUM_VALUE_HEADS - 1), axis=-1
            )
        else:
            # ``HEAD_REWARD_INDICES`` selects (latency_ns, peak_memory,
            # cosine_sim, frob_residual). cosine passes through symlog
            # unchanged (bounded [0,1] already); the cost channels are
            # symlog'd. The value head still learns the symlog of
            # estim_returns in the value loss; the GAE math in ``gae.py``
            # treats values as symlog'd (symexp back to "raw" — but with
            # the rewards now in symlog space, "raw" here is the symlog
            # scale, which is stable in the 10²-ish range).
            head_rewards = sl_reward[..., _HEAD_REWARD_INDICES_ARR]
        # PopArt: the value head emits NORMALIZED values, so de-normalize
        # before GAE (which works in raw reward units), then re-normalize the
        # resulting targets. With --advantage-norm zscore this is the identity
        # (mu=0, sigma=1 carried unchanged).
        use_popart = args.advantage_norm == "popart"
        popart_mu, popart_sigma = _popart_derive(
            popart_m1, popart_m2, popart_w, args.popart_sigma_min, 1e12)
        # TRUE OPTIMIZED RETURN (owner 2026-08-09): preference-weighted
        # PopArt-z of the post-gate post-symlog TERMINAL head rewards --
        # the exact scalar this update maximizes, in the space the
        # advantages live in. Valid for additive AND mult (the gate was
        # applied to traj_reward above). Logged as `scalarized_return`,
        # the same key az_gumbel already uses, so the two arms' headline
        # panels finally show the same quantity.
        _term_hr = head_rewards[:, -1, :]
        _term_z = (_term_hr - popart_mu) / popart_sigma
        _pref_t = (traj.preference[:, -1, :]
                   if traj.preference.ndim == 3 else traj.preference)
        true_scalar_return = jnp.mean(jnp.sum(_term_z * _pref_t, axis=-1))
        v_raw = traj.value * popart_sigma + popart_mu
        nv_raw = traj.next_value * popart_sigma + popart_mu
        # ONE value encoding, not two.
        #
        # `get_advantages` is make_get_advantages(use_symlog=True), so its scan
        # does `value_raw = symexp(value)`. That is correct ONLY when the value
        # head's output is symlog-encoded. Under PopArt it is not: the head
        # emits a z-score, and the affine `value * sigma + mu` above has already
        # decoded it. Feeding that to the symlog variant EXPONENTIATES an
        # already-decoded value.
        #
        # It survives exactly one update, which is why this looked like a slow
        # collapse rather than a type error. Round 1 PopArt is cold (sigma=1,
        # mu=0) so v_raw stays ~1.7 and symexp(1.7)~4.3 is harmless. Round 2 it
        # is warm (sigma=42, mu=92), v_raw reaches ~89, and symexp(89)~6e38
        # overflows float32 — the trace showed `advantages` pinned at 3.403e38,
        # FLT_MAX, with estim_returns already NaN and every finite advantage
        # crushed to 0 by the resulting sigma.
        _gae = _GAE_POPART if use_popart else get_advantages
        _, estim_returns, advantages = _gae(
            head_rewards,
            traj.done,
            v_raw if use_popart else traj.value,
            nv_raw if use_popart else traj.next_value,
            traj.discount,
            args.gae_lambda,
        )

        # T3 DIAGNOSTICS. `estim_returns` is what _popart_update consumes, and
        # its final line upstream is `returns = advantages + values` -- a
        # BOOTSTRAPPED target, bounded by the CRITIC, not by the reward. So
        # popart/mu_cos exceeding the reward ceiling (observed 1.874 against
        # 0.946 on v36) is critic overestimation, not the acc reward being
        # emitted more than once. Capture both sides here, in RAW units, before
        # the renormalisation below rewrites estim_returns in place.
        _dv = v_raw if use_popart else traj.value
        _diag_value_raw = jnp.mean(_dv, axis=tuple(range(_dv.ndim - 1)))
        _diag_return_raw = jnp.mean(
            estim_returns, axis=tuple(range(estim_returns.ndim - 1)))
        # Terminal-gate guard: env.py emits an all-zero reward vector for every
        # non-terminal step under --terminal-rewards-only, so at most ONE step
        # per env may carry a non-zero cosine. Max over envs; must stay <= 1.
        _diag_nonzero_cos_steps = jnp.max(jnp.sum(
            (traj.reward[..., int(REWARD_INDEX["cosine_sim"])] != 0.0
             ).astype(jnp.int32), axis=-1))

        # DEGENERATE STEPS ARE NEUTRAL, NOT CATASTROPHIC.
        # The env sentinels a plan that computed nothing (every cost = −1e10).
        # Letting that flow into the advantage would inject an enormous
        # artificial gradient and drag PopArt's per-channel sigma with it — the
        # cure would be worse than the collapse. Instead reuse the VALUE NET's
        # own prediction for those steps: set the advantage to 0 (the critic is
        # taken as correct there, so the TD error vanishes) and drop them from
        # the value target, so nothing trains ON the sentinel. The row is still
        # sentinelled everywhere it is RANKED (top-N, best_global, Pareto), so
        # a degenerate plan can never be crowned — it simply teaches nothing.
        # DEGENERACY DETECTION — must identify the SENTINEL, not "a big number".
        #
        # THE ep-39 CLIFF BUG (v16 post-mortem, found from the observation that
        # the PPO loss stayed smooth while muls_adds_fmas spiked): this read
        #     jnp.any(traj.reward <= SENTINEL_COST * 0.5)   # any channel <= -5e9
        # over ALL EIGHT channels. Channel 0 is -muls_adds_fmas and channel 3
        # is -max_io_sum — RAW SYMBOLIC COUNTS, legitimately ~4e12 on nn256.
        # So every plan doing more than 5e9 ops tripped the test, was declared
        # "degenerate", and had its advantage multiplied by zero. The ONLY
        # transitions that kept a policy gradient were the ones with tiny op
        # counts — i.e. the near-zero-work plans. The trainer was therefore
        # reinforcing degeneracy by construction, and the loss looked healthy
        # the whole time precisely BECAUSE almost every advantage was 0.
        #
        # The sentinel is an exact vector: -1e10 in all six cost channels
        # (cosine 0.0, frob -1.0). Require ALL cost channels at it — no real
        # plan is simultaneously 10 s slow, 10 GB, and 1e10-op in one row —
        # and use a tight bound so a merely expensive plan can never qualify.
        _SENT_CH = jnp.asarray(COMPUTE_REWARD_INDICES, dtype=jnp.int32)
        _is_degen = jnp.all(
            traj.reward[..., _SENT_CH] <= (SENTINEL_COST * 0.99), axis=-1
        )  # (E,T)
        _live = (~_is_degen).astype(jnp.float32)[..., None]                # (E,T,1)
        advantages = advantages * _live

        if use_popart:
            new_m1, new_m2, new_w = _popart_update(
                popart_m1, popart_m2, popart_w, estim_returns,
                args.popart_beta, args.popart_sigma_min, 1e12, 5.0,
            )
            new_mu, new_sigma = _popart_derive(
                new_m1, new_m2, new_w, args.popart_sigma_min, 1e12)
            # ART: output-preserving head rescale so the critic's predictions
            # survive the stats shift.
            agent = _popart_rescale_heads(
                agent, popart_mu, popart_sigma, new_mu, new_sigma)
            # POP: normalized critic targets + sigma-scaled advantages. The
            # per-channel sigma division is what puts the ~1e10 flops channel
            # and the [0,1] cosine channel on a comparable footing WITHOUT the
            # batch z-score's collapse ratchet (a uniformly-degenerate batch
            # drives std->0 and makes the opposing channel vanish).
            # Neutral target on degenerate steps: substitute the value net's
            # own prediction so the value loss for that step is ~0 and the
            # critic is not dragged toward the sentinel.
            #
            # #89: that substitution must be made in the SAME frame the
            # loss compares against. `traj.value` is the OLD head's
            # NORMALISED output, captured before `_popart_rescale_heads`
            # above rewrote the heads into (new_mu, new_sigma), while
            # `values` in the loss come from the RESCALED head. ART
            # preserves the RAW prediction, not the normalised one, so
            # feeding the stale normalised value made every degenerate
            # step contribute a non-zero loss pulling the critic toward
            # the old frame (measured 5.77 where the contract says ~0),
            # worst during the warm start when the stats move most.
            # Carry it across exactly as ART carries the head: de-normalise
            # with the OLD stats, re-normalise with the NEW ones.
            _neutral = (traj.value * popart_sigma + popart_mu
                        - new_mu) / new_sigma
            estim_returns = jnp.where(
                _live > 0.5, (estim_returns - new_mu) / new_sigma, _neutral)
            norm_adv_components = advantages / new_sigma
        elif args.advantage_norm == "none":
            new_m1, new_m2, new_w = popart_m1, popart_m2, popart_w
            # MANUAL-WEIGHT MODE (user-directed): no adaptive statistics
            # anywhere. The symlog channel compression is the only implicit
            # scaling; the --lambda-* weights (via traj.preference below) are
            # the explicit one. Reward semantics are stationary — a given
            # plan scores the same at ep 5 and ep 500, so "up and down then
            # stays down" cannot be a normaliser drifting under the policy.
            # Degenerate steps keep the PopArt-style neutral target: the
            # value loss compares symlog(target) to the head's own output, so
            # substituting symexp(value) makes that step's loss ~0.
            norm_adv_components = advantages
            estim_returns = jnp.where(
                _live > 0.5,
                estim_returns,
                inverse_reward_normalization_fn(traj.value),
            )
        else:
            new_m1, new_m2, new_w = popart_m1, popart_m2, popart_w

            def normalize(x):
                return (x - jnp.mean(x)) / (jnp.std(x) + 1e-7)

            norm_adv_components = jax.vmap(normalize, in_axes=-1, out_axes=-1)(
                advantages.reshape(-1, advantages.shape[-1])
            ).reshape(advantages.shape)
        if args.loss_mode == "scalar":
            # Single-channel path: only slot 0 carries signal — skip the
            # preference scalarization entirely so we don't multiply the
            # advantage by an unrelated CLI lambda twice.
            norm_adv = norm_adv_components[..., 0]
        else:
            # Always use the per-step preference for advantage weighting. In
            # the unconditioned (Stage A–E) path it's broadcast from the static
            # CLI --lambda-* weights; in Stage F it's the Dirichlet sample.
            norm_adv = jnp.sum(norm_adv_components * traj.preference, axis=-1)

        # ADVANTAGE / BASELINE DIAGNOSTIC (ALPHAGRAD_ADV_DIAG=1, default off).
        # `advantages` is what GAE produced, `norm_adv` is what the PPO ratio
        # is actually multiplied by, `estim_returns` is the critic's target
        # and `traj.value` is its prediction. One callback per episode (the
        # arrays are already materialised here), so the cost is one host sync
        # on a path that already has several.
        if _ADV_DIAG:
            # `new_sigma` only exists on the PopArt branch (Python-level `if`),
            # so fall back to the incoming accumulator elsewhere.
            _sig_diag = new_sigma if use_popart else popart_sigma

            def _adv_cb(_a, _na, _er, _v, _s):
                _ADV_STATS.clear()
                _ADV_STATS.update(_spread("adv_raw", _a))
                _ADV_STATS.update(_spread("adv_norm", _na))
                _ADV_STATS.update(_spread("estim_returns", _er))
                _ADV_STATS.update(_spread("value_pred", _v))
                _ADV_STATS.update(_spread("popart_sigma", _s))

            jax.debug.callback(
                _adv_cb, advantages, norm_adv, estim_returns, traj.value,
                _sig_diag)

        # Stage-by-stage NaN trace through the advantage/return path. The loss
        # localizer proved the NaN is ALREADY in norm_adv / estim_returns when
        # the batch is built, with every finite advantage exactly 0 and returns
        # still at raw op-count scale (1e11) rather than symlog scale (~25).
        # These prints say which stage first breaks that chain.
        if _DEBUG_NAN:
            def _st(name, x):
                return jax.debug.print(
                    "[trace] {n:<16} nan={b:<5} absmax={m:.4g}",
                    n=name,
                    b=jnp.sum(jnp.logical_not(jnp.isfinite(x))),
                    m=jnp.max(jnp.abs(jnp.nan_to_num(x))))
            _st("head_rewards", head_rewards)
            _st("traj.value", traj.value)
            _st("popart_mu", popart_mu)
            _st("popart_sigma", popart_sigma)
            _st("v_raw", v_raw)
            _st("estim_returns", estim_returns)
            _st("advantages", advantages)
            _st("norm_adv", norm_adv)

        # `old_*_dists` are the dynamic-mode equivalent of the legacy
        # old_{vertex,pair,factor}_dists fields — but TrainBatch only
        # carries the legacy ones plus the typed action sequence. The
        # dynamic loss path reads `traj.micro_op_dists` etc. directly via
        # the TrainBatch's micro_*_seq fields (the stored dists ARE the
        # old-policy snapshots since rollout_fn runs under stop-gradient
        # for batch construction). To keep TrainBatch flat we pack the
        # dynamic per-step dists into the same slots the legacy old_*
        # fields would occupy — when dynamic_substeps is on, the loss
        # path reads them via .pair_dists / .factor_dists slot reuse is
        # cleaner than threading another six fields through the whole
        # mini-batching pipeline. (TrainBatch's micro_*_seq fields below
        # carry the actions; we add micro_*_dists alongside them so the
        # old log-prob computation can index them.)
        # THE GRADIENT WINDOW (--grad-window K). The loss re-derives a step's
        # encoding by extending a STORED carry with a STORED delta, so
        # gradient reaches palimpsa through the last K deltas and stops at
        # the anchor carry. K = 1 selects step t's own anchor and its own
        # delta -- the previous behaviour, unrolled to the same single call.
        # Steps earlier than K-1 clamp the anchor to step 0 and mark the
        # missing entries with count 0, which `advance` runs as an exact
        # no-op (every row invalid), so the short prefix needs no special
        # case.
        # K = 0 is NOT a window: the loss scans the whole episode from the
        # base carry, so there is nothing to gather and nothing to anchor.
        # The (T, K) index materialises every delta K times and the anchor
        # fields are a per-step copy of the encoder state; both are dead here,
        # so the deltas are handed over as they were recorded and the anchor
        # slots carry a placeholder. TrainBatch's leaves must keep the leading
        # `(num_envs, T)` pair -- `shuffle_and_batch_by_trajectory` indexes the
        # env axis -- but nothing constrains what is under it.
        _T_steps = traj.delta_count.shape[1]
        _FULL_SCAN_EP = int(getattr(args, "grad_window", 1)) == 0
        if _FULL_SCAN_EP:
            _w_dtok = traj.delta_tokens
            _w_deqn = traj.delta_eqns
            _w_dcnt = traj.delta_count
            _w_down = traj.delta_owner
            _w_dpart = traj.delta_participants
            _dead = jnp.zeros(traj.delta_count.shape + (1,), jnp.float32)
            _w_encM = _w_encI = _w_ench = _w_encn = _w_encp = _dead
            _w_vs = _w_vc = _dead
        else:
            _wK = max(1, int(getattr(args, "grad_window", 1)))
            _w_t = (jnp.arange(_T_steps, dtype=jnp.int32)[:, None]
                    + (jnp.arange(_wK, dtype=jnp.int32) - (_wK - 1))[None, :])
            _w_idx = jnp.clip(_w_t, 0, _T_steps - 1)            # (T, K)
            _w_live = (_w_t >= 0)
            _w_anchor = _w_idx[:, 0]                            # (T,)
            _w_dtok = traj.delta_tokens[:, _w_idx]
            _w_deqn = traj.delta_eqns[:, _w_idx]
            _w_dcnt = jnp.where(_w_live[None], traj.delta_count[:, _w_idx], 0)
            _w_down = traj.delta_owner[:, _w_idx]
            _w_dpart = jnp.where(_w_live[None, ..., None],
                                 traj.delta_participants[:, _w_idx], 0.0)
            _w_encM = traj.enc_M[:, _w_anchor]
            _w_encI = traj.enc_I[:, _w_anchor]
            _w_ench = traj.enc_cumhist[:, _w_anchor]
            _w_encn = traj.enc_nvalid[:, _w_anchor]
            _w_encp = traj.enc_pos[:, _w_anchor]
            _w_vs = traj.vmem_sums[:, _w_anchor]
            _w_vc = traj.vmem_counts[:, _w_anchor]
        full_batch = TrainBatch(
            preference=traj.preference,
            vertex_idx=traj.vertex_idx,
            pair_seq=traj.pair_seq,
            factor_seq=traj.factor_seq,
            micro_op_seq=traj.micro_op_seq,
            micro_i_seq=traj.micro_i_seq,
            micro_j_seq=traj.micro_j_seq,
            micro_exp_seq=traj.micro_exp_seq,
            micro_factor_seq=traj.micro_factor_seq,
            micro_compress_kind_seq=traj.micro_compress_kind_seq,
            micro_quant_dtype_seq=traj.micro_quant_dtype_seq,
            micro_quant_scale_sign_seq=traj.micro_quant_scale_sign_seq,
            micro_quant_scale_frac_seq=traj.micro_quant_scale_frac_seq,
            old_vertex_dist=traj.vertex_dist,
            old_pair_dists=traj.pair_dists,
            old_factor_dists=traj.factor_dists,
            old_micro_op_dists=traj.micro_op_dists,
            old_micro_i_dists=traj.micro_i_dists,
            old_micro_j_dists=traj.micro_j_dists,
            old_micro_exp_dists=traj.micro_exp_dists,
            old_micro_kind_dists=traj.micro_kind_dists,
            old_micro_quant_logp=traj.micro_quant_logp,
            micro_pair_valid=traj.micro_pair_valid,
            micro_compress_valid=traj.micro_compress_valid,
            axis_state=traj.axis_state,
            axis_valid_mask=traj.axis_valid_mask,
            face_skip=traj.face_skip,
            face_op_type=traj.face_op_type,
            face_i=traj.face_i,
            face_j=traj.face_j,
            face_exponents=traj.face_exponents,
            face_factor=traj.face_factor,
            face_compress_kind=traj.face_compress_kind,
            face_quant_dtype=traj.face_quant_dtype,
            face_quant_scale_sign=traj.face_quant_scale_sign,
            face_quant_scale_frac=traj.face_quant_scale_frac,
            face_pair_valid=traj.face_pair_valid,
            face_comp_valid=traj.face_comp_valid,
            face_valid=traj.face_valid,
            face_endpoints=traj.face_endpoints,
            face_old_logp=traj.face_old_logp,
            face_counts=traj.face_counts,
            face_delta_tokens=traj.face_delta_tokens,
            face_delta_eqns=traj.face_delta_eqns,
            enc_M=_w_encM,
            enc_I=_w_encI,
            enc_cumhist=_w_ench,
            enc_nvalid=_w_encn,
            enc_pos=_w_encp,
            vmem_sums=_w_vs,
            vmem_counts=_w_vc,
            delta_tokens=_w_dtok,
            delta_eqns=_w_deqn,
            delta_count=_w_dcnt,
            delta_owner=_w_down,
            delta_participants=_w_dpart,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
            vertex_avail_mask=traj.vertex_avail_mask,
            # Threaded exactly like `delta_participants`: recorded per step in
            # the rollout, sliced by the same shuffle, read by the loss. All
            # four are None when the probe is off.
            probe_targets=traj.probe_targets,
            probe_extents=traj.probe_extents,
            probe_valid=traj.probe_valid,
            probe_step=traj.probe_step,
        )

        # The probes ride in the SAME scan carry (so their updates
        # accumulate across minibatches) but in their own slots, with their own
        # optimiser state. Both are None when the probe is off, and None is not
        # a pytree leaf, so the carry is byte-identical to before.
        dynamic_carry, static_carry = eqx.partition(
            (agent, opt_state, probes, probe_opt_state), eqx.is_array)

        # Stage D head LR warmup: thread a step counter through both scans
        # (epoch × minibatch) so the per-head LR multiplier ramps continuously
        # across episodes. The optimizer also has its own internal count
        # (cosine-decay schedule reads it) but we want a separate, explicit
        # ramp anchored at the start of this run.
        # ``shuffle_and_batch_by_trajectory`` slices ENVS into minibatches
        # (each mb keeps a whole trajectory together so the loss can
        # encode the residual jaxpr once per env). It's only usable when
        # there's at least one env per minibatch — otherwise the floor
        # division ``num_envs // minibatches`` returns 0 and every batch
        # is empty, producing ``jnp.mean(empty) = NaN`` for every loss
        # term. Fall back to the per-sample ``shuffle_and_batch`` in that
        # case; the loss path still cache-encodes per sample via the
        # ``elif args.cache_encoding`` branch in ``_dynamic_loss_fn`` /
        # ``loss_fn``.
        #
        # THE FULL-HORIZON PATH REQUIRES IT: `lax.scan` over an episode needs
        # that episode's steps, in order, in one minibatch. Shuffling steps
        # across envs (the default) would hand the scan an arbitrary
        # permutation of unrelated deltas. `main` has already clamped
        # `--minibatches` to `--num-envs` so no minibatch is empty.
        use_traj_batch = _FULL_SCAN_EP

        def epoch_step_fn(carry_with_step, epoch_key):
            carry, step = carry_with_step
            batches = (
                shuffle_and_batch_by_trajectory(full_batch, args.minibatches, epoch_key)
                if use_traj_batch
                else shuffle_and_batch(full_batch, args.minibatches, epoch_key)
            )
            mb_keys = jrand.split(epoch_key, args.minibatches)

            def mb_step_fn(c_with_step, batch_and_key):
                c, step = c_with_step
                (comb_agent, comb_opt_state, comb_probes,
                 comb_probe_opt) = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                grads, _aux = eqx.filter_grad(loss_fn, has_aux=True)(
                    comb_agent,
                    batch,
                    t_key,
                    pin_rules_to_exact_arg,
                    op_legality_override_arg,
                )
                # `has_aux` carries the probe's read-only view of the
                # representation alongside the metrics. filter_grad does not
                # differentiate aux, so this is a second, independent reason
                # the probe cannot appear in `grads`.
                if _PROBE_ON:
                    metrics, _probe_pack = _aux
                else:
                    metrics = _aux
                # Single fused per-leaf gradient scaling: combines the
                # Stage D head-LR ramp and the Stage G freeze mask. With an
                # all-True ``freeze_mask`` (regular training) the freeze
                # branch is a no-op; with the cal-mask the LR ramp is also
                # active for the trainable params (factor head, aggregator).
                grads = _scale_grads(
                    grads,
                    axis_mask,
                    factor_mask,
                    vertex_mask,
                    micro_mask,
                    freeze_mask,
                    _head_lr_mult(step, args.axis_warmup_steps),
                    _head_lr_mult(step, args.factor_warmup_steps),
                    vertex_mult_arg,
                    micro_mult_arg,
                )
                # The other half of the NaN fork: a FINITE loss whose gradient
                # is NaN (sqrt/norm/abs differentiated at exactly 0). The loss
                # localizer above cannot see this — by the time it fires the
                # params are already poisoned. Reported as a count so a single
                # bad leaf is visible against thousands of good ones.
                if _DEBUG_NAN:
                    # Name the leaves, not just count them. "190 leaves" says
                    # the poison has already spread; the SUBMODULE it starts
                    # in is what localises the trap, and a NaN gradient under
                    # a finite loss is always a specific op differentiated at
                    # a point its forward value hides (0*inf under jnp.where,
                    # sqrt/abs at exactly 0).
                    _gpl = [(jax.tree_util.keystr(kp), g) for kp, g
                            in jax.tree_util.tree_flatten_with_path(grads)[0]
                            if eqx.is_array(g)]
                    _gbad = sum(jnp.sum(jnp.logical_not(jnp.isfinite(g)))
                                for _p, g in _gpl)
                    for _pth, _g in _gpl:
                        _n = jnp.sum(jnp.logical_not(jnp.isfinite(_g)))
                        jax.lax.cond(
                            _n > 0,
                            lambda _n=_n, _pth=_pth: jax.debug.print(
                                "[nan]   leaf {p}: {n}", p=_pth, n=_n),
                            lambda: None,
                        )
                    jax.lax.cond(
                        _gbad > 0,
                        lambda: jax.debug.print(
                            "[nan] GRADIENT non-finite: {n} entries across "
                            "{k} leaves (loss itself was finite)",
                            n=_gbad, k=len(_gpl)),
                        lambda: None,
                    )
                updates, new_opt_state = optimizer.update(
                    grads, comb_opt_state, comb_agent
                )
                new_agent = eqx.apply_updates(comb_agent, updates)
                # THE PROBE'S OWN GRADIENT, ITS OWN CHAIN, ITS OWN TREE. The
                # agent is not an argument of `_probe_loss`, so no cotangent of
                # this scalar can reach a policy parameter -- and the probe
                # scalar is never added to `total_loss`, so the PPO gradient
                # above never saw a probe weight either.
                if _PROBE_ON:
                    _pgrads, _pmet = eqx.filter_grad(
                        _probe_loss, has_aux=True)(comb_probes, _probe_pack)
                    _pupd, new_probe_opt = probe_optimizer.update(
                        _pgrads, comb_probe_opt, comb_probes)
                    new_probes = eqx.apply_updates(comb_probes, _pupd)
                else:
                    new_probes = new_probe_opt = _pmet = None
                next_carry, _ = eqx.partition(
                    (new_agent, new_opt_state, new_probes, new_probe_opt),
                    eqx.is_array)
                return (next_carry, step + 1), (
                    (metrics, _pmet) if _PROBE_ON else metrics)

            return lax.scan(mb_step_fn, (carry, step), (batches, mb_keys))

        epoch_keys = jrand.split(subkey, args.ppo_epochs)
        # prof/postrollout: GAE + PopArt + the TrainBatch assembly.
        dynamic_carry, full_batch = _pp_mark(
            "prof/postrollout", (dynamic_carry, full_batch))
        (dynamic_carry, final_step), metrics_seq = lax.scan(
            epoch_step_fn,
            (dynamic_carry, global_step),
            epoch_keys,
        )
        # prof/update: ppo_epochs x minibatches of grad + optimizer step.
        dynamic_carry, metrics_seq = _pp_mark(
            "prof/update", (dynamic_carry, metrics_seq))

        (agent, opt_state, probes,
         probe_opt_state) = eqx.combine(dynamic_carry, static_carry)
        if _PROBE_ON:
            metrics_seq, _probe_seq = metrics_seq
            (_fp, _ft, _fv, _fs, _vp, _vt, _vv, _vs,
             _plf, _plv) = _probe_seq
            # ONE R2 per episode, over the FULL batch: the LAST epoch's
            # predictions (the most-trained probe), all minibatches
            # concatenated back into the whole episode. See the note in
            # _probe_loss for why per-minibatch R2 was wrong.
            def _last_epoch_flat(x):
                return x[-1].reshape((-1,) + tuple(x.shape[3:]))
            _r2 = _fprobe.within_step_r2(
                _last_epoch_flat(_fp), _last_epoch_flat(_ft),
                _last_epoch_flat(_fv),
                _last_epoch_flat(_fs).astype(jnp.int32), _PROBE_NSTEPS)
            _v_r2 = _fprobe.within_step_r2(
                _last_epoch_flat(_vp), _last_epoch_flat(_vt),
                _last_epoch_flat(_vv),
                _last_epoch_flat(_vs).astype(jnp.int32), _PROBE_NSTEPS)
            # The two losses keep the PPO metrics' reduction: mean over both
            # scan axes.
            probe_metrics = (_r2, _v_r2, jnp.mean(_plf), jnp.mean(_plv))
        else:
            probe_metrics = None
        # metrics_seq leaves have a leading (ppo_epochs, minibatches) pair.
        # Reduce by mean over those two scan axes only — scalars become
        # scalars, and the per-component KL slot (a (5,) array per step)
        # stays a (5,) array instead of being globally scalarized.
        metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=(0, 1)),
            metrics_seq,
        )
        # Legacy slots are zero-filled in --dynamic-substeps mode; the
        # micro_* slots carry the real typed action there. Pack both so
        # `host_log` can pick the right formatter.
        actions_pack = (
            traj.vertex_idx,
            traj.pair_seq,
            traj.factor_seq,
            traj.micro_op_seq,
            traj.micro_i_seq,
            traj.micro_j_seq,
            traj.micro_factor_seq,
            traj.micro_compress_kind_seq,
            traj.micro_quant_dtype_seq,
            # Final env-state face wires — the ONLY record of per-face
            # approximations (Trajectory carries none). Without these the
            # pareto archive stored unreplayable fronts (v43 post-mortem:
            # calls=[] on every entry while cos<<1).
            env_states.face_specs,
            env_states.face_skips,
        )
        # Pair / factor / preference marginals — Stage D / E / F diagnostics.
        # Average over (env, time, slot) — broad enough to detect global
        # collapse without exposing per-step noise.
        # P(END | t=0) averaged over (env, time) — the slot-0 END
        # probability is the highest-signal debugging metric: a spike
        # flags entropy collapse or reward-noise dominance. In the
        # legacy path "END at t=0" maps to slot-0 PAIR_STOP; in the
        # dynamic path it's the op_type head's OP_END mass at sub-step 0.
        if args.dynamic_substeps:
            p_stop_slot0 = jnp.mean(traj.micro_op_dists[..., 0, OP_END])
            # Mean op-type marginals over (env, time, sub-step). Useful
            # for spotting collapse in a specific op-type (e.g. policy
            # only emits END, never DIAG).
            op_marginals = jnp.mean(traj.micro_op_dists, axis=(0, 1, 2))
            # Mean sub-episode length: number of "active" sub-steps
            # before the first OP_END (inclusive of the END action itself).
            # Bounded by `max_substeps`. Tracking this catches the two
            # common failure modes: collapse to length-1 (always END) or
            # collapse to length=max_substeps (never END, gets truncated).
            is_end_seq = traj.micro_op_seq == OP_END
            prior_ends = jnp.cumsum(
                is_end_seq.astype(jnp.int32), axis=-1
            ) - is_end_seq.astype(jnp.int32)
            active_steps_per_rollout = (prior_ends == 0).astype(jnp.float32)
            mean_sub_episode_length = jnp.mean(
                jnp.sum(active_steps_per_rollout, axis=-1)
            )
        else:
            p_stop_slot0 = jnp.mean(traj.pair_dists[..., 0, PAIR_STOP])
            op_marginals = jnp.zeros((NUM_OPS,), dtype=jnp.float32)
            mean_sub_episode_length = jnp.array(0.0, dtype=jnp.float32)
        # REALIZED per-face approximation usage (the campaign's actual action
        # space under --live-faces, where the micro head is None and every
        # decision comes from the UnifiedFaceHead).
        #   face_skip  (E, T, F)      1 = face dropped entirely (SKIP_FACE)
        #   face_op    (E, T, F, S)   op per slot; OP_END(3) = "no approx here"
        # Padding faces carry skip=0 / op=OP_END, so the means below are over
        # all slots including padding — comparable across steps, and the
        # applied/skipped counters give the absolute reality check.
        # DENOMINATOR: valid faces only. F is the provable per-graph face
        # bound (196 here) while a vertex really has ~1.7 faces, so a plain
        # mean over the padded arrays reports 1 - mean_nf/F no matter what
        # the policy does. That is exactly what the old metric showed:
        # none 0.9919 measured vs 0.99148 predicted from padding alone.
        _fv = getattr(traj, "face_valid", None)
        if _fv is not None:
            _fv = _fv.astype(jnp.float32)
            _fv_n = jnp.maximum(jnp.sum(_fv), 1.0)
        if getattr(traj, "face_skip", None) is not None:
            _fsk = traj.face_skip.astype(jnp.float32)
            if _fv is not None:
                _face_skip_p = jnp.sum(_fsk * _fv) / _fv_n
            else:
                _face_skip_p = jnp.mean(_fsk)
        else:
            _face_skip_p = jnp.array(0.0, dtype=jnp.float32)
        _fop = getattr(traj, "face_op_type", None)
        if _fop is not None:
            _fop = _fop.astype(jnp.int32)
            if _fv is not None:
                # face_op_type is (..., F, S): broadcast validity over slots,
                # so the denominator is (valid faces) x (slots per face).
                #
                # WANDB AUDIT (2026-08-07) -- SKIP IS EXCLUSIVE. A SKIPPED face
                # still carries whatever op the head emitted for its slots, and
                # those slots used to be counted into the diag/compress/quant/
                # none classes even though the face was dropped before any
                # approximation could apply. az_gumbel counts a skipped face as
                # `skip` and nothing else, so the same key meant "P(op | face,
                # including dropped faces)" on one arm and "realized class"
                # on the other.
                #
                # Both arms now report the SAME partition of the SAME
                # denominator: one count per (valid face, slot), a skipped
                # face contributing all of its slots to `skip`. The five
                # classes {skip, none, diag, compress, quant} are mutually
                # exclusive and sum to 1 on both trainers.
                _fk = getattr(traj, "face_skip", None)
                if _fk is not None:
                    _keep = (1.0 - _fk.astype(jnp.float32))[..., None]
                else:
                    _keep = jnp.asarray(1.0, jnp.float32)
                _w = _fv[..., None] * _keep
                _wn = _fv_n * float(_fop.shape[-1])
                _face_op_freq = jnp.stack([
                    jnp.sum((_fop == k).astype(jnp.float32) * _w) / _wn
                    for k in range(4)
                ])
            else:
                _face_op_freq = jnp.stack([
                    jnp.mean((_fop == k).astype(jnp.float32))
                    for k in range(4)
                ])
        else:
            _face_op_freq = jnp.zeros((4,), dtype=jnp.float32)
        # Dilution readout: how loose the static width is versus reality.
        if _fv is not None:
            _n_steps = 1.0
            for _d in _fv.shape[:-1]:
                _n_steps *= float(_d)
            _face_mean_valid = jnp.sum(_fv) / max(_n_steps, 1.0)
        else:
            _face_mean_valid = jnp.array(0.0, dtype=jnp.float32)
        diag_pack = (
            jnp.mean(traj.pair_dists, axis=(0, 1, 2)),
            jnp.mean(traj.factor_dists, axis=(0, 1, 2)),
            jnp.mean(traj.preference, axis=(0, 1)),
            p_stop_slot0,
            op_marginals,
            mean_sub_episode_length,
            # T3, see the capture site above the PopArt update.
            _diag_value_raw,
            _diag_return_raw,
            _diag_nonzero_cos_steps,
            _face_skip_p,
            _face_op_freq,
            _face_mean_valid,
        )
        if _ATTN_ENTROPY_ON:
            # Reads the BASE buffer -- which is exactly what it read
            # before: `traj.tokens[0, 0]` was env 0 at step 0, i.e. the reset
            # state, i.e. the base stream inside a MAX_TOKENS-wide buffer.
            # Same tokens, same eqn ids, no padding tail, and no dependence
            # on a trajectory field that no longer exists.
            _attn_ent = attention_entropy_diagnostic(
                agent,
                _BASE_TOK,
                _BASE_EQN,
                traj.axis_state[0, 0],
                traj.axis_valid_mask[0, 0],
            )
        else:
            _attn_ent = jnp.asarray(float("nan"), jnp.float32)
        return (
            agent,
            opt_state,
            env_states,
            metrics,
            total_rewards_full,
            actions_pack,
            final_step,
            diag_pack,
            new_m1,
            new_m2,
            new_w,
            _attn_ent,
            true_scalar_return,
            probes,
            probe_opt_state,
            probe_metrics,
        )

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    # Reporting.
    # BEFORE any jit tracing: the transforms capture this as a constant.
    if getattr(args, "no_symlog", False):
        _NO_SYMLOG_ALL[0] = True
        if args.advantage_norm != "popart":
            print(
                "[warn] --no-symlog without --advantage-norm popart leaves the "
                "raw ~1e8 reward scale unnormalised; PopArt is what replaces "
                "symlog's magnitude compression.", flush=True)
        else:
            print("[cfg] symlog DISABLED; PopArt alone scales the channels.",
                  flush=True)

    _wandb_config = dict(vars(args))
    _wandb_config.update(_repo_commits())
    # WHICH quantity reward slot 6 holds, recorded on the run itself. The
    # per-channel keys (mean_quality, measure/quality/*, popart/mu_quality,
    # weighted_mean_quality) are deliberately metric-AGNOSTIC so two runs on
    # different metrics never silently share an axis; this is the key that
    # tells them apart. ``entropy/*`` panels are unaffected -- they describe
    # the policy heads, not the reward.
    _wandb_config["quality_metric_resolved"] = _QUALITY_METRIC
    wandb.init(
        project=getattr(args, "wandb_project", None) or "dsnn-vertex",
        entity=getattr(args, "wandb_entity", None) or None,
        name=args.name,
        config=_wandb_config,
        mode="disabled" if args.wandb == "disabled" else args.wandb,
    )
    # Pareto front over the three objectives the spec plots: compute cost,
    # memory, accuracy. All are stored "higher is better", matching the
    # archive's maximisation convention.
    # PopArt running stats, carried across episodes (identity under
    # --advantage-norm zscore).
    popart_m1 = jnp.zeros((NUM_VALUE_HEADS,), dtype=jnp.float32)
    popart_m2 = jnp.zeros((NUM_VALUE_HEADS,), dtype=jnp.float32)
    popart_w = jnp.zeros((NUM_VALUE_HEADS,), dtype=jnp.float32)

    from alphagrad.approx.common.pareto_archive import ParetoArchive
    pareto_archive = ParetoArchive(
        obj_names=(args.cmp_type, args.mem_type, "cosine_sim"),
        obj_idx=(cmp_idx, mem_idx, cosine_idx),
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])
    pbar = tqdm(total=args.episodes)

    host_state = {
        "samplecounts": 0,
        "best_global_return": -float("inf"),
        "best_global_act_seq": None,
        "top_n_total": [],
        "top_n_cmp": [],
        "top_n_mem": [],
        "top_n_acc": [],
        # WALL-CLOCK ORIGIN, stamped HERE and not at the first host_log call.
        # `_wall_t0` used to be set by the first logged episode, so
        # time/wall_seconds read exactly 0 on that row and the whole build +
        # first-trace cost (the single largest block of the run) fell outside
        # the axis. az_gumbel stamps `_t_start` at the same point in ITS setup
        # -- before the PopArt warm start -- so the shared time/* keys now
        # share an origin and time/sec_per_episode's first value is the honest
        # "setup + first episode" figure on both arms.
        "_wall_t0": _prof_time.perf_counter(),
    }

    def print_top_n(name, heap, reverse_val=True, log_to_wandb=True):
        print(f"\nTop {args.top_n} trajectories for {name}:")
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=reverse_val)
        weights = reward_weights_np
        table = None
        if log_to_wandb:
            table = wandb.Table(
                columns=[
                    "rank",
                    "episode",
                    "total_reward",
                    "cmp",
                    "acc",
                    "mem",
                    "sequence",
                ]
            )
        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            arr = np.array(rets)
            total_ret = float(np.sum(arr * weights))
            cmp_val = -float(arr[cmp_idx])  # display as positive cost
            mem_val = -float(arr[mem_idx])  # display as positive cost
            acc_val = float(arr[cosine_idx])  # quality channel, ~[0, 1]
            print(
                f"{rank}. Ep {ep} | Total Reward: {total_ret:.2e} | "
                f"CMP({args.cmp_type}): {cmp_val:.2e} | "
                f"Quality({_QUALITY_METRIC}): {acc_val:.4f} | "
                f"Mem({args.mem_type}): {mem_val:.2e}"
            )
            # ``seq`` is a list of (vertex, [callable_str, ...]) tuples; the
            # inner strings are already copy-pastable
            # ``diag(...)`` / ``compress(...)`` expressions, so render the
            # whole structure with the callable text un-quoted.
            seq_strs = []
            for v, calls in seq:
                joined = ", ".join(calls)
                seq_strs.append(f"({v}, [{joined}])")
            seq_repr = "[" + ", ".join(seq_strs) + "]"
            # THE PER-VERTEX CALLS ONLY. Under --face-actions the plan's
            # approximations live in the FACE wires (face_specs / face_skips),
            # which `_decode` does not see -- only `_decode_arch` (the pareto
            # archive) renders them. An empty "[]" here therefore means "no
            # PER-VERTEX micro-rule", NOT "no approximation": job 59045 ran at
            # applied_fraction 0.70-0.83 with every printed call list empty,
            # and reading those lines as approximation-free plans produced a
            # false P0 ("identical orders, different cosine"). Labelled, not
            # silently omitted.
            print(f"   Sequence (vertex, [per-vertex calls...]): {seq_repr}")
            if getattr(args, "face_actions", False):
                print("   (face-level approximations are NOT shown here — "
                      "see the pareto archive's `faces` field)")
            if table is not None:
                table.add_data(
                    rank, ep, total_ret, cmp_val, acc_val, mem_val, seq_repr
                )
        if table is not None:
            wandb.log({f"Top N {name}": table})

    def host_log(
        ep, all_rets, actions_pack, mean_r, mets, diag_pack=None,
        popart_stats=None, attn_entropy=None, warmup=False, true_return=None,
        probe_metrics=None):
        ep = int(ep)
        all_rets = np.array(all_rets)  # (num_envs, NUM_REWARDS)
        v_idx_arr = np.array(actions_pack[0])
        pair_arr = np.array(actions_pack[1])
        factor_arr = np.array(actions_pack[2])
        # Typed micro-action slots — populated only in --dynamic-substeps;
        # legacy mode passes zero-filled tensors. The decoder branch below
        # uses these whenever the dynamic policy is active.
        micro_op_arr = np.array(actions_pack[3])
        micro_i_arr = np.array(actions_pack[4])
        micro_j_arr = np.array(actions_pack[5])
        micro_factor_arr = np.array(actions_pack[6])
        micro_kind_arr = np.array(actions_pack[7])
        micro_quant_arr = np.array(actions_pack[8])
        face_specs_arr = (np.array(actions_pack[9])
                          if len(actions_pack) > 9 else None)
        face_skips_arr = (np.array(actions_pack[10])
                          if len(actions_pack) > 10 else None)

        def _decode(env_i):
            if args.dynamic_substeps:
                return _action_to_pylist_dynamic(
                    v_idx_arr[env_i],
                    micro_op_arr[env_i],
                    micro_i_arr[env_i],
                    micro_j_arr[env_i],
                    micro_factor_arr[env_i],
                    micro_kind_arr[env_i],
                    micro_quant_arr[env_i],
                    args.max_substeps,
                )
            raise RuntimeError(
                "the legacy (non-dynamic-substeps) action decoder was removed; "
                "--dynamic-substeps is the only supported mode"
            )

        def _decode_arch(env_i):
            """Archive form of _decode: seq + SPARSE per-slot face wires.
            Only the pareto archive consumes this (json-serializable dict);
            every other consumer keeps the plain (vertex, calls) list."""
            seq = _decode(env_i)
            if face_specs_arr is None or face_skips_arr is None:
                return seq
            fs, sk = face_specs_arr[env_i], face_skips_arr[env_i]
            faces = []
            for k in range(min(len(seq), int(fs.shape[0]))):
                rows_k, skip_k = fs[k], sk[k]
                live = [int(f) for f in range(int(rows_k.shape[0]))
                        if int(skip_k[f]) == 1
                        or bool((rows_k[f, :, 0] != -1).any())]
                if live:
                    faces.append({
                        "k": int(k),
                        "f": live,
                        "rows": [np.asarray(rows_k[f]).tolist()
                                 for f in live],
                        "skips": [int(skip_k[f]) for f in live],
                    })
            return {"seq": seq, "faces": faces} if faces else seq

        mean_r = np.atleast_1d(np.array(mean_r))
        # ---- LIVE (successfully measured) envs -------------------------------
        # WANDB AUDIT (2026-08-07). `mean_r` arrives as the mean over ALL
        # num_envs terminal reward vectors, sentinelled ones INCLUDED. A
        # sentinel is -1e10 on every cost channel, so ONE failed measurement in
        # 16 moves mean_latency_ns from ~-6e4 to ~-6e8: the panel plots the
        # collapse rate, not the latency. az_gumbel logs `mean_<channel>` for
        # the measurement it actually got (a failed measure `continue`s and
        # emits NO row at all), so under the shared key the two arms were
        # measuring different things.
        #
        # Recompute the mean over the LIVE envs only -- the same exact-sentinel
        # test `_is_degen` uses in the loss, so "live" means the same thing on
        # the reward panels as it does in the gradient. When nothing is live the
        # mean_*/mean_return/weighted_mean_* keys are DROPPED (see below) rather
        # than logged as -1e10: an absent key leaves a clean gap, matching both
        # az_gumbel and the warm-up branch's treatment of undefined keys.
        _sent_ch = np.asarray(COMPUTE_REWARD_INDICES, dtype=np.int64)
        if all_rets.ndim == 2 and all_rets.shape[1] > int(_sent_ch.max()):
            _live_env = ~np.all(
                all_rets[:, _sent_ch] <= (float(SENTINEL_COST) * 0.99), axis=-1)
        else:
            _live_env = np.ones((all_rets.shape[0],), dtype=bool)
        _any_live = bool(_live_env.any())
        if _any_live:
            mean_r = np.asarray(
                all_rets[_live_env].mean(axis=0), dtype=np.float64)

        host_state["samplecounts"] += num_envs * num_valid
        # mets is an 11-tuple: 9 scalars + two per-component arrays (KL is
        # (7,), entropy is (6,) -- slot 5 of the latter is the face head).
        # Slot 9 is per-component KL (vertex/op/i/j/exp in dynamic mode;
        # vertex/pair/factor/0/0 in legacy). Slot 10 is per-component
        # entropy (same slot layout; legacy = all zeros today).
        kl_div = float(mets[0])
        policy_entropy = float(mets[1])
        _fit_quality = float(mets[2])
        explained_var = float(mets[3])
        ppo_loss = float(mets[4])
        value_loss = float(mets[5])
        _entropy_loss = float(mets[6])
        total_loss = float(mets[7])
        _clipping_trigger_ratio = float(mets[8])
        kl_components = np.asarray(mets[9])
        entropy_components = np.asarray(mets[10])

        weights = reward_weights_np
        n_collapsed_this_ep = 0
        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
            # COLLAPSE GUARD (spec: "Don't allow collapsed values to count as
            # best"). A terminal reward row is degenerate when the plan zeroed
            # the computation (flops cost 0 / peak-memory cost 0), reported an
            # unmeasurably-fast latency while latency was actually measured,
            # or produced a near-zero cosine (destroyed Jacobian — after the
            # env fix a broken comparison also reads cosine 0). Such rows
            # still train (the reward fix handles that side) but must never
            # be crowned "best" or enter the top-N tables.
            # COSINE IS NOT A GATE (user-directed 2026-07-28). Requiring
            # cos > 1e-6 marked EVERY row collapsed from the moment the policy
            # stopped producing a non-trivial cosine (ep 25 of v17), which
            # silently emptied best_return, every measure/* panel, both Pareto
            # tables and all four top-N heaps for the entire run. Quality is
            # carried by frob in the reward; the collapse guard's job is only
            # to reject rows whose COSTS are degenerate (a zeroed computation
            # reports zero cost and would otherwise be crowned "best").
            collapsed = bool(
                rets[cmp_idx] >= 0.0
                or rets[mem_idx] >= 0.0
                or (measure_latency and rets[REWARD_INDEX["latency_ns"]] >= 0.0)
            )
            if collapsed:
                n_collapsed_this_ep += 1
                continue
            decoded = _decode(i)
            total_ret = float(np.sum(rets * weights))
            # Per-family heap keys: cmp uses the canonical compute index,
            # mem uses the canonical memory index, and acc tracks the quality
            # channel (loss_drop or cosine -- see env.quality_metric()).
            heaps_and_keys = [
                ("top_n_total", total_ret),
                ("top_n_cmp", float(rets[cmp_idx])),
                ("top_n_mem", float(rets[mem_idx])),
            ]
            for heap_name, key_val in heaps_and_keys:
                heap = host_state[heap_name]
                payload = (key_val, ep, list(rets), decoded)
                if len(heap) < args.top_n:
                    heapq.heappush(heap, payload)
                else:
                    heapq.heappushpop(heap, payload)

            acc_val = float(rets[cosine_idx])
            if args.capture_perfect_grads or acc_val < 0.999999:
                heap = host_state["top_n_acc"]
                payload = (acc_val, ep, list(rets), decoded)
                if len(heap) < args.top_n:
                    heapq.heappush(heap, payload)
                else:
                    heapq.heappushpop(heap, payload)

        host_state["collapsed_total"] = (
            host_state.get("collapsed_total", 0) + n_collapsed_this_ep
        )
        # best_global over NON-collapsed envs only. -inf when every env in the
        # episode collapsed (nothing eligible).
        weighted_sums = np.sum(all_rets * weights, axis=-1)
        eligible = np.array(
            [
                not (
                    all_rets[i][cmp_idx] >= 0.0
                    or all_rets[i][mem_idx] >= 0.0
                    or (
                        measure_latency
                        and all_rets[i][REWARD_INDEX["latency_ns"]] >= 0.0
                    )
                )
                for i in range(all_rets.shape[0])
            ],
            dtype=bool,
        )
        if eligible.any():
            masked = np.where(eligible, weighted_sums, -np.inf)
            best_idx = int(np.argmax(masked))
            best_ret = float(masked[best_idx])
            if best_ret > host_state["best_global_return"]:
                host_state["best_global_return"] = best_ret
                host_state["best_global_act_seq"] = _decode(best_idx)

        # #81/#96 ONE POLL PER host_log. Every actor-side `consume_*` POPS
        # its counter, so two independent merges would race: the first
        # would drain the actors and the second would read zeros. Merged
        # once here and read by BOTH the collapse/* entries in log_dict
        # below and the tokenization/* block further down. Best-effort:
        # no pool, a dead actor, or an actor too old to have the method
        # contributes nothing and never raises.
        _POOL_CS = {}
        try:
            from alphagrad.approx.common.measure_pool import (
                merge_pool_collapse_stats as _merge_cs)
            _POOL_CS = _merge_cs(getattr(env, "_remote_pool", None), {})
        except Exception:
            pass

        log_dict = {
            "best_return": host_state["best_global_return"],
            # TRUNCATED = refused by a resource limit (op cap / OOM) and
            # EXCLUDED from the gradient — "Time Limits in RL". ZERO_WORK =
            # computed nothing but was KEPT and punished by frob. They are
            # opposite treatments, so they get separate counters.
            # #96: these are module globals written inside the measurement
            # `_callback`, which runs in the ACTOR processes under
            # --ray-measure. `_POOL_CS` (merged once at the top of
            # host_log) carries the actors' share; without it the panels
            # read 0 while the collapse they count is happening.
            "collapse/truncated_this_ep": (
                consume_truncated_plan_count()
                + _POOL_CS.get("truncated", 0)),
            # UNTRACEABLE is a SUBSET of truncated: graphax could not build the
            # plan at all (the open canonical-output-order gap). Separated
            # because OOM scales with plan size and this scales with nothing we
            # control -- if it is a large fraction, the approx arm is sampling a
            # region the library cannot evaluate and the run is not comparable.
            "collapse/untraceable_this_ep": (
                consume_untraceable_plan_count()
                + _POOL_CS.get("untraceable", 0)),
            "collapse/zero_work_this_ep": (
                consume_zero_work_plan_count()
                + _POOL_CS.get("zero_work", 0)),
            "collapse/count_this_ep": n_collapsed_this_ep,
            "collapse/count_total": host_state["collapsed_total"],
            # DROPPED: collapse/fraction_this_ep (= count_this_ep / num_envs),
            # sample count (linear in the episode), and "KL divergence" --
            # that one summed the per-component micro KLs, which are point
            # masses under --live-faces, so it equalled kl/vertex AND
            # contained no face-head term at all. kl/approx is the real
            # joint (vertex+face) KL; kl/vertex is the component view.
            "entropy evolution": policy_entropy,
            "explained variance": explained_var,
            "ppo loss": ppo_loss,
            "value loss": value_loss,
            "total loss": total_loss,
        }
        # Per-component KL and entropy: slot semantics depend on the
        # trainer mode. In dynamic mode the components are vertex / op /
        # i / j / exp; in legacy mode they are vertex / pair / factor /
        # 0 / 0 (and entropy is currently all zeros in legacy — wiring it
        # would need an extra return-tuple from evaluate_action).
        if kl_components.shape[0] >= 5:
            if args.dynamic_substeps:
                names = ("vertex", "op", "i", "j", "exp")
            else:
                names = ("vertex", "pair", "factor", "_unused1", "_unused2")
            for j, nm in enumerate(names):
                log_dict[f"kl/{nm}"] = float(kl_components[j])
                log_dict[f"ent/{nm}"] = float(entropy_components[j])
            if args.dynamic_substeps:
                # Spec P2 entropy split: MACRO = the vertex-elimination
                # (pointer) head; MICRO = the approximation sub-episode heads
                # (op / i / j / exp) averaged. `entropy evolution` above is
                # the overall mean.
                log_dict["entropy/macro_vertex"] = float(entropy_components[0])
                # A2: cross-arm alias for the VERTEX-ELIMINATION head. az_gumbel
                # logs the entropy of the same distribution (the softmaxed
                # vertex logits over the legal set) under this exact key, so one
                # panel compares the two trainers' pointer heads directly.
                # entropy/macro_vertex and ent/vertex are kept for continuity.
                log_dict["entropy/ve_head"] = float(entropy_components[0])
                # entropy/micro_approx is GONE. It averaged the PER-VERTEX
                # MicroActionPolicy slots (1..4), which under --live-faces /
                # --no-approx-head belong to a head that is not constructed at
                # all -- a permanently-0 panel that read as a collapsed policy.
                # The per-slot numbers survive as ent/op, ent/i, ent/j, ent/exp.
                #
                # THE approximation-head entropy panel, arity-normalised.
                #
                # CAVEAT for the shared panel: PPO's entropy/approx_head is the
                # PER-FACE head (UnifiedFacePolicy -- one decision per live
                # face), while az_gumbel's key of the same name is the
                # PER-VERTEX head (MicroActionPolicy -- one sub-episode per
                # vertex). Both are divided by their OWN action arity, so the
                # two curves are comparable in SHAPE (trend, collapse, revival)
                # but NOT in absolute scale: they are entropies over different
                # action spaces with different alphabet sizes.
                if entropy_components.shape[0] > 5 and getattr(
                        args, "face_actions", False):
                    log_dict["entropy/approx_head"] = float(
                        entropy_components[5])
        # entropy/palimpsa -- the encoder's mean ATTENTION-ROW entropy. Logged
        # outside the dynamic-substeps branch because it is a property of the
        # ENCODER, not of any action head. NOT a policy entropy and NOT
        # comparable in units to entropy/ve_head or entropy/approx_head; see
        # attention_entropy_diagnostic.
        if attn_entropy is not None:
            _ae = float(np.asarray(attn_entropy))
            if np.isfinite(_ae):
                log_dict["entropy/palimpsa"] = _ae
        # NORMALISED companions, in [0,1] = fraction of that head's MAXIMUM
        # possible entropy. The raw nats are not comparable across heads: the
        # vertex head picks among ~total_v vertices (ln 13 = 2.56) while the
        # op head picks among 4 (ln 4 = 1.39), so macro sat at 1-1.6 and micro
        # at 0-0.3 because of ALPHABET SIZE, not because the micro heads were
        # more decided. Only these two get a normaliser: the i/j/exp heads
        # have a DYNAMIC alphabet (the legal dim/factor set depends on the
        # live tensor), so there is no static maximum to divide by and any
        # fixed constant would be wrong.
        #
        # WANDB AUDIT (2026-08-07): these two -- and the kl/ratio block below
        # -- used to be NESTED under `attn_entropy is not None` /
        # `np.isfinite(_ae)`, i.e. gated on an unrelated ENCODER probe.
        # `attention_entropy_diagnostic` returns NaN "when nothing applies",
        # and a CPU smoke with --set-pointer --live-faces hit exactly that:
        # entropy/palimpsa was absent AND it took entropy/macro_vertex_norm,
        # entropy/op_norm with it. Both blocks are functions of
        # `entropy_components` / `kl_components` alone, so a NaN from the probe
        # could also have taken ratio/max_log -- the one tail statistic that
        # catches the old/new log-prob mismatch a batch-averaged KL hides.
        if entropy_components.shape[0] >= 2:
            log_dict["entropy/macro_vertex_norm"] = float(
                entropy_components[0] / np.log(max(int(total_v), 2)))
            # entropy/op_norm only exists when the PER-VERTEX MicroActionPolicy
            # does. Under --live-faces / --no-approx-head that head is not
            # constructed, `ent_op` is a structural 0, and the panel would
            # plot a permanently collapsed op head. Same guard the other
            # micro-head panels (op_marginal/*, sub_episode_length) use.
            if not (bool(getattr(args, "live_faces", False))
                    or bool(getattr(args, "no_approx_head", False))):
                log_dict["entropy/op_norm"] = float(
                    entropy_components[1] / np.log(max(int(NUM_OPS), 2)))
        # T2: the unified head's live diagnostics. kl/approx is the KL on
        # the JOINT log-prob; ratio/max_log is the TAIL statistic that a
        # batch-averaged KL hides (this read +53.8 when the ratio was
        # broken, and must be ~0 at epoch 0).
        if kl_components.shape[0] > 6:
            log_dict["kl/approx"] = float(kl_components[5])
            log_dict["ratio/max_log"] = float(kl_components[6])
            log_dict["ratio/max"] = float(
                np.exp(np.clip(float(kl_components[6]), -700, 700)))
        # Measured-reward panels. Emitted ONLY when at least one env produced a
        # real measurement this episode (see the LIVE mask above) so the shared
        # `mean_<channel>` / `mean_return` keys carry the same quantity the
        # az_gumbel arm puts on them: a MEASUREMENT, never a sentinel.
        if _any_live:
            log_dict["mean_return"] = float(np.sum(mean_r * weights))
        # THE TRUE OPTIMIZED SCALAR (see train_episode). Emitted every
        # trained episode, live measurement or not -- it is a training-
        # space quantity, not a measurement.
        if true_return is not None:
            log_dict["scalarized_return"] = float(true_return)
            for j, name in enumerate(REWARD_NAMES):
                log_dict[f"mean_{name}"] = (
                    float(mean_r[j]) if j < len(mean_r) else 0.0)

        # ---- per-channel measurement stats (spec P2) ------------------------
        # best / mean / median / worst per channel, for THIS episode and
        # ALL-TIME, computed over the collapse-guarded (eligible) envs only so
        # a zero-compute plan can never own a "best". Channels are stored
        # "higher is better" (costs negated), so best = max, worst = min.
        elig_rets = all_rets[eligible] if eligible.any() else all_rets[:0]
        alltime = host_state.setdefault("channel_alltime", {})
        _lean = bool(getattr(args, "lean_logging", False))
        for j, name in enumerate(REWARD_NAMES):
            if elig_rets.shape[0] == 0 or _lean:
                continue
            col = elig_rets[:, j].astype(np.float64)
            b, w = float(col.max()), float(col.min())
            log_dict[f"measure/{name}/best_ep"] = b
            log_dict[f"measure/{name}/mean_ep"] = float(col.mean())
            log_dict[f"measure/{name}/median_ep"] = float(np.median(col))
            log_dict[f"measure/{name}/worst_ep"] = w
            prev = alltime.get(name)
            if prev is None:
                alltime[name] = {"best": b, "worst": w, "sum": float(col.sum()),
                                 "n": int(col.size), "vals": [float(np.median(col))]}
            else:
                prev["best"] = max(prev["best"], b)
                prev["worst"] = min(prev["worst"], w)
                prev["sum"] += float(col.sum())
                prev["n"] += int(col.size)
                prev["vals"].append(float(np.median(col)))
            a = alltime[name]
            log_dict[f"measure/{name}/best_alltime"] = a["best"]
            log_dict[f"measure/{name}/worst_alltime"] = a["worst"]
            log_dict[f"measure/{name}/mean_alltime"] = a["sum"] / max(1, a["n"])
            log_dict[f"measure/{name}/median_alltime"] = float(np.median(a["vals"]))

        # ---- PopArt / normalization stats (spec P2) -------------------------
        # popart/mu_* and popart/sigma_* are the CRITIC's running normaliser
        # and are defined regardless of whether this episode measured anything,
        # so they are logged unconditionally.
        #
        # CROSS-ARM CAVEAT (audited 2026-08-07, deliberately NOT renamed):
        # ppo's PopArt tracks `estim_returns` -- the GAE-bootstrapped DISCOUNTED
        # return over `_symlog_rewards(reward)` -- while az_gumbel's tracks the
        # RAW TERMINAL 4-vector. Under the campaign config
        # (--terminal-rewards-only --no-symlog) the two coincide up to the
        # critic bootstrap and the discount; under any other config they do
        # not. The key names the same ROLE on both arms (the normaliser the
        # value head is rescaled by), which is why it keeps one name; the
        # SPACES are only equal in that configuration.
        if popart_stats is not None:
            _mu, _sig = popart_stats
            for j, nm in enumerate(HEAD_NAMES):
                log_dict[f"popart/mu_{nm}"] = float(_mu[j])
                log_dict[f"popart/sigma_{nm}"] = float(_sig[j])

            # WEIGHTED (PopArt-lens) RETURN — Charts/ companion to mean_return.
            #
            # `mean_return` is a sum over RAW channels, so it is dominated by
            # whichever channel has the largest magnitude (peak_memory in
            # bytes, ~1e8): it tracked -peak_memory almost exactly and told us
            # nothing about the objective the policy actually optimises.
            # This is the same episode mean seen THROUGH PopArt — each trained
            # head normalised by its own running (mu, sigma) before the
            # preference-weighted sum, i.e. the units the advantage is
            # actually computed in. Comparable across episodes and across
            # channels, so a move here is a real move in the objective.
            # SAME SPACE AS (mu, sigma). PopArt is updated from
            # `estim_returns`, which is built from `_symlog_rewards(reward)`;
            # this used to z-score the RAW channel value against those stats,
            # so with symlog on (the default) it compared ~-6e4 against a mu of
            # ~-11 and every Phi(z) pinned to 0 or 1 -- a constant panel. The
            # projection is the IDENTITY under --no-symlog, so the campaign
            # runs are unchanged, but the key now means the same thing in both
            # configurations (and the same thing az_gumbel means by it: the
            # running-distribution percentile of this episode's channel value
            # under the critic's own normaliser).
            _hr = np.asarray(
                _symlog_rewards(jnp.asarray(mean_r, dtype=jnp.float32)),
                dtype=np.float64,
            )[np.asarray(HEAD_REWARD_INDICES, dtype=np.int64)]
            _z = (_hr - np.asarray(_mu, np.float64)) / np.maximum(
                np.asarray(_sig, np.float64), 1e-8
            )
            _hw = np.asarray(head_reward_weights_np, np.float64)
            # NORMALISE to [0,1] through PopArt rather than reporting a raw
            # z-score. Phi(z) is the running-distribution percentile of this
            # episode's channel value: 0.5 = exactly average for the run,
            # ->1 = far better than the running mean, ->0 = far worse. Bounded,
            # comparable across channels and across episodes.
            from math import erf as _erf
            _phi = np.array([0.5 * (1.0 + _erf(float(v) / np.sqrt(2.0)))
                             for v in _z], dtype=np.float64)
            _wsum = float(np.sum(np.abs(_hw)))
            _wn = (np.abs(_hw) / _wsum) if _wsum > 0 else np.zeros_like(_hw)
            # convex combination of [0,1] values -> itself in [0,1]
            # No "Charts/" folder: it was a wandb section prefix only (no
            # define_metric, no in-repo panel refers to it) and it buried the
            # headline score under a group name.
            if _any_live:
                log_dict["weighted_mean_return"] = float(np.sum(_phi * _wn))
                for j, nm in enumerate(HEAD_NAMES):
                    if _hw[j] != 0.0:
                        log_dict[f"weighted_mean_{nm}"] = float(_phi[j])

        # ---- wall clock: lets the wandb x-axis be switched from episode to
        # elapsed time, so a slowdown shows up as a flat stretch instead of
        # being invisible against a uniform episode axis.
        _now = _prof_time.perf_counter()
        _t0 = host_state.setdefault("_wall_t0", _now)
        _prev = host_state.get("_wall_prev", _t0)
        log_dict["time/wall_seconds"] = float(_now - _t0)
        log_dict["time/wall_minutes"] = float((_now - _t0) / 60.0)
        log_dict["time/sec_per_episode"] = float(_now - _prev)
        log_dict["time/episode"] = int(ep)
        # Shared x-axis with the AZ arm. One AZ episode is ONE measurement;
        # one PPO episode is num_envs of them, so plotting the two against
        # "episode" understates PPO's cost by ~num_envs.
        log_dict["n_meas"] = int(num_envs) * (int(ep) + 1)
        host_state["_wall_prev"] = _now

        # ---- per-face apply telemetry ---------------------------------------
        if args.per_face or args.face_actions:
            pf = consume_per_face_stats()
            # The per-face legality hooks run INSIDE the Ray measure actors,
            # so the trainer's own counters are always empty when the pool is
            # active -- poll the actors and merge, or approx_applied/* never
            # appears at all. No-op (and never raises) without a pool.
            try:
                from alphagrad.approx.common.measure_pool import (
                    merge_pool_face_stats as _merge_pf)
                pf = _merge_pf(getattr(env, "_remote_pool", None), pf)
            except Exception:
                pass
            # UNCONDITIONAL. The `if applied or skipped` gate this replaces
            # meant that a flat 0 -- the ONE reading that proves the face wires
            # never reached the measurement -- was logged as an ABSENT key
            # rather than a zero, which is exactly the failure the panel
            # exists to catch. az_gumbel emits these keys every episode; ppo
            # now does too, so the shared panel has the same support.
            log_dict["per_face/applied"] = pf.get("applied", 0)
            log_dict["per_face/skipped"] = pf.get("skipped", 0)
            log_dict["per_face/skipped_raised"] = pf.get("skipped_raised", 0)
            log_dict["per_face/applied_fraction"] = pf.get(
                "applied_fraction", 0.0)
            # REALITY histogram per approximation class (same keys on the
            # AZ runs): what the policy proposed is not what survived the
            # per-face legality mask.
            for _k in ("diag", "compress", "quant"):
                log_dict[f"approx_applied/{_k}"] = pf.get(f"applied_{_k}", 0)
                log_dict[f"approx_skipped/{_k}"] = pf.get(f"skipped_{_k}", 0)
            log_dict["approx_applied/total"] = pf.get("applied", 0)
            log_dict["approx_skipped/total"] = (
                pf.get("skipped", 0) + pf.get("skipped_raised", 0))
            log_dict["approx_applied/fraction"] = pf.get(
                "applied_fraction", 0.0)

        # ---- degenerate plans sentinelled by the env ------------------------
        _degen = consume_degenerate_plan_count()
        # DROPPED: collapse/degenerate_plans_this_ep -- _record_degenerate_plan
        # has no caller, and _DEGENERATE_PLANS is only bumped by the
        # truncated/untraceable recorders (which also bump _TRUNCATED_PLANS),
        # making it arithmetically identical to collapse/truncated_this_ep.
        _ = _degen

        # ---- XLA side-channel: approx/exact compression ratio ---------------
        # No second memory NUMBER here: peak_memory is the one memory channel
        # (logged raw as mean_peak_memory) and the static memory_analysis()
        # estimate only ever substitutes into it.
        xla_stats = consume_memory_compression_stats()
        if xla_stats["count"]:
            log_dict["measure/compression_ratio"] = xla_stats["compression_ratio"]
        # How many measurements this period had peak_memory replaced by the
        # STATIC estimate (structural on CPU). 0 = every reading is a real
        # runtime high-water mark. TRAINER-LOCAL: with --ray-measure the
        # substitution happens inside the measure actors, whose one-time stdout
        # note still surfaces in the driver log.
        log_dict["measure/peak_memory_static_fallback"] = int(
            consume_static_peak_fallbacks())

        # ---- tokenization truncation (was computed but never logged) --------
        # Oracle probe failures. Non-zero means graphax could not trace some
        # vertex elimination, so the legality oracle admitted NO approximation
        # there and the policy was forced to eliminate it exactly. A large
        # count means the approx arm is not really approximating.
        try:
            from alphagrad.approx.common.masks import consume_probe_failure_stats
            _probe = consume_probe_failure_stats()
            log_dict["oracle/probe_failures"] = int(_probe["count"])
            log_dict["oracle/probe_failed_vertices"] = int(_probe["n_vertices"])
            if _probe["count"] and not host_state.get("_probe_fail_printed"):
                print(f"[oracle] probe failed (fail-soft, vertex forced exact): "
                      f"{_probe['last']}", flush=True)
                host_state["_probe_fail_printed"] = True
        except Exception:
            pass
        trunc = consume_tokenization_truncation_stats()
        # `max_observed_len` is a MAX across processes, not a sum --
        # adding them would report a length no single step ever had.
        log_dict["tokenization/truncated_count"] = (
            trunc["count"] + _POOL_CS.get("trunc_count", 0))
        log_dict["tokenization/max_observed_len"] = max(
            trunc["max_observed_len"],
            _POOL_CS.get("trunc_max_observed_len", 0))
        log_dict["tokenization/overflow_sum_this_ep"] = (
            trunc["overflow_sum"] + _POOL_CS.get("trunc_overflow_sum", 0))
        # Token SIZE telemetry (not just loss). `delta_*` is one palimpsa
        # call's width -- the ONLY observation the policy gets per step -- and
        # is what must size ALPHAGRAD_MAX_DELTA_TOKENS. `stream_*` is now just
        # the base length (logged once per episode by `base_observation`); the
        # growing full stream MAX_TOKENS used to size is gone.
        # WATCH delta_max: `env._delta_observation` CLIPS at
        # MAX_DELTA_TOKENS, and the clipped tail is DROPPED, not deferred to
        # the next step. tokenization/truncated_count counts exactly that.
        _tl = consume_token_length_stats()
        log_dict["tokens/stream_mean"] = _tl["stream_mean"]
        log_dict["tokens/stream_max"] = _tl["stream_max"]
        log_dict["tokens/delta_mean"] = _tl["delta_mean"]
        log_dict["tokens/delta_max"] = _tl["delta_max"]
        log_dict["tokens/delta_count"] = _tl["delta_count"]
        # DROPPED: tokens/delta_budget (a constant) and tokens/delta_headroom
        # (= budget - tokens/delta_max, affine in a key already logged).

        # ---- Pareto front + hypervolume (spec P2) ---------------------------
        # Objectives are logged in "higher is better" form, so the archive's
        # maximisation convention applies directly.
        if pareto_archive is not None and elig_rets.shape[0]:
            elig_idx = [i for i in range(all_rets.shape[0]) if eligible[i]]
            pareto_archive.add_many(
                ((all_rets[i], _decode_arch(i)) for i in elig_idx), ep
            )
            # CROSS-ARM CAVEAT (audited 2026-08-07). `pareto/archive_size`
            # is the live front's cardinality and means the same thing on
            # az_gumbel. `pareto/hypervolume` uses the same ParetoArchive and
            # the same sweep, but is NOT comparable in absolute value across
            # the two arms, for two reasons:
            #   * the OBJECTIVE SET here is (args.cmp_type, args.mem_type,
            #     cosine_sim) -- config-dependent -- while az hardcodes
            #     (latency_ns, peak_memory, cosine_sim). They coincide only
            #     under --cmp-type latency --mem-type peak_memory (the
            #     campaign config).
            #   * ParetoArchive freezes its HV nadir at `pts.min(axis=0) - 1`
            #     on the FIRST non-empty call. That call sees num_envs points
            #     here and exactly one on az, so the reference boxes differ
            #     and each run's HV is only monotone WITHIN itself.
            log_dict["pareto/hypervolume"] = float(pareto_archive.hypervolume())
            log_dict["pareto/archive_size"] = len(pareto_archive.pts)
            # Persist the FRONT, not just these two scalars — see _dump_pareto.
            _pd = int(getattr(args, "pareto_dump_every", 50) or 0)
            if _pd > 0 and (ep % _pd == 0):
                _dump_pareto(pareto_archive, args, ep)
            if pareto_archive.pts:
                fx = np.stack(pareto_archive.pts).astype(np.float64)
                # 3 scatter tables: (latency|cmp x cos), (mem x cos), (cmp x mem)
                for key, (a, b) in {
                    "pareto/cmp_vs_cos": (0, 2),
                    "pareto/mem_vs_cos": (1, 2),
                    "pareto/cmp_vs_mem": (0, 1),
                }.items():
                    # Use the episode each point was ADMITTED at, not the
                    # current one: stamping `ep` on every row made the whole
                    # front look re-measured every episode.
                    _peps = getattr(pareto_archive, "eps", None) or []
                    if _lean:
                        continue
                    tbl = wandb.Table(columns=["x", "y", "episode"])
                    for _i, row in enumerate(fx):
                        _e = int(_peps[_i]) if _i < len(_peps) else int(ep)
                        tbl.add_data(float(row[a]), float(row[b]), _e)
                    log_dict[key] = tbl

        # Stage D/E/F marginals — pair-index distribution (axis-pair head),
        # factor-index distribution (Stage E ρ-collapse early-warning), and
        # the per-episode preference vector (Stage F sanity check).
        if diag_pack is not None:
            (
                pair_marg,
                factor_marg,
                pref_mean,
                p_stop_slot0,
                op_marginals,
                mean_sub_episode_length,
                value_raw,
                return_raw,
                nonzero_cos_steps,
                _face_skip_p,
                _face_op_freq,
                _face_mean_valid,
            ) = (np.asarray(x) for x in diag_pack)
            # T3. nonzero_cos_steps > 1 means the --terminal-rewards-only gate
            # in env.py has stopped holding. The value/return pair measures the
            # critic overestimate that puts popart/mu_cos above the reward
            # ceiling: the gap is return_raw - value_raw, and return_raw_acc
            # above ~1.0 is the critic, not the env.
            log_dict["reward/nonzero_cos_steps"] = int(nonzero_cos_steps)
            for j, nm in enumerate(HEAD_NAMES):
                if j < value_raw.shape[0]:
                    log_dict[f"diag/value_raw_{nm}"] = float(value_raw[j])
                    log_dict[f"diag/estim_return_raw_{nm}"] = float(return_raw[j])
            # Per-vertex micro-head panels: only meaningful when that head
            # is actually constructed. Under --live-faces / --no-approx-head
            # the micro dists are point masses and every one of these is a
            # constant (see the module note above the guard below).
            _has_micro_head = not (bool(getattr(args, "live_faces", False))
                                   or bool(getattr(args, "no_approx_head",
                                                   False)))
            if _has_micro_head:
                for j, p in enumerate(pair_marg):
                    log_dict[f"pair_marginal/{j}"] = float(p)
                for j, p in enumerate(factor_marg):
                    log_dict[f"factor_marginal/{j}"] = float(p)
            # Preferences are STATIC for the run -> they belong in the run
            # config, not in a time series that plots a flat line. Written
            # once, on the first episode that has them.
            if not host_state.get("_pref_logged"):
                try:
                    wandb.config.update(
                        {f"preference_{nm}": float(pref_mean[j])
                         for j, nm in enumerate(HEAD_NAMES)},
                        allow_val_change=True,
                    )
                except Exception:
                    pass
                host_state["_pref_logged"] = True
            if _has_micro_head:
                # == 1.0 exactly when the micro head is absent (END one-hot).
                log_dict["p_stop_slot0"] = float(p_stop_slot0)
            # Dynamic-substeps op-type marginals. Names MUST match
            # heads.py's op order (DIAG=0, COMPRESS=1, QUANT=2, END=3) —
            # this used to label index 2 "end", so the plotted "end" curve
            # was really QUANT and the true END mass was never logged, which
            # hid exactly the END-collapse mode the diagnostic exists for.
            if _has_micro_head:
                # (0,0,0,1) when the micro head is absent. The LIVE
                # approximation usage lives in approx_prob/* below, which
                # reads the per-face head that actually decides.
                for j, op_name in enumerate(
                        ("diag", "compress", "quant", "end")):
                    if j < op_marginals.shape[0]:
                        log_dict[f"op_marginal/{op_name}"] = float(
                            op_marginals[j])
            # POLICY PROBABILITY per approximation class, unified naming with
            # the AZ runs. ``end`` (emit nothing further) is the "none" class;
            # ``skip`` is the per-face gate's mass, which lives on a separate
            # head and is 0 when face actions are off.
            # The MICRO head's marginals (absent under --live-faces) stay on
            # their own panel; approx_prob/* reports the head that actually
            # decided, i.e. realized per-face usage.
            # DEFINITION (identical on az_gumbel, see _face_choice_counts):
            # one count per (VALID face, slot); a SKIPPED face contributes all
            # of its slots to approx_prob/skip and none to the op classes, so
            # {skip, diag, compress, quant, none} partition the episode's
            # realized per-face-slot decisions and sum to 1.
            _ap_names = ("diag", "compress", "quant", "none")
            # DROPPED: micro_op_marginal/* -- the SAME op_marginals array as
            # op_marginal/*, only relabelling index 3 end -> none.
            for j, _nm in enumerate(_ap_names):
                if j < _face_op_freq.shape[0]:
                    log_dict[f"approx_prob/{_nm}"] = float(_face_op_freq[j])
            log_dict["approx_prob/skip"] = float(_face_skip_p)
            # The width/reality gap the corrected denominator removes. If
            # mean_valid stays ~1.7 against width 196, the per-face arrays
            # are ~99% padding and any UNMASKED face mean is meaningless.
            log_dict["faces/mean_valid"] = float(_face_mean_valid)
            # DROPPED: faces/width -- the ENV_MAX_FACES constant. The
            # dilution it existed to expose is carried by faces/mean_valid.
            if _has_micro_head:
                # == 1.0 exactly without a micro head (the sub-episode is a
                # single forced END step).
                log_dict["sub_episode_length"] = float(
                    mean_sub_episode_length)
        # Populate the elimination-order table (it used to be created and
        # logged empty). Bounded: one row per episode for the best eligible
        # env, so the table stays small over a 1000-episode run.
        if eligible.any():
            try:
                elim_order_table.add_data(
                    ep, float(masked[best_idx]), repr(_decode(best_idx))
                )
            except Exception:
                pass
        _HEALTH_N[0] += 1
        if _HEALTH_N[0] <= int(os.environ.get("ALPHAGRAD_HEALTH_EPISODES", "3")):
            # The launch check, on stdout where a running job can be read
            # without wandb. ratio/max_log must be ~0 at epoch 0 BY
            # CONSTRUCTION -- the stored old log-prob IS the sampling
            # log-prob -- so anything else means the loss is reconstructing
            # the behaviour policy differently from how it sampled, which is
            # the bug that ran the ratio to 2.3e23 and was invisible in a
            # batch-averaged KL. mu_cos above the reward ceiling means critic
            # overestimation. Non-finite entropy means a poisoned gradient.
            print("[health ep%d] ppo=%.4g value=%.4g ent=%.4g "
                  "ratio/max_log=%.3g kl/approx=%.3g mu_quality=%.4g "
                  "sec/ep=%.1f" % (
                      _HEALTH_N[0] - 1, ppo_loss, value_loss, policy_entropy,
                      log_dict.get("ratio/max_log", float("nan")),
                      log_dict.get("kl/approx", float("nan")),
                      log_dict.get("popart/mu_quality", float("nan")),
                      log_dict.get("time/sec_per_episode", float("nan"))),
                  flush=True)
            if _LIVE_FACES is not None:
                # A chunk that fails soft is EMPTY, and an empty chunk leaves
                # the palimpsa carry where it was -- i.e. the head decides on
                # the vertex context alone, exactly the blindness --live-faces
                # exists to remove, while every metric still looks healthy.
                # `truncated` is the same failure by a different route: the
                # window kept only the tail of the contraction.
                print("[health ep%d] live-faces %s" % (
                    _HEALTH_N[0] - 1, _LIVE_FACES.consume_stats()),
                    flush=True)
        # ALPHAGRAD_DEBUG_APPROX_PROB=1: mirror the approximation telemetry to
        # stdout, so a --wandb disabled probe (or a crashed run's log) still
        # answers "is skip/none ever chosen, or is it masked?".
        if os.environ.get("ALPHAGRAD_DEBUG_APPROX_PROB", "0") == "1":
            _ap = {k: v for k, v in log_dict.items()
                   if k.startswith(("approx_prob/", "approx_applied/",
                                    "approx_skipped/", "per_face/"))}
            if _ap:
                print("[approx] " + " ".join(
                    f"{k.split('/')[-1]}={float(v):.4g}"
                    for k, v in sorted(_ap.items())), flush=True)
            # Same mirror for the three head entropies (az_gumbel prints the
            # identical line under the identical switch).
            _ek = {k: v for k, v in log_dict.items()
                   if k.startswith("entropy/")}
            if _ek:
                print("[entropy] " + " ".join(
                    f"{k.split('/')[-1]}={float(v):.4g}"
                    for k, v in sorted(_ek.items())), flush=True)
        if warmup:
            # POPART WARM-START episode. The rollout, the measurements and the
            # collapse counters are REAL and belong on their panels -- and the
            # wandb step counter must advance, so these episodes are not a
            # silent gap in every curve. But NO gradient step ran, so every key
            # derived from the loss is undefined. Drop them entirely rather
            # than logging NaN: wandb records NaN as a DATA POINT and it wrecks
            # the panel's y-range for the whole run, while an absent key leaves
            # a clean gap. (This is the same rule az_gumbel applies to "loss"
            # while its PopArt gate still holds.)
            _drop_exact = ("KL divergence", "entropy evolution",
                           "explained variance", "ppo loss", "value loss",
                           "total loss", "loss")
            for _k in list(log_dict):
                if _k in _drop_exact or _k.startswith(
                        ("kl/", "ent/", "entropy/", "ratio/")):
                    log_dict.pop(_k, None)
            log_dict["popart_init/warmup_episode"] = 1
            # The warm start is not spent from the measurement budget (az
            # says so explicitly and does not advance its `n_meas`), and every
            # warm-up row shares the outer `ep`, so both x-axis keys would
            # report a stack of identical values for measurements that were
            # never charged. Drop them: az's warm-up row carries neither.
            log_dict.pop("n_meas", None)
            log_dict.pop("time/episode", None)
        # FEATURE PROBE (ALPHAGRAD_FEATURE_PROBE=1). THE TWO CONTROLS ARE
        # THE HARNESS CHECK, not a bonus row: `stat_ln_i` / `stat_ln_j` are
        # recoverable from the endpoint vertex ids alone and scored 0.96-0.98
        # in EVERY offline arm. If they do not read ~0.97 here the plumbing is
        # broken and the five real numbers mean nothing -- so they are printed
        # on their own line every episode rather than left in a wandb panel.
        if probe_metrics is not None:
            _pr2, _pvr2, _plf, _plv = [np.asarray(_x) for _x in probe_metrics]
            for _i, _nm in enumerate(_fprobe.FACE_NAMES):
                log_dict[f"probe/r2_{_nm}"] = float(_pr2[_i])
                log_dict[f"probe/vertex_r2_{_nm}"] = float(_pvr2[_i])
            log_dict["probe/loss_face"] = float(_plf)
            log_dict["probe/loss_vertex"] = float(_plv)
            log_dict["probe/arm"] = _fprobe.PROBE_ARM
            _nctrl = len(_fprobe.FACE_TARGETS)
            print(
                "[probe] arm=%s | CONTROLS %s | targets %s | loss %.4g/%.4g"
                % (_fprobe.PROBE_ARM,
                   " ".join("%s=%.3f" % (_n, float(_pr2[_nctrl + _k]))
                            for _k, _n in enumerate(_fprobe.FACE_CONTROLS)),
                   " ".join("%s=%.3f" % (_n, float(_pr2[_k]))
                            for _k, _n in enumerate(_fprobe.FACE_TARGETS)),
                   float(_plf), float(_plv)),
                flush=True)
            # Host-side census, once per episode: how many faces each probe
            # callback saw, and whether any target build failed. A failure
            # used to be silently indistinguishable from "no faces".
            _hist = " ".join("%d:%d" % (_k, int(_c))
                             for _k, _c in enumerate(_PROBE_NF_HIST) if _c)
            _tf = int(_fprobe.TARGET_FAILS[0]) + int(_PROBE_FAILS[0])
            _cmsg = "[probe census] n_faces hist {%s}" % (_hist or "empty")
            if _tf:
                _cmsg += " | target-build FAILS %d (last: %s)" % (
                    _tf, _fprobe.LAST_TARGET_ERR[0] or "see FAILED lines")
            print(_cmsg, flush=True)
            _PROBE_NF_HIST[:] = 0
            _fprobe.TARGET_FAILS[0] = 0
            _PROBE_FAILS[0] = 0

        # PER-EPISODE JSONL SINK (ALPHAGRAD_UPDATE_JSONL, default off).
        # Written BEFORE wandb.log so a crashed/offline run still has it, and
        # append-only so a mid-run kill keeps every completed episode. Carries
        # what wandb cannot: the raw per-env terminal reward VECTORS (from
        # which best-so-far latency/memory is reconstructible even under
        # --lean-logging) and the ELIMINATION ORDERS themselves.
        if _UPD_JSONL:
            try:
                import json as _json
                _row = {"ep": int(ep), "warmup": bool(warmup)}
                for _k, _v in log_dict.items():
                    if isinstance(_v, bool):
                        _row[_k] = _v
                    elif isinstance(_v, (int, np.integer)):
                        _row[_k] = int(_v)
                    elif isinstance(_v, (float, np.floating)):
                        _row[_k] = float(_v)
                    elif isinstance(_v, str):
                        _row[_k] = _v
                    # wandb.Table / Image / anything else: skipped on purpose.
                _row["orders"] = np.asarray(v_idx_arr).tolist()
                _row["rets"] = np.asarray(all_rets, dtype=np.float64).tolist()
                _row["reward_names"] = list(REWARD_NAMES)
                _row["reward_weights"] = np.asarray(
                    reward_weights_np, dtype=np.float64).tolist()
                if true_return is not None:
                    _row["true_return"] = float(true_return)
                _row.update(_ADV_STATS)
                with open(_UPD_JSONL, "a") as _fh:
                    _fh.write(_json.dumps(_row) + "\n")
            except Exception as _je:
                print(f"[update-jsonl] write failed: {_je}", flush=True)
        wandb.log(log_dict)

        # Per-episode memory + JIT-cache diagnostic. Off by default; flip on
        # via ``ALPHAGRAD_DEBUG_MEM=1`` for empirical leak investigation
        # without recompiling. Captures:
        #   * process RSS from /proc/self/status (host RAM truth)
        #   * jax.live_arrays() count + total bytes + top-5 (shape, dtype)
        #     buckets — if a particular shape grows monotonically, that
        #     identifies the producer of the leak.
        #   * env.py LRU cache hits / misses / size — to confirm the cache
        #     is actually being hit and isn't thrashing.
        # Routed through ``tqdm.write`` so the progress bar pauses → prints
        # → redraws below cleanly, instead of a raw ``sys.stderr.write``
        # that the next ``\r`` bar repaint clobbers on a live terminal.
        # Total cost when enabled: one /proc read + one Python walk of
        # live arrays (~1-2 ms per episode in practice).
        # Experimental: brute-force flush JAX's C++ caches every K episodes.
        # ``jax.clear_caches()`` releases JIT cache + HLO compile cache +
        # backend executable registry. Useful as a leak-hunt probe — if a
        # leak lives in XLA's C++ state (not in our LRU's Python refs), this
        # call should reclaim it. The LRU's stored Executables remain valid
        # because we hold direct Python refs; only the backend's *internal*
        # registry of orphaned executables gets dropped.
        _clear_every = os.environ.get("ALPHAGRAD_CLEAR_JIT_CACHES_EVERY", "0")
        try:
            _clear_every = int(_clear_every)
        except ValueError:
            _clear_every = 0
        if _clear_every > 0 and (ep + 1) % _clear_every == 0:
            try:
                jax.clear_caches()
                # The cache flush drops the PYTHON refs; without a forced
                # collection the orphaned executables/cost-analysis dicts sit
                # in gen-2 until CPython gets around to them — on the leak-
                # prone measurement host that lag is real memory (the old
                # stack's facf622 fix paired clear_caches with gc for the
                # same reason).
                import gc as _gc

                _gc.collect()
                tqdm.write(
                    f"[experiment] jax.clear_caches() called at ep={ep}",
                    file=sys.stderr,
                )
            except Exception as _exc:
                tqdm.write(
                    f"[experiment] jax.clear_caches() failed: {_exc!r}",
                    file=sys.stderr,
                )

        if os.environ.get("ALPHAGRAD_DEBUG_MEM", "0") == "1":
            try:
                rss_kb = 0
                with open("/proc/self/status") as _f:
                    for _line in _f:
                        if _line.startswith("VmRSS:"):
                            rss_kb = int(_line.split()[1])
                            break
                live = jax.live_arrays()
                n_live = len(live)
                live_bytes = sum(a.nbytes for a in live)
                from collections import Counter as _Counter

                top_shapes = _Counter(
                    (a.shape, str(a.dtype)) for a in live
                ).most_common(5)
                shape_str = "; ".join(
                    f"{s}x{d}={c}" for (s, d), c in top_shapes
                )
                tqdm.write(
                    f"[mem ep={ep:3d}] rss={rss_kb / 1024:7.0f}MB  "
                    f"live={n_live:5d} arrays {live_bytes / 1024 / 1024:7.0f}MB  "
                    f"top={shape_str}",
                    file=sys.stderr,
                )
            except Exception as _exc:
                tqdm.write(
                    f"[mem ep={ep}] probe failed: {_exc!r}",
                    file=sys.stderr,
                )

        # Host-phase profile: where the episode's HOST seconds went (callback
        # phases from env.py + the policy-side oracle replays). The residual
        # vs the episode wall-clock is jit compute (rollout + PPO update) —
        # everything the host timers can't see.
        if os.environ.get("ALPHAGRAD_PROFILE", "0") == "1":
            try:
                _prof = _consume_profile()
                if _prof:
                    _items = sorted(
                        _prof.items(), key=lambda kv: -kv[1]
                    )
                    _tot = sum(v for _, v in _items)
                    # Prefix-cache engagement counters (consumed per episode,
                    # like the phase timers): nonzero ext == the O(T) fast
                    # paths are actually running, cold == full prefix replays.
                    try:
                        from alphagrad.approx import env as _envmod
                        _ss = _envmod._INCR_STREAM_STATS
                        _fs = _envmod._FACE_ENUM_STATS
                        _cache_line = (
                            f"  stream(hit/ext/cold/nostore)={_ss['hit']}/"
                            f"{_ss['ext']}/{_ss['cold']}/{_ss['nostore']}"
                            f"  face_enum(ext/cold/compress)="
                            f"{_fs['ext']}/{_fs['cold']}/"
                            f"{_fs['compress']}"
                            f"  face_enum(calls/elims/build)="
                            f"{_fs['calls']}/{_fs['elims']}/"
                            f"{_fs['build']}"
                            f"  live_chain="
                            f"{_envmod.consume_live_chain_stats()}")
                        _ss.update(hit=0, ext=0, cold=0, nostore=0)
                        _fs.update(ext=0, cold=0, compress=0, elims=0,
                                   calls=0, build=0)
                    except Exception:
                        _cache_line = ""
                    tqdm.write(
                        f"[prof ep={ep:3d}] host_total={_tot:6.1f}s  "
                        + "  ".join(f"{k}={v:.1f}s" for k, v in _items)
                        + _cache_line,
                        file=sys.stderr,
                    )
                # Phase-0 per-decision attribution + the size distributions
                # that size the bucket grids. Both are separately gated
                # (ALPHAGRAD_PROFILE_POLICY / ALPHAGRAD_PROFILE_DIST) and
                # empty when off.
                try:
                    from alphagrad.approx.env import (
                        consume_profile_samples as _cps,
                        consume_distributions as _cds,
                    )
                    _samp = _cps()
                    if _samp:
                        tqdm.write(
                            f"[ppdec ep={ep:3d}]\n  " + _pp_summary(_samp),
                            file=sys.stderr)
                    # EVENT TRACE (ALPHAGRAD_PROFILE_TRACE=1 +
                    # ALPHAGRAD_PROFILE_TRACE_FILE=path): the flat (t, label)
                    # log that establishes which timed phase CONTAINS which.
                    # Appended per episode; never read back by the trainer.
                    _tf = os.environ.get("ALPHAGRAD_PROFILE_TRACE_FILE", "")
                    if _tf:
                        from alphagrad.approx.env import consume_trace as _ct
                        _evs = _ct()
                        if _evs:
                            with open(_tf, "a") as _fh:
                                for _t, _lab in _evs:
                                    _fh.write(f"{ep}\t{_t:.9f}\t{_lab}\n")
                    _dst = _cds()
                    if _dst:
                        _caps = {
                            "faces_per_vertex": int(ENV_MAX_FACES),
                            "face_chunk_len": int(MAX_DELTA_TOKENS),
                            "delta_len": int(MAX_DELTA_TOKENS),
                        }
                        tqdm.write(
                            f"[ppdist ep={ep:3d}]\n  "
                            + _pp_dist_summary(_dst, _caps),
                            file=sys.stderr)
                        _fpv = _dst.get("faces_per_vertex")
                        _fcl = _dst.get("face_chunk_len")
                        if _fpv and _fcl:
                            _real = float(np.mean(_fpv)) * float(np.mean(_fcl))
                            _pad = float(ENV_MAX_FACES) * float(
                                MAX_DELTA_TOKENS)
                            tqdm.write(
                                f"[ppdist ep={ep:3d}] live-face buffer "
                                f"occupancy: mean_faces="
                                f"{np.mean(_fpv):.2f} x mean_chunk="
                                f"{np.mean(_fcl):.1f} = {_real:.0f} real "
                                f"slots vs {_pad:.0f} allocated "
                                f"({100.0 * _real / _pad:.4f}%)",
                                file=sys.stderr)
                except Exception as _exc:
                    tqdm.write(f"[ppdec ep={ep}] failed: {_exc!r}",
                               file=sys.stderr)
            except Exception as _exc:
                tqdm.write(f"[prof ep={ep}] failed: {_exc!r}", file=sys.stderr)

        # Per-episode tracemalloc diff. Top-K Python lines by allocated
        # bytes since the previous snapshot. ``host_state["_tm_prev"]``
        # caches the prior snapshot. With ALPHAGRAD_TRACEMALLOC=1 +
        # ALPHAGRAD_DEBUG_MEM=1 together, both probes fire.
        if os.environ.get("ALPHAGRAD_TRACEMALLOC", "0") == "1":
            try:
                import tracemalloc as _tm

                _snap = _tm.take_snapshot()
                _prev = host_state.get("_tm_prev")
                if _prev is not None:
                    _diff = _snap.compare_to(_prev, "lineno")
                    _top = _diff[:10]
                    _lines = []
                    for s in _top:
                        f = s.traceback[0]
                        _lines.append(
                            f"{f.filename.split('/')[-1]}:{f.lineno} "
                            f"+{s.size_diff / 1024 / 1024:.1f}MB "
                            f"+{s.count_diff} allocs"
                        )
                    tqdm.write(
                        f"[tm ep={ep:3d}] top-10 Δalloc since last ep:\n  "
                        + "\n  ".join(_lines),
                        file=sys.stderr,
                    )
                host_state["_tm_prev"] = _snap
            except Exception as _exc:
                tqdm.write(
                    f"[tm ep={ep}] probe failed: {_exc!r}",
                    file=sys.stderr,
                )

        pbar.update(1)
        # The progress bar shows the best ELIGIBLE env; when every env in the
        # episode collapsed there is no eligible row (best_idx is unset), so
        # fall back to the raw argmax purely for display. (Found by the
        # end-to-end smoke: episode 0 collapsed in all envs and this raised
        # UnboundLocalError.)
        _disp_idx = best_idx if eligible.any() else int(np.argmax(weighted_sums))
        b_ret_unnorm = np.abs(all_rets[_disp_idx])
        means_str = ", ".join(f"{float(x):.2e}" for x in np.abs(mean_r))
        b_ret_desc = ", ".join(f"{float(x):.2e}" for x in b_ret_unnorm)
        pbar.set_description(
            f"ent:{policy_entropy:.3f} best:{b_ret_desc} means:{means_str}"
        )

    # Training loop.
    # Stage D global step counter — increments by ppo_epochs * minibatches per
    # episode and feeds the per-head LR ramp from §3.2.
    global_step = jnp.array(0, dtype=jnp.int32)


    # Stage F: per-env preference sampling over the 3-head simplex (flops /
    # peak_memory / frob_residual). Uses a Dirichlet with the configured
    # concentration; values < 1 emphasise corners and edges. When
    # --preference-conditioned is off we broadcast the static
    # `head_reward_weights` so all downstream code sees a consistent
    # `(num_envs, NUM_VALUE_HEADS)` shape.
    static_pref = jnp.broadcast_to(head_reward_weights, (num_envs, NUM_VALUE_HEADS))

    # JAX profiler hook. ``ALPHAGRAD_JAX_TRACE_DIR=/path`` enables a
    # per-episode trace: starts on the episode index given by
    # ``ALPHAGRAD_JAX_TRACE_EP`` (default 2 — first ep after the cold
    # JIT compile so the trace doesn't drown in compile noise) and stops
    # on the next episode boundary. Output is a TensorBoard-compatible
    # ``plugins/profile/.../`` tree the user opens in
    # ``tensorboard --logdir`` or via Perfetto (open
    # ``trace.json.gz`` directly).
    _jax_trace_dir = os.environ.get("ALPHAGRAD_JAX_TRACE_DIR", "")
    try:
        _jax_trace_ep = int(os.environ.get("ALPHAGRAD_JAX_TRACE_EP", "2"))
    except ValueError:
        _jax_trace_ep = 2
    _jax_trace_active = False

    # DEVICE-LEVEL TRACE (ALPHAGRAD_XLA_TRACE_DIR=<dir> plus
    # ALPHAGRAD_XLA_TRACE_EPS=<comma-separated episode indices>). Wraps the
    # whole `train_episode` call in `jax.profiler.trace`, which records the
    # XLA op timeline as the device actually ran it. The `_pp_mark` timers
    # can only attribute HOST spans between two callbacks; they cannot say
    # which HLO op inside `env.step` costs what, and the last two host
    # attributions were both wrong until the mark anchor was fixed. Off by
    # default; when off this is one bool test per episode and zero HLO.
    #
    # DOES NOT WORK ON THE pgi15 GPU NODES. They run with the driver at
    # `RmProfilingAdminOnly: 1` (check /proc/driver/nvidia/params), so CUPTI
    # cannot attach; the failed attach poisons the CUDA context and the next
    # launch dies with `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
    # inside jit(train_episode) -- after the ~320 s compile, so it costs a
    # full run to rediscover (job 60018). Leave this off until a sysadmin
    # clears the restriction; use the ALPHAGRAD_PROFILE_TRACE event trace
    # (host timestamps around each callback) for attribution meanwhile.
    _xtr_dir = os.environ.get("ALPHAGRAD_XLA_TRACE_DIR", "")
    _xtr_eps = {int(_x) for _x in
                os.environ.get("ALPHAGRAD_XLA_TRACE_EPS", "").split(",")
                if _x.strip()}
    # The device `train_episode` runs on. Uncommitted arrays already live
    # here (it is the default device); naming it is what makes them
    # COMMITTED, which is the point -- see the loop body.
    _train_dev = jax.local_devices()[0]

    for ep in range(args.episodes):
        if _jax_trace_dir and ep == _jax_trace_ep and not _jax_trace_active:
            tqdm.write(
                f"[profiler] jax.profiler.start_trace -> {_jax_trace_dir} (ep={ep})",
                file=sys.stderr,
            )
            jax.profiler.start_trace(_jax_trace_dir)
            _jax_trace_active = True
        elif _jax_trace_active and ep == _jax_trace_ep + 1:
            jax.profiler.stop_trace()
            _jax_trace_active = False
            tqdm.write(
                f"[profiler] jax.profiler.stop_trace (after ep={ep - 1})",
                file=sys.stderr,
            )
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        if args.preference_conditioned:
            # Stage F mixture: each env independently draws its preference
            # from either the corner Dirichlet (α<1) or the uniform Dirichlet
            # (α=1). With ``--dirichlet-mix-ratio=0.5`` the trainer sees a
            # balanced supply of pure-corner / interior preferences so the
            # conditioned policy covers the whole Pareto front.
            corner_key, uniform_key, choice_key, ep_key = jrand.split(ep_key, 4)
            alpha_corner = jnp.full(
                (NUM_VALUE_HEADS,),
                args.dirichlet_alpha,
                dtype=jnp.float32,
            )
            alpha_uniform = jnp.full(
                (NUM_VALUE_HEADS,),
                args.dirichlet_alpha_uniform,
                dtype=jnp.float32,
            )
            corner_samples = jrand.dirichlet(
                corner_key,
                alpha_corner,
                shape=(num_envs,),
            )
            uniform_samples = jrand.dirichlet(
                uniform_key,
                alpha_uniform,
                shape=(num_envs,),
            )
            use_corner = (
                jrand.uniform(choice_key, (num_envs, 1)) < args.dirichlet_mix_ratio
            )
            preferences_per_env = jnp.where(
                use_corner,
                corner_samples,
                uniform_samples,
            )
        else:
            preferences_per_env = static_pref

        eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
        # ONE ray.put per EPISODE instead of one serialisation per STEP.
        # `eval_args_samples` rides the env, so it is a closed-over constant
        # of the step callback and the pool re-shipped the whole tuple (tens
        # of MB for TransformerLM) on every one of the ~95 decisions. The
        # pool already knows how to hold an ObjectRef -- it was simply never
        # handed one from here. Samples are regenerated per episode, so the
        # ref is refreshed here and nowhere else.
        _ep_pool = getattr(env, "_remote_pool", None)
        if _ep_pool is not None:
            try:
                _ep_pool.set_eval_samples(eval_samples)
            except Exception as _exc:
                tqdm.write(f"[pool] set_eval_samples failed ({_exc!r}); "
                           "falling back to per-call shipping",
                           file=sys.stderr)

        # THE BASE MEMORY for the ROLLOUT only: palimpsa's pass over the base
        # stream, scattered into the slot of the vertex that produced each
        # row. Params-dependent, so it is rebuilt every episode.
        #
        # THE LOSS DOES NOT USE THIS VALUE. `_dynamic_loss_fn` calls
        # `base_memory` itself, inside `filter_grad`. Handing this array to
        # the loss is precisely the defect that was removed: an encoder
        # output computed outside the gradient and read back as a constant.
        # Behaviour policy and target policy still agree at epoch 0 because
        # both are `base_memory(theta_old)` -- the loss recomputes the same
        # function, it does not read a stale copy of it.
        base_mem = _carry_stream.base_memory(
            agent, _BASE_TOK, _BASE_EQN, _BASE_N,
            window=_BASE_W, total_v=total_v, embd_dim=args.embd_dim,
            base_owners=_BASE_OWN,
        )

        env_states = reset_envs(env_episode)
        # Per-call head overrides — static for the whole run (the
        # ``--variant`` masks were resolved once at startup).
        stage_override = op_legality_override
        stage_vertex_mult = jnp.array(1.0, dtype=jnp.float32)
        stage_micro_mult = jnp.array(1.0, dtype=jnp.float32)
        stage_pin_rules = jnp.asarray(
            args.pin_rules_to_exact,
            dtype=jnp.bool_,
        )
        # ---- PopArt warm-start from RANDOM VALID plans -------------------
        # Runs once, before the first gradient step. Uses the ordinary rollout
        # so legality masking and measurement are identical to training; only
        # the vertex pointer is flattened to ~uniform.
        #
        # WHAT IS SEEDED: the DISCOUNTED RETURN, per channel, per step -- the
        # quantity `_popart_update` is actually fed (`estim_returns`) and the
        # quantity the critic regresses. It used to seed the LAST STEP'S REWARD
        # instead. Those are not the same distribution: with `gamma < 1` the
        # return at step t is the terminal reward scaled by `gamma^(T-1-t)`, so
        # seeding from the terminal reward alone overstates both mu and sigma
        # for every earlier step, and the very first advantage is divided by a
        # sigma the batch never had.
        #
        # The reward is pushed through the SAME pipeline as the loss --
        # `_apply_mult_gate` (under --reward-mode mult) then `_symlog_rewards`,
        # then `HEAD_REWARD_INDICES` -- because seeding in one space and
        # updating in another is exactly the raw-vs-symlog bug this warm start
        # was written to fix.
        if ep == 0 and int(getattr(args, "popart_init_episodes", 0)) > 0:
            _wt = jnp.asarray(
                float(getattr(args, "popart_init_temperature", 10.0)),
                dtype=jnp.float32)
            _wG, _wlive = [], []
            for _wi in range(int(args.popart_init_episodes)):
                _wkey, key = jrand.split(key)
                _wstates = reset_envs(env_episode)
                _wend, _wtraj, _wtot = rollout_fn(
                    agent, env_episode, num_valid, _wstates,
                    jrand.split(_wkey, num_envs), base_mem,
                    preferences_per_env, stage_override, stage_pin_rules,
                    _wt,   # positional: vmap in_axes is a positional tuple
                )
                _wr = _wtraj.reward                             # (E, T, R) raw
                # Sentinel test on the RAW vector, identical to `_is_degen` in
                # the loss. Testing it after symlog (as the old code did) can
                # never fire: symlog(-1e10) is -23, nowhere near -1e10, so
                # every failed measurement was silently defining the scale.
                _wl = np.asarray(
                    ~jnp.all(_wr[..., jnp.asarray(COMPUTE_REWARD_INDICES,
                                                  dtype=jnp.int32)]
                             <= (SENTINEL_COST * 0.99), axis=-1))
                if args.reward_mode == "mult":
                    _wr = _apply_mult_gate(
                        _wr, mult_cost_weights, args.gate_tau, args.gate_w,
                        args.anti_degen_penalty, args.anti_degen_tau,
                        gate_fidelity=args.gate_fidelity)
                _hr = np.asarray(
                    _symlog_rewards(_wr)[..., _HEAD_REWARD_INDICES_ARR],
                    dtype=np.float64)                           # (E, T, K)
                _dn = np.asarray(_wtraj.done, dtype=np.float64)
                _dc = np.asarray(_wtraj.discount, dtype=np.float64)
                # Monte-Carlo return, the lambda=1 limit of the GAE target the
                # critic is trained on: G_t = r_t + gamma*(1-done_t)*G_{t+1}.
                _g = np.zeros_like(_hr)
                _run = np.zeros(_hr.shape[::2], dtype=np.float64)   # (E, K)
                for _t in range(_hr.shape[1] - 1, -1, -1):
                    _run = _hr[:, _t, :] + (
                        _dc[:, _t, None] * (1.0 - _dn[:, _t, None]) * _run)
                    _g[:, _t, :] = _run
                _wG.append(_g.reshape(-1, _g.shape[-1]))
                _wlive.append(_wl.reshape(-1))
                # Log this warm-up episode like any other one, minus every
                # gradient-derived key (see host_log's `warmup` branch). NaN
                # metrics are passed only so the tuple shape is uniform; they
                # are dropped before the wandb call, never logged.
                _wmets = (
                    (float("nan"),) * 9
                    + (np.full((7,), np.nan), np.full((6,), np.nan))
                )
                host_log(
                    ep,
                    _wtot,
                    (
                        _wtraj.vertex_idx,
                        _wtraj.pair_seq,
                        _wtraj.factor_seq,
                        _wtraj.micro_op_seq,
                        _wtraj.micro_i_seq,
                        _wtraj.micro_j_seq,
                        _wtraj.micro_factor_seq,
                        _wtraj.micro_compress_kind_seq,
                        _wtraj.micro_quant_dtype_seq,
                        _wend.face_specs,
                        _wend.face_skips,
                    ),
                    jnp.mean(_wtot, axis=0),
                    _wmets,
                    None,
                    warmup=True,
                )
            _R = np.concatenate(_wG, axis=0)
            _live_m = np.concatenate(_wlive, axis=0)
            _ok = _live_m & np.isfinite(_R).all(axis=1)
            _R = _R[_ok]
            print(f"[popart-init] {int(args.popart_init_episodes)} random-plan "
                  f"rollouts -> {_R.shape[0]}/{_ok.shape[0]} usable "
                  f"(env, step) returns", flush=True)
            if _R.shape[0] >= 2:
                _mu0 = _R.mean(axis=0)
                _sd0 = _R.std(axis=0)
                # A CONSTANT channel is not seed-able and must stay COLD.
                # Random plans routinely return cosine == 0 at EVERY step (an
                # all-zero Jacobian and a half-destroyed one both read ~0), so
                # that channel's sample has sigma exactly 0 and carries no
                # scale information. Stamping w=1 on it anyway would claim the
                # accumulator is already warm, and _popart_update's debiasing
                # would then let the FIRST real measurement move mu by only
                # beta instead of adopting the batch exactly -- the seed would
                # actively slow down learning the one channel it knows nothing
                # about. So w is per-channel: warm where the sample has spread,
                # cold (w=0, the untouched init) where it does not.
                _warm = _sd0 > 1e-12
                popart_m1 = jnp.asarray(np.where(_warm, _mu0, 0.0),
                                        dtype=jnp.float32)
                popart_m2 = jnp.asarray(np.where(_warm, (_R ** 2).mean(axis=0),
                                                 0.0), dtype=jnp.float32)
                popart_w = jnp.asarray(_warm.astype(np.float32))
                # Variance of the NORMALISED target the critic will see. It
                # is 1.0 by construction UNLESS `--popart-sigma-min` floors the
                # channel's sigma, in which case the channel is being SHRUNK
                # rather than scaled and its advantage silently loses against
                # the others. Reported per channel so a floored channel is
                # visible at ep 0 instead of being inferred 500 episodes later.
                _sd_eff = np.maximum(_sd0, float(args.popart_sigma_min))
                _zvar = (((_R - _mu0) / _sd_eff) ** 2).mean(axis=0)
                for _k, _nm in enumerate(HEAD_NAMES):
                    if not _warm[_k]:
                        _fl = "  <-- CONSTANT in the sample, left COLD"
                    elif _sd0[_k] < _sd_eff[_k]:
                        _fl = "  <-- sigma FLOORED by --popart-sigma-min"
                    else:
                        _fl = ""
                    print(f"[popart-init]   {_nm}: mu={_mu0[_k]:.6g} "
                          f"sigma={_sd0[_k]:.6g} norm_var={_zvar[_k]:.4f}{_fl}",
                          flush=True)
                try:
                    _cfg = {}
                    for _k, _nm in enumerate(HEAD_NAMES):
                        _cfg[f"popart_init_mu_{_nm}"] = float(_mu0[_k])
                        _cfg[f"popart_init_sigma_{_nm}"] = float(_sd0[_k])
                        _cfg[f"popart_init_normvar_{_nm}"] = float(_zvar[_k])
                    _cfg["popart_init_samples"] = int(_R.shape[0])
                    wandb.config.update(_cfg, allow_val_change=True)
                except Exception:
                    pass
            else:
                print("[popart-init] too few usable returns; keeping zero init",
                      flush=True)
            env_states = reset_envs(env_episode)

        # ONE COMPILE, NOT TWO. `eqx.filter_jit`'s cache key includes every
        # argument's SHARDING, and JAX distinguishes an UNCOMMITTED array
        # (anything built host-side -- `jnp.zeros`, `jnp.asarray(nparray)`)
        # from a COMMITTED one (anything a jit returned). Six leaves flip:
        # the three PopArt statistics, which the calibration episode rebuilds
        # from numpy, and three int32 step counters inside the agent/optimiser
        # state. They arrive uncommitted on the first call and committed from
        # the first jit output onwards, so `train_episode` was traced and
        # compiled a SECOND time -- 327 s, on top of the first 327 s, against
        # a 15 s steady-state episode. Committing them up front makes every
        # episode present the same key.
        #
        # `jax.Array` leaves ONLY. Several arguments are static Python values
        # that shape the trace (`EnvState.max_steps`, the stage flags, the
        # jaxpr in `EnvConfig`); turning those into device arrays would change
        # WHAT is compiled, not just where the bytes live. numpy leaves are
        # left alone too -- they are uncommitted on every call already, so
        # they never contributed to the divergence.
        (agent, opt_state, global_step,
         popart_m1, popart_m2, popart_w) = jax.tree_util.tree_map(
            lambda _x: (jax.device_put(_x, _train_dev)
                        if isinstance(_x, jax.Array) else _x),
            (agent, opt_state, global_step,
             popart_m1, popart_m2, popart_w),
        )
        _xtr_on = bool(_xtr_dir) and ep in _xtr_eps
        if _xtr_on:
            jax.profiler.start_trace(os.path.join(_xtr_dir, f"ep{ep}"))
        (
            agent,
            opt_state,
            _,
            metrics,
            total_rewards_full,
            actions_pack,
            global_step,
            diag_pack,
            popart_m1,
            popart_m2,
            popart_w,
            attn_ent,
            true_scalar_return,
            probes,
            probe_opt_state,
            probe_metrics,
        ) = train_episode(
            agent,
            opt_state,
            env_states,
            env_episode,
            base_mem,
            preferences_per_env,
            global_step,
            ep_key,
            default_freeze_mask,
            stage_override,
            stage_vertex_mult,
            stage_pin_rules,
            stage_micro_mult,
            popart_m1,
            popart_m2,
            popart_w,
            probes,
            probe_opt_state,
        )
        if _xtr_on:
            # The trace has to stay open until the device is drained or the
            # timeline stops at the first output that happens to be ready.
            jax.block_until_ready(
                (agent, opt_state, metrics, total_rewards_full,
                 global_step, popart_m1, popart_m2, popart_w))
            jax.profiler.stop_trace()
        # SEEDED EQUIVALENCE DUMP (ALPHAGRAD_EQ_DUMP=<prefix>, off by default).
        # Any change to the env / callback / rollout path has to be proven
        # trajectory-identical against its parent commit, and the only honest
        # way to do that is to pickle the post-episode state and diff it leaf
        # by leaf (see ~/dsnn/_eq137_cmp.py). Pure instrumentation: nothing
        # downstream reads the file, and the branch is dead without the var.
        _eqp = os.environ.get("ALPHAGRAD_EQ_DUMP", "")
        if _eqp:
            import pickle as _pk
            _leaves = lambda t: [np.asarray(l) for l in
                                 jax.tree_util.tree_leaves(t)]
            with open(f"{_eqp}.ep{ep}.pkl", "wb") as _fh:
                _pk.dump({
                    "params": _leaves(eqx.filter(agent, eqx.is_array)),
                    "opt": _leaves(opt_state),
                    "metrics": _leaves(metrics),
                    "actions": _leaves(actions_pack),
                    "rewards": np.asarray(total_rewards_full),
                    "step": int(global_step),
                    "ret": float(true_scalar_return),
                }, _fh)
        host_log(
            ep,
            total_rewards_full,
            actions_pack,
            jnp.mean(total_rewards_full, axis=0),
            metrics,
            diag_pack,
            popart_stats=_popart_derive(
                popart_m1, popart_m2, popart_w,
                args.popart_sigma_min, 1e12),
            attn_entropy=attn_ent,
            true_return=float(true_scalar_return),
            probe_metrics=probe_metrics,
        )
        # Mid-training top-N snapshot. Skips the wandb table log so we
        # don't pollute the offline run with duplicate tables — only the
        # post-training / post-calibration dumps land in wandb.
        if (
            args.print_top_every > 0
            and (ep + 1) % args.print_top_every == 0
            and ep + 1 < args.episodes
        ):
            pbar.write(
                f"\n=== top-{args.top_n} after episode {ep + 1}/{args.episodes} ==="
            )
            print_top_n(
                "Total Reward",
                host_state["top_n_total"],
                log_to_wandb=False,
            )

    pbar.close()

    # Safe stop for the profiler in case the training loop ended before
    # the (trace_ep + 1) boundary (e.g. ``--episodes 3`` with
    # ``ALPHAGRAD_JAX_TRACE_EP=2``). Without this, JAX leaks an open
    # tracing session and the dump never lands on disk.
    if _jax_trace_active:
        jax.profiler.stop_trace()
        _jax_trace_active = False
        tqdm.write(
            "[profiler] jax.profiler.stop_trace (post-loop safe-stop)",
            file=sys.stderr,
        )

    _dump_pareto(pareto_archive, args, args.episodes, final=True)
    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n(f"CMP (Lowest {args.cmp_type})", host_state["top_n_cmp"])
    print_top_n(f"Memory (Lowest {args.mem_type})", host_state["top_n_mem"])
    print_top_n(f"Quality (Highest {_QUALITY_METRIC})", host_state["top_n_acc"])
    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
