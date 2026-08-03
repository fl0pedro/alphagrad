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
    compute_per_sample_vertex_features,
    compute_vertex_features,
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
    _AXIS_FEAT_GROUP_ID,
    consume_degenerate_plan_count,
    consume_truncated_plan_count,
    consume_untraceable_plan_count,
    consume_zero_work_plan_count,
    consume_per_face_stats,
    consume_tokenization_truncation_stats,
    consume_xla_memory_stats,
    _AXIS_FEAT_IS_COMPRESSED,
    _AXIS_FEAT_IS_OUTPUT,
    _AXIS_FEAT_SIZE,
    FACE_SLOTS,
    MAX_AXES_PER_VERTEX,
    MAX_FACES as ENV_MAX_FACES,
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
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
from alphagrad.approx.live_faces import LiveFaceStream
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
from alphagrad.utils import entropy, explained_variance

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
HEAD_NAMES: tuple[str, ...] = ("latency", "mem", "cos")

# Print every loss component the moment the total goes non-finite. Off by
# default because it forces a host callback inside the jitted update.
_DEBUG_NAN = os.environ.get("ALPHAGRAD_DEBUG_NAN", "0") == "1"
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


def _stream_len(tokens):
    """Number of real tokens in an append-only buffer (pad id is 0)."""
    return jnp.sum((tokens != 0).astype(jnp.int32))


def _zero_enc_carry_fields():
    """Degenerate (0,)-shaped stand-ins for the incremental-encode trajectory
    fields when the mode is off — zero memory, uniform pytree structure."""
    z = jnp.zeros((0,), jnp.float32)
    return dict(
        enc_M=z, enc_I=z, enc_cumhist=z,
        enc_nvalid=jnp.zeros((), jnp.float32),
        enc_pos=jnp.zeros((), jnp.int32),
        vmem_sums=z, vmem_counts=z,
        delta_owner=jnp.zeros((), jnp.int32),
    )


class Trajectory(NamedTuple):
    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array  # (V, embd_dim) at the start of this step
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
    face_old_logp: jax.Array      # () float32
    # Per-face chunk LENGTHS -- the only face-stream data the loss needs.
    # The chunks concatenate to exactly the step delta (pinned property), so
    # boundaries are the counts' cumsum and the loss pools the delta rows it
    # already computes in `_carry_heads`. The token windows that used to sit
    # here were the storage that made any face width expensive.
    face_counts: jax.Array        # (MAX_FACES,) int32
    # THIS step's emission (the current vertex's contractions + approx
    # echoes), sliced from next_state.tokens at the post-decision carry pos.
    # It is NOT in `tokens` (that buffer is the PRE-step stream), and it is
    # what the face chunks concatenate to -- the loss scans it once from the
    # stored carry's continuation and pools between chunk boundaries.
    face_delta_tokens: jax.Array  # (MAX_DELTA_TOKENS,) int32
    face_delta_eqns: jax.Array    # (MAX_DELTA_TOKENS,) int32
    # Phase 3b (--incremental-encode; degenerate (0,) shapes otherwise). The
    # PRE-step encoder carry + vertex memory, and the delta's owner vertex —
    # everything the loss needs to re-derive this step's encoding by
    # extending with the delta tokens only.
    enc_M: jax.Array        # (L, H, d, d)
    enc_I: jax.Array        # (L, H, d, d)
    enc_cumhist: jax.Array  # (MAX_EQNS,)
    enc_nvalid: jax.Array   # ()
    enc_pos: jax.Array      # () int32
    vmem_sums: jax.Array    # (V+1, E)
    vmem_counts: jax.Array  # (V+1,)
    delta_owner: jax.Array  # () int32 — vertex whose elimination emitted the delta
    discount: jax.Array
    vertex_avail_mask: jax.Array


class TrainBatch(NamedTuple):
    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array
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
    delta_owner: jax.Array
    estim_returns: jax.Array
    norm_adv: jax.Array
    vertex_avail_mask: jax.Array


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


class SetTransformerAggregator(eqx.Module):
    """Stage B.3 — permutation-invariant aggregator over calibration samples.

    Spec: "A small Set Transformer aggregates these features across the
    calibration samples, permutation-invariantly, to produce a per-vertex
    data embedding $d_v$." The per-sample variability is what tells the
    policy whether a vertex's behaviour is stable on the data distribution.

    Architecture (intentionally lightweight):

    * Embed the static op-type id (column 0) and concatenate with the
      remaining ``F-1`` continuous features → ``(S, V, hidden)``.
    * One self-attention block over the **sample** axis per vertex (each
      vertex's S samples form a set; attention is permutation-equivariant).
    * Mean-pool over samples → permutation-invariant ``(V, hidden)``.
    * Project to ``embd_dim`` → per-vertex data embedding.

    With 5 calibration samples this is essentially free; for larger sample
    counts the attention cost is O(S²) per vertex which is still tiny vs
    the main encoder.
    """

    op_embedding: eqx.nn.Embedding
    input_proj: eqx.nn.Linear
    sample_attn: RelationalMultiheadAttention
    output_proj: eqx.nn.Linear

    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    op_embd_dim: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_features,
        op_embd_dim,
        hidden_dim,
        num_heads,
        embd_dim,
        vocab_size,
        key,
    ):
        keys = jrand.split(key, 4)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.op_embd_dim = op_embd_dim
        self.num_features = num_features
        self.op_embedding = eqx.nn.Embedding(vocab_size, op_embd_dim, key=keys[0])
        # +op_embd_dim for op_emb, +num_features-1 for continuous features
        # (we drop column 0 since the op-type id is consumed by the embedding).
        self.input_proj = eqx.nn.Linear(
            op_embd_dim + num_features - 1,
            hidden_dim,
            key=keys[1],
        )
        self.sample_attn = RelationalMultiheadAttention(
            num_heads,
            hidden_dim,
            key=keys[2],
        )
        self.output_proj = eqx.nn.Linear(hidden_dim, embd_dim, key=keys[3])

    def __call__(self, per_sample_features, *, key):
        """``per_sample_features``: ``(S, V, F)``. Returns ``(V, embd_dim)``."""
        S, V, _ = per_sample_features.shape
        op_ids = per_sample_features[0, :, 0].astype(jnp.int32)
        op_emb = jax.vmap(self.op_embedding)(op_ids)  # (V, op_d)
        cont = per_sample_features[:, :, 1:]  # (S, V, F-1)
        op_emb_b = jnp.broadcast_to(op_emb[None, :, :], (S, V, op_emb.shape[-1]))
        combined = jnp.concatenate([op_emb_b, cont], axis=-1)  # (S, V, op_d+F-1)
        h = jax.vmap(jax.vmap(self.input_proj))(combined)  # (S, V, hidden)

        # Self-attention over samples, per-vertex independently.
        h_perm = jnp.transpose(h, (1, 0, 2))  # (V, S, hidden)
        attn_keys = jrand.split(key, V)
        h_attn = jax.vmap(lambda x, k: self.sample_attn(x, x, x, key=k))(
            h_perm, attn_keys
        )  # (V, S, hidden)

        pooled = jnp.mean(h_attn, axis=1)  # (V, hidden)
        return jax.vmap(self.output_proj)(pooled)  # (V, embd_dim)


class ResidualStateUpdate(eqx.Module):
    """Stage B.4: small recurrent update on the per-vertex residual state.

    Spec calls for tracking the cumulative effect of past eliminations so the
    decoder can condition on what's already happened without re-encoding the
    full residual jaxpr. The minimal-viable form here is a per-slot GRU-style
    gate: when vertex ``v`` is eliminated, blend its new representation into
    ``s_v`` with a learned scalar gate; other slots are unchanged.

    A future B.4.next can extend this to propagate updates through the DAG
    neighbours of the eliminated vertex (using the relation masks from B.1)
    so the residual carries dataflow information beyond just "this vertex
    was eliminated".
    """

    event_proj: eqx.nn.Linear
    decay_logit: jax.Array

    residual_dim: int = eqx.field(static=True)

    def __init__(self, *, residual_dim: int, key):
        keys = jrand.split(key, 2)
        self.residual_dim = residual_dim
        self.event_proj = eqx.nn.Linear(residual_dim, residual_dim, key=keys[0])
        # Initialised at 0 → gate sigmoid(0) = 0.5; first-step blend is
        # symmetric. Letting this become a learned scalar saves us from
        # picking a hyperparameter.
        self.decay_logit = jnp.zeros((), dtype=jnp.float32)

    def __call__(self, residual_state, vertex_idx, vertex_repr):
        """Update slot ``vertex_idx`` of ``residual_state`` with ``vertex_repr``.

        ``residual_state`` is `(V, residual_dim)`; ``vertex_repr`` is
        `(residual_dim,)`. Returns a new array of the same shape.
        """
        gate = jnn.sigmoid(self.decay_logit)
        update = self.event_proj(vertex_repr)
        new_slot = residual_state[vertex_idx] * (1.0 - gate) + update * gate
        return residual_state.at[vertex_idx].set(new_slot)


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
    vertex_feature_proj: eqx.nn.Linear
    # B.3: Set Transformer aggregator over calibration samples. Always
    # constructed; only invoked when `vertex_features` arrives with a leading
    # sample axis (rank-3) — the agent dispatches automatically.
    set_transformer_agg: SetTransformerAggregator
    residual_update: ResidualStateUpdate
    # B.4.next: residual-state → summary projection used by the cached-encoding
    # value path. Zero-initialised so the cached and re-encoding paths agree
    # on the initial value at episode start (residual_state == 0).
    residual_to_summary: eqx.nn.Linear
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
        vertex_feature_proj,
        set_transformer_agg,
        residual_update,
        residual_to_summary,
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
        self.vertex_feature_proj = vertex_feature_proj
        self.set_transformer_agg = set_transformer_agg
        self.residual_update = residual_update
        self.residual_to_summary = residual_to_summary
        self.pref_proj = pref_proj
        self.num_vertices = num_vertices
        self.num_value_heads = num_value_heads
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.embd_dim = embd_dim
        self.op_embd_dim = op_embd_dim

    def _data_embedding(self, vertex_features, *, agg_key=None):
        """Project per-vertex features to a per-vertex embedding ``(V, embd_dim)``.

        Two input shapes are accepted, dispatched on rank:

        * ``(V, NUM_VERTEX_FEATURES)`` — already aggregated across samples
          (Stage B.2.A path). Goes through :attr:`vertex_feature_proj`.
        * ``(S, V, NUM_VERTEX_FEATURES)`` — per-sample features (Stage B.3).
          The :class:`SetTransformerAggregator` is invoked to do a learned,
          permutation-invariant pool across the sample axis.
        """
        if vertex_features.ndim == 3:
            key = agg_key if agg_key is not None else jrand.PRNGKey(0)
            return self.set_transformer_agg(vertex_features, key=key)
        op_ids = vertex_features[:, 0].astype(jnp.int32)
        op_emb = jax.vmap(self.op_embedding)(op_ids)
        cont = vertex_features[:, 1:]
        combined = jnp.concatenate([op_emb, cont], axis=-1)
        return jax.vmap(self.vertex_feature_proj)(combined)

    def encode(
        self,
        tokens,
        eqn_ids=None,
        vertex_features=None,
        residual_state=None,
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

        # Stage B.2.A: fold per-vertex data features into the per-vertex
        # contexts so the rule policy and value head are conditioned on
        # the calibration-sample-derived signal alongside the IR encoding.
        if vertex_features is not None:
            vertex_contexts = vertex_contexts + self._data_embedding(vertex_features)

        # Stage B.4: add the cumulative-elimination residual state so the
        # decoder can condition on which vertices have already been picked
        # without paying for re-encoding the residual jaxpr from scratch.
        if residual_state is not None:
            vertex_contexts = vertex_contexts + residual_state

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
        vertex_features=None,
        residual_state=None,
        preference=None,
        key=None,
    ):
        _, _, value = self.encode(
            tokens,
            eqn_ids=eqn_ids,
            vertex_features=vertex_features,
            residual_state=residual_state,
            preference=preference,
            key=key,
        )
        return value

    def update_residual(self, residual_state, vertex_idx, vertex_repr):
        """Apply the per-slot recurrent update to ``residual_state``."""
        return self.residual_update(residual_state, vertex_idx, vertex_repr)

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
                      start=None):
        """Extend the palimpsa carry by ``count`` tokens read from
        ``tokens_buf`` at ``carry.pos`` (fixed static ``window``; pad steps
        freeze the carry, so the valid prefix is bitwise-independent of the
        window size). Returns ``(new_carry, rows, valid, eqn_window)`` with
        ``rows`` (window, E) zeroed on invalid steps.

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
        return self._extend_sequential(carry, toks, eqns, valid, count)

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
                q = jax.vmap(mixer.query_proj)(y).reshape(Bk, H, d)
                kk = jax.vmap(mixer.key_proj)(y).reshape(Bk, H, d)
                v = jax.vmap(mixer.value_proj)(y).reshape(Bk, H, d)
                b = jnn.softplus(
                    jax.vmap(mixer.bias_proj)(y)).reshape(Bk, H, d)
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

    def _extend_sequential(self, carry, toks, eqns, valid, count):
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
                q = mixer.query_proj(y).reshape(H, d)
                kk = mixer.key_proj(y).reshape(H, d)
                v = mixer.value_proj(y).reshape(H, d)
                b = jnn.softplus(mixer.bias_proj(y)).reshape(H, d)
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
        (M2, I2, ch2, nv2), rows = lax.scan(
            _step,
            (carry.M, carry.I, carry.cumhist, carry.nvalid),
            (toks, eqns, valid),
            unroll=int(os.environ.get("ALPHAGRAD_EXTEND_UNROLL", "8")),
        )
        new_carry = EncCarry(M=M2, I=I2, cumhist=ch2, nvalid=nv2,
                             pos=carry.pos + count)
        return new_carry, rows, valid, eqns

    def heads_from_memory(
        self,
        vmem_sums,
        vmem_counts,
        vertex_features=None,
        residual_state=None,
        preference=None,
    ):
        """The ``encode()`` head block, fed from the per-vertex memory instead
        of the raw (S, E) sequence: pointer via ``from_vertex_memory`` (same
        weights, keys/values = the V+1 pooled slots), value summary via the
        token-count-weighted slot mean (identical to the full path's masked
        token mean by associativity). Returns the same
        ``(vertex_logits, vertex_contexts, value)`` triple.
        """
        vmem_rows = _vmem.read(vmem_sums, vmem_counts)
        vmask = _vmem.occupancy(vmem_counts)
        from alphagrad.approx.set_pointer import SetPointerVertexPolicy
        if (isinstance(self.vertex_policy, SetPointerVertexPolicy)
                and vertex_features is not None):
            # The pointer must score CONTENT, and the vertices it has to
            # choose between are exactly the ones with EMPTY vmem slots
            # (slots fill on elimination). Occupancy-masked vmem alone gave
            # every candidate an identical zero slot -- the v31 uniform-pick
            # failure. Fold the per-vertex features (axis sizes, op
            # identity: content, not a learned V-table) into the slots and
            # let every vertex slot participate.
            _feat = self._data_embedding(vertex_features)
            slots = vmem_rows.at[: _feat.shape[0]].add(_feat)
            smask = jnp.ones_like(vmask)
            vertex_logits, vertex_contexts = (
                self.vertex_policy.from_vertex_memory(slots, smask))
        else:
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
        if vertex_features is not None:
            vertex_contexts = vertex_contexts + self._data_embedding(vertex_features)
        if residual_state is not None:
            vertex_contexts = vertex_contexts + residual_state
        summary = _vmem.summary(vmem_sums, vmem_counts)
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
        vertex_features=None,
        residual_state=None,
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
                vertex_features=vertex_features,
                residual_state=residual_state,
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
                 f_de) = self._face_loop(
                    v_context, features, factor_tables, face_key,
                    f_pair, f_comp, f_valid, enc_carry, face_chunk_fn,
                    vertex_idx, _vspecs, axis_state[vertex_idx],
                    op_legality_override,
                    (_n_faces if _n_faces is not None
                     else face_count_fn(vertex_idx)),
                )
            face_out = (fa, face_logp, face_ent, f_pair, f_comp, f_valid,
                        f_cnt, f_dt, f_de)

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
            return c2, self._face_pool(rows, valid, count)

        def _skip(c):
            return c, jnp.zeros((self.embd_dim,), jnp.float32)

        return lax.cond(count > 0, _run, _skip, carry)

    @staticmethod
    def _face_pool(rows, valid, count):
        """Mean palimpsa row over a face's chunk (zero when the chunk is
        empty -- an empty chunk must leave the context untouched, not inject
        a bias)."""
        w = valid.astype(jnp.float32)[:, None]
        return jnp.sum(rows * w, axis=0) / jnp.maximum(
            count.astype(jnp.float32), 1.0)

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

    def _face_loop(self, v_context, features, factor_tables, key,
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
               jnp.asarray(0, jnp.int32))

        def _body(st):
            f, carry, logp, ent, skips, cnts, rs, wa, ftok, feqn, off = st
            tk_f, eq_f, ct_f = face_chunk_fn(f, vertex_idx, vertex_specs,
                                             rs, skips)
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
                v_context, features, factor_tables, jrand.fold_in(key, f),
                f, f_pair[f], f_comp[f], f_valid[f], face_context=summ,
                op_legality_override=op_legality_override)
            rs = rs.at[f].set(self._face_row_specs(row, axis_state_v))
            skips = skips.at[f].set(sk.astype(jnp.int32))
            cnts = cnts.at[f].set(ct_eff)
            wa = tuple(w.at[f].set(row[k])
                       for w, k in zip(wa, self._WIRE_KEYS))
            return (f + 1, carry, logp + lp, ent + e, skips, cnts, rs, wa,
                    ftok, feqn, off + ct_eff)

        (_f, _c, logp, ent, skips, cnts, _rs, wa, ftok, feqn,
         _off) = lax.while_loop(lambda st: st[0] < n, _body, st0)
        fa = FaceAction(skip=skips, **dict(zip(self._WIRE_KEYS, wa)))
        return fa, logp, ent, cnts, ftok, feqn

    def _face_replay(self, v_context, features, factor_tables, fa,
                     f_pair, f_comp, f_valid, enc_carry, face_chunks,
                     op_legality_override):
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
        """
        pol = self.face_path_policy
        F = pol.max_faces
        f_cnt, f_toks, f_eqns = face_chunks
        total = jnp.sum(f_cnt.astype(jnp.int32))
        _, rows, _valid, _e = self.encode_extend(
            enc_carry, f_toks, f_eqns, total,
            window=MAX_DELTA_TOKENS, start=0)
        # Exclusive-prefix boundaries from the counts; pooled mean per span.
        # An empty chunk gives s == e -> zero context, matching the rollout's
        # `_face_encode` skip exactly.
        cum = jnp.concatenate(
            [jnp.zeros((1, rows.shape[-1]), rows.dtype),
             jnp.cumsum(rows, axis=0)], axis=0)          # (W+1, E)
        ends = jnp.cumsum(f_cnt.astype(jnp.int32))
        starts = ends - f_cnt.astype(jnp.int32)

        # scan, not a python loop: F is the provable bound (196 here), and
        # unrolling it multiplied the program by F. scan keeps reverse-mode
        # AD (while_loop would not); padding faces are gated to zero by the
        # stored face_valid, so the extra iterations cost one small MLP each.
        def _scan_f(acc, f):
            logp, ent, arity = acc
            span = (cum[jnp.clip(ends[f], 0, rows.shape[0])]
                    - cum[jnp.clip(starts[f], 0, rows.shape[0])])
            summ = span / jnp.maximum(f_cnt[f].astype(jnp.float32), 1.0)
            lp, e, ar, _sp, _od = pol.evaluate_face(
                v_context, features, factor_tables, fa, f,
                f_pair[f], f_comp[f], f_valid[f], face_context=summ,
                op_legality_override=op_legality_override)
            return (logp + lp, ent + e, arity + ar), None

        (logp, ent, arity), _ = lax.scan(
            _scan_f, (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
            jnp.arange(F))
        return logp, ent, arity

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
        vertex_features=None,
        residual_state=None,
        cached_encoding=None,
        preference=None,
        pair_valid=None,       # stored live DIAG mask for the chosen vertex
        compress_valid=None,   # stored live COMPRESS mask
        op_legality_override=None,  # (NUM_OPS,) variant mask — MUST match sample
        face_action: FaceAction | None = None,  # P1c stored per-path actions
        face_pair_valid=None,  # stored (F,N,N) sampling mask
        face_comp_valid=None,  # stored (F,N)
        face_valid=None,       # stored (F,)
        precomputed=None,      # 3b: (vertex_logits, vertex_contexts, value) from the carry path
        face_chunks=None,      # (counts, emission tokens, emission eqns)
        face_carry=None,       # carry2: where the sampling side carry branched
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
                    tokens, eqn_ids=eqn_ids, vertex_features=vertex_features,
                    residual_state=residual_state, preference=preference,
                    key=key)
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
            )
        if precomputed is not None:
            # 3b: same carry-derived triple the rollout sampled under (the
            # loss re-derives it from the stored pre-step carry + delta).
            vertex_logits, vertex_contexts, value = precomputed
        else:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
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
                f_logp, f_ent, f_arity = self._face_replay(
                    v_context, features, factor_tables, face_action,
                    face_pair_valid, face_comp_valid, face_valid,
                    face_carry, face_chunks, op_legality_override,
                )
            total_log_p = total_log_p + f_logp
            total_entropy = total_entropy + f_ent / jnp.maximum(f_arity, 1.0)
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
        "--incremental-encode", action="store_true",
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
             "ever share a device -- co-residency measured CV 0.0000% -> "
             "49.7%. Ray rather than threads because the per-measure XLA "
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
        "--popart-init-episodes", type=int, default=0,
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
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
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
        help="Allow the top-N accuracy heap to keep trajectories with cosine similarity == 1.0.",
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
    pos_enc = PositionalEncoder(args.embd_dim, MAX_TOKENS) if _use_pe else None
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
    # Continuous part of the feature vector = NUM_VERTEX_FEATURES - 1 (we drop
    # column 0 / op_type_id which is consumed by the embedding).
    proj_in_dim = args.op_embd_dim + NUM_VERTEX_FEATURES - 1
    vertex_feature_proj = eqx.nn.Linear(
        proj_in_dim,
        args.embd_dim,
        key=encoder_keys[7],
    )
    residual_update = ResidualStateUpdate(
        residual_dim=args.embd_dim,
        key=encoder_keys[8],
    )
    residual_to_summary = eqx.nn.Linear(
        args.embd_dim,
        args.embd_dim,
        key=encoder_keys[9],
    )
    # Hidden dim must be divisible by num_heads — keep it at embd_dim for
    # simplicity. The aggregator is small (one attention block over the
    # sample axis), so this isn't a meaningful parameter cost.
    set_transformer_agg = SetTransformerAggregator(
        num_features=NUM_VERTEX_FEATURES,
        op_embd_dim=args.op_embd_dim,
        hidden_dim=args.embd_dim,
        num_heads=args.num_heads,
        embd_dim=args.embd_dim,
        vocab_size=OP_TYPE_VOCAB_SIZE,
        key=encoder_keys[10],
    )
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
        vertex_feature_proj=vertex_feature_proj,
        set_transformer_agg=set_transformer_agg,
        residual_update=residual_update,
        residual_to_summary=residual_to_summary,
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
    # Stage B.2.A: zero the data-feature projection's output so the initial
    # data embedding is `0` and the agent's behaviour at step 0 matches the
    # B.1 agent. Gradient still flows in normally once training starts.
    agent = scale_module_weight(
        agent,
        lambda a: a.vertex_feature_proj.weight,
        0.0,
    )
    # B.3: zero the Set Transformer's output projection too. Same idea —
    # initial data embedding from the per-sample path is exactly zero, so
    # toggling --set-transformer-agg doesn't perturb the initial policy.
    agent = scale_module_weight(
        agent,
        lambda a: a.set_transformer_agg.output_proj.weight,
        0.0,
    )
    # Stage B.4: zero the residual-update event projection so the residual
    # state contributes nothing at initialisation. Same idea — keeps the
    # initial policy distribution uncorrupted by uninitialised additive paths.
    agent = scale_module_weight(
        agent,
        lambda a: a.residual_update.event_proj.weight,
        0.0,
    )
    # B.4.next: zero residual_to_summary so the cached-encoding value path
    # produces the same initial summary as the re-encoding path.
    agent = scale_module_weight(
        agent,
        lambda a: a.residual_to_summary.weight,
        0.0,
    )
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
}


def _cmp_reward_index(cmp_type: str) -> int:
    return REWARD_INDEX[_CMP_TYPE_TO_REWARD[cmp_type]]


def _mem_reward_index(mem_type: str) -> int:
    return REWARD_INDEX[_MEM_TYPE_TO_REWARD[mem_type]]



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
    "aggregator": ("set_transformer_agg",),
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




def _episode_vertex_features(
    args,
    jaxpr,
    consts: tuple,
    base_args: tuple,
    eval_samples,
    argnums: tuple,
) -> "jax.Array":
    """Single dispatch site for the per-vertex feature computation.

    Switches between mean-aggregated (Stage B.2.A) and per-sample (Stage B.3
    Set Transformer aggregator) features based on ``--set-transformer-agg``;
    used by the main training loop and the BC warm-start. Returns a
    `jnp.float32` array.
    """
    fn = (
        compute_per_sample_vertex_features
        if args.set_transformer_agg
        else compute_vertex_features
    )
    return jnp.asarray(
        fn(jaxpr, consts, base_args, eval_samples=eval_samples, argnums=argnums),
        dtype=jnp.float32,
    )


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
    if getattr(args, "incremental_encode", False):
        _pol = os.environ.get("ALPHAGRAD_POLICY", "transformer").strip().lower()
        if _pol != "palimpsa":
            raise ValueError(
                "--incremental-encode requires ALPHAGRAD_POLICY=palimpsa "
                f"(causal); got {_pol!r} (palimpsa_bi's reverse pass cannot "
                "ride a causal carry, and the transformer needs absolute "
                "positions)."
            )
        if os.environ.get("ALPHAGRAD_POS_ENC", "auto").strip().lower() in (
            "1", "true", "yes"
        ):
            raise ValueError(
                "--incremental-encode is incompatible with ALPHAGRAD_POS_ENC=1 "
                "(absolute positions have no incremental analogue)."
            )
        if os.environ.get("ALPHAGRAD_INCREMENTAL_TOKENS", "0") != "1":
            raise ValueError(
                "--incremental-encode requires ALPHAGRAD_INCREMENTAL_TOKENS=1 "
                "(the append-only stream is what the carry extends)."
            )
        if not args.dynamic_substeps:
            raise ValueError(
                "--incremental-encode is only wired for --dynamic-substeps."
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
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
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
    )

    # ---- --ray-measure: fan the measurement callback out over Ray actors ----
    if int(getattr(args, "ray_measure", 0) or 0) > 0:
        _n_actors = int(args.ray_measure)
        if getattr(args, "face_actions", False):
            raise ValueError(
                "--ray-measure is incompatible with --face-actions: the "
                "measurement pool's env is per-vertex and DROPS "
                "face_specs/face_skips, so it would measure a different plan "
                "than the policy chose."
            )
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
            _ray.init(ignore_reinit_error=True, include_dashboard=False)
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
            max_tokens=int(MAX_TOKENS),
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
                    _oracle_jaxpr, v, specs[k], is_last=(k == n - 1)
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
                    _oracle_jaxpr, v, specs[k], is_last=(k == n - 1)
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
                    _oracle_jaxpr, v, specs[k], is_last=(k == n - 1)
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
            except Exception:
                pass
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

    # ---- per-FACE token chunks (--live-faces) -------------------------
    # One callback per face. It replays the elimination prefix (cached) and
    # re-eliminates the CURRENT vertex with faces 0..f-1 carrying their
    # decided approximations, returning the tokens emitted between the
    # previous decision and this one. graphax has no resumable elimination,
    # so re-running is the only way to reach face f's contraction with face
    # f-1's approximation in place; the prefix is replayed once per distinct
    # prefix, so the cost is n_faces eliminations per env step.
    _LIVE_FACES = None
    if getattr(args, "live_faces", False):
        _LIVE_FACES = LiveFaceStream(
            _oracle_jaxpr, _oracle_argnums, _oracle_consts, _oracle_args,
            vocab=int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "248")),
            max_faces=_F_FACES, max_axes=_oracle_N,
            # A chunk is a slice of the step delta, so the delta cap is the
            # one honest window: truncation becomes impossible whenever the
            # delta itself fits, and the stored counts stay exact for the
            # loss's cumsum boundaries.
            window=MAX_DELTA_TOKENS,
        )

    def _live_face_host(order, spec_hist, step_count, vertex_idx,
                        vertex_specs, face_rows, face_skips, f):
        _pt0 = _prof_time.perf_counter()
        try:
            tok, ids, cnt, _nf = _LIVE_FACES.chunk(
                order, spec_hist, int(np.asarray(step_count)),
                int(np.asarray(vertex_idx)) + 1, vertex_specs,
                face_rows, face_skips, int(np.asarray(f)),
            )
            return tok, ids, np.asarray(cnt, np.int32)
        finally:
            _env_prof_add("faces.live_chunk",
                          _prof_time.perf_counter() - _pt0)

    def _live_face(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
                   face_rows, face_skips):
        W = MAX_DELTA_TOKENS
        # f rides as an OPERAND: inside the while_loop it is a tracer, and
        # a partial would freeze it into the callback as a python object
        # (TracerArrayConversionError at the first body run).
        return jax.pure_callback(
            _live_face_host,
            (jax.ShapeDtypeStruct((W,), jnp.int32),
             jax.ShapeDtypeStruct((W,), jnp.int32),
             jax.ShapeDtypeStruct((), jnp.int32)),
            order, spec_hist, step_count, vertex_idx, vertex_specs,
            face_rows, face_skips, f, vmap_method="sequential",
        )

    def _live_face_count_host(order, spec_hist, step_count, vertex_idx):
        return np.int32(_LIVE_FACES.n_faces(
            order, spec_hist, int(np.asarray(step_count)),
            int(np.asarray(vertex_idx)) + 1))

    def _live_face_count(order, spec_hist, step_count, vertex_idx):
        return jax.pure_callback(
            _live_face_count_host, jax.ShapeDtypeStruct((), jnp.int32),
            order, spec_hist, step_count, vertex_idx,
            vmap_method="sequential")

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
    cosine_idx = REWARD_INDEX["cosine_sim"]
    # ``--reward-mode mult``: cost weights for the cheapness term = the display
    # weights with the quality channels zeroed (the gate multiplies fidelity
    # back in); the preference collapses to one-hot on the cosine head so the
    # scalarization recovers the gated scalar exactly.
    mult_cost_weights_np = reward_weights_np.copy()
    mult_cost_weights_np[cosine_idx] = 0.0
    mult_cost_weights = jnp.asarray(mult_cost_weights_np, dtype=jnp.float32)
    if args.reward_mode == "mult":
        head_reward_weights_np = np.zeros(NUM_VALUE_HEADS, dtype=np.float32)
        head_reward_weights_np[HEAD_NAMES.index("cos")] = 1.0
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
    schedule = optax.cosine_decay_schedule(
        args.lr,
        args.episodes * args.ppo_epochs * args.minibatches,
        args.lr_decay_min_mult,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(schedule, b1=args.adam_b1, eps=args.adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

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
        vertex_features,
        preference,
        op_legality_override,
        pin_rules_to_exact_jax,
        vertex_temperature=None,
    ):
        keys = jrand.split(key, rollout_length)
        # Stage B.4: per-vertex residual state, initialised to zero at episode
        # start. Carried through the rollout's scan alongside env_state.
        init_residual = jnp.zeros((total_v, args.embd_dim), dtype=jnp.float32)

        encode_key, scan_key = jrand.split(keys[0], 2)
        initial_tokens = None
        initial_eqn_ids = None

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

        # 3b: consume the initial base stream ONCE (window = MAX_TOKENS),
        # fold its rows into the per-vertex memory (base eqn ids map to
        # vertex slots positionally; structural/overflow → global slot),
        # then each scan step extends by that step's delta only.
        if getattr(args, "incremental_encode", False):
            _enc0 = agent.carry_init()
            _base_count = _stream_len(env_state.tokens)
            _enc1, _rows0, _valid0, _eqns0 = agent.encode_extend(
                _enc0, env_state.tokens, env_state.eqn_ids, _base_count,
                window=MAX_TOKENS,
            )
            _base_ids = jnp.where(
                (_eqns0 >= 0) & (_eqns0 < total_v), _eqns0, -1
            )
            _vs0 = jnp.zeros((total_v + 1, args.embd_dim), jnp.float32)
            _vc0 = jnp.zeros((total_v + 1,), jnp.float32)
            _vs0, _vc0 = _vmem.update_ids(_vs0, _vc0, _rows0, _base_ids, _valid0)
            init_enc_state = (_enc1, _vs0, _vc0)
        else:
            init_enc_state = None

        def step_fn(carry, k):
            state, residual_state, elim_order, enc_state = carry
            sample_key, next_net_key = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid
            )

            precomputed = None
            if args.incremental_encode:
                enc_carry, vmem_s, vmem_c = enc_state
                # The delta appended since the carry's pos was emitted by the
                # PREVIOUS step's elimination (empty at step 0 — the base
                # stream is already consumed).
                delta_count = _stream_len(state.tokens) - enc_carry.pos
                delta_owner = jnp.where(
                    state.step_count > 0,
                    elim_order[jnp.maximum(state.step_count - 1, 0)],
                    jnp.array(-1, jnp.int32),
                ).astype(jnp.int32)
                enc_carry2, d_rows, d_valid, d_eqns = agent.encode_extend(
                    enc_carry, state.tokens, state.eqn_ids, delta_count,
                    window=MAX_DELTA_TOKENS,
                )
                d_ids = jnp.where(d_eqns >= 0, delta_owner, -1)
                vmem_s2, vmem_c2 = _vmem.update_ids(
                    vmem_s, vmem_c, d_rows, d_ids, d_valid
                )
                precomputed = agent.heads_from_memory(
                    vmem_s2, vmem_c2,
                    vertex_features=vertex_features,
                    residual_state=residual_state,
                    preference=(
                        preference if args.preference_conditioned else None
                    ),
                )

            face_chunk_fn = None
            face_count_fn = None
            if _LIVE_FACES is not None:
                _fc_o, _fc_s, _fc_k = (
                    state.order, state.sparsity_specs, state.step_count)

                def face_chunk_fn(_f, _v, _vspecs, _rows, _skips,
                                  _o=_fc_o, _s=_fc_s, _k=_fc_k):
                    return _live_face(_f, _o, _s, _k, _v,
                                      _vspecs, _rows, _skips)

                def face_count_fn(_v, _o=_fc_o, _s=_fc_s, _k=_fc_k):
                    return _live_face_count(_o, _s, _k, _v)

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
                    state.tokens,
                    vertex_avail_mask,
                    state.axis_state,
                    state.axis_valid_mask,
                    factor_tables,
                    op_legality_override,
                    sample_key,
                    eqn_ids=state.eqn_ids,
                    vertex_features=vertex_features,
                    residual_state=residual_state,
                    preference=preference if args.preference_conditioned else None,
                    oracle_pair_all=oracle_pair_all,
                    oracle_comp_all=oracle_comp_all,
                    face_masks_all=face_masks_all,
                    vertex_temperature=vertex_temperature,
                    precomputed=precomputed,
                    oracle_fn=oracle_one_fn,
                    face_chunk_fn=face_chunk_fn,
                    face_count_fn=face_count_fn,
                    enc_carry=(enc_carry2 if args.incremental_encode
                               else None),
                )
                # Record this vertex in the elimination prefix for the next
                # step's oracle replay.
                elim_order = elim_order.at[state.step_count].set(
                    vertex_idx.astype(elim_order.dtype))
                if face_out is not None:
                    (face_action, face_old_logp, _face_ent, face_pair_v,
                     face_comp_v, face_valid_v, face_cnt_v, face_dt_v,
                     face_de_v) = face_out
                else:
                    face_action = _zero_face_action()
                    face_old_logp = jnp.array(0.0)
                    face_pair_v = jnp.zeros(
                        (ENV_MAX_FACES, MAX_AXES_PER_VERTEX,
                         MAX_AXES_PER_VERTEX), jnp.float32)
                    face_comp_v = jnp.zeros(
                        (ENV_MAX_FACES, MAX_AXES_PER_VERTEX), jnp.float32)
                    face_valid_v = jnp.zeros((ENV_MAX_FACES,), jnp.float32)
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
            rewards = raw_rewards
            done = env_out.terminated.astype(jnp.float32)
            # Stage C potential-based shaping. Bootstraps a denser per-step
            # learning signal from the critic; provably preserves the
            # optimum (Ng et al. 1999). The shaping uses raw (un-symlog'd)
            # value space and is stop-gradient'd so the value head only
            # trains against the original returns. Disabled when the
            # coefficient is zero.

            new_residual = agent.update_residual(
                residual_state,
                vertex_idx,
                v_context,
            )

            pref_arg = preference if args.preference_conditioned else None
            if args.incremental_encode:
                # Bootstrap value at next_state: extend the post-decision
                # carry by the delta the JUST-CHOSEN elimination emitted.
                # This extension is recomputed by the next scan iteration
                # (kept self-contained rather than threading heads across
                # iterations); deltas are O(hundreds) of tokens, so the
                # duplicate extend is cheap next to a full re-encode.
                nv_count = _stream_len(next_state.tokens) - enc_carry2.pos
                _, nv_rows, nv_valid, nv_eqns = agent.encode_extend(
                    enc_carry2, next_state.tokens, next_state.eqn_ids,
                    nv_count, window=MAX_DELTA_TOKENS,
                )
                nv_ids = jnp.where(nv_eqns >= 0, vertex_idx.astype(jnp.int32), -1)
                nv_s, nv_c = _vmem.update_ids(
                    vmem_s2, vmem_c2, nv_rows, nv_ids, nv_valid
                )
                _, _, next_value = agent.heads_from_memory(
                    nv_s, nv_c,
                    vertex_features=vertex_features,
                    residual_state=new_residual,
                    preference=pref_arg,
                )
            else:
                next_value = (
                    agent.value_for(
                        next_state.tokens,
                        eqn_ids=next_state.eqn_ids,
                        vertex_features=vertex_features,
                        residual_state=new_residual,
                        preference=pref_arg,
                        key=next_net_key,
                    )
                )

            # In cache-encoding mode the trajectory's tokens/eqn_ids fields
            # carry the *initial* episode tokens (constant across the
            # rollout) so the loss path can encode_once + decode_from_cache
            # against the same reference. In the legacy path they remain
            # the per-step residual jaxpr tokens.
            traj_tokens = (
                state.tokens
            ).astype(jnp.int32)
            traj_eqn_ids = (
                state.eqn_ids
            ).astype(jnp.int32)

            if args.incremental_encode:
                # PRE-step snapshots (§7b): the carry/memory synced to the
                # PREVIOUS step's stream — the loss re-derives this step's
                # encoding by the same delta extension.
                _enc_fields = dict(
                    enc_M=enc_carry.M, enc_I=enc_carry.I,
                    enc_cumhist=enc_carry.cumhist,
                    enc_nvalid=enc_carry.nvalid, enc_pos=enc_carry.pos,
                    vmem_sums=vmem_s, vmem_counts=vmem_c,
                    delta_owner=delta_owner,
                )
            else:
                _enc_fields = _zero_enc_carry_fields()

            transition = Trajectory(
                tokens=traj_tokens,
                eqn_ids=traj_eqn_ids,
                residual_state=residual_state,
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
                face_old_logp=jnp.asarray(face_old_logp, jnp.float32),
                face_counts=face_cnt_v,
                face_delta_tokens=face_dt_v,
                face_delta_eqns=face_de_v,
                **_enc_fields,
                discount=jnp.array(args.discount),
                vertex_avail_mask=vertex_avail_mask,
            )
            next_enc_state = (
                (enc_carry2, vmem_s2, vmem_c2)
                if args.incremental_encode
                else enc_state
            )
            return (
                (next_state, new_residual, elim_order, next_enc_state),
                (transition, raw_rewards),
            )

        (final_state, _, _, _), (traj, all_raw_rewards) = lax.scan(
            step_fn,
            (env_state, init_residual, jnp.zeros((total_v,), dtype=jnp.int32),
             init_enc_state),
            keys,
        )
        return final_state, traj, all_raw_rewards[-1]

    def loss_fn(
        agent,
        batch: TrainBatch,
        vertex_features,
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
                agent, batch, vertex_features, key, op_legality_override
            )
    def _dynamic_loss_fn(
        agent, batch: TrainBatch, vertex_features, key, op_legality_override
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

        cached_flat = None
        keys = jrand.split(key, batch.tokens.shape[0])

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

        def _eval_dyn(toks, eids, rs, pref, vidx, action, vmask, ax_st, ax_vm,
                      cached, k, pv, cv, fa=None, fpv=None, fcv=None, fv=None,
                      pc3=None, fch=None, fcy=None):
            return agent.evaluate_action_dynamic(
                toks,
                vidx,
                action,
                vmask,
                ax_st,
                ax_vm,
                factor_tables,
                k,
                eqn_ids=eids,
                vertex_features=vertex_features,
                residual_state=rs,
                cached_encoding=cached,
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
                precomputed=pc3,
                face_chunks=fch,
                face_carry=fcy,
            )

        if args.incremental_encode:
            # 3b: re-derive each sample's encoding by extending its stored
            # PRE-step carry with the delta tokens (gather-sliced from the
            # stored stream at the carried pos), fold into the stored vertex
            # memory, and run the heads — the exact computation the rollout
            # sampled under, through CURRENT params (ratio 1 at epoch 0;
            # gradient flows through the delta + heads, truncating at the
            # stored carry by design).
            def _carry_heads(toks, eids, M, I, ch, nv, pos, owner, vs, vc,
                             rs, pref):
                carry = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
                count = _stream_len(toks) - pos
                carry2, rows, valid, eqw = agent.encode_extend(
                    carry, toks, eids, count, window=MAX_DELTA_TOKENS
                )
                ids = jnp.where(eqw >= 0, owner, -1)
                vs2, vc2 = _vmem.update_ids(vs, vc, rows, ids, valid)
                # carry2 is where the sampling side carry branched: the
                # stored pre-step carry advanced past the PREVIOUS delta.
                # The face replay continues from it over the stored emission
                # window. (The rows above are the previous delta's -- the
                # WRONG tokens for face contexts; see face_delta_tokens.)
                return agent.heads_from_memory(
                    vs2, vc2,
                    vertex_features=vertex_features,
                    residual_state=rs,
                    preference=pref_or_none(pref),
                ) + (carry2,)

            pc_logits, pc_ctx, pc_value, pc_carry = jax.vmap(_carry_heads)(
                batch.tokens, batch.eqn_ids,
                batch.enc_M, batch.enc_I, batch.enc_cumhist,
                batch.enc_nvalid, batch.enc_pos, batch.delta_owner,
                batch.vmem_sums, batch.vmem_counts,
                batch.residual_state, batch.preference,
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
            ) = (
                jax.vmap(
                    lambda toks, eids, rs, pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv, fa, fpv, fcv, fv, pl, pc, pvl,
                    fct, fdt, fde, cy:
                    _eval_dyn(
                        toks, eids, rs, pref, vidx, action, vmask, ax_st,
                        ax_vm, None, k, pv, cv, fa, fpv, fcv, fv,
                        pc3=(pl, pc, pvl),
                        # Chunk lengths + THIS step's emission window + the
                        # branch-point carry: everything the behaviour
                        # policy's face contexts were built from. Anything
                        # else and the ratio is not 1.
                        fch=((fct, fdt, fde) if _LIVE_FACES is not None
                             else None),
                        fcy=(cy if _LIVE_FACES is not None else None),
                    )
                )(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
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
                    pc_logits, pc_ctx, pc_value,
                    batch.face_counts,
                    batch.face_delta_tokens, batch.face_delta_eqns,
                    pc_carry,
                )
                if args.face_actions
                else jax.vmap(
                    lambda toks, eids, rs, pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv, pl, pc, pvl:
                    _eval_dyn(
                        toks, eids, rs, pref, vidx, action, vmask, ax_st,
                        ax_vm, None, k, pv, cv, pc3=(pl, pc, pvl)
                    )
                )(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
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
        elif cached_flat is None:
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
            ) = (
                jax.vmap(
                    lambda toks, eids, rs, pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv, fa, fpv, fcv, fv:
                    _eval_dyn(
                        toks, eids, rs, pref, vidx, action, vmask, ax_st,
                        ax_vm, None, k, pv, cv, fa, fpv, fcv, fv
                    )
                )(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
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
                )
                if args.face_actions
                else jax.vmap(
                    lambda toks, eids, rs, pref, vidx, action, vmask, ax_st,
                    ax_vm, k, pv, cv:
                    _eval_dyn(
                        toks, eids, rs, pref, vidx, action, vmask, ax_st,
                        ax_vm, None, k, pv, cv
                    )
                )(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
                    batch.preference,
                    batch.vertex_idx,
                    actions,
                    batch.vertex_avail_mask,
                    batch.axis_state,
                    batch.axis_valid_mask,
                    keys,
                    batch.micro_pair_valid,
                    batch.micro_compress_valid,
                )
            )
        else:
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
            ) = jax.vmap(_eval_dyn)(
                batch.tokens,
                batch.eqn_ids,
                batch.residual_state,
                batch.preference,
                batch.vertex_idx,
                actions,
                batch.vertex_avail_mask,
                batch.axis_state,
                batch.axis_valid_mask,
                cached_flat,
                keys,
                batch.micro_pair_valid,
                batch.micro_compress_valid,
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
        # 5-slot layout the legacy loss path returns.
        _entropy_components = (
            ent_vertex, ent_op, ent_i, ent_j, ent_exp + ent_kind + ent_quant,
        )

        total_loss = (
            ppo_loss
            + args.value_weight * value_loss
            - args.entropy_weight * entropy_loss
        )

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
        return total_loss, (
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
            # Per-component entropies (same 5-slot layout); legacy returns
            # zeros for the dynamic-only slots and the vertex entropy in
            # slot 0 if available.
            jnp.stack(_entropy_components),
        )

    def train_episode(
        agent,
        opt_state,
        env_states,
        env_obj,
        vertex_features,
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
    ):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, num_envs)

        env_states, traj, total_rewards_full = rollout_fn(
            agent,
            env_obj,
            num_valid,
            env_states,
            rollout_keys,
            vertex_features,
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
            # own (normalized) prediction so the value loss for that step is
            # ~0 and the critic is not dragged toward the sentinel.
            estim_returns = jnp.where(
                _live > 0.5, (estim_returns - new_mu) / new_sigma, traj.value)
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
        full_batch = TrainBatch(
            tokens=traj.tokens,
            eqn_ids=traj.eqn_ids,
            residual_state=traj.residual_state,
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
            face_old_logp=traj.face_old_logp,
            face_counts=traj.face_counts,
            face_delta_tokens=traj.face_delta_tokens,
            face_delta_eqns=traj.face_delta_eqns,
            enc_M=traj.enc_M,
            enc_I=traj.enc_I,
            enc_cumhist=traj.enc_cumhist,
            enc_nvalid=traj.enc_nvalid,
            enc_pos=traj.enc_pos,
            vmem_sums=traj.vmem_sums,
            vmem_counts=traj.vmem_counts,
            delta_owner=traj.delta_owner,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
            vertex_avail_mask=traj.vertex_avail_mask,
        )

        dynamic_carry, static_carry = eqx.partition((agent, opt_state), eqx.is_array)

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
        use_traj_batch = False

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
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
                    comb_agent,
                    batch,
                    vertex_features,
                    t_key,
                    pin_rules_to_exact_arg,
                    op_legality_override_arg,
                )
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
                next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
                return (next_carry, step + 1), metrics

            return lax.scan(mb_step_fn, (carry, step), (batches, mb_keys))

        epoch_keys = jrand.split(subkey, args.ppo_epochs)
        (dynamic_carry, final_step), metrics_seq = lax.scan(
            epoch_step_fn,
            (dynamic_carry, global_step),
            epoch_keys,
        )

        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
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
        )
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
            acc_val = float(arr[cosine_idx])  # cosine ∈ [0, 1]
            print(
                f"{rank}. Ep {ep} | Total Reward: {total_ret:.2e} | "
                f"CMP({args.cmp_type}): {cmp_val:.2e} | Acc: {acc_val:.4f} | "
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
            print(f"   Sequence (vertex, [calls...]): {seq_repr}")
            if table is not None:
                table.add_data(
                    rank, ep, total_ret, cmp_val, acc_val, mem_val, seq_repr
                )
        if table is not None:
            wandb.log({f"Top N {name}": table})

    def host_log(
        ep, all_rets, actions_pack, mean_r, mets, diag_pack=None,
        popart_stats=None,
    ):
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

        mean_r = np.atleast_1d(np.array(mean_r))

        host_state["samplecounts"] += num_envs * num_valid
        # mets is an 11-tuple: 9 scalars + two (5,) per-component arrays.
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
            # mem uses the canonical memory index, and acc tracks cosine_sim.
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

        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": float(np.sum(mean_r * weights)),
            # TRUNCATED = refused by a resource limit (op cap / OOM) and
            # EXCLUDED from the gradient — "Time Limits in RL". ZERO_WORK =
            # computed nothing but was KEPT and punished by frob. They are
            # opposite treatments, so they get separate counters.
            "collapse/truncated_this_ep": consume_truncated_plan_count(),
            # UNTRACEABLE is a SUBSET of truncated: graphax could not build the
            # plan at all (the open canonical-output-order gap). Separated
            # because OOM scales with plan size and this scales with nothing we
            # control -- if it is a large fraction, the approx arm is sampling a
            # region the library cannot evaluate and the run is not comparable.
            "collapse/untraceable_this_ep": consume_untraceable_plan_count(),
            "collapse/zero_work_this_ep": consume_zero_work_plan_count(),
            "collapse/count_this_ep": n_collapsed_this_ep,
            "collapse/count_total": host_state["collapsed_total"],
            "collapse/fraction_this_ep": n_collapsed_this_ep / max(1, all_rets.shape[0]),
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "explained variance": explained_var,
            "sample count": host_state["samplecounts"],
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
                log_dict["entropy/micro_approx"] = float(
                    np.mean(entropy_components[1:5])
                )
                # NORMALISED companions, in [0,1] = fraction of that head's
                # MAXIMUM possible entropy. The raw nats are not comparable
                # across heads: the vertex head picks among ~total_v vertices
                # (ln 13 = 2.56) while the op head picks among 4 (ln 4 = 1.39),
                # so macro sat at 1-1.6 and micro at 0-0.3 because of ALPHABET
                # SIZE, not because the micro heads were more decided.
                # Only these two get a normaliser: the i/j/exp heads have a
                # DYNAMIC alphabet (the legal dim/factor set depends on the
                # live tensor), so there is no static maximum to divide by and
                # any fixed constant would be wrong.
                log_dict["entropy/macro_vertex_norm"] = float(
                    entropy_components[0] / np.log(max(int(total_v), 2)))
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
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(mean_r[j]) if j < len(mean_r) else 0.0

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
            _hr = np.asarray(
                [float(mean_r[k]) for k in HEAD_REWARD_INDICES], dtype=np.float64
            )
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
            log_dict["Charts/weighted_mean_return"] = float(np.sum(_phi * _wn))
            for j, nm in enumerate(HEAD_NAMES):
                if _hw[j] != 0.0:
                    log_dict[f"Charts/weighted_mean_{nm}"] = float(_phi[j])

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
        host_state["_wall_prev"] = _now

        # ---- per-face apply telemetry ---------------------------------------
        if args.per_face or args.face_actions:
            pf = consume_per_face_stats()
            if pf.get("applied", 0) or pf.get("skipped", 0):
                log_dict["per_face/applied"] = pf.get("applied", 0)
                log_dict["per_face/skipped"] = pf.get("skipped", 0)
                log_dict["per_face/skipped_raised"] = pf.get("skipped_raised", 0)
                log_dict["per_face/applied_fraction"] = pf["applied_fraction"]

        # ---- degenerate plans sentinelled by the env ------------------------
        _degen = consume_degenerate_plan_count()
        log_dict["collapse/degenerate_plans_this_ep"] = _degen

        # ---- XLA side-channel: xla_peak_memory + compression ratio ----------
        xla_stats = consume_xla_memory_stats()
        if xla_stats["count"]:
            log_dict["measure/xla_peak_memory"] = xla_stats["xla_peak_memory"]
            log_dict["measure/compression_ratio"] = xla_stats["compression_ratio"]

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
        log_dict["tokenization/truncated_count"] = trunc["count"]
        log_dict["tokenization/max_observed_len"] = trunc["max_observed_len"]
        log_dict["tokenization/overflow_sum_this_ep"] = trunc["overflow_sum"]
        # Token SIZE telemetry (not just loss). `delta_*` is one palimpsa
        # call's width in the append-only path and is what should size
        # ALPHAGRAD_MAX_DELTA_TOKENS; `stream_*` is the full re-read length
        # that MAX_TOKENS has to cover while the full-buffer path is in use.
        # Watch delta_max: incremental_token_delta RAISES rather than clips,
        # because silently truncating a delta would desync the recurrence.
        _tl = consume_token_length_stats()
        log_dict["tokens/stream_mean"] = _tl["stream_mean"]
        log_dict["tokens/stream_max"] = _tl["stream_max"]
        log_dict["tokens/delta_mean"] = _tl["delta_mean"]
        log_dict["tokens/delta_max"] = _tl["delta_max"]
        log_dict["tokens/delta_count"] = _tl["delta_count"]
        log_dict["tokens/delta_budget"] = MAX_DELTA_TOKENS
        log_dict["tokens/delta_headroom"] = MAX_DELTA_TOKENS - _tl["delta_max"]

        # ---- Pareto front + hypervolume (spec P2) ---------------------------
        # Objectives are logged in "higher is better" form, so the archive's
        # maximisation convention applies directly.
        if pareto_archive is not None and elig_rets.shape[0]:
            elig_idx = [i for i in range(all_rets.shape[0]) if eligible[i]]
            pareto_archive.add_many(
                ((all_rets[i], _decode(i)) for i in elig_idx), ep
            )
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
            log_dict["p_stop_slot0"] = float(p_stop_slot0)
            # Dynamic-substeps op-type marginals. Names MUST match
            # heads.py's op order (DIAG=0, COMPRESS=1, QUANT=2, END=3) —
            # this used to label index 2 "end", so the plotted "end" curve
            # was really QUANT and the true END mass was never logged, which
            # hid exactly the END-collapse mode the diagnostic exists for.
            for j, op_name in enumerate(("diag", "compress", "quant", "end")):
                if j < op_marginals.shape[0]:
                    log_dict[f"op_marginal/{op_name}"] = float(op_marginals[j])
            log_dict["sub_episode_length"] = float(mean_sub_episode_length)
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
                  "ratio/max_log=%.3g kl/approx=%.3g mu_cos=%.4g "
                  "sec/ep=%.1f" % (
                      _HEALTH_N[0] - 1, ppo_loss, value_loss, policy_entropy,
                      log_dict.get("ratio/max_log", float("nan")),
                      log_dict.get("kl/approx", float("nan")),
                      log_dict.get("popart/mu_cos", float("nan")),
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
                            f"  face_enum(ext/cold)={_fs['ext']}/"
                            f"{_fs['cold']}")
                        _ss.update(hit=0, ext=0, cold=0, nostore=0)
                        _fs.update(ext=0, cold=0)
                    except Exception:
                        _cache_line = ""
                    tqdm.write(
                        f"[prof ep={ep:3d}] host_total={_tot:6.1f}s  "
                        + "  ".join(f"{k}={v:.1f}s" for k, v in _items)
                        + _cache_line,
                        file=sys.stderr,
                    )
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

        # Stage B.2.A / B.3: per-vertex features depend on the calibration
        # samples; recompute them per episode. The dispatch (mean-aggregated
        # vs Set-Transformer per-sample) is centralised in
        # ``_episode_vertex_features``.
        vertex_features = _episode_vertex_features(
            args,
            closed_jaxpr.jaxpr,
            tuple(closed_jaxpr.literals),
            tuple(xs),
            eval_samples=eval_samples,
            argnums=tuple(argnums),
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
                _, _wtraj, _ = rollout_fn(
                    agent, env_episode, num_valid, _wstates,
                    jrand.split(_wkey, num_envs), vertex_features,
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
        ) = train_episode(
            agent,
            opt_state,
            env_states,
            env_episode,
            vertex_features,
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
        )
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
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])
    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
