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
    _AXIS_FEAT_IS_COMPRESSED,
    _AXIS_FEAT_IS_OUTPUT,
    _AXIS_FEAT_SIZE,
    MAX_AXES_PER_VERTEX,
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_AXIS_PAIRS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
    micro_actions_to_rule_specs_jax,
)
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
    FactorTables,
    MicroAction,
    MicroActionPolicy,
    precompute_factor_tables,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder, make_encoder
from alphagrad.transformer.encoder import RelationalMultiheadAttention
from alphagrad.utils import entropy, explained_variance

# ---------------------------------------------------------------------------
# Constants shared by all three agent variants
# ---------------------------------------------------------------------------

# NUM_AXIS_PAIRS axis pairs + 1 STOP token marking end of a rule sequence.
NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS

# Four-head value/advantage configuration. The value head emits one scalar
# per training reward — (V_flops, V_mem, V_cos, V_frob) — and the per-episode
# preference vector `w` (stored on Trajectory.preference) weights these
# advantages when scalarizing for the PPO loss. The mapping into the env's
# 8-component reward vector is fixed:
#   head 0  flops          (REWARD_INDEX["flops"])
#   head 1  peak_memory    (REWARD_INDEX["peak_memory"])
#   head 2  cosine_sim     (REWARD_INDEX["cosine_sim"])
#   head 3  frob_residual  (REWARD_INDEX["frob_residual"])
# HISTORY: this used to be 3 heads (flops, mem, frob) with the frob head
# LABELED "acc" — cosine similarity, the accuracy metric everyone watches,
# never entered the training signal at all. Combined with the bounded frob
# penalty, zeroing the computation maximized 2 of 3 trained channels and the
# policy collapsed to flops=0/cos=0 (run 55403). cosine_sim is now a real
# trained head; "acc" in --rewards weights IT, and frob keeps its own
# --lambda-frob weight.
# The remaining env-reward components are still emitted for host-side
# logging / top-N heaps but do not enter the value head or advantage path.
HEAD_REWARD_INDICES: tuple[int, ...] = (
    REWARD_INDEX["flops"],
    REWARD_INDEX["peak_memory"],
    REWARD_INDEX["cosine_sim"],
    REWARD_INDEX["frob_residual"],
)
NUM_VALUE_HEADS = len(HEAD_REWARD_INDICES)
HEAD_NAMES: tuple[str, ...] = ("flops", "mem", "cos", "frob")
_HEAD_REWARD_INDICES_ARR = jnp.asarray(HEAD_REWARD_INDICES, dtype=jnp.int32)

# Cross-channel scale handling. Reward channels span ~10¹⁰ in flops, ~10⁹
# in peak_memory, ~70 in frob_residual, ~1 in cosine_sim. Without
# per-channel normalization the flops gradient (1e10) drowns out cosine_sim
# (1) by ten orders of magnitude, and the policy learns to ignore quality.
# Fix is two-fold:
#   1) symlog every monotone channel before the per-step weighted sum;
#      compresses the dynamic range to ~25 / ~21 / ~5 / ~1 across channels.
#   2) calibration measures mean |symlog(r_i)| over K rollouts of the
#      un-trained agent and rescales reward_weights[i] by 1/mean_abs_i so
#      each weighted channel contributes on a comparable scale.
# cosine_sim is intentionally excluded from both: it is already bounded
# to [0, 1] and ~order 1, so symlog is a near-identity that only complicates
# the threshold semantics, and the user's CLI lambda for cosine is already
# in usable units (reward per unit of cosine similarity).
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)
_NO_SYMLOG_MASK: "jax.Array" = (
    jnp.zeros((NUM_REWARDS,), dtype=jnp.bool_)
    .at[jnp.asarray(_NO_SYMLOG_REWARD_INDICES, dtype=jnp.int32)]
    .set(True)
)
_NO_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=np.bool_)
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


def _symlog_rewards(reward_vec: "jax.Array") -> "jax.Array":
    """Apply symlog elementwise on the last axis, leaving cosine_sim raw.

    ``reward_vec`` has trailing dim ``NUM_REWARDS``. The mask is broadcast
    against any leading batch / time dims so the call is shape-agnostic.
    """
    return jnp.where(
        _NO_SYMLOG_MASK,
        reward_vec,
        reward_normalization_fn(reward_vec),
    )


def _apply_mult_gate(
    rewards: "jax.Array",
    cost_weights: "jax.Array",
    gate_tau: float,
    gate_w: float,
    anti_degen_penalty: float,
    anti_degen_tau: float,
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
    cos = rewards[..., REWARD_INDEX["cosine_sim"]]              # (E, T)
    denom = jnp.maximum(1.0 - gate_tau, 1e-6)
    g = jnp.clip((cos - gate_tau) / denom, 0.0, 1.0)

    cost_mag = -rewards                                          # (E, T, R)
    cost_sl = jnp.sign(cost_mag) * jnp.log1p(jnp.abs(cost_mag))
    weighted_cost = jnp.sum(cost_sl * cost_weights, axis=-1)     # (E, T)
    cheapness = jnp.maximum(0.0, gate_w - weighted_cost)
    gated = g * cheapness

    # Shaped anti-degeneracy penalty on the TERMINAL step only.
    terminal = jnp.zeros(rewards.shape[:2], dtype=bool).at[:, -1].set(True)
    degen = (cos < anti_degen_tau) & terminal
    cos_basin = jnp.clip(cos, 0.0, anti_degen_tau)
    shaped = -(anti_degen_penalty - cos_basin * anti_degen_penalty)
    gated = jnp.where(degen, shaped, gated)

    out = jnp.zeros_like(rewards)
    return out.at[..., REWARD_INDEX["cosine_sim"]].set(gated)

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


class Trajectory(NamedTuple):
    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array  # (V, embd_dim) at the start of this step
    preference: jax.Array  # (NUM_VALUE_HEADS,) — weights V_flops/V_mem/V_cos/V_frob
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
    """Encoder + composable (vertex policy, rule policy) + four value heads.

    The value head is split into four single-output MLPs, one per training
    reward: ``value_head_flops``, ``value_head_mem``, ``value_head_cos``,
    ``value_head_frob``. Their concatenation is the (NUM_VALUE_HEADS,) = (4,)
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
    value_head_flops: MLP
    value_head_mem: MLP
    value_head_cos: MLP
    value_head_frob: MLP
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
        value_head_frob,
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
    ):
        self.embedding = embedding
        self.pos_enc = pos_enc
        self.encoder = encoder
        self.vertex_policy = vertex_policy
        self.micro_action_policy = micro_action_policy
        self.value_head_flops = value_head_flops
        self.value_head_mem = value_head_mem
        self.value_head_cos = value_head_cos
        self.value_head_frob = value_head_frob
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
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        # Stage B.1: when `eqn_ids` is provided, the encoder layers add
        # learned per-relation biases derived from it. When None, the encoder
        # falls back to vanilla self-attention so the path is preserved for
        # callers that haven't been wired yet.
        enc_x = self.encoder(x, eqn_ids=eqn_ids, key=enc_key)

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
        v_frob = self.value_head_frob(summary)
        value = jnp.concatenate([v_flops, v_mem, v_cos, v_frob], axis=-1)
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
        if self.micro_action_policy is None:
            raise RuntimeError(
                "sample_action_dynamic called but micro_action_policy is "
                "None — agent was built without --dynamic-substeps."
            )
        net_key, vertex_key, micro_key = jrand.split(key, 3)
        vertex_logits, vertex_contexts, value = self.encode(
            tokens,
            eqn_ids=eqn_ids,
            vertex_features=vertex_features,
            residual_state=residual_state,
            preference=preference,
            key=net_key,
        )

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        if vertex_temperature is not None:
            masked_v_logits = masked_v_logits / vertex_temperature
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        vertex_idx = distrax.Categorical(probs=vertex_dist).sample(seed=vertex_key)

        v_context = vertex_contexts[vertex_idx]
        features = _axis_features_from_state(
            axis_state[vertex_idx],
            axis_valid_mask[vertex_idx],
        )

        # Live per-vertex DIAG/COMPRESS masks for the CHOSEN vertex (oracle rows
        # are 1-based; vertex_idx is 0-based). These are the authoritative masks
        # that keep the micro DIAG head from proposing a per-face-invalid pair;
        # stored (v_pair/v_comp) so the loss re-masks identically -> ratio 1.
        if oracle_pair_all is not None:
            v_pair = oracle_pair_all[vertex_idx + 1]
            v_comp = oracle_comp_all[vertex_idx + 1]
            _sample_pair, _sample_comp = v_pair, v_comp
        else:
            _N = MAX_AXES_PER_VERTEX
            v_pair = jnp.zeros((_N, _N), jnp.float32)
            v_comp = jnp.zeros((_N,), jnp.float32)
            _sample_pair, _sample_comp = None, None  # tag-bit fallback

        # The policy doesn't itself know about the override mask; we
        # wrap by zeroing COMPRESS legality on the features' tag_bits
        # *before* the encoder sees them. Simpler: apply the override
        # to the per-step op_legality after the policy computes it.
        # We do that by patching the head's `op_head` at call time —
        # but that's awkward. Instead, since op_legality is computed
        # inside MicroActionPolicy._step_sample, the override is
        # threaded through by post-multiplying the op_dist's logits.
        # For the MVP we mask op_legality at construction time using a
        # boolean wrapper: zero COMPRESS legality everywhere by setting
        # `tag_bits[..., TAG_IS_COMPRESSED] = 1` on every axis (so
        # `compress_eligible` is empty). That's heavy-handed; instead
        # we just rely on the trainer's `--allow-compress` gating:
        # below we filter the returned action sequence and force every
        # COMPRESS to END if the override says so.
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
        # NOTE: pair_valid/compress_valid/quant_legality_mask default to None
        # here — the tag-bit fallback and all-dtypes QUANT. Threading the
        # oracle's authoritative per-edge masks (build_pair_valid_mask, already
        # wired in the static path) into this dynamic path is the immediate
        # follow-up now that the heads accept them.
        ) = self.micro_action_policy.sample(
            v_context,
            features,
            factor_tables,
            micro_key,
            pair_valid=_sample_pair,
            compress_valid=_sample_comp,
        )

        # Apply the op-legality override post-hoc to DIAG / COMPRESS /
        # QUANT: any disallowed op_type is rewritten to END. The
        # MicroActionPolicy.sample doesn't take the override directly —
        # it computes legality from axis-state (always allowing DIAG /
        # COMPRESS / QUANT when applicable). The override is how the
        # `--variant` setting forces `ve_only` (no DIAG / COMPRESS / QUANT)
        # or `compress` (no DIAG) or similar variants at sampling time.
        diag_allowed = op_legality_override[OP_DIAG] > 0.5
        compress_allowed = op_legality_override[OP_COMPRESS] > 0.5
        quant_allowed = op_legality_override[OP_QUANT] > 0.5
        is_diag = actions.op_type == OP_DIAG
        is_compress = actions.op_type == OP_COMPRESS
        is_quant = actions.op_type == OP_QUANT
        disallowed = (
            (is_diag & ~diag_allowed)
            | (is_compress & ~compress_allowed)
            | (is_quant & ~quant_allowed)
        )
        rewritten_op = jnp.where(
            disallowed,
            jnp.full_like(actions.op_type, OP_END),
            actions.op_type,
        )
        # When op_type is rewritten to END the sampled i / j / exponents /
        # factor / kind / quant_dtype are stale. Zero them so the recorded
        # action is canonical (matches what sample_step produces for genuine
        # END outputs).
        zeros_i = jnp.zeros_like(actions.i)
        zeros_j = jnp.zeros_like(actions.j)
        zeros_exp = jnp.zeros_like(actions.exponents)
        zeros_f = jnp.zeros_like(actions.factor)
        zeros_k = jnp.zeros_like(actions.compress_kind)
        zeros_q = jnp.zeros_like(actions.quant_dtype)
        ones_qs = jnp.ones_like(actions.quant_scale_sign)
        actions = MicroAction(
            op_type=rewritten_op,
            i=jnp.where(disallowed, zeros_i, actions.i),
            j=jnp.where(disallowed, zeros_j, actions.j),
            exponents=jnp.where(disallowed[..., None], zeros_exp, actions.exponents),
            factor=jnp.where(disallowed, zeros_f, actions.factor),
            compress_kind=jnp.where(disallowed, zeros_k, actions.compress_kind),
            quant_dtype=jnp.where(disallowed, zeros_q, actions.quant_dtype),
            quant_scale_sign=jnp.where(disallowed, ones_qs, actions.quant_scale_sign),
        )

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
            value,
            v_context,
        )

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
    ):
        """Joint log-prob / entropy for a stored typed action sequence.

        Mirrors :meth:`evaluate_action` but uses
        :class:`MicroActionPolicy.evaluate` for the rule path. Returns
        ``(total_log_p, total_entropy, value, vertex_dist,
        op_dists, i_dists, j_dists, exp_dists, sub_episode_length)``.
        """
        if self.micro_action_policy is None:
            raise RuntimeError(
                "evaluate_action_dynamic called but micro_action_policy is None."
            )
        vertex_logits, vertex_contexts, value = self.encode(
            tokens,
            eqn_ids=eqn_ids,
            vertex_features=vertex_features,
            residual_state=residual_state,
            preference=preference,
            key=key,
        )

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        log_p_vertex = jnp.log(vertex_dist[vertex_idx] + 1e-8)
        vertex_ent = entropy(vertex_dist)

        v_context = vertex_contexts[vertex_idx]
        features = _axis_features_from_state(
            axis_state[vertex_idx],
            axis_valid_mask[vertex_idx],
        )

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
        # Masks MUST match sample_action_dynamic (the stored live oracle masks;
        # all-dtypes QUANT) or the PPO ratio is not 1 at epoch 0.
        ) = self.micro_action_policy.evaluate(
            v_context,
            features,
            factor_tables,
            actions,
            pair_valid=pair_valid,
            compress_valid=compress_valid,
        )

        total_log_p = log_p_vertex + log_p_sub
        total_entropy = vertex_ent + ent_sub
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
        )
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=rule_specs,
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
    # select. `--lambda-frob` is new and defaults to 0 (Frobenius residual is
    # measured but not weighted unless explicitly opted into).
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
        help="Weight on the Frobenius-residual value head. Default 0.",
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
        "compute per-head GAE on (flops, peak_memory, frob_residual), and "
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
    pos_enc = PositionalEncoder(args.embd_dim, MAX_TOKENS)
    # Flag-gated token-mixer for the policy backbone. Default "transformer"
    # keeps behaviour byte-identical; ALPHAGRAD_POLICY=palimpsa swaps the
    # encoder self-attention for the verified Palimpsa linear-attention kernel.
    _policy = os.environ.get("ALPHAGRAD_POLICY", "transformer").strip().lower()
    if _policy not in ("transformer", "palimpsa", "palimpsa_bi"):
        raise ValueError(
            "ALPHAGRAD_POLICY must be 'transformer', 'palimpsa' or "
            f"'palimpsa_bi', got {_policy!r}"
        )
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
    vertex_policy = PointerVertexPolicy(
        num_vertices=total_v,
        embd_dim=args.embd_dim,
        num_heads=args.num_heads,
        key=encoder_keys[2],
    )
    # One single-output MLP per training reward (flops / peak_memory /
    # cosine_sim / frob_residual). Per-head split keeps gradient scales sane
    # across the qualitatively different reward families and matches the
    # per-head GAE and preference-vector scalarization in `train_episode`.
    value_dims = _parse_int_list(args.value_dims)
    value_head_flops = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[4])
    value_head_mem = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[5])
    value_head_cos = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[12])
    value_head_frob = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[14])
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
    if getattr(args, "dynamic_substeps", False):
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
    return Agent(
        embedding=embedding,
        pos_enc=pos_enc,
        encoder=encoder,
        vertex_policy=vertex_policy,
        value_head_flops=value_head_flops,
        value_head_mem=value_head_mem,
        value_head_cos=value_head_cos,
        value_head_frob=value_head_frob,
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
    )


def _scale_output_heads(agent, scale: float):
    """Scale policy-head weights so the initial action distribution is near-uniform."""
    agent = scale_module_weight(
        agent, lambda a: a.vertex_policy.pointer_proj.weight, scale
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
    return agent


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



def _build_reward_weights(args) -> np.ndarray:
    """Map legacy CLI flags to a (NUM_REWARDS,) advantage-weight vector.

    `--rewards` selects which families contribute; within a family the weight
    lands on the canonical component picked by `--cmp-type` / `--mem-type`.
    Quality terms: cosine gets weight 1.0 (matching legacy behaviour) when
    "acc" is in `--rewards`; Frobenius gets `--lambda-frob` (default 0).

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
    if args.lambda_frob != 0.0:
        weights[REWARD_INDEX["frob_residual"]] = args.lambda_frob
    return weights


def _build_head_weights(args) -> np.ndarray:
    """Build the (NUM_VALUE_HEADS,) = (4,) static preference vector.

    Indexes the four training rewards (flops / peak_memory / cosine_sim /
    frob_residual) — the value head and advantage path operate on exactly
    these. ``"acc" in --rewards`` weights the COSINE head (this used to
    silently weight frob while cosine never trained — the root cause of the
    zero-compute collapse); frob keeps its own ``--lambda-frob``.
    The `--cmp-type` and `--mem-type` flags only affect host-side display.
    """
    weights = np.zeros(NUM_VALUE_HEADS, dtype=np.float32)
    if "cmp" in args.rewards:
        weights[0] = args.lambda_cmp
    if "mem" in args.rewards:
        weights[1] = args.lambda_mem
    if "acc" in args.rewards:
        weights[2] = args.lambda_acc
    if args.lambda_frob != 0.0:
        weights[3] = args.lambda_frob
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
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(
        args.example, dataset=dataset_for_call, dataset_size=args.dataset_size
    )
    closed_jaxpr = _traced_inlined(target_fn, xs)
    # Always pass target_fun so flops/bytes_accessed/latency_ns/peak_memory
    # populate every step (see cpu_approx_worker.py for the full rationale).
    env_target_fun = target_fn
    argnums = infer_argnums(args.example)

    # Latency is the only optional component of the reward harness; auto-enable
    # measurement when the user has selected it as their primary compute metric
    # so the reward isn't silently zeroed out.
    measure_latency = args.measure_latency or args.cmp_type == "latency"
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
        latency_samples=int(getattr(args, "latency_samples", 1)),
        terminal_rewards_only=args.terminal_rewards_only,
    )

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

    def _oracle_masks_host(order, spec_hist, step_count):
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
    # value / advantage path operates on the 4-vec (flops / peak_memory /
    # cosine_sim / frob_residual); see HEAD_REWARD_INDICES and
    # `_build_head_weights`.
    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
    head_reward_weights_np = _build_head_weights(args)
    head_reward_weights = jnp.asarray(head_reward_weights_np, dtype=jnp.float32)
    cmp_idx = _cmp_reward_index(args.cmp_type)
    mem_idx = _mem_reward_index(args.mem_type)
    cosine_idx = REWARD_INDEX["cosine_sim"]
    frob_idx = REWARD_INDEX["frob_residual"]
    # ``--reward-mode mult``: cost weights for the cheapness term = the display
    # weights with the quality channels zeroed (the gate multiplies fidelity
    # back in); the preference collapses to one-hot on the cosine head so the
    # scalarization recovers the gated scalar exactly.
    mult_cost_weights_np = reward_weights_np.copy()
    mult_cost_weights_np[cosine_idx] = 0.0
    mult_cost_weights_np[frob_idx] = 0.0
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
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, 0, None, None))
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

        def step_fn(carry, k):
            state, residual_state, elim_order = carry
            sample_key, next_net_key = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid
            )

            if args.dynamic_substeps:
                # Live per-vertex DIAG/COMPRESS masks for the current graph
                # (replayed from the elimination prefix so far).
                oracle_pair_all, oracle_comp_all = _oracle_masks(
                    state.order, state.sparsity_specs, state.step_count)
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
                )
                # Record this vertex in the elimination prefix for the next
                # step's oracle replay.
                elim_order = elim_order.at[state.step_count].set(
                    vertex_idx.astype(elim_order.dtype))
                env_action = agent.to_env_action_dynamic(
                    vertex_idx,
                    micro_actions,
                    state.axis_state,
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
                discount=jnp.array(args.discount),
                vertex_avail_mask=vertex_avail_mask,
            )
            return (next_state, new_residual, elim_order), (transition, raw_rewards)

        (final_state, _, _), (traj, all_raw_rewards) = lax.scan(
            step_fn,
            (env_state, init_residual, jnp.zeros((total_v,), dtype=jnp.int32)),
            keys,
        )
        return final_state, traj, all_raw_rewards[-1]

    def loss_fn(
        agent,
        batch: TrainBatch,
        vertex_features,
        key,
        pin_rules_to_exact_jax,
    ):
        # Dynamic-substeps path branches off here so the legacy path
        # stays exactly as written. `_dynamic_loss_fn` lives below and
        # mirrors the same return shape — total_loss + 10-tuple of
        # metrics — so the train_episode plumbing doesn't care which
        # path was taken. (The dynamic path doesn't use pin_rules_to_exact;
        # the JAX-traced arg is ignored there.)
        if args.dynamic_substeps:
            return _dynamic_loss_fn(agent, batch, vertex_features, key)
    def _dynamic_loss_fn(agent, batch: TrainBatch, vertex_features, key):
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
        )

        def _eval_dyn(toks, eids, rs, pref, vidx, action, vmask, cached, k, pv, cv):
            return agent.evaluate_action_dynamic(
                toks,
                vidx,
                action,
                vmask,
                env.axis_state_static,
                env.axis_valid_static,
                factor_tables,
                k,
                eqn_ids=eids,
                vertex_features=vertex_features,
                residual_state=rs,
                cached_encoding=cached,
                preference=pref_or_none(pref),
                pair_valid=pv,
                compress_valid=cv,
            )

        if cached_flat is None:
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
            ) = jax.vmap(
                lambda toks, eids, rs, pref, vidx, action, vmask, k, pv, cv:
                _eval_dyn(
                    toks, eids, rs, pref, vidx, action, vmask, None, k, pv, cv
                )
            )(
                batch.tokens,
                batch.eqn_ids,
                batch.residual_state,
                batch.preference,
                batch.vertex_idx,
                actions,
                batch.vertex_avail_mask,
                keys,
                batch.micro_pair_valid,
                batch.micro_compress_valid,
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

        ratio = jnp.exp(log_probs - old_log_probs)
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
        entropy_loss = jnp.mean(entropies / jnp.maximum(sub_lengths, 1.0))

        # See the legacy loss path's value-mode switch for the rationale;
        # in scalar mode only slot 0 of (values, estim_returns) is alive.
        if args.loss_mode == "scalar":
            value_loss = jnp.mean(
                (values[..., 0] - reward_normalization_fn(batch.estim_returns[..., 0]))
                ** 2
            )
            explained_var = explained_variance(
                values[..., 0], batch.estim_returns[..., 0]
            )
        else:
            value_loss = jnp.mean(
                jnp.sum(
                    (values - reward_normalization_fn(batch.estim_returns)) ** 2,
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
        _kl_components = (kl_vertex, kl_op, kl_i, kl_j, kl_exp + kl_kind + kl_quant)

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
            )
        sl_reward = _symlog_rewards(traj_reward)  # (E, T, NUM_REWARDS)
        if args.loss_mode == "scalar":
            scalar_reward = jnp.sum(sl_reward * reward_weights, axis=-1)  # (E, T)
            zeros = jnp.zeros_like(scalar_reward)
            head_rewards = jnp.stack(
                [scalar_reward] + [zeros] * (NUM_VALUE_HEADS - 1), axis=-1
            )
        else:
            # ``HEAD_REWARD_INDICES`` selects (flops, peak_memory,
            # cosine_sim, frob_residual). cosine passes through symlog
            # unchanged (bounded [0,1] already); the cost channels are
            # symlog'd. The value head still learns the symlog of
            # estim_returns in the value loss; the GAE math in ``gae.py``
            # treats values as symlog'd (symexp back to "raw" — but with
            # the rewards now in symlog space, "raw" here is the symlog
            # scale, which is stable in the 10²-ish range).
            head_rewards = sl_reward[..., _HEAD_REWARD_INDICES_ARR]
        _, estim_returns, advantages = get_advantages(
            head_rewards,
            traj.done,
            traj.value,
            traj.next_value,
            traj.discount,
            args.gae_lambda,
        )

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
        )

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    # Reporting.
    wandb.init(
        project=getattr(args, "wandb_project", None) or "dsnn-vertex",
        entity=getattr(args, "wandb_entity", None) or None,
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else args.wandb,
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
                    "frob",
                    "sequence",
                ]
            )
        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            arr = np.array(rets)
            total_ret = float(np.sum(arr * weights))
            cmp_val = -float(arr[cmp_idx])  # display as positive cost
            mem_val = -float(arr[mem_idx])  # display as positive cost
            acc_val = float(arr[cosine_idx])  # cosine ∈ [0, 1]
            frob_val = -float(arr[frob_idx])  # display as positive residual
            print(
                f"{rank}. Ep {ep} | Total Reward: {total_ret:.2e} | "
                f"CMP({args.cmp_type}): {cmp_val:.2e} | Acc: {acc_val:.4f} | "
                f"Mem({args.mem_type}): {mem_val:.2e} | Frob: {frob_val:.4e}"
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
                    rank, ep, total_ret, cmp_val, acc_val, mem_val, frob_val, seq_repr
                )
        if table is not None:
            wandb.log({f"Top N {name}": table})

    def host_log(
        ep, all_rets, actions_pack, mean_r, mets, diag_pack=None
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
            collapsed = bool(
                rets[cmp_idx] >= 0.0
                or rets[mem_idx] >= 0.0
                or rets[cosine_idx] <= 1e-6
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
                    or all_rets[i][cosine_idx] <= 1e-6
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
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(mean_r[j]) if j < len(mean_r) else 0.0

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
            ) = (np.asarray(x) for x in diag_pack)
            for j, p in enumerate(pair_marg):
                log_dict[f"pair_marginal/{j}"] = float(p)
            for j, p in enumerate(factor_marg):
                log_dict[f"factor_marginal/{j}"] = float(p)
            for j, name in enumerate(HEAD_NAMES):
                log_dict[f"preference/{name}"] = float(pref_mean[j])
            log_dict["p_stop_slot0"] = float(p_stop_slot0)
            # Dynamic-substeps op-type marginals — DIAG / COMPRESS / END.
            # Zero in legacy mode (filled with zeros by train_episode).
            for j, op_name in enumerate(("diag", "compress", "end")):
                log_dict[f"op_marginal/{op_name}"] = float(op_marginals[j])
            log_dict["sub_episode_length"] = float(mean_sub_episode_length)
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
        b_ret_unnorm = np.abs(all_rets[best_idx])
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
        (
            agent,
            opt_state,
            _,
            metrics,
            total_rewards_full,
            actions_pack,
            global_step,
            diag_pack,
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
        )
        host_log(
            ep,
            total_rewards_full,
            actions_pack,
            jnp.mean(total_rewards_full, axis=0),
            metrics,
            diag_pack,
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

    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n(f"CMP (Lowest {args.cmp_type})", host_state["top_n_cmp"])
    print_top_n(f"Memory (Lowest {args.mem_type})", host_state["top_n_mem"])
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])
    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
