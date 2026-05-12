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
    OP_COMPRESS,
    OP_DIAG,
    OP_END,
    AxisTokenFeatures,
    FactorTables,
    MicroAction,
    MicroActionPolicy,
    precompute_factor_tables,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.transformer.encoder import RelationalMultiheadAttention
from alphagrad.utils import entropy, explained_variance

# ---------------------------------------------------------------------------
# Constants shared by all three agent variants
# ---------------------------------------------------------------------------

# NUM_AXIS_PAIRS axis pairs + 1 STOP token marking end of a rule sequence.
NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS

# Three-head value/advantage configuration. The value head emits one scalar
# per training reward — (V_flops, V_mem, V_acc) — and the per-episode
# preference vector `w` (stored on Trajectory.preference) weights these three
# advantages when scalarizing for the PPO loss. The mapping into the env's
# 8-component reward vector is fixed:
#   head 0  flops          (REWARD_INDEX["flops"])
#   head 1  peak_memory    (REWARD_INDEX["peak_memory"])
#   head 2  frob_residual  (REWARD_INDEX["frob_residual"])
# The remaining 5 env-reward components are still emitted for host-side
# logging / top-N heaps but do not enter the value head or advantage path.
HEAD_REWARD_INDICES: tuple[int, ...] = (
    REWARD_INDEX["flops"],
    REWARD_INDEX["peak_memory"],
    REWARD_INDEX["frob_residual"],
)
NUM_VALUE_HEADS = len(HEAD_REWARD_INDICES)
HEAD_NAMES: tuple[str, ...] = ("flops", "mem", "acc")
_HEAD_REWARD_INDICES_ARR = jnp.asarray(HEAD_REWARD_INDICES, dtype=jnp.int32)

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
    preference: jax.Array  # (NUM_VALUE_HEADS,) — weights V_flops/V_mem/V_acc
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
    old_vertex_dist: jax.Array
    old_pair_dists: jax.Array
    old_factor_dists: jax.Array
    old_micro_op_dists: jax.Array
    old_micro_i_dists: jax.Array
    old_micro_j_dists: jax.Array
    old_micro_exp_dists: jax.Array
    old_micro_kind_dists: jax.Array
    estim_returns: jax.Array
    norm_adv: jax.Array
    vertex_avail_mask: jax.Array


# ---------------------------------------------------------------------------
# Helpers used by every variant
# ---------------------------------------------------------------------------


def _stop_only_logits(num_pair_choices: int) -> jax.Array:
    """Logits that put all mass on the STOP pair index."""
    return jnp.where(jnp.arange(num_pair_choices) == PAIR_STOP, 0.0, -1e9).astype(
        jnp.float32
    )


def _pad_specs_to_env_slot(specs: jax.Array) -> jax.Array:
    """Pad/truncate per-vertex rule specs to the env's `(MAX_RULES_PER_VERTEX, 3)` slot shape.

    The agent's policy may emit fewer rules than the env can store (`--max-rules` <
    `MAX_RULES_PER_VERTEX`); unused trailing slots are filled with `(-1, -1, 0)` so
    the env's `_callback` correctly treats them as empty.
    """
    n = specs.shape[0]
    if n == MAX_RULES_PER_VERTEX:
        return specs
    if n > MAX_RULES_PER_VERTEX:
        return specs[:MAX_RULES_PER_VERTEX]
    pad = jnp.tile(
        jnp.array([-1, -1, 0], dtype=jnp.int32), (MAX_RULES_PER_VERTEX - n, 1)
    )
    return jnp.concatenate([specs, pad], axis=0)


def build_rule_specs(pair_seq, factor_seq, factor_table) -> jax.Array:
    """Convert per-slot `(pair_idx, factor_idx)` -> `(MAX_RULES_PER_VERTEX, 3)` rule specs.

    `pair_idx == PAIR_STOP` terminates the sequence; subsequent slots are
    written as unused (`base_idx1 = -1, factor = 0`). The output is always
    padded out to `MAX_RULES_PER_VERTEX` so the env's fixed-shape slot
    accepts it directly.
    """
    base = _PAIR_TO_BASE[pair_seq]
    factor_vals = factor_table[factor_seq]

    is_stop = pair_seq == PAIR_STOP
    has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0

    base_final = jnp.where(has_stopped[:, None], -1, base)
    factor_final = jnp.where(has_stopped, 0, factor_vals)
    specs = jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(
        jnp.int32
    )
    return _pad_specs_to_env_slot(specs)


def build_legacy_rule_specs(pair_seq) -> jax.Array:
    """Rule specs for the single-rule (factor=-1) variants. Ignores the factor table."""
    base = _PAIR_TO_BASE[pair_seq]
    factor = jnp.full(pair_seq.shape, -1, dtype=jnp.int32)

    is_stop = pair_seq == PAIR_STOP
    has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0

    base_final = jnp.where(has_stopped[:, None], -1, base)
    factor_final = jnp.where(has_stopped, 0, factor)
    specs = jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(
        jnp.int32
    )
    return _pad_specs_to_env_slot(specs)


def old_log_prob_for_action(
    vertex_idx, pair_seq, factor_seq, vertex_dist, pair_dists, factor_dists
):
    """Joint log-prob of an action under stored old-policy distributions."""
    log_p_v = jnp.log(vertex_dist[vertex_idx] + 1e-8)

    is_stop = pair_seq == PAIR_STOP
    prior_stops = jnp.cumsum(is_stop.astype(jnp.int32)) - is_stop.astype(jnp.int32)
    pair_active = (prior_stops == 0).astype(jnp.float32)
    factor_active = ((prior_stops == 0) & (~is_stop)).astype(jnp.float32)

    arange_r = jnp.arange(pair_seq.shape[0])
    pair_log_ps = jnp.log(pair_dists[arange_r, pair_seq] + 1e-8) * pair_active
    factor_log_ps = jnp.log(factor_dists[arange_r, factor_seq] + 1e-8) * factor_active
    return log_p_v + jnp.sum(pair_log_ps) + jnp.sum(factor_log_ps)


def old_micro_log_prob_for_action(
    vertex_idx,
    op_seq,
    i_seq,
    j_seq,
    exp_seq,
    kind_seq,
    vertex_dist,
    op_dists,
    i_dists,
    j_dists,
    exp_dists,
    kind_dists,
):
    """Joint log-prob of a typed micro-action sequence under stored dists.

    Dynamic-substeps analog of :func:`old_log_prob_for_action`. The vertex
    log-prob plus the per-sub-step (op_type, i, j, prime-exponents,
    compress_kind) contributions are summed, with the per-component
    activity gating matching :meth:`MicroActionHead.log_prob_step`:

    * i active for DIAG / COMPRESS.
    * j and prime-exponent active for DIAG only.
    * compress_kind active for COMPRESS only.
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

    return log_p_v + jnp.sum(log_p_op + log_p_i + log_p_j + log_p_exp + log_p_kind)


def _pad_seq(value: jax.Array, max_rules: int, pad: int = 0) -> jax.Array:
    """Pad a 1-d array out to `max_rules` with `pad` (used to fill unused slot indices)."""
    pad_len = max_rules - value.shape[0]
    if pad_len <= 0:
        return value[:max_rules]
    pad_arr = jnp.full((pad_len,), pad, dtype=value.dtype)
    return jnp.concatenate([value, pad_arr], axis=0)


def _pad_dists(
    active: jax.Array, max_rules: int, num_choices: int, fill_idx: int
) -> jax.Array:
    """Stack `active` (1, num_choices) with `(max_rules - 1)` degenerate one-hots at `fill_idx`."""
    pad_len = max_rules - active.shape[0]
    if pad_len <= 0:
        return active[:max_rules]
    pad_one_hot = jnp.broadcast_to(
        jnn.one_hot(fill_idx, num_choices), (pad_len, num_choices)
    )
    return jnp.concatenate([active, pad_one_hot], axis=0)


# ---------------------------------------------------------------------------
# Composable agent: VertexPolicy x RulePolicy
#
# `VertexPolicy` produces vertex logits (selection over `total_v` vertices)
# and a per-vertex context vector that conditions the rule head.
#   * `PointerVertexPolicy`: one learned vertex query per vertex cross-attends
#     over the encoded tokens; the cross-attended representation is the
#     context.
#   * `MLPVertexPolicy`: a single MLP off the masked-mean summary produces
#     vertex logits; per-vertex contexts are summary + vertex_embedding(v).
#
# `RulePolicy` consumes the chosen vertex's context (plus the per-vertex
# pair-validity mask) and emits a sequence of `(axis_pair, factor)` rules.
#   * `AutoregRulePolicy`: the autoregressive `RuleDecoder` scan from before;
#     emits up to `max_rules` rules with arbitrary factors from the table.
#   * `SingleRulePolicy`: a single per-vertex sp head emitting one rule with
#     factor fixed to -1 (the factor table is ignored).
#
# Both axes are independent: any of the four combinations is valid.
# ---------------------------------------------------------------------------


class RuleDecoder(eqx.Module):
    """Per-slot autoregressive head emitting `(axis_pair, factor)` rules.

    The pair head is conditioned on the previous slot's `(pair, factor)` and
    the chosen vertex; the factor head is conditioned on the *current* pair so
    the two heads are autoregressive within a single rule slot.
    """

    pair_embed: eqx.nn.Embedding
    factor_embed: eqx.nn.Embedding
    slot_embed: eqx.nn.Embedding
    rnn_proj: eqx.nn.Linear
    pair_head: eqx.nn.Linear
    factor_head: eqx.nn.Linear

    embd_dim: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)

    def __init__(self, embd_dim, max_rules, num_pair_choices, num_factors, key):
        keys = jrand.split(key, 6)
        self.embd_dim = embd_dim
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.pair_embed = eqx.nn.Embedding(num_pair_choices, embd_dim, key=keys[0])
        self.factor_embed = eqx.nn.Embedding(num_factors, embd_dim, key=keys[1])
        self.slot_embed = eqx.nn.Embedding(max_rules, embd_dim, key=keys[2])
        self.rnn_proj = eqx.nn.Linear(embd_dim * 4, embd_dim, key=keys[3])
        self.pair_head = eqx.nn.Linear(embd_dim, num_pair_choices, key=keys[4])
        self.factor_head = eqx.nn.Linear(embd_dim * 2, num_factors, key=keys[5])

    def step(self, vertex_repr, slot_idx, prev_pair, prev_factor, prev_h):
        slot_emb = self.slot_embed(slot_idx)
        pair_emb = self.pair_embed(prev_pair)
        factor_emb = self.factor_embed(prev_factor)
        inp = jnp.concatenate([vertex_repr, slot_emb, pair_emb, factor_emb])
        new_h = jnn.tanh(prev_h + self.rnn_proj(inp))
        return new_h, self.pair_head(new_h)

    def factor_logits_for(self, h, pair_idx):
        return self.factor_head(jnp.concatenate([h, self.pair_embed(pair_idx)]))


# === Vertex policies =======================================================


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


class MLPVertexPolicy(eqx.Module):
    """Vertex selection via a single MLP over the masked-mean summary.

    The per-vertex context fed into the rule head is `summary + vertex_embedding(v)`,
    so the rule policy still gets a meaningful, vertex-specific conditioning even
    though there is no pointer net.
    """

    vertex_embedding: eqx.nn.Embedding
    head: MLP

    num_vertices: int = eqx.field(static=True)

    def __init__(self, *, num_vertices, embd_dim, hidden_dims, key):
        keys = jrand.split(key, 2)
        self.num_vertices = num_vertices
        self.vertex_embedding = eqx.nn.Embedding(num_vertices, embd_dim, key=keys[0])
        self.head = MLP(embd_dim, num_vertices, hidden_dims, key=keys[1])

    def __call__(self, enc_x, token_mask):
        mask = token_mask[..., None]
        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        vertex_logits = self.head(summary)
        v_q = jax.vmap(self.vertex_embedding)(jnp.arange(self.num_vertices))
        per_vertex_context = v_q + summary[None, :]
        return vertex_logits, per_vertex_context


# === Rule policies =========================================================


class AutoregRulePolicy(eqx.Module):
    """Autoregressive (axis_pair, factor) rule decoder."""

    decoder: RuleDecoder

    embd_dim: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(self, *, embd_dim, max_rules, num_pair_choices, num_factors, key):
        self.embd_dim = embd_dim
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.use_factor_table = True
        self.decoder = RuleDecoder(
            embd_dim, max_rules, num_pair_choices, num_factors, key=key
        )

    def _init_carry(self):
        return (
            jnp.array(PAIR_STOP, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.zeros(self.embd_dim),
            jnp.array(True, dtype=jnp.bool_),
        )

    def sample(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        key,
        *,
        pin_factor_idx: int | None = None,
    ):
        """Sample ``(pair_seq, factor_seq)`` autoregressively.

        ``v_factor_mask`` is the per-(pair, factor) validity mask for the
        chosen vertex (``shape == (num_pair_choices, num_factors)``). For
        each emitted ``pair_idx`` we mask the factor logits with
        ``v_factor_mask[pair_idx]`` so the agent can only sample factors
        that ``apply_dynamic_sparsity`` will accept downstream.

        Stage D: when ``pin_factor_idx`` is set, every slot's factor is
        forced to that index regardless of the factor head's output. The
        next slot's autoregressive conditioning sees the forced factor as
        ``prev_factor``, so rollout and loss-time evaluation agree on the
        sequence — same trick the env uses for the rule_specs themselves.
        """
        slot_keys = jrand.split(key, self.max_rules)
        stop_only = _stop_only_logits(self.num_pair_choices)
        pin_static = pin_factor_idx  # capture in closure for jit

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, k = slot_data
            kp, kf = jrand.split(k)
            new_h, pair_logits = self.decoder.step(
                vertex_context, slot_idx, prev_pair, prev_factor, prev_h
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            pair_idx = distrax.Categorical(probs=pair_dist).sample(seed=kp)

            if pin_static is None:
                factor_logits = self.decoder.factor_logits_for(new_h, pair_idx)
                factor_logits = jnp.where(
                    v_factor_mask[pair_idx] > 0.5, factor_logits, -1e9
                )
                factor_dist = jnn.softmax(factor_logits, axis=-1)
                factor_idx = distrax.Categorical(probs=factor_dist).sample(seed=kf)
            else:
                factor_idx = jnp.asarray(pin_static, dtype=jnp.int32)
                factor_dist = jnn.one_hot(factor_idx, self.num_factors)

            new_active = active & (pair_idx != PAIR_STOP)
            return (pair_idx, factor_idx, new_h, new_active), (
                pair_idx,
                factor_idx,
                pair_dist,
                factor_dist,
            )

        _, (pair_seq, factor_seq, pair_dists, factor_dists) = lax.scan(
            rule_step, self._init_carry(), (jnp.arange(self.max_rules), slot_keys)
        )
        return pair_seq, factor_seq, pair_dists, factor_dists

    def evaluate(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        pair_seq,
        factor_seq,
        *,
        pin_factor_idx: int | None = None,
    ):
        stop_only = _stop_only_logits(self.num_pair_choices)
        pin_static = pin_factor_idx

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, true_pair, true_factor = slot_data
            new_h, pair_logits = self.decoder.step(
                vertex_context, slot_idx, prev_pair, prev_factor, prev_h
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            log_p_pair = jnp.log(pair_dist[true_pair] + 1e-8)

            if pin_static is None:
                factor_logits = self.decoder.factor_logits_for(new_h, true_pair)
                factor_logits = jnp.where(
                    v_factor_mask[true_pair] > 0.5, factor_logits, -1e9
                )
                factor_dist = jnn.softmax(factor_logits, axis=-1)
                log_p_factor = jnp.log(factor_dist[true_factor] + 1e-8)
                factor_ent_raw = entropy(factor_dist)
            else:
                # Factor is deterministic — log p = 0, entropy = 0. Returning
                # a degenerate one-hot for `factor_dist` keeps the KL terms
                # downstream well-defined (KL(δ‖δ) = 0).
                factor_dist = jnn.one_hot(
                    jnp.asarray(pin_static, dtype=jnp.int32), self.num_factors
                )
                log_p_factor = jnp.array(0.0, dtype=jnp.float32)
                factor_ent_raw = jnp.array(0.0, dtype=jnp.float32)

            active_f32 = active.astype(jnp.float32)
            log_p_pair_eff = log_p_pair * active_f32
            pair_ent = entropy(pair_dist) * active_f32

            factor_active = active & (true_pair != PAIR_STOP)
            factor_active_f32 = factor_active.astype(jnp.float32)
            log_p_factor_eff = log_p_factor * factor_active_f32
            factor_ent = factor_ent_raw * factor_active_f32

            new_active = active & (true_pair != PAIR_STOP)
            # Conditioning for the next slot uses `true_pair, true_factor` so
            # the autoregressive trace matches what `sample` produced — when
            # factor is pinned, both rollout and evaluate see the pinned
            # value as `prev_factor`.
            return (true_pair, true_factor, new_h, new_active), (
                log_p_pair_eff,
                log_p_factor_eff,
                pair_ent,
                factor_ent,
                pair_dist,
                factor_dist,
            )

        _, (lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists) = (
            lax.scan(
                rule_step,
                self._init_carry(),
                (jnp.arange(self.max_rules), pair_seq, factor_seq),
            )
        )
        return lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists

    def to_env_specs(self, pair_seq, factor_seq, factor_table):
        return build_rule_specs(pair_seq, factor_seq, factor_table)


class SparsityRatioRuleDecoder(eqx.Module):
    """Per-slot autoregressive head with rho-based factor parameterization.

    Same RNN backbone as :class:`RuleDecoder`, but the factor-side projection
    is a *single scalar* ``rho_logit`` instead of K factor logits. The discrete
    factor index used as the action is derived externally by snap-mixing this
    rho onto the precomputed sparsity-ratio table; this decoder just exposes
    the per-slot rho for the policy.

    Stage E prior anneal: ``rho_init_bias`` is added to the rho_head bias at
    construction time so that ``sigmoid(rho_init_bias) ≈ 1`` at step 0. This
    matches Stage D's strict-diagonal pin (the factor head behaves as if it
    was still pinned). Gradient through ``rho_head.bias`` then naturally
    anneals the prior toward the learned distribution as training proceeds —
    no explicit schedule needed because the bias is just a normal parameter.
    """

    pair_embed: eqx.nn.Embedding
    factor_embed: eqx.nn.Embedding
    slot_embed: eqx.nn.Embedding
    rnn_proj: eqx.nn.Linear
    pair_head: eqx.nn.Linear
    rho_head: eqx.nn.Linear

    embd_dim: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)

    def __init__(
        self,
        embd_dim,
        max_rules,
        num_pair_choices,
        num_factors,
        key,
        *,
        rho_init_bias: float = 0.0,
    ):
        keys = jrand.split(key, 6)
        self.embd_dim = embd_dim
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.pair_embed = eqx.nn.Embedding(num_pair_choices, embd_dim, key=keys[0])
        self.factor_embed = eqx.nn.Embedding(num_factors, embd_dim, key=keys[1])
        self.slot_embed = eqx.nn.Embedding(max_rules, embd_dim, key=keys[2])
        self.rnn_proj = eqx.nn.Linear(embd_dim * 4, embd_dim, key=keys[3])
        self.pair_head = eqx.nn.Linear(embd_dim, num_pair_choices, key=keys[4])
        rho_head = eqx.nn.Linear(embd_dim * 2, 1, key=keys[5])
        if rho_init_bias != 0.0 and rho_head.bias is not None:
            rho_head = eqx.tree_at(
                lambda l: l.bias,
                rho_head,
                rho_head.bias + jnp.asarray(rho_init_bias, dtype=rho_head.bias.dtype),
            )
        self.rho_head = rho_head

    def step(self, vertex_repr, slot_idx, prev_pair, prev_factor, prev_h):
        slot_emb = self.slot_embed(slot_idx)
        pair_emb = self.pair_embed(prev_pair)
        factor_emb = self.factor_embed(prev_factor)
        inp = jnp.concatenate([vertex_repr, slot_emb, pair_emb, factor_emb])
        new_h = jnn.tanh(prev_h + self.rnn_proj(inp))
        return new_h, self.pair_head(new_h)

    def rho_logit_for(self, h, pair_idx):
        return self.rho_head(jnp.concatenate([h, self.pair_embed(pair_idx)])).squeeze()


class SparsityRatioAutoregRulePolicy(eqx.Module):
    """Stage E: autoregressive (pair, factor) policy with sparsity-ratio reparam.

    The factor head emits a continuous coordinate ``ρ ∈ [0, 1]`` representing
    the fraction of the way from "no compression" to "strict diagonal" for
    the pair at hand. Given the precomputed sparsity ratios per factor in the
    table, ρ is snapped to the two adjacent factors ``(i, i+1)`` with mixing
    weight ``α = (ρ - ρ_i) / (ρ_{i+1} - ρ_i)``. The discrete factor index used
    in the rollout's action is sampled from ``{i, i+1}`` with probabilities
    ``(1-α, α)``; the log-probability of any sampled action has the closed
    form ``log α`` or ``log(1-α)``, so PPO works unchanged. The smooth
    relaxation lives entirely in the policy's gradient through ρ — no DARTS
    straight-through is needed for the action itself.

    The mixing weight α is the natural training-time signal: when the head
    is uncertain it spreads ρ between two factors, and the gradient through
    α tells it which way to slide. Collapse to "always strict diagonal"
    shows up as ρ saturating at 1, i.e. α stuck at 0 with idx_upper at the
    last factor; this is the marginal-distribution monitor the spec calls
    out as the early-warning signal.
    """

    decoder: SparsityRatioRuleDecoder
    sparsity_ratios: jax.Array  # (num_factors,) sorted ascending

    embd_dim: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        embd_dim,
        max_rules,
        num_pair_choices,
        num_factors,
        sparsity_ratios,
        key,
        rho_init_bias: float = 0.0,
    ):
        self.embd_dim = embd_dim
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.use_factor_table = True
        self.decoder = SparsityRatioRuleDecoder(
            embd_dim,
            max_rules,
            num_pair_choices,
            num_factors,
            key=key,
            rho_init_bias=rho_init_bias,
        )
        self.sparsity_ratios = jnp.asarray(sparsity_ratios, dtype=jnp.float32)

    def _init_carry(self):
        return (
            jnp.array(PAIR_STOP, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.zeros(self.embd_dim),
            jnp.array(True, dtype=jnp.bool_),
        )

    def _snap_mix(self, rho):
        """Given ``rho ∈ [0, 1]``, return ``(idx_lower, idx_upper, alpha)``.

        Indices are into ``self.sparsity_ratios``; α is the mixing weight on
        ``idx_upper``. When ρ is at or past the table extremes both indices
        collapse to the same value and α = 0/1 accordingly.
        """
        rho = jnp.clip(rho, self.sparsity_ratios[0], self.sparsity_ratios[-1])
        idx_upper = jnp.searchsorted(self.sparsity_ratios, rho).astype(jnp.int32)
        idx_upper = jnp.clip(idx_upper, 1, self.num_factors - 1)
        idx_lower = idx_upper - 1
        rho_lower = self.sparsity_ratios[idx_lower]
        rho_upper = self.sparsity_ratios[idx_upper]
        denom = jnp.maximum(rho_upper - rho_lower, 1e-9)
        alpha = jnp.clip((rho - rho_lower) / denom, 0.0, 1.0)
        return idx_lower, idx_upper, alpha

    def _factor_dist_from_snap(self, idx_lower, idx_upper, alpha):
        d = jnp.zeros(self.num_factors)
        d = d.at[idx_lower].add(1.0 - alpha)
        d = d.at[idx_upper].add(alpha)
        return d

    def sample(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        key,
        *,
        pin_factor_idx: int | None = None,
    ):
        slot_keys = jrand.split(key, self.max_rules)
        stop_only = _stop_only_logits(self.num_pair_choices)
        pin_static = pin_factor_idx

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, k = slot_data
            kp, kf = jrand.split(k)
            new_h, pair_logits = self.decoder.step(
                vertex_context,
                slot_idx,
                prev_pair,
                prev_factor,
                prev_h,
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            pair_idx = distrax.Categorical(probs=pair_dist).sample(seed=kp)

            if pin_static is None:
                rho_logit = self.decoder.rho_logit_for(new_h, pair_idx)
                rho = jnn.sigmoid(rho_logit)
                idx_lower, idx_upper, alpha = self._snap_mix(rho)
                u = jrand.uniform(kf)
                sample_upper = u < alpha
                factor_idx = jnp.where(sample_upper, idx_upper, idx_lower).astype(
                    jnp.int32
                )
                factor_dist = self._factor_dist_from_snap(idx_lower, idx_upper, alpha)
            else:
                factor_idx = jnp.asarray(pin_static, dtype=jnp.int32)
                factor_dist = jnn.one_hot(factor_idx, self.num_factors)

            new_active = active & (pair_idx != PAIR_STOP)
            return (pair_idx, factor_idx, new_h, new_active), (
                pair_idx,
                factor_idx,
                pair_dist,
                factor_dist,
            )

        _, (pair_seq, factor_seq, pair_dists, factor_dists) = lax.scan(
            rule_step,
            self._init_carry(),
            (jnp.arange(self.max_rules), slot_keys),
        )
        return pair_seq, factor_seq, pair_dists, factor_dists

    def evaluate(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        pair_seq,
        factor_seq,
        *,
        pin_factor_idx: int | None = None,
    ):
        stop_only = _stop_only_logits(self.num_pair_choices)
        pin_static = pin_factor_idx

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, true_pair, true_factor = slot_data
            new_h, pair_logits = self.decoder.step(
                vertex_context,
                slot_idx,
                prev_pair,
                prev_factor,
                prev_h,
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            log_p_pair = jnp.log(pair_dist[true_pair] + 1e-8)

            if pin_static is None:
                rho_logit = self.decoder.rho_logit_for(new_h, true_pair)
                rho = jnn.sigmoid(rho_logit)
                idx_lower, idx_upper, alpha = self._snap_mix(rho)
                # ``true_factor`` should always equal ``idx_upper`` or
                # ``idx_lower`` because we only sample from those — but the
                # action might have been recorded under a slightly different
                # ρ (post optimizer step). Compute log p robustly.
                p_true = jnp.where(
                    true_factor == idx_upper,
                    alpha,
                    jnp.where(true_factor == idx_lower, 1.0 - alpha, 1e-9),
                )
                log_p_factor = jnp.log(p_true + 1e-9)
                ent_factor_raw = -(
                    alpha * jnp.log(alpha + 1e-9)
                    + (1.0 - alpha) * jnp.log(1.0 - alpha + 1e-9)
                )
                factor_dist = self._factor_dist_from_snap(idx_lower, idx_upper, alpha)
            else:
                factor_dist = jnn.one_hot(
                    jnp.asarray(pin_static, dtype=jnp.int32),
                    self.num_factors,
                )
                log_p_factor = jnp.array(0.0, dtype=jnp.float32)
                ent_factor_raw = jnp.array(0.0, dtype=jnp.float32)

            active_f32 = active.astype(jnp.float32)
            log_p_pair_eff = log_p_pair * active_f32
            pair_ent = entropy(pair_dist) * active_f32

            factor_active = active & (true_pair != PAIR_STOP)
            factor_active_f32 = factor_active.astype(jnp.float32)
            log_p_factor_eff = log_p_factor * factor_active_f32
            factor_ent = ent_factor_raw * factor_active_f32

            new_active = active & (true_pair != PAIR_STOP)
            return (true_pair, true_factor, new_h, new_active), (
                log_p_pair_eff,
                log_p_factor_eff,
                pair_ent,
                factor_ent,
                pair_dist,
                factor_dist,
            )

        _, (lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists) = (
            lax.scan(
                rule_step,
                self._init_carry(),
                (jnp.arange(self.max_rules), pair_seq, factor_seq),
            )
        )
        return lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists

    def to_env_specs(self, pair_seq, factor_seq, factor_table):
        return build_rule_specs(pair_seq, factor_seq, factor_table)


class SingleRulePolicy(eqx.Module):
    """Single rule per vertex (factor=-1 fixed). Equivalent to the legacy sp head."""

    sp_head: MLP

    embd_dim: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(
        self, *, embd_dim, max_rules, num_pair_choices, num_factors, sp_dims, key
    ):
        self.embd_dim = embd_dim
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.use_factor_table = False
        self.sp_head = MLP(embd_dim, num_pair_choices, sp_dims, key=key)

    def _slot0_dist(self, vertex_context, v_pair_mask):
        sp_logits = self.sp_head(vertex_context)
        sp_logits = jnp.where(v_pair_mask > 0.5, sp_logits, -1e9)
        return jnn.softmax(sp_logits, axis=-1)

    def _padded_pair_dists(self, slot0_dist):
        return _pad_dists(
            slot0_dist[None, :], self.max_rules, self.num_pair_choices, PAIR_STOP
        )

    def _degenerate_factor_dists(self):
        return jnp.broadcast_to(
            jnn.one_hot(0, self.num_factors), (self.max_rules, self.num_factors)
        )

    def sample(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        key,
        *,
        pin_factor_idx: int | None = None,
    ):
        # SingleRulePolicy ignores v_factor_mask and the pin: factor is
        # already fixed to the legacy "-1 / first-table-entry" by construction.
        del v_factor_mask, pin_factor_idx
        slot0_dist = self._slot0_dist(vertex_context, v_pair_mask)
        pair_idx = distrax.Categorical(probs=slot0_dist).sample(seed=key)
        pair_seq = _pad_seq(
            jnp.atleast_1d(pair_idx).astype(jnp.int32), self.max_rules, pad=PAIR_STOP
        )
        factor_seq = jnp.zeros((self.max_rules,), dtype=jnp.int32)
        return (
            pair_seq,
            factor_seq,
            self._padded_pair_dists(slot0_dist),
            self._degenerate_factor_dists(),
        )

    def evaluate(
        self,
        vertex_context,
        v_pair_mask,
        v_factor_mask,
        pair_seq,
        factor_seq,
        *,
        pin_factor_idx: int | None = None,
    ):
        del v_factor_mask, pin_factor_idx
        slot0_dist = self._slot0_dist(vertex_context, v_pair_mask)
        pair_idx = pair_seq[0]
        log_p_pair = jnp.log(slot0_dist[pair_idx] + 1e-8)
        pair_ent = entropy(slot0_dist)

        # Slots > 0 are degenerate STOP and the factor head is degenerate; their
        # log-prob and entropy contributions are zero.
        lp_pairs = jnp.zeros((self.max_rules,), dtype=jnp.float32).at[0].set(log_p_pair)
        lp_factors = jnp.zeros((self.max_rules,), dtype=jnp.float32)
        ent_pairs = jnp.zeros((self.max_rules,), dtype=jnp.float32).at[0].set(pair_ent)
        ent_factors = jnp.zeros((self.max_rules,), dtype=jnp.float32)
        return (
            lp_pairs,
            lp_factors,
            ent_pairs,
            ent_factors,
            self._padded_pair_dists(slot0_dist),
            self._degenerate_factor_dists(),
        )

    def to_env_specs(self, pair_seq, factor_seq, factor_table):
        return build_legacy_rule_specs(pair_seq)


# === Composed agent ========================================================


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


class CachedEncoding(NamedTuple):
    """Per-rollout cache for the B.4.next "encode once" path.

    Captured from the initial residual jaxpr at episode start; reused for
    every step of the rollout (and at loss time) instead of re-running the
    transformer stack on the per-step residual jaxpr. The residual state is
    the only thing that varies inside an episode.
    """

    enc_x: jax.Array  # (T, embd_dim) — full encoder output
    token_mask: jax.Array  # (T,) bool, non-pad tokens
    summary: jax.Array  # (embd_dim,) masked-mean over enc_x
    vertex_logits_base: jax.Array  # (V,) before per-step masking
    vertex_contexts_base: jax.Array  # (V, embd_dim) pre-residual / pre-data


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
    reward: ``value_head_flops``, ``value_head_mem``, ``value_head_acc``.
    Their concatenation is the (NUM_VALUE_HEADS,) = (3,) value vector the
    trainer consumes; the per-head split keeps gradient scales sane across
    the qualitatively different reward families and matches the per-head
    GAE / preference-scalarization in ``train_episode``.

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
    rule_policy: eqx.Module
    # Optional: present iff `--dynamic-substeps` is on. When non-None, the
    # rollout / loss path routes through `sample_action_dynamic` and
    # `evaluate_action_dynamic` instead of the legacy `rule_policy` heads.
    # The two paths coexist on the same Agent so curriculum stages can
    # swap between them without rebuilding the whole module.
    micro_action_policy: MicroActionPolicy | None
    value_head_flops: MLP
    value_head_mem: MLP
    value_head_acc: MLP
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
        rule_policy,
        value_head_flops,
        value_head_mem,
        value_head_acc,
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
        self.rule_policy = rule_policy
        self.micro_action_policy = micro_action_policy
        self.value_head_flops = value_head_flops
        self.value_head_mem = value_head_mem
        self.value_head_acc = value_head_acc
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

    def encode_once(self, tokens, eqn_ids=None, *, key=None) -> CachedEncoding:
        """B.4.next: encode the residual jaxpr *once* per episode.

        Returns a :class:`CachedEncoding` capturing everything the per-step
        decoder path needs that doesn't depend on the residual state. Reuse
        this for every step of the rollout instead of re-running the
        transformer stack on the per-step residual jaxpr.
        """
        token_mask = tokens != 0
        mask = token_mask[..., None]
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        enc_x = self.encoder(x, eqn_ids=eqn_ids, key=enc_key)
        vertex_logits_base, vertex_contexts_base = self.vertex_policy(
            enc_x,
            token_mask,
        )
        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        return CachedEncoding(
            enc_x=enc_x,
            token_mask=token_mask,
            summary=summary,
            vertex_logits_base=vertex_logits_base,
            vertex_contexts_base=vertex_contexts_base,
        )

    def _decode_from_cache(
        self, cached, vertex_features, residual_state, preference=None
    ):
        """Combine cached encoding with per-step residual_state / data feats /
        Stage F preference vector."""
        vertex_logits = cached.vertex_logits_base
        vertex_contexts = cached.vertex_contexts_base
        if vertex_features is not None:
            vertex_contexts = vertex_contexts + self._data_embedding(vertex_features)
        if residual_state is not None:
            vertex_contexts = vertex_contexts + residual_state
            residual_summary = jnp.mean(residual_state, axis=0)
            summary_eff = cached.summary + self.residual_to_summary(residual_summary)
        else:
            summary_eff = cached.summary
        if preference is not None:
            pref_emb = self.pref_proj(preference)
            vertex_contexts = vertex_contexts + pref_emb[None, :]
            summary_eff = summary_eff + pref_emb
        v_flops = self.value_head_flops(summary_eff)
        v_mem = self.value_head_mem(summary_eff)
        v_acc = self.value_head_acc(summary_eff)
        value = jnp.concatenate([v_flops, v_mem, v_acc], axis=-1)
        return vertex_logits, vertex_contexts, value

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
        v_acc = self.value_head_acc(summary)
        value = jnp.concatenate([v_flops, v_mem, v_acc], axis=-1)
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

    def sample_action(
        self,
        tokens,
        vertex_avail_mask,
        pair_valid_mask,
        pair_factor_mask,
        key,
        eqn_ids=None,
        vertex_features=None,
        residual_state=None,
        cached_encoding=None,
        pin_rules_to_exact: bool = False,
        pin_factor_idx: int | None = None,
        preference=None,
        vertex_temperature=None,
    ):
        net_key, vertex_key, rule_key = jrand.split(key, 3)
        if cached_encoding is None:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                preference=preference,
                key=net_key,
            )
        else:
            vertex_logits, vertex_contexts, value = self._decode_from_cache(
                cached_encoding,
                vertex_features,
                residual_state,
                preference=preference,
            )

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        # Optional rollout-time logit temperature on the vertex head. The
        # stored vertex_dist reflects the tempered distribution, so the
        # downstream importance-ratio path (PPO/GDPO) corrects for it via
        # log_p_old. A `None` default keeps the existing fast path
        # bit-identical for callers that don't opt in.
        if vertex_temperature is not None:
            masked_v_logits = masked_v_logits / vertex_temperature
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        vertex_idx = distrax.Categorical(probs=vertex_dist).sample(seed=vertex_key)

        v_context = vertex_contexts[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]
        v_factor_mask = pair_factor_mask[vertex_idx]
        # Compute both pinned and sampled outputs unconditionally and select
        # by `pin_rules_to_exact` via jnp.where. This is the JAX-traced
        # equivalent of the original Python-level branch — accepts both
        # a static Python bool (fast path: jnp.where folds at trace time)
        # and a traced 0/1 scalar (used by the curriculum runner to swap
        # ve_only ↔ full without forcing a recompile). The 2x rule-head
        # compute is negligible next to the encoder.
        ps_pinned, fs_pinned, pd_pinned, fd_pinned = self._pinned_rule_outputs()
        ps_sampled, fs_sampled, pd_sampled, fd_sampled = self.rule_policy.sample(
            v_context,
            v_pair_mask,
            v_factor_mask,
            rule_key,
            pin_factor_idx=pin_factor_idx,
        )
        pin = jnp.asarray(pin_rules_to_exact, dtype=jnp.bool_)
        pair_seq = jnp.where(pin, ps_pinned, ps_sampled)
        factor_seq = jnp.where(pin, fs_pinned, fs_sampled)
        pair_dists = jnp.where(pin, pd_pinned, pd_sampled)
        factor_dists = jnp.where(pin, fd_pinned, fd_sampled)
        # `v_context` is the chosen vertex's representation as fed to the
        # rule head — exactly what the B.4 residual update wants to remember
        # about this elimination event.
        return (
            vertex_idx,
            pair_seq,
            factor_seq,
            vertex_dist,
            pair_dists,
            factor_dists,
            value,
            v_context,
        )

    def update_residual(self, residual_state, vertex_idx, vertex_repr):
        """Apply the per-slot recurrent update to ``residual_state``."""
        return self.residual_update(residual_state, vertex_idx, vertex_repr)

    def _pinned_rule_outputs(self):
        """Stage C: exact-AD pinned rule outputs.

        Returns ``(pair_seq, factor_seq, pair_dists, factor_dists)`` where
        every slot is STOP / factor 0 and the distributions are degenerate
        one-hots at those values. The policy is therefore identity in this
        mode — the rule head receives zero gradient and the env always sees
        a no-rules action (exact AD), letting the vertex pointer carry the
        whole learning signal as the spec calls for.
        """
        max_rules = self.max_rules
        pair_seq = jnp.full((max_rules,), PAIR_STOP, dtype=jnp.int32)
        factor_seq = jnp.zeros((max_rules,), dtype=jnp.int32)
        pair_dists = jnp.broadcast_to(
            jnn.one_hot(PAIR_STOP, self.num_pair_choices)[None, :],
            (max_rules, self.num_pair_choices),
        )
        factor_dists = jnp.broadcast_to(
            jnn.one_hot(0, self.num_factors)[None, :],
            (max_rules, self.num_factors),
        )
        return pair_seq, factor_seq, pair_dists, factor_dists

    def evaluate_action(
        self,
        tokens,
        vertex_idx,
        pair_seq,
        factor_seq,
        vertex_avail_mask,
        pair_valid_mask,
        pair_factor_mask,
        key,
        eqn_ids=None,
        vertex_features=None,
        residual_state=None,
        cached_encoding=None,
        pin_rules_to_exact: bool = False,
        pin_factor_idx: int | None = None,
        preference=None,
    ):
        if cached_encoding is None:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                preference=preference,
                key=key,
            )
        else:
            vertex_logits, vertex_contexts, value = self._decode_from_cache(
                cached_encoding,
                vertex_features,
                residual_state,
                preference=preference,
            )

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        log_p_vertex = jnp.log(vertex_dist[vertex_idx] + 1e-8)
        vertex_ent = entropy(vertex_dist)

        v_context = vertex_contexts[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]
        v_factor_mask = pair_factor_mask[vertex_idx]
        # Always run both branches and select via jnp.where on pin_rules_to_exact.
        # Pinned branch: the recorded action is always (STOP, factor 0); the
        # rule head's log-prob / entropy contributions are 0 and the returned
        # dists are degenerate one-hots. Sampled branch: the usual rule_policy
        # evaluate path.
        _, _, pd_pinned, fd_pinned = self._pinned_rule_outputs()
        (
            lp_pairs,
            lp_factors,
            ent_pairs,
            ent_factors,
            pd_sampled,
            fd_sampled,
        ) = self.rule_policy.evaluate(
            v_context,
            v_pair_mask,
            v_factor_mask,
            pair_seq,
            factor_seq,
            pin_factor_idx=pin_factor_idx,
        )
        pin = jnp.asarray(pin_rules_to_exact, dtype=jnp.bool_)
        total_log_p_sampled = log_p_vertex + jnp.sum(lp_pairs) + jnp.sum(lp_factors)
        total_entropy_sampled = vertex_ent + jnp.sum(ent_pairs) + jnp.sum(ent_factors)
        total_log_p = jnp.where(pin, log_p_vertex, total_log_p_sampled)
        total_entropy = jnp.where(pin, vertex_ent, total_entropy_sampled)
        pair_dists = jnp.where(pin, pd_pinned, pd_sampled)
        factor_dists = jnp.where(pin, fd_pinned, fd_sampled)
        return total_log_p, total_entropy, value, vertex_dist, pair_dists, factor_dists

    def to_env_action(self, vertex_idx, pair_seq, factor_seq, factor_table):
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=self.rule_policy.to_env_specs(
                pair_seq, factor_seq, factor_table
            ),
        )

    # ------------------------------------------------------------------
    # Dynamic-substeps path: typed micro-action sub-episodes via heads.py
    # ------------------------------------------------------------------

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
    ):
        """Same as :meth:`sample_action` but routes the rule head through
        :class:`MicroActionPolicy`. ``axis_state`` and ``axis_valid_mask``
        come straight from :class:`EnvState`; the per-vertex slice for
        the chosen vertex is converted to :class:`AxisTokenFeatures` and
        scanned by the policy. ``op_legality_override`` is a (3,) mask
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
        if cached_encoding is None:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                preference=preference,
                key=net_key,
            )
        else:
            vertex_logits, vertex_contexts, value = self._decode_from_cache(
                cached_encoding,
                vertex_features,
                residual_state,
                preference=preference,
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
        ) = self.micro_action_policy.sample(
            v_context,
            features,
            factor_tables,
            micro_key,
        )

        # Apply the op-legality override post-hoc on both DIAG and
        # COMPRESS: any disallowed op_type is rewritten to END. The
        # MicroActionPolicy.sample doesn't take the override directly —
        # it computes legality from axis-state (always allowing both
        # DIAG and COMPRESS when axes are available). The override is
        # how the curriculum runner forces `ve_only` (no DIAG, no
        # COMPRESS) or `compress` (no DIAG) or `diag_*` (no COMPRESS)
        # behaviour at sampling time.
        diag_allowed = op_legality_override[OP_DIAG] > 0.5
        compress_allowed = op_legality_override[OP_COMPRESS] > 0.5
        is_diag = actions.op_type == OP_DIAG
        is_compress = actions.op_type == OP_COMPRESS
        disallowed = (is_diag & ~diag_allowed) | (is_compress & ~compress_allowed)
        rewritten_op = jnp.where(
            disallowed,
            jnp.full_like(actions.op_type, OP_END),
            actions.op_type,
        )
        # When op_type is rewritten to END the sampled i / j / exponents /
        # factor / kind are stale. Zero them so the recorded action is
        # canonical (matches what sample_step produces for genuine END outputs).
        zeros_i = jnp.zeros_like(actions.i)
        zeros_j = jnp.zeros_like(actions.j)
        zeros_exp = jnp.zeros_like(actions.exponents)
        zeros_f = jnp.zeros_like(actions.factor)
        zeros_k = jnp.zeros_like(actions.compress_kind)
        actions = MicroAction(
            op_type=rewritten_op,
            i=jnp.where(disallowed, zeros_i, actions.i),
            j=jnp.where(disallowed, zeros_j, actions.j),
            exponents=jnp.where(disallowed[..., None], zeros_exp, actions.exponents),
            factor=jnp.where(disallowed, zeros_f, actions.factor),
            compress_kind=jnp.where(disallowed, zeros_k, actions.compress_kind),
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
        if cached_encoding is None:
            vertex_logits, vertex_contexts, value = self.encode(
                tokens,
                eqn_ids=eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                preference=preference,
                key=key,
            )
        else:
            vertex_logits, vertex_contexts, value = self._decode_from_cache(
                cached_encoding,
                vertex_features,
                residual_state,
                preference=preference,
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
        ) = self.micro_action_policy.evaluate(
            v_context,
            features,
            factor_tables,
            actions,
        )

        total_log_p = log_p_vertex + log_p_sub
        total_entropy = vertex_ent + ent_sub
        # Per-step dists are forwarded for KL tracking against the
        # rollout-time old-policy snapshots stored in the trajectory.
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
        DIAG micro-actions become rule rows; COMPRESS micro-actions are
        silently dropped (the legacy env path doesn't yet consume them
        — see ``--allow-compress``). The translator is JAX-traceable so
        the whole rollout step stays inside jit.

        ``actions.factor`` (stored on each MicroAction by ``sample_step``)
        is the integer factor consumed by the legacy spec — no re-
        derivation from exponents needed here.
        """
        axis_state_v = axis_state[vertex_idx]
        rule_specs = micro_actions_to_rule_specs_jax(
            actions.op_type,
            actions.i,
            actions.j,
            actions.factor,
            axis_state_v,
            compress_kinds=actions.compress_kind,
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
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument(
        "--exec-on-gpu",
        action="store_true",
        help="Pin training to GPU 0 and the env eval callback to GPU 1.",
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
        "--lambda-frob",
        type=float,
        default=0.0,
        help="Weight on the Frobenius-residual reward component (idx 7). Default 0.",
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
        "dominant rollout cost. PPO+GAE handles sparse rewards natively; the only "
        "knob to consider is reducing --potential-shaping (which assumes per-step "
        "value differences).",
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
    p.add_argument(
        "--no-ptr",
        action="store_true",
        help=(
            "Drop the pointer net for vertex selection and use a masked MLP head "
            "over the `total_v` vertices instead. Independent of --not-autoreg."
        ),
    )
    p.add_argument(
        "--not-autoreg",
        action="store_true",
        help=(
            "Replace the autoregressive RuleDecoder with a single per-vertex sp "
            "head (one rule per vertex, factor fixed to -1). Independent of --no-ptr."
        ),
    )

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

    # Curriculum: train through multiple variants with the same model. New
    # heads added in each stage warm up via cosine; existing heads from
    # earlier stages run at a reduced flat multiplier so their learned
    # weights aren't blown away but can still adapt to the new objective.
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help=(
            "Curriculum of variants to run in sequence. Format: "
            "`stage1:N1,stage2:N2,...` where each stage names a --variant "
            "and an episode count. Single optimizer carries across stages; "
            "per-head LR uses cosine_warmup_exp_decay_lr with period = N_i "
            "so the period matches the head-warmup window. Empty = single "
            "training run on --variant."
        ),
    )
    p.add_argument(
        "--curriculum-warmup-frac",
        type=float,
        default=0.3,
        help=(
            "Fraction of each curriculum stage's episodes spent in the "
            "cosine LR warm-up before the exponential-decay phase. 0.3 = "
            "first 30%% of the stage warms up, remaining 70%% decays."
        ),
    )
    p.add_argument(
        "--curriculum-existing-head-mult",
        type=float,
        default=0.3,
        help=(
            "Flat LR multiplier applied to heads introduced in an earlier "
            "curriculum stage. Keeps previously-learned concepts from being "
            "discarded but lets them adapt. Default 0.3."
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
        "--policy-dims",
        type=str,
        default="64,32",
        help="MLP policy/sp head hidden widths (comma-separated).",
    )
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
        "--pretrained-encoder",
        type=str,
        default=None,
        help="Path to a `PretrainModel` saved by `alphagrad.approx.pretrain` "
        "(Stage B.5). When set, the embedding/pos_enc/encoder modules "
        "of the freshly-built agent are replaced with the pretrained ones.",
    )
    p.add_argument(
        "--cache-encoding",
        action="store_true",
        help="Stage B.4.next: encode the residual jaxpr once at the start "
        "of each rollout episode and reuse the encoder output for every "
        "step. The residual state captures per-step variation; the value "
        "head consumes summary + W_residual @ mean(residual_state). At "
        "loss time the same cached path is used, so policy/value "
        "definitions match between rollout and training.",
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
        help="Stage D head curriculum (§3.2): linearly ramp the "
        "axis-pair head's LR multiplier from 1/3 → 1 over the "
        "first N optimizer steps after which point it stays at 1. "
        "0 = ramp disabled, full LR from step 0. The vertex and "
        "shared params always run at full base LR.",
    )
    p.add_argument(
        "--factor-warmup-steps",
        type=int,
        default=0,
        help="Stage E head curriculum: same ramp, applied to the "
        "factor head. 0 = no ramp.",
    )
    p.add_argument(
        "--sparsity-ratio",
        action="store_true",
        help="LEGACY ONLY (--no-dynamic-substeps): replace the "
        "categorical factor head with a scalar sparsity-ratio "
        "coordinate ρ ∈ [0, 1] that snap-mixes onto the two "
        "adjacent valid factors. Ignored in the default "
        "dynamic-substeps path (factors there come from the "
        "prime-exponent head).",
    )
    p.add_argument(
        "--rho-prior-bias",
        type=float,
        default=4.0,
        help="Stage E prior anneal: initial bias added to the "
        "rho_head's output. With the default 4.0 the initial "
        "ρ ≈ sigmoid(4) = 0.982, matching Stage D's "
        "strict-diagonal pin. Gradient pulls the bias down "
        "as training learns useful per-op factors. Set to "
        "0 to disable the prior anneal entirely.",
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
        "--lagrangian-constraint",
        action="append",
        default=[],
        metavar="NAME>=THRESH",
        help="Stage F hard-constraint deployment. Each occurrence adds a "
        "constraint of the form `<reward_name>>=<threshold>` (rewards "
        "are 'higher is better'; e.g. cosine_sim>=0.8, or for cost "
        "components flops>=-1e10 to cap FLOPs at 1e10). The trainer "
        "augments the per-step advantage with -λ_i · max(0, t_i - r_i) "
        "and updates λ_i ≥ 0 by dual ascent on the mean violation. "
        "May be repeated; default no constraints (= unconstrained PPO).",
    )
    p.add_argument(
        "--lagrangian-lr",
        type=float,
        default=1e-2,
        help="Dual-ascent step size on the Lagrangian multipliers, applied "
        "once per episode against the mean per-step violation.",
    )
    # Cosine-similarity band: keep cosine_sim in [lower, upper] via the
    # Lagrangian. Lower nudges the policy away from collapsing accuracy to
    # zero; upper prevents it from saturating at perfect agreement and
    # spending the remaining compute on quality nobody can spend. Set
    # ``--cosine-lower-bound <= 0`` or ``--cosine-upper-bound >= 1`` to
    # disable either side.
    p.add_argument(
        "--cosine-lower-bound",
        type=float,
        default=0.8,
        help="Floor on cosine_sim enforced via a Lagrangian multiplier. "
        "Default 0.8. Pass 0.0 to disable.",
    )
    p.add_argument(
        "--cosine-upper-bound",
        type=float,
        default=0.9,
        help="Ceiling on cosine_sim enforced via a Lagrangian multiplier. "
        "Default 0.9. Pass 1.0 to disable.",
    )
    p.add_argument(
        "--calibrate-steps",
        type=int,
        default=0,
        help="Stage G: after the main training loop, run this many "
        "few-shot calibration episodes against the env's "
        "calibration samples. Only the factor head and Set "
        "Transformer aggregator update; all other modules are "
        "frozen via gradient masking. Reward weights for these "
        "episodes are quality-focused (cosine + Frob) so the "
        "calibration signal matches what the spec calls for.",
    )
    p.add_argument(
        "--calibrate-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for the calibration optimizer. "
        "Independent of the main optimizer's state, since "
        "the calibration signal is a different objective.",
    )
    p.add_argument(
        "--potential-shaping",
        type=float,
        default=0.0,
        help="Stage C: scale on potential-based reward shaping. "
        "Augments per-step reward with `c · (γ V_ψ(s') - V_ψ(s))` "
        "(values stop-gradient'd) so long-horizon vertex-elim "
        "rollouts get a denser per-step learning signal "
        "without biasing the optimum (Ng et al. 1999). "
        "0.0 disables; spec recommends starting around 0.1.",
    )
    p.add_argument(
        "--bc-warmstart-steps",
        type=int,
        default=0,
        help="Stage C: number of supervised cross-entropy steps "
        "training the vertex pointer to match the Markowitz "
        "min-degree heuristic before PPO begins. 0 = skip.",
    )
    p.add_argument(
        "--bc-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the BC warm-start phase.",
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


def _compute_sparsity_ratios(factors_py: tuple[int, ...]) -> np.ndarray:
    """Map factor values to sparsity ratios per the spec (§2.2).

    For positive factors:  ``ρ_i = (1 - 1/f_i) / (1 - 1/f_max)``.
    Special values:  ``f == -1`` → ρ = 1 (gcd-collapse / strict-diagonal
    convention);  ``f == 0`` or ``f == 1`` → ρ = 0 (dense / no compression).

    Returns the ratios sorted ascending alongside an order array so the
    rule-policy decoder sees a monotone table for ``searchsorted``.
    """
    factors = np.asarray(factors_py, dtype=np.float32)
    pos = factors[factors > 1]
    if pos.size == 0:
        # Pure dense table — every factor maps to ρ=0 except f=-1 which
        # we lift to 1. Avoid 1/(1-1/1) = 0/0.
        f_max = 2.0
    else:
        f_max = float(pos.max())

    ratios = np.zeros_like(factors)
    for i, f in enumerate(factors):
        if f == -1:
            ratios[i] = 1.0
        elif f <= 1:
            ratios[i] = 0.0
        else:
            ratios[i] = (1.0 - 1.0 / f) / (1.0 - 1.0 / f_max)
    return ratios


def _build_factor_table(args, use_autoreg: bool):
    factors_py = tuple(_parse_int_list(args.factors))
    if not factors_py:
        raise ValueError("--factors must contain at least one factor value")

    if use_autoreg:
        if args.max_rules > MAX_RULES_PER_VERTEX:
            raise ValueError(
                f"--max-rules ({args.max_rules}) exceeds env-side "
                f"MAX_RULES_PER_VERTEX ({MAX_RULES_PER_VERTEX})"
            )
        max_rules = args.max_rules
    else:
        # Single-rule policy emits exactly one rule with factor=-1; the factor table is unused.
        factors_py = (-1,)
        max_rules = 1

    # Stage E: when sparsity-ratio reparam is on, the factor table must be
    # sorted by ρ ascending and have distinct ratios so the policy's
    # `searchsorted` snap is well-defined. We sort here so trajectory
    # factor_idx values index the same (sorted) table the policy sees.
    if use_autoreg and getattr(args, "sparsity_ratio", False):
        ratios = _compute_sparsity_ratios(factors_py)
        if len(set(float(r) for r in ratios)) != len(ratios):
            raise ValueError(
                f"--sparsity-ratio requires factors with distinct ρ-values. "
                f"Got factors {factors_py} → ratios {ratios.tolist()}. "
                f"e.g. `--factors=1,2,4` works (ρ ∈ {{0, 2/3, 1}}); "
                f"avoid mixing -1 with f≥f_max because both produce ρ=1."
            )
        perm = np.argsort(ratios)
        factors_py = tuple(int(factors_py[i]) for i in perm)

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

    `variant` overrides ``args.variant`` if given (used by the curriculum
    scheduler when stepping through stages). Raises if the preset is not
    yet wired (currently ``compress``, which depends on the atomic-COMPRESS
    action — see graphax.sparse.micro_actions and the heads.py rewrite).
    """
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


def make_curriculum_schedule(args, curriculum_stages):
    """Build an optax-compatible LR schedule for a curriculum run.

    Each stage gets its own ``cosine_warmup_exp_decay_lr`` hill whose
    ``period`` equals the stage's optimizer-step budget
    (``stage_episodes × ppo_epochs × minibatches``). Warm-up takes the
    first ``args.curriculum_warmup_frac`` of the stage; exponential
    decay covers the rest, ending at ``args.lr_decay_min_mult * lr``.
    Returns a JAX-traceable callable ``schedule(step) -> lr`` consumable
    by ``optax.adam(schedule)``.

    Boundaries are pre-computed at Python time so per-stage warmup
    counts are concrete inside the inner ``cosine_warmup_exp_decay_lr``
    calls (which read them with ``int(...)``).
    """
    steps_per_episode = args.ppo_epochs * args.minibatches
    stage_step_counts = [max(n * steps_per_episode, 1) for _, n in curriculum_stages]
    boundaries: list[int] = [0]
    for s in stage_step_counts:
        boundaries.append(boundaries[-1] + s)
    warmup_frac = float(args.curriculum_warmup_frac)

    def schedule(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        # Default LR (used when step falls outside any stage — shouldn't
        # happen but the optimizer will keep stepping after the last
        # episode if the trainer over-runs).
        lr = jnp.asarray(
            args.lr * args.lr_decay_min_mult,
            dtype=jnp.float32,
        )
        for i, stage_steps in enumerate(stage_step_counts):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            warmup_steps = max(int(stage_steps * warmup_frac), 1)
            local_step = step_f - float(lo)
            stage_lr = cosine_warmup_exp_decay_lr(
                local_step,
                args.lr,
                stage_steps,
                warmup_steps,
                end_mult=args.lr_decay_min_mult,
            )
            in_stage = (step_f >= float(lo)) & (step_f < float(hi))
            lr = jnp.where(in_stage, stage_lr, lr)
        return lr

    return schedule


def _current_stage_at(
    stages: list[tuple[str, int]],
    ep: int,
) -> str:
    """Map an episode index to the variant of the stage it falls into.

    Used by the curriculum runner to print a stage-transition message
    once per boundary. Returns the last stage's name if ``ep`` exceeds
    the total budget (shouldn't happen — total_episodes is set to the
    sum of stage counts — but harmless).
    """
    cumulative = 0
    for name, n in stages:
        if ep < cumulative + n:
            return name
        cumulative += n
    return stages[-1][0] if stages else ""


def _micro_introduction_stage(
    curriculum_stages: list[tuple[str, int]],
    allow_compress: bool,
) -> int:
    """First stage index where the dynamic head sees gradient signal.

    The micro_action_policy heads only carry useful signal when at least
    one of DIAG / COMPRESS is legal — ``ve_only`` forces END every
    sub-step, so the head's outputs are masked to 0 entropy / 0 log-prob
    contributions and gradients vanish. Returns ``len(stages)`` if the
    head is never introduced (i.e., all stages are ve_only) so the
    "past introduction" check stays well-defined.
    """
    for stage_idx, (variant, _) in enumerate(curriculum_stages):
        legal = _op_legality_for_variant(variant, allow_compress)
        if float(legal[0]) > 0.5 or float(legal[1]) > 0.5:
            return stage_idx
    return len(curriculum_stages)


def _pin_rules_for_variant(variant: str) -> bool:
    """Per-variant `pin_rules_to_exact` flag for the legacy rule head.

    Only ``ve_only`` requires the pinned (no-rules) output; every other
    variant lets the rule head sample normally. The dynamic-substeps
    path uses :func:`_op_legality_for_variant` instead.
    """
    return variant == "ve_only"


def _op_legality_for_variant(
    variant: str,
    allow_compress: bool,
) -> jax.Array:
    """Per-variant op-type legality mask for the dynamic action space.

    Maps a comparison-study variant to the ``(NUM_OPS,) = (DIAG,
    COMPRESS, END)`` float32 mask consumed by
    :meth:`Agent.sample_action_dynamic`. END is always legal so the
    sub-episode can terminate. ``allow_compress`` overrides the
    COMPRESS slot to 0 for any variant — useful while the graphax-side
    real-COMPRESS wiring is pending.

    * ``custom`` / ``full``: every op legal (gated by allow_compress).
    * ``ve_only``: force END at every sub-step (the dynamic-mode
      analogue of ``--pin-rules-to-exact``).
    * ``diag_gcd`` / ``diag_factor``: DIAG and END only.
    * ``compress``: COMPRESS and END only (requires allow_compress).
    """
    diag = 1.0
    compress = 1.0 if allow_compress else 0.0
    end = 1.0
    if variant == "ve_only":
        return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    if variant in ("diag_gcd", "diag_factor"):
        return jnp.array([diag, 0.0, end], dtype=jnp.float32)
    if variant == "compress":
        if not allow_compress:
            raise ValueError(
                "Variant 'compress' requires --allow-compress (and the "
                "graphax vertex_elimination_jaxpr rewrite to actually "
                "consume COMPRESS micro-actions through the env)."
            )
        return jnp.array([0.0, compress, end], dtype=jnp.float32)
    # `custom` and `full` (and anything else) get the unrestricted mask
    # gated by allow_compress.
    return jnp.array([diag, compress, end], dtype=jnp.float32)


def _parse_curriculum(spec: str) -> list[tuple[str, int]]:
    """Parse `stage1:N1,stage2:N2,...` into a list of (variant, episodes) pairs.

    Empty string -> empty list (no curriculum). Whitespace tolerated.
    Validates every variant name against ``VARIANT_PRESETS`` and every
    episode count is a positive integer.
    """
    if not spec.strip():
        return []
    stages: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Curriculum stage '{chunk}' missing ':'. Expected "
                "`variant:episode_count`."
            )
        name, n_str = chunk.split(":", 1)
        name = name.strip()
        if name not in VARIANT_PRESETS:
            raise ValueError(
                f"Curriculum stage variant '{name}' unknown. Valid: "
                f"{list(VARIANT_PRESETS)}."
            )
        try:
            n_episodes = int(n_str.strip())
        except ValueError as e:
            raise ValueError(
                f"Curriculum stage '{chunk}': episode count '{n_str}' is "
                "not an integer."
            ) from e
        if n_episodes <= 0:
            raise ValueError(f"Curriculum stage '{chunk}': episode count must be > 0.")
        stages.append((name, n_episodes))
    return stages


def _select_variant(args) -> tuple[bool, bool]:
    """Return `(use_pointer, use_autoreg)` — independent flags."""
    return (not args.no_ptr, not args.not_autoreg)


def _variant_label(use_pointer: bool, use_autoreg: bool) -> str:
    v = "pointer" if use_pointer else "mlp-vertex"
    r = "autoreg" if use_autoreg else "single-rule"
    return f"{v}+{r}"


def _build_agent(
    use_pointer: bool,
    use_autoreg: bool,
    args,
    total_v: int,
    num_factors: int,
    max_rules: int,
    key,
):
    encoder_keys = jrand.split(key, 14)
    embedding = eqx.nn.Embedding(args.vocab_size, args.embd_dim, key=encoder_keys[0])
    pos_enc = PositionalEncoder(args.embd_dim, MAX_TOKENS)
    encoder = Encoder(
        args.num_layers,
        args.num_heads,
        args.embd_dim,
        args.hidden_dim,
        key=encoder_keys[1],
    )
    if use_pointer:
        vertex_policy = PointerVertexPolicy(
            num_vertices=total_v,
            embd_dim=args.embd_dim,
            num_heads=args.num_heads,
            key=encoder_keys[2],
        )
    else:
        vertex_policy = MLPVertexPolicy(
            num_vertices=total_v,
            embd_dim=args.embd_dim,
            hidden_dims=_parse_int_list(args.policy_dims),
            key=encoder_keys[2],
        )
    if use_autoreg:
        # Sparsity-ratio is a legacy-only factor head; the dynamic-substeps
        # path uses the prime-exponent head instead. Silently downgrade to
        # plain AutoregRulePolicy when both flags are on so the unused
        # legacy rule_policy doesn't carry the heavier ρ module.
        use_sparsity_ratio = args.sparsity_ratio and not getattr(
            args, "dynamic_substeps", False
        )
        if use_sparsity_ratio:
            sparsity_ratios = _compute_sparsity_ratios(
                tuple(_parse_int_list(args.factors))
            )
            rule_policy = SparsityRatioAutoregRulePolicy(
                embd_dim=args.embd_dim,
                max_rules=max_rules,
                num_pair_choices=NUM_PAIR_CHOICES,
                num_factors=num_factors,
                sparsity_ratios=sparsity_ratios,
                key=encoder_keys[3],
                rho_init_bias=getattr(args, "rho_prior_bias", 4.0),
            )
        else:
            rule_policy = AutoregRulePolicy(
                embd_dim=args.embd_dim,
                max_rules=max_rules,
                num_pair_choices=NUM_PAIR_CHOICES,
                num_factors=num_factors,
                key=encoder_keys[3],
            )
    else:
        rule_policy = SingleRulePolicy(
            embd_dim=args.embd_dim,
            max_rules=max_rules,
            num_pair_choices=NUM_PAIR_CHOICES,
            num_factors=num_factors,
            sp_dims=_parse_int_list(args.policy_dims),
            key=encoder_keys[3],
        )
    # One single-output MLP per training reward (flops / peak_memory /
    # frob_residual). Per-head split keeps gradient scales sane across the
    # qualitatively different reward families and matches the per-head GAE
    # and preference-vector scalarization in `train_episode`.
    value_dims = _parse_int_list(args.value_dims)
    value_head_flops = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[4])
    value_head_mem = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[5])
    value_head_acc = MLP(args.embd_dim, 1, value_dims, key=encoder_keys[12])
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
        rule_policy=rule_policy,
        value_head_flops=value_head_flops,
        value_head_mem=value_head_mem,
        value_head_acc=value_head_acc,
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


def _scale_output_heads(agent, scale: float, use_pointer: bool, use_autoreg: bool):
    """Scale policy-head weights so the initial action distribution is near-uniform."""
    if use_pointer:
        agent = scale_module_weight(
            agent, lambda a: a.vertex_policy.pointer_proj.weight, scale
        )
    else:
        agent = scale_module_weight(
            agent, lambda a: a.vertex_policy.head.layers[-2].weight, scale
        )
    if use_autoreg:
        agent = scale_module_weight(
            agent, lambda a: a.rule_policy.decoder.pair_head.weight, scale
        )
        # The factor side is either ``factor_head`` (categorical, K logits) or
        # ``rho_head`` (sparsity-ratio scalar) depending on which autoreg
        # variant was constructed. Detect and scale the right one.
        if hasattr(agent.rule_policy.decoder, "rho_head"):
            agent = scale_module_weight(
                agent, lambda a: a.rule_policy.decoder.rho_head.weight, scale
            )
        else:
            agent = scale_module_weight(
                agent, lambda a: a.rule_policy.decoder.factor_head.weight, scale
            )
    else:
        agent = scale_module_weight(
            agent, lambda a: a.rule_policy.sp_head.layers[-2].weight, scale
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


def _action_to_pylist(vertex_seq, pair_seq, factor_seq, max_rules, factor_table_np):
    """Decode legacy `(vertex, pair, factor)` sequences into copy-pastable lists.

    Each entry is ``(vertex, [diag(idx1, idx2, factor), ...])`` — every emitted
    rule renders as a call to ``graphax.sparse.micro_actions.diag``. PAIR_STOP
    (and every slot past it) is dropped from the output so a no-rule vertex
    becomes the empty list ``[]``.

    This is the LEGACY-mode display formatter. In `--dynamic-substeps`
    mode the legacy `pair_seq` / `factor_seq` are zero-filled and the
    real action lives in the `micro_*_seq` fields — call
    :func:`_action_to_pylist_dynamic` instead.
    """
    out = []
    for v_idx, p_row, f_row in zip(vertex_seq, pair_seq, factor_seq):
        rules: list[str] = []
        for slot in range(max_rules):
            p = int(p_row[slot])
            if p == PAIR_STOP:
                break
            base = _PAIR_TO_BASE[p]
            base_idx1, base_idx2 = int(base[0]), int(base[1])
            factor = int(factor_table_np[int(f_row[slot])])
            rules.append(f"diag({base_idx1}, {base_idx2}, {factor})")
        out.append((int(v_idx) + 1, rules))
    return out


def _action_to_pylist_dynamic(
    vertex_seq,
    op_seq,
    i_seq,
    j_seq,
    factor_seq,
    kind_seq,
    max_substeps,
):
    """Decode typed micro-action sequences into copy-pastable per-vertex lists.

    Each entry is ``(vertex, [<call>, ...])`` where ``<call>`` is one of:

    * ``diag(i, j, factor)`` for an OP_DIAG sub-step.
    * ``compress("kind", axis)`` for an OP_COMPRESS sub-step (the kind
      is the string from :data:`COMPRESS_KINDS` at the sampled index).

    OP_END (and every sub-step past it) is dropped from the output, so a
    vertex whose sub-episode is just OP_END renders as ``(v, [])`` —
    matching how the user reads the no-approximation case.
    """
    out = []
    for v_idx, op_row, i_row, j_row, f_row, k_row in zip(
        vertex_seq,
        op_seq,
        i_seq,
        j_seq,
        factor_seq,
        kind_seq,
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


def parse_lagrangian_constraints(specs: list[str]) -> list[tuple[int, float, int]]:
    """Parse ``--lagrangian-constraint`` strings to ``(idx, threshold, sign)`` triples.

    Supported forms:

    * ``<reward_name>>=<threshold>``  → ``reward[idx] ≥ threshold``
      (sign +1; violation when reward dips below the floor).
    * ``<reward_name><=<threshold>``  → ``reward[idx] ≤ threshold``
      (sign −1; violation when reward exceeds the ceiling — useful for
      bounding quality terms like ``cosine_sim<=0.9`` so the policy is
      pushed to leave compute / memory headroom rather than maxing out
      perfect agreement).

    Rewards are higher-is-better (cost components are stored negated).
    """
    parsed: list[tuple[int, float, int]] = []
    for s in specs:
        if ">=" in s:
            op = ">="
            sign = 1
        elif "<=" in s:
            op = "<="
            sign = -1
        else:
            raise ValueError(
                f"--lagrangian-constraint must be of the form NAME>=THRESH "
                f"or NAME<=THRESH, got {s!r}"
            )
        name, thresh_s = s.split(op, 1)
        name = name.strip()
        if name not in REWARD_INDEX:
            raise ValueError(
                f"Unknown reward name {name!r} in constraint {s!r}; "
                f"valid names: {list(REWARD_INDEX.keys())}"
            )
        parsed.append((REWARD_INDEX[name], float(thresh_s.strip()), sign))
    return parsed


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
    """Build the (NUM_VALUE_HEADS,) = (3,) static preference vector.

    Indexes the three training rewards (flops / peak_memory / frob_residual)
    -- the value head and advantage path operate on exactly these three.
    The `--cmp-type` and `--mem-type` flags only affect host-side display:
    training always learns flops + peak_memory + frob_residual regardless
    of those settings.
    """
    weights = np.zeros(NUM_VALUE_HEADS, dtype=np.float32)
    if "cmp" in args.rewards:
        weights[0] = args.lambda_cmp
    if "mem" in args.rewards:
        weights[1] = args.lambda_mem
    if "acc" in args.rewards or args.lambda_frob != 0.0:
        weights[2] = args.lambda_frob if args.lambda_frob != 0.0 else 1.0
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
        "rule_policy.decoder.pair_head",
        "rule_policy.decoder.rnn_proj",
        "rule_policy.decoder.pair_embed",
        "rule_policy.decoder.slot_embed",
        "rule_policy.sp_head",  # legacy SingleRulePolicy axis head
    ),
    "factor": (
        "rule_policy.decoder.factor_head",
        "rule_policy.decoder.rho_head",  # Stage E sparsity-ratio variant
        "rule_policy.decoder.factor_embed",
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
    prime-exponent). The curriculum runner uses them to apply
    introduction-stage-aware LR scaling: a head's gradient runs at
    full LR in the first stage where it sees signal, then drops to
    ``--curriculum-existing-head-mult`` in subsequent stages.

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


def _build_calibration_mask(agent):
    """Stage G: True = trainable during calibration (factor head + aggregator),
    False = frozen (encoder, vertex head, axis-pair head, value heads, …)."""
    return _build_param_mask(
        agent,
        lambda p: _path_in(p, "factor") or _path_in(p, "aggregator"),
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

    Combines the Stage D head-LR ramp (per-head multiplier on axis / factor
    params), the curriculum existing-head multipliers on vertex_policy
    and micro_action_policy, and the Stage G freeze mask (zero gradient
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


def run_calibration_phase(
    agent,
    opt_state,
    env,
    env_episode_template_args,
    *,
    train_episode,
    reset_envs,
    num_envs,
    args,
    global_step,
    key,
):
    """Stage G: few-shot calibration loop.

    Freezes everything except the factor head + Set Transformer aggregator
    (via ``cal_mask``), forces a quality-focused preference vector, and
    runs ``args.calibrate_steps`` extra episodes through the same
    ``train_episode`` JIT'd path used during PPO. Reuses the main
    optimizer state so no JIT recompile fires.

    ``env_episode_template_args = (closed_jaxpr, base_args, argnums)``
    captures the jaxpr details needed to recompute per-vertex features each
    episode without pulling all of `main`'s closure into the call site.
    """
    closed_jaxpr, base_args, argnums = env_episode_template_args

    cal_mask = _build_calibration_mask(agent)
    leaves = jax.tree_util.tree_leaves(cal_mask)
    n_cal = sum(int(jnp.sum(l.astype(jnp.int32))) for l in leaves)
    n_total = sum(int(l.size) for l in leaves)
    print(
        f"\nStage G: calibrating for {args.calibrate_steps} episodes; "
        f"{n_cal}/{n_total} param leaves are trainable "
        f"(factor head + set_transformer_agg). All other modules frozen."
    )

    # Quality-focused preference: only the acc head (frob_residual) carries
    # the learning signal during calibration. flops and peak_memory weights
    # are 0 so the advantage scalarization is purely quality-driven, matching
    # the spec's "supervised against the Frobenius quality signal". cosine_sim
    # is no longer a value head under the 3-head split; its host-side display
    # is preserved via the 8-vec `traj.reward`.
    cal_pref = jnp.broadcast_to(
        jnp.zeros(NUM_VALUE_HEADS, dtype=jnp.float32).at[2].set(1.0),
        (num_envs, NUM_VALUE_HEADS),
    )

    # Stage G never enforces hard constraints — calibration is a supervised
    # quality nudge, not a constraint-satisfaction problem. Pass empty
    # constraint arrays so the augmentation in train_episode is a no-op.
    no_idx = jnp.zeros((0,), dtype=jnp.int32)
    no_thr = jnp.zeros((0,), dtype=jnp.float32)
    no_sign = jnp.zeros((0,), dtype=jnp.float32)
    no_lam = jnp.zeros((0,), dtype=jnp.float32)

    for step_idx in range(args.calibrate_steps):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
        vertex_features = _episode_vertex_features(
            args,
            closed_jaxpr.jaxpr,
            tuple(closed_jaxpr.literals),
            base_args,
            eval_samples=eval_samples,
            argnums=argnums,
        )
        env_states = reset_envs(env_episode)
        # Calibration uses the env's current op-legality (no per-stage
        # override). For dynamic-substeps calibration the caller will
        # provide the active override via args.
        if args.dynamic_substeps:
            cal_override = jnp.array(
                [1.0, 1.0 if args.allow_compress else 0.0, 1.0],
                dtype=jnp.float32,
            )
        else:
            cal_override = jnp.ones((NUM_OPS,), dtype=jnp.float32)
        (
            agent,
            opt_state,
            _,
            _metrics,
            totals,
            _actions,
            global_step,
            _new_lam,
            _diag_pack,
        ) = train_episode(
            agent,
            opt_state,
            env_states,
            env_episode,
            vertex_features,
            cal_pref,
            global_step,
            ep_key,
            cal_mask,
            no_lam,
            no_idx,
            no_thr,
            no_sign,
            cal_override,
            jnp.array(1.0, dtype=jnp.float32),  # calibration: no curriculum scaling
            # Calibration runs with the env's static pin_rules_to_exact:
            # we honour args.pin_rules_to_exact (Python bool) by lifting it
            # into a JAX scalar so the per-call API stays uniform.
            jnp.asarray(args.pin_rules_to_exact, dtype=jnp.bool_),
            jnp.array(1.0, dtype=jnp.float32),  # calibration: no micro scaling
        )
        cosine = float(jnp.mean(totals[:, REWARD_INDEX["cosine_sim"]]))
        neg_frob = float(jnp.mean(totals[:, REWARD_INDEX["frob_residual"]]))
        print(
            f"  cal step {step_idx:3d}/{args.calibrate_steps}  "
            f"cosine_sim={cosine:.4f}  frob_residual={-neg_frob:.4f}"
        )

    return agent, global_step, key


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
    used by the main training loop, the BC warm-start, and the calibration
    phase. Returns a `jnp.float32` array.
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


def _markowitz_bc_dataset(
    env, target_order, total_v, embd_dim, vertex_valid_static, num_valid
):
    """Roll the env forward following ``target_order`` to produce a list of
    (tokens, eqn_ids, vertex_avail, residual_state, target_vertex_idx) tuples.

    The agent is trained to pick the *index* of each Markowitz vertex (0-indexed
    over the original ``total_v`` action space).
    """
    state = env.reset(num_envs=0)
    residual_state = jnp.zeros((total_v, embd_dim), dtype=jnp.float32)
    samples = []
    empty_specs = (
        jnp.full(
            (MAX_RULES_PER_VERTEX, 3),
            -1,
            dtype=jnp.int32,
        )
        .at[..., 2]
        .set(0)
    )

    for v_id in target_order:
        vertex_avail = vertex_avail_at_step(
            state,
            vertex_valid_static,
            total_v,
            num_valid,
        )
        target_idx = jnp.asarray(int(v_id) - 1, dtype=jnp.int32)
        samples.append(
            (
                state.tokens.astype(jnp.int32),
                state.eqn_ids.astype(jnp.int32),
                vertex_avail,
                residual_state,
                target_idx,
            )
        )
        action = StepAction(
            target_vertex=jnp.asarray(int(v_id), dtype=jnp.int32),
            rule_specs=empty_specs,
        )
        out = env.step(state, action)
        state = out.state
        # Best-effort residual update — encode the current state and grab the
        # chosen vertex's context, like the rollout would. Compute is small.
        # (For BC the residual is only used to *condition* the policy; the
        # gradient signal still comes from the cross-entropy loss.)
        residual_state = jnp.zeros_like(residual_state)
    return samples


def _bc_warmstart_step(
    agent,
    opt_state,
    optimizer,
    sample,
    vertex_features,
    pair_valid_mask,
    pair_factor_mask,
    key,
):
    tokens, eqn_ids, vertex_avail, residual_state, target_idx = sample

    def loss_fn(agent):
        vertex_logits, _vertex_contexts, _value = agent.encode(
            tokens,
            eqn_ids=eqn_ids,
            vertex_features=vertex_features,
            residual_state=residual_state,
            key=key,
        )
        masked_logits = jnp.where(vertex_avail > 0.5, vertex_logits, -1e9)
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        return -log_probs[target_idx]

    loss, grads = eqx.filter_value_and_grad(loss_fn)(agent)
    updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        eqx.filter(agent, eqx.is_inexact_array),
    )
    new_agent = eqx.apply_updates(agent, updates)
    return new_agent, new_opt_state, loss


def run_bc_warmstart(
    agent,
    env,
    target_order,
    total_v,
    embd_dim,
    vertex_valid_static,
    num_valid,
    vertex_features,
    pair_valid_mask,
    pair_factor_mask,
    n_steps,
    lr,
    key,
):
    """Run ``n_steps`` of supervised cross-entropy on the vertex pointer head
    against the Markowitz heuristic. Returns the warm-started agent."""
    print(f"BC warmstart: {n_steps} steps against Markowitz order {list(target_order)}")
    bc_optimizer = optax.adam(lr)
    bc_opt_state = bc_optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    samples = _markowitz_bc_dataset(
        env,
        target_order,
        total_v,
        embd_dim,
        vertex_valid_static,
        num_valid,
    )
    losses = []
    for step in range(n_steps):
        sample = samples[step % len(samples)]
        step_key, key = jrand.split(key)
        agent, bc_opt_state, loss = _bc_warmstart_step(
            agent,
            bc_opt_state,
            bc_optimizer,
            sample,
            vertex_features,
            pair_valid_mask,
            pair_factor_mask,
            step_key,
        )
        losses.append(float(loss))
        if step == 0 or step == n_steps - 1 or step % max(1, n_steps // 10) == 0:
            print(f"  bc step {step:4d}/{n_steps}  loss={float(loss):7.4f}")
    print(f"  BC final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _setup_jax_compile_cache() -> None:
    """Enable XLA's persistent compile cache so successive runs of the same
    config skip JIT recompile. Honoured by JAX>=0.4.16 via env-var; we set a
    sensible default if the user hasn't already, then turn it on through the
    `jax.config` API too (belt-and-braces — JAX is in transition between
    the two surfaces). The cache lives at ``~/.cache/jax-compilation-cache``
    by default and is keyed on the HLO + flags + JAX/XLA version, so stale
    entries are never hit."""
    cache_dir = os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jax-compilation-cache"),
    )
    try:
        from jax.experimental.compilation_cache import compilation_cache

        compilation_cache.set_cache_dir(cache_dir)
    except Exception:
        pass


def main():
    args = make_argparser().parse_args()

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

    # Parse the curriculum spec early so misspelled stages fail fast. The
    # full runner (which would toggle pin_rules_to_exact across stages and
    # ramp per-head LR via cosine_warmup_exp_decay_lr) needs the heads.py
    # refactor: today's rollout_fn / loss_fn capture pin_rules_to_exact and
    # the factor table at JIT-compile time, so cross-stage transitions
    # require either re-jitting (acceptable but unimplemented) or threading
    # those values in as explicit per-call arguments (preferred — coming
    # with heads.py). For now we parse + validate the spec so the CLI
    # surface is stable, and raise a clear error if a non-empty curriculum
    # is requested.
    curriculum_stages = _parse_curriculum(args.curriculum)
    if curriculum_stages:
        # Both dynamic-substeps and legacy curricula are wired now —
        # stage transitions toggle (op_legality_override, vertex_mult,
        # pin_rules_to_exact) per call without forcing a recompile.
        # Legacy curricula still need to share the SAME --factors and
        # --max-rules across stages: the agent's rule head is sized at
        # build time and isn't rebuilt mid-run. Validate that here so
        # mis-specified stages fail fast rather than silently producing
        # wrong-shape rule outputs.
        if not args.dynamic_substeps:
            from collections import Counter

            stage_factors = []
            stage_max_rules = []
            for stage_name, _ in curriculum_stages:
                preset = VARIANT_PRESETS.get(stage_name)
                if preset is None:
                    raise NotImplementedError(
                        f"Curriculum stage '{stage_name}' is not yet wired "
                        "(blocked on the heads.py rewrite or atomic COMPRESS "
                        "in graphax)."
                    )
                stage_factors.append(preset.get("factors", args.factors))
                stage_max_rules.append(preset.get("max_rules", args.max_rules))
            f_counts = Counter(stage_factors)
            r_counts = Counter(stage_max_rules)
            if len(f_counts) > 1 or len(r_counts) > 1:
                raise NotImplementedError(
                    "Legacy-mode curriculum requires every stage to share "
                    "--factors and --max-rules (the rule head's output dim "
                    "is fixed at agent-build time). Got per-stage factors "
                    f"{stage_factors!r}, max_rules {stage_max_rules!r}. "
                    "Add --dynamic-substeps for the heads.py path that "
                    "doesn't have this restriction, or unify the stages' "
                    "factor table."
                )
        args.episodes = sum(n for _, n in curriculum_stages)
        mode_label = "dynamic-substeps" if args.dynamic_substeps else "legacy"
        print(
            "curriculum: "
            + " → ".join(f"{name}:{n}" for name, n in curriculum_stages)
            + f"  (total {args.episodes} episodes, {mode_label} mode)"
        )

    use_pointer, use_autoreg = _select_variant(args)
    variant_label = _variant_label(use_pointer, use_autoreg)

    main_device = _resolve_main_device(args)
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    _setup_jax_compile_cache()

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
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    env_target_fun = target_fn if "acc" in args.rewards else None
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
        terminal_rewards_only=args.terminal_rewards_only,
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
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(
        args, use_autoreg
    )
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
    # value / advantage path operates on the 3-vec (flops / peak_memory /
    # frob_residual); see HEAD_REWARD_INDICES and `_build_head_weights`.
    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
    head_reward_weights_np = _build_head_weights(args)
    head_reward_weights = jnp.asarray(head_reward_weights_np, dtype=jnp.float32)
    cmp_idx = _cmp_reward_index(args.cmp_type)
    mem_idx = _mem_reward_index(args.mem_type)
    cosine_idx = REWARD_INDEX["cosine_sim"]
    frob_idx = REWARD_INDEX["frob_residual"]

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
    # int32 / float32 lookup arrays); op_legality_override is a (3,)
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
    agent = _build_agent(
        use_pointer, use_autoreg, args, total_v, num_factors, max_rules, agent_key
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale, use_pointer, use_autoreg)
    if args.pretrained_encoder:
        from alphagrad.approx.pretrain import load_pretrained_encoder

        agent = load_pretrained_encoder(args.pretrained_encoder, agent)
        print(f"Loaded pretrained encoder from {args.pretrained_encoder}")

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

    if args.bc_warmstart_steps > 0:
        from alphagrad.approx.heuristics import markowitz_order

        target_order = markowitz_order(closed_jaxpr.jaxpr, env.valid_vertices)
        # BC needs vertex_features for the data-conditioned encode; reuse the
        # episode-0 features against the env's initial args so the supervised
        # signal is computed in a regime consistent with the upcoming PPO.
        bc_vertex_features = _episode_vertex_features(
            args,
            closed_jaxpr.jaxpr,
            tuple(closed_jaxpr.literals),
            tuple(xs),
            eval_samples=None,
            argnums=tuple(argnums),
        )
        bc_key, key = jrand.split(key)
        agent = run_bc_warmstart(
            agent,
            env,
            target_order,
            total_v,
            args.embd_dim,
            vertex_valid_static,
            num_valid,
            bc_vertex_features,
            pair_valid_mask,
            pair_factor_mask,
            args.bc_warmstart_steps,
            args.bc_lr,
            bc_key,
        )
    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent,
        )

    # Optimiser. Default = single cosine decay across the whole run.
    # When --curriculum is set, swap in a piecewise schedule that gives
    # each stage its own cosine-warmup + exp-decay hill so newly-
    # introduced variants ramp up gently and existing-variant heads
    # don't get blown out at stage boundaries.
    if curriculum_stages:
        schedule = make_curriculum_schedule(args, curriculum_stages)
        print(
            "curriculum LR: piecewise cosine_warmup_exp_decay across "
            + " → ".join(
                f"{name}({n * args.ppo_epochs * args.minibatches}st)"
                for name, n in curriculum_stages
            )
            + f" (warmup_frac={args.curriculum_warmup_frac}, "
            f"end_mult={args.lr_decay_min_mult})"
        )
    else:
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

        # Stage B.4.next: encode the initial residual jaxpr once if
        # --cache-encoding is on. The cached encoding is closed over by
        # step_fn — it doesn't change within an episode.
        encode_key, scan_key = jrand.split(keys[0], 2)
        if args.cache_encoding:
            cached_encoding = agent.encode_once(
                env_state.tokens,
                eqn_ids=env_state.eqn_ids,
                key=encode_key,
            )
            initial_tokens = env_state.tokens
            initial_eqn_ids = env_state.eqn_ids
        else:
            cached_encoding = None
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

        def step_fn(carry, k):
            state, residual_state = carry
            sample_key, next_net_key = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid
            )

            if args.dynamic_substeps:
                (
                    vertex_idx,
                    micro_actions,
                    vertex_dist,
                    micro_op_dists,
                    micro_i_dists,
                    micro_j_dists,
                    micro_exp_dists,
                    micro_kind_dists,
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
                    cached_encoding=cached_encoding,
                    preference=preference if args.preference_conditioned else None,
                )
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
            else:
                (
                    vertex_idx,
                    pair_seq,
                    factor_seq,
                    vertex_dist,
                    pair_dists,
                    factor_dists,
                    value,
                    v_context,
                ) = agent.sample_action(
                    state.tokens,
                    vertex_avail_mask,
                    pair_valid_mask,
                    pair_factor_mask,
                    sample_key,
                    eqn_ids=state.eqn_ids,
                    vertex_features=vertex_features,
                    residual_state=residual_state,
                    cached_encoding=cached_encoding,
                    pin_rules_to_exact=pin_rules_to_exact_jax,
                    pin_factor_idx=pin_factor_idx,
                    preference=preference if args.preference_conditioned else None,
                )
                env_action = agent.to_env_action(
                    vertex_idx, pair_seq, factor_seq, factor_table
                )
                # Dynamic fields zero-filled; legacy fields populated.
                micro_op_seq = _dyn_zero_op_seq
                micro_i_seq = _dyn_zero_i_seq
                micro_j_seq = _dyn_zero_j_seq
                micro_exp_seq = _dyn_zero_exp_seq
                micro_factor_seq = _dyn_zero_factor_seq
                micro_compress_kind_seq = _dyn_zero_kind_seq
                micro_op_dists = _dyn_zero_op_dists
                micro_i_dists = _dyn_zero_i_dists
                micro_j_dists = _dyn_zero_j_dists
                micro_exp_dists = _dyn_zero_exp_dists
                micro_kind_dists = _dyn_zero_kind_dists
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
                if not args.cache_encoding
                else (
                    agent._decode_from_cache(
                        cached_encoding,
                        vertex_features,
                        new_residual,
                        preference=pref_arg,
                    )[2]
                )
            )

            if args.potential_shaping != 0.0:
                v_raw = inverse_reward_normalization_fn(
                    lax.stop_gradient(value),
                )
                v_next_raw = inverse_reward_normalization_fn(
                    lax.stop_gradient(next_value),
                )
                # F_t = γ Φ(s_{t+1}) - Φ(s_t), with Φ(s_{T+1}) ≡ 0 at the
                # terminal step. The (1 - done) gate zeros only the bootstrap
                # term so the per-episode total reduces to the constant
                # -Φ(s_0) (Ng et al. 1999) — preserves the optimum exactly.
                shaping = args.discount * v_next_raw * (1.0 - done) - v_raw
                rewards = rewards + args.potential_shaping * shaping

            # In cache-encoding mode the trajectory's tokens/eqn_ids fields
            # carry the *initial* episode tokens (constant across the
            # rollout) so the loss path can encode_once + decode_from_cache
            # against the same reference. In the legacy path they remain
            # the per-step residual jaxpr tokens.
            traj_tokens = (
                initial_tokens if args.cache_encoding else state.tokens
            ).astype(jnp.int32)
            traj_eqn_ids = (
                initial_eqn_ids if args.cache_encoding else state.eqn_ids
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
                discount=jnp.array(args.discount),
                vertex_avail_mask=vertex_avail_mask,
            )
            return (next_state, new_residual), (transition, raw_rewards)

        (final_state, _), (traj, all_raw_rewards) = lax.scan(
            step_fn,
            (env_state, init_residual),
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
        # Three batching regimes share one ``evaluate_action`` vmap:
        #   1. Per-trajectory cache (B.4.next): batch arrives `(E, K, ...)`,
        #      encode once per trajectory and replicate K times.
        #   2. Per-step cache: flat `(B, ...)`, encode each entry separately.
        #   3. No cache: flat `(B, ...)`, no cached encoding (None).
        # `cached_flat` is the only thing that changes between regimes.
        pref_or_none = (
            (lambda p: p) if args.preference_conditioned else (lambda _: None)
        )

        if batch.tokens.ndim == 3:
            E, K = batch.tokens.shape[:2]
            enc_key, eval_key = jrand.split(key, 2)
            cached_per_traj = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens[:, 0], batch.eqn_ids[:, 0], jrand.split(enc_key, E))
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(E * K, *x.shape[2:]),
                batch,
            )
            cached_flat = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, K, axis=0),
                cached_per_traj,
            )
            keys = jrand.split(eval_key, E * K)
        elif args.cache_encoding:
            B = batch.tokens.shape[0]
            enc_key, eval_key = jrand.split(key, 2)
            cached_flat = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens, batch.eqn_ids, jrand.split(enc_key, B))
            keys = jrand.split(eval_key, B)
        else:
            cached_flat = None
            keys = jrand.split(key, batch.tokens.shape[0])

        def _eval_one(toks, eids, rs, pref, vidx, pseq, fseq, vmask, cached, k):
            return agent.evaluate_action(
                toks,
                vidx,
                pseq,
                fseq,
                vmask,
                pair_valid_mask,
                pair_factor_mask,
                k,
                eqn_ids=eids,
                vertex_features=vertex_features,
                residual_state=rs,
                cached_encoding=cached,
                pin_rules_to_exact=pin_rules_to_exact_jax,
                pin_factor_idx=pin_factor_idx,
                preference=pref_or_none(pref),
            )

        if cached_flat is None:
            # vmap can't ingest `None` over the cached axis; collapse to a
            # variant that hard-codes ``cached=None`` instead.
            (log_probs, entropies, values, vertex_dist, pair_dists, factor_dists) = (
                jax.vmap(
                    lambda toks, eids, rs, pref, vidx, pseq, fseq, vmask, k: _eval_one(
                        toks, eids, rs, pref, vidx, pseq, fseq, vmask, None, k
                    )
                )(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
                    batch.preference,
                    batch.vertex_idx,
                    batch.pair_seq,
                    batch.factor_seq,
                    batch.vertex_avail_mask,
                    keys,
                )
            )
        else:
            (log_probs, entropies, values, vertex_dist, pair_dists, factor_dists) = (
                jax.vmap(_eval_one)(
                    batch.tokens,
                    batch.eqn_ids,
                    batch.residual_state,
                    batch.preference,
                    batch.vertex_idx,
                    batch.pair_seq,
                    batch.factor_seq,
                    batch.vertex_avail_mask,
                    cached_flat,
                    keys,
                )
            )

        old_log_probs = jax.vmap(old_log_prob_for_action)(
            batch.vertex_idx,
            batch.pair_seq,
            batch.factor_seq,
            batch.old_vertex_dist,
            batch.old_pair_dists,
            batch.old_factor_dists,
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

        # Per-sample sub-episode length: 1 (vertex) + active pair slots +
        # active factor slots, matching the masking in `AutoregRulePolicy.evaluate`
        # and `old_log_prob_for_action`. Dividing the joint entropy by this
        # keeps long sub-episodes from dominating the entropy bonus.
        is_stop_seq = batch.pair_seq == PAIR_STOP
        prior_stops = jnp.cumsum(
            is_stop_seq.astype(jnp.int32), axis=-1
        ) - is_stop_seq.astype(jnp.int32)
        pair_active_mask = (prior_stops == 0).astype(jnp.float32)
        factor_active_mask = ((prior_stops == 0) & (~is_stop_seq)).astype(jnp.float32)
        sub_episode_lengths = (
            1.0
            + jnp.sum(pair_active_mask, axis=-1)
            + jnp.sum(factor_active_mask, axis=-1)
        )
        entropy_loss = jnp.mean(entropies / sub_episode_lengths)

        # In scalar mode only slot 0 of the (value, return) pair carries
        # signal; ignore the other heads so they don't drift training on
        # zero targets.
        if args.loss_mode == "scalar":
            value_loss = jnp.mean(
                (values[..., 0] - reward_normalization_fn(batch.estim_returns[..., 0]))
                ** 2
            )
            explained_var = explained_variance(
                batch.norm_adv, batch.estim_returns[..., 0]
            )
        else:
            value_loss = jnp.mean(
                jnp.sum(
                    (values - reward_normalization_fn(batch.estim_returns)) ** 2,
                    axis=-1,
                )
            )
            explained_var = explained_variance(
                batch.norm_adv, jnp.sum(batch.estim_returns, axis=-1)
            )

        vertex_kl = jnp.mean(
            optax.kl_divergence(jnp.log(vertex_dist + 1e-7), batch.old_vertex_dist)
        )
        pair_kl = jnp.mean(
            jnp.sum(
                optax.kl_divergence(jnp.log(pair_dists + 1e-7), batch.old_pair_dists),
                axis=-1,
            )
        )
        factor_kl = jnp.mean(
            jnp.sum(
                optax.kl_divergence(
                    jnp.log(factor_dists + 1e-7), batch.old_factor_dists
                ),
                axis=-1,
            )
        )
        kl_div = vertex_kl + pair_kl + factor_kl

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
            # Per-component KLs (vertex / op / i / j / exp). Legacy path
            # exposes the closest available signal: (vertex_kl, pair_kl,
            # factor_kl, 0, 0). The op/i/j/exp slot semantics are
            # dynamic-mode specific; the legacy values fill the first
            # three to preserve some signal for the per-component KL
            # dashboard panels.
            jnp.stack(
                [
                    vertex_kl,
                    pair_kl,
                    factor_kl,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                ]
            ),
            # Per-component entropies — legacy path doesn't currently
            # surface vertex/pair/factor entropies separately (the
            # evaluate_action return is summed), so all 5 slots are 0.
            # Wiring legacy per-component entropies would require an
            # extra return tuple from evaluate_action; deferring until
            # there's a concrete user for it.
            jnp.zeros((5,), dtype=jnp.float32),
        )

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

        if batch.tokens.ndim == 3:
            E, K = batch.tokens.shape[:2]
            enc_key, eval_key = jrand.split(key, 2)
            cached_per_traj = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens[:, 0], batch.eqn_ids[:, 0], jrand.split(enc_key, E))
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(E * K, *x.shape[2:]),
                batch,
            )
            cached_flat = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, K, axis=0),
                cached_per_traj,
            )
            keys = jrand.split(eval_key, E * K)
        elif args.cache_encoding:
            B = batch.tokens.shape[0]
            enc_key, eval_key = jrand.split(key, 2)
            cached_flat = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens, batch.eqn_ids, jrand.split(enc_key, B))
            keys = jrand.split(eval_key, B)
        else:
            cached_flat = None
            keys = jrand.split(key, batch.tokens.shape[0])

        actions = MicroAction(
            op_type=batch.micro_op_seq,
            i=batch.micro_i_seq,
            j=batch.micro_j_seq,
            exponents=batch.micro_exp_seq,
            factor=batch.micro_factor_seq,
            compress_kind=batch.micro_compress_kind_seq,
        )

        def _eval_dyn(toks, eids, rs, pref, vidx, action, vmask, cached, k):
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
            ) = jax.vmap(
                lambda toks, eids, rs, pref, vidx, action, vmask, k: _eval_dyn(
                    toks, eids, rs, pref, vidx, action, vmask, None, k
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
            )

        old_log_probs = jax.vmap(old_micro_log_prob_for_action)(
            batch.vertex_idx,
            batch.micro_op_seq,
            batch.micro_i_seq,
            batch.micro_j_seq,
            batch.micro_exp_seq,
            batch.micro_compress_kind_seq,
            batch.old_vertex_dist,
            batch.old_micro_op_dists,
            batch.old_micro_i_dists,
            batch.old_micro_j_dists,
            batch.old_micro_exp_dists,
            batch.old_micro_kind_dists,
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
                batch.norm_adv, batch.estim_returns[..., 0]
            )
        else:
            value_loss = jnp.mean(
                jnp.sum(
                    (values - reward_normalization_fn(batch.estim_returns)) ** 2,
                    axis=-1,
                )
            )
            explained_var = explained_variance(
                batch.norm_adv,
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
        kl_div = kl_vertex + kl_op + kl_i + kl_j + kl_exp + kl_kind
        # Stash per-component KLs so they can be logged separately — they're
        # the most useful single signal for debugging the dynamic head
        # (factor head and END decision are where collapse starts per the
        # design spec). The 5-slot layout is preserved for back-compat; the
        # kind KL is folded into the exponent slot since both gate on the
        # corresponding op-type (DIAG / COMPRESS respectively).
        _kl_components = (kl_vertex, kl_op, kl_i, kl_j, kl_exp + kl_kind)

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
        # Fold the kind entropy into the exp slot to keep the 5-slot
        # layout the legacy loss path returns.
        _entropy_components = (ent_vertex, ent_op, ent_i, ent_j, ent_exp + ent_kind)

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
        multipliers,
        constraint_indices,
        constraint_thresholds,
        constraint_signs,
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
        if args.loss_mode == "scalar":
            scalar_reward = jnp.sum(
                traj.reward * reward_weights, axis=-1
            )  # (E, T)
            zeros = jnp.zeros_like(scalar_reward)
            head_rewards = jnp.stack([scalar_reward, zeros, zeros], axis=-1)
        else:
            head_rewards = traj.reward[..., _HEAD_REWARD_INDICES_ARR]
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
            # CLI --lambda-* weights; in Stage F it's the Dirichlet sample; in
            # Stage G calibration it's the quality-focused override. Same code
            # path either way.
            norm_adv = jnp.sum(norm_adv_components * traj.preference, axis=-1)

        # Stage F Lagrangian: per-step constraint violations and multiplier
        # update. ``constraint_indices`` / ``constraint_thresholds`` /
        # ``constraint_signs`` are static-shape arrays (length 0 means no
        # constraints, in which case the gather is a no-op). For ``>=``
        # constraints (sign = +1) the violation is ``max(0, threshold -
        # reward)``; for ``<=`` constraints (sign = -1) it's
        # ``max(0, reward - threshold)``. Both collapse to
        # ``max(0, sign * (threshold - reward))``. The advantage is
        # penalized by ``-λ_i · violation_i`` so the policy is pushed away
        # from constraint-violating regions, and ``λ_i`` rises by dual
        # ascent on the mean violation.
        if constraint_indices.shape[0] > 0:
            picked = traj.reward[..., constraint_indices]  # (E, T, C)
            signed = constraint_signs * (constraint_thresholds - picked)
            violations = jnp.maximum(0.0, signed)  # (E, T, C)
            penalty = jnp.sum(violations * multipliers, axis=-1)  # (E, T)
            norm_adv = norm_adv - penalty
            mean_violations = jnp.mean(violations, axis=(0, 1))  # (C,)
            new_multipliers = jnp.maximum(
                0.0,
                multipliers + args.lagrangian_lr * mean_violations,
            )
        else:
            mean_violations = multipliers  # zero-length sentinel
            new_multipliers = multipliers

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
            old_vertex_dist=traj.vertex_dist,
            old_pair_dists=traj.pair_dists,
            old_factor_dists=traj.factor_dists,
            old_micro_op_dists=traj.micro_op_dists,
            old_micro_i_dists=traj.micro_i_dists,
            old_micro_j_dists=traj.micro_j_dists,
            old_micro_exp_dists=traj.micro_exp_dists,
            old_micro_kind_dists=traj.micro_kind_dists,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
            vertex_avail_mask=traj.vertex_avail_mask,
        )

        dynamic_carry, static_carry = eqx.partition((agent, opt_state), eqx.is_array)

        # Stage D head curriculum: thread a step counter through both scans
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
        use_traj_batch = args.cache_encoding and num_envs >= args.minibatches

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
            mean_violations,
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
            new_multipliers,
            diag_pack,
        )

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    # Reporting.
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
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
        ep, all_rets, actions_pack, mean_r, mets, diag_pack=None, multipliers_arr=None
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

        def _decode(env_i):
            if args.dynamic_substeps:
                return _action_to_pylist_dynamic(
                    v_idx_arr[env_i],
                    micro_op_arr[env_i],
                    micro_i_arr[env_i],
                    micro_j_arr[env_i],
                    micro_factor_arr[env_i],
                    micro_kind_arr[env_i],
                    args.max_substeps,
                )
            return _action_to_pylist(
                v_idx_arr[env_i],
                pair_arr[env_i],
                factor_arr[env_i],
                max_rules,
                factor_table_np,
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
        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
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

        weighted_sums = np.sum(all_rets * weights, axis=-1)
        best_idx = int(np.argmax(weighted_sums))
        best_ret = float(weighted_sums[best_idx])
        if best_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = best_ret
            host_state["best_global_act_seq"] = _decode(best_idx)

        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": float(np.sum(mean_r * weights)),
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
                mean_viol,
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
            if multipliers_arr is not None and np.size(multipliers_arr) > 0:
                lam = np.asarray(multipliers_arr)
                viol = np.asarray(mean_viol)
                for j, (idx, t, sign) in enumerate(constraint_specs):
                    op_str = ">=" if sign > 0 else "<="
                    name = f"{REWARD_NAMES[idx]}{op_str}{t:g}"
                    log_dict[f"lagrangian/{name}_lambda"] = float(lam[j])
                    log_dict[f"lagrangian/{name}_violation"] = float(viol[j])
        wandb.log(log_dict)

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

    # Stage F Lagrangian state. Constraint indices / thresholds / signs are
    # static-shape arrays (length-C jax arrays). Multipliers are dynamic
    # (length-C, ≥ 0, updated by dual ascent each episode). C == 0 → empty
    # arrays, in which case the augmentation in train_episode is a no-op.
    # The cosine-similarity bounds are added automatically; pass
    # ``--cosine-lower-bound <= 0`` / ``--cosine-upper-bound >= 1`` to disable.
    user_constraints = list(args.lagrangian_constraint)
    if args.cosine_lower_bound > 0.0:
        user_constraints.append(f"cosine_sim>={args.cosine_lower_bound}")
    if args.cosine_upper_bound < 1.0:
        user_constraints.append(f"cosine_sim<={args.cosine_upper_bound}")
    constraint_specs = parse_lagrangian_constraints(user_constraints)
    if constraint_specs:
        ops_for_print = {1: ">=", -1: "<="}
        print(
            f"Stage F Lagrangian: {len(constraint_specs)} constraints — "
            + ", ".join(
                f"{REWARD_NAMES[idx]}{ops_for_print[sign]}{t:g}"
                for idx, t, sign in constraint_specs
            )
        )
    constraint_indices = jnp.asarray(
        [idx for idx, _, _ in constraint_specs],
        dtype=jnp.int32,
    )
    constraint_thresholds = jnp.asarray(
        [t for _, t, _ in constraint_specs],
        dtype=jnp.float32,
    )
    constraint_signs = jnp.asarray(
        [float(sign) for _, _, sign in constraint_specs],
        dtype=jnp.float32,
    )
    multipliers = jnp.zeros(len(constraint_specs), dtype=jnp.float32)

    # Stage F: per-env preference sampling over the 3-head simplex (flops /
    # peak_memory / frob_residual). Uses a Dirichlet with the configured
    # concentration; values < 1 emphasise corners and edges. When
    # --preference-conditioned is off we broadcast the static
    # `head_reward_weights` so all downstream code sees a consistent
    # `(num_envs, NUM_VALUE_HEADS)` shape.
    static_pref = jnp.broadcast_to(head_reward_weights, (num_envs, NUM_VALUE_HEADS))

    # Per-head introduction-stage for the curriculum runner. Static
    # (depends on the curriculum spec, not on the current episode) so
    # compute once before the loop.
    if curriculum_stages:
        micro_intro_stage = _micro_introduction_stage(
            curriculum_stages,
            args.allow_compress,
        )
        print(
            f"curriculum: micro_action_policy first sees signal in stage "
            f"{micro_intro_stage}/{len(curriculum_stages)}"
            + (" (never)" if micro_intro_stage >= len(curriculum_stages) else "")
        )
    else:
        micro_intro_stage = 0  # unused outside curriculum runs

    for ep in range(args.episodes):
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
        # Curriculum stage resolution: figure out which stage `ep` falls
        # into and derive the per-call overrides from the stage's variant.
        # Empty curriculum (the default) uses the main-level values set at
        # startup.
        if curriculum_stages:
            cumulative = 0
            current_stage_name = curriculum_stages[-1][0]
            current_stage_idx = len(curriculum_stages) - 1
            for stage_idx, (stage_name, stage_n) in enumerate(curriculum_stages):
                if ep < cumulative + stage_n:
                    current_stage_name = stage_name
                    current_stage_idx = stage_idx
                    break
                cumulative += stage_n
            stage_override = _op_legality_for_variant(
                current_stage_name,
                args.allow_compress,
            )
            stage_pin_rules = jnp.asarray(
                _pin_rules_for_variant(current_stage_name),
                dtype=jnp.bool_,
            )
            # Per-head LR multipliers: each head runs at full LR in the
            # first stage where it sees signal, then drops to
            # --curriculum-existing-head-mult in subsequent stages.
            # vertex_policy is introduced at stage 0 (always); the
            # dynamic head's introduction is the first stage where DIAG
            # or COMPRESS is legal (see _micro_introduction_stage).
            stage_vertex_mult = jnp.array(
                1.0 if current_stage_idx == 0 else args.curriculum_existing_head_mult,
                dtype=jnp.float32,
            )
            stage_micro_mult = jnp.array(
                1.0
                if current_stage_idx <= micro_intro_stage
                else args.curriculum_existing_head_mult,
                dtype=jnp.float32,
            )
            # Log the stage boundary on transition (cheap host-side check).
            if ep == 0 or (
                ep > 0
                and _current_stage_at(curriculum_stages, ep - 1) != current_stage_name
            ):
                print(
                    f"[ep {ep}] curriculum stage → {current_stage_name}"
                    f"  (vertex_mult={float(stage_vertex_mult):.2f}, "
                    f"micro_mult={float(stage_micro_mult):.2f}, "
                    f"pin_rules={bool(stage_pin_rules)})"
                )
        else:
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
            multipliers,
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
            multipliers,
            constraint_indices,
            constraint_thresholds,
            constraint_signs,
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
            multipliers,
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

    # ------------------------------------------------------------------
    # Stage G: few-shot calibration
    # ------------------------------------------------------------------
    if args.calibrate_steps > 0:
        agent, global_step, key = run_calibration_phase(
            agent,
            opt_state,
            env,
            env_episode_template_args=(
                closed_jaxpr,
                tuple(xs),
                tuple(argnums),
            ),
            train_episode=train_episode,
            reset_envs=reset_envs,
            num_envs=num_envs,
            args=args,
            global_step=global_step,
            key=key,
        )

    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n(f"CMP (Lowest {args.cmp_type})", host_state["top_n_cmp"])
    print_top_n(f"Memory (Lowest {args.mem_type})", host_state["top_n_mem"])
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])
    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
