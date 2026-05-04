"""PPO trainer for the vertex-elimination env.

Three policy architectures are available and selectable via CLI flags:

* default — full transformer pointer net + autoregressive `RuleDecoder` that
  emits a sequence of `(axis_pair, factor)` rules per chosen vertex.
* `--not-autoreg` — transformer pointer net, but with a single per-vertex
  `sp_head` (one rule per vertex, factor fixed to -1).
* `--no-ptr` — original architecture with no pointer net: a single MLP policy
  head over the unrolled `(sp_type, vertex)` action space.

All three share the same trajectory layout, GAE machinery, and PPO loss; the
only differences live inside the agent classes (sampling and log-prob). All
hyperparameters (network widths, optimisation, PPO knobs, mask layout) are
behind CLI arguments.
"""

from __future__ import annotations

import argparse
import heapq
import os
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
import wandb
from tqdm import tqdm

from alphagrad.approx.common import (
    build_pair_valid_mask,
    build_vertex_valid_static,
    data_gen,
    generate_eval_samples,
    get_advantages,
    get_args,
    get_fn,
    get_num_clipping_triggers,
    infer_argnums,
    init_linear_weights,
    reward_normalization_fn,
    scale_module_weight,
    shuffle_and_batch,
    vertex_avail_at_step,
)
from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_AXIS_PAIRS,
    StepAction,
    VertexEliminationEnv,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy, explained_variance


# ---------------------------------------------------------------------------
# Constants shared by all three agent variants
# ---------------------------------------------------------------------------

# NUM_AXIS_PAIRS axis pairs + 1 STOP token marking end of a rule sequence.
NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS

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
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    next_value: jax.Array
    vertex_dist: jax.Array
    pair_dists: jax.Array
    factor_dists: jax.Array
    discount: jax.Array
    vertex_avail_mask: jax.Array


class TrainBatch(NamedTuple):
    tokens: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    old_vertex_dist: jax.Array
    old_pair_dists: jax.Array
    old_factor_dists: jax.Array
    estim_returns: jax.Array
    norm_adv: jax.Array
    vertex_avail_mask: jax.Array


# ---------------------------------------------------------------------------
# Helpers used by every variant
# ---------------------------------------------------------------------------


def _stop_only_logits(num_pair_choices: int) -> jax.Array:
    """Logits that put all mass on the STOP pair index."""
    return jnp.where(
        jnp.arange(num_pair_choices) == PAIR_STOP, 0.0, -1e9
    ).astype(jnp.float32)


def build_rule_specs(pair_seq, factor_seq, factor_table) -> jax.Array:
    """Convert per-slot `(pair_idx, factor_idx)` -> `(MAX_RULES, 3)` rule specs.

    `pair_idx == PAIR_STOP` terminates the sequence; subsequent slots are
    written as unused (`base_idx1 = -1, factor = 0`).
    """
    base = _PAIR_TO_BASE[pair_seq]  # (MAX_RULES, 2)
    factor_vals = factor_table[factor_seq]  # (MAX_RULES,)

    is_stop = pair_seq == PAIR_STOP
    has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0  # (MAX_RULES,) bool

    base_final = jnp.where(has_stopped[:, None], -1, base)
    factor_final = jnp.where(has_stopped, 0, factor_vals)
    return jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(
        jnp.int32
    )


def build_legacy_rule_specs(pair_seq) -> jax.Array:
    """Rule specs for the single-rule (factor=-1) variants. Ignores the factor table."""
    base = _PAIR_TO_BASE[pair_seq]  # (MAX_RULES, 2)
    factor = jnp.full(pair_seq.shape, -1, dtype=jnp.int32)

    is_stop = pair_seq == PAIR_STOP
    has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0

    base_final = jnp.where(has_stopped[:, None], -1, base)
    factor_final = jnp.where(has_stopped, 0, factor)
    return jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(
        jnp.int32
    )


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
    factor_log_ps = (
        jnp.log(factor_dists[arange_r, factor_seq] + 1e-8) * factor_active
    )
    return log_p_v + jnp.sum(pair_log_ps) + jnp.sum(factor_log_ps)


def _pad_seq(value: jax.Array, max_rules: int, pad: int = 0) -> jax.Array:
    """Pad a 1-d array out to `max_rules` with `pad` (used to fill unused slot indices)."""
    pad_len = max_rules - value.shape[0]
    if pad_len <= 0:
        return value[:max_rules]
    pad_arr = jnp.full((pad_len,), pad, dtype=value.dtype)
    return jnp.concatenate([value, pad_arr], axis=0)


def _pad_dists(active: jax.Array, max_rules: int, num_choices: int, fill_idx: int) -> jax.Array:
    """Stack `active` (1, num_choices) with `(max_rules - 1)` degenerate one-hots at `fill_idx`."""
    pad_len = max_rules - active.shape[0]
    if pad_len <= 0:
        return active[:max_rules]
    pad_one_hot = jnp.broadcast_to(
        jnn.one_hot(fill_idx, num_choices), (pad_len, num_choices)
    )
    return jnp.concatenate([active, pad_one_hot], axis=0)


# ---------------------------------------------------------------------------
# Variant A: full autoregressive pointer + RuleDecoder
# ---------------------------------------------------------------------------


class RuleDecoder(eqx.Module):
    """Per-slot autoregressive head emitting `(axis_pair, factor)` rules.

    The pair head is conditioned on the previous slot's `(pair, factor)` and the
    chosen vertex; the factor head is conditioned on the *current* pair so the
    two heads are autoregressive within a single rule slot.
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


class AutoregPointerAgent(eqx.Module):
    """Default variant: pointer net + autoregressive rule decoder."""

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    vertex_embedding: eqx.nn.Embedding
    cross_attn: eqx.nn.MultiheadAttention
    pointer_proj: eqx.nn.Linear
    rule_decoder: RuleDecoder
    value_head: MLP

    num_vertices: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_vertices,
        num_rewards,
        max_rules,
        num_pair_choices,
        num_factors,
        value_dims,
        seq_len,
        key,
    ):
        keys = jrand.split(key, 7)
        self.num_vertices = num_vertices
        self.num_rewards = num_rewards
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.embd_dim = embd_dim
        self.use_factor_table = True
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=keys[1])
        self.vertex_embedding = eqx.nn.Embedding(num_vertices, embd_dim, key=keys[2])
        self.cross_attn = eqx.nn.MultiheadAttention(num_heads, embd_dim, key=keys[3])
        self.pointer_proj = eqx.nn.Linear(embd_dim, 1, key=keys[4])
        self.rule_decoder = RuleDecoder(
            embd_dim, max_rules, num_pair_choices, num_factors, key=keys[5]
        )
        self.value_head = MLP(embd_dim, num_rewards, value_dims, key=keys[6])

    def encode(self, tokens, key=None):
        token_mask = tokens != 0
        mask = token_mask[..., None]
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        enc_x = self.encoder(x, key=enc_key)

        vertex_ids = jnp.arange(self.num_vertices)
        v_q = jax.vmap(self.vertex_embedding)(vertex_ids)
        attn_mask = jnp.broadcast_to(
            token_mask[None, :], (self.num_vertices, tokens.shape[0])
        )
        vertex_reprs = self.cross_attn(v_q, enc_x, enc_x, mask=attn_mask)

        vertex_logits = jax.vmap(self.pointer_proj)(vertex_reprs).squeeze(-1)
        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        return vertex_logits, vertex_reprs, self.value_head(summary)

    def sample_action(self, tokens, vertex_avail_mask, pair_valid_mask, key):
        net_key, vertex_key, decoder_key = jrand.split(key, 3)
        vertex_logits, vertex_reprs, value = self.encode(tokens, key=net_key)

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        vertex_idx = distrax.Categorical(probs=vertex_dist).sample(seed=vertex_key)
        v_repr = vertex_reprs[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]
        stop_only = _stop_only_logits(self.num_pair_choices)

        slot_keys = jrand.split(decoder_key, self.max_rules)

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, k = slot_data
            kp, kf = jrand.split(k)
            new_h, pair_logits = self.rule_decoder.step(
                v_repr, slot_idx, prev_pair, prev_factor, prev_h
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            pair_idx = distrax.Categorical(probs=pair_dist).sample(seed=kp)

            factor_logits = self.rule_decoder.factor_logits_for(new_h, pair_idx)
            factor_dist = jnn.softmax(factor_logits, axis=-1)
            factor_idx = distrax.Categorical(probs=factor_dist).sample(seed=kf)

            new_active = active & (pair_idx != PAIR_STOP)
            return (pair_idx, factor_idx, new_h, new_active), (
                pair_idx,
                factor_idx,
                pair_dist,
                factor_dist,
            )

        init_carry = (
            jnp.array(PAIR_STOP, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.zeros(self.embd_dim),
            jnp.array(True, dtype=jnp.bool_),
        )
        _, (pair_seq, factor_seq, pair_dists, factor_dists) = lax.scan(
            rule_step, init_carry, (jnp.arange(self.max_rules), slot_keys)
        )
        return (
            vertex_idx,
            pair_seq,
            factor_seq,
            vertex_dist,
            pair_dists,
            factor_dists,
            value,
        )

    def evaluate_action(
        self,
        tokens,
        vertex_idx,
        pair_seq,
        factor_seq,
        vertex_avail_mask,
        pair_valid_mask,
        key,
    ):
        vertex_logits, vertex_reprs, value = self.encode(tokens, key=key)

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        log_p_vertex = jnp.log(vertex_dist[vertex_idx] + 1e-8)
        vertex_entropy = entropy(vertex_dist)

        v_repr = vertex_reprs[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]
        stop_only = _stop_only_logits(self.num_pair_choices)

        def rule_step(carry, slot_data):
            prev_pair, prev_factor, prev_h, active = carry
            slot_idx, true_pair, true_factor = slot_data
            new_h, pair_logits = self.rule_decoder.step(
                v_repr, slot_idx, prev_pair, prev_factor, prev_h
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            pair_logits_eff = jnp.where(active, pair_logits, stop_only)
            pair_dist = jnn.softmax(pair_logits_eff, axis=-1)
            log_p_pair = jnp.log(pair_dist[true_pair] + 1e-8)

            factor_logits = self.rule_decoder.factor_logits_for(new_h, true_pair)
            factor_dist = jnn.softmax(factor_logits, axis=-1)
            log_p_factor = jnp.log(factor_dist[true_factor] + 1e-8)

            active_f32 = active.astype(jnp.float32)
            log_p_pair_eff = log_p_pair * active_f32
            pair_ent = entropy(pair_dist) * active_f32

            factor_active = active & (true_pair != PAIR_STOP)
            factor_active_f32 = factor_active.astype(jnp.float32)
            log_p_factor_eff = log_p_factor * factor_active_f32
            factor_ent = entropy(factor_dist) * factor_active_f32

            new_active = active & (true_pair != PAIR_STOP)
            return (true_pair, true_factor, new_h, new_active), (
                log_p_pair_eff,
                log_p_factor_eff,
                pair_ent,
                factor_ent,
                pair_dist,
                factor_dist,
            )

        init_carry = (
            jnp.array(PAIR_STOP, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.zeros(self.embd_dim),
            jnp.array(True, dtype=jnp.bool_),
        )
        _, (lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists) = (
            lax.scan(
                rule_step,
                init_carry,
                (jnp.arange(self.max_rules), pair_seq, factor_seq),
            )
        )

        total_log_p = log_p_vertex + jnp.sum(lp_pairs) + jnp.sum(lp_factors)
        total_entropy = vertex_entropy + jnp.sum(ent_pairs) + jnp.sum(ent_factors)
        return (
            total_log_p,
            total_entropy,
            value,
            vertex_dist,
            pair_dists,
            factor_dists,
        )

    def to_env_action(self, vertex_idx, pair_seq, factor_seq, factor_table):
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=build_rule_specs(pair_seq, factor_seq, factor_table),
        )

    def value_for(self, tokens, key=None):
        _, _, value = self.encode(tokens, key=key)
        return value


# ---------------------------------------------------------------------------
# Variant B: pointer net, single per-vertex sp head (no autoregression)
# ---------------------------------------------------------------------------


class PointerAgent(eqx.Module):
    """Pointer + per-vertex sp head. One rule per vertex, factor=-1 fixed."""

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    vertex_embedding: eqx.nn.Embedding
    cross_attn: eqx.nn.MultiheadAttention
    pointer_proj: eqx.nn.Linear
    sp_head: MLP
    value_head: MLP

    num_vertices: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_vertices,
        num_rewards,
        max_rules,
        num_pair_choices,
        num_factors,
        sp_dims,
        value_dims,
        seq_len,
        key,
    ):
        keys = jrand.split(key, 7)
        self.num_vertices = num_vertices
        self.num_rewards = num_rewards
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.embd_dim = embd_dim
        self.use_factor_table = False
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=keys[1])
        self.vertex_embedding = eqx.nn.Embedding(num_vertices, embd_dim, key=keys[2])
        self.cross_attn = eqx.nn.MultiheadAttention(num_heads, embd_dim, key=keys[3])
        self.pointer_proj = eqx.nn.Linear(embd_dim, 1, key=keys[4])
        # `sp_head` outputs `num_pair_choices` logits — one for each (pair, STOP) choice.
        self.sp_head = MLP(embd_dim, num_pair_choices, sp_dims, key=keys[5])
        self.value_head = MLP(embd_dim, num_rewards, value_dims, key=keys[6])

    def encode(self, tokens, key=None):
        token_mask = tokens != 0
        mask = token_mask[..., None]
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        enc_x = self.encoder(x, key=enc_key)

        vertex_ids = jnp.arange(self.num_vertices)
        v_q = jax.vmap(self.vertex_embedding)(vertex_ids)
        attn_mask = jnp.broadcast_to(
            token_mask[None, :], (self.num_vertices, tokens.shape[0])
        )
        vertex_reprs = self.cross_attn(v_q, enc_x, enc_x, mask=attn_mask)

        vertex_logits = jax.vmap(self.pointer_proj)(vertex_reprs).squeeze(-1)
        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        return vertex_logits, vertex_reprs, self.value_head(summary)

    def _slot0_dist(self, v_repr, v_pair_mask):
        """Compute the per-vertex pair distribution at slot 0 (with vertex-validity masking)."""
        sp_logits = self.sp_head(v_repr)
        sp_logits = jnp.where(v_pair_mask > 0.5, sp_logits, -1e9)
        return jnn.softmax(sp_logits, axis=-1)

    def _build_uniform_action(self, vertex_idx, vertex_dist, pair_idx, slot0_dist):
        """Pad (vertex, single pair) into the uniform autoreg-shaped action structure."""
        pair_seq = _pad_seq(jnp.atleast_1d(pair_idx).astype(jnp.int32), self.max_rules, pad=PAIR_STOP)
        factor_seq = jnp.zeros((self.max_rules,), dtype=jnp.int32)
        pair_dists = _pad_dists(slot0_dist[None, :], self.max_rules, self.num_pair_choices, PAIR_STOP)
        factor_dists = jnp.broadcast_to(
            jnn.one_hot(0, self.num_factors), (self.max_rules, self.num_factors)
        )
        return pair_seq, factor_seq, vertex_dist, pair_dists, factor_dists

    def sample_action(self, tokens, vertex_avail_mask, pair_valid_mask, key):
        net_key, vertex_key, sp_key = jrand.split(key, 3)
        vertex_logits, vertex_reprs, value = self.encode(tokens, key=net_key)

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        vertex_idx = distrax.Categorical(probs=vertex_dist).sample(seed=vertex_key)
        v_repr = vertex_reprs[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]

        slot0_dist = self._slot0_dist(v_repr, v_pair_mask)
        pair_idx = distrax.Categorical(probs=slot0_dist).sample(seed=sp_key)
        pair_seq, factor_seq, vd, pair_dists, factor_dists = self._build_uniform_action(
            vertex_idx, vertex_dist, pair_idx, slot0_dist
        )
        return vertex_idx, pair_seq, factor_seq, vd, pair_dists, factor_dists, value

    def evaluate_action(
        self,
        tokens,
        vertex_idx,
        pair_seq,
        factor_seq,
        vertex_avail_mask,
        pair_valid_mask,
        key,
    ):
        vertex_logits, vertex_reprs, value = self.encode(tokens, key=key)

        masked_v_logits = jnp.where(vertex_avail_mask > 0.5, vertex_logits, -1e9)
        vertex_dist = jnn.softmax(masked_v_logits, axis=-1)
        log_p_vertex = jnp.log(vertex_dist[vertex_idx] + 1e-8)
        vertex_ent = entropy(vertex_dist)

        v_repr = vertex_reprs[vertex_idx]
        v_pair_mask = pair_valid_mask[vertex_idx]

        slot0_dist = self._slot0_dist(v_repr, v_pair_mask)
        pair_idx = pair_seq[0]
        log_p_pair = jnp.log(slot0_dist[pair_idx] + 1e-8)
        pair_ent = entropy(slot0_dist)

        # Slots 1+ are degenerate STOP and the factor head is degenerate; their
        # log-prob and entropy contributions are zero.
        pair_dists = _pad_dists(
            slot0_dist[None, :], self.max_rules, self.num_pair_choices, PAIR_STOP
        )
        factor_dists = jnp.broadcast_to(
            jnn.one_hot(0, self.num_factors), (self.max_rules, self.num_factors)
        )

        total_log_p = log_p_vertex + log_p_pair
        total_entropy = vertex_ent + pair_ent
        return total_log_p, total_entropy, value, vertex_dist, pair_dists, factor_dists

    def to_env_action(self, vertex_idx, pair_seq, factor_seq, factor_table):
        # Single-rule, factor=-1: ignore factor_seq / factor_table.
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=build_legacy_rule_specs(pair_seq),
        )

    def value_for(self, tokens, key=None):
        _, _, value = self.encode(tokens, key=key)
        return value


# ---------------------------------------------------------------------------
# Variant C: original MLP head (no pointer net), unrolled (sp_type, vertex)
# ---------------------------------------------------------------------------


class MLPAgent(eqx.Module):
    """Encoder + MLP policy head over the unrolled `(sp_type, vertex)` action space."""

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP

    num_vertices: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)
    max_rules: int = eqx.field(static=True)
    num_pair_choices: int = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    use_factor_table: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_vertices,
        num_rewards,
        max_rules,
        num_pair_choices,
        num_factors,
        policy_dims,
        value_dims,
        seq_len,
        key,
    ):
        keys = jrand.split(key, 4)
        self.num_vertices = num_vertices
        self.num_rewards = num_rewards
        self.max_rules = max_rules
        self.num_pair_choices = num_pair_choices
        self.num_factors = num_factors
        self.embd_dim = embd_dim
        self.use_factor_table = False
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=keys[1])
        # Output dim == num_pair_choices * num_vertices: rows are sp-type / pair-index, cols are vertices.
        self.policy_head = MLP(
            embd_dim, num_pair_choices * num_vertices, policy_dims, key=keys[2]
        )
        self.value_head = MLP(embd_dim, num_rewards, value_dims, key=keys[3])

    def encode(self, tokens, key=None):
        token_mask = tokens != 0
        mask = token_mask[..., None]
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        enc_x = self.encoder(x, key=enc_key)
        summary = jnp.sum(enc_x * mask, axis=0) / jnp.maximum(
            jnp.sum(mask, axis=0), 1e-9
        )
        return self.policy_head(summary), self.value_head(summary)

    def _joint_dist_and_marginal(
        self, summary_logits, vertex_avail_mask, pair_valid_mask
    ):
        """Compute the full joint dist (pair × vertex) and the vertex marginal under masking."""
        joint_logits = summary_logits.reshape(self.num_pair_choices, self.num_vertices)
        # Mask: vertex must be available AND the (pair, vertex) combination must be valid.
        # `pair_valid_mask` is shape (num_vertices, num_pair_choices) -> transpose for joint layout.
        joint_mask = (
            vertex_avail_mask[None, :] * pair_valid_mask.T
        )  # (num_pair_choices, num_vertices)
        joint_logits = jnp.where(joint_mask > 0.5, joint_logits, -1e9)
        joint_flat = joint_logits.reshape(-1)
        joint_dist_flat = jnn.softmax(joint_flat, axis=-1)
        joint_dist = joint_dist_flat.reshape(
            self.num_pair_choices, self.num_vertices
        )
        vertex_dist = jnp.sum(joint_dist, axis=0)  # (num_vertices,)
        return joint_dist_flat, joint_dist, vertex_dist

    def _build_uniform_action(self, vertex_idx, pair_idx, vertex_dist, joint_dist):
        """Pad the chosen (vertex, pair) into the uniform autoreg-shaped action."""
        pair_seq = _pad_seq(jnp.atleast_1d(pair_idx).astype(jnp.int32), self.max_rules, pad=PAIR_STOP)
        factor_seq = jnp.zeros((self.max_rules,), dtype=jnp.int32)

        # Conditional pair distribution at the chosen vertex (used only for KL diagnostics).
        cond_denom = jnp.maximum(vertex_dist[vertex_idx], 1e-8)
        slot0_dist = joint_dist[:, vertex_idx] / cond_denom

        pair_dists = _pad_dists(
            slot0_dist[None, :], self.max_rules, self.num_pair_choices, PAIR_STOP
        )
        factor_dists = jnp.broadcast_to(
            jnn.one_hot(0, self.num_factors), (self.max_rules, self.num_factors)
        )
        return pair_seq, factor_seq, pair_dists, factor_dists, slot0_dist

    def sample_action(self, tokens, vertex_avail_mask, pair_valid_mask, key):
        net_key, act_key = jrand.split(key, 2)
        summary_logits, value = self.encode(tokens, key=net_key)

        joint_flat, joint_dist, vertex_dist = self._joint_dist_and_marginal(
            summary_logits, vertex_avail_mask, pair_valid_mask
        )
        action_idx = distrax.Categorical(probs=joint_flat).sample(seed=act_key)
        # Joint layout is (pair, vertex), so divmod by num_vertices.
        pair_idx = action_idx // self.num_vertices
        vertex_idx = action_idx % self.num_vertices

        pair_seq, factor_seq, pair_dists, factor_dists, _ = self._build_uniform_action(
            vertex_idx, pair_idx, vertex_dist, joint_dist
        )
        return (
            vertex_idx,
            pair_seq,
            factor_seq,
            vertex_dist,
            pair_dists,
            factor_dists,
            value,
        )

    def evaluate_action(
        self,
        tokens,
        vertex_idx,
        pair_seq,
        factor_seq,
        vertex_avail_mask,
        pair_valid_mask,
        key,
    ):
        summary_logits, value = self.encode(tokens, key=key)
        joint_flat, joint_dist, vertex_dist = self._joint_dist_and_marginal(
            summary_logits, vertex_avail_mask, pair_valid_mask
        )

        pair_idx = pair_seq[0]
        action_idx = pair_idx * self.num_vertices + vertex_idx
        log_p_action = jnp.log(joint_flat[action_idx] + 1e-8)
        joint_entropy = entropy(joint_flat)

        pair_seq, factor_seq, pair_dists, factor_dists, _ = self._build_uniform_action(
            vertex_idx, pair_idx, vertex_dist, joint_dist
        )
        return (
            log_p_action,
            joint_entropy,
            value,
            vertex_dist,
            pair_dists,
            factor_dists,
        )

    def to_env_action(self, vertex_idx, pair_seq, factor_seq, factor_table):
        return StepAction(
            target_vertex=jnp.asarray(vertex_idx + 1, dtype=jnp.int32),
            rule_specs=build_legacy_rule_specs(pair_seq),
        )

    def value_for(self, tokens, key=None):
        _, value = self.encode(tokens, key=key)
        return value


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
    p.add_argument("--wandb", type=str, default="offline",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument("--exec-on-gpu", action="store_true",
                   help="Pin training to GPU 0 and the env eval callback to GPU 1.")

    # Environment / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument("--disable-sparsification", action="store_true")
    p.add_argument("--cmp-type", type=str, default="flops",
                   choices=["graphax", "flops", "latency"])
    p.add_argument("--mem-type", type=str, default="peak_memory",
                   choices=["graphax", "bytes_accessed", "peak_memory"])
    p.add_argument("--rewards", nargs="+", type=str,
                   default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"])
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # Agent variant
    p.add_argument(
        "--no-ptr",
        action="store_true",
        help=(
            "Drop the pointer net and use the original MLP policy head over the "
            "unrolled (sp_type, vertex) action space. Implies non-autoregressive "
            "(combining with --not-autoreg gives the same MLP variant)."
        ),
    )
    p.add_argument(
        "--not-autoreg",
        action="store_true",
        help=(
            "With the pointer net, use a single per-vertex sp head (one rule, "
            "factor=-1) instead of the autoregressive RuleDecoder. Ignored when "
            "--no-ptr is also set."
        ),
    )

    # Multi-rule / autoregressive head config (ignored when --no-ptr or --not-autoreg)
    p.add_argument("--max-rules", type=int, default=MAX_RULES_PER_VERTEX,
                   help="Max number of (axis_pair, factor) rules per chosen vertex (autoregressive head only).")
    p.add_argument("--factors", type=str, default="-1,1,2,4",
                   help="Comma-separated factor choices for the per-rule factor head. -1 = gcd-based collapse (legacy).")

    # Network architecture
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32",
                   help="MLP policy/sp head hidden widths (comma-separated).")
    p.add_argument("--value-dims", type=str, default="64,32",
                   help="MLP value head hidden widths (comma-separated).")

    # Optimisation
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Parallel rollout envs. -1 = os.cpu_count() (or 16 for Vmapped examples).")
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
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1,
                   help="Cosine decay floor as a multiple of the initial learning rate.")
    p.add_argument("--head-init-scale", type=float, default=0.1,
                   help="Multiplier applied to output-head weights at startup for a near-uniform initial policy.")

    # Reporting
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--capture-perfect-grads", action="store_true",
                   help="Allow the top-N accuracy heap to keep trajectories with cosine similarity == 1.0.")

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


def _build_factor_table(args, variant: str):
    factors_py = tuple(_parse_int_list(args.factors))
    if not factors_py:
        raise ValueError("--factors must contain at least one factor value")

    if variant == "autoreg":
        if args.max_rules > MAX_RULES_PER_VERTEX:
            raise ValueError(
                f"--max-rules ({args.max_rules}) exceeds env-side "
                f"MAX_RULES_PER_VERTEX ({MAX_RULES_PER_VERTEX})"
            )
        max_rules = args.max_rules
    else:
        # Simpler agents emit a single rule per vertex with factor=-1 (factor_table unused).
        factors_py = (-1,)
        max_rules = 1

    factor_table = jnp.array(factors_py, dtype=jnp.int32)
    return factor_table, factors_py, factor_table.shape[0], max_rules


def _select_variant(args) -> str:
    if args.no_ptr:
        return "mlp"
    if args.not_autoreg:
        return "pointer"
    return "autoreg"


def _build_agent(variant: str, args, total_v: int, num_rewards: int, num_factors: int, max_rules: int, key):
    common_kwargs = dict(
        vocab_size=args.vocab_size,
        embd_dim=args.embd_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_vertices=total_v,
        num_rewards=num_rewards,
        max_rules=max_rules,
        num_pair_choices=NUM_PAIR_CHOICES,
        num_factors=num_factors,
        value_dims=_parse_int_list(args.value_dims),
        seq_len=MAX_TOKENS,
        key=key,
    )
    if variant == "mlp":
        return MLPAgent(policy_dims=_parse_int_list(args.policy_dims), **common_kwargs)
    if variant == "pointer":
        return PointerAgent(sp_dims=_parse_int_list(args.policy_dims), **common_kwargs)
    return AutoregPointerAgent(**common_kwargs)


def _scale_output_heads(agent, scale: float, variant: str):
    """Scale policy-head weights so the initial action distribution is near-uniform."""
    if variant == "mlp":
        return scale_module_weight(
            agent, lambda a: a.policy_head.layers[-2].weight, scale
        )

    agent = scale_module_weight(agent, lambda a: a.pointer_proj.weight, scale)
    if variant == "pointer":
        return scale_module_weight(
            agent, lambda a: a.sp_head.layers[-2].weight, scale
        )
    # autoreg: pointer + pair head + factor head
    agent = scale_module_weight(
        agent, lambda a: a.rule_decoder.pair_head.weight, scale
    )
    agent = scale_module_weight(
        agent, lambda a: a.rule_decoder.factor_head.weight, scale
    )
    return agent


def _action_to_pylist(vertex_seq, pair_seq, factor_seq, max_rules, factor_table_np):
    """Decode (vertex, pair, factor) sequences into a list of `(vertex, [(idx1, idx2, factor), ...])`."""
    out = []
    for v_idx, p_row, f_row in zip(vertex_seq, pair_seq, factor_seq):
        rules: list[tuple[int, int, int]] = []
        for slot in range(max_rules):
            p = int(p_row[slot])
            if p == PAIR_STOP:
                break
            base = _PAIR_TO_BASE[p]
            base_idx1, base_idx2 = int(base[0]), int(base[1])
            factor = int(factor_table_np[int(f_row[slot])])
            rules.append((base_idx1, base_idx2, factor))
        out.append((int(v_idx) + 1, rules))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = make_argparser().parse_args()
    variant = _select_variant(args)

    main_device = _resolve_main_device(args)
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    # Resolve example, build env, derive masks.
    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call, dataset_size=args.dataset_size)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    env_target_fun = target_fn if "acc" in args.rewards else None
    argnums = infer_argnums(args.example)

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
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args, variant)
    factor_table_np = np.array(factors_py, dtype=np.int32)
    num_envs = _resolve_num_envs(args.num_envs, args.example)
    num_rewards = len(args.rewards)

    print(
        f"variant={variant}, num_envs={num_envs}, max_rules={max_rules}, "
        f"factors={factors_py}, rollout_length={num_valid}, minibatches={args.minibatches}"
    )

    agent_key, init_key, key = jrand.split(key, 3)
    agent = _build_agent(variant, args, total_v, num_rewards, num_factors, max_rules, agent_key)
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale, variant)
    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent,
        )

    # Optimiser.
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
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def rollout_fn(agent, env_obj, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, k):
            sample_key, next_net_key = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid
            )

            (
                vertex_idx,
                pair_seq,
                factor_seq,
                vertex_dist,
                pair_dists,
                factor_dists,
                value,
            ) = agent.sample_action(
                state.tokens, vertex_avail_mask, pair_valid_mask, sample_key
            )

            env_action = agent.to_env_action(
                vertex_idx, pair_seq, factor_seq, factor_table
            )
            env_out = env_obj.step(state, env_action)
            next_state = env_out.state
            raw_rewards = env_out.reward
            rewards = jnp.array(
                [raw_rewards[0], raw_rewards[1], raw_rewards[2]]
            )[:num_rewards]
            done = env_out.terminated.astype(jnp.float32)

            next_value = agent.value_for(next_state.tokens, key=next_net_key)

            transition = Trajectory(
                tokens=state.tokens.astype(jnp.int32),
                vertex_idx=jnp.asarray(vertex_idx, dtype=jnp.int32),
                pair_seq=jnp.asarray(pair_seq, dtype=jnp.int32),
                factor_seq=jnp.asarray(factor_seq, dtype=jnp.int32),
                reward=jnp.atleast_1d(rewards),
                done=jnp.array(done, dtype=jnp.float32),
                value=jnp.atleast_1d(value),
                next_value=jnp.atleast_1d(next_value),
                vertex_dist=vertex_dist,
                pair_dists=pair_dists,
                factor_dists=factor_dists,
                discount=jnp.array(args.discount),
                vertex_avail_mask=vertex_avail_mask,
            )
            return next_state, (transition, raw_rewards)

        final_state, (traj, all_raw_rewards) = lax.scan(step_fn, env_state, keys)
        return final_state, traj, all_raw_rewards[-1]

    def loss_fn(agent, batch: TrainBatch, keys):
        eval_batched = jax.vmap(
            lambda toks, vidx, pseq, fseq, vmask, k: agent.evaluate_action(
                toks, vidx, pseq, fseq, vmask, pair_valid_mask, k
            )
        )
        (
            log_probs,
            entropies,
            values,
            vertex_dist,
            pair_dists,
            factor_dists,
        ) = eval_batched(
            batch.tokens,
            batch.vertex_idx,
            batch.pair_seq,
            batch.factor_seq,
            batch.vertex_avail_mask,
            keys,
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
        entropy_loss = jnp.mean(entropies)

        value_loss = jnp.mean(
            jnp.sum(
                (values - reward_normalization_fn(batch.estim_returns)) ** 2, axis=-1
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
        )

    def train_episode(agent, opt_state, env_states, env_obj, key):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, num_envs)

        env_states, traj, total_rewards_full = rollout_fn(
            agent, env_obj, num_valid, env_states, rollout_keys
        )

        _, estim_returns, advantages = get_advantages(
            traj.reward,
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
        adv_weights = jnp.array(
            [args.lambda_cmp, 1.0, args.lambda_mem]
        )[:num_rewards]
        norm_adv = jnp.sum(norm_adv_components * adv_weights, axis=-1)

        full_batch = TrainBatch(
            tokens=traj.tokens,
            vertex_idx=traj.vertex_idx,
            pair_seq=traj.pair_seq,
            factor_seq=traj.factor_seq,
            old_vertex_dist=traj.vertex_dist,
            old_pair_dists=traj.pair_dists,
            old_factor_dists=traj.factor_dists,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
            vertex_avail_mask=traj.vertex_avail_mask,
        )

        dynamic_carry, static_carry = eqx.partition((agent, opt_state), eqx.is_array)

        def train_epoch(carry, epoch_key):
            batches = shuffle_and_batch(full_batch, args.minibatches, epoch_key)
            mb_keys = jrand.split(epoch_key, args.minibatches)

            def train_minibatch(c, batch_and_key):
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                inner_keys = jrand.split(t_key, batch.tokens.shape[0])
                grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
                    comb_agent, batch, inner_keys
                )
                updates, new_opt_state = optimizer.update(
                    grads, comb_opt_state, comb_agent
                )
                new_agent = eqx.apply_updates(comb_agent, updates)
                next_carry, _ = eqx.partition(
                    (new_agent, new_opt_state), eqx.is_array
                )
                return next_carry, metrics

            return lax.scan(train_minibatch, carry, (batches, mb_keys))

        epoch_keys = jrand.split(subkey, args.ppo_epochs)
        dynamic_carry, metrics_seq = lax.scan(train_epoch, dynamic_carry, epoch_keys)

        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)
        actions_pack = (traj.vertex_idx, traj.pair_seq, traj.factor_seq)
        return agent, opt_state, env_states, metrics, total_rewards_full, actions_pack

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

    def host_log(ep, all_rets, actions_pack, mean_r, mets):
        ep = int(ep)
        all_rets = np.array(all_rets)
        v_idx_arr = np.array(actions_pack[0])
        pair_arr = np.array(actions_pack[1])
        factor_arr = np.array(actions_pack[2])
        mean_r = np.atleast_1d(np.array(mean_r))

        host_state["samplecounts"] += num_envs * num_valid
        (
            kl_div,
            policy_entropy,
            _fit_quality,
            explained_var,
            ppo_loss,
            value_loss,
            _entropy_loss,
            total_loss,
            _clipping_trigger_ratio,
        ) = [float(m) for m in mets]

        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:num_rewards]
        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
            decoded = _action_to_pylist(
                v_idx_arr[i], pair_arr[i], factor_arr[i], max_rules, factor_table_np
            )
            total_ret = float(np.sum(rets[:num_rewards] * weights))
            heaps_and_keys = [
                ("top_n_total", total_ret),
                ("top_n_cmp", float(rets[0])),
                ("top_n_mem", float(rets[2])),
            ]
            for heap_name, key_val in heaps_and_keys:
                heap = host_state[heap_name]
                payload = (key_val, ep, list(rets), decoded)
                if len(heap) < args.top_n:
                    heapq.heappush(heap, payload)
                else:
                    heapq.heappushpop(heap, payload)

            acc_val = float(rets[1])
            if args.capture_perfect_grads or acc_val < 0.999999:
                heap = host_state["top_n_acc"]
                payload = (acc_val, ep, list(rets), decoded)
                if len(heap) < args.top_n:
                    heapq.heappush(heap, payload)
                else:
                    heapq.heappushpop(heap, payload)

        best_idx = int(np.argmax(np.sum(all_rets[:, :num_rewards] * weights, axis=-1)))
        best_ret = float(np.sum(all_rets[best_idx, :num_rewards] * weights))
        if best_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = best_ret
            host_state["best_global_act_seq"] = _action_to_pylist(
                v_idx_arr[best_idx], pair_arr[best_idx], factor_arr[best_idx],
                max_rules, factor_table_np,
            )

        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": mean_r[0],
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "explained variance": explained_var,
            "sample count": host_state["samplecounts"],
            "ppo loss": ppo_loss,
            "value loss": value_loss,
            "total loss": total_loss,
        }
        for j in range(1, len(mean_r)):
            log_dict[f"mean_return_{j}"] = mean_r[j]
        wandb.log(log_dict)

        pbar.update(1)
        b_ret_unnorm = np.abs(all_rets[best_idx])
        means_str = ", ".join(f"{float(x):.2f}" for x in np.abs(mean_r))
        b_ret_desc = ", ".join(f"{float(x):.1f}" for x in b_ret_unnorm)
        pbar.set_description(
            f"ent:{policy_entropy:.3f} best:{b_ret_desc} means:{means_str}"
        )

    # Training loop.
    for ep in range(args.episodes):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)

        eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        env_states = reset_envs(env_episode)
        agent, opt_state, _, metrics, total_rewards_full, actions_pack = train_episode(
            agent, opt_state, env_states, env_episode, ep_key
        )
        host_log(
            ep,
            total_rewards_full,
            actions_pack,
            jnp.mean(total_rewards_full[:, :num_rewards], axis=0),
            metrics,
        )

    pbar.close()

    def print_top_n(name, heap, reverse_val=True):
        print(f"\nTop {args.top_n} trajectories for {name}:")
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=reverse_val)
        table = wandb.Table(
            columns=["rank", "episode", "total_reward", "cmp", "acc", "mem", "sequence"]
        )
        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:num_rewards]
        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            total_ret = np.sum(np.array(rets)[:num_rewards] * weights)
            cmp_val = -rets[0]
            acc_val = rets[1]
            mem_val = -rets[2]
            print(
                f"{rank}. Ep {ep} | Total Reward: {total_ret:.2f} | "
                f"CMP: {cmp_val:.1f} | Acc: {acc_val:.4f} | Mem: {mem_val:.1f}"
            )
            print(f"   Sequence (vertex, [(idx1, idx2, factor), ...]): {seq}")
            table.add_data(rank, ep, total_ret, cmp_val, acc_val, mem_val, str(seq))
        wandb.log({f"Top N {name}": table})

    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n("CMP (Lowest FLOPs)", host_state["top_n_cmp"])
    print_top_n("Memory (Lowest Bytes)", host_state["top_n_mem"])
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])
    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
