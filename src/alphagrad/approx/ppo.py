import argparse
import inspect
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
from graphax import examples
from tqdm import tqdm

import heapq
from alphagrad.approx.env import (
    MAX_TOKENS,
    MAX_RULES_PER_VERTEX,
    NUM_AXIS_PAIRS,
    StepAction,
    VertexEliminationEnv,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy, explained_variance, symexp, symlog

# Pair head: NUM_AXIS_PAIRS axis pairs + 1 STOP token marking end of the rule sequence.
NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS  # STOP index inside the pair distribution

# Mapping from pair index (0..NUM_AXIS_PAIRS-1) to (base_idx1, base_idx2). PAIR_STOP -> (-1, -1) sentinel.
_PAIR_TO_BASE = jnp.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [-1, -1],  # STOP
    ],
    dtype=jnp.int32,
)


class Trajectory(NamedTuple):
    tokens: jax.Array
    vertex_idx: jax.Array  # () int32 — 0-indexed within total_v
    pair_seq: jax.Array  # (MAX_RULES_PER_VERTEX,) int32
    factor_seq: jax.Array  # (MAX_RULES_PER_VERTEX,) int32 — index into the FACTORS table
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    next_value: jax.Array
    vertex_dist: jax.Array  # (num_vertices,)
    pair_dists: jax.Array  # (MAX_RULES_PER_VERTEX, NUM_PAIR_CHOICES)
    factor_dists: jax.Array  # (MAX_RULES_PER_VERTEX, NUM_FACTORS)
    discount: jax.Array
    vertex_avail_mask: jax.Array  # (num_vertices,)


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


NN_HIDDEN_DIM = 128
NN_VMAP_BATCH = 16

_DATASET_CACHE: dict = {}


def _dataset_dims(name):
    if name == "mnist":
        return 784, 10
    raise ValueError(f"Unknown dataset '{name}'")


def _load_dataset(name, dataset_size):
    cache_key = (name, dataset_size)
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if name == "mnist":
        import tensorflow_datasets as tfds

        ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
        x_np, y_np = tfds.as_numpy(ds)
        x_np = x_np.reshape(x_np.shape[0], -1).astype(np.float32) / 255.0
        y_np = np.eye(10, dtype=np.float32)[y_np]
        if dataset_size is not None and dataset_size > 0:
            x_np = x_np[:dataset_size]
            y_np = y_np[:dataset_size]
        result = (jnp.asarray(x_np), jnp.asarray(y_np))
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    _DATASET_CACHE[cache_key] = result
    return result


def data_gen(fn_str, dataset=None, dataset_size=-1):
    fn = None
    if fn_str == "Helmholtz":

        @jax.jit
        def fn(keys):
            x = jrand.uniform(keys[0], (4,))
            return (x / jnp.sum(x) * 0.9,)

        return fn

    if fn_str.endswith("NeuralNetwork"):
        if dataset is not None:
            x_data, y_data = _load_dataset(dataset, dataset_size)
            n_samples = int(x_data.shape[0])
            is_vmapped = fn_str.startswith("Vmapped")

            @jax.jit
            def fn(keys):
                if is_vmapped:
                    idx = jrand.randint(keys[0], (NN_VMAP_BATCH,), 0, n_samples)
                else:
                    idx = jrand.randint(keys[0], (), 0, n_samples)
                return x_data[idx], y_data[idx]

            return fn

        @jax.jit
        def fn(keys):
            if fn_str.startswith("Vmapped"):
                shape = (NN_VMAP_BATCH,)
            else:
                shape = ()
            r1 = jrand.uniform(keys[0], shape)
            th1 = jrand.uniform(keys[1], shape, minval=-jnp.pi, maxval=jnp.pi)
            r2 = jrand.uniform(keys[2], shape)
            th2 = jrand.uniform(keys[3], shape, minval=-jnp.pi, maxval=jnp.pi)

            x = jnp.stack([r1, th1 / jnp.pi, r2, th2 / jnp.pi], axis=-1)

            y = jnp.stack(
                [
                    r1 * jnp.cos(th1),
                    r1 * jnp.sin(th1),
                    r2 * jnp.cos(th2),
                    r2 * jnp.sin(th2),
                ],
                axis=-1,
            )

            y += 0.05 * jrand.normal(keys[4], y.shape)

            return x, y

        return fn

    if "Encoder" in fn_str or "Decoder" in fn_str:

        @jax.jit
        def fn(keys):
            if fn_str.startswith("Vmapped"):
                shape_x = (16, 4, 4)
                shape_y = (16, 4, 4)
            else:
                shape_x = (4, 4)
                shape_y = (4, 4)

            x = jrand.normal(keys[0], shape_x)

            y_base = jnp.sin(x * jnp.pi) + jnp.cos(x * jnp.pi)
            y = jax.nn.sigmoid(y_base) + 0.05 * jrand.normal(keys[1], shape_y)

            return x, y

        return fn


def _neural_network(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2


def get_args(fn_str, key, dataset=None):
    if fn_str.endswith("NeuralNetwork"):
        if dataset is not None:
            in_dim, out_dim = _dataset_dims(dataset)
            h = NN_HIDDEN_DIM
            shapes = [
                (in_dim,), (out_dim,),
                (h, in_dim), (h,),
                (out_dim, h), (out_dim,),
            ]
        else:
            shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
    elif fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,), (8,), (8,)]
    elif "EncoderDecoder" in fn_str:
        shapes = [(4, 4)] * 13 + [(4,)] * 8
    elif "Encoder" in fn_str:
        shapes = [(4, 4)] * 10 + [(4,)] * 6
    else:
        return {
            "Simple": (5.0, 7.0),
            "Lighthouse": (0.02,) * 4,
            "Helmholtz": (jnp.array([0.05, 0.15, 0.25, 0.35]),),
            "RobotArm_6DOF": (0.02,) * 6,
            "RoeFlux_1d": (0.01, 0.02, 0.02, 0.01, 0.03, 0.03),
            "RoeFlux_3d": (
                jnp.array([0.1]),
                jnp.array([0.1, 0.2, 0.3]),
                jnp.array([0.5]),
                jnp.array([0.2]),
                jnp.array([0.2, 0.2, 0.4]),
                jnp.array([0.6]),
            ),
            "BlackScholes_Jacobian": (1.0,) * 5,
        }[fn_str]

    if fn_str.startswith("Vmapped"):
        shapes[0] = (NN_VMAP_BATCH, *shapes[0])
        if "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron")):
            shapes[1] = (NN_VMAP_BATCH, *shapes[1])

    keys = jax.random.split(key, len(shapes))
    return [jax.random.normal(k, s) for k, s in zip(keys, shapes)]


def get_fn(fn_str):
    if fn_str.endswith("NeuralNetwork"):
        fn = _neural_network
    elif fn_str.endswith("Perceptron"):
        fn = examples.Perceptron
    else:
        fn = getattr(examples, fn_str, None)
        if fn is None:
            raise ValueError(f"Target function '{fn_str}' not found in examples.")

    if fn_str.startswith("Vmapped"):
        num_args = len(inspect.signature(fn).parameters)
        has_y = "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron"))

        mapped_axes = (0, 0) if has_y else (0,)
        static_axes = (None,) * (num_args - len(mapped_axes))

        fn = jax.vmap(fn, in_axes=mapped_axes + static_axes)

    return fn


class RuleDecoder(eqx.Module):
    """Autoregressive head that emits a sequence of (axis_pair, factor) rules per chosen vertex.

    Each step takes the previous (pair, factor, hidden_state) and emits the next pair logits;
    the factor head is conditioned on the pair just chosen so the two heads form an
    autoregressive pair within a single rule slot.
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
        pair_logits = self.pair_head(new_h)
        return new_h, pair_logits

    def factor_logits_for(self, h, pair_idx):
        pair_emb = self.pair_embed(pair_idx)
        return self.factor_head(jnp.concatenate([h, pair_emb]))


def _stop_only_logits(num_pair_choices: int) -> jax.Array:
    """Logits that put all mass on the STOP pair index, useful to force a stop after a previous stop."""
    return jnp.where(
        jnp.arange(num_pair_choices) == PAIR_STOP, 0.0, -1e9
    ).astype(jnp.float32)


class TransformerPointerPPOAgent(eqx.Module):
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

    def __init__(
        self,
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
        value = self.value_head(summary)
        return vertex_logits, vertex_reprs, value

    def sample_action(self, tokens, vertex_avail_mask, pair_valid_mask, key):
        """Sample (vertex, rule sequence). Returns dists for old-policy book-keeping."""
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
            jnp.array(PAIR_STOP, dtype=jnp.int32),  # BOS uses STOP token as a "no-op prev"
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
        """Teacher-forced log-prob, entropy, value, and per-head dists for a recorded action."""
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

            # Factor only "counts" when there is an actual rule (pair != STOP) and we are still active
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
        _, (lp_pairs, lp_factors, ent_pairs, ent_factors, pair_dists, factor_dists) = lax.scan(
            rule_step,
            init_carry,
            (jnp.arange(self.max_rules), pair_seq, factor_seq),
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


def build_rule_specs(pair_seq, factor_seq, factor_table):
    """Convert per-slot (pair_idx, factor_idx) sequence -> (MAX_RULES, 3) rule specs.

    pair_idx == PAIR_STOP marks the end of the sequence; that slot and all subsequent
    slots are written as unused (base_idx1 = -1, factor = 0).
    """
    base = _PAIR_TO_BASE[pair_seq]  # (MAX_RULES, 2)
    factor_vals = factor_table[factor_seq]  # (MAX_RULES,)

    is_stop = pair_seq == PAIR_STOP
    has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0  # (MAX_RULES,) bool

    base_final = jnp.where(has_stopped[:, None], -1, base)
    factor_final = jnp.where(has_stopped, 0, factor_vals)
    return jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(jnp.int32)


def old_log_prob_for_action(
    vertex_idx, pair_seq, factor_seq, vertex_dist, pair_dists, factor_dists
):
    """Recompute joint log-prob of an action given the *old* policy distributions stored at rollout time."""
    log_p_v = jnp.log(vertex_dist[vertex_idx] + 1e-8)

    is_stop = pair_seq == PAIR_STOP
    prior_stops = jnp.cumsum(is_stop.astype(jnp.int32)) - is_stop.astype(jnp.int32)
    pair_active = (prior_stops == 0).astype(jnp.float32)
    factor_active = (prior_stops == 0) & (~is_stop)
    factor_active_f32 = factor_active.astype(jnp.float32)

    arange_r = jnp.arange(pair_seq.shape[0])
    pair_log_ps = jnp.log(pair_dists[arange_r, pair_seq] + 1e-8) * pair_active
    factor_log_ps = jnp.log(factor_dists[arange_r, factor_seq] + 1e-8) * factor_active_f32

    return log_p_v + jnp.sum(pair_log_ps) + jnp.sum(factor_log_ps)


def init_linear_weights(model, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x) and x.bias is not None
    ]

    weights = get_weights(model)
    biases = get_biases(model)
    init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))

    new_weights = [
        init_fn(subkey, weight.shape)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model


def reward_normalization_fn(reward):
    return symlog(reward)


def inverse_reward_normalization_fn(reward):
    return symexp(reward)


def get_num_clipping_triggers(ratio, eps):
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj

        mask = 1.0 - done
        episodic_return = reward + discount * episodic_return * mask

        value_raw = inverse_reward_normalization_fn(value)
        next_value_raw = inverse_reward_normalization_fn(next_value)

        delta = reward + next_value_raw * discount * mask - value_raw
        advantage = delta + discount * gae_lambda * lastgaelam * mask

        estim_return = advantage + value_raw
        return (episodic_return, advantage), (episodic_return, estim_return, advantage)

    inputs = (rewards, dones, values, next_values, discounts)
    rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
    init_val = jnp.zeros_like(rewards[0])
    _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
    return jax.tree.map(lambda x: x[::-1], output)


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(tree, minibatches, key):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    num_envs, rollout_length = leaves[0].shape[:2]
    size = num_envs * rollout_length // minibatches
    valid_samples = size * minibatches

    indices = jrand.permutation(key, jnp.arange(num_envs * rollout_length))
    indices = indices[:valid_samples].reshape(minibatches, size)

    def _process(x):
        x = x.reshape(-1, *x.shape[2:])
        return x[indices]

    return jax.tree_util.tree_map(_process, tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="approx-ppo")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--wandb", type=str, default="offline")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--example", type=str, default="Helmholtz")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--disable-sparsification", action="store_true")
    parser.add_argument("--cmp-type", type=str, default="flops",
                        choices=["graphax", "flops", "latency"])
    parser.add_argument("--mem-type", type=str, default="peak_memory",
                        choices=["graphax", "bytes_accessed", "peak_memory"])
    parser.add_argument("--rewards", nargs="+", type=str, default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"])
    parser.add_argument("--lambda-cmp", type=float, default=1.0)
    parser.add_argument("--lambda-mem", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trajectories to capture for each metric")
    parser.add_argument("--capture-perfect-grads", action="store_true", help="Capture trajectories with perfect accuracy (cosine similarity = 1.0)")
    parser.add_argument("--exec-on-gpu", action="store_true", help="Enforce execution on GPU 0 and data on GPU 1")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"],
                        help="Dataset to use for NeuralNetwork examples. 'none' falls back to the synthetic 4-D generator.")
    parser.add_argument("--dataset-size", type=int, default=-1,
                        help="Number of training samples to draw batches from. -1 (default) uses the full training set.")
    parser.add_argument("--max-rules", type=int, default=MAX_RULES_PER_VERTEX,
                        help="Max number of (axis_pair, factor) rules per chosen vertex. The autoregressive rule head can emit up to this many before a STOP.")
    parser.add_argument("--factors", type=str, default="-1,1,2,4",
                        help="Comma-separated factor choices for the per-rule factor head. -1 means gcd-based collapse (legacy behaviour); positive ints set explicit block-sparse factors.")
    args = parser.parse_args()

    factor_table_py = tuple(int(x) for x in args.factors.split(",") if x.strip())
    if not factor_table_py:
        raise ValueError("--factors must contain at least one factor value")
    FACTOR_TABLE = jnp.array(factor_table_py, dtype=jnp.int32)
    NUM_FACTORS = FACTOR_TABLE.shape[0]
    if args.max_rules > MAX_RULES_PER_VERTEX:
        raise ValueError(
            f"--max-rules ({args.max_rules}) exceeds env-side MAX_RULES_PER_VERTEX ({MAX_RULES_PER_VERTEX})"
        )
    MAX_RULES = args.max_rules

    if args.exec_on_gpu:
        try:
            gpus = jax.devices("gpu")
        except:
            gpus = []
        if len(gpus) < 2:
            raise RuntimeError(f"Requested --exec-on-gpu but only {len(gpus)} GPU(s) found. "
                               "Check your --gpus argument and CUDA_VISIBLE_DEVICES.")
        main_device = gpus[0]
    else:
        main_device = None

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call, dataset_size=args.dataset_size)

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)

    env_target_fun = target_fn if "acc" in args.rewards else None
    argnums = None
    if args.example.endswith("NeuralNetwork"):
        argnums = (2, 3, 4, 5)
    elif args.example.endswith("Perceptron"):
        argnums = (2, 3, 4, 5, 6, 7)
    else:
        argnums = (0,)


    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr, args=xs, argnums=argnums, num_envs=0, data_gen=gen, target_fun=env_target_fun,
        cmp_type=args.cmp_type, mem_type=args.mem_type, exec_on_gpu=args.exec_on_gpu,
    )

    total_v = len(closed_jaxpr.jaxpr.eqns)
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_valid = len(env.valid_vertices)

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

    # Per-vertex validity for picking a vertex at all (decided by the env-discovered
    # `valid_vertices` set; refined further at rollout time by "not yet chosen").
    vertex_valid_static_np = np.zeros(total_v, dtype=np.float32)
    for v in env.valid_vertices:
        vertex_valid_static_np[v - 1] = 1.0
    vertex_valid_static = jnp.array(vertex_valid_static_np)

    # Per-vertex validity for each pair choice (4 axis pairs + STOP).
    # Pair index layout matches `_PAIR_TO_BASE`:
    #   0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1), 4 -> STOP
    pair_valid_mask_np = np.zeros((total_v, NUM_PAIR_CHOICES), dtype=np.float32)
    pair_valid_mask_np[:, PAIR_STOP] = 1.0  # STOP is always available

    for i, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        if args.disable_sparsification:
            continue
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_ndim = len(eqn.outvars[0].aval.shape)
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        if not invars:
            continue
        min_in_ndim = min(len(v.aval.shape) for v in invars)

        if out_ndim >= 1 and min_in_ndim >= 1:
            pair_valid_mask_np[i, 0] = 1.0  # (0,0)
        if out_ndim >= 1 and min_in_ndim >= 2:
            pair_valid_mask_np[i, 1] = 1.0  # (0,1)
        if out_ndim >= 2 and min_in_ndim >= 1:
            pair_valid_mask_np[i, 2] = 1.0  # (1,0)
        if out_ndim >= 2 and min_in_ndim >= 2:
            pair_valid_mask_np[i, 3] = 1.0  # (1,1)

    pair_valid_mask = jnp.array(pair_valid_mask_np)

    ENTROPY_WEIGHT = 0.05
    VALUE_WEIGHT = 0.5
    EPISODES = args.episodes
    # Synchronize environment parallelism with model batch size for Vmapped examples
    if "Vmapped" in args.example:
        NUM_ENVS = 16
    else:
        NUM_ENVS = os.cpu_count() or 64
    LR = 3e-4
    GAE_LAMBDA = 0.95
    EPS = 0.2
    MINIBATCHES = 32
    PPO_EPOCHS = 2

    NUM_REWARDS = len(args.rewards)
    OBS_SHAPE = MAX_TOKENS
    ROLLOUT_LENGTH = num_valid

    print(
        f"NUM_VERTICES={total_v}, MAX_RULES={MAX_RULES}, NUM_PAIR_CHOICES={NUM_PAIR_CHOICES}, "
        f"NUM_FACTORS={NUM_FACTORS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}, MINIBATCHES={MINIBATCHES}"
    )

    agent_key, init_key, key = jrand.split(key, 3)
    agent = TransformerPointerPPOAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_vertices=total_v,
        num_rewards=NUM_REWARDS,
        max_rules=MAX_RULES,
        num_pair_choices=NUM_PAIR_CHOICES,
        num_factors=NUM_FACTORS,
        value_dims=[64, 32],
        seq_len=OBS_SHAPE,
        key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)

    def get_pointer_weight(agent):
        return agent.pointer_proj.weight

    def get_pair_head_weight(agent):
        return agent.rule_decoder.pair_head.weight

    def get_factor_head_weight(agent):
        return agent.rule_decoder.factor_head.weight

    # Scale down output heads so initial action distributions are near-uniform (stable PPO start).
    agent = eqx.tree_at(get_pointer_weight, agent, get_pointer_weight(agent) * 0.1)
    agent = eqx.tree_at(get_pair_head_weight, agent, get_pair_head_weight(agent) * 0.1)
    agent = eqx.tree_at(get_factor_head_weight, agent, get_factor_head_weight(agent) * 0.1)

    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent
        )

    def generate_eval_samples(env_obj, key):
        config = env_obj.config
        args = env_obj.args

        def get_one_sample(k):
            dk, wk = jrand.split(k)
            e_args = list(args)
            if config.data_gen is not None:
                data = config.data_gen(jrand.split(dk, 5))
                for i, d in enumerate(data):
                    e_args[i] = d
            if config.argnums:
                w_keys = jrand.split(wk, len(config.argnums))
                for i, arg_idx in enumerate(config.argnums):
                    if config.data_gen is not None and arg_idx < len(data):
                        continue
                    curr_val = e_args[arg_idx]
                    e_args[arg_idx] = jrand.normal(
                        w_keys[i], curr_val.shape, curr_val.dtype
                    )

            return tuple(e_args)

        keys = jrand.split(key, 10)
        stacked_args = jax.vmap(get_one_sample)(keys)
        return stacked_args


    def reset_envs(env_obj):
        def _single_reset(_):
            return env_obj.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    LAMBDA_CMP = args.lambda_cmp
    LAMBDA_MEM = args.lambda_mem

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def rollout_fn(agent, env_obj, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            sample_key, next_net_key = jrand.split(key, 2)

            chosen = state.order
            step_idx = state.step_count

            arange_v = jnp.arange(num_valid)
            active_mask = (arange_v < jnp.expand_dims(step_idx, -1)).astype(jnp.float32)

            already_chosen = jnp.zeros(total_v, dtype=jnp.float32)
            already_chosen = already_chosen.at[chosen - 1].add(active_mask)
            vertex_avail_mask = vertex_valid_static * (
                1.0 - jnp.clip(already_chosen, 0.0, 1.0)
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

            target_vertex = jnp.array(vertex_idx + 1, dtype=jnp.int32)
            rule_specs = build_rule_specs(pair_seq, factor_seq, FACTOR_TABLE)
            env_action = StepAction(target_vertex=target_vertex, rule_specs=rule_specs)

            env_out = env_obj.step(state, env_action)
            next_state = env_out.state
            raw_rewards = env_out.reward
            rewards = jnp.array(
                [raw_rewards[0], raw_rewards[1], raw_rewards[2]]
            )[:NUM_REWARDS]
            done = env_out.terminated.astype(jnp.float32)

            _, _, next_value = agent.encode(next_state.tokens, key=next_net_key)

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
                discount=jnp.array(0.99),
                vertex_avail_mask=vertex_avail_mask,
            )

            return next_state, (transition, raw_rewards)

        final_state, (traj, all_raw_rewards) = lax.scan(step_fn, env_state, keys)
        return final_state, traj, all_raw_rewards[-1]


    schedule = optax.cosine_decay_schedule(LR, EPISODES * PPO_EPOCHS * MINIBATCHES, 0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(schedule, b1=0.9, eps=1e-7),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    def loss(agent, batch: TrainBatch, keys):
        # Vmap evaluate_action over the minibatch dimension. pair_valid_mask is shared.
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

        num_triggers = get_num_clipping_triggers(ratio, EPS)
        trigger_ratio = num_triggers / len(ratio)

        clipping_objective = jnp.minimum(
            ratio * batch.norm_adv,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * batch.norm_adv,
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

        # Per-head KL: sum over heads gives the joint KL of the factorized policy.
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
            ppo_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy_loss
        )

        return total_loss, (
            kl_div,
            entropy_loss,
            0.0,
            explained_var,
            ppo_loss,
            VALUE_WEIGHT * value_loss,
            ENTROPY_WEIGHT * entropy_loss,
            total_loss,
            trigger_ratio,
        )

    def train_episode(agent, opt_state, env_states, env_obj, key):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        env_states, traj, total_rewards_full = rollout_fn(agent, env_obj, ROLLOUT_LENGTH, env_states, rollout_keys)


        _, estim_returns, advantages = get_advantages(
            traj.reward,
            traj.done,
            traj.value,
            traj.next_value,
            traj.discount,
            GAE_LAMBDA,
        )

        def normalize(x):
            return (x - jnp.mean(x)) / (jnp.std(x) + 1e-7)

        norm_adv_components = jax.vmap(normalize, in_axes=-1, out_axes=-1)(
            advantages.reshape(-1, advantages.shape[-1])
        ).reshape(advantages.shape)
        
        weights = jnp.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]
        norm_adv = jnp.sum(norm_adv_components * weights, axis=-1)

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
            batches = shuffle_and_batch(full_batch, MINIBATCHES, epoch_key)
            mb_keys = jrand.split(epoch_key, MINIBATCHES)

            def train_minibatch(c, batch_and_key):
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                keys = jrand.split(t_key, batch.tokens.shape[0])
                grads, metrics = eqx.filter_grad(loss, has_aux=True)(
                    comb_agent, batch, keys
                )
                updates, new_opt_state = optimizer.update(
                    grads, comb_opt_state, comb_agent
                )
                new_agent = eqx.apply_updates(comb_agent, updates)
                next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
                return next_carry, metrics

            return lax.scan(train_minibatch, carry, (batches, mb_keys))

        epoch_keys = jrand.split(subkey, PPO_EPOCHS)
        dynamic_carry, metrics_seq = lax.scan(train_epoch, dynamic_carry, epoch_keys)

        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)

        # total_rewards_full shape: (NUM_ENVS, 3)
        # Pack the rich action sequence (vertex + per-slot pair/factor) for host-side reporting.
        actions_pack = (traj.vertex_idx, traj.pair_seq, traj.factor_seq)
        return agent, opt_state, env_states, metrics, total_rewards_full, actions_pack

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    pbar = tqdm(total=EPISODES)

    host_state = {
        "samplecounts": 0,
        "best_global_return": -float("inf"),
        "best_global_act_seq": None,
        "top_n_total": [],  # (value, episode, rewards, actions)
        "top_n_cmp": [],
        "top_n_mem": [],
        "top_n_acc": [],
    }

    factor_table_np = np.array(factor_table_py, dtype=np.int32)

    def _action_to_pylist(vertex_seq, pair_seq, factor_seq):
        """Decode (vertex, pair, factor) sequences into a list of (vertex, [(idx1, idx2, factor), ...]) for display."""
        out = []
        for v_idx, p_row, f_row in zip(vertex_seq, pair_seq, factor_seq):
            rules = []
            for slot in range(MAX_RULES):
                p = int(p_row[slot])
                if p == PAIR_STOP:
                    break
                base_idx1, base_idx2 = (
                    int(_PAIR_TO_BASE[p, 0]),
                    int(_PAIR_TO_BASE[p, 1]),
                )
                factor = int(factor_table_np[int(f_row[slot])])
                rules.append((base_idx1, base_idx2, factor))
            out.append((int(v_idx) + 1, rules))
        return out

    def host_log(ep, all_rets, actions_pack, mean_r, mets):
        ep = int(ep)
        all_rets = np.array(all_rets)
        v_idx_arr = np.array(actions_pack[0])  # (NUM_ENVS, ROLLOUT_LENGTH)
        pair_arr = np.array(actions_pack[1])  # (NUM_ENVS, ROLLOUT_LENGTH, MAX_RULES)
        factor_arr = np.array(actions_pack[2])  # (NUM_ENVS, ROLLOUT_LENGTH, MAX_RULES)
        mean_r = np.atleast_1d(np.array(mean_r))

        host_state["samplecounts"] += NUM_ENVS * ROLLOUT_LENGTH
        (
            kl_div,
            policy_entropy,
            fit_quality,
            explained_var,
            ppo_loss,
            value_loss,
            entropy_loss,
            total_loss,
            clipping_trigger_ratio,
        ) = [float(m) for m in mets]

        # Categories mapping in all_rets (raw_rewards from env.py):
        # 0: -cmp, 1: error (acc), 2: -mem
        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]

        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
            decoded = _action_to_pylist(v_idx_arr[i], pair_arr[i], factor_arr[i])

            # 1. Total Reward
            total_ret = float(np.sum(rets[:NUM_REWARDS] * weights))
            if len(host_state["top_n_total"]) < args.top_n:
                heapq.heappush(host_state["top_n_total"], (total_ret, ep, list(rets), decoded))
            else:
                heapq.heappushpop(host_state["top_n_total"], (total_ret, ep, list(rets), decoded))

            # 2. CMP (minimize FLOPs -> maximize -cmp)
            cmp_val = float(rets[0])
            if len(host_state["top_n_cmp"]) < args.top_n:
                heapq.heappush(host_state["top_n_cmp"], (cmp_val, ep, list(rets), decoded))
            else:
                heapq.heappushpop(host_state["top_n_cmp"], (cmp_val, ep, list(rets), decoded))

            # 3. Memory (minimize memory -> maximize -mem)
            mem_val = float(rets[2])
            if len(host_state["top_n_mem"]) < args.top_n:
                heapq.heappush(host_state["top_n_mem"], (mem_val, ep, list(rets), decoded))
            else:
                heapq.heappushpop(host_state["top_n_mem"], (mem_val, ep, list(rets), decoded))

            # 4. Accuracy (maximize cosine similarity)
            acc_val = float(rets[1])
            if args.capture_perfect_grads or acc_val < 0.999999:
                if len(host_state["top_n_acc"]) < args.top_n:
                    heapq.heappush(host_state["top_n_acc"], (acc_val, ep, list(rets), decoded))
                else:
                    heapq.heappushpop(host_state["top_n_acc"], (acc_val, ep, list(rets), decoded))

        # Update global best for traditional tracking
        best_of_batch_idx = int(np.argmax(np.sum(all_rets[:, :NUM_REWARDS] * weights, axis=-1)))
        best_of_batch_ret = float(np.sum(all_rets[best_of_batch_idx, :NUM_REWARDS] * weights))

        if best_of_batch_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = best_of_batch_ret
            host_state["best_global_act_seq"] = _action_to_pylist(
                v_idx_arr[best_of_batch_idx],
                pair_arr[best_of_batch_idx],
                factor_arr[best_of_batch_idx],
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
        if len(mean_r) > 1:
            for i in range(1, len(mean_r)):
                log_dict[f"mean_return_{i}"] = mean_r[i]

        wandb.log(log_dict)
        pbar.update(1)
        
        # Display best from CURRENT episode in pbar
        b_ret_unnorm = np.abs(all_rets[best_of_batch_idx])
        mean_r_unnorm = np.abs(mean_r)
        b_ret_desc = ", ".join([f"{float(x):.1f}" for x in b_ret_unnorm])
        means_str = ", ".join([f"{float(x):.2f}" for x in mean_r_unnorm])
        desc = f"ent:{policy_entropy:.3f} best:{b_ret_desc} means:{means_str}"
        pbar.set_description(desc)

    for ep in range(EPISODES):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        
        eval_samples = generate_eval_samples(env, ep_eval_key)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        env_states = reset_envs(env_episode)
        agent, opt_state, _, metrics, total_rewards_full, actions_pack = train_episode(
            agent, opt_state, env_states, env_episode, ep_key
        )

        host_log(
            ep,
            total_rewards_full,
            actions_pack,
            jnp.mean(total_rewards_full[:, :NUM_REWARDS], axis=0),
            metrics,
        )

    pbar.close()

    def print_top_n(name, heap, reverse_val=True):
        print(f"\nTop {args.top_n} trajectories for {name}:")
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=reverse_val)

        table = wandb.Table(columns=["rank", "episode", "total_reward", "cmp", "acc", "mem", "sequence"])

        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]

        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            total_ret = np.sum(np.array(rets)[:NUM_REWARDS] * weights)
            cmp_val = -rets[0]
            acc_val = rets[1]
            mem_val = -rets[2]

            # `seq` is a list of (vertex, [(idx1, idx2, factor), ...]) tuples — already decoded by host_log.
            print(f"{rank}. Ep {ep} | Total Reward: {total_ret:.2f} | CMP: {cmp_val:.1f} | Acc: {acc_val:.4f} | Mem: {mem_val:.1f}")
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
