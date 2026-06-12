"""MuZero trainer for the vertex-elimination env — hierarchical MCTS.

Each "real" elimination step is decomposed into ``1 + 2 · max_rules`` MCTS
depth levels:

  * depth 0           — vertex selection
  * depth 2k+1        — pair selection at rule-slot k (k = 0..max_rules-1)
  * depth 2k+2        — factor selection at rule-slot k
  * depth 1+2·max_rules — commit: at this depth, the ``last_reward`` field
    of the embedding holds the env reward predicted by the reward head
    (intermediate decision steps have last_reward = 0).

Unlike alpha0, mu0's MCTS runs entirely in the agent's latent space — the
environment is *not* called inside the search. ``dynamics(latent, action)``
produces the next latent and a predicted reward; ``prediction(latent)``
produces the unified-action policy logits and value estimate. The dynamics
learns to interpret the action ID via context in the latent: at vertex
depth the latent says "we're picking a vertex", at pair depth "we just
picked a vertex, picking pair_k", etc. No explicit depth conditioning is
applied to the dynamics — the latent's evolution carries the context.

Action space is unified to ``UNIFIED = max(total_v, NPC, num_factors)`` so
mctx sees a single fixed-size flat space; per-depth masks on the prior
zero out invalid actions. The env action that gets played in the *real*
rollout is built by sampling the rule sequence autoregressively through
the agent (one dynamics-call per decision, then prediction over the new
latent gives the next-decision logits).

CLI features ported from ``ppo.py`` for parity with the comparison study:

* ``--variant`` + ``--curriculum`` + ``full_curriculum`` — pre-canned
  configuration mapping to factors / max_rules / op-legality overrides
  (and an auto-curriculum when ``full_curriculum`` is selected). See
  :data:`VARIANT_PRESETS` and :func:`_apply_variant_preset`.
* ``--set-transformer-agg`` — permutation-invariant aggregation over
  calibration samples for the per-vertex data embedding.
* ``--cache-encoding`` — encode the residual jaxpr once per episode and
  reuse the encoder output for every step's representation.
* ``--cosine-lower-bound`` / ``--cosine-upper-bound`` — Lagrangian
  constraints on the ``cosine_sim`` reward channel.
* ``--calibrate-steps`` — pre-training reward-scale calibration that
  rescales ``reward_weights`` by ``1 / mean_abs(symlog(reward))`` per
  channel so the wide-magnitude reward families contribute comparably.
* ``--exec-on-gpu`` — pin training to GPU 0 and the env eval callback to
  GPU 1.
* ``--loss-mode scalar`` — mu0's default loss path is already a single
  scalar reward per step (the MuZero value head is scalar); the flag is
  accepted for CLI parity with ppo.py.
* ``--dynamic-substeps`` — accepted for CLI parity. mu0's unified action
  space already accommodates the per-decision typed action sequence;
  this flag is a no-op selector for the action layout (currently the
  unified path is the only one).
"""

from __future__ import annotations

import argparse
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
import mctx
import numpy as np
import optax
import wandb
from tqdm import tqdm

# Swap tqdm's default multiprocessing.RLock for a threading.RLock so the
# named POSIX semaphore behind it never gets created — otherwise it leaks
# on signal-kill. See ppo.py for the full rationale.
import threading as _threading
tqdm.set_lock(_threading.RLock())

from alphagrad.approx.common import (
    NUM_VERTEX_FEATURES,
    OP_TYPE_VOCAB_SIZE,
    SCHEDULES,
    build_pair_valid_mask,
    build_vertex_valid_static,
    compute_per_sample_vertex_features,
    compute_vertex_features,
    data_gen,
    extract_path_visits,
    generate_eval_samples,
    get_args,
    get_fn,
    infer_argnums,
    init_linear_weights,
    init_replay_buffer,
    load_replay_buffer,
    replay_add_batch,
    replay_sample,
    reward_normalization_fn,
    sample_preferences,
    save_replay_buffer,
    scale_module_weight,
    schedule_at,
    vertex_avail_at_step,
)
from alphagrad.approx.common.schedules import cosine_warmup_exp_decay_lr
from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_AXIS_PAIRS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)
from alphagrad.approx.mu0_args import make_argparser
from alphagrad.approx.variants import (
    VARIANT_PRESETS,
    _apply_variant_preset,
    _current_stage_at,
    _current_stage_index,
    _default_full_curriculum,
    _parse_curriculum,
    _pin_rules_for_variant,
)
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.transformer.encoder import RelationalMultiheadAttention


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PAIR_CHOICES = NUM_AXIS_PAIRS + 1
PAIR_STOP = NUM_AXIS_PAIRS

# Pair-index → ``(base_idx1, base_idx2)`` for the four real axis pairs;
# the STOP row is a sentinel that decodes to (-1, -1) inside the env.
_PAIR_TO_BASE = jnp.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [-1, -1],
    ],
    dtype=jnp.int32,
)

# Cross-channel scale handling for the calibration phase. Mirrors ppo.py:
# every reward channel except cosine_sim is symlog-compressed before the
# per-channel mean-abs is computed. cosine_sim is already bounded to ~1 so
# symlog would distort the Lagrangian thresholds.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)
_NO_SYMLOG_MASK: "jax.Array" = (
    jnp.zeros((NUM_REWARDS,), dtype=jnp.bool_)
    .at[jnp.asarray(_NO_SYMLOG_REWARD_INDICES, dtype=jnp.int32)]
    .set(True)
)
_NO_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=np.bool_)
_NO_SYMLOG_MASK_NP[list(_NO_SYMLOG_REWARD_INDICES)] = True


def _symlog_rewards(reward_vec: "jax.Array") -> "jax.Array":
    """Apply symlog elementwise on the last axis, leaving cosine_sim raw."""
    return jnp.where(
        _NO_SYMLOG_MASK,
        reward_vec,
        reward_normalization_fn(reward_vec),
    )


# ---------------------------------------------------------------------------
# Reward weighting (CLI mapping onto the canonical 8-vec)
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
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.reward_scaling.build_reward_weights`.

    Note: the canonical helper applies a `muls_adds_fmas` fallback when
    every channel ends up zero. Pre-refactor `mu0._build_reward_weights`
    silently returned all-zeros in that case (and the downstream
    `_per_channel_discounted_returns` would have multiplied by zero
    everywhere), so the new behaviour is strictly safer.
    """
    from alphagrad.approx.common.reward_scaling import build_reward_weights
    return build_reward_weights(args)


# ---------------------------------------------------------------------------
# Factor-table helpers (mirrors PPO's _build_factor_table)
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _build_factor_table(args) -> tuple[jax.Array, tuple[int, ...], int, int]:
    """Construct the factor table from ``--factors``. Returns
    ``(factor_table, factors_py, num_factors, max_rules)``.
    """
    factors_py = tuple(_parse_int_list(args.factors))
    if not factors_py:
        raise ValueError("--factors must list at least one factor value.")
    factor_table = jnp.array(factors_py, dtype=jnp.int32)
    return factor_table, factors_py, factor_table.shape[0], int(args.max_rules)


# ---------------------------------------------------------------------------
# Comparison-study variant presets
# ---------------------------------------------------------------------------

# Mirrors ppo.py's VARIANT_PRESETS. Each value is a dict of args fields to
# overwrite. ``custom`` is the no-op default. ``full_curriculum`` is a
# mu0-specific addition that triggers the default 3-stage curriculum
# (diag_gcd → diag_factor → full) when ``--curriculum`` is empty.
def make_curriculum_schedule(args, curriculum_stages):
    """Build a piecewise cosine-warmup + exp-decay LR schedule across stages."""
    steps_per_episode = max(args.minibatches, 1)
    stage_step_counts = [
        max(n * steps_per_episode, 1) for _, n in curriculum_stages
    ]
    boundaries: list[int] = [0]
    for s in stage_step_counts:
        boundaries.append(boundaries[-1] + s)
    warmup_frac = float(args.curriculum_warmup_frac)

    def schedule(step):
        step_f = jnp.asarray(step, dtype=jnp.float32)
        lr = jnp.asarray(args.lr * args.lr_decay_min_mult, dtype=jnp.float32)
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


# ---------------------------------------------------------------------------
# Lagrangian constraint parsing (mirrors ppo.py)
# ---------------------------------------------------------------------------


def parse_lagrangian_constraints(
    specs: list[str],
) -> list[tuple[int, float, int]]:
    """Parse ``NAME>=THRESH`` / ``NAME<=THRESH`` to ``(idx, threshold, sign)``."""
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


# ---------------------------------------------------------------------------
# Set Transformer aggregator (mirrors ppo.py B.3)
# ---------------------------------------------------------------------------


class SetTransformerAggregator(eqx.Module):
    """Permutation-invariant aggregator over calibration samples.

    Architecture mirrors ppo.py's SetTransformerAggregator: per-vertex
    self-attention over the sample axis followed by mean-pool and a
    projection to ``embd_dim``. With 5 calibration samples the attention
    cost is negligible compared to the main encoder.
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
        op_emb = jax.vmap(self.op_embedding)(op_ids)
        cont = per_sample_features[:, :, 1:]
        op_emb_b = jnp.broadcast_to(op_emb[None, :, :], (S, V, op_emb.shape[-1]))
        combined = jnp.concatenate([op_emb_b, cont], axis=-1)
        h = jax.vmap(jax.vmap(self.input_proj))(combined)
        h_perm = jnp.transpose(h, (1, 0, 2))  # (V, S, hidden)
        attn_keys = jrand.split(key, V)
        h_attn = jax.vmap(lambda x, k: self.sample_attn(x, x, x, key=k))(
            h_perm, attn_keys
        )
        pooled = jnp.mean(h_attn, axis=1)
        return jax.vmap(self.output_proj)(pooled)


# ---------------------------------------------------------------------------
# Cached encoding (mirrors ppo.py B.4.next)
# ---------------------------------------------------------------------------


class CachedEncoding(NamedTuple):
    """Per-rollout cache for the ``--cache-encoding`` path.

    Captured from the initial residual jaxpr at episode start; reused as the
    leading latent for every step of the rollout instead of re-running the
    transformer stack on the per-step residual jaxpr. The vertex_avail_mask
    is the only thing that varies inside an episode.
    """

    tokens: jax.Array
    eqn_ids: jax.Array
    latent: jax.Array  # (latent_dim,) — already includes pref_proj if any


# ---------------------------------------------------------------------------
# Hierarchical MCTS embedding
# ---------------------------------------------------------------------------


class DecisionEmbedding(NamedTuple):
    """State inside the hierarchical MCTS tree."""

    latent: jax.Array              # (latent_dim,)
    vertex_avail_mask: jax.Array   # (total_v,) float32 — 1 = available
    depth: jax.Array               # scalar int32 in [0, DECISION_DEPTH)
    vertex_idx: jax.Array          # scalar int32 — selected vertex
    pair_seq: jax.Array            # (max_rules,) int32
    factor_seq: jax.Array          # (max_rules,) int32
    active: jax.Array              # bool — true while still adding rules
    last_reward: jax.Array         # scalar — reward delivered to mctx on
                                   # the *next* recurrent_fn call.


def _empty_decision(
    latent, vertex_avail_mask, max_rules: int,
) -> DecisionEmbedding:
    return DecisionEmbedding(
        latent=latent,
        vertex_avail_mask=vertex_avail_mask,
        depth=jnp.array(0, dtype=jnp.int32),
        vertex_idx=jnp.array(0, dtype=jnp.int32),
        pair_seq=jnp.full((max_rules,), PAIR_STOP, dtype=jnp.int32),
        factor_seq=jnp.zeros((max_rules,), dtype=jnp.int32),
        active=jnp.array(True, dtype=jnp.bool_),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# MuZero agent
# ---------------------------------------------------------------------------


class MuZeroAgent(eqx.Module):
    """Encoder → latent representation; MLP dynamics; policy/value/reward heads.

    The dynamics is shared across all decision types — it learns to
    interpret a unified-space action id from context the latent has built
    up over prior decisions.

    Stage B.2.A / B.3 add a data-conditioned path that mirrors ppo.py: when
    ``vertex_features`` are supplied to :meth:`representation`, the per-vertex
    op-type embedding + continuous feature projection (or the
    :class:`SetTransformerAggregator` for per-sample features) is mean-pooled
    over vertices and added to the leading latent. Initialised to zero so the
    unconditioned baseline is preserved at step 0.
    """

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder

    action_embedding: eqx.nn.Embedding
    dynamics_mlp: MLP
    reward_head: MLP

    policy_head: MLP
    value_head: MLP

    # Stage F preference projection — adds ``w → latent_dim`` to the
    # initial latent so the policy/value/dynamics graph below conditions
    # on the preference.
    pref_proj: eqx.nn.Linear

    # Stage B.2.A / B.3: per-vertex feature processing.
    op_embedding: eqx.nn.Embedding
    vertex_feature_proj: eqx.nn.Linear
    set_transformer_agg: SetTransformerAggregator
    # Project the per-vertex data embedding (mean-pooled to a single vector)
    # into the latent so it adds to ``representation``'s output.
    vertex_features_to_latent: eqx.nn.Linear

    num_actions: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    op_embd_dim: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        latent_dim,
        num_actions,
        num_rewards,
        policy_dims,
        value_dims,
        op_embd_dim,
        seq_len,
        key,
    ):
        keys = jrand.split(key, 12)
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.num_rewards = num_rewards
        self.embd_dim = embd_dim
        self.op_embd_dim = op_embd_dim
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(
            num_layers, num_heads, embd_dim, hidden_dim, key=keys[1],
        )
        self.action_embedding = eqx.nn.Embedding(num_actions, latent_dim, key=keys[2])
        self.dynamics_mlp = MLP(
            latent_dim * 2, latent_dim, [hidden_dim], key=keys[3],
        )
        self.reward_head = MLP(latent_dim, 1, value_dims, key=keys[4])
        self.policy_head = MLP(latent_dim, num_actions, policy_dims, key=keys[5])
        self.value_head = MLP(latent_dim, 1, value_dims, key=keys[6])
        self.pref_proj = eqx.nn.Linear(num_rewards, latent_dim, key=keys[7])
        # Per-vertex feature processing.
        self.op_embedding = eqx.nn.Embedding(
            OP_TYPE_VOCAB_SIZE, op_embd_dim, key=keys[8],
        )
        proj_in_dim = op_embd_dim + NUM_VERTEX_FEATURES - 1
        self.vertex_feature_proj = eqx.nn.Linear(
            proj_in_dim, embd_dim, key=keys[9],
        )
        self.set_transformer_agg = SetTransformerAggregator(
            num_features=NUM_VERTEX_FEATURES,
            op_embd_dim=op_embd_dim,
            hidden_dim=embd_dim,
            num_heads=num_heads,
            embd_dim=embd_dim,
            vocab_size=OP_TYPE_VOCAB_SIZE,
            key=keys[10],
        )
        self.vertex_features_to_latent = eqx.nn.Linear(
            embd_dim, latent_dim, key=keys[11],
        )

    def _data_embedding(self, vertex_features, *, agg_key=None):
        """Project per-vertex features to a per-vertex embedding ``(V, embd_dim)``."""
        if vertex_features.ndim == 3:
            agg_key = agg_key if agg_key is not None else jrand.PRNGKey(0)
            return self.set_transformer_agg(vertex_features, key=agg_key)
        op_ids = vertex_features[:, 0].astype(jnp.int32)
        op_emb = jax.vmap(self.op_embedding)(op_ids)
        cont = vertex_features[:, 1:]
        combined = jnp.concatenate([op_emb, cont], axis=-1)
        return jax.vmap(self.vertex_feature_proj)(combined)

    def representation(
        self,
        tokens,
        eqn_ids=None,
        *,
        key=None,
        preference=None,
        vertex_features=None,
        agg_key=None,
    ):
        token_mask = (tokens != 0)
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        x = self.encoder(x, eqn_ids=eqn_ids, key=enc_key)
        mask = token_mask[..., None].astype(x.dtype)
        latent = jnp.sum(x * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-9)
        # Inject the preference once at the leading latent so the entire MCTS
        # sub-tree is conditioned without per-call plumbing.
        if preference is not None:
            latent = latent + self.pref_proj(preference)
        # Inject per-vertex data features (Stage B.2.A / B.3).
        if vertex_features is not None:
            data_emb = self._data_embedding(vertex_features, agg_key=agg_key)
            data_pool = jnp.mean(data_emb, axis=0)
            latent = latent + self.vertex_features_to_latent(data_pool)
        return latent

    def dynamics(self, latent, action):
        a_emb = self.action_embedding(action)
        x = jnp.concatenate([latent, a_emb], axis=-1)
        next_latent = self.dynamics_mlp(x)
        reward = self.reward_head(next_latent)[0]
        return next_latent, reward

    def prediction(self, latent):
        logits = self.policy_head(latent)
        value = self.value_head(latent)[0]
        return logits, value


# ---------------------------------------------------------------------------
# Trajectory layout — one row per real elimination step
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    tokens: jax.Array              # (T, MAX_TOKENS) int32
    eqn_ids: jax.Array             # (T, MAX_TOKENS) int32
    vertex_idx: jax.Array          # (T,) int32
    pair_seq: jax.Array            # (T, max_rules) int32
    factor_seq: jax.Array          # (T, max_rules) int32
    reward_vec: jax.Array          # (T, NUM_REWARDS) per-step reward vec
    scalar_reward: jax.Array       # (T,) — dot(reward_vec, weights)
    mcts_visits: jax.Array         # (T, DECISION_DEPTH, UNIFIED) — per-depth
                                   # visit-count distributions along the
                                   # chosen path through the search tree.
    mcts_value: jax.Array          # (T,) — root MCTS value
    preference: jax.Array          # (T, NUM_REWARDS) — per-env preference
                                   # broadcast to every step.


class TrajectoryWindow(NamedTuple):
    """A window of length ``UNROLL_STEPS + 1`` real steps, used by the loss."""

    tokens: jax.Array
    eqn_ids: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    scalar_reward: jax.Array
    target_value: jax.Array
    mcts_visits: jax.Array
    preference: jax.Array


# ---------------------------------------------------------------------------
# Helpers
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


def _scale_output_heads(agent, scale: float):
    agent = scale_module_weight(agent, lambda a: a.policy_head.layers[-2].weight, scale)
    agent = scale_module_weight(agent, lambda a: a.value_head.layers[-2].weight, scale)
    agent = scale_module_weight(agent, lambda a: a.reward_head.layers[-2].weight, scale)
    # Zero the data-feature projection's output so the initial behaviour
    # matches the baseline that doesn't consume vertex_features. Gradient
    # still flows in normally once training starts.
    agent = scale_module_weight(
        agent, lambda a: a.vertex_feature_proj.weight, 0.0,
    )
    agent = scale_module_weight(
        agent, lambda a: a.set_transformer_agg.output_proj.weight, 0.0,
    )
    agent = scale_module_weight(
        agent, lambda a: a.vertex_features_to_latent.weight, 0.0,
    )
    agent = scale_module_weight(agent, lambda a: a.pref_proj.weight, 0.0)
    return agent


def _setup_jax_compile_cache() -> None:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.cache.setup_jax_compile_cache`."""
    from alphagrad.approx.common.cache import setup_jax_compile_cache
    setup_jax_compile_cache()


def _discounted_returns(rewards: jax.Array, discount: float) -> jax.Array:
    """Per-step discounted return ``G_t = Σ_{j≥t} γ^{j-t} r_j``."""

    def step(carry, r):
        new = r + discount * carry
        return new, new

    init = jnp.zeros((), dtype=rewards.dtype)
    _, returns_rev = lax.scan(step, init, rewards[::-1])
    return returns_rev[::-1]


def _compute_traj_priorities(
    fresh_traj: Trajectory, reward_weights: jax.Array,
) -> jax.Array:
    del reward_weights
    episode_return = jnp.sum(fresh_traj.scalar_reward, axis=1)
    return episode_return - jnp.min(episode_return) + 1e-3


def _action_to_pylist(
    vertex_seq, pair_seq, factor_seq, factor_table_np,
) -> list[tuple[int, list]]:
    """Decode a per-step (vertex, pair_seq, factor_seq) trace to env-style."""
    pair_to_base = np.asarray(_PAIR_TO_BASE)
    out: list[tuple[int, list]] = []
    for t in range(vertex_seq.shape[0]):
        v = int(vertex_seq[t])
        rules: list[tuple[int, int, int]] = []
        for slot in range(pair_seq.shape[1]):
            p = int(pair_seq[t, slot])
            f_idx = int(factor_seq[t, slot])
            if p == PAIR_STOP:
                break
            base1 = int(pair_to_base[p, 0])
            base2 = int(pair_to_base[p, 1])
            factor = int(factor_table_np[f_idx])
            rules.append((base1, base2, factor))
        out.append((v + 1, rules))
    return out


def _episode_vertex_features(
    args,
    jaxpr,
    consts: tuple,
    base_args: tuple,
    eval_samples,
    argnums: tuple,
) -> "jax.Array":
    """Single dispatch site for the per-vertex feature computation.

    Switches between mean-aggregated and per-sample features based on
    ``--set-transformer-agg``.
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
# Batching
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def _shuffle_and_batch_windows(window_batch: TrajectoryWindow, minibatches: int, key):
    sample = window_batch.vertex_idx
    num_envs, num_windows = sample.shape[:2]
    mb_size = (num_envs * num_windows) // minibatches
    valid = mb_size * minibatches

    def reshape_one(x):
        x = x.reshape((num_envs * num_windows,) + x.shape[2:])
        x = jrand.permutation(key, x, axis=0)
        return x[:valid].reshape((minibatches, mb_size) + x.shape[1:])

    return jax.tree_util.tree_map(reshape_one, window_batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = make_argparser().parse_args()

    # Apply the --variant preset before anything else looks at args.factors
    # etc. Explicit CLI flags passed alongside --variant are overridden.
    auto_curriculum = args.variant == "full_curriculum" and not args.curriculum.strip()
    _apply_variant_preset(args)
    if args.variant != "custom":
        print(
            f"--variant={args.variant} applied: factors={args.factors!r}, "
            f"max_rules={args.max_rules}, "
            f"pin_rules_to_exact={args.pin_rules_to_exact}"
        )

    # Curriculum: parse the explicit spec or expand the implicit one for
    # ``full_curriculum``. The curriculum runner gates the rule head per
    # stage via the (pin_rules, op_legality) pair derived from the stage
    # name; the agent itself is built once with the final-stage action
    # footprint, mirroring ppo.py's behaviour.
    if auto_curriculum:
        # MuZero uses the legacy 3-stage curriculum
        # (diag_gcd → diag_factor → full). The 7-stage round-robin
        # curriculum is PPO-only until MuZero's prior-gating learns
        # to consume per-episode rotation masks. See
        # alphagrad/src/alphagrad/approx/CURRICULUM.md §"MuZero" for
        # the migration plan.
        curriculum_stages = _default_full_curriculum(args.episodes)
        print(
            f"--variant=full_curriculum: auto curriculum (3-stage, MuZero) "
            + " → ".join(f"{name}:{n}" for name, n in curriculum_stages)
        )
    else:
        curriculum_stages = _parse_curriculum(args.curriculum)
    if curriculum_stages:
        args.episodes = sum(n for _, n in curriculum_stages)
        print(
            "curriculum: "
            + " → ".join(f"{name}:{n}" for name, n in curriculum_stages)
            + f"  (total {args.episodes} episodes)"
        )

    main_device = _resolve_main_device(args)
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    _setup_jax_compile_cache()

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    # ---------------- Env ----------------
    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call, dataset_size=args.dataset_size)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    argnums = infer_argnums(args.example)

    # Always pass target_fun so flops/bytes_accessed/latency_ns/peak_memory
    # populate every step (see cpu_approx_worker.py for the full rationale).
    env_target_fun = target_fn
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

    total_v = len(closed_jaxpr.jaxpr.eqns)
    num_valid = len(env.valid_vertices)

    vertex_valid_static = build_vertex_valid_static(env.valid_vertices, total_v)
    pair_valid_mask = build_pair_valid_mask(
        closed_jaxpr.jaxpr, total_v,
        num_pair_choices=NUM_PAIR_CHOICES, pair_stop_idx=PAIR_STOP,
        disable_sparsification=args.disable_sparsification,
    )

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args)
    factor_table_np = np.array(factors_py, dtype=np.int32)
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    DECISION_DEPTH = 1 + 2 * max_rules
    UNIFIED_ACTION_SIZE = int(max(total_v, NUM_PAIR_CHOICES, num_factors))

    num_envs = _resolve_num_envs(args.num_envs, args.example)
    rollout_length = num_valid

    if (num_envs * rollout_length) // args.minibatches == 0:
        raise ValueError(
            f"--minibatches={args.minibatches} > num_envs * rollout "
            f"({num_envs} * {rollout_length} = {num_envs * rollout_length}). "
            "Each minibatch would be empty, so the loss becomes NaN and no "
            "learning happens. Lower --minibatches or raise --num-envs."
        )

    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)

    nonzero_w = ", ".join(
        f"{REWARD_NAMES[i]}={float(reward_weights_np[i]):+.3g}"
        for i in range(NUM_REWARDS)
        if reward_weights_np[i] != 0.0
    )
    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"num_envs={num_envs}, max_rules={max_rules}, "
        f"factors={factors_py}, decision_depth={DECISION_DEPTH}, "
        f"unified_action_size={UNIFIED_ACTION_SIZE}",
    )
    print(f"reward weights: {nonzero_w or '<all zero — debug only>'}")
    print(f"loss_mode={args.loss_mode}, dynamic_substeps={args.dynamic_substeps}, "
          f"cache_encoding={args.cache_encoding}, "
          f"set_transformer_agg={args.set_transformer_agg}")
    if args.mcts_mode == "hierarchical":
        mode_detail = (
            f"vertex MCTS (muzero_policy, max_depth=1) + "
            f"rule Gumbel (max_considered={args.gumbel_max_considered}, "
            f"scale={args.gumbel_scale})"
        )
    elif args.mcts_mode == "gumbel":
        mode_detail = (
            f"Gumbel MuZero (max_considered={args.gumbel_max_considered}, "
            f"scale={args.gumbel_scale})"
        )
    else:
        mode_detail = f"Sampled MuZero (K={args.sampled_k})"
    print(f"mcts_mode={args.mcts_mode}: {mode_detail}")

    # ---------------- Stage F Lagrangian state ----------------
    # ``--anti-degeneracy`` is desugared here so the corridor bounds used
    # by ``corridor/*_band_fraction`` instrumentation match the
    # constraints the trainer enforces (single source of truth across
    # PPO / MuZero / GFN).
    from alphagrad.approx.common.anti_degeneracy import (
        desugar_anti_degeneracy,
    )
    user_constraints, _corridor_low, _corridor_high = desugar_anti_degeneracy(
        list(args.lagrangian_constraint),
        getattr(args, "anti_degeneracy", "none"),
        float(getattr(args, "anti_degeneracy_delta", 0.01)),
        float(getattr(args, "cosine_lower_bound", 0.8)),
        float(getattr(args, "cosine_upper_bound", 0.9)),
    )
    # Legacy: if --anti-degeneracy is left at its default "none", honour
    # the explicit --cosine-lower-bound / --cosine-upper-bound bounds
    # for back-compat with existing mu0 dispatch scripts.
    if getattr(args, "anti_degeneracy", "none") == "none":
        if args.cosine_lower_bound > 0.0:
            user_constraints.append(f"cosine_sim>={args.cosine_lower_bound}")
            _corridor_low = float(args.cosine_lower_bound)
        if args.cosine_upper_bound < 1.0:
            user_constraints.append(f"cosine_sim<={args.cosine_upper_bound}")
            _corridor_high = float(args.cosine_upper_bound)
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
        [idx for idx, _, _ in constraint_specs], dtype=jnp.int32,
    )
    constraint_thresholds = jnp.asarray(
        [t for _, t, _ in constraint_specs], dtype=jnp.float32,
    )
    constraint_signs = jnp.asarray(
        [float(sign) for _, _, sign in constraint_specs], dtype=jnp.float32,
    )
    multipliers = jnp.zeros(len(constraint_specs), dtype=jnp.float32)

    # ---------------- Agent ----------------
    agent_key, init_key, key = jrand.split(key, 3)
    agent = MuZeroAgent(
        vocab_size=args.vocab_size,
        embd_dim=args.embd_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_actions=UNIFIED_ACTION_SIZE,
        num_rewards=NUM_REWARDS,
        policy_dims=_parse_int_list(args.policy_dims),
        value_dims=_parse_int_list(args.value_dims),
        op_embd_dim=args.op_embd_dim,
        seq_len=MAX_TOKENS,
        key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale)

    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent,
        )

    # Optimiser. Default = single cosine decay across the whole run. When
    # --curriculum (or auto_curriculum) is set, swap in a piecewise schedule.
    if curriculum_stages:
        schedule = make_curriculum_schedule(args, curriculum_stages)
        print(
            "curriculum LR: piecewise cosine_warmup_exp_decay across "
            + " → ".join(
                f"{name}({n * args.minibatches}st)"
                for name, n in curriculum_stages
            )
            + f" (warmup_frac={args.curriculum_warmup_frac}, "
            f"end_mult={args.lr_decay_min_mult})"
        )
    else:
        schedule = optax.cosine_decay_schedule(
            args.lr,
            args.episodes * args.minibatches,
            args.lr_decay_min_mult,
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, eps=args.adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    # ---------------- Helpers used inside MCTS / rollout ----------------
    def _pad_to_unified(logits, valid_size):
        out = jnp.full((UNIFIED_ACTION_SIZE,), -1e9, dtype=logits.dtype)
        return out.at[:valid_size].set(logits)

    def _initial_vertex_avail(state):
        return vertex_avail_at_step(
            state, vertex_valid_static, total_v, num_valid,
        ).astype(jnp.float32)

    def _build_action_mask_at_depth(emb: DecisionEmbedding, pin_rules):
        """Per-depth invalid-action mask (1 where invalid)."""
        depth = emb.depth
        v_invalid = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        v_invalid = v_invalid.at[:total_v].set(1.0 - emb.vertex_avail_mask)

        # Pair-slot mask. When pin_rules is on, force PAIR_STOP for every
        # pair slot (mu0's analogue of ppo.py's pin_rules_to_exact).
        p_invalid_unpinned = jnp.full(
            (UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32,
        )
        p_invalid_unpinned = p_invalid_unpinned.at[:NUM_PAIR_CHOICES].set(
            1.0 - pair_valid_mask[emb.vertex_idx],
        )
        p_invalid_pinned = jnp.ones((UNIFIED_ACTION_SIZE,), dtype=jnp.float32)
        p_invalid_pinned = p_invalid_pinned.at[PAIR_STOP].set(0.0)
        p_invalid = jnp.where(pin_rules, p_invalid_pinned, p_invalid_unpinned)

        factor_slot_k = (depth - 2) // 2
        pair_k = emb.pair_seq[jnp.maximum(factor_slot_k, 0)]
        f_invalid_full = jnp.full(
            (UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32,
        )
        f_invalid_full = f_invalid_full.at[:num_factors].set(
            1.0 - pair_factor_mask[emb.vertex_idx, pair_k],
        )

        is_vertex = depth == 0
        is_pair = (depth >= 1) & (depth % 2 == 1)
        return jnp.where(
            is_vertex, v_invalid,
            jnp.where(is_pair, p_invalid, f_invalid_full),
        )

    def _prior_at_depth(emb: DecisionEmbedding, pin_rules):
        logits, value = agent.prediction(emb.latent)
        invalid = _build_action_mask_at_depth(emb, pin_rules)
        masked = jnp.where(invalid > 0.5, -1e9, logits)
        return masked, value

    def _build_step_action(vertex_idx, pair_seq, factor_seq):
        target_vertex = jnp.asarray(vertex_idx + 1, dtype=jnp.int32)
        first_rows = []
        for slot in range(max_rules):
            p = pair_seq[slot]
            f_idx = factor_seq[slot]
            base = _PAIR_TO_BASE[p]
            is_stop = p == PAIR_STOP
            factor = jnp.where(is_stop, 0, factor_table[f_idx]).astype(jnp.int32)
            row = jnp.concatenate([base, factor[None]]).astype(jnp.int32)
            first_rows.append(row)
        if max_rules >= MAX_RULES_PER_VERTEX:
            specs = jnp.stack(first_rows[:MAX_RULES_PER_VERTEX], axis=0)
        else:
            stacked = jnp.stack(first_rows, axis=0)
            pad = jnp.tile(
                jnp.array([-1, -1, 0], dtype=jnp.int32),
                (MAX_RULES_PER_VERTEX - max_rules, 1),
            )
            specs = jnp.concatenate([stacked, pad], axis=0)
        return StepAction(target_vertex=target_vertex, rule_specs=specs)

    # ---------------- mctx callbacks ----------------
    #
    # mctx expects both ``root_fn`` and ``recurrent_fn`` to operate on
    # batched embeddings: ``root_fn`` is called once by the user with the
    # batched root, and ``mctx._src.search.expand`` slices the search-tree
    # embedding by ``(batch_range, parent_index)`` before calling
    # ``recurrent_fn`` (see search.py around line 222). Both callables
    # must return batched outputs.
    #
    # The agent's MLP heads operate on 1-D latents, and ``lax.cond``
    # requires scalar predicates, so we write the single-element logic
    # once and wrap each callable in ``jax.vmap`` over the batch axis.

    def make_root_fn(pin_rules):
        def root_fn_single(emb):
            return _prior_at_depth(emb, pin_rules)

        def root_fn(_agent, _rng_key, embedding):
            prior, value = jax.vmap(root_fn_single)(embedding)
            return mctx.RootFnOutput(
                prior_logits=prior, value=value, embedding=embedding,
            )
        return root_fn

    def make_recurrent_fn(pin_rules):
        def recurrent_step_single(action, embedding):
            depth = embedding.depth
            is_vertex = depth == 0
            is_pair = (depth >= 1) & (depth % 2 == 1)
            is_factor = (depth >= 2) & (depth % 2 == 0)
            pair_slot_k = (depth - 1) // 2
            factor_slot_k = (depth - 2) // 2

            new_vertex_idx = jnp.where(
                is_vertex, action.astype(jnp.int32), embedding.vertex_idx,
            )
            new_pair_seq = lax.cond(
                is_pair,
                lambda: embedding.pair_seq.at[
                    jnp.maximum(pair_slot_k, 0)
                ].set(action.astype(jnp.int32)),
                lambda: embedding.pair_seq,
            )
            new_factor_seq = lax.cond(
                is_factor,
                lambda: embedding.factor_seq.at[
                    jnp.maximum(factor_slot_k, 0)
                ].set(action.astype(jnp.int32)),
                lambda: embedding.factor_seq,
            )
            new_active = jnp.where(
                is_pair,
                embedding.active & (action.astype(jnp.int32) != PAIR_STOP),
                embedding.active,
            )

            next_latent, pred_reward = agent.dynamics(
                embedding.latent, action.astype(jnp.int32),
            )

            new_depth = depth + 1
            should_commit = new_depth == DECISION_DEPTH

            def commit_branch():
                new_avail = embedding.vertex_avail_mask.at[new_vertex_idx].set(0.0)
                fresh = _empty_decision(next_latent, new_avail, max_rules)
                return fresh._replace(last_reward=pred_reward)

            def no_commit_branch():
                return embedding._replace(
                    latent=next_latent,
                    depth=new_depth,
                    vertex_idx=new_vertex_idx,
                    pair_seq=new_pair_seq,
                    factor_seq=new_factor_seq,
                    active=new_active,
                    last_reward=pred_reward,
                )

            next_emb = lax.cond(should_commit, commit_branch, no_commit_branch)
            n_prior, n_value = _prior_at_depth(next_emb, pin_rules)

            return (n_prior, n_value, next_emb)

        def recurrent_fn(_agent, _rng_key, action, embedding):
            n_prior, n_value, next_emb = jax.vmap(recurrent_step_single)(
                action, embedding,
            )
            return (
                mctx.RecurrentFnOutput(
                    reward=next_emb.last_reward,
                    discount=jnp.full_like(
                        next_emb.last_reward, args.discount,
                    ),
                    prior_logits=n_prior,
                    value=n_value,
                ),
                next_emb,
            )
        return recurrent_fn

    # ---------------- Rollout ----------------
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, None, None))
    def rollout_fn(
        agent,
        temperature,
        env_obj,
        env_state,
        key,
        preference,
        vertex_features,
        pin_rules_jax,
    ):
        keys = jrand.split(key, rollout_length)

        # Cache-encoding: encode the initial state once. The cached latent
        # is the leading latent at every real step; the dynamics network
        # alone evolves it across real eliminations. The non-cache path
        # re-runs ``representation`` per step using the *current* env
        # state's tokens / eqn_ids.
        encode_key, scan_key = jrand.split(keys[0], 2)
        cached_latent = agent.representation(
            env_state.tokens,
            eqn_ids=env_state.eqn_ids,
            preference=preference,
            vertex_features=vertex_features,
            key=encode_key,
            agg_key=encode_key,
        )

        root_fn = make_root_fn(pin_rules_jax)
        recurrent_fn = make_recurrent_fn(pin_rules_jax)

        def _run_search(decision_state, search_key, traverse_key):
            """Dispatch the MCTS search based on ``args.mcts_mode``.

            Returns ``(mcts_visits, actions_path, mcts_value)`` with shapes
            ``((DECISION_DEPTH, UNIFIED_ACTION_SIZE), (DECISION_DEPTH,),
            ())``. ``args.mcts_mode`` is static at trace time so the
            Python branching here selects between the three search
            backends without runtime dispatch overhead.
            """
            embedding = jax.tree.map(
                lambda x: jnp.expand_dims(x, 0), decision_state,
            )
            roots = root_fn(agent, search_key, embedding)
            invalid_actions = _build_action_mask_at_depth(
                decision_state, pin_rules_jax,
            )

            if args.mcts_mode == "gumbel":
                # Full hierarchical tree, Gumbel sequential halving.
                policy_output = mctx.gumbel_muzero_policy(
                    params=agent,
                    rng_key=search_key,
                    root=roots,
                    recurrent_fn=recurrent_fn,
                    num_simulations=args.num_simulations,
                    invalid_actions=invalid_actions[None, :],
                    max_num_considered_actions=args.gumbel_max_considered,
                    gumbel_scale=args.gumbel_scale,
                )
                mcts_value = policy_output.search_tree.summary().value[0]
                mcts_visits, actions_path = extract_path_visits(
                    policy_output.search_tree, DECISION_DEPTH, traverse_key,
                )
                return mcts_visits, actions_path, mcts_value

            if args.mcts_mode == "sampled":
                # Sampled MuZero (root-only approximation): draw K actions
                # from the prior softmax (after masking invalid actions)
                # and let mctx.muzero_policy expand only those K.
                sample_key, search_inner = jrand.split(search_key)
                root_prior_masked = jnp.where(
                    invalid_actions > 0.5,
                    jnp.full_like(roots.prior_logits[0], -1e9),
                    roots.prior_logits[0],
                )
                sampled = jrand.categorical(
                    sample_key, root_prior_masked,
                    shape=(args.sampled_k,),
                )
                sample_keep = jnp.zeros(
                    (UNIFIED_ACTION_SIZE,), dtype=jnp.float32,
                ).at[sampled].set(1.0)
                sampled_invalid = jnp.maximum(
                    invalid_actions,
                    jnp.where(sample_keep > 0.5, 0.0, 1.0),
                )
                policy_output = mctx.muzero_policy(
                    params=agent,
                    rng_key=search_inner,
                    root=roots,
                    recurrent_fn=recurrent_fn,
                    num_simulations=args.num_simulations,
                    invalid_actions=sampled_invalid[None, :],
                    dirichlet_fraction=args.dirichlet_fraction,
                    dirichlet_alpha=args.dirichlet_alpha,
                    temperature=temperature,
                )
                mcts_value = policy_output.search_tree.summary().value[0]
                mcts_visits, actions_path = extract_path_visits(
                    policy_output.search_tree, DECISION_DEPTH, traverse_key,
                )
                return mcts_visits, actions_path, mcts_value

            # args.mcts_mode == "hierarchical":
            # Stage 1: shallow PUCT MCTS over the vertex head.
            v_search_key, r_search_key = jrand.split(search_key)
            r_traverse_key, _ = jrand.split(traverse_key)
            v_policy = mctx.muzero_policy(
                params=agent,
                rng_key=v_search_key,
                root=roots,
                recurrent_fn=recurrent_fn,
                num_simulations=args.num_simulations,
                invalid_actions=invalid_actions[None, :],
                max_depth=1,
                dirichlet_fraction=args.dirichlet_fraction,
                dirichlet_alpha=args.dirichlet_alpha,
                temperature=temperature,
            )
            v_visits_raw = v_policy.search_tree.summary().visit_counts[0]
            v_visits_sum = jnp.sum(v_visits_raw)
            v_visits = jnp.where(
                v_visits_sum > 0,
                v_visits_raw / jnp.maximum(v_visits_sum, 1e-8),
                jnp.full_like(v_visits_raw, 1.0 / v_visits_raw.shape[-1]),
            )
            v_value = v_policy.search_tree.summary().value[0]
            vertex_idx = v_policy.action[0].astype(jnp.int32)

            # Stage 2: Gumbel MCTS over the rule sub-sequence rooted at
            # the post-vertex-dynamics latent (depth=1).
            post_v_latent, _ = agent.dynamics(decision_state.latent, vertex_idx)
            new_avail = decision_state.vertex_avail_mask.at[vertex_idx].set(0.0)
            sub_decision = decision_state._replace(
                latent=post_v_latent,
                vertex_avail_mask=new_avail,
                depth=jnp.array(1, dtype=jnp.int32),
                vertex_idx=vertex_idx,
            )
            sub_embedding = jax.tree.map(
                lambda x: jnp.expand_dims(x, 0), sub_decision,
            )
            sub_roots = root_fn(agent, r_search_key, sub_embedding)
            sub_invalid = _build_action_mask_at_depth(
                sub_decision, pin_rules_jax,
            )
            sub_policy = mctx.gumbel_muzero_policy(
                params=agent,
                rng_key=r_search_key,
                root=sub_roots,
                recurrent_fn=recurrent_fn,
                num_simulations=args.num_simulations,
                invalid_actions=sub_invalid[None, :],
                max_num_considered_actions=args.gumbel_max_considered,
                gumbel_scale=args.gumbel_scale,
            )
            sub_visits, sub_actions = extract_path_visits(
                sub_policy.search_tree, DECISION_DEPTH - 1, r_traverse_key,
            )
            sub_value = sub_policy.search_tree.summary().value[0]

            combined_visits = jnp.concatenate(
                [v_visits[None, :], sub_visits], axis=0,
            )
            combined_actions = jnp.concatenate(
                [vertex_idx[None], sub_actions.astype(jnp.int32)], axis=0,
            )
            combined_value = 0.5 * (v_value + sub_value)
            return combined_visits, combined_actions, combined_value

        def step_fn(carry, k):
            state, evolved_latent = carry
            search_key, traverse_key, encode_key = jrand.split(k, 3)

            if args.cache_encoding:
                latent = evolved_latent
            else:
                latent = agent.representation(
                    state.tokens,
                    eqn_ids=state.eqn_ids,
                    preference=preference,
                    vertex_features=vertex_features,
                    key=encode_key,
                    agg_key=encode_key,
                )

            v_avail = _initial_vertex_avail(state)
            decision_state = _empty_decision(latent, v_avail, max_rules)

            mcts_visits, actions_path, mcts_value = _run_search(
                decision_state, search_key, traverse_key,
            )

            vertex_idx = actions_path[0]
            slot_indices = jnp.arange(max_rules)
            pair_seq = actions_path[1 + 2 * slot_indices].astype(jnp.int32)
            factor_seq = actions_path[2 + 2 * slot_indices].astype(jnp.int32)

            env_action = _build_step_action(vertex_idx, pair_seq, factor_seq)
            env_out = env_obj.step(state, env_action)
            scalar_reward = jnp.sum(env_out.reward * reward_weights)

            # Evolve the cached latent through the just-taken action sequence
            # so the next step starts from a latent that reflects what we did.
            # Each real step costs 1 + 2 · max_rules dynamics calls; the same
            # call count the search tree pays per simulation, so this is cheap.
            latent_v, _ = agent.dynamics(
                latent, vertex_idx.astype(jnp.int32),
            )

            def per_slot(carry, slot_idx):
                lat, _ = agent.dynamics(carry, pair_seq[slot_idx])
                lat, _ = agent.dynamics(lat, factor_seq[slot_idx])
                return lat, None

            next_latent_cached, _ = lax.scan(
                per_slot, latent_v, jnp.arange(max_rules),
            )

            transition = Trajectory(
                tokens=state.tokens.astype(jnp.int32),
                eqn_ids=state.eqn_ids.astype(jnp.int32),
                vertex_idx=vertex_idx,
                pair_seq=pair_seq,
                factor_seq=factor_seq,
                reward_vec=env_out.reward,
                scalar_reward=scalar_reward.astype(jnp.float32),
                mcts_visits=mcts_visits.astype(jnp.float32),
                mcts_value=jnp.asarray(mcts_value, dtype=jnp.float32),
                preference=preference.astype(jnp.float32),
            )
            return (env_out.state, next_latent_cached), transition

        (final_state, _), traj = lax.scan(
            step_fn, (env_state, cached_latent), keys,
        )
        return final_state, traj

    # ---------------- Loss ----------------
    def _build_action_seq(vertex_idx, pair_seq, factor_seq):
        out = jnp.zeros((DECISION_DEPTH,), dtype=jnp.int32)
        out = out.at[0].set(vertex_idx)
        for slot in range(max_rules):
            out = out.at[1 + 2 * slot].set(pair_seq[slot])
            out = out.at[2 + 2 * slot].set(factor_seq[slot])
        return out

    def _build_reward_seq(scalar_reward):
        out = jnp.zeros((DECISION_DEPTH,), dtype=jnp.float32)
        return out.at[DECISION_DEPTH - 1].set(scalar_reward)

    def loss_fn(agent, batch: TrajectoryWindow):
        def unroll_loss(window: TrajectoryWindow):
            tokens = window.tokens[0]
            eqn_ids = window.eqn_ids[0]
            preference = window.preference[0]
            latent = agent.representation(
                tokens, eqn_ids=eqn_ids, preference=preference,
            )

            l_pi = jnp.array(0.0, dtype=jnp.float32)
            l_v = jnp.array(0.0, dtype=jnp.float32)
            l_r = jnp.array(0.0, dtype=jnp.float32)

            for k in range(args.unroll_steps + 1):
                action_seq_k = _build_action_seq(
                    window.vertex_idx[k], window.pair_seq[k], window.factor_seq[k],
                )
                reward_seq_k = _build_reward_seq(window.scalar_reward[k])
                target_value_k = window.target_value[k]
                mcts_visits_k = window.mcts_visits[k]

                for d in range(DECISION_DEPTH):
                    logits, value = agent.prediction(latent)
                    target_d = mcts_visits_k[d]
                    l_pi = l_pi + (
                        -jnp.sum(target_d * jnn.log_softmax(logits))
                    )
                    l_v = l_v + 0.5 * jnp.square(value - target_value_k)

                    is_very_last = (
                        (k == args.unroll_steps) and (d == DECISION_DEPTH - 1)
                    )
                    if not is_very_last:
                        latent, pred_reward = agent.dynamics(
                            latent, action_seq_k[d],
                        )
                        l_r = l_r + 0.5 * jnp.square(
                            pred_reward - reward_seq_k[d],
                        )
                        if d == DECISION_DEPTH - 1:
                            latent = 0.5 * latent + 0.5 * lax.stop_gradient(latent)

            total = (
                l_pi
                + args.value_loss_weight * l_v
                + args.reward_loss_weight * l_r
            )
            return total, (l_pi, l_v, l_r)

        per_window_total, (lp, lv, lr) = jax.vmap(unroll_loss)(batch)
        return jnp.mean(per_window_total), (
            jnp.mean(lp), jnp.mean(lv), jnp.mean(lr),
        )

    @eqx.filter_jit
    def train_minibatch(agent, opt_state, batch):
        (loss_val, parts), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True,
        )(agent, batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(agent, eqx.is_inexact_array),
        )
        agent = eqx.apply_updates(agent, updates)
        return agent, opt_state, loss_val, parts

    # ---------------- Reporting ----------------
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else args.wandb,
    )
    elim_order_table = wandb.Table(
        columns=["episode", "return", "elimination order"],
    )
    pbar = tqdm(total=args.episodes)

    samplecounts = 0
    best_global_return = -float("inf")
    best_global_act_seq: list | None = None
    replay_buffer = None
    _resume_pending = bool(
        args.replay_checkpoint_path
        and os.path.exists(args.replay_checkpoint_path)
    )

    # ---------------- Stage G pre-training reward calibration ----------------
    # Mirrors ppo.py: ``--calibrate-steps`` un-trained rollouts measure
    # ``mean_abs(symlog(reward))`` per channel and rescale
    # ``reward_weights`` by the inverse so each weighted channel
    # contributes on a comparable scale. cosine_sim is excluded (already
    # bounded). Skipping the block leaves the raw lambdas in place.
    if args.calibrate_steps > 0:
        print(
            f"\nPre-training scale calibration: {args.calibrate_steps} rollouts "
            "(measuring |symlog(reward)| per channel under the initial policy)",
            flush=True,
        )
        abs_sum = np.zeros((NUM_REWARDS,), dtype=np.float32)
        cal_pin_rules = jnp.asarray(args.pin_rules_to_exact, dtype=jnp.bool_)
        for step_idx in range(args.calibrate_steps):
            cal_key, key = jrand.split(key)
            cal_eval_key, cal_rollout_key = jrand.split(cal_key)
            cal_rollout_keys = jrand.split(cal_rollout_key, num_envs)
            cal_eval_samples = generate_eval_samples(
                env, cal_eval_key, args.num_eval_samples,
            )
            cal_env = eqx.tree_at(
                lambda e: e.eval_args_samples, env, cal_eval_samples,
            )
            cal_vfeat = _episode_vertex_features(
                args,
                closed_jaxpr.jaxpr,
                tuple(closed_jaxpr.literals),
                tuple(xs),
                eval_samples=cal_eval_samples,
                argnums=tuple(argnums),
            )
            cal_env_states = reset_envs(cal_env)
            cal_uniform_pref = jnp.broadcast_to(
                jnp.full((NUM_REWARDS,), 1.0 / NUM_REWARDS, dtype=jnp.float32),
                (num_envs, NUM_REWARDS),
            )
            _, cal_traj = rollout_fn(
                agent,
                jnp.asarray(args.temperature, dtype=jnp.float32),
                cal_env,
                cal_env_states,
                cal_rollout_keys,
                cal_uniform_pref,
                cal_vfeat,
                cal_pin_rules,
            )
            sl_per_step = _symlog_rewards(cal_traj.reward_vec)
            mean_abs = np.asarray(jnp.mean(jnp.abs(sl_per_step), axis=(0, 1)))
            abs_sum = abs_sum + mean_abs
            print(
                f"  scale cal step {step_idx:3d}/{args.calibrate_steps}  "
                + "  ".join(
                    f"{REWARD_NAMES[i]}={mean_abs[i]:.2e}"
                    for i in range(NUM_REWARDS)
                    if mean_abs[i] > 0.0 or reward_weights_np[i] != 0.0
                ),
                flush=True,
            )
        mean_abs_final = abs_sum / args.calibrate_steps
        reward_scales_np = np.where(
            _NO_SYMLOG_MASK_NP, 1.0, 1.0 / (mean_abs_final + 1e-3)
        ).astype(np.float32)
        reward_weights_np = (
            reward_weights_np * reward_scales_np
        ).astype(np.float32)
        reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
        print(
            "calibrated reward_weights: "
            + ", ".join(
                f"{REWARD_NAMES[i]}={reward_weights_np[i]:+.3g}"
                for i in range(NUM_REWARDS)
                if reward_weights_np[i] != 0.0
            ),
            flush=True,
        )

    # ---------------- Training loop ----------------
    for ep in range(args.episodes):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        rollout_key, ep_key = jrand.split(ep_key)
        rollout_keys = jrand.split(rollout_key, num_envs)

        # Curriculum stage resolution
        if curriculum_stages:
            current_stage_name = _current_stage_at(curriculum_stages, ep)
            current_stage_idx = _current_stage_index(curriculum_stages, ep)
            stage_pin_rules = jnp.asarray(
                _pin_rules_for_variant(current_stage_name), dtype=jnp.bool_,
            )
            if ep == 0 or (
                ep > 0
                and _current_stage_at(curriculum_stages, ep - 1)
                != current_stage_name
            ):
                print(
                    f"[ep {ep}] curriculum stage → {current_stage_name}  "
                    f"(pin_rules={bool(stage_pin_rules)})"
                )
        else:
            stage_pin_rules = jnp.asarray(
                args.pin_rules_to_exact, dtype=jnp.bool_,
            )

        if args.num_eval_samples > 0:
            eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
            env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
        else:
            eval_samples = None
            env_episode = env

        vertex_features = _episode_vertex_features(
            args,
            closed_jaxpr.jaxpr,
            tuple(closed_jaxpr.literals),
            tuple(xs),
            eval_samples=eval_samples,
            argnums=tuple(argnums),
        )

        env_states = reset_envs(env_episode)
        progress = ep / max(args.episodes - 1, 1)
        temperature = jnp.asarray(
            schedule_at(
                progress, args.temperature, args.temperature_final,
                args.temperature_schedule,
            ),
            dtype=jnp.float32,
        )

        pref_key, ep_key = jrand.split(ep_key)
        if args.preference_conditioned:
            preferences = sample_preferences(
                pref_key, NUM_REWARDS, num_envs,
                dirichlet_alpha=args.preference_dirichlet_alpha,
            )
        else:
            preferences = jnp.zeros((num_envs, NUM_REWARDS), dtype=jnp.float32)

        _, fresh_traj = rollout_fn(
            agent, temperature, env_episode, env_states, rollout_keys,
            preferences, vertex_features, stage_pin_rules,
        )

        # ---------------- Lagrangian augmentation of scalar reward ----------------
        # Per-step augmentation: for each constraint i, add
        # ``-lambda_i * max(0, sign_i * (threshold_i - reward_vec[idx_i]))``
        # to the scalar reward used for the value target. Mirrors ppo.py.
        if constraint_specs:
            r = fresh_traj.reward_vec  # (E, T, NUM_REWARDS)
            # Per-constraint violation: positive when reward is on the wrong
            # side of the threshold.
            picked = r[..., constraint_indices]  # (E, T, C)
            viol = jnn.relu(
                constraint_signs * (constraint_thresholds - picked)
            )  # (E, T, C)
            augment = -jnp.sum(multipliers[None, None, :] * viol, axis=-1)
            scalar_with_lag = fresh_traj.scalar_reward + augment
            train_scalar = scalar_with_lag
        else:
            viol = None
            train_scalar = fresh_traj.scalar_reward

        # ---------------- Replay (lazy init, sample-and-mix) ----------------
        if args.replay_buffer_size > 0 and replay_buffer is None:
            replay_buffer = init_replay_buffer(
                jax.tree_util.tree_map(lambda x: x[0], fresh_traj),
                args.replay_buffer_size,
            )
            if _resume_pending:
                replay_buffer = load_replay_buffer(
                    args.replay_checkpoint_path, replay_buffer,
                )
                print(
                    f"[replay] resumed buffer from "
                    f"{args.replay_checkpoint_path} "
                    f"(size={int(replay_buffer.size)}/"
                    f"{replay_buffer.capacity})"
                )
                _resume_pending = False

        sample_key, ep_key = jrand.split(ep_key)
        if (
            args.replay_buffer_size > 0
            and replay_buffer is not None
            and ep >= args.replay_warmup
        ):
            target_batch_size = (
                args.replay_batch_size if args.replay_batch_size > 0
                else num_envs
            )
            n_fresh = min(
                int(target_batch_size * args.replay_fresh_fraction), num_envs,
            )
            n_replay = target_batch_size - n_fresh
            if n_replay > 0:
                replay_traj = replay_sample(
                    replay_buffer, n_replay, sample_key,
                    alpha=args.replay_priority_alpha,
                )
                if n_fresh > 0:
                    fresh_part = jax.tree_util.tree_map(
                        lambda x: x[:n_fresh], fresh_traj,
                    )
                    train_traj = jax.tree_util.tree_map(
                        lambda f, r: jnp.concatenate([f, r], axis=0),
                        fresh_part, replay_traj,
                    )
                else:
                    train_traj = replay_traj
            else:
                train_traj = jax.tree_util.tree_map(
                    lambda x: x[:target_batch_size], fresh_traj,
                )
        else:
            train_traj = fresh_traj

        if constraint_specs:
            # Re-augment the train scalar in case replay mixed in stored
            # trajectories whose scalar_reward predates the current
            # multipliers — keep it simple and recompute over train_traj's
            # full reward_vec.
            r_train = train_traj.reward_vec
            picked_train = r_train[..., constraint_indices]
            viol_train = jnn.relu(
                constraint_signs * (constraint_thresholds - picked_train)
            )
            augment_train = -jnp.sum(
                multipliers[None, None, :] * viol_train, axis=-1,
            )
            train_scalar_t = train_traj.scalar_reward + augment_train
        else:
            train_scalar_t = train_traj.scalar_reward

        discounted = jax.vmap(
            lambda r: _discounted_returns(r, args.discount)
        )(train_scalar_t)
        target_values = discounted

        windows = []
        for i in range(rollout_length - args.unroll_steps):
            sl = lambda x: x[:, i : i + args.unroll_steps + 1]
            windows.append(
                TrajectoryWindow(
                    tokens=sl(train_traj.tokens),
                    eqn_ids=sl(train_traj.eqn_ids),
                    vertex_idx=sl(train_traj.vertex_idx),
                    pair_seq=sl(train_traj.pair_seq),
                    factor_seq=sl(train_traj.factor_seq),
                    scalar_reward=sl(train_scalar_t),
                    target_value=sl(target_values),
                    mcts_visits=sl(train_traj.mcts_visits),
                    preference=sl(train_traj.preference),
                )
            )
        window_batch = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=1), *windows,
        )

        shuffle_key, ep_key = jrand.split(ep_key)
        batches = _shuffle_and_batch_windows(
            window_batch, args.minibatches, shuffle_key,
        )

        last_loss = None
        last_parts = None
        for i in range(args.minibatches):
            mb = jax.tree_util.tree_map(lambda x: x[i], batches)
            agent, opt_state, last_loss, last_parts = train_minibatch(
                agent, opt_state, mb,
            )
        p_loss, v_loss, r_loss = (float(x) for x in last_parts)

        # ---------------- Lagrangian dual ascent ----------------
        if constraint_specs:
            mean_viol = jnp.mean(viol, axis=(0, 1))  # (C,)
            multipliers = jnp.maximum(
                multipliers + args.lagrangian_lr * mean_viol, 0.0,
            )

        if args.replay_buffer_size > 0 and replay_buffer is not None:
            traj_priorities = _compute_traj_priorities(fresh_traj, reward_weights)
            replay_buffer = replay_add_batch(
                replay_buffer, fresh_traj, priorities=traj_priorities,
            )
            if (
                args.replay_checkpoint_path
                and (ep + 1) % args.replay_checkpoint_every == 0
            ):
                save_replay_buffer(
                    replay_buffer, args.replay_checkpoint_path,
                )

        episode_reward_vec = jnp.sum(fresh_traj.reward_vec, axis=1)
        episode_total = jnp.sum(episode_reward_vec * reward_weights, axis=-1)

        max_idx = int(jnp.argmax(episode_total))
        best_reward = float(episode_total[max_idx])
        best_seq = _action_to_pylist(
            np.asarray(fresh_traj.vertex_idx[max_idx]),
            np.asarray(fresh_traj.pair_seq[max_idx]),
            np.asarray(fresh_traj.factor_seq[max_idx]),
            factor_table_np,
        )
        if best_reward > best_global_return:
            best_global_return = best_reward
            best_global_act_seq = best_seq
            elim_order_table.add_data(ep, best_reward, str(best_seq))

        samplecounts += num_envs * rollout_length
        log_dict = {
            "best_return": best_global_return,
            "mean_return": float(jnp.mean(episode_total)),
            "policy loss": p_loss,
            "value loss": v_loss,
            "reward loss": r_loss,
            "total loss": float(last_loss),
            "sample count": samplecounts,
        }
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(jnp.mean(episode_reward_vec[:, j]))
        if constraint_specs:
            lam_np = np.asarray(multipliers)
            viol_np = np.asarray(jnp.mean(viol, axis=(0, 1)))
            for j, (idx, t, sign) in enumerate(constraint_specs):
                op_str = ">=" if sign > 0 else "<="
                name = f"{REWARD_NAMES[idx]}{op_str}{t:g}"
                log_dict[f"lagrangian/{name}_lambda"] = float(lam_np[j])
                log_dict[f"lagrangian/{name}_violation"] = float(viol_np[j])
        wandb.log(log_dict)

        pbar.update(1)
        pbar.set_description(
            f"best:{best_reward:.1f} mean:{float(jnp.mean(episode_total)):.1f} "
            f"pi:{p_loss:.3f} v:{v_loss:.3f} r:{r_loss:.3f}"
        )

    pbar.close()
    wandb.log({"Elimination order": elim_order_table})
    if best_global_act_seq is not None:
        print(f"\nBest elimination order (return={best_global_return:.2f}):")
        print(best_global_act_seq)


if __name__ == "__main__":
    main()
