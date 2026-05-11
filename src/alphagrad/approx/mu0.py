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

Loss unroll (per window of ``UNROLL_STEPS + 1`` real steps):
  * For each real step k and each decision depth d:
      - ``logits, value = prediction(latent)``
      - vertex (d=0):   CE with MCTS root visit counts at step k
      - rule (d ≥ 1):   log-likelihood of the *sampled* (pair_k, factor_k)
      - value loss:     ``(value − cumulative_return_k)²`` (target the
                        same across all d within real step k)
      - dynamics + reward: ``latent, pred_reward = dynamics(latent, a_d)``
                        with reward target ``0`` for d < DECISION_DEPTH-1
                        and ``scalar_env_reward_k`` at the last decision
  * Half-gradient ``latent = 0.5·latent + 0.5·sg(latent)`` at every real
    step boundary — preserves the original mu0 semantics while letting
    the per-step dynamics be unrolled fully.

DESIGN NOTE: rule-head training uses sampled-action LL (not visit counts).
Extracting per-depth visit counts from the mctx search tree adds tree
traversal in JAX for marginal gain at this stage; defer.
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
    SCHEDULES,
    build_pair_valid_mask,
    build_vertex_valid_static,
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
    sample_preferences,
    save_replay_buffer,
    scale_module_weight,
    schedule_at,
    vertex_avail_at_step,
)
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
from alphagrad.transformer import MLP, Encoder, PositionalEncoder


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


# ---------------------------------------------------------------------------
# Factor-table helpers (mirrors PPO's _build_factor_table)
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _build_factor_table(args) -> tuple[jax.Array, tuple[int, ...], int, int]:
    """Construct the factor table from ``--factors``. Returns
    ``(factor_table, factors_py, num_factors, max_rules)``.

    For mu0's hierarchical layout, ``num_factors == len(factors_py)`` and
    ``max_rules`` comes straight from CLI.
    """
    factors_py = tuple(_parse_int_list(args.factors))
    if not factors_py:
        raise ValueError("--factors must list at least one factor value.")
    factor_table = jnp.array(factors_py, dtype=jnp.int32)
    return factor_table, factors_py, factor_table.shape[0], int(args.max_rules)


# ---------------------------------------------------------------------------
# Hierarchical MCTS embedding
# ---------------------------------------------------------------------------


class DecisionEmbedding(NamedTuple):
    """State inside the hierarchical MCTS tree.

    ``vertex_avail_mask`` is the only non-latent piece of "real" state we
    carry — it tracks which vertices have been eliminated in the simulated
    path so the next vertex selection inside the tree masks them out.
    """

    latent: jax.Array              # (latent_dim,)
    vertex_avail_mask: jax.Array   # (total_v,) float32 — 1 = available
    depth: jax.Array               # scalar int32 in [0, DECISION_DEPTH)
    vertex_idx: jax.Array          # scalar int32 — selected vertex
    pair_seq: jax.Array            # (max_rules,) int32
    factor_seq: jax.Array          # (max_rules,) int32
    active: jax.Array              # bool — true while still adding rules
    last_reward: jax.Array         # scalar — reward delivered to mctx on
                                   # the *next* recurrent_fn call (so the
                                   # search backs up exactly the dynamics-
                                   # predicted reward at each transition).


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
# MuZero agent — unchanged from before except that ``num_actions`` now equals
# the unified action space (max(total_v, NPC, num_factors)).
# ---------------------------------------------------------------------------


class MuZeroAgent(eqx.Module):
    """Encoder → latent representation; MLP dynamics; policy/value/reward heads.

    The dynamics is shared across all decision types — it learns to
    interpret a unified-space action id from context the latent has built
    up over prior decisions.
    """

    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder

    action_embedding: eqx.nn.Embedding
    dynamics_mlp: MLP
    reward_head: MLP

    policy_head: MLP
    value_head: MLP

    # Stage F: per-episode preference projection. Adds ``w → latent_dim``
    # to the initial latent so the policy/value/dynamics graph below
    # conditions on the preference. Disabled by default — when called
    # without ``preference=...`` the projection is unused. Initialised to
    # zero by ``init_linear_weights`` in main()'s setup, so a fresh model
    # behaves identically to the unconditioned baseline at step 0.
    pref_proj: eqx.nn.Linear

    num_actions: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)

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
        seq_len,
        key,
    ):
        keys = jrand.split(key, 8)
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.num_rewards = num_rewards
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

    def representation(self, tokens, eqn_ids=None, *, key=None, preference=None):
        token_mask = (tokens != 0)
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        enc_key = key if key is not None else jrand.PRNGKey(0)
        x = self.encoder(x, eqn_ids=eqn_ids, key=enc_key)
        mask = token_mask[..., None].astype(x.dtype)
        latent = jnp.sum(x * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-9)
        # Inject the preference once at the leading latent. Subsequent
        # dynamics/prediction calls operate on this latent, so the
        # information propagates through the rest of the search tree
        # without needing per-call plumbing.
        if preference is not None:
            latent = latent + self.pref_proj(preference)
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
                                   # broadcast to every step (constant
                                   # within an episode).


class TrajectoryWindow(NamedTuple):
    """A window of length ``UNROLL_STEPS + 1`` real steps, used by the loss.

    Mirrors :class:`Trajectory` field-for-field — the loss reads the leading
    step's tokens for ``representation``, then unrolls dynamics over the
    rest of the window (each real step decomposes into DECISION_DEPTH
    dynamics calls).
    """

    tokens: jax.Array
    eqn_ids: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    scalar_reward: jax.Array
    target_value: jax.Array
    mcts_visits: jax.Array       # (W+1, DECISION_DEPTH, UNIFIED)
    preference: jax.Array        # (W+1, NUM_REWARDS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MuZero trainer for vertex-elimination (hierarchical MCTS).",
    )
    # Run / logging
    p.add_argument("--name", type=str, default="approx-muzero")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument("--wandb", type=str, default="offline",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")

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
    p.add_argument("--lambda-frob", type=float, default=0.0)
    p.add_argument("--measure-latency", action="store_true",
                   help="Run the compiled approx fn 10x per env step to populate "
                        "the latency reward component.")
    p.add_argument("--terminal-rewards-only", action="store_true",
                   help="Compute the env's reward vector only at the final "
                        "elimination step; intermediate steps return zeros.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # Network architecture
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")

    # Hierarchical action layout
    p.add_argument("--max-rules", type=int, default=1,
                   help="Rule-slot count per chosen vertex. With max_rules=1 "
                        "the MCTS has 3 depths (vertex, pair, factor); larger "
                        "values give 1 + 2·max_rules depths per elimination.")
    p.add_argument("--factors", type=str, default="-1,1,2,4",
                   help="Comma-separated factor values for the rule decoder.")

    # MuZero
    p.add_argument("--num-simulations", type=int, default=25)
    p.add_argument("--unroll-steps", type=int, default=2)
    p.add_argument("--dirichlet-fraction", type=float, default=0.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Initial MCTS visit-count temperature.")
    p.add_argument("--temperature-final", type=float, default=0.1)
    p.add_argument("--temperature-schedule", type=str, default="constant",
                   choices=SCHEDULES)
    p.add_argument("--reward-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)

    # Optimisation
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Parallel rollout envs. -1 = os.cpu_count().")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    p.add_argument("--discount", type=float, default=1.0,
                   help="Per-step discount factor inside MCTS / for value targets.")
    p.add_argument("--head-init-scale", type=float, default=0.1)

    # MuZero-style replay buffer (paper uses prioritised replay; we start
    # with uniform here, prioritised left as a follow-up).
    p.add_argument("--replay-buffer-size", type=int, default=0,
                   help="Replay buffer capacity (number of stored trajectories). "
                        "0 disables — train only on fresh rollouts.")
    p.add_argument("--replay-batch-size", type=int, default=0,
                   help="Trajectories sampled per training pass. 0 = use --num-envs.")
    p.add_argument("--replay-warmup", type=int, default=1,
                   help="Episodes (= rollouts) to fill before sampling.")
    p.add_argument("--replay-fresh-fraction", type=float, default=0.0,
                   help="Fraction of train batch from fresh rollout. 0.0 = "
                        "pure replay; 1.0 disables replay sampling.")
    # Prioritised sampling: weight = max(priority, eps)**alpha. The priority
    # for each stored trajectory is its scalar episode return (normalised
    # to be non-negative — see ``_compute_priorities``). alpha=0 reduces to
    # uniform; alpha=1 is fully proportional. Mid-range (0.5–0.7) is the
    # Schaul et al. 2016 default.
    p.add_argument("--replay-priority-alpha", type=float, default=0.0,
                   help="Power applied to per-slot priorities at sample "
                        "time. 0 = uniform.")
    # Disk checkpointing for the buffer. Useful for long runs that span
    # multiple processes / restarts.
    p.add_argument("--replay-checkpoint-path", type=str, default="",
                   help="If set, the buffer is saved here every "
                        "--replay-checkpoint-every episodes. Loaded at "
                        "startup if the file exists.")
    p.add_argument("--replay-checkpoint-every", type=int, default=10,
                   help="Episodes between buffer checkpoints.")
    # Preference conditioning. When on, each env in the rollout draws a
    # fresh ``w ∈ Δ^{NUM_REWARDS-1}`` per episode; the agent's
    # representation network adds ``pref_proj(w)`` to the leading latent
    # so the entire MCTS sub-tree is conditioned on the preference. A
    # single trained network then covers the whole reward simplex.
    p.add_argument("--preference-conditioned", action="store_true",
                   help="Sample a per-env Dirichlet preference each "
                        "episode and condition the latent on it.")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0,
                   help="Concentration for Dirichlet(α·1). 1=uniform on "
                        "the simplex; <1 corner-concentrated; >1 centre-"
                        "concentrated.")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_num_envs(arg_value: int) -> int:
    if arg_value > 0:
        return arg_value
    return os.cpu_count() or 64


def _scale_output_heads(agent, scale: float):
    agent = scale_module_weight(agent, lambda a: a.policy_head.layers[-2].weight, scale)
    agent = scale_module_weight(agent, lambda a: a.value_head.layers[-2].weight, scale)
    agent = scale_module_weight(agent, lambda a: a.reward_head.layers[-2].weight, scale)
    return agent


def _setup_jax_compile_cache() -> None:
    cache_dir = os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jax-compilation-cache"),
    )
    try:
        from jax.experimental.compilation_cache import compilation_cache
        compilation_cache.set_cache_dir(cache_dir)
    except Exception:
        pass


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
    """Per-trajectory priority = shifted-positive episode return.

    The stored ``scalar_reward`` is per-step; sum gives the (un-discounted)
    episode return per env. We shift by the batch min so all priorities
    are non-negative — required for ``priority**alpha`` to make sense
    when ``alpha < 1`` and rewards are negative-cost (the usual sign
    convention here).
    """
    del reward_weights  # ``scalar_reward`` is already weight-collapsed.
    episode_return = jnp.sum(fresh_traj.scalar_reward, axis=1)  # (E,)
    # Shift so the worst trajectory in the batch has priority ε > 0.
    return episode_return - jnp.min(episode_return) + 1e-3


def _action_to_pylist(
    vertex_seq, pair_seq, factor_seq, factor_table_np,
) -> list[tuple[int, list]]:
    """Decode a per-step (vertex, pair_seq, factor_seq) trace to env-style
    ``[(vertex, [(idx1, idx2, factor), ...])]``."""
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


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=1)
def _shuffle_and_batch_windows(window_batch: TrajectoryWindow, minibatches: int, key):
    sample = window_batch.vertex_idx  # (E, W, U+1)
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

    env_target_fun = target_fn if "acc" in args.rewards else None
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
        measure_latency=measure_latency,
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
    # All-ones placeholder — see ppo.py for rationale (apply_diag does
    # the divisibility check at apply time now).
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    DECISION_DEPTH = 1 + 2 * max_rules
    UNIFIED_ACTION_SIZE = int(max(total_v, NUM_PAIR_CHOICES, num_factors))

    num_envs = _resolve_num_envs(args.num_envs)
    rollout_length = num_valid

    # See ppo.py for rationale: empty minibatches → NaN losses → silent no-op.
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
        seq_len=MAX_TOKENS,
        key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(args.lr, eps=args.adam_eps),
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

    def _build_action_mask_at_depth(emb: DecisionEmbedding):
        """Per-depth invalid-action mask (1 where invalid)."""
        depth = emb.depth
        # depth==0 → vertex avail mask (size total_v); pad to UNIFIED.
        v_invalid = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        v_invalid = v_invalid.at[:total_v].set(1.0 - emb.vertex_avail_mask)

        # depth 2k+1 → per-vertex pair mask, padded to UNIFIED.
        p_invalid_full = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        p_invalid_full = p_invalid_full.at[:NUM_PAIR_CHOICES].set(
            1.0 - pair_valid_mask[emb.vertex_idx],
        )

        # depth 2k+2 → per-(vertex, pair_k) factor mask.
        factor_slot_k = (depth - 2) // 2
        pair_k = emb.pair_seq[jnp.maximum(factor_slot_k, 0)]
        f_invalid_full = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        f_invalid_full = f_invalid_full.at[:num_factors].set(
            1.0 - pair_factor_mask[emb.vertex_idx, pair_k],
        )

        is_vertex = depth == 0
        is_pair = (depth >= 1) & (depth % 2 == 1)
        return jnp.where(
            is_vertex, v_invalid,
            jnp.where(is_pair, p_invalid_full, f_invalid_full),
        )

    def _prior_at_depth(emb: DecisionEmbedding):
        """Run prediction(latent) and mask the unified-size logits to the
        valid action range for the current depth."""
        logits, value = agent.prediction(emb.latent)
        invalid = _build_action_mask_at_depth(emb)
        masked = jnp.where(invalid > 0.5, -1e9, logits)
        return masked, value

    def _build_step_action(vertex_idx, pair_seq, factor_seq):
        target_vertex = jnp.asarray(vertex_idx + 1, dtype=jnp.int32)
        # Construct rule_specs via the same factor-table indexing PPO uses.
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

    def root_fn(_agent, _rng_key, embedding):
        prior, value = _prior_at_depth(embedding)
        return mctx.RootFnOutput(
            prior_logits=prior, value=value, embedding=embedding,
        )

    def recurrent_fn(_agent, _rng_key, action, embedding):
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

        # Apply the dynamics: action goes through the (shared) action
        # embedding. The latent's evolution carries the depth context — we
        # don't pass depth explicitly (kept consistent with the original
        # MuZero design; the dynamics MLP discriminates from latent alone).
        next_latent, pred_reward = agent.dynamics(embedding.latent, action.astype(jnp.int32))

        new_depth = depth + 1
        should_commit = new_depth == DECISION_DEPTH

        def commit_branch():
            # Mark the chosen vertex as eliminated for the next vertex
            # selection inside this MCTS tree, then reset the decision
            # state. Latent is the dynamics-predicted "post-elimination"
            # latent; reward predicted by the reward head IS the env reward
            # (modulo learning) — pass it forward via last_reward.
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
        n_prior, n_value = _prior_at_depth(next_emb)

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

    # ---------------- Rollout ----------------
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0))
    def rollout_fn(agent, temperature, env_obj, env_state, key, preference):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, k):
            search_key, traverse_key = jrand.split(k, 2)

            # Latent and decision-state at the root of this real step.
            # When preference-conditioning is on, the per-env preference is
            # injected into the leading latent here; the dynamics network
            # carries it through the rest of the search tree without any
            # per-call plumbing inside MCTS.
            latent = agent.representation(
                state.tokens, eqn_ids=state.eqn_ids, preference=preference,
            )
            v_avail = _initial_vertex_avail(state)
            decision_state = _empty_decision(latent, v_avail, max_rules)

            # Hierarchical MCTS: ONE search per real step. The tree internally
            # spans 1 + 2·max_rules depth levels per simulated elimination.
            embedding = jax.tree.map(
                lambda x: jnp.expand_dims(x, 0), decision_state,
            )
            roots = root_fn(agent, search_key, embedding)
            invalid_actions = _build_action_mask_at_depth(decision_state)
            policy_output = mctx.muzero_policy(
                params=agent,
                rng_key=search_key,
                root=roots,
                recurrent_fn=recurrent_fn,
                num_simulations=args.num_simulations,
                invalid_actions=invalid_actions[None, :],
                dirichlet_fraction=args.dirichlet_fraction,
                dirichlet_alpha=args.dirichlet_alpha,
                temperature=temperature,
            )
            mcts_value = policy_output.search_tree.summary().value[0]

            # Walk the search tree along the chosen action path: at every
            # depth take the visit-count distribution as policy target and
            # sample the next action from it. Replaces the previous "sample
            # vertex from root visits, sample rules from agent prior" path.
            mcts_visits, actions_path = extract_path_visits(
                policy_output.search_tree, DECISION_DEPTH, traverse_key,
            )

            # Decode the unified action path into env-style components.
            vertex_idx = actions_path[0]
            slot_indices = jnp.arange(max_rules)
            pair_seq = actions_path[1 + 2 * slot_indices].astype(jnp.int32)
            factor_seq = actions_path[2 + 2 * slot_indices].astype(jnp.int32)

            # Apply the *real* env step with the full chosen action.
            env_action = _build_step_action(vertex_idx, pair_seq, factor_seq)
            env_out = env_obj.step(state, env_action)
            scalar_reward = jnp.sum(env_out.reward * reward_weights)

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
            return env_out.state, transition

        return lax.scan(step_fn, env_state, keys)

    # ---------------- Loss ----------------
    def _build_action_seq(vertex_idx, pair_seq, factor_seq):
        """Interleave (vertex, pair_0, factor_0, pair_1, factor_1, ...)."""
        out = jnp.zeros((DECISION_DEPTH,), dtype=jnp.int32)
        out = out.at[0].set(vertex_idx)
        for slot in range(max_rules):
            out = out.at[1 + 2 * slot].set(pair_seq[slot])
            out = out.at[2 + 2 * slot].set(factor_seq[slot])
        return out

    def _build_reward_seq(scalar_reward):
        """Per-decision rewards: ``[0, 0, ..., 0, scalar_reward]``."""
        out = jnp.zeros((DECISION_DEPTH,), dtype=jnp.float32)
        return out.at[DECISION_DEPTH - 1].set(scalar_reward)

    def loss_fn(agent, batch: TrajectoryWindow):
        def unroll_loss(window: TrajectoryWindow):
            tokens = window.tokens[0]
            eqn_ids = window.eqn_ids[0]
            preference = window.preference[0]  # constant within an episode
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
                # Per-depth visit-count distributions extracted from the
                # search tree at rollout time. Shape: (DECISION_DEPTH, UNIFIED).
                mcts_visits_k = window.mcts_visits[k]

                for d in range(DECISION_DEPTH):
                    logits, value = agent.prediction(latent)
                    # Visit-count CE at every depth (paper-style AlphaZero
                    # distillation). Invalid actions have zero visit
                    # probability so they contribute zero to the sum even
                    # when ``log_softmax`` returns very-negative values for
                    # masked logits.
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
                            # End of real step k → half-grad before stepping
                            # into real step k+1.
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
    # Replay buffer; lazy-initialised on episode 0 once we know the
    # trajectory pytree shape from the first rollout. ``_resume_pending``
    # tells the loop to overwrite the buffer's leaves from the checkpoint
    # path on the first iteration after init.
    replay_buffer = None
    _resume_pending = bool(
        args.replay_checkpoint_path
        and os.path.exists(args.replay_checkpoint_path)
    )

    # ---------------- Training loop ----------------
    for ep in range(args.episodes):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        rollout_key, ep_key = jrand.split(ep_key)
        rollout_keys = jrand.split(rollout_key, num_envs)

        if args.num_eval_samples > 0:
            eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
            env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
        else:
            env_episode = env

        env_states = reset_envs(env_episode)
        progress = ep / max(args.episodes - 1, 1)
        temperature = jnp.asarray(
            schedule_at(
                progress, args.temperature, args.temperature_final,
                args.temperature_schedule,
            ),
            dtype=jnp.float32,
        )

        # Per-env preference vectors. When --preference-conditioned is
        # off, all envs see the all-zero preference (treated as "no
        # conditioning"); the trained pref_proj column for that pseudo-w
        # gets gradient signal of zero, so the model behaves like the
        # unconditioned baseline.
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
            preferences,
        )

        # Replay buffer: lazy init on episode 0; sample-and-mix from
        # --replay-warmup onward. Fresh trajectories are added at the end
        # of the episode so we never train on the rollout that just
        # produced them.
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

        # Per-real-step value target = discounted MC return. Computed on
        # the *training* trajectory (which may include replay samples).
        discounted = jax.vmap(
            lambda r: _discounted_returns(r, args.discount)
        )(train_traj.scalar_reward)
        target_values = discounted

        # Build (E, num_windows, U+1, ...) windows.
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
                    scalar_reward=sl(train_traj.scalar_reward),
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

        # Add the fresh trajectories to the buffer *after* training so the
        # current episode never samples from itself. Priorities are the
        # shifted-positive episode return — high-return trajectories get
        # re-sampled more often when ``--replay-priority-alpha > 0``.
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

        # Per-env total reward for logging — uses fresh rollout, not mix.
        episode_reward_vec = jnp.sum(fresh_traj.reward_vec, axis=1)         # (E, NR)
        episode_total = jnp.sum(episode_reward_vec * reward_weights, axis=-1)  # (E,)

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
