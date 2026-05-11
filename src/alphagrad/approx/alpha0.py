"""AlphaZero trainer for the vertex-elimination env — hierarchical MCTS.

Each "real" elimination step is decomposed into ``1 + 2 · max_rules`` MCTS
depth levels:

  * depth 0           — vertex selection
  * depth 2k+1        — pair selection at rule-slot k (k = 0..max_rules-1)
  * depth 2k+2        — factor selection at rule-slot k
  * depth 1+2·max_rules — commit: env.step, residual_state update, reset

The MCTS embedding carries ``(env_state, residual_state, depth, vertex_idx,
pair_seq, factor_seq, decoder_h, active)``; ``decoder_h`` is PPO's
:class:`AutoregRulePolicy` carry, recomputed at each pair depth via
``decoder.step`` and stored for the next factor depth.

Action space is unified to ``UNIFIED = max(total_v, NPC, num_factors)`` so
mctx sees a single fixed-size flat space. Per-depth masks zero out invalid
actions; mctx ignores them.

Loss split (deliberate trade-off — see DESIGN NOTE below):
  * vertex head trains on cross-entropy with the MCTS root visit counts
    (the standard AlphaZero distillation signal).
  * rule heads train on log-likelihood of the autoregressive ``(pair_seq,
    factor_seq)`` actually sampled in the rollout. Visit-count distillation
    at depth ≥ 1 would require traversing the mctx search tree per chosen
    action — defer until the simpler signal is shown to be insufficient.
  * value head: per-component MSE against per-component cumulative returns.

DESIGN NOTE: rule sampling happens *outside* mctx, via the agent's autoreg
prior, after the vertex is committed. The hierarchical MCTS still simulates
rule choices internally (recurrent_fn descends through every rule depth to
the commit leaf where it gets the true env reward) — that's where the
search benefit accrues to the *vertex* visit counts, since the simulated
rule choices inform the value backups. We don't currently consume the
deeper-depth visit counts; sampling from the agent prior keeps the rollout
fast and doesn't require tree traversal in JAX.

Encoder cache (``--cache-encoding``) and ``--terminal-rewards-only``
continue to work; their plumbing is unchanged from the previous flat-
action version.
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
    schedule_at,
    shuffle_and_batch,
    shuffle_and_batch_by_trajectory,
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
from alphagrad.approx.ppo import (
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    _build_agent,
    _build_factor_table,
    _episode_vertex_features,
    _scale_output_heads,
    _setup_jax_compile_cache,
    _stop_only_logits,
)


# ---------------------------------------------------------------------------
# Action layout helpers
# ---------------------------------------------------------------------------


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
# Reward weighting
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


def _build_reward_weights(args) -> np.ndarray:
    weights = np.zeros(NUM_REWARDS, dtype=np.float32)
    if "cmp" in args.rewards:
        weights[REWARD_INDEX[_CMP_TYPE_TO_REWARD[args.cmp_type]]] = args.lambda_cmp
    if "mem" in args.rewards:
        weights[REWARD_INDEX[_MEM_TYPE_TO_REWARD[args.mem_type]]] = args.lambda_mem
    if "acc" in args.rewards:
        weights[REWARD_INDEX["cosine_sim"]] = 1.0
    if args.lambda_frob != 0.0:
        weights[REWARD_INDEX["frob_residual"]] = args.lambda_frob
    return weights


# ---------------------------------------------------------------------------
# Hierarchical MCTS embedding + decision-state helpers
# ---------------------------------------------------------------------------


class DecisionEmbedding(NamedTuple):
    """State inside the hierarchical MCTS tree.

    ``depth`` ranges over [0, 1 + 2·max_rules); the prior at each depth is
    one of ``{vertex, pair_k, factor_k}`` per the layout described in the
    module docstring.
    """

    env_state: object              # alphagrad EnvState pytree
    residual_state: jax.Array      # (total_v, embd_dim)
    depth: jax.Array               # scalar int32
    vertex_idx: jax.Array          # scalar int32 (valid for depth >= 1)
    pair_seq: jax.Array            # (max_rules,) int32
    factor_seq: jax.Array          # (max_rules,) int32
    decoder_h: jax.Array           # (embd_dim,) float32 — autoreg carry
    active: jax.Array              # bool — true while still adding rules
    last_reward: jax.Array         # scalar — accumulated reward to deliver
                                   # on the *next* recurrent_fn call (so
                                   # mctx receives the env reward at the
                                   # post-commit transition).


def _empty_decision(
    env_state, residual_state, max_rules: int, embd_dim: int
) -> DecisionEmbedding:
    return DecisionEmbedding(
        env_state=env_state,
        residual_state=residual_state,
        depth=jnp.array(0, dtype=jnp.int32),
        vertex_idx=jnp.array(0, dtype=jnp.int32),
        pair_seq=jnp.full((max_rules,), PAIR_STOP, dtype=jnp.int32),
        factor_seq=jnp.zeros((max_rules,), dtype=jnp.int32),
        decoder_h=jnp.zeros((embd_dim,), dtype=jnp.float32),
        active=jnp.array(True, dtype=jnp.bool_),
        last_reward=jnp.array(0.0, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Trajectory layout — one row per real elimination step
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    tokens: jax.Array              # (T, MAX_TOKENS) int32
    eqn_ids: jax.Array             # (T, MAX_TOKENS) int32
    residual_state: jax.Array      # (T, total_v, embd_dim) float32
    vertex_idx: jax.Array          # (T,) int32
    pair_seq: jax.Array            # (T, max_rules) int32
    factor_seq: jax.Array          # (T, max_rules) int32
    reward_vec: jax.Array          # (T, NUM_REWARDS) — overwritten with
                                   # per-component cumulative returns at
                                   # train time.
    scalar_reward: jax.Array       # (T,) — dot(reward_vec, weights)
    mcts_visits: jax.Array         # (T, DECISION_DEPTH, UNIFIED) — per-depth
                                   # visit-count distributions along the
                                   # chosen path through the search tree.
    preference: jax.Array          # (T, NUM_REWARDS) — per-env preference,
                                   # broadcast to every step (constant
                                   # within an episode).


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AlphaZero trainer for vertex-elimination (hierarchical MCTS)."
    )
    p.add_argument("--name", type=str, default="AlphaZero_Vertex")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument("--wandb", type=str, default="disabled",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Parallel rollout envs. -1 = os.cpu_count().")
    p.add_argument("--num-simulations", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--no-jit", action="store_true")

    # Network architecture (PPO-shaped).
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=64)
    p.add_argument("--op-embd-dim", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")
    p.add_argument("--head-init-scale", type=float, default=0.1)

    # Agent variant. Hierarchical MCTS uses ``use_autoreg=True`` so we get
    # the ``AutoregRulePolicy.decoder.step`` per-slot interface for free.
    p.add_argument("--no-ptr", action="store_true",
                   help="Use the MLP vertex policy instead of the pointer head.")
    p.add_argument("--max-rules", type=int, default=1,
                   help="Rule-slot count per chosen vertex. With --max-rules=1 "
                        "the MCTS has 3 depths (vertex, pair, factor); larger "
                        "values give 1 + 2·max_rules depths per elimination.")
    p.add_argument("--factors", type=str, default="-1,1,2,4",
                   help="Comma-separated factor values for the rule decoder. "
                        "Pinning to '-1' recovers the legacy single-factor "
                        "behaviour.")
    p.add_argument("--sparsity-ratio", action="store_true")
    p.add_argument("--rho-prior-bias", type=float, default=4.0)
    p.add_argument("--set-transformer-agg", action="store_true")

    # Reward selection.
    p.add_argument("--cmp-type", type=str, default="flops",
                   choices=["graphax", "flops", "latency"])
    p.add_argument("--mem-type", type=str, default="peak_memory",
                   choices=["graphax", "bytes_accessed", "peak_memory"])
    p.add_argument("--rewards", nargs="+", type=str,
                   default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"])
    p.add_argument("--lambda-cmp", type=float, default=1.0)
    p.add_argument("--lambda-mem", type=float, default=1.0)
    p.add_argument("--lambda-frob", type=float, default=0.0)
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--exec-on-gpu", action="store_true")
    p.add_argument("--terminal-rewards-only", action="store_true",
                   help="Compute the env's reward vector only at the final "
                        "elimination step; intermediate steps return zeros.")

    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)

    # MCTS visit-count temperature schedule.
    p.add_argument("--temperature-init", type=float, default=1.0)
    p.add_argument("--temperature-final", type=float, default=0.1)
    p.add_argument("--temperature-schedule", type=str, default="constant",
                   choices=SCHEDULES)
    p.add_argument("--dirichlet-fraction", type=float, default=0.25)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)

    p.add_argument("--cache-encoding", action="store_true",
                   help="Reuse the encoder output across the rollout via "
                        "encode-once + residual_state.")

    # Self-play replay buffer. AlphaZero-style: each episode appends fresh
    # rollouts; gradient steps sample from the buffer (uniformly here —
    # prioritised replay is left as a follow-up). 0 disables.
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
    # Prioritised sampling: weight = max(priority, eps)**alpha.
    p.add_argument("--replay-priority-alpha", type=float, default=0.0,
                   help="Power applied to per-slot priorities at sample "
                        "time. 0 = uniform.")
    # Disk checkpointing for the buffer.
    p.add_argument("--replay-checkpoint-path", type=str, default="",
                   help="If set, the buffer is saved here every "
                        "--replay-checkpoint-every episodes. Loaded at "
                        "startup if the file exists.")
    p.add_argument("--replay-checkpoint-every", type=int, default=10)
    # Preference conditioning. Each env in the rollout draws a fresh
    # ``w ∈ Δ^{NUM_REWARDS-1}`` per episode; PPO's ``Agent.pref_proj`` adds
    # ``w → embd_dim`` into the per-vertex contexts and value summary.
    p.add_argument("--preference-conditioned", action="store_true",
                   help="Sample a per-env Dirichlet preference each "
                        "episode and condition the policy/value on it.")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0)
    return p


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

    if args.cache_encoding:
        envs_per_mb = (
            (args.num_envs if args.num_envs > 0 else (os.cpu_count() or 64))
            // args.minibatches
        )
        if envs_per_mb < 1:
            raise ValueError(
                "--cache-encoding requires num_envs >= --minibatches; got "
                f"num_envs={args.num_envs} minibatches={args.minibatches}."
            )

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    # ---------------- Env ----------------
    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call, dataset_size=args.dataset_size)
    argnums = infer_argnums(args.example)

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    measure_latency = args.measure_latency or args.cmp_type == "latency"
    env_target_fun = target_fn if "acc" in args.rewards else None
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

    vertex_valid_static = build_vertex_valid_static(env.valid_vertices, total_v)
    pair_valid_mask = build_pair_valid_mask(
        closed_jaxpr.jaxpr,
        total_v,
        num_pair_choices=NUM_PAIR_CHOICES,
        pair_stop_idx=PAIR_STOP,
    )

    use_pointer = not args.no_ptr
    use_autoreg = True  # hierarchical MCTS leans on AutoregRulePolicy
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(
        args, use_autoreg,
    )
    factor_table_np = np.array(factors_py, dtype=np.int32)
    # All-ones placeholder — graphax's apply_diag validates factor
    # divisibility at apply time, so the legacy pre-mask is redundant.
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    DECISION_DEPTH = 1 + 2 * max_rules
    UNIFIED_ACTION_SIZE = int(max(total_v, NUM_PAIR_CHOICES, num_factors))

    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)

    EPISODES = args.episodes
    NUM_ENVS = args.num_envs if args.num_envs > 0 else (os.cpu_count() or 64)
    MINIBATCHES = args.minibatches
    NUM_SIMULATIONS = args.num_simulations
    ROLLOUT_LENGTH = num_valid

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"num_envs={NUM_ENVS}, max_rules={max_rules}, "
        f"factors={factors_py}, decision_depth={DECISION_DEPTH}, "
        f"unified_action_size={UNIFIED_ACTION_SIZE}"
    )
    nonzero_w = ", ".join(
        f"{REWARD_NAMES[i]}={float(reward_weights_np[i]):+.3g}"
        for i in range(NUM_REWARDS)
        if reward_weights_np[i] != 0.0
    )
    print(f"reward weights: {nonzero_w or '<all zero — debug only>'}")

    # ---------------- Agent ----------------
    agent_key, init_key, key = jrand.split(key, 3)
    agent = _build_agent(
        use_pointer, use_autoreg, args, total_v, num_factors, max_rules, agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(
        agent, args.head_init_scale, use_pointer, use_autoreg,
    )

    # ---------------- Rollout helpers ----------------
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(NUM_ENVS))

    def get_vertex_avail(state):
        return vertex_avail_at_step(
            state, vertex_valid_static, total_v, num_valid,
        )

    def _pad_to_unified(logits, valid_size):
        """Embed an action-specific logit vector inside ``UNIFIED_ACTION_SIZE``.

        Slots beyond ``valid_size`` get a strongly-negative sentinel so mctx
        treats them as invalid.
        """
        out = jnp.full((UNIFIED_ACTION_SIZE,), -1e9, dtype=logits.dtype)
        return out.at[:valid_size].set(logits)

    # ---------------- Hierarchical MCTS ----------------
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0, None, 0))
    def rollout_fn(agent, temperature, env_state, key, vertex_features, preference):
        keys = jrand.split(key, ROLLOUT_LENGTH)
        init_residual = jnp.zeros((total_v, args.embd_dim), dtype=jnp.float32)

        # Cache the initial residual jaxpr's encoding once per env.
        encode_key, scan_key = jrand.split(keys[0], 2)
        if args.cache_encoding:
            cached_encoding = agent.encode_once(
                env_state.tokens, eqn_ids=env_state.eqn_ids, key=encode_key,
            )
        else:
            cached_encoding = None
        # Trajectory tokens carry the initial reference when caching.
        traj_init_tokens = (
            env_state.tokens if args.cache_encoding else None
        )
        traj_init_eqn_ids = (
            env_state.eqn_ids if args.cache_encoding else None
        )

        def _decode(state, residual_state):
            """Return ``(vertex_logits, vertex_contexts, value_vec)``.

            ``preference`` is closed over from the rollout's per-env axis;
            PPO's encode / _decode_from_cache add ``pref_proj(w)`` to the
            per-vertex contexts and summary so the entire MCTS sub-tree
            sees the preference.
            """
            if args.cache_encoding:
                return agent._decode_from_cache(
                    cached_encoding, vertex_features, residual_state,
                    preference=preference,
                )
            return agent.encode(
                state.tokens, eqn_ids=state.eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                preference=preference, key=jrand.PRNGKey(0),
            )

        def _vertex_prior(state, residual):
            vlogits, _, value_vec = _decode(state, residual)
            scalar_value = jnp.sum(value_vec * reward_weights)
            vertex_avail = get_vertex_avail(state)
            vlogits_masked = jnp.where(vertex_avail > 0.5, vlogits, -1e9)
            return _pad_to_unified(vlogits_masked, total_v), scalar_value

        def _pair_prior_and_h(
            state, residual, vertex_idx, slot_k, decoder_h,
            prev_pair, prev_factor, active,
        ):
            _, vctx, value_vec = _decode(state, residual)
            scalar_value = jnp.sum(value_vec * reward_weights)
            v_context = vctx[vertex_idx]
            v_pair_mask = pair_valid_mask[vertex_idx]

            new_h, pair_logits = agent.rule_policy.decoder.step(
                v_context, slot_k, prev_pair, prev_factor, decoder_h,
            )
            pair_logits = jnp.where(v_pair_mask > 0.5, pair_logits, -1e9)
            stop_only = _stop_only_logits(NUM_PAIR_CHOICES)
            pair_logits = jnp.where(active, pair_logits, stop_only)
            return _pad_to_unified(pair_logits, NUM_PAIR_CHOICES), scalar_value, new_h

        def _factor_prior(
            state, residual, vertex_idx, decoder_h, pair_k,
        ):
            _, _, value_vec = _decode(state, residual)
            scalar_value = jnp.sum(value_vec * reward_weights)
            v_factor_mask = pair_factor_mask[vertex_idx, pair_k]
            factor_logits = agent.rule_policy.decoder.factor_logits_for(
                decoder_h, pair_k,
            )
            factor_logits = jnp.where(v_factor_mask > 0.5, factor_logits, -1e9)
            return _pad_to_unified(factor_logits, num_factors), scalar_value

        def _prior_at_depth(emb: DecisionEmbedding):
            """Dispatch by ``emb.depth`` to one of the three branch-priors.

            Computes all three branches and selects with ``jnp.where`` —
            JIT fuses the unused arms to no-ops at trace time.
            """
            depth = emb.depth
            is_vertex = depth == 0
            is_pair = (depth >= 1) & (depth % 2 == 1)

            pair_slot_k = (depth - 1) // 2
            factor_slot_k = (depth - 2) // 2
            prev_pair = jnp.where(
                pair_slot_k > 0, emb.pair_seq[jnp.maximum(pair_slot_k - 1, 0)],
                jnp.array(PAIR_STOP, dtype=jnp.int32),
            )
            prev_factor = jnp.where(
                pair_slot_k > 0, emb.factor_seq[jnp.maximum(pair_slot_k - 1, 0)],
                jnp.array(0, dtype=jnp.int32),
            )

            v_prior, v_value = _vertex_prior(emb.env_state, emb.residual_state)
            p_prior, p_value, p_new_h = _pair_prior_and_h(
                emb.env_state, emb.residual_state, emb.vertex_idx,
                pair_slot_k, emb.decoder_h, prev_pair, prev_factor, emb.active,
            )
            f_prior, f_value = _factor_prior(
                emb.env_state, emb.residual_state, emb.vertex_idx,
                emb.decoder_h, emb.pair_seq[jnp.maximum(factor_slot_k, 0)],
            )

            prior = jnp.where(
                is_vertex, v_prior,
                jnp.where(is_pair, p_prior, f_prior),
            )
            value = jnp.where(
                is_vertex, v_value,
                jnp.where(is_pair, p_value, f_value),
            )
            new_h = jnp.where(is_pair, p_new_h, emb.decoder_h)
            return prior, value, new_h

        def _build_step_action(vertex_idx, pair_seq, factor_seq):
            """Construct an env :class:`StepAction` from the decided sequence."""
            target_vertex = jnp.asarray(vertex_idx + 1, dtype=jnp.int32)
            specs = agent.rule_policy.to_env_specs(
                pair_seq, factor_seq, factor_table,
            )
            return StepAction(target_vertex=target_vertex, rule_specs=specs)

        def _commit(emb: DecisionEmbedding):
            """At depth == DECISION_DEPTH-1 + post-action: apply env step,
            evolve residual_state, reset depth=0 for the next vertex."""
            env_action = _build_step_action(
                emb.vertex_idx, emb.pair_seq, emb.factor_seq,
            )
            env_out = env.step(emb.env_state, env_action)
            scalar_reward = jnp.sum(env_out.reward * reward_weights)
            _, vctx, _ = _decode(emb.env_state, emb.residual_state)
            new_residual = agent.update_residual(
                emb.residual_state, emb.vertex_idx, vctx[emb.vertex_idx],
            )
            return env_out.state, new_residual, scalar_reward, env_out.reward

        # ----- mctx callbacks -----

        def root_fn(_agent, _rng_key, embedding):
            prior, value, _ = _prior_at_depth(embedding)
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

            # Update embedding fields based on what the action just chose.
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

            # Recompute decoder_h on a pair step (cheap; mirrors the prior
            # path which already produced the same ``new_h``).
            def update_h():
                _, vctx, _ = _decode(
                    embedding.env_state, embedding.residual_state,
                )
                v_context = vctx[new_vertex_idx]
                prev_pair = jnp.where(
                    pair_slot_k > 0,
                    embedding.pair_seq[jnp.maximum(pair_slot_k - 1, 0)],
                    jnp.array(PAIR_STOP, dtype=jnp.int32),
                )
                prev_factor = jnp.where(
                    pair_slot_k > 0,
                    embedding.factor_seq[jnp.maximum(pair_slot_k - 1, 0)],
                    jnp.array(0, dtype=jnp.int32),
                )
                new_h, _ = agent.rule_policy.decoder.step(
                    v_context, pair_slot_k, prev_pair, prev_factor,
                    embedding.decoder_h,
                )
                return new_h

            new_decoder_h = lax.cond(
                is_pair, update_h, lambda: embedding.decoder_h,
            )

            new_depth = embedding.depth + 1
            should_commit = new_depth == DECISION_DEPTH

            # If we just chose the final factor, commit: env.step + residual.
            def commit_branch():
                interim = embedding._replace(
                    vertex_idx=new_vertex_idx,
                    pair_seq=new_pair_seq,
                    factor_seq=new_factor_seq,
                    decoder_h=new_decoder_h,
                    active=new_active,
                )
                next_state, next_residual, scalar_reward, _ = _commit(interim)
                # Reset decision state for the next vertex.
                fresh = _empty_decision(
                    next_state, next_residual, max_rules, args.embd_dim,
                )
                fresh = fresh._replace(
                    last_reward=scalar_reward,
                )
                return fresh

            def no_commit_branch():
                return embedding._replace(
                    depth=new_depth,
                    vertex_idx=new_vertex_idx,
                    pair_seq=new_pair_seq,
                    factor_seq=new_factor_seq,
                    decoder_h=new_decoder_h,
                    active=new_active,
                    last_reward=jnp.array(0.0, dtype=jnp.float32),
                )

            next_emb = lax.cond(should_commit, commit_branch, no_commit_branch)

            # Compute next prior + value at the *new* depth.
            n_prior, n_value, _ = _prior_at_depth(next_emb)

            return (
                mctx.RecurrentFnOutput(
                    reward=next_emb.last_reward,
                    discount=jnp.ones_like(next_emb.last_reward),
                    prior_logits=n_prior,
                    value=n_value,
                ),
                next_emb,
            )

        # ----- per-step rollout (one real elimination = many MCTS depths) -----

        def step_fn(carry, k):
            state, residual_state = carry
            search_key, traverse_key = jrand.split(k, 2)

            # Initial decision state (depth 0).
            decision_state = _empty_decision(
                state, residual_state, max_rules, args.embd_dim,
            )

            # The hierarchical MCTS is *one* search run rooted at depth=0.
            # mctx walks the tree by calling root_fn once and recurrent_fn
            # `num_simulations` times; the embedding internally advances
            # through every depth and commits at depth==DECISION_DEPTH.
            embedding = jax.tree.map(
                lambda x: jnp.expand_dims(x, 0), decision_state,
            )
            roots = root_fn(agent, search_key, embedding)
            invalid_actions = jnp.zeros(
                (1, UNIFIED_ACTION_SIZE,), dtype=jnp.float32,
            )
            invalid_actions = invalid_actions.at[0, :].set(
                1.0 - (
                    jnp.arange(UNIFIED_ACTION_SIZE) < total_v
                ).astype(jnp.float32),
            )
            # Combine with vertex availability — the root mask is depth-0.
            v_avail = get_vertex_avail(state)
            invalid_actions = invalid_actions.at[
                0, :total_v
            ].set(1.0 - v_avail.astype(jnp.float32))

            policy_output = mctx.muzero_policy(
                params=agent,
                rng_key=search_key,
                root=roots,
                recurrent_fn=recurrent_fn,
                num_simulations=NUM_SIMULATIONS,
                invalid_actions=invalid_actions,
                dirichlet_fraction=args.dirichlet_fraction,
                dirichlet_alpha=args.dirichlet_alpha,
                temperature=temperature,
            )

            # Walk the search tree along the chosen action path: at every
            # depth take the visit-count distribution as policy target and
            # sample the next action from it. Replaces the previous "sample
            # vertex from root visits, sample rules from agent prior" pattern
            # — every decision in the rollout is now MCTS-informed.
            mcts_visits, actions_path = extract_path_visits(
                policy_output.search_tree, DECISION_DEPTH, traverse_key,
            )

            # Decode the unified action path into env-style components.
            vertex_idx = actions_path[0]
            slot_indices = jnp.arange(max_rules)
            pair_seq = actions_path[1 + 2 * slot_indices].astype(jnp.int32)
            factor_seq = actions_path[2 + 2 * slot_indices].astype(jnp.int32)

            # Look up the chosen vertex's context for the residual update;
            # it's the same vertex_contexts the policy saw at depth 0.
            _, vctx, _ = _decode(state, residual_state)

            # Apply env step + evolve residual.
            env_action = _build_step_action(vertex_idx, pair_seq, factor_seq)
            env_out = env.step(state, env_action)
            scalar_reward = jnp.sum(env_out.reward * reward_weights)
            new_residual = agent.update_residual(
                residual_state, vertex_idx, vctx[vertex_idx],
            )

            transition = Trajectory(
                tokens=(
                    traj_init_tokens if args.cache_encoding else state.tokens
                ).astype(jnp.int32),
                eqn_ids=(
                    traj_init_eqn_ids if args.cache_encoding else state.eqn_ids
                ).astype(jnp.int32),
                residual_state=residual_state,
                vertex_idx=vertex_idx,
                pair_seq=pair_seq,
                factor_seq=factor_seq,
                reward_vec=env_out.reward,
                scalar_reward=scalar_reward.astype(jnp.float32),
                mcts_visits=mcts_visits.astype(jnp.float32),
                preference=preference.astype(jnp.float32),
            )
            return (env_out.state, new_residual), transition

        carry = (env_state, init_residual)
        _, traj = lax.scan(step_fn, carry, keys)
        return traj

    # ---------------- Loss ----------------
    schedule = optax.cosine_decay_schedule(args.lr, EPISODES, 0.0)
    optimizer = optax.chain(
        optax.adamw(schedule, eps=1e-7),
        optax.clip_by_global_norm(1.0),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    def loss_fn(agent, batch: Trajectory, vertex_features, key):
        # Shape dispatch — same pattern as the previous version.
        if batch.tokens.ndim == 3:
            E, K = batch.tokens.shape[:2]
            enc_key, _ = jrand.split(key, 2)
            cached_per_traj = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens[:, 0], batch.eqn_ids[:, 0], jrand.split(enc_key, E))
            batch_flat = jax.tree_util.tree_map(
                lambda x: x.reshape(E * K, *x.shape[2:]), batch,
            )
            cached_flat = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, K, axis=0), cached_per_traj,
            )

            def per_sample(cached, residual_state, vertex_idx, pair_seq, factor_seq, preference):
                vlogits, vctx, value = agent._decode_from_cache(
                    cached, vertex_features, residual_state,
                    preference=preference,
                )
                v_context = vctx[vertex_idx]
                v_pair_mask = pair_valid_mask[vertex_idx]
                v_factor_mask = pair_factor_mask[vertex_idx]
                # ``evaluate`` returns the per-slot pair/factor *distributions*
                # — used as the policy term for visit-count CE at depths ≥ 1.
                _, _, _, _, pair_dists, factor_dists = agent.rule_policy.evaluate(
                    v_context, v_pair_mask, v_factor_mask, pair_seq, factor_seq,
                )
                vlogits_padded = _pad_to_unified(vlogits, total_v)
                return vlogits_padded, value, pair_dists, factor_dists

            vlogits_b, value_b, pair_dists_b, factor_dists_b = jax.vmap(per_sample)(
                cached_flat, batch_flat.residual_state,
                batch_flat.vertex_idx, batch_flat.pair_seq, batch_flat.factor_seq,
                batch_flat.preference,
            )
            target_visits = batch_flat.mcts_visits        # (B, DECISION_DEPTH, UNIFIED)
            target_returns = batch_flat.reward_vec
        else:
            B = batch.tokens.shape[0]
            enc_key, _ = jrand.split(key, 2)
            enc_keys = jrand.split(enc_key, B)

            def per_sample_full(toks, eids, rs, vidx, pseq, fseq, pref, k):
                vlogits, vctx, value = agent.encode(
                    toks, eqn_ids=eids, vertex_features=vertex_features,
                    residual_state=rs, preference=pref, key=k,
                )
                v_context = vctx[vidx]
                v_pair_mask = pair_valid_mask[vidx]
                v_factor_mask = pair_factor_mask[vidx]
                _, _, _, _, pair_dists, factor_dists = agent.rule_policy.evaluate(
                    v_context, v_pair_mask, v_factor_mask, pseq, fseq,
                )
                vlogits_padded = _pad_to_unified(vlogits, total_v)
                return vlogits_padded, value, pair_dists, factor_dists

            vlogits_b, value_b, pair_dists_b, factor_dists_b = jax.vmap(
                per_sample_full,
            )(
                batch.tokens, batch.eqn_ids, batch.residual_state,
                batch.vertex_idx, batch.pair_seq, batch.factor_seq,
                batch.preference, enc_keys,
            )
            target_visits = batch.mcts_visits             # (B, DECISION_DEPTH, UNIFIED)
            target_returns = batch.reward_vec

        # Vertex CE — depth 0 of the per-depth visit-count tensor. Visits
        # beyond ``total_v`` are zero by construction (the rollout's mask
        # zeros them in the search tree), so the cross-entropy is well-
        # defined even though ``vlogits_b`` is padded with ``-1e9`` past
        # ``total_v``.
        log_pi_v = jnn.log_softmax(vlogits_b, axis=-1)
        target_v = target_visits[:, 0, :]
        p_loss_vertex = jnp.mean(-jnp.sum(target_v * log_pi_v, axis=-1))

        # Per-slot pair CE — odd depths 1, 3, ..., 2·max_rules-1.
        # ``pair_dists_b`` is already softmax-normalised by AutoregRulePolicy;
        # take ``log`` directly for the CE log-policy term.
        log_pair = jnp.log(pair_dists_b + 1e-8)
        target_pairs = target_visits[:, 1::2, :NUM_PAIR_CHOICES]
        ce_pair = -jnp.sum(target_pairs * log_pair, axis=-1)        # (B, max_rules)
        p_loss_pairs = jnp.mean(jnp.sum(ce_pair, axis=-1))

        # Per-slot factor CE — even depths 2, 4, ..., 2·max_rules.
        log_factor = jnp.log(factor_dists_b + 1e-8)
        target_factors = target_visits[:, 2::2, :num_factors]
        ce_factor = -jnp.sum(target_factors * log_factor, axis=-1)
        p_loss_factors = jnp.mean(jnp.sum(ce_factor, axis=-1))

        p_loss_rules = p_loss_pairs + p_loss_factors

        # Multi-reward value loss — per-component MSE summed across components.
        v_loss = jnp.mean(jnp.sum((value_b - target_returns) ** 2, axis=-1))

        return p_loss_vertex + p_loss_rules + v_loss, (
            p_loss_vertex, p_loss_rules, v_loss,
        )

    @eqx.filter_jit
    def train_minibatch(agent, opt_state, batch, vertex_features, key):
        grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
            agent, batch, vertex_features, key,
        )
        updates, opt_state = optimizer.update(grads, opt_state, agent)
        new_agent = eqx.apply_updates(agent, updates)
        return new_agent, opt_state, metrics

    # ---------------- Reporting ----------------
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else args.wandb,
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    pbar = tqdm(range(EPISODES))
    best_global_return = -float("inf")
    best_global_act_seq = None
    # Self-play replay buffer; lazily initialised on episode 0 (see below)
    # so we don't need to know the trajectory pytree shape ahead of time.
    replay_buffer = None
    _resume_pending = bool(
        args.replay_checkpoint_path
        and os.path.exists(args.replay_checkpoint_path)
    )

    def _action_seq_to_pylist(vertex_seq, pair_seq, factor_seq):
        """Decode a per-step (vertex, pair_seq, factor_seq) trace to env-style
        ``[(vertex, [(idx1, idx2, factor), ...])]`` for logging/inspection."""
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

    # ---------------- Training loop ----------------
    for episode in pbar:
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        rollout_key, ep_key = jrand.split(ep_key)
        train_key, ep_key = jrand.split(ep_key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        if args.num_eval_samples > 0:
            eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
            env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
        else:
            eval_samples = None
            env_episode = env

        vertex_features = _episode_vertex_features(
            args, closed_jaxpr.jaxpr, tuple(closed_jaxpr.literals),
            tuple(xs), eval_samples=eval_samples, argnums=tuple(argnums),
        )

        env_states = reset_envs(env_episode)
        progress = episode / max(EPISODES - 1, 1)
        temperature = jnp.asarray(
            schedule_at(
                progress, args.temperature_init, args.temperature_final,
                args.temperature_schedule,
            ),
            dtype=jnp.float32,
        )

        pref_key, ep_key = jrand.split(ep_key)
        if args.preference_conditioned:
            preferences = sample_preferences(
                pref_key, NUM_REWARDS, NUM_ENVS,
                dirichlet_alpha=args.preference_dirichlet_alpha,
            )
        else:
            preferences = jnp.zeros((NUM_ENVS, NUM_REWARDS), dtype=jnp.float32)

        fresh_traj = rollout_fn(
            agent, temperature, env_states, rollout_keys, vertex_features,
            preferences,
        )

        # Per-component cumulative return for the multi-reward value head.
        # Stored *back* into reward_vec so the loss path reads the right
        # target without changing its signature.
        returns_vec = jnp.cumsum(
            fresh_traj.reward_vec[:, ::-1, :], axis=1,
        )[:, ::-1, :]
        fresh_traj = fresh_traj._replace(reward_vec=returns_vec)

        # Replay buffer: lazy init on episode 0; from --replay-warmup onward
        # we mix in samples from past rollouts. Trajectories are stored
        # *after* returns are computed so the buffer leaves match what the
        # loss expects.
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

        sample_key, train_key = jrand.split(train_key)
        if (
            args.replay_buffer_size > 0
            and replay_buffer is not None
            and episode >= args.replay_warmup
        ):
            target_batch_size = (
                args.replay_batch_size if args.replay_batch_size > 0
                else NUM_ENVS
            )
            n_fresh = min(
                int(target_batch_size * args.replay_fresh_fraction), NUM_ENVS,
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

        shuffle_key, train_key = jrand.split(train_key)
        batches = (
            shuffle_and_batch_by_trajectory(train_traj, MINIBATCHES, shuffle_key)
            if args.cache_encoding
            else shuffle_and_batch(train_traj, MINIBATCHES, shuffle_key)
        )

        for i in range(MINIBATCHES):
            mb = jax.tree_util.tree_map(lambda x: x[i], batches)
            mb_key = jrand.fold_in(train_key, i)
            agent, opt_state, metrics = train_minibatch(
                agent, opt_state, mb, vertex_features, mb_key,
            )
        p_vertex_loss, p_rules_loss, v_loss = metrics

        # Add the fresh trajectories to the buffer *after* the gradient
        # step — that way this episode doesn't sample from itself. Priority
        # is the shifted-positive scalar episode return (taken from the
        # already-cumsumed reward_vec at t=0).
        if args.replay_buffer_size > 0 and replay_buffer is not None:
            ep_returns = jnp.sum(
                fresh_traj.reward_vec[:, 0, :] * reward_weights, axis=-1,
            )
            traj_priorities = ep_returns - jnp.min(ep_returns) + 1e-3
            replay_buffer = replay_add_batch(
                replay_buffer, fresh_traj, priorities=traj_priorities,
            )
            if (
                args.replay_checkpoint_path
                and (episode + 1) % args.replay_checkpoint_every == 0
            ):
                save_replay_buffer(
                    replay_buffer, args.replay_checkpoint_path,
                )

        # Logging uses the *fresh* rollout, not the training mix.
        ep_total = jnp.sum(returns_vec[:, 0, :] * reward_weights, axis=-1)
        max_idx = int(jnp.argmax(ep_total))
        best_reward = float(ep_total[max_idx])
        best_seq = _action_seq_to_pylist(
            np.asarray(fresh_traj.vertex_idx[max_idx]),
            np.asarray(fresh_traj.pair_seq[max_idx]),
            np.asarray(fresh_traj.factor_seq[max_idx]),
        )

        if best_reward > best_global_return:
            best_global_return = best_reward
            best_global_act_seq = best_seq
            elim_order_table.add_data(episode, best_reward, str(best_seq))

        log_dict = {
            "best_return": best_reward,
            "mean_return": float(jnp.mean(ep_total)),
            "policy loss (vertex)": float(p_vertex_loss),
            "policy loss (rules)": float(p_rules_loss),
            "value loss": float(v_loss),
        }
        per_component_means = jnp.mean(returns_vec[:, 0, :], axis=0)
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(per_component_means[j])
        wandb.log(log_dict)

        pbar.set_description(
            f"best: {best_reward:.1f}, mean: {float(jnp.mean(ep_total)):.1f}"
        )

    wandb.log({"Elimination order": elim_order_table})
    if best_global_act_seq is not None:
        print(
            f"\nBest elimination order (return={best_global_return:.2f}):\n"
            f"{best_global_act_seq}"
        )


if __name__ == "__main__":
    main()
