"""GDPO trainer — Group reward-Decoupled normalization Policy Optimization.

Faithful port of Liu et al. 2026 (arXiv:2601.05242), now riding on PPO's
autoregressive policy stack (vertex pointer + composable rule decoder) so
the GDPO advantage path reaches the same action space as PPO/GFN. The loss
remains GRPO-style:

  * Per-rollout per-reward returns:
        R_k^(j) = Σ_t r_k^(j,t)
  * Per-reward group-relative normalisation (paper Eq. 4):
        A_k^(j) = (R_k^(j) − μ_g(R_k)) / (σ_g(R_k) + ε)
  * Sum across reward dimensions (Eq. 5):
        A_sum^(j) = Σ_k A_k^(j)
  * Batch-wise renormalisation (Eq. 6):
        Â_sum^(j) = (A_sum^(j) − μ_b(A_sum)) / (σ_b(A_sum) + ε)
  * Same Â_sum^(j) applied uniformly to every step of rollout j.
  * GRPO clipped surrogate (Eq. 3) — joint log-prob ``log π(vertex, pairs,
    factors)`` slots in as a single scalar; importance ratio against the
    stored old distributions corrects the rollout-time logit temperature.

Critic-free: no value head is used in the loss. The Agent's value heads still
exist (the base PPO Agent always builds them), but their outputs are simply
ignored — the simplest port that doesn't fork the agent module.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from typing import NamedTuple

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
    SCHEDULES,
    build_pair_valid_mask,
    build_vertex_valid_static,
    data_gen,
    generate_eval_samples,
    get_args,
    get_fn,
    get_num_clipping_triggers,
    infer_argnums,
    init_linear_weights,
    sample_preferences,
    schedule_at,
    shuffle_and_batch,
    shuffle_and_batch_by_trajectory,
    vertex_avail_at_step,
)
from alphagrad.approx.env import (
    MAX_TOKENS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    VertexEliminationEnv,
)
from alphagrad.approx.ppo import (
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    _action_to_pylist,
    _build_agent,
    _build_factor_table,
    _episode_vertex_features,
    _scale_output_heads,
    _select_variant,
    _setup_jax_compile_cache,
    _variant_label,
    old_log_prob_for_action,
)


# ---------------------------------------------------------------------------
# Trajectory / TrainBatch
# ---------------------------------------------------------------------------


class Trajectory(NamedTuple):
    """One step of a rollout — fields needed both for the GDPO advantage path
    and for re-evaluating the joint log-prob under the current policy at loss
    time. Mirrors gfn.py's layout, plus the three old-policy distributions
    needed by the GRPO importance ratio."""

    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    vertex_avail_mask: jax.Array
    old_vertex_dist: jax.Array
    old_pair_dists: jax.Array
    old_factor_dists: jax.Array
    reward: jax.Array         # (NUM_REWARDS,) — full env vector each step
    preference: jax.Array     # (NUM_REWARDS,) — per-env preference,
                              # broadcast to every step.


class TrainBatch(NamedTuple):
    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    vertex_avail_mask: jax.Array
    old_vertex_dist: jax.Array
    old_pair_dists: jax.Array
    old_factor_dists: jax.Array
    advantage: jax.Array      # scalar Â_sum^(j) broadcast to every step
    preference: jax.Array     # (NUM_REWARDS,)


# ---------------------------------------------------------------------------
# Reward index helpers — GDPO selects which dims contribute to the decoupled
# advantage (paper Eq. 4 needs one A_k per dim).
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


def _build_reward_indices(args) -> np.ndarray:
    """Pick which reward components contribute to the GDPO signal.

    Each selected component gets its own per-reward group-normalised
    advantage (paper Eq. 4); they're then summed equally (Eq. 5).
    """
    indices: list[int] = []
    if "cmp" in args.rewards:
        indices.append(REWARD_INDEX[_CMP_TYPE_TO_REWARD[args.cmp_type]])
    if "mem" in args.rewards:
        indices.append(REWARD_INDEX[_MEM_TYPE_TO_REWARD[args.mem_type]])
    if "acc" in args.rewards:
        indices.append(REWARD_INDEX["cosine_sim"])
    if not indices:
        raise ValueError("--rewards must select at least one of cmp/mem/acc")
    return np.asarray(indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _resolve_num_envs(arg_value: int, example: str) -> int:
    if arg_value > 0:
        return arg_value
    if "Vmapped" in example:
        return 16
    return os.cpu_count() or 64


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "GDPO — Group reward-Decoupled normalization Policy Optimization "
            "with PPO's autoregressive policy stack."
        ),
    )
    # Run / logging
    p.add_argument("--name", type=str, default="GDPO")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument("--wandb", type=str, default="disabled",
                   choices=["disabled", "offline", "online"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--no-jit", action="store_true")
    p.add_argument("--exec-on-gpu", action="store_true")

    # Env / reward
    p.add_argument("--example", type=str, default="Helmholtz")
    p.add_argument("--disable-sparsification", action="store_true")
    p.add_argument("--disable-eval", action="store_true",
                   help="Skip the exact-Jacobian compile/exec; the cosine-sim "
                        "head is unavailable. If 'acc' is in --rewards it is "
                        "silently dropped.")
    p.add_argument("--cmp-type", type=str, default="flops",
                   choices=["graphax", "flops", "latency"])
    p.add_argument("--mem-type", type=str, default="peak_memory",
                   choices=["graphax", "bytes_accessed", "peak_memory"])
    p.add_argument("--rewards", nargs="+", type=str,
                   default=["cmp", "acc"], choices=["cmp", "mem", "acc"])
    p.add_argument("--measure-latency", action="store_true",
                   help="Run the compiled approx fn 10x per env step to populate "
                        "the latency reward component. Significantly slower; turn "
                        "on only when --cmp-type=latency.")
    p.add_argument("--terminal-rewards-only", action="store_true",
                   help="Compute the env's reward vector only at the final "
                        "elimination step; intermediate steps return zeros. "
                        "Skips per-step jacve compile/exec — the dominant "
                        "rollout cost. GDPO's per-rollout aggregation collapses "
                        "to the terminal reward, so this is the paper-native form.")
    p.add_argument("--num-eval-samples", type=int, default=0,
                   help="Calibration samples drawn from data_gen and pinned per "
                        "episode. 0 = use the env's static args. Must be >= 10 "
                        "when --measure-latency is on.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)

    # Agent variant — same independent flags as PPO. The four combinations
    # are pointer/mlp-vertex × autoreg/single-rule.
    p.add_argument("--no-ptr", action="store_true",
                   help="Use the MLP vertex policy instead of the pointer head.")
    p.add_argument("--not-autoreg", action="store_true",
                   help="Use the single-rule policy (matches the legacy GDPO "
                        "layout: one (pair, factor) per chosen vertex).")
    p.add_argument("--max-rules", type=int, default=4)
    p.add_argument("--factors", type=str, default="-1,1,2,4")
    p.add_argument("--sparsity-ratio", action="store_true")
    p.add_argument("--rho-prior-bias", type=float, default=4.0)
    p.add_argument("--set-transformer-agg", action="store_true")

    # Network architecture (PPO-shaped — _build_agent reads these)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument("--op-embd-dim", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")
    p.add_argument("--head-init-scale", type=float, default=0.1)

    # Optim — GRPO/GDPO subset of PPO knobs (no GAE-λ, no value-weight)
    p.add_argument("--num-envs", type=int, default=-1,
                   help="Group size G — number of parallel rollouts that share "
                        "the GDPO group statistics. -1 = os.cpu_count() (16 for "
                        "Vmapped examples).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ppo-clip-eps", type=float, default=0.2)
    p.add_argument("--minibatches", type=int, default=32)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--entropy-weight", type=float, default=0.0,
                   help="Optional entropy bonus on the policy. Paper omits it; "
                        "default 0 matches.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.0)
    p.add_argument("--adam-b1", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    # Rollout-time logit temperature schedule. T scales the masked vertex
    # logits before sampling; the stored old_vertex_dist is the tempered
    # distribution so the GRPO ratio in loss_fn corrects for it implicitly.
    p.add_argument("--rollout-temperature", type=float, default=1.0,
                   help="Initial vertex-logit temperature for action sampling.")
    p.add_argument("--rollout-temperature-final", type=float, default=1.0)
    p.add_argument("--rollout-temperature-schedule", type=str,
                   default="constant", choices=SCHEDULES)
    # Encoder cache: encode the initial residual jaxpr's tokens once per
    # episode (per env in the rollout, per trajectory in the loss) and reuse
    # via the residual_state path. Saves ~T-fold encoder work in both phases.
    # Requires num_envs to be divisible by --minibatches when on (loss uses
    # per-trajectory batching).
    p.add_argument("--cache-encoding", action="store_true",
                   help="Reuse the encoder output across all T steps of a "
                        "rollout via PPO's encode-once + residual_state path.")
    # Preference conditioning.
    p.add_argument("--preference-conditioned", action="store_true",
                   help="Sample a per-env Dirichlet preference each "
                        "episode and condition the policy/value on it.")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0)
    return p


# ---------------------------------------------------------------------------
# GDPO advantage (paper Eqs. 4–6)
# ---------------------------------------------------------------------------


def gdpo_advantages(per_reward_returns: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Compute the GDPO advantage from per-rollout per-reward total returns.

    Inputs
    ------
    per_reward_returns : (G, num_rewards)
        Sum-over-time of each rollout's per-reward signal — i.e. R_k^(j).

    Returns
    -------
    advantage : (G,)
        Â_sum^(j), the batch-renormalised sum of per-reward group-normalised
        advantages.

    Implements paper Eqs. 4–6 verbatim:
      Eq. 4: ``A_k^(j) = (R_k^(j) − μ_g(R_k)) / (σ_g(R_k) + ε)``
      Eq. 5: ``A_sum^(j) = Σ_k A_k^(j)``
      Eq. 6: ``Â_sum^(j) = (A_sum^(j) − μ_b(A_sum)) / (σ_b(A_sum) + ε)``
    """
    mu_k = jnp.mean(per_reward_returns, axis=0, keepdims=True)
    sigma_k = jnp.std(per_reward_returns, axis=0, keepdims=True)
    A_k = (per_reward_returns - mu_k) / (sigma_k + eps)
    A_sum = jnp.sum(A_k, axis=-1)
    return (A_sum - jnp.mean(A_sum)) / (jnp.std(A_sum) + eps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = make_argparser().parse_args()
    use_pointer, use_autoreg = _select_variant(args)
    variant_label = _variant_label(use_pointer, use_autoreg)

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    _setup_jax_compile_cache()

    if args.disable_eval and "acc" in args.rewards:
        print("Note: --disable-eval set; dropping 'acc' from --rewards.")
        args.rewards = [r for r in args.rewards if r != "acc"]
    if args.measure_latency and args.num_eval_samples < 10:
        raise ValueError(
            "--measure-latency requires --num-eval-samples >= 10 (the env "
            f"draws 10 samples per step); got {args.num_eval_samples}."
        )
    if args.cache_encoding:
        # Per-trajectory batching reshapes (num_envs, T, ...) → (mb, envs_per_mb, T,
        # ...); envs_per_mb = num_envs // minibatches must be ≥ 1 (and ideally
        # divides evenly so no trajectories are dropped).
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

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(
        args, use_autoreg,
    )
    factor_table_np = np.array(factors_py, dtype=np.int32)
    # All-ones placeholder — see ppo.py for rationale.
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    # ---------------- Reward selection ----------------
    reward_indices_np = _build_reward_indices(args)
    reward_indices = jnp.asarray(reward_indices_np)
    num_rewards = int(reward_indices_np.shape[0])
    selected_names = [REWARD_NAMES[i] for i in reward_indices_np]
    print(f"Group-decoupled reward heads ({num_rewards}): {selected_names}")

    # ---------------- Hyperparameters ----------------
    num_envs = _resolve_num_envs(args.num_envs, args.example)
    rollout_length = num_valid

    print(
        f"variant={variant_label}, GROUP_SIZE={num_envs}, "
        f"max_rules={max_rules}, factors={factors_py}, "
        f"rollout_length={rollout_length}, MINIBATCHES={args.minibatches}"
    )

    # ---------------- Agent ----------------
    agent_key, init_key, key = jrand.split(key, 3)
    agent = _build_agent(
        use_pointer, use_autoreg, args, total_v, num_factors, max_rules,
        agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(
        agent, args.head_init_scale, use_pointer, use_autoreg,
    )

    # ---------------- Optimiser ----------------
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

    # ---------------- Rollout ----------------
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, 0))
    def rollout_fn(agent, env_obj, temperature, env_state, key, vertex_features, preference):
        keys = jrand.split(key, rollout_length)
        init_residual = jnp.zeros((total_v, args.embd_dim), dtype=jnp.float32)

        # Optional cache_encoding path: encode the residual jaxpr's initial
        # tokens once per env. The cached encoding is closed over by step_fn
        # and reused for every step's sample_action call — only the
        # residual_state varies across the rollout.
        encode_key, _ = jrand.split(keys[0], 2)
        if args.cache_encoding:
            cached_encoding = agent.encode_once(
                env_state.tokens, eqn_ids=env_state.eqn_ids, key=encode_key,
            )
        else:
            cached_encoding = None

        def step_fn(carry, k):
            state, residual_state = carry
            sample_key, _ = jrand.split(k, 2)
            vertex_avail_mask = vertex_avail_at_step(
                state, vertex_valid_static, total_v, num_valid,
            )
            (
                vertex_idx,
                pair_seq,
                factor_seq,
                vertex_dist,
                pair_dists,
                factor_dists,
                _value,
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
                vertex_temperature=temperature,
                preference=preference,
            )
            env_action = agent.to_env_action(
                vertex_idx, pair_seq, factor_seq, factor_table,
            )
            env_out = env_obj.step(state, env_action)
            new_residual = agent.update_residual(
                residual_state, vertex_idx, v_context,
            )

            transition = Trajectory(
                tokens=state.tokens.astype(jnp.int32),
                eqn_ids=state.eqn_ids.astype(jnp.int32),
                residual_state=residual_state,
                vertex_idx=jnp.asarray(vertex_idx, dtype=jnp.int32),
                pair_seq=jnp.asarray(pair_seq, dtype=jnp.int32),
                factor_seq=jnp.asarray(factor_seq, dtype=jnp.int32),
                vertex_avail_mask=vertex_avail_mask,
                old_vertex_dist=vertex_dist,
                old_pair_dists=pair_dists,
                old_factor_dists=factor_dists,
                reward=jnp.atleast_1d(env_out.reward),
                preference=preference.astype(jnp.float32),
            )
            return (env_out.state, new_residual), transition

        (final_state, _), traj = lax.scan(
            step_fn, (env_state, init_residual), keys,
        )
        return final_state, traj

    # ---------------- Loss ----------------
    def loss_fn(agent, batch: TrainBatch, vertex_features, key):
        # Two batch shapes:
        #   * Flat (default): leading dim = mb_size, all per-(env,t) samples
        #     interleaved by `shuffle_and_batch`.
        #   * Per-trajectory (cache mode): leading two dims = (envs_per_mb, T)
        #     from `shuffle_and_batch_by_trajectory`. We encode once per
        #     trajectory, broadcast the cached encoding to T steps, then
        #     flatten before vmapping `evaluate_action`.
        if batch.tokens.ndim == 3:
            E, K = batch.tokens.shape[:2]
            enc_key, eval_key = jrand.split(key, 2)
            cached_per_traj = jax.vmap(
                lambda t, e, k: agent.encode_once(t, eqn_ids=e, key=k)
            )(batch.tokens[:, 0], batch.eqn_ids[:, 0], jrand.split(enc_key, E))
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(E * K, *x.shape[2:]), batch,
            )
            cached_flat = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, K, axis=0), cached_per_traj,
            )
            keys = jrand.split(eval_key, E * K)
        else:
            cached_flat = None
            keys = jrand.split(key, batch.tokens.shape[0])

        def _eval_one(toks, eids, rs, vidx, pseq, fseq, vmask, pref, cached, k):
            return agent.evaluate_action(
                toks, vidx, pseq, fseq, vmask,
                pair_valid_mask, pair_factor_mask, k,
                eqn_ids=eids, vertex_features=vertex_features,
                residual_state=rs, cached_encoding=cached,
                preference=pref,
            )

        if cached_flat is None:
            # vmap can't ingest `None` over a vmapped axis; bind a no-cache
            # variant that closes over `None` for the cached argument.
            log_probs, entropies, _value, prob_v, prob_p, prob_f = jax.vmap(
                lambda toks, eids, rs, vidx, pseq, fseq, vmask, pref, k:
                    _eval_one(toks, eids, rs, vidx, pseq, fseq, vmask, pref, None, k),
            )(
                batch.tokens, batch.eqn_ids, batch.residual_state,
                batch.vertex_idx, batch.pair_seq, batch.factor_seq,
                batch.vertex_avail_mask, batch.preference, keys,
            )
        else:
            log_probs, entropies, _value, prob_v, prob_p, prob_f = jax.vmap(
                _eval_one,
            )(
                batch.tokens, batch.eqn_ids, batch.residual_state,
                batch.vertex_idx, batch.pair_seq, batch.factor_seq,
                batch.vertex_avail_mask, batch.preference, cached_flat, keys,
            )

        old_log_probs = jax.vmap(old_log_prob_for_action)(
            batch.vertex_idx, batch.pair_seq, batch.factor_seq,
            batch.old_vertex_dist, batch.old_pair_dists, batch.old_factor_dists,
        )
        ratio = jnp.exp(log_probs - old_log_probs)

        num_triggers = get_num_clipping_triggers(ratio, args.ppo_clip_eps)
        trigger_ratio = num_triggers / len(ratio)

        # GRPO clipped surrogate (paper Eq. 3) with the GDPO advantage.
        clipping_objective = jnp.minimum(
            ratio * batch.advantage,
            jnp.clip(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
            * batch.advantage,
        )
        policy_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)

        kl_div = jnp.mean(
            optax.kl_divergence(jnp.log(prob_v + 1e-7), batch.old_vertex_dist)
        )

        total_loss = policy_loss - args.entropy_weight * entropy_loss
        return total_loss, (
            kl_div, entropy_loss, policy_loss, total_loss, trigger_ratio,
        )

    # ---------------- Train episode ----------------
    def train_episode(
        agent, opt_state, env_states, env_obj, vertex_features, temperature,
        preferences, key,
    ):
        rollout_key, loss_key = jrand.split(key, 2)
        rollout_keys = jrand.split(rollout_key, num_envs)

        env_states, traj = rollout_fn(
            agent, env_obj, temperature, env_states, rollout_keys, vertex_features,
            preferences,
        )

        # GDPO advantage: per-rollout per-reward returns → group-normalise →
        # sum across rewards → batch-renormalise → broadcast to every step.
        # traj.reward shape: (num_envs, T, NUM_REWARDS); we pick the selected
        # subset before aggregating.
        per_reward_per_step = traj.reward[..., reward_indices]   # (E, T, n)
        per_reward_totals = jnp.sum(per_reward_per_step, axis=1)  # (E, n)
        adv_per_rollout = gdpo_advantages(per_reward_totals)      # (E,)
        adv_per_step = jnp.broadcast_to(
            adv_per_rollout[:, None], (num_envs, rollout_length),
        )

        full_batch = TrainBatch(
            tokens=traj.tokens,
            eqn_ids=traj.eqn_ids,
            residual_state=traj.residual_state,
            vertex_idx=traj.vertex_idx,
            pair_seq=traj.pair_seq,
            factor_seq=traj.factor_seq,
            vertex_avail_mask=traj.vertex_avail_mask,
            old_vertex_dist=traj.old_vertex_dist,
            old_pair_dists=traj.old_pair_dists,
            old_factor_dists=traj.old_factor_dists,
            advantage=adv_per_step,
            preference=traj.preference,
        )

        dynamic_carry, static_carry = eqx.partition(
            (agent, opt_state), eqx.is_array,
        )

        def train_epoch(carry, epoch_key):
            batches = (
                shuffle_and_batch_by_trajectory(
                    full_batch, args.minibatches, epoch_key,
                )
                if args.cache_encoding
                else shuffle_and_batch(full_batch, args.minibatches, epoch_key)
            )
            mb_keys = jrand.split(epoch_key, args.minibatches)

            def train_minibatch(c, batch_and_key):
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
                    comb_agent, batch, vertex_features, t_key,
                )
                updates, new_opt_state = optimizer.update(
                    grads, comb_opt_state, comb_agent,
                )
                new_agent = eqx.apply_updates(comb_agent, updates)
                next_carry, _ = eqx.partition(
                    (new_agent, new_opt_state), eqx.is_array,
                )
                return next_carry, metrics

            return lax.scan(train_minibatch, carry, (batches, mb_keys))

        epoch_keys = jrand.split(loss_key, args.ppo_epochs)
        dynamic_carry, metrics_seq = lax.scan(
            train_epoch, dynamic_carry, epoch_keys,
        )
        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)

        # (num_envs, NUM_REWARDS) — sum over rollout time of every reward dim,
        # for diagnostics. The selected subset drives the loss; the full
        # vector is kept so wandb can plot every component.
        ep_per_reward_full = jnp.sum(traj.reward, axis=1)
        actions_pack = (traj.vertex_idx, traj.pair_seq, traj.factor_seq)
        return agent, opt_state, env_states, metrics, ep_per_reward_full, actions_pack

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    # ---------------- Reporting ----------------
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else args.wandb,
    )
    elim_order_table = wandb.Table(
        columns=["episode", "joint return", "elimination order"],
    )
    pbar = tqdm(total=args.episodes)
    host_state = {
        "samplecounts": 0,
        "best_global_return": -float("inf"),
        "best_global_act_seq": None,
    }

    weights_full = np.zeros(len(REWARD_NAMES), dtype=np.float32)
    for i in reward_indices_np:
        weights_full[int(i)] = 1.0  # uniform across selected dims for reporting

    def host_log(ep, ep_per_reward_full, actions_pack, mets):
        ep = int(ep)
        ep_per_reward_full = np.asarray(ep_per_reward_full)
        v_idx_arr = np.asarray(actions_pack[0])
        pair_arr = np.asarray(actions_pack[1])
        factor_arr = np.asarray(actions_pack[2])

        # "Joint return" for ranking trajectories: sum of selected dims, same
        # set the GDPO advantage operates on (uniform weights — paper's Eq. 5
        # default). Diagnostic only; not used in the loss.
        joint = ep_per_reward_full @ weights_full
        best_idx = int(np.argmax(joint))
        best_ret = float(joint[best_idx])
        mean_per_reward_full = ep_per_reward_full.mean(axis=0)

        host_state["samplecounts"] += num_envs * rollout_length
        kl_div, policy_entropy, policy_loss, total_loss, trigger_ratio = (
            float(m) for m in mets
        )

        if best_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = best_ret
            host_state["best_global_act_seq"] = _action_to_pylist(
                v_idx_arr[best_idx], pair_arr[best_idx], factor_arr[best_idx],
                max_rules, factor_table_np,
            )
            per_r = ", ".join(
                f"{REWARD_NAMES[i]}={float(ep_per_reward_full[best_idx, i]):.3f}"
                for i in reward_indices_np
            )
            print(f"\nNew best joint return: {best_ret:.4f}  ({per_r})")
            print(f"Action sequence: {host_state['best_global_act_seq']}")
            elim_order_table.add_data(
                ep, best_ret, str(host_state["best_global_act_seq"]),
            )

        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": float(joint.mean()),
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "sample count": host_state["samplecounts"],
            "policy loss": policy_loss,
            "total loss": total_loss,
            "clip trigger ratio": trigger_ratio,
        }
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(mean_per_reward_full[j])
        wandb.log(log_dict)

        means_str = ", ".join(
            f"{REWARD_NAMES[i]}={float(mean_per_reward_full[i]):.2f}"
            for i in reward_indices_np
        )
        pbar.update(1)
        pbar.set_description(
            f"ent:{policy_entropy:.3f} best:{best_ret:.2f} {means_str}"
        )

    # ---------------- Training loop ----------------
    for ep in range(args.episodes):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)

        if args.num_eval_samples > 0:
            eval_samples = generate_eval_samples(
                env, ep_eval_key, args.num_eval_samples,
            )
            env_episode = eqx.tree_at(
                lambda e: e.eval_args_samples, env, eval_samples,
            )
        else:
            eval_samples = None
            env_episode = env

        vertex_features = _episode_vertex_features(
            args, closed_jaxpr.jaxpr, tuple(closed_jaxpr.literals),
            tuple(xs), eval_samples=eval_samples, argnums=tuple(argnums),
        )

        env_states = reset_envs(env_episode)
        progress = ep / max(args.episodes - 1, 1)
        temperature = jnp.asarray(
            schedule_at(
                progress, args.rollout_temperature,
                args.rollout_temperature_final,
                args.rollout_temperature_schedule,
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
        agent, opt_state, _, metrics, ep_per_reward_full, actions_pack = train_episode(
            agent, opt_state, env_states, env_episode, vertex_features,
            temperature, preferences, ep_key,
        )
        host_log(ep, ep_per_reward_full, actions_pack, metrics)

    pbar.close()
    wandb.log({"Elimination order": elim_order_table})
    if host_state["best_global_act_seq"] is not None:
        print(
            f"\nBest vertex elimination sequence after {args.episodes} episodes: "
            f"{host_state['best_global_act_seq']} with joint return "
            f"{host_state['best_global_return']:.4f}."
        )


if __name__ == "__main__":
    main()
