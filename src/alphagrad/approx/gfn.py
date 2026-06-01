"""Trajectory-Balance GFlowNet trainer for the vertex-elimination env.

Architecturally this is a thin layer over the PPO trainer's :class:`Agent`
(encoder + composable vertex/rule policies + value head). The only addition
is a scalar learnable ``logZ`` parameter wrapped into :class:`GFNAgent`, used
in the Trajectory-Balance objective

    L(τ) = (logZ + Σ_t log π_F(a_t | s_t)
                  - Σ_t log P_B(s_{t-1} | s_t)
                  - β · log R(x))²

with R(x) the *terminal* environment reward (the env's per-step reward at the
last step is the cost of the fully-eliminated graph; intermediate steps cost
prefixes only). The backward policy is the standard uniform-over-parents
approximation P_B(s_{t-1} | s_t) = 1/t. In our MDP the state encodes the
ordered prefix, so each non-initial state has a unique parent — log P_B then
contributes a constant -log(N!) over the trajectory and does not affect the
gradient. We keep it in the loss as a placeholder for non-uniform variants.

Everything else (vertex / rule policy variant flags, reward family weights,
factor table, encoder size, optimiser) reuses PPO's CLI conventions so a run
config can be moved between trainers with minimal edits. PPO-specific knobs
(GAE-λ, PPO clip, multi-epoch minibatching, Stage C/D/E/F/G phases) are not
ported.
"""

from __future__ import annotations

import argparse
import heapq
import os
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
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
    generate_eval_samples,
    data_gen,
    get_args,
    get_fn,
    infer_argnums,
    init_linear_weights,
    init_replay_buffer,
    load_replay_buffer,
    build_pair_valid_mask,
    build_vertex_valid_static,
    replay_add_batch,
    replay_sample,
    sample_preferences,
    save_replay_buffer,
    schedule_at,
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
    Agent,
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    _action_to_pylist,
    _build_agent,
    _build_factor_table,
    _build_reward_weights,
    _cmp_reward_index,
    _episode_vertex_features,
    _mem_reward_index,
    _resolve_num_envs,
    _scale_output_heads,
    _select_variant,
    _setup_jax_compile_cache,
    _variant_label,
)
from alphagrad.utils import symlog


# ---------------------------------------------------------------------------
# GFN-specific module: PPO Agent + scalar logZ
# ---------------------------------------------------------------------------


class GFNAgent(eqx.Module):
    """Wraps a PPO :class:`Agent` and adds the learnable Trajectory-Balance ``logZ``.

    All policy / encode / value logic is delegated to ``base_agent``. The
    value head is unused here (TB doesn't need a critic) but we leave it
    constructed so the agent can be initialised, scaled, and optimised by
    the same helpers PPO uses.
    """

    base_agent: Agent
    logZ: jax.Array

    def __init__(self, base_agent: Agent, logZ_init: float = 0.0):
        self.base_agent = base_agent
        self.logZ = jnp.asarray(logZ_init, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Trajectory layout — fields needed to recompute log π_F(a_t | s_t) at loss
# time and to read off the terminal reward.
# ---------------------------------------------------------------------------


class GFNTrajectory(NamedTuple):
    tokens: jax.Array
    eqn_ids: jax.Array
    residual_state: jax.Array
    vertex_idx: jax.Array
    pair_seq: jax.Array
    factor_seq: jax.Array
    vertex_avail_mask: jax.Array
    reward: jax.Array     # (T, NUM_REWARDS) per-step env reward vector
    done: jax.Array       # (T,) float32; 1.0 only at the terminating step
    step_count: jax.Array  # (T,) float32; state.step_count *before* the action
    preference: jax.Array  # (T, NUM_REWARDS) — per-env preference, broadcast


# ---------------------------------------------------------------------------
# Argparse — GFN-specific subset of PPO's CLI
# ---------------------------------------------------------------------------


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trajectory-Balance GFlowNet trainer for vertex-elimination.",
    )

    # Run / logging
    p.add_argument("--name", type=str, default="approx-gfn")
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
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--terminal-rewards-only", action="store_true",
                   help="Compute the env's reward vector only at the final "
                        "elimination step; intermediate steps return zeros. "
                        "Skips per-step jacve compile/exec. TB already reads "
                        "only traj.reward[:, -1, :], so this is a pure speed "
                        "win with identical training behaviour.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)

    # Agent variant — same independent flags as PPO. The four combinations
    # are pointer/mlp-vertex × autoreg/single-rule.
    p.add_argument("--no-ptr", action="store_true")
    p.add_argument("--not-autoreg", action="store_true")
    p.add_argument("--max-rules", type=int, default=4)
    p.add_argument("--factors", type=str, default="-1,1,2,4")
    # Stage E knobs are kept off in GFN by default — `_build_agent`/_build_factor_table
    # read them, so we surface them here to keep the call sites identical.
    p.add_argument("--sparsity-ratio", action="store_true")
    p.add_argument("--rho-prior-bias", type=float, default=4.0)
    p.add_argument("--set-transformer-agg", action="store_true")

    # Network architecture
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--embd-dim", type=int, default=32)
    p.add_argument("--op-embd-dim", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--policy-dims", type=str, default="64,32")
    p.add_argument("--value-dims", type=str, default="64,32")
    p.add_argument("--head-init-scale", type=float, default=0.1)

    # Optimisation
    p.add_argument("--num-envs", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--logZ-lr", type=float, default=1e-2,
                   help="Separate LR for the scalar logZ — TB convergence is "
                        "much faster when logZ has its own larger step size.")
    p.add_argument("--logZ-init", type=float, default=0.0)
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Gradient steps per rollout. Each step re-evaluates "
                        "log π_F under the current agent against the stored actions.")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--adam-b1", type=float, default=0.9)
    p.add_argument("--adam-eps", type=float, default=1e-7)
    p.add_argument("--lr-decay-min-mult", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.0,
                   help="Optional entropy bonus added to the TB objective. "
                        "Default 0 — the TB residual already encourages "
                        "diversity through logZ.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="TB inverse temperature. log R(x) = β · symlog(weighted "
                        "terminal reward). Larger β → policy concentrates on "
                        "best trajectories; β → 0 recovers a uniform policy. "
                        "Acts as the initial value when --beta-schedule != "
                        "constant.")
    p.add_argument("--beta-final", type=float, default=1.0,
                   help="Final β when --beta-schedule != constant. A common "
                        "recipe is to warm up β: 0.1 → 1.0 over training "
                        "(broad exploration → mode-focused) — set --beta=0.1 "
                        "--beta-final=1.0 --beta-schedule=linear.")
    p.add_argument("--beta-schedule", type=str, default="constant",
                   choices=SCHEDULES)
    # Encoder cache: encode the initial residual jaxpr's tokens once per
    # episode (per env in the rollout, per trajectory in the loss) and reuse
    # via the residual_state path. Saves ~T-fold encoder work in both phases.
    p.add_argument("--cache-encoding", action="store_true",
                   help="Reuse the encoder output across all T steps of a "
                        "rollout via PPO's encode-once + residual_state path.")

    # SEED-style replay buffer. TB is off-policy by construction (paper
    # §3.3.3), so retaining past trajectories and sampling from them at
    # gradient-step time is a clean diversity boost: the policy keeps
    # "rehearsing" old high-reward modes while still exploring new ones
    # via fresh rollouts.
    p.add_argument("--replay-buffer-size", type=int, default=0,
                   help="Replay buffer capacity (number of stored trajectories). "
                        "0 disables — train only on fresh rollouts.")
    p.add_argument("--replay-batch-size", type=int, default=0,
                   help="Trajectories sampled from buffer per gradient step. "
                        "0 = use --num-envs (matches the fresh rollout's batch). "
                        "Ignored when --replay-buffer-size == 0.")
    p.add_argument("--replay-warmup", type=int, default=1,
                   help="Episodes (= rollouts) to fill before sampling from "
                        "the buffer. Until then we train on fresh data.")
    p.add_argument("--replay-fresh-fraction", type=float, default=0.0,
                   help="Fraction of the train batch drawn from the *fresh* "
                        "rollout instead of the buffer. 0.0 = pure replay; "
                        "1.0 disables replay sampling. Useful for tuning the "
                        "off-policy/on-policy mix.")
    # Prioritised sampling: high-reward trajectories get re-sampled more
    # often. For TB this corresponds to "rehearsing winners" which can
    # accelerate mode discovery — at the cost of some entropy.
    p.add_argument("--replay-priority-alpha", type=float, default=0.0,
                   help="Power applied to per-slot priorities at sample "
                        "time. 0 = uniform.")
    # Disk checkpointing.
    p.add_argument("--replay-checkpoint-path", type=str, default="",
                   help="If set, the buffer is saved here every "
                        "--replay-checkpoint-every episodes; loaded at "
                        "startup if the file exists.")
    p.add_argument("--replay-checkpoint-every", type=int, default=10)

    # Preference conditioning.
    p.add_argument("--preference-conditioned", action="store_true",
                   help="Sample a per-env Dirichlet preference each "
                        "episode and condition the policy on it.")
    p.add_argument("--preference-dirichlet-alpha", type=float, default=1.0)

    # Reporting
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--capture-perfect-grads", action="store_true")

    return p


# ---------------------------------------------------------------------------
# Build helpers — thin wrappers over PPO's so optimisation / scaling is shared
# ---------------------------------------------------------------------------


def _build_gfn_agent(
    use_pointer: bool,
    use_autoreg: bool,
    args,
    total_v: int,
    num_factors: int,
    max_rules: int,
    key,
) -> GFNAgent:
    base_agent = _build_agent(
        use_pointer, use_autoreg, args, total_v, num_factors, max_rules, key,
    )
    return GFNAgent(base_agent=base_agent, logZ_init=args.logZ_init)


def _logZ_mask(gfn_agent: GFNAgent):
    """Scalar-bool pytree aligned with ``eqx.filter(gfn_agent, eqx.is_inexact_array)``
    where the lone True leaf is the ``logZ`` scalar.

    ``optax.masked`` requires scalar Python bools at each leaf (it tests the
    leaf with a plain ``if`` to decide whether to substitute ``MaskedNode``),
    so we return ``True`` / ``False`` rather than 0-d arrays.
    """
    params = eqx.filter(gfn_agent, eqx.is_inexact_array)
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    return treedef.unflatten([
        jax.tree_util.keystr(path).endswith(".logZ")
        for path, _ in leaves_with_path
    ])


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

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(args.example, dataset=dataset_for_call, dataset_size=args.dataset_size)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    # Always pass target_fun so flops/bytes_accessed/latency_ns/peak_memory
    # populate every step (see cpu_approx_worker.py for the full rationale).
    env_target_fun = target_fn
    argnums = infer_argnums(args.example)

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
        exec_on_gpu=False,
        measure_latency=measure_latency,
        latency_samples=int(getattr(args, "latency_samples", 1)),
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

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args, use_autoreg)
    factor_table_np = np.array(factors_py, dtype=np.int32)
    # All-ones placeholder — see ppo.py for rationale.
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    num_envs = _resolve_num_envs(args.num_envs, args.example)
    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
    cmp_idx = _cmp_reward_index(args.cmp_type)
    mem_idx = _mem_reward_index(args.mem_type)
    cosine_idx = REWARD_INDEX["cosine_sim"]
    frob_idx = REWARD_INDEX["frob_residual"]

    print(
        f"variant={variant_label}, num_envs={num_envs}, max_rules={max_rules}, "
        f"factors={factors_py}, rollout_length={num_valid}, beta={args.beta}"
    )
    nonzero_w = ", ".join(
        f"{REWARD_NAMES[i]}={float(reward_weights_np[i]):+.3g}"
        for i in range(NUM_REWARDS)
        if reward_weights_np[i] != 0.0
    )
    print(f"reward weights: {nonzero_w or '<all zero — debug only>'}")

    agent_key, init_key, key = jrand.split(key, 3)
    gfn_agent = _build_gfn_agent(
        use_pointer, use_autoreg, args, total_v, num_factors, max_rules, agent_key,
    )
    gfn_agent = eqx.tree_at(
        lambda g: g.base_agent,
        gfn_agent,
        init_linear_weights(gfn_agent.base_agent, init_key),
    )
    gfn_agent = eqx.tree_at(
        lambda g: g.base_agent,
        gfn_agent,
        _scale_output_heads(
            gfn_agent.base_agent, args.head_init_scale, use_pointer, use_autoreg,
        ),
    )

    # Per-leaf LR schedule. Matches PPO's cosine decay for the policy params;
    # logZ rides a faster constant LR (TB literature consistently reports
    # this is needed for the TB residual to track the policy quickly enough).
    schedule = optax.cosine_decay_schedule(
        args.lr, args.episodes * args.gradient_steps, args.lr_decay_min_mult,
    )
    logZ_mask = _logZ_mask(gfn_agent)
    # ``optax.masked`` walks both masks with plain Python ``if``; keep the
    # leaves as scalar bools (no jnp lifting).
    not_logZ_mask = jax.tree_util.tree_map(lambda b: not b, logZ_mask)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.masked(
            optax.adam(args.logZ_lr, b1=args.adam_b1, eps=args.adam_eps),
            logZ_mask,
        ),
        optax.masked(
            optax.adam(schedule, b1=args.adam_b1, eps=args.adam_eps),
            not_logZ_mask,
        ),
    )
    opt_state = optimizer.init(eqx.filter(gfn_agent, eqx.is_inexact_array))

    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, 0))
    def rollout_fn(gfn_agent, env_obj, rollout_length, env_state, key, vertex_features, preference):
        keys = jrand.split(key, rollout_length)
        init_residual = jnp.zeros((total_v, args.embd_dim), dtype=jnp.float32)

        # Optional cache_encoding path: encode the initial residual jaxpr
        # once per env, reuse for every step's sample_action via the cached
        # encoding + per-step residual_state.
        encode_key, _ = jrand.split(keys[0], 2)
        if args.cache_encoding:
            cached_encoding = gfn_agent.base_agent.encode_once(
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
                _vertex_dist,
                _pair_dists,
                _factor_dists,
                _value,
                v_context,
            ) = gfn_agent.base_agent.sample_action(
                state.tokens,
                vertex_avail_mask,
                pair_valid_mask,
                pair_factor_mask,
                sample_key,
                eqn_ids=state.eqn_ids,
                vertex_features=vertex_features,
                residual_state=residual_state,
                cached_encoding=cached_encoding,
                preference=preference,
            )
            env_action = gfn_agent.base_agent.to_env_action(
                vertex_idx, pair_seq, factor_seq, factor_table,
            )
            env_out = env_obj.step(state, env_action)
            new_residual = gfn_agent.base_agent.update_residual(
                residual_state, vertex_idx, v_context,
            )

            transition = GFNTrajectory(
                tokens=state.tokens.astype(jnp.int32),
                eqn_ids=state.eqn_ids.astype(jnp.int32),
                residual_state=residual_state,
                vertex_idx=jnp.asarray(vertex_idx, dtype=jnp.int32),
                pair_seq=jnp.asarray(pair_seq, dtype=jnp.int32),
                factor_seq=jnp.asarray(factor_seq, dtype=jnp.int32),
                vertex_avail_mask=vertex_avail_mask,
                reward=jnp.atleast_1d(env_out.reward),
                done=env_out.terminated.astype(jnp.float32),
                step_count=state.step_count.astype(jnp.float32),
                preference=preference.astype(jnp.float32),
            )
            return (env_out.state, new_residual), transition

        (final_state, _), traj = lax.scan(
            step_fn, (env_state, init_residual), keys,
        )
        return final_state, traj

    def loss_fn(gfn_agent, traj: GFNTrajectory, vertex_features, beta, key):
        """Trajectory-Balance loss.

        The loss operates on the full ``(num_envs, T)`` trajectory tensor —
        re-evaluates ``log π_F(stored_action | s_t)`` per step under the
        current agent (so multi-step training is well-defined), sums over
        time to get per-trajectory ``log_pf``, and combines with the
        terminal log-reward.

        ``beta`` is the TB inverse-temperature, fed in as a runtime scalar so
        it can vary per episode under a schedule (warm-up, cosine, etc.).
        """
        E, T = traj.tokens.shape[:2]
        enc_key, eval_key = jrand.split(key, 2)

        # When cache_encoding is on: encode each env's initial tokens once
        # (E encodings) and broadcast that cached encoding to all T steps
        # of the rollout. Without it, the per-(env, t) flat path runs the
        # full encoder E*T times.
        if args.cache_encoding:
            cached_per_env = jax.vmap(
                lambda t, e, k: gfn_agent.base_agent.encode_once(
                    t, eqn_ids=e, key=k,
                )
            )(traj.tokens[:, 0], traj.eqn_ids[:, 0], jrand.split(enc_key, E))
            cached_flat = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, T, axis=0), cached_per_env,
            )
        else:
            cached_flat = None

        flat = jax.tree_util.tree_map(
            lambda x: x.reshape(E * T, *x.shape[2:]), traj,
        )
        eval_keys = jrand.split(eval_key, E * T)

        def _eval_one(toks, eids, rs, vidx, pseq, fseq, vmask, pref, cached, k):
            return gfn_agent.base_agent.evaluate_action(
                toks, vidx, pseq, fseq, vmask,
                pair_valid_mask, pair_factor_mask, k,
                eqn_ids=eids, vertex_features=vertex_features,
                residual_state=rs, cached_encoding=cached,
                preference=pref,
            )

        if cached_flat is None:
            log_p, ent, _value, _vd, _pd, _fd = jax.vmap(
                lambda toks, eids, rs, vidx, pseq, fseq, vmask, pref, k:
                    _eval_one(toks, eids, rs, vidx, pseq, fseq, vmask, pref, None, k),
            )(
                flat.tokens, flat.eqn_ids, flat.residual_state,
                flat.vertex_idx, flat.pair_seq, flat.factor_seq,
                flat.vertex_avail_mask, flat.preference, eval_keys,
            )
        else:
            log_p, ent, _value, _vd, _pd, _fd = jax.vmap(_eval_one)(
                flat.tokens, flat.eqn_ids, flat.residual_state,
                flat.vertex_idx, flat.pair_seq, flat.factor_seq,
                flat.vertex_avail_mask, flat.preference, cached_flat, eval_keys,
            )
        log_p = log_p.reshape(E, T)
        ent = ent.reshape(E, T)

        # Per-trajectory log_pf — every per-step log-prob counts, including
        # the terminating step where the policy commits the last vertex.
        sum_log_pf = jnp.sum(log_p, axis=-1)

        # Uniform backward policy P_B(s_{t-1} | s_t) = 1/t. ``state.step_count``
        # is the number of vertices already eliminated *before* the action,
        # so t = step_count + 1 after the action lands.
        log_pb_per_step = -jnp.log(traj.step_count + 1.0)
        sum_log_pb = jnp.sum(log_pb_per_step, axis=-1)

        # Terminal reward. The env's per-step reward is the *cumulative* cost
        # of the partial elimination prefix, so the last step's reward is the
        # full trajectory's cost. symlog squashes the typically-large flop /
        # memory magnitudes into a range where β ~ O(1) is well-behaved.
        terminal_rewards = traj.reward[:, -1, :]                  # (E, NUM_REWARDS)
        weighted_terminal = jnp.sum(terminal_rewards * reward_weights, axis=-1)  # (E,)
        log_R = beta * symlog(weighted_terminal)

        tb_residual = gfn_agent.logZ + sum_log_pf - sum_log_pb - log_R
        tb_loss = jnp.mean(tb_residual ** 2)
        entropy_term = jnp.mean(jnp.sum(ent, axis=-1))

        total_loss = tb_loss - args.entropy_weight * entropy_term

        metrics = (
            tb_loss,
            gfn_agent.logZ,
            jnp.mean(sum_log_pf),
            jnp.mean(sum_log_pb),
            jnp.mean(log_R),
            jnp.mean(weighted_terminal),
            entropy_term,
            total_loss,
        )
        return total_loss, metrics

    # Split rollout + training so the host loop can interleave a replay
    # buffer between the two without baking the buffer state into the
    # JIT'd training graph.
    def do_rollout(gfn_agent, env_obj, env_states, vertex_features, preferences, key):
        rollout_keys = jrand.split(key, num_envs)
        env_states, traj = rollout_fn(
            gfn_agent, env_obj, num_valid, env_states, rollout_keys, vertex_features,
            preferences,
        )
        return env_states, traj

    def do_train_steps(gfn_agent, opt_state, train_traj, vertex_features, beta, key):
        dynamic_carry, static_carry = eqx.partition((gfn_agent, opt_state), eqx.is_array)
        step_keys = jrand.split(key, args.gradient_steps)

        def grad_step_fn(carry, t_key):
            comb_agent, comb_opt_state = eqx.combine(carry, static_carry)
            grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
                comb_agent, train_traj, vertex_features, beta, t_key,
            )
            updates, new_opt_state = optimizer.update(
                grads, comb_opt_state, comb_agent,
            )
            new_agent = eqx.apply_updates(comb_agent, updates)
            next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
            return next_carry, metrics

        dynamic_carry, metrics_seq = lax.scan(grad_step_fn, dynamic_carry, step_keys)
        gfn_agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)
        return gfn_agent, opt_state, metrics

    if not args.no_jit:
        do_rollout = eqx.filter_jit(do_rollout)
        do_train_steps = eqx.filter_jit(do_train_steps)

    # Reporting.
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )
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
    replay_buffer = None  # lazily initialised on episode 0 (see training loop)
    _resume_pending = bool(
        args.replay_checkpoint_path
        and os.path.exists(args.replay_checkpoint_path)
    )

    def host_log(ep, all_rets, actions_pack, mets):
        ep = int(ep)
        all_rets = np.array(all_rets)  # (num_envs, NUM_REWARDS)
        v_idx_arr = np.array(actions_pack[0])
        pair_arr = np.array(actions_pack[1])
        factor_arr = np.array(actions_pack[2])

        host_state["samplecounts"] += num_envs * num_valid
        (
            tb_loss, logZ, mean_log_pf, mean_log_pb, mean_log_R,
            mean_terminal, entropy_term, total_loss,
        ) = [float(m) for m in mets]

        weights = reward_weights_np
        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
            decoded = _action_to_pylist(
                v_idx_arr[i], pair_arr[i], factor_arr[i], max_rules, factor_table_np,
            )
            total_ret = float(np.sum(rets * weights))
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
            host_state["best_global_act_seq"] = _action_to_pylist(
                v_idx_arr[best_idx], pair_arr[best_idx], factor_arr[best_idx],
                max_rules, factor_table_np,
            )

        mean_r = np.mean(all_rets, axis=0)
        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": float(np.sum(mean_r * weights)),
            "tb_loss": tb_loss,
            "logZ": logZ,
            "mean_log_pf": mean_log_pf,
            "mean_log_pb": mean_log_pb,
            "mean_log_R": mean_log_R,
            "mean_terminal_reward": mean_terminal,
            "entropy": entropy_term,
            "total_loss": total_loss,
            "sample count": host_state["samplecounts"],
        }
        for j, name in enumerate(REWARD_NAMES):
            log_dict[f"mean_{name}"] = float(mean_r[j])
        wandb.log(log_dict)

        pbar.update(1)
        means_str = ", ".join(f"{float(x):.2f}" for x in np.abs(mean_r))
        b_ret_unnorm = np.abs(all_rets[best_idx])
        b_ret_desc = ", ".join(f"{float(x):.1f}" for x in b_ret_unnorm)
        pbar.set_description(
            f"tb:{tb_loss:.2f} logZ:{logZ:.2f} best:{b_ret_desc} means:{means_str}"
        )

    for ep in range(args.episodes):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)

        eval_samples = generate_eval_samples(env, ep_eval_key, args.num_eval_samples)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        vertex_features = _episode_vertex_features(
            args, closed_jaxpr.jaxpr, tuple(closed_jaxpr.literals),
            tuple(xs), eval_samples=eval_samples, argnums=tuple(argnums),
        )

        env_states = reset_envs(env_episode)
        progress = ep / max(args.episodes - 1, 1)
        beta = jnp.asarray(
            schedule_at(
                progress, args.beta, args.beta_final, args.beta_schedule,
            ),
            dtype=jnp.float32,
        )
        rollout_key, sample_key, pref_key, train_key = jrand.split(ep_key, 4)

        if args.preference_conditioned:
            preferences = sample_preferences(
                pref_key, NUM_REWARDS, num_envs,
                dirichlet_alpha=args.preference_dirichlet_alpha,
            )
        else:
            preferences = jnp.zeros((num_envs, NUM_REWARDS), dtype=jnp.float32)

        env_states, fresh_traj = do_rollout(
            gfn_agent, env_episode, env_states, vertex_features,
            preferences, rollout_key,
        )

        # Replay buffer: SEED-style off-policy sampling. Initialise lazily
        # from the first episode's rollout so we don't need to know the
        # trajectory pytree shape ahead of time. Fresh trajectories are
        # added to the buffer *after* the gradient step — that way the
        # episode that just rolled doesn't sample from itself.
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
                int(target_batch_size * args.replay_fresh_fraction),
                num_envs,
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

        gfn_agent, opt_state, metrics = do_train_steps(
            gfn_agent, opt_state, train_traj, vertex_features, beta, train_key,
        )

        if args.replay_buffer_size > 0 and replay_buffer is not None:
            # Priority = shifted-positive weighted *terminal* reward. TB
            # only ever reads the terminal reward, so this is the natural
            # "trajectory quality" signal.
            terminal_weighted = jnp.sum(
                fresh_traj.reward[:, -1, :] * reward_weights, axis=-1,
            )
            traj_priorities = (
                terminal_weighted - jnp.min(terminal_weighted) + 1e-3
            )
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

        terminal_rewards_full = fresh_traj.reward[:, -1, :]
        actions_pack = (
            fresh_traj.vertex_idx, fresh_traj.pair_seq, fresh_traj.factor_seq,
        )
        host_log(ep, terminal_rewards_full, actions_pack, metrics)

    pbar.close()

    def print_top_n(name, heap):
        print(f"\nTop {args.top_n} trajectories for {name}:")
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
        table = wandb.Table(
            columns=["rank", "episode", "total_reward", "cmp", "acc", "mem", "frob", "sequence"],
        )
        weights = reward_weights_np
        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            arr = np.array(rets)
            total_ret = float(np.sum(arr * weights))
            cmp_val = -float(arr[cmp_idx])
            mem_val = -float(arr[mem_idx])
            acc_val = float(arr[cosine_idx])
            frob_val = -float(arr[frob_idx])
            print(
                f"{rank}. Ep {ep} | Total Reward: {total_ret:.2f} | "
                f"CMP({args.cmp_type}): {cmp_val:.1f} | Acc: {acc_val:.4f} | "
                f"Mem({args.mem_type}): {mem_val:.1f} | Frob: {frob_val:.4f}"
            )
            print(f"   Sequence (vertex, [(idx1, idx2, factor), ...]): {seq}")
            table.add_data(
                rank, ep, total_ret, cmp_val, acc_val, mem_val, frob_val, str(seq),
            )
        wandb.log({f"Top N {name}": table})

    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n(f"CMP (Lowest {args.cmp_type})", host_state["top_n_cmp"])
    print_top_n(f"Memory (Lowest {args.mem_type})", host_state["top_n_mem"])
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])


if __name__ == "__main__":
    main()
