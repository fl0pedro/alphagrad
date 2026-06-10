from __future__ import annotations

import contextlib
import os
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import mctx
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from alphagrad.approx.common import (
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
    vertex_avail_at_step,
)
from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_AXIS_PAIRS,
    NUM_REWARDS,
    QUALITY_REWARD_INDICES,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)
from alphagrad.utils import symlog as _symlog
from alphagrad.approx.mu0 import (
    _NO_SYMLOG_MASK_NP,
    _PAIR_TO_BASE,
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    DecisionEmbedding,
    MuZeroAgent,
    Trajectory,
    TrajectoryWindow,
    _action_to_pylist,
    _build_factor_table,
    _build_reward_weights,
    _compute_traj_priorities,
    _discounted_returns,
    _empty_decision,
    _episode_vertex_features,
    _parse_int_list,
    _resolve_num_envs,
    _scale_output_heads,
    _setup_jax_compile_cache,
    _shuffle_and_batch_windows,
    make_curriculum_schedule,
    parse_lagrangian_constraints,
)
from alphagrad.approx.variants import (
    _apply_variant_preset,
    _current_stage_at,
    _default_full_curriculum,
    _parse_curriculum,
    _pin_rules_for_variant,
)


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Back-compat re-export of
    :func:`alphagrad.approx.common.ray_runtime._args_from_dict`."""
    from alphagrad.approx.common.ray_runtime import _args_from_dict as _impl
    return _impl(args_dict)


# Per-channel "do not discount" mask. ``cosine_sim`` and ``frob_residual``
# (env.QUALITY_REWARD_INDICES = (6, 7)) are quality metrics computed
# against the *final* approximated Jacobian — they don't accumulate over
# the rollout the way the per-step compute/memory costs do, so applying
# ``--discount < 1`` to them would penalise the terminal quality signal
# by ``γ^T`` for no good reason. The per-step compute/memory channels
# still see ``--discount`` normally.
_NO_DISCOUNT_MASK = jnp.zeros((NUM_REWARDS,), dtype=jnp.bool_).at[
    jnp.asarray(list(QUALITY_REWARD_INDICES), dtype=jnp.int32)
].set(True)


def _per_channel_discounted_returns(
    reward_vec: "jax.Array",
    weights: "jax.Array",
    discount: float,
) -> "jax.Array":
    """Per-step return where some channels are not discounted.

    ``reward_vec`` is (T, NUM_REWARDS); ``weights`` is (NUM_REWARDS,);
    returns (T,) — the per-step ``G_t = Σ_{j≥t} γ_c^{j-t} w_c r_{c,j}``
    with ``γ_c = 1.0`` for channels flagged in ``_NO_DISCOUNT_MASK``
    and ``γ_c = discount`` otherwise.

    Implementation: keep per-channel accumulators in a single ``lax.scan``
    over time (reversed) so the cost is the same as the original
    single-channel ``_discounted_returns`` (one scan + a final sum)
    regardless of how many channels are involved.

    Phase 4a: clamp raw rewards to ``±(SENTINEL - 1)`` before weighting
    so any pool sentinel that slipped past the cpu_approx_pool's mask
    can't blow up the value loss. ``peak_memory ≈ 1e9`` is real, so we
    only bound at ``1e10 - 1`` (just below the sentinel magnitude).
    """
    from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
    bound = jnp.float32(abs(SENTINEL_REWARD_VALUE) - 1.0)
    reward_vec = jnp.clip(reward_vec, -bound, bound)
    weighted = reward_vec * weights  # (T, NUM_REWARDS)
    gammas = jnp.where(
        _NO_DISCOUNT_MASK,
        jnp.float32(1.0),
        jnp.float32(discount),
    )  # (NUM_REWARDS,)

    def step(carry, x):
        # carry: (NUM_REWARDS,) per-channel discounted-future sum
        # x:     (NUM_REWARDS,) per-step weighted reward
        new = x + gammas * carry
        return new, new

    init = jnp.zeros_like(gammas)
    _, returns_rev = lax.scan(step, init, weighted[::-1])
    # Sum channels to get the scalar return per step.
    return returns_rev[::-1].sum(axis=-1)


def _build_actor_state(
    args_dict: dict,
    variant: str,
    actor_seed: int,
    is_spmd: bool = False,
    cpu_workers: list = None,
    remote_pool: object = None,
    remote_timeout_s: float = 60.0,
) -> dict:
    args = _args_from_dict(args_dict)
    _apply_variant_preset(args, variant)

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _setup_jax_compile_cache()

    key = jrand.PRNGKey(actor_seed)
    key, args_key = jrand.split(key)

    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = dataset_arg is not None and args.example.endswith("NeuralNetwork")
    dataset_for_call = dataset_arg if use_dataset else None
    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(
        args.example, dataset=dataset_for_call, dataset_size=args.dataset_size
    )
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
        # Measurement-pool + latency-estimator knobs (getattr defaults so this
        # is a no-op until mu0_args defines the flags — the exec loop now sizes
        # the pool from num_data_points×reps_per_point, not latency_samples).
        num_data_points=int(getattr(args, "num_data_points", 5)),
        reps_per_point=int(getattr(args, "reps_per_point", 4)),
        percentile_keep=float(getattr(args, "percentile_keep", 0.60)),
        latency_inner_reps=int(getattr(args, "latency_inner_reps", 1)),
        latency_warmup=int(getattr(args, "latency_warmup", 0)),
        latency_winsor=float(getattr(args, "latency_winsor", 0.0)),
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
        disable_sparsification=args.disable_sparsification,
    )

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args)
    factor_table_np = np.array(factors_py, dtype=np.int32)
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32
    )

    DECISION_DEPTH = 1 + 2 * max_rules
    UNIFIED_ACTION_SIZE = int(max(total_v, NUM_PAIR_CHOICES, num_factors))
    num_envs = _resolve_num_envs(args.num_envs, args.example)
    rollout_length = num_valid

    # --- START AUTO-TUNE ---
    num_devs = len(jax.devices()) if is_spmd else 1
    if not getattr(args, "strict_config", False) and num_devs > 1:
        old_envs = num_envs
        if num_envs % num_devs != 0:
            num_envs = max(num_devs, round(num_envs / num_devs) * num_devs)

        tw = num_envs * max(1, rollout_length - args.unroll_steps)
        base = tw // num_devs

        possible_m = [i for i in range(1, base + 1) if base % i == 0]
        if possible_m:
            best_m = min(possible_m, key=lambda x: abs(x - args.minibatches))
            if old_envs != num_envs or best_m != args.minibatches:
                print(f"\n[Auto-Tune] Optimizing for {num_devs} GPUs:")
                if old_envs != num_envs:
                    print(f"  * num_envs: {old_envs} -> {num_envs}")
                if args.minibatches != best_m:
                    print(f"  * minibatches: {args.minibatches} -> {best_m}")
                print(
                    f"  * Resulting batch size: {tw // best_m} ({tw // best_m // num_devs} per GPU)\n"
                )
            args.minibatches = best_m
        args.num_envs = num_envs
    # --- END AUTO-TUNE ---

    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)

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

    curriculum_stages = _parse_curriculum(getattr(args, "curriculum", ""))
    schedule = (
        make_curriculum_schedule(args, curriculum_stages)
        if curriculum_stages
        else optax.cosine_decay_schedule(
            args.lr,
            args.episodes * args.minibatches,
            args.lr_decay_min_mult,
        )
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, eps=args.adam_eps),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    if is_spmd:
        mesh = Mesh(np.array(jax.devices()), axis_names=("dev",))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        data_sharding = NamedSharding(mesh, PartitionSpec("dev"))
        # Add this: Shard dim 1 (batch), keep dim 0 (minibatch) unsharded for lax.scan
        scan_data_sharding = NamedSharding(mesh, PartitionSpec(None, "dev"))

        def shard_leaf(x):
            return jax.device_put(x, replicated_sharding) if eqx.is_array(x) else x

        agent = jax.tree.map(shard_leaf, agent)
        opt_state = jax.tree.map(shard_leaf, opt_state)
    else:
        data_sharding = None
        scan_data_sharding = None
        mesh = None

    key, eval_key = jrand.split(key)
    eval_samples = generate_eval_samples(env, eval_key, args.num_eval_samples)
    env_with_samples = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

    # Inject the CPU-approx Ray pool (if any) BEFORE building rollout_fn,
    # because make_rollout_fn closes over env_with_samples and JIT-traces
    # env.step() — which captures the Python object identity of the
    # ``tokenize()`` closure. With the pool installed, tokenize() returns
    # a closure that dispatches to Ray; without it, tokenize() returns
    # the inline ``_callback`` path. We don't want to JIT a path and
    # then change tokenize() later.
    #
    # ``_remote_pool`` rides in the env's tree_flatten aux_data (it's an
    # opaque Python object — Ray actor handles + a deque), so
    # round-tripping the pytree preserves it.
    if remote_pool is not None:
        children, aux_data = env_with_samples.tree_flatten()
        # aux_data layout: (config, valid_vertices, num_envs,
        #                   _remote_pool, _remote_timeout_s)
        aux_data = (
            aux_data[0], aux_data[1], aux_data[2],
            remote_pool, float(remote_timeout_s),
        )
        env_with_samples = type(env_with_samples).tree_unflatten(
            aux_data, children,
        )
        # ``ray.put`` the eval_samples tuple once and cache the ref on
        # the pool, so per-step ``actor.evaluate.remote(...)`` calls ship
        # only the (tiny) ObjectRef instead of re-serialising the same
        # multi-MB tuple of per-sample model inputs on every
        # io_callback. Convert to numpy first — JAX arrays serialise
        # through host RAM anyway, and we want the actor to receive a
        # plain numpy tuple matching ``CpuApproximationServer.evaluate``'s
        # ``eval_samples`` contract.
        eval_samples_np = tuple(np.asarray(x) for x in eval_samples)
        remote_pool.set_eval_samples(eval_samples_np)
    vertex_features = _episode_vertex_features(
        args,
        closed_jaxpr.jaxpr,
        tuple(closed_jaxpr.literals),
        tuple(xs),
        eval_samples=eval_samples,
        argnums=tuple(argnums),
    )

    env_states = jax.vmap(lambda _: env_with_samples.reset())(jnp.arange(num_envs))

    # Action Mask Helpers & Callbacks (Abridged core implementation matching mu0_ray_worker)
    def _build_action_mask_at_depth(emb, pin_rules):
        depth = emb.depth
        v_invalid = (
            jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
            .at[:total_v]
            .set(1.0 - emb.vertex_avail_mask)
        )
        p_invalid = jnp.where(
            pin_rules,
            jnp.ones((UNIFIED_ACTION_SIZE,), dtype=jnp.float32).at[PAIR_STOP].set(0.0),
            jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
            .at[:NUM_PAIR_CHOICES]
            .set(1.0 - pair_valid_mask[emb.vertex_idx]),
        )
        pair_k = emb.pair_seq[jnp.maximum((depth - 2) // 2, 0)]
        f_invalid = (
            jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
            .at[:num_factors]
            .set(1.0 - pair_factor_mask[emb.vertex_idx, pair_k])
        )
        return jnp.where(
            depth == 0,
            v_invalid,
            jnp.where((depth >= 1) & (depth % 2 == 1), p_invalid, f_invalid),
        )

    def _prior_at_depth(agent_local, emb, pin_rules):
        logits, value = agent_local.prediction(emb.latent)
        invalid = _build_action_mask_at_depth(emb, pin_rules)
        return jnp.where(invalid > 0.5, -1e9, logits), value

    def make_root_fn(pin_rules, agent_local):
        def root_fn(_params, _rng_key, embedding):
            prior, value = jax.vmap(
                lambda emb: _prior_at_depth(agent_local, emb, pin_rules)
            )(embedding)
            return mctx.RootFnOutput(
                prior_logits=prior, value=value, embedding=embedding
            )

        return root_fn

    def make_recurrent_fn(pin_rules, agent_local):
        def recurrent_step_single(action, embedding):
            depth, is_v, is_p, is_f = (
                embedding.depth,
                embedding.depth == 0,
                (embedding.depth >= 1) & (embedding.depth % 2 == 1),
                (embedding.depth >= 2) & (embedding.depth % 2 == 0),
            )
            n_v_idx = jnp.where(is_v, action.astype(jnp.int32), embedding.vertex_idx)
            n_p_seq = lax.cond(
                is_p,
                lambda: embedding.pair_seq.at[jnp.maximum((depth - 1) // 2, 0)].set(
                    action.astype(jnp.int32)
                ),
                lambda: embedding.pair_seq,
            )
            n_f_seq = lax.cond(
                is_f,
                lambda: embedding.factor_seq.at[jnp.maximum((depth - 2) // 2, 0)].set(
                    action.astype(jnp.int32)
                ),
                lambda: embedding.factor_seq,
            )
            n_active = jnp.where(
                is_p,
                embedding.active & (action.astype(jnp.int32) != PAIR_STOP),
                embedding.active,
            )
            next_latent, pred_reward = agent_local.dynamics(
                embedding.latent, action.astype(jnp.int32)
            )

            should_commit = (depth + 1) == DECISION_DEPTH
            next_emb = lax.cond(
                should_commit,
                lambda: _empty_decision(
                    next_latent,
                    embedding.vertex_avail_mask.at[n_v_idx].set(0.0),
                    max_rules,
                )._replace(last_reward=pred_reward),
                lambda: embedding._replace(
                    latent=next_latent,
                    depth=depth + 1,
                    vertex_idx=n_v_idx,
                    pair_seq=n_p_seq,
                    factor_seq=n_f_seq,
                    active=n_active,
                    last_reward=pred_reward,
                ),
            )
            n_prior, n_value = _prior_at_depth(agent_local, next_emb, pin_rules)
            return n_prior, n_value, next_emb

        def recurrent_fn(_params, _rng_key, action, embedding):
            n_prior, n_value, next_emb = jax.vmap(recurrent_step_single)(
                action, embedding
            )
            return mctx.RecurrentFnOutput(
                reward=next_emb.last_reward,
                discount=jnp.full_like(next_emb.last_reward, args.discount),
                prior_logits=n_prior,
                value=n_value,
            ), next_emb

        return recurrent_fn

    def make_rollout_fn(agent_local):
        @eqx.filter_jit
        @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None))
        def rollout_fn(
            agent_param,
            temperature,
            env_state,
            key,
            preference,
            vertex_features_local,
            reward_weights_local,
            pin_rules_jax,
        ):
            keys = jrand.split(key, rollout_length)
            cached_latent = agent_param.representation(
                env_state.tokens,
                eqn_ids=env_state.eqn_ids,
                preference=preference,
                vertex_features=vertex_features_local,
                key=keys[0],
                agg_key=keys[0],
            )

            def step_fn(carry, k):
                state, evolved_latent = carry
                s_key, t_key, e_k = jrand.split(k, 3)
                latent = (
                    evolved_latent
                    if args.cache_encoding
                    else agent_param.representation(
                        state.tokens,
                        eqn_ids=state.eqn_ids,
                        preference=preference,
                        vertex_features=vertex_features_local,
                        key=e_k,
                        agg_key=e_k,
                    )
                )
                v_avail = vertex_avail_at_step(
                    state, vertex_valid_static, total_v, num_valid
                ).astype(jnp.float32)
                decision_state = _empty_decision(latent, v_avail, max_rules)

                roots = make_root_fn(pin_rules_jax, agent_param)(
                    agent_param,
                    s_key,
                    jax.tree.map(lambda x: jnp.expand_dims(x, 0), decision_state),
                )
                policy_out = mctx.gumbel_muzero_policy(
                    params=agent_param,
                    rng_key=s_key,
                    root=roots,
                    recurrent_fn=make_recurrent_fn(pin_rules_jax, agent_param),
                    num_simulations=args.num_simulations,
                    invalid_actions=_build_action_mask_at_depth(
                        decision_state, pin_rules_jax
                    )[None, :],
                    max_num_considered_actions=args.gumbel_max_considered,
                    gumbel_scale=args.gumbel_scale,
                )
                mcts_visits, actions_path = extract_path_visits(
                    policy_out.search_tree, DECISION_DEPTH, t_key
                )

                v_idx, slot_indices = actions_path[0], jnp.arange(max_rules)
                p_seq, f_seq = (
                    actions_path[1 + 2 * slot_indices].astype(jnp.int32),
                    actions_path[2 + 2 * slot_indices].astype(jnp.int32),
                )

                is_stop = p_seq == PAIR_STOP
                has_stopped = jnp.cumsum(is_stop.astype(jnp.int32)) > 0
                base_final = jnp.where(has_stopped[:, None], -1, _PAIR_TO_BASE[p_seq])
                # Use the already-jnp ``factor_table`` (built once in
                # ``_build_factor_table`` above as ``jnp.array(..., int32)``)
                # rather than re-converting ``factor_table_np`` each scan
                # step. Identical semantics, ~5-8 s/episode saved at
                # rollout_length=50, num_envs=4 (was profiling-survey
                # finding #3 — safe).
                factor_final = jnp.where(has_stopped, 0, factor_table[f_seq])
                specs = jnp.concatenate([base_final, factor_final[:, None]], axis=-1).astype(jnp.int32)
                
                pad_len = MAX_RULES_PER_VERTEX - specs.shape[0]
                if pad_len > 0:
                    specs = jnp.concatenate([specs, jnp.tile(jnp.array([-1, -1, 0], dtype=jnp.int32), (pad_len, 1))], axis=0)
                else:
                    specs = specs[:MAX_RULES_PER_VERTEX]

                env_action = StepAction(
                    target_vertex=jnp.asarray(v_idx + 1, jnp.int32),
                    rule_specs=specs,
                )
                env_out = env_with_samples.step(state, env_action)

                next_latent_cached, _ = lax.scan(
                    lambda lat, idx: (
                        agent_param.dynamics(
                            agent_param.dynamics(lat, p_seq[idx])[0], f_seq[idx]
                        )[0],
                        None,
                    ),
                    agent_param.dynamics(latent, v_idx.astype(jnp.int32))[0],
                    jnp.arange(max_rules),
                )

                return (env_out.state, next_latent_cached), Trajectory(
                    tokens=state.tokens.astype(jnp.int32),
                    eqn_ids=state.eqn_ids.astype(jnp.int32),
                    vertex_idx=v_idx,
                    pair_seq=p_seq,
                    factor_seq=f_seq,
                    reward_vec=env_out.reward,
                    scalar_reward=jnp.sum(env_out.reward * reward_weights_local).astype(
                        jnp.float32
                    ),
                    mcts_visits=mcts_visits.astype(jnp.float32),
                    mcts_value=jnp.asarray(
                        policy_out.search_tree.summary().value[0], dtype=jnp.float32
                    ),
                    preference=preference.astype(jnp.float32),
                )

            return lax.scan(step_fn, (env_state, cached_latent), keys)[1]

        return rollout_fn

    def loss_fn(agent_local, batch):
        def unroll_loss(w):
            latent = agent_local.representation(
                w.tokens[0], eqn_ids=w.eqn_ids[0], preference=w.preference[0]
            )
            l_pi, l_v, l_r = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)
            for k in range(args.unroll_steps + 1):
                act_seq = (
                    jnp.zeros((DECISION_DEPTH,), jnp.int32).at[0].set(w.vertex_idx[k])
                )
                for s in range(max_rules):
                    act_seq = (
                        act_seq.at[1 + 2 * s]
                        .set(w.pair_seq[k][s])
                        .at[2 + 2 * s]
                        .set(w.factor_seq[k][s])
                    )
                rew_seq = (
                    jnp.zeros((DECISION_DEPTH,), jnp.float32)
                    .at[DECISION_DEPTH - 1]
                    .set(w.scalar_reward[k])
                )
                for d in range(DECISION_DEPTH):
                    logits, value = agent_local.prediction(latent)
                    l_pi += -jnp.sum(w.mcts_visits[k][d] * jnn.log_softmax(logits))
                    # Symlog the target value + reward (Pohlen squashing
                    # analogue) before the squared-error loss. Without
                    # this the cost-family rewards (~1e10 raw) blow the
                    # value head up: ``(value - target)^2 ≈ 1e20``
                    # observed in wandb run gtpk5xiz before this fix.
                    # PPO does the same on its legacy scalar path; this
                    # brings MuZero in line. The value / dynamics outputs
                    # are interpreted as symlog'd predictions everywhere
                    # downstream (MCTS uses them as comparable scalars
                    # under the monotone symlog, so the relative order
                    # of children is preserved).
                    l_v += 0.5 * jnp.square(value - _symlog(w.target_value[k]))
                    if not ((k == args.unroll_steps) and (d == DECISION_DEPTH - 1)):
                        latent, pred_r = agent_local.dynamics(latent, act_seq[d])
                        l_r += 0.5 * jnp.square(pred_r - _symlog(rew_seq[d]))
                        if d == DECISION_DEPTH - 1:
                            latent = 0.5 * latent + 0.5 * lax.stop_gradient(latent)
            return (
                l_pi + args.value_loss_weight * l_v + args.reward_loss_weight * l_r,
                (l_pi, l_v, l_r),
            )

        per_w_tot, (lp, lv, lr) = jax.vmap(unroll_loss)(batch)
        return jnp.mean(per_w_tot), (jnp.mean(lp), jnp.mean(lv), jnp.mean(lr))

    @eqx.filter_jit
    def train_minibatches_scanned(agent_local, opt_state_local, all_batches):
        dynamic_carry, static_carry = eqx.partition(
            (agent_local, opt_state_local), eqx.is_array
        )

        def scan_body(dyn_carry, batch_i):
            agent_c, opt_state_c = eqx.combine(dyn_carry, static_carry)

            (loss_val, parts), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                agent_c, batch_i
            )
            # Phase 4e: NaN-skip — if loss isn't finite (calibration
            # poisoning, cold-cache compile error, or any value-head
            # blowup that grad clipping at max_grad_norm couldn't
            # rescue), keep agent + opt_state unchanged so a single bad
            # batch doesn't propagate through the rest of the minibatch
            # scan. Increments a per-call counter the driver can log.
            loss_finite = jnp.isfinite(loss_val)
            zero_grads = jax.tree.map(jnp.zeros_like, grads)
            safe_grads = jax.tree.map(
                lambda g, z: jnp.where(loss_finite, g, z), grads, zero_grads,
            )
            updates, opt_state_new = optimizer.update(
                safe_grads, opt_state_c, eqx.filter(agent_c, eqx.is_inexact_array)
            )
            opt_state_new = jax.tree.map(
                lambda new, old: jnp.where(loss_finite, new, old),
                opt_state_new, opt_state_c,
            )

            new_agent = eqx.apply_updates(agent_c, updates)
            new_dyn_carry, _ = eqx.partition((new_agent, opt_state_new), eqx.is_array)

            return new_dyn_carry, (loss_val, parts, jnp.where(loss_finite, 0, 1).astype(jnp.int32))

        final_dyn_carry, (losses, parts, nan_flags) = lax.scan(
            scan_body, dynamic_carry, all_batches
        )
        final_agent, final_opt = eqx.combine(final_dyn_carry, static_carry)

        return (
            final_agent, final_opt,
            jnp.mean(losses), jax.tree.map(jnp.mean, parts),
            jnp.sum(nan_flags),
        )

    return {
        "args": args,
        "variant": variant,
        "env": env_with_samples,
        "env_states": env_states,
        "vertex_features": vertex_features,
        "factor_table_np": factor_table_np,
        "max_rules": max_rules,
        "DECISION_DEPTH": DECISION_DEPTH,
        "UNIFIED_ACTION_SIZE": UNIFIED_ACTION_SIZE,
        "num_envs": num_envs,
        "rollout_length": rollout_length,
        "reward_weights": reward_weights,
        "agent": agent,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "rollout_fn": make_rollout_fn(agent),
        "train_minibatches": train_minibatches_scanned,
        "scan_data_sharding": scan_data_sharding,
        "key": key,
        "data_sharding": data_sharding,
        "mesh": mesh,
    }


class SPMDServerWorker:
    def __init__(
        self,
        args_dict: dict,
        variant: str,
        seed: int = 0,
        cpu_workers: list = None,
        *,
        callback_timeout_s: float = 120.0,
        initial_timeout_s: float | None = None,
        warm_after: int = 3,
        recycle_every: int = 50,
        cpu_actor_options: dict | None = None,
        starting_actor_id: int = 1000,
    ):
        # Build the CPU-approx pool first (if any cpu_workers were
        # provided) so we can hand it to ``_build_actor_state`` and have
        # it baked into the env BEFORE rollout_fn JIT-traces. The pool
        # itself is JAX-free; building it doesn't trigger any compile.
        pool = None
        self._pool = None
        self._recycle_every = int(recycle_every) if recycle_every else 0
        self._callback_timeout_s = float(callback_timeout_s)
        self._episodes_since_recycle = 0
        if cpu_workers:
            from alphagrad.approx.cpu_approx_pool import CpuApproxPool
            from alphagrad.approx.env import (
                MAX_TOKENS as _MAX_TOKENS,
                NUM_REWARDS as _NUM_REWARDS,
                REWARD_INDEX as _REWARD_INDEX,
            )
            from alphagrad.approx.mu0_ray_actors import CPUApproximationActor

            # Counter starts above the initial pool's IDs (driver gave
            # 0..N-1) so log lines are visually distinguishable. We use
            # a mutable single-element list so the closure can mutate
            # it without ``nonlocal``.
            _actor_counter = [int(starting_actor_id)]
            _opts = dict(cpu_actor_options or {})
            _args_dict = args_dict
            _variant = variant

            def _respawn_factory():
                aid = _actor_counter[0]
                _actor_counter[0] += 1
                return CPUApproximationActor.options(**_opts).remote(
                    _args_dict, _variant, aid,
                )

            pool = CpuApproxPool(
                cpu_workers,
                timeout_s=callback_timeout_s,
                initial_timeout_s=initial_timeout_s,
                warm_after=warm_after,
                respawn_factory=_respawn_factory,
                max_tokens=_MAX_TOKENS,
                num_rewards=_NUM_REWARDS,
                cosine_sim_idx=_REWARD_INDEX["cosine_sim"],
                frob_residual_idx=_REWARD_INDEX["frob_residual"],
            )
            self._pool = pool

        self.state = _build_actor_state(
            args_dict,
            variant,
            seed,
            is_spmd=True,
            cpu_workers=cpu_workers,
            remote_pool=pool,
            remote_timeout_s=callback_timeout_s,
        )
        self.args = self.state["args"]
        self.variant = variant
        self._key = self.state["key"]
        self._pin_rules_default = _pin_rules_for_variant(variant)
        self.replay_buffer = None
        self.train_step_counter = 0
        self._checkpoint_path = getattr(self.args, "checkpoint_path", "") or ""
        self._checkpoint_every = int(getattr(self.args, "checkpoint_every", 0))

        # Optional resume from a previously-written checkpoint. Same
        # contract as the PPO worker (see ppo_ray_worker.py):
        # eqx.tree_deserialise_leaves needs templates from the freshly-
        # built agent + opt_state, so we restore right after the state
        # dict is constructed.
        if self._checkpoint_path:
            try:
                from alphagrad.approx.common.checkpoint import (
                    install_sigterm_handler, load_state,
                )
                restored = load_state(
                    self._checkpoint_path,
                    template_agent=self.state["agent"],
                    template_opt_state=self.state["opt_state"],
                )
                if restored is not None:
                    if restored["agent"] is not None:
                        self.state["agent"] = restored["agent"]
                    if restored["opt_state"] is not None:
                        self.state["opt_state"] = restored["opt_state"]
                    self.train_step_counter = int(restored["episode_counter"])
                    if restored["reward_weights"] is not None:
                        self.state["reward_weights"] = jnp.asarray(
                            restored["reward_weights"], dtype=jnp.float32,
                        )
                    print(
                        f"[mu0_ray_worker] resumed from {self._checkpoint_path} "
                        f"at train_step {self.train_step_counter}"
                    )
                install_sigterm_handler(self._save_checkpoint_safe)
            except Exception as exc:
                print(f"[mu0_ray_worker] checkpoint resume failed: {exc}")

    def run_rollout_and_train(
        self,
        rng_seed: int,
        preference_np: Any = None,
        pin_rules: Any = None,
        reset_env: bool = True,
        train_steps: int = 1,
    ) -> dict:
        pr = jnp.asarray(
            bool(pin_rules if pin_rules is not None else self._pin_rules_default),
            dtype=jnp.bool_,
        )
        if reset_env:
            self.state["env_states"] = jax.vmap(lambda _: self.state["env"].reset())(
                jnp.arange(self.state["num_envs"])
            )

        pref = (
            jnp.zeros((self.state["num_envs"], NUM_REWARDS), jnp.float32)
            if preference_np is None
            else jnp.asarray(preference_np, jnp.float32)
        )

        keys = jrand.split(jrand.PRNGKey(int(rng_seed)), self.state["num_envs"])

        ds = self.state.get("data_sharding")
        mesh = self.state.get("mesh")
        num_devs = len(jax.devices()) if ds is not None else 1

        @contextlib.contextmanager
        def active_mesh():
            if mesh is not None:
                with mesh:
                    yield
            else:
                yield

        scan_ds = self.state.get("scan_data_sharding")

        with active_mesh():
            traj = self.state["rollout_fn"](
                self.state["agent"],
                jnp.asarray(self.args.temperature, jnp.float32),
                self.state["env_states"],
                keys,
                pref,
                self.state["vertex_features"],
                self.state["reward_weights"],
                pr,
            )

        rw_np = np.asarray(self.state["reward_weights"])
        r_vec_np = np.asarray(traj.reward_vec)
        # Sentinel-aware per-env per-channel return: any transition
        # whose cost channel == SENTINEL_REWARD_VALUE (the -1e10
        # cpu_approx_pool timeout marker) is zeroed out of the sum so
        # the running per-channel "best" doesn't latch onto a sentinel
        # trajectory and report `flop=-1e+10` as if it were a real
        # value. Mirrors the PPO worker's aggregate_per_channel_stats
        # path (alphagrad/src/alphagrad/approx/ppo_ray_worker.py).
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            filter_sentinel_mask as _filter_sentinel_mask,
        )

        # r_vec_np shape: (num_envs, T, NUM_REWARDS). Mask is per
        # (env, t); zeroed rows don't contribute to the per-env sum.
        valid_mask = _filter_sentinel_mask(r_vec_np, SENTINEL_REWARD_VALUE)
        masked_r_vec = np.where(valid_mask[:, :, None], r_vec_np, 0.0)
        r_per_env = masked_r_vec.sum(axis=1)  # (num_envs, NUM_REWARDS)
        weighted_per_env = r_per_env * rw_np  # (num_envs, NUM_REWARDS)
        per_env_tot = weighted_per_env.sum(axis=-1)  # (num_envs,)
        best_idx = int(per_env_tot.argmax())

        v_np = np.asarray(traj.vertex_idx)  # (E, T)
        p_np = np.asarray(traj.pair_seq)    # (E, T, max_rules)
        f_np = np.asarray(traj.factor_seq)  # (E, T, max_rules)
        ftab = self.state["factor_table_np"]

        # For each *tuned* reward channel (non-zero weight), find the env
        # whose raw per-channel return is highest and grab its action
        # sequence. Channels with zero weight are skipped to keep the
        # per-reward report focused on what the user is actually
        # optimising.
        best_per_reward = {}
        for j in range(NUM_REWARDS):
            if float(rw_np[j]) == 0.0:
                continue
            bidx = int(r_per_env[:, j].argmax())
            # Full 8-channel snapshot of the env that won this channel
            # — mirrors PPO's aggregate_per_channel_stats output so the
            # shared JSON dump (best_sequences_snapshot) records
            # ``(a_i, b_i, c_i, r_i)`` for every per-channel best.
            all_raw = {
                REWARD_NAMES[k]: float(r_per_env[bidx, k])
                for k in range(NUM_REWARDS)
            }
            all_weighted = {
                REWARD_NAMES[k]: float(weighted_per_env[bidx, k])
                for k in range(NUM_REWARDS)
            }
            best_per_reward[REWARD_NAMES[j]] = {
                "raw_value": float(r_per_env[bidx, j]),
                "weighted_value": float(weighted_per_env[bidx, j]),
                "weighted_total": float(per_env_tot[bidx]),
                "env_idx": bidx,
                "seq": _action_to_pylist(v_np[bidx], p_np[bidx], f_np[bidx], ftab),
                "all_raw": all_raw,
                "all_weighted": all_weighted,
            }

        # Mean MCTS visit-distribution entropy. ``mcts_visits`` has shape
        # ``(E, T, DECISION_DEPTH, UNIFIED_ACTION_SIZE)`` and is the raw
        # visit count per action at every decision node along the path.
        # Normalise within each (env, step, depth) row, then ``-sum(p log p)``
        # over the action axis. Mean is taken over the valid entries
        # (rows where total visits > 0 — terminal/masked positions have
        # zero visits and contribute nothing).
        mv = np.asarray(traj.mcts_visits)  # (E, T, D, A)
        v_sum = mv.sum(axis=-1, keepdims=True)
        probs = mv / np.maximum(v_sum, 1e-8)
        ent = -np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0).sum(axis=-1)
        valid = (v_sum.squeeze(-1) > 0).astype(np.float32)
        ent_total = (ent * valid).sum()
        valid_count = max(float(valid.sum()), 1.0)
        mean_entropy = float(ent_total / valid_count)
        # Entropy at depth 0 specifically — that's the strategic vertex
        # choice, before pair/factor sub-decisions. Useful to see
        # exploration of *which vertex to eliminate next* separately
        # from the rule sub-policy entropy.
        root_valid = valid[..., 0]
        root_count = max(float(root_valid.sum()), 1.0)
        root_entropy = float((ent[..., 0] * root_valid).sum() / root_count)

        # Per-channel raw rewards of the overall-best (highest weighted-sum)
        # trajectory this episode. Distinct from ``best_per_reward`` —
        # that one optimises each channel independently and may pick a
        # different env per channel. ``best_overall_rewards`` is the
        # cross-section "for the env that won the overall scalar".
        best_overall_rewards = {
            REWARD_NAMES[j]: float(r_per_env[best_idx, j])
            for j in range(NUM_REWARDS)
        }
        best_overall_weighted = {
            REWARD_NAMES[j]: float(weighted_per_env[best_idx, j])
            for j in range(NUM_REWARDS)
        }

        per_reward_means = (
            {
                REWARD_NAMES[j]: float(
                    r_vec_np[..., j][valid_mask].mean()
                )
                for j in range(NUM_REWARDS)
            }
            if valid_mask.any()
            else {REWARD_NAMES[j]: 0.0 for j in range(NUM_REWARDS)}
        )
        # Terminal-step rewards (each env's last transition). mu0
        # trajectories are fixed-length per env, so the per-env
        # terminal step is always ``T - 1``. ``r_vec_np`` is
        # (num_envs, T, NUM_REWARDS); slice the last step per env.
        if r_vec_np.shape[1] > 0:
            term_rewards = r_vec_np[:, -1, :]  # (num_envs, NUM_REWARDS)
            term_valid_env = valid_mask[:, -1] if valid_mask.shape[1] > 0 else np.ones(
                r_vec_np.shape[0], dtype=bool
            )
            if term_valid_env.any():
                term_mean = term_rewards[term_valid_env].mean(axis=0)
                term_max = term_rewards[term_valid_env].max(axis=0)
            else:
                term_mean = np.zeros((NUM_REWARDS,), dtype=np.float32)
                term_max = np.zeros((NUM_REWARDS,), dtype=np.float32)
            terminal_means = {
                REWARD_NAMES[j]: float(term_mean[j]) for j in range(NUM_REWARDS)
            }
            best_terminal = {
                REWARD_NAMES[j]: float(term_max[j]) for j in range(NUM_REWARDS)
            }
            terminal_cs = term_rewards[term_valid_env, REWARD_INDEX["cosine_sim"]]
        else:
            terminal_means = {n: 0.0 for n in REWARD_NAMES}
            best_terminal = {n: 0.0 for n in REWARD_NAMES}
            terminal_cs = np.zeros((0,), dtype=np.float32)

        stats = {
            "best_return": float(per_env_tot[best_idx]),
            "mean_return": float(per_env_tot.mean()),
            "best_seq": _action_to_pylist(
                v_np[best_idx], p_np[best_idx], f_np[best_idx], ftab,
            ),
            "best_overall_rewards": best_overall_rewards,
            "best_overall_weighted": best_overall_weighted,
            "per_reward_means": per_reward_means,
            "terminal_means": terminal_means,
            "best_terminal": best_terminal,
            "best_per_reward": best_per_reward,
            "entropy_mean": mean_entropy,
            "entropy_root": root_entropy,
        }
        # Unified `reward/{cost,quality}/{per_step,terminal,best_terminal}/<name>`
        # keys + corridor instrumentation. Legacy `reward_mean/*` keys
        # (emitted by the driver from ``per_reward_means``) remain.
        from alphagrad.approx.common.reward_scaling import (
            build_unified_reward_log_dict,
        )
        stats.update(
            build_unified_reward_log_dict(
                stats,
                corridor_low=getattr(self, "_corridor_low", None),
                corridor_high=getattr(self, "_corridor_high", None),
                terminal_cossims=terminal_cs,
            )
        )

        if self.replay_buffer is None and self.args.replay_buffer_size > 0:
            self.replay_buffer = init_replay_buffer(
                jax.tree_util.tree_map(lambda x: x[0], traj),
                self.args.replay_buffer_size,
            )
            if self.args.replay_checkpoint_path and os.path.exists(
                self.args.replay_checkpoint_path
            ):
                self.replay_buffer = load_replay_buffer(
                    self.args.replay_checkpoint_path, self.replay_buffer
                )

        if self.replay_buffer is not None:
            self.replay_buffer = replay_add_batch(
                self.replay_buffer,
                traj,
                priorities=_compute_traj_priorities(traj, self.state["reward_weights"]),
            )

        with active_mesh():
            for _ in range(train_steps):
                if self.replay_buffer is not None and int(
                    self.replay_buffer.size
                ) >= max(self.args.replay_warmup * self.state["num_envs"], 1):
                    s_key, self._key = jrand.split(self._key)
                    t_traj = replay_sample(
                        self.replay_buffer,
                        self.args.replay_batch_size
                        if self.args.replay_batch_size > 0
                        else self.state["num_envs"],
                        s_key,
                        alpha=self.args.replay_priority_alpha,
                    )
                    # Per-channel discount: cosine_sim and frob_residual
                    # are terminal quality metrics (env emits them only at
                    # the final step), so γ=1.0 for those channels and
                    # ``args.discount`` for the per-step cmp/mem channels.
                    # Reads ``reward_vec`` (B, T, NUM_REWARDS) directly
                    # instead of the already-aggregated ``scalar_reward``
                    # so we can apply different γ per channel before the
                    # final sum.
                    rw = self.state["reward_weights"]
                    t_vals = jax.vmap(
                        lambda rv: _per_channel_discounted_returns(
                            rv, rw, self.args.discount,
                        )
                    )(t_traj.reward_vec)

                    w_batch = jax.tree_util.tree_map(
                        lambda *xs: jnp.stack(xs, axis=1),
                        *[
                            TrajectoryWindow(
                                tokens=t_traj.tokens[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                eqn_ids=t_traj.eqn_ids[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                vertex_idx=t_traj.vertex_idx[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                pair_seq=t_traj.pair_seq[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                factor_seq=t_traj.factor_seq[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                scalar_reward=t_traj.scalar_reward[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                target_value=t_vals[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                mcts_visits=t_traj.mcts_visits[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                                preference=t_traj.preference[
                                    :, i : i + self.args.unroll_steps + 1
                                ],
                            )
                            for i in range(
                                self.state["rollout_length"] - self.args.unroll_steps
                            )
                        ],
                    )

                    sh_key, self._key = jrand.split(self._key)
                    batches = _shuffle_and_batch_windows(
                        w_batch, self.args.minibatches, sh_key
                    )

                    b_size = jax.tree_util.tree_leaves(batches)[0].shape[1]

                    if scan_ds is not None and b_size % num_devs == 0:
                        batches = jax.tree.map(
                            lambda x: jax.device_put(x, scan_ds), batches
                        )

                    (
                        self.state["agent"],
                        self.state["opt_state"],
                        last_loss,
                        last_parts,
                        nan_skip_count,
                    ) = self.state["train_minibatches"](
                        self.state["agent"],
                        self.state["opt_state"],
                        batches,
                    )

                    self.train_step_counter += self.args.minibatches
                    stats.update(
                        {
                            "policy_loss": float(last_parts[0]),
                            "value_loss": float(last_parts[1]),
                            "reward_loss": float(last_parts[2]),
                            "total_loss": float(last_loss),
                            "nan_skip_count": int(nan_skip_count),
                        }
                    )

        stats.update(
            {
                "buffer_size": int(self.replay_buffer.size)
                if self.replay_buffer
                else 0,
                "train_step": self.train_step_counter,
            }
        )

        # Expose pool telemetry + maybe recycle. The pool is the
        # only piece of state in this worker whose memory grows
        # unboundedly with episode count (each actor accumulates
        # ``cost_analysis()`` C++ state per call; ~9 MB × hundreds
        # of calls ≈ GB-scale residual per ep). Recycle bounds it.
        if self._pool is not None:
            stats.update({f"pool/{k}": v for k, v in self._pool.stats().items()})
            # Per-episode timeout delta (see PPO equivalent — sentinel-fire
            # detector; should stay at 0 with the pool=num_envs fix).
            try:
                stats["pool/timeouts_this_episode"] = (
                    self._pool.fetch_timeout_delta()
                )
            except Exception:
                stats["pool/timeouts_this_episode"] = 0
            # Mirror PPO: pop per-rollout jaxpr-tokenization truncation
            # counters from each CPU actor's per-process state. All
            # values are PER-EPISODE (the actor counters reset on
            # consume). Best-effort — a Ray hiccup contributes zero.
            try:
                trunc = self._pool.fetch_tokenization_truncation_stats()
                count = int(trunc.get("count", 0))
                overflow_sum = int(trunc.get("overflow_sum", 0))
                max_len = int(trunc.get("max_observed_len", 0))
            except Exception:
                count = 0
                overflow_sum = 0
                max_len = 0
            stats["tokenization/truncated_count"] = count
            stats["tokenization/overflow_sum_this_ep"] = overflow_sum
            stats["tokenization/mean_overflow_per_trunc"] = (
                float(overflow_sum / count) if count > 0 else 0.0
            )
            stats["tokenization/max_observed_len"] = max_len
            if self._recycle_every > 0:
                # Cascading recycle (mirror of PPO worker): kill ONE
                # actor every ``recycle_every / N`` episodes so the
                # pool rotates over the full ``recycle_every`` window
                # but the memory spike is smeared. Mathematically
                # equivalent to the old all-at-once ``recycle()`` for
                # per-actor lifetime; just quieter at the system
                # level.
                n_pool = max(self._pool.size(), 1)
                interval = max(1, self._recycle_every // n_pool)
                if (
                    self.train_step_counter > 0
                    and (self.train_step_counter // max(self.args.minibatches, 1))
                    % interval == 0
                ):
                    new_size = self._pool.recycle_one()
                    stats["pool/recycled_at_step"] = self.train_step_counter
                    stats["pool/size_after_recycle"] = new_size

        # Periodic checkpoint (PPO mirror). The SIGTERM hook installed
        # at __init__ also fires a final save on slurm timeout — this
        # one bounds the progress lost on a non-clean exit.
        if (
            self._checkpoint_path
            and self._checkpoint_every > 0
            and self.train_step_counter > 0
            and self.train_step_counter // max(self.args.minibatches, 1) % self._checkpoint_every == 0
        ):
            self._save_checkpoint_safe()
            stats["checkpoint/saved_at_step"] = self.train_step_counter

        return stats

    def get_pool_stats(self) -> dict:
        """Driver-side polling hook for the CPU-approx pool. Empty
        when no pool is attached."""
        if self._pool is None:
            return {}
        return self._pool.stats()

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Per-channel calibration statistics over `num_rollouts`
        zero-pref rollouts, with sentinel transitions filtered out.

        Returns a dict with per-channel arrays (shape ``(NUM_REWARDS,)``):
          * ``mean`` — per-channel mean (legacy ``mean_abs(symlog)`` path)
          * ``median``, ``q25``, ``q75`` — raw-space quartiles
          * ``median_symlog``, ``q25_symlog``, ``q75_symlog`` — symlog-space
            quartiles so the driver can compute IQR-on-symlog
          * ``count`` — number of valid samples (scalar)

        Phase 4b: previously this averaged across ALL transitions and
        a single cold-cache timeout could drag the per-channel mean
        way down. Sentinel filtering protects against that; switching
        to a robust statistic (IQR) on top further insulates the
        calibration from outliers.
        """
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            filter_sentinel_mask,
            symlog_np,
        )

        all_rows: list[np.ndarray] = []
        ds = self.state.get("data_sharding")
        mesh = self.state.get("mesh")
        num_devs = len(jax.devices()) if ds is not None else 1

        @contextlib.contextmanager
        def active_mesh():
            if mesh is not None:
                with mesh:
                    yield
            else:
                yield

        with active_mesh():
            for i in range(num_rollouts):
                pref = jnp.zeros((self.state["num_envs"], NUM_REWARDS), jnp.float32)
                env_states = jax.vmap(lambda _: self.state["env"].reset())(
                    jnp.arange(self.state["num_envs"])
                )
                keys = jrand.split(
                    jrand.PRNGKey(int(rng_seed) + i * 31 + 1), self.state["num_envs"]
                )

                if ds is not None and self.state["num_envs"] % num_devs == 0:
                    pref = jax.device_put(pref, ds)
                    env_states = jax.tree.map(
                        lambda x: jax.device_put(x, ds), env_states
                    )
                    keys = jax.device_put(keys, ds)

                traj = self.state["rollout_fn"](
                    self.state["agent"],
                    jnp.asarray(self.args.temperature, jnp.float32),
                    env_states,
                    keys,
                    pref,
                    self.state["vertex_features"],
                    self.state["reward_weights"],
                    jnp.asarray(self._pin_rules_default, dtype=jnp.bool_),
                )
                rv = np.asarray(traj.reward_vec)  # (N, T, NUM_REWARDS)
                flat = rv.reshape(-1, NUM_REWARDS)
                mask = filter_sentinel_mask(flat, SENTINEL_REWARD_VALUE)
                if mask.any():
                    all_rows.append(flat[mask])
        if not all_rows:
            zeros = np.zeros((NUM_REWARDS,), dtype=np.float32)
            return {
                "mean": zeros, "median": zeros, "q25": zeros, "q75": zeros,
                "median_symlog": zeros, "q25_symlog": zeros, "q75_symlog": zeros,
                "count": 0,
            }
        samples = np.concatenate(all_rows, axis=0).astype(np.float32)
        samples_sl = symlog_np(samples).astype(np.float32)
        return {
            "mean": samples.mean(axis=0).astype(np.float32),
            "median": np.median(samples, axis=0).astype(np.float32),
            "q25": np.quantile(samples, 0.25, axis=0).astype(np.float32),
            "q75": np.quantile(samples, 0.75, axis=0).astype(np.float32),
            "median_symlog": np.median(samples_sl, axis=0).astype(np.float32),
            "q25_symlog": np.quantile(samples_sl, 0.25, axis=0).astype(np.float32),
            "q75_symlog": np.quantile(samples_sl, 0.75, axis=0).astype(np.float32),
            "count": int(samples.shape[0]),
        }

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        self.state["reward_weights"] = jnp.asarray(weights_np, dtype=jnp.float32)

    def checkpoint_replay(self, path: str) -> None:
        if self.replay_buffer is not None and path:
            save_replay_buffer(self.replay_buffer, path)

    def _save_checkpoint_safe(self) -> None:
        """Wrapped `save_state` that never raises. Used by the periodic
        save below and the SIGTERM handler installed at __init__.
        """
        if not self._checkpoint_path:
            return
        try:
            from alphagrad.approx.common.checkpoint import save_state
            save_state(
                self._checkpoint_path,
                agent=self.state["agent"],
                opt_state=self.state["opt_state"],
                episode_counter=self.train_step_counter,
                reward_weights=np.asarray(self.state["reward_weights"]),
                multipliers=None,
                replay_buffer=self.replay_buffer,
                extras={"variant": str(self.variant)},
            )
        except Exception as exc:
            print(f"[mu0_ray_worker] checkpoint save failed: {exc}")

    def ready(self) -> bool:
        return True


class CPUApproximationWorker:
    """Real CPU-side approximation worker — wraps :class:`CpuApproximationServer`.

    Was previously a stub returning ``{}``. The SPMDActor spawns one of
    these per pool slot in :func:`alphagrad.approx.mu0_ray._run_one_variant`,
    and the env's ``io_callback`` dispatches per-step ``(order, specs,
    step)`` requests here via the :class:`CpuApproxPool` rather than
    running ``jax.jit(jacve(...)).lower().compile()`` inline on the
    GPU-owning SPMD process.

    Each worker owns its own ``VertexEliminationEnv`` instance and
    eval-samples bundle, built lazily on first ``evaluate`` so actor
    spawn doesn't block on the (potentially expensive)
    ``generate_eval_samples`` call.
    """

    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        self.args_dict = args_dict
        self.variant = variant
        self.actor_id = actor_id
        self._server = None  # lazy-init in ``_ensure_server``

    def _ensure_server(self):
        if self._server is None:
            from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
            self._server = CpuApproximationServer.from_args_dict(
                self.args_dict,
                variant=self.variant,
                seed=int(self.args_dict.get("seed", 0)) + self.actor_id,
            )
        return self._server

    def compile_approximations(self) -> dict:
        """Pre-warm the worker's compile cache.

        Called once at startup by the driver — the first ``evaluate``
        call will trigger an expensive XLA compile, so warm it eagerly
        so the first real rollout doesn't pay that cost on the
        critical path. Returns a small status dict.
        """
        srv = self._ensure_server()
        return {"status": "ready", "actor_id": self.actor_id,
                "valid_vertices": len(srv._env.valid_vertices)}

    def evaluate(
        self,
        order,
        sparsity_specs,
        step,
        eval_samples=None,
        init: bool = False,
    ):
        """Run one ``_callback`` evaluation out-of-process.

        Mirrors the signature ``env._callback`` is invoked with via
        ``io_callback``. Returns ``(tokens, eqn_ids, reward)`` numpy
        arrays. See :meth:`CpuApproximationServer.evaluate` for the
        sentinel-on-error contract.
        """
        srv = self._ensure_server()
        return srv.evaluate(
            order, sparsity_specs, step,
            eval_samples=eval_samples, init=init,
        )

    def reset_caches(self) -> dict:
        srv = self._ensure_server()
        return srv.reset_caches()

    def ready(self) -> bool:
        # Don't force ``_ensure_server`` here — Ray uses ``ready()`` as
        # a "did the actor's __init__ complete" probe, and we want
        # that to return fast (~ms). Lazy init of the env is fine.
        return True
