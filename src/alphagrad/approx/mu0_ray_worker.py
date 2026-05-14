"""Gumbel MuZero training kernels (JAX side).

Defines plain Python classes :class:`LearnerWorker` and
:class:`RolloutWorker` that hold the JAX state for off-policy Gumbel
MuZero training. They are wrapped as Ray actors in
:mod:`alphagrad.approx.mu0_ray_actors`, which is a JAX-free shim — the
driver in :mod:`alphagrad.approx.mu0_ray` imports only the shim, so JAX
never gets imported into the driver process. (Mixing Ray's actor-spawn
machinery with JAX in the same process is known to deadlock on
``fork``-based platforms; isolating JAX to worker processes is the fix.)

Importing this module triggers JAX initialisation — only call it from
within Ray actor processes.

Two worker classes:

* :class:`LearnerWorker` — owns the replay buffer, the agent params,
  and the optimiser state. The N rollout workers push fresh trajectories
  into the buffer with ``add_trajectories`` and the driver pulls weight
  refreshes with ``get_params_numpy``. ``train_step`` samples a
  minibatch from the buffer and runs one gradient update.

* :class:`RolloutWorker` — holds a stale-by-design copy of the agent
  params and runs ``rollout_one`` to produce a Trajectory pytree on the
  host. The worker builds the env + vmapped rollout function once at
  startup; subsequent calls just plug in fresh keys and the latest
  params.

The rollout / loss / training code mirrors the corresponding paths in
:mod:`alphagrad.approx.mu0` but is restricted to the
``--mcts-mode=gumbel`` branch — the user picked Gumbel MuZero for this
migration, so the hierarchical and sampled paths are not reproduced
here. Other ports (cache-encoding, set-transformer agg, Lagrangian
constraints, curriculum LR, reward calibration) are preserved.

The off-policy posture: rollouts arrive at the learner asynchronously
via ``add_trajectories``. ``train_step`` always samples from the replay
buffer (mixing in fresh rollouts is gated by ``--replay-fresh-fraction``,
default 0.0 in the driver). The buffer lives inside the learner worker
— the N→1 fan-in is the natural serialisation point and avoids an extra
host↔device roundtrip per training batch that a separate buffer actor
would force.
"""

from __future__ import annotations

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
    reward_normalization_fn,
    sample_preferences,
    save_replay_buffer,
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
from alphagrad.approx.mu0 import (
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    _NO_SYMLOG_MASK_NP,
    _PAIR_TO_BASE,
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
    _resolve_main_device,
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


# ---------------------------------------------------------------------------
# Shared JAX-side setup
# ---------------------------------------------------------------------------


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Reconstruct an argparse-Namespace-like object from a plain dict.

    Ray serialises args across the driver→actor boundary as a plain dict
    (so the Namespace's class doesn't need to be importable on both sides
    of the boundary). Inside the actor we want attribute access, so wrap
    the dict in a SimpleNamespace.
    """
    return SimpleNamespace(**args_dict)


def _build_actor_state(args_dict: dict, variant: str, actor_seed: int) -> dict:
    """Build the JAX-side training state shared by both actor classes.

    Returns a dict with:
      * ``args`` (Namespace), ``variant`` (str)
      * ``env``, ``env_states`` (batched), ``vertex_features`` cached
      * ``agent``, ``optimizer``, ``opt_state``
      * ``rollout_fn`` (JIT-compiled), ``train_minibatch`` (JIT-compiled)
      * sizing scalars (``num_envs``, ``rollout_length``, ``DECISION_DEPTH``,
        ``UNIFIED_ACTION_SIZE``, ``max_rules``, ``num_factors``, ``total_v``,
        ``num_valid``)
      * ``factor_table_np`` for decoding actions on the host
      * ``reward_weights`` (mutable; replaced on ``set_reward_weights``)
      * ``constraint_specs`` and Lagrangian state
      * ``key`` (a PRNG key state — each rollout/train step splits from it)
    """
    args = _args_from_dict(args_dict)
    _apply_variant_preset(args, variant)

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    # Don't touch CUDA_VISIBLE_DEVICES — Ray manages GPU isolation per actor.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _setup_jax_compile_cache()

    key = jrand.PRNGKey(actor_seed)
    key, args_key = jrand.split(key)

    # ---- Env ----
    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = (
        dataset_arg is not None and args.example.endswith("NeuralNetwork")
    )
    dataset_for_call = dataset_arg if use_dataset else None
    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(
        args.example, dataset=dataset_for_call, dataset_size=args.dataset_size,
    )
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
    vertex_valid_static = build_vertex_valid_static(env.valid_vertices, total_v)
    pair_valid_mask = build_pair_valid_mask(
        closed_jaxpr.jaxpr, total_v,
        num_pair_choices=NUM_PAIR_CHOICES, pair_stop_idx=PAIR_STOP,
        disable_sparsification=args.disable_sparsification,
    )

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args)
    factor_table_np = np.array(factors_py, dtype=np.int32)
    # pair_factor_mask: all-ones placeholder; the env-side legality check
    # has already pruned invalid pairs via pair_valid_mask, and the legacy
    # path doesn't expose a per-pair factor-legality grid.
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    DECISION_DEPTH = 1 + 2 * max_rules
    UNIFIED_ACTION_SIZE = int(max(total_v, NUM_PAIR_CHOICES, num_factors))

    num_envs = _resolve_num_envs(args.num_envs, args.example)
    rollout_length = num_valid

    if (num_envs * rollout_length) // args.minibatches == 0:
        raise ValueError(
            f"--minibatches={args.minibatches} > num_envs * rollout_length "
            f"({num_envs} * {rollout_length}). Lower --minibatches or raise "
            "--num-envs."
        )

    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)

    # ---- Lagrangian constraint state ----
    user_constraints = list(args.lagrangian_constraint)
    if args.cosine_lower_bound > 0.0:
        user_constraints.append(f"cosine_sim>={args.cosine_lower_bound}")
    if args.cosine_upper_bound < 1.0:
        user_constraints.append(f"cosine_sim<={args.cosine_upper_bound}")
    constraint_specs = parse_lagrangian_constraints(user_constraints)
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

    # ---- Agent ----
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

    # ---- Optimiser ----
    # We don't need a curriculum here for the driver-side per-variant
    # sweep (each variant is its own training run). The auto-curriculum
    # for ``full_curriculum`` is expanded driver-side before
    # _apply_variant_preset, so by the time we get here ``curriculum`` is
    # the explicit stages string (if any).
    curriculum_stages = _parse_curriculum(getattr(args, "curriculum", ""))
    if curriculum_stages:
        schedule = make_curriculum_schedule(args, curriculum_stages)
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

    # ---- Vertex-feature cache for the initial eval samples ----
    key, eval_key = jrand.split(key)
    eval_samples = generate_eval_samples(env, eval_key, args.num_eval_samples)
    env_with_samples = eqx.tree_at(
        lambda e: e.eval_args_samples, env, eval_samples,
    )
    vertex_features = _episode_vertex_features(
        args,
        closed_jaxpr.jaxpr,
        tuple(closed_jaxpr.literals),
        tuple(xs),
        eval_samples=eval_samples,
        argnums=tuple(argnums),
    )

    # ---- Env reset (batched) ----
    def reset_envs(env_obj):
        return jax.vmap(lambda _: env_obj.reset())(jnp.arange(num_envs))

    env_states = reset_envs(env_with_samples)

    # ---- Per-depth mask helpers (used by mctx callbacks) ----
    def _build_action_mask_at_depth(emb: DecisionEmbedding, pin_rules):
        depth = emb.depth
        v_invalid = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        v_invalid = v_invalid.at[:total_v].set(1.0 - emb.vertex_avail_mask)
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
        f_invalid = jnp.full((UNIFIED_ACTION_SIZE,), 1.0, dtype=jnp.float32)
        f_invalid = f_invalid.at[:num_factors].set(
            1.0 - pair_factor_mask[emb.vertex_idx, pair_k],
        )
        is_vertex = depth == 0
        is_pair = (depth >= 1) & (depth % 2 == 1)
        return jnp.where(
            is_vertex, v_invalid,
            jnp.where(is_pair, p_invalid, f_invalid),
        )

    def _prior_at_depth(agent_local, emb: DecisionEmbedding, pin_rules):
        logits, value = agent_local.prediction(emb.latent)
        invalid = _build_action_mask_at_depth(emb, pin_rules)
        masked = jnp.where(invalid > 0.5, -1e9, logits)
        return masked, value

    def _build_step_action(vertex_idx, pair_seq, factor_seq):
        target_vertex = jnp.asarray(vertex_idx + 1, dtype=jnp.int32)
        rows = []
        for slot in range(max_rules):
            p = pair_seq[slot]
            f_idx = factor_seq[slot]
            base = _PAIR_TO_BASE[p]
            is_stop = p == PAIR_STOP
            factor = jnp.where(is_stop, 0, factor_table[f_idx]).astype(jnp.int32)
            row = jnp.concatenate([base, factor[None]]).astype(jnp.int32)
            rows.append(row)
        if max_rules >= MAX_RULES_PER_VERTEX:
            specs = jnp.stack(rows[:MAX_RULES_PER_VERTEX], axis=0)
        else:
            stacked = jnp.stack(rows, axis=0)
            pad = jnp.tile(
                jnp.array([-1, -1, 0], dtype=jnp.int32),
                (MAX_RULES_PER_VERTEX - max_rules, 1),
            )
            specs = jnp.concatenate([stacked, pad], axis=0)
        return StepAction(target_vertex=target_vertex, rule_specs=specs)

    # ---- mctx callbacks (Gumbel mode only) ----
    def make_root_fn(pin_rules, agent_local):
        def root_fn_single(emb):
            return _prior_at_depth(agent_local, emb, pin_rules)
        def root_fn(_params, _rng_key, embedding):
            prior, value = jax.vmap(root_fn_single)(embedding)
            return mctx.RootFnOutput(
                prior_logits=prior, value=value, embedding=embedding,
            )
        return root_fn

    def make_recurrent_fn(pin_rules, agent_local):
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
            next_latent, pred_reward = agent_local.dynamics(
                embedding.latent, action.astype(jnp.int32),
            )
            new_depth = depth + 1
            should_commit = new_depth == DECISION_DEPTH

            def commit_branch():
                new_avail = embedding.vertex_avail_mask.at[new_vertex_idx].set(
                    0.0,
                )
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
            n_prior, n_value = _prior_at_depth(agent_local, next_emb, pin_rules)
            return n_prior, n_value, next_emb

        def recurrent_fn(_params, _rng_key, action, embedding):
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

    # ---- Rollout function (vmapped, Gumbel mode only) ----
    def make_rollout_fn(agent_local):
        @eqx.filter_jit
        @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, None, None, None))
        def rollout_fn(
            agent_param,  # passed in for re-tracing on weight refresh
            temperature,
            env_state,
            key,
            preference,
            vertex_features_local,
            reward_weights_local,
            pin_rules_jax,
        ):
            keys = jrand.split(key, rollout_length)
            encode_key, _ = jrand.split(keys[0], 2)
            cached_latent = agent_param.representation(
                env_state.tokens,
                eqn_ids=env_state.eqn_ids,
                preference=preference,
                vertex_features=vertex_features_local,
                key=encode_key,
                agg_key=encode_key,
            )

            root_fn = make_root_fn(pin_rules_jax, agent_param)
            recurrent_fn = make_recurrent_fn(pin_rules_jax, agent_param)

            def _run_search(decision_state, search_key, traverse_key):
                embedding = jax.tree.map(
                    lambda x: jnp.expand_dims(x, 0), decision_state,
                )
                roots = root_fn(agent_param, search_key, embedding)
                invalid_actions = _build_action_mask_at_depth(
                    decision_state, pin_rules_jax,
                )
                policy_output = mctx.gumbel_muzero_policy(
                    params=agent_param,
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

            def step_fn(carry, k):
                state, evolved_latent = carry
                search_key, traverse_key, encode_k = jrand.split(k, 3)
                if args.cache_encoding:
                    latent = evolved_latent
                else:
                    latent = agent_param.representation(
                        state.tokens,
                        eqn_ids=state.eqn_ids,
                        preference=preference,
                        vertex_features=vertex_features_local,
                        key=encode_k,
                        agg_key=encode_k,
                    )
                v_avail = vertex_avail_at_step(
                    state, vertex_valid_static, total_v, num_valid,
                ).astype(jnp.float32)
                decision_state = _empty_decision(latent, v_avail, max_rules)

                mcts_visits, actions_path, mcts_value = _run_search(
                    decision_state, search_key, traverse_key,
                )

                vertex_idx = actions_path[0]
                slot_indices = jnp.arange(max_rules)
                pair_seq = actions_path[1 + 2 * slot_indices].astype(jnp.int32)
                factor_seq = actions_path[2 + 2 * slot_indices].astype(jnp.int32)

                env_action = _build_step_action(vertex_idx, pair_seq, factor_seq)
                env_out = env.step(state, env_action)
                scalar_reward = jnp.sum(env_out.reward * reward_weights_local)

                # Evolve cached latent through the just-taken action sequence
                # so the next step starts from a latent reflecting the move.
                lat_v, _ = agent_param.dynamics(latent, vertex_idx.astype(jnp.int32))

                def per_slot(carry_lat, slot_idx):
                    lat, _ = agent_param.dynamics(carry_lat, pair_seq[slot_idx])
                    lat, _ = agent_param.dynamics(lat, factor_seq[slot_idx])
                    return lat, None

                next_latent_cached, _ = lax.scan(
                    per_slot, lat_v, jnp.arange(max_rules),
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

        return rollout_fn

    rollout_fn = make_rollout_fn(agent)

    # ---- Loss + train step ----
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

    def loss_fn(agent_local, batch: TrajectoryWindow):
        def unroll_loss(window: TrajectoryWindow):
            tokens = window.tokens[0]
            eqn_ids = window.eqn_ids[0]
            preference = window.preference[0]
            latent = agent_local.representation(
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
                    logits, value = agent_local.prediction(latent)
                    target_d = mcts_visits_k[d]
                    l_pi = l_pi + (-jnp.sum(target_d * jnn.log_softmax(logits)))
                    l_v = l_v + 0.5 * jnp.square(value - target_value_k)
                    is_very_last = (
                        (k == args.unroll_steps) and (d == DECISION_DEPTH - 1)
                    )
                    if not is_very_last:
                        latent, pred_reward = agent_local.dynamics(
                            latent, action_seq_k[d],
                        )
                        l_r = l_r + 0.5 * jnp.square(
                            pred_reward - reward_seq_k[d],
                        )
                        if d == DECISION_DEPTH - 1:
                            latent = 0.5 * latent + 0.5 * lax.stop_gradient(
                                latent,
                            )
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
    def train_minibatch(agent_local, opt_state_local, batch):
        (loss_val, parts), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True,
        )(agent_local, batch)
        updates, opt_state_new = optimizer.update(
            grads, opt_state_local, eqx.filter(agent_local, eqx.is_inexact_array),
        )
        agent_new = eqx.apply_updates(agent_local, updates)
        return agent_new, opt_state_new, loss_val, parts

    return {
        "args": args,
        "variant": variant,
        "env": env_with_samples,
        "env_states": env_states,
        "closed_jaxpr": closed_jaxpr,
        "xs": xs,
        "argnums": argnums,
        "vertex_features": vertex_features,
        "factor_table_np": factor_table_np,
        "factors_py": factors_py,
        "num_factors": num_factors,
        "max_rules": max_rules,
        "DECISION_DEPTH": DECISION_DEPTH,
        "UNIFIED_ACTION_SIZE": UNIFIED_ACTION_SIZE,
        "total_v": total_v,
        "num_valid": num_valid,
        "num_envs": num_envs,
        "rollout_length": rollout_length,
        "reward_weights": reward_weights,
        "pair_valid_mask": pair_valid_mask,
        "pair_factor_mask": pair_factor_mask,
        "vertex_valid_static": vertex_valid_static,
        "agent": agent,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "rollout_fn": rollout_fn,
        "train_minibatch": train_minibatch,
        "key": key,
        "constraint_specs": constraint_specs,
        "constraint_indices": constraint_indices,
        "constraint_thresholds": constraint_thresholds,
        "constraint_signs": constraint_signs,
        "multipliers": multipliers,
    }


def _agent_to_numpy(agent: MuZeroAgent) -> list:
    """Extract the array-valued leaves of ``agent`` as a flat list of numpy arrays.

    Used by both actors to ship params across the Ray boundary without
    sending JAX device arrays or eqx's ``_Missing`` sentinel (which
    doesn't survive Ray's pickle/unflatten cycle). The receiver
    reconstructs the pytree by flattening its own freshly-built
    skeleton agent and zipping our leaves in.
    """
    params = eqx.filter(agent, eqx.is_inexact_array)
    leaves = jax.tree_util.tree_leaves(params)
    return [np.asarray(x) for x in leaves]


def _agent_from_numpy(skeleton: MuZeroAgent, leaves_np: list) -> MuZeroAgent:
    """Inverse of :func:`_agent_to_numpy`.

    Uses the skeleton's own treedef to unflatten the incoming leaves,
    then merges with the skeleton's static side via ``eqx.combine``.
    Both sender and receiver must construct their skeleton with
    identical args (and seed when the structure depends on it) so the
    leaf order / count match.
    """
    params_skel = eqx.filter(skeleton, eqx.is_inexact_array)
    leaves_skel, treedef = jax.tree_util.tree_flatten(params_skel)
    if len(leaves_skel) != len(leaves_np):
        raise ValueError(
            f"param-leaf count mismatch: skeleton has {len(leaves_skel)}, "
            f"sender shipped {len(leaves_np)}. Did the two actors build "
            "their agent with different args?"
        )
    leaves_jax = [jnp.asarray(x) for x in leaves_np]
    new_params = jax.tree_util.tree_unflatten(treedef, leaves_jax)
    return eqx.combine(new_params, skeleton)


def _trajectory_to_numpy(traj: Trajectory) -> dict:
    """Convert a Trajectory pytree to a plain dict of numpy arrays."""
    return {
        "tokens": np.asarray(traj.tokens),
        "eqn_ids": np.asarray(traj.eqn_ids),
        "vertex_idx": np.asarray(traj.vertex_idx),
        "pair_seq": np.asarray(traj.pair_seq),
        "factor_seq": np.asarray(traj.factor_seq),
        "reward_vec": np.asarray(traj.reward_vec),
        "scalar_reward": np.asarray(traj.scalar_reward),
        "mcts_visits": np.asarray(traj.mcts_visits),
        "mcts_value": np.asarray(traj.mcts_value),
        "preference": np.asarray(traj.preference),
    }


def _trajectory_from_numpy(traj_np: dict) -> Trajectory:
    """Inverse of :func:`_trajectory_to_numpy`."""
    return Trajectory(
        tokens=jnp.asarray(traj_np["tokens"]),
        eqn_ids=jnp.asarray(traj_np["eqn_ids"]),
        vertex_idx=jnp.asarray(traj_np["vertex_idx"]),
        pair_seq=jnp.asarray(traj_np["pair_seq"]),
        factor_seq=jnp.asarray(traj_np["factor_seq"]),
        reward_vec=jnp.asarray(traj_np["reward_vec"]),
        scalar_reward=jnp.asarray(traj_np["scalar_reward"]),
        mcts_visits=jnp.asarray(traj_np["mcts_visits"]),
        mcts_value=jnp.asarray(traj_np["mcts_value"]),
        preference=jnp.asarray(traj_np["preference"]),
    )


# ---------------------------------------------------------------------------
# Learner actor
# ---------------------------------------------------------------------------


class LearnerWorker:
    """Owns the replay buffer, agent params, and optimiser state.

    Wrapped by :class:`alphagrad.approx.mu0_ray_actors.LearnerActor`,
    which is the Ray-side proxy. The worker builds its own env + agent
    + optimiser in ``__init__`` — symmetric with :class:`RolloutWorker`
    so both share the same JAX import / JIT compile cost. The buffer is
    allocated lazily on the first ``add_trajectories`` call (its leaf
    shapes depend on the trajectory pytree, which only stabilises once
    the rollout workers return their first batch).
    """

    def __init__(self, args_dict: dict, variant: str, learner_seed: int = 0):
        self.state = _build_actor_state(args_dict, variant, learner_seed)
        self.args = self.state["args"]
        self.variant = variant
        self.replay_buffer = None
        self.train_step_counter = 0
        self._resume_pending = bool(
            self.args.replay_checkpoint_path
            and os.path.exists(self.args.replay_checkpoint_path)
        )
        self._key = self.state["key"]

    def add_trajectories(self, traj_np: dict) -> None:
        traj = _trajectory_from_numpy(traj_np)
        if self.replay_buffer is None and self.args.replay_buffer_size > 0:
            sample_traj = jax.tree_util.tree_map(lambda x: x[0], traj)
            self.replay_buffer = init_replay_buffer(
                sample_traj, self.args.replay_buffer_size,
            )
            if self._resume_pending:
                self.replay_buffer = load_replay_buffer(
                    self.args.replay_checkpoint_path, self.replay_buffer,
                )
                self._resume_pending = False

        if self.replay_buffer is not None:
            priorities = _compute_traj_priorities(
                traj, self.state["reward_weights"],
            )
            self.replay_buffer = replay_add_batch(
                self.replay_buffer, traj, priorities=priorities,
            )

    def train_step(self) -> dict:
        if self.replay_buffer is None:
            return {"skipped": True, "buffer_size": 0,
                    "train_step": self.train_step_counter}
        min_size = max(
            self.args.replay_warmup * self.state["num_envs"], 1,
        )
        buffer_size = int(self.replay_buffer.size)
        if buffer_size < min_size:
            return {"skipped": True, "buffer_size": buffer_size,
                    "train_step": self.train_step_counter}

        target_batch_size = (
            self.args.replay_batch_size
            if self.args.replay_batch_size > 0
            else self.state["num_envs"]
        )
        sample_key, self._key = jrand.split(self._key)
        train_traj = replay_sample(
            self.replay_buffer, target_batch_size, sample_key,
            alpha=self.args.replay_priority_alpha,
        )

        # Build training windows + scalar-reward targets exactly as in
        # mu0.py: discount the per-step scalar rewards along the trajectory,
        # slice into ``unroll_steps + 1``-step windows, then stack.
        discounted = jax.vmap(
            lambda r: _discounted_returns(r, self.args.discount)
        )(train_traj.scalar_reward)
        target_values = discounted

        rollout_length = self.state["rollout_length"]
        windows = []
        for i in range(rollout_length - self.args.unroll_steps):
            def _sl(x, i=i):
                return x[:, i: i + self.args.unroll_steps + 1]
            windows.append(
                TrajectoryWindow(
                    tokens=_sl(train_traj.tokens),
                    eqn_ids=_sl(train_traj.eqn_ids),
                    vertex_idx=_sl(train_traj.vertex_idx),
                    pair_seq=_sl(train_traj.pair_seq),
                    factor_seq=_sl(train_traj.factor_seq),
                    scalar_reward=_sl(train_traj.scalar_reward),
                    target_value=_sl(target_values),
                    mcts_visits=_sl(train_traj.mcts_visits),
                    preference=_sl(train_traj.preference),
                )
            )
        window_batch = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=1), *windows,
        )
        shuffle_key, self._key = jrand.split(self._key)
        batches = _shuffle_and_batch_windows(
            window_batch, self.args.minibatches, shuffle_key,
        )

        last_loss = jnp.asarray(0.0)
        last_parts = (jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))
        train_minibatch = self.state["train_minibatch"]
        agent = self.state["agent"]
        opt_state = self.state["opt_state"]
        for i in range(self.args.minibatches):
            mb = jax.tree_util.tree_map(lambda x: x[i], batches)
            agent, opt_state, last_loss, last_parts = train_minibatch(
                agent, opt_state, mb,
            )
        self.state["agent"] = agent
        self.state["opt_state"] = opt_state
        self.train_step_counter += self.args.minibatches

        p_loss, v_loss, r_loss = (float(x) for x in last_parts)
        return {
            "skipped": False,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "reward_loss": r_loss,
            "total_loss": float(last_loss),
            "buffer_size": int(self.replay_buffer.size),
            "train_step": self.train_step_counter,
        }

    def get_params_numpy(self) -> list:
        return _agent_to_numpy(self.state["agent"])

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        self.state["reward_weights"] = jnp.asarray(weights_np, dtype=jnp.float32)

    def checkpoint_replay(self, path: str) -> None:
        if self.replay_buffer is not None and path:
            save_replay_buffer(self.replay_buffer, path)

    def get_stats(self) -> dict:
        return {
            "buffer_size": (
                int(self.replay_buffer.size) if self.replay_buffer else 0
            ),
            "train_step": self.train_step_counter,
        }

    def ready(self) -> bool:
        # Cheap "init-finished" probe the driver uses to block on actor
        # spawn before sending the first remote call.
        return True


# ---------------------------------------------------------------------------
# Rollout actor
# ---------------------------------------------------------------------------


class RolloutWorker:
    """Runs one vmapped Gumbel MuZero rollout per ``rollout_one`` call.

    Wrapped by :class:`alphagrad.approx.mu0_ray_actors.RolloutActor`,
    which is the Ray-side proxy. Holds a private copy of the agent
    params, which the driver refreshes periodically via
    ``set_params_numpy``. The env state is held in the worker and reset
    at every rollout — episodes are independent (mirrors mu0.py's
    per-episode reset).
    """

    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        # Each actor gets a distinct seed so their rollouts diverge.
        seed = int(args_dict.get("seed", 0)) + 1000 * (actor_id + 1)
        self.state = _build_actor_state(args_dict, variant, seed)
        self.args = self.state["args"]
        self.variant = variant
        self.actor_id = actor_id
        self._key = self.state["key"]
        self._pin_rules_default = _pin_rules_for_variant(variant)

    def set_params_numpy(self, params_np: list) -> None:
        new_agent = _agent_from_numpy(self.state["agent"], params_np)
        self.state["agent"] = new_agent

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        self.state["reward_weights"] = jnp.asarray(weights_np, dtype=jnp.float32)

    def rollout_one(
        self,
        rng_seed: int,
        preference_np: np.ndarray | None = None,
        pin_rules: bool | None = None,
        reset_env: bool = True,
    ) -> tuple[dict, dict]:
        if pin_rules is None:
            pin_rules = self._pin_rules_default
        pin_rules_jax = jnp.asarray(bool(pin_rules), dtype=jnp.bool_)

        agent = self.state["agent"]
        num_envs = self.state["num_envs"]
        env = self.state["env"]

        if reset_env:
            self.state["env_states"] = jax.vmap(
                lambda _: env.reset()
            )(jnp.arange(num_envs))

        rollout_key = jrand.PRNGKey(int(rng_seed))
        rollout_keys = jrand.split(rollout_key, num_envs)

        if preference_np is None:
            preference = jnp.zeros((num_envs, NUM_REWARDS), dtype=jnp.float32)
        else:
            preference = jnp.asarray(preference_np, dtype=jnp.float32)

        temperature = jnp.asarray(self.args.temperature, dtype=jnp.float32)
        _, traj = self.state["rollout_fn"](
            agent,
            temperature,
            self.state["env_states"],
            rollout_keys,
            preference,
            self.state["vertex_features"],
            self.state["reward_weights"],
            pin_rules_jax,
        )

        # Stats summary computed on the host (after a small device→host
        # transfer of the reward vectors). Keeps the actor's return value
        # to a flat dict that Ray can pickle quickly.
        reward_weights_np = np.asarray(self.state["reward_weights"])
        reward_vec_np = np.asarray(traj.reward_vec)
        per_env_total = (reward_vec_np.sum(axis=1) * reward_weights_np).sum(
            axis=-1,
        )
        best_idx = int(per_env_total.argmax())
        best_seq = _action_to_pylist(
            np.asarray(traj.vertex_idx[best_idx]),
            np.asarray(traj.pair_seq[best_idx]),
            np.asarray(traj.factor_seq[best_idx]),
            self.state["factor_table_np"],
        )
        stats = {
            "actor_id": self.actor_id,
            "best_return": float(per_env_total[best_idx]),
            "mean_return": float(per_env_total.mean()),
            "best_seq": best_seq,
            "per_reward_means": {
                REWARD_NAMES[j]: float(reward_vec_np[..., j].mean())
                for j in range(NUM_REWARDS)
            },
        }
        return _trajectory_to_numpy(traj), stats

    def reward_vec_means(
        self, rng_seed: int, num_rollouts: int,
    ) -> np.ndarray:
        """Run ``num_rollouts`` rollouts under the current params and return
        the per-channel mean reward vector (numpy, shape ``(NUM_REWARDS,)``).

        Used by the driver's reward-calibration phase before training
        begins. The rollouts here do NOT mutate the replay buffer.
        """
        sum_vec = np.zeros((NUM_REWARDS,), dtype=np.float32)
        for i in range(num_rollouts):
            traj_np, _ = self.rollout_one(
                rng_seed=int(rng_seed) + i * 31 + 1,
                preference_np=None,
                pin_rules=self._pin_rules_default,
                reset_env=True,
            )
            sum_vec += traj_np["reward_vec"].sum(axis=(0, 1))
        denom = float(
            num_rollouts * self.state["num_envs"] * self.state["rollout_length"]
        )
        return sum_vec / max(denom, 1.0)

    def ready(self) -> bool:
        return True
