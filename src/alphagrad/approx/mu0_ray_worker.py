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
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)
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
    return SimpleNamespace(**args_dict)


def _build_actor_state(
    args_dict: dict, variant: str, actor_seed: int, is_spmd: bool = False
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
    if not getattr(args, 'strict_config', False) and num_devs > 1:
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
                print(f"  * Resulting batch size: {tw // best_m} ({tw // best_m // num_devs} per GPU)\n")
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

                env_action = StepAction(
                    target_vertex=jnp.asarray(v_idx + 1, jnp.int32),
                    rule_specs=jnp.stack(
                        [
                            jnp.concatenate(
                                [
                                    _PAIR_TO_BASE[p],
                                    jnp.where(p == PAIR_STOP, 0, factor_table[f])[None],
                                ]
                            ).astype(jnp.int32)
                            for p, f in zip(p_seq, f_seq)
                        ]
                        + [jnp.array([-1, -1, 0], jnp.int32)]
                        * max(0, MAX_RULES_PER_VERTEX - max_rules)
                    )[:MAX_RULES_PER_VERTEX],
                )
                env_out = env.step(state, env_action)

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
                    l_v += 0.5 * jnp.square(value - w.target_value[k])
                    if not ((k == args.unroll_steps) and (d == DECISION_DEPTH - 1)):
                        latent, pred_r = agent_local.dynamics(latent, act_seq[d])
                        l_r += 0.5 * jnp.square(pred_r - rew_seq[d])
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
        def scan_body(carry, batch_i):
            agent_c, opt_state_c = carry
            (loss_val, parts), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                agent_c, batch_i
            )
            updates, opt_state_new = optimizer.update(
                grads, opt_state_c, eqx.filter(agent_c, eqx.is_inexact_array)
            )
            return (eqx.apply_updates(agent_c, updates), opt_state_new), (loss_val, parts)

        (final_agent, final_opt), (losses, parts) = lax.scan(
            scan_body, 
            (agent_local, opt_state_local), 
            all_batches
        )
        
        return final_agent, final_opt, jnp.mean(losses), jax.tree.map(jnp.mean, parts)

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
    def __init__(self, args_dict: dict, variant: str, seed: int = 0):
        self.state = _build_actor_state(args_dict, variant, seed, is_spmd=True)
        self.args = self.state["args"]
        self.variant = variant
        self._key = self.state["key"]
        self._pin_rules_default = _pin_rules_for_variant(variant)
        self.replay_buffer = None
        self.train_step_counter = 0

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
                with mesh: yield
            else:
                yield
        
        scan_ds = self.state.get("scan_data_sharding")

        with active_mesh():
            for _ in range(train_steps):
                if self.replay_buffer is not None and int(self.replay_buffer.size) >= max(
                    self.args.replay_warmup * self.state["num_envs"], 1
                ):
                    s_key, self._key = jrand.split(self._key)
                    t_traj = replay_sample(...)
                    t_vals = jax.vmap(...)(t_traj.scalar_reward)
                    w_batch = jax.tree_util.tree_map(...)
                    
                    sh_key, self._key = jrand.split(self._key)
                    
                    batches = _shuffle_and_batch_windows(
                        w_batch, self.args.minibatches, sh_key
                    )

                    b_size = jax.tree_util.tree_leaves(batches)[0].shape[1]
                    
                    if scan_ds is not None and b_size % num_devs == 0:
                        batches = jax.tree.map(lambda x: jax.device_put(x, scan_ds), batches)

                    (
                        self.state["agent"],
                        self.state["opt_state"],
                        mean_loss,
                        mean_parts,
                    ) = self.state["train_minibatches"](
                        self.state["agent"],
                        self.state["opt_state"],
                        batches,
                    )
                    
                    self.train_step_counter += self.args.minibatches
                    stats.update(
                        {
                            "policy_loss": float(mean_parts[0]),
                            "value_loss": float(mean_parts[1]),
                            "reward_loss": float(mean_parts[2]),
                            "total_loss": float(mean_loss),
                        }
                    )

        rw_np = np.asarray(self.state["reward_weights"])
        r_vec_np = np.asarray(traj.reward_vec)
        per_env_tot = (r_vec_np.sum(axis=1) * rw_np).sum(axis=-1)
        best_idx = int(per_env_tot.argmax())

        stats = {
            "best_return": float(per_env_tot[best_idx]),
            "mean_return": float(per_env_tot.mean()),
            "best_seq": _action_to_pylist(
                np.asarray(traj.vertex_idx[best_idx]),
                np.asarray(traj.pair_seq[best_idx]),
                np.asarray(traj.factor_seq[best_idx]),
                self.state["factor_table_np"],
            ),
            "per_reward_means": {
                REWARD_NAMES[j]: float(r_vec_np[..., j].mean())
                for j in range(NUM_REWARDS)
            },
        }

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
                if self.replay_buffer is not None and int(self.replay_buffer.size) >= max(
                    self.args.replay_warmup * self.state["num_envs"], 1
                ):
                    s_key, self._key = jrand.split(self._key)
                    t_traj = replay_sample(
                        self.replay_buffer,
                        self.args.replay_batch_size
                        if self.args.replay_batch_size > 0
                        else self.state["num_envs"],
                        s_key,
                        alpha=self.args.replay_priority_alpha,
                    )
                    t_vals = jax.vmap(lambda r: _discounted_returns(r, self.args.discount))(
                        t_traj.scalar_reward
                    )

                    w_batch = jax.tree_util.tree_map(
                        lambda *xs: jnp.stack(xs, axis=1),
                        *[
                            TrajectoryWindow(
                                tokens=t_traj.tokens[:, i : i + self.args.unroll_steps + 1],
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
                                target_value=t_vals[:, i : i + self.args.unroll_steps + 1],
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

                    for i in range(self.args.minibatches):
                        batch_i = jax.tree_util.tree_map(lambda x: x[i], batches)
                        b_size = jax.tree_util.tree_leaves(batch_i)[0].shape[0]
                        
                        if ds is not None and b_size % num_devs == 0:
                            batch_i = jax.tree.map(lambda x: jax.device_put(x, ds), batch_i)

                        (
                            self.state["agent"],
                            self.state["opt_state"],
                            last_loss,
                            last_parts,
                        ) = self.state["train_minibatch"](
                            self.state["agent"],
                            self.state["opt_state"],
                            batch_i,
                        )
                    self.train_step_counter += self.args.minibatches
                    stats.update(
                        {
                            "policy_loss": float(last_parts[0]),
                            "value_loss": float(last_parts[1]),
                            "reward_loss": float(last_parts[2]),
                            "total_loss": float(last_loss),
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
        return stats

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> np.ndarray:
        sum_vec = np.zeros((NUM_REWARDS,), dtype=np.float32)
        ds = self.state.get("data_sharding")
        mesh = self.state.get("mesh")
        num_devs = len(jax.devices()) if ds is not None else 1

        @contextlib.contextmanager
        def active_mesh():
            if mesh is not None:
                with mesh: yield
            else:
                yield

        with active_mesh():
            for i in range(num_rollouts):
                pref = jnp.zeros((self.state["num_envs"], NUM_REWARDS), jnp.float32)
                env_states = jax.vmap(lambda _: self.state["env"].reset())(jnp.arange(self.state["num_envs"]))
                keys = jrand.split(jrand.PRNGKey(int(rng_seed) + i * 31 + 1), self.state["num_envs"])
                
                if ds is not None and self.state["num_envs"] % num_devs == 0:
                    pref = jax.device_put(pref, ds)
                    env_states = jax.tree.map(lambda x: jax.device_put(x, ds), env_states)
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
                sum_vec += np.asarray(traj.reward_vec).sum(axis=(0, 1))
        return sum_vec / max(
            float(num_rollouts * self.state["num_envs"] * self.state["rollout_length"]),
            1.0,
        )

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        self.state["reward_weights"] = jnp.asarray(weights_np, dtype=jnp.float32)

    def checkpoint_replay(self, path: str) -> None:
        if self.replay_buffer is not None and path:
            save_replay_buffer(self.replay_buffer, path)

    def ready(self) -> bool:
        return True


class CPUApproximationWorker:
    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        self.args = _args_from_dict(args_dict)
        self.variant = variant
        self.actor_id = actor_id

    def compile_approximations(self) -> dict:
        time.sleep(1)
        return {"status": "compiled"}

    def ready(self) -> bool:
        return True