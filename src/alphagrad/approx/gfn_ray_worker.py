"""SPMD GFlowNet trainer worker — lifts gfn.py into a Ray actor body.

Mirrors :mod:`alphagrad.approx.mu0_ray_worker` minus the MCTS layer. The
algorithm body (rollout, Trajectory-Balance loss, fresh + replay mix,
scanned gradient steps) is functionally identical to
:func:`alphagrad.approx.gfn.main`'s inner closures (gfn.py:451–880); it
just lives inside :class:`GFNServerWorker` so a Ray driver can spawn it
on a multi-GPU SPMD actor and route per-step jacve/tokenize work
through a :class:`alphagrad.approx.cpu_approx_pool.CpuApproxPool`.

Notable design points
---------------------
* **logZ under SPMD.** ``logZ`` is a scalar :class:`equinox.Module` field
  on :class:`alphagrad.approx.gfn.GFNAgent`. ``shard_leaf`` puts every
  array leaf on the ``replicated_sharding`` (``PartitionSpec()``), so the
  scalar lands replicated across all data-parallel devices. ``jax.grad``
  automatically all-reduces the per-device contribution, which is the
  intended semantics for a global normalising constant.

* **Replay priorities use symlog scale.** The single-process trainer in
  gfn.py:864-869 stores the *raw* weighted terminal reward as the
  priority. With ``cmp ~ 1e10`` and ``--replay-priority-alpha > 0`` that
  spans ten decades — sampling collapses onto the single highest-reward
  trajectory after one rollout. Here we store ``symlog(weighted_terminal)``-
  scaled priorities so the priority space matches the TB loss's
  ``β·symlog(weighted_terminal)`` term. At the default
  ``--replay-priority-alpha 0.0`` this doesn't matter; the change makes
  the ``alpha > 0`` path safe for later tuning.

* **--terminal-rewards-only is correct for TB.** The loss reads exactly
  ``traj.reward[:, -1, :]`` (mirror of gfn.py:603). ``log P_B = -log(t+1)``
  uses ``step_count``, not reward. Intermediate-step rewards never enter
  the gradient.
"""

from __future__ import annotations

import contextlib
import os
from functools import partial
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from alphagrad.approx.common import (
    build_pair_valid_mask,
    build_vertex_valid_static,
    data_gen,
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
    vertex_avail_at_step,
)
from alphagrad.approx.env import (
    MAX_TOKENS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    VertexEliminationEnv,
)
from alphagrad.approx.gfn import (
    GFNAgent,
    GFNTrajectory,
    _build_gfn_agent,
    _logZ_mask,
)
from alphagrad.approx.heads import MicroAction, precompute_factor_tables
from alphagrad.approx.ppo import (
    NUM_PAIR_CHOICES,
    NUM_VALUE_HEADS,
    PAIR_STOP,
    _action_to_pylist,
    _action_to_pylist_dynamic,
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
from alphagrad.approx.variants import _apply_variant_preset
from alphagrad.utils import symlog as _symlog


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Back-compat re-export of
    :func:`alphagrad.approx.common.ray_runtime._args_from_dict`."""
    from alphagrad.approx.common.ray_runtime import _args_from_dict as _impl
    return _impl(args_dict)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(raw).split(",") if x.strip())


# ---------------------------------------------------------------------------
# _build_actor_state — heavyweight init, mirror of mu0_ray_worker:147-340
# ---------------------------------------------------------------------------


def _build_actor_state(
    args_dict: dict,
    variant: str,
    actor_seed: int,
    is_spmd: bool = False,
    cpu_workers: list | None = None,
    remote_pool: object | None = None,
    remote_timeout_s: float = 60.0,
) -> dict:
    """Build the per-variant SPMD training state.

    Mirrors :func:`alphagrad.approx.mu0_ray_worker._build_actor_state`
    end-to-end. The mu0 version owns the MuZero agent + the MCTS-specific
    rollout/loss; this version owns the GFN agent + the
    :func:`alphagrad.approx.gfn` rollout/loss/train closures.

    Returns a dict containing all the JAX-compiled callables and state
    that :class:`GFNServerWorker` reads from at training time.
    """
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
    # Gradient mode: measure value_and_grad of the scalar loss — build the
    # jaxpr + env target from the reduced scalar (shared wrap).
    from alphagrad.approx.common import maybe_scalar_loss
    target_fn, measure_grad = maybe_scalar_loss(args, target_fn)
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
        num_data_points=int(getattr(args, "num_data_points", 5)),
        reps_per_point=int(getattr(args, "reps_per_point", 4)),
        percentile_keep=float(getattr(args, "percentile_keep", 0.60)),
        latency_inner_reps=int(getattr(args, "latency_inner_reps", 1)),
        latency_warmup=int(getattr(args, "latency_warmup", 0)),
        latency_winsor=float(getattr(args, "latency_winsor", 0.0)),
        measure_grad=measure_grad,
        latency_timer=str(getattr(args, "latency_timer", "perf_counter")),
        slow_exec_cutoff_seconds=float(
            getattr(args, "slow_exec_cutoff_seconds", 8.0)
        ),
        flop_gate_threshold=float(getattr(args, "flop_gate_threshold", 0.0)),
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

    use_pointer, use_autoreg = _select_variant(args)
    variant_label = _variant_label(use_pointer, use_autoreg)

    factor_table, factors_py, num_factors, max_rules = _build_factor_table(
        args, use_autoreg,
    )
    factor_table_np = np.array(factors_py, dtype=np.int32)
    # Pre-converted jnp factor table — hoisted once at init so the
    # rollout's per-step scan doesn't re-array-ify on every iteration
    # (free-win optimization #3 from commit b6d366a).
    factor_table_j = jnp.asarray(factor_table_np, dtype=jnp.int32)
    pair_factor_mask = jnp.ones(
        (total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32,
    )

    # Full micro-action scheme (DIAG/COMPRESS/QUANT, up to max_substeps rules
    # per vertex). The MicroActionPolicy consumes the env's STATIC per-vertex
    # axis structure (same arrays ppo.py's own dynamic loss uses) + a prime/
    # exponent factor table. ``op_legality_j`` is all-ones for --variant full
    # (every op type legal); the env still gates COMPRESS/QUANT to the final
    # vertex internally.
    factor_tables = precompute_factor_tables(int(getattr(args, "max_axis_size", 1024)))
    axis_state_static = env.axis_state_static
    axis_valid_static = env.axis_valid_static
    op_legality_j = jnp.ones((4,), dtype=jnp.float32)
    max_substeps = int(getattr(args, "max_substeps", 16))

    num_envs = _resolve_num_envs(args.num_envs, args.example)
    rollout_length = num_valid

    # --- SPMD auto-tune (mirror of mu0_ray_worker:216-240) -----------------
    num_devs = len(jax.devices()) if is_spmd else 1
    if not getattr(args, "strict_config", False) and num_devs > 1:
        old_envs = num_envs
        if num_envs % num_devs != 0:
            num_envs = max(num_devs, round(num_envs / num_devs) * num_devs)

        tw = num_envs
        base = tw // num_devs

        possible_m = [i for i in range(1, base + 1) if base % i == 0] or [1]
        best_m = min(possible_m, key=lambda x: abs(x - args.minibatches))
        if old_envs != num_envs or best_m != args.minibatches:
            print(f"\n[Auto-Tune] Optimizing for {num_devs} GPUs:")
            if old_envs != num_envs:
                print(f"  * num_envs: {old_envs} -> {num_envs}")
            if args.minibatches != best_m:
                print(f"  * minibatches: {args.minibatches} -> {best_m}")
            per_gpu = max(num_envs // num_devs, 1)
            print(f"  * Per-GPU envs after sharding: {per_gpu}\n")
        args.minibatches = best_m
        args.num_envs = num_envs
    # --- END auto-tune ---------------------------------------------------

    reward_weights_np = _build_reward_weights(args)
    reward_weights = jnp.asarray(reward_weights_np, dtype=jnp.float32)
    cmp_idx = _cmp_reward_index(args.cmp_type)
    mem_idx = _mem_reward_index(args.mem_type)
    cosine_idx = REWARD_INDEX["cosine_sim"]
    frob_idx = REWARD_INDEX["frob_residual"]

    # MOGFN-PC channel indices: which NUM_REWARDS channels the 3-dim
    # preference vector indexes. Validates length == NUM_VALUE_HEADS.
    pref_channels_spec = getattr(
        args, "preference_channels", "flops,peak_memory,cosine_sim",
    )
    pref_channel_indices = _parse_preference_channels(pref_channels_spec)
    pref_indices_j = jnp.asarray(pref_channel_indices, dtype=jnp.int32)
    print(
        f"[gfn_ray] MOGFN-PC preference channels: "
        f"{[REWARD_NAMES[i] for i in pref_channel_indices]} "
        f"(indices {list(pref_channel_indices)})"
    )

    nonzero_w = ", ".join(
        f"{REWARD_NAMES[i]}={float(reward_weights_np[i]):+.3g}"
        for i in range(NUM_REWARDS)
        if reward_weights_np[i] != 0.0
    )
    print(
        f"[gfn_ray] variant={variant_label}, num_envs={num_envs}, "
        f"max_rules={max_rules}, factors={factors_py}, "
        f"rollout_length={rollout_length}, beta={args.beta}\n"
        f"[gfn_ray] reward weights: {nonzero_w or '<all zero — debug only>'}"
    )

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

    # Per-leaf LR schedule. Matches gfn.py:431-449 exactly: cosine decay
    # for the policy params, fast constant LR for the scalar logZ.
    schedule = optax.cosine_decay_schedule(
        args.lr,
        max(args.episodes * args.gradient_steps, 1),
        args.lr_decay_min_mult,
    )
    logZ_mask_bool = _logZ_mask(gfn_agent)
    not_logZ_mask_bool = jax.tree_util.tree_map(lambda b: not b, logZ_mask_bool)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.masked(
            optax.adam(args.logZ_lr, b1=args.adam_b1, eps=args.adam_eps),
            logZ_mask_bool,
        ),
        optax.masked(
            optax.adam(schedule, b1=args.adam_b1, eps=args.adam_eps),
            not_logZ_mask_bool,
        ),
    )
    opt_state = optimizer.init(eqx.filter(gfn_agent, eqx.is_inexact_array))

    # --- SPMD mesh + replicated sharding (mirror of mu0_ray_worker:280-295)
    if is_spmd:
        mesh = Mesh(np.array(jax.devices()), axis_names=("dev",))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        data_sharding = NamedSharding(mesh, PartitionSpec("dev"))
        scan_data_sharding = NamedSharding(mesh, PartitionSpec(None, "dev"))

        def shard_leaf(x):
            return jax.device_put(x, replicated_sharding) if eqx.is_array(x) else x

        # ``logZ`` is a scalar jax.Array on the GFNAgent — gets
        # PartitionSpec() (replicated). jax.grad under data-parallel
        # then all-reduces the per-device contribution automatically.
        gfn_agent = jax.tree.map(shard_leaf, gfn_agent)
        opt_state = jax.tree.map(shard_leaf, opt_state)
    else:
        data_sharding = None
        scan_data_sharding = None
        mesh = None

    # --- Eval samples + ray.put once via the pool (free-win #2) ----------
    key, eval_key = jrand.split(key)
    eval_samples = generate_eval_samples(env, eval_key, args.num_eval_samples)
    env_with_samples = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
    if remote_pool is not None:
        children, aux_data = env_with_samples.tree_flatten()
        aux_data = (
            aux_data[0], aux_data[1], aux_data[2],
            remote_pool, float(remote_timeout_s),
        )
        env_with_samples = type(env_with_samples).tree_unflatten(
            aux_data, children,
        )
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

    # ---------- Closures: rollout / loss / training -----------------------
    # Lifted directly from gfn.py:454-655. We close over module-level args,
    # the env's vertex_valid_static / pair_valid_mask / pair_factor_mask,
    # the hoisted factor_table_j, reward_weights, and the optimizer.

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, 0))
    def rollout_fn(
        gfn_agent_local,
        env_obj,
        rollout_length_arg,
        env_state,
        rkey,
        vertex_features_local,
        preference,
    ):
        keys = jrand.split(rkey, rollout_length_arg)
        init_residual = jnp.zeros((total_v, args.embd_dim), dtype=jnp.float32)
        encode_key, _ = jrand.split(keys[0], 2)
        if args.cache_encoding:
            cached_encoding = gfn_agent_local.base_agent.encode_once(
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
                actions,
                _vertex_dist,
                _op_dists,
                _i_dists,
                _j_dists,
                _exp_dists,
                _kind_dists,
                _quant_dists,
                _value,
                v_context,
            ) = gfn_agent_local.base_agent.sample_action_dynamic(
                state.tokens,
                vertex_avail_mask,
                axis_state_static,
                axis_valid_static,
                factor_tables,
                op_legality_j,
                sample_key,
                eqn_ids=state.eqn_ids,
                vertex_features=vertex_features_local,
                residual_state=residual_state,
                cached_encoding=cached_encoding,
                preference=preference,
            )
            env_action = gfn_agent_local.base_agent.to_env_action_dynamic(
                vertex_idx, actions, axis_state_static,
            )
            env_out = env_obj.step(state, env_action)
            new_residual = gfn_agent_local.base_agent.update_residual(
                residual_state, vertex_idx, v_context,
            )

            transition = GFNTrajectory(
                tokens=state.tokens.astype(jnp.int32),
                eqn_ids=state.eqn_ids.astype(jnp.int32),
                residual_state=residual_state,
                vertex_idx=jnp.asarray(vertex_idx, dtype=jnp.int32),
                pair_seq=jnp.zeros((max_rules,), dtype=jnp.int32),
                factor_seq=jnp.zeros((max_rules,), dtype=jnp.int32),
                micro_op_seq=actions.op_type.astype(jnp.int32),
                micro_i_seq=actions.i.astype(jnp.int32),
                micro_j_seq=actions.j.astype(jnp.int32),
                micro_exp_seq=actions.exponents.astype(jnp.int32),
                micro_factor_seq=actions.factor.astype(jnp.int32),
                micro_kind_seq=actions.compress_kind.astype(jnp.int32),
                micro_quant_seq=actions.quant_dtype.astype(jnp.int32),
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

    def loss_fn(
        gfn_agent_local,
        traj: GFNTrajectory,
        vertex_features_local,
        beta,
        reward_stats_mean,   # (NUM_REWARDS,) — EMA mean of symlog(reward), runtime arg
        reward_stats_var,    # (NUM_REWARDS,) — EMA var of  symlog(reward), runtime arg
        lkey,
    ):
        E, T = traj.tokens.shape[:2]
        enc_key, eval_key = jrand.split(lkey, 2)

        if args.cache_encoding:
            cached_per_env = jax.vmap(
                lambda t, e, kk: gfn_agent_local.base_agent.encode_once(
                    t, eqn_ids=e, key=kk,
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

        def _eval_one(toks, eids, rs, vidx, mo, mi, mj, me, mf, mk, mq, vmask, pref, cached, kk):
            actions = MicroAction(
                op_type=mo, i=mi, j=mj, exponents=me,
                factor=mf, compress_kind=mk, quant_dtype=mq,
            )
            out = gfn_agent_local.base_agent.evaluate_action_dynamic(
                toks, vidx, actions, vmask,
                axis_state_static, axis_valid_static, factor_tables, kk,
                eqn_ids=eids, vertex_features=vertex_features_local,
                residual_state=rs, cached_encoding=cached, preference=pref,
            )
            return out[0], out[1]  # (total_log_p, total_entropy)

        _mfields = (
            flat.micro_op_seq, flat.micro_i_seq, flat.micro_j_seq,
            flat.micro_exp_seq, flat.micro_factor_seq,
            flat.micro_kind_seq, flat.micro_quant_seq,
        )
        if cached_flat is None:
            log_p, ent = jax.vmap(
                lambda toks, eids, rs, vidx, mo, mi, mj, me, mf, mk, mq, vmask, pref, kk:
                    _eval_one(toks, eids, rs, vidx, mo, mi, mj, me, mf, mk, mq,
                              vmask, pref, None, kk),
            )(
                flat.tokens, flat.eqn_ids, flat.residual_state, flat.vertex_idx,
                *_mfields, flat.vertex_avail_mask, flat.preference, eval_keys,
            )
        else:
            log_p, ent = jax.vmap(_eval_one)(
                flat.tokens, flat.eqn_ids, flat.residual_state, flat.vertex_idx,
                *_mfields, flat.vertex_avail_mask, flat.preference, cached_flat, eval_keys,
            )
        log_p = log_p.reshape(E, T)
        ent = ent.reshape(E, T)

        sum_log_pf = jnp.sum(log_p, axis=-1)
        log_pb_per_step = -jnp.log(traj.step_count + 1.0)
        sum_log_pb = jnp.sum(log_pb_per_step, axis=-1)

        terminal_rewards = traj.reward[:, -1, :]   # (E, NUM_REWARDS)

        # Always compute the legacy weighted-terminal for metric / replay
        # priority continuity. It's NOT used in the loss when --mogfn-pc
        # is on; it's still emitted so progress-bar / wandb logging
        # stays comparable across runs.
        weighted_terminal = jnp.sum(terminal_rewards * reward_weights, axis=-1)

        # Conditional log-partition log Z_θ(ω) (MOGFN-PC, Jain et al. 2023):
        # the partition function is a learned function of the preference, not a
        # scalar. ``traj.preference`` is (E, T, NUM_VALUE_HEADS), constant over
        # T, so take t=0.
        w_pref = traj.preference[:, 0, :]                       # (E, NUM_VALUE_HEADS)
        log_Z_w = jax.vmap(gfn_agent_local.log_Z)(w_pref)       # (E,)

        if args.mogfn_pc:
            # Per-channel normalized log-reward components z_k:
            #   z_k = (symlog(r_k) - μ_k) / σ_k   (zscore)  |  symlog(r_k) (none)
            # The env's rewards span ~1e8 (latency) … 1e-7 (frob), so the
            # normalization keeps the scalarization well-conditioned. R̃_k =
            # exp(z_k) > 0 is then a positive per-objective reward, which the
            # paper's scalarizations require.
            terminal_symlog = jnp.sign(terminal_rewards) * jnp.log1p(
                jnp.abs(terminal_rewards),
            )
            if args.reward_normalization == "zscore":
                std = jnp.sqrt(reward_stats_var + 1e-6)
                terminal_normalized = (terminal_symlog - reward_stats_mean) / std
            else:
                terminal_normalized = terminal_symlog
            z = terminal_normalized[:, pref_indices_j]          # (E, n_pref)
            w = w_pref / jnp.maximum(jnp.sum(w_pref, axis=-1, keepdims=True), 1e-8)
            scal = getattr(args, "scalarization", "ws")
            if scal == "ws":
                # Weighted-Sum (paper default/best): R(x|ω)=Σ_k w_k R̃_k used
                # directly as the GFN reward ⇒ log R = β·log Σ_k w_k exp(z_k)
                #                                    = β·logsumexp(z_k + log w_k).
                log_R = beta * jax.nn.logsumexp(
                    z + jnp.log(w + 1e-8), axis=-1,
                )                                               # (E,)
            elif scal == "wt":
                # Weighted-Tchebycheff: R = -max_k w_k (z*_k - z_k); z* = batch max.
                zstar = jnp.max(z, axis=0, keepdims=True)
                g = jnp.max(w * (zstar - z), axis=-1)
                log_R = -beta * g
            else:  # "wl" — Weighted-log-sum (geometric): log R = β·Σ_k w_k z_k
                log_R = beta * jnp.sum(w * z, axis=-1)
        else:
            # Legacy: scalar weighted sum, then symlog. Kept for ablation.
            log_R = beta * _symlog(weighted_terminal)

        tb_residual = log_Z_w + sum_log_pf - sum_log_pb - log_R
        tb_loss = jnp.mean(tb_residual ** 2)
        entropy_term = jnp.mean(jnp.sum(ent, axis=-1))

        total_loss = tb_loss - args.entropy_weight * entropy_term

        metrics = (
            tb_loss,
            jnp.mean(log_Z_w),
            jnp.mean(sum_log_pf),
            jnp.mean(sum_log_pb),
            jnp.mean(log_R),
            jnp.mean(weighted_terminal),
            entropy_term,
            total_loss,
        )
        return total_loss, metrics

    def do_rollout(gfn_agent_local, env_obj, env_states_local,
                   vertex_features_local, preferences, rkey):
        rollout_keys = jrand.split(rkey, num_envs)
        env_states_new, traj = rollout_fn(
            gfn_agent_local, env_obj, num_valid, env_states_local,
            rollout_keys, vertex_features_local, preferences,
        )
        return env_states_new, traj

    def do_train_steps(
        gfn_agent_local,
        opt_state_local,
        train_traj,
        vertex_features_local,
        beta,
        reward_stats_mean,
        reward_stats_var,
        tkey,
    ):
        dynamic_carry, static_carry = eqx.partition(
            (gfn_agent_local, opt_state_local), eqx.is_array,
        )
        step_keys = jrand.split(tkey, args.gradient_steps)

        def grad_step_fn(carry, t_key):
            comb_agent, comb_opt_state = eqx.combine(carry, static_carry)
            grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(
                comb_agent, train_traj, vertex_features_local, beta,
                reward_stats_mean, reward_stats_var, t_key,
            )
            # Pass the FILTERED params (None at non-inexact-array leaves)
            # so the pytree structure matches the masked-optimizer's
            # ``logZ_mask`` (also built via ``eqx.filter(gfn_agent,
            # eqx.is_inexact_array)`` in _build_actor_state). Without
            # this, optax.masked.mask_pytree does
            # ``jax.tree.map(lambda m, p: ..., mask_tree, params)``
            # which under recent JAX no longer treats ``None`` as a
            # tree prefix of function leaves — yielding
            # ``ValueError: Expected None, got <function MLP.<lambda>>``.
            # (gfn.py:646-648 passes ``comb_agent`` unfiltered and has
            # the same latent bug; the Ray path fixes it here so the
            # cluster JAX/optax versions don't trip on it.)
            filtered_params = eqx.filter(comb_agent, eqx.is_inexact_array)
            updates, new_opt_state = optimizer.update(
                grads, comb_opt_state, filtered_params,
            )
            new_agent = eqx.apply_updates(comb_agent, updates)
            next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
            return next_carry, metrics

        dynamic_carry, metrics_seq = lax.scan(grad_step_fn, dynamic_carry, step_keys)
        gfn_agent_new, opt_state_new = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)
        return gfn_agent_new, opt_state_new, metrics

    if not args.no_jit:
        do_rollout = eqx.filter_jit(do_rollout)
        do_train_steps = eqx.filter_jit(do_train_steps)

    return {
        "args": args,
        "variant": variant,
        "variant_label": variant_label,
        "use_pointer": use_pointer,
        "use_autoreg": use_autoreg,
        "env": env_with_samples,
        "env_states": env_states,
        "vertex_features": vertex_features,
        "factor_table_np": factor_table_np,
        "max_rules": max_rules,
        "max_substeps": max_substeps,
        "num_envs": num_envs,
        "rollout_length": rollout_length,
        "total_v": total_v,
        "num_valid": num_valid,
        "reward_weights": reward_weights,
        "cmp_idx": cmp_idx,
        "mem_idx": mem_idx,
        "cosine_idx": cosine_idx,
        "frob_idx": frob_idx,
        "agent": gfn_agent,
        "optimizer": optimizer,
        "opt_state": opt_state,
        "do_rollout": do_rollout,
        "do_train_steps": do_train_steps,
        "scan_data_sharding": scan_data_sharding,
        "data_sharding": data_sharding,
        "mesh": mesh,
        "key": key,
    }


# ---------------------------------------------------------------------------
# GFNServerWorker — runs inside the Ray SPMD actor process
# ---------------------------------------------------------------------------


class _RewardStats:
    """Per-channel running EMA mean/var of ``symlog(reward)``.

    Used by the MOGFN-PC scalarization in ``loss_fn`` to normalize each
    reward dim before the weighted sum. Without this, channels with
    much larger raw magnitudes (e.g. ``symlog(flops)~23`` vs
    ``symlog(cos_sim)~0.5``) still dominate even in log-space, and the
    Pareto-front exploration collapses to whatever dim has the highest
    natural scale. Decay = standard EMA momentum knob (0.99 default →
    slow, stable).
    """

    def __init__(self, num_channels: int, decay: float = 0.99):
        self.num_channels = int(num_channels)
        self.decay = float(decay)
        self.mean = np.zeros((num_channels,), dtype=np.float32)
        self.var = np.ones((num_channels,), dtype=np.float32)
        self.warmed = False

    @staticmethod
    def _symlog_np(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def update(self, raw_rewards: np.ndarray) -> None:
        """``raw_rewards`` is ``(N, num_channels)`` — typically a flat batch
        of per-(env, t) reward vectors or per-env terminal rewards.
        """
        if raw_rewards.size == 0:
            return
        sl = self._symlog_np(raw_rewards.astype(np.float32))
        batch_mean = sl.mean(axis=0)
        batch_var = sl.var(axis=0) + 1e-8
        if not self.warmed:
            self.mean = batch_mean
            self.var = batch_var
            self.warmed = True
        else:
            d = self.decay
            self.mean = d * self.mean + (1.0 - d) * batch_mean
            self.var = d * self.var + (1.0 - d) * batch_var


def _parse_preference_channels(spec: str) -> tuple[int, ...]:
    """Map a comma-separated reward-name list to channel indices.

    Validates the result is exactly NUM_VALUE_HEADS long (matches the
    agent's pref_proj input dim, see ppo.py:3069). Raises ValueError on
    bad input — caught at worker init so the failure is fast and loud.
    """
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) != NUM_VALUE_HEADS:
        raise ValueError(
            f"--preference-channels expects exactly {NUM_VALUE_HEADS} names "
            f"(matches NUM_VALUE_HEADS, the agent's pref_proj input dim); "
            f"got {len(names)}: {names}"
        )
    unknown = [n for n in names if n not in REWARD_INDEX]
    if unknown:
        raise ValueError(
            f"--preference-channels contains unknown reward names: "
            f"{unknown}. Valid: {list(REWARD_INDEX)}"
        )
    return tuple(REWARD_INDEX[n] for n in names)


def _symlog_priorities(weighted_per_env: np.ndarray) -> jnp.ndarray:
    """Symlog-scaled, strictly-positive priorities for the replay buffer.

    The TB loss reads ``β · symlog(weighted_terminal)`` (see
    :func:`alphagrad.approx.gfn.main`'s loss_fn). Priorities therefore
    live on the same symlog'd reward scale: shift so the minimum is at
    ``+1e-3`` (strictly positive, so ``priority**alpha`` is well-defined
    for any α).
    """
    sl = np.sign(weighted_per_env) * np.log1p(np.abs(weighted_per_env))
    sl = sl - sl.min() + 1e-3
    return jnp.asarray(sl, dtype=jnp.float32)


class GFNServerWorker:
    """SEED-style off-policy GFN trainer body, run on the SPMD GPU actor.

    Surface mirrors :class:`alphagrad.approx.mu0_ray_worker.SPMDServerWorker`
    so the Ray driver loop in ``gfn_ray.py`` is structurally identical
    to ``mu0_ray.py``.
    """

    def __init__(
        self,
        args_dict: dict,
        variant: str,
        seed: int = 0,
        cpu_workers: list | None = None,
        *,
        callback_timeout_s: float = 120.0,
        initial_timeout_s: float | None = None,
        warm_after: int = 3,
        recycle_every: int = 50,
        cpu_actor_options: dict | None = None,
        starting_actor_id: int = 1000,
    ):
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

            _actor_counter = [int(starting_actor_id)]
            _opts = dict(cpu_actor_options or {})
            _args_dict_closure = args_dict
            _variant_closure = variant

            def _respawn_factory():
                aid = _actor_counter[0]
                _actor_counter[0] += 1
                return CPUApproximationActor.options(**_opts).remote(
                    _args_dict_closure, _variant_closure, aid,
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
        self.replay_buffer = None
        self.train_step_counter = 0
        self._episode_counter = 0
        self._checkpoint_path = getattr(self.args, "checkpoint_path", "") or ""
        self._checkpoint_every = int(getattr(self.args, "checkpoint_every", 0))

        # MOGFN-PC per-channel reward stats (running EMA of symlog'd
        # terminal rewards). Updated on the host side at the end of each
        # rollout from raw terminal rewards; consumed by the JIT'd
        # ``do_train_steps`` as runtime args so each call sees fresh
        # stats. ``warmed=False`` until the first update; while cold,
        # mean=0 / var=1 → no-op normalization.
        self.reward_stats = _RewardStats(
            num_channels=NUM_REWARDS,
            decay=float(getattr(self.args, "reward_stats_decay", 0.99)),
        )

        # Resume from a previously-written checkpoint if one exists. Same
        # contract as the mu0 / PPO workers.
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
                        f"[gfn_ray_worker] resumed from {self._checkpoint_path} "
                        f"at train_step {self.train_step_counter}"
                    )
                install_sigterm_handler(self._save_checkpoint_safe)
            except Exception as exc:
                print(f"[gfn_ray_worker] checkpoint resume failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _active_mesh(self):
        mesh = self.state.get("mesh")
        if mesh is not None:
            with mesh:
                yield
        else:
            yield

    def _sample_preferences(self, pref_key):
        # PPO's Agent.pref_proj is ``Linear(NUM_VALUE_HEADS=3, embd_dim)``
        # (ppo.py:3069). The 3 heads carry whatever the user assigns
        # via --preference-channels (default flops, peak_memory,
        # cosine_sim).
        #
        # MOGFN-PC implies preference-conditioning, so we sample a
        # Dirichlet vector whenever either --mogfn-pc or
        # --preference-conditioned is set. Without either, we emit
        # zeros — the agent's pref_proj is zero-initialised so a zero
        # preference == no-op in the encoder.
        args = self.args
        if getattr(args, "mogfn_pc", False) or args.preference_conditioned:
            return sample_preferences(
                pref_key,
                NUM_VALUE_HEADS,
                self.state["num_envs"],
                dirichlet_alpha=args.preference_dirichlet_alpha,
            )
        return jnp.zeros(
            (self.state["num_envs"], NUM_VALUE_HEADS), dtype=jnp.float32,
        )

    # ------------------------------------------------------------------
    # Main entry point — one episode
    # ------------------------------------------------------------------

    def run_rollout_and_train(
        self,
        rng_seed: int,
        preference_np: Any = None,
        pin_rules: Any = None,  # accepted for CLI parity; no effect in TB
        reset_env: bool = True,
        train_steps: int = 1,
    ) -> dict:
        args = self.args

        if reset_env:
            self.state["env_states"] = jax.vmap(
                lambda _: self.state["env"].reset()
            )(jnp.arange(self.state["num_envs"]))

        # ---- β schedule (mirror of gfn.py:776-782) ------------------------
        progress = float(self._episode_counter) / max(args.episodes - 1, 1)
        beta = jnp.asarray(
            schedule_at(
                progress, args.beta, args.beta_final, args.beta_schedule,
            ),
            dtype=jnp.float32,
        )

        # ---- Per-episode preferences -------------------------------------
        pref_key, rollout_key, sample_key, train_key, self._key = jrand.split(
            jrand.fold_in(self._key, int(rng_seed)), 5,
        )

        if preference_np is None:
            preferences = self._sample_preferences(pref_key)
        else:
            preferences = jnp.asarray(preference_np, dtype=jnp.float32)

        # ---- Rollout ------------------------------------------------------
        # GFN's train_traj has shape ``(B, T, ...)`` where B is the batch
        # of trajectories (num_envs for fresh-only, replay_batch_size for
        # off-policy). axis 0 is the natural batch dim, so we want
        # ``PartitionSpec("dev")`` (data_sharding) — NOT mu0's
        # ``PartitionSpec(None, "dev")`` (scan_data_sharding) which
        # shards axis 1 (mu0's outer "minibatch" reshape). Using
        # scan_data_sharding here causes the whole batch to land on
        # each device, so the attention ``f32[E*T*2, 4096, 4096]``
        # tensor blew past the 24 GiB on the 3090 in run 45301.
        data_ds = self.state.get("data_sharding")
        with self._active_mesh():
            env_states_new, fresh_traj = self.state["do_rollout"](
                self.state["agent"],
                self.state["env"],
                self.state["env_states"],
                self.state["vertex_features"],
                preferences,
                rollout_key,
            )
            self.state["env_states"] = env_states_new

        # ---- Numpy snapshot for stats / best tracking --------------------
        reward_weights_np = np.asarray(self.state["reward_weights"])
        terminal_rewards_full = np.asarray(fresh_traj.reward[:, -1, :])
        per_env_terminal = terminal_rewards_full  # (E, NUM_REWARDS)
        weighted_per_env = per_env_terminal * reward_weights_np
        per_env_tot = weighted_per_env.sum(axis=-1)
        best_idx = int(per_env_tot.argmax())

        # Update per-channel running stats (MOGFN-PC normalization).
        # Update from the FRESH per-env terminal rewards each episode —
        # cheap (E rows × NUM_REWARDS cols, numpy ops) and gives the
        # loss a moving baseline that tracks the policy's evolving
        # reward distribution. Stats are warmed lazily on the first
        # update; while cold, mean=0 / var=1 so normalization is a
        # no-op the first ep.
        self.reward_stats.update(terminal_rewards_full)
        reward_stats_mean_j = jnp.asarray(self.reward_stats.mean, dtype=jnp.float32)
        reward_stats_var_j = jnp.asarray(self.reward_stats.var, dtype=jnp.float32)

        v_np = np.asarray(fresh_traj.vertex_idx)
        # Typed micro-action sequences (DIAG/COMPRESS/QUANT, up to max_substeps
        # rules per vertex) → decode each env's full per-vertex rule list.
        mop_np = np.asarray(fresh_traj.micro_op_seq)
        mi_np = np.asarray(fresh_traj.micro_i_seq)
        mj_np = np.asarray(fresh_traj.micro_j_seq)
        mf_np = np.asarray(fresh_traj.micro_factor_seq)
        mk_np = np.asarray(fresh_traj.micro_kind_seq)
        mq_np = np.asarray(fresh_traj.micro_quant_seq)
        _max_substeps = int(self.state["max_substeps"])

        def _decode_seq(idx: int):
            return _action_to_pylist_dynamic(
                v_np[idx], mop_np[idx], mi_np[idx], mj_np[idx],
                mf_np[idx], mk_np[idx], mq_np[idx], _max_substeps,
            )

        best_per_reward: dict[str, dict] = {}
        for j in range(NUM_REWARDS):
            if float(reward_weights_np[j]) == 0.0:
                continue
            bidx = int(per_env_terminal[:, j].argmax())
            all_raw = {
                REWARD_NAMES[k]: float(per_env_terminal[bidx, k])
                for k in range(NUM_REWARDS)
            }
            all_weighted = {
                REWARD_NAMES[k]: float(weighted_per_env[bidx, k])
                for k in range(NUM_REWARDS)
            }
            best_per_reward[REWARD_NAMES[j]] = {
                "raw_value": float(per_env_terminal[bidx, j]),
                "weighted_value": float(weighted_per_env[bidx, j]),
                "weighted_total": float(per_env_tot[bidx]),
                "env_idx": bidx,
                "seq": _decode_seq(bidx),
                "all_raw": all_raw,
                "all_weighted": all_weighted,
            }

        best_overall_rewards = {
            REWARD_NAMES[j]: float(per_env_terminal[best_idx, j])
            for j in range(NUM_REWARDS)
        }
        best_overall_weighted = {
            REWARD_NAMES[j]: float(weighted_per_env[best_idx, j])
            for j in range(NUM_REWARDS)
        }
        per_reward_means = {
            REWARD_NAMES[j]: float(per_env_terminal[:, j].mean())
            for j in range(NUM_REWARDS)
        }
        # In GFN the reward is *only* defined at terminal — ``per_reward_means``
        # above is computed on per_env_terminal, so terminal_means == per_reward_means.
        # Still emit terminal_means / best_terminal for naming-consistency
        # with the unified PPO / MuZero logging layout (Infra 1).
        terminal_means = dict(per_reward_means)
        best_terminal = {
            REWARD_NAMES[j]: float(per_env_terminal[:, j].max())
            for j in range(NUM_REWARDS)
        }
        terminal_cs = per_env_terminal[:, REWARD_INDEX["cosine_sim"]]

        # MOGFN Pareto-archive input: each (non-sentinel) sampled terminal's
        # objective vector PAIRED with the full typed micro-action sequence
        # that produced it (mirrors the C-MORL Pareto front). Sentinel
        # filtering semantics live in the shared helper — see its docstring
        # for why exact equality (not a magnitude threshold) is required.
        from alphagrad.approx.common.cache import (
            SENTINEL_REWARD_VALUE as _SENTINEL,
        )
        from alphagrad.approx.common.reward_scaling import (
            build_terminal_solutions as _build_terminal_solutions,
        )
        terminal_solutions = _build_terminal_solutions(
            per_env_terminal, _decode_seq, _SENTINEL,
        )

        stats: dict = {
            "best_return": float(per_env_tot[best_idx]),
            "mean_return": float(per_env_tot.mean()),
            "terminal_solutions": terminal_solutions,
            "best_seq": _decode_seq(best_idx),
            "best_overall_rewards": best_overall_rewards,
            "best_overall_weighted": best_overall_weighted,
            "per_reward_means": per_reward_means,
            "terminal_means": terminal_means,
            "best_terminal": best_terminal,
            "best_per_reward": best_per_reward,
            # TB has no MCTS visit-distribution — entropy is the agent's
            # forward-policy entropy averaged over the rollout (computed
            # inside loss_fn below, fed into stats once we have it).
            "entropy_mean": float("nan"),
            "entropy_root": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": 0.0,
            "reward_loss": 0.0,
        }
        # Unified `reward/{cost,quality}/{per_step,terminal,best_terminal}/<name>`
        # keys + corridor instrumentation (Infra 1). Legacy
        # ``per_reward_means`` / ``reward_mean/*`` remain for back-compat.
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

        # ---- Replay buffer lifecycle -------------------------------------
        if self.replay_buffer is None and args.replay_buffer_size > 0:
            self.replay_buffer = init_replay_buffer(
                jax.tree_util.tree_map(lambda x: x[0], fresh_traj),
                args.replay_buffer_size,
            )
            if args.replay_checkpoint_path and os.path.exists(
                args.replay_checkpoint_path
            ):
                try:
                    self.replay_buffer = load_replay_buffer(
                        args.replay_checkpoint_path, self.replay_buffer,
                    )
                    print(
                        f"[gfn_ray_worker] replay resumed from "
                        f"{args.replay_checkpoint_path} "
                        f"(size={int(self.replay_buffer.size)}/"
                        f"{self.replay_buffer.capacity})"
                    )
                except Exception as exc:
                    print(f"[gfn_ray_worker] replay load failed: {exc}")

        # ---- Training: outer × inner gradient steps -----------------------
        ep = self._episode_counter
        with self._active_mesh():
            for _ in range(max(int(train_steps), 1)):
                # Compose train batch: fresh + replay mix (gfn.py:820-854)
                if (
                    args.replay_buffer_size > 0
                    and self.replay_buffer is not None
                    and ep >= args.replay_warmup
                ):
                    target_batch_size = (
                        args.replay_batch_size if args.replay_batch_size > 0
                        else self.state["num_envs"]
                    )
                    n_fresh = min(
                        int(target_batch_size * args.replay_fresh_fraction),
                        self.state["num_envs"],
                    )
                    n_replay = target_batch_size - n_fresh
                    if n_replay > 0:
                        rep_key, sample_key = jrand.split(sample_key)
                        replay_traj = replay_sample(
                            self.replay_buffer, n_replay, rep_key,
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

                if data_ds is not None:
                    train_traj = jax.tree.map(
                        lambda x: jax.device_put(x, data_ds), train_traj,
                    )

                t_key, train_key = jrand.split(train_key)
                self.state["agent"], self.state["opt_state"], metrics = (
                    self.state["do_train_steps"](
                        self.state["agent"], self.state["opt_state"],
                        train_traj, self.state["vertex_features"], beta,
                        reward_stats_mean_j, reward_stats_var_j, t_key,
                    )
                )
                self.train_step_counter += int(args.gradient_steps)

        # Unpack the last train_steps' worth of metrics into stats.
        (
            tb_loss, logZ_val, mean_log_pf, mean_log_pb, mean_log_R,
            mean_terminal, entropy_term, total_loss,
        ) = [float(x) for x in metrics]
        stats.update({
            "policy_loss": tb_loss,
            "total_loss": total_loss,
            "tb_loss": tb_loss,
            "logZ": logZ_val,
            "mean_log_pf": mean_log_pf,
            "mean_log_pb": mean_log_pb,
            "mean_log_R": mean_log_R,
            "mean_terminal_reward": mean_terminal,
            "entropy_mean": entropy_term,
            "beta": float(beta),
        })

        # ---- Add fresh trajectories to the replay buffer ------------------
        if self.replay_buffer is not None:
            traj_priorities = _symlog_priorities(per_env_tot)
            self.replay_buffer = replay_add_batch(
                self.replay_buffer, fresh_traj, priorities=traj_priorities,
            )
            if (
                args.replay_checkpoint_path
                and (ep + 1) % max(int(args.replay_checkpoint_every), 1) == 0
            ):
                try:
                    save_replay_buffer(
                        self.replay_buffer, args.replay_checkpoint_path,
                    )
                except Exception as exc:
                    print(f"[gfn_ray_worker] replay save failed: {exc}")

        stats["buffer_size"] = (
            int(self.replay_buffer.size) if self.replay_buffer else 0
        )
        stats["train_step"] = self.train_step_counter

        # ---- Pool telemetry + cascading recycle (mirror mu0_ray_worker) ---
        if self._pool is not None:
            stats.update({f"pool/{k}": v for k, v in self._pool.stats().items()})
            # Per-episode timeout delta (see ppo_ray_worker equivalent).
            try:
                stats["pool/timeouts_this_episode"] = (
                    self._pool.fetch_timeout_delta()
                )
            except Exception:
                stats["pool/timeouts_this_episode"] = 0
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
                n_pool = max(self._pool.size(), 1)
                interval = max(1, self._recycle_every // n_pool)
                # Recycle on episode boundary — the TB inner loop runs
                # ``train_steps × gradient_steps`` gradient updates,
                # but the pool is exercised once per rollout, so we
                # gate on episode count not step count.
                if (ep + 1) % interval == 0:
                    new_size = self._pool.recycle_one()
                    stats["pool/recycled_at_ep"] = ep
                    stats["pool/size_after_recycle"] = new_size

        # ---- Periodic checkpoint -----------------------------------------
        if (
            self._checkpoint_path
            and self._checkpoint_every > 0
            and (ep + 1) % self._checkpoint_every == 0
        ):
            self._save_checkpoint_safe()
            stats["checkpoint/saved_at_ep"] = ep

        self._episode_counter += 1
        return stats

    # ------------------------------------------------------------------
    # Calibration / reward-weight writeback / pool poll / replay save
    # ------------------------------------------------------------------

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Per-channel calibration statistics over ``num_rollouts``
        zero-pref rollouts. Same dict schema as
        :meth:`alphagrad.approx.mu0_ray_worker.SPMDServerWorker.reward_vec_means`.
        """
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            filter_sentinel_mask,
            symlog_np,
        )

        all_rows: list[np.ndarray] = []
        ds = self.state.get("data_sharding")
        num_devs = len(jax.devices()) if ds is not None else 1

        with self._active_mesh():
            for i in range(num_rollouts):
                # NUM_VALUE_HEADS, not NUM_REWARDS — see _sample_preferences
                # for the rationale.
                pref = jnp.zeros(
                    (self.state["num_envs"], NUM_VALUE_HEADS), jnp.float32,
                )
                env_states = jax.vmap(lambda _: self.state["env"].reset())(
                    jnp.arange(self.state["num_envs"])
                )
                keys = jrand.PRNGKey(int(rng_seed) + i * 31 + 1)

                if ds is not None and self.state["num_envs"] % num_devs == 0:
                    pref = jax.device_put(pref, ds)
                    env_states = jax.tree.map(
                        lambda x: jax.device_put(x, ds), env_states,
                    )

                _, traj = self.state["do_rollout"](
                    self.state["agent"],
                    self.state["env"],
                    env_states,
                    self.state["vertex_features"],
                    pref,
                    keys,
                )
                rv = np.asarray(traj.reward)  # (E, T, NUM_REWARDS)
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

    def get_pool_stats(self) -> dict:
        if self._pool is None:
            return {}
        return self._pool.stats()

    def _save_checkpoint_safe(self) -> None:
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
            print(f"[gfn_ray_worker] checkpoint save failed: {exc}")

    def ready(self) -> bool:
        return True
