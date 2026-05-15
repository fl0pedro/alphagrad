"""Ray-actor host of PPO training using the external-tokenizer path.

First cut of the Phase-3+5 PPO Ray version from the migration plan:

* JAX-side rollout uses `env.step_external_jax_part` (no `io_callback`).
  Per timestep the worker JIT-runs vmap(act + step_external) over the
  env batch, ships the resulting `(order, specs)` triples to a pool of
  `CpuApproximationActor` Ray actors for the graphax tokenize / jacve
  compile / cost-analysis work, then stitches the tokenizer output back
  in via `env.assemble_step_result`. The result is functionally
  identical to `env.step()` but moves the host-side Python pass off the
  trainer thread.

* PPO update is a deliberately simple clip-loss + value-MSE + entropy.
  We do NOT replicate every feature of the single-process `ppo.py`
  trainer here (no curriculum / Lagrangian / dynamic-substeps /
  preference Dirichlet / replay buffer — those will land as follow-ups
  if the basic path proves out). The trainer scalarises the env's
  8-dim reward vector to a single scalar via a fixed weight vector
  derived from `--rewards / --cmp-type / --mem-type / --lambda-*` —
  same shape as `ppo._build_reward_weights`.

* Single-GPU for now. The mu0_ray_worker SPMD shard (Mesh + NamedSharding)
  is the natural follow-up but kept out of the first cut to limit
  surface area.

Interface (matches `mu0_ray_worker.SPMDServerWorker` shape so the driver
in `ppo_ray.py` can drop in identically):

    worker = PPORayWorker(args_dict, seed, cpu_workers=cpu_workers)
    stats = worker.run_rollout_and_train(rng_seed)
"""

from __future__ import annotations

import os
from functools import partial
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
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
    vertex_avail_at_step,
)
from alphagrad.approx.common.gae import get_advantages, reward_normalization_fn
from alphagrad.utils import symlog
from alphagrad.approx.env import (
    MAX_AXES_PER_VERTEX,
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
    micro_actions_to_rule_specs_jax,
)
from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END


# Stage F: which reward channels skip symlog when used in the
# Lagrangian comparison. Currently just cosine_sim — it's already
# bounded to [0, 1], so symlog'ing it would only complicate the
# threshold semantics. Mirrored from ppo._NO_SYMLOG_REWARD_INDICES.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)


def _parse_lagrangian_constraints(specs: list) -> list[tuple[int, float, int]]:
    """Parse ``--lagrangian-constraint`` strings to (idx, threshold, sign).

    * ``NAME>=THRESH`` → sign=+1, violation when reward < threshold
    * ``NAME<=THRESH`` → sign=−1, violation when reward > threshold

    Duplicates ``ppo.parse_lagrangian_constraints`` (kept local so this
    file doesn't pull in the 6k-line single-process ppo module).
    """
    parsed: list[tuple[int, float, int]] = []
    for s in specs or []:
        if ">=" in s:
            op, sign = ">=", 1
        elif "<=" in s:
            op, sign = "<=", -1
        else:
            raise ValueError(
                f"--lagrangian-constraint must be NAME>=THRESH or NAME<=THRESH, "
                f"got {s!r}"
            )
        name, thresh_s = s.split(op, 1)
        name = name.strip()
        if name not in REWARD_INDEX:
            raise ValueError(
                f"Unknown reward name {name!r} in {s!r}; "
                f"valid names: {list(REWARD_INDEX.keys())}"
            )
        parsed.append((REWARD_INDEX[name], float(thresh_s.strip()), sign))
    return parsed


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    return SimpleNamespace(**args_dict)


def _setup_jax_compile_cache() -> None:
    """Match the cache-dir setup used by `ppo.py` so the disk-cache
    path is shared across process boundaries. Mirrored from
    `mu0._setup_jax_compile_cache` (kept inline to avoid pulling in
    mu0's NumPy-import side effects)."""
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "jax-compile"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


def _build_reward_weights(args) -> np.ndarray:
    """Compose CLI lambda flags into a single (NUM_REWARDS,) scalarising
    weight vector. Strict subset of `ppo._build_reward_weights`'s logic —
    we omit the per-head split (one head, one scalar) since the first
    cut uses a single value head only.
    """
    w = np.zeros(NUM_REWARDS, dtype=np.float32)
    if "cmp" in args.rewards:
        cmp_idx = {
            "graphax": REWARD_INDEX["muls_adds_fmas"],
            "flops": REWARD_INDEX["flops"],
            "latency": REWARD_INDEX["latency_ns"],
        }[args.cmp_type]
        w[cmp_idx] = float(getattr(args, "lambda_cmp", 1.0))
    if "mem" in args.rewards:
        mem_idx = {
            "graphax": REWARD_INDEX["max_io_sum"],
            "bytes_accessed": REWARD_INDEX["bytes_accessed"],
            "peak_memory": REWARD_INDEX["peak_memory"],
        }[args.mem_type]
        w[mem_idx] = float(getattr(args, "lambda_mem", 1.0))
    if "acc" in args.rewards:
        w[REWARD_INDEX["cosine_sim"]] = 1.0
    lam_frob = float(getattr(args, "lambda_frob", 0.0))
    if lam_frob != 0.0:
        w[REWARD_INDEX["frob_residual"]] = lam_frob
    # If --rewards somehow ended up empty, fall back to muls_adds_fmas:
    # we need *some* signal for the loss to be non-degenerate.
    if not np.any(w):
        w[REWARD_INDEX["muls_adds_fmas"]] = 1.0
    return w


# ---------------------------------------------------------------------------
# Minimal Agent — small footprint by design.
#
# Stripped-down twin of `ppo.Agent`: encoder + vertex pointer head + single
# value head. We deliberately do NOT carry the rule policy / micro-action
# policy / per-head value split / preference projection — those features
# are not in the first-cut PPO loop. The action this agent emits is a
# pure (vertex_idx, no-rules) StepAction, which is the same as the
# `ve_only` variant in `variants.py`.
#
# Reusing the building blocks from `alphagrad.transformer` keeps params
# / encoder identical to the bigger Agent so we can grow into the missing
# features by swapping module fields in.
# ---------------------------------------------------------------------------
class SimplePPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: Any
    encoder: Any
    vertex_logits_head: Any  # MLP token-pooled -> (num_vertices,)
    value_head: Any          # MLP token-pooled -> ()
    # Dynamic-substeps heads (Phase C). Present iff
    # ``--dynamic-substeps`` is on. Each emits logits for one
    # axis/op/factor component of a 1-substep micro-action per vertex.
    # ``None`` when the flag is off — the agent then emits empty
    # rule_specs (the legacy `ve_only` variant).
    op_type_head: Any  # MLP token-pooled -> 2  (DIAG, END)
    i_head: Any        # MLP token-pooled -> MAX_AXES_PER_VERTEX
    j_head: Any        # MLP token-pooled -> MAX_AXES_PER_VERTEX
    factor_head: Any   # MLP token-pooled -> num_factors

    embd_dim: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)
    dynamic_substeps: bool = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size: int,
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_vertices: int,
        policy_dims: tuple[int, ...],
        value_dims: tuple[int, ...],
        key,
        dynamic_substeps: bool = False,
        num_factors: int = 4,
    ):
        from alphagrad.transformer import MLP, Encoder, PositionalEncoder

        keys = jrand.split(key, 8)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=keys[1])
        self.vertex_logits_head = MLP(
            embd_dim, num_vertices, policy_dims, key=keys[2],
        )
        self.value_head = MLP(embd_dim, 1, value_dims, key=keys[3])
        if dynamic_substeps:
            # 3-way op-type head: 0=DIAG, 1=COMPRESS, 2=END. The
            # `compress_kinds` parameter on the env's translator stays
            # at the default (0=mean) for now; learning the kind is an
            # easy follow-up (add a 5th categorical head over
            # COMPRESS_KINDS) but isn't needed for the first FULL run.
            self.op_type_head = MLP(embd_dim, 3, policy_dims, key=keys[4])
            self.i_head = MLP(embd_dim, MAX_AXES_PER_VERTEX, policy_dims, key=keys[5])
            self.j_head = MLP(embd_dim, MAX_AXES_PER_VERTEX, policy_dims, key=keys[6])
            self.factor_head = MLP(embd_dim, num_factors, policy_dims, key=keys[7])
        else:
            self.op_type_head = None
            self.i_head = None
            self.j_head = None
            self.factor_head = None
        self.embd_dim = embd_dim
        self.num_vertices = num_vertices
        self.dynamic_substeps = dynamic_substeps
        self.num_factors = num_factors

    def encode(self, tokens, key):
        """Return token-pooled context vector ``(embd_dim,)``."""
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        x = self.encoder(x, key=key)
        # Mean-pool over non-pad tokens. tokens==0 is the pad token in the
        # graphax tokenizer; the mask is 1 for real tokens, 0 for pad.
        mask = (tokens > 0).astype(x.dtype)[:, None]
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        return jnp.sum(x * mask, axis=0) / denom

    def policy_logits(self, tokens, key):
        ctx = self.encode(tokens, key=key)
        return self.vertex_logits_head(ctx)

    def value(self, tokens, key):
        ctx = self.encode(tokens, key=key)
        return jnp.squeeze(self.value_head(ctx), axis=-1)

    def policy_and_value(self, tokens, key):
        ctx = self.encode(tokens, key=key)
        logits = self.vertex_logits_head(ctx)
        value = jnp.squeeze(self.value_head(ctx), axis=-1)
        return logits, value

    def micro_action_logits(self, tokens, key):
        """Return ``(op_logits, i_logits, j_logits, factor_logits)``.

        Each one is a flat categorical over its component's choice set.
        Only callable when ``dynamic_substeps`` is on; the caller is
        responsible for not invoking this on a non-dynamic agent.
        """
        ctx = self.encode(tokens, key=key)
        return (
            self.op_type_head(ctx),
            self.i_head(ctx),
            self.j_head(ctx),
            self.factor_head(ctx),
        )

    def all_logits(self, tokens, key):
        """Single-encode variant: ``(vertex_logits, value, op, i, j, factor)``.

        Combined entry point that runs the encoder once and dispatches
        to every head. Previously the rollout / loss paths called
        ``policy_and_value`` and ``micro_action_logits`` back-to-back,
        each of which re-encoded the tokens — i.e. one encoder forward
        pass per head group. The transformer is the dominant model
        cost, so collapsing the two calls into one is a ~2× win on the
        per-step / per-sample GPU compute when ``dynamic_substeps`` is
        on (no behavioural change otherwise — same softmax inputs).

        On a non-dynamic agent the micro heads are ``None``; the
        returned slots are placeholder zero arrays so the caller's
        downstream unpacking stays uniform.
        """
        ctx = self.encode(tokens, key=key)
        vertex_logits = self.vertex_logits_head(ctx)
        value = jnp.squeeze(self.value_head(ctx), axis=-1)
        if self.dynamic_substeps:
            return (
                vertex_logits, value,
                self.op_type_head(ctx),
                self.i_head(ctx),
                self.j_head(ctx),
                self.factor_head(ctx),
            )
        # Placeholders that match the dynamic shapes — the rollout /
        # loss paths gate on ``self.dynamic_substeps`` before reading
        # these, so the values are never observed.
        zero_op = jnp.zeros((3,), dtype=jnp.float32)  # DIAG / COMPRESS / END
        zero_axes = jnp.zeros((MAX_AXES_PER_VERTEX,), dtype=jnp.float32)
        zero_factor = jnp.zeros((max(self.num_factors, 1),), dtype=jnp.float32)
        return vertex_logits, value, zero_op, zero_axes, zero_axes, zero_factor


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class PPORayWorker:
    """Owns env + agent + opt_state for one Ray trainer."""

    def __init__(self, args_dict: dict, seed: int = 0, cpu_workers: list | None = None):
        self.args = _args_from_dict(args_dict)
        self.seed = int(seed)
        self.cpu_workers = list(cpu_workers) if cpu_workers else []
        if self.args.no_jit:
            jax.config.update("jax_disable_jit", True)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        _setup_jax_compile_cache()

        key = jrand.PRNGKey(self.seed)
        key, args_key, eval_key, agent_key, init_key = jrand.split(key, 5)
        self._rng_key = key

        # Build env exactly like cpu_approx_worker but keep the JAX-side
        # references locally — we need both `args` (for tokenizer/eval)
        # and `valid_vertices` for the action mask.
        dataset_arg = None if self.args.dataset == "none" else self.args.dataset
        use_dataset = (
            dataset_arg is not None
            and self.args.example.endswith("NeuralNetwork")
        )
        dataset_for_call = dataset_arg if use_dataset else None
        target_fn = get_fn(self.args.example)
        xs = get_args(self.args.example, args_key, dataset=dataset_for_call)
        gen = data_gen(
            self.args.example,
            dataset=dataset_for_call,
            dataset_size=self.args.dataset_size,
        )
        closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
        argnums = infer_argnums(self.args.example)
        env_target_fun = target_fn if "acc" in self.args.rewards else None
        measure_latency = bool(
            getattr(self.args, "measure_latency", False)
            or self.args.cmp_type == "latency"
        )
        env = VertexEliminationEnv.from_jaxpr(
            closed_jaxpr,
            args=xs,
            argnums=argnums,
            num_envs=0,
            data_gen=gen,
            target_fun=env_target_fun,
            cmp_type=self.args.cmp_type,
            mem_type=self.args.mem_type,
            exec_on_gpu=getattr(self.args, "exec_on_gpu", False),
            measure_latency=measure_latency,
            terminal_rewards_only=getattr(
                self.args, "terminal_rewards_only", False,
            ),
        )
        eval_samples = generate_eval_samples(
            env, eval_key, int(self.args.num_eval_samples),
        )
        self.env = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        self.total_v = len(closed_jaxpr.jaxpr.eqns)
        self.num_valid = len(env.valid_vertices)
        self.vertex_valid_static = build_vertex_valid_static(
            env.valid_vertices, self.total_v,
        )
        self.rollout_length = self.num_valid
        # SPMD mesh — replicate agent / opt_state across all visible
        # devices, shard env_states along the env axis. When there's only
        # one device, the sharding annotations are no-ops; the same code
        # path runs on a laptop CPU and a 4-GPU node.
        devices = jax.devices()
        self.num_devices = len(devices)
        self.mesh = Mesh(np.asarray(devices), axis_names=("dev",))
        self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("dev"))
        # Update_step batches go through a 2-D layout
        # ``(num_minibatches, mb_size, ...)`` — the data-parallel axis is
        # the mb_size dimension (axis 1) since the minibatch axis is the
        # Python-side loop variable. Mirrors `scan_data_sharding` in
        # mu0_ray_worker. Keeps the per-device working set bounded by
        # mb_size / num_devices instead of by mb_size.
        self.scan_data_sharding = NamedSharding(
            self.mesh, PartitionSpec(None, "dev"),
        )

        # Auto-tune num_envs to a multiple of num_devices so the shard
        # split is even. Mirrors mu0_ray_worker's tuning — we silently
        # round up rather than reject the config so the user can pass
        # round numbers without having to remember the device count.
        requested_envs = int(getattr(self.args, "num_envs", 4))
        if self.num_devices > 1 and requested_envs % self.num_devices != 0:
            self.num_envs = (
                max(self.num_devices, round(requested_envs / self.num_devices))
                * self.num_devices
            )
            print(
                f"[ppo_ray] num_envs tuned: {requested_envs} -> "
                f"{self.num_envs} (multiple of {self.num_devices} devices)"
            )
        else:
            self.num_envs = max(requested_envs, self.num_devices)
        self.minibatches = max(int(getattr(self.args, "minibatches", 1)), 1)
        self.ppo_eps = float(getattr(self.args, "ppo_eps", 0.2))
        self.value_coef = float(getattr(self.args, "value_coef", 0.5))
        self.entropy_coef = float(getattr(self.args, "entropy_coef", 0.01))
        self.gae_lambda = float(getattr(self.args, "gae_lambda", 0.95))
        self.discount = float(getattr(self.args, "discount", 0.99))
        self.reward_weights_np = _build_reward_weights(self.args)
        self.reward_weights = jnp.asarray(
            self.reward_weights_np, dtype=jnp.float32,
        )

        # Stage F Lagrangian state. Stored as numpy because the multipliers
        # update with simple dual-ascent steps after each episode — no
        # need to keep them on the device. Empty config = no constraints
        # (the violation path becomes a no-op).
        constraint_specs = _parse_lagrangian_constraints(
            getattr(self.args, "lagrangian_constraint", []) or []
        )
        if constraint_specs:
            self.constraint_indices_np = np.array(
                [c[0] for c in constraint_specs], dtype=np.int32,
            )
            self.constraint_thresholds_np = np.array(
                [c[1] for c in constraint_specs], dtype=np.float32,
            )
            self.constraint_signs_np = np.array(
                [c[2] for c in constraint_specs], dtype=np.float32,
            )
            self.constraint_names = [
                REWARD_NAMES[c[0]] + (">=" if c[2] > 0 else "<=") + str(c[1])
                for c in constraint_specs
            ]
            # No-symlog mask gathered per-constraint, so we can skip the
            # symlog transform on the cosine-sim channel (and anything
            # else flagged in _NO_SYMLOG_REWARD_INDICES).
            self.constraint_no_symlog_np = np.array(
                [c[0] in _NO_SYMLOG_REWARD_INDICES for c in constraint_specs],
                dtype=np.bool_,
            )
        else:
            self.constraint_indices_np = np.zeros((0,), dtype=np.int32)
            self.constraint_thresholds_np = np.zeros((0,), dtype=np.float32)
            self.constraint_signs_np = np.zeros((0,), dtype=np.float32)
            self.constraint_names = []
            self.constraint_no_symlog_np = np.zeros((0,), dtype=np.bool_)
        self.multipliers_np = np.zeros(
            (self.constraint_indices_np.shape[0],), dtype=np.float32,
        )
        self.lagrangian_lr = float(getattr(self.args, "lagrangian_lr", 1e-3))

        # Agent + optimizer.
        policy_dims = self._parse_int_list(self.args.policy_dims)
        value_dims = self._parse_int_list(self.args.value_dims)
        self.dynamic_substeps = bool(
            getattr(self.args, "dynamic_substeps", False)
        )
        self.factor_table = self._parse_int_list(
            getattr(self.args, "factors", "-1,2,3,4")
        )
        if not self.factor_table:
            self.factor_table = (-1, 2, 3, 4)
        self.factor_table_j = jnp.asarray(self.factor_table, dtype=jnp.int32)
        self.num_factors = len(self.factor_table)
        self.agent = SimplePPOAgent(
            vocab_size=int(self.args.vocab_size),
            embd_dim=int(self.args.embd_dim),
            num_layers=int(self.args.num_layers),
            num_heads=int(self.args.num_heads),
            hidden_dim=int(self.args.hidden_dim),
            num_vertices=self.total_v,
            policy_dims=policy_dims,
            value_dims=value_dims,
            key=agent_key,
            dynamic_substeps=self.dynamic_substeps,
            num_factors=self.num_factors,
        )
        self.agent = init_linear_weights(self.agent, init_key)
        # Replicate the agent across all devices — under SPMD this is
        # cheap (params are < 100 MB) and lets `act_step` / loss path
        # run sharded without the trainer having to think about it.
        self.agent = self._replicate(self.agent)

        schedule = optax.cosine_decay_schedule(
            float(self.args.lr),
            int(self.args.episodes) * self.minibatches,
            float(getattr(self.args, "lr_decay_min_mult", 0.1)),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(self.args.max_grad_norm)),
            optax.adamw(schedule, eps=float(self.args.adam_eps)),
        )
        self.opt_state = self.optimizer.init(
            eqx.filter(self.agent, eqx.is_inexact_array),
        )
        self.opt_state = self._replicate(self.opt_state)

        # Pre-reset batched env state. The env axis (axis 0) is the
        # data-parallel dimension; shard it across devices so each
        # GPU owns ``num_envs // num_devices`` envs.
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )

        # Cache the per-env vertex-valid mask. It's static (depends only on
        # the env's `valid_vertices`), so we vectorise once.
        self._vertex_valid_static_j = jnp.asarray(
            self.vertex_valid_static, dtype=jnp.float32,
        )

        self._episode_counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_int_list(raw: str) -> tuple[int, ...]:
        if not raw:
            return ()
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())

    def _replicate(self, tree):
        """Place every array leaf onto every device (replicated sharding).

        Static (non-array) leaves pass through unchanged. Matches the
        `shard_leaf` pattern in `mu0_ray_worker._build_actor_state`.
        """
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.replicated_sharding)
            if eqx.is_array(x) else x,
            tree,
        )

    def _vertex_avail(self, state):
        """`(num_envs, num_vertices)` 0/1 mask of vertices still in the
        active suffix of `state.order`. Matches the action-mask convention
        the policy uses to invalidate finished vertices."""
        return jax.vmap(
            lambda s: vertex_avail_at_step(
                s,
                self._vertex_valid_static_j,
                self.total_v,
                self.num_valid,
            ),
        )(state)

    def _scalar_reward(self, reward_vec):
        """Reduce the env's 8-component reward vector to a scalar via
        the per-component weight vector. Symlog is applied downstream
        by `get_advantages` (per the gae.py contract)."""
        return jnp.sum(reward_vec * self.reward_weights, axis=-1)

    # ------------------------------------------------------------------
    # JIT'd per-step act + JIT-side env update
    # ------------------------------------------------------------------
    def _make_act_step_fn(self):
        env = self.env
        dynamic = self.dynamic_substeps
        factor_table = self.factor_table_j

        @eqx.filter_jit
        def act_step(agent, state_batch, vert_avail_batch, key):
            keys = jrand.split(key, self.num_envs)

            def per_env(state_i, avail_i, k_i):
                # Up to 5 sub-keys: 1 for the encode pass, 4 for the
                # action heads (vertex / op / i / j / factor). With
                # `dynamic_substeps` off only the first two are used;
                # the others are simply unused.
                k_enc, k_v, k_op, k_i, k_j, k_f = jrand.split(k_i, 6)
                # Single encoder pass shared across the vertex / value
                # heads AND the micro-action heads — see
                # `SimplePPOAgent.all_logits`. When dynamic_substeps
                # is off, the four micro slots are placeholders that
                # the `if dynamic:` branch below never reads.
                logits, value, op_l, i_l, j_l, f_l = agent.all_logits(
                    state_i.tokens, key=k_enc,
                )
                masked = jnp.where(avail_i > 0.5, logits, -1e9)
                log_probs_v = jax.nn.log_softmax(masked)
                vertex_action = jrand.categorical(k_v, masked)
                log_prob_v = log_probs_v[vertex_action]
                vertex_id = vertex_action + 1  # env vertex IDs are 1-indexed

                if dynamic:
                    # Mask i / j logits by the chosen vertex's axis_valid.
                    axis_valid_v = state_i.axis_valid_mask[
                        vertex_id - jnp.int32(1)
                    ]  # (MAX_AXES_PER_VERTEX,)
                    i_l_masked = jnp.where(axis_valid_v > 0.5, i_l, -1e9)
                    j_l_masked = jnp.where(axis_valid_v > 0.5, j_l, -1e9)
                    op_sample = jrand.categorical(k_op, op_l)
                    i_sample = jrand.categorical(k_i, i_l_masked)
                    j_sample = jrand.categorical(k_j, j_l_masked)
                    f_sample = jrand.categorical(k_f, f_l)
                    log_prob_op = jax.nn.log_softmax(op_l)[op_sample]
                    log_prob_i = jax.nn.log_softmax(i_l_masked)[i_sample]
                    log_prob_j = jax.nn.log_softmax(j_l_masked)[j_sample]
                    log_prob_f = jax.nn.log_softmax(f_l)[f_sample]
                    # 3-way op-type: 0→DIAG, 1→COMPRESS, 2→END. The
                    # env's translator routes COMPRESS to a Compress
                    # transform only when the chosen vertex is the
                    # last one in the partial elimination order (see
                    # env._callback's `v_idx != last_v_idx` guard);
                    # for any earlier vertex the COMPRESS slot is
                    # silently dropped. The policy gradient still
                    # flows through the op-type head — the env's
                    # behaviour just degrades gracefully to "no rule"
                    # on the bad placements.
                    op_type = jnp.where(
                        op_sample == 0, OP_DIAG,
                        jnp.where(op_sample == 1, OP_COMPRESS, OP_END),
                    )
                    factor_val = factor_table[f_sample]
                    rule_specs = micro_actions_to_rule_specs_jax(
                        op_types=jnp.array([op_type], dtype=jnp.int32),
                        i_indices=jnp.array([i_sample], dtype=jnp.int32),
                        j_indices=jnp.array([j_sample], dtype=jnp.int32),
                        factors=jnp.array([factor_val], dtype=jnp.int32),
                        axis_state_for_vertex=state_i.axis_state[
                            vertex_id - jnp.int32(1)
                        ],
                    )
                else:
                    op_sample = jnp.int32(0)
                    i_sample = jnp.int32(0)
                    j_sample = jnp.int32(0)
                    f_sample = jnp.int32(0)
                    log_prob_op = jnp.float32(0.0)
                    log_prob_i = jnp.float32(0.0)
                    log_prob_j = jnp.float32(0.0)
                    log_prob_f = jnp.float32(0.0)
                    rule_specs = jnp.full(
                        (MAX_RULES_PER_VERTEX, 3), -1, dtype=jnp.int32,
                    ).at[..., 2].set(0)

                env_action = StepAction(
                    target_vertex=jnp.asarray(vertex_id, dtype=jnp.int32),
                    rule_specs=rule_specs,
                )
                partial, order, specs, step = env.step_external_jax_part(
                    state_i, env_action,
                )
                # Joint log-prob over the five action components. PPO
                # clip operates on this sum; equivalently the ratio is
                # the product of per-head ratios. Heads that didn't
                # participate (everything except vertex when dynamic
                # is off) contributed log_prob=0 and ratio=1.
                log_prob_total = (
                    log_prob_v + log_prob_op + log_prob_i + log_prob_j + log_prob_f
                )
                return (
                    vertex_action,
                    op_sample, i_sample, j_sample, f_sample,
                    log_prob_total,
                    value,
                    partial, order, specs, step,
                )

            return jax.vmap(per_env)(state_batch, vert_avail_batch, keys)

        return act_step

    def _make_assemble_fn(self):
        env = self.env

        @eqx.filter_jit
        def assemble(state_before, partial, tokens, eqn_ids, reward):
            def per_env(s_b, p, t, e, r):
                return env.assemble_step_result(s_b, p, t, e, r).state
            return jax.vmap(per_env)(
                state_before, partial, tokens, eqn_ids, reward,
            )

        return assemble

    # ------------------------------------------------------------------
    # PPO loss + minibatch update
    # ------------------------------------------------------------------
    def _make_update_step(self):
        clip_eps = self.ppo_eps
        value_coef = self.value_coef
        entropy_coef = self.entropy_coef
        dynamic = self.dynamic_substeps
        axis_valid_static_j = jnp.asarray(
            self.env.axis_valid_static, dtype=jnp.float32,
        )

        def loss_fn(agent, batch, key):
            (
                tokens, actions, op_a, i_a, j_a, f_a,
                vertex_idx_for_mask,
                old_log_probs, returns, advantages,
            ) = batch
            keys = jrand.split(key, tokens.shape[0])

            def per_sample(tok, v_act, op, i_s, j_s, f_s, v_for_mask,
                            olp, ret, adv, k):
                # Single encoder forward for both the vertex/value
                # heads and the micro-action heads; mirrors the
                # equivalent share in `act_step`. When dynamic is off
                # the (op_l, i_l, j_l, f_l) slots are placeholders and
                # the `if dynamic:` branch below skips them.
                logits, value, op_l, i_l, j_l, f_l = agent.all_logits(
                    tok, key=k,
                )
                log_probs = jax.nn.log_softmax(logits)
                lp_v = log_probs[v_act]
                p = jax.nn.softmax(logits)
                ent_v = -jnp.sum(p * log_probs)

                if dynamic:
                    axis_valid = axis_valid_static_j[v_for_mask - 1]
                    i_l_m = jnp.where(axis_valid > 0.5, i_l, -1e9)
                    j_l_m = jnp.where(axis_valid > 0.5, j_l, -1e9)
                    lp_op = jax.nn.log_softmax(op_l)[op]
                    lp_i = jax.nn.log_softmax(i_l_m)[i_s]
                    lp_j = jax.nn.log_softmax(j_l_m)[j_s]
                    lp_f = jax.nn.log_softmax(f_l)[f_s]
                    p_op = jax.nn.softmax(op_l)
                    ent_op = -jnp.sum(p_op * jax.nn.log_softmax(op_l))
                    # i / j entropy is over the valid-axes subset; mask
                    # the softmax denominator so the term doesn't see
                    # the -1e9 logits as low-but-nonzero probability.
                    p_i = jax.nn.softmax(i_l_m)
                    ent_i = -jnp.sum(p_i * jax.nn.log_softmax(i_l_m))
                    p_j = jax.nn.softmax(j_l_m)
                    ent_j = -jnp.sum(p_j * jax.nn.log_softmax(j_l_m))
                    p_f = jax.nn.softmax(f_l)
                    ent_f = -jnp.sum(p_f * jax.nn.log_softmax(f_l))
                    new_log_prob = lp_v + lp_op + lp_i + lp_j + lp_f
                    # Mean across the 5 heads so the entropy bonus has
                    # comparable scale to the single-head case.
                    entropy = (ent_v + ent_op + ent_i + ent_j + ent_f) / 5.0
                else:
                    new_log_prob = lp_v
                    entropy = ent_v

                ratio = jnp.exp(new_log_prob - olp)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -jnp.minimum(surr1, surr2)
                value_loss = (value - symlog(ret)) ** 2
                return policy_loss, value_loss, entropy

            p_l, v_l, ent = jax.vmap(per_sample)(
                tokens, actions, op_a, i_a, j_a, f_a,
                vertex_idx_for_mask,
                old_log_probs, returns, advantages, keys,
            )
            ppo_loss = jnp.mean(p_l)
            value_loss = jnp.mean(v_l)
            entropy_loss = -jnp.mean(ent)
            total = ppo_loss + value_coef * value_loss + entropy_coef * entropy_loss
            aux = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "entropy": jnp.mean(ent),
                "total_loss": total,
            }
            return total, aux

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        @eqx.filter_jit
        def update_step(agent, opt_state, batch, key):
            (loss, aux), grads = grad_fn(agent, batch, key)
            updates, new_opt_state = self.optimizer.update(
                grads, opt_state, agent,
            )
            new_agent = eqx.apply_updates(agent, updates)
            return new_agent, new_opt_state, aux

        return update_step

    # ------------------------------------------------------------------
    # Driver entry — one episode (rollout + train)
    # ------------------------------------------------------------------
    def run_rollout_and_train(self, rng_seed: int) -> dict:
        """One on-policy rollout + ppo-epochs * minibatches gradient steps.

        Returns a metrics dict ready to log to wandb.
        """
        if not hasattr(self, "_act_step"):
            self._act_step = self._make_act_step_fn()
            self._assemble = self._make_assemble_fn()
            self._update_step = self._make_update_step()

        key = jrand.PRNGKey(int(rng_seed))
        key, reset_key = jrand.split(key)

        # Fresh on-policy episode: reset state for every env. Preserve
        # the sharded layout (data-parallel along the env axis).
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )

        # Rollout buffers — numpy is fine; we re-stage to JAX once for
        # the update step.
        T = int(self.rollout_length)
        N = int(self.num_envs)
        buf_tokens = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        buf_actions = np.zeros((T, N), dtype=np.int32)
        # Per-head action samples — only meaningful when dynamic_substeps
        # is on. When off they stay zero and the loss treats them as
        # log_prob = 0 (ratio = 1, no policy contribution).
        buf_op = np.zeros((T, N), dtype=np.int32)
        buf_i = np.zeros((T, N), dtype=np.int32)
        buf_j = np.zeros((T, N), dtype=np.int32)
        buf_f = np.zeros((T, N), dtype=np.int32)
        # Vertex-id ↦ axis_state[v] gather slot. Stored so the loss path
        # can re-mask i / j logits identically to the rollout (the
        # axis_valid_mask is static per env across the episode, so we
        # only need to remember which vertex was acted on).
        buf_vertex_for_loss = np.zeros((T, N), dtype=np.int32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T, N), dtype=np.float32)
        buf_rewards = np.zeros((T, N), dtype=np.float32)
        # Raw 8-vec reward per step — Lagrangian violations and per-channel
        # diagnostics read from this buffer; the scalar `buf_rewards` is
        # only what the policy sees as the scalarised return.
        buf_reward_vec = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)

        state = self.env_states
        for t in range(T):
            key, sub = jrand.split(key)
            avail = self._vertex_avail(state)
            (
                actions, op_a, i_a, j_a, f_a,
                log_probs, values,
                partial, order, specs, step,
            ) = self._act_step(self.agent, state, avail, sub)

            # Convert to numpy for Ray fan-out.
            order_np = np.asarray(order)
            specs_np = np.asarray(specs)
            step_np = np.asarray(step)

            tokens_np, eqn_ids_np, reward_np = self._fan_out_tokenize(
                order_np, specs_np, step_np,
            )

            # Shard the Ray-returned numpy back to the env axis so the
            # assemble JIT runs SPMD (otherwise JAX would gather the
            # already-sharded `state` / `partial` onto one device).
            tokens_j = jax.device_put(
                jnp.asarray(tokens_np, dtype=jnp.int32), self.data_sharding,
            )
            eqn_ids_j = jax.device_put(
                jnp.asarray(eqn_ids_np, dtype=jnp.int32), self.data_sharding,
            )
            reward_j = jax.device_put(
                jnp.asarray(reward_np, dtype=jnp.float32), self.data_sharding,
            )
            state = self._assemble(state, partial, tokens_j, eqn_ids_j, reward_j)

            # Record. We store the *post-step* tokens so the next-step
            # policy gradient targets see the same obs the policy used.
            buf_tokens[t] = np.asarray(state.tokens)
            buf_actions[t] = np.asarray(actions)
            buf_op[t] = np.asarray(op_a)
            buf_i[t] = np.asarray(i_a)
            buf_j[t] = np.asarray(j_a)
            buf_f[t] = np.asarray(f_a)
            buf_vertex_for_loss[t] = np.asarray(actions) + 1  # 1-indexed for env
            buf_log_probs[t] = np.asarray(log_probs)
            buf_values[t] = np.asarray(values)
            buf_rewards[t] = np.asarray(
                self._scalar_reward(reward_j),
            )
            buf_reward_vec[t] = reward_np
            buf_dones[t] = np.asarray(state.terminated).astype(np.float32)

        # Bootstrap value at the final state (for the GAE next_value
        # term on the last timestep). `state` is sharded along the env
        # axis under data_sharding; shard the keys to match so the
        # vmap doesn't gather the state onto one device.
        key, boot_key = jrand.split(key)
        boot_keys = jax.device_put(
            jrand.split(boot_key, N), self.data_sharding,
        )
        bootstrap = np.asarray(
            jax.vmap(lambda s, k: self.agent.value(s.tokens, key=k))(state, boot_keys),
        )

        # GAE over the rollout. get_advantages is vmapped over the batch
        # dim and scans over time — we transpose (T, N) → (N, T) to match.
        rewards_b = jnp.asarray(buf_rewards.T)              # (N, T)
        dones_b = jnp.asarray(buf_dones.T)
        values_b = jnp.asarray(buf_values.T)
        # next_value is the value at the next timestep; for the final
        # step it's the bootstrap. Shift values by one and append.
        next_values_b = jnp.concatenate(
            [values_b[:, 1:], jnp.asarray(bootstrap)[:, None]], axis=1,
        )
        discounts_b = jnp.full_like(rewards_b, self.discount)
        # Symlog is applied inside `get_advantages` via the helper, so
        # we feed the raw scalar reward here.
        _episodic_return, returns_b, advantages_b = get_advantages(
            rewards_b, dones_b, values_b, next_values_b, discounts_b,
            self.gae_lambda,
        )

        # Stage F Lagrangian: penalise the advantage by the per-step
        # constraint violation, weighted by the current multipliers.
        # The violation is `max(0, sign * (threshold - reward))` —
        # zero when the constraint is satisfied. After applying the
        # penalty we update each multiplier by gradient ascent on the
        # mean violation (clamped >= 0). Empty-constraint case is a
        # no-op (the gather has zero columns).
        violation_stats: dict = {}
        if self.constraint_indices_np.shape[0] > 0:
            # Pull the constrained channels across the rollout buffer.
            # (T, N, C) where C = #constraints. Reshape to match the (N, T)
            # advantages layout used by GAE.
            picked = buf_reward_vec[:, :, self.constraint_indices_np]  # (T, N, C)
            picked = np.transpose(picked, (1, 0, 2))  # (N, T, C)
            # Symlog cost-family channels so the multipliers don't have to
            # bridge ~10⁹× channel-scale gaps; leave cosine_sim raw.
            no_symlog = self.constraint_no_symlog_np[None, None, :]  # (1,1,C)
            picked_sl = np.where(
                no_symlog, picked, np.sign(picked) * np.log1p(np.abs(picked)),
            )
            thresh_sl = np.where(
                self.constraint_no_symlog_np,
                self.constraint_thresholds_np,
                np.sign(self.constraint_thresholds_np)
                * np.log1p(np.abs(self.constraint_thresholds_np)),
            )  # (C,)
            signed = self.constraint_signs_np * (thresh_sl[None, None, :] - picked_sl)
            violations = np.maximum(0.0, signed)  # (N, T, C)
            penalty = np.sum(violations * self.multipliers_np[None, None, :], axis=-1)  # (N, T)
            advantages_b = advantages_b - jnp.asarray(penalty)
            mean_violations = violations.mean(axis=(0, 1))  # (C,)
            # Dual ascent on the multipliers — applied AFTER the penalty
            # has shaped this episode's gradient so the policy sees a
            # consistent multiplier within the rollout.
            self.multipliers_np = np.maximum(
                0.0,
                self.multipliers_np + self.lagrangian_lr * mean_violations,
            )
            for j, name in enumerate(self.constraint_names):
                violation_stats[f"lagrangian/{name}_lambda"] = float(self.multipliers_np[j])
                violation_stats[f"lagrangian/{name}_violation"] = float(mean_violations[j])

        # Normalise advantages per-rollout — mean 0 std 1, with the
        # `+1e-8` floor that's standard PPO hygiene. Applied AFTER the
        # Lagrangian penalty so the policy still sees zero-mean
        # advantages even when the multipliers shift the distribution.
        adv_flat = advantages_b.reshape(-1)
        adv_mean = jnp.mean(adv_flat)
        adv_std = jnp.std(adv_flat) + 1e-8
        advantages_b = (advantages_b - adv_mean) / adv_std

        # Stage for the update. Flatten (N, T) -> (N*T,) along the env
        # axis (each transition is independent for PPO).
        flat_tokens = jnp.asarray(buf_tokens.transpose(1, 0, 2).reshape(N * T, MAX_TOKENS))
        flat_actions = jnp.asarray(buf_actions.T.reshape(N * T))
        flat_op = jnp.asarray(buf_op.T.reshape(N * T))
        flat_i = jnp.asarray(buf_i.T.reshape(N * T))
        flat_j = jnp.asarray(buf_j.T.reshape(N * T))
        flat_f = jnp.asarray(buf_f.T.reshape(N * T))
        flat_vmask = jnp.asarray(buf_vertex_for_loss.T.reshape(N * T))
        flat_log_probs = jnp.asarray(buf_log_probs.T.reshape(N * T))
        flat_returns = returns_b.reshape(N * T)
        flat_advantages = advantages_b.reshape(N * T)

        # Single epoch × `minibatches` mini-batches. We rotate the batch
        # split by a deterministic permutation so each episode's first
        # minibatch isn't always envs 0..k-1.
        perm_key = jrand.fold_in(key, int(self._episode_counter))
        perm = jrand.permutation(perm_key, N * T)
        flat_tokens = flat_tokens[perm]
        flat_actions = flat_actions[perm]
        flat_op = flat_op[perm]
        flat_i = flat_i[perm]
        flat_j = flat_j[perm]
        flat_f = flat_f[perm]
        flat_vmask = flat_vmask[perm]
        flat_log_probs = flat_log_probs[perm]
        flat_returns = flat_returns[perm]
        flat_advantages = flat_advantages[perm]

        total = N * T
        mb_size = total // self.minibatches
        if mb_size == 0:
            mb_size = total
            mb_count = 1
        else:
            mb_count = self.minibatches
        # The data-parallel axis (mb_size) must be evenly splittable
        # across devices for the SPMD shard. If it isn't, fall back to
        # a single un-sharded minibatch — the user will see the warning
        # and can pick --minibatches accordingly.
        if mb_size % self.num_devices != 0:
            print(
                f"[ppo_ray] mb_size={mb_size} is not divisible by "
                f"num_devices={self.num_devices}; update_step will run "
                f"un-sharded (likely host-bound + may OOM). Pick a "
                f"--minibatches such that (num_envs*rollout_length)/"
                f"minibatches is a multiple of {self.num_devices}.",
            )
            mb_sharded = False
        else:
            mb_sharded = True

        # Reshape to ``(mb_count, mb_size, ...)`` so we can shard the
        # data axis (axis 1) across devices. Each minibatch index `i`
        # then yields a per-mb tensor that already lives in the
        # data-parallel layout — the loss vmap's leading axis is the
        # sharded one, and XLA distributes the work without a gather.
        def _reshape_mb(flat, mb_extra_shape=()):
            return flat.reshape((mb_count, mb_size, *mb_extra_shape))

        mb_tokens = _reshape_mb(flat_tokens, (MAX_TOKENS,))
        mb_actions = _reshape_mb(flat_actions)
        mb_op = _reshape_mb(flat_op)
        mb_i = _reshape_mb(flat_i)
        mb_j = _reshape_mb(flat_j)
        mb_f = _reshape_mb(flat_f)
        mb_vmask = _reshape_mb(flat_vmask)
        mb_log_probs = _reshape_mb(flat_log_probs)
        mb_returns = _reshape_mb(flat_returns)
        mb_advantages = _reshape_mb(flat_advantages)

        if mb_sharded:
            _shard = lambda x: jax.device_put(x, self.scan_data_sharding)
            mb_tokens = _shard(mb_tokens)
            mb_actions = _shard(mb_actions)
            mb_op = _shard(mb_op)
            mb_i = _shard(mb_i)
            mb_j = _shard(mb_j)
            mb_f = _shard(mb_f)
            mb_vmask = _shard(mb_vmask)
            mb_log_probs = _shard(mb_log_probs)
            mb_returns = _shard(mb_returns)
            mb_advantages = _shard(mb_advantages)

        agent = self.agent
        opt_state = self.opt_state
        last_aux = {}
        for i in range(mb_count):
            batch = (
                mb_tokens[i], mb_actions[i],
                mb_op[i], mb_i[i], mb_j[i], mb_f[i], mb_vmask[i],
                mb_log_probs[i], mb_returns[i], mb_advantages[i],
            )
            key, mb_key = jrand.split(key)
            agent, opt_state, aux = self._update_step(
                agent, opt_state, batch, mb_key,
            )
            last_aux = {k: float(v) for k, v in aux.items()}

        self.agent = agent
        self.opt_state = opt_state
        self._episode_counter += 1

        # Aggregate rewards across the batch for the wandb log dict.
        # Pull the raw 8-vec rewards back from buf_rewards (which is
        # already scalarised); for diagnostic per-channel reporting we'd
        # need to also buffer the raw vec — left out of the first cut
        # so the buffer stays simple.
        episode_return = float(jnp.sum(rewards_b, axis=1).mean())
        best_return = float(jnp.sum(rewards_b, axis=1).max())
        last_aux.update({
            "episode_return_mean": episode_return,
            "episode_return_max": best_return,
            "rollout_length": T,
            "num_envs": N,
        })
        # Per-channel raw-reward means so the driver can log them — same
        # spirit as `mu0_ray.run_rollout_and_train`'s `per_reward_means`.
        per_channel_means = buf_reward_vec.mean(axis=(0, 1))
        for j, name in enumerate(REWARD_NAMES):
            last_aux[f"reward_mean/{name}"] = float(per_channel_means[j])
        last_aux.update(violation_stats)
        return last_aux

    def _fan_out_tokenize(self, order_np, specs_np, step_np):
        """Ship the (order, specs, step) triple for each env to a CPU
        actor and collect (tokens, eqn_ids, reward) back. If no Ray
        workers are configured (e.g. local smoke test), fall back to
        an in-process `CpuApproximationServer`.
        """
        N = order_np.shape[0]
        if not self.cpu_workers:
            # In-process fallback — useful for local smoke tests where
            # we don't want to spin up a Ray cluster.
            if not hasattr(self, "_in_proc_server"):
                from alphagrad.approx.cpu_approx_worker import (
                    CpuApproximationServer,
                )
                self._in_proc_server = CpuApproximationServer.from_env(self.env)
            out = [
                self._in_proc_server.evaluate(
                    order_np[i], specs_np[i], int(step_np[i]),
                )
                for i in range(N)
            ]
        else:
            import ray
            W = len(self.cpu_workers)
            futures = [
                self.cpu_workers[i % W].evaluate.remote(
                    order_np[i], specs_np[i], int(step_np[i]),
                )
                for i in range(N)
            ]
            out = ray.get(futures)
        tokens = np.stack([r[0] for r in out])
        eqn_ids = np.stack([r[1] for r in out])
        rewards = np.stack([r[2] for r in out])
        return tokens, eqn_ids, rewards

    # ------------------------------------------------------------------
    # Lifecycle helpers used by the driver
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return True
