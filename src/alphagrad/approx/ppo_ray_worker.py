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
from alphagrad.approx.common.gae import get_advantages
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
from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT
from graphax.sparse.micro_actions import NUM_QUANT_DTYPES


# Stage F: which reward channels skip symlog when used in the
# Lagrangian comparison. Currently just cosine_sim — it's already
# bounded to [0, 1], so symlog'ing it would only complicate the
# threshold semantics. Mirrored from ppo._NO_SYMLOG_REWARD_INDICES.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Back-compat re-export of the canonical helper."""
    from alphagrad.approx.common.ray_runtime import _args_from_dict as _impl
    return _impl(args_dict)


def _setup_jax_compile_cache() -> None:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.cache.setup_jax_compile_cache`."""
    from alphagrad.approx.common.cache import setup_jax_compile_cache
    setup_jax_compile_cache()


def _build_reward_weights(args) -> np.ndarray:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.reward_scaling.build_reward_weights`."""
    from alphagrad.approx.common.reward_scaling import build_reward_weights
    return build_reward_weights(args)


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
    # axis/op/factor/quant component of a 1-substep micro-action per
    # vertex. ``None`` when the flag is off — the agent then emits empty
    # rule_specs (the legacy `ve_only` variant).
    op_type_head: Any        # MLP token-pooled -> 4  (DIAG, COMPRESS, QUANT, END)
    i_head: Any              # MLP token-pooled -> MAX_AXES_PER_VERTEX
    j_head: Any              # MLP token-pooled -> MAX_AXES_PER_VERTEX
    factor_head: Any         # MLP token-pooled -> num_factors
    quant_dtype_head: Any    # MLP token-pooled -> NUM_QUANT_DTYPES

    embd_dim: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)
    dynamic_substeps: bool = eqx.field(static=True)
    num_factors: int = eqx.field(static=True)
    num_quant_dtypes: int = eqx.field(static=True)

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

        keys = jrand.split(key, 9)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=keys[1])
        self.vertex_logits_head = MLP(
            embd_dim, num_vertices, policy_dims, key=keys[2],
        )
        # Per-channel value head — emits ``NUM_REWARDS`` scalars per
        # sample so the GDPO advantage path can do per-channel GAE +
        # per-channel z-scoring + per-channel value-loss. The legacy
        # `--advantage-norm scalar` path collapses these K=NUM_REWARDS
        # outputs to a single scalar via dot-product with
        # ``reward_weights`` at the bootstrap / loss boundary; the
        # agent architecture is identical regardless of mode.
        self.value_head = MLP(embd_dim, NUM_REWARDS, value_dims, key=keys[3])
        if dynamic_substeps:
            # 4-way op-type head: 0=DIAG, 1=COMPRESS, 2=QUANT, 3=END.
            # QUANT and COMPRESS each carry a sub-categorical head
            # (NUM_QUANT_DTYPES / not yet wired for COMPRESS kinds); the
            # gating in `_make_act_step_fn` reads the op-type sample and
            # routes through `micro_actions_to_rule_specs_jax` which knows
            # how to translate each.
            self.op_type_head = MLP(embd_dim, 4, policy_dims, key=keys[4])
            self.i_head = MLP(embd_dim, MAX_AXES_PER_VERTEX, policy_dims, key=keys[5])
            self.j_head = MLP(embd_dim, MAX_AXES_PER_VERTEX, policy_dims, key=keys[6])
            self.factor_head = MLP(embd_dim, num_factors, policy_dims, key=keys[7])
            self.quant_dtype_head = MLP(
                embd_dim, NUM_QUANT_DTYPES, policy_dims, key=keys[8],
            )
        else:
            self.op_type_head = None
            self.i_head = None
            self.j_head = None
            self.factor_head = None
            self.quant_dtype_head = None
        self.embd_dim = embd_dim
        self.num_vertices = num_vertices
        self.dynamic_substeps = dynamic_substeps
        self.num_factors = num_factors
        self.num_quant_dtypes = NUM_QUANT_DTYPES

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
        # Per-channel value: returns shape ``(NUM_REWARDS,)``. Legacy
        # callers expecting a scalar must collapse via dot-product with
        # reward weights — see ``_scalar_value`` on the worker.
        ctx = self.encode(tokens, key=key)
        return self.value_head(ctx)

    def policy_and_value(self, tokens, key):
        ctx = self.encode(tokens, key=key)
        logits = self.vertex_logits_head(ctx)
        value = self.value_head(ctx)  # (NUM_REWARDS,)
        return logits, value

    def micro_action_logits(self, tokens, key):
        """Return ``(op_logits, i_logits, j_logits, factor_logits, quant_logits)``.

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
            self.quant_dtype_head(ctx),
        )

    def all_logits(self, tokens, key):
        """Single-encode variant returning
        ``(vertex_logits, value, op, i, j, factor, quant)``.

        Combined entry point that runs the encoder once and dispatches
        to every head. Previously the rollout / loss paths called
        ``policy_and_value`` and ``micro_action_logits`` back-to-back,
        each of which re-encoded the tokens. The transformer dominates
        per-step compute, so collapsing the calls is roughly a ~2× win
        on the GPU when ``dynamic_substeps`` is on (no behavioural
        change otherwise — same softmax inputs).

        On a non-dynamic agent the micro heads are ``None``; the
        returned slots are placeholder zero arrays so the caller's
        downstream unpacking stays uniform.
        """
        ctx = self.encode(tokens, key=key)
        vertex_logits = self.vertex_logits_head(ctx)
        value = self.value_head(ctx)  # (NUM_REWARDS,)
        if self.dynamic_substeps:
            return (
                vertex_logits, value,
                self.op_type_head(ctx),
                self.i_head(ctx),
                self.j_head(ctx),
                self.factor_head(ctx),
                self.quant_dtype_head(ctx),
            )
        # Placeholders that match the dynamic shapes — the rollout /
        # loss paths gate on ``self.dynamic_substeps`` before reading
        # these, so the values are never observed. The op-type slot is
        # shape (4,) so the 4-way head matches.
        zero_op = jnp.zeros((4,), dtype=jnp.float32)  # DIAG/COMPRESS/QUANT/END
        zero_axes = jnp.zeros((MAX_AXES_PER_VERTEX,), dtype=jnp.float32)
        zero_factor = jnp.zeros((max(self.num_factors, 1),), dtype=jnp.float32)
        zero_quant = jnp.zeros((max(NUM_QUANT_DTYPES, 1),), dtype=jnp.float32)
        return (
            vertex_logits, value,
            zero_op, zero_axes, zero_axes, zero_factor, zero_quant,
        )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class PPORayWorker:
    """Owns env + agent + opt_state for one Ray trainer."""

    def __init__(self, args_dict: dict, seed: int = 0, cpu_workers: list | None = None):
        self.args = _args_from_dict(args_dict)
        self.seed = int(seed)
        self.cpu_workers = list(cpu_workers) if cpu_workers else []
        # Phase 3: optional CpuApproxPool for timeout-bounded tokenisation.
        # Created lazily by `init_server` (mirrors mu0_ray_actors.SPMDActor's
        # API). When None, `_fan_out_tokenize` falls back to the naive
        # per-actor `ray.get` path (no timeout — original behaviour).
        self._cpu_pool = None
        # Checkpoint path is captured here; loading happens after the
        # agent + opt_state templates are constructed (so eqx has
        # something to fill the leaves into).
        self._checkpoint_path = getattr(self.args, "checkpoint_path", "") or ""
        self._checkpoint_every = int(getattr(self.args, "checkpoint_every", 0))
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
        # Always pass target_fun so flops/bytes_accessed/latency_ns/peak_memory
        # populate every step (see cpu_approx_worker.py for the full rationale).
        env_target_fun = target_fn
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
            latency_samples=int(getattr(self.args, "latency_samples", 1)),
            num_data_points=int(getattr(self.args, "num_data_points", 5)),
            reps_per_point=int(getattr(self.args, "reps_per_point", 4)),
            percentile_keep=float(getattr(self.args, "percentile_keep", 0.60)),
            slow_exec_cutoff_seconds=float(
                getattr(self.args, "slow_exec_cutoff_seconds", 15.0)
            ),
            flop_gate_threshold=float(
                getattr(self.args, "flop_gate_threshold", 0.0)
            ),
            terminal_rewards_only=not bool(getattr(
                self.args, "intermediate_rewards", False,
            )),
        )
        # Per-rollout resampling stores the bank size here so we can
        # refresh the same way each episode (see run_rollout_and_train).
        self._eval_sample_count = int(getattr(
            self.args, "num_data_points",
            int(self.args.num_eval_samples),
        ))
        eval_samples = generate_eval_samples(
            env, eval_key, self._eval_sample_count,
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
        self.ppo_epochs = max(int(getattr(self.args, "ppo_epochs", 4)), 1)
        self.ppo_eps = float(getattr(self.args, "ppo_eps", 0.2))
        self.value_coef = float(getattr(self.args, "value_coef", 0.5))
        self.entropy_coef_init = float(getattr(self.args, "entropy_coef", 0.05))
        self.entropy_coef_final = float(getattr(self.args, "entropy_coef_final", 0.001))
        self.entropy_coef = self.entropy_coef_init
        # Reward scalarization is static: symlog per channel, then a fixed
        # priority-weighted sum (the per-channel "lambda" weights). No dynamic
        # per-channel σ normalization — so no reward-EMA state is kept.
        self.gae_lambda = float(getattr(self.args, "gae_lambda", 0.95))
        self.discount = float(getattr(self.args, "discount", 0.99))
        self.reward_weights_np = _build_reward_weights(self.args)
        self.reward_weights = jnp.asarray(
            self.reward_weights_np, dtype=jnp.float32,
        )

        # RQ8 / Pitch A — reward pipeline. When --reward-pipeline pca2,
        # the 6 cost channels (indices 0-5) get z-scored + PCA-2
        # compressed into 2 latents inline, replacing buf_reward_vec
        # contents at indices 0-1 with the latents and zeroing indices
        # 2-5. The lambdas for the 6 cost channels are re-derived so
        # only indices 0, 1 (the latents) carry non-zero weight; the
        # weight of the original primary-cost channel (whichever
        # --cmp-type pointed to) is summed onto index 0 and the
        # original primary-mem onto index 1, so the scalar reward
        # magnitude stays comparable to the legacy pipeline.
        self.reward_pipeline = str(
            getattr(self.args, "reward_pipeline", "legacy")
        )
        self._pca2_state = None
        if self.reward_pipeline == "pca2":
            from alphagrad.approx.common.reward_pca import PCA2State

            self._pca2_state = PCA2State.create(
                num_cost_channels=6,
                refit_every=int(getattr(self.args, "pca_refit_every", 100)),
                warmup_episodes=int(
                    getattr(self.args, "pca_warmup_episodes", 50)
                ),
            )
            # Reroute cost-channel weight onto the two PCA latents.
            # Sum the existing cost-channel weights so the latent-sum
            # scalar reward stays on the original cost magnitude
            # baseline; clear indices 2-5 to avoid double-counting.
            cost_w_sum = float(np.sum(self.reward_weights_np[:6]))
            if cost_w_sum == 0.0:
                cost_w_sum = 1.0  # fall back to unit weight on each latent
            self.reward_weights_np = self.reward_weights_np.copy()
            # Half the cost weight goes on each latent (sum stays
            # invariant; latents are unit-variance after whitening so
            # this means each contributes roughly 50/50 by magnitude).
            self.reward_weights_np[0] = 0.5 * cost_w_sum
            self.reward_weights_np[1] = 0.5 * cost_w_sum
            self.reward_weights_np[2:6] = 0.0
            self.reward_weights = jnp.asarray(
                self.reward_weights_np, dtype=jnp.float32,
            )

        # RQ8 / Pitch A bonus — telescoping max-over-vertices reduction
        # for channels like peak_memory / max_io_sum that are
        # conceptually "rollout-wide max" rather than additive cost
        # streams. ``--running-max-channels peak_memory,max_io_sum``
        # converts those channels' per-step values to telescoping
        # increments (cumsum == rollout-wide max) BEFORE GAE — keeps
        # GAE's additivity assumption intact while making the
        # per-channel sum equal the true rollout peak.
        self._running_max_indices: list[int] = []
        running_max_spec = str(
            getattr(self.args, "running_max_channels", "") or ""
        )
        if running_max_spec:
            from alphagrad.approx.env import REWARD_INDEX
            for name in running_max_spec.split(","):
                name = name.strip()
                if not name:
                    continue
                if name not in REWARD_INDEX:
                    print(
                        f"[ppo_ray] --running-max-channels: unknown "
                        f"channel {name!r}; ignoring"
                    )
                    continue
                self._running_max_indices.append(int(REWARD_INDEX[name]))

        # Advantage-normalisation strategy. ``gdpo`` activates the
        # per-channel z-score → priority-weighted sum → batch-norm
        # pipeline from arXiv:2601.05242. ``scalar`` preserves the
        # legacy scalarise-then-normalise behaviour for A/B comparison.
        self.advantage_norm = str(getattr(self.args, "advantage_norm", "gdpo"))
        if self.advantage_norm not in ("gdpo", "scalar"):
            raise ValueError(
                f"--advantage-norm must be 'gdpo' or 'scalar', got "
                f"{self.advantage_norm!r}",
            )
        self._use_symlog_in_gae = self.advantage_norm == "scalar"
        # Channel masks used by ``gdpo_normalise_advantages``. The
        # channel mask is 1 only for reward channels with non-zero
        # weight (user opted in via --rewards / --lambda-*). The sparse
        # mask flags channels (cosine_sim, frob_residual) whose value
        # is only meaningful at the terminal elimination step — z-score
        # uses rescale-without-recentre for them so the all-zero
        # intermediate steps don't artificially boost the std.
        from alphagrad.approx.common.reward_scaling import (
            SPARSE_TERMINAL_MASK_NP as _SPARSE_TERMINAL_MASK_NP,
        )
        self._channel_mask_j = jnp.asarray(
            (self.reward_weights_np != 0.0).astype(np.float32)
        )
        self._sparse_mask_j = jnp.asarray(
            _SPARSE_TERMINAL_MASK_NP.astype(np.float32)
        )
        # Cache the no-symlog GAE variant for the gdpo path. ``get_advantages``
        # (imported at module scope) is the legacy symlog'd variant used by
        # the scalar path. The two are jit'd independently — building both
        # up-front keeps the rollout loop free of import / construction
        # cost on every call.
        from alphagrad.approx.common.gae import (
            make_get_advantages as _make_get_advantages,
        )
        self._gae_no_symlog = _make_get_advantages(use_symlog=False)

        # Phase D — conditioned-reward gates. The user-supplied
        # ``--reward-condition`` specs are parsed once at init; the
        # rollout loop walks the list and zeroes the easier reward at
        # any transition where the gate fails. Sparse-terminal harder
        # channels (cosine_sim, frob_residual) only carry a signal at
        # the terminal step, so the gate is forced True on
        # intermediate steps to avoid zeroing the easier channel
        # everywhere.
        from alphagrad.approx.common.reward_scaling import (
            parse_reward_conditions as _parse_reward_conditions,
            SPARSE_TERMINAL_INDICES as _SPARSE_TI,
        )
        self._reward_conditions: list[tuple[int, int, str, float]] = (
            _parse_reward_conditions(
                getattr(self.args, "reward_condition", []) or []
            )
        )
        self._sparse_terminal_idx_set = set(_SPARSE_TI)

        # Anti-degeneracy corridor bounds for the corridor/* instrumentation
        # (build_unified_reward_log_dict reads self._corridor_low/_high). The
        # Lagrangian constraint machinery was removed — the reward is a static
        # priority-weighted scalar — so only the corridor bounds are derived
        # here, from --anti-degeneracy / --cosine-*-bound.
        from alphagrad.approx.common.anti_degeneracy import (
            desugar_anti_degeneracy,
        )
        _, self._corridor_low, self._corridor_high = desugar_anti_degeneracy(
            [],
            getattr(self.args, "anti_degeneracy", "none"),
            float(getattr(self.args, "anti_degeneracy_delta", 0.01)),
            float(getattr(self.args, "cosine_lower_bound", 0.8)),
            float(getattr(self.args, "cosine_upper_bound", 0.9)),
        )

        # Agent + optimizer.
        policy_dims = self._parse_int_list(self.args.policy_dims)
        value_dims = self._parse_int_list(self.args.value_dims)
        self.dynamic_substeps = bool(
            getattr(self.args, "dynamic_substeps", False)
        )
        # Variant / curriculum support. When ``--variant`` is anything
        # other than ``custom``, we build the agent with the UNION
        # factor table (so curriculum stage transitions can mask via
        # the policy logits rather than rebuilding the agent) and
        # restrict the action space per stage via mask flags below.
        # ``custom`` falls back to the legacy ``--factors`` value.
        from alphagrad.approx.variants import (
            compute_ppo_variant_masks,
            ppo_full_factor_table,
        )
        self.current_variant = str(getattr(self.args, "variant", "custom"))
        if self.current_variant == "custom":
            self.factor_table = self._parse_int_list(
                getattr(self.args, "factors", "-1,2,3,4")
            )
            if not self.factor_table:
                self.factor_table = (-1, 2, 3, 4)
        else:
            self.factor_table = ppo_full_factor_table()
        self.factor_table_j = jnp.asarray(self.factor_table, dtype=jnp.int32)
        # Initial action masks from the starting variant. For custom
        # there's no restriction (all-True). For curriculum-aware
        # variants the mask is the stage-specific allowed set.
        # ``compute_union_variant_masks`` accepts both single variants
        # and the compound "A+B" / "all_simple" strings the 7-stage
        # curriculum's rotation slots emit.
        from alphagrad.approx.variants import (
            compute_union_variant_masks as _compute_union_variant_masks,
        )
        from graphax.sparse.micro_actions import (
            NUM_QUANT_DTYPES as _NUM_QUANT_DTYPES_VAR,
        )
        self._num_quant_dtypes_for_mask = int(_NUM_QUANT_DTYPES_VAR)
        masks = _compute_union_variant_masks(
            self.current_variant,
            tuple(self.factor_table),
            self._num_quant_dtypes_for_mask,
        )
        self._current_op_mask_j = jnp.asarray(
            masks["op_type_mask"].astype(np.float32),
        )
        self._current_factor_mask_j = jnp.asarray(
            masks["factor_mask"].astype(np.float32),
        )
        self._current_quant_mask_j = jnp.asarray(
            masks["quant_dtype_mask"].astype(np.float32),
        )
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
            int(self.args.episodes) * self.minibatches * self.ppo_epochs,
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

        # 2-phase cost pipeline state. ``always_full`` is the default
        # (legacy behaviour, all channels every step). ``cheap_first``
        # starts with the cheap path (target_fun=None, no JIT-exec) and
        # swaps to the full path at the configured cutover. Wandb event
        # ``phase/cutover_ep`` records when the swap fires.
        self._cost_schedule = str(getattr(
            self.args, "cost_pipeline_schedule", "always_full",
        ))
        self._cost_phase_full = (self._cost_schedule == "always_full")
        self._cost_cutover_ep = int(getattr(
            self.args, "phase_cutover_ep", 0,
        ))

        # Trainer-side leak profile (mirror of the per-actor profiler in
        # cpu_approx_worker). Lets the user diff trainer-process RSS
        # growth against the worker-process growth in
        # ``slurm/logs/${JOB_ID}/leak-trainer-${PID}.log`` vs.
        # ``leak-actor-${PID}.log`` — confirms (or refutes) that the
        # JIT compile leak lives in the worker, not the trainer.
        from alphagrad.approx.common.leak_profile import maybe_install
        self._leak_profile = maybe_install("trainer")

        # Optional resume from a previously-written checkpoint. Must
        # happen AFTER the agent + opt_state are constructed (we need
        # template trees for eqx.tree_deserialise_leaves).
        if self._checkpoint_path:
            try:
                from alphagrad.approx.common.checkpoint import (
                    install_sigterm_handler, load_state,
                )
                restored = load_state(
                    self._checkpoint_path,
                    template_agent=self.agent,
                    template_opt_state=self.opt_state,
                )
                if restored is not None:
                    if restored["agent"] is not None:
                        self.agent = self._replicate(restored["agent"])
                    if restored["opt_state"] is not None:
                        self.opt_state = self._replicate(restored["opt_state"])
                    self._episode_counter = int(restored["episode_counter"])
                    if restored["reward_weights"] is not None:
                        self.reward_weights_np = restored["reward_weights"].astype(np.float32)
                        self.reward_weights = jnp.asarray(self.reward_weights_np, dtype=jnp.float32)
                    print(
                        f"[ppo_ray_worker] resumed from {self._checkpoint_path} "
                        f"at episode {self._episode_counter}"
                    )
                # Install the SIGTERM handler regardless of whether a
                # checkpoint already existed — we want the next SLURM
                # timeout to drop a checkpoint even if we started cold.
                install_sigterm_handler(self._save_checkpoint_safe)
            except Exception as exc:
                print(f"[ppo_ray_worker] checkpoint resume failed: {exc}")

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
        def act_step(agent, state_batch, vert_avail_batch,
                     op_mask, factor_mask, quant_mask, key):
            """``op_mask``, ``factor_mask``, ``quant_mask`` are per-stage
            curriculum masks (shapes ``(4,)``, ``(F,)``, ``(NUM_QUANT_DTYPES,)``,
            1.0 for allowed, 0.0 otherwise). Passed as traced inputs so
            variant transitions don't trigger a re-jit. When
            ``dynamic_substeps`` is off they're ignored — the
            static-action path doesn't sample op_type / factor /
            quant_dtype anyway."""
            keys = jrand.split(key, self.num_envs)

            def per_env(state_i, avail_i, k_i):
                # Up to 7 sub-keys: 1 encode + 6 action heads
                # (vertex / op / i / j / factor / quant). With
                # `dynamic_substeps` off only the first two are used;
                # the others are simply unused.
                k_enc, k_v, k_op, k_i, k_j, k_f, k_q = jrand.split(k_i, 7)
                # Single encoder pass shared across the vertex / value
                # heads AND the micro-action heads — see
                # `SimplePPOAgent.all_logits`. When dynamic_substeps
                # is off, the five micro slots are placeholders that
                # the `if dynamic:` branch below never reads.
                logits, value, op_l, i_l, j_l, f_l, q_l = agent.all_logits(
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
                    # Apply the curriculum masks to op_type, factor, AND
                    # quant_dtype logits. Disallowed actions get -1e9 → ~0
                    # probability in the softmax. The log_prob_* below use
                    # the SAME masked logits so the PPO ratio matches.
                    op_l_curr = jnp.where(op_mask > 0.5, op_l, -1e9)
                    f_l_curr = jnp.where(factor_mask > 0.5, f_l, -1e9)
                    q_l_curr = jnp.where(quant_mask > 0.5, q_l, -1e9)
                    op_sample = jrand.categorical(k_op, op_l_curr)
                    i_sample = jrand.categorical(k_i, i_l_masked)
                    j_sample = jrand.categorical(k_j, j_l_masked)
                    f_sample = jrand.categorical(k_f, f_l_curr)
                    q_sample = jrand.categorical(k_q, q_l_curr)
                    log_prob_op = jax.nn.log_softmax(op_l_curr)[op_sample]
                    log_prob_i = jax.nn.log_softmax(i_l_masked)[i_sample]
                    log_prob_j = jax.nn.log_softmax(j_l_masked)[j_sample]
                    log_prob_f = jax.nn.log_softmax(f_l_curr)[f_sample]
                    log_prob_q = jax.nn.log_softmax(q_l_curr)[q_sample]
                    # 4-way op-type: 0→DIAG, 1→COMPRESS, 2→QUANT, 3→END.
                    # The env's translator routes COMPRESS / QUANT only
                    # when the chosen vertex is the last in the partial
                    # elimination order (see env._callback's last-vertex
                    # guard); for any earlier vertex the slot is
                    # silently dropped. Policy gradient still flows
                    # through op_type — the env's behaviour just
                    # degrades gracefully to "no rule" on bad placements.
                    op_type = jnp.where(
                        op_sample == 0, OP_DIAG,
                        jnp.where(
                            op_sample == 1, OP_COMPRESS,
                            jnp.where(op_sample == 2, OP_QUANT, OP_END),
                        ),
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
                        quant_dtypes=jnp.array([q_sample], dtype=jnp.int32),
                    )
                else:
                    op_sample = jnp.int32(0)
                    i_sample = jnp.int32(0)
                    j_sample = jnp.int32(0)
                    f_sample = jnp.int32(0)
                    q_sample = jnp.int32(0)
                    log_prob_op = jnp.float32(0.0)
                    log_prob_i = jnp.float32(0.0)
                    log_prob_j = jnp.float32(0.0)
                    log_prob_f = jnp.float32(0.0)
                    log_prob_q = jnp.float32(0.0)
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
                # Joint log-prob over the six action components. PPO
                # clip operates on this sum; equivalently the ratio is
                # the product of per-head ratios. Heads that didn't
                # participate (everything except vertex when dynamic
                # is off) contributed log_prob=0 and ratio=1.
                log_prob_total = (
                    log_prob_v + log_prob_op + log_prob_i + log_prob_j
                    + log_prob_f + log_prob_q
                )
                return (
                    vertex_action,
                    op_sample, i_sample, j_sample, f_sample, q_sample,
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
        dynamic = self.dynamic_substeps
        # Mode-aware capture so the closure switches on the right path
        # without recompiling per call.
        is_gdpo = self.advantage_norm == "gdpo"
        channel_mask_j = self._channel_mask_j        # (NUM_REWARDS,)
        sparse_mask_j = self._sparse_mask_j          # (NUM_REWARDS,)
        priority_weights_j = self.reward_weights     # (NUM_REWARDS,)
        # Per-channel value-loss mask + active-channel count for the
        # gdpo path. Dead channels (weight 0) carry zero gradient.
        active_count = jnp.maximum(jnp.sum(channel_mask_j), jnp.float32(1.0))
        axis_valid_static_j = jnp.asarray(
            self.env.axis_valid_static, dtype=jnp.float32,
        )

        # Lazy import — at module scope we already imported the legacy
        # entry point; the new helper lives in the same module so this
        # is a cheap re-import.
        from alphagrad.approx.common.gae import gdpo_normalise_advantages

        def loss_fn(
            agent, batch, op_mask, factor_mask, quant_mask,
            key, entropy_coef,
        ):
            (
                tokens, actions, op_a, i_a, j_a, f_a, q_a,
                vertex_idx_for_mask,
                old_log_probs, returns, advantages,
            ) = batch
            # GDPO path: advantages enter as (B, K), normalise to scalar
            # (B,) via per-channel z-score → priority sum → batch-norm.
            # Scalar path: advantages enter as (B,) already z-scored
            # rollout-wide upstream.
            if is_gdpo:
                adv_scalar = gdpo_normalise_advantages(
                    advantages, channel_mask_j, sparse_mask_j,
                    priority_weights_j,
                )  # (B,)
            else:
                adv_scalar = advantages
            keys = jrand.split(key, tokens.shape[0])

            def per_sample(tok, v_act, op, i_s, j_s, f_s, q_s, v_for_mask,
                            olp, ret, adv, k):
                # Single encoder forward for both the vertex/value
                # heads and the micro-action heads; mirrors the
                # equivalent share in `act_step`. When dynamic is off
                # the (op_l, i_l, j_l, f_l, q_l) slots are placeholders
                # and the `if dynamic:` branch below skips them.
                logits, value, op_l, i_l, j_l, f_l, q_l = agent.all_logits(
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
                    # Apply the same curriculum masks the rollout used,
                    # so log_prob_op / log_prob_f / log_prob_q match the
                    # sampling distribution. Without this the PPO ratio
                    # sees a distribution different from the behaviour
                    # policy and the on-policy assumption breaks.
                    op_l_c = jnp.where(op_mask > 0.5, op_l, -1e9)
                    f_l_c = jnp.where(factor_mask > 0.5, f_l, -1e9)
                    q_l_c = jnp.where(quant_mask > 0.5, q_l, -1e9)
                    lp_op = jax.nn.log_softmax(op_l_c)[op]
                    lp_i = jax.nn.log_softmax(i_l_m)[i_s]
                    lp_j = jax.nn.log_softmax(j_l_m)[j_s]
                    lp_f = jax.nn.log_softmax(f_l_c)[f_s]
                    lp_q = jax.nn.log_softmax(q_l_c)[q_s]
                    # Entropy is computed over the LEGAL action set
                    # (-1e9 logits contribute ~0 probability and ~0 to
                    # entropy). Masked categorical heads naturally have
                    # smaller entropy ceilings as the action space
                    # shrinks (which is the point of the curriculum).
                    p_op = jax.nn.softmax(op_l_c)
                    ent_op = -jnp.sum(p_op * jax.nn.log_softmax(op_l_c))
                    # i / j entropy is over the valid-axes subset; mask
                    # the softmax denominator so the term doesn't see
                    # the -1e9 logits as low-but-nonzero probability.
                    p_i = jax.nn.softmax(i_l_m)
                    ent_i = -jnp.sum(p_i * jax.nn.log_softmax(i_l_m))
                    p_j = jax.nn.softmax(j_l_m)
                    ent_j = -jnp.sum(p_j * jax.nn.log_softmax(j_l_m))
                    p_f = jax.nn.softmax(f_l_c)
                    ent_f = -jnp.sum(p_f * jax.nn.log_softmax(f_l_c))
                    p_q = jax.nn.softmax(q_l_c)
                    ent_q = -jnp.sum(p_q * jax.nn.log_softmax(q_l_c))
                    new_log_prob = lp_v + lp_op + lp_i + lp_j + lp_f + lp_q
                    # Mean across the 6 heads so the entropy bonus
                    # stays comparable in scale to the single-head case.
                    per_head = jnp.stack(
                        [ent_v, ent_op, ent_i, ent_j, ent_f, ent_q],
                    )
                    entropy = jnp.mean(per_head)
                else:
                    new_log_prob = lp_v
                    entropy = ent_v
                    # Pad per-head with zeros for the static-shape
                    # contract so the vmap'd return type is stable.
                    z = jnp.float32(0.0)
                    per_head = jnp.stack([ent_v, z, z, z, z, z])

                ratio = jnp.exp(new_log_prob - olp)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -jnp.minimum(surr1, surr2)
                # Value-loss branches by mode:
                #   * gdpo:  per-channel MSE with channel_mask + std-rescale.
                #            ``value`` and ``ret`` are both (NUM_REWARDS,).
                #   * scalar: collapse the K-head value to scalar via the
                #            priority dot product, then legacy
                #            (scalar - symlog(scalar_ret))^2.
                if is_gdpo:
                    per_channel_se = (value - ret) ** 2           # (K,)
                    normaliser = jnp.maximum(jnp.abs(ret), 1.0)   # (K,)
                    value_loss = jnp.sum(
                        per_channel_se / (normaliser ** 2) * channel_mask_j
                    ) / active_count
                    pred_scalar = jnp.sum(value * priority_weights_j)
                    target_scalar = jnp.sum(ret * priority_weights_j)
                else:
                    # Scalar path: rewards are symlog'd and the per-channel
                    # balance is set by the static priority weights, so the
                    # priority-weighted scalar `ret` is the target directly.
                    pred_scalar = jnp.sum(value * priority_weights_j)
                    target_scalar = ret
                    value_loss = (pred_scalar - target_scalar) ** 2
                return policy_loss, value_loss, entropy, per_head, pred_scalar, target_scalar

            p_l, v_l, ent, per_head_ent, v_pred, v_target = jax.vmap(per_sample)(
                tokens, actions, op_a, i_a, j_a, f_a, q_a,
                vertex_idx_for_mask,
                old_log_probs, returns, adv_scalar, keys,
            )
            ppo_loss = jnp.mean(p_l)
            value_loss = jnp.mean(v_l)
            entropy_loss = -jnp.mean(ent)
            total = ppo_loss + value_coef * value_loss + entropy_coef * entropy_loss
            # explained_variance = 1 - Var(target - pred) / Var(target). Near 0
            # means the value head is just predicting the mean; near 1 means it
            # captures per-state structure. Falls back to 0 when targets are
            # constant (e.g. terminal-only rewards + tiny symlog'd spread).
            var_target = jnp.var(v_target)
            explained_var = jnp.where(
                var_target > 1e-12,
                1.0 - jnp.var(v_target - v_pred) / (var_target + 1e-12),
                0.0,
            )
            # Per-head entropy means: useful for diagnosing which
            # categorical head is collapsing (vertex / op_type / i /
            # j / factor / quant). Mean over the batch axis.
            head_means = jnp.mean(per_head_ent, axis=0)  # (6,)
            aux = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "explained_variance": explained_var,
                "entropy": jnp.mean(ent),
                "entropy/vertex": head_means[0],
                "entropy/op_type": head_means[1],
                "entropy/axis_i": head_means[2],
                "entropy/axis_j": head_means[3],
                "entropy/factor": head_means[4],
                "entropy/quant": head_means[5],
                "total_loss": total,
            }
            return total, aux

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        @eqx.filter_jit
        def update_step(
            agent, opt_state, batch,
            op_mask, factor_mask, quant_mask, key,
            entropy_coef,
        ):
            (loss, aux), grads = grad_fn(
                agent, batch, op_mask, factor_mask, quant_mask, key,
                entropy_coef,
            )
            # NaN-skip guard: if loss is non-finite (sentinel
            # poisoning, cold-cache compile-error fallback, etc.),
            # return the agent and opt state unchanged so a single bad
            # batch doesn't blow up the run. The `nan_skip_count`
            # counter is incremented in `aux` so the driver can log
            # it. See Phase 4e of the unification plan.
            loss_finite = jnp.isfinite(loss)
            zero_grads = jax.tree.map(jnp.zeros_like, grads)
            safe_grads = jax.tree.map(
                lambda g, z: jnp.where(loss_finite, g, z), grads, zero_grads,
            )
            updates, new_opt_state = self.optimizer.update(
                safe_grads, opt_state, agent,
            )
            new_agent = eqx.apply_updates(agent, updates)
            # When skipping, keep the opt_state from before the (no-op)
            # update — optax produces zero-magnitude updates from zero
            # grads but moment buffers still tick forward; freezing them
            # is the more conservative choice.
            new_opt_state = jax.tree.map(
                lambda new, old: jnp.where(loss_finite, new, old),
                new_opt_state, opt_state,
            )
            aux = dict(aux)
            aux["nan_skip"] = jnp.where(loss_finite, 0, 1).astype(jnp.int32)
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

        # Coarse per-phase profiling (gated on ALPHAGRAD_DBG_TIMING). Each
        # phase accumulator is printed at the end of the episode so we can
        # see where the per-ep wall-clock actually goes: rollout act_step
        # (GPU policy) vs fan_out_tokenize (Ray → CPU actors, incl. the
        # terminal measurement) vs GAE vs the PPO update.
        import time as _pt
        _prof = os.environ.get("ALPHAGRAD_DBG_TIMING", "0") == "1"
        _ph = {
            "reset": 0.0, "act_step": 0.0, "fan_out": 0.0,
            "fan_out_terminal": 0.0, "assemble": 0.0,
            "bootstrap_gae": 0.0, "update": 0.0, "stats": 0.0,
        }
        _ep_t0 = _pt.time()

        key = jrand.PRNGKey(int(rng_seed))
        key, reset_key, sample_key = jrand.split(key, 3)

        # Per-rollout eval-sample refresh — DISABLED to isolate hang in
        # first run_rollout_and_train call. The init-time eval_samples
        # bank is reused across all episodes (matches RQ1 behaviour).
        # TODO: re-enable once we understand the hang interaction with
        # data_gen + Ray actors + JAX vmap.
        _ = sample_key  # silence unused warning

        # Fresh on-policy episode: reset state for every env. Preserve
        # the sharded layout (data-parallel along the env axis).
        _t = _pt.time()
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )
        if _prof:
            jax.block_until_ready(self.env_states.tokens)
            _ph["reset"] += _pt.time() - _t

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
        buf_q = np.zeros((T, N), dtype=np.int32)
        # Vertex-id ↦ axis_state[v] gather slot. Stored so the loss path
        # can re-mask i / j logits identically to the rollout (the
        # axis_valid_mask is static per env across the episode, so we
        # only need to remember which vertex was acted on).
        buf_vertex_for_loss = np.zeros((T, N), dtype=np.int32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        # Per-channel value buffer (post-refactor). The value head is
        # always K=NUM_REWARDS wide; the scalar-mode path collapses
        # via dot product with ``reward_weights`` after rollout
        # collection, the gdpo-mode path keeps the K-vector through
        # GAE and the advantage stack.
        buf_values = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        # Raw 8-vec reward per step — Lagrangian violations and per-channel
        # diagnostics read from this buffer. The scalar reward (legacy
        # scalar-mode path) is derived from this buffer + ``reward_weights``
        # after the rollout loop terminates.
        buf_reward_vec = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)
        # Phase 3 (b): per-step sentinel mask. True when the CPU pool
        # timed out / errored on that env's tokenisation. Before GAE we
        # force `dones[t]=1` and `rewards[t]=0` for sentinel slots so
        # on-policy PPO doesn't bootstrap value over a -1e10 reward
        # (which would otherwise survive symlog'ing in GAE).
        buf_sentinel = np.zeros((T, N), dtype=bool)

        state = self.env_states
        for t in range(T):
            key, sub = jrand.split(key)
            _t = _pt.time()
            avail = self._vertex_avail(state)
            (
                actions, op_a, i_a, j_a, f_a, q_a,
                log_probs, values,
                partial, order, specs, step,
            ) = self._act_step(
                self.agent, state, avail,
                self._current_op_mask_j, self._current_factor_mask_j,
                self._current_quant_mask_j, sub,
            )

            # Convert to numpy for Ray fan-out.
            order_np = np.asarray(order)
            specs_np = np.asarray(specs)
            step_np = np.asarray(step)
            if _prof:
                _ph["act_step"] += _pt.time() - _t
                _t = _pt.time()

            # Terminal step + --measure-queue: use the global per-point
            # measurement queue (full core utilisation). Non-terminal
            # steps stay on the cheap tokenize-only batch path.
            if (
                t == T - 1
                and getattr(self.args, "measure_queue", False)
                and not bool(getattr(self.args, "intermediate_rewards", False))
            ):
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_terminal_queue(order_np, specs_np, step_np)
                )
            else:
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_tokenize(order_np, specs_np, step_np)
                )
            if _prof:
                _dt = _pt.time() - _t
                # Last rollout step = terminal (the measurement burst).
                if t == T - 1:
                    _ph["fan_out_terminal"] += _dt
                else:
                    _ph["fan_out"] += _dt
                _t = _pt.time()
            buf_sentinel[t] = sentinel_mask

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
            if _prof:
                jax.block_until_ready(state.tokens)
                _ph["assemble"] += _pt.time() - _t

            # Record. We store the *post-step* tokens so the next-step
            # policy gradient targets see the same obs the policy used.
            buf_tokens[t] = np.asarray(state.tokens)
            buf_actions[t] = np.asarray(actions)
            buf_op[t] = np.asarray(op_a)
            buf_i[t] = np.asarray(i_a)
            buf_j[t] = np.asarray(j_a)
            buf_f[t] = np.asarray(f_a)
            buf_q[t] = np.asarray(q_a)
            buf_vertex_for_loss[t] = np.asarray(actions) + 1  # 1-indexed for env
            buf_log_probs[t] = np.asarray(log_probs)
            buf_values[t] = np.asarray(values)  # (N, NUM_REWARDS)
            buf_reward_vec[t] = reward_np
            buf_dones[t] = np.asarray(state.terminated).astype(np.float32)

        _t = _pt.time()
        # Bootstrap value at the final state (for the GAE next_value
        # term on the last timestep). The value head emits K=NUM_REWARDS
        # heads; the bootstrap is shape ``(N, NUM_REWARDS)``.
        key, boot_key = jrand.split(key)
        boot_keys = jax.device_put(
            jrand.split(boot_key, N), self.data_sharding,
        )
        bootstrap = np.asarray(
            jax.vmap(lambda s, k: self.agent.value(s.tokens, key=k))(state, boot_keys),
        )  # (N, NUM_REWARDS)

        # Mask sentinel transitions (CPU pool timeouts).
        if buf_sentinel.any():
            buf_reward_vec = np.where(
                buf_sentinel[..., None], 0.0, buf_reward_vec,
            )
            buf_dones = np.where(buf_sentinel, 1.0, buf_dones)

        # ----- per-channel reward normalization (Option-A layer 1) -----
        # Two-stage Dreamer-V3 style: symlog crushes the per-channel
        # dynamic range (~13 decades between flops and frob), then EMA-
        # tracked σ in symlog space brings each channel to ~N(0, 1).
        # STATIC scalarization: symlog only, NO dynamic per-channel σ-divide.
        # The per-channel balancing is done by FIXED reward weights (static
        # lambdas chosen offline from the sampler's channel scales so each
        # rewarded channel contributes ~unit after symlog·λ) applied at the
        # weighted-sum below — not by an EMA of the running variance. This
        # replaces the old dynamic reweighting with a frozen, inspectable one.
        rewards_b_raw = jnp.asarray(
            np.transpose(buf_reward_vec, (1, 0, 2)),
        )  # (N, T, K)
        rewards_b_normalized = symlog(rewards_b_raw)
        dones_b = jnp.asarray(buf_dones.T)
        values_b = jnp.asarray(np.transpose(buf_values, (1, 0, 2)))
        next_values_b = jnp.concatenate(
            [values_b[:, 1:, :], jnp.asarray(bootstrap)[:, None, :]], axis=1,
        )
        discounts_b = jnp.full_like(dones_b, self.discount)

        # Per-channel GAE on the symlog'd rewards (identity inverse transform;
        # the static priority weights set the per-channel balance at the sum).
        from alphagrad.approx.common.gae import get_advantages_running_norm
        identity_mean = jnp.zeros(NUM_REWARDS, dtype=jnp.float32)
        identity_std = jnp.ones(NUM_REWARDS, dtype=jnp.float32)
        _episodic_return, returns_b, advantages_b = get_advantages_running_norm(
            rewards_b_normalized, dones_b, values_b, next_values_b, discounts_b,
            self.gae_lambda, identity_mean, identity_std,
        )
        # Priority-weighted sum across channels. After the symlog + per-
        # channel σ pre-norm, every channel is ~N(0, 1), so a plain
        # weighted sum mixes them fairly already.
        weights_j = self.reward_weights
        returns_b = jnp.sum(returns_b * weights_j, axis=-1)
        advantages_b = jnp.sum(advantages_b * weights_j, axis=-1)

        # (Lagrangian dual-ascent / cosine-constraint penalty removed: the
        # reward is now a purely static-weighted scalar — no constraint.)

        # Flatten (N, T) -> (N*T,) for minibatch slicing.
        flat_tokens = jnp.asarray(buf_tokens.transpose(1, 0, 2).reshape(N * T, MAX_TOKENS))
        flat_actions = jnp.asarray(buf_actions.T.reshape(N * T))
        flat_op = jnp.asarray(buf_op.T.reshape(N * T))
        flat_i = jnp.asarray(buf_i.T.reshape(N * T))
        flat_j = jnp.asarray(buf_j.T.reshape(N * T))
        flat_f = jnp.asarray(buf_f.T.reshape(N * T))
        flat_q = jnp.asarray(buf_q.T.reshape(N * T))
        flat_vmask = jnp.asarray(buf_vertex_for_loss.T.reshape(N * T))
        flat_log_probs = jnp.asarray(buf_log_probs.T.reshape(N * T))
        flat_returns = returns_b.reshape(N * T)
        flat_advantages = advantages_b.reshape(N * T)

        total = N * T
        mb_size = total // self.minibatches
        if mb_size == 0:
            mb_size = total
            mb_count = 1
        else:
            mb_count = self.minibatches
        mb_sharded = (mb_size % self.num_devices == 0)

        if _prof:
            jax.block_until_ready((flat_returns, flat_advantages))
            _ph["bootstrap_gae"] += _pt.time() - _t
            _t = _pt.time()

        agent = self.agent
        opt_state = self.opt_state
        aux_accum = {}
        nan_skip_total = 0
        n_updates = 0

        # Entropy coefficient annealing: linear decay from init to final.
        total_episodes = max(int(getattr(self.args, "episodes", 500)), 1)
        progress = min(self._episode_counter / total_episodes, 1.0)
        self.entropy_coef = (
            self.entropy_coef_init
            + (self.entropy_coef_final - self.entropy_coef_init) * progress
        )
        ent_coef_j = jnp.array(self.entropy_coef, dtype=jnp.float32)

        for _epoch in range(self.ppo_epochs):
            # Per-epoch reshuffle + z-score renormalization.
            key, perm_key = jrand.split(key)
            perm = jrand.permutation(perm_key, total)

            def _permute_and_mb(flat, extra=()):
                return flat[perm].reshape((mb_count, mb_size, *extra))

            mb_tokens = _permute_and_mb(flat_tokens, (MAX_TOKENS,))
            mb_actions = _permute_and_mb(flat_actions)
            mb_op = _permute_and_mb(flat_op)
            mb_i = _permute_and_mb(flat_i)
            mb_j = _permute_and_mb(flat_j)
            mb_f = _permute_and_mb(flat_f)
            mb_q = _permute_and_mb(flat_q)
            mb_vmask = _permute_and_mb(flat_vmask)
            mb_log_probs = _permute_and_mb(flat_log_probs)
            mb_returns = _permute_and_mb(flat_returns)
            # Per-epoch advantage z-score so normalization stays fresh.
            # Per-epoch advantage z-score (standard PPO scalar-advantage
            # normalization; this is NOT the per-channel reward reweighting).
            epoch_adv = flat_advantages[perm]
            adv_mean = jnp.mean(epoch_adv)
            adv_std = jnp.std(epoch_adv) + 1e-8
            epoch_adv = (epoch_adv - adv_mean) / adv_std
            mb_advantages = epoch_adv.reshape((mb_count, mb_size))

            if mb_sharded:
                _shard = lambda x: jax.device_put(x, self.scan_data_sharding)
                mb_tokens = _shard(mb_tokens)
                mb_actions = _shard(mb_actions)
                mb_op = _shard(mb_op)
                mb_i = _shard(mb_i)
                mb_j = _shard(mb_j)
                mb_f = _shard(mb_f)
                mb_q = _shard(mb_q)
                mb_vmask = _shard(mb_vmask)
                mb_log_probs = _shard(mb_log_probs)
                mb_returns = _shard(mb_returns)
                mb_advantages = _shard(mb_advantages)

            for i in range(mb_count):
                batch = (
                    mb_tokens[i], mb_actions[i],
                    mb_op[i], mb_i[i], mb_j[i], mb_f[i], mb_q[i],
                    mb_vmask[i],
                    mb_log_probs[i], mb_returns[i], mb_advantages[i],
                )
                key, mb_key = jrand.split(key)
                agent, opt_state, aux = self._update_step(
                    agent, opt_state, batch,
                    self._current_op_mask_j, self._current_factor_mask_j,
                    self._current_quant_mask_j, mb_key,
                    ent_coef_j,
                )
                nan_skip_total += int(aux.pop("nan_skip", 0))
                for k, v in aux.items():
                    aux_accum[k] = aux_accum.get(k, 0.0) + float(v)
                n_updates += 1

        if _prof:
            _ph["update"] += _pt.time() - _t
            _ep_total = _pt.time() - _ep_t0
            _acct = sum(_ph.values())
            print(
                "[DBG-prof] ep=%d total=%.1fs | reset=%.1f act_step=%.1f "
                "fan_out_rollout=%.1f fan_out_TERMINAL=%.1f assemble=%.1f "
                "boot_gae=%.1f update=%.1f | unaccounted=%.1f"
                % (
                    self._episode_counter, _ep_total, _ph["reset"],
                    _ph["act_step"], _ph["fan_out"], _ph["fan_out_terminal"],
                    _ph["assemble"], _ph["bootstrap_gae"], _ph["update"],
                    _ep_total - _acct,
                ),
                flush=True,
            )

        last_aux = {k: v / max(n_updates, 1) for k, v in aux_accum.items()}
        last_aux["entropy_coef"] = float(self.entropy_coef)
        # (reward_norm_symlog/* and lagrangian/* logging removed along with the
        # dynamic σ-reweighting and the Lagrangian constraint.)
        # Per-channel rollout means: raw, weighted (static λ), and terminal-only.
        _rv_nt = np.transpose(buf_reward_vec, (1, 0, 2))  # (N, T, K)
        _term_mask = buf_dones.T.astype(bool)             # (N, T)
        _wj = np.asarray(self.reward_weights)
        _has_term = bool(_term_mask.any())
        for _k, _name in enumerate(REWARD_NAMES):
            _raw = float(_rv_nt[..., _k].mean())
            last_aux[f"reward_mean/{_name}_raw"] = _raw
            last_aux[f"reward_mean/{_name}_weighted"] = _raw * float(_wj[_k])
            if _has_term:
                last_aux[f"reward_mean/{_name}_terminal"] = float(
                    _rv_nt[_term_mask, _k].mean(),
                )
        self.agent = agent
        self.opt_state = opt_state
        self._episode_counter += 1

        violation_stats: dict = {}
        reward_condition_stats: dict = {}

        if (
            not self._cost_phase_full
            and self._cost_cutover_ep > 0
            and self._episode_counter >= self._cost_cutover_ep
        ):
            import ray
            try:
                refs = [
                    a.set_cost_mode_full.remote()
                    for a in self.cpu_workers
                ]
                _ = ray.get(refs, timeout=60.0)
                self._cost_phase_full = True
                last_aux["phase/cutover_ep"] = int(self._episode_counter)
                last_aux["phase/cost_mode_full"] = 1
                print(
                    f"[ppo_ray] phase cutover: episode "
                    f"{self._episode_counter} — swapped {len(self.cpu_workers)} "
                    f"CPU actors from cheap to full cost mode",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[ppo_ray] phase cutover FAILED at ep "
                    f"{self._episode_counter}: {exc}; continuing in cheap mode",
                    flush=True,
                )

        # Best / mean return computed from the RAW per-channel buffer +
        # the (calibration-aware) reward weights — NOT from the
        # scalarised `rewards_b` which has been symlog'd inside GAE
        # (and which masks magnitude ordering for large rewards). The
        # pre-refactor calc at this site is why PPO's best stayed at
        # ep 0 in the wandb log run-vsh0bv2c — `jnp.sum(rewards_b,
        # axis=1).max()` was comparing symlog'd sums.
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            aggregate_per_channel_stats,
            REWARD_NAMES as _RS_REWARD_NAMES,
        )

        per_env_weighted_sum = (
            buf_reward_vec * self.reward_weights_np
        ).sum(axis=(0, 2))  # (N,) — raw weighted per-env return
        episode_return = float(per_env_weighted_sum.mean())
        best_return = float(per_env_weighted_sum.max())

        # Per-env action sequence — what landed in the env at each
        # step. For the simple policy each entry is just the vertex
        # index; for ``dynamic_substeps`` we record the full 6-tuple
        # ``(vertex, op, i, j, factor, quant)`` so the JSON dump can
        # reproduce the exact micro-action sequence that won each
        # channel. ``buf_*`` are shape ``(T, N)``; transpose so the
        # outer dimension is per-env (N) and the inner is per-step (T).
        per_env_actions: list[list]
        if self.dynamic_substeps:
            per_env_actions = [
                [
                    [
                        int(buf_actions[t, n]),
                        int(buf_op[t, n]),
                        int(buf_i[t, n]),
                        int(buf_j[t, n]),
                        int(buf_f[t, n]),
                        int(buf_q[t, n]),
                    ]
                    for t in range(T)
                ]
                for n in range(N)
            ]
        else:
            per_env_actions = [
                [int(buf_actions[t, n]) for t in range(T)]
                for n in range(N)
            ]

        # Per-channel best / mean / overall — same shape as MuZero's
        # `mu0_ray_worker` returns so the driver can use the shared
        # logging helpers. ``dones_mask`` (T, N) splits the per-step
        # mean into per_step / terminal so the sparse-terminal quality
        # channels (cossim / frob) report a non-diluted value.
        ch_stats = aggregate_per_channel_stats(
            buf_reward_vec.astype(np.float32),
            self.reward_weights_np.astype(np.float32),
            sentinel=SENTINEL_REWARD_VALUE,
            action_seq=per_env_actions,
            dones_mask=buf_dones.astype(bool),
        )

        last_aux.update({
            "episode_return_mean": episode_return,
            "episode_return_max": best_return,
            "mean_return": episode_return,
            "best_return": best_return,
            "rollout_length": T,
            "num_envs": N,
            "nan_skip_count": int(nan_skip_total),
            "train_step": int(self._episode_counter),
        })
        # Per-channel raw-reward means — same key namespace MuZero
        # uses. Already part of `per_reward_means` in `ch_stats` but
        # re-emit as flat `reward_mean/...` keys for the driver's
        # back-compat wandb log dict.
        for name, val in ch_stats["per_reward_means"].items():
            last_aux[f"reward_mean/{name}"] = float(val)
        last_aux["per_reward_means"] = ch_stats["per_reward_means"]
        last_aux["best_per_reward"] = ch_stats["best_per_reward"]
        last_aux["best_overall_rewards"] = ch_stats["best_overall_rewards"]
        last_aux["best_overall_weighted"] = ch_stats["best_overall_weighted"]
        # Per-channel 5-number summary over the rollout envs (distribution
        # tracker, per-episode). Numeric stats -> wandb via reward_dist/;
        # the quantile sequences ride along to best_sequences.json through
        # update_running_bests (dropped from wandb by the prefix whitelist).
        for _name, _q in ch_stats.get("reward_quantiles", {}).items():
            for _lab, _v in _q.items():
                last_aux[f"reward_dist/{_name}/{_lab}"] = float(_v)
        last_aux["quantile_sequences"] = ch_stats.get("quantile_sequences", {})
        # Unified `reward/{cost,quality}/{per_step,terminal,best_terminal}/<name>`
        # keys + corridor instrumentation. The legacy `reward_mean/*`
        # keys above remain for backward compat with existing dashboards.
        from alphagrad.approx.common.reward_scaling import (
            COSINE_SIM_IDX,
            build_unified_reward_log_dict,
        )
        terminal_mask_np = buf_dones.astype(bool)
        terminal_cs = (
            buf_reward_vec[terminal_mask_np, COSINE_SIM_IDX]
            if terminal_mask_np.any()
            else np.zeros((0,), dtype=np.float32)
        )
        last_aux.update(
            build_unified_reward_log_dict(
                ch_stats,
                corridor_low=getattr(self, "_corridor_low", None),
                corridor_high=getattr(self, "_corridor_high", None),
                terminal_cossims=terminal_cs,
            )
        )
        # ``best_seq`` is the canonical key the driver's
        # ``update_running_bests`` reads to populate
        # ``state["best_global_seq"]``. Pass through the overall-best
        # env's action sequence so the JSON dump can reconstruct
        # ``(a_i, b_i, c_i, r_i)`` AND the sequence that produced it.
        last_aux["best_seq"] = ch_stats.get("best_overall_seq", [])
        last_aux["best_overall_env"] = ch_stats.get("best_overall_env", -1)
        last_aux.update(violation_stats)
        last_aux.update(reward_condition_stats)

        # Phase 3 follow-up: periodic CPU-pool recycle (matches the
        # MuZero side at mu0_ray_worker.py:1041-1050). Without this,
        # each ``CpuApproximationActor`` accumulates the ~9 MB-per-call
        # ``cost_analysis()`` C++ residual; on the 188 GB pgi15 nodes
        # this OOM-killed the PPO run at episode ~605 with 16 workers
        # holding 10 GB each. The recycle is a kill + respawn, takes a
        # few seconds, and resets each actor's RSS back to baseline.
        if self._cpu_pool is not None:
            last_aux.update(
                {f"pool/{k}": v for k, v in self._cpu_pool.stats().items()}
            )
            # Per-episode timeout delta — the cumulative ``pool/timeouts``
            # is monotonic so the running total alone hides episode-level
            # spikes. We log the delta so a wandb time-series shows when
            # (if ever) sentinel-replacement fires. Mirrors the dead-code
            # decision check at end-of-run (see ``training_summary`` log).
            try:
                last_aux["pool/timeouts_this_episode"] = (
                    self._cpu_pool.fetch_timeout_delta()
                )
            except Exception:
                last_aux["pool/timeouts_this_episode"] = 0
            # Aggregate the per-actor tokenization-truncation telemetry.
            # All values are PER-EPISODE (the pool's
            # ``consume_*`` reset is invoked once per rollout). Mirrors
            # the ``nan_skip_count`` pattern. Best-effort — a Ray hiccup
            # here returns zeros rather than crashing the training step.
            #   * ``truncated_count``       — how often the clip bit this episode
            #   * ``overflow_sum_this_ep``  — total tokens of information thrown
            #                                 away this episode (sum of
            #                                 ``raw_len - MAX_TOKENS``)
            #   * ``mean_overflow_per_trunc`` — overflow_sum / count, or 0 if no
            #                                 truncations
            #   * ``max_observed_len``      — largest raw jaxpr length this episode;
            #                                 ``MAX_TOKENS`` is the live clip
            try:
                trunc = self._cpu_pool.fetch_tokenization_truncation_stats()
                count = int(trunc.get("count", 0))
                overflow_sum = int(trunc.get("overflow_sum", 0))
                max_len = int(trunc.get("max_observed_len", 0))
            except Exception:
                count = 0
                overflow_sum = 0
                max_len = 0
            last_aux["tokenization/truncated_count"] = count
            last_aux["tokenization/overflow_sum_this_ep"] = overflow_sum
            last_aux["tokenization/mean_overflow_per_trunc"] = (
                float(overflow_sum / count) if count > 0 else 0.0
            )
            last_aux["tokenization/max_observed_len"] = max_len
            recycle_every = getattr(self, "_cpu_pool_recycle_every", 0)
            if recycle_every > 0:
                # Cascading recycle: kill ONE actor every
                # ``recycle_every / N`` episodes so the full pool
                # rotates over ``recycle_every`` total episodes — same
                # effective per-actor lifetime as the old all-at-once
                # ``recycle()``, but the memory spike is smeared.
                n_pool = max(self._cpu_pool.size(), 1)
                interval = max(1, recycle_every // n_pool)
                if (
                    self._episode_counter > 0
                    and self._episode_counter % interval == 0
                ):
                    new_size = self._cpu_pool.recycle_one()
                    last_aux["pool/recycled_at_ep"] = self._episode_counter
                    last_aux["pool/size_after_recycle"] = new_size

        # Periodic checkpoint. The SIGTERM handler also fires a save
        # on slurm timeout, but periodic writes bound the worst-case
        # progress lost on a non-clean crash (OOM, segfault, network).
        if (
            self._checkpoint_path
            and self._checkpoint_every > 0
            and self._episode_counter > 0
            and self._episode_counter % self._checkpoint_every == 0
        ):
            self._save_checkpoint_safe()
            last_aux["checkpoint/saved_at_ep"] = self._episode_counter

        # Trainer-side leak sample. The actor-side profiler counts
        # `evaluate` calls; here we count episodes (each ep == one
        # full rollout + minibatch update). If RSS growth is local to
        # the trainer process, this log will show it; if the leak is
        # in the worker actors (where the per-call jit().compile()
        # lives), trainer RSS should stay roughly flat after JIT-warm.
        if self._leak_profile is not None:
            # Force-record on every episode (not gated by the every-N
            # counter) by sampling at a counter the profiler always
            # accepts — use the episode counter * `every` so the
            # ``n_call % every == 0`` gate always fires.
            self._leak_profile.record_call(
                self._episode_counter * self._leak_profile._every
            )
            last_aux["leak/trainer_rss_mb"] = float(
                self._leak_profile.rss_bytes() / 1024 / 1024
            )

        return last_aux

    # ------------------------------------------------------------------
    # Calibration support — methods called by
    # `alphagrad.approx.common.calibration.run_calibration`.
    # ------------------------------------------------------------------
    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Run `num_rollouts` zero-pref rollouts of the un-trained
        agent and return per-channel calibration statistics.

        Returns a dict (shape ``(NUM_REWARDS,)`` per entry):
          * ``mean``      — per-channel mean of valid transitions
          * ``median``    — per-channel median (raw)
          * ``q25``,``q75`` — quartiles (raw); ``iqr = q75 - q25``
          * ``median_symlog``, ``q25_symlog``, ``q75_symlog`` — same in
            symlog space (so IQR-on-symlog can be computed driver-side)
          * ``count``     — number of valid samples per channel
            (scalar — same across channels because the mask is row-wise)

        Sentinel transitions (any cost channel == SENTINEL_REWARD_VALUE)
        are filtered out — see Phase 4b note in the docstring of the
        MuZero sibling for why naive averaging breaks under sentinel
        contamination.

        Used by :func:`alphagrad.approx.common.calibration.run_calibration`,
        which picks the statistic (mean_abs / iqr / std) to derive the
        per-channel weight rescale from this dict.
        """
        from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            NUM_REWARDS as _NUM_REWARDS_RS,
            filter_sentinel_mask,
            symlog_np,
        )
        if not hasattr(self, "_act_step"):
            self._act_step = self._make_act_step_fn()
            self._assemble = self._make_assemble_fn()
            self._update_step = self._make_update_step()

        T = int(self.num_valid)
        N = int(self.num_envs)
        # Accumulate the FLAT valid samples per channel; small (≈ T·N
        # ·num_rollouts × K × 4 bytes ≈ 250 KB at defaults) so a single
        # contiguous buffer keeps the median/quartile computation simple.
        all_rows: list[np.ndarray] = []
        for r in range(int(num_rollouts)):
            key = jrand.PRNGKey(int(rng_seed) + r)
            key, reset_key = jrand.split(key)
            env_states = jax.vmap(lambda _: self.env.reset())(jnp.arange(N))
            state = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding)
                if eqx.is_array(x) else x,
                env_states,
            )
            buf_reward_vec = np.zeros((T, N, _NUM_REWARDS_RS), dtype=np.float32)
            for t in range(T):
                key, sub = jrand.split(key)
                avail = self._vertex_avail(state)
                act_out = self._act_step(
                    self.agent, state, avail,
                    self._current_op_mask_j, self._current_factor_mask_j,
                    self._current_quant_mask_j, sub,
                )
                partial = act_out[-4]
                order = act_out[-3]
                specs = act_out[-2]
                step = act_out[-1]
                tokens_np, eqn_ids_np, reward_np, _sentinel_mask = (
                    self._fan_out_tokenize(
                        np.asarray(order),
                        np.asarray(specs),
                        np.asarray(step),
                    )
                )
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
                buf_reward_vec[t] = reward_np
            valid = filter_sentinel_mask(buf_reward_vec, SENTINEL_REWARD_VALUE)
            if valid.any():
                all_rows.append(buf_reward_vec[valid])  # (n_valid, K)
        if not all_rows:
            zeros = np.zeros((_NUM_REWARDS_RS,), dtype=np.float32)
            return {
                "mean": zeros, "median": zeros, "q25": zeros, "q75": zeros,
                "median_symlog": zeros, "q25_symlog": zeros, "q75_symlog": zeros,
                "count": 0,
            }
        samples = np.concatenate(all_rows, axis=0).astype(np.float32)  # (M, K)
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

    def set_variant_masks(self, variant: str) -> dict:
        """Switch to a different curriculum stage by updating the
        per-stage op_type / factor / quant masks. Called by the driver
        between episodes when ``--curriculum`` is set. The agent itself
        is not rebuilt — only the masks change, so the existing policy
        weights carry over across stages.

        Accepts both single-variant strings (``"diag_gcd"``) and the
        compound strings the 7-stage round-robin curriculum emits
        (``"diag_gcd+compress_scalar"`` or ``"all_simple"``); union
        masks are computed via
        :func:`alphagrad.approx.variants.compute_union_variant_masks`.

        Returns the new mask shapes for confirmation logging.
        """
        from alphagrad.approx.variants import compute_union_variant_masks
        masks = compute_union_variant_masks(
            variant,
            tuple(self.factor_table),
            self._num_quant_dtypes_for_mask,
        )
        self.current_variant = variant
        self._current_op_mask_j = jnp.asarray(
            masks["op_type_mask"].astype(np.float32),
        )
        self._current_factor_mask_j = jnp.asarray(
            masks["factor_mask"].astype(np.float32),
        )
        self._current_quant_mask_j = jnp.asarray(
            masks["quant_dtype_mask"].astype(np.float32),
        )
        return {
            "variant": variant,
            "op_type_legal_count": int(masks["op_type_mask"].sum()),
            "factor_legal_count": int(masks["factor_mask"].sum()),
            "quant_dtype_legal_count": int(masks["quant_dtype_mask"].sum()),
        }

    def set_reward_weights(self, weights_np) -> None:
        """Replace the scalarising weight vector in-place. Called by
        `run_calibration` after it computes the per-channel scaling.
        """
        weights_np = np.asarray(weights_np, dtype=np.float32)
        assert weights_np.shape == self.reward_weights_np.shape, (
            f"weights shape mismatch: got {weights_np.shape}, "
            f"expected {self.reward_weights_np.shape}"
        )
        self.reward_weights_np = weights_np
        self.reward_weights = jnp.asarray(weights_np, dtype=jnp.float32)
        # Keep the GDPO channel mask in sync — calibration can change
        # which channels carry non-zero weight if a previously-disabled
        # channel gets a non-trivial scaling factor.
        self._channel_mask_j = jnp.asarray(
            (self.reward_weights_np != 0.0).astype(np.float32)
        )

    def _fan_out_tokenize(self, order_np, specs_np, step_np):
        """Ship the (order, specs, step) triple for each env to a CPU
        actor and collect (tokens, eqn_ids, reward) back.

        Returns ``(tokens, eqn_ids, rewards, sentinel_mask)`` where
        ``sentinel_mask`` is ``(N,) bool`` — True for env slots whose
        callback timed out / errored. The naive in-process and
        unpooled-ray paths never produce sentinels (they raise or
        block instead). Only the pooled path can mark a slot as
        sentinel.
        """
        N = order_np.shape[0]
        if self._cpu_pool is not None:
            # Phase 3 path: timeout-bounded fan-out via the shared
            # pool. Sentinels for timed-out slots flow back to the
            # rollout loop so it can mask the transition before GAE.
            tokens, eqn_ids, rewards, sentinel_mask = (
                self._cpu_pool.evaluate_batch(
                    order_np, specs_np, step_np, eval_samples=None, init=False,
                )
            )
            return tokens, eqn_ids, rewards, sentinel_mask

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
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens, eqn_ids, rewards, sentinel_mask

    def _fan_out_terminal_queue(self, order_np, specs_np, step_np):
        """Global measurement queue for the TERMINAL step.

        Instead of N env-tasks each measuring n_points data points
        SEQUENTIALLY on one actor's fixed core slice (which leaves
        gated/cheap envs' cores idle while a straggler grinds), decompose
        into ``N × n_points`` per-(env, data-point) tasks and dispatch
        them through ``ray.util.ActorPool``. Free actors pull the next
        task, so an env's points run in parallel and gated tasks
        immediately free workers for stragglers — full core utilisation.
        Each task measures ONE data point (R reps) on a fixed core slice,
        so the per-reading latency stays consistent. The driver then
        P60-aggregates each env's n_points reward vectors per channel.

        Returns the same ``(tokens, eqn_ids, rewards, sentinel_mask)``
        contract as ``_fan_out_tokenize``.
        """
        import ray
        from ray.util import ActorPool

        N = order_np.shape[0]
        n_points = max(int(getattr(self.args, "num_data_points", 5)), 1)
        pk = float(getattr(self.args, "percentile_keep", 0.60))

        actors = None
        if self._cpu_pool is not None:
            actors = self._cpu_pool.live_actors()
        elif self.cpu_workers:
            actors = list(self.cpu_workers)
        if not actors:
            # No Ray pool (in-proc tests) — fall back to the per-env path.
            return self._fan_out_tokenize(order_np, specs_np, step_np)

        # Build (env, point) task list, env-major so results[env*n_points+p].
        tasks = [
            (int(e), int(p), order_np[e], specs_np[e], int(step_np[e]))
            for e in range(N)
            for p in range(n_points)
        ]
        pool = ActorPool(actors)
        results = list(
            pool.map(
                lambda a, t: a.evaluate.remote(
                    t[2], t[3], t[4], point_idx=t[1],
                ),
                tasks,
            )
        )  # preserves task order

        tokens_out = np.zeros((N, MAX_TOKENS), dtype=np.int32)
        eqn_ids_out = np.zeros((N, MAX_TOKENS), dtype=np.int32)
        rewards_out = np.zeros((N, NUM_REWARDS), dtype=np.float32)
        for e in range(N):
            base = e * n_points
            # Per-point reward vectors for this env (each already P60'd
            # over its R reps inside _callback).
            pr = np.stack([results[base + p][2] for p in range(n_points)])  # (P, K)
            # Aggregate across points with the same "keep worst pk" rule
            # the single-call path uses. Cost channels are stored negated
            # (reward = -cost), so the worst-pk cost = -percentile(-reward).
            # cosine_sim (index COSINE) is "higher better" + weight 0 — use
            # the mean; everything else is a cost channel.
            agg = np.empty((NUM_REWARDS,), dtype=np.float32)
            _cos_idx = REWARD_INDEX["cosine_sim"]
            for k in range(NUM_REWARDS):
                if k == _cos_idx:
                    agg[k] = float(np.mean(pr[:, k]))
                else:
                    agg[k] = -float(np.percentile(-pr[:, k], pk * 100.0))
            rewards_out[e] = agg
            tokens_out[e] = results[base][0]
            eqn_ids_out[e] = results[base][1]
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens_out, eqn_ids_out, rewards_out, sentinel_mask

    # ------------------------------------------------------------------
    # Lifecycle helpers used by the driver
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return True

    def _save_checkpoint_safe(self) -> None:
        """Wrapped `save_state` that never propagates exceptions — used
        by both the periodic save in `run_rollout_and_train` and the
        SIGTERM handler. Logs to stdout instead of raising so a buggy
        serialisation doesn't take training down.
        """
        if not self._checkpoint_path:
            return
        try:
            from alphagrad.approx.common.checkpoint import save_state
            save_state(
                self._checkpoint_path,
                agent=self.agent,
                opt_state=self.opt_state,
                episode_counter=self._episode_counter,
                reward_weights=self.reward_weights_np,
            )
        except Exception as exc:
            print(f"[ppo_ray_worker] checkpoint save failed: {exc}")

    def init_server(
        self,
        cpu_workers: list,
        *,
        callback_timeout_s: float = 120.0,
        initial_timeout_s: float = 600.0,
        warm_after: int = 3,
        recycle_every: int = 50,
        cpu_actor_options: dict | None = None,
        starting_actor_id: int = 0,
    ) -> bool:
        """Construct the timeout-bounded `CpuApproxPool` from a list of
        Ray actor handles. Mirrors `mu0_ray_actors.SPMDActor.init_server`.

        After this call, `_fan_out_tokenize` dispatches through the
        pool — slow / hung actors are killed + respawned per the cold
        (`initial_timeout_s`) / warm (`callback_timeout_s`) budget,
        and timed-out env slots get sentinel rewards rather than
        blocking the rollout.
        """
        from alphagrad.approx.cpu_approx_pool import CpuApproxPool
        from alphagrad.approx.env import (
            MAX_TOKENS as _MAX_TOKENS,
            NUM_REWARDS as _NUM_REWARDS,
            REWARD_INDEX as _REWARD_INDEX,
        )

        self.cpu_workers = list(cpu_workers) if cpu_workers else []

        if cpu_actor_options is not None:
            from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
            args_dict = vars(self.args)
            next_id = [int(starting_actor_id)]

            def _factory():
                next_id[0] += 1
                return CpuApproximationActor.options(**cpu_actor_options).remote(
                    args_dict, variant=None, actor_id=next_id[0],
                )
        else:
            _factory = None  # pool runs without respawn

        self._cpu_pool = CpuApproxPool(
            self.cpu_workers,
            timeout_s=float(callback_timeout_s),
            initial_timeout_s=float(initial_timeout_s),
            warm_after=int(warm_after),
            respawn_factory=_factory,
            max_tokens=int(_MAX_TOKENS),
            num_rewards=int(_NUM_REWARDS),
            cosine_sim_idx=int(_REWARD_INDEX["cosine_sim"]),
            frob_residual_idx=int(_REWARD_INDEX["frob_residual"]),
        )
        self._cpu_pool_recycle_every = int(recycle_every)
        return True
