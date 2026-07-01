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
    grad_target_setup,
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
from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT
from graphax.sparse.micro_actions import NUM_QUANT_DTYPES


# Stage F: which reward channels skip symlog when used in the
# Lagrangian comparison. Currently just cosine_sim — it's already
# bounded to [0, 1], so symlog'ing it would only complicate the
# threshold semantics. Mirrored from ppo._NO_SYMLOG_REWARD_INDICES.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)

# Channel-index aliases used by the multiplicative cosine-gate reward.
COSINE_SIM_IDX: int = REWARD_INDEX["cosine_sim"]
FROB_RESIDUAL_IDX: int = REWARD_INDEX["frob_residual"]

# ADDITIVE_SYMLOG_COST: when ALPHAGRAD_ADDITIVE_SYMLOG_COST=1 (additive mode),
# symlog-compress the COST channels in the reward buffer before the scalar
# weighted sum + GAE. Without this the raw cost magnitudes (peak_memory ~1.7e6,
# latency ~2.4e4) dominate the scalar reward by 4-6 orders of magnitude over
# the [0,1] quality channels (cosine_sim / bkstep_acc), so bkstep(fracred) and
# the capped-cossim guide can never influence the advantage direction — the
# "flat-zero-basin" is really a scale-domination basin. Symlog brings
# latency->~10, peak_memory->~14 so lambda_cmp/lambda_mem (~0.005) put them on
# ~0.05-0.07 footing, commensurate with lambda_acc*bkstep (~0.5) and the guide.
# Mirrors ppo._symlog_rewards. cosine_sim + bkstep_acc are [0,1] quality
# channels and stay RAW (excluded from the mask).
_COST_SYMLOG_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_REWARDS)
    if REWARD_NAMES[i] not in ("cosine_sim", "bkstep_acc")
)
_COST_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
_COST_SYMLOG_MASK_NP[list(_COST_SYMLOG_INDICES)] = True


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
        # ALPHAGRAD_LOCAL_TOKENIZE=1: compute non-terminal state-tokens
        # in-process on the trainer (skip the per-step Ray round-trip).
        # Default OFF -> byte-identical to the all-Ray path. Only valid
        # under ``terminal_rewards_only`` (non-terminal rewards zeroed,
        # tokens a pure deterministic fn of (order, specs, step)).
        self._local_tokenize_enabled = (
            os.environ.get("ALPHAGRAD_LOCAL_TOKENIZE", "0") == "1"
        )
        # Process-local in-proc server for the local-tokenize path.
        # Built once in ``init_server`` (and lazily on first use as a
        # fallback). Reuses the same ``env._callback`` the CPU pool runs.
        self._in_proc_server = None
        # ``--terminal-rewards-only`` was renamed to
        # ``--intermediate-rewards`` (BooleanOptionalAction, default
        # False). Derive terminal_rewards_only the same way the CPU
        # actor does (cpu_approx_worker._build_env_from_args), so the
        # trainer's own env (and the in-proc server built from it) agree
        # with the pool on whether non-terminal steps skip the heavy
        # jacve compile/exec. Fall back to the legacy key.
        if hasattr(self.args, "intermediate_rewards"):
            self._terminal_rewards_only = not bool(
                self.args.intermediate_rewards
            )
        else:
            self._terminal_rewards_only = bool(
                getattr(self.args, "terminal_rewards_only", False)
            )
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
        # GRAD MODE (--measure-grad): wrap into the scalar-loss (+ optional
        # seed-vertex) graph and shift argnums, IDENTICALLY to the measure
        # actor (cpu_approx_worker._build_env_from_args). This makes the POLICY
        # env's jaxpr / valid_vertices / argnums match the graph the actor
        # actually differentiates, so the policy emits a COMPLETE elimination
        # order over the grad graph (incl. the output / loss-reduction
        # vertices). A Jacobian-graph order is incomplete for the grad graph
        # and graphax value_and_grad then returns a structurally-zero gradient.
        target_fn, xs, argnums = grad_target_setup(
            self.args, target_fn, xs, self.args.example
        )
        closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
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
            terminal_rewards_only=self._terminal_rewards_only,
            measure_grad=bool(getattr(self.args, "measure_grad", False)),
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

        # --------------------------------------------------------------
        # Multiplicative cosine-gate reward (ALPHAGRAD_REWARD_MODE=mult).
        # --------------------------------------------------------------
        # The default ("additive") path is byte-identical to the legacy
        # behaviour: the scalar reward is sum(reward_vec * weights), with
        # the cost channels stored negated (r = -cost, so cost->0 => r->0
        # = its max) and cosine_sim in [0,1] * lambda_acc. That additive
        # form is reward-hackable: a degenerate order can drive cost->0 and
        # cosine->0 and still beat a faithful order, because the cost term
        # alone reaches its maximum (0) while a faithful order pays cost.
        #
        # "mult" mode replaces the additive scalar with a FIDELITY-GATED
        # CHEAPNESS product computed in ``_apply_mult_reward_gate``:
        #     reward = g(cosine) * max(cheapness, 0)
        # where g(cosine) in [0,1] -> 0 as cosine -> 0, and ``cheapness``
        # is POSITIVE-oriented (larger = cheaper). cosine -> 0 => g -> 0 =>
        # reward -> ~0 regardless of how cheap the order is, so the
        # degenerate order can no longer be the reward-max.
        #
        # Implementation: the gated scalar is written into the cosine_sim
        # channel of the reward buffer and the scalarising weights are
        # collapsed to a one-hot on cosine_sim (weight 1.0). Every
        # downstream consumer that scalarises via ``reward_weights``
        # (per-channel GAE -> dot product, the scalar value-loss
        # ``priority_weights`` dot product, and the milestone best/mean
        # return) then picks out exactly the gated scalar. The RAW
        # per-channel buffer is preserved untouched for per-channel
        # logging (cosine_sim/cost means still report measured values).
        self.reward_mode = str(
            os.environ.get("ALPHAGRAD_REWARD_MODE", "additive")
        ).strip().lower()
        if self.reward_mode not in ("additive", "mult"):
            raise ValueError(
                f"ALPHAGRAD_REWARD_MODE must be 'additive' or 'mult', got "
                f"{self.reward_mode!r}",
            )
        # ADDITIVE COST-SYMLOG (opt-in, default OFF = plain weighted sum).
        # The additive scalar reward is ``sum(reward_vec * weights)`` over the
        # RAW per-channel buffer: cost channels are stored as ``-cost`` in their
        # native units (latency_ns ~1e6, peak_memory ~1e8), so a plain weighted
        # sum is dominated by the largest raw cost and is reward-hackable (a
        # degenerate cost->0/cossim->0 rule scores 0, beating a faithful rule's
        # huge negative cost). ``ALPHAGRAD_ADDITIVE_SYMLOG_COST=1`` symlog's the
        # COST channels (NOT cosine_sim / frob_residual) BEFORE the weighted sum
        # so the scalar becomes
        #     lam_cmp*(-symlog(latency)) + lam_mem*(-symlog(peak_mem))
        #         + lam_acc*cosine_sim
        # i.e. costs land in the ~14-21 symlog band, comparable to an O(10-30)
        # lam_acc*cosine term, so O(1)-O(30) lambdas balance the objective and a
        # large-enough lam_acc makes the cossim term strictly dominate the cost
        # gain (degenerate can never be the argmax). buf_reward_vec_raw stays RAW
        # for Lagrangian / per-channel telemetry. No effect in mult mode (the
        # gate already symlog's cost in ``cheapness``).
        self.additive_symlog_cost = (
            os.environ.get("ALPHAGRAD_ADDITIVE_SYMLOG_COST", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # Gate hyperparameters (only consulted in mult mode).
        #   tau: fidelity floor; cosine <= tau => g = 0.
        #   W:   cheapness offset; cheapness = W - sum_c |w_c|*symlog(cost_c).
        #        Pick W >= typical max weighted symlog(cost) so faithful
        #        orders keep cheapness > 0 (cheaper => larger). Clamped to
        #        >= 0 so an unusually expensive order floors at 0 rather
        #        than flipping sign (which would invert the incentive).
        self.reward_gate_tau = float(
            os.environ.get("ALPHAGRAD_REWARD_GATE_TAU", "0.5")
        )
        self.reward_gate_w = float(
            os.environ.get("ALPHAGRAD_REWARD_GATE_W", "5.0")
        )
        # ------------------------------------------------------------------
        # ANTI-DEGENERACY PENALTY (ALPHAGRAD_ANTI_DEGEN, default ON).
        # ------------------------------------------------------------------
        # The bare mult-gate makes the degenerate basin FLAT-ZERO: a
        # terminal rule with cosine_sim < tau drives g(cos)->0, so
        # gated = g*cheapness -> 0 REGARDLESS of how cheap it is. That
        # stops the policy from *being rewarded* for cossim->0 (good, no
        # hacking) but it does NOT *punish* it — the whole cossim<tau
        # region is a 0-gradient plateau the policy drifts into and can
        # never climb out of (every direction reads 0). Compounded: a
        # skipped (mem-gate) / sentineled (shape-storm) / exceptioned
        # measurement returns ZEROED cost channels which ``cheapness``
        # reads as "free perfect" (cheapness = W - symlog(0) = W), and a
        # cossim=0 zeroed row would also collapse to gated=0 — so the
        # degenerate and the failed-measure cases share the same flat-0
        # plateau and can corrupt best_overall as "free" rules.
        #
        # FIX: any TERMINAL transition that is degenerate (cosine_sim <
        # tau_deg, which also captures ||approx||~=0 since the env emits
        # cosine_sim=0 for a collapsed/failed Jacobian) OR a
        # failed/skipped measure gets a strictly NEGATIVE gated reward
        # -P, making the basin a penalised region with a gradient OUT
        # toward higher cosine. P is set strictly worse than the worst
        # VALID rule (cheapness floors at 0 -> worst valid gated == 0),
        # so -P < 0 <= any valid gated, i.e. degenerate/failed rules are
        # always the argmin and never the argmax / never best_overall.
        self.anti_degen = (
            os.environ.get("ALPHAGRAD_ANTI_DEGEN", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # Degeneracy fidelity floor: terminal cosine_sim below this is
        # treated as degenerate. Default 0.1 (well below the gate tau so
        # genuinely low-but-real fidelity rules still earn the small
        # positive g*cheapness rather than the penalty).
        self.anti_degen_tau = float(
            os.environ.get("ALPHAGRAD_ANTI_DEGEN_TAU", "0.1")
        )
        # Penalty magnitude. Default 2*W so a degenerate rule (gated=-P)
        # is strictly worse than the cheapest possible VALID rule
        # (gated -> g*W <= W). Tunable; clamped to be > 0.
        self.anti_degen_penalty = float(
            os.environ.get(
                "ALPHAGRAD_ANTI_DEGEN_PENALTY",
                str(2.0 * self.reward_gate_w),
            )
        )
        if self.anti_degen_penalty <= 0.0:
            self.anti_degen_penalty = 2.0 * self.reward_gate_w
        # In mult mode the gated scalar lives entirely on the cosine_sim
        # channel; collapse the scalarising weights to a one-hot so the
        # GAE / value / milestone dot products recover it exactly. The
        # gate already folds in the cost lambdas via ``cheapness``, so the
        # raw cost weights would double-count if left in the dot product.
        if self.reward_mode == "mult":
            # Preserve the user's full weight vector for per-channel
            # telemetry (the canonical weights collapse to one-hot below,
            # which would otherwise zero the cost columns in the
            # per-channel logging).
            self._logging_weights_np = self.reward_weights_np.astype(np.float32).copy()
            # Remember the user lambdas (|weight| per active cost channel)
            # BEFORE collapsing — the gate uses them to weight per-channel
            # symlog(cost) in ``cheapness``.
            self._mult_cost_weights_np = np.abs(
                self.reward_weights_np.astype(np.float32)
            )
            self._mult_cost_weights_np[COSINE_SIM_IDX] = 0.0
            self._mult_cost_weights_np[FROB_RESIDUAL_IDX] = 0.0
            collapsed = np.zeros_like(self.reward_weights_np)
            collapsed[COSINE_SIM_IDX] = 1.0
            self.reward_weights_np = collapsed.astype(np.float32)
            print(
                f"[ppo_ray] REWARD_MODE=mult: cosine-gate reward active "
                f"(tau={self.reward_gate_tau}, W={self.reward_gate_w}); "
                f"cost weights {self._mult_cost_weights_np.tolist()} folded "
                f"into cheapness; scalarising weight = one-hot[cosine_sim]."
            )
            if self.anti_degen:
                print(
                    f"[ppo_ray] ANTI_DEGEN active (SHAPED): terminal "
                    f"cosine_sim < {self.anti_degen_tau} -> gated reward ramps "
                    f"linearly from -{self.anti_degen_penalty} (cos=0) up to "
                    f"-{self.anti_degen_penalty * (1.0 - self.anti_degen_tau):.4g} "
                    f"(cos=tau); failed/sentinel measure -> "
                    f"-{self.anti_degen_penalty}. Positive d/dcos gives a "
                    f"gradient OUT of the cossim-0 basin; still strictly worse "
                    f"than the cheapest valid rule (excluded from best_overall)."
                )
        else:
            self._mult_cost_weights_np = np.zeros_like(self.reward_weights_np)
            self._logging_weights_np = self.reward_weights_np.astype(np.float32).copy()

        self.reward_weights = jnp.asarray(
            self.reward_weights_np, dtype=jnp.float32,
        )
        # ADDITIVE_SYMLOG_COST: symlog-compress cost channels in the reward
        # buffer before the scalar sum (see _COST_SYMLOG_MASK_NP). additive
        # mode only — mult mode does its own cost handling. Default off.
        self._additive_symlog_cost = (
            self.reward_mode == "additive"
            and os.environ.get("ALPHAGRAD_ADDITIVE_SYMLOG_COST", "0") == "1"
        )

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
            # Sparse-terminal mask: channels whose value is meaningful
            # only at the terminal elimination step. Used below to
            # zero out violations on non-terminal steps for those
            # channels (otherwise ``threshold - 0`` looks like a full
            # violation on every intermediate step and floods the
            # Lagrangian penalty term).
            from alphagrad.approx.common.reward_scaling import (
                SPARSE_TERMINAL_INDICES as _SPARSE_TERMINAL_INDICES,
            )
            self.constraint_is_sparse_np = np.array(
                [c[0] in _SPARSE_TERMINAL_INDICES for c in constraint_specs],
                dtype=np.bool_,
            )
        else:
            self.constraint_indices_np = np.zeros((0,), dtype=np.int32)
            self.constraint_thresholds_np = np.zeros((0,), dtype=np.float32)
            self.constraint_signs_np = np.zeros((0,), dtype=np.float32)
            self.constraint_names = []
            self.constraint_no_symlog_np = np.zeros((0,), dtype=np.bool_)
            self.constraint_is_sparse_np = np.zeros((0,), dtype=np.bool_)
        self.multipliers_np = np.zeros(
            (self.constraint_indices_np.shape[0],), dtype=np.float32,
        )
        self.lagrangian_lr = float(getattr(self.args, "lagrangian_lr", 1e-3))
        # Phase 5: bound multiplier growth and warm up before applying.
        # See the unification plan for why this is necessary — the prior
        # behaviour drove cosine_sim>=0.5 multipliers into the tens
        # within 100 episodes, killing exploration via overwhelming
        # penalty.
        self.lagrangian_multiplier_max = float(
            getattr(self.args, "lagrangian_multiplier_max", 1.0)
        )
        self.lagrangian_warmup_eps = int(
            getattr(self.args, "lagrangian_warmup_eps", 0)
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
                    if restored["multipliers"] is not None and self.constraint_indices_np.shape[0] > 0:
                        self.multipliers_np = restored["multipliers"].astype(np.float32)
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

    def _apply_mult_reward_gate(
        self,
        buf_reward_vec: np.ndarray,
        terminal_mask: np.ndarray | None = None,
        failed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform the raw per-channel reward buffer into the gated
        representation used by the multiplicative cosine-gate reward.

        ``buf_reward_vec`` is ``(T, N, NUM_REWARDS)`` of RAW per-channel
        rewards: cost channels stored NEGATED (r = -cost, more negative =
        worse), cosine_sim in [0,1]. Returns a NEW buffer (the caller
        keeps the original for per-channel logging).

        Gate::

            g(cos)    = clip((cos - tau) / (1 - tau), 0, 1)      # fidelity
            cheapness = max(0, W - sum_c |w_c| * symlog(cost_c))  # >0 = cheap
            reward    = g(cos) * cheapness

        where ``cost_c = -r_c`` (cost is the negated stored reward) and
        symlog(cost_c) = symlog(-r_c) grows with cost, so a CHEAP order
        (small cost) yields a LARGE positive cheapness. The gated scalar
        is written into the cosine_sim channel and every other channel is
        zeroed; the scalarising weights (one-hot[cosine_sim], set in
        ``__init__``) then recover it exactly downstream.

        SIGN: ``cheapness`` is built from the POSITIVE cost magnitude
        (``-r_c``), NOT the negative stored reward, so it is never
        multiplied by g while carrying a negative sign — a low-cosine
        order (g -> 0) yields reward -> 0, never a spuriously-large
        less-negative value. cosine -> 0 => reward -> ~0 regardless of
        cheapness, which kills the cost->0/cosine->0 hack.

        ANTI-DEGENERACY (``self.anti_degen``): the bare gate above makes
        the cossim<tau region a flat-0 plateau (no *punishment* for
        cossim->0, just no reward) — the policy can drift in and never
        climb out. When enabled, any TERMINAL transition that is
        degenerate (cosine_sim < ``anti_degen_tau``, capturing
        ||approx||~=0 since the env emits cosine_sim=0 for a
        collapsed/failed Jacobian) OR a failed/skipped/sentineled
        measurement (``failed_mask``) gets a strictly NEGATIVE gated
        reward ``-anti_degen_penalty``. With cheapness floored at 0 the
        worst VALID gated reward is 0, so the penalty makes degenerate /
        failed rules strictly the argmin (never best_overall) and turns
        the flat plateau into a penalised slope with a gradient toward
        higher cosine. ``terminal_mask`` / ``failed_mask`` are ``(T, N)``
        bool; when both are None the penalty is skipped (back-compat).
        """
        out = np.array(buf_reward_vec, dtype=np.float32, copy=True)
        cos = out[..., COSINE_SIM_IDX]                       # (T, N), in [0,1]
        tau = np.float32(self.reward_gate_tau)
        denom = np.maximum(np.float32(1.0) - tau, np.float32(1e-6))
        g = np.clip((cos - tau) / denom, 0.0, 1.0)           # (T, N)

        # Per-channel positive cost magnitude = -stored_reward; symlog and
        # weight by the user's |lambda| for each active cost channel. Only
        # cost channels carry nonzero weight here (cosine/frob zeroed in
        # __init__), so this never picks up the quality channels.
        cost_mag = -out                                      # (T, N, R)
        cost_sl = np.sign(cost_mag) * np.log1p(np.abs(cost_mag))
        w = self._mult_cost_weights_np.astype(np.float32)    # (R,)
        weighted_cost = (cost_sl * w[None, None, :]).sum(axis=-1)  # (T, N)
        cheapness = np.maximum(
            0.0, np.float32(self.reward_gate_w) - weighted_cost,
        )                                                    # (T, N)

        gated = (g * cheapness).astype(np.float32)           # (T, N)

        # Anti-degeneracy penalty: replace the flat-0 plateau with a
        # strictly-negative reward for degenerate / failed TERMINAL
        # transitions so the policy gets a gradient out of the basin.
        if self.anti_degen:
            P = np.float32(self.anti_degen_penalty)
            tau = np.float32(self.anti_degen_tau)
            degen = cos < tau                                  # (T, N)
            fail = None
            if failed_mask is not None:
                fail = np.asarray(failed_mask, dtype=bool)
                degen = degen | fail
            # Only penalise terminal transitions — intermediate steps
            # legitimately carry cosine_sim=0 (sparse-terminal channel)
            # and must stay at the gated value (0) so GAE isn't poisoned
            # with -P on every non-terminal step.
            if terminal_mask is not None:
                degen = degen & np.asarray(terminal_mask, dtype=bool)
            # SHAPED penalty: instead of a FLAT -P (zero advantage inside the
            # whole basin -> value collapse, the cossim-0 absorbing attractor),
            # give the penalty a positive slope in cos so the policy gets a
            # gradient pointing OUT of the basin. Linear ramp on cos in [0, tau]:
            #   cos=0   -> -P
            #   cos=tau -> -P*(1-tau)   (still strictly < 0 for tau in (0,1))
            # which stays <= 0 <= the worst VALID gated reward (cheapness>=0 =>
            # gated>=0 for a valid rule), so best_overall ordering and the
            # anti-hack guarantee are preserved. Failed/sentinel measures have
            # no meaningful cos -> pinned to the floor -P.
            cos_basin = np.clip(cos, 0.0, tau).astype(np.float32)
            shaped = (-(P - cos_basin * P)).astype(np.float32)   # (T, N)
            if fail is not None:
                shaped = np.where(fail, -P, shaped).astype(np.float32)
            gated = np.where(degen, shaped, gated).astype(np.float32)
            # Record for telemetry / caller (best_overall exclusion).
            self._last_degen_mask = degen
        else:
            self._last_degen_mask = np.zeros_like(gated, dtype=bool)

        out[...] = 0.0
        out[..., COSINE_SIM_IDX] = gated
        return out

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
        entropy_coef = self.entropy_coef
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

        def loss_fn(agent, batch, op_mask, factor_mask, quant_mask, key):
            """``op_mask`` / ``factor_mask`` / ``quant_mask`` are the
            curriculum masks for the CURRENT stage. They must match
            the masks used at rollout time (in ``act_step``) —
            otherwise the PPO log-prob ratio is computed against a
            different distribution from the one that produced the
            samples, breaking the on-policy assumption."""
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
                else:
                    value_scalar = jnp.sum(value * priority_weights_j)
                    value_loss = (value_scalar - symlog(ret)) ** 2
                return policy_loss, value_loss, entropy, per_head

            p_l, v_l, ent, per_head_ent = jax.vmap(per_sample)(
                tokens, actions, op_a, i_a, j_a, f_a, q_a,
                vertex_idx_for_mask,
                old_log_probs, returns, adv_scalar, keys,
            )
            ppo_loss = jnp.mean(p_l)
            value_loss = jnp.mean(v_l)
            entropy_loss = -jnp.mean(ent)
            total = ppo_loss + value_coef * value_loss + entropy_coef * entropy_loss
            # Per-head entropy means: useful for diagnosing which
            # categorical head is collapsing (vertex / op_type / i /
            # j / factor / quant). Mean over the batch axis.
            head_means = jnp.mean(per_head_ent, axis=0)  # (6,)
            aux = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
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
        def update_step(agent, opt_state, batch,
                        op_mask, factor_mask, quant_mask, key):
            (loss, aux), grads = grad_fn(
                agent, batch, op_mask, factor_mask, quant_mask, key,
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

            # Terminal step + --measure-queue: use the global per-point
            # measurement queue (full core utilisation). Non-terminal
            # steps stay on the cheap tokenize-only batch path. Grafted
            # from the approx branch — additive, gated entirely on the
            # ``--measure-queue`` flag (default off keeps theirs' path).
            _is_terminal_step = t == T - 1
            _local_tokenize = (
                not _is_terminal_step
                and self._local_tokenize_enabled
                and self._terminal_rewards_only
            )
            if (
                _is_terminal_step
                and getattr(self.args, "measure_queue", False)
                and self._terminal_rewards_only
            ):
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_terminal_queue(order_np, specs_np, step_np)
                )
            elif _local_tokenize:
                # Non-terminal step under ``terminal_rewards_only`` with
                # ALPHAGRAD_LOCAL_TOKENIZE=1: the reward is zeroed
                # (env.py early-return) and the tokens are a pure
                # deterministic fn of (order, specs, step), so compute
                # them in-process and skip the ~2 s/step Ray round-trip.
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._tokenize_local(order_np, specs_np, step_np)
                )
            else:
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_tokenize(order_np, specs_np, step_np)
                )
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

        # ANTI-DEGEN / failed-measure mask. A measurement is "failed"
        # when EITHER the CPU pool timed out / errored (``buf_sentinel``)
        # OR the env returned a sentinel reward vector (mem-gate skip /
        # shape-storm / exception path: cost channels == -1e10). Both
        # currently get their reward channels ZEROED below for GAE
        # stability — but a zeroed cost row reads as "free perfect"
        # (cheapness = W - symlog(0) = W) in the mult-gate and would
        # corrupt best_overall, while a zeroed cossim=0 row collapses to
        # gated=0 (the flat plateau). We capture the mask HERE, before
        # zeroing, so the mult-gate can stamp these with -P (anti-degen
        # penalty) and best_overall can exclude them — never zero.
        from alphagrad.approx.common.cache import (
            SENTINEL_REWARD_VALUE as _SENTINEL_RV,
        )
        # Env-side sentinel: any COST channel exactly == SENTINEL value.
        _cost_idx = [
            i for i in range(buf_reward_vec.shape[-1])
            if i not in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX)
        ]
        _env_sentinel = np.any(
            buf_reward_vec[..., _cost_idx] == np.float32(_SENTINEL_RV),
            axis=-1,
        )  # (T, N)
        buf_failed = buf_sentinel | _env_sentinel  # (T, N)

        # Phase 3 (b): mask sentinel transitions. Zero the per-channel
        # reward vector AND force dones=1 so GAE treats sentinel
        # timesteps as terminal and the value head bootstraps cleanly
        # at the next state. Per-channel zero (vs the legacy scalar
        # zero-out) means the gdpo path also sees zero reward on
        # those steps. NB: in mult+anti_degen mode the gate stamps the
        # cosine channel of these rows with -P AFTER this zeroing, so the
        # net effect is a strictly-negative gated reward (a gradient out),
        # NOT the misleading "free" zero — the zeroing here only clears the
        # raw -1e10 cost channels that would otherwise blow up symlog/GAE.
        if buf_failed.any():
            buf_reward_vec = np.where(
                buf_failed[..., None], 0.0, buf_reward_vec,
            )
            buf_dones = np.where(buf_failed, 1.0, buf_dones)
            n_sentinels = int(buf_failed.sum())
            print(
                f"[ppo_ray] {n_sentinels}/{T*N} failed/sentinel transitions "
                f"this episode (pool timeout / mem-gate / shape-storm); "
                f"zeroing raw rewards, forcing dones"
                + (
                    f", anti-degen penalty -{self.anti_degen_penalty} applied"
                    if (self.anti_degen and self.reward_mode == 'mult')
                    else ""
                )
                + "."
            )

        # ADDITIVE_SYMLOG_COST: symlog-compress the cost channels so the raw
        # ~1e6 cost magnitudes don't dwarf the [0,1] quality channels in the
        # scalar weighted sum (see _COST_SYMLOG_MASK_NP). Applied AFTER sentinel
        # zeroing (symlog(0)==0, so zeroed rows stay 0) and BEFORE reward
        # conditions / GAE. additive mode only.
        if self._additive_symlog_cost:
            _m = _COST_SYMLOG_MASK_NP[None, None, :]  # (1,1,NUM_REWARDS)
            buf_reward_vec = np.where(
                _m,
                np.sign(buf_reward_vec) * np.log1p(np.abs(buf_reward_vec)),
                buf_reward_vec,
            ).astype(np.float32)

        # Phase D: apply conditioned-reward gates. For each spec
        # ``(easier, harder, op, threshold)``, zero the easier channel
        # at any (t, n) where the harder channel doesn't satisfy
        # ``op(harder, threshold)``. Threshold comparison happens in
        # symlog space for cost channels (so the user can write
        # ``flops>=1e9`` without thinking about the squashing) and in
        # raw space for bounded channels (cosine_sim).
        reward_condition_stats: dict = {}
        if self._reward_conditions:
            for easier_idx, harder_idx, op, thresh in self._reward_conditions:
                harder_val = buf_reward_vec[..., harder_idx]  # (T, N)
                if harder_idx in _NO_SYMLOG_REWARD_INDICES:
                    h_compare = harder_val
                    t_compare = np.float32(thresh)
                else:
                    h_compare = np.sign(harder_val) * np.log1p(np.abs(harder_val))
                    t_compare = (
                        np.sign(np.float32(thresh))
                        * np.log1p(np.abs(np.float32(thresh)))
                    )
                if op == ">=":
                    gate = (h_compare >= t_compare).astype(np.float32)
                else:
                    gate = (h_compare <= t_compare).astype(np.float32)
                # Sparse-terminal harder channels carry signal only at
                # the terminal step. Force the gate True on
                # intermediate steps to avoid zeroing the easier
                # channel everywhere when the policy hasn't reached
                # the terminal step yet.
                if harder_idx in self._sparse_terminal_idx_set:
                    gate[:-1, :] = 1.0
                # ``buf_reward_vec`` is the canonical reward buffer; we
                # rebuild a writable copy because the sentinel path
                # above may have produced a view.
                buf_reward_vec = np.array(buf_reward_vec, copy=True)
                buf_reward_vec[..., easier_idx] = (
                    buf_reward_vec[..., easier_idx] * gate
                )
                gated_fraction = float(1.0 - gate.mean())
                easier_name = REWARD_NAMES[easier_idx]
                harder_name = REWARD_NAMES[harder_idx]
                reward_condition_stats[
                    f"reward_condition/{easier_name}_gated_fraction"
                ] = gated_fraction
                reward_condition_stats[
                    f"reward_condition/{harder_name}_threshold_{op}"
                ] = float(thresh)

        # Keep the RAW per-channel buffer for per-channel logging
        # (cosine_sim / cost means) — the mult-gate below overwrites
        # the canonical buffer with the gated scalar, which must not
        # corrupt the measured-channel telemetry.
        buf_reward_vec_raw = np.array(buf_reward_vec, dtype=np.float32, copy=True)

        # ADDITIVE COST-SYMLOG (ALPHAGRAD_ADDITIVE_SYMLOG_COST=1). Squash the
        # COST channels of the canonical buffer into symlog space so the plain
        # weighted sum below is magnitude-balanced (cost ~symlog 14-21 vs the
        # O(10-30) lam_acc*cosine term) instead of raw-cost dominated. Only the
        # cost channels are transformed; cosine_sim / frob_residual (the quality
        # signals, already O(1)) are left raw. ``buf_reward_vec_raw`` above keeps
        # the untransformed values for the Lagrangian / per-channel telemetry.
        # Sign-preserving: stored cost is negative (-cost), symlog keeps the sign
        # so a cheaper (smaller |cost|) order still scores a smaller-magnitude
        # negative term. No-op in mult mode (cheapness already symlog's cost).
        if self.reward_mode == "additive" and self.additive_symlog_cost:
            _cost_idx_sl = [
                i for i in range(buf_reward_vec.shape[-1])
                if i not in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX)
            ]
            buf_reward_vec = np.array(buf_reward_vec, copy=True)
            _c = buf_reward_vec[..., _cost_idx_sl]
            buf_reward_vec[..., _cost_idx_sl] = (
                np.sign(_c) * np.log1p(np.abs(_c))
            )

        # Multiplicative cosine-gate reward (ALPHAGRAD_REWARD_MODE=mult).
        # Collapse the per-channel buffer into the fidelity-gated cheapness
        # scalar on the cosine_sim channel BEFORE GAE/scalarisation. The
        # one-hot[cosine_sim] weights set in __init__ then make every
        # downstream dot product (GAE scalarise, scalar value loss, the
        # milestone best/mean return) pick out exactly the gated scalar.
        # ``additive`` mode leaves the buffer untouched (byte-identical to
        # the legacy path).
        if self.reward_mode == "mult":
            _term_mask = buf_dones.astype(bool)              # (T, N)
            buf_reward_vec = self._apply_mult_reward_gate(
                buf_reward_vec,
                terminal_mask=_term_mask,
                failed_mask=buf_failed,
            )

        # GAE over the rollout. Both modes use the same per-channel
        # tensor contract — the difference is whether symlog squashing
        # is applied inside GAE (scalar mode keeps it; gdpo mode drops
        # it because the per-mb z-score handles magnitude). The
        # factory caches the jit'd variant per mode.
        if self._use_symlog_in_gae:
            _gae = get_advantages  # legacy with symlog
        else:
            _gae = self._gae_no_symlog
        # rewards_b shape: (N, T, K). dones / discounts broadcast over K.
        rewards_b = jnp.asarray(np.transpose(buf_reward_vec, (1, 0, 2)))
        dones_b = jnp.asarray(buf_dones.T)             # (N, T)
        values_b = jnp.asarray(np.transpose(buf_values, (1, 0, 2)))  # (N, T, K)
        next_values_b = jnp.concatenate(
            [values_b[:, 1:, :], jnp.asarray(bootstrap)[:, None, :]], axis=1,
        )  # (N, T, K)
        discounts_b = jnp.full_like(dones_b, self.discount)  # (N, T)
        _episodic_return, returns_b, advantages_b = _gae(
            rewards_b, dones_b, values_b, next_values_b, discounts_b,
            self.gae_lambda,
        )
        # returns_b / advantages_b shape: (N, T, K) in gdpo mode.
        # In scalar mode we still ran per-channel GAE above; collapse
        # to scalar via the priority-weighted dot product so the
        # downstream legacy path (global z-score, scalar value loss)
        # receives (N, T) tensors.
        if self.advantage_norm == "scalar":
            weights_j = self.reward_weights
            returns_b = jnp.sum(returns_b * weights_j, axis=-1)      # (N, T)
            advantages_b = jnp.sum(advantages_b * weights_j, axis=-1)  # (N, T)

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
            # Use the RAW per-channel buffer: the Lagrangian thresholds
            # are expressed in measured-channel units (e.g. cosine_sim>=0.5,
            # peak_memory<=1e8), so they must see the raw channels, not the
            # mult-gate's collapsed cosine scalar. (`_raw` == canonical in
            # additive mode.)
            picked = buf_reward_vec_raw[:, :, self.constraint_indices_np]  # (T, N, C)
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
            # Per-constraint normalisation so the same `lagrangian_lr`
            # works for both `cosine_sim>=0.5` (small magnitudes) and
            # `peak_memory<=1e8` (symlog'd to ~18). Without this the
            # cosine_sim multiplier accumulated ~50× faster than other
            # channels and dominated the objective by ep ~100.
            thresh_scale = np.maximum(np.abs(thresh_sl), 1e-3)  # (C,)
            signed = (
                self.constraint_signs_np
                * (thresh_sl[None, None, :] - picked_sl)
                / thresh_scale[None, None, :]
            )
            violations = np.maximum(0.0, signed)  # (N, T, C)
            # Sparse-terminal mask: for constraints on sparse channels
            # (cosine_sim, frob_residual), only the terminal step
            # carries a real value — graphax's ``jacve`` returns a
            # zero-norm Jacobian for any partial elimination order, so
            # ``reward[cos_idx] = 0`` at intermediate steps. Without
            # this mask the Lagrangian sees a ``threshold - 0`` full
            # violation on every non-terminal step (T-1 spurious
            # violations per env per episode), accumulating spurious
            # penalty and floor-flooding the dual multipliers.
            if self.constraint_is_sparse_np.any():
                # Last step (t = T-1) is the terminal vertex elimination.
                # ``violations`` shape: (N, T, C).
                is_terminal_step = np.zeros((N, T), dtype=np.float32)
                is_terminal_step[:, -1] = 1.0
                sparse_step_mask = np.where(
                    self.constraint_is_sparse_np[None, None, :],  # (1,1,C)
                    is_terminal_step[:, :, None],                  # (N,T,1)
                    1.0,                                           # dense: keep
                )
                violations = violations * sparse_step_mask
            # Phase 5: gate penalty + multiplier update on warm-up.
            # During the first --lagrangian-warmup-eps episodes we still
            # compute violations (for logging) but don't shape the
            # advantage and don't ascend the multipliers — gives the
            # policy a chance to learn the scalar reward before the
            # constraint mechanic kicks in.
            in_warmup = self._episode_counter < self.lagrangian_warmup_eps
            if not in_warmup:
                penalty = np.sum(
                    violations * self.multipliers_np[None, None, :], axis=-1,
                )  # (N, T)
                # Broadcast the scalar penalty over the channel axis in
                # gdpo mode so the per-channel z-score sees the same
                # violation pressure on every active channel. In scalar
                # mode advantages_b is already (N, T) and the broadcast
                # is a no-op.
                penalty_j = jnp.asarray(penalty)
                if advantages_b.ndim == 3:
                    penalty_j = penalty_j[..., None]
                advantages_b = advantages_b - penalty_j
                mean_violations = violations.mean(axis=(0, 1))  # (C,)
                # Dual ascent with multiplier cap. Without the cap a
                # structurally hard constraint sees the multiplier grow
                # unbounded and overwhelm the rest of the objective.
                self.multipliers_np = np.clip(
                    self.multipliers_np
                    + self.lagrangian_lr * mean_violations,
                    0.0,
                    self.lagrangian_multiplier_max,
                )
            else:
                mean_violations = violations.mean(axis=(0, 1))
            for j, name in enumerate(self.constraint_names):
                violation_stats[f"lagrangian/{name}_lambda"] = float(
                    self.multipliers_np[j]
                )
                violation_stats[f"lagrangian/{name}_violation"] = float(
                    mean_violations[j]
                )
            violation_stats["lagrangian/warmup_active"] = int(in_warmup)

        # Normalise advantages — scalar mode keeps the legacy
        # rollout-wide z-score (mean 0, std 1, +1e-8 floor) applied AFTER
        # the Lagrangian penalty. gdpo mode skips the global z-score
        # entirely — per-channel z-scoring runs per-minibatch in the
        # update step (see gdpo_normalise_advantages) so the global pass
        # would double-normalise and wash out cross-minibatch signal.
        if self.advantage_norm == "scalar":
            adv_flat = advantages_b.reshape(-1)
            adv_mean = jnp.mean(adv_flat)
            adv_std = jnp.std(adv_flat) + 1e-8
            advantages_b = (advantages_b - adv_mean) / adv_std

        # Stage for the update. Flatten (N, T) -> (N*T,) along the env
        # axis (each transition is independent for PPO). Returns and
        # advantages carry a trailing K axis in gdpo mode (per-channel
        # values flow into the per-mb gdpo normalisation inside the
        # loss); scalar mode collapsed them above and they're already
        # 1D here.
        flat_tokens = jnp.asarray(buf_tokens.transpose(1, 0, 2).reshape(N * T, MAX_TOKENS))
        flat_actions = jnp.asarray(buf_actions.T.reshape(N * T))
        flat_op = jnp.asarray(buf_op.T.reshape(N * T))
        flat_i = jnp.asarray(buf_i.T.reshape(N * T))
        flat_j = jnp.asarray(buf_j.T.reshape(N * T))
        flat_f = jnp.asarray(buf_f.T.reshape(N * T))
        flat_q = jnp.asarray(buf_q.T.reshape(N * T))
        flat_vmask = jnp.asarray(buf_vertex_for_loss.T.reshape(N * T))
        flat_log_probs = jnp.asarray(buf_log_probs.T.reshape(N * T))
        if self.advantage_norm == "gdpo":
            flat_returns = returns_b.reshape(N * T, NUM_REWARDS)
            flat_advantages = advantages_b.reshape(N * T, NUM_REWARDS)
        else:
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
        flat_q = flat_q[perm]
        flat_vmask = flat_vmask[perm]
        flat_log_probs = flat_log_probs[perm]
        flat_returns = flat_returns[perm]
        flat_advantages = flat_advantages[perm]

        total = N * T
        mb_size = total // self.minibatches
        if mb_size == 0:
            mb_size = total
            
        # Guarantee that mb_size is a multiple of num_devices so we can always shard
        if mb_size % self.num_devices != 0:
            mb_size = max(self.num_devices, (mb_size // self.num_devices) * self.num_devices)
            
        mb_count = total // mb_size
        valid_total = mb_count * mb_size
        
        if valid_total < total:
            print(f"[ppo_ray] truncating batch from {total} to {valid_total} to ensure mb_size={mb_size} is a multiple of {self.num_devices}")
            flat_tokens = flat_tokens[:valid_total]
            flat_actions = flat_actions[:valid_total]
            flat_op = flat_op[:valid_total]
            flat_i = flat_i[:valid_total]
            flat_j = flat_j[:valid_total]
            flat_f = flat_f[:valid_total]
            flat_q = flat_q[:valid_total]
            flat_vmask = flat_vmask[:valid_total]
            flat_log_probs = flat_log_probs[:valid_total]
            flat_returns = flat_returns[:valid_total]
            flat_advantages = flat_advantages[:valid_total]
            
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
        mb_q = _reshape_mb(flat_q)
        mb_vmask = _reshape_mb(flat_vmask)
        mb_log_probs = _reshape_mb(flat_log_probs)
        # Returns + advantages carry a trailing K=NUM_REWARDS axis in
        # gdpo mode and are scalar (no trailing axis) in scalar mode.
        if self.advantage_norm == "gdpo":
            mb_returns = _reshape_mb(flat_returns, (NUM_REWARDS,))
            mb_advantages = _reshape_mb(flat_advantages, (NUM_REWARDS,))
        else:
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
            mb_q = _shard(mb_q)
            mb_vmask = _shard(mb_vmask)
            mb_log_probs = _shard(mb_log_probs)
            mb_returns = _shard(mb_returns)
            mb_advantages = _shard(mb_advantages)

        agent = self.agent
        opt_state = self.opt_state
        last_aux = {}
        nan_skip_total = 0
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
            )
            nan_skip_total += int(aux.pop("nan_skip", 0))
            last_aux = {k: float(v) for k, v in aux.items()}

        self.agent = agent
        self.opt_state = opt_state
        self._episode_counter += 1

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
        # logging helpers.
        # Per-channel telemetry uses the RAW measured buffer + the user's
        # full weight vector. In mult mode the canonical ``buf_reward_vec``
        # has been collapsed to the gated scalar on the cosine channel and
        # ``reward_weights_np`` is one-hot, so feeding those here would
        # zero out cost-channel reporting and overstate cosine. ``_raw``
        # and ``_logging_weights_np`` are byte-identical to the canonical
        # ones in additive mode.
        # ANTI-DEGEN best_overall exclusion. The failed-measure zeroing
        # above turns a sentinel/mem-gate/shape-storm terminal into an
        # ALL-ZERO raw row, which ``aggregate_per_channel_stats`` would
        # read as "free perfect" (cost=0) and could crown as best_overall
        # — corrupting the milestone best. Likewise a genuinely degenerate
        # terminal (cosine_sim < anti_degen_tau) must never be best. Stamp
        # the cost channels of these envs' TERMINAL rows in the
        # aggregation copy with the SENTINEL value so the built-in
        # ``filter_sentinel_mask`` drops them from every per-channel mean,
        # the per-env sum, and the best_overall argmax. (The canonical
        # ``buf_reward_vec`` already carries -P on these via the gate, so
        # ``best_return`` from the gated buffer is correct independently.)
        buf_raw_for_agg = buf_reward_vec_raw.astype(np.float32)
        if self.anti_degen and self.reward_mode == "mult":
            _term = buf_dones.astype(bool)                    # (T, N)
            _degen_terminal = (
                (buf_reward_vec_raw[..., COSINE_SIM_IDX]
                 < np.float32(self.anti_degen_tau))
                | buf_failed
            ) & _term                                        # (T, N)
            if _degen_terminal.any():
                buf_raw_for_agg = buf_raw_for_agg.copy()
                _cost_idx_arr = np.array(_cost_idx, dtype=np.int64)
                # Mark every cost channel of the degenerate terminal rows
                # as SENTINEL so filter_sentinel_mask excludes them.
                for _ci in _cost_idx_arr:
                    buf_raw_for_agg[..., int(_ci)] = np.where(
                        _degen_terminal,
                        np.float32(SENTINEL_REWARD_VALUE),
                        buf_raw_for_agg[..., int(_ci)],
                    )
                print(
                    f"[ppo_ray] anti-degen: excluded "
                    f"{int(_degen_terminal.sum())} degenerate/failed terminal "
                    f"env(s) from best_overall."
                )

        ch_stats = aggregate_per_channel_stats(
            buf_raw_for_agg,
            self._logging_weights_np.astype(np.float32),
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
        # Per-channel terminal-step means — the honest per-episode signal
        # for the sparse-terminal quality channels (cossim / frob), which
        # the per-step `reward_mean/<name>` dilutes with intermediate
        # zero-reward steps. Restored to match the pre-merge PPO behaviour
        # (and the C-MORL `reward_mean/<name>_terminal` definition):
        # mean over the dones-masked terminal transitions.
        for name, val in ch_stats.get("terminal_means", {}).items():
            last_aux[f"reward_mean/{name}_terminal"] = float(val)
        last_aux["per_reward_means"] = ch_stats["per_reward_means"]
        last_aux["per_reward_means_terminal"] = ch_stats.get("terminal_means", {})
        last_aux["best_per_reward"] = ch_stats["best_per_reward"]
        last_aux["best_overall_rewards"] = ch_stats["best_overall_rewards"]
        last_aux["best_overall_weighted"] = ch_stats["best_overall_weighted"]
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

    def _ensure_in_proc_server(self):
        """Lazily build the process-local ``CpuApproximationServer``.

        Reuses the already-constructed env so the in-proc path runs the
        exact same ``env._callback`` the CPU pool's actors run — the
        tokenizer output is therefore byte-identical to the pool path
        for the (deterministic) non-terminal steps.
        """
        if self._in_proc_server is None:
            from alphagrad.approx.cpu_approx_worker import (
                CpuApproximationServer,
            )
            self._in_proc_server = CpuApproximationServer.from_env(self.env)
            # Pin the non-terminal tokenize to a CPU device. The
            # tokenizer is pure symbolic/graph work (graphax
            # _build_graph) — it does NOT need the GPU, and running it
            # on the PPOActor's GPU (device 0) steals memory from
            # ``jit_update_step`` and OOMs the PPO weight update. The CPU
            # pool actors run on CPU devices too, so this also keeps the
            # local path byte-identical to the pool path.
            try:
                self._tokenize_device = jax.devices("cpu")[0]
            except Exception:
                self._tokenize_device = None
        return self._in_proc_server

    def _tokenize_local(self, order_np, specs_np, step_np):
        """Compute non-terminal state-tokens IN-PROCESS (no Ray).

        For the NON-TERMINAL steps under ``terminal_rewards_only`` the
        reward is zeroed (``env._callback`` early-returns before any
        jacve compile / cost_analysis) and the tokens are a pure
        deterministic fn of ``(order, specs, step)``. So we run the same
        ``env._callback`` locally via the in-proc server, skipping the
        ~2 s/step Ray round-trip (~99% of which is Ray overhead).

        Returns the same ``(tokens, eqn_ids, rewards, sentinel_mask)``
        contract as ``_fan_out_tokenize`` — ``rewards`` is all-zero and
        ``sentinel_mask`` is all-False (the local path never times out;
        a callback exception still raises, surfacing the bug rather than
        masking it on the cheap deterministic path).
        """
        server = self._ensure_in_proc_server()
        N = order_np.shape[0]
        dev = getattr(self, "_tokenize_device", None)
        if dev is not None:
            with jax.default_device(dev):
                out = [
                    server.evaluate(order_np[i], specs_np[i], int(step_np[i]))
                    for i in range(N)
                ]
        else:
            out = [
                server.evaluate(order_np[i], specs_np[i], int(step_np[i]))
                for i in range(N)
            ]
        tokens = np.stack([r[0] for r in out])
        eqn_ids = np.stack([r[1] for r in out])
        rewards = np.zeros((N, NUM_REWARDS), dtype=np.float32)
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens, eqn_ids, rewards, sentinel_mask

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
            server = self._ensure_in_proc_server()
            out = [
                server.evaluate(
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
                multipliers=getattr(self, "multipliers_np", None),
                extras={
                    "lagrangian_warmup_eps": int(getattr(self, "lagrangian_warmup_eps", 0)),
                },
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
        # Build the process-local in-proc server up front when the
        # local-tokenize path is enabled, so the first non-terminal step
        # doesn't pay the env-reuse setup cost mid-rollout.
        if self._local_tokenize_enabled:
            self._ensure_in_proc_server()
        return True
