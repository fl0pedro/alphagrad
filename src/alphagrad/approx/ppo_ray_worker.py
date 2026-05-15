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
from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_REWARDS,
    REWARD_INDEX,
    StepAction,
    VertexEliminationEnv,
)


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

    embd_dim: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)

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
    ):
        from alphagrad.transformer import MLP, Encoder, PositionalEncoder

        k_emb, k_enc, k_pol, k_val = jrand.split(key, 4)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k_emb)
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k_enc)
        self.vertex_logits_head = MLP(
            embd_dim, num_vertices, policy_dims, key=k_pol,
        )
        self.value_head = MLP(embd_dim, 1, value_dims, key=k_val)
        self.embd_dim = embd_dim
        self.num_vertices = num_vertices

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
        self.num_envs = int(getattr(self.args, "num_envs", 4))
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

        # Agent + optimizer.
        policy_dims = self._parse_int_list(self.args.policy_dims)
        value_dims = self._parse_int_list(self.args.value_dims)
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
        )
        self.agent = init_linear_weights(self.agent, init_key)

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

        # Pre-reset batched env state (the JAX side only — tokens come
        # from a real callback once on construction since the very first
        # rollout needs valid tokens; subsequent resets use the external
        # path).
        self.env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
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

        @eqx.filter_jit
        def act_step(agent, state_batch, vert_avail_batch, key):
            keys = jrand.split(key, self.num_envs)

            def per_env(state_i, avail_i, k_i):
                pol_key, samp_key = jrand.split(k_i, 2)
                logits, value = agent.policy_and_value(state_i.tokens, key=pol_key)
                masked = jnp.where(avail_i > 0.5, logits, -1e9)
                # Categorical sample with manual log_prob (avoids the
                # distrax dependency here — we already import it
                # transitively via the alphagrad common helpers).
                log_probs_all = jax.nn.log_softmax(masked)
                action = jrand.categorical(k_i, masked)
                log_prob = log_probs_all[action]
                # `action` is an index into `0..total_v-1`; vertex IDs
                # in the env are 1-indexed.
                vertex_id = action + 1
                env_action = StepAction(
                    target_vertex=jnp.asarray(vertex_id, dtype=jnp.int32),
                    rule_specs=jnp.full(
                        (MAX_RULES_PER_VERTEX, 3), -1, dtype=jnp.int32,
                    ).at[..., 2].set(0),
                )
                partial, order, specs, step = env.step_external_jax_part(
                    state_i, env_action,
                )
                del samp_key  # silence unused-var warning under JIT trace
                return action, log_prob, value, partial, order, specs, step

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

        def loss_fn(agent, batch, key):
            tokens, actions, old_log_probs, returns, advantages = batch
            # vmap so each leading dim is one transition. Each per-sample
            # call gets a fresh fold of `key` so dropout (if added later)
            # doesn't share state across the batch.
            keys = jrand.split(key, tokens.shape[0])

            def per_sample(tok, act, olp, ret, adv, k):
                logits, value = agent.policy_and_value(tok, key=k)
                log_probs = jax.nn.log_softmax(logits)
                new_log_prob = log_probs[act]
                ratio = jnp.exp(new_log_prob - olp)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -jnp.minimum(surr1, surr2)
                value_loss = (value - ret) ** 2
                # Categorical entropy = -sum(p log p).
                p = jax.nn.softmax(logits)
                entropy = -jnp.sum(p * log_probs)
                return policy_loss, value_loss, entropy

            p_l, v_l, ent = jax.vmap(per_sample)(
                tokens, actions, old_log_probs, returns, advantages, keys,
            )
            ppo_loss = jnp.mean(p_l)
            value_loss = jnp.mean(v_l)
            entropy_loss = -jnp.mean(ent)  # we want to *maximise* entropy
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

        # Fresh on-policy episode: reset state for every env.
        self.env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )

        # Rollout buffers — numpy is fine; we re-stage to JAX once for
        # the update step.
        T = int(self.rollout_length)
        N = int(self.num_envs)
        buf_tokens = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        buf_actions = np.zeros((T, N), dtype=np.int32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T, N), dtype=np.float32)
        buf_rewards = np.zeros((T, N), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)

        state = self.env_states
        for t in range(T):
            key, sub = jrand.split(key)
            avail = self._vertex_avail(state)
            actions, log_probs, values, partial, order, specs, step = self._act_step(
                self.agent, state, avail, sub,
            )

            # Convert to numpy for Ray fan-out.
            order_np = np.asarray(order)
            specs_np = np.asarray(specs)
            step_np = np.asarray(step)

            tokens_np, eqn_ids_np, reward_np = self._fan_out_tokenize(
                order_np, specs_np, step_np,
            )

            tokens_j = jnp.asarray(tokens_np, dtype=jnp.int32)
            eqn_ids_j = jnp.asarray(eqn_ids_np, dtype=jnp.int32)
            reward_j = jnp.asarray(reward_np, dtype=jnp.float32)
            state = self._assemble(state, partial, tokens_j, eqn_ids_j, reward_j)

            # Record. We store the *post-step* tokens so the next-step
            # policy gradient targets see the same obs the policy used.
            buf_tokens[t] = np.asarray(state.tokens)
            buf_actions[t] = np.asarray(actions)
            buf_log_probs[t] = np.asarray(log_probs)
            buf_values[t] = np.asarray(values)
            buf_rewards[t] = np.asarray(
                self._scalar_reward(reward_j),
            )
            buf_dones[t] = np.asarray(state.terminated).astype(np.float32)

        # Bootstrap value at the final state (for the GAE next_value
        # term on the last timestep).
        key, boot_key = jrand.split(key)
        boot_keys = jrand.split(boot_key, N)
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

        # Normalise advantages per-rollout — mean 0 std 1, with the
        # `+1e-8` floor that's standard PPO hygiene.
        adv_flat = advantages_b.reshape(-1)
        adv_mean = jnp.mean(adv_flat)
        adv_std = jnp.std(adv_flat) + 1e-8
        advantages_b = (advantages_b - adv_mean) / adv_std

        # Stage for the update. Flatten (N, T) -> (N*T,) along the env
        # axis (each transition is independent for PPO).
        flat_tokens = jnp.asarray(buf_tokens.transpose(1, 0, 2).reshape(N * T, MAX_TOKENS))
        flat_actions = jnp.asarray(buf_actions.T.reshape(N * T))
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
        flat_log_probs = flat_log_probs[perm]
        flat_returns = flat_returns[perm]
        flat_advantages = flat_advantages[perm]

        mb_size = (N * T) // self.minibatches
        if mb_size == 0:
            mb_size = N * T  # fall back to a single mini-batch
            mb_count = 1
        else:
            mb_count = self.minibatches

        agent = self.agent
        opt_state = self.opt_state
        last_aux = {}
        for i in range(mb_count):
            sl = slice(i * mb_size, (i + 1) * mb_size)
            batch = (
                flat_tokens[sl],
                flat_actions[sl],
                flat_log_probs[sl],
                flat_returns[sl],
                flat_advantages[sl],
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
