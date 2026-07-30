"""ray_vertex_ppo_v2.py
====================

A pure-JAX rewrite of ``ray_vertex_ppo.py`` that

1. drops the Torch / FLA learner in favour of a Flax NNX agent (mirroring the
   actor-topology of ``alphagrad.approx.mu0_ray``);

2. moves the graphax compilation work (``extract_jaxpr`` /
   ``vertex_elimination_jaxpr``) out of ``RayVertexGame``'s ``jax.pure_callback``
   sites and into a dedicated, cache-owning ``JitCompilationActor`` that the
   rollout workers RPC into. This decouples policy inference from symbolic
   compilation, lets the workers stay JAX-only/JIT-friendly for the network
   forward pass, and lets the (slow, Python-bound) graphax work be amortised
   across the worker pool by a single trie cache.

Actor topology
--------------
* **JitCompilationActor** (CPU)
    Owns the jaxpr / argnums / consts / sparse flag, the list of valid
    vertices, and the ``TrieCache``. Exposes ``tokenize`` and ``step_counts``
    (and batched variants) for workers, plus ``info`` for the driver. Runs
    no JAX devices itself - only graphax + numpy.

* **PPOLearner** (GPU)
    Owns the NNX agent + an ``optax`` optimiser. Single ``nnx.jit``'d
    ``train_step`` that runs the clipped-PPO loss with symlog value targets.
    ``weights()`` returns the agent's NNX state as a numpy pytree for cheap
    cross-actor transport.

* **RolloutWorker** (CPU, JAX_PLATFORMS=cpu)
    Pulls weights from the learner, builds a local NNX agent, and steps the
    environment Python-side (no ``lax.scan``): every step it runs a jitted
    forward pass to get logits/value, samples an action under an availability
    mask, then RPCs the JitCompilationActor for the new tokens / fma counts.
    Returns numpy trajectories.

Why Python-side stepping
------------------------
A ``jax.pure_callback`` cannot RPC into a Ray actor (the callback runs on the
JAX device thread; ray.get inside it would deadlock and break tracing). Since
the graphax work is the *only* reason ``RayVertexGame.step`` was inside JAX in
v1, lifting it into an actor naturally pulls the rollout out of ``lax.scan``.
The agent forward pass is still ``nnx.jit``'d per step, which is what
dominates wall-clock anyway for small models on CPU.

Run
---
    python -m alphagrad.ppo.ray_vertex_ppo_v2 [--episodes N] [--local-test]
        [--num-gpus G] [--num-jit-actors K]
        [--ray-py /path/to/python] [--ray-cpu-py /path/to/python]
        [--ray-gpu-py /path/to/python]

If ``--ray-*-py`` is omitted the code defaults to ``$ALPHAGRAD_RAY_*_PY`` env
vars and finally to ``sys.executable`` - that lets the same script run on the
laptop and on the cluster without code edits.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ray
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants -- kept in sync with alphagrad.vertexgame.ray_vertex_game.
# ---------------------------------------------------------------------------
MAX_TOKENS = 1024
VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# GAE (numpy, runs on the driver between rollout and learner train_step).
# ---------------------------------------------------------------------------
def symlog_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def symexp_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1.0)


def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    """Vanilla GAE-lambda with symexp-decoded value baselines."""
    _, T = rewards.shape
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[:, t]
        v_raw = symexp_np(values[:, t])
        v_next_raw = symexp_np(next_values[:, t])
        delta = rewards[:, t] + v_next_raw * discounts[:, t] * mask - v_raw
        advantage = delta + discounts[:, t] * gae_lambda * lastgaelam * mask
        lastgaelam = advantage
        advantages[:, t] = advantage
        returns[:, t] = advantage + v_raw

    return advantages, returns


# ---------------------------------------------------------------------------
# Inline NNX agent. Self-contained so that v2 has no transitive dependency on
# the (currently broken) ``alphagrad.transformer.gated_deltanet.model`` path
# that v1 imports. Mirrors v1's interface: ``forward(tokens) -> (logits, value)``
# where ``tokens`` is shape ``(B, MAX_TOKENS)`` int32 and the model pools over
# the sequence into a single decision.
# ---------------------------------------------------------------------------
def build_nnx_agent_classes():
    """Importing flax.nnx is expensive (pulls JAX); do it lazily inside actors."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    class MLP(nnx.Module):
        def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, *, rngs: nnx.Rngs):
            self.layers = []
            curr = in_dim
            for h in hidden_dims:
                self.layers.append(nnx.Linear(curr, h, rngs=rngs))
                curr = h
            self.head = nnx.Linear(curr, out_dim, rngs=rngs)

        def __call__(self, x):
            for layer in self.layers:
                x = nnx.relu(layer(x))
            return self.head(x)

    class SelfAttentionBlock(nnx.Module):
        """Pre-norm multi-head self-attention + feed-forward, residual."""

        def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs):
            assert d_model % n_heads == 0, "d_model must divide num_heads"
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
            self.q = nnx.Linear(d_model, d_model, rngs=rngs)
            self.k = nnx.Linear(d_model, d_model, rngs=rngs)
            self.v = nnx.Linear(d_model, d_model, rngs=rngs)
            self.o = nnx.Linear(d_model, d_model, rngs=rngs)
            self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
            self.ff1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
            self.ff2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)

        def __call__(self, x):
            # x: (B, T, D)
            h = self.norm1(x)
            B, T, _ = h.shape
            q = self.q(h).reshape(B, T, self.n_heads, self.head_dim)
            k = self.k(h).reshape(B, T, self.n_heads, self.head_dim)
            v = self.v(h).reshape(B, T, self.n_heads, self.head_dim)
            # (B, H, T, T)
            scores = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(self.head_dim)
            attn = jax.nn.softmax(scores, axis=-1)
            ctx = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, self.d_model)
            x = x + self.o(ctx)
            h2 = self.norm2(x)
            x = x + self.ff2(nnx.gelu(self.ff1(h2)))
            return x

    class NNXPPOAgent(nnx.Module):
        def __init__(
            self,
            *,
            vocab_size: int,
            embd_dim: int,
            num_layers: int,
            num_heads: int,
            num_actions: int,
            policy_dims: list[int],
            value_dims: list[int],
            seq_len: int = MAX_TOKENS,
            rngs: nnx.Rngs,
        ):
            self.num_actions = num_actions
            self.embd_dim = embd_dim
            self.embedding = nnx.Embed(vocab_size, embd_dim, rngs=rngs)
            self.pos_enc = nnx.Param(
                jax.random.normal(rngs.params(), (1, seq_len, embd_dim)) * 0.02
            )
            self.blocks = [
                SelfAttentionBlock(embd_dim, num_heads, rngs=rngs)
                for _ in range(num_layers)
            ]
            self.policy_head = MLP(embd_dim, policy_dims, num_actions, rngs=rngs)
            self.value_head = MLP(embd_dim, value_dims, 1, rngs=rngs)

        def __call__(self, tokens):
            # tokens: (B, T) int
            x = self.embedding(tokens)
            x = x + self.pos_enc[:, : x.shape[1], :]
            for block in self.blocks:
                x = block(x)
            summary = jnp.mean(x, axis=1)
            logits = self.policy_head(summary)
            value = self.value_head(summary).squeeze(-1)
            return logits, value

    return nnx, NNXPPOAgent


# ---------------------------------------------------------------------------
# JitCompilationActor: owns the jaxpr + trie cache and runs all graphax calls.
# ---------------------------------------------------------------------------
@ray.remote
class JitCompilationActor:
    """Centralised owner of the graphax symbolic-compilation work.

    In v1 the same logic ran inside ``jax.pure_callback`` calls embedded in
    ``RayVertexGame.{reset,step}``, which meant every rollout worker had its
    own private trie cache and re-tokenised order prefixes the others had
    already seen. By promoting the cache into a Ray actor we (a) share work
    across the pool and (b) keep JAX threads free of the global-interpreter-
    locked graphax code.
    """

    def __init__(self, env_config: dict[str, Any]):
        # Late imports so this actor's process is pure-Python at construction.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp

        if "target_fn" in env_config:
            target_fn_val = env_config["target_fn"]
            if isinstance(target_fn_val, str):
                import graphax.examples as examples
                target_fn = getattr(examples, target_fn_val)
            else:
                target_fn = target_fn_val

            args_raw = env_config["args"]
            jax_args = tuple(jnp.array(x) for x in args_raw)
            closed_jaxpr = jax.make_jaxpr(target_fn)(*jax_args)
            self.jaxpr = closed_jaxpr.jaxpr
            self.consts = tuple(jax.device_get(c) for c in closed_jaxpr.literals)
            self.argnums = tuple(range(len(jax_args)))
            self.args_np = tuple(jax.device_get(x) for x in jax_args)
        else:
            self.jaxpr = env_config["jaxpr"]
            self.argnums = tuple(env_config["argnums"])
            self.args_np = tuple(jax.device_get(a) for a in env_config["args"])
            self.consts = tuple(jax.device_get(c) for c in env_config["consts"])

        self.sparse = env_config.get("sparse", False)

        valid_vertices = env_config.get("valid_vertices")
        if valid_vertices is None:
            from graphax.core import _build_graph

            _, _, _, _, vo_vertices = _build_graph(
                self.jaxpr, self.args_np, self.consts
            )
            valid: list[int] = []
            for i, eqn in enumerate(self.jaxpr.eqns, 1):
                if eqn.outvars[0] not in self.jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        self.valid_vertices = list(valid_vertices)
        self.total_v = len(self.jaxpr.eqns)
        self.num_valid = len(self.valid_vertices)

        # Plain-Python trie cache. Worker processes never touch it directly;
        # they go through the ``tokenize`` / ``step_counts`` RPCs and so the
        # cache is automatically shared across the whole worker pool.
        self._tokens_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._counts_cache: dict[tuple[int, ...], int] = {}

    # ---- read-only metadata ---------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "total_v": self.total_v,
            "num_valid": self.num_valid,
            "valid_vertices": list(self.valid_vertices),
            "max_tokens": MAX_TOKENS,
            "vocab_size": VOCAB_SIZE,
        }

    # ---- graphax work ---------------------------------------------------
    def _tokenize(self, prefix_tuple: tuple[int, ...]) -> np.ndarray:
        cached = self._tokens_cache.get(prefix_tuple)
        if cached is not None:
            return cached
        from graphax.core import extract_jaxpr

        ve = extract_jaxpr(
            self.jaxpr,
            self.argnums,
            list(prefix_tuple),
            self.sparse,
            self.args_np,
            self.consts,
        )
        result = np.zeros(MAX_TOKENS, dtype=np.int32)
        toks = ve.tokenized()
        n = min(len(toks), MAX_TOKENS)
        result[:n] = toks[:n]
        np.clip(result, 0, VOCAB_SIZE - 1, out=result)
        self._tokens_cache[prefix_tuple] = result
        return result

    def _step_counts(self, prefix_tuple: tuple[int, ...]) -> int:
        cached = self._counts_cache.get(prefix_tuple)
        if cached is not None:
            return cached
        from graphax.core import vertex_elimination_jaxpr

        _, aux = vertex_elimination_jaxpr(
            self.jaxpr,
            list(prefix_tuple),
            self.consts,
            *self.args_np,
            argnums=self.argnums,
            count_ops=True,
            sparse_representation=self.sparse,
        )
        counts = int(aux["fmas"])
        self._counts_cache[prefix_tuple] = counts
        return counts

    def tokenize(self, prefix: list[int]) -> np.ndarray:
        return self._tokenize(tuple(int(x) for x in prefix))

    def step_counts(self, prefix: list[int]) -> int:
        return self._step_counts(tuple(int(x) for x in prefix))

    def transition(self, prefix: list[int]) -> dict[str, Any]:
        """Convenience: bundle a tokenize + step_counts in a single RPC.

        Workers call this once per env step; that's one round-trip per step
        instead of two and matches what ``RayVertexGame.step`` did under v1.
        """
        prefix_tuple = tuple(int(x) for x in prefix)
        return {
            "tokens": self._tokenize(prefix_tuple),
            "fmas": self._step_counts(prefix_tuple),
        }

    def cache_stats(self) -> dict[str, int]:
        return {
            "tokens_entries": len(self._tokens_cache),
            "counts_entries": len(self._counts_cache),
        }


# ---------------------------------------------------------------------------
# PPOLearner (GPU): NNX agent + optax + jitted train_step.
# ---------------------------------------------------------------------------
@ray.remote
class PPOLearner:
    def __init__(self, agent_config: dict[str, Any]):
        # Late, in-actor imports keep the driver free of JAX (mirrors mu0_ray).
        import jax
        import jax.numpy as jnp
        import optax
        from flax import nnx

        self._jax = jax
        self._jnp = jnp
        self._optax = optax
        self._nnx = nnx

        _, NNXPPOAgent = build_nnx_agent_classes()
        self._AgentCls = NNXPPOAgent

        rngs = nnx.Rngs(int(agent_config.get("seed", 0)))
        self.agent = NNXPPOAgent(
            vocab_size=agent_config["vocab_size"],
            embd_dim=agent_config["embd_dim"],
            num_layers=agent_config["num_layers"],
            num_heads=agent_config["num_heads"],
            num_actions=agent_config["num_actions"],
            policy_dims=agent_config["policy_dims"],
            value_dims=agent_config["value_dims"],
            seq_len=MAX_TOKENS,
            rngs=rngs,
        )

        self.optimizer = nnx.Optimizer(
            self.agent,
            optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(agent_config["lr"]),
            ),
            wrt=nnx.Param,
        )

        self.eps = float(agent_config["eps"])
        self.value_weight = float(agent_config["value_weight"])
        self.entropy_weight = float(agent_config["entropy_weight"])
        self.config = dict(agent_config)
        self._build_train_step()

    # -- jitted PPO update -------------------------------------------------
    def _build_train_step(self):
        nnx = self._nnx
        jax = self._jax
        jnp = self._jnp

        eps = self.eps
        v_w = self.value_weight
        e_w = self.entropy_weight

        def loss_fn(agent, tokens, actions, old_log_probs, advantages, returns):
            logits, values = agent(tokens)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            curr_lp = jnp.take_along_axis(log_probs, (actions - 1)[:, None], axis=-1).squeeze(-1)

            ratio = jnp.exp(curr_lp - old_log_probs)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
            ppo_loss = -jnp.minimum(surr1, surr2).mean()

            # Symlog targets so the value head can fit wide-magnitude returns
            # without saturating; matches v1's torch loss.
            sign = jnp.sign(returns)
            value_targets = sign * jnp.log1p(jnp.abs(returns))
            value_loss = jnp.mean(jnp.square(values - value_targets))

            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

            total = ppo_loss + v_w * value_loss - e_w * entropy
            return total, (ppo_loss, value_loss, entropy)

        @nnx.jit
        def train_step(agent, optimizer, tokens, actions, old_log_probs, advantages, returns):
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (total, (ppo_loss, value_loss, entropy)), grads = grad_fn(
                agent, tokens, actions, old_log_probs, advantages, returns
            )
            optimizer.update(agent, grads)
            return total, ppo_loss, value_loss, entropy

        self._train_step = train_step

    # -- ray-callable surface ---------------------------------------------
    def get_status(self) -> dict[str, Any]:
        return {
            "PID": os.getpid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "devices": [str(d) for d in self._jax.devices()],
        }

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        jnp = self._jnp
        tokens = jnp.asarray(batch["tokens"], dtype=jnp.int32)
        actions = jnp.asarray(batch["actions"], dtype=jnp.int32)
        old_lp = jnp.asarray(batch["log_probs"], dtype=jnp.float32)
        adv = jnp.asarray(batch["advantages"], dtype=jnp.float32)
        ret = jnp.asarray(batch["returns"], dtype=jnp.float32)

        total, ppo_l, val_l, ent = self._train_step(
            self.agent, self.optimizer, tokens, actions, old_lp, adv, ret
        )
        return {
            "loss": float(total),
            "ppo_loss": float(ppo_l),
            "value_loss": float(val_l),
            "entropy": float(ent),
        }

    def weights(self) -> dict[str, np.ndarray]:
        """Return the agent's variable state as a flat ``{path: ndarray}`` dict.

        Worker actors rebuild a model with the same architecture and overwrite
        their state with this dict. Using ``flatten`` rather than the full
        nested ``State`` keeps the wire format trivially picklable.
        """
        nnx = self._nnx
        state = nnx.state(self.agent, nnx.Param)
        flat = {}
        for path, leaf in state.flat_state().items():
            key = ".".join(str(p) for p in path)
            arr = leaf.value if hasattr(leaf, "value") else leaf
            flat[key] = np.asarray(arr)
        return flat


# ---------------------------------------------------------------------------
# RolloutWorker (CPU): runs the env step-by-step against the JitCompilationActor.
# ---------------------------------------------------------------------------
@ray.remote
class RolloutWorker:
    def __init__(self, worker_id: int, agent_config: dict[str, Any], env_info: dict[str, Any]):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import jax.numpy as jnp
        from flax import nnx

        self._jax = jax
        self._jnp = jnp
        self._nnx = nnx
        self._AgentCls = build_nnx_agent_classes()[1]
        self.worker_id = worker_id
        self.config = dict(agent_config)
        self.total_v = int(env_info["total_v"])
        self.num_valid = int(env_info["num_valid"])
        self.valid_vertices = np.asarray(env_info["valid_vertices"], dtype=np.int32)

        # Pre-build the agent at construction time so the worker is JIT-warm
        # before the first sample(); weights get *overwritten* every rollout
        # so the initial random init is throwaway.
        self.rngs = nnx.Rngs(int(agent_config.get("seed", 0)) + 100 + worker_id)
        self.agent = self._AgentCls(
            vocab_size=agent_config["vocab_size"],
            embd_dim=agent_config["embd_dim"],
            num_layers=agent_config["num_layers"],
            num_heads=agent_config["num_heads"],
            num_actions=agent_config["num_actions"],
            policy_dims=agent_config["policy_dims"],
            value_dims=agent_config["value_dims"],
            seq_len=MAX_TOKENS,
            rngs=self.rngs,
        )

        @nnx.jit
        def forward(agent, tokens):
            return agent(tokens[None, :])  # (1, T) -> (1, A), (1,)

        self._forward = forward
        # Warm trace
        self._forward(self.agent, jnp.zeros((MAX_TOKENS,), dtype=jnp.int32))

    # ----- weight sync ----------------------------------------------------
    def _load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Overwrite the local agent's Param state with the learner snapshot."""
        nnx = self._nnx
        jnp = self._jnp
        state = nnx.state(self.agent, nnx.Param)
        for path, leaf in list(state.flat_state().items()):
            key = ".".join(str(p) for p in path)
            if key not in weights:
                # Should not happen if architectures match; surface loudly.
                raise KeyError(f"weight {key!r} missing from learner snapshot")
            arr = jnp.asarray(weights[key])
            if hasattr(leaf, "value"):
                leaf.value = arr
            else:
                state.replace_by_pure_dict({path: arr})
        nnx.update(self.agent, state)

    # ----- env helpers (Python-side) -------------------------------------
    def _initial_state(self, jit_actor) -> dict[str, Any]:
        order = self.valid_vertices.copy()
        step_count = 0
        # Initial tokens correspond to the empty prefix (no eliminations yet).
        tokens = ray.get(jit_actor.tokenize.remote([]))
        return {
            "order": order,
            "step_count": step_count,
            "fmas": 0,
            "tokens": np.asarray(tokens, dtype=np.int32),
            "terminated": False,
        }

    def _apply_action(self, state: dict[str, Any], action: int) -> dict[str, Any]:
        """Pure-numpy version of v1's RayVertexGame.step order-update logic."""
        order = state["order"]
        idx = state["step_count"]
        # Position of the chosen vertex inside the order array.
        pos = int(np.argmax(order == action))
        if pos > idx:
            shifted = np.roll(order, 1)
            mask = (np.arange(len(order)) > idx) & (np.arange(len(order)) <= pos)
            new_order = np.where(mask, shifted, order)
        else:
            new_order = order.copy()
        new_order[idx] = action
        return {
            "order": new_order,
            "step_count": idx + 1,
        }

    # ----- main rollout ---------------------------------------------------
    def sample(
        self,
        rollout_length: int,
        learner,
        jit_actor,
        rng_seed: int,
    ) -> list[dict[str, Any]]:
        jnp = self._jnp

        weights = ray.get(learner.weights.remote())
        self._load_weights(weights)

        rng = np.random.default_rng(rng_seed)
        state = self._initial_state(jit_actor)
        trajs: list[dict[str, Any]] = []

        for _ in range(rollout_length):
            tokens_jnp = jnp.asarray(state["tokens"], dtype=jnp.int32)
            logits, value = self._forward(self.agent, tokens_jnp)
            logits_np = np.asarray(logits[0])
            value_scalar = float(value[0])

            # Availability mask: a vertex is available iff it's in valid_vertices
            # and has not yet been chosen (i.e. its current position in `order`
            # is at or beyond `step_count`).
            available = np.zeros(self.total_v, dtype=np.float32)
            for v in self.valid_vertices:
                pos = int(np.argmax(state["order"] == v))
                if pos >= state["step_count"]:
                    available[v - 1] = 1.0

            masked = np.where(available > 0.5, logits_np, -1e9)
            # Numerically stable softmax sample.
            shifted = masked - masked.max()
            exps = np.exp(shifted)
            probs = exps / max(exps.sum(), 1e-12)
            action_idx = int(rng.choice(self.total_v, p=probs))
            action = action_idx + 1
            log_prob = float(np.log(max(probs[action_idx], 1e-7)))

            # Apply the order update Python-side and ask the JIT actor for
            # the resulting tokens + fma counts in a single RPC.
            updated = self._apply_action(state, action)
            new_step = updated["step_count"]
            prefix = updated["order"][:new_step].tolist()
            transition = ray.get(jit_actor.transition.remote(prefix))
            new_tokens = np.asarray(transition["tokens"], dtype=np.int32)
            new_fmas = int(transition["fmas"])
            reward = float(state["fmas"] - new_fmas)

            done = new_step >= self.num_valid
            trajs.append(
                {
                    "obs": state["tokens"].copy(),
                    "action": action,
                    "reward": reward,
                    "done": bool(done),
                    "value": value_scalar,
                    "log_prob": log_prob,
                    "discount": 0.99,
                }
            )

            if done:
                state = self._initial_state(jit_actor)
            else:
                state = {
                    "order": updated["order"],
                    "step_count": new_step,
                    "fmas": new_fmas,
                    "tokens": new_tokens,
                    "terminated": False,
                }

        return trajs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def find_workspace_root() -> Path:
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "graphax").exists() and (parent / "alphagrad").exists():
            return parent
    return Path(__file__).resolve().parents[4]


def _resolve_py_executable(arg_value: str | None, env_var: str) -> str:
    """Fall back chain for ``runtime_env={"py_executable": ...}`` paths.

    1. CLI arg (``--ray-py`` etc.)
    2. Environment variable
    3. ``sys.executable``
    """
    if arg_value:
        return arg_value
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO + Ray on the vertex elimination env (v2)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rollout-length", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="CPU rollout workers; 0 = auto (cpu_count - 1)")
    parser.add_argument("--num-jit-actors", type=int, default=1,
                        help="JitCompilationActor replicas; >1 keeps caches independent.")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPUs requested by ray.init; None = let ray autodetect.")
    parser.add_argument("--learner-gpus", type=float, default=1.0,
                        help="GPUs the learner actor reserves; 0 to run learner on CPU.")
    parser.add_argument("--local-test", action="store_true",
                        help="Cap CPUs/workers for laptop runs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-fn", type=str, default="Helmholtz",
                        help="graphax.examples.<name> (e.g. Helmholtz, g, ...)")
    parser.add_argument("--ray-py", type=str, default=None)
    parser.add_argument("--ray-cpu-py", type=str, default=None)
    parser.add_argument("--ray-gpu-py", type=str, default=None)
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 4
    root = find_workspace_root()

    ray_py = _resolve_py_executable(args.ray_py, "ALPHAGRAD_RAY_PY")
    ray_cpu_py = _resolve_py_executable(args.ray_cpu_py, "ALPHAGRAD_RAY_CPU_PY")
    ray_gpu_py = _resolve_py_executable(args.ray_gpu_py, "ALPHAGRAD_RAY_GPU_PY")

    if not ray.is_initialized():
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        new_paths = "alphagrad/src:graphax/src"
        merged_pythonpath = (
            f"{new_paths}:{current_pythonpath}" if current_pythonpath else new_paths
        )

        env_vars = {"PYTHONPATH": merged_pythonpath}
        if os.environ.get("UV_CACHE_DIR"):
            env_vars["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]

        ray_kwargs: dict[str, Any] = {
            "num_cpus": 8 if args.local_test else cpu_count,
            "runtime_env": {
                "working_dir": str(root),
                "env_vars": env_vars,
                "py_executable": ray_py,
                "excludes": [
                    ".ray_*",
                    "python3.13", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
                    "**/pyproject.toml", "**/uv.lock",
                    "dann", "snnax", "synaptax", "torchneuromorphic", "graphax-og",
                    "graphax_bak",
                    "wandb", "dist", "build", "target", "eval", "docs", "tests", "~",
                    "*.zip", "*.lp", "*.mps", "*.log", "**/.*_cache",
                ],
            },
        }
        if args.num_gpus is not None:
            ray_kwargs["num_gpus"] = args.num_gpus
        ray.init(**ray_kwargs)

    # --- Env config -------------------------------------------------------
    if args.target_fn == "Helmholtz":
        xs_np = [np.random.uniform(0, 1, (4,)).astype(np.float32)]
    elif args.target_fn == "g":
        xs_np = [np.random.uniform(0, 1, (1,)).astype(np.float32) for _ in range(15)]
    else:
        # Generic single-arg fallback. Override with --target-fn at your own risk.
        xs_np = [np.random.uniform(0, 1, (4,)).astype(np.float32)]

    env_config = {
        "target_fn": args.target_fn,
        "args": xs_np,
        "sparse": False,
    }

    # --- Spin up the JIT compilation actor pool --------------------------
    jit_runtime_env = {
        "py_executable": ray_cpu_py,
        "env_vars": {"JAX_PLATFORMS": "cpu"},
    }
    jit_actors = [
        JitCompilationActor.options(
            num_cpus=1, num_gpus=0, runtime_env=jit_runtime_env
        ).remote(env_config)
        for _ in range(max(1, args.num_jit_actors))
    ]
    primary_jit = jit_actors[0]
    env_info = ray.get(primary_jit.info.remote())
    print(f"[v2] env: total_v={env_info['total_v']}  num_valid={env_info['num_valid']}")

    # --- Agent / learner -------------------------------------------------
    agent_config = {
        "vocab_size": VOCAB_SIZE,
        "embd_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "num_actions": env_info["total_v"],
        "policy_dims": [128, 64],
        "value_dims": [64],
        "lr": 3e-4,
        "eps": 0.2,
        "value_weight": 0.5,
        "entropy_weight": 0.01,
        "seed": args.seed,
    }

    learner_gpus = 0.0 if args.local_test else args.learner_gpus
    learner_runtime_env = {
        "py_executable": ray_gpu_py if learner_gpus > 0 else ray_cpu_py,
        "env_vars": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
            **({"JAX_PLATFORMS": "cpu"} if learner_gpus == 0 else {}),
        },
    }
    learner = PPOLearner.options(
        num_gpus=learner_gpus, runtime_env=learner_runtime_env
    ).remote(agent_config)

    # --- Rollout worker pool ---------------------------------------------
    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = 4 if args.local_test else max(1, cpu_count - 1)

    worker_runtime_env = {
        "py_executable": ray_cpu_py,
        "env_vars": {"JAX_PLATFORMS": "cpu"},
    }
    workers = [
        RolloutWorker.options(
            num_cpus=1, num_gpus=0, runtime_env=worker_runtime_env
        ).remote(i, agent_config, env_info)
        for i in range(num_workers)
    ]

    # --- Wait for all actors to be JIT-warm ------------------------------
    status = ray.get(learner.get_status.remote())
    print(f"[v2] learner status: {status}")
    print(f"[v2] {num_workers} rollout workers, {len(jit_actors)} jit-actors")

    pbar = tqdm(range(args.episodes))
    for ep in pbar:
        # Round-robin workers across the JIT-actor replicas.
        futures = [
            workers[i].sample.remote(
                args.rollout_length,
                learner,
                jit_actors[i % len(jit_actors)],
                rng_seed=args.seed * 1_000_003 + ep * 1_009 + i,
            )
            for i in range(num_workers)
        ]
        results = ray.get(futures)
        all_trajs = [item for sublist in results for item in sublist]
        T = args.rollout_length

        obs_batch = np.stack([t["obs"] for t in all_trajs], axis=0)
        actions = np.array([t["action"] for t in all_trajs], dtype=np.int32)
        rewards = np.array([t["reward"] for t in all_trajs], dtype=np.float32).reshape(num_workers, T)
        dones = np.array([t["done"] for t in all_trajs], dtype=np.float32).reshape(num_workers, T)
        values = np.array([t["value"] for t in all_trajs], dtype=np.float32).reshape(num_workers, T)
        log_probs = np.array([t["log_prob"] for t in all_trajs], dtype=np.float32)
        discounts = np.array([t["discount"] for t in all_trajs], dtype=np.float32).reshape(num_workers, T)

        # next-value bootstrap: last step uses 0 (matches v1).
        next_values = np.zeros_like(values)
        next_values[:, :-1] = values[:, 1:]

        advs, rets = get_advantages(rewards, dones, values, next_values, discounts, 0.95)

        train_data = {
            "tokens": obs_batch,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advs.flatten(),
            "returns": rets.flatten(),
        }
        metrics = ray.get(learner.train_step.remote(train_data))
        pbar.set_description(
            f"loss={metrics['loss']:+.3g} ppo={metrics['ppo_loss']:+.3g} "
            f"v={metrics['value_loss']:+.3g} H={metrics['entropy']:+.3g} "
            f"r̄={rewards.mean():+.3g}"
        )

    cache_stats = ray.get([a.cache_stats.remote() for a in jit_actors])
    print(f"[v2] PPO training finished. jit-cache stats: {cache_stats}")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
