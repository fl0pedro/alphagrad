"""CPU-side host of the per-step approximation reward harness.

Today the env runs the (tokenize → jacve compile → execute → cost_analysis
→ reward) pipeline inside a `jax.experimental.io_callback`, which puts the
work on the host main thread under the GIL and pins three problematic
costs to *every* env-step:

* graphax jaxpr elimination + re-tokenization (single-threaded Python).
* JIT-compile of the approximated jaxpr (~100 MB – 1 GB Executables).
* `compiled_approx.cost_analysis()` — ~9 MB C++ allocation per call that
  isn't reliably released, currently mitigated with the
  `ALPHAGRAD_SKIP_COST_ANALYSIS=1` env var (see `env.py:1182`).

This module factors the same computation out of `io_callback` into a
plain Python class (`CpuApproximationServer`) suitable for owning from a
Ray actor. The driver gets to fan tokenization across N cores instead of
serialising it through the trainer process, and the cost-analysis C++
leak is bounded by recycling the actor every K episodes (the driver's
job, see `ppo_ray.py`/`mu0_ray.py`).

The class is deliberately *not* a Ray actor itself — the `@ray.remote`
wrapper lives in `cpu_approx_actors.py` so this file stays importable
without Ray (for the in-process roundtrip tests and for the
`engineering:debug` flow that wants to call `evaluate(...)` directly).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Sequence


def _setup_jax_compile_cache() -> None:
    """Enable JAX's persistent disk compile cache for this process.

    No-op if JAX is already configured. The cache dir defaults to
    ``~/.cache/jax-compile`` and is shared across Ray actors on the
    same host — duplicated compiles in different actors hit the same
    on-disk cache and skip the HLO-generation pass.

    Idempotent: re-runs of this function (e.g. from multiple
    ``CpuApproximationServer.__init__``s in the same process) only
    do anything once.
    """
    import jax

    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "jax-compile"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


class CpuApproximationServer:
    """Process-local wrapper around `env._callback`.

    Two construction modes:

    * `from_args_dict(args_dict, variant, ...)` — replays just enough of
      `mu0_ray_worker._build_actor_state` to reconstruct the env-side
      state (`config`, `args`, `consts`, `eval_args_samples`). This is
      the Ray-friendly path: the args_dict is the same JSON-able
      shape the SPMD actor receives.
    * `from_env(env)` — convenience for in-process tests; reuses an
      already-constructed `VertexEliminationEnv` so unit tests don't
      re-run the full env-build dance.

    The interface is intentionally numpy-in / numpy-out so the values
    survive a Ray RPC unchanged. Internally everything is converted to
    `jax.numpy` on the way in and back to `numpy` on the way out — JAX
    arrays don't travel cleanly through Ray's object store (they're
    bound to a device handle).
    """

    def __init__(self, env, eval_samples=None):
        # Take the env by reference. We never mutate it; everything we
        # need is on `env.config` / `env.args` / `env.consts` / the
        # optional `env.eval_args_samples` field.
        self._env = env
        self._config = env.config
        self._args = tuple(env.args)
        self._consts = tuple(env.consts)
        self._eval_samples = (
            tuple(eval_samples)
            if eval_samples is not None
            else (tuple(env.eval_args_samples) if env.eval_args_samples is not None else ())
        )
        # Wire the JAX persistent disk cache so each per-call
        # `jax.jit(jacve(...)).lower(...).compile()` re-uses the XLA
        # passes from the previous compile of the same (o_list, transforms).
        # Without this the CPU worker pays the full HLO-generation cost
        # on every step (~50-200 ms each). Cache is shared across all
        # Ray actors on the host because the dir lives under
        # ``~/.cache/jax-compile`` by default. JAX's hashing keys on
        # the lowered HLO, so unrelated workers' caches don't poison
        # each other.
        _setup_jax_compile_cache()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env, eval_samples=None) -> "CpuApproximationServer":
        return cls(env, eval_samples=eval_samples)

    @classmethod
    def from_args_dict(
        cls,
        args_dict: dict,
        variant: str | None = None,
        *,
        seed: int = 0,
    ) -> "CpuApproximationServer":
        env = _build_env_from_args(args_dict, variant, seed=seed)
        return cls(env)

    # ------------------------------------------------------------------
    # Core evaluation API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        order: Any,
        sparsity_specs: Any,
        step: int,
        *,
        eval_samples: Sequence | None = None,
        init: bool = False,
    ):
        """Run the per-step reward pipeline once.

        Mirrors the signature `env._callback` is called with from
        `io_callback`:

        * `order`        — (num_valid,) int32. Current elimination order.
        * `sparsity_specs` — (num_valid, MAX_RULES_PER_VERTEX, 3) int32.
        * `step`         — int. Same `step_count + 1` value the env-side
                           caller would pass; `_callback` uses it as
                           the `partial_order[:stop]` upper bound.
        * `eval_samples` — optional; overrides the per-server default
                           the constructor stored. Each entry is a
                           stacked-leaf array matching one positional
                           arg position; the per-sample `arg[i]` indexing
                           inside `_callback` does the rest.
        * `init`         — True only for the reset-time tokenization (no
                           reward computation).

        Returns
        -------
        (tokens_np, eqn_ids_np, reward_vec_np)
            All numpy arrays, dtypes match the JAX-side counterparts
            (int32, int32, float32). Safe to ship over Ray.
        """
        import jax.numpy as jnp
        import numpy as np
        from alphagrad.approx.env import MAX_TOKENS, NUM_REWARDS, _callback

        order_j = jnp.asarray(order, dtype=jnp.int32)
        specs_j = jnp.asarray(sparsity_specs, dtype=jnp.int32)
        es = (
            tuple(eval_samples)
            if eval_samples is not None
            else self._eval_samples
        )
        try:
            tokens, eqn_ids, reward = _callback(
                self._config,
                self._args,
                self._consts,
                order_j,
                specs_j,
                int(step),
                *es,
                init=bool(init),
            )
            return np.asarray(tokens), np.asarray(eqn_ids), np.asarray(reward)
        except Exception as exc:
            # graphax can raise on transforms that produce shape-incompatible
            # edges (e.g. a DIAG whose factor doesn't divide some primal axis,
            # or a COMPRESS in the middle of the elimination order). Killing
            # the whole rollout for one bad action is over-strict — the
            # rollout buffer would lose all preceding work. Emit a sentinel
            # `(zeros, -1e10 cost reward)` so the policy gradient is pushed
            # away from the bad action and the rollout proceeds. The actual
            # tokens stay zero (the next-step policy will see an unhelpful
            # obs for one step, then training continues from the new state).
            #
            # Stash the message so a curious caller can grep diagnostics —
            # `last_eval_error` is updated atomically (just Python reference
            # assignment).
            self.last_eval_error = (type(exc).__name__, str(exc)[:200])
            sentinel_tokens = np.zeros((MAX_TOKENS,), dtype=np.int32)
            sentinel_eqn_ids = np.zeros((MAX_TOKENS,), dtype=np.int32)
            sentinel_reward = np.full((NUM_REWARDS,), -1e10, dtype=np.float32)
            # cosine_sim is "higher is better, capped at 1" — set to 0 so the
            # acc head doesn't see a weirdly bad positive signal.
            from alphagrad.approx.env import REWARD_INDEX
            sentinel_reward[REWARD_INDEX["cosine_sim"]] = 0.0
            sentinel_reward[REWARD_INDEX["frob_residual"]] = -1e10
            return sentinel_tokens, sentinel_eqn_ids, sentinel_reward

    def evaluate_batch(self, batch: Sequence[tuple]):
        """Sequential fallback for callers that want to ship multiple
        (order, specs, step) requests in one Ray roundtrip.

        Each tuple is `(order_np, specs_np, step)` or `(order_np,
        specs_np, step, eval_samples)`; the latter overrides
        per-element. Returns a list of `(tokens, eqn_ids, reward)`
        triples, preserving input order.

        Useful when one actor owns multiple envs — the Ray RPC cost is
        ~0.5 ms / call, so batching 8 envs amortises the roundtrip.
        """
        out = []
        for item in batch:
            if len(item) == 3:
                order, specs, step = item
                es = None
            elif len(item) == 4:
                order, specs, step, es = item
            else:
                raise ValueError(
                    f"evaluate_batch item must have 3 or 4 elements; got {len(item)}"
                )
            out.append(
                self.evaluate(order, specs, step, eval_samples=es)
            )
        return out

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset_caches(self) -> dict:
        """No-op stub.

        Previously dropped the in-process LRU around the per-call
        compile in `env._callback`. That LRU was thrashing under the
        actual rollout distribution (~2 % hit at startup, climbing
        slowly), so it was removed entirely. The only durable
        in-process growth left is the C++ `cost_analysis()` allocation,
        which the LRU couldn't have bounded anyway — the driver in
        `ppo_ray.py` / `mu0_ray.py` recycles this whole actor (`ray.kill`
        + respawn) every `--cpu-worker-recycle-every` episodes to
        return that allocation to the OS.

        Kept as a method so callers that already issued
        `actor.reset_caches.remote()` don't break; returns an empty
        dict for shape-stability with the old return type.
        """
        return {}

    def ready(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Internal: replay the env-build dance from mu0_ray_worker, scoped to the
# bits the reward harness actually needs (no agent, no optimizer, no
# sharding). Kept private — callers should use the `from_args_dict`
# classmethod above.
# ---------------------------------------------------------------------------
def _build_env_from_args(args_dict: dict, variant: str | None, *, seed: int = 0):
    """Rebuild a `VertexEliminationEnv` from a serialisable args_dict.

    This is a stripped-down twin of
    `mu0_ray_worker._build_actor_state`'s env-build branch — same
    `target_fn / xs / closed_jaxpr / argnums / from_jaxpr` sequence, then
    one `generate_eval_samples` call to populate `env.eval_args_samples`.
    The agent / optimizer / sharding work that follows in the SPMD
    server is intentionally omitted; this worker never sees a model.
    """
    import equinox as eqx
    import jax
    import jax.random as jrand
    from alphagrad.approx.common import (
        data_gen,
        generate_eval_samples,
        get_args,
        get_fn,
        infer_argnums,
    )
    from alphagrad.approx.env import VertexEliminationEnv
    from alphagrad.approx.variants import _apply_variant_preset

    args = SimpleNamespace(**args_dict)
    if variant is not None:
        _apply_variant_preset(args, variant)
    # Mirror the trainer's default — the CPU worker never benefits from
    # the GPU preallocator (it usually lands on CPU jax devices anyway),
    # and the env's `_callback` can transiently allocate GPU buffers via
    # `exec_on_gpu` mode.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    key = jrand.PRNGKey(int(getattr(args, "seed", seed)))
    key, args_key, eval_key = jrand.split(key, 3)

    dataset_arg = None if args.dataset == "none" else args.dataset
    use_dataset = (
        dataset_arg is not None and args.example.endswith("NeuralNetwork")
    )
    dataset_for_call = dataset_arg if use_dataset else None
    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key, dataset=dataset_for_call)
    gen = data_gen(
        args.example, dataset=dataset_for_call, dataset_size=args.dataset_size
    )
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    argnums = infer_argnums(args.example)

    env_target_fun = target_fn if "acc" in args.rewards else None
    measure_latency = bool(
        getattr(args, "measure_latency", False)
        or getattr(args, "cmp_type", "") == "latency"
    )
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=xs,
        argnums=argnums,
        num_envs=0,
        data_gen=gen,
        target_fun=env_target_fun,
        cmp_type=args.cmp_type,
        mem_type=args.mem_type,
        exec_on_gpu=getattr(args, "exec_on_gpu", False),
        measure_latency=measure_latency,
        terminal_rewards_only=getattr(args, "terminal_rewards_only", False),
    )

    num_eval = int(getattr(args, "num_eval_samples", 10) or 10)
    eval_samples = generate_eval_samples(env, eval_key, num_eval)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
