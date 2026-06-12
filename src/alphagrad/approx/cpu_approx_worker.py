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
    """Thin back-compat wrapper around the canonical helper.

    The real implementation lives in :func:`alphagrad.approx.common.cache.setup_jax_compile_cache`
    so every trainer (single-process ppo/mu0, Ray PPO worker, CPU
    approx worker) hits the same per-SLURM-job, per-node cache and
    we can fix the cross-node AOT-loader contamination in one place.
    """
    from alphagrad.approx.common.cache import setup_jax_compile_cache

    setup_jax_compile_cache()


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
        # Per-actor leak profiler. Lazy-initialised on first call to
        # ``evaluate`` so we don't pay the import cost when profiling
        # is disabled. Gated by env-var ``ALPHAGRAD_LEAK_PROFILE`` so
        # production runs aren't taxed.
        self._n_calls = 0
        self._leak_profile = None  # set by `_maybe_init_leak_profile`
        # One-shot env-var banner: prints which ALPHAGRAD_* /
        # JAX_COMPILATION_* keys the actor's runtime_env actually
        # received. Lets us catch driver-side passthrough bugs (e.g.
        # an ``ALPHAGRAD_DEBUG_QUALITY`` set in sbatch that never
        # reaches the actor) without grepping by hand.
        _alpha_keys = sorted(
            (k, v) for k, v in os.environ.items()
            if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_")
        )
        if _alpha_keys:
            print(
                f"[cpu_approx_worker pid={os.getpid()}] env: "
                + " ".join(f"{k}={v}" for k, v in _alpha_keys),
                flush=True,
            )

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
        server = cls(env)
        # Stash for set_cost_mode_full — we need to rebuild target_fn
        # without shipping a Callable through Ray, so reconstruct from
        # the example name at swap time.
        server._args_dict = dict(args_dict)
        return server

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
        point_idx: int = -1,
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
        # Optional per-actor leak profile. Off by default — enable with
        # ``ALPHAGRAD_LEAK_PROFILE=1``. Logs each `evaluate` call's RSS
        # so we can attribute the leak to *this* process (not the
        # GPU trainer that dispatched the request via Ray).
        self._maybe_init_leak_profile()
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
                point_idx=int(point_idx),
            )
            self._n_calls += 1
            if self._leak_profile is not None:
                self._leak_profile.record_call(self._n_calls)
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

    def set_cost_mode_full(self) -> bool:
        """Switch from phase-1 cheap (target_fun=None) to phase-2 full
        (target_fun=target_fn). Idempotent — calling when already in full
        mode is a no-op.

        Implementation: rebuild the target_fn from the cached args_dict
        (cheaper than shipping a Callable through Ray), swap the env's
        ``config.target_fun`` via ``eqx.tree_at``, and replace this
        server's stored env with the new one. JAX compile caches are
        keyed by (order, specs, shape) and survive the swap.
        """
        if self._config.target_fun is not None:
            return True  # already in full mode
        import equinox as eqx
        from alphagrad.approx.common import get_fn
        from alphagrad.approx.env import EnvConfig

        # Rebuild target_fn locally — Ray actor holds the args_dict from
        # construction.
        args_dict = getattr(self, "_args_dict", None)
        if args_dict is None:
            return False  # paranoid; nothing to rebuild from
        target_fn = get_fn(args_dict["example"])
        # Match the grad-mode scalar-loss wrapping used at env-build time so the
        # swapped-in target stays the same scalar-loss graph (shared wrap).
        from alphagrad.approx.common import maybe_scalar_loss
        target_fn, _ = maybe_scalar_loss(args_dict, target_fn)
        new_config = self._config._replace(target_fun=target_fn)
        # Swap on the env via eqx.tree_at so the JAX-side state survives.
        self._env = eqx.tree_at(lambda e: e.config, self._env, new_config)
        self._config = new_config
        return True

    def _maybe_init_leak_profile(self) -> None:
        """Activate the per-actor RSS / tracemalloc profiler on first
        ``evaluate`` if ``ALPHAGRAD_LEAK_PROFILE=1`` is set."""
        if self._leak_profile is not None:
            return
        from alphagrad.approx.common.leak_profile import maybe_install
        self._leak_profile = maybe_install("actor")


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
    # Gradient mode: THIS env (inside the CpuApproximationActor) does the actual
    # pooled measurement, so the scalar-loss wrapping + jaxpr must mirror the
    # trainer exactly, or grad-mode runs would silently measure the Jacobian.
    from alphagrad.approx.common import maybe_scalar_loss
    target_fn, measure_grad = maybe_scalar_loss(args, target_fn)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    argnums = infer_argnums(args.example)

    # Always pass target_fun so env._callback runs the JIT-compile +
    # cost_analysis + ResourceMonitor path on every step — this is what
    # populates flops, bytes_accessed, latency_ns, peak_memory. Previously
    # this was gated on ``"acc" in args.rewards`` (skip exec when only
    # symbolic counters are needed) but that left 4 of 6 cost channels at
    # 0 in best_sequences.json / wandb, which broke downstream comparison.
    # The cossim/frob comparison work (the *expensive* part beyond bare
    # exec) is still skipped on non-terminal steps via the ``is_terminal``
    # guard inside _callback, so the per-step extra cost is bounded by
    # one JIT-exec (or 10× with --measure-latency).
    #
    # 2-phase schedule (``--cost-pipeline-schedule cheap_first``): start
    # with target_fun=None (cheap-only — graphax symbolic counts only,
    # no JIT-exec). The actor's ``set_cost_mode_full`` method swaps it
    # to ``target_fn`` after the policy stabilises (KL-or-ep cutover).
    schedule = str(getattr(args, "cost_pipeline_schedule", "always_full"))
    env_target_fun = None if schedule == "cheap_first" else target_fn
    measure_latency = bool(
        getattr(args, "measure_latency", False)
        or getattr(args, "cmp_type", "") == "latency"
    )
    # ``--terminal-rewards-only`` was renamed to ``--intermediate-rewards``
    # (BooleanOptionalAction, default False). The trainer's args_dict now
    # carries ``intermediate_rewards``; derive terminal_rewards_only the
    # same way the PPO worker does. Fall back to the legacy key for any
    # caller that still sets it directly.
    if hasattr(args, "intermediate_rewards"):
        terminal_rewards_only = not bool(args.intermediate_rewards)
    else:
        terminal_rewards_only = bool(getattr(args, "terminal_rewards_only", False))
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
        latency_samples=int(getattr(args, "latency_samples", 1)),
        # New noisy-channel measurement params — must mirror the trainer's
        # CLI, otherwise the CPU actor (which does the actual measurement)
        # silently falls back to the EnvConfig defaults (5×4=20 measures).
        num_data_points=int(getattr(args, "num_data_points", 5)),
        reps_per_point=int(getattr(args, "reps_per_point", 4)),
        percentile_keep=float(getattr(args, "percentile_keep", 0.60)),
        # Latency-measurement knobs — THIS env (inside the CpuApproximationActor)
        # does the actual pooled measurement, so the flags must be forwarded
        # here or --latency-winsor/--latency-inner-reps/--latency-warmup are
        # silently inert in every pooled Ray run.
        latency_inner_reps=int(getattr(args, "latency_inner_reps", 1)),
        latency_warmup=int(getattr(args, "latency_warmup", 0)),
        latency_winsor=float(getattr(args, "latency_winsor", 0.0)),
        measure_grad=measure_grad,
        latency_timer=str(getattr(args, "latency_timer", "perf_counter")),
        slow_exec_cutoff_seconds=float(
            getattr(args, "slow_exec_cutoff_seconds", 15.0)
        ),
        flop_gate_threshold=float(getattr(args, "flop_gate_threshold", 0.0)),
        terminal_rewards_only=terminal_rewards_only,
    )

    num_eval = int(getattr(args, "num_eval_samples", 10) or 10)
    eval_samples = generate_eval_samples(env, eval_key, num_eval)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
