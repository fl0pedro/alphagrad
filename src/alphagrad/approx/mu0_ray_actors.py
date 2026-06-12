from __future__ import annotations

from typing import Any

import numpy as np
import ray


@ray.remote
class SPMDActor:
    """Centralized JAX SPMD Actor that handles natively sharded MCTS and Training."""

    def __init__(self, args_dict: dict, variant: str, seed: int = 0):
        self.args_dict = args_dict
        self.variant = variant
        self.seed = seed
        self._impl = None

    def init_server(
        self,
        cpu_workers,
        *,
        callback_timeout_s: float = 120.0,
        initial_timeout_s: float | None = None,
        warm_after: int = 3,
        recycle_every: int = 50,
        cpu_actor_options: dict | None = None,
        starting_actor_id: int = 1000,
    ):
        """Construct the SPMD-side training state and wire the
        CPU approx pool into the env's ``io_callback``.

        ``initial_timeout_s`` / ``warm_after`` enable a cold-cache
        budget — first ``warm_after`` calls per actor get a longer
        timeout before each actor switches to ``callback_timeout_s``.
        Default ``initial_timeout_s=None`` uses ``callback_timeout_s``
        for all calls (legacy behaviour).
        """
        from alphagrad.approx.mu0_ray_worker import SPMDServerWorker

        self._impl = SPMDServerWorker(
            self.args_dict,
            self.variant,
            self.seed,
            cpu_workers=cpu_workers,
            callback_timeout_s=callback_timeout_s,
            initial_timeout_s=initial_timeout_s,
            warm_after=warm_after,
            recycle_every=recycle_every,
            cpu_actor_options=cpu_actor_options,
            starting_actor_id=starting_actor_id,
        )
        return True

    def run_rollout_and_train(
        self,
        rng_seed: int,
        preference_np: Any = None,
        pin_rules: Any = None,
        reset_env: bool = True,
        train_steps: int = 1,
    ) -> dict:
        return self._impl.run_rollout_and_train(
            rng_seed=rng_seed,
            preference_np=preference_np,
            pin_rules=pin_rules,
            reset_env=reset_env,
            train_steps=train_steps,
        )

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """See ``CPUApproximationWorker.reward_vec_means`` for the dict
        schema. Returns per-channel mean + median + quartile stats."""
        return self._impl.reward_vec_means(rng_seed, num_rollouts)

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        return self._impl.set_reward_weights(weights_np)

    def checkpoint_replay(self, path: str) -> None:
        return self._impl.checkpoint_replay(path)

    def get_pool_stats(self) -> dict:
        """Forward telemetry from the CPU-approx pool to the driver."""
        if self._impl is None:
            return {}
        return self._impl.get_pool_stats()

    def ready(self) -> bool:
        if self._impl is None:
            return False
        return self._impl.ready()


@ray.remote
class CPUApproximationActor:
    """CPU-only Ray actor wrapping :class:`CPUApproximationWorker`.

    Spawned in pools by :func:`alphagrad.approx.mu0_ray._run_one_variant`
    and consumed by the env's ``io_callback`` via
    :class:`alphagrad.approx.cpu_approx_pool.CpuApproxPool`. Each call
    runs one ``jax.jit(jacve(...)).lower().compile()`` + execution
    out-of-process so the SPMD actor (which owns the GPUs) never
    blocks on XLA's uninterruptible C++ compile.
    """

    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        from alphagrad.approx.mu0_ray_worker import CPUApproximationWorker

        self._impl = CPUApproximationWorker(args_dict, variant, actor_id)

    def compile_approximations(self) -> dict:
        return self._impl.compile_approximations()

    def evaluate(
        self,
        order,
        sparsity_specs,
        step,
        eval_samples=None,
        init: bool = False,
    ):
        """Forward to ``CPUApproximationWorker.evaluate``.

        Explicit method (not ``__getattr__``-based) so Ray's method
        lookup is unambiguous and the actor signature is stable when
        we add cancellation / timing later.
        """
        return self._impl.evaluate(
            order, sparsity_specs, step,
            eval_samples=eval_samples, init=init,
        )

    def reset_caches(self) -> dict:
        return self._impl.reset_caches()

    def ready(self) -> bool:
        return self._impl.ready()

    def consume_tokenization_truncation_stats(self) -> dict:
        """Pop per-process truncation counters; see PPO sibling for the
        contract."""
        from alphagrad.approx.env import (
            consume_tokenization_truncation_stats as _consume,
        )
        return _consume()
