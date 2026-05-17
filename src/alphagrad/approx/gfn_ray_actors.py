"""Ray actor wrappers for the SEED-style off-policy GFlowNet trainer.

Two actor classes:

* :class:`GFNSPMDActor` — centralised JAX SPMD trainer. Owns the
  GFN agent, the JAX mesh, the replay buffer, and the host-side
  :class:`alphagrad.approx.cpu_approx_pool.CpuApproxPool` that the
  env's ``io_callback`` dispatches to. Surface mirrors
  :class:`alphagrad.approx.mu0_ray_actors.SPMDActor` 1:1 so the driver
  loop in ``gfn_ray.py`` is structurally identical to ``mu0_ray.py``.

* ``CPUApproximationActor`` — re-exported from
  :mod:`alphagrad.approx.mu0_ray_actors`. The CPU-side approximation
  actor is algorithm-agnostic (it wraps
  :class:`alphagrad.approx.cpu_approx_worker.CpuApproximationServer`
  which only depends on the env / variant / args_dict, not on the RL
  algorithm), so we re-use it verbatim instead of duplicating the
  class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray

# Re-export the CPU-side actor unchanged. Driver code can keep importing
# ``CPUApproximationActor`` from this module the same way ``mu0_ray.py``
# imports it from ``mu0_ray_actors``.
from alphagrad.approx.mu0_ray_actors import CPUApproximationActor  # noqa: F401


@ray.remote
class GFNSPMDActor:
    """Centralised JAX SPMD actor for SEED-style off-policy GFN training.

    Holds the GFN agent, optimiser state, JAX mesh, replay buffer, and
    the host-side :class:`CpuApproxPool` that the env's ``io_callback``
    dispatches per-step jacve/tokenize work to. All methods are thin
    forwarders to :class:`alphagrad.approx.gfn_ray_worker.GFNServerWorker`.
    """

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
        """Construct the SPMD-side training state and wire the CPU
        approx pool into the env's ``io_callback``.

        ``initial_timeout_s`` / ``warm_after`` enable a cold-cache
        budget — first ``warm_after`` calls per actor get a longer
        timeout before each actor switches to ``callback_timeout_s``.
        Default ``initial_timeout_s=None`` uses ``callback_timeout_s``
        for all calls (legacy behaviour).
        """
        from alphagrad.approx.gfn_ray_worker import GFNServerWorker

        self._impl = GFNServerWorker(
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
        """Per-channel calibration statistics over ``num_rollouts``
        zero-pref rollouts. See ``mu0_ray_actors.SPMDActor`` for the
        dict schema."""
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
