"""Ray-remote wrappers around :class:`PPORayWorker`.

Mirrors the `mu0_ray_actors.SPMDActor` shape so the driver in
`ppo_ray.py` can spawn this actor / the CPU worker pool with the same
options + ready-handshake pattern as `mu0_ray.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray


@ray.remote
class PPOActor:
    """One JAX trainer per process. Wraps :class:`PPORayWorker`."""

    def __init__(self, args_dict: dict, seed: int = 0):
        self.args_dict = args_dict
        self.seed = int(seed)
        self._impl = None  # built when `init_worker` is called with cpu_workers

    def init_worker(self, cpu_workers: list | None = None) -> bool:
        from alphagrad.approx.ppo_ray_worker import PPORayWorker

        self._impl = PPORayWorker(
            self.args_dict, seed=self.seed, cpu_workers=cpu_workers,
        )
        return True

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
        """Construct the timeout-bounded CPU pool. Mirrors
        ``mu0_ray_actors.SPMDActor.init_server`` so the driver can
        spawn either trainer with the same wire-up dance.

        Calls ``init_worker(cpu_workers)`` first if no worker has been
        built yet (allows the caller to skip the separate
        ``init_worker`` call).
        """
        if self._impl is None:
            self.init_worker(cpu_workers)
        return self._impl.init_server(
            cpu_workers,
            callback_timeout_s=callback_timeout_s,
            initial_timeout_s=initial_timeout_s,
            warm_after=warm_after,
            recycle_every=recycle_every,
            cpu_actor_options=cpu_actor_options,
            starting_actor_id=starting_actor_id,
        )

    def run_rollout_and_train(self, rng_seed: int) -> dict:
        return self._impl.run_rollout_and_train(int(rng_seed))

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Calibration entry point — see PPORayWorker.reward_vec_means
        for the returned per-channel stats dict schema (mean / median /
        quartiles in raw + symlog space)."""
        return self._impl.reward_vec_means(int(rng_seed), int(num_rollouts))

    def set_reward_weights(self, weights_np) -> None:
        """Calibration sink — push rescaled weights back into the worker."""
        return self._impl.set_reward_weights(weights_np)

    def set_variant_masks(self, variant: str) -> dict:
        """Curriculum stage transition — see
        ``PPORayWorker.set_variant_masks`` for the contract. Driver
        calls this between episodes when ``--curriculum`` is set."""
        return self._impl.set_variant_masks(str(variant))

    def ready(self) -> bool:
        if self._impl is None:
            return False
        return self._impl.ready()
