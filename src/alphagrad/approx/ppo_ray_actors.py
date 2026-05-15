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

    def run_rollout_and_train(self, rng_seed: int) -> dict:
        return self._impl.run_rollout_and_train(int(rng_seed))

    def ready(self) -> bool:
        if self._impl is None:
            return False
        return self._impl.ready()
