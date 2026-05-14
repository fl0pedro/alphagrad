"""JAX-free Ray actor shims for :mod:`alphagrad.approx.mu0_ray_worker`.

The driver (:mod:`alphagrad.approx.mu0_ray`) imports actor classes from
this module so it can spawn Ray actors without itself importing JAX.
The JAX-heavy worker module is imported lazily inside each actor's
``__init__`` — Ray executes that code in a freshly-forked worker
process, so JAX initialisation never happens in the driver. (Mixing
Ray's actor-spawn machinery with JAX in the same process has known
fork/CUDA deadlock failure modes; this shim is the boundary that keeps
them separated.)

Every method is a thin forward to the corresponding ``LearnerWorker`` /
``RolloutWorker`` method. The actors accept and return numpy /
nested-dict data only — no JAX arrays cross the Ray boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray


@ray.remote
class LearnerActor:
    """Ray actor wrapping :class:`mu0_ray_worker.LearnerWorker`."""

    def __init__(self, args_dict: dict, variant: str, learner_seed: int = 0):
        # JAX import deferred to inside the actor process.
        from alphagrad.approx.mu0_ray_worker import LearnerWorker
        self._impl = LearnerWorker(args_dict, variant, learner_seed)

    def add_trajectories(self, traj_np: dict) -> None:
        return self._impl.add_trajectories(traj_np)

    def train_step(self) -> dict:
        return self._impl.train_step()

    def get_params_numpy(self) -> list:
        return self._impl.get_params_numpy()

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        return self._impl.set_reward_weights(weights_np)

    def checkpoint_replay(self, path: str) -> None:
        return self._impl.checkpoint_replay(path)

    def get_stats(self) -> dict:
        return self._impl.get_stats()

    def ready(self) -> bool:
        return self._impl.ready()


@ray.remote
class RolloutActor:
    """Ray actor wrapping :class:`mu0_ray_worker.RolloutWorker`."""

    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        from alphagrad.approx.mu0_ray_worker import RolloutWorker
        self._impl = RolloutWorker(args_dict, variant, actor_id)

    def set_params_numpy(self, params_np: list) -> None:
        return self._impl.set_params_numpy(params_np)

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        return self._impl.set_reward_weights(weights_np)

    def rollout_one(
        self,
        rng_seed: int,
        preference_np: Any = None,
        pin_rules: Any = None,
        reset_env: bool = True,
    ) -> tuple[dict, dict]:
        return self._impl.rollout_one(
            rng_seed=rng_seed,
            preference_np=preference_np,
            pin_rules=pin_rules,
            reset_env=reset_env,
        )

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> np.ndarray:
        return self._impl.reward_vec_means(rng_seed, num_rollouts)

    def ready(self) -> bool:
        return self._impl.ready()
