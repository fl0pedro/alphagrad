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

    def init_server(self, cpu_workers):
        from alphagrad.approx.mu0_ray_worker import SPMDServerWorker

        self._impl = SPMDServerWorker(
            self.args_dict, self.variant, self.seed, cpu_workers=cpu_workers
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

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> np.ndarray:
        return self._impl.reward_vec_means(rng_seed, num_rollouts)

    def set_reward_weights(self, weights_np: np.ndarray) -> None:
        return self._impl.set_reward_weights(weights_np)

    def checkpoint_replay(self, path: str) -> None:
        return self._impl.checkpoint_replay(path)

    def ready(self) -> bool:
        if self._impl is None:
            return False
        return self._impl.ready()


@ray.remote
class CPUApproximationActor:
    """CPU-only worker for compiling variants and generating dataset approximations."""

    def __init__(self, args_dict: dict, variant: str, actor_id: int):
        from alphagrad.approx.mu0_ray_worker import CPUApproximationWorker

        self._impl = CPUApproximationWorker(args_dict, variant, actor_id)

    def compile_approximations(self) -> dict:
        return self._impl.compile_approximations()

    def evaluate_graph(self, o_list, transforms, eval_samples):
        return self._impl.evaluate_graph(o_list, transforms, eval_samples)

    def ready(self) -> bool:
        return self._impl.ready()
