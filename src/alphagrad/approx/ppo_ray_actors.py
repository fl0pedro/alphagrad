"""DEPRECATED — DOES NOT RUN against this tree.

Verified by two independent audits (2026-07-26): the Ray line dies before the
first update with at least five independent failures — the 8-vs-10 reward
channel skew (PPORayWorker.__init__ indexes channel 9 of an 8-vector), the
unported FactoredQuantHead (`quant_dtype_head.proj` no longer exists; MicroAction
needs `quant_scale_sign`), and 100% sentinel measurements (`_callback` does not
accept the `point_idx=` this line always passes, so every measurement is
swallowed into -1e10). The advertised async pipeline cannot start at all
(no `--p3o` flag; its learner `train_on_trajs` was removed but is still called).

KEPT AS A DESIGN REFERENCE ONLY. The parts worth reading are catalogued in
alphagrad/COMPONENTS.md; the ones worth having have been ported to ppo.py
(PopArt, Pareto + hypervolume, the multiplicative cosine gate, per-component KL).
Use `ppo.py`.
"""

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
        # Apply ALPHAGRAD_* policy switches threaded via args_dict BEFORE the
        # policy module (heads) is imported/traced, so ALPHAGRAD_QUANT_ALLOWED /
        # ALPHAGRAD_SUBSTEP_NO_END actually take effect in this actor (runtime_env
        # delivery is unreliable at heads-import time). heads also reads lazily.
        import os
        os.environ.update(self.args_dict.get("_alphagrad_env", {}) or {})
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

    # ---- Async pipeline (Stage 1) Ray entrypoints ----
    def get_weights(self):
        return self._impl.get_weights()

    def set_weights(self, weights_np, learner_step: int = 0) -> int:
        return int(self._impl.set_weights(weights_np, int(learner_step)))

    def set_compile_actor(self, compile_actor) -> bool:
        return bool(self._impl.set_compile_actor(compile_actor))

    def collect_traj(self, rng_seed: int):
        return self._impl.collect_traj(int(rng_seed))

    def train_on_trajs(self, trajs, synced_learner_steps, n_updates: int = 1):
        return self._impl.train_on_trajs(
            list(trajs), list(synced_learner_steps), int(n_updates),
        )

    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Calibration entry point — see PPORayWorker.reward_vec_means
        for the returned per-channel stats dict schema (mean / median /
        quartiles in raw + symlog space)."""
        return self._impl.reward_vec_means(int(rng_seed), int(num_rollouts))

    def set_reward_weights(self, weights_np) -> None:
        """Calibration sink — push rescaled weights back into the worker."""
        return self._impl.set_reward_weights(weights_np)

    def ready(self) -> bool:
        if self._impl is None:
            return False
        return self._impl.ready()

    def pool_stats(self) -> dict:
        """Cumulative CPU-pool counters (calls / timeouts / errors).

        Used by the driver's end-of-run summary to emit the loud
        ``timeouts total: N (dead code if 0)`` line. Returns ``{}`` when
        the worker / pool aren't initialised so the driver can fall back
        cleanly."""
        if self._impl is None or getattr(self._impl, "_cpu_pool", None) is None:
            return {}
        return self._impl._cpu_pool.stats()
