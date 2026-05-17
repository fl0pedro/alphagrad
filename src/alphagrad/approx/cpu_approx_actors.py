"""Ray-remote wrapper around :class:`CpuApproximationServer`.

Kept separate from `cpu_approx_worker.py` so the underlying server
class stays importable without Ray (the in-process parity tests in
`tests/test_cpu_approx_worker.py` exercise the server directly). This
file is the only place ray is imported at module top level.

Wire-up pattern from the driver (mirrors `mu0_ray.py`'s SPMD actor):

    cpu_workers = [
        CpuApproximationActor.options(num_cpus=1).remote(
            args_dict, variant=variant, actor_id=i,
        )
        for i in range(num_cpu_workers)
    ]
    ray.get([w.ready.remote() for w in cpu_workers])
    # ... per env step (Python loop in driver):
    results = ray.get([
        w.evaluate.remote(order_np, specs_np, step) for w, ... in ...
    ])
    # ... periodically (drains the cost_analysis C++ leak):
    if ep % args.cpu_worker_recycle_every == 0:
        for w in cpu_workers:
            ray.kill(w)
        cpu_workers = [... respawn ...]
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import ray


@ray.remote
class CpuApproximationActor:
    """Owns one `CpuApproximationServer` per Ray actor.

    Constructed with `args_dict` + optional `variant`, matching the
    shape `mu0_ray.py` already uses for `CPUApproximationActor`. The
    actor spins up the env (and its eval_args_samples) once at
    `__init__`; every `evaluate` call afterwards just dispatches to the
    server's pure-Python `evaluate` method.

    `actor_id` is plumbed through for parity with the existing stub and
    for log-tagging; the server doesn't read it.
    """

    def __init__(
        self,
        args_dict: dict,
        variant: str | None = None,
        actor_id: int = 0,
        seed: int = 0,
    ):
        # Deferred import — keeps the driver process JAX-free until a
        # worker actor actually starts. Mirrors the
        # `mu0_ray_actors.SPMDActor` pattern.
        from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

        self._actor_id = int(actor_id)
        self._impl = CpuApproximationServer.from_args_dict(
            args_dict, variant=variant, seed=seed,
        )

    def evaluate(
        self,
        order: np.ndarray,
        sparsity_specs: np.ndarray,
        step: int,
        eval_samples: Sequence | None = None,
        init: bool = False,
    ):
        return self._impl.evaluate(
            order, sparsity_specs, step,
            eval_samples=eval_samples, init=init,
        )

    def evaluate_batch(self, batch: Sequence[tuple]):
        return self._impl.evaluate_batch(batch)

    def reset_caches(self) -> dict:
        return self._impl.reset_caches()

    def actor_id(self) -> int:
        return self._actor_id

    def ready(self) -> bool:
        return self._impl.ready()

    def compile_approximations(self) -> dict:
        """Pool warm-up handshake.

        The underlying :class:`CpuApproximationServer` is constructed in
        ``__init__`` (env build + JAX cache wiring), so this is mostly a
        handshake — but we expose it for parity with the MuZero pool
        (`mu0_ray_worker.CPUApproximationWorker.compile_approximations`)
        so the shared `run_calibration` helper can `ray.get` a warm-up
        future on either trainer before issuing zero-pref rollouts.
        """
        return {"status": "ready", "actor_id": self._actor_id}

    def consume_tokenization_truncation_stats(self) -> dict:
        """Pop the per-actor jaxpr-tokenization truncation counters.

        Returns ``{"count": int, "max_observed_len": int}`` and resets
        the per-process counters in ``env.py`` so subsequent rollouts
        report only their own deltas. Pool aggregates across actors.
        """
        from alphagrad.approx.env import (
            consume_tokenization_truncation_stats as _consume,
        )
        return _consume()
