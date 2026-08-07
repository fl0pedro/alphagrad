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


@ray.remote(num_cpus=0)
class _CoreAllocator:
    """Cluster-wide, per-node disjoint CPU-core dispenser.

    With multiple PPO jobs sharing one Ray cluster (the 2-node
    pooled-measurement setup), each job's CpuApproximationActors would
    otherwise independently compute the SAME affinity slice
    (actor_id 0 → cores 0..k) and collide — N-fold oversubscription, the
    exact pathology the affinity pinning exists to prevent. This named
    actor (lifetime=detached, one per cluster) tracks how many cores are
    already claimed PER NODE and hands the next disjoint slice to whoever
    asks — independent of which job the requester belongs to. Robust to
    any number of concurrent jobs/actors.
    """

    def __init__(self):
        self._claimed: dict[str, int] = {}  # node_id -> next free core index
        self._by_key: dict[str, list[int]] = {}  # claim_key -> cores (idempotent)

    def claim(
        self, node_id: str, ncores_total: int, n: int, claim_key: str | None = None,
    ) -> list[int]:
        """Claim ``n`` consecutive core indices on ``node_id``. Wraps
        modulo ``ncores_total`` if the node is over-subscribed (more
        actors than cores/n) — degrades to sharing rather than erroring.

        ``claim_key`` makes the claim IDEMPOTENT: a re-claim with the same
        key (e.g. the same logical actor respawned after a recycle/timeout
        kill) returns its existing slice instead of advancing the per-node
        counter — otherwise respawns leak slots forever and eventually wrap
        onto cores still pinned by live actors.
        """
        if claim_key is not None and claim_key in self._by_key:
            return self._by_key[claim_key]
        start = self._claimed.get(node_id, 0)
        cores = [(start + k) % ncores_total for k in range(n)]
        self._claimed[node_id] = (start + n) % ncores_total
        if claim_key is not None:
            self._by_key[claim_key] = cores
        return cores

    def reset(self) -> None:
        self._claimed.clear()
        self._by_key.clear()


def _get_core_allocator():
    """Fetch-or-create the cluster's core allocator named actor."""
    try:
        return ray.get_actor("dsnn_core_allocator")
    except Exception:
        try:
            return _CoreAllocator.options(
                name="dsnn_core_allocator", lifetime="detached",
                get_if_exists=True,
            ).remote()
        except Exception:
            return None


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
        self._actor_id = int(actor_id)

        # CPU-affinity pinning to a disjoint core SLICE per actor. The
        # approx-Jacobian exec scales ~linearly with cores, so each actor
        # needs MANY cores — but N actors each defaulting to all 64 cores
        # oversubscribe catastrophically (~48s/exec vs 0.45s). Pinning
        # each actor to ``cores // n_workers`` disjoint cores *before* the
        # jax import sizes its Eigen pool to that slice: multi-core execs,
        # no cross-actor oversubscription, full utilisation. Profiled
        # 2026-06-06. (Setting affinity before the deferred jax import is
        # what makes XLA size the pool to the slice rather than all cores.)
        # Best-effort: skipped on platforms without sched_setaffinity.
        try:
            import os as _os
            n_workers = max(int(args_dict.get("num_cpu_workers", 1) or 1), 1)
            avail = sorted(_os.sched_getaffinity(0))
            ncores = len(avail)
            reserved = int(args_dict.get("reserved_driver_cores", 0) or 0)
            reserved = max(0, min(reserved, ncores - n_workers))
            pool = avail[reserved:] if reserved < ncores else avail
            npool = len(pool)
            # ``cpu_cores_per_actor > 0`` forces an exact slice width (e.g. 1
            # for single-core-per-actor measurement: cleanest per-reading CV,
            # at the cost of single-threaded exec — pair with
            # num_cpu_workers ≈ #cores so every core hosts one actor and total
            # throughput is recovered by cross-actor parallelism). 0 = the
            # legacy auto slice ``npool // n_workers``.
            _cpa = int(args_dict.get("cpu_cores_per_actor", 0) or 0)
            per = _cpa if _cpa > 0 else max(1, npool // n_workers)
            # Disjoint slice. When multiple jobs share the cluster
            # (--cpu-cores-shared), claim from the per-node allocator so
            # the 4 jobs' actors don't collide on the same cores. Else
            # use the single-job static slice (actor_id * per).
            base = (self._actor_id * per) % npool
            if args_dict.get("cpu_cores_shared", False):
                try:
                    import socket
                    node_id = socket.gethostname()
                    alloc = _get_core_allocator()
                    if alloc is not None:
                        # Idempotent per (run, actor) so a respawn reuses its
                        # slice instead of leaking a fresh one.
                        claim_key = f"{args_dict.get('name', '')}:{self._actor_id}"
                        base = int(ray.get(
                            alloc.claim.remote(node_id, npool, per, claim_key)
                        )[0])
                except Exception as _exc:
                    # Falling back to the static base risks the cross-job
                    # collision the allocator exists to prevent — surface it.
                    print(
                        f"[measure-actor {self._actor_id}] core-allocator "
                        f"claim failed ({_exc}); using static slice base={base}",
                        flush=True,
                    )
            if 0 < per < ncores:
                cores = {pool[(base + k) % npool] for k in range(per)}
                _os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, ValueError):
            pass

        # Deferred import — keeps the driver process JAX-free until a
        # worker actor actually starts. Mirrors the
        # `mu0_ray_actors.SPMDActor` pattern.
        from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

        self._impl = CpuApproximationServer.from_args_dict(
            args_dict, variant=variant, seed=seed,
        )
        # Node/GPU tracking: log where this measurement actor landed.
        try:
            import socket as _sock, os as _os2, jax as _jax
            print(
                f"[measure-actor {self._actor_id}] host={_sock.gethostname()} "
                f"CUDA_VISIBLE_DEVICES={_os2.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"jax_devices={_jax.devices()}", flush=True,
            )
        except Exception:
            pass

    def evaluate(
        self,
        order: np.ndarray,
        sparsity_specs: np.ndarray,
        step: int,
        eval_samples: Sequence | None = None,
        init: bool = False,
        point_idx: int = -1,
        face_specs=None,
        face_skips=None,
    ):
        return self._impl.evaluate(
            order, sparsity_specs, step,
            eval_samples=eval_samples, init=init, point_idx=point_idx,
            face_specs=face_specs, face_skips=face_skips,
        )

    def evaluate_batch(self, batch: Sequence[tuple]):
        return self._impl.evaluate_batch(batch)

    def precompile(self, order, sparsity_specs, step: int) -> bool:
        """STAGE-2 async compile-actor entrypoint — warm the shared cache for
        this order (compile-only, no measure). See
        ``CpuApproximationServer.precompile``."""
        return bool(self._impl.precompile(order, sparsity_specs, int(step)))

    def reset_caches(self) -> dict:
        return self._impl.reset_caches()

    def pop_oom_flag(self) -> bool:
        """Return-and-reset whether the most recent ``evaluate`` OOM-ed.
        Ray-remote wrapper; see CpuApproximationServer.pop_oom_flag. Drives
        CpuApproxPool's recycle+retry-on-OOM path."""
        return bool(self._impl.pop_oom_flag())

    def actor_id(self) -> int:
        return self._actor_id

    def ready(self) -> bool:
        return self._impl.ready()

    def set_cost_mode_full(self) -> bool:
        """Phase-2 cutover: swap target_fun=None → target_fun=target_fn
        so subsequent ``evaluate`` calls run the full cost-channel path
        (XLA cost_analysis + ResourceMonitor + compiled_exact at terminal).

        Idempotent. See CpuApproximationServer.set_cost_mode_full for
        implementation details — this is the Ray-remote wrapper."""
        return bool(self._impl.set_cost_mode_full())

    def compile_approximations(self) -> dict:
        """Pool warm-up handshake.

        The underlying :class:`CpuApproximationServer` is constructed in
        ``__init__`` (env build + JAX cache wiring), so this is mostly a
        handshake — but we expose it for parity with the MuZero pool
        (`mu0_ray_worker.CPUApproximationWorker.compile_approximations`)
        so either trainer's driver can `ray.get` a warm-up future
        before issuing the first rollouts.
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

    def consume_face_stats(self) -> dict:
        """Pop this actor's per-face apply counters.

        The hooks that decide per-face legality run HERE (the measurement
        ``_callback`` executes inside the actor process, not the trainer),
        so ``env._PER_FACE_STATS`` only ever fills up in this process; the
        trainer's own dict is always empty while the pool is active.
        """
        from alphagrad.approx.env import consume_per_face_stats as _consume_pf
        return _consume_pf()

    def consume_collapse_stats(self) -> dict:
        """Pop this actor's truncation / collapse counters (#81, #96).

        Same reason as :meth:`consume_face_stats`: these are module
        globals written inside the measurement ``_callback``, which runs
        in THIS process. The trainer reads its own copy, which is always
        zero while the pool is active -- so the panels read 0 while the
        clipping/collapse they count is actually happening.
        """
        from alphagrad.approx import env as _e
        out = {}
        try:
            _t = _e.consume_tokenization_truncation_stats()
            out["trunc_count"] = int(_t.get("count", 0))
            out["trunc_overflow_sum"] = int(_t.get("overflow_sum", 0))
            out["trunc_max_observed_len"] = int(
                _t.get("max_observed_len", 0))
        except Exception:
            pass
        for _key, _fn in (("truncated", "consume_truncated_plan_count"),
                          ("untraceable", "consume_untraceable_plan_count"),
                          ("zero_work", "consume_zero_work_plan_count"),
                          ("degenerate", "consume_degenerate_plan_count")):
            try:
                out[_key] = int(getattr(_e, _fn)())
            except Exception:
                pass
        return out
