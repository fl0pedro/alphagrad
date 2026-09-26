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
import os

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
        core_ids: Sequence[int] | None = None,
        slot: int | None = None,
        gpu_uuid: str | None = None,
    ):
        self._actor_id = int(actor_id)
        # What every plan record of this actor is stamped with: the device it
        # saw and the process that timed the plan (dsnn-dfw.229).
        self._slot = None if slot is None else int(slot)
        self._gpu_uuid = None if gpu_uuid is None else str(gpu_uuid)
        self._device = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._pid = os.getpid()

        # AN EXPLICIT SLICE WINS (owner ruling Q3, 2026-09-18). The node budget
        # is computed once in the driver from the launcher constants, so the
        # trainer, the timing actors and the oracle hold disjoint cores; the
        # legacy `npool // n_workers` arithmetic below cannot express that
        # because it knows only its own kind of actor. ABSOLUTE cpu ids: the
        # trainer has already narrowed its own mask when it starts the raylet,
        # so what this process inherited is not the node.
        if core_ids is not None:
            _ids = {int(c) for c in core_ids}
            if not _ids:
                raise ValueError("core_ids must not be empty")
            os.sched_setaffinity(0, _ids)

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
        _legacy_pin = core_ids is None
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
            if _legacy_pin and args_dict.get("cpu_cores_shared", False):
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
            if _legacy_pin and 0 < per < ncores:
                cores = {pool[(base + k) % npool] for k in range(per)}
                _os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, ValueError):
            pass

        # Deferred import — keeps the driver process JAX-free until a
        # worker actor actually starts. Mirrors the
        # `mu0_ray_actors.SPMDActor` pattern.
        from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
        from alphagrad.approx.common.rsnn_shd import temporal_rule_list

        # ONE SERVER PER GRAPH (owner ruling 2026-09-22). A run that
        # alternates between two graphs of one target measures whichever
        # graph the episode is on, so the actor holds both and `evaluate`
        # names one. Both are built HERE, at __init__, and not on first use:
        # a graph built inside an episode's callback would stall the whole
        # rollout behind a trace, and a build that fails must fail at
        # startup where the pool can see it.
        self._rules = temporal_rule_list(args_dict.get("temporal_rule"))
        if len(self._rules) <= 1:
            self._impls = {None: CpuApproximationServer.from_args_dict(
                args_dict, variant=variant, seed=seed)}
        else:
            # THE WIDER FACE BOUND WINS. env.MAX_FACES is process state and
            # the trainer fills ONE face wire, so both graphs must decode it
            # at the same width; each build reports the bound it derived and
            # the second pass installs the larger.
            from alphagrad.approx.env import (
                configure_max_faces as _cfg_faces)
            self._impls = {
                r: CpuApproximationServer.from_args_dict(
                    args_dict, variant=variant, seed=seed, rule=r)
                for r in self._rules
            }
            _bounds = [getattr(s._env, "_derived_max_faces", None)
                       for s in self._impls.values()]
            _bounds = [int(b) for b in _bounds if b is not None]
            if _bounds:
                _cfg_faces(max(_bounds))
                print(f"[measure-actor {self._actor_id}] graphs "
                      f"{list(self._rules)}, face bounds {_bounds}, "
                      f"in force {max(_bounds)}", flush=True)
        self._impl = next(iter(self._impls.values()))
        # Node/GPU tracking: log where this measurement actor landed.
        try:
            import socket as _sock, os as _os2, jax as _jax
            print(
                f"[measure-actor {self._actor_id}] host={_sock.gethostname()} "
                f"pid={self._pid} slot={self._slot} "
                f"CUDA_VISIBLE_DEVICES={_os2.environ.get('CUDA_VISIBLE_DEVICES', '')} "
                f"gpu_uuid={self._gpu_uuid} "
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
        episode: int | None = None,
        env_row: int | None = None,
        rule: str | None = None,
        timeout_s: float | None = None,
    ):
        # ``rule`` IS THE GRAPH THIS PLAN WAS ACTED ON (owner ruling
        # 2026-09-22). A run that alternates sends it on every dispatch and
        # this actor measures that graph; a run with one graph sends None and
        # gets the one server it built. Measuring the other graph would score
        # a plan against a program the policy never saw, which is the one
        # failure this whole per-graph split exists to prevent, so an
        # unknown rule RAISES rather than falling back to the default.
        #
        # ``episode`` IS NOT OPTIONAL AT THE WIRE. 4c4d872 gave
        # ``CpuApproxPool.evaluate`` / ``evaluate_batch`` an ``episode``
        # field (A3's pooled walk rotation) and both forward it to
        # ``actor.evaluate.remote(...)``, but this wrapper -- the only thing
        # between the pool and ``CpuApproximationServer.evaluate``, which
        # HAS accepted ``episode`` all along -- was never given the
        # parameter. Every pooled dispatch therefore died with
        #   TypeError: got an unexpected keyword argument 'episode'
        # the pool sentinelled the row, killed the actor, and the remaining
        # slots came back "pool-drained (actor died)": under --ray-measure
        # NO plan was measured at all, every terminal reward was the
        # degenerate sentinel, and the failure was visible only as
        # [SENTINEL] lines the trainer does not gate on.
        #
        # ``env_row`` IS THE SAME DEFECT, A SECOND TIME. The per-environment
        # step-position draw of 2026-09-16 gave the pool an ``env_rows``
        # field and both dispatch sites forward it as ``env_row=``, and the
        # server has accepted it from the start -- but this wrapper did not.
        # MEASURED, job 66096 on pgi15-gpu19: every pooled dispatch of a
        # three-episode run died with
        #   TypeError: got an unexpected keyword argument 'env_row'
        # and all four terminal plans of every episode came back
        # sentinelled. It is not specific to the recurrent target: the field
        # is passed on EVERY pooled call, so the whole campaign would have
        # measured nothing.
        # ``tests/pool_dispatch_signature_test.py`` now binds the pool's own
        # dispatch keywords against this signature and the server's, so a
        # third one cannot land.
        return self._server(rule).evaluate(
            order, sparsity_specs, step,
            eval_samples=eval_samples, init=init, point_idx=point_idx,
            face_specs=face_specs, face_skips=face_skips,
            episode=episode, env_row=env_row, timeout_s=timeout_s,
        )

    def _server(self, rule: str | None):
        """The server that holds ``rule``'s graph."""
        if len(self._impls) == 1:
            return self._impl
        if rule is None:
            raise ValueError(
                f"measure actor {self._actor_id} holds {sorted(self._impls)} "
                f"and the dispatch named no graph. Picking one would measure "
                f"the wrong graph on half the episodes in silence.")
        impl = self._impls.get(str(rule))
        if impl is None:
            raise ValueError(
                f"measure actor {self._actor_id} was asked to measure the "
                f"{str(rule)!r} graph and holds {sorted(self._impls)}. The "
                f"plan was acted on a graph this actor never built, so "
                f"measuring it here would score it against another program.")
        return impl

    def evaluate_batch(self, batch: Sequence[tuple]):
        return self._impl.evaluate_batch(batch)

    def precompile(self, order, sparsity_specs, step: int) -> bool:
        """STAGE-2 async compile-actor entrypoint — warm the shared cache for
        this order (compile-only, no measure). See
        ``CpuApproximationServer.precompile``."""
        return bool(self._impl.precompile(order, sparsity_specs, int(step)))

    def reset_caches(self) -> dict:
        # EVERY GRAPH THIS ACTOR HOLDS. The retention bound exists to return
        # executables to the device; clearing one graph's caches and keeping
        # the other's would leave half the memory held.
        out = {}
        for impl in self._impls.values():
            out.update(impl.reset_caches() or {})
        return out

    def pop_oom_flag(self) -> bool:
        """Return-and-reset whether the most recent ``evaluate`` OOM-ed.
        Ray-remote wrapper; see CpuApproximationServer.pop_oom_flag. Drives
        CpuApproxPool's recycle+retry-on-OOM path. An OOM on EITHER graph is
        this actor's OOM, and every flag is reset so none is read twice."""
        return bool(sum(int(bool(impl.pop_oom_flag()))
                        for impl in self._impls.values()))

    def actor_id(self) -> int:
        return self._actor_id

    def ready(self) -> bool:
        return all(impl.ready() for impl in self._impls.values())

    def set_cost_mode_full(self) -> bool:
        """Phase-2 cutover: swap target_fun=None → target_fun=target_fn
        so subsequent ``evaluate`` calls run the full cost-channel path
        (XLA cost_analysis + ResourceMonitor + compiled_exact at terminal).

        Idempotent. See CpuApproximationServer.set_cost_mode_full for
        implementation details — this is the Ray-remote wrapper. The cutover
        is the RUN's, so every graph this actor holds cuts over together."""
        return all(bool(impl.set_cost_mode_full())
                   for impl in self._impls.values())

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

    def consume_plan_records(self, flush_episode: bool = True) -> dict:
        """Pop this actor's A6 plan-log records (every terminal plan).

        Same reason as :meth:`consume_face_stats`: the measurement
        ``_callback`` -- and therefore the plan-log recorder -- runs in
        THIS process. Under ``--ray-measure`` the trainer's own list is
        empty for every pooled row, so without this method the plan log
        would silently contain only the trainer-local terminals.

        Returns ``{"records": [...], "dropped": int, "pid": int}``; the
        actor id is stamped by the merger, which is the only side that
        knows it.
        """
        from alphagrad.approx.env import (
            check_mem_parity_complete as _mp_check,
            consume_plan_records as _consume)
        from alphagrad.approx.common.plan_log import (
            stamp_provenance as _stamp)
        out = _consume(flush_episode=flush_episode)
        out["actor_id"] = self._actor_id
        _stamp(out.get("records") or (),
               device={"cuda_visible_devices": getattr(self, "_device", None),
                       "gpu_uuid": getattr(self, "_gpu_uuid", None)},
               actor_id={"pid": getattr(self, "_pid", os.getpid()),
                         "slot": getattr(self, "_slot", None),
                         "actor": self._actor_id})
        # MEMORY PARITY (ticket .49, folded from .32): every plan THIS
        # actor measured must have left a (temp, watermark) record. Checked
        # HERE, in the process that measured; the fault class rides the
        # toolchain-fault escalation, so the pool does not swallow it.
        _mp_check(out.get("mem_parity") or {},
                  f"measure actor {self._actor_id} (pid {os.getpid()})")
        return out

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
        # THE REFUSAL RATE, PER KIND. Flat numeric keys so
        # `merge_pool_collapse_stats` sums them across actors exactly as it
        # sums everything else here. `refused_total` is the numerator of the
        # rate the trainer logs; the per-kind keys say WHY.
        for _k, _v in _e.consume_refused_counts().items():
            out[f"refused_{_k}"] = int(_v)
        return out

    def consume_call_telemetry(self) -> dict:
        # The pool calls this right after each terminal call, so a kill or a recycle takes nothing with it (dsnn-dfw.201).
        return {"plan": self.consume_plan_records(flush_episode=False),
                "collapse": self.consume_collapse_stats(),
                "face": self.consume_face_stats()}
