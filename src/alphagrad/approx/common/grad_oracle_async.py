"""THE GRADIENT ORACLE, ASYNCHRONOUS AND RETROACTIVE (owner ruling 2026-09-18,
moved off a thread onto a Ray CPU actor by owner ruling 2026-09-18 / dsnn-dfw.22).

THE ORACLE IS A SANITY CHECK, NOT PART OF THE SCORING. It compares the exact
vertex-elimination gradient of an elimination order against ``jax.grad`` of the
same target. Until this module it ran INSIDE the measurement callback, on the
measure actor's GPU, before the plan's quality reference was used -- so its own
float64 compile could fail and REFUSE the plan. On Blackwell it did exactly
that for 15.6 percent of the plans of an oracle-due episode (agent-sentinel
report, 2026-09-18, section 4). The apparatus was scoring the apparatus.

WHAT THIS MODULE IS. One worker OWNS the oracle: a Ray CPU actor
(``num_cpus=4, num_gpus=0``) built with the measurement pool and killed with
it, or -- when nothing built an actor factory for it, which is what every test
in this file does -- one worker THREAD in the trainer process, kept as the
local mode. Either way the trainer hands the worker the distinct elimination
orders of an oracle-due episode and goes on. The check runs on the CPU device
in float64, where every order sits at 1e-14 (agent-df8 report, 2026-09-16).
Results arrive whenever they arrive and are written RETROACTIVELY, as a late
``oracle_result`` record in the append-only plan log and as counters in wandb.

THE TRAINER NEVER WAITS. There is no ticket, no drain, no exclusion of a plan
from the update, and no path by which a slow or failing check can change a
reward, a gradient or the pace of an episode. The only thing the oracle can do
to the run is STOP it: a disagreement in float64 on the CPU is a real graphax
defect and not noise, so the trainer raises at the next episode boundary, after
that episode's checkpoint is written, and the run resumes from that checkpoint
once the defect is fixed.

TIME IS BOUNDED BY THE SUBMITTER, NOT BY THE WORKER. In the ACTOR case, once a
job has been in flight longer than the timeout, :meth:`take_results` kills the
actor (``ray.kill``, ``no_restart=True``) and replaces it with a fresh one from
the same factory, so the CPU quota a hung check held is reclaimed -- the worker
is no longer disowned. Every other job still queued on the killed actor times
out with it, because its answer can no longer arrive. In the THREAD (local)
case a Python thread cannot be interrupted, so a check that runs too long is
not killed: it is DISOWNED, and if it finishes later its answer is dropped.
Either way, once a job is over its timeout, :meth:`take_results` counts it as
``timeout`` and writes its late record; a record for an episode the run has
already passed a decision on would say the check was made in time when it was
not. Nothing blocks either way, so a backlog costs pending count (and, in the
actor case, a kill) and nothing else the run's numbers depend on.

WHY A MEMO. "Once per process and order" is the oracle's own rule: the exact
gradient of an order does not depend on the episode. An order that has been
checked returns its recorded answer at once, so a run whose policy converges on
a few orders pays for each of them once. The memo is keyed on the order alone
-- the approximation rules of a plan are not part of what the oracle checks --
which is why one result can carry several plan hashes.
"""

from __future__ import annotations

import queue
import threading
import time

# The three terminal states of one check. ``pass`` and ``fail`` are the
# oracle's answer; ``timeout`` is the apparatus saying it has no answer, which
# under the owner's error rule is counted and reported and does not stop the
# run (an unanswered sanity check is missing data, not a bad plan).
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"

# CPUs reserved for the oracle's Ray actor. The check itself is one CPU
# elimination and one jax.grad, but XLA:CPU's own thread pool wants more than
# one core to not be the slowest part of a check that otherwise takes 1-30s
# (section 2 of the agent-oracle-async report, 2026-09-18).
ORACLE_ACTOR_NUM_CPUS = 4


def make_ray_oracle_actor_factory(args_dict: dict,
                                   num_cpus: int = ORACLE_ACTOR_NUM_CPUS):
    """A zero-arg callable returning a fresh Ray actor that runs the check.

    ONE ACTOR IS ONE PROCESS, ``num_gpus=0``: the check never touches a GPU
    and must not compete with the trainer's own device for one.

    THE ACTOR BUILDS ITS OWN ``EnvConfig``, ONCE, FROM ``args_dict`` (the
    parsed CLI namespace as a plain dict -- every value a string, number or
    bool). It does NOT receive the trainer's live ``env.config``: that object
    carries ``target_fun``, a closure the check must CALL, and on job 66267 /
    66288 Ray's own serializer refused to ship it (found the whole argument
    tuple non-serializable, tracing to a ``PjitFunction`` inside it). This is
    the same problem ``CpuApproximationActor``'s pool already solved --
    ``cpu_approx_worker._build_env_from_args`` rebuilds ``target_fun`` and
    the jaxpr from ``args_dict`` LOCALLY, inside the worker process, instead
    of shipping the live callable -- so the oracle actor reuses that exact
    function rather than inventing a second way to do the same rebuild.

    ``ray`` and ``_build_env_from_args`` are imported here, lazily, so a
    process that never asks for a Ray oracle -- every test in this file --
    never has to have Ray or the rest of alphagrad importable.
    """
    import ray

    @ray.remote(num_cpus=num_cpus, num_gpus=0)
    class _GradOracleActor:
        def __init__(self, args_dict):
            # Ray's num_cpus is a scheduling hint only. XLA:CPU sizes its
            # thread pool from the affinity mask at the first jax import, so
            # pin the mask here, the way the measure actors already do.
            try:
                import os as _os
                _avail = sorted(_os.sched_getaffinity(0))
                if 0 < num_cpus < len(_avail):
                    _os.sched_setaffinity(0, set(_avail[-num_cpus:]))
            except (AttributeError, OSError, ValueError):
                pass
            self._args_dict = dict(args_dict)
            self._config = None       # built lazily, once, on first check()

        def _config_once(self):
            if self._config is None:
                from alphagrad.approx.cpu_approx_worker import (
                    _build_env_from_args)
                self._config = _build_env_from_args(
                    self._args_dict, None).config
            return self._config

        def check(self, args_np, order, probe_seed):
            from alphagrad.approx import env as _env
            return _env.grad_oracle_cpu_check(
                self._config_once(), args_np, order, probe_seed)

    def _factory():
        return _GradOracleActor.remote(args_dict)

    return _factory


class AsyncGradOracle:
    """One worker running ``check(order, probe_seed, episode)``, or, when
    ``actor_factory`` is given, one Ray CPU actor running the same check
    remotely, one order at a time.

    ``check`` (LOCAL / THREAD MODE, used when ``actor_factory`` is None --
    every test in this file) returns ``(status, rel_l2)`` with ``status`` in
    ``{"pass", "fail"}``; anything it raises is re-raised in the trainer at
    the next boundary exactly as a ``fail`` is, because an oracle that cannot
    run is a defect of the same apparatus and must not be silent. This is the
    mode the tests drive: no Ray needed, so a fake check can be slow, can
    fail and can hang without costing a compile or an actor.

    ``actor_factory`` (RAY / ACTOR MODE, used by the trainer) is a zero-arg
    callable returning a fresh actor handle whose ``.check.remote(args_np,
    order, probe_seed)`` runs the same check in its own process, against a
    ``config`` the actor built ITSELF (see :func:`make_ray_oracle_actor_factory`
    -- a live ``config.target_fun`` cannot cross Ray's wire). ``arg_resolver
    (episode)`` returns the ``args_np`` that episode froze; it plays the role
    the driver-side closure over ``_GRAD_ORACLE_ARGS`` played in thread mode,
    because the actor has none of the trainer's per-episode state and must be
    handed it as a call argument.

    The class is deliberately ignorant of JAX, of env.py and of the plan log:
    it schedules, times and counts. ``env.grad_oracle_cpu_check`` is the real
    check and the tests pass a fake one.
    """

    def __init__(self, check, *, timeout_s: float = 600.0, log=None,
                 actor_factory=None, arg_resolver=None):
        self._check = check
        self.timeout_s = float(timeout_s)
        self._log = log
        self._jobs: queue.Queue = queue.Queue()
        self._done: queue.Queue = queue.Queue()
        # job id -> the submitted job, for every job whose result has not been
        # taken yet. Read and written by the TRAINER thread only.
        self._pending: dict = {}
        self._next_id = 0
        # order -> (status, rel_l2). Written by the WORKER thread only, read by
        # the worker only, so it needs no lock.
        self._memo: dict = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.n_pass = 0
        self.n_fail = 0
        self.n_timeout = 0
        self.n_late = 0          # answers that arrived after their timeout
        self.n_submitted = 0
        # -- actor mode only --
        self._actor_factory = actor_factory
        self._arg_resolver = arg_resolver
        self._actor = None
        self._actor_generation = 0
        self.n_actor_kills = 0
        if self._actor_factory is not None:
            if self._arg_resolver is None:
                raise ValueError(
                    "actor_factory needs arg_resolver: the actor has none of "
                    "the trainer's state and must be handed (config, "
                    "args_np) explicitly for every job")
            self._actor = self._actor_factory()

    # -- the worker ---------------------------------------------------------
    def _start(self) -> None:
        if self._thread is not None:
            return
        # A DAEMON THREAD. The trainer drains at the end of the run and waits
        # up to the timeout; a check still running after that must not keep a
        # finished run alive, because it can no longer change anything the run
        # reports.
        self._thread = threading.Thread(
            target=self._run, name="grad-oracle", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            t0 = time.monotonic()
            memo = self._memo.get(job["order"])
            if memo is not None:
                status, rel = memo
                seconds = 0.0
                error = None
            else:
                try:
                    status, rel = self._check(
                        job["order"], job["probe_seed"], job["episode"])
                    status = str(status)
                    rel = (None if rel is None else float(rel))
                    error = None
                except BaseException as exc:            # noqa: BLE001
                    # THE ORACLE'S OWN FAULT IS STILL A FAULT. It is reported
                    # as a fail with the exception text; the trainer raises on
                    # it at the next boundary like any other fail, so an
                    # apparatus that cannot check anything cannot go unnoticed.
                    status, rel = STATUS_FAIL, None
                    error = f"{type(exc).__name__}: {exc}"
                seconds = time.monotonic() - t0
                self._memo[job["order"]] = (status, rel)
            self._done.put({
                "id": job["id"],
                "episode": job["episode"],
                "order": job["order"],
                "plan_hashes": job["plan_hashes"],
                "status": status,
                "rel_l2": rel,
                "error": error,
                "seconds": seconds,
                "from_memo": memo is not None,
            })

    # -- the trainer's side -------------------------------------------------
    def submit(self, episode: int, probe_seed: int, jobs) -> int:
        """Hand the worker the distinct orders of one episode. Returns how
        many jobs were queued. NEVER BLOCKS and never raises on a full queue:
        the queue is unbounded on purpose, because a bound would put the
        trainer's pace back in the oracle's hands. A ``.remote()`` call
        (actor mode) is itself non-blocking, same as a queue put (thread
        mode)."""
        n = 0
        for job in jobs:
            order = tuple(int(v) for v in job["order"])
            self._next_id += 1
            rec = {
                "id": self._next_id,
                "episode": int(episode),
                "probe_seed": int(probe_seed),
                "order": order,
                "plan_hashes": [str(h) for h in job.get("plan_hashes", ())],
                "submitted_at": time.monotonic(),
            }
            self._pending[rec["id"]] = rec
            if self._actor_factory is not None:
                self._submit_to_actor(rec)
            else:
                self._jobs.put(rec)
            n += 1
        self.n_submitted += n
        if n and self._actor_factory is None:
            self._start()
        return n

    def _submit_to_actor(self, rec: dict) -> None:
        """ACTOR MODE ONLY. A memoized order resolves at once, off the memo,
        with no remote call; a new order is dispatched to the actor and its
        object ref is kept until :meth:`take_results` finds it ready or
        stale."""
        memo = self._memo.get(rec["order"])
        rec["actor_gen"] = self._actor_generation
        if memo is not None:
            status, rel = memo
            rec["ref"] = None
            rec["memo_result"] = (status, rel)
            return
        args_np = self._arg_resolver(rec["episode"])
        rec["ref"] = self._actor.check.remote(
            args_np, rec["order"], rec["probe_seed"])
        rec["memo_result"] = None

    def _count(self, res) -> None:
        if res["status"] == STATUS_PASS:
            self.n_pass += 1
        elif res["status"] == STATUS_TIMEOUT:
            self.n_timeout += 1
        else:
            self.n_fail += 1

    def take_results(self, now: float | None = None) -> list:
        """Every answer that has arrived, plus a ``timeout`` for every job that
        has been in flight longer than ``timeout_s``. Removes them from
        pending. NEVER BLOCKS. In actor mode a timeout also KILLS AND
        RECREATES the actor (see :meth:`_kill_and_respawn`), so a hung check
        no longer disowns the worker the way a thread's did."""
        now = time.monotonic() if now is None else float(now)
        out = []
        if self._actor_factory is not None:
            self._poll_actor(now)
        while True:
            try:
                res = self._done.get_nowait()
            except queue.Empty:
                break
            job = self._pending.pop(res["id"], None)
            if job is None:
                # Already counted as a timeout. The answer is dropped: the
                # episode boundary it belonged to has passed.
                self.n_late += 1
                continue
            out.append(res)
        stale = [(jid, job) for jid, job in sorted(self._pending.items())
                 if now - job["submitted_at"] >= self.timeout_s]
        if stale and self._actor_factory is not None:
            # ONE KILL COVERS EVERY STALE JOB ON THAT ACTOR GENERATION: they
            # were all queued behind the hung check and none of their answers
            # can still arrive once the actor that held them is gone.
            dead_gens = {job["actor_gen"] for _, job in stale
                         if job.get("ref") is not None}
            if dead_gens:
                self._kill_and_respawn()
                stale = [(jid, job) for jid, job in sorted(self._pending.items())
                         if job.get("actor_gen") in dead_gens
                         and job.get("ref") is not None]
        for jid, job in stale:
            out.append({
                "id": jid,
                "episode": job["episode"],
                "order": job["order"],
                "plan_hashes": job["plan_hashes"],
                "status": STATUS_TIMEOUT,
                "rel_l2": None,
                "error": None,
                "seconds": now - job["submitted_at"],
                "from_memo": False,
            })
        for res in out:
            self._pending.pop(res["id"], None)
            self._count(res)
        return out

    def _poll_actor(self, now: float) -> None:
        """ACTOR MODE ONLY. Move every ready job from ``self._pending`` into
        ``self._done``, exactly what the worker thread does for itself in
        ``_run`` -- ``take_results`` cannot tell the two apart afterwards."""
        memoized = [(jid, job) for jid, job in self._pending.items()
                    if job.get("ref") is None and job.get("memo_result") is not None]
        for jid, job in memoized:
            status, rel = job["memo_result"]
            self._done.put({
                "id": jid, "episode": job["episode"], "order": job["order"],
                "plan_hashes": job["plan_hashes"], "status": status,
                "rel_l2": rel, "error": None, "seconds": 0.0,
                "from_memo": True,
            })
        live = [(jid, job) for jid, job in self._pending.items()
                if job.get("ref") is not None]
        if not live:
            return
        refs = [job["ref"] for _, job in live]
        import ray
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        ready_set = set(ready)  # ray.ObjectRef is hashable and comparable
        for jid, job in live:
            if job["ref"] not in ready_set:
                continue
            try:
                status, rel = ray.get(job["ref"])
                status = str(status)
                rel = None if rel is None else float(rel)
                error = None
            except BaseException as exc:                 # noqa: BLE001
                # THE ACTOR'S OWN FAULT IS STILL A FAULT, same as the
                # thread's: reported as a fail with the exception text.
                status, rel = STATUS_FAIL, None
                error = f"{type(exc).__name__}: {exc}"
            self._memo[job["order"]] = (status, rel)
            self._done.put({
                "id": jid, "episode": job["episode"], "order": job["order"],
                "plan_hashes": job["plan_hashes"], "status": status,
                "rel_l2": rel, "error": error,
                "seconds": now - job["submitted_at"], "from_memo": False,
            })

    def _kill_and_respawn(self) -> None:
        """ACTOR MODE ONLY. ``ray.kill`` the hung actor and replace it with a
        fresh one from the same factory. The CPU quota the hung check held is
        reclaimed; a Python thread could never do this."""
        import ray
        try:
            ray.kill(self._actor, no_restart=True)
        except Exception:
            pass
        self.n_actor_kills += 1
        self._actor_generation += 1
        self._actor = self._actor_factory()

    def drain(self, timeout_s: float | None = None) -> list:
        """Wait for the pending checks at process exit, up to the timeout, and
        return every result. Whatever has not answered by then is counted as a
        timeout, so a 1000-episode run ends with every check accounted for."""
        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + limit
        out = []
        while self._pending and time.monotonic() < deadline:
            out.extend(self.take_results())
            if not self._pending:
                break
            time.sleep(0.05)
        out.extend(self.take_results(now=time.monotonic() + limit + 1.0))
        return out

    def pending_episodes(self) -> set:
        """The episodes at least one in-flight check still belongs to. The
        trainer holds one frozen argument set per such episode and drops the
        rest, so a 1000-episode run never accumulates them."""
        return {int(job["episode"]) for job in self._pending.values()}

    def counts(self) -> dict:
        c = {
            "pass": int(self.n_pass),
            "fail": int(self.n_fail),
            "timeout": int(self.n_timeout),
            "pending": int(len(self._pending)),
            "submitted": int(self.n_submitted),
            "late": int(self.n_late),
        }
        # ACTOR MODE ONLY, so the exact dict shape thread-mode tests pin is
        # untouched: how many times a hung check cost the worker its process.
        if self._actor_factory is not None:
            c["actor_kills"] = int(self.n_actor_kills)
        return c

    def close(self) -> None:
        """ACTOR MODE: kill the actor -- built with the measurement pool, torn
        down with it. THREAD MODE: ask the worker to stop after its current
        job, without joining, because a check in flight cannot be interrupted
        and must not hold the run."""
        if self._actor_factory is not None:
            if self._actor is not None:
                import ray
                try:
                    ray.kill(self._actor, no_restart=True)
                except Exception:
                    pass
                self._actor = None
            return
        self._stop.set()
        self._jobs.put(None)


def result_record(res, *, plan_hash: str, tol: float) -> dict:
    """ONE late ``oracle_result`` line of the plan log.

    THE PLAN LOG IS APPEND-ONLY, so a result that arrives long after the plan
    was written cannot be a field on the plan's own line. It is its own record
    instead, joined to the plan by ``plan_hash`` -- the content hash of the
    five wires that decide the executable -- and by ``episode``. ``kind``
    distinguishes it from a terminal-plan record for every reader that scans
    the file.

    ``episode`` is the episode the PLAN was measured in. ``checked_at_episode``
    is the episode boundary at which the answer arrived and this line was
    written. They differ by however long the check took, which is the whole
    point of the asynchronous oracle and is therefore recorded.
    """
    return {
        "kind": "oracle_result",
        "plan_hash": str(plan_hash),
        "episode": int(res["episode"]),
        "order": [int(v) for v in res["order"]],
        "oracle": {
            "status": str(res["status"]),
            "rel_l2": (None if res.get("rel_l2") is None
                       else float(res["rel_l2"])),
            "checked_at_episode": int(res.get("checked_at_episode",
                                              res["episode"])),
            "tol": float(tol),
            "seconds": float(res.get("seconds", 0.0)),
            "device": "cpu",
            "dtype": "float64",
            "error": res.get("error"),
        },
    }
