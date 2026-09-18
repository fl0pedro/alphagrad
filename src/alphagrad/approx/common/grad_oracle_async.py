"""THE GRADIENT ORACLE, ASYNCHRONOUS AND RETROACTIVE (owner ruling 2026-09-18).

THE ORACLE IS A SANITY CHECK, NOT PART OF THE SCORING. It compares the exact
vertex-elimination gradient of an elimination order against ``jax.grad`` of the
same target. Until this module it ran INSIDE the measurement callback, on the
measure actor's GPU, before the plan's quality reference was used -- so its own
float64 compile could fail and REFUSE the plan. On Blackwell it did exactly
that for 15.6 percent of the plans of an oracle-due episode (agent-sentinel
report, 2026-09-18, section 4). The apparatus was scoring the apparatus.

WHAT THIS MODULE IS. One worker thread in the TRAINER process owns the oracle.
The trainer hands it the distinct elimination orders of an oracle-due episode
and goes on. The check runs on the CPU device in float64, where every order
sits at 1e-14 (agent-df8 report, 2026-09-16). Results arrive whenever they
arrive and are written RETROACTIVELY, as a late ``oracle_result`` record in the
append-only plan log and as counters in wandb.

THE TRAINER NEVER WAITS. There is no ticket, no drain, no exclusion of a plan
from the update, and no path by which a slow or failing check can change a
reward, a gradient or the pace of an episode. The only thing the oracle can do
to the run is STOP it: a disagreement in float64 on the CPU is a real graphax
defect and not noise, so the trainer raises at the next episode boundary, after
that episode's checkpoint is written, and the run resumes from that checkpoint
once the defect is fixed.

TIME IS BOUNDED BY THE SUBMITTER, NOT BY THE WORKER. A Python thread cannot be
interrupted, so a check that runs too long is not killed: it is DISOWNED. Once
a job has been in flight longer than the timeout, :meth:`take_results` counts
it as ``timeout``, writes its late record and forgets it; if the worker
finishes it later the answer is dropped, because a record for an episode the
run has already passed a decision on would say the check was made in time when
it was not. Nothing blocks either way, so a backlog costs pending count and
nothing else.

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


class AsyncGradOracle:
    """One worker thread running ``check(order, probe_seed, episode)``.

    ``check`` returns ``(status, rel_l2)`` with ``status`` in
    ``{"pass", "fail"}``; anything it raises is re-raised in the trainer at the
    next boundary exactly as a ``fail`` is, because an oracle that cannot run
    is a defect of the same apparatus and must not be silent.

    The class is deliberately ignorant of JAX, of env.py and of the plan log:
    it schedules, times and counts. ``env.grad_oracle_cpu_check`` is the real
    check and the tests pass a fake one.
    """

    def __init__(self, check, *, timeout_s: float = 600.0, log=None):
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
        """Hand the thread the distinct orders of one episode. Returns how
        many jobs were queued. NEVER BLOCKS and never raises on a full queue:
        the queue is unbounded on purpose, because a bound would put the
        trainer's pace back in the oracle's hands."""
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
            self._jobs.put(rec)
            n += 1
        self.n_submitted += n
        if n:
            self._start()
        return n

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
        pending. NEVER BLOCKS."""
        now = time.monotonic() if now is None else float(now)
        out = []
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
        for jid, job in sorted(self._pending.items()):
            if now - job["submitted_at"] < self.timeout_s:
                continue
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
        return {
            "pass": int(self.n_pass),
            "fail": int(self.n_fail),
            "timeout": int(self.n_timeout),
            "pending": int(len(self._pending)),
            "submitted": int(self.n_submitted),
            "late": int(self.n_late),
        }

    def close(self) -> None:
        """Ask the worker to stop after its current job. Does not join: a
        check in flight cannot be interrupted and must not hold the run."""
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
