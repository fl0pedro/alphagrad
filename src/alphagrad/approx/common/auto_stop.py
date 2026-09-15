"""AUTO-STOP: end a 1000-episode run early when the result is already clear.

Ticket ``dsnn-dfw.6``, owner ruling of 2026-09-15. OFF by default, opt-in
through ``--auto-stop``, and never on the gate arm.

WHY THIS EXISTS. The thesis matrix runs arms A, B and C for 1000 episodes
each. Arm A is predicted to collapse to quality 0 and arm B is predicted to
stay at the identity. An arm that has already done so learns nothing in the
remaining 750 episodes, and the node hours it burns are node hours arm C does
not get. So a run whose result is settled ends, and it ends in a state that is
indistinguishable from a run that was configured to be that long: a checkpoint
at the quiescent point, the front dumped, the top-N printed, return code 0.

THE THREE CONDITIONS, AND WHY THEY ARE THREE. Each one alone is a bad stop
rule.

* The archive alone. A front that admitted nothing for 100 episodes can still
  be a policy that is climbing towards it. The front is a record of the best
  points ever seen, and a policy that is improving on average admits nothing
  until it passes the incumbent.
* The return alone. A flat mean return is the signature of a converged run AND
  of a run whose reward channel is saturated while the policy still moves.
  PPO returns are noisy at this scale, so a 2 percent band over 100 episodes
  is a weak statement on its own.
* The plan alone. A policy can emit one plan for 100 episodes and still be
  moving in parameter space towards another one.

Together they say: the search found nothing new, the objective did not move,
and the policy is not producing anything different. That is the definition of
a settled run this project uses.

THE WINDOW CONVENTION. A check point at episode E is evaluated at the
QUIESCENT POINT at the top of the trainer's iteration ``ep == E``, so episodes
with index 0 to E-1 are complete and E episodes have been run. The RECENT
window is the episode indices ``E-W .. E-1`` and the PREVIOUS window is
``E-2W .. E-W-1``, with ``W`` the window size (100). At the ruled check point
250 that is the indices 150..249 against 50..149. A stop there produces a run
of exactly 250 episodes, which is the length the thesis plan's cost table
assumes.

Both windows must be COMPLETE. A check point whose two windows are not fully
observed does not stop and says so. That is reachable only on a resume from a
checkpoint taken inside the previous window; see `AutoStopMonitor.to_json`.

WHAT IS COMPUTED FROM WHAT. Every quantity comes from the episode the trainer
already has in `host_log`, and nothing here measures anything of its own.

* ``admitted`` is the return value of ``ParetoArchive.add_many``. The trainer
  discarded it; the auto-stop call site keeps it. It is the number of points
  that were non-dominated ON ARRIVAL, which is exactly "the archive admitted a
  new point".
* ``scalar_return`` is ``true_scalar_return``, the episode's mean scalar
  return over the environments, before PopArt normalisation. The normalised
  one moves when the normaliser moves and would call a shifting scale a
  change in the result.
* ``qualities`` are the quality channel of the terminal reward vectors of the
  LIVE environments (the same exact-sentinel test the reward panels use). A
  sentinelled environment has no quality, not a quality of -1e10.
* ``n_approx`` is a count of the approximations the terminal plan ASKS FOR,
  taken from the decoded plan: the micro-action calls per vertex plus the live
  face-wire rows. Applied is at most requested, so ``requested == 0`` is
  ``applied == 0`` exactly. This is a count and not the batch-wide
  ``approx_applied_est`` estimate, which ppo.py's own comment forbids using as
  a count.
* ``plan_hash`` is a digest of the archive form of the plan -- the elimination
  order together with the per-face approximation wires, which is the
  definition of Plan in CONTEXT.md. Two plans with the same hash are the same
  plan.

THE TWO COLLAPSE MODES. Arm A is predicted to collapse to quality 0, so a
median terminal quality under 0.05 over the window is a collapse. Arm B is
predicted to stay at the identity, so zero approximations in every terminal
plan of the window is a collapse. Either one fires the detector.

THE MEDIAN OVER THE WINDOW is the median of the per-episode median terminal
qualities, not the median over the pooled plans. The monitor keeps one row per
episode so that the history fits in the checkpoint's ``meta.json`` and
survives a resume; a pooled median would need every plan's quality of 200
episodes in that file. The pooled FRACTION under the collapse threshold is
carried as well (it is a sum of per-episode counts, so it is exact) and is
written into the reason, so a reader who wants the pooled view has it.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

#: How many episodes one window holds. The ruled value.
WINDOW = 100

#: The episodes, counted as "this many episodes are complete", at which the
#: decision is made. The ruled values.
CHECK_POINTS = (250, 500)

#: The mean scalar return must move by LESS than this against the previous
#: window. Strictly less: a move of exactly 2 percent does not stop.
RETURN_TOLERANCE = 0.02

#: Arm A's collapse: the median terminal quality over the window is below
#: this.
COLLAPSE_QUALITY_MEDIAN = 0.05

#: The history format in `meta.json`. Bump when a field changes meaning.
HISTORY_VERSION = 1

#: The file a stop writes into the run directory.
AUTO_STOP_FILENAME = "auto_stop.json"


class AutoStopError(RuntimeError):
    """A configuration of --auto-stop that cannot do what it promises."""


# ---------------------------------------------------------------------------
# Arguments.
# ---------------------------------------------------------------------------

def add_auto_stop_args(p) -> None:
    """Install ``--auto-stop`` and its two hidden test-only knobs."""
    import argparse

    p.add_argument(
        "--auto-stop", action="store_true",
        help="End the run early at a check point (after 250 and after 500 "
             "episodes) when the result is already clear: the Pareto archive "
             "admitted no new point over the last 100 episodes, AND the mean "
             "scalar return moved by less than 2 percent against the previous "
             "100 episodes, AND either the collapse detector fired (median "
             "terminal quality below 0.05, or zero approximations in every "
             "terminal plan) or the terminal plan did not change over the "
             "window. On a stop the run writes auto_stop.json into the run "
             "directory, writes the reason to the wandb summary and to the "
             "plan log, takes a final checkpoint at the quiescent point and "
             "exits 0. OFF by default. Needs --checkpoint-every above 0: a "
             "run that stops with nothing to resume from cannot be continued. "
             "A resumed run with --auto-stop still on re-checks at the NEXT "
             "check point, so raising --episodes past a stop continues the "
             "run.")
    # TEST-ONLY. 250 episodes is an hour on the smallest arm, which no test
    # can pay for, so the check points and the window size are parameters
    # with the ruled values as their defaults. They are SUPPRESSed from the
    # help, they are part of the argument namespace (so a resume checks them
    # like every other argument), and no launcher sets them.
    p.add_argument(
        "--auto-stop-check-at", type=str, default="250,500",
        metavar="N,N", help=argparse.SUPPRESS)
    p.add_argument(
        "--auto-stop-window", type=int, default=WINDOW,
        metavar="N", help=argparse.SUPPRESS)


def check_points(args) -> tuple:
    """The check points of this command line, as a sorted tuple."""
    raw = str(getattr(args, "auto_stop_check_at", "") or "")
    if not raw.strip():
        raise AutoStopError(
            "--auto-stop-check-at is empty, so --auto-stop has nowhere to "
            "check. The ruled value is '250,500'.")
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError as exc:
            raise AutoStopError(
                f"--auto-stop-check-at holds {piece!r}, which is not a whole "
                f"number of episodes.") from exc
        if value <= 0:
            raise AutoStopError(
                f"--auto-stop-check-at holds {value}, which is not a positive "
                f"episode count.")
        out.append(value)
    if not out:
        raise AutoStopError(
            "--auto-stop-check-at parsed to no check point at all.")
    return tuple(sorted(set(out)))


def window_size(args) -> int:
    value = int(getattr(args, "auto_stop_window", WINDOW) or 0)
    if value <= 0:
        raise AutoStopError(
            f"--auto-stop-window must be positive, got {value}.")
    return value


def check_auto_stop_args(args) -> None:
    """RAISE unless ``--auto-stop`` can keep its promise on this command line.

    Called right after `parse_args`, beside the checkpoint's own refusal, so
    a run that cannot stop correctly does not start.
    """
    if not bool(getattr(args, "auto_stop", False)):
        return
    every = int(getattr(args, "checkpoint_every", 0) or 0)
    if every <= 0:
        raise AutoStopError(
            "--auto-stop with --checkpoint-every 0 is refused. An auto-stopped "
            "run ends before the episode count it was given, so the only way "
            "to continue it is to resume from a checkpoint, and with "
            "--checkpoint-every 0 there is nothing to resume from. Either give "
            "--checkpoint-every a positive number (the ruled value for the "
            "thesis arms is 50) or drop --auto-stop.")
    points = check_points(args)
    window = window_size(args)
    episodes = int(getattr(args, "episodes", 0) or 0)
    for point in points:
        if point < 2 * window:
            raise AutoStopError(
                f"--auto-stop cannot check after {point} episodes with a "
                f"window of {window}: the decision compares the last {window} "
                f"episodes against the {window} before them, which needs "
                f"{2 * window} complete episodes.")
    if episodes and min(points) > episodes:
        print(f"[auto-stop] no check point is reachable: the first is after "
              f"{min(points)} episodes and --episodes is {episodes}. The run "
              f"will finish normally.", flush=True)


# ---------------------------------------------------------------------------
# The per-episode observation.
# ---------------------------------------------------------------------------

def plan_hash(plan) -> str:
    """A content hash of one terminal plan.

    `plan` is the archive form ppo.py's `_decode_arch` produces: either the
    plain ``[(vertex, [call, ...]), ...]`` elimination order, or a dict
    ``{"seq": ..., "faces": [...]}`` when the face head is on. Both are built
    from plain ints and strs in a fixed order, so `repr` is a faithful,
    process-stable encoding of the plan's content.
    """
    return hashlib.sha256(repr(plan).encode("utf-8")).hexdigest()[:16]


def _digest(hashes) -> str:
    """One hash standing for a SET of plan hashes."""
    joined = "|".join(sorted(set(hashes)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


# ---------------------------------------------------------------------------
# The monitor.
# ---------------------------------------------------------------------------

class AutoStopMonitor:
    """The per-episode history and the decision made from it.

    One instance per run. `record` is called once per episode from the point
    in `host_log` where the archive is offered this episode's plans; `decide`
    is called at a check point, at the quiescent point of the loop.
    """

    def __init__(self, *, window: int = WINDOW, points=CHECK_POINTS,
                 return_tolerance: float = RETURN_TOLERANCE,
                 collapse_quality_median: float = COLLAPSE_QUALITY_MEDIAN):
        self.window = int(window)
        if self.window <= 0:
            raise AutoStopError(
                f"an auto-stop window must be positive, got {window}.")
        self.points = tuple(sorted(int(p) for p in points))
        self.return_tolerance = float(return_tolerance)
        self.collapse_quality_median = float(collapse_quality_median)
        #: episode index -> one row. Pruned to the last 2*window rows.
        self.rows: dict = {}

    # -- recording ------------------------------------------------------

    def record(self, *, episode: int, admitted: int, scalar_return: float,
               qualities, plan_hashes, n_approx, n_skip=None) -> dict:
        """Take one episode's observation. Returns the row it stored.

        `qualities` are the quality channel of the LIVE environments only.
        `plan_hashes`, `n_approx` and `n_skip` cover EVERY environment: a plan
        exists whether or not its measurement came back.
        """
        episode = int(episode)
        hashes = [str(h) for h in plan_hashes]
        approx = [int(a) for a in n_approx]
        skips = [int(s) for s in (n_skip or ())]
        quals = [float(q) for q in np.asarray(qualities, dtype=np.float64).ravel()
                 if math.isfinite(float(q))]
        if not math.isfinite(float(scalar_return)):
            raise AutoStopError(
                f"episode {episode} offered a non-finite scalar return "
                f"({scalar_return!r}). The auto-stop decision reads it, and a "
                f"non-finite return is a broken episode, not a flat one.")
        row = {
            "episode": episode,
            "admitted": int(admitted),
            "ret": float(scalar_return),
            "n_plans": len(hashes),
            "n_quality": len(quals),
            "q_median": _median(quals),
            "q_min": (min(quals) if quals else None),
            "q_max": (max(quals) if quals else None),
            "q_below": int(sum(
                1 for q in quals if q < self.collapse_quality_median)),
            "approx_max": (max(approx) if approx else 0),
            "approx_sum": int(sum(approx)),
            "skip_sum": int(sum(skips)),
            "n_distinct": len(set(hashes)),
            "plans": _digest(hashes),
        }
        self.rows[episode] = row
        self._prune(episode)
        return row

    def _prune(self, episode: int) -> None:
        keep = 2 * self.window
        if len(self.rows) <= keep:
            return
        for stale in sorted(self.rows)[:len(self.rows) - keep]:
            del self.rows[stale]

    # -- the decision ---------------------------------------------------

    def is_check_point(self, episode: int) -> bool:
        return int(episode) in self.points

    def _slice(self, lo: int, hi: int):
        """The rows for episode indices `lo`..`hi` inclusive, or None when one
        of them was never observed."""
        rows = []
        for ep in range(int(lo), int(hi) + 1):
            row = self.rows.get(ep)
            if row is None:
                return None
            rows.append(row)
        return rows

    def decide(self, episode: int) -> dict:
        """The decision at a check point after `episode` complete episodes.

        Returns a dict with ``stop`` (bool), ``reason`` (the numbers) and
        ``message`` (one line of plain English). It never raises on an
        incomplete history: it declines to stop and says which episodes are
        missing.
        """
        episode = int(episode)
        w = self.window
        recent = self._slice(episode - w, episode - 1)
        previous = self._slice(episode - 2 * w, episode - w - 1)
        base = {
            "version": HISTORY_VERSION,
            "check_point": episode,
            "window": w,
            "recent_window": [episode - w, episode - 1],
            "previous_window": [episode - 2 * w, episode - w - 1],
            "return_tolerance": self.return_tolerance,
            "collapse_quality_median": self.collapse_quality_median,
        }
        if recent is None or previous is None:
            have = sorted(self.rows)
            missing = [ep for ep in range(episode - 2 * w, episode)
                       if ep not in self.rows]
            base.update(
                stop=False,
                conditions={},
                incomplete=True,
                observed=[have[0], have[-1]] if have else [],
                missing_count=len(missing),
                missing_first=(missing[0] if missing else None),
                missing_last=(missing[-1] if missing else None),
            )
            base["message"] = (
                f"auto-stop does not decide after {episode} episodes: "
                f"{len(missing)} of the {2 * w} episodes the two windows need "
                f"were not observed by this process (first {missing[0] if missing else None}, "
                f"last {missing[-1] if missing else None}). The run continues.")
            return {"stop": False, "reason": base,
                    "message": base["message"]}

        # (1) the archive admitted no new point over the recent window.
        admitted = int(sum(r["admitted"] for r in recent))
        archive_quiet = admitted == 0

        # (2) the mean scalar return moved by less than the tolerance.
        m_recent = float(np.mean([r["ret"] for r in recent]))
        m_previous = float(np.mean([r["ret"] for r in previous]))
        if m_previous == 0.0:
            rel = 0.0 if m_recent == 0.0 else float("inf")
        else:
            rel = abs(m_recent - m_previous) / abs(m_previous)
        return_flat = rel < self.return_tolerance

        # (3a) arm A's collapse: the median terminal quality is under 0.05.
        ep_medians = [r["q_median"] for r in recent if r["q_median"] is not None]
        q_window_median = _median(ep_medians)
        collapse_quality = (q_window_median is not None
                            and q_window_median < self.collapse_quality_median)
        n_quality = int(sum(r["n_quality"] for r in recent))
        n_below = int(sum(r["q_below"] for r in recent))

        # (3b) arm B's collapse: no terminal plan of the window asked for a
        # single approximation. Requested is an upper bound on applied, so
        # requested == 0 is applied == 0 exactly.
        approx_max = int(max(r["approx_max"] for r in recent))
        collapse_identity = approx_max == 0
        collapse = bool(collapse_quality or collapse_identity)

        # (3c) the terminal plan did not change: the set of distinct terminal
        # plans over the whole window has size 1.
        digests = {r["plans"] for r in recent}
        plan_frozen = (len(digests) == 1
                       and all(r["n_distinct"] == 1 for r in recent))
        third = bool(collapse or plan_frozen)

        stop = bool(archive_quiet and return_flat and third)
        base.update(
            stop=stop,
            incomplete=False,
            conditions={
                "archive_admitted_nothing": archive_quiet,
                "return_moved_less_than_tolerance": return_flat,
                "collapse_or_plan_frozen": third,
                "collapse_detector_fired": collapse,
                "collapse_quality": collapse_quality,
                "collapse_identity": collapse_identity,
                "plan_did_not_change": plan_frozen,
            },
            numbers={
                "admitted_recent": admitted,
                "mean_return_recent": m_recent,
                "mean_return_previous": m_previous,
                "relative_return_move": rel,
                "quality_median_of_episode_medians": q_window_median,
                "quality_plans_in_window": n_quality,
                "quality_plans_below_threshold": n_below,
                "max_approximations_in_any_terminal_plan": approx_max,
                "total_approximations_in_window": int(
                    sum(r["approx_sum"] for r in recent)),
                "total_skips_in_window": int(
                    sum(r["skip_sum"] for r in recent)),
                "distinct_terminal_plans_in_window": (
                    1 if plan_frozen else _distinct_estimate(recent)),
                "terminal_plan_digest": (
                    sorted(digests)[0] if plan_frozen else None),
            },
        )
        base["message"] = _message(base)
        return {"stop": stop, "reason": base, "message": base["message"]}

    # -- persistence -----------------------------------------------------

    def to_json(self) -> dict:
        """The history, for the checkpoint's `meta.json`.

        A resume needs it: the window at check point 500 reaches back to
        episode 300, and a run resumed from the checkpoint at 350 has observed
        nothing before 350. Without the history such a resume could not decide
        at all. One row per episode, at most 2*window rows, so this is of the
        order of ten kilobytes beside the 25 kilobytes `meta.json` already
        holds.
        """
        return {
            "version": HISTORY_VERSION,
            "window": self.window,
            "points": list(self.points),
            "return_tolerance": self.return_tolerance,
            "collapse_quality_median": self.collapse_quality_median,
            "rows": [self.rows[ep] for ep in sorted(self.rows)],
        }

    def load_json(self, doc: dict) -> None:
        """Restore a history written by `to_json`. RAISES on a mismatch.

        The window and the check points are part of the argument namespace, so
        a resume has already been refused if they differ; this check is the
        second one, against the history itself, and it exists because a
        history taken under another window is a history of other numbers.
        """
        if not isinstance(doc, dict):
            raise AutoStopError(
                "the checkpoint holds no auto-stop history, but this run has "
                "--auto-stop on. The history is what the decision is made "
                "from, and a resume without it could not decide at the next "
                "check point.")
        version = int(doc.get("version", -1))
        if version != HISTORY_VERSION:
            raise AutoStopError(
                f"the checkpoint's auto-stop history is version {version}; "
                f"this build reads version {HISTORY_VERSION}.")
        saved_window = int(doc.get("window", -1))
        if saved_window != self.window:
            raise AutoStopError(
                f"the checkpoint's auto-stop window is {saved_window} and this "
                f"run's is {self.window}. The stored rows are per episode, but "
                f"the decision they feed is a comparison of two windows, so "
                f"two windows are two different questions.")
        saved_points = tuple(int(p) for p in doc.get("points", ()))
        if saved_points != self.points:
            raise AutoStopError(
                f"the checkpoint's auto-stop check points are {list(saved_points)} "
                f"and this run's are {list(self.points)}.")
        self.rows = {}
        for row in doc.get("rows", ()):
            self.rows[int(row["episode"])] = dict(row)


def _distinct_estimate(recent) -> int:
    """A LOWER BOUND on the number of distinct terminal plans in the window.

    The monitor keeps one digest per episode rather than every plan hash, so
    the exact union is not recoverable. The bound is enough for the reason
    text: the condition it serves ("the set has size 1") is decided exactly by
    the digests, and this number is only reported when that condition already
    failed.
    """
    return max(len({r["plans"] for r in recent}),
               max(int(r["n_distinct"]) for r in recent))


def _message(reason: dict) -> str:
    c = reason["conditions"]
    n = reason["numbers"]
    lo, hi = reason["recent_window"]
    plo, phi = reason["previous_window"]
    held = []
    held.append(
        f"the archive admitted {n['admitted_recent']} new points over "
        f"episodes {lo}..{hi}"
        + (" (condition 1 HOLDS)" if c["archive_admitted_nothing"]
           else " (condition 1 does not hold)"))
    held.append(
        f"the mean scalar return moved {100.0 * n['relative_return_move']:.3f} "
        f"percent, from {n['mean_return_previous']:.6g} over episodes "
        f"{plo}..{phi} to {n['mean_return_recent']:.6g}"
        + (f" (condition 2 HOLDS, the tolerance is "
           f"{100.0 * reason['return_tolerance']:.1f} percent)"
           if c["return_moved_less_than_tolerance"]
           else " (condition 2 does not hold)"))
    third = []
    if c["collapse_quality"]:
        third.append(
            f"the median terminal quality is "
            f"{n['quality_median_of_episode_medians']:.4g}, below "
            f"{reason['collapse_quality_median']:g}")
    if c["collapse_identity"]:
        third.append(
            "no terminal plan of the window asked for a single approximation")
    if c["plan_did_not_change"]:
        third.append(
            f"every terminal plan of the window is the same plan "
            f"({n['terminal_plan_digest']})")
    if third:
        held.append("; ".join(third) + " (condition 3 HOLDS)")
    else:
        qm = n["quality_median_of_episode_medians"]
        held.append(
            f"the collapse detector did not fire (median terminal quality "
            f"{'unknown' if qm is None else format(qm, '.4g')}, "
            f"{n['max_approximations_in_any_terminal_plan']} approximations in "
            f"the busiest terminal plan) and the terminal plan changed at "
            f"least {n['distinct_terminal_plans_in_window']} times over the "
            f"window (condition 3 does not hold)")
    verdict = ("STOPS" if reason["stop"] else "does not stop")
    return (f"auto-stop {verdict} after {reason['check_point']} episodes: "
            + "; ".join(held) + ".")


def _strict(value):
    """Make one value strict-JSON writable.

    A relative return move against a previous window of exactly zero is
    `inf`, and `json.dumps` writes `Infinity` for it, which is not JSON and
    which several readers refuse. Non-finite floats become the STRINGS
    "inf" / "-inf" / "nan", the same convention `plan_log.jsonable` uses, so
    they are distinguishable from a missing value. This is the only value in
    the reason that can be non-finite, and it is never non-finite on a file
    that was actually written (a stop needs the move to be under the
    tolerance), but the reason is printed and summarised on a NON-stop too.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): _strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(v) for v in value]
    return value


def write_auto_stop_json(run_dir: str, reason: dict) -> str:
    """Write ``auto_stop.json`` into the run directory. Returns the path."""
    import os

    path = os.path.join(run_dir or ".", AUTO_STOP_FILENAME)
    with open(path, "w") as fh:
        json.dump(_strict(reason), fh, indent=2, sort_keys=True,
                  allow_nan=False)
    return path


def summary_fields(reason: dict) -> dict:
    """The wandb summary fields a stop writes. Flat scalars and strings."""
    out = {
        "auto_stop/stopped": True,
        "auto_stop/episodes": int(reason["check_point"]),
        "auto_stop/window": int(reason["window"]),
        "auto_stop/recent_window_first": int(reason["recent_window"][0]),
        "auto_stop/recent_window_last": int(reason["recent_window"][1]),
        "auto_stop/message": str(reason["message"]),
    }
    for name, value in reason.get("conditions", {}).items():
        out[f"auto_stop/{name}"] = bool(value)
    for name, value in reason.get("numbers", {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[f"auto_stop/{name}"] = bool(value)
        elif isinstance(value, (int, float)):
            out[f"auto_stop/{name}"] = _strict(float(value))
        else:
            out[f"auto_stop/{name}"] = str(value)
    return out


def plan_log_record(reason: dict) -> dict:
    """The one record a stop appends to the plan log.

    The plan log is a stream of terminal plans, and a reader that hits this
    row must not read it as one. `record` says what it is, and the two fields
    every plan row has (`episode`, `plan_index`) are deliberately absent.
    """
    return {
        "record": "auto_stop",
        "auto_stop": dict(reason),
    }
