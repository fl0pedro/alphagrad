"""THE READOUT (owner ruling 2026-09-26, Q2 c; ticket dsnn-dfw.291).

After training, the final checkpoint is loaded, plans are sampled from the
policy with no further update, and they are measured with the same instrument
the training episodes are measured with. `--readout N` samples N plans and the
argmax plan (the argmax at every step). The archive front says what the search
found, and the readout says what the policy learned (CONTEXT.md, Readout);
both are reported, each with its own claim.

JAX-free. Everything here is host arithmetic over plan records and JSON, so it
is testable without building an agent. `ppo.main` runs the rollouts and the
measurement: at the end of a training run under `--readout N`, and for
`tools/readout.py` on the checkpoint of a finished run.

ONE RECORD PER PLAN, in `readout.jsonl` in the run directory. Every record is
marked `"record": "readout"`, and `draw` says which of the two a plan is. A
failed plan keeps its record and its failed-plan scores, a dropped measurement
keeps its record with no quality, and a plan whose plan record never reached
the trainer keeps its record too: nothing is dropped from the readout.
"""
from __future__ import annotations

import json
import os

import numpy as np

from alphagrad.approx.common.checkpoint import (
    CheckpointError, list_checkpoints, read_ppo_meta)

#: The readout's file, beside `pareto_front.json` in the run directory.
READOUT_FILE = "readout.jsonl"

#: The marker every readout record carries, in the plan log's `record` field
#: convention (an auto-stop row is `"record": "auto_stop"`).
RECORD = "readout"

#: How a plan was drawn from the policy.
SAMPLED = "sampled"
ARGMAX = "argmax"

#: Folded into the seed, so the readout's key stream is its own and not a
#: stretch of the training key chain.
KEY_TAG = 291

#: What a plan's measurement came to, in the glossary's words.
#: `measured`: a paired measurement. `failed`: a failed plan, scored as one
#: (the plan record calls it `refused`). `dropped`: a dropped measurement, the
#: undefined quality the update leaves out. `missing`: no plan record reached
#: the trainer for this plan; its reward vector is all there is.
STATUSES = ("measured", "failed", "dropped", "missing")

#: The fields a table row carries, in order (the wandb summary's table).
TABLE_COLUMNS = ("draw", "index", "plan_hash", "status", "quality",
                 "latency_log_ratio", "memory_log_ratio", "refused")


def schedule(n_plans: int, num_envs: int) -> list:
    """One entry per readout rollout: the draw that rollout makes.

    A rollout measures `num_envs` plans, so N sampled plans take N / num_envs
    rollouts, and the argmax plan takes one more, whose environments all draw
    the same plan. RAISES rather than rounding: a readout that quietly measured
    48 plans where 64 were asked for would be reported as 64.
    """
    n, e = int(n_plans), int(num_envs)
    if n < 1 or e < 1:
        raise ValueError(
            f"a readout needs a positive plan count and environment count, "
            f"got --readout {n_plans!r} and --num-envs {num_envs!r}")
    if n % e:
        raise ValueError(
            f"--readout {n} is not a multiple of --num-envs {e}. One rollout "
            f"measures {e} plans, so only multiples of {e} are measured "
            f"exactly.")
    return [SAMPLED] * (n // e) + [ARGMAX]


def plan_status(plan_record, dropped: bool) -> str:
    """`measured`, `failed`, `dropped` or `missing` (see STATUSES).

    `dropped` is the trainer's own exclusion predicate (every cost channel of
    the reward row at the sentinel), handed in by the caller, so the readout
    and the update agree on which measurements are dropped.
    """
    if dropped:
        return "dropped"
    if plan_record is None:
        return "missing"
    return "failed" if plan_record.get("refused") else "measured"


def record(*, draw, index, rollout, env, provenance, plan_hash, plan,
           rewards, quality, numbers, plan_record, dropped) -> dict:
    """One readout record.

    `provenance` names the readout (checkpoint, its episode, seed, run name,
    and the sha256 of the policy's array leaves it read out);
    `numbers` holds the paired log ratios read off the plan record the way
    the archive front reads them (`latency_log_ratio`, `memory_log_ratio`,
    `memory_source`, `latency_quantiles`). A dropped measurement has no
    quality: it is undefined, which is why the update leaves it out.
    """
    status = plan_status(plan_record, dropped)
    out = {
        "record": RECORD,
        "draw": str(draw),
        "index": int(index),
        "rollout": int(rollout),
        "env": int(env),
        **provenance,
        "status": status,
        "plan_hash": plan_hash,
        "plan": plan,
        "rewards": [float(x) for x in rewards],
        "quality": None if status == "dropped" else float(quality),
        "latency_log_ratio": numbers.get("latency_log_ratio"),
        "memory_log_ratio": numbers.get("memory_log_ratio"),
        "memory_source": numbers.get("memory_source"),
        "latency_quantiles": numbers.get("latency_quantiles"),
        "refused": (None if plan_record is None
                    else plan_record.get("refused")),
        "plan_record": plan_record,
    }
    return out


def _median(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def summary(records, quality_floor=None) -> dict:
    """The flat fields the readout writes into the wandb summary.

    Over the SAMPLED plans: the count of each status, the distinct plans, and
    the medians of quality and of the two paired log ratios over the plans
    that were scored, which counts a failed plan at its failed-plan scores.
    Then the argmax plan's own numbers, and whether the sample holds it.
    """
    sampled = [r for r in records if r["draw"] == SAMPLED]
    argmax = [r for r in records if r["draw"] == ARGMAX]
    if len(argmax) != 1:
        raise ValueError(
            f"a readout holds exactly one argmax plan, got {len(argmax)}")
    a = argmax[0]
    out = {
        "readout/plans": len(records),
        "readout/sampled": len(sampled),
        "readout/checkpoint_episode": int(a["checkpoint_episode"]),
        "readout/seed": int(a["seed"]),
        "readout/distinct": len({r["plan_hash"] for r in sampled}),
        "readout/argmax_in_sample": bool(
            any(r["plan_hash"] == a["plan_hash"] for r in sampled)),
    }
    for st in STATUSES:
        out[f"readout/{st}"] = sum(1 for r in sampled if r["status"] == st)
    scored = [r for r in sampled if r["status"] in ("measured", "failed")]
    for name in ("quality", "latency_log_ratio", "memory_log_ratio"):
        med = _median(r[name] for r in scored)
        if med is not None:
            out[f"readout/{name}_median"] = med
    if quality_floor is not None and scored:
        out["readout/feasible_fraction"] = float(np.mean(
            [float(r["quality"]) >= float(quality_floor) for r in scored]))
    out["readout/argmax/status"] = a["status"]
    out["readout/argmax/plan_hash"] = a["plan_hash"]
    for name in ("quality", "latency_log_ratio", "memory_log_ratio"):
        if a[name] is not None:
            out[f"readout/argmax/{name}"] = float(a[name])
    return out


def table_rows(records) -> list:
    """One row per plan, in TABLE_COLUMNS order."""
    return [[r[c] for c in TABLE_COLUMNS] for r in records]


def write_records(path: str, records) -> str:
    """Write the readout's records to `path`. REFUSES to overwrite.

    A readout names one checkpoint and one seed, and a second readout written
    over the first would leave no trace of the first. Pass another directory.
    """
    from alphagrad.approx.common.plan_log import append_records

    if os.path.exists(path):
        raise FileExistsError(
            f"{path} exists: a readout does not overwrite another readout. "
            f"Write this one to another directory.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    append_records(path, records)
    return path


def load_records(path: str) -> list:
    """The records of a readout file, every one checked for the marker."""
    from alphagrad.approx.common.plan_log import read_records

    out = read_records(path)
    for i, rec in enumerate(out):
        if rec.get("record") != RECORD:
            raise ValueError(
                f"{path}: record {i} is not a readout record "
                f"(record={rec.get('record')!r})")
    return out


def resolve_checkpoint(path: str, episodes: int) -> str:
    """The checkpoint a readout reads.

    PATH itself when it is a checkpoint directory. Otherwise PATH is a run
    directory, and the checkpoint is the run's FINAL one: the newest, which
    must sit at the episode the run ends at, `--episodes`, or the episode
    `auto_stop.json` names when the run stopped early. A run whose newest
    checkpoint is short of that did not finish, and RAISES: its newest
    checkpoint is not the policy the run ended with.
    """
    from alphagrad.approx.common.auto_stop import AUTO_STOP_FILENAME

    if os.path.exists(os.path.join(path, "meta.json")):
        return path
    found = list_checkpoints(path)
    if not found:
        raise CheckpointError(
            f"{path!r} is neither a checkpoint directory nor a run directory "
            f"holding one, so there is no checkpoint to read out.")
    last = found[-1]
    end = int(episodes)
    stop = os.path.join(path, AUTO_STOP_FILENAME)
    if os.path.exists(stop):
        with open(stop) as fh:
            end = int(json.load(fh)["check_point"])
    got = int(read_ppo_meta(last)["episode"])
    if got != end:
        raise CheckpointError(
            f"the newest checkpoint in {path!r} is at episode {got} and the "
            f"run ends at episode {end}, so the run did not finish and has no "
            f"final checkpoint. Name a checkpoint directory to read that one "
            f"out.")
    return last
