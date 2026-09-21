"""The inference-time preference sweep: load, pin, write, compare.

JAX-free. Everything here is host arithmetic over saved JSON, so the loader
and the front writer are testable on a CPU node without building an agent.

THE LOADER is not a resume. ``--resume`` is argument-locked by design
(common/checkpoint.check_resume_args) because a resume continues the run the
checkpoint is a state of. A sweep READS the checkpoint: it rebuilds the
argument namespace the run was started with, overrides the handful of
arguments that decide where output goes and that no array depends on, and
never takes a gradient step. The guarantee the resume gets from the argument
check, the sweep gets from `load_ppo_tree`'s manifest comparison: a namespace
that does not rebuild this checkpoint's tree cannot deserialise it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from alphagrad.approx.common.checkpoint import CheckpointError, args_to_json
from alphagrad.approx.common.pareto_archive import hypervolume


#: The sweep's own arguments. They are absent from every saved namespace, so
#: the loader neither expects nor refuses them.
SWEEP_ONLY_ARGS = frozenset({
    "preference_sweep_checkpoint",
    "preference_sweep_weights",
    "preference_sweep_plans",
    "preference_sweep_out",
})

#: What a sweep MAY change about the run it reads. Everything else comes from
#: the checkpoint. `name`/`wandb*` name the output, `episodes` is unused (the
#: sweep's length is its weight list), and the three below turn off machinery
#: that only a training run has any use for.
SWEEP_OVERRIDABLE_ARGS = frozenset({
    "name", "wandb", "wandb_entity", "wandb_project", "episodes",
    "checkpoint_every", "resume", "auto_stop", "grad_oracle",
    "pareto_dump_every", "print_top_every",
})


def load_sweep_args(meta: dict, parser, overrides: dict | None = None):
    """The saved argument namespace, rebuilt as an `argparse.Namespace`.

    RAISES when the checkpoint and this build disagree about which arguments
    exist. A saved argument this build does not define, or a defined argument
    the checkpoint does not carry, means the namespace would be completed
    from this build's defaults -- which is the silent drift the resume path
    refuses too.
    """
    saved = meta.get("args")
    if not isinstance(saved, dict):
        raise CheckpointError(
            "the checkpoint carries no argument namespace, so the run that "
            "wrote it cannot be rebuilt.")
    defaults = vars(parser.parse_args([]))
    known = set(defaults) - SWEEP_ONLY_ARGS
    missing = sorted(known - set(saved))
    extra = sorted(set(saved) - set(defaults))
    if missing or extra:
        raise CheckpointError(
            "the checkpoint's argument namespace does not match this build:\n"
            + "".join(f"  {n}: defined here, absent from the checkpoint\n"
                      for n in missing)
            + "".join(f"  {n}: in the checkpoint, not defined here\n"
                      for n in extra))
    out = dict(defaults)
    for name in known:
        out[name] = _as_default_type(defaults[name], saved[name])
    for name, value in (overrides or {}).items():
        if name not in SWEEP_OVERRIDABLE_ARGS and name not in SWEEP_ONLY_ARGS:
            raise CheckpointError(
                f"a preference sweep may not override {name!r}: it is part of "
                f"the state the checkpoint is a state of. Overridable: "
                f"{sorted(SWEEP_OVERRIDABLE_ARGS)}")
        out[name] = value
    return argparse.Namespace(**out)


def _as_default_type(default, value):
    """`value` in the container type this build's default uses."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def sweep_args_round_trip(meta: dict, parser) -> bool:
    """True when the rebuilt namespace writes back the saved one exactly."""
    rebuilt = args_to_json(load_sweep_args(meta, parser))
    for name in SWEEP_ONLY_ARGS:
        rebuilt.pop(name, None)
    return rebuilt == meta["args"]


# ---------------------------------------------------------------------------
# The preference grid.
# ---------------------------------------------------------------------------

def edge_weights(n: int) -> list:
    """`n` points on the latency-memory edge, both corners included.

    Under `--reward-mode lagrangian --preference-conditioned` the policy reads
    (latency, memory, lambda): `_lag_preferences` drops the quality
    coordinate, renormalises the two cost coordinates to sum to 1 and writes
    lambda into the quality slot. So a preference on that edge IS one number,
    and this is the grid the sweep walks.
    """
    if int(n) < 2:
        raise ValueError(
            f"an edge sweep needs at least the two corners, got {n!r}")
    return [(float(t), float(1.0 - t), 0.0)
            for t in np.linspace(0.0, 1.0, int(n))]


def parse_weights(spec: str, num_heads: int) -> list:
    """`--preference-sweep-weights`: "edge:N", or explicit vectors.

    Explicit form: semicolon-separated vectors of `num_heads` comma-separated
    numbers ("1,0,0;0.5,0.5,0"). The quality coordinate is READ AND KEPT -- in
    lagrangian mode the trainer overwrites it with lambda, and a sweep that
    silently dropped it would report a preference the policy never saw.
    """
    spec = str(spec or "").strip()
    if not spec:
        raise ValueError("--preference-sweep-weights is empty")
    if spec.startswith("edge:"):
        return edge_weights(int(spec.split(":", 1)[1]))
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vec = [float(x) for x in chunk.split(",")]
        if len(vec) != int(num_heads):
            raise ValueError(
                f"preference vector {chunk!r} has {len(vec)} components; this "
                f"policy is conditioned on {int(num_heads)}")
        out.append(tuple(vec))
    if not out:
        raise ValueError(f"--preference-sweep-weights {spec!r} names no vector")
    return out


def sweep_schedule(weights, plans_per_weight: int, num_envs: int) -> list:
    """One entry per sweep episode: the weight that episode pins.

    An episode rolls out `num_envs` plans, so a weight that wants more than
    that gets consecutive episodes. RAISES rather than rounding: a sweep that
    quietly measured 16 plans where 24 were asked for would be reported as 24.
    """
    n = int(plans_per_weight)
    e = int(num_envs)
    if n < 1 or e < 1:
        raise ValueError(
            f"plans per weight and environments must both be positive, got "
            f"{plans_per_weight!r} and {num_envs!r}")
    if n % e:
        raise ValueError(
            f"--preference-sweep-plans {n} is not a multiple of the "
            f"checkpoint's --num-envs {e}. One episode measures {e} plans, so "
            f"only multiples of {e} are measured exactly.")
    out = []
    for w in weights:
        out.extend([tuple(float(x) for x in w)] * (n // e))
    return out


# ---------------------------------------------------------------------------
# The two fronts, compared.
# ---------------------------------------------------------------------------

def front_points(doc: dict, objectives) -> np.ndarray:
    """The (P, 2) medians of a `pareto_front.json`, in the named order.

    MINIMISATION coordinates, which is what the file holds: a log ratio
    against the paired rev-exact reference, 0 at parity, lower is better.
    """
    names = list(doc.get("objectives") or ())
    for nm in objectives:
        if nm not in names:
            raise ValueError(
                f"the front carries objectives {names}, which do not include "
                f"{nm!r}; the two fronts are not in one space")
    pts = [[float(p["obj"][nm]) for nm in objectives]
           for p in doc.get("front") or ()]
    return np.asarray(pts, dtype=np.float64).reshape(-1, len(objectives))


def set_coverage(a, b) -> float:
    """C(A, B): the fraction of B weakly dominated by some point of A.

    Minimisation. 0 when B is empty, which is the only honest answer to
    "how much of nothing does A cover".
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"set coverage wants two point sets in one space, got shapes "
            f"{a.shape} and {b.shape}")
    le = np.all(a[:, None, :] <= b[None, :, :], axis=-1)
    return float(np.mean(np.any(le, axis=0)))


def shared_nadir(*fronts, margin: float = 1.0) -> np.ndarray:
    """ONE nadir for both fronts, from both fronts, in MAXIMISATION space.

    Both archive classes freeze their own nadir on the first non-empty call,
    so `pareto/hypervolume` is monotone within one run and means nothing
    across two. A comparison therefore computes its own, records it, and
    hands it to the module-level `hypervolume`.
    """
    pts = [np.asarray(f, dtype=np.float64) for f in fronts if np.size(f)]
    if not pts:
        raise ValueError("a shared nadir needs at least one non-empty front")
    allp = -np.concatenate(pts, axis=0)
    return allp.min(axis=0) - float(margin)


def compare_fronts(swept: np.ndarray, archive: np.ndarray,
                   objectives) -> dict:
    """Coverage both ways and hypervolume for both, under ONE nadir."""
    ref = shared_nadir(swept, archive)
    return {
        "objectives": list(objectives),
        "nadir": [float(x) for x in ref],
        "nadir_space": "maximisation (negated log ratios)",
        "num_swept": int(np.asarray(swept).reshape(-1, len(objectives)).shape[0]),
        "num_archive": int(
            np.asarray(archive).reshape(-1, len(objectives)).shape[0]),
        "coverage_swept_of_archive": set_coverage(swept, archive),
        "coverage_archive_of_swept": set_coverage(archive, swept),
        "hypervolume_swept": hypervolume(-np.asarray(swept, dtype=np.float64),
                                         ref),
        "hypervolume_archive": hypervolume(
            -np.asarray(archive, dtype=np.float64), ref),
    }


def per_weight_table(plans, quality_floor: float) -> list:
    """One row per preference weight: the median, its band, feasibility.

    THE BAND IS THE 90 PERCENT MEDIAN INTERVAL OF THE PLANS AT THAT WEIGHT,
    fitted by the same `median_band` the archive fits its points with. It is
    NOT the union of the per-plan bands: that is an envelope, one bad plan
    sets it, and it says nothing about where the median of this weight's
    plans is.

    `plans` are the sweep's plan records. A plan that was refused or
    sentinelled carries no band: it is excluded and counted, the standing
    rule for a refused measurement.
    """
    from alphagrad.approx.common.pareto_archive import median_band

    rows = {}
    for p in plans:
        key = tuple(round(float(x), 6) for x in p["w"])
        row = rows.setdefault(key, {"w": key, "n": 0, "refused": 0,
                                    "lat": [], "mem": [], "feasible": 0,
                                    "quality": []})
        if p.get("refused") or p.get("band") is None:
            row["refused"] += 1
            continue
        row["n"] += 1
        band = p["band"]
        row["lat"].append(float(band["latency"]["median"]))
        row["mem"].append(float(band["memory"]["median"]))
        row["quality"].append(float(p["quality"]))
        row["feasible"] += int(float(p["quality"]) >= float(quality_floor))
    out = []
    for key in sorted(rows):
        r = rows[key]
        if not r["n"]:
            out.append({"w": list(key), "n": 0, "refused": r["refused"],
                        "latency_median": None, "memory_median": None,
                        "feasible_fraction": None})
            continue
        lat_med, lat_lo, lat_hi = median_band(r["lat"])
        mem_med, mem_lo, mem_hi = median_band(r["mem"])
        out.append({
            "w": list(key),
            "n": r["n"],
            "refused": r["refused"],
            "latency_median": float(lat_med),
            "latency_lo": float(lat_lo),
            "latency_hi": float(lat_hi),
            "memory_median": float(mem_med),
            "memory_lo": float(mem_lo),
            "memory_hi": float(mem_hi),
            "quality_median": float(np.median(r["quality"])),
            "feasible_fraction": float(r["feasible"]) / float(r["n"]),
        })
    return out


def write_plan_records(path: str, plans) -> None:
    """One JSON line per rollout: the weight, the plan, the band it measured."""
    with open(path, "w") as fh:
        for p in plans:
            fh.write(json.dumps(p) + "\n")


def load_plan_records(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def swept_front_from_plans(plans, objectives, cap: int = 64,
                           quality_floor=None):
    """A `RatioBandArchive` holding the sweep's non-dominated plans.

    The SAME class the trainer's archive is, fed the same per-window paired
    log ratios, so the file it dumps is the file `tools/landscape_map.py
    --archive` already reads.
    """
    from alphagrad.approx.common.pareto_archive import RatioBandArchive

    archive = RatioBandArchive(obj_names=tuple(objectives), cap=int(cap),
                               quality_floor=quality_floor)
    for i, p in enumerate(plans):
        if p.get("refused") or p.get("band") is None:
            continue
        dist = {objectives[0]: p["band"]["latency"]["windows"],
                objectives[1]: p["band"]["memory"]["windows"]}
        archive.add(dist, p["seq"], int(p.get("episode", i)),
                    quality=float(p["quality"]))
    return archive
