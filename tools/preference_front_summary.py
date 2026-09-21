"""The five seeds' preference sweeps, read as one set (dsnn-dfw.86).

Takes the per-seed `preference_front_comparison.json` files, re-reads the
fronts they name, and writes:

* `preference_sweep_summary.md` -- one table per weight, POOLED over the
  seeds, and one table per seed with the coverages, the front sizes and the
  hypervolumes.
* `preference_sweep_summary.json` -- the same numbers, plus the ONE nadir.

THE NADIR IS TAKEN OVER ALL THE SEEDS AT ONCE. Each per-seed comparison has
its own, which is right for that pair and wrong for five: a hypervolume is a
volume above a corner, and five volumes above five different corners are not
comparable. So every hypervolume here is recomputed above the corner of the
whole set, and that corner is in the output.

Pure host arithmetic over JSON. CPU node, no JAX.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from alphagrad.approx.common import preference_sweep as psweep


#: What each window reading means, on the table that uses it.
WINDOW_NOTE = {
    "admitted": "A point is late when it ENTERED the front inside the "
                "window.",
    "members": "A point is late when a MEMBER PLAN of it measured inside its "
               "band inside the window (owner ruling 2026-09-21). A plan "
               "re-measured late counts as late, so this set contains the "
               "admitted one.",
}


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", action="append", default=[],
                   help="a per-seed preference_front_comparison.json; repeat")
    p.add_argument("--archive", action="append", default=[],
                   help="SEED=path/to/pareto_front.json for a seed whose "
                        "sweep has not run yet. The row then carries the "
                        "archive columns and the swept ones read '-', so the "
                        "archive side of the set can be read before the GPU "
                        "jobs land. Repeat.")
    p.add_argument("--archive-window", type=int, default=200,
                   help="the window for --archive rows; a --comparison row "
                        "keeps the window its own run used")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--objectives", default="latency,peak_memory")
    p.add_argument("--quality-floor", type=float, default=0.9)
    return p


def _load(path, objectives):
    with open(path) as fh:
        cmp_doc = json.load(fh)
    with open(cmp_doc["swept_front"]) as fh:
        swept_doc = json.load(fh)
    with open(cmp_doc["archive_front"]) as fh:
        archive_doc = json.load(fh)
    window = int(cmp_doc["archive_window_episodes"])
    out = {
        "comparison": cmp_doc,
        "seed": int(cmp_doc.get("seed", -1)),
        "plans": psweep.load_plan_records(cmp_doc["plans"]),
        "swept": psweep.front_points(swept_doc, objectives),
        "archive": psweep.front_points(archive_doc, objectives),
        "archive_window_episodes": window,
    }
    out.update(_windows(archive_doc, window, objectives))
    return out


def _windows(archive_doc, window, objectives) -> dict:
    """Both readings of the window (owner ruling 2026-09-21), as points."""
    out = {}
    for by in psweep.WINDOW_SETS:
        doc = psweep.front_window(archive_doc, int(window), by=by)
        out[f"late_{by}"] = psweep.front_points(doc, objectives)
        out["archive_window_since_episode"] = int(doc["window_since_episode"])
    return out


def _load_archive_only(spec, window, objectives):
    """A seed whose sweep has not run: the archive side of the row only."""
    if "=" not in spec:
        raise ValueError(f"--archive wants SEED=PATH, got {spec!r}")
    seed, path = spec.split("=", 1)
    with open(path) as fh:
        archive_doc = json.load(fh)
    out = {
        "comparison": None,
        "seed": int(seed),
        "plans": [],
        "swept": np.empty((0, len(objectives)), dtype=np.float64),
        "archive": psweep.front_points(archive_doc, objectives),
        "archive_window_episodes": int(window),
    }
    out.update(_windows(archive_doc, window, objectives))
    return out


def seed_table(rows, by: str) -> str:
    """One table per window reading. `by` names the set every column uses."""
    lines = [
        f"| seed | swept | archive | late ({by}) | C(swept,arch) | "
        f"C(arch,swept) | C(swept,late) | C(late,swept) | HV swept | "
        f"HV archive | HV late ({by}) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for r in rows:
        if not r["swept_measured"]:
            lines.append(
                f"| {r['seed']} | - | {r['num_archive']} | "
                f"{r[f'num_late_{by}']} | - | - | - | - | - | "
                f"{r['hypervolume_archive']:.4g} | "
                f"{r[f'hypervolume_late_{by}']:.4g} |")
            continue
        lines.append(
            f"| {r['seed']} | {r['num_swept']} | {r['num_archive']} | "
            f"{r[f'num_late_{by}']} | {r['coverage_swept_of_archive']:.2f} | "
            f"{r['coverage_archive_of_swept']:.2f} | "
            f"{r[f'coverage_swept_of_late_{by}']:.2f} | "
            f"{r[f'coverage_late_{by}_of_swept']:.2f} | "
            f"{r['hypervolume_swept']:.4g} | "
            f"{r['hypervolume_archive']:.4g} | "
            f"{r[f'hypervolume_late_{by}']:.4g} |")
    return "\n".join(lines) + "\n"


def weight_table(table, objectives) -> str:
    lines = [
        f"| w (latency, memory) | plans | seeds | {objectives[0]} median "
        f"[90% band] | {objectives[1]} median [90% band] | quality median | "
        f"feasible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in table:
        if not r["n"]:
            lines.append(f"| {r['w'][0]:.2f}, {r['w'][1]:.2f} | 0 | "
                         f"{r.get('seeds', 0)} | - | - | - | - |")
            continue
        lines.append(
            f"| {r['w'][0]:.2f}, {r['w'][1]:.2f} | {r['n']} | "
            f"{r.get('seeds', 0)} | "
            f"{r['latency_median']:+.4f} [{r['latency_lo']:+.4f}, "
            f"{r['latency_hi']:+.4f}] | "
            f"{r['memory_median']:+.4f} [{r['memory_lo']:+.4f}, "
            f"{r['memory_hi']:+.4f}] | {r['quality_median']:.4f} | "
            f"{r['feasible_fraction']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    a = make_argparser().parse_args()
    objectives = tuple(x.strip() for x in a.objectives.split(","))
    if len(objectives) != 2:
        raise ValueError(f"--objectives wants two names, got {a.objectives!r}")
    runs = [_load(p, objectives) for p in a.comparison]
    runs += [_load_archive_only(s, a.archive_window, objectives)
             for s in a.archive]
    if not runs:
        raise ValueError("a summary needs at least one --comparison or "
                         "--archive")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"two inputs name the same seed {sorted(seeds)}. A summary over "
            f"five seeds that read one seed twice is not a summary over five "
            f"seeds.")
    ref = psweep.shared_nadir(*[r[k] for r in runs
                                for k in ("swept", "archive")])
    rows = []
    for r in sorted(runs, key=lambda x: x["seed"]):
        cmp_doc = r["comparison"] or {}
        row = {
            "seed": r["seed"],
            "swept_measured": r["comparison"] is not None,
            "num_swept": int(r["swept"].shape[0]),
            "num_archive": int(r["archive"].shape[0]),
            "coverage_swept_of_archive": psweep.set_coverage(
                r["swept"], r["archive"]),
            "coverage_archive_of_swept": psweep.set_coverage(
                r["archive"], r["swept"]),
            "hypervolume_swept": psweep.hypervolume_of(r["swept"], ref),
            "hypervolume_archive": psweep.hypervolume_of(r["archive"], ref),
            "rollouts": int(cmp_doc.get("rollouts", 0)),
            "rollouts_measured": int(cmp_doc.get("rollouts_measured", 0)),
            "archive_window_episodes": int(cmp_doc.get(
                "archive_window_episodes",
                r.get("archive_window_episodes", a.archive_window))),
            "archive_window_since_episode": int(cmp_doc.get(
                "archive_window_since_episode",
                r.get("archive_window_since_episode", -1))),
        }
        for by in psweep.WINDOW_SETS:
            late = r[f"late_{by}"]
            row[f"num_late_{by}"] = int(late.shape[0])
            row[f"coverage_swept_of_late_{by}"] = psweep.set_coverage(
                r["swept"], late)
            row[f"coverage_late_{by}_of_swept"] = psweep.set_coverage(
                late, r["swept"])
            row[f"hypervolume_late_{by}"] = psweep.hypervolume_of(late, ref)
        rows.append(row)
    pooled_plans = [p for r in runs for p in r["plans"]]
    pooled = psweep.per_weight_table(pooled_plans, a.quality_floor)
    by_w = {}
    for r in runs:
        for p in r["plans"]:
            key = tuple(round(float(x), 6) for x in p["w"])
            by_w.setdefault(key, set()).add(r["seed"])
    for row in pooled:
        row["seeds"] = len(by_w.get(tuple(row["w"]), ()))
    os.makedirs(a.out, exist_ok=True)
    doc = {
        "objectives": list(objectives),
        "seeds": sorted(seeds),
        "nadir": [float(x) for x in ref],
        "nadir_space": "maximisation (negated log ratios), over every swept "
                       "and archive front of every seed",
        "quality_floor": float(a.quality_floor),
        "rollouts": int(sum(r["rollouts"] for r in rows)),
        "rollouts_measured": int(sum(r["rollouts_measured"] for r in rows)),
        "per_seed": rows,
        "per_weight_pooled": pooled,
    }
    out_json = os.path.join(a.out, "preference_sweep_summary.json")
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2)
    swept_seeds = [r["seed"] for r in rows if r["swept_measured"]]
    body = (
        f"# The preference sweep over {len(rows)} condC NN256 seeds\n\n"
        f"Swept: {swept_seeds or 'none yet'}. "
        f"{doc['rollouts']} rollouts, {doc['rollouts_measured']} measured. "
        f"Quality floor {a.quality_floor:g}. A seed with no sweep yet carries "
        f"its archive columns and '-' in the swept ones. Every hypervolume is "
        f"above ONE nadir, {np.array2string(np.asarray(ref), precision=6)}, "
        f"in {doc['nadir_space']}.\n\n"
        f"## Per weight, pooled over the swept seeds\n\n"
        + weight_table(pooled, objectives)
        + "".join(f"\n## Per seed, the window taken by {by}\n\n"
                  + WINDOW_NOTE[by] + "\n\n" + seed_table(rows, by)
                  for by in psweep.WINDOW_SETS))
    out_md = os.path.join(a.out, "preference_sweep_summary.md")
    with open(out_md, "w") as fh:
        fh.write(body)
    print(body, flush=True)
    print(f"[summary] {out_json}\n[summary] {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
