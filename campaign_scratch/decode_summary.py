"""decode_summary: final table + learning curves from the decode_* json runs."""
import json, glob, os, sys
import numpy as np

R = "/Users/assmuth/dsnn"
runs = {}
for p in sorted(glob.glob(f"{R}/decode_[ABC]*.json")):
    tag = os.path.basename(p)[7:-5]
    try:
        runs[tag] = json.load(open(p))
    except Exception as e:
        print("skip", p, e)
oracle = json.load(open(f"{R}/decode_oracle.json"))
base = list(runs.values())[0]["baselines"]
KEYS = ["t0", "t20", "t50", "t70", "tALL"]
NAMES = list(oracle.keys())


def fmt(e, k, w=True):
    v = e.get(k, {}).get("r2w" if w else "r2", float("nan"))
    return "  n/a " if v != v else f"{v:6.3f}"


print("=" * 100)
print("WITHIN-STEP R^2 on HELD-OUT trajectories (the ranking-relevant contrast)")
print("=" * 100)
for nm in NAMES:
    print(f"\n--- {nm} ---")
    print(f"  {'model':28s} " + " ".join(f"{k:>7s}" for k in KEYS))
    rows = [("per-vertex constant", base["const_vertex"][nm]),
            ("per-(vertex,step) constant", base["const_vertex_step"][nm])]
    for tag, r in runs.items():
        last = r["curve"][-1]
        rows.append((f"TRAINED {tag} (test)", last["test"][nm]))
        rows.append((f"TRAINED {tag} (trainfit)", last["train"][nm]))
    rows.append(("CEILING: MLP(v, elim-set)", oracle[nm]))
    for lbl, e in rows:
        print(f"  {lbl:28s} " + " ".join(fmt(e, k) for k in KEYS))

print("\n" + "=" * 100)
print("LEARNING CURVES -- within-step R^2 (test) for `fill` and `elim_nb`")
print("=" * 100)
for tag, r in runs.items():
    print(f"\n{tag}:")
    print(f"  {'step':>7s} {'loss':>8s} {'fill tALL':>10s} {'fill t70':>9s} "
          f"{'elimnb tALL':>12s} {'fill TRAINFIT':>14s} {'wall_s':>8s}")
    for c in r["curve"]:
        print(f"  {c['step']:7d} {c['loss']:8.4f} "
              f"{c['test']['fill']['tALL']['r2w']:10.3f} "
              f"{c['test']['fill']['t70']['r2w']:9.3f} "
              f"{c['test']['elim_nb']['tALL']['r2w']:12.3f} "
              f"{c['train']['fill']['tALL']['r2w']:14.3f} {c['wall']:8.0f}")
