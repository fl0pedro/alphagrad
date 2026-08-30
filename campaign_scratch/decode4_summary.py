"""decode4_summary: four-pool comparison, 5 seeds, train beside test.

Within-step R2 is the metric (per-vertex signal is constant inside a step
group and contributes exactly 0).  Everything is reported as a DISTRIBUTION
over seeds -- a flat best-so-far after one early hit is luck, not a result.
"""
import json, sys, os, glob
import numpy as np

THR = [0.6, 0.8, 0.9]
POOLS = ["mean", "last", "sumtok", "attn"]

# The published bars, from the prior decode3 arm-A table (within-step R2).
BAR = {
    "ln_factor": (-0.185, 0.429, 0.474, 0.600),
    "n_diag":    (-0.146, 0.391, 0.475, 0.477),
    "n_paired":  (0.020, 0.536, 0.569, 0.498),
    "n_comp":    (0.117, 0.577, 0.612, 0.557),
    "ln_stored": (0.095, 0.719, 0.741, 0.671),
}

runs = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "face_*_s*.json"))):
    D = json.load(open(path))
    if not D.get("curve"):
        continue
    a = D["args"]
    runs.setdefault(a["pool"], []).append(D)

if not runs:
    print("no runs found"); raise SystemExit(0)

names = list(list(runs.values())[0][0]["curve"][-1]["test"].keys())
degen = set(list(runs.values())[0][0]["curve"][-1].get("degenerate", []))


def final(D, nm, key):
    return D["curve"][-1][key][nm]["r2w"]


def steps_to(D, nm, th):
    for c in D["curve"]:
        v = c["test"][nm]["r2w"]
        if v == v and v >= th:
            return c["step"]
    return None


print("\n" + "=" * 100)
print("FINAL WITHIN-STEP R2 -- median over seeds [min, max], TEST | TRAIN")
print("=" * 100)
hdr = f"{'target':13s}" + "".join(f"{p:>21s}" for p in POOLS)
print(hdr)
for nm in names:
    if nm in degen:
        print(f"{nm:13s}    DEGENERATE (target identically constant)")
        continue
    line = f"{nm:13s}"
    for p in POOLS:
        if p not in runs:
            line += f"{'-':>21s}"; continue
        te = np.array([final(D, nm, "test") for D in runs[p]], float)
        tr = np.array([final(D, nm, "train") for D in runs[p]], float)
        line += f"  {np.nanmedian(te):6.3f}|{np.nanmedian(tr):6.3f} n{len(te)}"
    print(line)

print("\n" + "=" * 100)
print("SEED SPREAD (test within-step R2, min..max over seeds)")
print("=" * 100)
for nm in names:
    if nm in degen:
        continue
    line = f"{nm:13s}"
    for p in POOLS:
        if p not in runs:
            line += f"{'-':>21s}"; continue
        te = np.array([final(D, nm, "test") for D in runs[p]], float)
        line += f"  {np.nanmin(te):6.3f}..{np.nanmax(te):6.3f}   "
    print(line)

print("\n" + "=" * 100)
print("STEPS TO TEST WITHIN-STEP R2 >= 0.6 / 0.8 / 0.9  (median over seeds; "
      "'never' = at least one seed never reached it)")
print("=" * 100)
for nm in names:
    if nm in degen:
        continue
    line = f"{nm:13s}"
    for p in POOLS:
        if p not in runs:
            line += f"{'-':>26s}"; continue
        cell = []
        for th in THR:
            hits = [steps_to(D, nm, th) for D in runs[p]]
            cell.append("never" if any(h is None for h in hits)
                        else str(int(np.median([h for h in hits]))))
        line += "  " + "/".join(f"{c:>5s}" for c in cell)
    print(line)

print("\n" + "=" * 100)
print("DOES ANY POOL LIFT THE TOKENS-ONLY ARM TO THE *_sizes BAR?")
print("  columns: tokens-only(prior)  sizes_A  sizes_B  +extents-bar  then "
      "each pool's median")
print("=" * 100)
for nm, (t0, sA, sB, ex) in BAR.items():
    line = (f"{nm:13s} {t0:7.3f} {sA:7.3f} {sB:7.3f} {ex:7.3f}   |")
    for p in POOLS:
        if p not in runs:
            line += f"{'-':>9s}"; continue
        te = float(np.nanmedian([final(D, nm, "test") for D in runs[p]]))
        mark = "<" if te >= sA else " "
        line += f" {te:7.3f}{mark}"
    print(line)
print("  '<' = this pool's median reaches the arm-A *_sizes value, i.e. the "
      "information was already in the tokens and the mean was destroying it.")

print("\n" + "=" * 100)
print("DAG-AGNOSTICISM (from each run's own audit)")
print("=" * 100)
for p in POOLS:
    if p not in runs:
        continue
    f = runs[p][0].get("dag_agnostic_fails", [])
    print(f"  {p:8s} {'FAIL: ' + str(f) if f else 'PASS -- no trainable leaf is dimensioned by V, T, F or NSTEP'}")

print("\nwall (s), per pool:", {p: [int(D["curve"][-1]["wall"]) for D in runs[p]]
                               for p in runs})
