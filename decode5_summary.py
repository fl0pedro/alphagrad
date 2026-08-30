"""decode5_summary: the 2x2x2 {pool} x {mp} x {extents} factorial.

Within-step R2 is the metric (per-vertex signal is constant inside a step
group and contributes exactly 0).  Everything is a DISTRIBUTION over seeds --
a flat best-so-far after one early hit is luck, not a result -- and TRAIN is
printed beside TEST everywhere, because train 0.6 / test 0.0 (overfitting) and
train 0.4 / test 0.4 (under-training) are indistinguishable from one curve.
"""
import json, sys, os, glob, itertools
import numpy as np

THR = [0.6, 0.8, 0.9]
# the five live-contraction targets the decision is about, then the controls
KEY = ["ln_factor", "n_diag", "n_paired", "n_comp", "ln_stored"]
CTRL = ["stat_ln_i", "stat_ln_j"]

# decode4's measured pool A/B (5 seeds, within-step R2 TEST) and the *_sizes
# bar the owner deferred extents against.
D4 = {
    "ln_factor": dict(mean=-0.131, attn=-0.131, bar=0.600),
    "n_diag":    dict(mean=-0.137, attn=-0.151, bar=0.477),
    "n_paired":  dict(mean=0.066,  attn=0.078,  bar=0.498),
    "n_comp":    dict(mean=0.121,  attn=0.120,  bar=0.557),
    "ln_stored": dict(mean=0.110,  attn=0.116,  bar=0.671),
}

runs = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "f5_*.json"))):
    D = json.load(open(path))
    if not D.get("curve"):
        print("  (no curve, skipped)", os.path.basename(path))
        continue
    a = D["args"]
    runs.setdefault((a["pool"], int(a["mp"]), int(a["extents"])), []).append(D)

if not runs:
    print("no runs found"); raise SystemExit(0)

any_run = list(runs.values())[0][0]
names = list(any_run["curve"][-1]["test"].keys())
degen = set(any_run["curve"][-1].get("degenerate", []))
ARMS = [k for k in itertools.product(["attn", "mean"], [0, 1], [0, 1])
        if k in runs]


def tag(k):
    return f"{k[0]}/mp{k[1]}/ex{k[2]}"


def final(D, nm, key):
    return D["curve"][-1][key][nm]["r2w"]


def med(k, nm, key="test"):
    v = [final(D, nm, key) for D in runs[k]]
    return float(np.nanmedian(v)), float(np.nanmin(v)), float(np.nanmax(v)), len(v)


def steps_to(D, nm, th):
    for c in D["curve"]:
        v = c["test"][nm]["r2w"]
        if v == v and v >= th:
            return c["step"]
    return None


print("\nARMS FOUND (seeds):",
      {tag(k): len(v) for k, v in sorted(runs.items())})

print("\n" + "=" * 118)
print("HARNESS CONTROL -- static per-endpoint targets must land ~0.97 or the "
      "metric, not the representation, is broken")
print("=" * 118)
print(f"{'target':13s}" + "".join(f"{tag(k):>16s}" for k in ARMS))
for nm in CTRL:
    line = f"{nm:13s}"
    for k in ARMS:
        m = med(k, nm)[0]
        line += f"{m:16.3f}"
    print(line + ("   <-- CONTROL" if nm == CTRL[-1] else ""))

print("\n" + "=" * 118)
print("FINAL WITHIN-STEP R2 @ last eval -- median over seeds, TEST | TRAIN")
print("  TEST << TRAIN is OVERFITTING (the information is not in the "
      "representation); TEST ~= TRAIN and both low is UNDER-training.")
print("=" * 118)
print(f"{'target':13s}" + "".join(f"{tag(k):>16s}" for k in ARMS))
for nm in names:
    if nm in degen:
        print(f"{nm:13s}    DEGENERATE (target identically constant on this "
              f"graph -- reported as degenerate, NOT as a NaN result)")
        continue
    line = f"{nm:13s}"
    for k in ARMS:
        te = med(k, nm, "test")[0]; tr = med(k, nm, "train")[0]
        line += f"  {te:6.3f}|{tr:6.3f}"
    print(line)

print("\n" + "=" * 118)
print("SEED SPREAD on the five decision targets (TEST within-step R2, "
      "min..max, n seeds)")
print("=" * 118)
for nm in KEY:
    line = f"{nm:13s}"
    for k in ARMS:
        m, lo, hi, n = med(k, nm)
        line += f"  {lo:6.3f}..{hi:6.3f} n{n}"
    print(line)

print("\n" + "=" * 118)
print("STEPS TO TEST WITHIN-STEP R2 >= 0.6 / 0.8 / 0.9 (median over seeds; "
      "'never' = at least one seed never got there)")
print("=" * 118)
for nm in KEY + CTRL:
    line = f"{nm:13s}"
    for k in ARMS:
        cell = []
        for th in THR:
            hits = [steps_to(D, nm, th) for D in runs[k]]
            cell.append("never" if any(h is None for h in hits)
                        else str(int(np.median(hits))))
        line += "  " + "/".join(f"{c:>5s}" for c in cell)
    print(line)

print("\n" + "=" * 118)
print("MAIN EFFECTS on the five decision targets (mean over targets of the "
      "per-arm median TEST within-step R2)")
print("=" * 118)


def score(k):
    return float(np.nanmean([med(k, nm)[0] for nm in KEY]))


for k in sorted(ARMS, key=score, reverse=True):
    print(f"  {tag(k):16s} mean-over-5-targets  {score(k):7.3f}   "
          f"(train {np.nanmean([med(k, nm, 'train')[0] for nm in KEY]):6.3f})")


def eff(pred):
    on = [score(k) for k in ARMS if pred(k)]
    off = [score(k) for k in ARMS if not pred(k)]
    if not on or not off:
        return None
    return float(np.mean(on) - np.mean(off))


for label, pred in [("EXTENTS on - off ", lambda k: k[2] == 1),
                    ("MP      on - off ", lambda k: k[1] == 1),
                    ("POOL attn - mean ", lambda k: k[0] == "attn")]:
    e = eff(pred)
    print(f"  main effect  {label} = "
          f"{'n/a' if e is None else format(e, '+.3f')}")

# The question the owner actually asked: does MP add anything ON TOP OF
# extents?  That is an INTERACTION, not a main effect.
for p in ["attn", "mean"]:
    for ex in [0, 1]:
        a, b = (p, 1, ex), (p, 0, ex)
        if a in runs and b in runs:
            print(f"  MP effect at pool={p:4s} extents={ex}: "
                  f"{score(a) - score(b):+.3f}   "
                  f"({tag(a)} {score(a):.3f} vs {tag(b)} {score(b):.3f})")

print("\n" + "=" * 118)
print("AGAINST THE BAR -- decode4's failing pools and the *_sizes bar the "
      "owner deferred extents against")
print("  cols: decode4 mean | decode4 attn | *_sizes BAR | then each arm's "
      "median")
print("=" * 118)
for nm in KEY:
    r = D4[nm]
    line = f"{nm:13s} {r['mean']:7.3f} {r['attn']:7.3f} {r['bar']:7.3f}  |"
    for k in ARMS:
        m = med(k, nm)[0]
        line += f" {m:6.3f}{'<' if m >= r['bar'] else ' '}"
    print(line)
print("  '<' = this arm reaches the *_sizes bar, i.e. the face channel is "
      "actually decodable in that arm.")

print("\n" + "=" * 118)
print("DAG-AGNOSTICISM (each run's own parameter-shape audit)")
print("=" * 118)
for k in ARMS:
    f = runs[k][0].get("dag_agnostic_fails", [])
    print(f"  {tag(k):16s} "
          f"{'flagged: ' + str(f) if f else 'PASS -- no flagged leaf'}")
print("  NOTE: on THIS graph 3*embd_dim(96) == V(96) and OP_TYPE_VOCAB_SIZE"
      "(71) == NSTEP(71).  Both are coincidences; see the ALT-SHAPE RE-AUDIT "
      "in the log, which rebuilds every arm at V=137 / NSTEP=53 and compares "
      "leaf shapes elementwise.")

print("\nwall (s) per arm:",
      {tag(k): [int(D["curve"][-1]["wall"]) for D in runs[k]]
       for k in sorted(runs)})
