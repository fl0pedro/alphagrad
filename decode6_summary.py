"""decode6_summary: {shapes} x {extents} -- can per-equation SHAPES in the
token stream replace the explicit per-face EXTENTS side channel?

ONE variable changed against the settled decode4/decode5 comparison: the
dataset.  Arm A is the production ``IncrementalPathTokenizer``; arm B is the
``ShapeTokenizer`` subclass that states every equation's OUTPUT SHAPE.  The
decode4 four-pool A/B ran EVERY arm on $DA (decode4_run.sbatch:20) -- the
shapes dataset was never put in front of the pools.  That is the gap here.

METRIC: WITHIN-STEP R2 (per-vertex signal is constant inside a step group and
contributes exactly 0).  TRAIN is printed beside TEST everywhere: train 0.6 /
test 0.0 (overfitting) and train 0.4 / test 0.4 (under-training) are
indistinguishable from one curve, and this repo has misread one for the other.

``ln_maxdim`` is algebraically max(f_ext), so in any +extents arm it is a
TAUTOLOGY.  It is EXCLUDED from every headline mean and reported separately.
``n_compressed`` / ``is_lowrank`` are identically 0 on TLM -> DEGENERATE, never
reported as a NaN result.

Usage: decode6_summary.py DECODE6_OUT [DECODE5_OUT]
"""
import json, sys, os, glob
import numpy as np

THR = [0.6, 0.8, 0.9]
KEY = ["ln_factor", "n_diag", "n_paired", "n_comp", "ln_stored"]
CTRL = ["stat_ln_i", "stat_ln_j"]
TAUT = ["ln_maxdim"]

# decode4's *_sizes bar: the +extents range the owner deferred the side
# channel against.  An arm CLEARS the bar on a target if it reaches this.
BAR = dict(ln_factor=0.600, n_diag=0.477, n_paired=0.498, n_comp=0.557,
           ln_stored=0.671)
D4POOL = dict(ln_factor=-0.131, n_diag=-0.137, n_paired=0.066, n_comp=0.121,
              ln_stored=0.110)   # decode4 pool=mean, dsA, no extents

runs = {}


def add(path, ds_override=None):
    D = json.load(open(path))
    if not D.get("curve"):
        print("  (no curve, skipped)", os.path.basename(path))
        return
    a = D["args"]
    ds = ds_override or ("B" if "dsB" in a["data"] else "A")
    runs.setdefault((ds, a["pool"], int(a["mp"]), int(a["extents"])),
                    []).append(D)


for p in sorted(glob.glob(os.path.join(sys.argv[1], "d6_*.json"))):
    add(p)
if len(sys.argv) > 2:
    for p in sorted(glob.glob(os.path.join(sys.argv[2], "f5_*.json"))):
        add(p, "A")           # stage-B reference arms, same script, same seeds

if not runs:
    print("no runs found"); raise SystemExit(0)

any_run = list(runs.values())[0][0]
names = list(any_run["curve"][-1]["test"].keys())
degen = set(any_run["curve"][-1].get("degenerate", []))
ARMS = sorted(runs, key=lambda k: (k[0], k[2], k[3], k[1]))


def tag(k):
    return f"{k[0]}/{k[1][:4]}/mp{k[2]}/ex{k[3]}"


def final(D, nm, key):
    return D["curve"][-1][key][nm]["r2w"]


def med(k, nm, key="test"):
    v = [final(D, nm, key) for D in runs[k]]
    return (float(np.nanmedian(v)), float(np.nanmin(v)), float(np.nanmax(v)),
            len(v))


def score(k, key="test"):
    return float(np.nanmean([med(k, nm, key)[0] for nm in KEY]))


def steps_to(D, nm, th):
    for c in D["curve"]:
        v = c["test"][nm]["r2w"]
        if v == v and v >= th:
            return c["step"]
    return None


print("\nARMS FOUND (dataset/pool/mp/extents -> n seeds):")
for k in ARMS:
    print(f"  {tag(k):22s} n={len(runs[k])}  seeds "
          f"{sorted(D['args']['seed'] for D in runs[k])}")
print("  dataset A = production tokenizer (NO per-eqn shapes), "
      "B = ShapeTokenizer (per-eqn OUTPUT SHAPES stated)")
print("DEGENERATE on this graph:", sorted(degen) or "none")

print("\n" + "=" * 130)
print("HARNESS CONTROLS -- static per-endpoint targets must land ~0.97, "
      "otherwise the metric is broken, not the representation")
print("=" * 130)
print(f"{'target':13s}" + "".join(f"{tag(k):>22s}" for k in ARMS))
for nm in CTRL:
    print(f"{nm:13s}" + "".join(f"{med(k, nm)[0]:22.3f}" for k in ARMS))

print("\n" + "=" * 130)
print("FINAL WITHIN-STEP R2 @ last eval -- median over seeds, TEST | TRAIN")
print("  TEST << TRAIN = OVERFITTING (the information is not in the "
      "representation).  TEST ~= TRAIN and both low = UNDER-training.")
print("=" * 130)
print(f"{'target':13s}" + "".join(f"{tag(k):>22s}" for k in ARMS))
for nm in names:
    if nm in degen:
        print(f"{nm:13s}    DEGENERATE (identically constant on this graph)")
        continue
    line = f"{nm:13s}"
    for k in ARMS:
        line += f"   {med(k, nm, 'test')[0]:7.3f}|{med(k, nm, 'train')[0]:7.3f}"
    print(line + ("   <-- TAUTOLOGY in any +extents arm (== max f_ext)"
                  if nm in TAUT else ""))

print("\n" + "=" * 130)
print("SEED SPREAD, TEST within-step R2 on the five decision targets "
      "(min..max, n)")
print("=" * 130)
for nm in KEY:
    line = f"{nm:13s}"
    for k in ARMS:
        m, lo, hi, n = med(k, nm)
        line += f"  {lo:6.3f}..{hi:6.3f}n{n}"
    print(line)

print("\n" + "=" * 130)
print("STEPS TO TEST WITHIN-STEP R2 >= 0.6 / 0.8 / 0.9 "
      "(median over seeds; 'never' = at least one seed never got there)")
print("=" * 130)
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

print("\n" + "=" * 130)
print("ARM RANKING -- mean over the five decision targets "
      "(ln_maxdim EXCLUDED: tautological under extents)")
print("=" * 130)
for k in sorted(ARMS, key=score, reverse=True):
    print(f"  {tag(k):22s} TEST {score(k):7.3f}   TRAIN {score(k, 'train'):7.3f}"
          f"   n={len(runs[k])}")

print("\n" + "=" * 130)
print("MAIN EFFECTS + THE INTERACTION THAT DECIDES THE SIDE CHANNEL")
print("=" * 130)


def eff(pred):
    on = [score(k) for k in ARMS if pred(k)]
    off = [score(k) for k in ARMS if not pred(k)]
    if not on or not off:
        return None
    return float(np.mean(on) - np.mean(off))


for label, pred in [("SHAPES  B - A    ", lambda k: k[0] == "B"),
                    ("EXTENTS on - off ", lambda k: k[3] == 1),
                    ("MP      on - off ", lambda k: k[2] == 1)]:
    e = eff(pred)
    print(f"  main effect  {label} = "
          f"{'n/a' if e is None else format(e, '+.3f')}")
print()
for mp in (0, 1):
    for pool in ("mean", "attn", "sumtok", "last"):
        a = ("B", pool, mp, 0); b = ("A", pool, mp, 1)
        c = ("A", pool, mp, 0); dd = ("B", pool, mp, 1)
        if a in runs and b in runs:
            print(f"  THE QUESTION  pool={pool:6s} mp={mp}:  "
                  f"shapes+NOextents {score(a):6.3f}   vs   "
                  f"+extents reference {score(b):6.3f}   "
                  f"=> {'CLEARS' if score(a) >= score(b) else 'SHORT BY ' + format(score(b) - score(a), '.3f')}")
        if a in runs and c in runs:
            print(f"    shapes alone buys (B/ex0 - A/ex0): "
                  f"{score(a) - score(c):+.3f}")
        if dd in runs and b in runs:
            print(f"    shapes ON TOP OF extents (B/ex1 - A/ex1): "
                  f"{score(dd) - score(b):+.3f}")

print("\n" + "=" * 130)
print("AGAINST THE BAR, PER TARGET  (bar = decode4's *_sizes / +extents range;"
      " '<' = this arm reaches it)")
print("  cols: decode4 pool=mean dsA noext | BAR | then each arm's median")
print("=" * 130)
for nm in KEY:
    line = f"{nm:13s} {D4POOL[nm]:7.3f} {BAR[nm]:7.3f}  |"
    for k in ARMS:
        m = med(k, nm)[0]
        line += f" {m:6.3f}{'<' if m >= BAR[nm] else ' '}"
    print(line)
print("  header order: " + "  ".join(tag(k) for k in ARMS))

print("\nwall (s) per arm:",
      {tag(k): [int(D["curve"][-1]["wall"]) for D in runs[k]] for k in ARMS})
