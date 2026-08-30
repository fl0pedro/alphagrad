"""decode2_summary: learning curves, TRAIN-vs-TEST and steps-to-threshold."""
import json, sys, glob, os
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "decode2_out"
THR = [0.6, 0.8, 0.9]


def s2t(curve, get, ths=None):
    out = {}
    for th in (ths or THR):
        hit = None
        for c in curve:
            v = get(c)
            if v == v and v >= th:
                hit = c["step"]; break
        out[th] = hit
    return out


print("=" * 100)
print("PART B -- VERTEX.  headline = DV_FILL within-step R2 on held-out "
      "trajectories (tALL)")
print("=" * 100)
rows = []
for f in sorted(glob.glob(os.path.join(R, "vert_*.json"))):
    d = json.load(open(f))
    c = d["curve"]
    nm = os.path.basename(f)[5:-5]
    g = lambda x: x["test"]["fill"]["tALL"]["r2w"]
    gt = lambda x: x["train"]["fill"]["tALL"]["r2w"]
    st = s2t(c, g)
    last = c[-1]
    rows.append((nm, last["step"], g(last), gt(last),
                 last["test"]["static_ln"]["tALL"]["r2w"],
                 last["test"]["max_jac"]["tALL"]["r2w"],
                 last["test"]["elim_nb"]["tALL"]["r2w"],
                 st, last["wall"]))
print(f"{'variant':14s} {'steps':>6s} {'fill test':>9s} {'fill train':>10s} "
      f"{'static_ln':>9s} {'max_jac':>8s} {'elim_nb':>8s}  steps->0.6/0.8/0.9")
for r in sorted(rows, key=lambda x: -(x[2] if x[2] == x[2] else -9)):
    print(f"{r[0]:14s} {r[1]:6d} {r[2]:9.3f} {r[3]:10.3f} {r[4]:9.3f} "
          f"{r[5]:8.3f} {r[6]:8.3f}   "
          + "/".join(str(r[7][t]) for t in THR) + f"   [{r[8]/60:.0f} min]")

for f in sorted(glob.glob(os.path.join(R, "vert_*.json"))):
    d = json.load(open(f)); c = d["curve"]
    nm = os.path.basename(f)[5:-5]
    print(f"\n-- {nm}: fill within-step R2 (test | train) per eval --")
    print("   " + "  ".join(
        f"{x['step']}:{x['test']['fill']['tALL']['r2w']:.2f}|"
        f"{x['train']['fill']['tALL']['r2w']:.2f}" for x in c))

print()
print("=" * 100)
print("PART A -- FACE.  pooled R2 on held-out trajectories")
print("=" * 100)
FKEY = ["ln_out", "ln_prim", "ln_stored", "gain", "ln_factor", "n_diag",
        "n_comp", "n_paired", "n_dims", "stat_ln_i", "stat_ln_j"]
base_printed = False
for f in sorted(glob.glob(os.path.join(R, "face_*.json"))):
    d = json.load(open(f)); c = d["curve"]; nm = os.path.basename(f)[5:-5]
    if not base_printed:
        print("\nNON-LEARNED BASELINES (test, pooled R2):")
        print(f"  {'baseline':20s} " + " ".join(f"{k:>10s}" for k in FKEY))
        for bn, br in d["baselines"].items():
            print(f"  {bn:20s} " + " ".join(f"{br[k]['r2']:10.3f}"
                                            for k in FKEY))
        base_printed = True
    last = c[-1]
    print(f"\n-- arm {nm} @ {last['step']} steps ({last['wall']/60:.0f} min) --")
    print(f"  {'target':13s} {'test R2':>8s} {'train R2':>9s} {'test within':>12s}"
          f" {'peak':>7s}   ->0.6/0.8/0.9 pooled | w->0.2/0.3/0.4")
    for k in FKEY:
        st = s2t(c, lambda x, k=k: x["test"][k]["r2"])
        stw = s2t(c, lambda x, k=k: x["test"][k]["r2w"], [0.2, 0.3, 0.4])
        pk = max(x["test"][k]["r2"] for x in c)
        print(f"  {k:13s} {last['test'][k]['r2']:8.3f} "
              f"{last['train'][k]['r2']:9.3f} {last['test'][k]['r2w']:12.3f}  "
              f"{pk:7.3f}  " + "/".join(str(st[t]) for t in THR)
              + "   w:" + "/".join(str(stw[t]) for t in [0.2, 0.3, 0.4]))

print("\n\nFACE learning curves (test pooled R2 of ln_factor / n_diag / "
      "n_paired / stat_ln_j):")
for f in sorted(glob.glob(os.path.join(R, "face_*.json"))):
    d = json.load(open(f)); c = d["curve"]; nm = os.path.basename(f)[5:-5]
    for k in ["ln_factor", "n_diag", "n_paired", "stat_ln_j"]:
        print(f"  {nm:16s} {k:10s} " + " ".join(
            f"{x['step']}:{x['test'][k]['r2']:.2f}" for x in c[::max(1, len(c)//12)]))
