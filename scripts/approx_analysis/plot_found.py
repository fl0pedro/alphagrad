"""Plot a variant's RL-found orders (10x8 re-measured) over the 256 sampler
search-space background, in cost-vs-accuracy space, and print the cost-Pareto
frontier picks (for downstream training). Run via SLURM CPU.

Usage: python plot_found.py --variant diag_factor
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
a = ap.parse_args()
FOUND = os.path.expanduser(f"~/dsnn/found_measured/{a.variant}.jsonl")
BG = os.path.expanduser("~/dsnn/search_space_full/results_seed12345.jsonl")
OUT = os.path.expanduser(f"~/dsnn/found_measured/{a.variant}_plot.png")

def agg_bg(path):
    acc = defaultdict(lambda: {"lat": [], "peak": [], "cos": [], "frob": [], "flops": None})
    for l in open(path):
        try: o = json.loads(l)
        except Exception: continue
        if o.get("invalid") or o.get("crashed") or o.get("fatal"): continue
        x = acc[o["idx"]]
        x["lat"] += o.get("latency_ns_samples", []); x["peak"] += o.get("peak_memory_samples", [])
        x["cos"] += [c for c in o.get("cosine_sim_per_point", []) if np.isfinite(c)]
        x["frob"] += [f for f in o.get("frob_residual_per_point", []) if np.isfinite(f)]
        if x["flops"] is None: x["flops"] = o.get("flops")
    F, L, P, C, FL = [], [], [], [], []
    for x in acc.values():
        if not (x["lat"] and x["cos"] and x["frob"]): continue
        F.append(float(np.clip(np.median(x["frob"]), 0, 2))); L.append(np.median(x["lat"]) / 1e6)
        P.append(np.median(x["peak"]) / 1e6); C.append(float(np.clip(np.median(x["cos"]), -1, 1)))
        FL.append(x["flops"])
    return map(np.array, (F, L, P, C, FL))

# found orders: aggregate per label (the 10x8)
acc = defaultdict(lambda: {"lat": [], "peak": [], "cos": [], "frob": [], "flops": None, "label": None, "nops": 0})
for l in open(FOUND):
    o = json.loads(l)
    if o.get("invalid"): continue
    x = acc[o["idx"]]
    x["lat"] += o.get("latency_ns_samples", []); x["peak"] += o.get("peak_memory_samples", [])
    x["cos"] += [c for c in o.get("cosine_sim_per_point", []) if np.isfinite(c)]
    x["frob"] += [f for f in o.get("frob_residual_per_point", []) if np.isfinite(f)]
    if x["flops"] is None: x["flops"] = o.get("flops"); x["label"] = o.get("label"); x["nops"] = o.get("n_ops")
fr, fl_, fp, fc, ffl, flab, fnops = [], [], [], [], [], [], []
for i, x in acc.items():
    if not (x["lat"] and x["cos"]): continue
    fr.append(float(np.clip(np.median(x["frob"]), 0, 2))); fl_.append(np.median(x["lat"]) / 1e6)
    fp.append(np.median(x["peak"]) / 1e6); fc.append(float(np.clip(np.median(x["cos"]), -1, 1)))
    ffl.append(x["flops"]); flab.append(x["label"]); fnops.append(x["nops"])
fr, fl_, fp, fc, ffl = map(np.array, (fr, fl_, fp, fc, ffl))
print(f"{a.variant}: {len(fr)} found orders re-measured; cosine range [{fc.min():.3f},{fc.max():.3f}]")

bF, bL, bP, bC, bFL = agg_bg(BG)

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
for ax, (by, fy, ylab, logy) in zip(axes, [
        (bL, fl_, "latency (ms) ↓", False),
        (bFL, ffl, "flops ↓", True),
        (bP, fp, "peak_memory (MB) ↓", False)]):
    ax.scatter(bF, by, s=8, alpha=0.12, color="gray", label="256 search space")
    sc = ax.scatter(fr, fy, c=fc, cmap="viridis", vmin=0, vmax=1, s=90,
                    edgecolors="k", linewidths=0.5, zorder=5, label=f"{a.variant} found")
    ax.set_xlabel("frob residual (0=exact)"); ax.set_ylabel(ylab)
    if logy: ax.set_yscale("log")
    ax.set_xlim(-0.05, 1.15); ax.grid(alpha=0.25); ax.legend(loc="upper right", fontsize=9)
fig.colorbar(sc, ax=axes, shrink=0.6, label="cosine_sim")
fig.suptitle(f"RL-found orders ({a.variant}) re-measured 10x8, over 256 search space", fontsize=14)
fig.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT)

# cost-Pareto over found orders (minimize latency & flops; cosine is the accuracy)
# pick non-dominated by (frob, latency) and report for training
idx = np.lexsort((fl_, fr)); front = []; best = np.inf
for i in idx:
    if fl_[i] < best - 1e-9: front.append(i); best = fl_[i]
print("\n=== cost-Pareto picks (frob, latency_ms, flops, cosine, nops, label) ===")
for i in front:
    print(f"  frob={fr[i]:.3f} lat={fl_[i]:.3f}ms flops={ffl[i]:.3g} cos={fc[i]:.3f} nops={fnops[i]} <- {flab[i]}")
# also the single cheapest-flops and the best_overall
chx = int(np.argmin(ffl))
print(f"cheapest-flops: {flab[chx]} flops={ffl[chx]:.3g} lat={fl_[chx]:.3f}ms cos={fc[chx]:.3f}")
