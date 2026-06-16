"""Plot the MORL Pareto front before vs after re-measuring cosine_sim. Per source:
cosine-vs-latency and cosine-vs-memory faces (old front, new front, candidate cloud),
the cosine_sim shift histogram, and a 3-D view. Objectives are stored negated
(higher=better); converted to positive latency(us)/memory(MB) for display.
  uv run python src/alphagrad/approx/plot_recos_front.py
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

FR = os.path.expanduser("~/dsnn/morl_fronts")
RM = os.path.expanduser("~/dsnn/train_exp/remeasure")
OUT = os.path.expanduser("~/dsnn/morl_fronts/plots"); os.makedirs(OUT, exist_ok=True)


def lat_us(o): return -o["latency_ns"] / 1e3
def mem_mb(o): return -o["xla_peak_memory"] / 1e6


def load(src):
    cands = json.load(open(f"{FR}/{src}_allcand.json"))["candidates"]
    rm = {}
    for f in glob.glob(f"{RM}/{src}_*.jsonl"):
        for line in open(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("cos_new") == d.get("cos_new"):
                rm[d["i"]] = d["cos_new"]
    c_lat = np.array([lat_us(c["obj"]) for c in cands])
    c_mem = np.array([mem_mb(c["obj"]) for c in cands])
    c_old = np.array([c["obj"]["cosine_sim"] for c in cands])
    c_new = np.array([rm.get(i, c_old[i]) for i in range(len(cands))])
    of = json.load(open(f"{FR}/{src}_front.json"))["front"]
    nf = json.load(open(f"{FR}/{src}_front_recos.json"))["front"]
    O = dict(lat=np.array([lat_us(p["obj"]) for p in of]), mem=np.array([mem_mb(p["obj"]) for p in of]),
             cos=np.array([p["obj"]["cosine_sim"] for p in of]))
    N = dict(lat=np.array([lat_us(p["obj"]) for p in nf]), mem=np.array([mem_mb(p["obj"]) for p in nf]),
             cos=np.array([p["obj"]["cosine_sim"] for p in nf]))
    return dict(c_lat=c_lat, c_mem=c_mem, c_old=c_old, c_new=c_new, O=O, N=N, n=len(cands))


def plot(src, color):
    D = load(src); O, N = D["O"], D["N"]
    fig = plt.figure(figsize=(14, 11)); tag = src.upper()
    fig.suptitle(f"{tag}: Pareto front [latency, xla_peak_memory, cosine_sim] — old vs re-measured cosine", fontsize=13)

    # 1) cosine vs latency
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(D["c_lat"], D["c_new"], s=3, c="lightgray", alpha=.4, label="all candidates (new cos)")
    ax.scatter(O["lat"], O["cos"], s=42, facecolors="none", edgecolors="tab:red", label=f"OLD front (n={len(O['cos'])})")
    ax.scatter(N["lat"], N["cos"], s=30, c="tab:blue", alpha=.8, label=f"NEW front (n={len(N['cos'])})")
    ax.set_xscale("log"); ax.set_xlabel("latency (µs, lower=left)"); ax.set_ylabel("cosine_sim")
    ax.set_title("cosine vs latency"); ax.legend(fontsize=8)

    # 2) cosine vs memory
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(D["c_mem"], D["c_new"], s=3, c="lightgray", alpha=.4)
    ax.scatter(O["mem"], O["cos"], s=42, facecolors="none", edgecolors="tab:red")
    ax.scatter(N["mem"], N["cos"], s=30, c="tab:blue", alpha=.8)
    ax.set_xscale("log"); ax.set_xlabel("xla_peak_memory (MB, lower=left)"); ax.set_ylabel("cosine_sim")
    ax.set_title("cosine vs memory")

    # 3) cosine shift histogram
    ax = fig.add_subplot(2, 2, 3)
    ax.hist(D["c_old"], bins=40, alpha=.5, color="tab:red", label=f"old (mean {D['c_old'].mean():.2f})")
    ax.hist(D["c_new"], bins=40, alpha=.5, color="tab:blue", label=f"new (mean {D['c_new'].mean():.2f})")
    ax.set_xlabel("cosine_sim"); ax.set_ylabel("# candidates"); ax.set_yscale("log")
    ax.set_title("cosine_sim distribution (all candidates)"); ax.legend(fontsize=8)

    # 4) 3D
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.scatter(np.log10(O["lat"]), O["mem"], O["cos"], s=30, c="tab:red", label="OLD front", depthshade=False)
    ax.scatter(np.log10(N["lat"]), N["mem"], N["cos"], s=30, c="tab:blue", label="NEW front", depthshade=False)
    ax.set_xlabel("log10 latency µs"); ax.set_ylabel("xla_peak MB"); ax.set_zlabel("cosine_sim")
    ax.set_title("3-D front"); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, .97])
    p = f"{OUT}/{src}_front_old_vs_new.png"; fig.savefig(p, dpi=140); plt.close(fig)
    print(f"saved {p}")


plot("cmorl", "tab:blue")
plot("mogfn", "tab:orange")
print("done")
