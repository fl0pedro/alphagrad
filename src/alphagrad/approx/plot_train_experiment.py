"""Plot the train-rules experiment: per-rule MNIST test accuracy (20 seeds) +
latency vs the reverse-mode baseline, 2-D Pareto cuts (cossim-latency,
cossim-memory), and a 3-D plot overlaying the all-candidate dump. Separate
figures for C-MORL and MOGFN. Plain matplotlib.

  uv run plot_train_experiment.py --exp ~/dsnn/train_exp --fronts ~/dsnn/morl_fronts --out ~/dsnn/train_exp/plots
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=os.path.expanduser("~/dsnn/train_exp"))
ap.add_argument("--fronts", default=os.path.expanduser("~/dsnn/morl_fronts"))
ap.add_argument("--out", default=os.path.expanduser("~/dsnn/train_exp/plots"))
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)


def load_source(src):
    rows, rev = [], None
    for f in glob.glob(os.path.join(A.exp, f"{src}_shard*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("results", []):
            if r.get("label") == "reverse_mode":
                rev = r
            elif "error" not in r and "acc_mean" in r:
                rows.append(r)
    rows.sort(key=lambda r: r.get("idx", 0))
    return rows, rev


# Reverse-mode grad latency measured under the SAME 1-core-affinity mechanism the
# front used (the experiment's own re-measured latency used a ~5x-slower thread-cap
# mechanism, so we use the front's affinity latency for rules and these for reverse).
_rl = os.path.expanduser("~/dsnn/train_exp/rev_lat.json")
REV_LAT_US = json.load(open(_rl)) if os.path.exists(_rl) else {"cmorl": None, "mogfn": None}


def lat_us(r):  # affinity latency (µs): front rules carry the front's (negated) cost
    if "front_obj" in r:
        return -r["front_obj"]["latency_ns"] / 1e3
    return r["lat_ns_mean"] / 1e3  # fallback (reverse handled via REV_LAT_US)


def cos(r):
    return r["front_obj"]["cosine_sim"]


def mem_mb(r):
    return -r["front_obj"]["xla_peak_memory"] / 1e6


def plot_source(src, color):
    rows, rev = load_source(src)
    if not rows:
        print(f"[{src}] no results yet"); return
    accs = np.array([r["acc_mean"] for r in rows])
    astd = np.array([r["acc_std"] for r in rows])
    lats = np.array([lat_us(r) for r in rows])
    lstd = np.array([r["lat_ns_std"] / 1e3 for r in rows])
    coss = np.array([cos(r) for r in rows])
    mems = np.array([mem_mb(r) for r in rows])
    n = len(rows)
    rev_acc = rev["acc_mean"] if rev else None
    rev_lat = REV_LAT_US.get(src)
    tag = src.upper()

    # 1) accuracy per rule (sorted by front cosine) with std
    o = np.argsort(coss)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(range(n), accs[o], yerr=astd[o], fmt="o", ms=4, color=color, capsize=2)
    if rev_acc is not None:
        ax.axhline(rev_acc, ls="--", color="k", label=f"reverse-mode ({rev_acc:.3f})")
    ax.set_xlabel("rule (sorted by gradient cosine)"); ax.set_ylabel("MNIST test accuracy")
    ax.set_title(f"{tag}: test accuracy per rule (mean±std, 20 seeds)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{A.out}/{src}_accuracy.png", dpi=160); plt.close(fig)

    # 2) latency per rule with std (log)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(range(n), lats[o], yerr=lstd[o], fmt="s", ms=4, color=color, capsize=2)
    if rev_lat is not None:
        ax.axhline(rev_lat, ls="--", color="k", label=f"reverse-mode ({rev_lat:.0f}µs)")
    ax.set_yscale("log"); ax.set_xlabel("rule (sorted by gradient cosine)")
    ax.set_ylabel("grad latency (µs)"); ax.set_title(f"{tag}: per-step latency (mean±std)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{A.out}/{src}_latency.png", dpi=160); plt.close(fig)

    # 3) accuracy vs gradient cosine (does cossim predict trainability?)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coss, accs, c=lats, cmap="viridis", norm=matplotlib.colors.LogNorm(), s=30)
    if rev_acc is not None:
        ax.axhline(rev_acc, ls="--", color="k", lw=0.8)
    ax.set_xlabel("gradient cosine_sim (front)"); ax.set_ylabel("MNIST test accuracy")
    ax.set_title(f"{tag}: accuracy vs gradient cosine"); fig.colorbar(sc, label="latency µs")
    fig.tight_layout(); fig.savefig(f"{A.out}/{src}_acc_vs_cos.png", dpi=160); plt.close(fig)

    # 4) 2-D Pareto cut: cosine vs latency, colored by accuracy
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(coss, lats, c=accs, cmap="plasma", s=34, vmin=0, vmax=1)
    ax.set_yscale("log"); ax.set_xlabel("gradient cosine_sim"); ax.set_ylabel("latency (µs)")
    ax.set_title(f"{tag}: Pareto cut — cosine vs latency"); fig.colorbar(sc, label="test acc")
    fig.tight_layout(); fig.savefig(f"{A.out}/{src}_pareto_cos_lat.png", dpi=160); plt.close(fig)

    # 5) 2-D Pareto cut: cosine vs memory
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(coss, mems, c=accs, cmap="plasma", s=34, vmin=0, vmax=1)
    ax.set_xlabel("gradient cosine_sim"); ax.set_ylabel("xla_peak (MB)")
    ax.set_title(f"{tag}: Pareto cut — cosine vs memory"); fig.colorbar(sc, label="test acc")
    fig.tight_layout(); fig.savefig(f"{A.out}/{src}_pareto_cos_mem.png", dpi=160); plt.close(fig)

    # 6) 3-D: cosine x latency x memory, front (acc-colored) + all-candidate dump (gray)
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    cf = os.path.join(A.fronts, f"{src}_allcand.json")
    if os.path.exists(cf):
        cand = json.load(open(cf)).get("candidates", [])
        cc = np.array([c["reward_vec"][6] for c in cand])
        cl = np.array([-c["reward_vec"][2] / 1e3 for c in cand])
        cm = np.array([-c["reward_vec"][8] / 1e6 for c in cand])
        good = np.isfinite(cc) & np.isfinite(cl) & np.isfinite(cm) & (cl > 0)
        ax.scatter(cc[good], np.log10(cl[good]), cm[good], c="lightgray", s=4, alpha=0.3, label="all candidates")
    p = ax.scatter(coss, np.log10(lats), mems, c=accs, cmap="plasma", s=36, vmin=0, vmax=1, label="front (trained)")
    ax.set_xlabel("cosine_sim"); ax.set_ylabel("log10 latency µs"); ax.set_zlabel("xla_peak MB")
    ax.set_title(f"{tag}: front (trained) over all candidates"); fig.colorbar(p, label="test acc", shrink=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(f"{A.out}/{src}_3d.png", dpi=160); plt.close(fig)

    print(f"[{src}] {n} rules plotted (reverse acc={rev_acc}, lat={rev_lat})")


plot_source("cmorl", "tab:blue")
plot_source("mogfn", "tab:orange")
print(f"figures -> {A.out}")
