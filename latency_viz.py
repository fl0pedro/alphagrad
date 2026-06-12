"""Minimal matplotlib figures for the latency-measurement experiment.
Plain defaults, no styling. Data from /tmp/lat_1core.txt and /tmp/lat_Ncores.txt."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    txt = open(path).read()
    start = txt.find("\n=== JSON ===\n")
    return json.loads(txt[start + len("\n=== JSON ===\n"):])


rows = load_rows("/tmp/lat_1core.txt") + load_rows("/tmp/lat_Ncores.txt")


def mean_over(rows, **filt):
    sub = [r for r in rows if all(r[k] == v for k, v in filt.items())]
    return {m: float(np.mean([r[m] for r in sub])) for m in ("discrim", "spearman", "cv")}


# ── Fig 1: timer x cpu affinity ───────────────────────────────────────────────
combos = [("rm", "Ncores_unpinned", "RM\nN-core"),
          ("rm", "1core_pinned",    "RM\n1-core"),
          ("pc", "Ncores_unpinned", "PC\nN-core"),
          ("pc", "1core_pinned",    "PC\n1-core")]
disc = [mean_over(rows, timer=t, cpu_label=c)["discrim"] for t, c, _ in combos]
spear = [mean_over(rows, timer=t, cpu_label=c)["spearman"] for t, c, _ in combos]
labels = [l for _, _, l in combos]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.bar(labels, disc)
ax1.set_yscale("log")
ax1.set_ylabel("discriminability")
ax1.set_title("(a) discriminability")
ax2.bar(labels, spear)
ax2.set_ylabel("Spearman rho")
ax2.set_ylim(0, 1)
ax2.set_title("(b) ranking accuracy")
fig.tight_layout()
fig.savefig("/tmp/lat_fig1_paper.png", dpi=200)
plt.close(fig)


# ── Fig 2: cumulative technique gains ─────────────────────────────────────────
steps = [
    ("baseline",    mean_over(rows, timer="rm", cpu_label="Ncores_unpinned")["discrim"]),
    ("+1-core",     mean_over(rows, timer="rm", cpu_label="1core_pinned")["discrim"]),
    ("+perf_ctr",   mean_over(rows, timer="pc", cpu_label="1core_pinned")["discrim"]),
    ("+warmup+min", next(r["discrim"] for r in rows
                         if r["timer"] == "pc" and r["cpu_label"] == "1core_pinned"
                         and r["warmup"] == 3 and r["estimator"] == "min"
                         and r["inner_reps"] == 4)),
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(len(steps)), [s[1] for s in steps], "o-")
ax.set_yscale("log")
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([s[0] for s in steps])
ax.set_ylabel("discriminability")
ax.set_title("cumulative technique gains")
fig.tight_layout()
fig.savefig("/tmp/lat_fig2_paper.png", dpi=200)
plt.close(fig)


# ── Fig 3: estimator x cpu affinity ───────────────────────────────────────────
ests = ["min", "p60", "winsor20", "mean"]
sp_1c = [mean_over(rows, estimator=e, cpu_label="1core_pinned", timer="pc")["spearman"] for e in ests]
sp_nc = [mean_over(rows, estimator=e, cpu_label="Ncores_unpinned", timer="pc")["spearman"] for e in ests]
x = np.arange(len(ests))
# Crop the y-range to the data (all values cluster high) for readability.
_lo = np.floor((min(sp_1c + sp_nc) - 0.01) * 20) / 20
_hi = min(np.ceil((max(sp_1c + sp_nc) + 0.01) * 20) / 20, 1.0)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - 0.2, sp_1c, 0.4, label="1-core")
ax.bar(x + 0.2, sp_nc, 0.4, label="N-core")
ax.set_xticks(x)
ax.set_xticklabels(ests)
ax.set_ylabel("Spearman rho")
ax.set_ylim(_lo, _hi)
ax.set_title("estimator vs CPU affinity")
ax.legend()
fig.tight_layout()
fig.savefig("/tmp/lat_fig3_paper.png", dpi=200)
plt.close(fig)


# ── Fig 4: leaderboard (top 12) ───────────────────────────────────────────────
top = sorted(rows, key=lambda r: r["discrim"], reverse=True)[:12]
names = [f"{r['timer']} {'1c' if r['cpu_label']=='1core_pinned' else 'Nc'} "
         f"wu{r['warmup']} {r['estimator']}" for r in top]
vals = [r["discrim"] for r in top]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(range(len(top)), vals)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("discriminability")
ax.set_title("top 12 configurations")
fig.tight_layout()
fig.savefig("/tmp/lat_fig4_paper.png", dpi=200)
plt.close(fig)


# ── Summary 2x2 ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].bar(labels, disc); axes[0, 0].set_yscale("log")
axes[0, 0].set_ylabel("discriminability"); axes[0, 0].set_title("(a) timer x affinity")
axes[0, 1].plot(range(len(steps)), [s[1] for s in steps], "o-")
axes[0, 1].set_yscale("log"); axes[0, 1].set_xticks(range(len(steps)))
axes[0, 1].set_xticklabels([s[0] for s in steps], fontsize=8)
axes[0, 1].set_ylabel("discriminability"); axes[0, 1].set_title("(b) cumulative gains")
axes[1, 0].bar(x - 0.2, sp_1c, 0.4, label="1-core")
axes[1, 0].bar(x + 0.2, sp_nc, 0.4, label="N-core")
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(ests)
axes[1, 0].set_ylabel("Spearman rho"); axes[1, 0].set_ylim(_lo, _hi)
axes[1, 0].set_title("(c) estimator x affinity"); axes[1, 0].legend()
axes[1, 1].barh(range(len(top)), vals)
axes[1, 1].set_yticks(range(len(top))); axes[1, 1].set_yticklabels(names, fontsize=6)
axes[1, 1].invert_yaxis(); axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("discriminability"); axes[1, 1].set_title("(d) leaderboard")
fig.tight_layout()
fig.savefig("/tmp/lat_summary_paper.png", dpi=200)
plt.close(fig)

print("wrote 5 figures")
