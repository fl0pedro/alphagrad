import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = os.path.expanduser("~/dsnn/out/downstream")
files = {"jax_grad (exact)": "diag_factor_jax_grad.csv",
         "best_overall Diag×1": "diag_factor_best_overall.csv",
         "mulsmax Diag×3": "diag_factor_mulsmax.csv"}
def fcol(rows, c):
    out = []
    for r in rows:
        v = r.get(c)
        try: out.append(float(v))
        except (TypeError, ValueError): out.append(float("nan"))
    return out
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for label, fn in files.items():
    p = os.path.join(D, fn)
    if not os.path.exists(p): continue
    rows = list(csv.DictReader(open(p)))
    step = fcol(rows, "step")
    axes[0].plot(step, fcol(rows, "test_acc"), "-o", ms=3, label=label, lw=1.8)
    axes[1].plot(step, fcol(rows, "train_loss"), "-", label=label, lw=1.8)
    axes[2].plot(step, fcol(rows, "grad_cossim_vs_exact"), "-", label=label, lw=1.8)
for ax, t in zip(axes, ["test accuracy ↑", "train loss ↓", "grad cosine vs exact ↑"]):
    ax.set_xlabel("step"); ax.set_title(t); ax.grid(alpha=0.3); ax.legend(fontsize=9)
axes[0].axhline(0.1, ls="--", color="gray", lw=0.8)
fig.suptitle("Downstream MNIST training — diag_factor Pareto picks vs exact (2000 steps)", fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(D, "diag_factor_curves.png"), dpi=120); print("wrote")
