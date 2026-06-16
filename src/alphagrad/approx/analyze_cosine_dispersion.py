"""Phase 0 (dispersion gate) — analysis. Reads measure_dispersion.py output and
decides GO/NO-GO for the distributional (CVaR) reward layer: does cosine genuinely
vary across inputs for configs that matter, enough that a tail-summary (CVaR) would
rank configs differently from the mean?

The cosine distribution across configs is bimodal (most configs sit at ~0 or ~1
on every input), so plain medians collapse to 0 and are uninformative. We therefore
gate on the TAIL of the dispersion distribution and on the contested (non-degenerate)
band specifically:

  GO if ALL of:
    (1) >=10% of usable configs have within-config std >= STD_THR (0.05): real
        per-input variation exists somewhere;
    (2) >=1 such dispersed config also lies on the 3-D non-dominated front (it would
        actually be deployed);
    (3) among NON-DEGENERATE configs (not pinned at 0/1), >=20% reorder by >=2
        positions between ordering-by-mean and ordering-by-CVaR_alpha.

  uv run python src/alphagrad/approx/analyze_cosine_dispersion.py --dir ~/dsnn/train_exp/dispersion
"""
import os, glob, json, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/dsnn/train_exp/dispersion"))
ap.add_argument("--alpha", type=float, default=0.10, help="CVaR lower-tail fraction")
ap.add_argument("--std-thr", type=float, default=0.05, help="within-config std threshold for 'dispersed'")
ap.add_argument("--out", default=os.path.expanduser("~/dsnn/morl_fronts/plots/cosine_dispersion.png"))
A = ap.parse_args()


def nondominated(P):
    nd = np.ones(len(P), dtype=bool)
    for i in range(len(P)):
        if not nd[i]:
            continue
        dom = (P >= P[i]).all(1) & (P > P[i]).any(1); dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


def rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(int)


def cvar(c, a):
    s = np.sort(c); k = max(1, int(np.floor(a * len(s))))
    return float(s[:k].mean())


# ---- load ----
rows = []
for f in glob.glob(os.path.join(A.dir, "d_*.jsonl")):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        c = d.get("cos_per_point") or []
        if len(c) >= 4 and np.isfinite(d["latency_ns"]) and np.isfinite(d["xla_peak_memory"]):
            d["_c"] = np.asarray(c, float); rows.append(d)
if not rows:
    raise SystemExit(f"no usable measurements in {A.dir}")

C = [r["_c"] for r in rows]
m = np.array([c.mean() for c in C])
sd = np.array([c.std() for c in C])
cv = np.array([cvar(c, A.alpha) for c in C])
lat = np.array([r["latency_ns"] for r in rows])
xla = np.array([r["xla_peak_memory"] for r in rows])
n = len(rows)

# ---- distribution diagnostics ----
deg0 = m < 0.05; deg1 = m > 0.95; mid = ~deg0 & ~deg1
dispersed = sd >= A.std_thr
frac_disp = float(dispersed.mean())

# ---- 3-D non-dominated front (maximize latency_reward, xla_reward, mean cosine) ----
front = np.where(nondominated(np.column_stack([lat, xla, m])))[0]
front_disp = int(dispersed[front].sum())

# ---- reorder among NON-DEGENERATE configs: mean-order vs CVaR-order ----
nd_band = mid | dispersed                       # configs CVaR could plausibly move
idx = np.where(nd_band)[0]
if len(idx) > 1:
    rm = rank(m[idx]); rc = rank(cv[idx])
    reorder = float(np.mean(np.abs(rm - rc) >= 2))
else:
    reorder = 0.0

go = (frac_disp >= 0.10) and (front_disp >= 1) and (reorder >= 0.20)

print("================ Phase 0: cosine dispersion gate (robust) ================")
print(f" usable configs = {n}   front size = {len(front)}   CVaR alpha = {A.alpha}  std-thr = {A.std_thr}")
print(f" mean-cosine bands:  ~0 (<0.05): {int(deg0.sum())}   mid (0.05-0.95): {int(mid.sum())}   ~1 (>0.95): {int(deg1.sum())}")
print(f" within-config std:  median={np.median(sd):.4f}  mean={sd.mean():.4f}  p90={np.quantile(sd,0.9):.4f}  max={sd.max():.4f}")
print(f"   dispersed configs (std>={A.std_thr}):  {int(dispersed.sum())}/{n} = {frac_disp:.2f}   (need >=0.10)")
print(f"   dispersed AND on front: {front_disp}   (need >=1)")
print(f" mid-band/dispersed reorder (mean-order vs CVaR_{A.alpha}-order, |Δrank|>=2): {reorder:.2f}  (need >=0.20)   over n={len(idx)}")
if mid.sum():
    print(f" mid-band std: median={np.median(sd[mid]):.4f}  max={sd[mid].max():.4f}   dispersed in mid: {int(dispersed[mid].sum())}/{int(mid.sum())}")
od = np.argsort(-sd)[:6]
print(" most-dispersed configs (mean / std / min / max / CVaR):")
for i in od:
    print(f"   mean={m[i]:.3f} std={sd[i]:.3f} min={C[i].min():.3f} max={C[i].max():.3f} cvar={cv[i]:.3f}")
print(f"\n  >>> GATE: {'GO — dispersion is real where it matters; build the CVaR layer' if go else 'NO-GO — cosine is near-deterministic per config; CVaR adds nothing, use mean cosine'} <<<")

# ---- plot ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Phase 0 cosine dispersion gate — {'GO' if go else 'NO-GO'}  "
             f"(dispersed frac {frac_disp:.2f}, on-front {front_disp}, reorder {reorder:.2f})")
ax[0].hist(sd, bins=40, color="tab:blue"); ax[0].axvline(A.std_thr, color="k", ls="--", label=f"std={A.std_thr}")
ax[0].set_yscale("log"); ax[0].set_xlabel("within-config cosine std"); ax[0].set_ylabel("# configs (log)")
ax[0].set_title("dispersion distribution"); ax[0].legend(fontsize=8)
sc = ax[1].scatter(m, sd, s=14, c=dispersed, cmap="coolwarm")
ax[1].scatter(m[front], sd[front], s=60, facecolors="none", edgecolors="k", label="front")
ax[1].axhline(A.std_thr, color="k", ls="--", lw=0.8)
ax[1].set_xlabel("mean cosine"); ax[1].set_ylabel("within-config std"); ax[1].set_title("dispersion vs mean (○=front)"); ax[1].legend(fontsize=8)
ax[2].scatter(m[nd_band], cv[nd_band], s=16, c="tab:purple"); lim = [min(m[nd_band].min(), cv[nd_band].min()), 1.0] if len(idx) else [0, 1]
ax[2].plot(lim, lim, "k--", lw=0.8, label="mean=CVaR")
ax[2].set_xlabel("mean cosine"); ax[2].set_ylabel(f"CVaR_{A.alpha} cosine"); ax[2].set_title("mean vs CVaR (non-degenerate)"); ax[2].legend(fontsize=8)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(A.out, dpi=140); plt.close(fig)
print(f"\n saved plot -> {A.out}")
json.dump(dict(n=n, front_size=len(front), frac_dispersed=frac_disp, front_dispersed=front_disp,
               reorder=reorder, n_mid=int(mid.sum()), std_median=float(np.median(sd)),
               std_max=float(sd.max()), alpha=A.alpha, std_thr=A.std_thr, go=bool(go)),
          open(os.path.join(A.dir, "gate_result.json"), "w"), indent=2)
