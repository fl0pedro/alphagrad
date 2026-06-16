"""Define the Pareto fronts. (A) Representative points of the re-stated MORL fronts.
(B) The non-dominated front discovered by the 190k random search-space sweep, over the
same objectives [latency_ns, xla_peak_memory, cosine_sim] (maximization; latency/memory
stored negated). Aggregates the sweep's multi-pass measurements per sequence (mean),
computes the 3-D maxima via an O(n log n) BIT sweep, prints + saves + plots it.
  uv run python src/alphagrad/approx/define_fronts.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FR = os.path.expanduser("~/dsnn/morl_fronts")
SWEEP = os.path.expanduser("~/dsnn/search_space_grad/results_seed12345.jsonl")
OUT = os.path.expanduser("~/dsnn/morl_fronts/plots"); os.makedirs(OUT, exist_ok=True)
us = lambda l: -l / 1e3
mb = lambda m: -m / 1e6


# ---------- (A) MORL re-stated front representative points ----------
print("############ (A) MORL re-stated fronts ############")
for s in ("cmorl", "mogfn"):
    nf = json.load(open(f"{FR}/{s}_front_recos.json"))["front"]
    P = [(p["obj"]["latency_ns"], p["obj"]["xla_peak_memory"], p["obj"]["cosine_sim"],
          p["obj"].get("cosine_sim_old"), len(p["seq"])) for p in nf]
    print(f"\n=== {s.upper()} front: {len(P)} points ===")
    for nm, p in (("best cosine", max(P, key=lambda x: x[2])),
                  ("best latency", max(P, key=lambda x: x[0])),
                  ("best memory", max(P, key=lambda x: x[1]))):
        print(f" {nm:11s}: cos={p[2]:.3f}(old {p[3]:.3f})  lat={us(p[0]):9.1f}us  mem={mb(p[1]):6.2f}MB  vtx={p[4]}")
    A = np.array([(us(l), mb(m), c) for l, m, c, _, _ in P], float)
    nz = (A - A.min(0)) / (np.ptp(A, 0) + 1e-9); nz[:, 0] = 1 - nz[:, 0]; nz[:, 1] = 1 - nz[:, 1]
    p = P[int(nz.sum(1).argmax())]
    print(f" knee(bal.) : cos={p[2]:.3f}(old {p[3]:.3f})  lat={us(p[0]):9.1f}us  mem={mb(p[1]):6.2f}MB  vtx={p[4]}")
    print(f" cos>0.9: {sum(1 for x in P if x[2]>0.9)}/{len(P)}   cos<0.05: {sum(1 for x in P if x[2]<0.05)}/{len(P)}")


# ---------- (B) sweep front ----------
print("\n############ (B) random-sweep Pareto front ############")
agg = {}  # idx -> [sum_lat, sum_xla, sum_cos, n]
rows = 0
for line in open(SWEEP):
    try:
        d = json.loads(line)
    except Exception:
        continue
    rows += 1
    if d.get("invalid"):
        continue
    rv = d.get("reward_vec")
    if not rv or not np.isfinite(rv[6]):
        continue
    a = agg.setdefault(d["idx"], [0.0, 0.0, 0.0, 0])
    a[0] += rv[2]; a[1] += rv[8]; a[2] += rv[6]; a[3] += 1
idxs = [i for i, a in agg.items() if a[3] > 0]
lat = np.array([agg[i][0] / agg[i][3] for i in idxs])   # negated (higher=better)
xla = np.array([agg[i][1] / agg[i][3] for i in idxs])
cos = np.array([agg[i][2] / agg[i][3] for i in idxs])
print(f" sweep rows read={rows}  valid sequences={len(idxs)}")


def maxima3d(o0, o1, o2):
    """Non-dominated (maximization) mask via BIT sweep. O(n log n)."""
    n = len(o0)
    order = np.lexsort((-o2, -o1, -o0))  # o0 desc primary
    # compress o1 ascending, then reverse so 'o1 >= x' is a prefix
    r1 = np.searchsorted(np.unique(o1), o1) + 1
    m = int(r1.max()); r1 = m - r1 + 1
    tree = np.full(m + 1, -np.inf)
    def upd(i, v):
        while i <= m:
            if v > tree[i]: tree[i] = v
            i += i & -i
    def qry(i):
        r = -np.inf
        while i > 0:
            if tree[i] > r: r = tree[i]
            i -= i & -i
        return r
    nd = np.zeros(n, dtype=bool)
    o0s = o0[order]
    # process in groups of equal o0 (query strictly-greater o0 only)
    i = 0
    while i < n:
        j = i
        while j < n and o0s[j] == o0s[i]:
            j += 1
        grp = order[i:j]
        for k in grp:
            if qry(r1[k]) < o2[k] - 1e-12:   # nobody with o1>= and o2>= seen yet
                nd[k] = True
        for k in grp:
            upd(r1[k], o2[k])
        i = j
    return nd


nd = maxima3d(lat, xla, cos)
fi = np.where(nd)[0]
print(f" SWEEP FRONT size={len(fi)} (non-dominated over {len(idxs)} sampled sequences)")
P = [(lat[i], xla[i], cos[i]) for i in fi]
for nm, key in (("best cosine", lambda x: x[2]), ("best latency", lambda x: x[0]), ("best memory", lambda x: x[1])):
    p = max(P, key=key)
    print(f" {nm:11s}: cos={p[2]:.3f}  lat={us(p[0]):9.1f}us  mem={mb(p[1]):6.2f}MB")
A = np.array([(us(l), mb(m), c) for l, m, c in P], float)
nz = (A - A.min(0)) / (np.ptp(A, 0) + 1e-9); nz[:, 0] = 1 - nz[:, 0]; nz[:, 1] = 1 - nz[:, 1]
p = P[int(nz.sum(1).argmax())]
print(f" knee(bal.) : cos={p[2]:.3f}  lat={us(p[0]):9.1f}us  mem={mb(p[1]):6.2f}MB")
print(f" front cos>0.9: {sum(1 for x in P if x[2]>0.9)}/{len(P)}")

json.dump(dict(objectives=["latency_ns", "xla_peak_memory", "cosine_sim"], num_points=len(fi),
               note="non-dominated front over the 190k random sweep (per-seq mean over valid passes)",
               front=[dict(idx=int(idxs[i]), latency_ns=float(lat[i]), xla_peak_memory=float(xla[i]),
                           cosine_sim=float(cos[i])) for i in fi]),
          open(f"{FR}/sweep_front.json", "w"))

# plot sweep front
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle(f"Random-sweep Pareto front ({len(fi)} pts of {len(idxs)} sampled sequences)")
ax[0].scatter(us(lat), cos, s=2, c="lightgray", alpha=.3)
ax[0].scatter(us(lat[fi]), cos[fi], s=14, c="tab:green")
ax[0].set_xscale("log"); ax[0].set_xlabel("latency µs"); ax[0].set_ylabel("cosine_sim"); ax[0].set_title("cosine vs latency")
ax[1].scatter(mb(xla), cos, s=2, c="lightgray", alpha=.3)
ax[1].scatter(mb(xla[fi]), cos[fi], s=14, c="tab:green")
ax[1].set_xscale("log"); ax[1].set_xlabel("xla_peak MB"); ax[1].set_ylabel("cosine_sim"); ax[1].set_title("cosine vs memory")
ax[2].scatter(us(lat[fi]), mb(xla[fi]), s=14, c=cos[fi], cmap="viridis")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("latency µs"); ax[2].set_ylabel("xla_peak MB")
ax[2].set_title("front: latency vs memory (color=cosine)")
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(f"{OUT}/sweep_front.png", dpi=140); plt.close(fig)
print(f"\n saved {OUT}/sweep_front.png  +  {FR}/sweep_front.json")
