"""Restate the MORL Pareto front using the re-measured cosine_sim. Loads the dumped
candidates (latency_ns, xla_peak_memory kept) + the freshly re-measured cosine, then
recomputes the non-dominated set over [latency_ns, xla_peak_memory, cosine_sim] and
reports how the front changed vs the original dump.
  uv run python src/alphagrad/approx/remeasure_front.py --src cmorl
"""
import os, glob, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--rmdir", default=os.path.expanduser("~/dsnn/train_exp/remeasure"))
A = ap.parse_args()

cands = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{A.src}_allcand.json")))["candidates"]
rm = {}
for f in glob.glob(f"{A.rmdir}/{A.src}_*.jsonl"):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        rm[d["i"]] = d

n = len(cands)
have = sum(1 for i in range(n) if i in rm)
lat = np.array([c["obj"]["latency_ns"] for c in cands])
mem = np.array([c["obj"]["xla_peak_memory"] for c in cands])
cos_old = np.array([c["obj"]["cosine_sim"] for c in cands])
cos_new = np.array([rm[i]["cos_new"] if i in rm else np.nan for i in range(n)])
nan_new = int(np.sum(~np.isfinite(cos_new)))
# fall back to old cosine where re-measure missing/failed, so the front stays defined
cos_use = np.where(np.isfinite(cos_new), cos_new, cos_old)


def nondominated(P):
    """Maximization. Returns boolean mask of non-dominated rows."""
    nd = np.ones(len(P), dtype=bool)
    for i in range(len(P)):
        if not nd[i]:
            continue
        ge = (P >= P[i]).all(1); gt = (P > P[i]).any(1)
        dom = ge & gt; dom[i] = False
        if dom.any():
            nd[i] = False
    return nd


P_old = np.column_stack([lat, mem, cos_old])
P_new = np.column_stack([lat, mem, cos_use])
fo = nondominated(P_old)
fn = nondominated(P_new)


def hv_mc(P, mask, ref, hi, n_mc=400000, seed=0):
    rng = np.random.default_rng(seed)
    F = P[mask]
    S = rng.uniform(ref, hi, size=(n_mc, 3))
    dom = np.zeros(n_mc, dtype=bool)
    for f in F:
        dom |= (S <= f).all(1)   # point in box is "covered" if some front pt dominates it (>= for max)
    vol = np.prod(hi - ref)
    return float(dom.mean() * vol)


ref = np.minimum(P_old.min(0), P_new.min(0))
hi = np.maximum(P_old.max(0), P_new.max(0))
hv_old = hv_mc(P_old, fo, ref, hi)
hv_new = hv_mc(P_new, fn, ref, hi)

d = cos_use - cos_old
rescued = int(np.sum((cos_old < 0.5) & (cos_use > 0.9)))
demoted = int(np.sum((cos_old > 0.9) & (cos_use < 0.5)))
survivors = int(np.sum(fo & fn))
newcomers = int(np.sum(fn & ~fo))
dropped = int(np.sum(fo & ~fn))

print(f"\n================ {A.src.upper()}: front re-stated with re-measured cosine ================")
print(f" candidates={n}  re-measured={have}  (failed/NaN -> fell back to old cos: {nan_new})")
print(f"\n cosine_sim shift (new - old) over all candidates:")
print(f"   mean={d.mean():+.3f}  median={np.median(d):+.3f}  improved>0.05: {int(np.sum(d>0.05))}  worsened<-0.05: {int(np.sum(d<-0.05))}")
print(f"   RESCUED (old<0.5 -> new>0.9): {rescued}   DEMOTED (old>0.9 -> new<0.5): {demoted}")
print(f"   cosine mean: old={cos_old.mean():.3f} -> new={cos_use.mean():.3f}   max: old={cos_old.max():.3f} -> new={cos_use.max():.3f}")
print(f"\n Pareto front [latency_ns, xla_peak_memory, cosine_sim] (non-dominated over dumped candidates):")
print(f"   size: old={int(fo.sum())}  new={int(fn.sum())}")
print(f"   survivors (on both)={survivors}   newcomers (new only)={newcomers}   dropped (old only)={dropped}")
print(f"   hypervolume (MC, shared box): old={hv_old:.4g}  new={hv_new:.4g}  ({100*(hv_new-hv_old)/(hv_old+1e-30):+.1f}%)")
print(f"\n front cosine_sim distribution:")
print(f"   OLD front by old cos: mean={cos_old[fo].mean():.3f} min={cos_old[fo].min():.3f} max={cos_old[fo].max():.3f}")
print(f"   NEW front by new cos: mean={cos_use[fn].mean():.3f} min={cos_use[fn].min():.3f} max={cos_use[fn].max():.3f}")

# save new front (seqs + objectives)
new_front = [dict(i=int(i), seq=cands[i]["seq"],
                  obj=dict(latency_ns=float(lat[i]), xla_peak_memory=float(mem[i]),
                           cosine_sim=float(cos_use[i]), cosine_sim_old=float(cos_old[i])))
             for i in np.where(fn)[0]]
out = os.path.expanduser(f"~/dsnn/morl_fronts/{A.src}_front_recos.json")
json.dump(dict(objectives=["latency_ns", "xla_peak_memory", "cosine_sim"],
               num_points=len(new_front), hypervolume_mc=hv_new,
               note="front recomputed over dumped candidates with re-measured (current-impl) cosine_sim",
               front=new_front), open(out, "w"))
print(f"\n saved new front -> {out}")
