import json, os, numpy as np
from collections import defaultdict
RES = os.path.expanduser("~/dsnn/search_space_full/results_seed12345.jsonl")
CH = ["muls_adds_fmas","flops","latency_ns","max_io_sum","bytes_accessed","peak_memory","cosine_sim","frob_residual"]
acc = defaultdict(lambda: {c: [] for c in ["lat","peak","cos","frob"]} | {"det": None})
vals = {c: [] for c in CH}
for l in open(RES):
    try: o = json.loads(l)
    except Exception: continue
    if o.get("invalid") or o.get("crashed") or o.get("fatal"): continue
    # per-config medians, then aggregate
    for c, key in [("latency_ns","latency_ns_samples"),("peak_memory","peak_memory_samples"),
                   ("cosine_sim","cosine_sim_per_point"),("frob_residual","frob_residual_per_point")]:
        s = [x for x in o.get(key, []) if x==x and abs(x)!=float("inf")]
        if s: vals[c].append(float(np.median(s)))
    for c in ["muls_adds_fmas","flops","max_io_sum","bytes_accessed"]:
        if o.get(c) is not None: vals[c].append(float(o[c]))
def symlog(x): return np.sign(x)*np.log1p(abs(x))
print(f"{'channel':16} {'median|raw|':>13} {'symlog(med)':>12} {'static λ=1/symlog':>17}")
lam = {}
for c in CH:
    if not vals[c]: continue
    med = float(np.median(np.abs(vals[c])))
    sl = symlog(med)
    l = (1.0/sl) if abs(sl) > 1e-6 else 0.0
    lam[c] = l
    print(f"{c:16} {med:>13.4g} {sl:>12.4g} {l:>17.4g}")
print("\nrewarded channels (latency_ns, peak_memory, frob_residual) suggested static λ:")
for c in ["latency_ns","peak_memory","frob_residual"]:
    print(f"  {c}: {lam.get(c,0):.4g}")
