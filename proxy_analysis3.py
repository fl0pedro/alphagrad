import json, math
from collections import defaultdict
import numpy as np
PATH="/Users/assmuth/dsnn/search_space_full/results_seed12345.jsonl"
lat=defaultdict(list); det={}
with open(PATH) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get("invalid"): continue
        ls=r.get("latency_ns_samples")
        if not ls: continue
        lat[r["idx"]].extend([x for x in ls if x and x>0])
        det[r["idx"]]=(r["muls_adds_fmas"],r["flops"],r["max_io_sum"],r["bytes_accessed"])
def tm(v,p=.2):
    a=np.sort(np.asarray(v,float));n=len(a);k=int(n*p)
    return float(np.median(a)) if n-2*k<=0 else float(a[k:n-k].mean())
rows=[(i,tm(v),*det[i]) for i,v in lat.items() if len(v)>=40 and min(det[i])>0]
idx=np.array([r[0] for r in rows]); L=np.array([r[1] for r in rows])
mm=np.array([r[2] for r in rows]);fl=np.array([r[3] for r in rows])
io=np.array([r[4] for r in rows]);by=np.array([r[5] for r in rows])
def sp(x,y): return float(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1])

# How well does each proxy rank the BEST (lowest-latency) configs? Pareto search cares about the head.
truerank=np.argsort(L)
k=int(0.1*len(L)); true_top=set(truerank[:k])
print("Top-10% (fastest) recall by each single proxy (rank ascending=best):")
for name,p in [("flops",fl),("muls",mm),("max_io",io),("bytes",by)]:
    got=set(np.argsort(p)[:k])
    print(f"  {name:<8} recall={len(true_top&got)/k:.3f}  Spearman={sp(p,L):.3f}")

# float8/quant paradox: among configs with NEARLY IDENTICAL flops, latency spread?
# bin configs into flops deciles, report within-bin latency CV (var the proxy can't see)
print("\nWithin-flops-decile latency spread (variation invisible to a flops proxy):")
dec=np.argsort(np.argsort(fl))*10//len(fl)
for d in range(10):
    sel=dec==d
    if sel.sum()<10: continue
    ll=np.log(L[sel])
    print(f"  decile {d}: n={sel.sum():4d} log-lat std={ll.std():.3f} max/min lat ratio={L[sel].max()/L[sel].min():.1f}x")
# overall: fraction of total log-lat variance that is WITHIN flops-deciles (proxy-blind)
within=np.mean([np.log(L[dec==d]).var() for d in range(10) if (dec==d).sum()>1])
total=np.log(L).var()
print(f"\nWithin-decile (flops-blind) variance = {within/total*100:.0f}% of total log-lat variance")
