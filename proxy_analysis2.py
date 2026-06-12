import json, math
from collections import defaultdict
import numpy as np

PATH = "/Users/assmuth/dsnn/search_space_full/results_seed12345.jsonl"

lat_readings = defaultdict(list)
det = {}
frob = defaultdict(list)
cos  = defaultdict(list)
rvec = {}

with open(PATH) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        if r.get("invalid"): continue
        idx=r["idx"]; ls=r.get("latency_ns_samples")
        if not ls: continue
        lat_readings[idx].extend([x for x in ls if x and x>0])
        det[idx]=(r["muls_adds_fmas"],r["flops"],r["max_io_sum"],r["bytes_accessed"])
        fr=r.get("frob_residual_per_point");  cs=r.get("cosine_sim_per_point")
        if fr: frob[idx].extend([x for x in fr if x is not None])
        if cs: cos[idx].extend([x for x in cs if x is not None])
        if "reward_vec" in r: rvec[idx]=r["reward_vec"]

def tmean(v,p=0.20):
    a=np.sort(np.asarray(v,float)); n=len(a); k=int(math.floor(n*p))
    return float(np.median(a)) if n-2*k<=0 else float(a[k:n-k].mean())

rows=[]
for idx,vals in lat_readings.items():
    if len(vals)<40: continue
    mm,fl,io,by=det[idx]
    if min(mm,fl,io,by)<=0: continue
    rows.append(dict(idx=idx,vals=np.array(vals),lat=tmean(vals),mm=mm,fl=fl,io=io,by=by,
                     frob=float(np.mean(frob[idx])) if frob[idx] else np.nan,
                     cos=float(np.mean(cos[idx])) if cos[idx] else np.nan))

N=len(rows)
lat=np.array([r['lat'] for r in rows]); llat=np.log(lat)
def sp(x,y):
    return float(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1])

# --- A. Reliability of the 'truth' itself: split-half stability of the trimmed-mean ranking ---
rng=np.random.default_rng(1)
half1=[]; half2=[]
for r in rows:
    v=r['vals'].copy(); rng.shuffle(v); m=len(v)//2
    half1.append(tmean(v[:m])); half2.append(tmean(v[m:]))
half1=np.log(np.array(half1)); half2=np.log(np.array(half2))
print(f"N configs={N}")
print(f"[A] split-half Spearman of trimmed-mean latency ranking = {sp(half1,half2):.3f}")
print(f"    -> ceiling on any latency-recovery score (truth is itself noisy)")

# --- B. frob_residual distribution: is it a usable quant/approx indicator? ---
fr=np.array([r['frob'] for r in rows]); cosv=np.array([r['cos'] for r in rows])
m=np.isfinite(fr)
print(f"\n[B] frob_residual: ==1.0 (no approx applied) frac={np.mean(np.isclose(fr[m],1.0)):.3f}; "
      f"<1.0 (approx) frac={np.mean(fr[m]<0.999):.3f}; >1.0 frac={np.mean(fr[m]>1.001):.3f}")
print(f"    cosine_sim: ==0 frac={np.mean(np.isclose(cosv[m],0.0)):.3f}; >0.5 frac={np.mean(cosv[m]>0.5):.3f}")
# define 'approx/quant' configs = cosine_sim high (output preserved despite approx) OR frob != 1
approx = (np.abs(fr-1.0)>0.01) | (cosv>0.01)
print(f"    'approx/quant-like' configs (frob!=1 or cos>0): {np.mean(approx[m])*100:.1f}%")

# --- C. flops paradox: do high-flops configs sometimes run faster? bin by flops, see latency ---
fl=np.array([r['fl'] for r in rows]); mm=np.array([r['mm'] for r in rows])
io=np.array([r['io'] for r in rows]); by=np.array([r['by'] for r in rows])
# residual of latency vs flops; check if approx configs are the fast-for-their-flops ones
X=np.column_stack([np.log(fl),np.ones(N)])
b,*_=np.linalg.lstsq(X,llat,rcond=None); resid=llat-X@b
ma=approx&m
print(f"\n[C] mean log-latency residual vs flops:")
print(f"    approx/quant configs:     {np.mean(resid[ma]):+.3f}  (n={ma.sum()})")
print(f"    non-approx configs:       {np.mean(resid[m&~approx]):+.3f}  (n={(m&~approx).sum()})")
print(f"    -> negative resid = FASTER than flops predicts")

# --- D. Best model R2 vs the achievable ceiling (truth split-half R2) ---
# achievable R2 ceiling = correlation^2 between two independent half-estimates of truth
ceil_r2 = np.corrcoef(half1,half2)[0,1]**2
def fit(cols):
    X=np.column_stack([np.log(c) for c in cols]+[np.ones(N)])
    bb,*_=np.linalg.lstsq(X,llat,rcond=None); pr=X@bb; rs=llat-pr
    return 1-np.sum(rs**2)/np.sum((llat-llat.mean())**2)
r2_full=fit([mm,fl,io,by])
print(f"\n[D] proxy full-model R2={r2_full:.3f}; truth self-consistency R2 ceiling={ceil_r2:.3f}")
print(f"    proxy captures {r2_full/ceil_r2*100:.0f}% of the *reliably-measurable* between-config variance")

# --- E. measurement budget vs ranking quality (how many samples needed) ---
print(f"\n[E] ranking recovery vs measurement budget (Spearman to full trimmed-mean):")
allv=[r['vals'] for r in rows]
for k in (1,2,4,8,16,32):
    est=[]
    for v in allv:
        kk=min(k,len(v)); est.append(np.mean(rng.choice(v,kk,replace=False)))
    print(f"    {k:>3} samples: Spearman={sp(np.array(est),lat):.3f}")
print(f"    proxy-only (flops): Spearman={sp(fl,lat):.3f}")
