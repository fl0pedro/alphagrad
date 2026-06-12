import json, math
from collections import defaultdict
import numpy as np

PATH = "/Users/assmuth/dsnn/search_space_full/results_seed12345.jsonl"

# Per-config aggregation
lat_readings = defaultdict(list)        # idx -> list of latency_ns samples (all valid passes)
det = {}                                # idx -> deterministic scalars
frob = defaultdict(list)                # idx -> frob_residual values (mean per pass)
cos  = defaultdict(list)
peak = defaultdict(list)
n_valid_pass = defaultdict(int)

with open(PATH) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("invalid"): continue
        idx = r["idx"]
        ls = r.get("latency_ns_samples")
        if not ls: continue
        lat_readings[idx].extend([x for x in ls if x is not None and x > 0])
        det[idx] = (r["muls_adds_fmas"], r["flops"], r["max_io_sum"], r["bytes_accessed"])
        fr = r.get("frob_residual_per_point")
        if fr: frob[idx].extend([x for x in fr if x is not None])
        cs = r.get("cosine_sim_per_point")
        if cs: cos[idx].extend([x for x in cs if x is not None])
        pm = r.get("peak_memory_samples")
        if pm: peak[idx].extend([x for x in pm if x is not None])
        n_valid_pass[idx]+=1

def trimmed_mean(vals, p=0.20):
    a = np.sort(np.asarray(vals, float))
    n = len(a)
    k = int(math.floor(n*p))
    if n - 2*k <= 0:
        return float(np.median(a))
    return float(a[k:n-k].mean())

# Build per-config table: require enough valid readings (>=40 of 80)
rows = []
for idx, vals in lat_readings.items():
    if len(vals) < 40:    # "enough valid readings"
        continue
    mm, fl, io, by = det[idx]
    if min(mm, fl, io, by) <= 0:
        continue
    lat = trimmed_mean(vals, 0.20)
    cv = float(np.std(vals)/np.mean(vals)) if np.mean(vals)>0 else float('nan')
    frob_mean = float(np.mean(frob[idx])) if frob[idx] else float('nan')
    cos_mean  = float(np.mean(cos[idx]))  if cos[idx]  else float('nan')
    peak_mean = float(np.mean(peak[idx])) if peak[idx] else 0.0
    rows.append(dict(idx=idx, lat=lat, cv=cv, n=len(vals),
                     mm=mm, fl=fl, io=io, by=by,
                     frob=frob_mean, cos=cos_mean, peak=peak_mean))

print(f"configs with >=40 valid readings: {len(rows)}")
print(f"total configs seen valid: {len(lat_readings)}")
med_cv = np.median([r['cv'] for r in rows])
print(f"median per-config raw-latency CV: {med_cv:.3f}")

lat = np.array([r['lat'] for r in rows])
mm  = np.array([r['mm'] for r in rows])
fl  = np.array([r['fl'] for r in rows])
io  = np.array([r['io'] for r in rows])
by  = np.array([r['by'] for r in rows])
frob_arr = np.array([r['frob'] for r in rows])

def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0,1])

def pearson(x, y):
    return float(np.corrcoef(x, y)[0,1])

llat = np.log(lat)
proxies = {"muls_adds_fmas": mm, "flops": fl, "max_io_sum": io, "bytes_accessed": by}

print("\n=== 1. PROXY CORRELATION ===")
print(f"{'proxy':<16}{'Spearman':>10}{'logPearson':>12}")
sp_results = {}
for name, p in proxies.items():
    sp = spearman(p, lat)
    pe = pearson(np.log(p), llat)
    sp_results[name] = (sp, pe)
    print(f"{name:<16}{sp:>10.3f}{pe:>12.3f}")

# === 2. BEST LOG-LINEAR MODEL ===
print("\n=== 2. BEST DETERMINISTIC LOG MODEL ===")
def fit(cols, names):
    X = np.column_stack([np.log(c) for c in cols] + [np.ones(len(llat))])
    beta, *_ = np.linalg.lstsq(X, llat, rcond=None)
    pred = X @ beta
    resid = llat - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((llat - llat.mean())**2)
    r2 = 1 - ss_res/ss_tot
    resid_cv = float(np.exp(np.std(resid)) - 1)   # approx multiplicative spread
    return r2, resid, beta, resid_cv

# single best
best_single = max(sp_results, key=lambda k: abs(sp_results[k][1]))
r2s, _, _, cvs = fit([proxies[best_single]], [best_single])
print(f"single best ({best_single}): R2={r2s:.3f}  resid mult-CV={cvs:.3f}")

# full model
r2f, residf, betaf, cvf = fit([mm, fl, io, by], list(proxies))
print(f"full (mm+flops+io+bytes): R2={r2f:.3f}  resid mult-CV={cvf:.3f}")
print(f"  coeffs log[mm,flops,io,bytes,const] = {np.round(betaf,3).tolist()}")

# add frob residual as a feature (quant indicator), where available
mask_fr = np.isfinite(frob_arr)
def fit_mask(cols, mask):
    X = np.column_stack([np.log(c[mask]) for c in cols[:-1]] + [cols[-1][mask]] + [np.ones(mask.sum())])
    y = llat[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X@beta; resid=y-pred
    r2 = 1 - np.sum(resid**2)/np.sum((y-y.mean())**2)
    return r2, beta
r2_fr, _ = fit_mask([mm, fl, io, by, frob_arr], mask_fr)
print(f"full + frob_residual: R2={r2_fr:.3f}  (n={int(mask_fr.sum())})")

# === 3. IRREPLACEABLE / LARGEST RESIDUALS ===
print("\n=== 3. WHERE MEASUREMENT IS IRREPLACEABLE ===")
# residuals from full model (log space). positive resid = measured SLOWER than proxy predicts
abs_resid = np.abs(residf)
order = np.argsort(abs_resid)[::-1]
mult_err = np.exp(residf)  # measured/predicted ratio
print(f"residual log-space std={np.std(residf):.3f}; "
      f"mult error p50={np.percentile(np.abs(mult_err-1),50):.3f} "
      f"p90={np.percentile(np.abs(mult_err-1),90):.3f} "
      f"p99={np.percentile(np.abs(mult_err-1),99):.3f}")

# fraction with proxy error > 2x
for thr in (1.5, 2.0, 3.0):
    frac = np.mean((mult_err > thr) | (mult_err < 1/thr))
    print(f"  fraction off by > {thr:.1f}x : {frac*100:.1f}%")

# characterize top-residual configs by frob (quant/approx indicator)
top = order[:int(0.05*len(rows))]   # worst 5%
rest = order[int(0.05*len(rows)):]
def frob_stats(ix):
    f = frob_arr[ix]; f=f[np.isfinite(f)]
    return np.mean(f), np.median(f)
ft_m, ft_md = frob_stats(top)
fr_m, fr_md = frob_stats(rest)
print(f"  worst-5% residual configs: mean frob_residual={ft_m:.3f} median={ft_md:.3f}")
print(f"  rest:                      mean frob_residual={fr_m:.3f} median={fr_md:.3f}")
# correlation between |resid| and frob
m2 = np.isfinite(frob_arr)
print(f"  Spearman(|resid|, frob_residual)={spearman(abs_resid[m2], frob_arr[m2]):.3f}")
# how many worst configs are "approximated" (frob>0.01)
print(f"  worst-5%: fraction frob>0.01 = {np.mean(frob_arr[top][np.isfinite(frob_arr[top])]>0.01)*100:.1f}%")
print(f"  overall : fraction frob>0.01 = {np.mean(frob_arr[m2]>0.01)*100:.1f}%")

# === 4. HYBRID STRATEGY ===
print("\n=== 4. HYBRID STRATEGY / DISCRIMINABILITY ===")
# 'true' ranking ~ trimmed mean latency (best available). single-pass noisy = pick one pass.
# Compare ranking recovery (Spearman vs true) of:
#  a) proxy prediction (full model)
#  b) single noisy measurement (1 pass ~ 8 samples mean)
#  c) hybrid: proxy, but for the configs the proxy is least sure about, use measurement
pred_full = residf*0 + (llat - residf)  # = model prediction in log space
true = llat

sp_proxy = spearman(pred_full, true)
print(f"a) pure proxy ranking vs trimmed-mean truth: Spearman={sp_proxy:.3f}")

# single noisy reading: simulate by sampling 8 readings per config
rng = np.random.default_rng(0)
single = []
for r in rows:
    v = np.array(lat_readings[r['idx']])
    single.append(np.mean(rng.choice(v, size=min(8,len(v)), replace=False)))
single = np.log(np.array(single))
sp_single = spearman(single, true)
print(f"b) single 8-sample measurement vs truth:     Spearman={sp_single:.3f}")

# hybrid: rank by proxy, then re-measure top fraction & the high-uncertainty (high frob) ones.
# discriminability metric: how well does each recover the TOP-K best (lowest-latency) configs?
def topk_recall(score, k_frac=0.10):
    k = int(k_frac*len(true))
    true_top = set(np.argsort(true)[:k])           # lowest latency = best
    got_top  = set(np.argsort(score)[:k])
    return len(true_top & got_top)/k

for kf in (0.05, 0.10, 0.20):
    print(f"  top-{int(kf*100)}% recall: proxy={topk_recall(pred_full,kf):.3f}  single-meas={topk_recall(single,kf):.3f}")

# Hybrid: use proxy to pick top 30% candidates, then MEASURE those (use trimmed mean truth as 'measured'),
# rank within by measurement. Cost = 30% of full measurement budget.
def hybrid_recall(k_frac=0.10, gate=0.30):
    g = int(gate*len(true))
    cand = np.argsort(pred_full)[:g]               # proxy-selected candidates
    # measure candidates (use single noisy meas to be realistic)
    score = np.full(len(true), np.inf)
    score[cand] = single[cand]
    k = int(k_frac*len(true))
    true_top = set(np.argsort(true)[:k])
    got_top  = set(np.argsort(score)[:k])
    return len(true_top & got_top)/k

for kf in (0.05, 0.10):
    print(f"  top-{int(kf*100)}% recall HYBRID(proxy-gate 30%->measure): {hybrid_recall(kf,0.30):.3f}  (meas budget=30%)")
    print(f"  top-{int(kf*100)}% recall HYBRID(proxy-gate 50%->measure): {hybrid_recall(kf,0.50):.3f}  (meas budget=50%)")

# SNR: between-config variance of true latency vs within-config measurement noise
between = np.var(llat)
within = np.mean([ (np.std(np.log(np.array(lat_readings[r['idx']])))**2) for r in rows ])
print(f"\nSNR (between-config var / within-config log-var) = {between/within:.2f}")
print(f"  between-config log-var={between:.3f}  within-config log-var(mean)={within:.3f}")
# proxy 'explains' fraction of between-config var = R2_full
print(f"  proxy explains {r2f*100:.1f}% of between-config latency variance")
