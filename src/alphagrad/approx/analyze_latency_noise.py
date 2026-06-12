"""Empirically characterize latency-measurement noise from the search-space
sampler output, and rank candidate aggregation estimators by split-half
test-retest reliability + discriminability. Pure numpy; reads the raw
per-(config,pass) jsonl (80 latency readings/config)."""
import json, sys, collections
import numpy as np

F = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/assmuth/dsnn/search_space_full/results_seed12345.jsonl"

pool = collections.defaultdict(list)
with open(F) as fh:
    for ln in fh:
        d = json.loads(ln)
        if d.get("invalid"):
            continue
        for x in d.get("latency_ns_samples", []):
            if x and np.isfinite(x) and x > 0:
                pool[d["idx"]].append(float(x))

cfgs = {k: np.array(v) for k, v in pool.items() if len(v) >= 60}
print("configs with >=60 valid latency readings:", len(cfgs))


def q(a, p):
    return float(np.percentile(a, p))


ratios = np.array([q(a, 60) / a.min() for a in cfgs.values()])
mxmn = np.array([a.max() / a.min() for a in cfgs.values()])
cv = np.array([a.std() / a.mean() for a in cfgs.values()])
print("within-config noise (across configs):")
print("  P60/min : median=%.2f p90=%.2f" % (np.median(ratios), q(ratios, 90)))
print("  max/min : median=%.2f p90=%.2f" % (np.median(mxmn), q(mxmn, 90)))
print("  CV      : median=%.2f p90=%.2f" % (np.median(cv), q(cv, 90)))


def best_m(a, frac):
    k = max(1, int(len(a) * frac))
    return float(np.sort(a)[:k].mean())


ESTS = {
    "min": lambda a: float(a.min()),
    "p10": lambda a: q(a, 10),
    "p20": lambda a: q(a, 20),
    "best25%mean": lambda a: best_m(a, .25),
    "median": lambda a: q(a, 50),
    "p60(current)": lambda a: q(a, 60),
    "mean": lambda a: float(a.mean()),
    "trim20%sym": lambda a: float(np.sort(a)[int(.2 * len(a)):len(a) - int(.2 * len(a))].mean()),
    "gmean": lambda a: float(np.exp(np.log(a).mean())),
}


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


rng = np.random.default_rng(0)
print("\nsplit-half test-retest (higher rho, lower medRelDiff, higher SNR = better):")
print("  %-14s %8s %11s %12s" % ("estimator", "rho", "medRelDiff", "discrim_SNR"))
for name, fn in ESTS.items():
    A, B = [], []
    for a in cfgs.values():
        p = rng.permutation(a)
        h = len(p) // 2
        A.append(fn(p[:h]))
        B.append(fn(p[h:]))
    A = np.array(A)
    B = np.array(B)
    rho = spearman(A, B)
    reldiff = float(np.median(np.abs(A - B) / B))
    snr = float(np.std((A + B) / 2) / (np.std(A - B) / np.sqrt(2)))
    print("  %-14s %8.4f %11.3f %12.2f" % (name, rho, reldiff, snr))

print("\nsample-size curve (best25%mean split-half rel diff vs n per half):")
for n in [8, 16, 24, 40, 60]:
    diffs = []
    for a in cfgs.values():
        if len(a) < 2 * n:
            continue
        p = rng.permutation(a)
        diffs.append(abs(best_m(p[:n], .25) - best_m(p[n:2 * n], .25)) / best_m(p[n:2 * n], .25))
    if diffs:
        print("  n=%3d: medRelDiff=%.3f (configs=%d)" % (n, np.median(diffs), len(diffs)))
