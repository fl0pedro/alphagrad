"""Signal N: fixed-budget depth-vs-breadth of the rollout probe. At a constant total
step budget (reps x steps = 40), is it better to average many SHORT independent runs
or take a few DEEP ones? Each rep = a fresh seed trained `steps` Adam steps with the
rule's own approx grad; signal = mean test accuracy over reps. Computed from one set
of NSEED 40-step rollouts (snapshot acc at the needed step counts), then each config
reads mean-over-first-`reps`-seeds of acc-at-`steps`.
  uv run python src/alphagrad/approx/qsig_N_fixedbudget.py
"""
import os, json
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
NSEED, MAXSTEP = 40, 40
CKPTS = [1, 2, 4, 5, 8, 10, 20, 40]
CONFIGS = [(40, 1), (20, 2), (10, 4), (8, 5), (5, 8), (4, 10), (2, 20), (1, 40)]  # reps x steps = 40
CIDX = {c: i for i, c in enumerate(CKPTS)}


def rollout(fn):
    A = np.full((NSEED, len(CKPTS)), np.nan)
    cset = set(CKPTS)
    for s in range(NSEED):
        W = q.init_w(s); opt = optax.adam(1e-3); ost = opt.init(W)
        rng = np.random.default_rng(s); ci = 0
        for t in range(1, MAXSTEP + 1):
            i = rng.integers(0, xtr.shape[0], 16); xb, yb = xtr[i], ytr[i]
            va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
            upd, ost = opt.update(ga, ost, W); W = optax.apply_updates(W, upd)
            if t in cset:
                A[s, ci] = q.accuracy(W, xte[:5000], yte[:5000]); ci += 1
    return A


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); A = rollout(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); A = np.full((NSEED, len(CKPTS)), np.nan)
    recs.append(dict(src=src, idx=idx, acc=lab["acc"], lat=lab["lat"], A=A.tolist()))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return pear(rank(x[m]), rank(y[m]))


acc = np.array([r["acc"] for r in recs]); src = np.array([r["src"] for r in recs])
AA = np.array([r["A"] for r in recs])   # (nrules, NSEED, nckpts)
cm = src == "cmorl"; mo = src == "mogfn"
print("\n== fixed-budget (reps x steps = 40): mean test-acc as signal, Spearman vs label "
      "(aligned_cos=0.78; best rollout so far 1x40=0.831) ==")
for reps, steps in CONFIGS:
    sig = np.nanmean(AA[:, :reps, CIDX[steps]], axis=1)
    print(f"  {reps:2d} reps x {steps:2d} steps: BOTH={spear(sig,acc):+.3f} "
          f"C={spear(sig[cm],acc[cm]):+.3f} M={spear(sig[mo],acc[mo]):+.3f}")

json.dump(recs, open(os.path.expanduser("~/dsnn/train_exp/qsig_N_fixedbudget.json"), "w"))
print("\nsaved -> ~/dsnn/train_exp/qsig_N_fixedbudget.json")
