"""B-frontier: the cost/accuracy tradeoff of the only signal that beats the ~0.78
plateau (B_kstep, the training-rollout probe). In ONE pass per rule, run 80 Adam
steps x 3 seeds with the rule's own approx grad and snapshot MNIST test accuracy at
{10,20,40,80} steps. Then report Spearman vs the 20-seed label for every
(n_seeds, n_steps) budget — so we can pick the cheapest rollout that retains B's
0.844. Loss-proxy variant also reported (no test-set eval) for an even cheaper option.
  uv run python src/alphagrad/approx/qsig_B2_frontier.py
"""
import os, json
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
CKPTS = [10, 20, 40, 80]
SEEDS = (0, 1, 2)


def rollout(fn):
    A = np.full((len(SEEDS), len(CKPTS)), np.nan)   # test acc
    Ltr = np.full((len(SEEDS), len(CKPTS)), np.nan)  # train-loss proxy (lower=better)
    for si, s in enumerate(SEEDS):
        W = q.init_w(s); opt = optax.adam(1e-3); ost = opt.init(W)
        rng = np.random.default_rng(s); ci = 0
        for t in range(1, max(CKPTS) + 1):
            i = rng.integers(0, xtr.shape[0], 16); xb, yb = xtr[i], ytr[i]
            va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
            upd, ost = opt.update(ga, ost, W); W = optax.apply_updates(W, upd)
            if t in CKPTS:
                A[si, ci] = q.accuracy(W, xte[:5000], yte[:5000])
                j = rng.integers(0, xtr.shape[0], 512)
                Ltr[si, ci] = float(q.LOSS(xtr[j], ytr[j], *W))
                ci += 1
    return A, Ltr


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); A, Ltr = rollout(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}")
        A = np.full((len(SEEDS), len(CKPTS)), np.nan); Ltr = A.copy()
    recs.append(dict(src=src, idx=idx, acc=lab["acc"], lat=lab["lat"], roll=A.tolist(), loss=Ltr.tolist()))


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    return pear(rank(x[m]), rank(y[m]))


acc = np.array([r["acc"] for r in recs]); src = np.array([r["src"] for r in recs])
R = np.array([r["roll"] for r in recs])   # (nrules, nseeds, nckpts)
L = np.array([r["loss"] for r in recs])
cm = src == "cmorl"; mo = src == "mogfn"

print("\n== B-frontier: Spearman vs 20-seed label, by budget (baseline aligned_cos=0.78; B@80x3=0.844) ==")
print("   [accuracy probe]")
for ns in (1, 3):
    for ci, ck in enumerate(CKPTS):
        sig = np.nanmean(R[:, :ns, ci], axis=1)
        print(f"   seeds={ns} steps={ck:3d}: BOTH={spear(sig,acc):+.3f} C={spear(sig[cm],acc[cm]):+.3f} M={spear(sig[mo],acc[mo]):+.3f}  (~{ns*ck} steps/rule)")
print("   [train-loss probe — no test-set eval, cheaper] (negated so higher=better)")
for ns in (1, 3):
    for ci, ck in enumerate(CKPTS):
        sig = -np.nanmean(L[:, :ns, ci], axis=1)
        print(f"   seeds={ns} steps={ck:3d}: BOTH={spear(sig,acc):+.3f} C={spear(sig[cm],acc[cm]):+.3f} M={spear(sig[mo],acc[mo]):+.3f}  (~{ns*ck} steps/rule)")

json.dump(recs, open(os.path.expanduser("~/dsnn/train_exp/qsig_B2_frontier.json"), "w"))
print("\nsaved -> ~/dsnn/train_exp/qsig_B2_frontier.json")
