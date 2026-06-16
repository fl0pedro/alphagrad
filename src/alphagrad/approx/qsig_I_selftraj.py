"""Signal I: self-trajectory alignment. Distills WHY the K-step probe (B) works,
cheaply: run K=25 Adam steps with the rule's OWN approx grad and, at each step,
record the aligned cosine to the exact grad AT THE CURRENT (self-)weights. Signal
= mean cosine along the rule's own path (3 seeds). Tests whether directional
consistency as the rule moves into its own region predicts trainability better than
alignment measured only at init (A) or over reverse-mode weights (D). No accuracy
eval and ~3x fewer steps than B.
  uv run python src/alphagrad/approx/qsig_I_selftraj.py
"""
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()


def fcos(ga, ge):
    a = np.concatenate([np.asarray(g).ravel() for g in ga])
    e = np.concatenate([np.asarray(g).ravel() for g in ge])
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))


def sigI(fn, K=25, seeds=(0, 1, 2)):
    means = []
    for s in seeds:
        W = q.init_w(s); opt = optax.adam(1e-3); ost = opt.init(W)
        rng = np.random.default_rng(s); cs = []
        for t in range(K):
            i = rng.integers(0, xtr.shape[0], 16); xb, yb = xtr[i], ytr[i]
            va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
            _, ge = q.EXACT(xb, yb, *W); cs.append(fcos(ga, ge))
            upd, ost = opt.update(ga, ost, W); W = optax.apply_updates(W, upd)
        means.append(np.mean(cs))
    return float(np.mean(means))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigI(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("I_selftraj", recs)
