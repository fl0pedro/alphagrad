"""Signal B: K-step closed-loop probe. Run a short real training (K Adam steps,
3 seeds) using the rule's OWN approximate gradient, then read MNIST test accuracy.
Directly simulates 'would training with this learning rule work'.
  uv run python src/alphagrad/approx/qsig_B_kstep.py
"""
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()


def sigB(fn, n_steps=80, seeds=(0, 1, 2)):
    accs = []
    for s in seeds:
        W = q.init_w(s); opt = optax.adam(1e-3); ost = opt.init(W)
        rng = np.random.default_rng(s)
        for t in range(n_steps):
            i = rng.integers(0, xtr.shape[0], 16); xb, yb = xtr[i], ytr[i]
            va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
            upd, ost = opt.update(ga, ost, W); W = optax.apply_updates(W, upd)
        accs.append(q.accuracy(W, xte[:5000], yte[:5000]))
    return float(np.mean(accs))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigB(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("B_kstep", recs)
