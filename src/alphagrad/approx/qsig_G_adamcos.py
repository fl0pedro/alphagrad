"""Signal G: Adam-geometry alignment. Training uses Adam, which rescales each
coordinate by 1/sqrt(E[g^2]); raw-space cosine (A, D) ignores this. At each
reference weight point, estimate the Adam preconditioner p = 1/(sqrt(mean exact
g^2)+eps) over R minibatches, then take the cosine of the preconditioned MEAN
approx vs exact directions. This is D (noise-averaged cosine) re-expressed in the
optimizer's actual geometry.
  uv run python src/alphagrad/approx/qsig_G_adamcos.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigG(fn, R=8, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for xb, yb, Wj, ge in pool:
        ga_acc, ge_acc = [], []
        for r in range(R):
            i = rng.integers(0, xtr.shape[0], 16); xa, ya = xtr[i], ytr[i]
            _, gee = q.EXACT(xa, ya, *Wj); va, gaa = fn(xa, ya, *Wj); gaa = q.align_grads(gaa, Wj)
            ge_acc.append(np.concatenate([np.asarray(g).ravel() for g in gee]))
            ga_acc.append(np.concatenate([np.asarray(g).ravel() for g in gaa]))
        E = np.array(ge_acc); A = np.array(ga_acc)
        p = 1.0 / (np.sqrt(np.mean(E ** 2, 0)) + 1e-8)   # Adam preconditioner
        em = E.mean(0) * p; am = A.mean(0) * p
        vals.append(float(am @ em / (np.linalg.norm(am) * np.linalg.norm(em) + 1e-30)))
    return float(np.mean(vals))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigG(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("G_adamcos", recs)
