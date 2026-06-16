"""Signal J: magnitude-weighted sign agreement (signSGD/Adam-Linf geometry). Adam
behaves like sign-descent, so descent is governed by per-coordinate sign agreement
weighted by exact-grad magnitude, NOT L2 cosine. At each reference weight point,
accumulate the per-coordinate MEAN approx and MEAN exact grad over R minibatches,
then score = sum_i |em_i| * 1[sign(gm_i)==sign(em_i)] / sum_i |em_i|. Range [0,1],
chance 0.5. Genuinely orthogonal to the saturated cosine family.
  uv run python src/alphagrad/approx/qsig_J_signagree.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigJ(fn, R=8, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for xb, yb, Wj, ge in pool:
        ga_acc, ge_acc = [], []
        for r in range(R):
            i = rng.integers(0, xtr.shape[0], 16); xa, ya = xtr[i], ytr[i]
            _, gee = q.EXACT(xa, ya, *Wj); va, gaa = fn(xa, ya, *Wj); gaa = q.align_grads(gaa, Wj)
            ge_acc.append(np.concatenate([np.asarray(g).ravel() for g in gee]))
            ga_acc.append(np.concatenate([np.asarray(g).ravel() for g in gaa]))
        em = np.mean(ge_acc, 0); gm = np.mean(ga_acc, 0)
        w = np.abs(em); agree = (np.sign(gm) == np.sign(em)).astype(float)
        vals.append(float((w * agree).sum() / (w.sum() + 1e-12)))
    return float(np.mean(vals))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigJ(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("J_signagree", recs)
