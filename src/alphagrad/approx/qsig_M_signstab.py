"""Signal M: sign self-stability across minibatches (signSGD success-probability /
discrete ZiCo cousin). signSGD convergence needs per-coordinate P(correct sign)>1/2
to survive noise-induced flips. Measure the rule's OWN sign consistency under data
noise: at each reference weight point, draw m minibatches, aligned approx grad per
batch; per coordinate majority sign maj_i, stability p_i = frac(sign==maj_i) in
[0.5,1]; magnitude-weight by across-batch mean |d|; point score = sum_i mbar_i*(2p_i
-1)/sum_i mbar_i (margin above coin flip). Exact grad NOT used; orthogonal to J
(approx-vs-exact) and to cosine.
  uv run python src/alphagrad/approx/qsig_M_signstab.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigM(fn, m=8, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for xb, yb, Wj, ge in pool:
        sgns, mags = [], []
        for b in range(m):
            i = rng.integers(0, xtr.shape[0], 16); xa, ya = xtr[i], ytr[i]
            va, gaa = fn(xa, ya, *Wj); gaa = q.align_grads(gaa, Wj)
            f = np.concatenate([np.asarray(g).ravel() for g in gaa])
            sgns.append(np.sign(f)); mags.append(np.abs(f))
        S = np.array(sgns); maj = np.sign(S.sum(0)); p = (S == maj).mean(0); mbar = np.mean(mags, 0)
        vals.append(float((mbar * (2 * p - 1)).sum() / (mbar.sum() + 1e-12)))
    return float(np.mean(vals))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigM(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("M_signstab", recs)
