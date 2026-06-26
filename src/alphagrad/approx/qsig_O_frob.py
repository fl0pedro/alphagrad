"""Signal O: aligned Frobenius residual vs cosine. Is the (relative) frobenius
distance between approx and exact grad a better static signal than cosine? Computed
at init (the measurement point where cosine A scored best). Reports BOTH the relative
residual ||ga-ge||/||ge|| (negated; conflates magnitude ratio + cosine) and the pure
magnitude ratio ||ga||/||ge||, so we can see whether magnitude carries any signal
beyond cosine. Under Adam (per-coordinate scale-invariant) magnitude should be noise.
  uv run python src/alphagrad/approx/qsig_O_frob.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()

recs, magrecs = [], []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq)
        W = q.init_w(0); i = np.arange(16); xb, yb = xtr[i], ytr[i]
        va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
        _, ge = q.EXACT(xb, yb, *W)
        fa = np.concatenate([np.asarray(g).ravel() for g in ga])
        fe = np.concatenate([np.asarray(g).ravel() for g in ge])
        na, ne = np.linalg.norm(fa), np.linalg.norm(fe)
        relfrob = float(np.linalg.norm(fa - fe) / (ne + 1e-30))
        magratio = float(na / (ne + 1e-30))
        v = -relfrob                       # higher = better
        mv = -abs(np.log(magratio + 1e-30))  # higher = magnitude closer to exact
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = mv = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))
    magrecs.append(dict(src=src, idx=idx, sig=mv, acc=lab["acc"], lat=lab["lat"]))

q.report("O_frob", recs)
q.report("O_magratio", magrecs)
