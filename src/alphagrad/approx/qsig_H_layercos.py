"""Signal H: per-layer (bottleneck) alignment. The global flattened cosine (A) is
dominated by W1 (200k params) and is BLIND to the output layer W2 (2.5k params),
whose alignment may gate learning. Compute aligned cosine per weight matrix and
report the WORST-aligned weight layer (min over W1, W2). Same cost as aligned_cos.
  uv run python src/alphagrad/approx/qsig_H_layercos.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()


def lcos(a, e):
    a = np.asarray(a).ravel(); e = np.asarray(e).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq)
        W = q.init_w(0); i = np.arange(16); xb, yb = xtr[i], ytr[i]
        va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
        _, ge = q.EXACT(xb, yb, *W)
        cW1 = lcos(ga[0], ge[0]); cW2 = lcos(ga[2], ge[2])  # tuple = (W1,b1,W2,b2)
        v = float(min(cW1, cW2))
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan"); cW1 = cW2 = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, cos_W1=cW1, cos_W2=cW2, acc=lab["acc"], lat=lab["lat"]))

q.report("H_layercos", recs)
