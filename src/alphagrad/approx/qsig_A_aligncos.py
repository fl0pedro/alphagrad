"""Signal A (baseline): layout-aligned gradient cosine at init. Reproduces the
shipped aligned_cos signal in THIS run for an apples-to-apples baseline.
  uv run python src/alphagrad/approx/qsig_A_aligncos.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()

recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq)
        W = q.init_w(0); i = np.arange(16); xb, yb = xtr[i], ytr[i]
        va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
        fa = np.concatenate([np.asarray(g).ravel() for g in ga])
        _, ge = q.EXACT(xb, yb, *W)
        fe = np.concatenate([np.asarray(g).ravel() for g in ge])
        cos = float(fa @ fe / (np.linalg.norm(fa) * np.linalg.norm(fe) + 1e-30))
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); cos = float("nan")
    recs.append(dict(src=src, idx=idx, sig=cos, acc=lab["acc"], lat=lab["lat"]))

q.report("A_aligncos", recs)
