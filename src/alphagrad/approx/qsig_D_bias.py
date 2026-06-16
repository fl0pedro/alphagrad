"""Signal D: directional bias (noise-averaged cosine). At each reference weight
point, average the rule's approx gradient over R minibatches and the exact
gradient over the same batches, then take the cosine of the two MEAN directions.
Isolates systematic (bias) error from minibatch noise that per-batch cosine
conflates.
  uv run python src/alphagrad/approx/qsig_D_bias.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigD(fn, R=8, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for xb, yb, Wj, ge in pool:
        gs, es = [], []
        for r in range(R):
            i = rng.integers(0, xtr.shape[0], 16); xa, ya = xtr[i], ytr[i]
            _, gee = q.EXACT(xa, ya, *Wj); va, ga = fn(xa, ya, *Wj); ga = q.align_grads(ga, Wj)
            gs.append(np.concatenate([np.asarray(g).ravel() for g in ga]))
            es.append(np.concatenate([np.asarray(g).ravel() for g in gee]))
        gm = np.mean(gs, 0); em = np.mean(es, 0)
        vals.append(float(gm @ em / (np.linalg.norm(gm) * np.linalg.norm(em) + 1e-30)))
    return float(np.mean(vals))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigD(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("D_bias", recs)
