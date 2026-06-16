"""Signal K: ZiCo-rule (cross-minibatch gradient SNR of the APPROX grad). The only
zero-cost NAS proxy that is rule-dependent in our dual (fixed-net, varying-operator)
regime, and orthogonal to every cosine signal because it never references the exact
direction. At each reference weight point, draw m minibatches, compute the aligned
approx grad per batch; per coordinate mu_i=mean_b|d_i|, sd_i=std_b(d_i); per leaf
score = log(sum_i mu_i/(sd_i+eps)); point score = sum over leaves. Higher SNR =
more trainable. Exact grad NOT used.
  uv run python src/alphagrad/approx/qsig_K_zico.py
"""
import numpy as np
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigK(fn, m=8, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for xb, yb, Wj, ge in pool:
        per_batch = []
        for b in range(m):
            i = rng.integers(0, xtr.shape[0], 16); xa, ya = xtr[i], ytr[i]
            va, gaa = fn(xa, ya, *Wj); gaa = q.align_grads(gaa, Wj)
            per_batch.append([np.asarray(g) for g in gaa])
        score = 0.0
        for l in range(len(per_batch[0])):
            stack = np.stack([per_batch[b][l] for b in range(m)], 0)
            mu = np.mean(np.abs(stack), 0).ravel(); sd = np.std(stack, 0).ravel()
            score += float(np.log(np.sum(mu / (sd + 1e-8)) + 1e-30))
        vals.append(score)
    return float(np.mean(vals))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigK(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("K_zico", recs)
