"""Signal E: one-Adam-step true-loss decrease. At each reference weight point,
take a single Adam step with the rule's approx grad and measure the actual
decrease in the TRUE loss. Signal = mean (l0 - l1); positive = the approx grad
is a genuine descent direction on the real objective.
  uv run python src/alphagrad/approx/qsig_E_onestep.py
"""
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigE(fn):
    dls = []
    for xb, yb, Wj, ge in pool:
        l0 = float(q.LOSS(xb, yb, *Wj)); va, ga = fn(xb, yb, *Wj); ga = q.align_grads(ga, Wj)
        opt = optax.adam(1e-3); ost = opt.init(Wj); upd, ost = opt.update(ga, ost, Wj); W1 = optax.apply_updates(Wj, upd)
        l1 = float(q.LOSS(xb, yb, *W1)); dls.append(l0 - l1)
    return float(np.mean(dls))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigE(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("E_onestep", recs)
