"""Signal C: trajectory-shadowing descent ratio. Along a reverse-mode reference
trajectory (arc-length-stratified checkpoints), compare the true-loss decrease
from one Adam step with the rule's approx grad vs. with the exact grad. Signal =
mean ratio (approx descent / exact descent).
  uv run python src/alphagrad/approx/qsig_C_shadow.py
"""
import numpy as np, jax.numpy as jnp, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()
pool = q.build_pool(M=16)


def sigC(fn):
    rs = []
    for xb, yb, Wj, ge in pool:
        l0 = float(q.LOSS(xb, yb, *Wj)); opt = optax.adam(1e-3)
        ge_j = tuple(jnp.asarray(g) for g in ge)
        oe = opt.init(Wj); ue, oe = opt.update(ge_j, oe, Wj); We = optax.apply_updates(Wj, ue); le = float(q.LOSS(xb, yb, *We))
        va, ga = fn(xb, yb, *Wj); ga = q.align_grads(ga, Wj)
        oa = opt.init(Wj); ua, oa = opt.update(ga, oa, Wj); Wa = optax.apply_updates(Wj, ua); la = float(q.LOSS(xb, yb, *Wa))
        de = l0 - le; da = l0 - la; rs.append(da / (de + 1e-12))
    return float(np.mean(rs))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigC(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("C_shadow", recs)
