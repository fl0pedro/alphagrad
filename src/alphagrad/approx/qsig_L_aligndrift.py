"""Signal L: alignment drift (align-then-memorise early slope). B_kstep works
because good rules raise their gradient alignment in the FIRST few steps. Capture
that mechanism cheaply and non-tautologically: run K=6 Adam steps with the rule's
own aligned approx grad; record flattened aligned cosine c_t at each step; signal =
least-squares slope of c_t vs t (3 seeds). Distinct from I_selftraj, which reports
the mean LEVEL of cosine, not its TREND — a rule can have high mean cosine that is
flat or decreasing. No accuracy read, so not tautological.
  uv run python src/alphagrad/approx/qsig_L_aligndrift.py
"""
import numpy as np, optax
import alphagrad.approx.qsig_common as q

xtr, ytr, xte, yte = q.mnist()
env, ev = q.build_env()


def fcos(ga, ge):
    a = np.concatenate([np.asarray(g).ravel() for g in ga])
    e = np.concatenate([np.asarray(g).ravel() for g in ge])
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))


def sigL(fn, K=6, seeds=(0, 1, 2)):
    slopes = []
    for s in seeds:
        W = q.init_w(s); opt = optax.adam(1e-3); ost = opt.init(W)
        rng = np.random.default_rng(s); cs = []
        for t in range(K):
            i = rng.integers(0, xtr.shape[0], 16); xb, yb = xtr[i], ytr[i]
            va, ga = fn(xb, yb, *W); ga = q.align_grads(ga, W)
            _, ge = q.EXACT(xb, yb, *W); cs.append(fcos(ga, ge))
            upd, ost = opt.update(ga, ost, W); W = optax.apply_updates(W, upd)
        slopes.append(np.polyfit(np.arange(K), np.array(cs), 1)[0])
    return float(np.mean(slopes))


recs = []
for src, idx, seq, lab in q.rules():
    try:
        fn = q.capture_gfn(env, ev, seq); v = sigL(fn)
    except Exception as e:
        print(f"  [skip {src}#{idx}] {e}"); v = float("nan")
    recs.append(dict(src=src, idx=idx, sig=v, acc=lab["acc"], lat=lab["lat"]))

q.report("L_aligndrift", recs)
