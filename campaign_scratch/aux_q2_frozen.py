"""aux_q2: hard ceiling for a FROZEN per-vertex representation.

PPO's per-candidate slot is bitwise constant within an episode and identical
across episodes (it is a function of the static token stream only).  So the
best ANY aux head reading that slot can output for a dynamic feature is a
per-VERTEX CONSTANT.  This script measures:
  (a) R^2 of a per-vertex fixed effect for each dynamic feature (the ceiling),
  (b) the ranking gain of frozen-vs-real dynamic features,
  (c) how the gain splits over episode phase.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import (build_static, extract, DYN_VERTEX_DIM, STATIC_DIM,
    DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL, DV_MAX_JAC)

NPLAN = int(sys.argv[1]) if len(sys.argv) > 1 else 250
MODE = sys.argv[2] if len(sys.argv) > 2 else "best"

fn, args_, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
static = build_static(env)

rows = []
for s in (0, 1, 2):
    for li, line in enumerate(open(os.path.expanduser(f"~/dsnn/elimrl_m3_s{s}/measurements.jsonl"))):
        d = json.loads(line)
        if d.get("status") == "ok" and d.get("latency_ns") and d["tag"] == "pomo":
            rows.append((s, li, d["latency_ns"], d["order"]))
if MODE == "best":
    sel = sorted(rows, key=lambda r: r[2])[:NPLAN]
else:
    rng = np.random.default_rng(0); sel = [rows[i] for i in rng.choice(len(rows), NPLAN, replace=False)]
print(f"MODE={MODE} nplan={len(sel)}", flush=True)

DYNCOL = [DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL, DV_MAX_JAC]
DYNN = ["in_deg", "out_deg", "markowitz", "fill", "max_jac"]
assert DYNCOL == [0,1,2,3,5], DYNCOL
Xs, ROW, ch, off, step_of, plan_of = [], [], [], [0], [], []
t0 = time.time()
for pi, (s, li, lat, order) in enumerate(sel):
    env.reset()
    for t, vid in enumerate(order):
        stt = env.state(); legal = stt.legal_vertices
        if len(legal) < 2: break
        sf = extract(stt, static)
        lr = np.asarray([static.row_of(j) for j in legal], np.int32)
        Xs.append(np.concatenate([sf.vert_dyn[np.ix_(lr, DYNCOL)], static.feat[lr]], 1).astype(np.float32))
        ROW.append(lr)
        ch.append(legal.index(vid)); off.append(off[-1] + len(legal))
        step_of.append(t); plan_of.append(pi)
        env.step(("V", int(vid)))
X = np.concatenate(Xs, 0).astype(np.float64); ROW = np.concatenate(ROW)
ch = np.asarray(ch); off = np.asarray(off); sz = np.diff(off)
step_of = np.asarray(step_of); plan_of = np.asarray(plan_of)
G = len(ch); gid = np.repeat(np.arange(G), sz)
print(f"groups={G} rows={len(X)} wall={time.time()-t0:.0f}s", flush=True)

NDYN, D = 5, 5 + STATIC_DIM
# ---- (a) per-vertex fixed-effect R^2, on WITHIN-STEP-CENTRED values ----
def wcentre(v):
    m = np.bincount(gid, v) / sz
    return v - m[gid]
nrow = static.n_rows
print("\n=== (a) CEILING for a frozen per-vertex representation ===")
print("    R2_vertex = fraction of the within-step-contrast variance of the feature")
print("    that a per-VERTEX CONSTANT can reproduce.  1-R2 is unreachable by any")
print("    encoder whose per-candidate output cannot change during the episode.")
for k in range(NDYN):
    v = wcentre(X[:, k])
    cnt = np.bincount(ROW, minlength=nrow); s1 = np.bincount(ROW, v, minlength=nrow)
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    pred = mu[ROW]
    r2 = 1.0 - ((v - pred) ** 2).mean() / max(v.var(), 1e-12)
    print(f"  dyn_{DYNN[k]:11s} var={v.var():.4f}  R2_vertex={r2:.4f}   unreachable={1-r2:.4f}")
for k, nm in ((0, "st_topo_idx"), (1, "st_height")):
    pass
print("  (static features are by construction per-vertex constants: R2_vertex = 1.0)")

# ---- build frozen versions ----
Xfroz = X.copy()
for k in range(NDYN):
    v = X[:, k]
    cnt = np.bincount(ROW, minlength=nrow); s1 = np.bincount(ROW, v, minlength=nrow)
    mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    Xfroz[:, k] = mu[ROW]

def prep(M):
    mg = np.zeros((G, M.shape[1])); np.add.at(mg, gid, M); mg /= sz[:, None]
    C = M - mg[gid]
    sd = C.std(0); lv = sd > 1e-9
    C[:, lv] /= sd[lv]; C[:, ~lv] = 0.0
    return C, lv

Xc, live = prep(X)
Xfc, livef = prep(Xfroz)

rng = np.random.default_rng(0)
pids = np.unique(plan_of); rng.shuffle(pids)
test_p = set(pids[:max(1, len(pids)//4)].tolist())
te = np.asarray([p in test_p for p in plan_of]); tr = ~te

def _sub(M, cols, mask_g):
    idx = np.nonzero(mask_g)[0]
    keep = np.concatenate([np.arange(off[i], off[i+1]) for i in idx])
    return M[keep][:, cols], sz[idx], ch[idx], idx

def fit_eval(M, cols, l2=1e-3, sub_te=None):
    Xi, szi, chi0, _ = _sub(M, cols, tr)
    gi = np.repeat(np.arange(len(szi)), szi); o = np.concatenate([[0], np.cumsum(szi)])
    chi = chi0 + o[:-1]
    def nll(w):
        z = Xi @ w
        m = np.full(len(szi), -np.inf); np.maximum.at(m, gi, z)
        e = np.exp(z - m[gi]); S = np.bincount(gi, e)
        ll = z[chi] - m - np.log(S); p = e / S[gi]
        g = (Xi.T @ p - Xi[chi].sum(0)) / len(szi) + 2*l2*w
        return -ll.mean() + l2*w@w, g
    w = minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=400)).x
    mask = te if sub_te is None else sub_te
    Xj, szj, chj0, idxj = _sub(M, cols, mask)
    gj = np.repeat(np.arange(len(szj)), szj); oj = np.concatenate([[0], np.cumsum(szj)])
    chj = chj0 + oj[:-1]
    z = Xj @ w
    m = np.full(len(szj), -np.inf); np.maximum.at(m, gj, z)
    e = np.exp(z - m[gj]); S = np.bincount(gj, e)
    nl = -(z[chj] - m - np.log(S))
    zmax = np.full(len(szj), -np.inf); np.maximum.at(zmax, gj, z)
    top1 = (z[chj] >= zmax - 1e-12)
    base = np.log(szj)
    return dict(gain=(base - nl).mean(), top1=top1.mean(),
                per_group_gain=base - nl, per_group_top1=top1, idx=idxj, w=w)

DYNC = [k for k in range(NDYN) if live[k]]
STAC = [k for k in range(NDYN, D) if live[k]]
print("\n=== (b) RANKING GAIN: real dynamic vs frozen (per-vertex-constant) dynamic ===")
res = {}
for tag, M, cols in (("static only", Xc, STAC),
                     ("frozen dynamic only", Xfc, DYNC),
                     ("static + frozen dynamic", Xfc, STAC + DYNC),
                     ("REAL dynamic only", Xc, DYNC),
                     ("static + REAL dynamic", Xc, STAC + DYNC)):
    e = fit_eval(M, cols)
    res[tag] = e
    print(f"  {tag:26s} k={len(cols):2d} gain {e['gain']:.4f}  top1 {100*e['top1']:5.1f}%", flush=True)
a = res["static + frozen dynamic"]['gain']; b = res["static + REAL dynamic"]['gain']
s = res["static only"]['gain']
print(f"\n  headroom a FROZEN rep could still capture : {a - s:+.4f} gain "
      f"({100*(a-s)/max(b-s,1e-9):.1f}% of the live-graph gain)")
print(f"  headroom that REQUIRES live-graph state   : {b - a:+.4f} gain "
      f"({100*(b-a)/max(b-s,1e-9):.1f}% of the live-graph gain)")

print("\n=== (c) gain by episode phase (test groups) ===")
e_real = res["static + REAL dynamic"]; e_froz = res["static + frozen dynamic"]
st = step_of[e_real['idx']]
for lo, hi in ((0, 10), (10, 30), (30, 50), (50, 70), (70, 96)):
    m = (st >= lo) & (st < hi)
    if m.sum() == 0: continue
    print(f"  steps {lo:2d}-{hi:2d}  n={m.sum():6d}  real gain {e_real['per_group_gain'][m].mean():.4f} "
          f"top1 {100*e_real['per_group_top1'][m].mean():5.1f}%   |  frozen gain "
          f"{e_froz['per_group_gain'][m].mean():.4f} top1 {100*e_froz['per_group_top1'][m].mean():5.1f}%")

print("\n=== (d) how fast do the dynamic features move? ===")
print("    mean |feature(t) - feature(0)| for a vertex still legal at step t")
for k in range(NDYN):
    v = X[:, k]
    first = {}
    dif = []
    for t_lo, t_hi in ((1, 10), (10, 30), (30, 60), (60, 96)):
        m = (np.repeat(step_of, sz) >= t_lo) & (np.repeat(step_of, sz) < t_hi)
        p0 = np.repeat(step_of, sz) == 0
        base = np.zeros(nrow); cnt0 = np.bincount(ROW[p0], minlength=nrow)
        s0 = np.bincount(ROW[p0], v[p0], minlength=nrow)
        base = np.where(cnt0 > 0, s0 / np.maximum(cnt0, 1), 0.0)
        dif.append((t_lo, t_hi, np.abs(v[m] - base[ROW[m]]).mean()))
    print(f"  dyn_{DYNN[k]:11s} " + "  ".join(f"t{a}-{b}:{d:.3f}" for a, b, d in dif))
