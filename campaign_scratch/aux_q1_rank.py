"""aux_q1: which per-vertex features carry POMO's decision signal (conditional logit)."""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import (build_static, extract, DYN_VERTEX_DIM,
    DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL, DV_ELIMINATED, DV_MAX_JAC, DV_IS_LEGAL,
    S_DOT_LHS_C, S_DOT_RHS_C, S_DOT_LHS_B, S_DOT_RHS_B, S_OUT_RANK, S_OUT_EXT,
    S_OUT_LN, S_DTYPE, S_INVAR, S_ROLE, S_TOPO, STATIC_DIM, MAX_RANK)

MODE = sys.argv[1]           # best | random | late
NPLAN = int(sys.argv[2])
MAXSTEP = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

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
elif MODE == "late":
    sel = sorted(rows, key=lambda r: -r[1])[:NPLAN]
else:
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.choice(len(rows), size=min(NPLAN, len(rows)), replace=False)]
print(f"MODE={MODE} nplan={len(sel)} lat_min={min(r[2] for r in sel):.0f} "
      f"lat_med={np.median([r[2] for r in sel]):.0f} lat_max={max(r[2] for r in sel):.0f}", flush=True)

# ---- feature name table (dyn 7 + static 40) ----
names = ["dyn_in_deg","dyn_out_deg","dyn_markowitz","dyn_fill","dyn_eliminated","dyn_max_jac","dyn_is_legal"]
names += [f"st_dot_lhsC{i}" for i in range(MAX_RANK)] + [f"st_dot_rhsC{i}" for i in range(MAX_RANK)]
names += [f"st_dot_lhsB{i}" for i in range(MAX_RANK)] + [f"st_dot_rhsB{i}" for i in range(MAX_RANK)]
names += ["st_out_rank"] + [f"st_out_ext{i}" for i in range(MAX_RANK)] + ["st_out_ln","st_dtype"]
names += [f"st_invar{i}" for i in range(9)]
names += ["st_role_param","st_role_data","st_role_out","st_role_seed","st_role_stopgrad"]
names += ["st_topo_idx","st_depth","st_height"]
assert len(names) == DYN_VERTEX_DIM + STATIC_DIM, (len(names), DYN_VERTEX_DIM+STATIC_DIM)
D = len(names)

# ---- replay & collect (ragged) ----
Xs, ch, off, plan_of, step_of = [], [], [0], [], []
t0 = time.time()
for pi, (s, li, lat, order) in enumerate(sel):
    env.reset()
    for t, vid in enumerate(order):
        stt = env.state()
        legal = stt.legal_vertices
        if len(legal) < 2 or t >= MAXSTEP:
            break
        sf = extract(stt, static)
        lr = np.asarray([static.row_of(j) for j in legal], np.int32)
        Xs.append(np.concatenate([sf.vert_dyn[lr], static.feat[lr]], 1).astype(np.float32))
        ch.append(legal.index(vid)); off.append(off[-1] + len(legal))
        plan_of.append(pi); step_of.append(t)
        env.step(("V", int(vid)))
    if pi % 50 == 0: print(f"  plan {pi} {time.time()-t0:.0f}s", flush=True)
X = np.concatenate(Xs, 0); del Xs
ch = np.asarray(ch, np.int32); off = np.asarray(off, np.int64)
plan_of = np.asarray(plan_of, np.int32); step_of = np.asarray(step_of, np.int32)
G = len(ch); sz = np.diff(off)
print(f"groups={G} rows={len(X)} collect_wall={time.time()-t0:.0f}s", flush=True)

# ---- within-group centre + global scale (softmax is shift-invariant per group) ----
gid = np.repeat(np.arange(G), sz)
mean_g = np.zeros((G, D), np.float64)
np.add.at(mean_g, gid, X.astype(np.float64))
mean_g /= sz[:, None]
Xc = (X - mean_g[gid]).astype(np.float64)
sd = Xc.std(0); live = sd > 1e-9
print("DEAD (no within-step variation, cannot affect any ranker):",
      [names[i] for i in range(D) if not live[i]], flush=True)
Xc[:, live] /= sd[live]
Xc[:, ~live] = 0.0

# ---- split by PLAN ----
rng = np.random.default_rng(0)
pids = np.unique(plan_of); rng.shuffle(pids)
test_p = set(pids[:max(1, len(pids)//4)].tolist())
te = np.asarray([p in test_p for p in plan_of])
tr = ~te
print(f"train groups {tr.sum()} test groups {te.sum()}", flush=True)

logC = np.log(sz.astype(np.float64))

def fit(cols, mask_g, l2=1e-3):
    cols = np.asarray(cols, int)
    idx = np.nonzero(mask_g)[0]
    keep = np.concatenate([np.arange(off[i], off[i+1]) for i in idx])
    Xi = Xc[np.ix_(keep, cols)]
    szi = sz[idx]; gi = np.repeat(np.arange(len(idx)), szi)
    o = np.concatenate([[0], np.cumsum(szi)])
    chi = ch[idx] + o[:-1]
    def nll(w):
        z = Xi @ w
        m = np.full(len(idx), -np.inf); np.maximum.at(m, gi, z)
        e = np.exp(z - m[gi]); Ssum = np.bincount(gi, e)
        ll = z[chi] - m - np.log(Ssum)
        p = e / Ssum[gi]
        g = np.zeros(len(cols))
        g -= Xi[chi].sum(0)
        g += Xi.T @ p
        return -ll.mean() + l2*w@w, g/len(idx) + 2*l2*w
    r = minimize(nll, np.zeros(len(cols)), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=400))
    return r.x

def evaluate(w, cols, mask_g):
    cols = np.asarray(cols, int)
    idx = np.nonzero(mask_g)[0]
    keep = np.concatenate([np.arange(off[i], off[i+1]) for i in idx])
    Xi = Xc[np.ix_(keep, cols)]
    szi = sz[idx]; gi = np.repeat(np.arange(len(idx)), szi)
    o = np.concatenate([[0], np.cumsum(szi)])
    chi = ch[idx] + o[:-1]
    z = Xi @ w
    m = np.full(len(idx), -np.inf); np.maximum.at(m, gi, z)
    e = np.exp(z - m[gi]); Ssum = np.bincount(gi, e)
    nll = -(z[chi] - m - np.log(Ssum))
    zmax = np.full(len(idx), -np.inf); np.maximum.at(zmax, gi, z)
    top1 = (z[chi] >= zmax - 1e-12).mean()
    # rank percentile of chosen (0 = highest scored)
    better = np.bincount(gi, (z > z[chi][gi]).astype(np.float64))
    rankpct = (better / np.maximum(szi - 1, 1)).mean()
    base = logC[idx].mean()
    return dict(nll=nll.mean(), base_nll=base,
                gain=base - nll.mean(), gain_frac=(base - nll.mean())/base,
                top1=top1, rankpct=rankpct)

live_cols = [i for i in range(D) if live[i]]
DYN = [i for i in range(DYN_VERTEX_DIM) if live[i]]
STA = [i for i in live_cols if i >= DYN_VERTEX_DIM]

def run(tag, cols):
    if not len(cols): return None
    w = fit(cols, tr)
    e = evaluate(w, cols, te)
    print(f"{tag:34s} k={len(cols):2d}  NLL {e['nll']:.4f} (base {e['base_nll']:.4f}) "
          f" gain {e['gain']:.4f} ({100*e['gain_frac']:.1f}%)  top1 {100*e['top1']:5.1f}%  "
          f"rankpct {e['rankpct']:.4f}", flush=True)
    return e, w

print("\n=== BLOCKS (test = held-out plans) ===", flush=True)
full = run("ALL live features", live_cols)
run("DYNAMIC only (live-graph)", DYN)
run("STATIC only (jaxpr-intrinsic)", STA)
run("STATIC topo only (idx,depth,height)", [i for i in STA if names[i].startswith("st_topo") or names[i] in ("st_depth","st_height")])

print("\n=== SINGLE FEATURE ALONE (sorted by gain) ===", flush=True)
singles = []
for i in live_cols:
    w = fit([i], tr); e = evaluate(w, [i], te)
    singles.append((e['gain'], i, e))
for g, i, e in sorted(singles, reverse=True):
    print(f"  {names[i]:22s} gain {g:.4f} ({100*e['gain_frac']:5.1f}%) top1 {100*e['top1']:5.1f}% "
          f"rankpct {e['rankpct']:.4f}", flush=True)

print("\n=== LEAVE-ONE-OUT from ALL (loss vs full) ===", flush=True)
loo = []
for i in live_cols:
    cols = [c for c in live_cols if c != i]
    w = fit(cols, tr); e = evaluate(w, cols, te)
    loo.append((full[0]['gain'] - e['gain'], i, e))
for d, i, e in sorted(loo, reverse=True):
    print(f"  drop {names[i]:22s} dGain {d:+.5f}  top1 {100*e['top1']:5.1f}% rankpct {e['rankpct']:.4f}", flush=True)

print("\n=== DYNAMIC LOO (within DYNAMIC-only model) ===", flush=True)
dfull = evaluate(fit(DYN, tr), DYN, te)
for i in DYN:
    cols = [c for c in DYN if c != i]
    w = fit(cols, tr); e = evaluate(w, cols, te)
    print(f"  drop {names[i]:22s} dGain {dfull['gain']-e['gain']:+.5f}  top1 {100*e['top1']:5.1f}%", flush=True)

print("\n=== ADD-ONE dynamic on top of ALL-STATIC (incremental value of live-graph info) ===", flush=True)
sfull = evaluate(fit(STA, tr), STA, te)
print(f"  STATIC base gain {sfull['gain']:.4f} top1 {100*sfull['top1']:.1f}%", flush=True)
for i in DYN:
    cols = STA + [i]
    w = fit(cols, tr); e = evaluate(w, cols, te)
    print(f"  +{names[i]:22s} dGain {e['gain']-sfull['gain']:+.5f} top1 {100*e['top1']:5.1f}%", flush=True)
w = fit(STA + DYN, tr); e = evaluate(w, STA + DYN, te)
print(f"  +ALL DYNAMIC            dGain {e['gain']-sfull['gain']:+.5f} top1 {100*e['top1']:5.1f}%", flush=True)

print("\n=== REDUNDANCY: within-step-centred correlation (|r|>0.5 pairs) ===", flush=True)
C = np.corrcoef(Xc[:, live_cols].T)
for a in range(len(live_cols)):
    for b in range(a+1, len(live_cols)):
        if abs(C[a, b]) > 0.5:
            print(f"  {names[live_cols[a]]:22s} ~ {names[live_cols[b]]:22s} r={C[a,b]:+.3f}", flush=True)
print("\nmarkowitz vs in+out deg: r(mark, in)=%.3f r(mark,out)=%.3f" % (
    np.corrcoef(Xc[:, DV_MARKOWITZ], Xc[:, DV_IN_DEG])[0,1],
    np.corrcoef(Xc[:, DV_MARKOWITZ], Xc[:, DV_OUT_DEG])[0,1]), flush=True)
# residual of markowitz on (in,out)
A = np.stack([Xc[:, DV_IN_DEG], Xc[:, DV_OUT_DEG], np.ones(len(Xc))], 1)
coef, *_ = np.linalg.lstsq(A, Xc[:, DV_MARKOWITZ], rcond=None)
res = Xc[:, DV_MARKOWITZ] - A @ coef
print("markowitz R^2 explained by (in_deg,out_deg): %.4f" % (1 - res.var()/Xc[:, DV_MARKOWITZ].var()), flush=True)
