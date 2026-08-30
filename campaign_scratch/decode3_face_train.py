"""decode3_face_train: can the FACE level be read off the tokenized
append-only jaxpr ALONE, under the 2026-08-14 architecture?

  --arm new         THE LIVE PATH, exactly: [ctx_i || ctx_j || face_latent]
                    at 3E.  ctx_i/ctx_j are the two ENDPOINT vertices' own
                    contexts, gathered from the pointer's per-vertex contexts
                    (identity pool + participation dynamic, ctx_proj'd back to
                    E); face_latent is the mean palimpsa row over that face's
                    OWN token chunk.  An endpoint that is a jaxpr input has no
                    vertex, so its block is ZERO -- never a substitute vector.
                    NO explicit input of any kind.
  --arm new_sizes   `new` + the face's EXPLICIT log2 extents.  The upper
                    bound deferred task #94 would buy, measured on the SAME
                    representation, so the gap is attributable.
  --arm sizes_only  extents + the (deleted) hand vertex features and nothing
                    else -- how much of each target is pure extent
                    arithmetic.
  --arm old_today   the PREVIOUS architecture's face input re-measured here:
                    the eliminated vertex's context + the chunk mean + its
                    hand features.

METRIC.  WITHIN-STEP R2 (faces of ONE elimination step of ONE trajectory) is
the number that matters: anything per-vertex is constant inside a step group
and contributes exactly 0, so within-step isolates the per-face channel.
Steps-to-threshold at 0.6 / 0.8 / 0.9, train printed beside test.

POSITIVE CONTROLS: stat_ln_i / stat_ln_j, the log2 numel of the two endpoint
variables -- purely static, must reach ~1.0 or the harness is broken.
DEGENERATE: n_compressed and is_lowrank are identically 0 on this graph (and
there is no LOWRANK action), so they are reported as degenerate, not as R2.
"""
import argparse, json, os, time
import numpy as np
import decode3_arch as ARCH
import jax, jax.numpy as jnp, jax.random as jrand
import equinox as eqx, optax

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--arm", default="new",
               choices=["new", "new_sizes", "sizes_only", "old_today"])
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--chunk-cap", type=int, default=2048)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--trace-only", action="store_true",
               help="abstract-trace forward AND backward, then exit -- catches every shape/vmap bug in seconds instead of paying a 20-minute XLA compile per arm")
A = P.parse_args()

d = np.load(A.data, allow_pickle=True)
TOK, OWN, EQN, DID = d["tok"], d["own"], d["eqn"], d["did"]
NTOK, MODE, VFEAT, PART = d["ntok"], d["mode"], d["vfeat"], d["part"]
FN = [str(x) for x in d["f_names"]]
NV = VFEAT.shape[0]
N, L = TOK.shape
NFT = len(FN)
NDIM = d["f_ext"].shape[1]
NSTEP = PART.shape[1]
E = A.embd_dim
KV = VFEAT.shape[1]
print(f"data N={N} L={L} NV={NV} nsteps={NSTEP} faces={len(d['f_traj'])} "
      f"arm={'B' if int(d['shapes']) else 'A'}", flush=True)

f_traj, f_q = d["f_traj"], d["f_q"]
f_start, f_split = d["f_start"], d["f_split"]
f_si, f_sj, f_v = d["f_si"], d["f_sj"], d["f_v"]
f_tgt, f_ext = d["f_tgt"], d["f_ext"]
cnt = np.bincount(f_traj, minlength=N)
FMAX = int(cnt.max())
print(f"faces/traj min={cnt.min()} max={FMAX} mean={cnt.mean():.1f}", flush=True)
FQ = np.zeros((N, FMAX), np.int32); FS = np.zeros((N, FMAX), np.int32)
FP = np.zeros((N, FMAX), np.int32); FI = np.full((N, FMAX), NV, np.int32)
FJ = np.full((N, FMAX), NV, np.int32); FV = np.zeros((N, FMAX), np.int32)
FT = np.zeros((N, FMAX, NFT), np.float32); FE = np.zeros((N, FMAX, NDIM), np.float32)
FM = np.zeros((N, FMAX), bool)
_w = np.zeros(N, np.int32)
for k in range(len(f_traj)):
    i = int(f_traj[k]); j = int(_w[i]); _w[i] += 1
    FQ[i, j] = f_q[k]; FS[i, j] = f_start[k]; FP[i, j] = f_split[k]
    FI[i, j] = f_si[k] if f_si[k] >= 0 else NV
    FJ[i, j] = f_sj[k] if f_sj[k] >= 0 else NV
    FV[i, j] = f_v[k]; FT[i, j] = f_tgt[k]; FE[i, j] = f_ext[k]; FM[i, j] = True
CLEN = np.maximum(FP - FS, 1)
CAP = int(min(A.chunk_cap, max(1, CLEN[FM].max())))
print(f"chunk len: mean {CLEN[FM].mean():.0f} max {CLEN[FM].max()} cap {CAP} "
      f"truncated {(CLEN[FM] > CAP).mean()*100:.1f}%", flush=True)

rng = np.random.default_rng(A.seed)
te = []
for m in (0, 1):
    ids = np.nonzero(MODE == m)[0]; rng.shuffle(ids)
    te.append(ids[: A.n_test // 2])
te_idx = np.sort(np.concatenate(te))
tr_idx = np.setdiff1d(np.arange(N), te_idx)
print(f"train {len(tr_idx)} test {len(te_idx)}", flush=True)

sel = FM[tr_idx].reshape(-1)
flat = FT[tr_idx].reshape(-1, NFT)[sel]
MU = flat.mean(0); SD = np.maximum(flat.std(0), 1e-6)
DEGEN = [FN[c] for c in range(NFT) if flat[:, c].std() < 1e-9]
print("target mu", np.round(MU, 2), "\n       sd", np.round(SD, 2), flush=True)
print("DEGENERATE (identically constant on train):", DEGEN, flush=True)
FTZ = ((FT - MU) / SD).astype(np.float32)
EXT_MU = FE[tr_idx].reshape(-1, NDIM)[sel].mean(0)
EXT_SD = np.maximum(FE[tr_idx].reshape(-1, NDIM)[sel].std(0), 1e-6)
FEZ = ((FE - EXT_MU) / EXT_SD).astype(np.float32)

STEP_W = 16384
BUCK = np.minimum(((NTOK + STEP_W - 1) // STEP_W) * STEP_W, L)
print("buckets", {int(w): int((BUCK == w).sum()) for w in np.unique(BUCK)},
      flush=True)

k = jrand.split(jrand.PRNGKey(A.seed), 4)
agent = ARCH.build(E, A.num_layers, A.num_heads, A.pointer_blocks, NV, k[0])
VF = jnp.asarray(VFEAT)
VFP = jnp.concatenate([VF, jnp.zeros((1, KV))], 0)   # slot NV = "no vertex"

DIN = {"new": 3 * E, "new_sizes": 3 * E + NDIM,
       "sizes_only": NDIM + KV, "old_today": 2 * E + KV}[A.arm]
print(f"arm {A.arm} din={DIN}", flush=True)


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NFT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


head = Readout(DIN, A.head_width, k[1])


def per_traj(agent, head, tok, eqn, did, own, ntok, part, fq, fs, fp, fi, fj,
             fv, fe, W):
    rows, w = ARCH.encode(agent, tok, eqn, ntok, W)
    ident = ARCH.identity(agent, rows, w, own, did, NV)
    tab = ARCH.memory_tables(rows, w, own, eqn, did, part, NV, NSTEP)

    def _ctx(t):
        S, C = ARCH.mem_at(tab, t, NV, E, NSTEP)
        c, _ = ARCH.heads(agent, S, C, ident)
        return c
    ctx_all = jax.vmap(_ctx)(jnp.arange(NSTEP, dtype=jnp.int32))   # (T, NV, E)
    # index NV is the explicit "endpoint is a jaxpr input" ZERO row.
    ctx_all = jnp.concatenate(
        [ctx_all, jnp.zeros((NSTEP, 1, E), jnp.float32)], 1)

    def one(q, s_, p_, i_, j_, v_, e_):
        lat = ARCH.face_latent(rows, w, s_, p_, CAP)
        if A.arm == "new":
            return jnp.concatenate([ctx_all[q, i_], ctx_all[q, j_], lat])
        if A.arm == "new_sizes":
            return jnp.concatenate([ctx_all[q, i_], ctx_all[q, j_], lat, e_])
        if A.arm == "sizes_only":
            return jnp.concatenate([e_, VFP[v_]])
        return jnp.concatenate([ctx_all[q, v_], lat, VFP[v_]])

    h = jax.vmap(one)(fq, fs, fp, fi, fj, fv, fe)
    return head(h)


def batched(agent, head, *a):
    return jax.vmap(per_traj, in_axes=(None, None) + (0,) * 13 + (None,))(
        agent, head, *a)


def loss_fn(params, statics, tok, eqn, did, own, ntok, part, fq, fs, fp, fi,
            fj, fv, fe, tz, fm, W):
    agent, head = eqx.combine(params, statics)
    pr = batched(agent, head, tok, eqn, did, own, ntok, part, fq, fs, fp, fi,
                 fj, fv, fe, W)
    m = fm[..., None].astype(jnp.float32)
    return jnp.sum(((pr - tz) ** 2) * m) / jnp.maximum(jnp.sum(m), 1.0)


params, statics = eqx.partition((agent, head), eqx.is_inexact_array)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(A.lr))
ostate = opt.init(params)


@eqx.filter_jit
def train_step(params, ostate, statics, *a):
    v, g = jax.value_and_grad(loss_fn)(params, statics, *a)
    u, ostate = opt.update(g, ostate, params)
    return eqx.apply_updates(params, u), ostate, v


@eqx.filter_jit
def eval_step(params, statics, *a):
    agent, head = eqx.combine(params, statics)
    return batched(agent, head, *a)


def dev(idx, W):
    return (jnp.asarray(TOK[idx, :W]), jnp.asarray(EQN[idx, :W]),
            jnp.asarray(DID[idx, :W].astype(np.int32)),
            jnp.asarray(OWN[idx, :W]), jnp.asarray(NTOK[idx]),
            jnp.asarray(PART[idx].astype(np.float32)),
            jnp.asarray(FQ[idx]), jnp.asarray(FS[idx]), jnp.asarray(FP[idx]),
            jnp.asarray(FI[idx]), jnp.asarray(FJ[idx]), jnp.asarray(FV[idx]),
            jnp.asarray(FEZ[idx]))


if A.trace_only:
    # Abstract trace of forward AND backward: catches every shape /
    # vmap / concat bug in seconds, without paying the ~20 min XLA
    # compile of a 16k-step palimpsa scan that a real step needs.
    _b = tr_idx[:A.batch]
    _w = int(BUCK[_b].max())
    _o = jax.eval_shape(
        lambda p, *a: jax.value_and_grad(loss_fn)(p, statics, *a, _w),
        params, *dev(_b, _w), jnp.asarray(FTZ[_b]), jnp.asarray(FM[_b]))
    print(f"TRACE OK arm={A.arm} window={_w} loss={_o[0]}", flush=True)
    raise SystemExit(0)


def collect(params, idx):
    out = np.zeros((len(idx), FMAX, NFT), np.float32)
    for wv in np.unique(BUCK[idx]):
        sub = np.nonzero(BUCK[idx] == wv)[0]
        for i in range(0, len(sub), A.batch):
            b = sub[i:i + A.batch]
            out[b] = np.asarray(eval_step(params, statics, *dev(idx[b], int(wv)),
                                          int(wv)))
    return out


def r2_report(pred, idx):
    y = FTZ[idx]; m = FM[idx]; q = FQ[idx]
    rep = {}
    for c, nm in enumerate(FN):
        a = y[:, :, c][m]; b = pred[:, :, c][m]
        r2 = float("nan") if a.var() < 1e-12 else 1.0 - ((a - b) ** 2).mean() / a.var()
        yy, pp = [], []
        for t in range(len(idx)):
            steps = np.unique(q[t][m[t]])
            for s in steps:
                s2 = m[t] & (q[t] == s)
                if s2.sum() < 2:
                    continue
                aa = y[t, :, c][s2]; bb = pred[t, :, c][s2]
                yy.append(aa - aa.mean()); pp.append(bb - bb.mean())
        if yy:
            yy = np.concatenate(yy); pp = np.concatenate(pp)
            r2w = (float("nan") if yy.var() < 1e-12 else
                   1.0 - ((yy - pp) ** 2).mean() / yy.var())
            nw = len(yy)
        else:
            r2w, nw = float("nan"), 0
        rep[nm] = dict(r2=float(r2), r2w=float(r2w), n=int(m.sum()), nw=nw)
    return rep


def _group_const(keyf):
    tab = {}
    for t in tr_idx:
        for j in np.nonzero(FM[t])[0]:
            tab.setdefault(keyf(t, j), []).append(FTZ[t, j])
    tab = {k_: np.mean(v, 0) for k_, v in tab.items()}
    gm = np.zeros(NFT, np.float32)
    out = np.zeros((len(te_idx), FMAX, NFT), np.float32)
    for a, t in enumerate(te_idx):
        for j in np.nonzero(FM[t])[0]:
            out[a, j] = tab.get(keyf(t, j), gm)
    return out


base_rep = {
    "const_global": r2_report(np.zeros((len(te_idx), FMAX, NFT), np.float32),
                              te_idx),
    "const_facekey": r2_report(
        _group_const(lambda t, j: (int(FI[t, j]), int(FJ[t, j]))), te_idx),
}
print("\n=== FACE BASELINES (test) R2 / R2_within ===", flush=True)
for nm, rep in base_rep.items():
    print(f"  {nm:16s} " + "  ".join(
        f"{k_}:{v['r2']:6.3f}/{v['r2w']:6.3f}" for k_, v in rep.items()),
        flush=True)

EVALS = sorted(set([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 140, 200,
                    280, 400, 550, 750, 1000, 1400, 2000, 2500, 3000]
                   + [A.steps]))
EVALS = [s for s in EVALS if s <= A.steps]

curve = []
t0 = time.time()
tr_by_b = {int(wv): tr_idx[BUCK[tr_idx] == wv] for wv in np.unique(BUCK[tr_idx])}
ws = np.asarray(list(tr_by_b))
pw = np.asarray([len(tr_by_b[int(wv)]) for wv in ws], np.float64); pw /= pw.sum()
tr_eval = tr_idx[:len(te_idx)]
for step in range(1, A.steps + 1):
    wv = int(rng.choice(ws, p=pw))
    pool = tr_by_b[wv]
    b = rng.choice(pool, size=min(A.batch, len(pool)),
                   replace=len(pool) < A.batch)
    params, ostate, v = train_step(params, ostate, statics, *dev(b, wv),
                                   jnp.asarray(FTZ[b]), jnp.asarray(FM[b]), wv)
    if step in EVALS:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_eval)
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_eval)
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0, degenerate=DEGEN))
        print(f"--- eval @ {step}  loss {float(v):.4f}  "
              f"wall {time.time()-t0:.0f}s ---", flush=True)
        for nm in FN:
            tag = "  DEGENERATE" if nm in DEGEN else ""
            print(f"  {nm:13s} TEST r2 {rte[nm]['r2']:7.3f} within "
                  f"{rte[nm]['r2w']:7.3f}   TRAIN within "
                  f"{rtr[nm]['r2w']:7.3f}{tag}", flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), baselines=base_rep, curve=curve), f)
print("done", time.time() - t0, flush=True)
