"""decode2_face_train: is a PER-FACE approximation decision decodable from
what the face head actually sees?

Same protocol as decode_train (real palimpsa encoder trained end-to-end with a
small read-out MLP, held-out TRAJECTORIES, R2 against z-scored targets), but
the query is a FACE, not a vertex.

  --arm today     v_context[v]  +  MEAN of the face's own token chunk
                  + the eliminated vertex's static features.
                  This is a STRICT SUPERSET of what UnifiedFacePolicy builds
                  today (ppo.py:2364 -> unified_face_policy._ctx: it ADDS
                  v_context and the chunk mean and pools them through the
                  AxisSetEncoder with the VERTEX's axis features).  Concat
                  beats add, so a failure here is a failure of the live path.
  --arm today_bnd same, but the chunk's LAST causal row instead of its mean.
  --arm endpoints [ctx_i, ctx_j, latent] -- the two ENDPOINT vertex
                  representations gathered on demand plus the palimpsa latent
                  at the decision point, modelled on POMO's face_embed
                  (elimrl/encoder.py:181).
  --arm endpoints_sizes  the above + the face's EXPLICIT log2 extents: the
                  upper bound that plumbing task #94 would buy.
"""
import argparse, json, os, sys, time
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_CHUNKED_EXTEND", "1")
os.environ.setdefault("ALPHAGRAD_CHUNK_BLOCK", "512")
import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand
import equinox as eqx, optax
from types import SimpleNamespace

from alphagrad.approx.ppo import _build_agent

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--arm", default="today",
               choices=["today", "today_bnd", "today_sizes", "endpoints",
                        "endpoints_sizes", "sizes_only"])
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--eval-every", type=int, default=100)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--chunk-cap", type=int, default=1536)
P.add_argument("--seed", type=int, default=0)
A = P.parse_args()

d = np.load(A.data, allow_pickle=True)
TOK, OWN, EQN, DID = d["tok"], d["own"], d["eqn"], d["did"]
NTOK, MODE, VFEAT = d["ntok"], d["mode"], d["vfeat"]
FN = [str(x) for x in d["f_names"]]
NV = VFEAT.shape[0]
N, L = TOK.shape
NFT = len(FN)
NDIM = d["f_ext"].shape[1]
NSTEP = int(DID.max()) + 1
E = A.embd_dim
print(f"data N={N} L={L} NV={NV} nsteps={NSTEP} faces={len(d['f_traj'])} "
      f"targets={FN}", flush=True)

# ---------------- per-trajectory padded face tables ----------------------
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

# ---------------- split by trajectory, stratified on the order mode -------
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
print("target mu", np.round(MU, 2), "\n       sd", np.round(SD, 2), flush=True)
FTZ = ((FT - MU) / SD).astype(np.float32)
EXT_MU = FE[tr_idx].reshape(-1, NDIM)[sel].mean(0)
EXT_SD = np.maximum(FE[tr_idx].reshape(-1, NDIM)[sel].std(0), 1e-6)
FEZ = ((FE - EXT_MU) / EXT_SD).astype(np.float32)

STEP_W = 16384
BUCK = np.minimum(((NTOK + STEP_W - 1) // STEP_W) * STEP_W, L)
print("buckets", {int(w): int((BUCK == w).sum()) for w in np.unique(BUCK)},
      flush=True)

args = SimpleNamespace(vocab_size=512, embd_dim=E, op_embd_dim=8,
                       num_layers=A.num_layers, num_heads=A.num_heads,
                       hidden_dim=64, value_dims="64,32", set_pointer=True,
                       set_pointer_blocks=A.pointer_blocks,
                       dynamic_substeps=False, no_approx_head=True,
                       live_faces=False, face_actions=False,
                       unified_head=False, unified_face_head=False,
                       max_substeps=1)
k = jrand.split(jrand.PRNGKey(A.seed), 6)
agent = _build_agent(args, NV, 1, 1, k[0])
VF = jnp.asarray(VFEAT)
KV = VFEAT.shape[1]
VFP = jnp.concatenate([VF, jnp.zeros((1, KV))], 0)      # slot NV = "no vertex"

DIN = {"today": 2 * E + KV,
       "today_bnd": 2 * E + KV,
       "today_sizes": 2 * E + KV + NDIM,
       "endpoints": 3 * E + 2 * KV,
       "endpoints_sizes": 3 * E + 2 * KV + NDIM,
       "sizes_only": NDIM + KV}[A.arm]
print(f"arm {A.arm} din={DIN}", flush=True)


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NFT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


head = Readout(DIN, A.head_width, k[1])


def per_traj(agent, head, tok, eqn, did, own, ntok, fq, fs, fp, fi, fj, fv,
             fe, W):
    """One trajectory: encode once, rebuild the per-step vertex memory, then
    read out every face."""
    c0 = agent.carry_init()
    _, rows, valid, _ = agent.encode_extend(c0, tok, eqn, ntok, window=W,
                                            start=0, chunk=0)
    w = valid.astype(jnp.float32)
    rw = rows * w[:, None]

    # --- per-step vertex memory (base fold + the deltas that precede t) ----
    oid = jnp.where(own < 0, NV, jnp.minimum(own, NV - 1)).astype(jnp.int32)
    is_base = (did < 0)
    bw = w * is_base.astype(jnp.float32)
    base_s = jax.ops.segment_sum(rows * bw[:, None], oid, num_segments=NV + 1)
    base_c = jax.ops.segment_sum(bw, oid, num_segments=NV + 1)
    dsel = (~is_base).astype(jnp.float32) * w
    d_id = jnp.where(did < 0, 0, did).astype(jnp.int32)
    dsum = jax.ops.segment_sum(rows * (dsel[:, None]), d_id, num_segments=NSTEP)
    dcnt = jax.ops.segment_sum(dsel, d_id, num_segments=NSTEP)
    downer = jax.ops.segment_max(
        jnp.where(is_base, -1, oid), d_id, num_segments=NSTEP)
    downer = jnp.clip(downer, 0, NV)
    oh = jax.nn.one_hot(downer, NV + 1)                       # (T, NV+1)
    contrib_s = oh[:, :, None] * dsum[:, None, :]             # (T, NV+1, E)
    contrib_c = oh * dcnt[:, None]
    cs = jnp.cumsum(contrib_s, axis=0) - contrib_s            # exclusive
    cc = jnp.cumsum(contrib_c, axis=0) - contrib_c
    S = base_s[None] + cs
    C = base_c[None] + cc

    def _ctx(s_, c_):
        _, ctx, _ = agent.heads_from_memory(s_, c_, vertex_features=VF)
        return ctx
    ctx_all = jax.vmap(_ctx)(S, C)
    # slot NV is the explicit "endpoint is an input var / no vertex" row
    ctx_all = jnp.concatenate(
        [ctx_all[:, :NV], jnp.zeros((ctx_all.shape[0], 1, E))], 1)

    # --- per-face gathers -------------------------------------------------
    ar = jnp.arange(CAP, dtype=jnp.int32)

    def one(q, s_, p_, i_, j_, v_, e_):
        idx = jnp.minimum(s_ + ar, W - 1)
        m = ((s_ + ar) < p_).astype(jnp.float32) * w[idx]
        ch = jnp.sum(rows[idx] * m[:, None], 0) / jnp.maximum(m.sum(), 1.0)
        last = rows[jnp.clip(p_ - 1, 0, W - 1)]
        vc = ctx_all[q, v_]
        if A.arm == "today":
            return jnp.concatenate([vc, ch, VFP[v_]])
        if A.arm == "today_bnd":
            return jnp.concatenate([vc, last, VFP[v_]])
        if A.arm == "today_sizes":
            return jnp.concatenate([vc, ch, VFP[v_], e_])
        if A.arm == "endpoints":
            return jnp.concatenate([ctx_all[q, i_], ctx_all[q, j_], last,
                                    VFP[i_], VFP[j_]])
        if A.arm == "endpoints_sizes":
            return jnp.concatenate([ctx_all[q, i_], ctx_all[q, j_], last,
                                    VFP[i_], VFP[j_], e_])
        return jnp.concatenate([e_, VFP[v_]])

    h = jax.vmap(one)(fq, fs, fp, fi, fj, fv, fe)             # (FMAX, DIN)
    return head(h)


def batched(agent, head, *a):
    return jax.vmap(per_traj, in_axes=(None, None) + (0,) * 12 + (None,))(
        agent, head, *a)


def loss_fn(params, statics, tok, eqn, did, own, ntok, fq, fs, fp, fi, fj, fv,
            fe, tz, fm, W):
    agent, head = eqx.combine(params, statics)
    pr = batched(agent, head, tok, eqn, did, own, ntok, fq, fs, fp, fi, fj, fv,
                 fe, W)
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
            jnp.asarray(FQ[idx]), jnp.asarray(FS[idx]), jnp.asarray(FP[idx]),
            jnp.asarray(FI[idx]), jnp.asarray(FJ[idx]), jnp.asarray(FV[idx]),
            jnp.asarray(FEZ[idx]))


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
        # within (trajectory, elimination step) groups of >= 2 faces
        yy, pp = [], []
        for t in range(len(idx)):
            steps = np.unique(q[t][m[t]])
            for s in steps:
                sel = m[t] & (q[t] == s)
                if sel.sum() < 2:
                    continue
                aa = y[t, :, c][sel]; bb = pred[t, :, c][sel]
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


# ---------------- non-learned baselines -----------------------------------
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
    "const_vertex": r2_report(_group_const(lambda t, j: int(FV[t, j])), te_idx),
    "const_facekey": r2_report(
        _group_const(lambda t, j: (int(FI[t, j]), int(FJ[t, j]))), te_idx),
    "const_facekey_step": r2_report(
        _group_const(lambda t, j: (int(FI[t, j]), int(FJ[t, j]),
                                   int(FQ[t, j]) // 10)), te_idx),
}
print("\n=== FACE BASELINES (test) R2 / R2_within ===", flush=True)
for nm, rep in base_rep.items():
    print(f"  {nm:20s} " + "  ".join(
        f"{k_}:{v['r2']:6.3f}/{v['r2w']:6.3f}" for k_, v in rep.items()),
        flush=True)

curve = []
t0 = time.time()
tr_by_b = {int(wv): tr_idx[BUCK[tr_idx] == wv] for wv in np.unique(BUCK[tr_idx])}
ws = np.asarray(list(tr_by_b))
pw = np.asarray([len(tr_by_b[int(wv)]) for wv in ws], np.float64); pw /= pw.sum()
for step in range(1, A.steps + 1):
    wv = int(rng.choice(ws, p=pw))
    pool = tr_by_b[wv]
    b = rng.choice(pool, size=min(A.batch, len(pool)),
                   replace=len(pool) < A.batch)
    params, ostate, v = train_step(params, ostate, statics, *dev(b, wv),
                                   jnp.asarray(FTZ[b]), jnp.asarray(FM[b]), wv)
    if step % 25 == 0:
        print(f"step {step:5d} loss {float(v):.4f} wall {time.time()-t0:.0f}s",
              flush=True)
    if step % A.eval_every == 0 or step == A.steps:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_idx[:len(te_idx)])
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_idx[:len(te_idx)])
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0))
        print(f"--- eval @ {step}  (test R2 | train R2) ---", flush=True)
        for nm in FN:
            print(f"  {nm:13s} test {rte[nm]['r2']:7.3f} (w {rte[nm]['r2w']:6.3f})"
                  f"   train {rtr[nm]['r2']:7.3f}", flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), baselines=base_rep, curve=curve), f)
print("done", time.time() - t0, flush=True)
