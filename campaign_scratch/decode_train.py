"""decode_train: can palimpsa + the set-pointer's vertex memory DECODE the
live per-vertex graph features from its own representation?

Trains the REAL path end-to-end (token embedding -> palimpsa encoder ->
per-vertex memory fold -> SetPointerVertexPolicy.from_vertex_memory ->
Agent.heads_from_memory) plus a small read-out MLP that is queried with ONE
vertex's per-candidate representation, against the live ElimEnv features at
the same step.

  --model slot   the real path (default)
  --model attn   same encoder rows, but the read-out cross-attends the WHOLE
                 row sequence (architecture ablation: routing vs content)
"""
import argparse, json, os, sys, time
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_CHUNKED_EXTEND", "1")
os.environ.setdefault("ALPHAGRAD_CHUNK_BLOCK", "512")
import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand, jax.nn as jnn
import equinox as eqx, optax
from types import SimpleNamespace

from alphagrad.approx.ppo import _build_agent

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--model", default="slot", choices=["slot", "attn"])
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=8)
P.add_argument("--lr", type=float, default=3e-4)
P.add_argument("--eval-every", type=int, default=100)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--seed", type=int, default=0)
A = P.parse_args()

d = np.load(A.data, allow_pickle=True)
TOK, OWN, EQN = d["tok"], d["own"], d["eqn"]
NTOK, PREF, TGT, LEG = d["ntok"], d["pref"], d["tgt"], d["legal"]
MODE, VFEAT = d["mode"], d["vfeat"]
QSTEPS = list(d["qsteps"]); NAMES = [str(x) for x in d["names"]]
N, L = TOK.shape
NV = VFEAT.shape[0]
NQ, NT = len(QSTEPS), len(NAMES)
print("[decode] building", flush=True)
print(f"data N={N} L={L} NV={NV} qsteps={QSTEPS} targets={NAMES}", flush=True)

# ---- split by trajectory, stratified over the two order distributions ----
rng = np.random.default_rng(A.seed)
te_idx = []
for m in (0, 1):
    ids = np.nonzero(MODE == m)[0]; rng.shuffle(ids)
    te_idx.append(ids[: A.n_test // 2])
te_idx = np.sort(np.concatenate(te_idx))
tr_idx = np.setdiff1d(np.arange(N), te_idx)
print(f"train {len(tr_idx)} test {len(te_idx)}", flush=True)

# ---- z-scoring from the TRAIN pool (legal candidates only) ----
sel = LEG[tr_idx].reshape(-1)
flat = TGT[tr_idx].reshape(-1, NT)[sel]
MU = flat.mean(0); SD = np.maximum(flat.std(0), 1e-6)
print("target mu", np.round(MU, 3), "sd", np.round(SD, 3), flush=True)
TGTZ = ((TGT - MU) / SD).astype(np.float32)

# ---- length buckets so a batch's scan window tracks its own streams ----
STEP_W = 16384
BUCK = np.minimum(((NTOK + STEP_W - 1) // STEP_W) * STEP_W, L)
print("bucket sizes", {int(w): int((BUCK == w).sum()) for w in np.unique(BUCK)},
      flush=True)

args = SimpleNamespace(vocab_size=512, embd_dim=A.embd_dim, op_embd_dim=8,
                       num_layers=A.num_layers, num_heads=A.num_heads,
                       hidden_dim=64, value_dims="64,32", set_pointer=True,
                       set_pointer_blocks=A.pointer_blocks,
                       dynamic_substeps=False, no_approx_head=True,
                       live_faces=False, face_actions=False,
                       unified_head=False, unified_face_head=False,
                       max_substeps=1)
k = jrand.split(jrand.PRNGKey(A.seed), 6)
agent = _build_agent(args, NV, 1, 1, k[0])
E = A.embd_dim


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


class AttnRead(eqx.Module):
    """Read-out that cross-attends the WHOLE encoder row sequence."""
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    own_emb: eqx.nn.Embedding
    mlp: eqx.nn.MLP
    nh: int = eqx.field(static=True)

    def __init__(self, E, width, nh, key):
        ks = jrand.split(key, 5)
        self.q_proj = eqx.nn.Linear(2 * E, E, key=ks[0])
        self.k_proj = eqx.nn.Linear(2 * E, E, key=ks[1])
        self.v_proj = eqx.nn.Linear(2 * E, E, key=ks[2])
        self.own_emb = eqx.nn.Embedding(NV + 1, E, key=ks[3])
        self.mlp = eqx.nn.MLP(2 * E, NT, width, depth=2, key=ks[4])
        self.nh = nh

    def __call__(self, hslot, rows, own, ok):
        # hslot (NV, E) the pointer's own per-vertex repr = the query source
        kv_in = jnp.concatenate(
            [rows, jax.vmap(self.own_emb)(jnp.where(own < 0, NV, own))], -1)
        K = jax.vmap(self.k_proj)(kv_in)
        V = jax.vmap(self.v_proj)(kv_in)
        Q = jax.vmap(self.q_proj)(jnp.concatenate([hslot, hslot], -1))
        H, dh = self.nh, E // self.nh
        Qh = Q.reshape(NV, H, dh); Kh = K.reshape(-1, H, dh); Vh = V.reshape(-1, H, dh)
        s = jnp.einsum("vhd,thd->hvt", Qh, Kh) / jnp.sqrt(jnp.float32(dh))
        s = jnp.where(ok[None, None, :], s, -jnp.inf)
        p = jnn.softmax(s, axis=-1)
        o = jnp.einsum("hvt,thd->vhd", p, Vh).reshape(NV, E)
        return jax.vmap(self.mlp)(jnp.concatenate([hslot, o], -1))


if A.model == "slot":
    head = Readout(E, A.head_width, k[1])
else:
    head = AttnRead(E, A.head_width, A.num_heads, k[1])

VF = jnp.asarray(VFEAT)


def encode_one(agent, tok, eqn, ntok, W):
    c0 = agent.carry_init()
    _, rows, valid, _ = agent.encode_extend(c0, tok, eqn, ntok, window=W,
                                            start=0, chunk=0)
    return rows, valid


def fold(rows, own, w):
    ids = jnp.where(own < 0, NV, jnp.minimum(own, NV - 1)).astype(jnp.int32)
    s = jax.ops.segment_sum(rows * w[:, None], ids, num_segments=NV + 1)
    c = jax.ops.segment_sum(w, ids, num_segments=NV + 1)
    return s, c


def predict_one(agent, head, tok, own, eqn, ntok, pref, W):
    rows, valid = encode_one(agent, tok, eqn, ntok, W)
    ar = jnp.arange(W, dtype=jnp.int32)

    def per_q(p):
        w = (valid & (ar < p)).astype(jnp.float32)
        s, c = fold(rows, own, w)
        _, ctx, _ = agent.heads_from_memory(s, c, vertex_features=VF)
        if A.model == "slot":
            return head(ctx)
        return head(ctx, rows, own, w > 0)

    return jax.vmap(per_q)(pref)            # (NQ, NV, NT)


def batched_pred(agent, head, tok, own, eqn, ntok, pref, W):
    return jax.vmap(predict_one, in_axes=(None, None, 0, 0, 0, 0, 0, None))(
        agent, head, tok, own, eqn, ntok, pref, W)


def loss_fn(params, statics, tok, own, eqn, ntok, pref, tz, lm, W):
    agent, head = eqx.combine(params, statics)
    pr = batched_pred(agent, head, tok, own, eqn, ntok, pref, W)
    m = lm[..., None].astype(jnp.float32)
    return jnp.sum(((pr - tz) ** 2) * m) / jnp.maximum(jnp.sum(m), 1.0)


params, statics = eqx.partition((agent, head), eqx.is_inexact_array)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(A.lr))
ostate = opt.init(params)


@eqx.filter_jit
def train_step(params, ostate, statics, tok, own, eqn, ntok, pref, tz, lm, W):
    v, g = jax.value_and_grad(loss_fn)(params, statics, tok, own, eqn, ntok,
                                       pref, tz, lm, W)
    u, ostate = opt.update(g, ostate, params)
    return eqx.apply_updates(params, u), ostate, v


@eqx.filter_jit
def eval_step(params, statics, tok, own, eqn, ntok, pref, W):
    agent, head = eqx.combine(params, statics)
    return batched_pred(agent, head, tok, own, eqn, ntok, pref, W)


def dev(idx, W):
    return (jnp.asarray(TOK[idx, :W]), jnp.asarray(OWN[idx, :W]),
            jnp.asarray(EQN[idx, :W]), jnp.asarray(NTOK[idx]),
            jnp.asarray(PREF[idx]))


def collect(params, idx):
    out = np.zeros((len(idx), NQ, NV, NT), np.float32)
    for w in np.unique(BUCK[idx]):
        sub = np.nonzero(BUCK[idx] == w)[0]
        for i in range(0, len(sub), A.batch):
            b = sub[i:i + A.batch]
            pr = eval_step(params, statics, *dev(idx[b], int(w)), int(w))
            out[b] = np.asarray(pr)
    return out


def r2_report(pred, idx):
    """R^2 per target: raw, and WITHIN-STEP (per (traj, qstep) group centred)."""
    tz = TGTZ[idx]; lm = LEG[idx]
    rep = {}
    for c, nm in enumerate(NAMES):
        e = {}
        for qi, q in list(enumerate(QSTEPS)) + [(None, "ALL")]:
            if qi is None:                      # pooled over every query step
                m = lm
                y = tz[:, :, :, c][m]; p = pred[:, :, :, c][m]
                r2 = (float("nan") if y.var() < 1e-12 else
                      1.0 - ((y - p) ** 2).mean() / y.var())
                yy, pp = [], []
                for j in range(len(idx)):
                    for qj in range(NQ):
                        mm = lm[j, qj]
                        if mm.sum() < 2:
                            continue
                        a = tz[j, qj, :, c][mm]; b = pred[j, qj, :, c][mm]
                        yy.append(a - a.mean()); pp.append(b - b.mean())
                yy = np.concatenate(yy); pp = np.concatenate(pp)
                r2w = (float("nan") if yy.var() < 1e-12 else
                       1.0 - ((yy - pp) ** 2).mean() / yy.var())
                e["tALL"] = dict(r2=float(r2), r2w=float(r2w),
                                 var=float(y.var()))
                continue
            m = lm[:, qi]
            y = tz[:, qi, :, c][m]; p = pred[:, qi, :, c][m]
            if y.var() < 1e-12:
                e[f"t{q}"] = dict(r2=float("nan"), r2w=float("nan"),
                                  var=float(y.var()))
                continue
            r2 = 1.0 - ((y - p) ** 2).mean() / y.var()
            # within-step: centre both inside each trajectory's step group
            yy, pp = [], []
            for j in range(len(idx)):
                mm = lm[j, qi]
                if mm.sum() < 2:
                    continue
                a = tz[j, qi, :, c][mm]; b = pred[j, qi, :, c][mm]
                yy.append(a - a.mean()); pp.append(b - b.mean())
            yy = np.concatenate(yy); pp = np.concatenate(pp)
            r2w = (float("nan") if yy.var() < 1e-12 else
                   1.0 - ((yy - pp) ** 2).mean() / yy.var())
            e[f"t{q}"] = dict(r2=float(r2), r2w=float(r2w), var=float(y.var()))
        rep[nm] = e
    return rep


# ---------------- non-learned baselines (fit on train, scored on test) -----
def const_baselines():
    ytr = TGTZ[tr_idx]; ltr = LEG[tr_idx]
    pv = np.zeros((NV, NT)); cv = np.zeros((NV, 1))
    pvt = np.zeros((NQ, NV, NT)); cvt = np.zeros((NQ, NV, 1))
    for qi in range(NQ):
        m = ltr[:, qi]                                   # (n, NV)
        cnt = m.sum(0)[:, None]
        s = (ytr[:, qi] * m[..., None]).sum(0)
        pvt[qi] = s / np.maximum(cnt, 1); cvt[qi] = cnt
        pv += s; cv += cnt
    pv = pv / np.maximum(cv, 1)
    b1 = np.broadcast_to(pv[None, None], (len(te_idx), NQ, NV, NT)).copy()
    b2 = np.broadcast_to(pvt[None], (len(te_idx), NQ, NV, NT)).copy()
    return b1, b2


b_v, b_vt = const_baselines()
base_rep = {"const_vertex": r2_report(b_v, te_idx),
            "const_vertex_step": r2_report(b_vt, te_idx)}
print("\n=== BASELINES (test) : R2 / R2_within ===", flush=True)
for nm, rep in base_rep.items():
    for t, e in rep.items():
        print(f"  {nm:18s} {t:10s} " + "  ".join(
            f"{k}:{v['r2']:6.3f}/{v['r2w']:6.3f}" for k, v in e.items()),
            flush=True)

curve = []
t0 = time.time()
tr_by_b = {int(w): tr_idx[BUCK[tr_idx] == w] for w in np.unique(BUCK[tr_idx])}
ws = np.asarray(list(tr_by_b)); pw = np.asarray([len(tr_by_b[int(w)]) for w in ws],
                                                np.float64)
pw = pw / pw.sum()
for step in range(1, A.steps + 1):
    w = int(rng.choice(ws, p=pw))
    pool = tr_by_b[w]
    b = rng.choice(pool, size=min(A.batch, len(pool)), replace=len(pool) < A.batch)
    params, ostate, v = train_step(params, ostate, statics, *dev(b, w),
                                   jnp.asarray(TGTZ[b]), jnp.asarray(LEG[b]), w)
    if step % 25 == 0:
        print(f"step {step:5d} loss {float(v):.4f} wall {time.time()-t0:.0f}s",
              flush=True)
    if step % A.eval_every == 0 or step == A.steps:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_idx[:len(te_idx)])
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_idx[:len(te_idx)])
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0))
        print(f"--- eval @ {step} (test R2/R2within) ---", flush=True)
        keys = [f"t{q}" for q in QSTEPS] + ["tALL"]
        for nm in NAMES:
            print(f"  {nm:10s} " + "  ".join(
                f"{kk}:{rte[nm][kk]['r2']:6.3f}/{rte[nm][kk]['r2w']:6.3f}"
                for kk in keys), flush=True)
        print("  TRAINFIT  " + "  ".join(
            f"fill {kk}:{rtr['fill'][kk]['r2w']:6.3f}" for kk in keys)
            + "   elim_nb tALL:%.3f" % rtr['elim_nb']['tALL']['r2w'],
            flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), baselines=base_rep, curve=curve), f)
print("done", time.time() - t0, flush=True)
