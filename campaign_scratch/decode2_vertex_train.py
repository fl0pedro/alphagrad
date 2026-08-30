"""decode2_vertex_train: PART B -- vertex-side encodings judged by
STEPS-TO-THRESHOLD, not by final R2.

Every variant trains the SAME palimpsa encoder end-to-end with the SAME
read-out MLP, the same optimiser and the same batches; only the way a
per-vertex vector is assembled from the encoder's rows differs.

  --variant fold        TODAY. agent.heads_from_memory over the mean-pooled
                        owner fold -- the 0.932 baseline (decode_train --model
                        slot).
  --variant boundary    the causal row at the END of the vertex's equation
                        span in the BASE encode (a compile-time-constant
                        gather) + the causal latent at the query prefix.
  --variant split_part  a STATIC identity key (mean of the vertex's base rows,
                        never overwritten) + a dynamic local state whose delta
                        rows are attributed by PARTICIPATION (every vertex a
                        step's rows touch) instead of by AUTHORSHIP
                        (carry_stream.py:100, why an un-eliminated candidate's
                        slot never moves).
  --variant split_auth  the control for the above: identical split, but the
                        deltas attributed by AUTHORSHIP.  Isolates
                        participation from the split.
  --variant mix_nobias  own 1-block self-attention mixer over the slots
                        (replaces the set pointer), unordered set, no
                        adjacency.
  --variant mix_adj     the same mixer with an ADJACENCY attention bias
                        (a1*A + a2*A^2 + a3*I, three learned scalars).
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
P.add_argument("--variant", default="fold",
               choices=["fold", "boundary", "split_part", "split_auth",
                        "mix_nobias", "mix_adj"])
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=8)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--eval-every", type=int, default=100)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--seed", type=int, default=0)
A = P.parse_args()

d = np.load(A.data, allow_pickle=True)
TOK, OWN, EQN, DID = d["tok"], d["own"], d["eqn"], d["did"]
NTOK, PREF, TGT, LEG = d["ntok"], d["pref"], d["tgt"], d["legal"]
MODE, VFEAT = d["mode"], d["vfeat"]
PART, BND = d["part"], d["bnd"]
QSTEPS = [int(x) for x in d["qsteps"]]
NAMES = [str(x) for x in d["names"]]
N, L = TOK.shape
NV = VFEAT.shape[0]
NQ, NT = len(QSTEPS), len(NAMES)
NSTEP = PART.shape[1]
KPART = PART.shape[2]
E = A.embd_dim
KV = VFEAT.shape[1]
print(f"data N={N} L={L} NV={NV} q={QSTEPS} nstep={NSTEP} kpart={KPART}",
      flush=True)

# ---- adjacency of the ORIGINAL jaxpr (static; variant mix_adj) -----------
ADJ = np.zeros((NV, NV), np.float32)
if A.variant == "mix_adj":
    from alphagrad.elimrl.baselines import tlm_target
    _fn, _a, _an = tlm_target(seq=32, dmodel=128, vocab=1024)
    _jx = jax.make_jaxpr(_fn)(*_a).jaxpr
    _of = {}
    for pos, e_ in enumerate(_jx.eqns, start=1):
        for ov in e_.outvars:
            _of[id(ov)] = pos
    for pos, e_ in enumerate(_jx.eqns, start=1):
        for iv in e_.invars:
            u = _of.get(id(iv))
            if u is not None and u != pos:
                ADJ[pos - 1, u - 1] = 1.0; ADJ[u - 1, pos - 1] = 1.0
    print(f"adjacency density {ADJ.mean():.4f}", flush=True)
ADJ2 = np.minimum(ADJ @ ADJ, 1.0) if A.variant == "mix_adj" else ADJ
AJ = jnp.asarray(ADJ); AJ2 = jnp.asarray(ADJ2)
EYE = jnp.eye(NV, dtype=jnp.float32)

rng = np.random.default_rng(A.seed)
te = []
for m in (0, 1):
    ids = np.nonzero(MODE == m)[0]; rng.shuffle(ids)
    te.append(ids[: A.n_test // 2])
te_idx = np.sort(np.concatenate(te))
tr_idx = np.setdiff1d(np.arange(N), te_idx)
print(f"train {len(tr_idx)} test {len(te_idx)}", flush=True)

sel = LEG[tr_idx].reshape(-1)
flat = TGT[tr_idx].reshape(-1, NT)[sel]
MU = flat.mean(0); SD = np.maximum(flat.std(0), 1e-6)
TGTZ = ((TGT - MU) / SD).astype(np.float32)

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
k = jrand.split(jrand.PRNGKey(A.seed), 8)
agent = _build_agent(args, NV, 1, 1, k[0])
VF = jnp.asarray(VFEAT)
# per-step query index: memory at query q covers deltas < QSTEPS[q]
QIDX = jnp.asarray(np.asarray(QSTEPS, np.int32))


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


class Mixer(eqx.Module):
    """One self-attention block over the NV slots, optionally with an
    adjacency-derived attention bias (3 scalars, no new pathway)."""
    inp: eqx.nn.Linear
    q: eqx.nn.Linear
    kk: eqx.nn.Linear
    v: eqx.nn.Linear
    o: eqx.nn.Linear
    b: jnp.ndarray
    use_adj: bool = eqx.field(static=True)

    def __init__(self, din, e, use_adj, key):
        ks = jrand.split(key, 5)
        self.inp = eqx.nn.Linear(din, e, key=ks[0])
        self.q = eqx.nn.Linear(e, e, key=ks[1])
        self.kk = eqx.nn.Linear(e, e, key=ks[2])
        self.v = eqx.nn.Linear(e, e, key=ks[3])
        self.o = eqx.nn.Linear(e, e, key=ks[4])
        self.b = jnp.zeros(3)
        self.use_adj = use_adj

    def __call__(self, x):
        h = jax.vmap(self.inp)(x)
        Q = jax.vmap(self.q)(h); K = jax.vmap(self.kk)(h); V = jax.vmap(self.v)(h)
        s = (Q @ K.T) / jnp.sqrt(jnp.float32(h.shape[-1]))
        if self.use_adj:
            s = s + self.b[0] * AJ + self.b[1] * AJ2 + self.b[2] * EYE
        a = jnn.softmax(s, axis=-1)
        return jnp.concatenate([h, jax.vmap(self.o)(a @ V)], -1)


DIN = {"fold": E,
       "boundary": 2 * E + KV,
       "split_part": 2 * E + 1 + KV,
       "split_auth": 2 * E + 1 + KV,
       "mix_nobias": 2 * E,
       "mix_adj": 2 * E}[A.variant]
MIX_IN = 2 * E + 1 + KV
mixer = (Mixer(MIX_IN, E, A.variant == "mix_adj", k[2])
         if A.variant in ("mix_nobias", "mix_adj") else None)
head = Readout(DIN, A.head_width, k[1])
print(f"variant {A.variant} din={DIN}", flush=True)


def per_traj(agent, head, mixer, tok, eqn, did, own, ntok, pref, bnd, part, W):
    c0 = agent.carry_init()
    _, rows, valid, _ = agent.encode_extend(c0, tok, eqn, ntok, window=W,
                                            start=0, chunk=0)
    w = valid.astype(jnp.float32)
    ar = jnp.arange(W, dtype=jnp.int32)
    oid = jnp.where(own < 0, NV, jnp.minimum(own, NV - 1)).astype(jnp.int32)
    is_base = did < 0
    d_id = jnp.where(is_base, 0, did).astype(jnp.int32)

    if A.variant == "fold":
        def per_q(p):
            ww = w * (ar < p)
            s = jax.ops.segment_sum(rows * ww[:, None], oid, num_segments=NV + 1)
            c = jax.ops.segment_sum(ww, oid, num_segments=NV + 1)
            _, ctx, _ = agent.heads_from_memory(s, c, vertex_features=VF)
            return head(ctx[:NV])
        return jax.vmap(per_q)(pref)

    # ---- pieces shared by every other variant ----------------------------
    bw = w * is_base
    base_s = jax.ops.segment_sum(rows * bw[:, None], oid, num_segments=NV + 1)
    base_c = jax.ops.segment_sum(bw, oid, num_segments=NV + 1)
    key_static = base_s[:NV] / jnp.maximum(base_c[:NV, None], 1.0)
    dw = w * (~is_base)
    dsum = jax.ops.segment_sum(rows * dw[:, None], d_id, num_segments=NSTEP)
    dcnt = jax.ops.segment_sum(dw, d_id, num_segments=NSTEP)
    downer = jnp.clip(jax.ops.segment_max(jnp.where(is_base, -1, oid), d_id,
                                          num_segments=NSTEP), 0, NV)

    if A.variant == "boundary":
        bidx = jnp.clip(bnd, 0, W - 1)
        hb = rows[bidx] * (bnd >= 0)[:, None]

        def per_q(p):
            lat = rows[jnp.clip(p - 1, 0, W - 1)]
            h = jnp.concatenate(
                [hb, jnp.broadcast_to(lat, (NV, E)), VF], -1)
            return head(h)
        return jax.vmap(per_q)(pref)

    # participation / authorship scatter, exclusive cumsum over steps
    if A.variant == "split_part" or A.variant in ("mix_nobias", "mix_adj"):
        oh = jnp.zeros((NSTEP, NV + 1), jnp.float32)
        for kk_ in range(KPART):
            idx = jnp.where(part[:, kk_] < 0, NV, part[:, kk_]).astype(jnp.int32)
            oh = oh + jax.nn.one_hot(idx, NV + 1) * (part[:, kk_] >= 0)[:, None]
    else:
        oh = jax.nn.one_hot(downer, NV + 1)
    cs = jnp.cumsum(oh[:, :, None] * dsum[:, None, :], axis=0)
    cc = jnp.cumsum(oh * dcnt[:, None], axis=0)

    if A.variant in ("split_part", "split_auth"):
        def per_q(qi):
            t = QIDX[qi]
            s_ = jnp.where(t > 0, cs[jnp.maximum(t - 1, 0)], 0.0)[:NV]
            c_ = jnp.where(t > 0, cc[jnp.maximum(t - 1, 0)], 0.0)[:NV]
            dyn = s_ / jnp.maximum(c_[:, None], 1.0)
            h = jnp.concatenate([key_static, dyn,
                                 jnp.log1p(c_)[:, None], VF], -1)
            return head(h)
        return jax.vmap(per_q)(jnp.arange(pref.shape[0]))

    # mixers
    def per_q(qi):
        t = QIDX[qi]
        s_ = jnp.where(t > 0, cs[jnp.maximum(t - 1, 0)], 0.0)[:NV]
        c_ = jnp.where(t > 0, cc[jnp.maximum(t - 1, 0)], 0.0)[:NV]
        dyn = s_ / jnp.maximum(c_[:, None], 1.0)
        x = jnp.concatenate([key_static, dyn, jnp.log1p(c_)[:, None], VF], -1)
        return head(mixer(x))
    return jax.vmap(per_q)(jnp.arange(pref.shape[0]))


def batched(agent, head, mixer, *a):
    return jax.vmap(per_traj, in_axes=(None, None, None) + (0,) * 8 + (None,))(
        agent, head, mixer, *a)


def loss_fn(params, statics, tok, eqn, did, own, ntok, pref, bnd, part, tz, lm, W):
    agent, head, mixer = eqx.combine(params, statics)
    pr = batched(agent, head, mixer, tok, eqn, did, own, ntok, pref, bnd, part, W)
    m = lm[..., None].astype(jnp.float32)
    return jnp.sum(((pr - tz) ** 2) * m) / jnp.maximum(jnp.sum(m), 1.0)


params, statics = eqx.partition((agent, head, mixer), eqx.is_inexact_array)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(A.lr))
ostate = opt.init(params)


@eqx.filter_jit
def train_step(params, ostate, statics, *a):
    v, g = jax.value_and_grad(loss_fn)(params, statics, *a)
    u, ostate = opt.update(g, ostate, params)
    return eqx.apply_updates(params, u), ostate, v


@eqx.filter_jit
def eval_step(params, statics, *a):
    agent, head, mixer = eqx.combine(params, statics)
    return batched(agent, head, mixer, *a)


def dev(idx, W):
    return (jnp.asarray(TOK[idx, :W]), jnp.asarray(EQN[idx, :W]),
            jnp.asarray(DID[idx, :W].astype(np.int32)),
            jnp.asarray(OWN[idx, :W]), jnp.asarray(NTOK[idx]),
            jnp.asarray(PREF[idx]), jnp.asarray(BND[idx]),
            jnp.asarray(PART[idx].astype(np.int32)))


def collect(params, idx):
    out = np.zeros((len(idx), NQ, NV, NT), np.float32)
    for wv in np.unique(BUCK[idx]):
        sub = np.nonzero(BUCK[idx] == wv)[0]
        for i in range(0, len(sub), A.batch):
            b = sub[i:i + A.batch]
            out[b] = np.asarray(eval_step(params, statics, *dev(idx[b], int(wv)),
                                          int(wv)))
    return out


def r2_report(pred, idx):
    tz = TGTZ[idx]; lm = LEG[idx]
    rep = {}
    for c, nm in enumerate(NAMES):
        e = {}
        for qi, q in list(enumerate(QSTEPS)) + [(None, "ALL")]:
            if qi is None:
                m = lm; y = tz[:, :, :, c][m]; p = pred[:, :, :, c][m]
                rng_q = range(NQ)
            else:
                m = lm[:, qi]; y = tz[:, qi, :, c][m]; p = pred[:, qi, :, c][m]
                rng_q = [qi]
            r2 = (float("nan") if y.var() < 1e-12 else
                  1.0 - ((y - p) ** 2).mean() / y.var())
            yy, pp = [], []
            for j in range(len(idx)):
                for qj in rng_q:
                    mm = lm[j, qj]
                    if mm.sum() < 2:
                        continue
                    a = tz[j, qj, :, c][mm]; b = pred[j, qj, :, c][mm]
                    yy.append(a - a.mean()); pp.append(b - b.mean())
            if yy:
                yy = np.concatenate(yy); pp = np.concatenate(pp)
                r2w = (float("nan") if yy.var() < 1e-12 else
                       1.0 - ((yy - pp) ** 2).mean() / yy.var())
            else:
                r2w = float("nan")
            e[f"t{q}"] = dict(r2=float(r2), r2w=float(r2w))
        rep[nm] = e
    return rep


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
                                   jnp.asarray(TGTZ[b]), jnp.asarray(LEG[b]), wv)
    if step % 25 == 0:
        print(f"step {step:5d} loss {float(v):.4f} wall {time.time()-t0:.0f}s",
              flush=True)
    if step % A.eval_every == 0 or step == A.steps:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_idx[:len(te_idx)])
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_idx[:len(te_idx)])
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0))
        keys = [f"t{q}" for q in QSTEPS] + ["tALL"]
        print(f"--- eval @ {step} (test R2/R2within) ---", flush=True)
        for nm in NAMES:
            print(f"  {nm:10s} " + "  ".join(
                f"{kk}:{rte[nm][kk]['r2']:6.3f}/{rte[nm][kk]['r2w']:6.3f}"
                for kk in keys), flush=True)
        print(f"  TRAINFIT fill tALL:{rtr['fill']['tALL']['r2w']:6.3f}  "
              f"static_ln tALL:{rte['static_ln']['tALL']['r2w']:6.3f}  "
              f"elim_nb tALL:{rte['elim_nb']['tALL']['r2w']:6.3f}", flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), curve=curve), f)
print("done", time.time() - t0, flush=True)
