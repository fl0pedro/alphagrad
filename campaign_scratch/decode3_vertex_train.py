"""decode3_vertex_train: can the VERTEX level be read off the tokenized
append-only jaxpr ALONE, under the 2026-08-14 architecture?

Every arm trains the SAME palimpsa encoder end to end with the SAME read-out
MLP, the same optimiser and the same batches.  Only how a per-vertex vector is
assembled differs.

  --arm new        THE LIVE PATH.  identity = the learned attention pool over
                   the vertex's own BASE-token span; dynamic = the
                   participation-credited delta memory; the two CONCATENATED,
                   scored by the SetPointer over V+2 slots, contexts projected
                   back to E by ctx_proj.  Read out from vertex_contexts --
                   what the face head and the value path actually receive.
                   NO explicit feature of any kind.
  --arm new_slots  the same representation read out BEFORE the pointer mixes
                   it: [identity || dynamic || log1p(count)].  Separates "the
                   information is in the slots" from "the pointer preserves
                   it".
  --arm anchor     the old split+participation bar: a STATIC MEAN identity
                   (not the pool) + the same dynamic + log1p(count) + the
                   hand-written vertex features.  This is the 0.930 / 10-30-310
                   number the previous architecture reached, re-measured here
                   so the comparison is on one dataset and one seed.
  --arm no_ident   `new` with the identity half ZEROED -- what the dynamic
                   channel alone carries.
  --arm no_dyn     `new` with the dynamic half ZEROED -- what identity alone
                   carries.  Must fail on DV_FILL at t>0 or the target is not
                   dynamic.

METRIC.  STEPS-TO-THRESHOLD on WITHIN-STEP R2 (0.6 / 0.8 / 0.9), with the
train curve printed beside the test curve at every eval.  Anything constant
inside a step group contributes exactly 0 to a within-step R2, so it is the
only reading that isolates the per-vertex channel.
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
               choices=["new", "new_slots", "anchor", "no_ident", "no_dyn"])
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=8)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--trace-only", action="store_true",
               help="abstract-trace forward AND backward, then exit -- catches every shape/vmap bug in seconds instead of paying a 20-minute XLA compile per arm")
A = P.parse_args()

d = np.load(A.data, allow_pickle=True)
TOK, OWN, EQN, DID = d["tok"], d["own"], d["eqn"], d["did"]
NTOK, PREF, TGT, LEG = d["ntok"], d["pref"], d["tgt"], d["legal"]
MODE, VFEAT, PART = d["mode"], d["vfeat"], d["part"]
QSTEPS = [int(x) for x in d["qsteps"]]
NAMES = [str(x) for x in d["names"]]
N, L = TOK.shape
NV = VFEAT.shape[0]
NQ, NT = len(QSTEPS), len(NAMES)
NSTEP = PART.shape[1]
E = A.embd_dim
KV = VFEAT.shape[1]
print(f"data N={N} L={L} NV={NV} q={QSTEPS} nstep={NSTEP} "
      f"arm={'B' if int(d['shapes']) else 'A'}", flush=True)

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

k = jrand.split(jrand.PRNGKey(A.seed), 4)
agent = ARCH.build(E, A.num_layers, A.num_heads, A.pointer_blocks, NV, k[0])
VF = jnp.asarray(VFEAT)
QIDX = jnp.asarray(np.asarray(QSTEPS, np.int32))

DIN = {"new": E, "new_slots": 2 * E + 1, "anchor": 2 * E + 1 + KV,
       "no_ident": E, "no_dyn": E}[A.arm]
print(f"arm {A.arm} din={DIN}", flush=True)


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


head = Readout(DIN, A.head_width, k[1])


def per_traj(agent, head, tok, eqn, did, own, ntok, part, W):
    rows, w = ARCH.encode(agent, tok, eqn, ntok, W)
    ident = ARCH.identity(agent, rows, w, own, did, NV)
    if A.arm == "no_ident":
        ident = jnp.zeros_like(ident)
    tab = ARCH.memory_tables(rows, w, own, eqn, did, part, NV, NSTEP)

    if A.arm == "anchor":
        # OLD static identity: the plain MEAN of the vertex's base rows.
        is_base = (did < 0).astype(jnp.float32)
        oid = jnp.where(own < 0, NV, jnp.minimum(own, NV - 1)).astype(jnp.int32)
        bw = w * is_base
        bs = jax.ops.segment_sum(rows * bw[:, None], oid, num_segments=NV + 1)
        bc = jax.ops.segment_sum(bw, oid, num_segments=NV + 1)
        key_static = bs[:NV] / jnp.maximum(bc[:NV, None], 1.0)

    def per_q(qi):
        t = QIDX[qi]
        S, C = ARCH.mem_at(tab, t, NV, E, NSTEP)
        if A.arm == "no_dyn":
            S = jnp.zeros_like(S)
            C = jnp.zeros_like(C)
        if A.arm == "new" or A.arm == "no_ident" or A.arm == "no_dyn":
            ctx, _ = ARCH.heads(agent, S, C, ident)
            return head(ctx)
        vrows = S / jnp.maximum(C, 1.0)[:, None]
        if A.arm == "new_slots":
            h = jnp.concatenate(
                [ident[:NV], vrows[:NV], jnp.log1p(C[:NV])[:, None]], -1)
        else:
            h = jnp.concatenate(
                [key_static, vrows[:NV], jnp.log1p(C[:NV])[:, None], VF], -1)
        return head(h)

    return jax.vmap(per_q)(jnp.arange(NQ))


def batched(agent, head, *a):
    return jax.vmap(per_traj, in_axes=(None, None) + (0,) * 6 + (None,))(
        agent, head, *a)


def loss_fn(params, statics, tok, eqn, did, own, ntok, part, tz, lm, W):
    agent, head = eqx.combine(params, statics)
    pr = batched(agent, head, tok, eqn, did, own, ntok, part, W)
    m = lm[..., None].astype(jnp.float32)
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
            jnp.asarray(PART[idx].astype(np.float32)))


if A.trace_only:
    # Abstract trace of forward AND backward: catches every shape /
    # vmap / concat bug in seconds, without paying the ~20 min XLA
    # compile of a 16k-step palimpsa scan that a real step needs.
    _b = tr_idx[:A.batch]
    _w = int(BUCK[_b].max())
    _o = jax.eval_shape(
        lambda p, *a: jax.value_and_grad(loss_fn)(p, statics, *a, _w),
        params, *dev(_b, _w), jnp.asarray(TGTZ[_b]), jnp.asarray(LEG[_b]))
    print(f"TRACE OK arm={A.arm} window={_w} loss={_o[0]}", flush=True)
    raise SystemExit(0)


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


# LOG-SPACED evals: steps-to-threshold needs resolution at 10 steps and at
# 3000, and a fixed stride cannot give both.
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
                                   jnp.asarray(TGTZ[b]), jnp.asarray(LEG[b]), wv)
    if step in EVALS:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_eval)
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_eval)
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0))
        keys = [f"t{q}" for q in QSTEPS] + ["tALL"]
        print(f"--- eval @ {step}  loss {float(v):.4f}  "
              f"wall {time.time()-t0:.0f}s  (test R2/R2within) ---", flush=True)
        for nm in NAMES:
            print(f"  {nm:10s} " + "  ".join(
                f"{kk}:{rte[nm][kk]['r2']:6.3f}/{rte[nm][kk]['r2w']:6.3f}"
                for kk in keys), flush=True)
        print(f"  TRAIN  fill tALL w:{rtr['fill']['tALL']['r2w']:6.3f}   "
              f"TEST fill tALL w:{rte['fill']['tALL']['r2w']:6.3f}   "
              f"| controls TEST static_ln w:{rte['static_ln']['tALL']['r2w']:6.3f} "
              f"elim_nb w:{rte['elim_nb']['tALL']['r2w']:6.3f}", flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), curve=curve), f)
print("done", time.time() - t0, flush=True)
