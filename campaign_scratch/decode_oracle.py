"""decode_oracle: the INFORMATION CEILING.

The dynamic targets are a deterministic function of (vertex, ELIMINATED SET),
and the eliminated set is unambiguously present in the token stream (every
delta block is owned by the vertex whose elimination emitted it). This fits a
small MLP on exactly that pair -- no palimpsa, no vmem -- on the SAME
trajectories, split and metric. Whatever it reaches is what a representation
that preserved the elimination history could reach.
"""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand, equinox as eqx, optax

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import build_static

DS = sys.argv[1]
OUT = sys.argv[2]
d = np.load(DS, allow_pickle=True)
TGT, LEG, MODE, VFEAT = d["tgt"], d["legal"], d["mode"], d["vfeat"]
QSTEPS = [int(x) for x in d["qsteps"]]; NAMES = [str(x) for x in d["names"]]
N, NQ, NV, NT = TGT.shape
print("N", N, "NQ", NQ, "NV", NV, "NT", NT, flush=True)

fn, args_, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
static = build_static(env)
MAXSTEP = max(QSTEPS)

# replay the SAME trajectories (decode_data.gen uses seed 1000+i, mode by parity)
ELIM = np.zeros((N, NQ, NV), np.float32)
t0 = time.time()
for i in range(N):
    rng = np.random.default_rng(1000 + i)
    mode = "rev" if i % 2 == 0 else "rand"
    env.reset()
    for t in range(MAXSTEP + 1):
        st = env.state(); legal = st.legal_vertices
        if not legal:
            break
        if t in QSTEPS:
            qi = QSTEPS.index(t)
            for j in st.eliminated:
                ELIM[i, qi, j - 1] = 1.0
        if t == MAXSTEP:
            break
        v = (max(legal) if (mode == "rev" and rng.random() > 0.15)
             else int(rng.choice(legal)))
        env.step(("V", int(v)))
print(f"replayed {N} trajectories in {time.time()-t0:.0f}s "
      f"(mean |S| at t70 = {ELIM[:, -1].sum(1).mean():.1f})", flush=True)

rng = np.random.default_rng(0)
te_idx = []
for m in (0, 1):
    ids = np.nonzero(MODE == m)[0]; rng.shuffle(ids)
    te_idx.append(ids[: 32 // 2])
te_idx = np.sort(np.concatenate(te_idx)); tr_idx = np.setdiff1d(np.arange(N), te_idx)

sel = LEG[tr_idx].reshape(-1); flat = TGT[tr_idx].reshape(-1, NT)[sel]
MU = flat.mean(0); SD = np.maximum(flat.std(0), 1e-6)
TGTZ = ((TGT - MU) / SD).astype(np.float32)

EYE = np.eye(NV, dtype=np.float32)
SF = VFEAT.astype(np.float32)
SF = (SF - SF.mean(0)) / np.maximum(SF.std(0), 1e-6)


def build(idx):
    X, Y, M = [], [], []
    for j in idx:
        for qi in range(NQ):
            e = ELIM[j, qi]
            X.append(np.concatenate(
                [EYE, np.broadcast_to(e, (NV, NV)), SF], 1))
            Y.append(TGTZ[j, qi]); M.append(LEG[j, qi])
    return (np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.float32), np.concatenate(M))


Xtr, Ytr, Mtr = build(tr_idx)
Xte, Yte, Mte = build(te_idx)
print("Xtr", Xtr.shape, "kept", Mtr.sum(), flush=True)
Xtr, Ytr = Xtr[Mtr], Ytr[Mtr]
Xte_f, Yte_f = Xte[Mte], Yte[Mte]

key = jrand.PRNGKey(0)
mlp = eqx.nn.MLP(Xtr.shape[1], NT, 512, depth=3, key=key)
opt = optax.adam(1e-3)
params, statics = eqx.partition(mlp, eqx.is_inexact_array)
ost = opt.init(params)


@eqx.filter_jit
def step(params, ost, xb, yb):
    def L(p):
        m = eqx.combine(p, statics)
        return jnp.mean((jax.vmap(m)(xb) - yb) ** 2)
    v, g = jax.value_and_grad(L)(params)
    u, ost = opt.update(g, ost, params)
    return eqx.apply_updates(params, u), ost, v


@eqx.filter_jit
def pred(params, x):
    m = eqx.combine(params, statics)
    return jax.vmap(m)(x)


rng2 = np.random.default_rng(1)
t0 = time.time()
for it in range(1, 6001):
    b = rng2.choice(len(Xtr), 2048, replace=False)
    params, ost, v = step(params, ost, jnp.asarray(Xtr[b]), jnp.asarray(Ytr[b]))
    if it % 1000 == 0:
        print(f"  it {it} loss {float(v):.4f} wall {time.time()-t0:.0f}s", flush=True)

P = np.zeros((len(Xte_f), NT), np.float32)
for i in range(0, len(Xte_f), 8192):
    P[i:i + 8192] = np.asarray(pred(params, jnp.asarray(Xte_f[i:i + 8192])))

# scatter back so the within-step grouping is available
full = np.zeros_like(Yte); full[Mte] = P
FP = full.reshape(len(te_idx), NQ, NV, NT)
res = {}
print("\n=== INFORMATION CEILING: MLP on (vertex id, eliminated set, static) ===")
print("    test R2 / R2_within")
for c, nm in enumerate(NAMES):
    row = {}
    for qi, q in list(enumerate(QSTEPS)) + [(None, "ALL")]:
        if qi is None:
            m = LEG[te_idx]
            y = TGTZ[te_idx][..., c][m]; p = FP[..., c][m]
            gi = [(j, qj) for j in range(len(te_idx)) for qj in range(NQ)]
        else:
            m = LEG[te_idx][:, qi]
            y = TGTZ[te_idx][:, qi, :, c][m]; p = FP[:, qi, :, c][m]
            gi = [(j, qi) for j in range(len(te_idx))]
        r2 = (float("nan") if y.var() < 1e-12
              else 1.0 - ((y - p) ** 2).mean() / y.var())
        yy, pp = [], []
        for (j, qj) in gi:
            mm = LEG[te_idx][j, qj]
            if mm.sum() < 2:
                continue
            a = TGTZ[te_idx][j, qj, :, c][mm]; b = FP[j, qj, :, c][mm]
            yy.append(a - a.mean()); pp.append(b - b.mean())
        yy = np.concatenate(yy); pp = np.concatenate(pp)
        r2w = (float("nan") if yy.var() < 1e-12
               else 1.0 - ((yy - pp) ** 2).mean() / yy.var())
        row["t%s" % q] = dict(r2=float(r2), r2w=float(r2w))
    res[nm] = row
    print(f"  {nm:10s} " + "  ".join(
        f"{k}:{v['r2']:6.3f}/{v['r2w']:6.3f}" for k, v in row.items()), flush=True)
json.dump(res, open(OUT, "w"))
