"""decode5_face_train: the DECIDING 2x2x2 FACTORIAL for the face representation.

decode4 settled the READOUT question with a 4-way 5-seed pool A/B: all four
pools FAIL every live-contraction target and the choice barely matters
(attn ~= mean > sumtok > last).  So the readout was not the bottleneck.  What
is left are the two factors that can actually add INFORMATION rather than
re-read it:

  --pool {attn,mean}   the two surviving pools from decode4 (tied).
  --mp {0,1}           MESSAGE PASSING AFTER POOLING.  Faces of the same
                       eliminated vertex, and their endpoint vertices,
                       exchange information before the readout.  Reuses POMO's
                       `EdgeCondLayer` / `layer_rows` BY IMPORT from
                       alphagrad.elimrl.encoder -- separate in/out (fwd/bwd)
                       aggregation, which is what a DAG needs.
                       Graph: nodes = the (NSTEP, V+1) per-step vertex
                       contexts, flattened (block-diagonal by step, so no
                       information crosses an elimination step).  Each face
                       (i, v, j) contributes TWO directed edges  i -> v  and
                       v -> j , both carrying that face's own edge feature, so
                       the ELIMINATED VERTEX v is a genuine hub every face of
                       the step touches: any two faces of a step are 2 hops
                       apart and 3 layers mix the step completely.  The
                       endpoint ids are already on device (commit 9253614
                       carries them out of the chunk callback for the
                       [ctx_i || ctx_j || face_latent] gather), so MP adds NO
                       host traffic -- it re-uses f_si/f_sj/f_v, the very
                       arrays that gather already uses.
  --extents {0,1}      the per-face log2 axis extents (`f_ext`, exactly
                       `LiveVertexMaskOracle.face_features`, common/masks.py
                       :619).  ALREADY LOG2 at build time and z-scored here
                       against TRAIN-only statistics, i.e. SCALE-FREE: raw
                       extents once saturated every downstream logit to
                       p_skip ~ 1e-16 (an absorbing state), so raw is never
                       used.  probe_faces already builds the live SparseTensor
                       for the mask, so this costs NO extra elimination.

METRIC: WITHIN-STEP R2 (faces of ONE elimination step of ONE trajectory).
Anything per-vertex is constant inside a step group and contributes exactly 0,
so within-step isolates the per-face channel.  TRAIN is printed beside TEST at
every eval: the face arms show train 0.57-0.73 against ~0 test, i.e.
OVERFITTING, and a separate study once read 0.41 and called it "not learnable"
purely from UNDER-training.  One curve cannot tell those apart.

POSITIVE CONTROLS: stat_ln_i / stat_ln_j (log2 numel of the two endpoint
variables, purely static) must reach ~1.0 or the harness is broken.
DEGENERATE: n_compressed / is_lowrank are identically 0 on this graph.

DAG-AGNOSTICISM: --audit prints every trainable leaf with its shape and FAILS
if any is dimensioned by V (vertex count), T (stream length) or F (face
count).  --audit also runs the ALT-SHAPE re-audit: the whole model is rebuilt
at V'=137, NSTEP'=53, F'=201, T'=99991 and the leaf shapes are compared
elementwise.  That is strictly stronger than "the flags vanish on another
graph": identical shapes at two different V PROVE no leaf is V-dimensioned,
and it disambiguates decode4's two known coincidences on this graph
(op vocabulary 71 == NSTEP 71; 3*embd_dim 96 == V 96).

ORIGINAL decode4 docstring follows.
--------------------------------------------------------------------------
decode4_face_train: WHICH READOUT recovers face-level information?

The face representation is fixed at the live path's `new` arm
([ctx_i || ctx_j || face_latent], 3E, NO explicit input of any kind).  The
ONLY thing that varies is how the (T_chunk, E) palimpsa rows of a face's own
token chunk collapse to the (E,) `face_latent`:

  --pool mean     TODAY (`Agent._face_pool`).  Unweighted mean over the
                  chunk's rows.  THE CONTROL.  Palimpsa rows are already a
                  gated accumulation (M_t = v_t (x) k_t + decay * M_{t-1}), so
                  averaging them re-accumulates with UNIFORM weights and
                  throws away the gating the encoder learned.
  --pool last     The chunk's LAST row.  Free, and causal palimpsa means that
                  row has seen the whole chunk -- but its QUERY is whatever
                  the final real token's q happens to be, not one learned for
                  summarising.  A boundary read failed badly on the VERTEX
                  side (within-step R2 -0.217, train 0.765: fits, transfers
                  nothing), in a different setup.
  --pool sumtok   An appended learned SUMMARY TOKEN.  One extra token is
                  inserted into the stream at each chunk's end; its row is the
                  face latent.  The encoder produces the summary ITSELF via
                  mu_t . q_t, where q_t comes from that token's (learned)
                  embedding -- ONE new vocabulary row, E params.  The stream
                  is rebuilt once, in numpy, at load; every offset is remapped
                  so nothing else changes.
  --pool attn     A learned attention pool over the chunk's rows: one query,
                  k/v/out projections, softmax within the chunk.  Structurally
                  identical to `VertexIdentityPool`, the treatment that took
                  the VERTEX side from 1300 steps to 7.

METRIC: WITHIN-STEP R2 (faces of ONE elimination step of ONE trajectory).
Anything per-vertex is constant inside a step group and contributes exactly 0,
so within-step isolates the per-face channel.  TRAIN is printed beside TEST at
every eval: the face arms show train 0.57-0.73 against ~0 test, i.e.
OVERFITTING, and a separate study once read 0.41 and called it "not learnable"
purely from UNDER-training.  One curve cannot tell those apart.

POSITIVE CONTROLS: stat_ln_i / stat_ln_j (log2 numel of the two endpoint
variables, purely static) must reach ~1.0 or the harness is broken.
DEGENERATE: n_compressed / is_lowrank are identically 0 on this graph.

DAG-AGNOSTICISM: --audit prints every trainable leaf with its shape and FAILS
if any is dimensioned by V (vertex count), T (stream length) or F (face
count).  Only the COMPILATION SHAPE may vary per DAG.
"""
import argparse, json, os, time
import numpy as np
import decode3_arch as ARCH
import jax, jax.numpy as jnp, jax.random as jrand
import equinox as eqx, optax

P = argparse.ArgumentParser()
P.add_argument("--data", required=True)
P.add_argument("--out", required=True)
P.add_argument("--pool", default="mean",
               choices=["mean", "last", "sumtok", "attn"])
P.add_argument("--mp", type=int, default=0, choices=[0, 1],
               help="message passing AFTER pooling (EdgeCondLayer x --mp-layers)")
P.add_argument("--mp-layers", type=int, default=3)
P.add_argument("--mp-width", type=int, default=64)
P.add_argument("--edge-dim", type=int, default=16)
P.add_argument("--extents", type=int, default=0, choices=[0, 1],
               help="feed the per-face log2 axis extents (f_ext, z-scored on "
                    "TRAIN only) to the readout AND to the MP edge feature")
P.add_argument("--embd-dim", type=int, default=32)
P.add_argument("--num-layers", type=int, default=3)
P.add_argument("--num-heads", type=int, default=2)
P.add_argument("--pointer-blocks", type=int, default=2)
P.add_argument("--head-width", type=int, default=256)
P.add_argument("--steps", type=int, default=3000)
P.add_argument("--batch", type=int, default=4)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--n-test", type=int, default=32)
P.add_argument("--chunk-cap", type=int, default=4096)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--audit", action="store_true",
               help="print the parameter-shape audit and exit")
P.add_argument("--trace-only", action="store_true")
P.add_argument("--load-params", default=None,
               help="DAG-TRANSFER: deserialise weights trained on ANOTHER "
                    "graph into a model built for THIS one. It only works if "
                    "no leaf is dimensioned by V/T/F -- a shape mismatch here "
                    "IS the negative result.")
P.add_argument("--norm", default=None,
               help="use the SOURCE run's target mu/sd, so transfer is "
                    "measured in the units the readout was trained in")
P.add_argument("--eval-only", action="store_true",
               help="no training: one within-step R2 report over EVERY "
                    "trajectory of --data")
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

f_traj, f_q = d["f_traj"], d["f_q"]
f_start, f_split = d["f_start"], d["f_split"]
f_si, f_sj, f_v = d["f_si"], d["f_sj"], d["f_v"]
f_tgt, f_ext = d["f_tgt"], d["f_ext"]
print(f"data N={N} L={L} NV={NV} nsteps={NSTEP} faces={len(f_traj)} "
      f"arm={'B' if int(d['shapes']) else 'A'} pool={A.pool}", flush=True)

# ---------------------------------------------------------------- sumtok
# The SUMMARY TOKEN is a real token in a real stream: one extra id, inserted
# at each chunk's end, whose row IS the face latent.  np.insert's `obj` is
# indexed against the ORIGINAL array and inserts BEFORE each position, in
# order for ties -- exactly the semantics needed, so the remap is closed form:
#   old token p  -> p + #{sentinels with split <= p}
#   sentinel r (0-based in stable split order) -> split_r + r
SUMTOK = 512          # max real token id on this tokenizer is 511 (measured)
VOCAB = 256
f_sum = np.zeros(len(f_traj), np.int32)      # unused unless pool == sumtok
if A.pool == "sumtok":
    VOCAB = 513
    by_traj = [[] for _ in range(N)]
    for k in range(len(f_traj)):
        by_traj[int(f_traj[k])].append(k)
    add = np.asarray([len(x) for x in by_traj], np.int64)
    L2 = int((NTOK.astype(np.int64) + add).max())
    T2 = np.zeros((N, L2), np.int32)
    O2 = np.full((N, L2), -1, np.int32)
    Q2 = np.full((N, L2), -1, np.int32)
    D2 = np.full((N, L2), -1, np.int16)
    NT2 = np.zeros(N, np.int32)
    ns_ = np.array(f_start, np.int64).copy()
    for i in range(N):
        ks = by_traj[i]
        n = int(NTOK[i])
        if not ks:
            T2[i, :n] = TOK[i, :n]; O2[i, :n] = OWN[i, :n]
            Q2[i, :n] = EQN[i, :n]; D2[i, :n] = DID[i, :n]
            NT2[i] = n
            continue
        sp = np.asarray([f_split[k] for k in ks], np.int64)
        order = np.argsort(sp, kind="stable")
        ssp = sp[order]
        # the sentinel belongs to the delta the chunk ends inside
        dsrc = DID[i, np.maximum(ssp - 1, 0)]
        t2 = np.insert(TOK[i, :n], ssp, SUMTOK)
        o2 = np.insert(OWN[i, :n], ssp, -1)
        q2 = np.insert(EQN[i, :n], ssp, -1)
        d2 = np.insert(DID[i, :n], ssp, dsrc)
        m = len(t2)
        T2[i, :m] = t2; O2[i, :m] = o2; Q2[i, :m] = q2; D2[i, :m] = d2
        NT2[i] = m
        rank = np.empty(len(ks), np.int64)
        rank[order] = np.arange(len(ks))
        for a, k in enumerate(ks):
            f_sum[k] = ssp[rank[a]] + rank[a]
            ns_[k] = f_start[k] + int(np.searchsorted(ssp, f_start[k],
                                                      side="right"))
        # sanity: the sentinel must sit exactly where the chunk ends
        assert (t2[f_sum[ks]] == SUMTOK).all(), i
    TOK, OWN, EQN, DID, NTOK = T2, O2, Q2, D2, NT2
    f_start = ns_.astype(np.int32)
    N, L = TOK.shape
    print(f"sumtok: stream L {int(d['tok'].shape[1])} -> {L}, "
          f"+{int(add.sum())} summary tokens, vocab {VOCAB}", flush=True)

cnt = np.bincount(f_traj, minlength=N)
FMAX = int(cnt.max())
print(f"faces/traj min={cnt.min()} max={FMAX} mean={cnt.mean():.1f}", flush=True)
FQ = np.zeros((N, FMAX), np.int32); FS = np.zeros((N, FMAX), np.int32)
FP = np.zeros((N, FMAX), np.int32); FI = np.full((N, FMAX), NV, np.int32)
FJ = np.full((N, FMAX), NV, np.int32); FV = np.zeros((N, FMAX), np.int32)
FU = np.zeros((N, FMAX), np.int32)
FT = np.zeros((N, FMAX, NFT), np.float32); FE = np.zeros((N, FMAX, NDIM), np.float32)
FM = np.zeros((N, FMAX), bool)
_w = np.zeros(N, np.int32)
for k in range(len(f_traj)):
    i = int(f_traj[k]); j = int(_w[i]); _w[i] += 1
    FQ[i, j] = f_q[k]; FS[i, j] = f_start[k]
    FP[i, j] = f_sum[k] if A.pool == "sumtok" else f_split[k]
    FU[i, j] = f_sum[k]
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
print("DEGENERATE (identically constant on train):", DEGEN, flush=True)
if A.norm:
    _n = np.load(A.norm, allow_pickle=True)
    MU, SD = _n["mu"], _n["sd"]
    print("using SOURCE normalisation from", A.norm, flush=True)
FTZ = ((FT - MU) / SD).astype(np.float32)
EXT_MU = FE[tr_idx].reshape(-1, NDIM)[sel].mean(0)
EXT_SD = np.maximum(FE[tr_idx].reshape(-1, NDIM)[sel].std(0), 1e-6)
FEZ = ((FE - EXT_MU) / EXT_SD).astype(np.float32)

STEP_W = 16384
BUCK = np.minimum(((NTOK + STEP_W - 1) // STEP_W) * STEP_W, L)
print("buckets", {int(w): int((BUCK == w).sum()) for w in np.unique(BUCK)},
      flush=True)

k = jrand.split(jrand.PRNGKey(A.seed), 4)
agent = ARCH.build(E, A.num_layers, A.num_heads, A.pointer_blocks, NV, k[0],
                   vocab=VOCAB)


# ------------------------------------------------------------------ pools
class FaceAttnPool(eqx.Module):
    """Learned attention pool over ONE face's chunk rows.

    Structurally `VertexIdentityPool` restricted to a single segment: one
    learned query, k/v/out projections, softmax over the chunk's live rows.
    Every parameter is (E, E) or (E,) -- nothing is dimensioned by the number
    of vertices, faces, or tokens, so the same weights apply to any DAG.
    """
    q: jax.Array
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim, *, key):
        ks = jrand.split(key, 4)
        self.embd_dim = embd_dim
        self.q = jrand.normal(ks[0], (embd_dim,)) * (embd_dim ** -0.5)
        self.k_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[1])
        self.v_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[2])
        self.out_proj = eqx.nn.Linear(embd_dim, embd_dim, key=ks[3])

    def __call__(self, rows_c, m):
        live = m > 0
        kk = jax.vmap(self.k_proj)(rows_c)
        vv = jax.vmap(self.v_proj)(rows_c)
        sc = (kk @ self.q) / jnp.sqrt(jnp.asarray(self.embd_dim, rows_c.dtype))
        # -inf never reaches exp: dead rows are zeroed arithmetically.
        mx = jnp.max(jnp.where(live, sc, -jnp.inf))
        mx = jnp.where(jnp.isfinite(mx), mx, 0.0)
        e = jnp.where(live, jnp.exp(sc - mx), 0.0)
        z = jnp.sum(e)
        wgt = e / jnp.maximum(z, 1e-9)
        pooled = jnp.sum(vv * wgt[:, None], axis=0)
        return jnp.where(z > 0, self.out_proj(pooled), 0.0)


pool = FaceAttnPool(E, key=k[2]) if A.pool == "attn" else None


# ------------------------------------------------------- message passing
# POMO's edge-conditioned layer, BY IMPORT.  Its composition pattern there is
# embed_rows -> 3 x EdgeCondLayer -> vertex_embed/face_embed; here the
# "embed_rows" step is the palimpsa encoder + pointer contexts that already
# produce ctx_all, and the face_embed step is the existing
# [ctx_i || ctx_j || face_latent] concat, so only the middle is new.
from alphagrad.elimrl.encoder import EdgeCondLayer, layer_rows   # noqa: E402

EXT_DIM = NDIM if A.extents else 0


class FaceMP(eqx.Module):
    """Directional message passing over the per-step face incidence graph.

    Every parameter is dimensioned by (embd_dim, edge_dim, width) only.
    Nothing is dimensioned by V, T, F or NSTEP: the node COUNT changes per
    DAG, the node WIDTH does not, which is exactly what a segment_sum GNN
    needs and what --audit checks.
    """
    e_proj: eqx.nn.Linear
    layers: tuple

    def __init__(self, embd_dim, in_dim, edge_dim, width, n_layers, *, key):
        ks = jrand.split(key, n_layers + 1)
        self.e_proj = eqx.nn.Linear(in_dim, edge_dim, key=ks[0])
        self.layers = tuple(
            EdgeCondLayer(embd_dim, edge_dim, width, ks[1 + i])
            for i in range(n_layers))

    def __call__(self, ctx_all, lat, extz, fq, fi, fj, fv, fm):
        """``ctx_all`` (NSTEP, NV+1, E) -> refined (NSTEP, NV+1, E).

        Faces are edges; the eliminated vertex is the hub.  Blocks are keyed
        by step, so nothing leaks across an elimination step.
        """
        S, R, Ed = ctx_all.shape
        n = S * R
        h = ctx_all.reshape(n, Ed)
        ein = lat if extz is None else jnp.concatenate([lat, extz], -1)
        ef = jax.vmap(self.e_proj)(ein)                      # (F, edge_dim)
        base = fq.astype(jnp.int32) * R
        gi = base + jnp.clip(fi, 0, R - 1)
        gj = base + jnp.clip(fj, 0, R - 1)
        gv = base + jnp.clip(fv, 0, R - 1)
        src = jnp.concatenate([gi, gv])
        dst = jnp.concatenate([gv, gj])
        e = jnp.concatenate([ef, ef], 0)
        # padded faces carry weight 0: their messages are multiplied to 0
        # BEFORE the segment sum, and adding 0.0 is exact in fp.
        em = jnp.concatenate([fm, fm]).astype(jnp.float32)
        rows = jnp.arange(n, dtype=jnp.int32)
        for layer in self.layers:
            h = layer_rows(layer, h, e, src, dst, em, rows)
        return h.reshape(S, R, Ed)


mp = (FaceMP(E, E + EXT_DIM, A.edge_dim, A.mp_width, A.mp_layers, key=k[3])
      if A.mp else None)
print(f"ARM pool={A.pool} mp={A.mp} extents={A.extents} "
      f"(EXT_DIM={EXT_DIM}, mp_layers={A.mp_layers if A.mp else 0})",
      flush=True)


def face_latent(pool, rows, w, s_, p_, u_):
    if A.pool == "mean":
        return ARCH.face_latent(rows, w, s_, p_, CAP)
    if A.pool == "last":
        i_ = jnp.clip(p_ - 1, 0, rows.shape[0] - 1)
        ok = (p_ > s_) & (w[i_] > 0)
        return jnp.where(ok, rows[i_], 0.0)
    if A.pool == "sumtok":
        i_ = jnp.clip(u_, 0, rows.shape[0] - 1)
        return jnp.where(w[i_] > 0, rows[i_], 0.0)
    ar = jnp.arange(CAP, dtype=jnp.int32)
    idx = jnp.minimum(s_ + ar, rows.shape[0] - 1)
    m = ((s_ + ar) < p_).astype(jnp.float32) * w[idx]
    return pool(rows[idx], m)


VF = jnp.asarray(VFEAT)
DIN = 3 * E + EXT_DIM


class Readout(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, din, width, key):
        self.mlp = eqx.nn.MLP(din, NFT, width, depth=2, key=key)

    def __call__(self, h):
        return jax.vmap(self.mlp)(h)


head = Readout(DIN, A.head_width, k[1])


# --------------------------------------------------------- DAG-agnosticism
def audit(model, label):
    """FAIL if any trainable leaf is dimensioned by V, T or F."""
    bad = {"V(vertices)": NV, "V+1": NV + 1, "V+2": NV + 2,
           "NSTEP": NSTEP, "F(faces/traj)": FMAX, "T(stream)": L,
           "CAP(chunk)": CAP, "N(traj)": N}
    leaves = jax.tree_util.tree_leaves_with_path(
        eqx.filter(model, eqx.is_inexact_array))
    print(f"\n=== PARAMETER-SHAPE AUDIT [{label}] ===", flush=True)
    tot, fails = 0, []
    for path, v in leaves:
        nm = jax.tree_util.keystr(path)
        hit = [k_ for k_, n_ in bad.items() if n_ in v.shape]
        tot += int(np.prod(v.shape))
        if hit:
            fails.append((nm, v.shape, hit))
        print(f"  {nm:70s} {str(v.shape):16s} "
              f"{'*** ' + ','.join(hit) if hit else ''}", flush=True)
    print(f"  total trainable scalars: {tot}", flush=True)
    if fails:
        print("  NOT DAG-AGNOSTIC:", fails, flush=True)
    else:
        print("  DAG-AGNOSTIC: no parameter is dimensioned by V, T, F or "
              "NSTEP; the only per-DAG quantity is the COMPILATION SHAPE.",
              flush=True)
    return fails


_fails = audit((agent, head, pool, mp),
               f"pool={A.pool} mp={A.mp} extents={A.extents}")
if A.audit:
    # ALT-SHAPE RE-AUDIT.  Rebuild the WHOLE model against a fabricated graph
    # with a different vertex count / step count / face count / stream length
    # and compare leaf shapes elementwise.  Identical shapes at two different
    # V is a machine-checked proof that no leaf is V/T/F/NSTEP-dimensioned,
    # and it kills decode4's two documented coincidences on this graph
    # (op vocabulary 71 == NSTEP 71, and 3*embd_dim 96 == V 96).
    from alphagrad.approx.common import OP_TYPE_VOCAB_SIZE
    print(f"\nOP_TYPE_VOCAB_SIZE = {OP_TYPE_VOCAB_SIZE}  <- a hardcoded jaxpr "
          f"PRIMITIVE-NAME table in common/instrumentation.py, not a property "
          f"of any DAG.  It equals NSTEP({NSTEP}) here by coincidence.",
          flush=True)
    NV2, NSTEP2 = 137, 53
    k2 = jrand.split(jrand.PRNGKey(A.seed + 991), 4)
    agent2 = ARCH.build(E, A.num_layers, A.num_heads, A.pointer_blocks, NV2,
                        k2[0], vocab=VOCAB)
    head2 = Readout(DIN, A.head_width, k2[1])
    pool2 = FaceAttnPool(E, key=k2[2]) if A.pool == "attn" else None
    mp2 = (FaceMP(E, E + EXT_DIM, A.edge_dim, A.mp_width, A.mp_layers,
                  key=k2[3]) if A.mp else None)
    lv = lambda m: [(jax.tree_util.keystr(p), v.shape) for p, v in
                    jax.tree_util.tree_leaves_with_path(
                        eqx.filter(m, eqx.is_inexact_array))]
    a1, a2 = lv((agent, head, pool, mp)), lv((agent2, head2, pool2, mp2))
    print(f"\n=== ALT-SHAPE RE-AUDIT  V {NV}->{NV2}, NSTEP {NSTEP}->{NSTEP2},"
          f" F {FMAX}->201, T {L}->99991 ===", flush=True)
    diff = [(n1, s1, s2) for (n1, s1), (_, s2) in zip(a1, a2) if s1 != s2]
    if len(a1) != len(a2):
        print(f"  LEAF COUNT CHANGED {len(a1)} -> {len(a2)}: NOT DAG-AGNOSTIC",
              flush=True)
    elif diff:
        print(f"  {len(diff)} LEAF SHAPES CHANGED WITH V => NOT DAG-AGNOSTIC",
              flush=True)
        for n1, s1, s2 in diff:
            print(f"    {n1:70s} {s1} -> {s2}", flush=True)
    else:
        print(f"  ALL {len(a1)} LEAF SHAPES IDENTICAL AT V={NV} AND V={NV2}."
              f"  DAG-AGNOSTIC: no trainable parameter is dimensioned by the "
              f"graph.  The only per-DAG quantity is the COMPILATION SHAPE.",
              flush=True)
    raise SystemExit(0)


def per_traj(agent, head, pool, mp, tok, eqn, did, own, ntok, part, fq, fs, fp,
             fu, fi, fj, fv, fe, fm, W):
    rows, w = ARCH.encode(agent, tok, eqn, ntok, W)
    ident = ARCH.identity(agent, rows, w, own, did, NV)
    tab = ARCH.memory_tables(rows, w, own, eqn, did, part, NV, NSTEP)

    def _ctx(t):
        S, C = ARCH.mem_at(tab, t, NV, E, NSTEP)
        c, _ = ARCH.heads(agent, S, C, ident)
        return c
    ctx_all = jax.vmap(_ctx)(jnp.arange(NSTEP, dtype=jnp.int32))
    ctx_all = jnp.concatenate(
        [ctx_all, jnp.zeros((NSTEP, 1, E), jnp.float32)], 1)

    # POOL FIRST -- message passing is strictly AFTER pooling, so the two
    # factors stay separable.
    lat = jax.vmap(lambda s_, p_, u_: face_latent(pool, rows, w, s_, p_, u_))(
        fs, fp, fu)
    extz = fe if A.extents else None
    ctx_u = (mp(ctx_all, lat, extz, fq, fi, fj, fv, fm) if mp is not None
             else ctx_all)

    def one(q, i_, j_, lt, ex):
        parts = [ctx_u[q, i_], ctx_u[q, j_], lt]
        if A.extents:
            parts.append(ex)
        return jnp.concatenate(parts)

    return head(jax.vmap(one)(fq, fi, fj, lat, fe))


def batched(agent, head, pool, mp, *a):
    # exactly one in_axes entry per positional argument (4 statics, 15 mapped
    # per-trajectory arrays, then the static window W).
    return jax.vmap(per_traj,
                    in_axes=(None,) * 4 + (0,) * 15 + (None,))(
        agent, head, pool, mp, *a)


def loss_fn(params, statics, *a):
    # a = the 15 per-trajectory arrays produced by dev(), then tz, then W.
    tz, W = a[-2], a[-1]
    fm = a[-3]
    agent, head, pool, mp = eqx.combine(params, statics)
    pr = batched(agent, head, pool, mp, *a[:-2], W)
    m = fm[..., None].astype(jnp.float32)
    return jnp.sum(((pr - tz) ** 2) * m) / jnp.maximum(jnp.sum(m), 1.0)


params, statics = eqx.partition((agent, head, pool, mp), eqx.is_inexact_array)
if A.load_params:
    # Shape-checked by construction: tree_deserialise_leaves refuses a leaf
    # whose shape differs, so a clean load is machine-checked evidence that
    # every parameter is V/T/F-independent.
    params = eqx.tree_deserialise_leaves(A.load_params, params)
    print("loaded params from", A.load_params, flush=True)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(A.lr))
ostate = opt.init(params)


@eqx.filter_jit
def train_step(params, ostate, statics, *a):
    v, g = jax.value_and_grad(loss_fn)(params, statics, *a)
    u, ostate = opt.update(g, ostate, params)
    return eqx.apply_updates(params, u), ostate, v


@eqx.filter_jit
def eval_step(params, statics, *a):
    agent, head, pool, mp = eqx.combine(params, statics)
    return batched(agent, head, pool, mp, *a)


def dev(idx, W):
    return (jnp.asarray(TOK[idx, :W]), jnp.asarray(EQN[idx, :W]),
            jnp.asarray(DID[idx, :W].astype(np.int32)),
            jnp.asarray(OWN[idx, :W]), jnp.asarray(NTOK[idx]),
            jnp.asarray(PART[idx].astype(np.float32)),
            jnp.asarray(FQ[idx]), jnp.asarray(FS[idx]), jnp.asarray(FP[idx]),
            jnp.asarray(FU[idx]), jnp.asarray(FI[idx]), jnp.asarray(FJ[idx]),
            jnp.asarray(FV[idx]), jnp.asarray(FEZ[idx]),
            jnp.asarray(FM[idx]))


if A.trace_only:
    _b = tr_idx[:A.batch]
    _w = int(BUCK[_b].max())
    _o = jax.eval_shape(
        lambda p, *a: jax.value_and_grad(loss_fn)(p, statics, *a, _w),
        params, *dev(_b, _w), jnp.asarray(FTZ[_b]))
    print(f"TRACE OK pool={A.pool} mp={A.mp} extents={A.extents} "
          f"window={_w} loss={_o[0]}", flush=True)
    raise SystemExit(0)


def collect(params, idx):
    out = np.zeros((len(idx), FMAX, NFT), np.float32)
    for wv in np.unique(BUCK[idx]):
        sub = np.nonzero(BUCK[idx] == wv)[0]
        for i in range(0, len(sub), A.batch):
            b = sub[i:i + A.batch]
            out[b] = np.asarray(eval_step(params, statics,
                                          *dev(idx[b], int(wv)), int(wv)))
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


if A.eval_only:
    all_idx = np.arange(N)
    pr = collect(params, all_idx)
    rep = r2_report(pr, all_idx)
    print(f"\n=== TRANSFER EVAL pool={A.pool} data={A.data} ===", flush=True)
    for nm in FN:
        tag = "  DEGENERATE" if nm in DEGEN else ""
        print(f"  {nm:13s} r2 {rep[nm]['r2']:7.3f}  within "
              f"{rep[nm]['r2w']:7.3f}{tag}", flush=True)
    json.dump(dict(args=vars(A), transfer=rep, degenerate=DEGEN),
              open(A.out, "w"))
    raise SystemExit(0)

base_rep = {"const_global": r2_report(
    np.zeros((len(te_idx), FMAX, NFT), np.float32), te_idx)}

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
    pl = tr_by_b[wv]
    b = rng.choice(pl, size=min(A.batch, len(pl)), replace=len(pl) < A.batch)
    params, ostate, v = train_step(params, ostate, statics, *dev(b, wv),
                                   jnp.asarray(FTZ[b]), wv)
    if step in EVALS:
        pte = collect(params, te_idx)
        ptr = collect(params, tr_eval)
        rte = r2_report(pte, te_idx); rtr = r2_report(ptr, tr_eval)
        curve.append(dict(step=step, loss=float(v), test=rte, train=rtr,
                          wall=time.time() - t0, degenerate=DEGEN,
                          pool=A.pool, mp=A.mp, extents=A.extents,
                          seed=A.seed))
        print(f"--- eval @ {step}  pool {A.pool} mp {A.mp} ext {A.extents} "
              f"seed {A.seed}  "
              f"loss {float(v):.4f}  wall {time.time()-t0:.0f}s ---", flush=True)
        for nm in FN:
            tag = "  DEGENERATE" if nm in DEGEN else ""
            print(f"  {nm:13s} TEST r2 {rte[nm]['r2']:7.3f} within "
                  f"{rte[nm]['r2w']:7.3f}   TRAIN within "
                  f"{rtr[nm]['r2w']:7.3f}{tag}", flush=True)
        with open(A.out, "w") as f:
            json.dump(dict(args=vars(A), baselines=base_rep, curve=curve,
                           dag_agnostic_fails=[list(map(str, x))
                                               for x in _fails]), f)

# The trained weights, for the DAG-TRANSFER test (decode4_transfer.py): the
# audit above proves no leaf is dimensioned by V/T/F, so the SAME file must
# deserialise against a model built for a DIFFERENT graph. If it does not,
# that is the DAG-agnosticism verdict, not a bug.
eqx.tree_serialise_leaves(A.out + ".eqx", params)
np.savez(A.out + ".norm.npz", mu=MU, sd=SD, pool=np.asarray(A.pool),
         embd_dim=np.asarray(E), nft=np.asarray(NFT))
print("saved params ->", A.out + ".eqx", flush=True)
print("done", time.time() - t0, flush=True)
