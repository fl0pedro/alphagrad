"""decode2_face_data: per-FACE decodability dataset (+ vertex-side extras).

Extends decode_data.py; SAME target graph, SAME seeds, SAME env/tokenizer
loop, so the vertex-side numbers stay comparable with decode_ds_192.npz.

NEW, per query step, for the vertex that is about to be eliminated:
  * the per-FACE token chunk [start, split) of graphax's ``last_face_segments``
    -- exactly the span ``live_faces.chunk`` hands the live per-face head
    (in the EXACT arm a face's chunk is its own header + contraction, because
    the preceding face's approximation tail is empty).
  * the two ENDPOINT vertex slots of the face key ``(vidx[in_edge],
    vidx[out_edge])`` -- the gather indices arm (b) needs.
  * TARGETS read off the LIVE SparseTensor of that face
    (``LiveVertexMaskOracle.probe_faces`` / ``face_masks``): block extents,
    rule legality + largest legal block factor, true block structure, a
    stored-vs-dense cost gain, and two PURELY STATIC positive controls
    (log2 numel of the two endpoint vars).

ALSO (Part B):
  * BND   last BASE-stream token index owned by each vertex slot (variant 2)
  * DIDX  per-token delta index (-1 = base)
  * PART  per-delta PARTICIPATION table: every vertex slot the step's rows
          touch (v plus the endpoint slots of all of v's faces), not just the
          author (variant 3)

Usage: decode2_face_data.py OUT.npz N_TRAJ
"""
import os, sys, time, math, json
import numpy as np
import jax

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import (
    build_static, extract, DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL,
    DV_MAX_JAC, S_OUT_LN)
from alphagrad.approx.common.instrumentation import compute_vertex_features
from alphagrad.approx.common import masks as MK
from graphax import IncrementalPathTokenizer
from graphax.core import _vidx_for
from graphax.sparse import lattice as LAT

OUT = sys.argv[1]
NTRAJ = int(sys.argv[2]) if len(sys.argv) > 2 else 64
QSTEPS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                           else ["0", "20", "50", "70"])]
MAXSTEP = max(QSTEPS)
# FACE query steps are separate: faces are only 1-3 per step early on, so the
# face probe runs at EVERY step while the (much bigger) per-vertex target
# tensor stays on the four checkpoints Part B is comparable on.
FSTEPS = list(range(MAXSTEP + 1)) if os.environ.get("D2_FACE_EVERY", "1") == "1" \
    else list(QSTEPS)
VOCAB = 512
MAXF = 64                    # face slots kept per query step
KPART = 24                   # participation fan-out cap per delta
NDIM = 12                    # extent slots kept per face

TGT_COLS = [DV_FILL, DV_OUT_DEG, DV_IN_DEG, DV_MARKOWITZ, DV_MAX_JAC]
TGT_NAMES = ["fill", "out_deg", "in_deg", "markowitz", "max_jac", "static_ln",
             "elim_nb", "n_elim"]

FT_NAMES = [
    "ln_out",        # log2 prod(logical out_dims)      -- block extent
    "ln_prim",       # log2 prod(logical primal_dims)   -- block extent
    "ln_maxdim",     # log2 max logical dim
    "ln_stored",     # log2 stored elements of the live block
    "gain",          # log2(dense / stored): sparsity already present
    "ln_factor",     # log2 largest LEGAL block factor (the gcd rule)
    "n_diag",        # number of legal DIAG pairs (rule legality)
    "n_comp",        # number of legal COMPRESS axes
    "n_paired",      # dims with other_id  -- true block-diagonal structure
    "n_compressed",  # compressed dims
    "is_lowrank",    # classify_edge(t) == LOWRANK
    "n_dims",        # total dims of the block
    "stat_ln_i",     # STATIC CONTROL: log2 numel of endpoint var i
    "stat_ln_j",     # STATIC CONTROL: log2 numel of endpoint var j
]
NFT = len(FT_NAMES)

T0 = time.time()
fn, args_, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
static = build_static(env)
closed = jax.make_jaxpr(fn)(*args_)
jaxpr, consts = closed.jaxpr, closed.literals
NV = len(jaxpr.eqns)
print(f"[{time.time()-T0:.0f}s] n_eqns={NV} n_rows={static.n_rows} "
      f"jacve={len(env.jacve_vertices)}", flush=True)

vfeat = compute_vertex_features(jaxpr, tuple(consts), tuple(args_),
                                eval_samples=None, argnums=tuple(argnums))
slot_row = np.asarray([static.row_of(s + 1) for s in range(NV)], np.int32)
static_ln = static.feat[slot_row, S_OUT_LN].astype(np.float32)

# ---- var index -> vertex slot / static log2 numel ------------------------
VIDX = _vidx_for(jaxpr)
NVIDX = (max(VIDX.values()) + 1) if VIDX else 1
slot_of_vidx = np.full(NVIDX + 4, -1, np.int32)
ln_of_vidx = np.zeros(NVIDX + 4, np.float32)
for _v, _i in VIDX.items():
    av = getattr(_v, "aval", None)
    n = 1
    if av is not None and getattr(av, "shape", None) is not None:
        for s in av.shape:
            n *= int(s)
    ln_of_vidx[_i] = math.log2(max(n, 1))
for pos, eqn_ in enumerate(jaxpr.eqns, start=1):
    for ov in eqn_.outvars:
        j = VIDX.get(ov)
        if j is not None:
            slot_of_vidx[j] = pos - 1

# ---- original-graph neighbours (dynamic vertex control, as decode_data) ---
_vertex_of = {}
for pos, eqn_ in enumerate(jaxpr.eqns, start=1):
    for ov in eqn_.outvars:
        _vertex_of[id(ov)] = pos
NB = [set() for _ in range(NV + 1)]
for pos, eqn_ in enumerate(jaxpr.eqns, start=1):
    for iv in eqn_.invars:
        u = _vertex_of.get(id(iv))
        if u is not None and u != pos:
            NB[pos].add(u); NB[u].add(pos)


def _dims_of(st):
    return list(getattr(st, "out_dims", ()) or ()), \
           list(getattr(st, "primal_dims", ()) or ())


def face_targets(st, pair_k, comp_k, gnom_k, ki, kj):
    """The supervised target vector for ONE face's live block."""
    t = np.zeros(NFT, np.float32)
    od, pd = _dims_of(st)
    dims = od + pd
    lo = 1
    for d in od:
        lo *= max(int(getattr(d, "logical_size", 1) or 1), 1)
    lp = 1
    for d in pd:
        lp *= max(int(getattr(d, "logical_size", 1) or 1), 1)
    mx = 1
    for d in dims:
        mx = max(mx, int(getattr(d, "logical_size", 1) or 1))
    val = getattr(st, "val", None)
    shp = tuple(getattr(val, "shape", ()) or ())
    stored = 1
    for s in shp:
        stored *= int(s)
    n_paired = sum(1 for d in dims if getattr(d, "other_id", None) is not None)
    n_comp_d = sum(1 for d in dims if getattr(d, "is_compressed", False))
    try:
        lr = 1.0 if LAT.classify_edge(st) is LAT.L.LOWRANK else 0.0
    except Exception:
        lr = 0.0
    t[0] = math.log2(max(lo, 1))
    t[1] = math.log2(max(lp, 1))
    t[2] = math.log2(max(mx, 1))
    t[3] = math.log2(max(stored, 1))
    t[4] = math.log2(max(lo * lp, 1)) - math.log2(max(stored, 1))
    t[5] = math.log2(max(gnom_k, 1))
    t[6] = float(pair_k.sum() * 0.5)
    t[7] = float(comp_k.sum())
    t[8] = float(n_paired)
    t[9] = float(n_comp_d)
    t[10] = lr
    t[11] = float(len(dims))
    t[12] = ln_of_vidx[ki] if 0 <= ki < len(ln_of_vidx) else 0.0
    t[13] = ln_of_vidx[kj] if 0 <= kj < len(ln_of_vidx) else 0.0
    ext = np.zeros(NDIM, np.float32)
    for a, d in enumerate(dims[:NDIM]):
        ext[a] = math.log2(max(int(getattr(d, "logical_size", 1) or 1), 1))
    return t, ext


def best_gnom(vertex, pair_k):
    """Largest legal nominal block factor for this face = max gcd over the
    pairs face_masks admitted (the ``gcd(N_i,N_j)`` rule at ppo.py:2924)."""
    eqn = jaxpr.eqns[vertex - 1]
    out_shape = tuple(eqn.outvars[0].aval.shape)
    out_len = len(out_shape)
    primal_shapes = [tuple(iv.aval.shape) for iv in eqn.invars
                     if hasattr(iv, "aval")]
    if not primal_shapes:
        return 1
    n_primal = min(len(ps) for ps in primal_shapes)
    best = 1
    for bi1 in range(out_len):
        n1 = int(out_shape[bi1])
        for bi2 in range(n_primal):
            i, j = MK_ROW2PAIR(vertex, bi1, bi2)
            if i == j or i >= 8 or j >= 8 or not pair_k[i, j]:
                continue
            g = n1
            for ps in primal_shapes:
                g = math.gcd(g, int(ps[bi2]))
            best = max(best, g)
    return best


from alphagrad.approx.env import diag_row_to_pair as _d2p


def MK_ROW2PAIR(vertex, bi1, bi2):
    return _d2p(jaxpr, vertex, bi1, bi2)


FSET = set()   # filled after FSTEPS is known
STATS = dict(steps=0, probe_empty=0, len_mismatch=0, key_unmapped=0,
             faces=0, overflow=0)


def gen(seed, mode):
    rng = np.random.default_rng(seed)
    env.reset()
    tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts),
                                  list(args_), vocab_size=VOCAB)
    oracle = MK.LiveVertexMaskOracle(jaxpr, list(consts), list(args_),
                                     tuple(argnums), max_axes=8)
    base = np.asarray([int(x) for x in tk.base_tokens()], np.int32)
    bown = np.asarray([int(x) for x in tk.last_owner_ids()], np.int32)
    beqn = np.asarray([int(x) for x in tk.last_eqn_ids()], np.int32)
    bslot = np.where(bown > 0, bown - 1, -1).astype(np.int32)

    toks = [base]; owns = [bslot]; eqns = [beqn]
    didx = [np.full(len(base), -1, np.int16)]
    cum = len(base)
    prefix_len = {}
    tgt = np.zeros((len(QSTEPS), NV, len(TGT_NAMES)), np.float32)
    legal_m = np.zeros((len(QSTEPS), NV), bool)
    order = []
    part = np.full((MAXSTEP + 1, KPART), -1, np.int16)
    faces = []                       # per emitted face at a query step

    for t in range(MAXSTEP + 1):
        st_env = env.state()
        legal = st_env.legal_vertices
        if not legal:
            break
        qi = QSTEPS.index(t) if t in QSTEPS else None
        if qi is not None:
            prefix_len[qi] = cum
            sf = extract(st_env, static)
            vd = sf.vert_dyn[slot_row]
            for c, col in enumerate(TGT_COLS):
                tgt[qi, :, c] = vd[:, col]
            tgt[qi, :, len(TGT_COLS)] = static_ln
            gone = set(st_env.eliminated)
            tgt[qi, :, len(TGT_COLS) + 1] = np.asarray(
                [len(NB[s + 1] & gone) for s in range(NV)], np.float32)
            tgt[qi, :, len(TGT_COLS) + 2] = float(len(gone))
            for j in legal:
                legal_m[qi, j - 1] = True

        if mode == "rev":
            v = max(legal) if rng.random() > 0.15 else int(rng.choice(legal))
        else:
            v = int(rng.choice(legal))

        # ---- participation: v plus the endpoint slots of all of v's faces --
        try:
            keys_all = list(tk.ij.faces(int(v)))
        except Exception:
            keys_all = []
        pset = {v - 1}
        for (ki, kj) in keys_all:
            for kk in (ki, kj):
                if kk is None:
                    continue
                s_ = int(slot_of_vidx[kk]) if 0 <= kk < len(slot_of_vidx) else -1
                if s_ >= 0:
                    pset.add(s_)
        pl = sorted(pset)
        if len(pl) > KPART:
            STATS["overflow"] += 1
            pl = pl[:KPART]
        part[t, :len(pl)] = np.asarray(pl, np.int16)

        probes = None
        pair = comp = None
        fq = t if t in FSET else None
        if fq is not None:
            STATS["steps"] += 1
            try:
                probes = oracle.probe_faces(int(v), approx=True)
            except Exception:
                probes = []
            try:
                pair, comp, nfm = oracle.face_masks(int(v), max_faces=MAXF)
            except Exception:
                pair = np.zeros((MAXF, 8, 8), bool)
                comp = np.zeros((MAXF, 8), bool)

        d = np.asarray([int(x) for x in tk.eliminate(int(v))], np.int32)
        de = np.asarray([int(x) for x in tk.last_eqn_ids()], np.int32)
        ds = np.where(de >= 0, v - 1, -1).astype(np.int32)

        if fq is not None:
            try:
                segs = list(tk.last_face_segments())
            except Exception:
                segs = []
            ij = tk.ij
            sink = getattr(ij, "face_sink", None)
            recs = list(ij.step_faces(len(ij.steps) - 1)) if sink else []
            vidx = sink.vidx if (sink and sink.vidx is not None) else VIDX
            ekeys = [(vidx.get(fr.in_edge), vidx.get(fr.out_edge)) for fr in recs]
            if not probes:
                STATS["probe_empty"] += 1
            elif len(probes) != len(segs) or len(ekeys) != len(segs):
                STATS["len_mismatch"] += 1
            else:
                for k in range(min(len(segs), MAXF)):
                    s_, sp_, e_ = segs[k]
                    ki, kj = ekeys[k]
                    if ki is None or kj is None:
                        STATS["key_unmapped"] += 1
                        continue
                    si = int(slot_of_vidx[ki]) if 0 <= ki < len(slot_of_vidx) else -1
                    sj = int(slot_of_vidx[kj]) if 0 <= kj < len(slot_of_vidx) else -1
                    g = best_gnom(int(v), pair[k]) if pair is not None else 1
                    ft, ext = face_targets(probes[k], pair[k], comp[k], g, ki, kj)
                    faces.append((fq, cum + int(s_), cum + int(sp_), si, sj,
                                  v - 1, ft, ext))
                    STATS["faces"] += 1

        toks.append(d); owns.append(ds); eqns.append(de)
        didx.append(np.full(len(d), t, np.int16))
        cum += len(d)
        order.append(int(v))
        env.step(("V", int(v)))
        try:
            oracle.advance(int(v))
        except Exception:
            pass

    for i in range(len(QSTEPS)):
        prefix_len.setdefault(i, cum)
    return (np.concatenate(toks), np.concatenate(owns), np.concatenate(eqns),
            np.concatenate(didx),
            np.asarray([prefix_len[i] for i in range(len(QSTEPS))], np.int32),
            tgt, legal_m, np.asarray(order, np.int32), part, faces)


FSET = set(FSTEPS)
print("face query steps:", len(FSET), flush=True)
recs = []
t = time.time()
for i in range(NTRAJ):
    mode = "rev" if i % 2 == 0 else "rand"
    recs.append((mode,) + gen(1000 + i, mode))
    if i % 8 == 0:
        print(f"  traj {i}/{NTRAJ} len={len(recs[-1][1])} "
              f"faces={len(recs[-1][-1])} wall={time.time()-t:.0f}s {STATS}",
              flush=True)
L = max(len(r[1]) for r in recs)
N = len(recs)
print(f"max stream len {L}  N={N}  {STATS}", flush=True)

TOK = np.zeros((N, L), np.int32)
OWN = np.full((N, L), -1, np.int32)
EQN = np.full((N, L), -1, np.int32)
DID = np.full((N, L), -1, np.int16)
NTOK = np.zeros(N, np.int32)
PREF = np.zeros((N, len(QSTEPS)), np.int32)
TGT = np.zeros((N, len(QSTEPS), NV, len(TGT_NAMES)), np.float32)
LEG = np.zeros((N, len(QSTEPS), NV), bool)
MODE = np.zeros(N, np.int32)
PART = np.full((N, MAXSTEP + 1, KPART), -1, np.int16)
BND = np.full((N, NV), -1, np.int32)

f_traj, f_q, f_s, f_sp, f_si, f_sj, f_v = [], [], [], [], [], [], []
f_t, f_e = [], []
for i, (mode, tk_, ow, eq, dd, pf, tg, lg, od, pt, fs) in enumerate(recs):
    n = len(tk_)
    TOK[i, :n] = tk_; OWN[i, :n] = ow; EQN[i, :n] = eq; DID[i, :n] = dd
    NTOK[i] = n; PREF[i] = pf; TGT[i] = tg; LEG[i] = lg
    MODE[i] = 0 if mode == "rev" else 1
    PART[i] = pt
    nb = int(pf[0])                      # base = everything before delta 0
    ob = ow[:nb]
    for s in range(NV):
        w = np.nonzero(ob == s)[0]
        if len(w):
            BND[i, s] = int(w[-1])
    for (qi, s_, sp_, si, sj, vs, ft, ext) in fs:
        f_traj.append(i); f_q.append(qi); f_s.append(s_); f_sp.append(sp_)
        f_si.append(si); f_sj.append(sj); f_v.append(vs)
        f_t.append(ft); f_e.append(ext)

FT = np.asarray(f_t, np.float32) if f_t else np.zeros((0, NFT), np.float32)
FE = np.asarray(f_e, np.float32) if f_e else np.zeros((0, NDIM), np.float32)
np.savez(OUT, tok=TOK, own=OWN, eqn=EQN, did=DID, ntok=NTOK, pref=PREF,
         tgt=TGT, legal=LEG, mode=MODE, vfeat=vfeat.astype(np.float32),
         qsteps=np.asarray(QSTEPS), names=np.asarray(TGT_NAMES),
         slot_row=slot_row, part=PART, bnd=BND,
         f_traj=np.asarray(f_traj, np.int32), f_q=np.asarray(f_q, np.int32),
         f_start=np.asarray(f_s, np.int32), f_split=np.asarray(f_sp, np.int32),
         f_si=np.asarray(f_si, np.int32), f_sj=np.asarray(f_sj, np.int32),
         f_v=np.asarray(f_v, np.int32), f_tgt=FT, f_ext=FE,
         f_names=np.asarray(FT_NAMES), stats=np.asarray(json.dumps(STATS)))
print(f"saved {OUT} tok{TOK.shape} faces{FT.shape} wall={time.time()-T0:.0f}s",
      flush=True)
print("STATS", STATS, flush=True)
if len(FT):
    for c, nm in enumerate(FT_NAMES):
        v = FT[:, c]
        print(f"  face {nm:12s} mean={v.mean():8.3f} std={v.std():8.3f} "
              f"min={v.min():8.3f} max={v.max():8.3f}", flush=True)
    span = np.asarray(f_sp) - np.asarray(f_s)
    print(f"  chunk len mean={span.mean():.1f} min={span.min()} "
          f"max={span.max()} zero={(span == 0).sum()}", flush=True)
