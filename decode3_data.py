"""decode3_data: per-FACE + per-VERTEX decodability dataset for the
2026-08-14 architecture (identity pool + participation + endpoint face head).

Derived from decode2_face_data.py -- SAME target graph, SAME seeds, SAME
targets, SAME metrics.  What is new:

  * ARM A / ARM B.  ``--shapes`` switches the tokenizer between
      A: ``IncrementalPathTokenizer`` exactly as production runs it, and
      B: a LOCAL subclass that additionally states each equation's OUTPUT
         SHAPE after the outvar name.
    graphax is READ ONLY: the append-only tokenizer has no ``show_shapes``
    flag at all (that flag belongs to ``VEJaxpr.tokenized()``, which only the
    legacy ``vertexgame`` package calls), so arm B is implemented here, on the
    alphagrad side, by overriding ``_emit_eqns``.

  * PART is stored as an explicit (NSTEP, V+1) participation MASK, the exact
    object ``Agent.participation_mask`` builds, instead of a capped id list.

  * per-delta length + vocab statistics, so a truncated arm B is visible as a
    budget failure rather than as a failure of the idea.

Usage: decode3_data.py OUT.npz N_TRAJ [QSTEPS] [--shapes]
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
from jax import core as jcore
from graphax.jaxpr import _op_fn_key, _opaque_symbols

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SHAPES = "--shapes" in sys.argv
OUT = ARGS[0]
NTRAJ = int(ARGS[1]) if len(ARGS) > 1 else 64
QSTEPS = [int(x) for x in (ARGS[2].split(",") if len(ARGS) > 2
                           else ["0", "20", "50", "70"])]
MAXSTEP = max(QSTEPS)
FSTEPS = list(range(MAXSTEP + 1))
VOCAB = 512
MAXF = 64
NDIM = 12


# ---------------------------------------------------------------- ARM B
class ShapeTokenizer(IncrementalPathTokenizer):
    """Arm B: state every equation's OUTPUT SHAPE in the stream.

    A verbatim copy of ``IncrementalPathTokenizer._emit_eqns`` with ONE
    addition -- ``_format_shape(ov.aval.shape)`` after each outvar name.  It
    is an override, not an edit: graphax is untouched.

    ``_emit_eqns`` is the single emitter for BOTH ``base_tokens`` and every
    ``eliminate`` delta, so the append-only path gets the shapes too.  The
    span bookkeeping (``_cur_eqn_spans`` / ``_cur_spans`` / face segments) is
    recorded from ``len(out)`` AFTER the insertion, so every offset stays
    consistent by construction.

    The shape symbols ('<', '*', '>', digits) are the SAME atoms the input
    list already emits through ``_format_shape``, so no new name symbol is
    minted and the vocabulary cannot grow.  Verified, not assumed: the script
    prints max_token_id for both arms.
    """

    def _emit_eqns(self, eqns, out):
        _seg_start = len(out)
        _src_out = [] if getattr(self, "_cur_eqn_spans", None) is not None \
            else None
        flat = self._flatten(eqns, src_out=_src_out)
        render = []
        new_defs = []
        for outs, prim, params, ins in flat:
            keep = [v for v in outs if not isinstance(v, jcore.DropVar)]
            if not keep:
                render.append(None)
                continue
            pk = _op_fn_key(prim, params)
            if not pk:
                render.append(("op", prim))
            else:
                key = (prim, pk)
                name = self._fns.get(key)
                if name is None:
                    name = next(self._namegen)
                    self._fns[key] = name
                    new_defs.append((name, prim, params))
                render.append(("fn", name))
        if new_defs:
            self._emit_word("fns", out)
            for name, prim, params in new_defs:
                self._emit_atoms(name, out)
                out.append(self.vocab["="])
                self._emit_op_params(prim, params, out)
        out.append(self.vocab["{"])
        for _fi, ((outs, prim, params, ins), r) in enumerate(
                zip(flat, render)):
            if r is None:
                continue
            _eq_start = len(out)
            keep = [v for v in outs if not isinstance(v, jcore.DropVar)]
            for i, ov in enumerate(keep):
                if i:
                    out.append(self.vocab["_"])
                self._emit_atom(ov, out)
                # >>> ARM B, the only change <<<
                av = getattr(ov, "aval", None)
                shp = getattr(av, "shape", None)
                if shp:
                    self._emit_atoms(self._format_shape(shp), out)
            if r[0] == "fn":
                out.append(self.vocab["="])
                self._emit_atoms(r[1], out)
                out.append(self.vocab[":"])
            elif prim in self.vocab:
                out.append(self.vocab[prim])
            else:
                self._emit_atoms(
                    _opaque_symbols(prim, self.digit_base, kind="op"), out)
            for j, iv in enumerate(ins):
                if j:
                    out.append(self.vocab["_"])
                self._emit_atom(iv, out)
            out.append(self.vocab["\n"])
            if _src_out is not None and _fi < len(_src_out):
                self._cur_eqn_spans.append(
                    (_eq_start, len(out), _src_out[_fi]))
        if out and out[-1] == self.vocab["\n"]:
            out.pop()
        out.append(self.vocab["}"])
        if getattr(self, "_cur_spans", None) is not None and len(out) > _seg_start:
            self._cur_spans.append((_seg_start, len(out), self._eqn_seg))
            self._eqn_seg += 1


TKCLS = ShapeTokenizer if SHAPES else IncrementalPathTokenizer

TGT_COLS = [DV_FILL, DV_OUT_DEG, DV_IN_DEG, DV_MARKOWITZ, DV_MAX_JAC]
TGT_NAMES = ["fill", "out_deg", "in_deg", "markowitz", "max_jac", "static_ln",
             "elim_nb", "n_elim"]

FT_NAMES = [
    "ln_out", "ln_prim", "ln_maxdim", "ln_stored", "gain", "ln_factor",
    "n_diag", "n_comp", "n_paired", "n_compressed", "is_lowrank", "n_dims",
    "stat_ln_i", "stat_ln_j",
]
NFT = len(FT_NAMES)

T0 = time.time()
print(f"ARM {'B (output shapes stated)' if SHAPES else 'A (production)'}",
      flush=True)
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


from alphagrad.approx.env import diag_row_to_pair as _d2p


def MK_ROW2PAIR(vertex, bi1, bi2):
    return _d2p(jaxpr, vertex, bi1, bi2)


def best_gnom(vertex, pair_k):
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


FSET = set(FSTEPS)
STATS = dict(steps=0, probe_empty=0, len_mismatch=0, key_unmapped=0,
             faces=0, max_delta=0, max_tok_id=0)


def gen(seed, mode):
    rng = np.random.default_rng(seed)
    env.reset()
    tk = TKCLS(jaxpr, tuple(argnums), list(consts), list(args_),
               vocab_size=VOCAB)
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
    # PARTICIPATION MASK over (V+1) slots -- the trailing entry is the GLOBAL
    # slot, exactly Agent.participation_mask's layout.
    part = np.zeros((MAXSTEP + 1, NV + 1), np.uint8)
    faces = []

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

        try:
            keys_all = list(tk.ij.faces(int(v)))
        except Exception:
            keys_all = []
        part[t, v - 1] = 1
        for (ki, kj) in keys_all:
            for kk in (ki, kj):
                if kk is None:
                    continue
                s_ = int(slot_of_vidx[kk]) if 0 <= kk < len(slot_of_vidx) else -1
                if s_ >= 0:
                    part[t, s_] = 1

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
        STATS["max_delta"] = max(STATS["max_delta"], int(len(d)))
        if len(d):
            STATS["max_tok_id"] = max(STATS["max_tok_id"], int(d.max()))

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
    STATS["max_tok_id"] = max(STATS["max_tok_id"], int(base.max()))
    return (np.concatenate(toks), np.concatenate(owns), np.concatenate(eqns),
            np.concatenate(didx),
            np.asarray([prefix_len[i] for i in range(len(QSTEPS))], np.int32),
            tgt, legal_m, np.asarray(order, np.int32), part, faces)


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
PART = np.zeros((N, MAXSTEP + 1, NV + 1), np.uint8)

f_traj, f_q, f_s, f_sp, f_si, f_sj, f_v = [], [], [], [], [], [], []
f_t, f_e = [], []
for i, (mode, tk_, ow, eq, dd, pf, tg, lg, od, pt, fs) in enumerate(recs):
    n = len(tk_)
    TOK[i, :n] = tk_; OWN[i, :n] = ow; EQN[i, :n] = eq; DID[i, :n] = dd
    NTOK[i] = n; PREF[i] = pf; TGT[i] = tg; LEG[i] = lg
    MODE[i] = 0 if mode == "rev" else 1
    PART[i] = pt
    for (qi, s_, sp_, si, sj, vs, ft, ext) in fs:
        f_traj.append(i); f_q.append(qi); f_s.append(s_); f_sp.append(sp_)
        f_si.append(si); f_sj.append(sj); f_v.append(vs)
        f_t.append(ft); f_e.append(ext)

FT = np.asarray(f_t, np.float32) if f_t else np.zeros((0, NFT), np.float32)
FE = np.asarray(f_e, np.float32) if f_e else np.zeros((0, NDIM), np.float32)
np.savez(OUT, tok=TOK, own=OWN, eqn=EQN, did=DID, ntok=NTOK, pref=PREF,
         tgt=TGT, legal=LEG, mode=MODE, vfeat=vfeat.astype(np.float32),
         qsteps=np.asarray(QSTEPS), names=np.asarray(TGT_NAMES),
         slot_row=slot_row, part=PART,
         f_traj=np.asarray(f_traj, np.int32), f_q=np.asarray(f_q, np.int32),
         f_start=np.asarray(f_s, np.int32), f_split=np.asarray(f_sp, np.int32),
         f_si=np.asarray(f_si, np.int32), f_sj=np.asarray(f_sj, np.int32),
         f_v=np.asarray(f_v, np.int32), f_tgt=FT, f_ext=FE,
         f_names=np.asarray(FT_NAMES), shapes=np.asarray(int(SHAPES)),
         stats=np.asarray(json.dumps(STATS)))
print(f"saved {OUT} tok{TOK.shape} faces{FT.shape} wall={time.time()-T0:.0f}s",
      flush=True)
print("STATS", STATS, flush=True)
print(f"BUDGET  max delta tokens = {STATS['max_delta']}  "
      f"(ALPHAGRAD_MAX_DELTA_TOKENS default 1024)", flush=True)
print(f"VOCAB   max token id = {STATS['max_tok_id']}  (embedding table "
      f"vocab_size={VOCAB}) -> {'OVERFLOW' if STATS['max_tok_id'] >= VOCAB else 'ok'}",
      flush=True)
_pl = np.asarray([len(r[1]) for r in recs])
print(f"STREAM  len mean={_pl.mean():.0f} max={_pl.max()}", flush=True)
if len(FT):
    for c, nm in enumerate(FT_NAMES):
        v = FT[:, c]
        print(f"  face {nm:12s} mean={v.mean():8.3f} std={v.std():8.3f} "
              f"min={v.min():8.3f} max={v.max():8.3f}", flush=True)
    span = np.asarray(f_sp) - np.asarray(f_s)
    print(f"  chunk len mean={span.mean():.1f} min={span.min()} "
          f"max={span.max()} zero={(span == 0).sum()}", flush=True)
