"""decode_data: build the supervised decodability dataset.

INPUT  = exactly what the policy sees: the palimpsa token stream (base +
per-elimination deltas) with the SAME per-token owner ids the per-vertex
memory is keyed on (carry_stream.init_carry / advance).
TARGET = the live per-vertex dynamic features of elimrl.features.extract on
the ElimEnv state at the same step.

Usage: decode_data.py OUT.npz N_TRAJ [MAX_STEP]
"""
import os, sys, time
import numpy as np
import jax

from alphagrad.elimrl.baselines import tlm_target
from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.features import (
    build_static, extract, DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL,
    DV_MAX_JAC, S_OUT_LN)
from alphagrad.approx.common.instrumentation import compute_vertex_features
from graphax import IncrementalPathTokenizer

OUT = sys.argv[1]
NTRAJ = int(sys.argv[2]) if len(sys.argv) > 2 else 64
QSTEPS = [0, 20, 50, 70]
MAXSTEP = max(QSTEPS)
VOCAB = 256

TGT_COLS = [DV_FILL, DV_OUT_DEG, DV_IN_DEG, DV_MARKOWITZ, DV_MAX_JAC]
TGT_NAMES = ["fill", "out_deg", "in_deg", "markowitz", "max_jac", "static_ln",
             "elim_nb", "n_elim"]

T0 = time.time()
fn, args_, argnums = tlm_target(seq=32, dmodel=128, vocab=1024)
env = ElimEnv(fn, args_, argnums, vertex_only=True, symbolic=True)
static = build_static(env)
closed = jax.make_jaxpr(fn)(*args_)
jaxpr, consts = closed.jaxpr, closed.literals
NV = len(jaxpr.eqns)                       # vmem slots 0..NV-1  <->  vertex s+1
print(f"[{time.time()-T0:.0f}s] n_eqns={NV} n_rows={static.n_rows} "
      f"jacve={len(env.jacve_vertices)}", flush=True)

vfeat = compute_vertex_features(jaxpr, tuple(consts), tuple(args_),
                                eval_samples=None, argnums=tuple(argnums))
print("vertex_features", vfeat.shape, flush=True)

# slot -> feature row, and the static log2-numel control per slot
slot_row = np.asarray([static.row_of(s + 1) for s in range(NV)], np.int32)
static_ln = static.feat[slot_row, S_OUT_LN].astype(np.float32)

# --- DYNAMIC POSITIVE CONTROL -------------------------------------------
# `elim_nb` = how many of v's ORIGINAL jaxpr neighbours are already gone.
# It is pure "who has been eliminated" x static structure -- the minimal
# live quantity the stream unambiguously contains (every delta is owned by
# the vertex that emitted it). If the representation cannot even carry this,
# nothing dynamic survives the routing; if it carries this but not `fill`,
# what is lost is graph STRUCTURE, not the elimination history.
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


def gen(seed, mode):
    """One trajectory: concatenated stream + owners + eqn ids + targets."""
    rng = np.random.default_rng(seed)
    env.reset()
    tk = IncrementalPathTokenizer(jaxpr, tuple(argnums), list(consts),
                                  list(args_), vocab_size=VOCAB)
    base = np.asarray([int(x) for x in tk.base_tokens()], np.int32)
    bown = np.asarray([int(x) for x in tk.last_owner_ids()], np.int32)
    beqn = np.asarray([int(x) for x in tk.last_eqn_ids()], np.int32)
    assert len(bown) == len(base) == len(beqn), (len(base), len(bown), len(beqn))
    # init_carry: owner is 1-based; slot = owner-1; 0 (no owner) -> -1 (global)
    bslot = np.where(bown > 0, bown - 1, -1).astype(np.int32)

    toks = [base]; owns = [bslot]; eqns = [beqn]
    cum = len(base)
    prefix_len = {}
    tgt = np.zeros((len(QSTEPS), NV, len(TGT_NAMES)), np.float32)
    legal_m = np.zeros((len(QSTEPS), NV), bool)
    order = []
    for t in range(MAXSTEP + 1):
        st = env.state()
        legal = st.legal_vertices
        if not legal:
            break
        if t in QSTEPS:
            qi = QSTEPS.index(t)
            prefix_len[qi] = cum
            sf = extract(st, static)
            vd = sf.vert_dyn[slot_row]                     # (NV, DYN)
            for c, col in enumerate(TGT_COLS):
                tgt[qi, :, c] = vd[:, col]
            tgt[qi, :, len(TGT_COLS)] = static_ln
            gone = set(st.eliminated)
            tgt[qi, :, len(TGT_COLS) + 1] = np.asarray(
                [len(NB[s + 1] & gone) for s in range(NV)], np.float32)
            tgt[qi, :, len(TGT_COLS) + 2] = float(len(gone))
            for j in legal:
                legal_m[qi, j - 1] = True
        if t == MAXSTEP:
            break
        if mode == "rev":
            v = max(legal) if rng.random() > 0.15 else int(rng.choice(legal))
        else:
            v = int(rng.choice(legal))
        d = np.asarray([int(x) for x in tk.eliminate(int(v))], np.int32)
        de = np.asarray([int(x) for x in tk.last_eqn_ids()], np.int32)
        assert len(de) == len(d)
        # advance(): ids = where(eqn >= 0, owner_slot, -1); owner slot = v-1
        ds = np.where(de >= 0, v - 1, -1).astype(np.int32)
        toks.append(d); owns.append(ds); eqns.append(de)
        cum += len(d)
        order.append(int(v))
        env.step(("V", int(v)))
    return (np.concatenate(toks), np.concatenate(owns), np.concatenate(eqns),
            np.asarray([prefix_len[i] for i in range(len(QSTEPS))], np.int32),
            tgt, legal_m, np.asarray(order, np.int32))


recs = []
t = time.time()
for i in range(NTRAJ):
    mode = "rev" if i % 2 == 0 else "rand"
    recs.append((mode,) + gen(1000 + i, mode))
    if i % 16 == 0:
        print(f"  traj {i}/{NTRAJ} len={len(recs[-1][1])} wall={time.time()-t:.0f}s",
              flush=True)
L = max(len(r[1]) for r in recs)
print(f"max stream len {L}  (pad to {L})", flush=True)
N = len(recs)
TOK = np.zeros((N, L), np.int32)
OWN = np.full((N, L), -1, np.int32)
EQN = np.full((N, L), -1, np.int32)
NTOK = np.zeros(N, np.int32)
PREF = np.zeros((N, len(QSTEPS)), np.int32)
TGT = np.zeros((N, len(QSTEPS), NV, len(TGT_NAMES)), np.float32)
LEG = np.zeros((N, len(QSTEPS), NV), bool)
MODE = np.zeros(N, np.int32)
for i, (mode, tk_, ow, eq, pf, tg, lg, od) in enumerate(recs):
    n = len(tk_)
    TOK[i, :n] = tk_; OWN[i, :n] = ow; EQN[i, :n] = eq
    NTOK[i] = n; PREF[i] = pf; TGT[i] = tg; LEG[i] = lg
    MODE[i] = 0 if mode == "rev" else 1
np.savez(OUT, tok=TOK, own=OWN, eqn=EQN, ntok=NTOK, pref=PREF, tgt=TGT,
         legal=LEG, mode=MODE, vfeat=vfeat.astype(np.float32),
         qsteps=np.asarray(QSTEPS), names=np.asarray(TGT_NAMES),
         slot_row=slot_row)
print(f"saved {OUT}  tok{TOK.shape} tgt{TGT.shape} wall={time.time()-T0:.0f}s",
      flush=True)
for qi, q in enumerate(QSTEPS):
    for c, nm in enumerate(TGT_NAMES):
        v = TGT[:, qi][LEG[:, qi]][:, c]
        print(f"  t={q:3d} {nm:10s} mean={v.mean():7.3f} std={v.std():7.3f}",
              flush=True)
