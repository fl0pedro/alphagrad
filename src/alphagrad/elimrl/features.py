"""alphagrad.elimrl.features -- M2: typed features for the elimination GNN.

Two timescales, matching the MDP:

* STATIC per-vertex features are computed ONCE per target from the jaxpr
  (:func:`build_static`): primitive-vocab index (the learned embedding lives
  in the encoder), dot_general contracting/batch dimension-number multi-hots,
  output rank / log2 axis extents / log2 numel / dtype bit-width, aggregated
  per-invar stats + literal flags, role flags (is_param / is_data_input /
  is_output / is_seed / is_stop_gradient) and topology (normalised topo
  index, depth-from-inputs, height-to-outputs).

* DYNAMIC features are re-derived from each :class:`ElimState`
  (:func:`extract`): per-VERTEX degrees / Markowitz product / incident-fill
  count / eliminated flag / largest incident edge Jacobian, and per-EDGE
  features for every live line-dag Z node (log2 extents of the edge Jacobian
  out (+) in sides, sparsity-class one-hot derived from the SparseTensor dim
  metadata, stored-vs-logical log2 size, fill flag, log2 product cost).
  FACE candidates are assembled on demand from the two edge rows + three
  vertex rows + scalars (log2 product cost, absorb flag) -- the line graph is
  NEVER materialised.

Vertex-id -> row convention (fixed per target; the primal node set never
grows during an episode -- only edges change):

    invar vid -(i+1)  ->  row i
    eqn   vid j >= 1  ->  row n_invars + j - 1
    vid 0 (unmapped)  ->  row n_rows - 1        (catch-all)

Known state-interface gaps (M2 report): NodeMeta dim rows encode
``(class, id, size, val_dim, other_id)`` but the current graphax ``Index``
carries ``axis`` / ``block_size`` / ``block_axis`` -- there is no ``val_dim``
attribute, so that slot is always -1, and explicit BLOCK SIZES are not
recoverable. Sparsity CLASS still is (``other_id`` + subclass name); for
block sizes we substitute the stored-vs-logical log2-size gap, which subsumes
them numerically. Seed vertices (graphax.seed_vertices) are not marked in the
jaxpr either; :func:`build_static` accepts ``seed_vertices=`` from the caller.

Imports: numpy + jax core types only (no equinox, no alphagrad.approx).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from jax._src.core import Literal as _Literal, Var as _Var


MAX_RANK = 4                       # axis-extent slots; overflow folds into the last
PRIM_UNK = "<unk>"
PRIM_INPUT = "<input>"

# ---------------------------------------------------------------------------
# static per-vertex feature layout (see build_static)
# ---------------------------------------------------------------------------
S_DOT_LHS_C = slice(0, MAX_RANK)                       # dot_general lhs contract
S_DOT_RHS_C = slice(MAX_RANK, 2 * MAX_RANK)            # dot_general rhs contract
S_DOT_LHS_B = slice(2 * MAX_RANK, 3 * MAX_RANK)        # dot_general lhs batch
S_DOT_RHS_B = slice(3 * MAX_RANK, 4 * MAX_RANK)        # dot_general rhs batch
S_OUT_RANK = 4 * MAX_RANK                              # output rank / MAX_RANK
S_OUT_EXT = slice(4 * MAX_RANK + 1, 5 * MAX_RANK + 1)  # log2 output extents
S_OUT_LN = 5 * MAX_RANK + 1                            # log2 output numel
S_DTYPE = 5 * MAX_RANK + 2                             # dtype bits / 32
S_INVAR = slice(5 * MAX_RANK + 3, 5 * MAX_RANK + 12)   # 9 aggregated invar stats
S_ROLE = slice(5 * MAX_RANK + 12, 5 * MAX_RANK + 17)   # 5 role flags
S_TOPO = slice(5 * MAX_RANK + 17, 5 * MAX_RANK + 20)   # topo idx, depth, height
STATIC_DIM = 5 * MAX_RANK + 20                         # = 40 for MAX_RANK=4

# role flag order inside S_ROLE
ROLE_PARAM, ROLE_DATA, ROLE_OUTPUT, ROLE_SEED, ROLE_STOPGRAD = range(5)

# ---------------------------------------------------------------------------
# dynamic per-vertex feature layout (see extract)
# ---------------------------------------------------------------------------
DV_IN_DEG, DV_OUT_DEG, DV_MARKOWITZ, DV_FILL, DV_ELIMINATED, DV_MAX_JAC, \
    DV_IS_LEGAL = range(7)
DYN_VERTEX_DIM = 7

# ---------------------------------------------------------------------------
# per-edge feature layout (see extract)
# ---------------------------------------------------------------------------
E_OUT_EXT = slice(0, MAX_RANK)                 # log2 extents, edge-Jacobian out side
E_IN_EXT = slice(MAX_RANK, 2 * MAX_RANK)       # log2 extents, edge-Jacobian in side
E_OUT_LN = 2 * MAX_RANK                        # log2 numel out side
E_IN_LN = 2 * MAX_RANK + 1                     # log2 numel in side
E_CLS = slice(2 * MAX_RANK + 2, 2 * MAX_RANK + 6)   # one-hot, order below
E_HAS_VAL = 2 * MAX_RANK + 6                   # a buffer is actually stored
E_N_PAIRED = 2 * MAX_RANK + 7                  # paired (diagonal) dims / 2R
E_N_DENSE = 2 * MAX_RANK + 8                   # unpaired dims / 2R
E_STORED = 2 * MAX_RANK + 9                    # log2 stored elements
E_LOGICAL = 2 * MAX_RANK + 10                  # log2 logical (dense) elements
E_GAP = 2 * MAX_RANK + 11                      # stored - logical (block/sparsity)
E_IS_FILL = 2 * MAX_RANK + 12
E_PRODCOST = 2 * MAX_RANK + 13                 # log2 fma cost through this edge
EDGE_DIM = 2 * MAX_RANK + 14                   # = 22 for MAX_RANK=4

# E_CLS one-hot order
CLS_NOMETA, CLS_DENSE, CLS_DIAGONAL, CLS_COMPRESSED = range(4)

FACE_SCALAR_DIM = 2                            # log2 product cost, absorb flag


def _log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


def _shape_stats(shape) -> Tuple[float, np.ndarray, float]:
    """(rank, log2 extents padded to MAX_RANK (overflow folded), log2 numel)."""
    ext = np.zeros(MAX_RANK, np.float32)
    ln = 0.0
    for i, s in enumerate(shape):
        l2 = _log2(int(s))
        ln += l2
        ext[min(i, MAX_RANK - 1)] += l2
    return float(len(shape)), ext, ln


@dataclass(frozen=True)
class StaticFeatures:
    """Per-target static tables (computed once from the jaxpr)."""
    n_invars: int
    n_eqns: int
    n_rows: int                 # n_invars + n_eqns + 1 (catch-all row last)
    prim_vocab: Dict[str, int]  # primitive name -> embedding index
    prim_idx: np.ndarray        # (n_rows,) int32
    feat: np.ndarray            # (n_rows, STATIC_DIM) float32
    log2_extents: np.ndarray    # (n_rows, MAX_RANK) float32 -- output var shape
    lognumel: np.ndarray        # (n_rows,) float32
    max_init_nid: int           # line-dag nids > this are FILL nodes

    def row_of(self, vid: int) -> int:
        if vid < 0:
            return -vid - 1
        if vid == 0:
            return self.n_rows - 1
        return self.n_invars + vid - 1


def build_static(env, seed_vertices: Sequence[int] = ()) -> StaticFeatures:
    """Build the static vertex tables from a FRESH (unstepped) ElimEnv.

    ``seed_vertices``: graphax vertex ids of tangent/adjoint seed vertices
    (graphax.seed_vertices) -- the jaxpr itself does not mark them.
    """
    if getattr(env, "steps", 0) != 0:
        raise ValueError("build_static needs a freshly-reset env "
                         "(fill-node detection anchors on the initial nids)")
    jaxpr = env.jaxpr
    invars = list(jaxpr.invars)
    eqns = list(jaxpr.eqns)
    n_i, n_e = len(invars), len(eqns)
    n_rows = n_i + n_e + 1

    vocab: Dict[str, int] = {PRIM_UNK: 0, PRIM_INPUT: 1}
    for name in sorted({e.primitive.name for e in eqns}):
        vocab.setdefault(name, len(vocab))

    prim_idx = np.zeros(n_rows, np.int32)
    feat = np.zeros((n_rows, STATIC_DIM), np.float32)
    log2_ext = np.zeros((n_rows, MAX_RANK), np.float32)
    lognumel = np.zeros(n_rows, np.float32)

    # --- producing vertex of every var; depth/height over the primal DAG ---
    producer: Dict[int, int] = {}           # id(var) -> vid
    for i, iv in enumerate(invars):
        producer[id(iv)] = -(i + 1)
    for j, eqn in enumerate(eqns, start=1):
        for ov in eqn.outvars:
            producer[id(ov)] = j

    depth = np.zeros(n_rows, np.float32)    # longest path from the inputs
    consumers: Dict[int, list] = {}
    for j, eqn in enumerate(eqns, start=1):
        row = n_i + j - 1
        d = 0.0
        for v in eqn.invars:
            if isinstance(v, _Var) and id(v) in producer:
                pvid = producer[id(v)]
                prow = (-pvid - 1) if pvid < 0 else n_i + pvid - 1
                d = max(d, depth[prow] + 1.0)
                consumers.setdefault(prow, []).append(row)
        depth[row] = d
    height = np.zeros(n_rows, np.float32)   # longest path to any consumer sink
    for j in range(n_e, 0, -1):             # reverse topo (eqns are topo-sorted)
        row = n_i + j - 1
        for c in consumers.get(row, ()):
            height[row] = max(height[row], height[c] + 1.0)
    for i in range(n_i):
        for c in consumers.get(i, ()):
            height[i] = max(height[i], height[c] + 1.0)
    dmax = max(float(depth.max()), 1.0)
    hmax = max(float(height.max()), 1.0)

    outset = {id(ov) for ov in jaxpr.outvars}
    argset = set(env.argnums)
    seedset = set(int(s) for s in seed_vertices)

    # --- invar rows ---
    for i, iv in enumerate(invars):
        row = i
        prim_idx[row] = vocab[PRIM_INPUT]
        rank, ext, ln = _shape_stats(iv.aval.shape)
        feat[row, S_OUT_RANK] = rank / MAX_RANK
        feat[row, S_OUT_EXT] = ext
        feat[row, S_OUT_LN] = ln
        feat[row, S_DTYPE] = iv.aval.dtype.itemsize * 8 / 32.0
        roles = feat[row, S_ROLE]
        roles[ROLE_PARAM] = 1.0 if i in argset else 0.0
        roles[ROLE_DATA] = 0.0 if i in argset else 1.0
        feat[row, S_TOPO] = (0.0, 0.0, height[row] / hmax)
        log2_ext[row] = ext
        lognumel[row] = ln

    # --- eqn rows ---
    for j, eqn in enumerate(eqns, start=1):
        row = n_i + j - 1
        prim_idx[row] = vocab.get(eqn.primitive.name, vocab[PRIM_UNK])
        # dot_general dimension-number multi-hots
        if eqn.primitive.name == "dot_general":
            (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
            for sl, axes in ((S_DOT_LHS_C, lc), (S_DOT_RHS_C, rc),
                             (S_DOT_LHS_B, lb), (S_DOT_RHS_B, rb)):
                block = feat[row, sl]
                for a in axes:
                    block[min(int(a), MAX_RANK - 1)] = 1.0
        # output block (first outvar; multi-outvar eqns share the vertex)
        ov = eqn.outvars[0]
        rank, ext, ln = _shape_stats(ov.aval.shape)
        feat[row, S_OUT_RANK] = rank / MAX_RANK
        feat[row, S_OUT_EXT] = ext
        feat[row, S_OUT_LN] = ln
        feat[row, S_DTYPE] = ov.aval.dtype.itemsize * 8 / 32.0
        log2_ext[row] = ext
        lognumel[row] = ln
        # aggregated invar stats + literal flags
        ranks, lns, mexts = [], [], []
        n_lit = 0
        for v in eqn.invars:
            if isinstance(v, _Literal):
                n_lit += 1
            av = v.aval
            r, e, l = _shape_stats(av.shape)
            ranks.append(r)
            lns.append(l)
            mexts.append(float(e.max()) if len(av.shape) else 0.0)
        n_in = max(len(eqn.invars), 1)
        inv = feat[row, S_INVAR]
        if ranks:
            inv[0] = float(np.mean(ranks)) / MAX_RANK
            inv[1] = float(np.max(ranks)) / MAX_RANK
            inv[2] = float(np.mean(lns))
            inv[3] = float(np.max(lns))
            inv[4] = float(np.mean(mexts))
            inv[5] = float(np.max(mexts))
        inv[6] = _log2(1 + len(eqn.invars))
        inv[7] = float(n_lit)
        inv[8] = (len(eqn.invars) - n_lit) / n_in
        # role flags
        roles = feat[row, S_ROLE]
        roles[ROLE_OUTPUT] = 1.0 if any(id(o) in outset for o in eqn.outvars) else 0.0
        roles[ROLE_SEED] = 1.0 if j in seedset else 0.0
        roles[ROLE_STOPGRAD] = 1.0 if eqn.primitive.name == "stop_gradient" else 0.0
        # topology
        feat[row, S_TOPO] = ((j - 1) / max(n_e - 1, 1),
                             depth[row] / dmax, height[row] / hmax)

    st0 = env.state()
    max_init_nid = max((n.nid for n in st0.nodes), default=-1)
    return StaticFeatures(
        n_invars=n_i, n_eqns=n_e, n_rows=n_rows, prim_vocab=vocab,
        prim_idx=prim_idx, feat=feat, log2_extents=log2_ext,
        lognumel=lognumel, max_init_nid=max_init_nid)


# ---------------------------------------------------------------------------
# dynamic extraction
# ---------------------------------------------------------------------------
@dataclass
class StepFeatures:
    """Per-step arrays: vertex-dynamic rows, edge rows (one per live line-dag
    Z node), and candidate index sets. All numpy, unpadded."""
    vert_dyn: np.ndarray        # (n_rows, DYN_VERTEX_DIM) float32
    edge_feat: np.ndarray       # (E, EDGE_DIM) float32
    edge_src: np.ndarray        # (E,) int32 -- primal ROW of the source vertex
    edge_dst: np.ndarray        # (E,) int32
    edge_nid: np.ndarray        # (E,) int64 -- line-dag node id (edge identity)
    nid_index: Dict[int, int]   # nid -> position in the edge arrays
    face_edge_u: np.ndarray     # (F,) int32 -- edge-array index of face's u
    face_edge_w: np.ndarray     # (F,) int32
    face_rows: np.ndarray       # (F, 3) int32 -- rows of (i, j, k)
    face_scal: np.ndarray       # (F, FACE_SCALAR_DIM) float32
    vert_rows: np.ndarray       # (V,) int32 -- rows of legal vertices


def _classify(node) -> Tuple[int, float, int, int]:
    """(class, has_val, n_paired_dims, n_dense_dims) from NodeMeta dim rows.

    class: CLS_NOMETA (no dim metadata at all -- symbolic fill), CLS_COMPRESSED
    (any BandedIndex/SetIndex/ToeplitzIndex row), CLS_DIAGONAL (any paired dim,
    other_id set), CLS_DENSE otherwise.
    """
    rows = node.out_dims + node.primal_dims
    has_val = 1.0 if node.val_shape is not None else 0.0
    if not rows and node.val_shape is None:
        return CLS_NOMETA, has_val, 0, 0
    n_paired = sum(1 for r in rows if r[4] is not None)
    n_dense = len(rows) - n_paired
    if any(r[0] != "Index" for r in rows):
        return CLS_COMPRESSED, has_val, n_paired, n_dense
    if n_paired:
        return CLS_DIAGONAL, has_val, n_paired, n_dense
    return CLS_DENSE, has_val, n_paired, n_dense


def extract(state, static: StaticFeatures) -> StepFeatures:
    """All dynamic features of one ElimState (numpy, no jax)."""
    n_rows = static.n_rows
    row_of = static.row_of
    ln = static.lognumel

    z_nodes = [n for n in state.nodes if n.kind == "Z"]
    E = len(z_nodes)
    edge_feat = np.zeros((E, EDGE_DIM), np.float32)
    edge_src = np.zeros(E, np.int32)
    edge_dst = np.zeros(E, np.int32)
    edge_nid = np.zeros(E, np.int64)
    nid_index: Dict[int, int] = {}

    for p, n in enumerate(z_nodes):
        srow, drow = row_of(n.src_vertex), row_of(n.snk_vertex)
        edge_src[p], edge_dst[p], edge_nid[p] = srow, drow, n.nid
        nid_index[n.nid] = p
        f = edge_feat[p]
        f[E_OUT_EXT] = static.log2_extents[drow]
        f[E_IN_EXT] = static.log2_extents[srow]
        f[E_OUT_LN] = ln[drow]
        f[E_IN_LN] = ln[srow]
        cls, has_val, n_paired, n_dense = _classify(n)
        f[E_CLS.start + cls] = 1.0
        f[E_HAS_VAL] = has_val
        f[E_N_PAIRED] = n_paired / (2 * MAX_RANK)
        f[E_N_DENSE] = n_dense / (2 * MAX_RANK)
        stored = 0.0
        if n.val_shape is not None:
            stored = float(sum(_log2(s) for s in n.val_shape))
        logical = float(ln[drow] + ln[srow])
        f[E_STORED] = stored
        f[E_LOGICAL] = logical
        f[E_GAP] = stored - logical if has_val else 0.0
        f[E_IS_FILL] = 1.0 if n.nid > static.max_init_nid else 0.0
        f[E_PRODCOST] = logical

    # --- vertex dynamics (vectorised) ---
    vd = np.zeros((n_rows, DYN_VERTEX_DIM), np.float32)
    in_deg = np.bincount(edge_dst, minlength=n_rows).astype(np.float32)
    out_deg = np.bincount(edge_src, minlength=n_rows).astype(np.float32)
    fill_mask = edge_feat[:, E_IS_FILL] > 0
    fill_cnt = (np.bincount(edge_dst[fill_mask], minlength=n_rows)
                + np.bincount(edge_src[fill_mask], minlength=n_rows)).astype(np.float32)
    max_jac = np.zeros(n_rows, np.float32)
    if E:
        logical_col = edge_feat[:, E_LOGICAL]
        np.maximum.at(max_jac, edge_dst, logical_col)
        np.maximum.at(max_jac, edge_src, logical_col)
    vd[:, DV_IN_DEG] = np.log2(1.0 + in_deg)
    vd[:, DV_OUT_DEG] = np.log2(1.0 + out_deg)
    vd[:, DV_MARKOWITZ] = np.log2(1.0 + in_deg * out_deg)
    vd[:, DV_FILL] = np.log2(1.0 + fill_cnt)
    vd[:, DV_MAX_JAC] = max_jac
    for j in state.eliminated:
        vd[row_of(j), DV_ELIMINATED] = 1.0
    for j in state.legal_vertices:
        vd[row_of(j), DV_IS_LEGAL] = 1.0

    # --- FACE candidates (derived on demand; no line graph) ---
    F = len(state.faces)
    face_u = np.zeros(F, np.int32)
    face_w = np.zeros(F, np.int32)
    face_rows = np.zeros((F, 3), np.int32)
    face_scal = np.zeros((F, FACE_SCALAR_DIM), np.float32)
    for p, fm in enumerate(state.faces):
        face_u[p] = nid_index[fm.u]
        face_w[p] = nid_index[fm.w]
        ri, rj, rk = row_of(fm.i_vertex), row_of(fm.j_vertex), row_of(fm.k_vertex)
        face_rows[p] = (ri, rj, rk)
        face_scal[p, 0] = ln[ri] + ln[rj] + ln[rk]   # log2 dense fma cost
        face_scal[p, 1] = 1.0 if fm.absorb else 0.0

    vert_rows = np.asarray([row_of(j) for j in state.legal_vertices], np.int32)
    return StepFeatures(
        vert_dyn=vd, edge_feat=edge_feat, edge_src=edge_src, edge_dst=edge_dst,
        edge_nid=edge_nid, nid_index=nid_index,
        face_edge_u=face_u, face_edge_w=face_w, face_rows=face_rows,
        face_scal=face_scal, vert_rows=vert_rows)


def changed_rows(prev: StepFeatures, cur: StepFeatures) -> Tuple[np.ndarray, np.ndarray]:
    """Delta between two consecutive steps, at the FEATURE level (robust: no
    graph-theoretic reasoning; anything that alters an input alters a row).

    Returns (d0, edge_ep):
      d0:      rows whose vertex-dynamic features changed (h0 must be redone);
      edge_ep: rows incident to an added / removed / feature-changed edge
               (their h1+ must be redone even if their h0 did not change).
    """
    d0 = np.nonzero((prev.vert_dyn != cur.vert_dyn).any(axis=1))[0].astype(np.int32)
    ep = set()
    for nid, i in cur.nid_index.items():
        j = prev.nid_index.get(nid)
        if j is None or (prev.edge_feat[j] != cur.edge_feat[i]).any():
            ep.add(int(cur.edge_src[i]))
            ep.add(int(cur.edge_dst[i]))
    for nid, j in prev.nid_index.items():
        if nid not in cur.nid_index:
            ep.add(int(prev.edge_src[j]))
            ep.add(int(prev.edge_dst[j]))
    return d0, np.asarray(sorted(ep), np.int32)
