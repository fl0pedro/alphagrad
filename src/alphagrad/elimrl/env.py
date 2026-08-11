"""alphagrad.elimrl.env -- M0: ONE unified face/vertex elimination MDP (EXACT only).

Engine provenance
-----------------
The line-dag face-elimination engine below is copied and adapted from the
loose (non-repo) prototypes at the top level of pgi15:~/dsnn:

    dual_face_v2.py   -- transform-aware ``_contract``, fill/absorb + merge rules,
                         validated arbitrary-order-correct vs jax.jacrev.
    face_env.py       -- the stepable-env shape (dynamic ``faces()`` action set).
    naumann_dense.py  -- dense numpy reference (Rule 1, arXiv:2303.16087 /
                         Naumann 2004); source of the duplicate-block SUM fix.

Semantics: Naumann Rule 1 face elimination on the LINE DAG with DISTINCT
vertex identities (path labels). Eliminating the face (u=(i..j), w=(j..k)):

    1. absorb-or-fill: if a live Z node t with source i, sink k,
       P(t) == P(u) and S(t) == S(w) exists, absorb into it; otherwise fill a
       new node inheriting P(u) up-links and S(w) down-links. absorb-vs-create
       is READ FROM THE STATE (never a policy choice) -- see
       :meth:`ElimEnv.absorb_target` and ``FaceMeta.absorb``.
    2. fma:  cf[t] += cf[w] . cf[u]      (transform-aware ``_contract``)
    3. remove the face edge u -> w.
    4./5. u and w are isolated-dropped, or merged with a same-endpoints,
       same-neighbourhood sibling.

Terminal when the line dag is BIPARTITE: no Z->Z edge remains (``faces()``
empty). A vertex elimination is a *sequence* of face eliminations, so the
VERTEX(j) macro-action (eliminate ALL faces with middle j) reproduces
graphax ``jacve``'s per-vertex step; ``vertex_only=True`` restricts the MDP
to exactly jacve's search space (see ``jacve_vertices``).

One deliberate fix vs dual_face_v2: :meth:`ElimEnv.jacobian` SUMS duplicate
surviving (x, y) blocks (naumann_dense ``collect(mode="sum")``) instead of
dict-overwriting them -- overwrite silently drops path mass whenever two
finished (x, y) nodes coexist at termination.

M0 contract: EXACT accumulation only. The prototypes' per-face ``transforms``
hooks are intentionally NOT ported. Imports: graphax + jax + numpy (+ stdlib)
ONLY.

Action encoding (JSON-serialisable, replayable across processes)
----------------------------------------------------------------
    ("F", u, w)   FACE micro-action: eliminate live face u -> w, where u/w are
                  line-dag node ids as enumerated by ``faces()`` / ``state()``.
    ("V", j)      VERTEX macro-action: eliminate ALL faces whose middle var is
                  an outvar of graphax vertex j (eqn j-1; ids start at 1,
                  jacve numbering).

Node-id determinism: nodes are created in a fixed order (argnums invars, then
outvars, then primal edges in ``_build_graph`` insertion order), and every
container iterated for choices is a dict (insertion-ordered) or an int-set
(value-ordered), so identical (fn, args, argnums, action-prefix) yields
identical node ids in any process -- recorded ``history`` replays exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np  # noqa: F401  (kept: part of the documented M0 import contract)
import jax

from jax._src.core import Var as _Var

from graphax.core import (
    _acts_as_identity,
    _build_graph,
    _compressed_dims,
    _drain_or_unload_pre,
    _drain_transforms,
    _force,
    _identity_passthrough,
    _is_scalar_st,
    _materialize_for_op,
    append_pre_transforms,
    prepend_post_transforms,
    unload_post_transforms,
)


# ---------------------------------------------------------------------------
# transform-aware edge composition (verbatim from dual_face_v2._contract)
# ---------------------------------------------------------------------------
def _contract(post_val, pre_val):
    """Transform-aware edge composition, faithful to graphax
    ``_eliminate_vertex``'s inner loop: drain reshape/transpose transforms
    BEFORE the sparse matmul, then re-queue post/pre transforms on the result.
    (A bare ``post @ pre`` silently mis-lays-out reshape/transpose edges.)
    Copied from pgi15:~/dsnn/dual_face_v2.py (validated ViT-correct)."""
    if len(pre_val.post_transforms) > 0 and post_val.val is not None:
        _post_val = unload_post_transforms(post_val, pre_val)
    else:
        _post_val = post_val.copy()
    _post_val, _pre_val = _drain_or_unload_pre(post_val, pre_val, _post_val)
    need = ((pre_val.val is not None and post_val.val is not None)
            or (post_val.val is None and not _acts_as_identity(_post_val))
            or (pre_val.val is None and not _acts_as_identity(_pre_val)))
    if need:
        if _is_scalar_st(_post_val) and _is_scalar_st(_pre_val):
            edge_outval = _post_val * _pre_val
        else:
            edge_outval = _post_val @ _pre_val
    elif pre_val.val is not None:
        edge_outval = _identity_passthrough(_pre_val, _post_val, "pre")
    else:
        edge_outval = _identity_passthrough(_post_val, _pre_val, "post")
    if len(post_val.post_transforms) > 0:
        edge_outval = prepend_post_transforms(post_val, edge_outval)
    if len(pre_val.pre_transforms) > 0:
        edge_outval = append_pre_transforms(pre_val, edge_outval)
    if _compressed_dims(edge_outval):
        edge_outval = _materialize_for_op(edge_outval)
    return edge_outval


def _merge_cf(a, b):
    """Accumulate two coefficient SparseTensors (graphax merge semantics:
    drain queued transforms before the add). Either side may be None."""
    if a is None:
        return b
    if b is None:
        return a
    return _drain_transforms(a) + _drain_transforms(b)


# ---------------------------------------------------------------------------
# state-object interface (the M2 encoder builds against these)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeMeta:
    """One live line-dag node.

    kind:        'X' (input boundary) | 'Z' (Jacobian-carrying) | 'Y' (output
                 boundary).
    src_vertex / snk_vertex: graphax vertex ids of the source/sink primal var
                 (eqn index + 1; invars are -(position+1); X/Y boundary nodes
                 carry their own var in both slots).
    val_shape / val_dtype: shape/dtype of the stored SparseTensor buffer
                 (None for structural-identity edges with ``val is None`` and
                 for X/Y boundary nodes).
    out_dims / primal_dims: SparseTensor dimension metadata, encoded as
                 tuples of (class_name, id, size, val_dim, other_id) with
                 val_dim/other_id -1/None when absent.
    """
    nid: int
    kind: str
    src_vertex: int
    snk_vertex: int
    val_shape: Optional[Tuple[int, ...]]
    val_dtype: Optional[str]
    out_dims: Tuple[Tuple[Any, ...], ...]
    primal_dims: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class FaceMeta:
    """One legal FACE action: eliminate the live face u -> w.

    i_vertex/j_vertex/k_vertex: graphax vertex ids of (source, middle, sink)
    primal vars of the face.
    absorb: True iff the contraction will be ABSORBED into an existing target
    node (``target`` = its nid); False iff it will CREATE a fill node. Read
    from the state, never a policy choice.
    """
    u: int
    w: int
    i_vertex: int
    j_vertex: int
    k_vertex: int
    absorb: bool
    target: Optional[int]


@dataclass(frozen=True)
class ElimState:
    """Snapshot of the MDP state.

    nodes:            live line-dag nodes (sorted by nid) incl. fill nodes.
    edges:            live line-dag adjacency as (from_nid, to_nid), sorted.
    faces:            the legal FACE actions (empty tuple in vertex_only mode).
    legal_vertices:   the legal VERTEX macro-actions (graphax vertex ids).
    eliminated:       jacve-eliminable vertex ids that are fully eliminated.
    """
    nodes: Tuple[NodeMeta, ...]
    edges: Tuple[Tuple[int, int], ...]
    faces: Tuple[FaceMeta, ...]
    legal_vertices: Tuple[int, ...]
    eliminated: frozenset
    vertex_only: bool
    done: bool
    steps: int


def _st_meta(st):
    """(val_shape, val_dtype, out_dims, primal_dims) for a SparseTensor|None."""
    if st is None:
        return None, None, (), ()
    val = getattr(st, "val", None)
    shape = tuple(int(s) for s in val.shape) if val is not None else None
    dtype = str(val.dtype) if val is not None else None

    def enc(dims):
        rows = []
        for d in dims:
            vd = getattr(d, "val_dim", None)
            rows.append((type(d).__name__,
                         int(getattr(d, "id", -1)),
                         int(getattr(d, "size", 0) or 0),
                         -1 if vd is None else int(vd),
                         getattr(d, "other_id", None)))
        return tuple(rows)

    return shape, dtype, enc(getattr(st, "out_dims", ())), enc(getattr(st, "primal_dims", ()))


# ---------------------------------------------------------------------------
# the unified MDP
# ---------------------------------------------------------------------------
class ElimEnv:
    """Unified face/vertex elimination MDP over the line dag of ``fn``'s jaxpr.

    Args:
        fn:          function to differentiate (same contract as graphax.jacve).
        args:        concrete (or traced) example arguments.
        argnums:     differentiable input slots.
        vertex_only: restrict the action space to VERTEX macros == exactly the
                     jacve search space; FACE actions raise.
        symbolic:    structural mode -- no coefficient math is performed
                     (``jacobian()`` unavailable); used for cheap plan
                     generation where only graph dynamics matter.
    """

    def __init__(self, fn: Callable, args: Sequence, argnums: Sequence[int] = (0,),
                 vertex_only: bool = False, symbolic: bool = False):
        self.fn = fn
        self.args = tuple(args)
        self.argnums = tuple(int(a) for a in argnums)
        self.vertex_only = bool(vertex_only)
        self.symbolic = bool(symbolic)

        closed = jax.make_jaxpr(fn)(*self.args)
        self.jaxpr = closed.jaxpr

        # topo rank of each var (inputs = -1) -- drives the "rev" policy.
        self.rank = {}
        for iv in self.jaxpr.invars:
            self.rank[id(iv)] = -1
        for pos, eqn in enumerate(self.jaxpr.eqns):
            for ov in eqn.outvars:
                self.rank[id(ov)] = pos

        # var <-> graphax vertex id (eqns are 1-based like jacve orders;
        # invars get -(position+1)).
        self._vertex_of = {}
        self._vars_of_vertex = {}
        for i, iv in enumerate(self.jaxpr.invars):
            self._vertex_of[id(iv)] = -(i + 1)
        for pos, eqn in enumerate(self.jaxpr.eqns, start=1):
            self._vars_of_vertex[pos] = tuple(eqn.outvars)
            for ov in eqn.outvars:
                self._vertex_of[id(ov)] = pos

        prim_env, g0, _tg0, vo = _build_graph(
            self.jaxpr, self.args, closed.literals, self.argnums)
        self._prim_env = prim_env
        self._g0 = g0
        self._vo = vo

        self._X_list = [iv for i, iv in enumerate(self.jaxpr.invars)
                        if i in self.argnums]
        self.X = set(self._X_list)
        self.Y = set(self.jaxpr.outvars)

        # jacve's action space, mirroring graphax.core._checkify_order: an eqn
        # is eliminable iff any outvar is not a pure output (intermediate, or
        # in vo_vertices = intermediate AND output).
        outvars = self.jaxpr.outvars

        def _should(eqn):
            return any((ov not in outvars) or (ov in vo)
                       for ov in eqn.outvars if isinstance(ov, _Var))

        self.jacve_vertices = frozenset(
            i for i, eqn in enumerate(self.jaxpr.eqns, start=1) if _should(eqn))

        self.reset()

    # -- construction / reset ------------------------------------------------
    def reset(self) -> "ElimState":
        self.src, self.snk, self.typ, self.cf = {}, {}, {}, {}
        self.succ, self.pred = {}, {}
        self._nid = 0
        self.steps = 0
        self.history: list = []

        # DETERMINISTIC creation order (cross-process replayability): X nodes
        # in invar order, Y nodes in outvar order, Z nodes in _build_graph
        # insertion (eqn) order. dual_face_v2 iterated the X/Y *sets* here,
        # which is id-hash ordered and NOT stable across processes.
        self._v_done: set = set()          # vids consumed by a VERTEX action
        xv = {x: self._new(x, x, 'X') for x in self._X_list}
        yv = {}
        for y in self.jaxpr.outvars:
            if y not in yv:
                yv[y] = self._new(y, y, 'Y')
        Z = {}
        for a in self._g0:
            for b in self._g0[a]:
                t = _force(self._g0[a][b])
                if t is None:
                    continue
                Z[(a, b)] = self._new(a, b, 'Z', t)
        for (a, b), v in Z.items():                      # E_X / E_Y boundary links
            if a in self.X:
                self._link(xv[a], v)
            if b in self.Y:
                self._link(v, yv[b])
        for (a, b), u in Z.items():                      # E_Z faces: (a,b)->(b,c)
            for (b2, c) in Z:
                if b2 == b:
                    self._link(u, Z[(b2, c)])
        # jacve's graph is NOT argnums-pruned, so its order space contains
        # vertices with no live faces here (unreachable from the argnums
        # inputs); eliminating those in jacve is a legal no-op. Record which
        # eliminable vertices are actually present so face-driven elimination
        # can be told apart from never-was-there.
        self._reset_present = frozenset(self._present_vids())
        return self.state()

    # -- primitives -----------------------------------------------------------
    def _new(self, s, k, t, c=None) -> int:
        v = self._nid
        self._nid += 1
        self.src[v], self.snk[v], self.typ[v], self.cf[v] = s, k, t, c
        self.succ[v], self.pred[v] = set(), set()
        return v

    def _link(self, a, b):
        self.succ[a].add(b)
        self.pred[b].add(a)

    def _drop(self, v):
        for p in list(self.pred[v]):
            self.succ[p].discard(v)
        for s in list(self.succ[v]):
            self.pred[s].discard(v)
        for d in (self.src, self.snk, self.typ, self.cf, self.succ, self.pred):
            del d[v]

    def _vid(self, var) -> int:
        return self._vertex_of.get(id(var), 0)

    # -- legality (regenerated from the live graph every step) ----------------
    def faces(self) -> list:
        """All live Z->Z faces (u, w), canonically sorted by node id."""
        return sorted((u, w) for u in self.succ if self.typ.get(u) == 'Z'
                      for w in self.succ[u] if self.typ.get(w) == 'Z')

    def legal_vertices(self) -> Tuple[int, ...]:
        """Legal VERTEX actions == jacve legality EXACTLY: every jacve-
        eliminable vertex that has not been eliminated yet (by a VERTEX
        action, or -- in mixed mode -- fully by FACE actions). A legal vertex
        with no live faces (argnums-pruned, unreachable from the diff inputs)
        eliminates as a 0-face no-op, exactly like jacve's step on it."""
        gone = self.eliminated_vertices()
        return tuple(sorted(j for j in self.jacve_vertices if j not in gone))

    def absorb_target(self, u: int, w: int) -> Optional[int]:
        """The existing node the face (u, w) would absorb into, or None (fill).
        Read from the state -- exists t: src==i, snk==k, P(t)==P(u), S(t)==S(w)."""
        i, k = self.src[u], self.snk[w]
        for cand in list(self.succ):
            if cand == u or cand == w or self.typ.get(cand) != 'Z':
                continue
            if (self.src[cand] == i and self.snk[cand] == k
                    and self.pred[cand] == self.pred[u]
                    and self.succ[cand] == self.succ[w]):
                return cand
        return None

    @property
    def done(self) -> bool:
        """Terminal iff the line dag is bipartite (no Z->Z edge left)."""
        for u in self.succ:
            if self.typ.get(u) != 'Z':
                continue
            for w in self.succ[u]:
                if self.typ.get(w) == 'Z':
                    return False
        return True

    def _present_vids(self) -> set:
        """Vertex ids whose var appears on BOTH sides (as some live Z node's
        src AND some live Z node's snk) -- i.e. a face with that middle exists
        or can still reappear."""
        src_vids, snk_vids = set(), set()
        for v in self.succ:
            if self.typ.get(v) == 'Z':
                src_vids.add(self._vid(self.src[v]))
                snk_vids.add(self._vid(self.snk[v]))
        return src_vids & snk_vids

    def eliminated_vertices(self) -> frozenset:
        """Vertices consumed by a VERTEX action, plus vertices that were
        present at reset but have been fully eliminated by FACE actions.
        (Vertices absent from reset on -- argnums-pruned -- stay LEGAL no-ops
        until used, mirroring jacve's unpruned order space.)"""
        present = self._present_vids()
        return frozenset(self._v_done | {
            j for j in self.jacve_vertices
            if j in self._reset_present and j not in present})

    # -- dynamics --------------------------------------------------------------
    def _eliminate_face(self, u: int, w: int):
        i, k = self.src[u], self.snk[w]
        # 1. absorb-or-fill (state-determined)
        t = self.absorb_target(u, w)
        if t is None:
            t = self._new(i, k, 'Z', None)
            for p in list(self.pred[u]):
                self._link(p, t)                          # up-links P(u)
            for s in list(self.succ[w]):
                self._link(t, s)                          # down-links S(w)
        # 2. fma
        if not self.symbolic:
            prod = _contract(self.cf[w], self.cf[u])
            self.cf[t] = _merge_cf(self.cf[t], prod)
        # 3. remove the face edge
        self.succ[u].discard(w)
        self.pred[w].discard(u)
        # 4. cleanup u: isolated-drop or same-endpoints/neighbourhood merge
        if u in self.succ:
            if not self.succ[u]:
                self._drop(u)
            else:
                for up in [x for x in list(self.succ) if x != u
                           and self.typ.get(x) == 'Z'
                           and self.src.get(x) == self.src[u]
                           and self.snk.get(x) == self.snk[u]
                           and self.pred.get(x) == self.pred[u]
                           and self.succ.get(x) == self.succ[u]]:
                    if not self.symbolic:
                        self.cf[u] = _merge_cf(self.cf[u], self.cf[up])
                    self._drop(up)
                    break
        # 5. cleanup w
        if w in self.pred:
            if not self.pred[w]:
                self._drop(w)
            else:
                for wp in [x for x in list(self.pred) if x != w
                           and self.typ.get(x) == 'Z'
                           and self.src.get(x) == self.src[w]
                           and self.snk.get(x) == self.snk[w]
                           and self.pred.get(x) == self.pred[w]
                           and self.succ.get(x) == self.succ[w]]:
                    if not self.symbolic:
                        self.cf[w] = _merge_cf(self.cf[w], self.cf[wp])
                    self._drop(wp)
                    break
        self.steps += 1

    def _eliminate_vertex(self, vid: int) -> int:
        """VERTEX macro: eliminate ALL faces whose middle var is an outvar of
        graphax vertex ``vid`` -- equals jacve's single-vertex elimination
        (a vertex elimination is a sequence of face eliminations; fill nodes
        created here never have middle ``vid``, so this terminates)."""
        mids = {id(v) for v in self._vars_of_vertex.get(vid, ())}
        n = 0
        while True:
            fs = [f for f in self.faces() if id(self.snk[f[0]]) in mids]
            if not fs:
                break
            self._eliminate_face(*fs[0])
            n += 1
        return n

    def apply(self, action) -> dict:
        """Apply one action: ("F", u, w) | ("V", j). Raises ValueError on an
        illegal action. Returns an info dict; appends to ``history``."""
        action = tuple(action)
        kind = action[0]
        if kind in ("F", "FACE"):
            if self.vertex_only:
                raise ValueError("vertex_only env: FACE actions are disabled")
            u, w = int(action[1]), int(action[2])
            if (self.typ.get(u) != 'Z' or self.typ.get(w) != 'Z'
                    or w not in self.succ.get(u, ())):
                raise ValueError(f"illegal face action {(u, w)}: not a live Z->Z face")
            self._eliminate_face(u, w)
            self.history.append(("F", u, w))
            return {"kind": "F", "eliminations": 1, "steps": self.steps}
        if kind in ("V", "VERTEX"):
            j = int(action[1])
            if j not in self.jacve_vertices:
                raise ValueError(f"vertex {j} is not jacve-eliminable")
            if j in self.eliminated_vertices():
                raise ValueError(f"vertex {j} is already eliminated")
            n = self._eliminate_vertex(j)
            self._v_done.add(j)
            self.history.append(("V", j))
            return {"kind": "V", "eliminations": n, "steps": self.steps}
        raise ValueError(f"unknown action kind {kind!r}")

    def step(self, action):
        """Gym-ish step: returns (state, reward, done, info). The M0 reward is
        0.0 -- real objectives come from the isolated measurement worker."""
        info = self.apply(action)
        st = self.state()
        return st, 0.0, st.done, info

    # -- readout ----------------------------------------------------------------
    def jacobian(self, sparse_representation: bool = False) -> list:
        """Accumulated Jacobian blocks ordered outvars x argnums-invars
        (jacve/jax.jacrev block order); None for a missing (zero) block.
        Duplicate surviving (x, y) nodes are SUMMED (naumann_dense
        collect(mode="sum")); dual_face_v2 overwrote, which drops path mass."""
        if self.symbolic:
            raise RuntimeError("symbolic env carries no coefficient values")
        J = {}
        for v in list(self.succ):
            if (self.typ.get(v) == 'Z' and self.src[v] in self.X
                    and self.snk[v] in self.Y and self.cf[v] is not None):
                key = (self.src[v], self.snk[v])
                J[key] = _merge_cf(J.get(key), self.cf[v])
        if not sparse_representation:
            J = {k: _drain_transforms(v).dense() for k, v in J.items()}
        else:
            J = {k: _drain_transforms(v) for k, v in J.items()}
        return [J.get((iv, ov)) for ov in self.jaxpr.outvars for iv in self._X_list]

    def primal_outputs(self) -> list:
        """Primal outputs of fn (value-and-jacobian, graphax has_aux sense)."""
        return [self._prim_env[v] for v in self.jaxpr.outvars]

    def state(self) -> ElimState:
        nodes = []
        for v in sorted(self.typ):
            shape, dtype, od, pd = _st_meta(self.cf[v])
            nodes.append(NodeMeta(
                nid=v, kind=self.typ[v],
                src_vertex=self._vid(self.src[v]),
                snk_vertex=self._vid(self.snk[v]),
                val_shape=shape, val_dtype=dtype,
                out_dims=od, primal_dims=pd))
        edges = tuple(sorted((a, b) for a in self.succ for b in self.succ[a]))
        faces_meta: Tuple[FaceMeta, ...] = ()
        if not self.vertex_only:
            fm = []
            for (u, w) in self.faces():
                t = self.absorb_target(u, w)
                fm.append(FaceMeta(
                    u=u, w=w,
                    i_vertex=self._vid(self.src[u]),
                    j_vertex=self._vid(self.snk[u]),
                    k_vertex=self._vid(self.snk[w]),
                    absorb=t is not None, target=t))
            faces_meta = tuple(fm)
        return ElimState(
            nodes=tuple(nodes), edges=edges, faces=faces_meta,
            legal_vertices=self.legal_vertices(),
            eliminated=self.eliminated_vertices(),
            vertex_only=self.vertex_only, done=self.done, steps=self.steps)


# ---------------------------------------------------------------------------
# built-in policies
# ---------------------------------------------------------------------------
def rev_policy(env: ElimEnv):
    """Reverse-ish face policy (dual_face_v2 "rev"): eliminate the face whose
    MIDDLE var is latest in topo order; ties break to the first face in the
    canonical (sorted) enumeration. Structural only -- safe under tracing."""
    F = env.faces()
    if not F:
        raise RuntimeError("rev_policy called on a terminal env")
    u, w = max(F, key=lambda uw: env.rank.get(id(env.snk[uw[0]]), 0))
    return ("F", u, w)
