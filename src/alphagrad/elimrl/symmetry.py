"""Symmetry machinery for elimination orders on jaxpr computation graphs (M1).

Two exploitable symmetries:

1. **Mazurkiewicz traces / Foata normal form** (`trace_key`). Two adjacent
   eliminations u, v commute iff they are *path-independent* in the CURRENT
   graph (no directed path between them, fill edges included). On the
   connectivity level, eliminating a vertex w adds exactly pred(w) x succ(w)
   fill edges, i.e. every fill edge corresponds to an original path and every
   original path contracts to a current path. Hence reachability among the
   REMAINING vertices of the evolving graph is identical to reachability in
   the ORIGINAL graph -- the path-independence relation is *static* (it is
   incomparability in the original DAG's reachability partial order). That
   makes the commutation relation a genuine Mazurkiewicz independence and the
   Foata normal form a sound canonical representative. `independent_at`
   implements the literal evolving-graph definition; the property test asserts
   it coincides with the static relation.

   Consequences used by the tests:
   - swapping adjacent independent eliminations preserves `trace_key`,
     the resulting graph, and the Markowitz cost profile (as a multiset);
   - swapping adjacent *dependent* eliminations flips the orientation of a
     dependent pair, hence changes the key (Foata NF is a complete trace
     invariant);
   - two orders with equal key produce the identical graph state after any
     equal number of Foata blocks, and identical final graphs.

2. **Coloured automorphism orbits** (`orbits`). The computation DAG is
   coloured by structural features (primitive name, significant params such
   as dot_general dimension numbers, output avals, literal operands, role
   flags; inputs by shape/dtype/differentiability). Orbits of the coloured
   automorphism group give exchangeable vertices: an automorphism applied to
   a trajectory (`augment`) yields another legal trajectory of identical
   structural cost. Two backends:
   - pynauty (exact, preferred) when importable;
   - 1-WL colour refinement + explicit per-pair automorphism verification
     (WL classes are unions of orbits; only verified pairs are merged, so the
     reported orbits are sound -- classes whose verification exceeds the
     budget are reported as `unverified` WL classes, not as orbits).

Pure graph algorithms; jax/graphax are imported lazily and only by the
builders (`build_elim_graph`, `build_example_graph`).
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ElimGraph",
    "trace_key",
    "independent_at",
    "foata_normal_form",
    "eliminate_vertex",
    "graph_after",
    "markowitz_profile",
    "orbits",
    "OrbitResult",
    "augment",
    "is_automorphism",
    "wl_refine",
    "find_automorphism",
    "build_elim_graph",
    "build_example_graph",
]


# =====================================================================
# Graph container
# =====================================================================

class ElimGraph:
    """Connectivity view of a graphax elimination graph.

    Nodes are ints: equations are 1..N (1-based eqn position, graphax's vertex
    ids), graph inputs are negative (-(invar_index+1), constvars continue the
    negative range) -- the same convention as graphax's per-path transform ids.

    `succ`/`pred` cover every live node. `eliminable` is the sorted tuple of
    vertex ids a full elimination order permutes.
    """

    def __init__(self,
                 succ: Mapping[int, Iterable[int]],
                 eliminable: Sequence[int],
                 colors: Optional[Mapping[int, object]] = None,
                 in_positions: Optional[Mapping[int, Sequence[Tuple[int, int]]]] = None,
                 position_sensitive: Optional[Iterable[int]] = None,
                 meta: Optional[dict] = None):
        nodes = set(succ.keys())
        for vs in succ.values():
            nodes.update(vs)
        self.nodes: Tuple[int, ...] = tuple(sorted(nodes))
        self.succ: Dict[int, FrozenSet[int]] = {
            n: frozenset(succ.get(n, ())) for n in self.nodes}
        pred: Dict[int, set] = {n: set() for n in self.nodes}
        for u, vs in self.succ.items():
            for v in vs:
                pred[v].add(u)
        self.pred: Dict[int, FrozenSet[int]] = {
            n: frozenset(ps) for n, ps in pred.items()}
        self.eliminable: Tuple[int, ...] = tuple(sorted(eliminable))
        self.colors: Dict[int, object] = dict(colors or {})
        # eqn vertex -> ((invar position, source node id), ...) for edges that
        # exist; used to encode argument-position sensitivity via port nodes.
        self.in_positions: Dict[int, Tuple[Tuple[int, int], ...]] = {
            k: tuple(v) for k, v in (in_positions or {}).items()}
        # vertices whose primitive is argument-position sensitive
        self.position_sensitive: FrozenSet[int] = frozenset(position_sensitive or ())
        self.meta: dict = dict(meta or {})
        self._desc: Optional[Dict[int, FrozenSet[int]]] = None

    @classmethod
    def from_edges(cls, edges: Iterable[Tuple[int, int]],
                   eliminable: Optional[Sequence[int]] = None,
                   colors: Optional[Mapping[int, object]] = None) -> "ElimGraph":
        succ: Dict[int, set] = defaultdict(set)
        nodes = set()
        for u, v in edges:
            succ[u].add(v)
            nodes.update((u, v))
        if eliminable is None:
            # default: internal nodes (those with both preds and succs)
            haspred = {v for vs in succ.values() for v in vs}
            eliminable = sorted(n for n in nodes if n in succ and n in haspred)
        return cls(succ, eliminable, colors=colors)

    # -- reachability -------------------------------------------------
    def descendants(self) -> Dict[int, FrozenSet[int]]:
        """node -> frozenset of strict descendants in the ORIGINAL graph."""
        if self._desc is None:
            desc: Dict[int, FrozenSet[int]] = {}
            # reverse topological order (DAG)
            for n in self._topo_order()[::-1]:
                acc: set = set()
                for s in self.succ[n]:
                    acc.add(s)
                    acc |= desc[s]
                desc[n] = frozenset(acc)
            self._desc = desc
        return self._desc

    def _topo_order(self) -> List[int]:
        indeg = {n: len(self.pred[n]) for n in self.nodes}
        stack = [n for n in self.nodes if indeg[n] == 0]
        out: List[int] = []
        while stack:
            n = stack.pop()
            out.append(n)
            for s in self.succ[n]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    stack.append(s)
        if len(out) != len(self.nodes):
            raise ValueError("graph has a cycle; not a DAG")
        return out

    def dependent(self, u: int, v: int) -> bool:
        """Path-dependence in the original DAG (== in any evolved state)."""
        d = self.descendants()
        return v in d[u] or u in d[v]

    # -- mutable copies for elimination replay -------------------------
    def mutable(self) -> Tuple[Dict[int, set], Dict[int, set]]:
        return ({n: set(vs) for n, vs in self.succ.items()},
                {n: set(ps) for n, ps in self.pred.items()})


# =====================================================================
# Elimination engine (connectivity level)
# =====================================================================

def eliminate_vertex(succ: Dict[int, set], pred: Dict[int, set], v: int) -> int:
    """Eliminate v in place: connect pred(v) x succ(v), remove v.

    Returns the Markowitz product |pred(v)| * |succ(v)| at elimination time.
    """
    ps, ss = pred[v], succ[v]
    mark = len(ps) * len(ss)
    for p in ps:
        succ[p].discard(v)
        succ[p].update(ss)
    for s in ss:
        pred[s].discard(v)
        pred[s].update(ps)
    del succ[v]
    del pred[v]
    return mark


def graph_after(graph: ElimGraph, order: Sequence[int]) -> FrozenSet[Tuple[int, int]]:
    """Edge set of the graph state after eliminating `order` (a prefix is fine)."""
    succ, pred = graph.mutable()
    for v in order:
        eliminate_vertex(succ, pred, v)
    return frozenset((u, w) for u, vs in succ.items() for w in vs)


def markowitz_profile(graph: ElimGraph, order: Sequence[int]) -> Tuple[int, ...]:
    """Sequence of Markowitz products along the trajectory (cheap cost proxy)."""
    succ, pred = graph.mutable()
    return tuple(eliminate_vertex(succ, pred, v) for v in order)


def _reachable(succ: Mapping[int, set], src: int, dst: int) -> bool:
    if src == dst:
        return True
    seen = {src}
    stack = [src]
    while stack:
        n = stack.pop()
        for s in succ.get(n, ()):  # noqa: B909
            if s == dst:
                return True
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return False


def independent_at(graph: ElimGraph, order: Sequence[int], i: int) -> bool:
    """Literal evolving-graph independence of order[i], order[i+1].

    Replays order[:i] (fill edges included) and checks that no directed path
    connects order[i] and order[i+1] in the CURRENT graph state. Equivalent to
    static incomparability (see module docstring); kept as the reference
    semantics and exercised against the static relation by the tests.
    """
    u, w = order[i], order[i + 1]
    succ, pred = graph.mutable()
    for v in order[:i]:
        eliminate_vertex(succ, pred, v)
    return not (_reachable(succ, u, w) or _reachable(succ, w, u))


# =====================================================================
# Foata normal form / trace key
# =====================================================================

def foata_normal_form(order: Sequence[int],
                      dependent: Callable[[int, int], bool]) -> Tuple[Tuple[int, ...], ...]:
    """Foata NF of `order` under the (static) dependence relation.

    Greedy level assignment: level(v) = 1 + max level of an earlier dependent
    letter (0 if none). Blocks are sets of pairwise-independent letters,
    canonically sorted. Complete invariant of the trace class. O(n^2).
    """
    level: Dict[int, int] = {}
    for v in order:
        lv = 0
        for u, lu in level.items():
            if lu > lv and dependent(u, v):
                lv = lu
        level[v] = lv + 1
    if not level:
        return ()
    blocks: List[List[int]] = [[] for _ in range(max(level.values()))]
    for v in order:
        blocks[level[v] - 1].append(v)
    return tuple(tuple(sorted(b)) for b in blocks)


def trace_key(order: Sequence[int], graph: ElimGraph) -> Tuple[Tuple[int, ...], ...]:
    """Canonical trace representative of an elimination sequence.

    Two orders that differ by swaps of adjacent path-independent eliminations
    map to the same key; orders differing in the relative orientation of any
    path-dependent pair map to different keys. Works for full orders and
    prefixes alike.
    """
    if len(set(order)) != len(order):
        raise ValueError("elimination order contains repeated vertices")
    elim = set(graph.eliminable)
    for v in order:
        if v not in elim:
            raise ValueError(f"vertex {v} is not eliminable")
    return foata_normal_form(order, graph.dependent)


# =====================================================================
# WL refinement, automorphism search, orbits
# =====================================================================

def _canon_color(c: object) -> str:
    try:
        hash(c)
        return repr(c)
    except TypeError:
        return repr(c)


def wl_refine(nodes: Sequence[int],
              succ: Mapping[int, Iterable[int]],
              pred: Mapping[int, Iterable[int]],
              init: Mapping[int, int],
              max_rounds: int = 0) -> Dict[int, int]:
    """1-WL colour refinement on a directed graph. Returns stable colour ids.

    Signature of a node = (own colour, sorted multiset of in-neighbour
    colours, sorted multiset of out-neighbour colours).
    """
    color = dict(init)
    n = len(nodes)
    rounds = max_rounds or (n + 1)
    for _ in range(rounds):
        sigs = {}
        for v in nodes:
            sigs[v] = (color[v],
                       tuple(sorted(color[u] for u in pred.get(v, ()))),
                       tuple(sorted(color[u] for u in succ.get(v, ()))))
        remap: Dict[tuple, int] = {}
        new = {}
        for v in nodes:
            new[v] = remap.setdefault(sigs[v], len(remap))
        if len(set(new.values())) == len(set(color.values())):
            color = new
            break
        color = new
    return color


class _Budget:
    def __init__(self, n: int):
        self.left = n

    def tick(self) -> bool:
        self.left -= 1
        return self.left >= 0


def find_automorphism(nodes: Sequence[int],
                      succ: Mapping[int, FrozenSet[int]],
                      pred: Mapping[int, FrozenSet[int]],
                      colors: Mapping[int, int],
                      pin: Optional[Tuple[int, int]] = None,
                      budget: int = 200_000) -> Optional[Dict[int, int]]:
    """Backtracking search for a colour-preserving digraph automorphism.

    `pin=(a, b)` forces a -> b. Returns a full permutation dict, or None if
    none exists (exact if the search completes) or the budget is exhausted
    (raises TimeoutError so callers can distinguish "no" from "unknown").
    """
    base = wl_refine(nodes, succ, pred, colors)
    if pin is not None:
        a, b = pin
        if base[a] != base[b]:
            return None
        # individualize the pin and re-refine: prunes the candidate sets hard
        ind = dict(base)
        mx = max(ind.values()) + 1
        ind[a] = mx  # a gets a unique colour...
        ref_a = wl_refine(nodes, succ, pred, ind)
        ind2 = dict(base)
        ind2[b] = mx
        ref_b = wl_refine(nodes, succ, pred, ind2)
        # the two individualized refinements must have matching class
        # histograms, else a->b is impossible
        if sorted(Counter(ref_a.values()).values()) != sorted(Counter(ref_b.values()).values()):
            return None
        cls_a = defaultdict(list)
        cls_b = defaultdict(list)
        # classes are matched via the ORIGINAL base colour signature of their
        # members combined with refined-class sizes; simplest sound approach:
        # candidates(v) = nodes w with base[w]==base[v]; the pin constraint is
        # enforced by assignment, refinement only orders the search.
        del cls_a, cls_b

    by_color: Dict[int, List[int]] = defaultdict(list)
    for v in nodes:
        by_color[base[v]].append(v)
    cand = {v: by_color[base[v]] for v in nodes}

    order = sorted(nodes, key=lambda v: (len(cand[v]),
                                         -(len(succ.get(v, ())) + len(pred.get(v, ())))))
    if pin is not None:
        order = [pin[0]] + [v for v in order if v != pin[0]]

    mapping: Dict[int, int] = {}
    used: set = set()
    bud = _Budget(budget)

    def consistent(v: int, w: int) -> bool:
        # edges to already-mapped nodes must be preserved in both directions
        for u in succ.get(v, ()):  # noqa: B909
            if u in mapping and mapping[u] not in succ.get(w, ()):
                return False
        for u in pred.get(v, ()):  # noqa: B909
            if u in mapping and mapping[u] not in pred.get(w, ()):
                return False
        for u, mu in mapping.items():
            if v in succ.get(u, ()) and w not in succ.get(mu, ()):
                return False
            if v in pred.get(u, ()) and w not in pred.get(mu, ()):
                return False
        return True

    def bt(idx: int) -> bool:
        if idx == len(order):
            return True
        v = order[idx]
        opts = [pin[1]] if (pin is not None and v == pin[0]) else cand[v]
        for w in opts:
            if w in used:
                continue
            if len(succ.get(v, ())) != len(succ.get(w, ())):
                continue
            if len(pred.get(v, ())) != len(pred.get(w, ())):
                continue
            if not bud.tick():
                raise TimeoutError("automorphism search budget exhausted")
            if not consistent(v, w):
                continue
            mapping[v] = w
            used.add(w)
            if bt(idx + 1):
                return True
            del mapping[v]
            used.discard(w)
        return False

    ok = bt(0)
    return dict(mapping) if ok else None


@dataclass
class OrbitResult:
    orbits: List[FrozenSet[int]]
    method: str                       # 'pynauty' | 'wl+verify'
    exact: bool                       # True when orbits are proven orbits
    unverified: List[FrozenSet[int]] = field(default_factory=list)
    automorphisms: List[Dict[int, int]] = field(default_factory=list)
    group_size: Optional[float] = None

    def sizes(self) -> List[int]:
        return sorted((len(o) for o in self.orbits), reverse=True)

    def histogram(self) -> Dict[int, int]:
        return dict(Counter(len(o) for o in self.orbits))


# argument-position-INsensitive primitives: their in-edges need no port nodes
_COMMUTATIVE_PRIMS = {
    "add", "add_any", "mul", "max", "min", "and", "or", "xor", "eq", "ne",
}


def _automorphism_encoding(graph: ElimGraph, colored: bool):
    """(nodes, succ, pred, colors) with port nodes encoding argument position.

    Port nodes make argument order part of the digraph structure for
    position-sensitive primitives (sub, div, dot_general, ...), so a digraph
    automorphism can never exchange lhs/rhs of a non-commutative op. Only used
    for the colored encoding; the uncolored variant is the bare digraph.
    """
    succ: Dict[int, set] = {n: set(graph.succ[n]) for n in graph.nodes}
    colors: Dict[object, object] = {}
    if not colored:
        base = {n: 0 for n in graph.nodes}
        return list(graph.nodes), {n: frozenset(s) for n, s in succ.items()}, \
            _pred_of(succ), base
    for n in graph.nodes:
        colors[n] = ("node", _canon_color(graph.colors.get(n, ("?",))))
    next_port = max(graph.nodes) + 1 if graph.nodes else 1
    for v in sorted(graph.position_sensitive):
        for pos, src in graph.in_positions.get(v, ()):
            if src not in graph.succ or v not in graph.succ[src]:
                continue
            port = next_port
            next_port += 1
            succ[src].discard(v)
            succ[src] = succ[src] | {port}
            succ[port] = {v}
            colors[port] = ("port", pos)
    nodes = sorted(succ.keys() | {w for vs in succ.values() for w in vs})
    for n in nodes:
        succ.setdefault(n, set())
    fsucc = {n: frozenset(s) for n, s in succ.items()}
    cint: Dict[int, int] = {}
    remap: Dict[object, int] = {}
    for n in nodes:
        cint[n] = remap.setdefault(_canon_color(colors[n]), len(remap))
    return nodes, fsucc, _pred_of(succ), cint


def _pred_of(succ: Mapping[int, Iterable[int]]) -> Dict[int, FrozenSet[int]]:
    pred: Dict[int, set] = defaultdict(set)
    for u, vs in succ.items():
        pred[u]  # touch
        for v in vs:
            pred[v].add(u)
    return {n: frozenset(ps) for n, ps in pred.items()}


def _try_pynauty(nodes, succ, colors_int):
    try:
        import pynauty  # type: ignore
    except Exception:
        return None
    idx = {n: i for i, n in enumerate(nodes)}
    adj = {idx[u]: [idx[v] for v in vs] for u, vs in succ.items()}
    cells: Dict[int, set] = defaultdict(set)
    for n in nodes:
        cells[colors_int[n]].add(idx[n])
    coloring = [cells[c] for c in sorted(cells)]
    g = pynauty.Graph(number_of_vertices=len(nodes), directed=True,
                      adjacency_dict=adj, vertex_coloring=coloring)
    gens, grpsize1, grpsize2, orbit_ids, numorbits = pynauty.autgrp(g)
    rev = {i: n for n, i in idx.items()}
    by_orbit: Dict[int, set] = defaultdict(set)
    for i, o in enumerate(orbit_ids):
        by_orbit[o].add(rev[i])
    autos = [{rev[i]: rev[p[i]] for i in range(len(nodes))} for p in gens]
    return (sorted((frozenset(s) for s in by_orbit.values()), key=lambda s: (-len(s), min(s))),
            autos, grpsize1 * (10 ** grpsize2))


def orbits(graph: ElimGraph, colored: bool = True, method: str = "auto",
           budget: int = 200_000, restrict_to_real: bool = True) -> OrbitResult:
    """Automorphism orbits of the (coloured) computation digraph.

    method: 'auto' (pynauty if importable, else WL+verify), 'pynauty', 'wl'.
    Orbits/automorphisms are restricted to the real graph nodes (port nodes
    used for the position encoding are dropped).
    """
    nodes, succ, pred, cint = _automorphism_encoding(graph, colored)
    real = set(graph.nodes)

    if method in ("auto", "pynauty"):
        res = _try_pynauty(nodes, succ, cint)
        if res is not None:
            orbs, autos, gsize = res
            if restrict_to_real:
                orbs = [frozenset(o & real) for o in orbs if o & real]
                autos = [{k: v for k, v in a.items() if k in real} for a in autos]
            return OrbitResult(orbits=orbs, method="pynauty", exact=True,
                               automorphisms=autos, group_size=gsize)
        if method == "pynauty":
            raise ImportError("pynauty requested but not importable")

    # ---- WL + explicit verification fallback -------------------------
    refined = wl_refine(nodes, succ, pred, cint)
    classes: Dict[int, List[int]] = defaultdict(list)
    for v in nodes:
        classes[refined[v]].append(v)

    parent: Dict[int, int] = {v: v for v in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    autos: List[Dict[int, int]] = []
    unverified: List[FrozenSet[int]] = []
    for cls in classes.values():
        if len(cls) == 1:
            continue
        members = sorted(cls)
        pending = list(members[1:])
        reps = [members[0]]
        timed_out = False
        while pending:
            u = pending.pop(0)
            placed = False
            for r in reps:
                try:
                    perm = find_automorphism(nodes, succ, pred, cint,
                                             pin=(r, u), budget=budget)
                except TimeoutError:
                    timed_out = True
                    perm = None
                if perm is not None:
                    union(u, r)
                    autos.append(perm)
                    placed = True
                    break
            if not placed:
                reps.append(u)  # new candidate orbit representative
        if timed_out:
            unverified.append(frozenset(members))

    by_root: Dict[int, set] = defaultdict(set)
    for v in nodes:
        by_root[find(v)].add(v)
    orbs = sorted((frozenset(s) for s in by_root.values()),
                  key=lambda s: (-len(s), min(s)))
    if restrict_to_real:
        orbs = [frozenset(o & real) for o in orbs if o & real]
        autos = [{k: v for k, v in a.items() if k in real} for a in autos]
    return OrbitResult(orbits=orbs, method="wl+verify", exact=not unverified,
                       unverified=unverified, automorphisms=autos)


# =====================================================================
# Trajectory augmentation
# =====================================================================

def is_automorphism(graph: ElimGraph, perm: Mapping[int, int],
                    colored: bool = True) -> bool:
    """Check perm is a colour- and edge-preserving bijection of graph.nodes."""
    nodes = set(graph.nodes)
    img = set(perm.get(n, n) for n in nodes)
    if img != nodes:
        return False
    for u in nodes:
        pu = perm.get(u, u)
        if colored and _canon_color(graph.colors.get(u)) != _canon_color(graph.colors.get(pu)):
            return False
        if {perm.get(v, v) for v in graph.succ[u]} != set(graph.succ[pu]):
            return False
    return True


def augment(trajectory: Sequence[int], automorphism: Mapping[int, int]) -> List[int]:
    """Relabel a trajectory by an automorphism: v_i -> pi(v_i).

    If pi is an automorphism of the elimination graph fixing the eliminable
    set, the result is a legal elimination order with an identical structural
    cost profile (sequence of Markowitz products), because pi is a bijection
    that commutes with the elimination operation on the connectivity level.
    """
    return [automorphism.get(v, v) for v in trajectory]


# =====================================================================
# Builders (lazy jax/graphax imports)
# =====================================================================

def _params_key(params: dict) -> tuple:
    out = []
    for k in sorted(params):
        v = params[k]
        try:
            out.append((k, repr(v)))
        except Exception:
            out.append((k, f"<{type(v).__name__}>"))
    return tuple(out)


def build_elim_graph(fn, args: Sequence, argnums: Sequence[int]) -> ElimGraph:
    """Trace fn once and extract the graphax elimination graph connectivity.

    Mirrors graphax's own pipeline (`jacve` -> `_inline_call_primitives` ->
    `_build_graph(argnums)` -> `_prune_graph`), then drops the partial
    Jacobians and keeps pure connectivity + structural colours. Read-only use
    of graphax; nothing is modified.
    """
    import jax
    from jax._src import core as jcore
    from graphax.core import (_build_graph, _inline_call_primitives,
                              _prune_graph, prune_enabled)

    closed = jax.make_jaxpr(fn)(*args)
    jaxpr, consts = _inline_call_primitives(closed.jaxpr, closed.literals)
    _env, g, tg, vo_vertices = _build_graph(jaxpr, list(args), consts,
                                            tuple(argnums))
    if prune_enabled():
        _prune_graph(g, tg, jaxpr, tuple(argnums))

    var2id: Dict[object, int] = {}
    for i, eqn in enumerate(jaxpr.eqns, start=1):
        for ov in eqn.outvars:
            if isinstance(ov, jcore.Var):
                var2id[ov] = i
    for j, iv in enumerate(jaxpr.invars):
        var2id.setdefault(iv, -(j + 1))
    for k, cv in enumerate(jaxpr.constvars):
        var2id.setdefault(cv, -(len(jaxpr.invars) + k + 1))

    succ: Dict[int, set] = defaultdict(set)
    for src, inner in g.items():
        if src not in var2id:
            continue
        for dst in inner:
            if dst in var2id and inner[dst] is not None:
                a, b = var2id[src], var2id[dst]
                if a != b:
                    succ[a].add(b)

    live = set(succ.keys()) | {v for vs in succ.values() for v in vs}
    outset = set(jaxpr.outvars)

    def _should_eliminate(eqn):
        return any(ov not in outset or ov in vo_vertices
                   for ov in eqn.outvars if isinstance(ov, jcore.Var))

    eliminable = [i for i, eqn in enumerate(jaxpr.eqns, start=1)
                  if _should_eliminate(eqn) and i in live]

    colors: Dict[int, object] = {}
    in_positions: Dict[int, List[Tuple[int, int]]] = {}
    pos_sensitive: List[int] = []
    argset = set(argnums)
    for i, eqn in enumerate(jaxpr.eqns, start=1):
        if i not in live:
            continue
        lits = tuple((p, repr(getattr(v, "val", None))[:48])
                     for p, v in enumerate(eqn.invars)
                     if isinstance(v, jcore.Literal))
        outav = tuple((tuple(ov.aval.shape), str(ov.aval.dtype))
                      for ov in eqn.outvars if isinstance(ov, jcore.Var))
        is_out = any(ov in outset for ov in eqn.outvars)
        colors[i] = ("eqn", eqn.primitive.name, _params_key(eqn.params),
                     outav, lits, is_out)
        pos = []
        for p, v in enumerate(eqn.invars):
            if isinstance(v, jcore.Var) and v in var2id and var2id[v] in live:
                if i in succ.get(var2id[v], set()):
                    pos.append((p, var2id[v]))
        in_positions[i] = pos
        if eqn.primitive.name not in _COMMUTATIVE_PRIMS and len(pos) > 1:
            pos_sensitive.append(i)
    for j, iv in enumerate(jaxpr.invars):
        nid = -(j + 1)
        if nid in live:
            colors[nid] = ("in", tuple(iv.aval.shape), str(iv.aval.dtype),
                           j in argset)
    for k, cv in enumerate(jaxpr.constvars):
        nid = -(len(jaxpr.invars) + k + 1)
        if nid in live:
            colors[nid] = ("const", tuple(cv.aval.shape), str(cv.aval.dtype))

    prim_names = {i: eqn.primitive.name
                  for i, eqn in enumerate(jaxpr.eqns, start=1) if i in live}
    return ElimGraph(succ, eliminable, colors=colors,
                     in_positions=in_positions,
                     position_sensitive=pos_sensitive,
                     meta={"n_eqns": len(jaxpr.eqns), "prim_names": prim_names})


def build_example_graph(name: str = "TransformerLM", dataset: Optional[str] = None,
                        seed: int = 0) -> ElimGraph:
    """Build the elimination graph for an alphagrad example target, offline."""
    import jax
    from alphagrad.approx.common.examples import (get_args, get_fn,
                                                  infer_argnums,
                                                  scalar_loss_fn)
    fn = scalar_loss_fn(get_fn(name))
    argnums = infer_argnums(name)
    args = get_args(name, jax.random.PRNGKey(seed), dataset=dataset)
    return build_elim_graph(fn, args, argnums)
