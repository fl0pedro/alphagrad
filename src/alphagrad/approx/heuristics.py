"""Heuristic vertex-elimination orders for behaviour-clone warm-start.

Stage C calls for an optional BC pass against a heuristic before PPO. The
classic heuristic for sparse vertex elimination is **Markowitz min-degree**:
at each step pick the vertex with the smallest fan-in × fan-out product.
That minimises the immediate fill-in (= number of new edges created by
eliminating the vertex) and gives a strong baseline that the learned policy
should at least match on small graphs.

This module operates purely at the *jaxpr* level — same view of the graph
the env starts from. No graphax dependency beyond the original jaxpr's
``invars`` / ``eqns``. The output is a list of 1-indexed vertex IDs in the
same convention as :class:`alphagrad.approx.env.VertexEliminationEnv`.
"""

from __future__ import annotations

from collections import defaultdict


def _build_initial_graph(jaxpr, valid_vertices):
    """Return ``(predecessors, successors)`` dicts keyed on 1-indexed vertex id.

    ``predecessors[v]`` is the set of vertices whose output is consumed by
    eqn ``v - 1``; ``successors[v]`` is the set of vertices that consume ``v``.
    Only vertices in ``valid_vertices`` participate; anything outside (input
    args, terminal outputs, value-only branches) is treated as constant edge
    boundary and dropped from the degree counts.
    """
    valid_set = set(int(v) for v in valid_vertices)
    var_producer: dict = {}
    for i, eqn in enumerate(jaxpr.eqns, 1):
        if i not in valid_set:
            continue
        for v in eqn.outvars:
            var_producer[id(v)] = i

    predecessors: dict = defaultdict(set)
    successors: dict = defaultdict(set)
    for i, eqn in enumerate(jaxpr.eqns, 1):
        if i not in valid_set:
            continue
        for inv in eqn.invars:
            src = var_producer.get(id(inv))
            if src is None or src == i:
                continue
            predecessors[i].add(src)
            successors[src].add(i)
    return dict(predecessors), dict(successors)


def markowitz_order(jaxpr, valid_vertices) -> list[int]:
    """Greedy Markowitz min-degree elimination order.

    At every step picks the still-uneliminated valid vertex with the smallest
    ``len(predecessors) * len(successors)``. Ties are broken by smaller
    vertex id (deterministic — important so the BC target is reproducible
    across seeds).
    """
    predecessors, successors = _build_initial_graph(jaxpr, valid_vertices)
    remaining = set(int(v) for v in valid_vertices)
    order: list[int] = []
    while remaining:
        best_v = None
        best_score = None
        for v in sorted(remaining):
            score = len(predecessors.get(v, set())) * len(successors.get(v, set()))
            if best_score is None or score < best_score:
                best_v = v
                best_score = score
        order.append(int(best_v))
        # Update graph: connect every predecessor to every successor of the
        # eliminated vertex (this is the fill-in step) and drop best_v.
        preds = predecessors.pop(best_v, set())
        succs = successors.pop(best_v, set())
        for p in preds:
            successors[p].discard(best_v)
        for s in succs:
            predecessors[s].discard(best_v)
        for p in preds:
            for s in succs:
                if p == s:
                    continue
                successors.setdefault(p, set()).add(s)
                predecessors.setdefault(s, set()).add(p)
        remaining.discard(best_v)
    return order
