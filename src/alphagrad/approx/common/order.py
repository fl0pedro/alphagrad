"""The fixed elimination order (ticket dsnn-3qm.64).

ONE implementation of every static order the trainer, the sweep tool and the
probes can pin a plan to. ``--fixed-order`` on the trainer and ``--order`` on
``landscape_map`` both resolve here, so the two can never disagree about which
vertex comes at which step.

Orders (``FIXED_ORDER_CHOICES``):

  free        no pin; the vertex head chooses (trainer only).
  reverse     the valid vertices in descending id -- the order of the paired
              rev-exact reference and of the drift floor (CONTEXT.md: reverse
              stays the reference's order only).
  markowitz   the STATIC minimum Markowitz degree order (finding 59): greedy on
              the EXACT live graph, degree = |predecessors| x |successors| of
              the vertex on the current graph, ties to the lowest vertex id,
              computed once at env build. A SKIP removes paths and changes the
              live graph, so a per-step greedy order would diverge plan by
              plan; the static table pins every plan of a run to the same
              order.

The table is a ``numpy.int32`` array of vertex ids (1-based, the jaxpr eqn
index plus one), one per step, and is the ``fixed_order`` argument of
``masks.vertex_avail_at_step``.

``ALPHAGRAD_FORCE_REV_ORDER`` is gone. It was read at import time of
``common/masks.py`` and had no flag; a process that still exports it fails
loudly there (args only, no env vars, no fallback period).
"""
from __future__ import annotations

import numpy as np

FIXED_ORDER_CHOICES: tuple[str, ...] = ("free", "reverse", "markowitz")


def reverse_order(valid_vertices) -> np.ndarray:
    """The valid vertices in descending id.

    This is the order the old ``ALPHAGRAD_FORCE_REV_ORDER=1`` pin produced:
    ``vertex_avail_at_step`` kept only the HIGHEST remaining valid vertex, so
    the forced order is the valid vertices in descending id, not
    ``range(n, 0, -1)``.
    """
    return np.array(sorted((int(v) for v in valid_vertices), reverse=True),
                    dtype=np.int32)


def markowitz_order(jaxpr, argnums, consts, args, valid_vertices) -> np.ndarray:
    """Greedy minimum Markowitz degree order over the valid vertices.

    Walks graphax's ``IncrementalJaxpr`` on the exact graph: at every step the
    vertex with the smallest ``|preds| x |succs|`` on the CURRENT graph is
    eliminated (ties to the lowest vertex id), and the graph advances by that
    exact elimination. Intermediate equations with small degree (elementwise
    ops, embeddings, projections) therefore go before the contraction into the
    scalar loss, which keeps non-empty Jacobian out-dimensions alive for the
    structural approximations (finding 59: Diag legal on rhs, new and old
    sites under this order, never under reverse).
    """
    from graphax.incremental import IncrementalJaxpr

    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=False)
    eliminable = set(int(v) for v in valid_vertices)
    order: list[int] = []
    while eliminable:
        scores = {}
        for v in eliminable:
            v_var = jaxpr.eqns[v - 1].outvars[0]
            preds = [u for u in ij.graph if v_var in ij.graph[u]]
            succs = list(ij.graph.get(v_var, {}).keys())
            scores[v] = len(preds) * len(succs)
        best_v = min(scores.keys(), key=lambda v: (scores[v], v))
        order.append(best_v)
        eliminable.remove(best_v)
        ij.eliminate(best_v, (), None)
    return np.array(order, dtype=np.int32)


def fixed_order_table(kind: str, jaxpr, argnums, consts, args,
                      valid_vertices) -> np.ndarray | None:
    """The static order table for ``kind``, or ``None`` for ``free``."""
    if kind not in FIXED_ORDER_CHOICES:
        raise ValueError(
            f"--fixed-order {kind!r} is not one of {list(FIXED_ORDER_CHOICES)}")
    if kind == "free":
        return None
    if kind == "reverse":
        return reverse_order(valid_vertices)
    return markowitz_order(jaxpr, argnums, consts, args, valid_vertices)


def fixed_order_for_env(kind: str, env) -> np.ndarray | None:
    """``fixed_order_table`` from a ``VertexEliminationEnv`` (its config holds
    the jaxpr and argnums, the env the consts and args)."""
    cfg = env.config
    return fixed_order_table(kind, cfg.jaxpr, cfg.argnums, env.consts,
                             env.args, env.valid_vertices)
