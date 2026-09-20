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


#: The name of the order :func:`reference_order_face_count` counts on, printed
#: beside the number so a reader never has to guess which walk produced it.
REFERENCE_ORDER_NAME = "reverse (the paired rev-exact reference order)"


def face_count_on_order(jaxpr, argnums, consts, args, order) -> int:
    """How many LIVE FACES a plan that follows ``order`` offers in total.

    One elimination walk, counting ``graphax.faces_of`` at every step on the
    live graph the previous steps left -- the same enumeration
    ``env._face_transforms_for_order`` and ``landscape_map.face_inventory``
    ride, so this number is the number of face DECISIONS the face head makes
    over one episode, not a bound.

    The count is a property of the ORDER, never of the graph alone:
    eliminating a vertex rewires its neighbours, so vertex k's face count
    depends on the k-1 before it (tools/faces_per_vertex.py).
    """
    from graphax import faces_of
    from graphax.incremental import IncrementalJaxpr

    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=False)
    total = 0
    for v in order:
        v = int(v)
        total += len(faces_of(ij.graph, ij.tgraph, v, jaxpr))
        ij.eliminate(v, (), None)
    return int(total)


def reference_order_face_count(env) -> tuple[int, str, np.ndarray]:
    """``(F, order_name, order)`` for the REVERSE-MODE REFERENCE ORDER.

    F is the live-face count the normalized face-head init
    (``--face-init-approx-per-plan`` / ``--face-init-skips-per-plan``)
    normalizes by, under the owner's ruling of 2026-09-20.

    THE ORDER IS THE PAIRED REFERENCE'S OWN. ``env.py`` measures every
    candidate against rev-exact -- ``sorted(o_list, reverse=True)``, the
    reverse order over the same vertex set the candidate eliminated, no rule
    and no face action -- and a complete plan eliminates every valid vertex,
    so :func:`reverse_order` over ``env.valid_vertices`` IS that order. The
    reward the policy chases is a ratio against this walk, so the init's
    normalizer and the reward's denominator name the same plan.

    Under ``--fixed-order free`` the policy samples a DIFFERENT order every
    episode and every environment, each with its own face count; F is the
    reference's count, not any sampled plan's, and it is computed once.
    """
    cfg = env.config
    order = reverse_order(env.valid_vertices)
    F = face_count_on_order(cfg.jaxpr, cfg.argnums, env.consts, env.args,
                            order)
    return F, REFERENCE_ORDER_NAME, order
