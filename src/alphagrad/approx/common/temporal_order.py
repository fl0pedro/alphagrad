"""The TEMPORAL order constraint, and the step tag every vertex carries.

TWO INDEPENDENT CONSTRAINTS (CONTEXT.md, owner ruling 2026-09-15). On a
recurrent target the elimination graph is one BASE block plus one block per
time step, and there are two different questions to ask about the order:

  SPATIAL  (``--fixed-order {free,reverse,markowitz}``, ``common/order.py``)
           the order among the vertices of ONE time step.
  TEMPORAL (``--fixed-temporal-order {free,reverse,forward}``, this module)
           the order across the step COPIES. ``reverse`` = later steps first =
           backpropagation through time. ``forward`` = earlier steps first =
           real-time recurrent learning. ``markowitz`` is deliberately NOT a
           temporal choice: it is a degree heuristic on the live graph and says
           nothing about time.

Either can be free or pinned, and every one of the nine combinations is legal.
A PINNED CONSTRAINT IS A PARTIAL ORDER THE POLICY FILLS IN:

  pinned temporal, free spatial  every vertex of one copy is eliminated before
                                 any vertex of the next copy in the pinned
                                 direction; inside a copy the policy chooses.
  free temporal, pinned spatial  each copy's vertices keep the pinned RELATIVE
                                 order; the policy chooses which copy advances.
  both pinned                    a total order.
  both free                      no mask at all.

WHY A MASK AND NOT A TABLE. ``common/order.py`` pins an order by handing
``masks.vertex_avail_at_step`` a table of one vertex per step, which can only
express a TOTAL order. A partial order needs a predicate, so this module builds
the small static arrays the predicate reads and ``vertex_avail_at_step``
evaluates it. Vertex elimination is exact in ANY complete order, so a
constraint never makes a plan wrong -- it only makes it smaller.

THE STEP TAG. graphax wraps each unrolled step body of a temporal model in
``jax.named_scope("snn_step_<t>")`` (``graphax.examples.neuromorphic``), and
that scope survives both ``jax.make_jaxpr`` and graphax's
``inline_call_primitives`` on ``eqn.source_info.name_stack``. :func:`step_tags`
reads it. An equation in no such scope is BASE.

WHERE THE BASE GOES (the ruling this module records). The detached pre-window
forward is not in the traced graph at all -- it runs in the args builder and
arrives as the carry -- so the only untagged equations are the READOUT that
closes the loss, and the readout is DOWNSTREAM of every step. The base
therefore gets a STEP OF ITS OWN, numbered ONE PAST the last window step. Under
``reverse`` (later first) it is eliminated first, under ``forward`` (earlier
first) it is eliminated last, which is what backpropagation through time and
real-time recurrent learning respectively do. This is the right number only
while the untagged block is downstream of the steps, which is a property of
every target in ``snn_shd.TEMPORAL_TARGETS``; a future target with a
once-computed input block would need its own tag, not this default.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common.order import fixed_order_table
from alphagrad.approx.common.snn_shd import has_time_steps

FIXED_TEMPORAL_ORDER_CHOICES: tuple[str, ...] = ("free", "reverse", "forward")

#: ``step_tags`` writes this where an equation is in no step scope.
BASE_STEP = -1


def _scope_step(name_stack, prefix: str) -> int:
    """The step index encoded in ``name_stack``, or :data:`BASE_STEP`.

    ``name_stack`` stringifies to a ``/``-joined path, so a scope entered
    inside another one still shows ``snn_step_7`` as one of its segments.
    """
    if name_stack is None:
        return BASE_STEP
    text = str(name_stack)
    if not text:
        return BASE_STEP
    tag = prefix + "_"
    for part in text.split("/"):
        if part.startswith(tag):
            suffix = part[len(tag):]
            if suffix.isdigit():
                return int(suffix)
    return BASE_STEP


def step_tags(jaxpr, prefix: str | None = None) -> np.ndarray:
    """``(len(jaxpr.eqns),) int32`` step index per equation, BASE_STEP if none.

    Vertex ``v`` is equation ``v - 1`` (``common/order.py``), so entry ``v - 1``
    of this array is vertex ``v``'s step.
    """
    if prefix is None:
        from graphax.examples.neuromorphic import SNN_STEP_SCOPE
        prefix = SNN_STEP_SCOPE
    return np.array(
        [_scope_step(getattr(e.source_info, "name_stack", None), prefix)
         for e in jaxpr.eqns],
        dtype=np.int32)


class OrderConstraint(NamedTuple):
    """The static arrays :func:`masks.vertex_avail_at_step` evaluates.

    ``group_of`` and ``rank_in_group`` are indexed by ``vertex id - 1`` and
    cover every equation of the jaxpr; ``group_rank`` and ``group_size`` are
    indexed by group. A group is one time-step copy, or the base block.
    """

    group_of: jnp.ndarray        # (total_v,) int32, group index per vertex
    rank_in_group: jnp.ndarray   # (total_v,) int32, spatial rank, -1 = free
    group_rank: jnp.ndarray      # (n_groups,) int32, temporal position, -1 = free
    group_size: jnp.ndarray      # (n_groups,) int32, valid vertices per group
    spatial_pinned: bool
    temporal_pinned: bool

    @property
    def n_groups(self) -> int:
        return int(self.group_size.shape[0])


def _require_temporal_target(temporal: str, example: str | None) -> None:
    if temporal == "free":
        return
    if not has_time_steps(example):
        raise ValueError(
            f"--fixed-temporal-order {temporal} was passed with --example "
            f"{example}, which has NO time steps. A temporal order is the "
            f"order across step COPIES; this target has one copy. Drop the "
            f"flag or change the target.")


def build_order_constraint(spatial: str, temporal: str, env,
                           example: str | None):
    """``(fixed_order_table, order_constraint)`` for the two flags.

    Exactly one of the two is ever not ``None``, and both are ``None`` when
    neither constraint bites:

      * both free                      -> ``(None, None)``, no mask.
      * spatial pinned, no time steps  -> ``(table, None)``, the ticket .64
        path, BYTE FOR BYTE what every non-temporal arm has run since
        2026-09-05 (one group makes the predicate the same one-hot the table
        gathers, but the table path is kept so nothing about those arms moves).
      * anything temporal, or a pinned spatial order on a target WITH time
        steps -> ``(None, OrderConstraint)``.
    """
    if temporal not in FIXED_TEMPORAL_ORDER_CHOICES:
        raise ValueError(
            f"--fixed-temporal-order {temporal!r} is not one of "
            f"{list(FIXED_TEMPORAL_ORDER_CHOICES)}")
    _require_temporal_target(temporal, example)
    timed = has_time_steps(example)
    if temporal == "free" and not timed:
        return fixed_order_table(spatial, env.config.jaxpr, env.config.argnums,
                                 env.consts, env.args, env.valid_vertices), None
    if temporal == "free" and spatial == "free":
        return None, None

    cfg = env.config
    tags = step_tags(cfg.jaxpr)
    total_v = len(cfg.jaxpr.eqns)
    valid = sorted(int(v) for v in env.valid_vertices)
    steps = sorted({int(tags[v - 1]) for v in valid if tags[v - 1] != BASE_STEP})
    if not steps:
        raise ValueError(
            f"--example {example} is registered as a temporal target but not "
            f"one of its {len(valid)} valid vertices carries a step scope. "
            f"The model's unrolled step body must sit inside "
            f"graphax.examples.neuromorphic.snn_step_scope(t); without it the "
            f"temporal order has nothing to order.")

    # THE GROUPS: one per step copy, in time order, then the base one past the
    # last step (see the module docstring for why).
    step_to_group = {s: i for i, s in enumerate(steps)}
    base_group = len(steps)
    n_groups = base_group + 1

    group_of = np.zeros(total_v, dtype=np.int32)
    for v in range(1, total_v + 1):
        t = int(tags[v - 1])
        group_of[v - 1] = base_group if t == BASE_STEP else step_to_group[t]

    group_size = np.zeros(n_groups, dtype=np.int32)
    for v in valid:
        group_size[group_of[v - 1]] += 1

    # THE SPATIAL RANK: the vertex's position in the pinned global table,
    # RESTRICTED to its own group. Only the relative order inside a group is
    # read, so reverse and markowitz both give one consistent total order per
    # copy without either flag learning about the other.
    rank_in_group = np.full(total_v, -1, dtype=np.int32)
    if spatial != "free":
        table = fixed_order_table(spatial, cfg.jaxpr, cfg.argnums, env.consts,
                                  env.args, env.valid_vertices)
        seen = np.zeros(n_groups, dtype=np.int32)
        for v in (int(x) for x in table):
            g = group_of[v - 1]
            rank_in_group[v - 1] = seen[g]
            seen[g] += 1

    # THE TEMPORAL RANK: the group's position in the pinned direction. reverse
    # eliminates the LAST copy first (BPTT), forward the FIRST (RTRL); the base
    # sits one past the last step either way, so reverse takes it first.
    group_rank = np.full(n_groups, -1, dtype=np.int32)
    if temporal != "free":
        order = (list(range(n_groups - 1, -1, -1)) if temporal == "reverse"
                 else list(range(n_groups)))
        for pos, g in enumerate(order):
            group_rank[g] = pos

    return None, OrderConstraint(
        group_of=jnp.asarray(group_of, dtype=jnp.int32),
        rank_in_group=jnp.asarray(rank_in_group, dtype=jnp.int32),
        group_rank=jnp.asarray(group_rank, dtype=jnp.int32),
        group_size=jnp.asarray(group_size, dtype=jnp.int32),
        spatial_pinned=spatial != "free",
        temporal_pinned=temporal != "free",
    )


def describe(constraint: "OrderConstraint | None", table, spatial: str,
             temporal: str) -> str:
    """One line for the trainer's ``[cfg]`` block."""
    if constraint is None and table is None:
        return (f"order: spatial {spatial}, temporal {temporal} -- no mask, "
                f"the vertex head chooses freely")
    if constraint is None:
        return (f"order: spatial {spatial} (static table, {len(table)} steps), "
                f"temporal {temporal} -- one copy, so the pin is a total order")
    sizes = np.asarray(constraint.group_size)
    return (f"order: spatial {spatial}, temporal {temporal}; "
            f"{constraint.n_groups} groups "
            f"({constraint.n_groups - 1} step copies of "
            f"{int(sizes[0])} vertices + a base of {int(sizes[-1])}), "
            f"partial order filled in by the policy")
