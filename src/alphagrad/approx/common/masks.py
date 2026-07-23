"""Action-mask helpers shared between the approx-RL trainers.

The trainers differ in their action-space layout (legacy unrolled `sp_type ×
vertex` vs. the new factorised `vertex × pair × factor`), but the underlying
shape inspection — "for each vertex, what is the output ndim and the smallest
input ndim?" — is the same. These helpers expose that primitive plus a few
convenience builders for the most common mask layouts.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def vertex_axis_dims(jaxpr, total_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (out_ndim, min_in_ndim) — entries are 0 if the vertex has no shape info."""
    out_ndims = np.zeros(total_v, dtype=np.int32)
    in_ndims = np.zeros(total_v, dtype=np.int32)
    for i, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_ndims[i] = len(eqn.outvars[0].aval.shape)
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        if invars:
            in_ndims[i] = min(len(v.aval.shape) for v in invars)
    return out_ndims, in_ndims


def build_pair_valid_mask(
    jaxpr,
    total_v: int,
    num_pair_choices: int,
    pair_stop_idx: int,
    disable_sparsification: bool = False,
):
    """Build the per-vertex axis-pair validity mask used by the factorised policy head.

    The pair-index layout matches the trainer's `_PAIR_TO_BASE` table:
    `0 → (0,0), 1 → (0,1), 2 → (1,0), 3 → (1,1), pair_stop_idx → STOP`. STOP is
    always available; the four real pairs gate on the vertex's `out_ndim` and
    `min_in_ndim`. Returns a `jnp.float32` array of shape `(total_v, num_pair_choices)`.
    """
    out_ndims, in_ndims = vertex_axis_dims(jaxpr, total_v)
    mask = np.zeros((total_v, num_pair_choices), dtype=np.float32)
    mask[:, pair_stop_idx] = 1.0  # STOP is always valid

    if disable_sparsification:
        return jnp.array(mask)

    for i in range(total_v):
        out_n = int(out_ndims[i])
        in_n = int(in_ndims[i])
        if out_n == 0 or in_n == 0:
            continue
        if out_n >= 1 and in_n >= 1:
            mask[i, 0] = 1.0  # (0, 0)
        if out_n >= 1 and in_n >= 2:
            mask[i, 1] = 1.0  # (0, 1)
        if out_n >= 2 and in_n >= 1:
            mask[i, 2] = 1.0  # (1, 0)
        if out_n >= 2 and in_n >= 2:
            mask[i, 3] = 1.0  # (1, 1)
    return jnp.array(mask)


# Pair-index → (out_axis, primal_axis) — kept for the legacy autoreg rule
# head's spec construction. The per-(vertex, pair, factor) validity mask
# that used to live in this file has been removed: graphax's typed
# transform API (apply_diag / apply_compress) now validates factors at
# apply time, so the per-factor pre-mask is dead weight. Callers that
# previously consumed it (the autoreg factor head) should pass an
# all-ones tensor of shape ``(total_v, num_pair_choices, num_factors)``
# or migrate to the prime-exponent head in heads.py which does its own
# per-pair gcd-based factor legality.
_PAIR_TO_BASE = ((0, 0), (0, 1), (1, 0), (1, 1))


def build_legacy_sp_valid_mask(
    jaxpr,
    total_v: int,
    *,
    num_sp_types: int,
    use_min_in_ndim: bool,
    disable_sparsification: bool = False,
):
    """Build the legacy `(num_sp_types, total_v)` mask used by the unrolled action space.

    `num_sp_types == 5` matches the original PPO layout where the four real
    `sp_types` correspond to (output axis, input axis) ∈ {(0,0), (0,1), (1,0),
    (1,1)} and row 0 is "vertex valid".

    `num_sp_types == 3` matches the `alpha0`/`mu0`/`gdpo` layout: row 1 = at
    least one axis pair, row 2 = at least two axis pairs available on the input.

    `use_min_in_ndim` switches between using the min (legacy ppo) or max (legacy
    alpha0/mu0/gdpo) of the input variable ndims. Both behaviours are preserved
    so we can move the trainers onto this helper without semantic drift.
    """
    if num_sp_types not in (3, 5):
        raise ValueError(f"num_sp_types must be 3 or 5, got {num_sp_types}")

    out_ndims = np.zeros(total_v, dtype=np.int32)
    in_ndims = np.zeros(total_v, dtype=np.int32)
    for i, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_ndims[i] = len(eqn.outvars[0].aval.shape)
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        if invars:
            in_ndims[i] = (
                min(len(v.aval.shape) for v in invars)
                if use_min_in_ndim
                else max(len(v.aval.shape) for v in invars)
            )

    sp_mask = np.zeros((num_sp_types, total_v), dtype=np.float32)
    for i in range(total_v):
        sp_mask[0, i] = 1.0
        if disable_sparsification:
            continue
        out_n = int(out_ndims[i])
        in_n = int(in_ndims[i])
        if num_sp_types == 5:
            if out_n == 2 and in_n == 2:
                sp_mask[1:5, i] = 1.0
            elif out_n == 1 and in_n == 2:
                sp_mask[1:3, i] = 1.0
            elif out_n == 2 and in_n == 1:
                sp_mask[1, i] = 1.0
                sp_mask[3, i] = 1.0
            elif out_n == 1 and in_n == 1:
                sp_mask[1, i] = 1.0
        else:  # num_sp_types == 3
            if out_n >= 1 and in_n >= 1:
                sp_mask[1, i] = 1.0
            if out_n >= 1 and in_n >= 2:
                sp_mask[2, i] = 1.0
    return jnp.array(sp_mask)


def build_vertex_valid_static(valid_vertices, total_v: int):
    """Static per-vertex 0/1 mask for "this vertex can be eliminated at all"."""
    arr = np.zeros(total_v, dtype=np.float32)
    for v in valid_vertices:
        arr[v - 1] = 1.0
    return jnp.array(arr)


def vertex_avail_at_step(state, vertex_valid_static, total_v: int, num_valid: int):
    """Per-vertex availability at the current rollout step.

    `state.order` keeps the chosen vertices in slots `[0, step_count)` (slots
    beyond `step_count` retain the original valid-vertices ordering). We zero
    out the entries that have already been chosen.
    """
    chosen = state.order
    step_idx = state.step_count
    arange_v = jnp.arange(num_valid)
    active = (arange_v < jnp.expand_dims(step_idx, -1)).astype(jnp.float32)
    already_chosen = (
        jnp.zeros(total_v, dtype=jnp.float32).at[chosen - 1].add(active)
    )
    return vertex_valid_static * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-EDGE micro-action validity (Diag / Compress)
# ---------------------------------------------------------------------------
# graphax's typed transform API fails LOUDLY when a micro-action does not fit
# the edge it lands on ("TRANSFORM DID NOT FIT ... Mask invalid actions up
# front instead of discovering them by throwing"). These helpers are that
# mask, and they live here rather than in graphax because the action space is
# the env's concern, not the AD kernel's.
#
# `build_pair_valid_mask` above is per-VERTEX and infers legality from jaxpr
# eqn ndims. That is too coarse for the per-face policy: the lhs / rhs / res
# slots of a single face carry three DIFFERENT geometries. The helpers below
# read the SparseTensor's own index structure, so they are exact per edge.


def diag_valid_mask(st, max_dims: int) -> np.ndarray:  # noqa: F821 (uses diag_pair_factor_space, defined below)
    """``(max_dims, max_dims)`` bool mask of legal ``Diag(i, j)`` on edge ``st``.

    ``i`` and ``j`` index the CONCATENATED ``out_dims + primal_dims`` list --
    the numbering :class:`graphax.sparse.micro_actions.Diag` uses. A pair is
    legal iff all of:

    * both lie in ``[0, len(out_dims) + len(primal_dims))`` and ``i != j``;
    * the pair is **split across the bipartite boundary** -- a Jacobian
      diagonal ties one OUT axis to one PRIMAL axis, so out<->out and
      primal<->primal pairs are meaningless and graphax rejects them;
    * **neither side is already spoken for** -- a dim with ``is_sparse`` is
      already paired through ``other_id``, so its only legal partner is that
      partner. Two dense (free) dims may always be paired;
    * **neither side is implicit** (``axis is None``) -- such a dim has no
      physical axis to diagonalise, either because it was never materialised
      or because an earlier ``Compress`` in the same sub-episode dropped it.

    Entries past the tensor's real rank stay ``False``, so the mask can be
    emitted at a fixed ``max_dims`` for a statically-shaped policy head.
    """
    mask = np.zeros((max_dims, max_dims), dtype=bool)
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    n_out = len(st.out_dims)
    n = min(len(dims), max_dims)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i < n_out) == (j < n_out):
                continue  # both out, or both primal -> not a diagonal
            di, dj = dims[i], dims[j]
            # An IMPLICIT dim (axis is None) has no physical axis to
            # block-diagonalise -- it was never materialised, or a prior
            # Compress in this same sub-episode dropped it. graphax rejects
            # such a pair outright, so it must be masked out here.
            if di.axis is None or dj.axis is None:
                continue
            # A sparse dim is already half of a pair; only its partner is legal.
            if di.is_sparse and di.other_id != dj.id:
                continue
            if dj.is_sparse and dj.other_id != di.id:
                continue
            # A pair that affords only the no-op is not an action. Without
            # this the factor head would be handed an all-masked support and
            # its softmax -- uniform over illegal factors -- would look like a
            # confident choice. Costs one gcd per surviving pair.
            _base, span = diag_pair_factor_space(st, i, j)
            if span <= 1:
                continue
            mask[i, j] = True
    return mask


def diag_pair_gcd(st, i: int, j: int) -> int:
    """Largest legal ``Diag.factor`` for pair ``(i, j)`` on ``st``.

    ``factor`` must be a positive divisor of BOTH logical sizes, so the legal
    factors are exactly the divisors of this gcd -- which is what the
    prime-exponent factor head in ``heads.py`` enumerates. Returns ``0`` for an
    out-of-range pair.

    This is the FREE-pair answer. An already-coupled pair carries the extra
    constraint that the new factor must be a multiple of its current meta
    count; use :func:`diag_pair_factor_space`, which folds both rules together.
    """
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    if not (0 <= i < len(dims) and 0 <= j < len(dims)):
        return 0
    return math.gcd(int(dims[i].logical_size), int(dims[j].logical_size))


def compress_valid_mask(st, max_axes: int) -> np.ndarray:
    """``(max_axes,)`` bool mask of legal ``Compress`` axes on edge ``st``.

    ``Compress.axes`` are PHYSICAL positions into ``st.val``, so the legal
    range is ``[0, val.ndim)`` -- ``val.ndim - 1`` is the highest axis, from 0
    up. An edge with no materialised ``val`` (a pure-structure Jacobian, e.g.
    a constant ``-1``) masks to all-``False``: graphax *accepts* a Compress
    there, but it is a guaranteed no-op, and spending one of the sub-episode's
    micro-actions on a no-op is never what the policy wants. The mask is
    therefore deliberately stricter than ``apply_compress`` in this one case --
    strictness is safe, looseness would crash.
    """
    mask = np.zeros(max_axes, dtype=bool)
    val = getattr(st, "val", None)
    if val is None:
        return mask
    ndim = int(getattr(val, "ndim", 0))
    if ndim > 0:
        mask[: min(ndim, max_axes)] = True
    return mask


def diag_pair_factor_space(st, i: int, j: int) -> tuple[int, int]:
    """``(base, span)`` describing the legal ``Diag.factor`` set for ``(i, j)``.

    The legal factors are exactly ``base * d`` for every divisor ``d`` of
    ``span`` **with ``d > 1``** -- a shape the prime-exponent factor head can
    enumerate directly, since it already works by picking a divisor of a single
    integer.

    ``d`` is the RELATIVE subdivision: how many times finer the blocks get.
    ``Diag.factor`` itself is ABSOLUTE -- it is the resulting meta count (the
    number of diagonal blocks), and the block sizes come out as
    ``(N_i // factor, N_j // factor)``. So halving the blocks of an existing
    meta-2 pair is ``d = 2``, i.e. ``factor = 4``, not ``factor = 2``.

    ``d = 1`` is excluded because it is a pure no-op: for a free pair
    ``factor = 1`` leaves the tensor untouched (``apply_diag`` returns the very
    same object), and for a coupled pair ``factor == meta`` is likewise
    defined as a no-op. Spending a micro-action slot to do nothing is never
    what the policy wants, so ``span == 1`` means the pair affords NO legal
    action at all -- which is why :func:`diag_valid_mask` masks such a pair
    out entirely rather than leaving the factor head with empty support.

    Two regimes, both enforced by graphax:

    * **free pair** -- ``base = 1``, ``span = gcd(N_i, N_j)``. Any divisor of
      the gcd divides both logical sizes, which is the whole requirement.
    * **already-coupled pair** -- ``base = m``, the current meta count, and
      ``span = gcd(N_i, N_j) // m``. A further Diag on a coupled pair may only
      SUBDIVIDE it: graphax accepts ``factor == m`` as a no-op and
      ``factor = m*k`` when it still divides both logical sizes, but rejects a
      coarser or non-nesting re-mask outright rather than silently applying a
      lighter approximation than asked for.

    Returns ``(0, 0)`` for an out-of-range pair.
    """
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    if not (0 <= i < len(dims) and 0 <= j < len(dims)):
        return 0, 0
    di, dj = dims[i], dims[j]
    g = math.gcd(int(di.logical_size), int(dj.logical_size))
    coupled = (
        di.is_sparse and dj.is_sparse
        and di.other_id == dj.id and dj.other_id == di.id
        and di.size == dj.size
    )
    if not coupled:
        return 1, g
    m = int(di.size)
    if m <= 0 or g % m != 0:
        return 0, 0
    return m, g // m


# ---------------------------------------------------------------------------
# Concrete legal actions, and the per-slot chooser graphax calls
# ---------------------------------------------------------------------------
# The lhs / rhs / res slots of a face are JOIN INTERMEDIATES -- lhs is the fresh
# contraction product, rhs the edge accumulated so far, res their sum -- so none
# of them exists before `eliminate` runs. A policy therefore cannot be handed a
# precomputed per-face mask: the operand's index structure is not knowable until
# the moment the transform is applied.
#
# graphax closes this by letting a slot hold a CALLABLE. Handed the live
# operand, it returns the micro-action it chose, which graphax then applies and
# records exactly as it would a literal one. These helpers build that callable
# so the choice is made against an accurate mask.


def _divisors_above_one(n: int):
    return [d for d in range(2, int(n) + 1) if n % d == 0]


def legal_diag_actions(st, max_dims: int = 8) -> list:
    """Every ``Diag`` graphax will accept on ``st``, as concrete actions."""
    from graphax.sparse.micro_actions import Diag

    mask = diag_valid_mask(st, max_dims)
    out = []
    for i in range(max_dims):
        for j in range(max_dims):
            if not mask[i, j]:
                continue
            base, span = diag_pair_factor_space(st, i, j)
            out.extend(Diag(i, j, base * d) for d in _divisors_above_one(span))
    return out


def legal_compress_actions(st, max_axes: int = 8,
                           kinds: tuple = ("mean",)) -> list:
    """Every single-axis ``Compress`` graphax will accept on ``st``."""
    from graphax.sparse.micro_actions import Compress

    mask = compress_valid_mask(st, max_axes)
    return [Compress(axes=(a,), kind=k)
            for a in range(max_axes) if mask[a] for k in kinds]


# ---------------------------------------------------------------------------
# LIVE per-VERTEX masks (the per-vertex ``transforms`` path)
# ---------------------------------------------------------------------------
# WHAT THE PER-VERTEX SITE ACTUALLY TRANSFORMS
# --------------------------------------------
# It is tempting to think the per-vertex ``transforms`` list acts on the
# vertex's OWN elemental Jacobian -- a tensor that exists before the vertex is
# eliminated and whose geometry is readable straight off the jaxpr equation.
# It does not. ``graphax.core._eliminate_vertex`` applies the list to
# ``edge_outval``, which at that point is
#
#     edge_outval = (out-edge Jacobian) @ (in-edge Jacobian)   [+ the existing
#                   parallel edge, then drained]
#
# for EACH face ``(in_edge -> vertex -> out_edge)`` -- literally the same site
# as the per-face ``res`` slot, which is applied on the next line. So it is a
# JOIN INTERMEDIATE just like ``lhs`` / ``rhs`` / ``res``: it does not exist
# until the vertex is eliminated, its logical rank is the rank of a DIFFERENT
# pair of graph variables (out_dims = the out-edge var's shape, primal_dims =
# the in-edge var's shape), and its index structure depends on the whole
# elimination prefix. Measured on the nn256 graph: vertex 5 is ``tanh`` with
# nominal ``out=(16,63) primal=(16,63)``, but under the forward order the
# tensor its per-vertex transform receives is ``out=(16,10) primal=(63,)`` --
# a different rank, different sizes, different everything.
#
# Consequently the nominal ``(out_shape ++ primal_shape)`` model the env uses
# for its axis tokens CANNOT decide legality, and the two failure families the
# PPO runs emit are exactly its two blind spots:
#
#   * ``Diag`` on a pair that is ALREADY a coupled diagonal (every elementwise
#     vertex's edge is), where graphax only accepts a factor that is a multiple
#     of the current meta count;
#   * ``Compress`` on a physical axis past the LIVE ``val.ndim`` (a diagonal
#     pair stores two logical dims in one physical axis).
#
# The only sound way to decide is to look at the live tensor. This oracle does
# that WITHOUT committing: it keeps a structural elimination in step with the
# episode and, for each candidate vertex, replays that one vertex on a COPY of
# the graph with a recording no-op transform in the per-vertex slot -- the same
# slot, the same tensor, zero side effects.


def _shallow_copy_graph(graph):
    """Two-level copy of a ``{src: {dst: SparseTensor}}`` elimination graph.

    ``_eliminate_vertex`` REPLACES entries (``_set_inner`` / ``_del_inner``)
    rather than mutating the tensors in place, so copying the two dict levels
    is enough to leave the original untouched.
    """
    return {k: dict(v) for k, v in graph.items()}


class LiveVertexMaskOracle:
    """Exact per-vertex ``Diag`` / ``Compress`` legality, read off LIVE edges.

    Usage mirrors the episode::

        oracle = LiveVertexMaskOracle(jaxpr, consts, args, argnums)
        for step in episode:
            pair_valid, compress_valid = oracle.masks(candidate_vertices)
            v, rules = policy(...)          # masked with the above
            oracle.advance(v, rules)        # keep in step with the env

    ``masks`` returns ``(V+1, N, N)`` / ``(V+1, N)`` float32 arrays indexed by
    1-based vertex id (row 0 is unused padding) so a policy can gather the row
    for whichever vertex it sampled inside a jit.

    CLOSED LOOP OVER THE WIRE FORMAT. The policy does not hand graphax a
    ``Diag`` directly -- it emits axis-token indices which
    :func:`~alphagrad.approx.env.micro_actions_to_rule_specs_jax` packs into a
    ``[bi1, bi2, factor]`` row and
    :func:`~alphagrad.approx.env.rule_specs_to_transforms` unpacks again. This
    oracle asks the question that actually matters -- "if the policy picks
    token pair (i, j), is the transform the env WILL EMIT legal on every face
    of this vertex?" -- so it screens the round trip, not an idealised action.

    NO ARGUMENT DATA IS USED. The builder runs on fresh jaxpr tracers, and only
    ``SparseTensor`` index metadata and ``val.shape`` are read; the throwaway
    equations the probes trace are discarded. Cost is ~4 ms per candidate
    vertex on the nn256 graph (two probes, see ``_DISPATCH_MODES``).
    """

    def __init__(self, jaxpr, consts, args, argnums, *, max_axes: int = 8):
        from graphax.incremental import IncrementalJaxpr

        self.jaxpr = jaxpr
        self.max_axes = int(max_axes)
        self.total_v = len(jaxpr.eqns)
        self._argnums = tuple(int(a) for a in argnums)
        self._consts = list(consts)
        self._args = list(args)
        self._IncrementalJaxpr = IncrementalJaxpr
        self.reset()

    # -- episode bookkeeping ------------------------------------------------
    # TWO graphs, one per elemental-dispatch setting. ``_callback`` runs the
    # SAME order through ``vertex_elimination_jaxpr`` (which sets
    # ``dispatch.approx_active``) for the op counts and, under
    # ALPHAGRAD_MEASURE_VIA_AOJ=1, through ``IncrementalJaxpr`` (which does
    # not) for the measured executable. The flag gates the elemental
    # composition layer inside ``sparse_matmul``, so the two paths build
    # STRUCTURALLY DIFFERENT edges from the same order (measured on nn256
    # vertex 7: ``val=(16,10,63)`` vs ``val=(10,63)`` with an implicit pair).
    # An action has to be legal on both or the first one to run sentinels the
    # measurement, so the oracle tracks both and intersects.
    _DISPATCH_MODES = (True, False)

    def reset(self):
        """Rewind to the un-eliminated graph (call at env reset)."""
        self._incrs = {
            m: self._IncrementalJaxpr(
                self.jaxpr, self._argnums, list(self._consts),
                list(self._args),
            )
            for m in self._DISPATCH_MODES
        }
        self._eliminated: set[int] = set()

    def advance(self, vertex: int, rules=()):
        """Commit one elimination so later masks see the post-step graph.

        ``rules`` must be the transforms the env ACTUALLY applied at this
        vertex (``rule_specs_to_transforms``' output for it) -- an earlier
        approximation changes the structure of every downstream edge, so a mask
        computed against an exact-elimination replay would be wrong.
        """
        from graphax.sparse.elemental.dispatch import (
            approx_active, set_approx_active,
        )

        vertex = int(vertex)
        prev = approx_active()
        try:
            for mode, incr in self._incrs.items():
                set_approx_active(mode)
                incr.eliminate(vertex, rules=tuple(rules))
        finally:
            set_approx_active(prev)
        self._eliminated.add(vertex)

    @property
    def eliminated(self):
        return frozenset(self._eliminated)

    # -- the probe ----------------------------------------------------------
    def probe_faces(self, vertex: int, approx: bool = True):
        """The live tensors this vertex's per-vertex transforms would receive.

        One entry per face, in the order ``_eliminate_vertex`` visits them.
        Runs on a copy of the graph and drops the equations it traced, so the
        oracle's own state is unchanged and the caller may probe every
        candidate before choosing one.

        ``approx`` selects the ``graphax.sparse.elemental.dispatch``
        ``approx_active`` flag. It is NOT cosmetic: the flag gates the elemental
        composition layer inside ``sparse_matmul``, so the two settings produce
        contractions with genuinely different index structure (measured on
        nn256 vertex 7: ``val=(16,10,63)`` with the flag on vs ``val=(10,63)``
        with a compressed-away implicit pair with it off). ``_callback``
        exercises BOTH — ``vertex_elimination_jaxpr`` sets the flag for its
        op-count pass, the AOJ builder (``ALPHAGRAD_MEASURE_VIA_AOJ=1``) never
        does — so :meth:`vertex_mask` intersects over both.
        """
        from jax._src import core as _jcore
        from graphax.core import _eliminate_vertex
        from graphax.sparse.elemental.dispatch import (
            approx_active, set_approx_active,
        )

        seen: list = []

        def _record(st):
            seen.append(st)
            return st

        incr = self._incrs[bool(approx)]
        graph = _shallow_copy_graph(incr.graph)
        tgraph = _shallow_copy_graph(incr.tgraph)
        n_eqns0 = len(incr.trace.frame.tracing_eqns)
        # The probe transform is a CALLABLE, which is what puts core.py on its
        # approx code path -- the same path the real run takes.
        prev = approx_active()
        set_approx_active(bool(approx))
        try:
            with _jcore.set_current_trace(incr.trace):
                _eliminate_vertex(
                    int(vertex), incr.jaxpr, graph, tgraph, incr.vo, False,
                    transforms=(_record,), face_transforms=None,
                )
        finally:
            set_approx_active(prev)
            # Throw away the equations the probe traced; nothing ever
            # materialises this builder's jaxpr, but the list would grow
            # without bound over an episode.
            del incr.trace.frame.tracing_eqns[n_eqns0:]
        return seen

    # -- the masks ----------------------------------------------------------
    def vertex_mask(self, vertex: int):
        """``(pair_valid (N, N), compress_valid (N,))`` bool for one vertex.

        A token pair / axis is admitted only when the transform the env would
        emit for it is legal on EVERY face -- the per-vertex list is applied
        uniformly to all of them, so anything less is a crash waiting for the
        second face.
        """
        from alphagrad.approx.env import diag_row_to_pair

        N = self.max_axes
        pair = np.zeros((N, N), dtype=bool)
        comp = np.zeros((N,), dtype=bool)
        vertex = int(vertex)
        if not (1 <= vertex <= self.total_v) or vertex in self._eliminated:
            return pair, comp

        eqn = self.jaxpr.eqns[vertex - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            return pair, comp
        out_shape = tuple(eqn.outvars[0].aval.shape)
        out_len = len(out_shape)
        primal_shapes = [
            tuple(iv.aval.shape) for iv in eqn.invars if hasattr(iv, "aval")
        ]
        if not primal_shapes:
            return pair, comp

        # Intersect over BOTH elemental-dispatch settings (see _DISPATCH_MODES).
        faces = []
        for mode in self._DISPATCH_MODES:
            faces += self.probe_faces(vertex, approx=mode)
        if not faces:
            return pair, comp

        # --- COMPRESS: the emitted axis is the token index verbatim ---------
        # Env screens (rule_specs_to_transforms): the axis must exist on every
        # invar, and the FULL-REDUCTION CAP forbids dropping the edge's last
        # physical axis. Both are nominal; the live ``val.ndim`` test on top of
        # them is what actually keeps apply_compress from raising.
        edge_phys_axes = out_len + max(
            (len(ps) for ps in primal_shapes), default=0
        )
        if edge_phys_axes > 1:
            face_comp = [compress_valid_mask(st, N) for st in faces]
            for a in range(N):
                if a >= out_len and any(
                    a - out_len >= len(ps) for ps in primal_shapes
                ):
                    continue
                if all(m[a] for m in face_comp):
                    comp[a] = True

        # --- DIAG: token pair -> the (i, j) the env will emit ---------------
        # The policy's primal tokens come from the FIRST invar
        # (``compute_static_axis_state``'s primal proxy) while the env screens
        # ``bi2`` against EVERY invar, so the reachable ``bi2`` range is the
        # narrowest of them.
        face_diag = [diag_valid_mask(st, N) for st in faces]
        n_primal = min(len(ps) for ps in primal_shapes)
        for bi1 in range(out_len):
            n1 = int(out_shape[bi1])
            for bi2 in range(n_primal):
                i, j = diag_row_to_pair(self.jaxpr, vertex, bi1, bi2)
                if i == j or i >= N or j >= N:
                    continue
                # Every factor the prime-exponent head can emit is a divisor of
                # gcd(size_i, size_j) over the sizes the policy SEES, and the
                # env additionally screens ``factor | ps[bi2]`` for every
                # invar. Admitting the pair therefore promises that EVERY
                # divisor > 1 of that joint gcd is live-legal.
                g_nom = n1
                for ps in primal_shapes:
                    g_nom = math.gcd(g_nom, int(ps[bi2]))
                if g_nom <= 1:
                    continue  # only factor 1 reachable -> a guaranteed no-op
                ok = True
                for st, dm in zip(faces, face_diag):
                    if not dm[i, j]:
                        ok = False
                        break
                    base, span = diag_pair_factor_space(st, i, j)
                    # base > 1 is an already-coupled pair: its legal factors are
                    # base*d, which the head (emitting bare divisors) cannot
                    # express. g_nom | span makes every emittable divisor legal.
                    if base != 1 or span % g_nom != 0:
                        ok = False
                        break
                if ok:
                    pair[i, j] = True
                    pair[j, i] = True  # the row format canonicalises the order
        return pair, comp

    def masks(self, candidates=None):
        """``(pair_valid, compress_valid)`` for every vertex, 1-based rows.

        Shapes ``(total_v + 1, N, N)`` and ``(total_v + 1, N)``, float32 so
        they drop straight into the policy's masked softmaxes. ``candidates``
        restricts the probing to the vertices still available (everything else
        stays all-zero); ``None`` probes every un-eliminated vertex.
        """
        N = self.max_axes
        pair = np.zeros((self.total_v + 1, N, N), dtype=np.float32)
        comp = np.zeros((self.total_v + 1, N), dtype=np.float32)
        if candidates is None:
            candidates = [v for v in range(1, self.total_v + 1)
                          if v not in self._eliminated]
        for v in candidates:
            v = int(v)
            p, c = self.vertex_mask(v)
            pair[v] = p.astype(np.float32)
            comp[v] = c.astype(np.float32)
        return pair, comp


def masked_micro_chooser(pick, max_dims: int = 8, max_axes: int = 8,
                         kinds: tuple = ("mean",)):
    """Wrap ``pick`` into the slot callable graphax invokes per face slot.

    ``pick(tensor, actions)`` receives the live operand and the list of actions
    that are legal ON THAT OPERAND, and returns one of them (or ``None`` to
    leave the operand exact). Because the list is enumerated from the tensor
    itself, an illegal action is not merely unlikely -- it is unrepresentable.

    Returning the chosen action rather than a transformed tensor is what keeps
    it visible to the AOJ's transform log; a callable that returns a tensor is
    applied but never recorded.
    """
    def _chooser(st):
        actions = (legal_diag_actions(st, max_dims)
                   + legal_compress_actions(st, max_axes, kinds))
        return pick(st, actions)

    return _chooser
