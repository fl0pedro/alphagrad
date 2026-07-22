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
