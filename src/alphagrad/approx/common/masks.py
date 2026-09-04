"""Action-mask helpers shared between the approx-RL trainers.

The trainers differ in their action-space layout (legacy unrolled `sp_type ×
vertex` vs. the new factorised `vertex × pair × factor`), but the underlying
shape inspection — "for each vertex, what is the output ndim and the smallest
input ndim?" — is the same. These helpers expose that primitive plus a few
convenience builders for the most common mask layouts.
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

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


# FORCE-REV (import-time constant so it is static under jit): with the
# flag on, vertex_avail_at_step keeps only the HIGHEST remaining vertex.
import os as _os
_FORCE_REV = _os.environ.get("ALPHAGRAD_FORCE_REV_ORDER", "0") == "1"
if _FORCE_REV:
    print("[cfg] FORCE REV ORDER: vertex choice pinned to reverse "
          "elimination; only approximations are learned", flush=True)


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
    avail = vertex_valid_static * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))
    if _FORCE_REV:
        # Keep ONLY the highest-indexed available vertex ('rev' order:
        # [n, ..., 2, 1]). All-zero avail (terminal) stays all-zero:
        # argmax lands on 0 and avail[0] is 0 there.
        _score = avail * (jnp.arange(total_v, dtype=jnp.float32) + 1.0)
        _top = jnp.argmax(_score)
        avail = jnp.zeros_like(avail).at[_top].set(
            (avail[_top] > 0).astype(avail.dtype))
    return avail


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
    An IMPLICIT side (``axis is None``) is NO LONGER excluded: such a dim is
    broadcast-constant, not absent, so block-masking it is a valid tightening
    that needs no physical axis (measured bit-identical to the dense oracle).
    The one implicit case that stays out -- an already-coupled pure diagonal --
    is removed by the ``span <= 1`` no-op filter below.

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
            # COMPRESS -> DIAG IS LEGAL (2026-07-28).
            #
            # This used to skip any pair with an implicit (axis=None) side,
            # mirroring a graphax guard that has since been shown wrong by
            # measurement: mixed and both-implicit-dense pairs densify
            # bit-identically to the dense block-mask oracle, and end-to-end
            # jacve with [Compress, Diag] matches at cos = 1.0. An implicit dim
            # is UNIFORM (broadcast-constant), not absent, so block-masking it
            # is a real tightening that costs no val axis.
            #
            # This mask mattered beyond correctness: COMPRESS made dims
            # implicit, which made DIAG illegal, which eventually made DIAG
            # globally illegal at a vertex -- while QUANT is never masked at
            # all. So the only always-available operator was also the
            # annihilating one, and the environment ratcheted the policy
            # toward destruction. Admitting these pairs removes that ratchet.
            #
            # Still excluded, by the span<=1 no-op filter below rather than
            # here: an already-COUPLED implicit pair is a pure diagonal (c*I);
            # blocking it would widen its support.
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


def dim_logical_sizes(st, max_dims: int = 8) -> np.ndarray:
    """``(max_dims,)`` int32 LOGICAL sizes of ``out_dims ++ primal_dims``.

    The size vector indexed the way ``Diag(i, j)``, :func:`diag_valid_mask`,
    :func:`diag_pair_factor_space` and :func:`rule_is_legal` are indexed -- so
    ``gcd(sizes[i], sizes[j])`` is exactly the ``span`` those functions compute
    for a free pair. Entries past the tensor's rank stay 0, which is what marks
    them invalid when they reach the head as ``AxisTokenFeatures``.
    """
    dims = tuple(st.out_dims) + tuple(st.primal_dims)
    out = np.zeros((int(max_dims),), dtype=np.int32)
    for a, d in enumerate(dims[:int(max_dims)]):
        out[a] = int(d.logical_size)
    return out


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
                           kinds: tuple = ("mean",),
                           strict: bool = False) -> list:
    """Every single-axis ``Compress`` graphax will accept on ``st``.

    ``strict`` swaps :func:`compress_valid_mask` (bounded by ``val.ndim``) for
    :func:`compress_slot_mask` (bounded by the CANONICAL SLOT count, which is
    the bound ``apply_compress`` actually enforces). Default ``False`` keeps
    the historical enumeration byte-for-byte -- ``masked_micro_chooser`` and
    ``face_masks`` both read it, and both are on paths that must not move when
    ``--per-face-masks`` is off.
    """
    from graphax.sparse.micro_actions import Compress

    mask = (compress_slot_mask(st, max_axes) if strict
            else compress_valid_mask(st, max_axes))
    return [Compress(axes=(a,), kind=k)
            for a in range(max_axes) if mask[a] for k in kinds]


# ---------------------------------------------------------------------------
# PER-FACE DIAG masking  (``--diag-per-face``)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS, and what it is NOT fixing.
#
# The unified face head does not choose a DIAG factor: ``_rows``
# (unified_face_policy.py) hardcodes ``factor = gcd(N_i, N_j)`` over the
# vertex's NOMINAL jaxpr sizes, and the (i, j) pair comes from a mask built by
# ``face_masks`` against a PROBE of the face. Both are per-VERTEX quantities
# dressed up as per-face ones: the probe records the tensor handed to the
# per-vertex ``transforms`` callable, while the rule is then applied to the
# face's THREE slot operands (lhs / rhs / res), whose index structures differ
# from the probe's and from each other.
#
# MEASURED (NeuralNetwork, all three slots planted with the mask-admitted pair,
# 2026-08-27). Of 155 live rejections in ``rule_is_legal``:
#
#   128 (83%)  the pair is ALREADY COUPLED at exactly this factor
#              (base = factor, span = 1) -- an idempotent re-request, and a
#              correct skip: applying it again would change nothing;
#    20        one side is coupled to a DIFFERENT partner;
#     8        the operand has rank 0 at this point;
#     0        any factor clause (``factor % base``, ``d == 1``, ``span % d``).
#
# So the factor arithmetic was never the blocker -- the hypothesis that
# ``d = factor // base == 1`` holds by construction is FALSE, because a free
# pair returns ``base = 1``, not ``base = gcd``. What is missing is that the
# requested (pair, factor) is not drawn from THIS operand's own legal set.
#
# ``diag_pair_factor_space`` IS that set -- legal factors are ``base * d`` for
# divisors ``d > 1`` of ``span`` -- and the hook below is the only place in the
# system where the live operand exists. So this is where the per-face mask has
# to be applied. When enabled, an illegal DIAG is PROJECTED onto the operand's
# legal set instead of being dropped:
#
#   1. keep the requested pair when it is legal here, and snap the factor to a
#      legal one;
#   2. otherwise fall back to a legal pair on this operand (deterministic
#      order: prefer the requested ``i``, then the requested ``j``, then
#      ascending index) and snap the factor there;
#   3. otherwise leave the operand exact -- today's behaviour.
#
# The ``d``-selection rule is DETERMINISTIC and stated, not learned: the factor
# is still not an action in the 94-slot head layout. ``largest`` (the default)
# picks the finest blocks, matching what the head asks for today; ``nearest``
# picks the legal factor closest to the requested one; ``smallest`` picks the
# coarsest, i.e. the mildest approximation. It is a flag so that it can later
# be replaced by a learned factor field without moving anything else.
#
# DEFAULT OFF. With the flag off this module behaves exactly as before, down to
# the telemetry keys -- pinned by ``tests/diag_per_face_test.py``.

_DIAG_PER_FACE = [os.environ.get("ALPHAGRAD_DIAG_PER_FACE", "0")
                  not in ("0", "", "false", "False", "no")]
_DIAG_PER_FACE_RULE = [os.environ.get("ALPHAGRAD_DIAG_PER_FACE_RULE",
                                      "largest")]
_DIAG_PER_FACE_REPAIR_PAIR = [os.environ.get(
    "ALPHAGRAD_DIAG_PER_FACE_REPAIR_PAIR", "1")
    not in ("0", "", "false", "False", "no")]
_DIAG_RULES = ("largest", "nearest", "smallest")


def set_diag_per_face(enabled: bool, rule: str = "largest",
                      repair_pair: bool = True) -> None:
    """Install the ``--diag-per-face`` setting process-wide.

    Also republished to the ENVIRONMENT so the Ray measure actors -- which run
    ``env._callback`` in their own processes and import this module fresh --
    inherit it. Same "one env var read by one module" discipline the quality
    channel uses; two paths that must agree is this codebase's dominant bug
    class.
    """
    rule = str(rule)
    if rule not in _DIAG_RULES:
        raise ValueError(
            f"--diag-per-face-rule must be one of {_DIAG_RULES}, got {rule!r}")
    _DIAG_PER_FACE[0] = bool(enabled)
    _DIAG_PER_FACE_RULE[0] = rule
    _DIAG_PER_FACE_REPAIR_PAIR[0] = bool(repair_pair)
    os.environ["ALPHAGRAD_DIAG_PER_FACE"] = "1" if enabled else "0"
    os.environ["ALPHAGRAD_DIAG_PER_FACE_RULE"] = rule
    os.environ["ALPHAGRAD_DIAG_PER_FACE_REPAIR_PAIR"] = (
        "1" if repair_pair else "0")


def diag_per_face_enabled() -> bool:
    return bool(_DIAG_PER_FACE[0])


def diag_pair_legal_factors(st, i: int, j: int) -> list:
    """THE per-face legal ``Diag.factor`` set for ``(i, j)`` on ``st``.

    Exactly the enumeration :func:`diag_pair_factor_space` describes: ``base *
    d`` for every divisor ``d > 1`` of ``span``, ascending. Empty when the pair
    affords nothing but the no-op -- which is the case that makes an
    already-coupled pair unrepairable rather than merely mis-factored.
    """
    base, span = diag_pair_factor_space(st, i, j)
    if base <= 0 or span <= 1:
        return []
    return [base * d for d in _divisors_above_one(span)]


def _pick_diag_factor(factors: list, want: int, rule: str) -> int | None:
    if not factors:
        return None
    if rule == "largest":
        return factors[-1]
    if rule == "smallest":
        return factors[0]
    return min(factors, key=lambda f: (abs(f - int(want)), f))


def project_diag_to_face(st, rule, *, max_dims: int = 8,
                         factor_rule: str | None = None,
                         repair_pair: bool | None = None):
    """``rule`` re-expressed inside THIS operand's legal set, or ``None``.

    Returns ``None`` when the operand affords no legal DIAG at all -- notably
    for the idempotent case (the pair is already coupled at this granularity,
    so ``span == 1`` and there is nothing finer to ask for) and for a rank-0
    operand. ``None`` means "leave exact", never "apply something illegal".
    """
    from graphax.sparse.micro_actions import Diag

    if factor_rule is None:
        factor_rule = _DIAG_PER_FACE_RULE[0]
    if repair_pair is None:
        repair_pair = _DIAG_PER_FACE_REPAIR_PAIR[0]
    want = int(getattr(rule, "factor", 0))
    vm = diag_valid_mask(st, max_dims)
    i, j = int(rule.i), int(rule.j)
    if 0 <= i < max_dims and 0 <= j < max_dims and vm[i, j]:
        f = _pick_diag_factor(diag_pair_legal_factors(st, i, j), want,
                              factor_rule)
        return None if f is None else Diag(i, j, f)
    if not repair_pair:
        return None
    # The requested pair does not exist on this operand. Deterministic
    # fallback -- a stable order matters more than a clever one, because the
    # policy has to be able to learn what its own action does.
    cands = [(a, b) for a in range(max_dims) for b in range(max_dims)
             if vm[a, b]]
    if not cands:
        return None
    cands.sort(key=lambda ab: (ab[0] != i, ab[1] != j, ab))
    for a, b in cands:
        f = _pick_diag_factor(diag_pair_legal_factors(st, a, b), want,
                              factor_rule)
        if f is not None:
            return Diag(a, b, f)
    return None


# ---------------------------------------------------------------------------
# PER-FACE MASKING FOR EVERY APPROXIMATION  (``--per-face-masks``)
# ---------------------------------------------------------------------------
# WHAT THIS GENERALISES, AND WHY IT IS NOT JUST "--diag-per-face FOR THE OTHER
# TWO OPS".
#
# Approximation legality is decided at THREE layers that disagree about
# granularity:
#
#   L1  NOMINAL / per-VERTEX  -- ``env.decode_vertex_rule_specs`` builds the
#       rule from the jaxpr equation's ``out_shape ++ primal_shape``;
#   L2  ORACLE PROBE          -- ``LiveVertexMaskOracle.face_masks`` is
#       per-face, but it screens the pair with a gcd taken over L1's NOMINAL
#       sizes (``g_nom``), and the head then derives its DIAG factor from the
#       same nominal sizes;
#   L3  APPLY TIME            -- ``rule_is_legal`` reads the LIVE operand the
#       slot actually holds, whose dims are neither L1's nor (necessarily)
#       the probe's.
#
# L3 is the layer that decides, so L1 and L2 must be expressed in L3's terms
# or the policy is choosing in a coordinate system nothing downstream uses.
# Measured consequence on TLM: DIAG applied 0 of 103 requested rules, QUANT
# had no per-face mask at all (``diag_valid_mask``'s note: "QUANT is never
# masked at all"), and COMPRESS's per-face mask existed but its misses were
# dropped silently.
#
# Two independent halves, both behind the one flag:
#
#   (a) THE SIZES THE HEAD READS. ``LiveVertexMaskOracle.face_dim_sizes``
#       returns the LIVE ``logical_size`` of ``out_dims ++ primal_dims`` per
#       face -- index-aligned with ``diag_valid_mask`` /
#       ``diag_pair_factor_space`` / ``rule_is_legal``, i.e. with L3. Handed to
#       the head as that face's ``AxisTokenFeatures.size`` it makes both the
#       ``pair_ok`` (gcd > 1) gate and the hardcoded ``factor = gcd(N_i, N_j)``
#       per-face and, for the free pairs the mask admits, LEGAL BY
#       CONSTRUCTION (a free pair has ``base = 1`` and ``span = gcd``, so
#       ``factor = gcd`` is exactly ``base * span``, the coarsest-to-finest
#       endpoint of :func:`diag_pair_legal_factors`).
#
#       NOTE this is deliberately NOT ``face_features``, which keeps
#       ``val.shape`` -- the PHYSICAL axes of the sparse tensor. Those are not
#       the numbering the wire format, the masks or ``Diag(i, j)`` use (a
#       coupled pair stores two logical dims in one physical axis), so feeding
#       them to the head would align nothing.
#
#   (b) THE APPLY-TIME PROJECTION. ``make_live_masked_hook`` gains the
#       COMPRESS and QUANT counterparts of ``project_diag_to_face``, each
#       picking from the action list :func:`face_legal_actions` enumerates FROM
#       the live operand -- the ``masked_micro_chooser`` property ("an illegal
#       action is unrepresentable"), applied to the per-vertex hook protocol.
#       Every projection is re-verified with :func:`rule_is_legal` before it is
#       applied, so ``skipped_raised`` cannot rise.
#
# DEFAULT OFF, and with the flag off every path here is bypassed: same masks,
# same rules, same telemetry keys, same trace.

_PER_FACE_MASKS = [os.environ.get("ALPHAGRAD_PER_FACE_MASKS", "0")
                   not in ("0", "", "false", "False", "no")]
_PER_FACE_REPAIR_AXIS = [os.environ.get(
    "ALPHAGRAD_PER_FACE_REPAIR_AXIS", "1")
    not in ("0", "", "false", "False", "no")]


def set_per_face_masks(enabled: bool, repair_axis: bool = True) -> None:
    """Install the ``--per-face-masks`` setting process-wide.

    Republished to the ENVIRONMENT for the same reason
    :func:`set_diag_per_face` is: the Ray measure actors import this module
    fresh in their own processes and would otherwise run the flag-off hook
    against a flag-on policy.
    """
    _PER_FACE_MASKS[0] = bool(enabled)
    _PER_FACE_REPAIR_AXIS[0] = bool(repair_axis)
    os.environ["ALPHAGRAD_PER_FACE_MASKS"] = "1" if enabled else "0"
    os.environ["ALPHAGRAD_PER_FACE_REPAIR_AXIS"] = "1" if repair_axis else "0"


def per_face_masks_enabled() -> bool:
    return bool(_PER_FACE_MASKS[0])


# --face-slot-frames (ticket dsnn-3qm.18, defects D2 and D3). A face has
# three operand slots -- lhs (d central / d in_edge), rhs (d out_edge /
# d central), new (d out_edge / d in_edge) -- and they are NOT alike: under a
# scalar loss rhs and new have an EMPTY out side on every face (finding 54)
# while lhs carries the vertex's own out dims. ``slot`` (the default) decodes
# each slot's wire row in the frame of the live tensor that slot is handed
# (``env.make_slot_frame_hook``) and masks each slot of the 94-logit head with
# that slot's own sizes / Diag-pair / Reduce-axis / Quant legality
# (``LiveFaceStream.face_slot_legality`` -> ``UnifiedFacePolicy``).
# ``vertex`` is the pre-ticket behaviour -- one vertex frame and one legality
# vector, probed from the result tensor, broadcast to all three slots -- kept
# only so the flag-off bit-identity gate (ALPHAGRAD_EQ_DUMP) can reach it.
# Same discipline as --per-face-masks: republished to the environment so the
# Ray measure actors decode the wire in the same frame as the trainer.
_FACE_SLOT_FRAMES = [os.environ.get("ALPHAGRAD_FACE_SLOT_FRAMES", "1")
                     not in ("0", "", "false", "False", "no")]


def set_face_slot_frames(enabled: bool) -> None:
    """Install the ``--face-slot-frames`` setting process-wide (and in the
    environment, for the measure actors)."""
    _FACE_SLOT_FRAMES[0] = bool(enabled)
    os.environ["ALPHAGRAD_FACE_SLOT_FRAMES"] = "1" if enabled else "0"


def face_slot_frames_enabled() -> bool:
    return bool(_FACE_SLOT_FRAMES[0])


# The 94-slot face head's dtype field is a BERNOULLI over exactly these two
# (``unified_face_policy._rows``: ``_BF16_SLOT`` / ``_F32_SLOT``), so this --
# not the full ``QUANT_DTYPES`` catalog -- is the QUANT action set a face can
# actually request. Kept here rather than imported from ``heads`` so this
# module stays importable inside the measure actors without pulling equinox.
FACE_QUANT_DTYPES = ("float32", "bfloat16")


def compress_slot_mask(st, max_axes: int = 8) -> np.ndarray:
    """``(max_axes,)`` bool mask of legal ``Compress`` SLOTS on ``st``.

    THE CORRECTION :func:`compress_valid_mask` NEEDS, and the reason this had
    to be found rather than assumed. ``Compress.axes`` are CANONICAL LOGICAL
    SLOTS (``graphax.sparse.micro_actions.canonical_axis_order``), not physical
    ``val`` axes: a coupled pair contributes ONE meta slot for its TWO dims
    (plus a block slot only when it has a block size), and an implicit
    component contributes a slot that resolves to ``None``.
    :func:`compress_valid_mask` bounds the axis by ``val.ndim`` instead, which
    is neither an upper nor a lower bound on the slot count -- so it admits
    slots ``apply_compress`` then rejects with "Compress.axes entry {a} out of
    range".

    MEASURED, not theorised: with the per-face COMPRESS projection snapping
    out-of-range axes to the top of the ``val.ndim`` range, the ``_mlp``
    fixture produced 3 ``skipped_raised`` -- the first non-zero that counter
    has ever shown here. This mask is what takes it back to 0.

    Slots resolving to ``None``, and slots whose physical extent is 1, are
    excluded: ``apply_compress`` returns the operand unchanged for both
    (``if not drops: return st``; "reducing an implicit or extent-1 slot is the
    identity"), so they are guaranteed no-ops -- the same reason
    ``compress_valid_mask`` excludes ``val is None``.
    """
    mask = np.zeros((int(max_axes),), dtype=bool)
    val = getattr(st, "val", None)
    if val is None:
        return mask
    try:
        from graphax.sparse.micro_actions import canonical_axis_order
        slots = canonical_axis_order(st)
    except Exception:
        # Never be LOOSER than the historical mask on a graphax we cannot
        # introspect: fall back to it rather than admitting everything.
        return compress_valid_mask(st, max_axes)
    shape = tuple(getattr(val, "shape", ()) or ())
    for a, phys in enumerate(slots[: int(max_axes)]):
        if phys is None or not (0 <= int(phys) < len(shape)):
            continue
        if int(shape[int(phys)]) > 1:
            mask[a] = True
    return mask


def compress_is_noop(st, rule, *, max_axes: int = 8) -> bool:
    """Would this ``Compress`` leave ``st`` unchanged?

    ``val is None`` (uniform tensor), or every addressed slot is implicit /
    extent-1 -- in both cases ``apply_compress`` returns the operand itself.
    """
    if getattr(st, "val", None) is None:
        return True
    mask = compress_slot_mask(st, max_axes)
    axes = rule.axes if isinstance(rule.axes, tuple) else (rule.axes,)
    return not any(0 <= int(a) < max_axes and mask[int(a)] for a in axes)


def quant_is_noop(st, dtype_name: str) -> bool:
    """Would ``Quant(dtype_name)`` leave ``st`` bit-identical?

    ``apply_quant`` returns the operand UNCHANGED when ``val is None`` (a
    pure-structure Jacobian) or when ``val.dtype`` already is the target. Both
    are legal and both do nothing, which is why QUANT's applied count has
    always looked healthy while carrying no approximation: a face head that
    draws ``float32`` on a ``float32`` operand is scored as having applied an
    approximation it did not make. This is QUANT's exact analogue of DIAG's
    idempotent re-request.
    """
    val = getattr(st, "val", None)
    if val is None:
        return True
    return str(getattr(val, "dtype", "")) == str(dtype_name)


def quant_valid_mask(st, dtypes: tuple = FACE_QUANT_DTYPES) -> np.ndarray:
    """Per-dtype bool mask of QUANTs that are legal AND not a no-op on ``st``.

    THE per-face QUANT mask. Before this, QUANT was the one operator with no
    per-face legality at all -- which is what made it the only always-available
    action, and therefore the one the policy could always fall back on.
    """
    return np.array(
        [bool(quant_chain_ok(st, d)) and not quant_is_noop(st, d)
         for d in dtypes], dtype=bool)


def legal_quant_actions(st, dtypes: tuple = FACE_QUANT_DTYPES,
                        scale_sign: int = 1) -> list:
    """Every ``Quant`` that both fits and CHANGES ``st``, as concrete actions."""
    from graphax.sparse.micro_actions import Quant

    mask = quant_valid_mask(st, dtypes)
    return [Quant(dtype=d, scale_sign=scale_sign)
            for d, ok in zip(dtypes, mask) if ok]


def face_legal_actions(st, *, max_dims: int = 8, max_axes: int = 8,
                       kinds: tuple = ("mean",),
                       quant_dtypes: tuple = ()) -> list:
    """THE operand's own legal action set -- what a per-face mask *is*.

    The enumeration :func:`masked_micro_chooser` hands its picker, factored out
    so the apply-time projection can pick from exactly the same list rather
    than re-deriving legality with a second copy of the rules (two paths that
    must agree is this codebase's dominant bug class).

    ``quant_dtypes`` defaults to EMPTY so the historical chooser -- which never
    offered QUANT -- is unchanged; pass :data:`FACE_QUANT_DTYPES` for the
    per-face action set the 94-slot head can actually express.
    """
    out = (legal_diag_actions(st, max_dims)
           + legal_compress_actions(st, max_axes, kinds))
    if quant_dtypes:
        out += legal_quant_actions(st, quant_dtypes)
    return out


def project_compress_to_face(st, rule, *, max_axes: int = 8,
                             repair_axis: bool | None = None):
    """``rule`` re-expressed inside THIS operand's legal COMPRESS set, or None.

    ``Compress.axes`` are PHYSICAL positions into ``st.val``, and a face's
    live ``val.ndim`` is routinely smaller than the nominal rank the axis the
    head drew was indexed against (a coupled pair stores two logical dims in
    one physical axis, and every prior COMPRESS on the same operand drops one).
    An out-of-range axis is not a wrong *kind* of request, it is the right
    request in the wrong coordinates -- so it is snapped to the nearest legal
    axis rather than dropped.

    Deterministic, and stated: NEAREST legal axis, ties to the LOWER index. A
    stable rule matters more than a clever one, because the policy has to be
    able to learn what its own action does. ``repair_axis=False`` keeps the
    strict behaviour (drop rather than snap).

    ``None`` when the operand affords no COMPRESS at all -- notably the
    ``val is None`` pure-structure Jacobian, where a Compress is accepted by
    graphax but is a guaranteed no-op.
    """
    from graphax.sparse.micro_actions import Compress

    if repair_axis is None:
        repair_axis = _PER_FACE_REPAIR_AXIS[0]
    kind = getattr(rule, "kind", "mean")
    # STRICT: the CANONICAL SLOT range, the bound apply_compress enforces. A
    # repair that snapped into the val.ndim range instead is exactly what
    # produced this module's first-ever non-zero `skipped_raised`.
    legal_axes = [int(a.axes[0])
                  for a in legal_compress_actions(st, max_axes, kinds=(kind,),
                                                  strict=True)]
    if not legal_axes:
        return None
    axes = rule.axes if isinstance(rule.axes, tuple) else (rule.axes,)
    out: list[int] = []
    for a in axes:
        a = int(a)
        if a in legal_axes:
            b = a
        elif not repair_axis:
            continue
        else:
            b = min(legal_axes, key=lambda c: (abs(c - a), c))
        if b not in out:
            out.append(b)
    if not out:
        return None
    return Compress(axes=tuple(out), kind=kind)


def project_quant_to_face(st, rule, *, dtypes: tuple = FACE_QUANT_DTYPES):
    """``rule`` re-expressed inside THIS operand's legal QUANT set, or None.

    THERE IS DELIBERATELY NO DTYPE FALLBACK, and that asymmetry with DIAG /
    COMPRESS is the point. A DIAG projected to a different factor, or a
    COMPRESS to a different axis, is the SAME approximation re-expressed in the
    operand's own coordinates. A QUANT projected to a different dtype is a
    DIFFERENT approximation -- swapping a requested ``float32`` (a no-op on a
    ``float32`` operand) for ``bfloat16`` would invent a numerical change the
    policy never asked for, and attribute its reward to an action the policy
    did not take.

    So QUANT's per-face treatment is a MASK, not a repair: the request stands
    when it is a legal, non-idempotent cast on this operand, and is otherwise
    left exact and counted as a no-op rather than as an approximation.
    """
    if not quant_valid_mask(st, (rule.dtype,))[0]:
        return None
    return rule


def project_rule_to_face(st, rule, *, max_dims: int = 8, max_axes: int = 8,
                         factor_rule: str | None = None,
                         repair_pair: bool | None = None,
                         repair_axis: bool | None = None):
    """Dispatch ``rule`` to its per-kind projection. ``None`` = leave exact."""
    from graphax.sparse.micro_actions import Compress, Diag, Quant

    if isinstance(rule, Diag):
        return project_diag_to_face(st, rule, max_dims=max_dims,
                                    factor_rule=factor_rule,
                                    repair_pair=repair_pair)
    if isinstance(rule, Compress):
        return project_compress_to_face(st, rule, max_axes=max_axes,
                                        repair_axis=repair_axis)
    if isinstance(rule, Quant):
        return project_quant_to_face(st, rule)
    return None


def rule_is_idempotent_noop(st, rule, *, max_dims: int = 8,
                            max_axes: int = 8) -> bool:
    """Would applying ``rule`` to ``st`` change nothing? A CORRECT skip.

    THE HONEST DENOMINATOR. ``applied / requested`` reads an idempotent
    re-request as a failure, which is how DIAG came to look inert: 83% of the
    live DIAG rejections measured on NeuralNetwork were pairs already coupled
    at exactly the requested granularity. The metric that means something is
    ``applied / (requested - idempotent no-ops)``, and this is the predicate
    that splits the denominator.
    """
    from graphax.sparse.micro_actions import Compress, Diag, Quant

    if isinstance(rule, Diag):
        base, span = diag_pair_factor_space(st, rule.i, rule.j)
        # Already coupled at this granularity: nothing finer is legal (span
        # == 1), or the request IS the current meta count (graphax's own
        # definition of a Diag no-op).
        return base > 1 and (span <= 1 or int(rule.factor) == base)
    if isinstance(rule, Compress):
        return compress_is_noop(st, rule, max_axes=max_axes)
    if isinstance(rule, Quant):
        return quant_is_noop(st, rule.dtype)
    return False


def face_rule_is_legal(st, rule, *, max_dims: int = 8,
                       max_axes: int = 8) -> bool:
    """:func:`rule_is_legal`, TIGHTENED to the live operand's own addressing.

    THE per-face legality predicate (``--per-face-masks``). Everything
    ``rule_is_legal`` rejects is still rejected; on top of it:

    * COMPRESS is bounded by the CANONICAL SLOT count rather than ``val.ndim``
      (:func:`compress_slot_mask`) -- the bound ``apply_compress`` enforces,
      and the one whose absence is the only way this hook ever raised;
    * an IDEMPOTENT request is not "legal", it is a no-op
      (:func:`rule_is_idempotent_noop`) and is accounted as one rather than as
      an approximation the policy made.

    Kept separate from :func:`rule_is_legal` rather than folded into it:
    that one is read by ``face_masks``, ``masked_face_transforms`` and the
    flag-off hook, none of which may move.
    """
    from graphax.sparse.micro_actions import Compress

    if not rule_is_legal(st, rule, max_dims=max_dims, max_axes=max_axes):
        return False
    if isinstance(rule, Compress):
        m = compress_slot_mask(st, max_axes)
        axes = rule.axes if isinstance(rule.axes, tuple) else (rule.axes,)
        if not all(0 <= int(a) < max_axes and m[int(a)] for a in axes):
            return False
    return not rule_is_idempotent_noop(st, rule, max_dims=max_dims,
                                       max_axes=max_axes)


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

    def advance(self, vertex: int, rules=(), face_wires=None):
        """Commit one elimination so later masks see the post-step graph.

        ``rules`` must be the transforms the env ACTUALLY applied at this
        vertex (``rule_specs_to_transforms``' output for it) -- an earlier
        approximation changes the structure of every downstream edge, so a mask
        computed against an exact-elimination replay would be wrong.

        ``face_wires`` (probe-path hygiene, docs/FACE_LATENT_INFO_LOSS.md
        section 4): the step's REALIZED per-face decisions as
        ``(face_rows (F, S, 3), face_skips (F,))`` wire rows. Under
        --live-faces the per-vertex ``rules`` are all-exact and the real
        approximations live in these wires, so a replay that drops them
        rebuilds an exact graph the measurement never built -- targets
        decoded from it drift as soon as the face head approximates.
        Decoding mirrors ``live_faces.LiveFaceStream._decided`` (position-
        independent wire decode, ``make_live_masked_hook`` wrap, SKIP_FACE
        for skips), with keys from :func:`graphax.core.faces_of` on each
        dispatch mode's own live graph. Default ``None`` = the historical
        exact-face replay every mask consumer keeps.
        """
        from graphax.sparse.elemental.dispatch import (
            approx_active, set_approx_active,
        )

        vertex = int(vertex)
        prev = approx_active()
        try:
            for mode, incr in self._incrs.items():
                set_approx_active(mode)
                ft = None
                if face_wires is not None:
                    try:
                        ft = self._face_ft(incr, vertex, face_wires)
                    except Exception:
                        # A face decode must never take the replay down --
                        # worst case is the historical exact-face behaviour.
                        ft = None
                incr.eliminate(vertex, rules=tuple(rules),
                               face_transforms=ft or None)
        finally:
            set_approx_active(prev)
        self._eliminated.add(vertex)

    def _face_ft(self, incr, vertex, face_wires):
        """``{face_key: slots | SKIP_FACE}`` decoded from realized wires.

        Returns ``None`` for an all-inert step (no skip, every row still
        the -1 fill) so the exact path pays no ``faces_of`` enumeration.
        """
        face_rows, face_skips = face_wires
        fr = np.asarray(face_rows)
        fs = np.asarray(face_skips)
        if not (np.any(fs == 1) or np.any(fr >= 0)):
            return None
        from graphax import SKIP_FACE
        from graphax.core import faces_of
        from alphagrad.approx.env import (
            FACE_SLOTS, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs)

        keys = faces_of(incr.graph, incr.tgraph, int(vertex), incr.jaxpr)
        ft: dict = {}
        for f in range(min(len(keys), fr.shape[0])):
            if int(fs[f]) == 1:
                ft[keys[f]] = SKIP_FACE
                continue
            slots = []
            for s in range(FACE_SLOTS):
                row = [[int(x) for x in fr[f][s]]] + [
                    [-1, -1, 0]] * (MAX_RULES_PER_VERTEX - 1)
                try:
                    rls = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row)
                except Exception:
                    rls = ()
                slots.append(make_live_masked_hook(tuple(rls))
                             if rls else None)
            if any(sl is not None for sl in slots):
                ft[keys[f]] = tuple(slots)
        return ft or None

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
                try:
                    _eliminate_vertex(
                        int(vertex), incr.jaxpr, graph, tgraph, incr.vo, False,
                        transforms=(_record,), face_transforms=None,
                    )
                except Exception as exc:
                    # graphax cannot trace this elimination (observed on the
                    # xent graph: an add of a tensor and its own transpose,
                    # (16,10,784,256) vs (16,10,256,784), from the open
                    # canonical-output-order gap in _normalize_inputs).
                    # A vertex that cannot be eliminated has no LEGAL
                    # approximation, so return the faces seen so far and let
                    # vertex_mask's all() intersection admit nothing -- the
                    # policy then eliminates it exactly. Never transpose a
                    # side to make the add fit: guessing the axis order
                    # silently corrupts the Jacobian the reward is built on.
                    _record_probe_failure(int(vertex), bool(approx), exc)
        finally:
            set_approx_active(prev)
            # Throw away the equations the probe traced; nothing ever
            # materialises this builder's jaxpr, but the list would grow
            # without bound over an episode.
            del incr.trace.frame.tracing_eqns[n_eqns0:]
        return seen

    def face_features(self, vertex: int, max_faces: int = 8):
        """PER-FACE axis sizes of the LIVE contraction: (F, N) int32, n_faces.

        THE POINT OF THIS METHOD. The approximation head used to be handed the
        vertex's static AxisTokenFeatures plus a face EMBEDDING -- a label,
        carrying no information about the face it names. Every face therefore
        looked identical to the head, which makes a per-face decision
        meaningless: it was approximating a contraction it had never read.

        probe_faces already builds the live SparseTensor for every face (it
        is how :meth: decides legality); this simply keeps the
        SHAPE instead of reducing it to booleans, so the head conditions on the
        actual thing it is approximating. No extra elimination is run -- the
        probe is shared with the mask path.

        Rows >= n_faces are zero padding, matching face_masks.
        """
        N = self.max_axes
        F = int(max_faces)
        sizes = np.zeros((F, N), dtype=np.int32)
        vertex = int(vertex)
        if not (1 <= vertex <= self.total_v) or vertex in self._eliminated:
            return sizes, 0
        faces = self.probe_faces(vertex, approx=True)
        if not faces:
            return sizes, 0
        n_faces = min(len(faces), F)
        for k in range(n_faces):
            st = faces[k]
            val = getattr(st, "val", None)
            shp = tuple(getattr(val, "shape", ()) or ())
            for a, n in enumerate(shp[:N]):
                sizes[k, a] = int(n)
        return sizes, n_faces

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

    def face_dim_sizes(self, vertex: int, max_faces: int = 8):
        """PER-FACE LIVE ``logical_size`` of ``out_dims ++ primal_dims``.

        ``(F, N) int32, n_faces``. THE NUMBERING MATTERS: this is the
        concatenated-dims numbering :class:`graphax.sparse.micro_actions.Diag`
        uses, which is also what :func:`diag_valid_mask`,
        :func:`diag_pair_factor_space` and :func:`rule_is_legal` index -- i.e.
        the numbering the decision is ACTUALLY made in at apply time. Handed to
        the head as a face's ``AxisTokenFeatures.size`` it puts the head's
        ``gcd``-derived factor and its ``pair_ok`` gate in the same coordinate
        system as the mask that admits the pair.

        Contrast :meth:`face_features`, which keeps ``val.shape`` -- the
        PHYSICAL axes. Those are a different (shorter, and order-dependent)
        list, so they align with nothing the policy emits.

        Rows ``>= n_faces`` are zero padding, matching :meth:`face_masks`.
        """
        _p, _c, sizes, _q, n = self.face_masks_and_sizes(vertex, max_faces)
        return sizes, n

    def face_masks(self, vertex: int, max_faces: int = 8):
        """PER-FACE ``(pair_valid (F, N, N), compress_valid (F, N), n_faces)``.

        The per-face counterpart of :meth:`vertex_mask`. Same env screens (wire
        format, full-reduction cap, "every emittable divisor is legal"), but the
        per-face masks are kept SEPARATE instead of AND-ed together.

        That difference is the point of per-face approximation: a per-vertex rule
        list is applied uniformly to every face, so a pair is admissible only if
        it fits ALL of them -- which is why per-vertex DIAG is structurally
        almost always illegal. A per-face slot only has to fit ITS OWN operand,
        so the per-face masks are a strict superset of the intersected one.

        Faces are index-aligned across ``_DISPATCH_MODES`` (``probe_faces``
        returns them in ``_eliminate_vertex`` visit order, the same for both), so
        face ``k`` is intersected only with face ``k`` of the other mode -- an
        action must still be legal on BOTH dispatch paths, just not on other
        faces. Rows ``>= n_faces`` stay all-zero padding.

        Deliberately a separate method rather than a refactor of
        :meth:`vertex_mask`: that one is on the live per-vertex training path and
        must not be perturbed.
        """
        pair, comp, _sizes, _quant, n_faces = self.face_masks_and_sizes(
            vertex, max_faces, per_face=False)
        return pair, comp, n_faces

    def face_masks_and_sizes(self, vertex: int, max_faces: int = 8, *,
                             per_face: bool = True,
                             quant_dtypes: tuple = FACE_QUANT_DTYPES):
        """``(pair (F,N,N), comp (F,N), sizes (F,N) int32, quant (F,), n)``.

        ONE probe pass for all four outputs. :meth:`face_masks` already runs
        ``probe_faces`` once per dispatch mode and those probes trace a whole
        elimination, so deriving the sizes and the QUANT mask from a SEPARATE
        call would have added 50% to the oracle's host cost for data the same
        probe already holds.

        ``per_face`` selects the screen:

        * ``False`` -- exactly :meth:`face_masks`, byte for byte: a DIAG pair
          is admitted when every emittable divisor of the NOMINAL gcd is legal
          on every mode's operand. ``sizes`` and ``quant`` come back as zeros
          (nothing consumes them on that path).
        * ``True`` (``--per-face-masks``) -- the same screen with the nominal
          gcd replaced by ``g_face``, the gcd over THIS FACE's own live
          logical sizes, which is what the head will derive its factor from
          once it is handed those sizes. The two must be the same quantity or
          the mask admits a pair whose factor the operand then rejects.

          Soundness across dispatch modes is unchanged in form: the pair is
          admitted only when ``span % g_face == 0`` on EVERY mode, and since a
          free pair has ``span = gcd(N_i, N_j)`` that implies ``g_face``
          divides both dims on every mode -- so ``factor = g_face`` is legal
          everywhere, exactly as ``g_nom`` was.

        ``quant[k]`` is 1 iff at least one dtype in ``quant_dtypes`` is a
        legal, NON-IDEMPOTENT cast on every mode's operand for face ``k``.
        That is the per-face QUANT mask that did not exist before: QUANT was
        the only operator with no per-face legality, hence the only one always
        available.
        """
        from alphagrad.approx.env import diag_row_to_pair

        N = self.max_axes
        F = int(max_faces)
        pair = np.zeros((F, N, N), dtype=bool)
        comp = np.zeros((F, N), dtype=bool)
        sizes = np.zeros((F, N), dtype=np.int32)
        quant = np.zeros((F,), dtype=bool)
        vertex = int(vertex)
        if not (1 <= vertex <= self.total_v) or vertex in self._eliminated:
            return pair, comp, sizes, quant, 0

        eqn = self.jaxpr.eqns[vertex - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            return pair, comp, sizes, quant, 0
        out_shape = tuple(eqn.outvars[0].aval.shape)
        out_len = len(out_shape)
        primal_shapes = [
            tuple(iv.aval.shape) for iv in eqn.invars if hasattr(iv, "aval")
        ]
        if not primal_shapes:
            return pair, comp, sizes, quant, 0

        probes = {m: self.probe_faces(vertex, approx=m)
                  for m in self._DISPATCH_MODES}
        per_mode = [fs for fs in probes.values() if fs]
        if not per_mode:
            return pair, comp, sizes, quant, 0
        n_faces = min(min(len(fs) for fs in per_mode), F)
        # The size source is the approx=True graph -- the one
        # ``vertex_elimination_jaxpr`` builds for the op counts, and the one
        # ``face_features`` already reads. Falls back to the other mode only
        # if that probe produced nothing at all.
        sizes_src = probes.get(True) or probes.get(False) or []

        edge_phys_axes = out_len + max((len(ps) for ps in primal_shapes), default=0)
        n_primal = min(len(ps) for ps in primal_shapes)

        for k in range(n_faces):
            faces_k = [fs[k] for fs in per_mode]

            if per_face:
                if k < len(sizes_src):
                    sizes[k] = dim_logical_sizes(sizes_src[k], N)
                qm = np.ones((len(quant_dtypes),), dtype=bool)
                for st in faces_k:
                    qm &= quant_valid_mask(st, quant_dtypes)
                quant[k] = bool(qm.any())

            # --- COMPRESS: vertex_mask's screens, this face only ------------
            if edge_phys_axes > 1:
                face_comp = [compress_valid_mask(st, N) for st in faces_k]
                for a in range(N):
                    if a >= out_len and any(
                        a - out_len >= len(ps) for ps in primal_shapes
                    ):
                        continue
                    if all(m[a] for m in face_comp):
                        comp[k, a] = True

            # --- DIAG: vertex_mask's screens, this face only -----------------
            face_diag = [diag_valid_mask(st, N) for st in faces_k]
            for bi1 in range(out_len):
                n1 = int(out_shape[bi1])
                for bi2 in range(n_primal):
                    i, j = diag_row_to_pair(self.jaxpr, vertex, bi1, bi2)
                    if i == j or i >= N or j >= N:
                        continue
                    if per_face:
                        # The gcd the HEAD will compute, over the sizes it will
                        # be handed. Same quantity on both sides by
                        # construction -- that is the whole fix.
                        g_ref = math.gcd(int(sizes[k, i]), int(sizes[k, j]))
                    else:
                        g_ref = n1
                        for ps in primal_shapes:
                            g_ref = math.gcd(g_ref, int(ps[bi2]))
                    if g_ref <= 1:
                        continue
                    ok = True
                    for st, dm in zip(faces_k, face_diag):
                        if not dm[i, j]:
                            ok = False
                            break
                        base, span = diag_pair_factor_space(st, i, j)
                        if base != 1 or span % g_ref != 0:
                            ok = False
                            break
                    if ok:
                        pair[k, i, j] = True
                        pair[k, j, i] = True
        return pair, comp, sizes, quant, n_faces

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
                         kinds: tuple = ("mean",),
                         quant_dtypes: tuple = ()):
    """Wrap ``pick`` into the slot callable graphax invokes per face slot.

    ``pick(tensor, actions)`` receives the live operand and the list of actions
    that are legal ON THAT OPERAND, and returns one of them (or ``None`` to
    leave the operand exact). Because the list is enumerated from the tensor
    itself, an illegal action is not merely unlikely -- it is unrepresentable.

    Returning the chosen action rather than a transformed tensor is what keeps
    it visible to the AOJ's transform log; a callable that returns a tensor is
    applied but never recorded.

    ``quant_dtypes`` defaults to EMPTY, i.e. the historical DIAG+COMPRESS
    enumeration; pass :data:`FACE_QUANT_DTYPES` to include the QUANTs the
    94-slot face head can express. The enumeration itself now lives in
    :func:`face_legal_actions`, which the apply-time projection also picks
    from -- one definition of "legal on this operand", not two.
    """
    def _chooser(st):
        return pick(st, face_legal_actions(
            st, max_dims=max_dims, max_axes=max_axes, kinds=kinds,
            quant_dtypes=quant_dtypes))

    return _chooser


# ---------------------------------------------------------------------------
# PER-FACE approximation through jacve (the MVP path)
# ---------------------------------------------------------------------------
# graphax's per-vertex ``transforms`` entry accepts a CALLABLE, and invokes it
# ONCE PER FACE, handing it that face's accumulated contraction -- verified:
# a callable registered on one vertex of `tanh(x@y)*sin(x@y)` fires twice with
# different live layouts. So per-path approximation does NOT need the AOJ
# measurement path: jacve already provides the granularity, and the callable is
# the only place the operand's real index structure is known.
#
# NOTE the per-vertex callable protocol differs from the per-face slot one:
# here the return value IS the tensor, so "leave exact" means returning the
# operand unchanged. Returning None crashes graphax's consistency assert.


def quant_chain_ok(st, dtype_name: str) -> bool:
    """Is quantising ``st`` to ``dtype_name`` a legal CHAIN?

    Measured on device over the full catalog: every dtype casts fine from
    float32, and the only failing chains are narrow-float -> integer
    (``float8_*`` / ``float4_*`` -> ``int*``/``uint*``/``bool``), which raise
    ``TypePromotionError`` inside ``apply_quant``. Everything else composes.
    """
    val = getattr(st, "val", None)
    if val is None:
        return True
    cur = str(getattr(val, "dtype", ""))
    narrow_float = cur.startswith("float8") or cur.startswith("float4")
    to_integral = (dtype_name.startswith("int") or dtype_name.startswith("uint")
                   or dtype_name == "bool")
    return not (narrow_float and to_integral)


def rule_is_legal(st, rule, *, max_dims: int = 8, max_axes: int = 8) -> bool:
    """Is ``rule`` legal on the LIVE tensor ``st``?"""
    from graphax.sparse.micro_actions import Compress, Diag, Quant

    if isinstance(rule, Diag):
        if not (0 <= rule.i < max_dims and 0 <= rule.j < max_dims):
            return False
        if not diag_valid_mask(st, max_dims)[rule.i, rule.j]:
            return False
        base, span = diag_pair_factor_space(st, rule.i, rule.j)
        if span <= 1 or base <= 0 or rule.factor % base:
            return False
        d = rule.factor // base
        return d > 1 and span % d == 0
    if isinstance(rule, Compress):
        mask = compress_valid_mask(st, max_axes)
        axes = rule.axes if isinstance(rule.axes, tuple) else (rule.axes,)
        return bool(axes) and all(0 <= a < max_axes and mask[a] for a in axes)
    if isinstance(rule, Quant):
        return quant_chain_ok(st, rule.dtype)
    return False


def hook_rule_is_legal(st, rule, *, max_dims: int = 8,
                       max_axes: int = 8) -> bool:
    """THE legality predicate :func:`make_live_masked_hook` applies.

    ``--per-face-masks`` on: :func:`face_rule_is_legal` (the slot-count bound
    on COMPRESS, idempotent requests refused); off: :func:`rule_is_legal`.
    Module-level so :func:`slot_legality` can hand the head EXACTLY the
    verdict the hook will reach -- one predicate, not two that must agree.
    """
    if _PER_FACE_MASKS[0]:
        return face_rule_is_legal(st, rule, max_dims=max_dims,
                                  max_axes=max_axes)
    return rule_is_legal(st, rule, max_dims=max_dims, max_axes=max_axes)


class SlotLegality(NamedTuple):
    """ONE face slot's legal action set, read off its live tensor.

    ``sizes`` -- ``dim_logical_sizes``, the numbering ``Diag(i, j)`` and the
    wire use; ``n_out`` -- how many of those are out dims (the wire's
    ``bi2 = j - n_out``); ``pair[i, j]`` -- the head's Diag on ``(i, j)`` with
    its own factor ``gcd(sizes[i], sizes[j])`` is applied by the hook;
    ``comp[a]`` -- ``Compress(axes=(a,))`` decodes in this slot's frame AND
    is applied by the hook; ``quant[d]`` -- ``FACE_QUANT_DTYPES[d]`` is a
    legal, non-idempotent cast.
    """
    sizes: np.ndarray    # (N,) int32
    n_out: int
    pair: np.ndarray     # (N, N) bool
    comp: np.ndarray     # (N,) bool
    quant: np.ndarray    # (len(FACE_QUANT_DTYPES),) bool


def slot_legality(st, max_axes: int = 8,
                  dtypes: tuple = FACE_QUANT_DTYPES) -> SlotLegality:
    """Per-slot legality FROM THE SLOT'S OWN LIVE TENSOR (ticket .18, D3).

    Every entry is the answer of :func:`hook_rule_is_legal` -- the predicate
    the apply-time hook uses -- to the concrete rule the 94-logit head would
    put on the wire for that choice, after the slot-frame decode
    (``env.decode_rule_specs_in_frame``) has admitted it. So "the mask admits
    it" and "the hook applies it" are the same statement, slot by slot; that
    equality is what ``tests/face_slot_frames_test.py`` pins and what .59's
    apply-rate-equals-request-rate test can stand on.
    """
    from graphax.sparse.micro_actions import Compress, Diag

    N = int(max_axes)
    sizes = dim_logical_sizes(st, N)
    n_out = len(st.out_dims)
    n_log = n_out + len(st.primal_dims)
    pair = np.zeros((N, N), dtype=bool)
    comp = np.zeros((N,), dtype=bool)
    for i in range(min(n_log, N)):
        for j in range(min(n_log, N)):
            if i == j or (i < n_out) == (j < n_out):
                continue  # a Jacobian diagonal ties one OUT to one PRIMAL
            io, jp = (i, j) if i < n_out else (j, i)
            g = math.gcd(int(sizes[i]), int(sizes[j]))
            if g <= 1:
                continue  # the decoder drops 0/1 factors: a no-op
            pair[i, j] = hook_rule_is_legal(
                st, Diag(i=io, j=jp, factor=g), max_dims=N, max_axes=N)
    for a in range(min(n_log, N)):
        comp[a] = hook_rule_is_legal(
            st, Compress(axes=(a,), kind="mean"), max_dims=N, max_axes=N)
    return SlotLegality(sizes=sizes, n_out=int(n_out), pair=pair, comp=comp,
                        quant=quant_valid_mask(st, dtypes))


def couple_quant_rules(rules, applied_dtype=None):
    """Quant contraction coupling: **one quantization per turn**, and the
    second operand of a contraction INHERITS the first's dtype.

    The hardware compat matrix is largely self-only (int4 only dots with int4),
    so letting the policy pick an independent dtype for the post operand mostly
    produces an illegal contraction. Rather than mask that choice away, the
    spec's resolution is to make it automatic: the first Quant in a turn is the
    policy's decision, and any later Quant in the same turn is REWRITTEN to the
    same dtype (the "inherit") — leaving the genuinely-new choice for the
    result edge on the next turn.

    Returns ``(coupled_rules, dtype_in_force)``. ``applied_dtype`` carries the
    dtype already in force from earlier in the same turn (None = free choice).
    """
    from graphax.sparse.micro_actions import Quant

    out, in_force = [], applied_dtype
    for r in rules:
        if isinstance(r, Quant):
            if in_force is None:
                in_force = r.dtype
                out.append(r)
            elif r.dtype == in_force:
                out.append(r)            # already consistent
            else:
                # inherit: same dtype, keep the policy's sign choice
                out.append(Quant(dtype=in_force,
                                 scale_sign=getattr(r, "scale_sign", 1)))
        else:
            out.append(r)
    return tuple(out), in_force


# --- measured-elimination arming -------------------------------------------
# One hook OBJECT is invoked by several consumers: the structural face-enum
# replay (env._face_transforms_for_order), the tokenizer replay
# (env._incremental_stream_tokens), the optional count pass, and the MEASURED
# jacve trace. A sink that counted all of them would report 2-3x the truth --
# a fabricated "reality" number, which is worse than the key being absent.
# Hooks built with ``gated=True`` therefore bump their counters only while
# :func:`arm_face_counts` is in force; env.py arms exactly around the measured
# trace. Ungated hooks (direct callers, tests) keep counting unconditionally.
# A DEPTH counter, not a flag, so nesting cannot disarm an outer scope.
_COUNT_ARMED = [0]


def arm_face_counts() -> None:
    """Enter a scope whose gated hook invocations ARE the measurement."""
    _COUNT_ARMED[0] += 1


def disarm_face_counts() -> None:
    _COUNT_ARMED[0] = max(0, _COUNT_ARMED[0] - 1)


def face_counts_armed() -> bool:
    return _COUNT_ARMED[0] > 0


def make_live_masked_hook(rules, *, max_dims: int = 8, max_axes: int = 8,
                          stats: dict | None = None, gated: bool = False):
    """Wrap ``rules`` into the per-vertex callable graphax applies PER FACE.

    Each requested rule is applied iff it is legal on *this* face's operand;
    otherwise it is skipped and the operand passes through untouched. A face on
    which nothing is legal is therefore left exact -- which is the correct
    behaviour when the action is fully masked, and is why this cannot raise
    TRANSFORM DID NOT FIT.

    ``stats`` (optional dict) accumulates ``applied`` / ``skipped`` counts so a
    run can report how much of the policy's intent actually survived masking
    rather than silently approximating nothing. ``gated=True`` restricts those
    counts to an :func:`arm_face_counts` scope -- see the note above; use it
    whenever the same hook is replayed on graphs that are not the measured one.
    """
    from graphax.sparse.micro_actions import (
        apply_compress, apply_diag, apply_quant, Compress, Diag, Quant)

    def _bump(key):
        if stats is None or (gated and not _COUNT_ARMED[0]):
            return
        stats[key] = stats.get(key, 0) + 1

    coupled, _ = couple_quant_rules(rules)

    def _kind_of(rule):
        """Telemetry label for a rule — the REALITY histogram's bucket."""
        if isinstance(rule, Diag):
            return "diag"
        if isinstance(rule, Compress):
            return "compress"
        if isinstance(rule, Quant):
            return "quant"
        return "other"

    def _project_on(rule) -> bool:
        """Is the per-face projection armed for THIS rule's kind?

        ``--diag-per-face`` is the DIAG-only predecessor and stays exactly
        that; ``--per-face-masks`` arms all three. Both off = the historical
        drop-on-illegal hook.
        """
        if _PER_FACE_MASKS[0]:
            return True
        return isinstance(rule, Diag) and _DIAG_PER_FACE[0]

    def _legal(st, rule):
        """The arm's legality predicate. ONE definition, used by BOTH the
        initial test and the post-projection re-verify -- a projection cleared
        by a laxer predicate than the one that rejected the original is how a
        repair turns into a raise."""
        return hook_rule_is_legal(st, rule, max_dims=max_dims,
                                  max_axes=max_axes)

    def _hook(st):
        cur = st
        for rule in coupled:
            _kind = _kind_of(rule)
            live = rule
            if not _legal(cur, live):
                # PER-FACE MASKING. The requested rule was chosen against the
                # vertex's nominal sizes and a probe of the face; this operand
                # has its own legal set, so ask for the nearest thing in it
                # rather than dropping the action. Off by default.
                _alt = None
                if _project_on(rule):
                    _alt = project_rule_to_face(cur, rule, max_dims=max_dims,
                                                max_axes=max_axes)
                    if _alt is not None and not _legal(cur, _alt):
                        # The projection is meant to make this unreachable;
                        # never apply an action the mask has not cleared.
                        _alt = None
                if _alt is None:
                    _bump("skipped")
                    _bump(f"skipped_{_kind}")
                    if _project_on(rule) and rule_is_idempotent_noop(
                            cur, rule, max_dims=max_dims, max_axes=max_axes):
                        # Distinguish the idempotent re-request (DIAG: already
                        # coupled at exactly this granularity; COMPRESS: every
                        # addressed slot implicit or extent-1; QUANT: already
                        # this dtype) from a genuine miss. Without this split
                        # `applied_fraction` reads a correct no-op as a
                        # failure, which is how DIAG came to look inert -- and
                        # how QUANT came to look like it always worked.
                        _bump(f"skipped_{_kind}_noop")
                    continue
                if _alt is not rule:
                    live = _alt
                    _bump(f"repaired_{_kind}")
            try:
                if isinstance(live, Diag):
                    cur = apply_diag(cur, live)
                elif isinstance(live, Compress):
                    cur = apply_compress(cur, live)
                elif isinstance(live, Quant):
                    cur = apply_quant(cur, live)
                else:
                    _bump("skipped")
                    _bump(f"skipped_{_kind}")
                    continue
                _bump("applied")
                _bump(f"applied_{_kind}")
            except ValueError:
                # The mask is meant to make this unreachable; if a case slips
                # through, leaving the operand exact is strictly better than
                # killing the episode's measurement.
                _bump("skipped_raised")
                _bump(f"skipped_{_kind}")
        return cur

    return _hook


# ---------------------------------------------------------------------------
# Per-PATH / per-SLOT / per-EDGE addressing
# ---------------------------------------------------------------------------
# A face key is ``(in_edge_vid, out_edge_vid)``. With pre1/pre2 and post1/post2
# a vertex has four local paths, and graphax's face_transforms gives each of
# them three independently addressable slots:
#
#     lhs -> pre_val   (the in_edge  Jacobian)  | both BEFORE the contraction
#     rhs -> post_val  (the out_edge Jacobian)  |
#     res -> the contraction result             | after
#
# Two levels of addressing are therefore useful, and both are exact:
#   * PER PATH  -- key on the whole face; new1..new4 can each differ.
#   * PER EDGE  -- key on the in/out vid; "approximate pre1" means every face
#                  whose in_edge is pre1. This is the natural handle when the
#                  policy reasons about the edge Jacobians themselves.
#
# Verified numerically on a 2-pre x 2-post vertex: exact / pre1 / pre2 / post1 /
# post2 / (pre1+post2) all produce DIFFERENT Jacobians (6/6 distinct), and a
# ``None`` slot leaves that operand exact -- which is the per-path skip.


def face_transforms_from_edges(face_keys, *, pre=None, post=None, res=None):
    """Per-EDGE / per-PATH choices -> the ``{face_key: (lhs, rhs, res)}`` dict.

    ``pre`` is keyed by in_edge vid, ``post`` by out_edge vid, ``res`` by the
    full face key. Anything absent stays ``None``, i.e. that slot is left exact
    -- so skipping a path is simply omitting it.
    """
    pre, post, res = pre or {}, post or {}, res or {}
    return {k: (pre.get(k[0]), post.get(k[1]), res.get(k)) for k in face_keys}


def masked_face_transforms(face_keys, *, pre=None, post=None, res=None,
                           max_dims: int = 8, max_axes: int = 8,
                           stats: dict | None = None):
    """Same, but every entry is a rule LIST wrapped in a live-masked hook.

    Each slot's rules are checked against that slot's own operand at apply
    time, so an illegal rule is skipped rather than raising -- and because the
    three slots see three different tensors, they mask independently.
    """
    def _wrap(rules):
        if not rules:
            return None
        return make_live_masked_hook(list(rules), max_dims=max_dims,
                                     max_axes=max_axes, stats=stats)

    pre = {k: _wrap(v) for k, v in (pre or {}).items()}
    post = {k: _wrap(v) for k, v in (post or {}).items()}
    res = {k: _wrap(v) for k, v in (res or {}).items()}
    return face_transforms_from_edges(face_keys, pre=pre, post=post, res=res)


# --- oracle probe failure accounting ---------------------------------------
# probe_faces fails SOFT (see the except clause there), which is silent by
# construction: an approx run whose oracle rejects every vertex degrades into
# an exact run while every other metric still looks healthy. These counters are
# what make that visible, so they are not optional decoration.
_PROBE_FAILURES = {"count": 0, "vertices": set(), "last": ""}


def _record_probe_failure(vertex: int, approx: bool, exc: BaseException) -> None:
    _PROBE_FAILURES["count"] += 1
    _PROBE_FAILURES["vertices"].add(int(vertex))
    _PROBE_FAILURES["last"] = f"v{vertex} approx={approx} {type(exc).__name__}: {exc}"


def consume_probe_failure_stats() -> dict:
    """Per-episode probe-failure counts; resets the accumulator."""
    out = {
        "count": int(_PROBE_FAILURES["count"]),
        "n_vertices": len(_PROBE_FAILURES["vertices"]),
        "last": _PROBE_FAILURES["last"],
    }
    _PROBE_FAILURES["count"] = 0
    _PROBE_FAILURES["vertices"] = set()
    return out
