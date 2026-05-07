"""Action-mask helpers shared between the approx-RL trainers.

The trainers differ in their action-space layout (legacy unrolled `sp_type ×
vertex` vs. the new factorised `vertex × pair × factor`), but the underlying
shape inspection — "for each vertex, what is the output ndim and the smallest
input ndim?" — is the same. These helpers expose that primitive plus a few
convenience builders for the most common mask layouts.
"""

from __future__ import annotations

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


# Pair-index → (out_axis, primal_axis) — must match the trainer's `_PAIR_TO_BASE`.
_PAIR_TO_BASE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _vertex_pair_dim_sizes(jaxpr, total_v: int, num_pair_choices: int):
    """Per-(vertex, pair) ``(d1, d2)`` axis sizes, or ``(0, 0)`` if invalid.

    ``d1`` is the size of output axis ``pair.bi1``; ``d2`` is the size of the
    same-position axis on the *first non-literal* input variable. For pair
    indices that exceed the vertex's available axes (or the STOP slot),
    both are ``0`` so every factor will be marked invalid for that slot.
    """
    sizes = np.zeros((total_v, num_pair_choices, 2), dtype=np.int32)
    for i, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_shape = eqn.outvars[0].aval.shape
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        if not invars:
            continue
        primal_shape = invars[0].aval.shape  # first input as the primal proxy
        for pair_idx, (bi1, bi2) in enumerate(_PAIR_TO_BASE):
            if pair_idx >= num_pair_choices:
                break
            if bi1 >= len(out_shape) or bi2 >= len(primal_shape):
                continue
            sizes[i, pair_idx, 0] = int(out_shape[bi1])
            sizes[i, pair_idx, 1] = int(primal_shape[bi2])
    return sizes


def _factor_is_valid(factor: int, d1: int, d2: int) -> bool:
    """Does ``apply_dynamic_sparsity`` produce a finite, matmul-compatible
    SparseTensor for this ``(d1, d2)`` pair under ``factor``?

    Mirrors the contract that the rewritten `apply_dynamic_sparsity` honours:
    * ``factor == -1`` — always valid (gcd-collapse).
    * ``factor == 0``  — always valid (drops the axes).
    * ``factor == 1``  — always valid (no-op; the rule is silently dropped).
    * ``factor > 1``   — valid iff it divides both ``d1`` and ``d2`` *and*
                          is at most ``gcd(d1, d2)``.

    Returns ``False`` only when the slot itself is unused (``d1 == 0`` or
    ``d2 == 0``), so the policy never picks an action for a pair the
    vertex doesn't actually have.
    """
    if d1 == 0 or d2 == 0:
        return False
    if factor in (-1, 0, 1):
        return True
    if factor < 0:
        return False
    return d1 % factor == 0 and d2 % factor == 0 and factor <= min(d1, d2)


def build_pair_factor_valid_mask(
    jaxpr,
    total_v: int,
    num_pair_choices: int,
    factor_table,
    pair_stop_idx: int,
):
    """Build the per-(vertex, pair, factor) validity mask used by the
    autoregressive factor head.

    Returns ``jnp.float32`` shape ``(total_v, num_pair_choices, num_factors)``.
    The STOP pair gets ``factor_table[0]`` enabled so the categorical over
    factors is never fully masked when the rule terminates (the chosen
    factor is ignored anyway in the env).
    """
    factor_table_np = np.asarray(factor_table)
    num_factors = factor_table_np.shape[0]
    sizes = _vertex_pair_dim_sizes(jaxpr, total_v, num_pair_choices)

    mask = np.zeros((total_v, num_pair_choices, num_factors), dtype=np.float32)
    for v in range(total_v):
        for p in range(num_pair_choices):
            if p == pair_stop_idx:
                # Keep at least one factor valid so the categorical sums to 1
                # even when the agent picks STOP. The factor is unused.
                mask[v, p, 0] = 1.0
                continue
            d1 = int(sizes[v, p, 0])
            d2 = int(sizes[v, p, 1])
            for f_idx in range(num_factors):
                f = int(factor_table_np[f_idx])
                if _factor_is_valid(f, d1, d2):
                    mask[v, p, f_idx] = 1.0
    return jnp.array(mask)


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
