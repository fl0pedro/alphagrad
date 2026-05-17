"""Tests for the sparse-terminal reward handling.

Covers three pieces that work together to ensure ``cosine_sim`` and
``frob_residual`` — which the env only computes meaningfully at the
final elimination step — don't pollute downstream training signals:

1. ``SPARSE_TERMINAL_INDICES`` is the single source of truth.
2. MuZero's ``_per_channel_discounted_returns`` with γ=1 for the
   sparse channels broadcasts the terminal value back to every
   timestep without decay (so the value head sees the same target
   from t=0 as from t=T-1).
3. PPO's Lagrangian violation mask zeros intermediate-step
   "violations" for sparse channels so the dual multiplier doesn't
   accumulate ``threshold - 0`` penalty on every partial-order step.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_sparse_terminal_indices_match_canonical_channels():
    from alphagrad.approx.common.reward_scaling import (
        COSINE_SIM_IDX,
        FROB_RESIDUAL_IDX,
        SPARSE_TERMINAL_INDICES,
        SPARSE_TERMINAL_MASK_NP,
    )
    assert set(SPARSE_TERMINAL_INDICES) == {COSINE_SIM_IDX, FROB_RESIDUAL_IDX}
    assert SPARSE_TERMINAL_MASK_NP.dtype == bool
    assert SPARSE_TERMINAL_MASK_NP[COSINE_SIM_IDX]
    assert SPARSE_TERMINAL_MASK_NP[FROB_RESIDUAL_IDX]
    # Cost channels must NOT be flagged.
    for idx in range(SPARSE_TERMINAL_MASK_NP.shape[0]):
        if idx not in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX):
            assert not SPARSE_TERMINAL_MASK_NP[idx]


def test_muzero_no_discount_propagates_terminal_value():
    """The γ=1 mask for sparse channels makes the discounted return
    G_t broadcast the terminal value back to every timestep without
    decay. Without this, a γ=0.99^T factor would shrink the value
    target by ~T·log(γ) by t=0 — which would be wrong because the
    sparse channel only fires once (at t=T-1), not per step.
    """
    import jax
    import jax.numpy as jnp

    # Mu0 worker's discounted-return implementation lives in
    # ``alphagrad.approx.mu0_ray_worker``. Import it lazily here so
    # the test doesn't pull MuZero's full JAX stack if not needed.
    from alphagrad.approx.mu0_ray_worker import _per_channel_discounted_returns
    from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX

    T = 12
    cos_idx = REWARD_INDEX["cosine_sim"]
    flops_idx = REWARD_INDEX["flops"]

    # Build a fake rollout: cos=0 at t=0..T-2, cos=0.8 at t=T-1.
    # flops is dense: -1.0 per step (real cost channel).
    reward_vec = jnp.zeros((T, NUM_REWARDS), dtype=jnp.float32)
    reward_vec = reward_vec.at[-1, cos_idx].set(0.8)        # terminal cos
    reward_vec = reward_vec.at[:, flops_idx].set(-1.0)      # dense per-step

    weights = jnp.zeros((NUM_REWARDS,), dtype=jnp.float32)
    weights = weights.at[cos_idx].set(1.0)
    weights = weights.at[flops_idx].set(1.0)

    # discount=0.99 → cost channel decays; cos channel must not (γ=1).
    returns = _per_channel_discounted_returns(reward_vec, weights, discount=0.99)
    returns = np.asarray(returns)

    # G_t = cos_contribution + flops_contribution
    # cos contribution at every t = 1.0 * 0.8 (no decay)
    # flops contribution at t = sum_{j>=t} 0.99^(j-t) * (-1.0)
    expected_cos_part = 0.8
    expected_flops_part = np.array(
        [-sum(0.99 ** k for k in range(T - t)) for t in range(T)],
        dtype=np.float32,
    )
    expected = expected_cos_part + expected_flops_part
    np.testing.assert_allclose(returns, expected, atol=1e-4)


def test_lagrangian_sparse_mask_zeros_intermediate_violations():
    """When a constraint targets a sparse channel (cosine_sim), the
    Lagrangian violation tensor must be zero at all non-terminal
    timesteps and equal to the threshold-based violation at the
    terminal step only.
    """
    from alphagrad.approx.common.reward_scaling import (
        COSINE_SIM_IDX,
        SPARSE_TERMINAL_INDICES,
    )

    # Synthesize the violation computation in isolation (the PPO
    # worker's path uses the same formula). buf_reward_vec has cos=0
    # at intermediate steps and cos=0.3 at terminal — below the 0.8
    # threshold, so the terminal-only violation should be positive.
    N, T = 2, 4
    NUM_REWARDS = 8
    buf = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    buf[-1, :, COSINE_SIM_IDX] = 0.3  # terminal cos

    # Constraint: cosine_sim >= 0.8, sign=+1, threshold=0.8
    constraint_indices = np.array([COSINE_SIM_IDX], dtype=np.int32)
    constraint_thresholds = np.array([0.8], dtype=np.float32)
    constraint_signs = np.array([+1.0], dtype=np.float32)
    constraint_is_sparse = np.array(
        [COSINE_SIM_IDX in SPARSE_TERMINAL_INDICES], dtype=bool,
    )

    # Replicate the worker's violation block.
    picked = buf[:, :, constraint_indices]              # (T, N, C)
    picked = np.transpose(picked, (1, 0, 2))            # (N, T, C)
    # No-symlog because cosine_sim is in NO_SYMLOG_REWARD_INDICES.
    picked_sl = picked  # bypass symlog for cos
    thresh_sl = constraint_thresholds  # (C,)
    thresh_scale = np.maximum(np.abs(thresh_sl), 1e-3)
    signed = (
        constraint_signs * (thresh_sl[None, None, :] - picked_sl)
        / thresh_scale[None, None, :]
    )
    violations = np.maximum(0.0, signed)                # (N, T, C)

    # Apply sparse-terminal mask.
    is_terminal_step = np.zeros((N, T), dtype=np.float32)
    is_terminal_step[:, -1] = 1.0
    sparse_step_mask = np.where(
        constraint_is_sparse[None, None, :],
        is_terminal_step[:, :, None],
        1.0,
    )
    violations_masked = violations * sparse_step_mask

    # Intermediate steps: violation MUST be zero.
    for t in range(T - 1):
        assert np.all(violations_masked[:, t, :] == 0.0), (
            f"sparse-channel violation at non-terminal step t={t} should "
            f"be 0 but got {violations_masked[:, t, :]}"
        )
    # Terminal step: violation should be (0.8 - 0.3) / 0.8 = 0.625.
    np.testing.assert_allclose(
        violations_masked[:, -1, 0], 0.625, atol=1e-5,
    )


def test_lagrangian_dense_channel_unaffected_by_mask():
    """For a dense (non-sparse) constraint like ``peak_memory<=1e8``,
    the mask must NOT zero intermediate-step violations."""
    from alphagrad.approx.common.reward_scaling import (
        SPARSE_TERMINAL_INDICES,
        REWARD_INDEX,
    )

    peak_idx = REWARD_INDEX["peak_memory"]
    N, T = 2, 3
    NUM_REWARDS = 8

    # peak_memory is a cost channel; assume rewards are large negatives.
    buf = np.full((T, N, NUM_REWARDS), -1e7, dtype=np.float32)

    constraint_indices = np.array([peak_idx], dtype=np.int32)
    constraint_thresholds = np.array([-1e8], dtype=np.float32)  # cap
    constraint_signs = np.array([-1.0], dtype=np.float32)       # <= → sign -1
    constraint_is_sparse = np.array(
        [peak_idx in SPARSE_TERMINAL_INDICES], dtype=bool,
    )

    picked = buf[:, :, constraint_indices]
    picked = np.transpose(picked, (1, 0, 2))
    # Apply symlog (peak_memory IS in symlog channels).
    picked_sl = np.sign(picked) * np.log1p(np.abs(picked))
    thresh_sl = np.sign(constraint_thresholds) * np.log1p(np.abs(constraint_thresholds))
    thresh_scale = np.maximum(np.abs(thresh_sl), 1e-3)
    signed = (
        constraint_signs * (thresh_sl[None, None, :] - picked_sl)
        / thresh_scale[None, None, :]
    )
    violations = np.maximum(0.0, signed)
    is_terminal_step = np.zeros((N, T), dtype=np.float32)
    is_terminal_step[:, -1] = 1.0
    sparse_step_mask = np.where(
        constraint_is_sparse[None, None, :],
        is_terminal_step[:, :, None],
        1.0,
    )
    violations_masked = violations * sparse_step_mask
    # Dense channel: violations preserved at EVERY step, not just terminal.
    np.testing.assert_allclose(violations_masked, violations)
