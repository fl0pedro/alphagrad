"""Tests for the sparse-terminal reward handling.

Covers two pieces that work together to ensure ``cosine_sim`` and
``frob_residual`` — which the env only computes meaningfully at the
final elimination step — don't pollute downstream training signals:

1. ``SPARSE_TERMINAL_INDICES`` is the single source of truth.
2. MuZero's ``_per_channel_discounted_returns`` with γ=1 for the
   sparse channels broadcasts the terminal value back to every
   timestep without decay (so the value head sees the same target
   from t=0 as from t=T-1).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_sparse_terminal_indices_match_canonical_channels():
    from alphagrad.approx.common.reward_scaling import (
        BKSTEP_ACC_IDX,
        COSINE_SIM_IDX,
        FROB_RESIDUAL_IDX,
        SPARSE_TERMINAL_INDICES,
        SPARSE_TERMINAL_MASK_NP,
    )
    # bkstep_acc joined cossim/frob as a third terminal-only QUALITY channel
    # (the B_kstep accuracy is only meaningful once the graph is eliminated).
    quality = {COSINE_SIM_IDX, FROB_RESIDUAL_IDX, BKSTEP_ACC_IDX}
    assert set(SPARSE_TERMINAL_INDICES) == quality
    assert SPARSE_TERMINAL_MASK_NP.dtype == bool
    for idx in quality:
        assert SPARSE_TERMINAL_MASK_NP[idx]
    # Cost channels must NOT be flagged.
    for idx in range(SPARSE_TERMINAL_MASK_NP.shape[0]):
        if idx not in quality:
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
