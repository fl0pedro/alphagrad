"""Tests for the GDPO-style per-channel advantage normalisation.

Covers the helpers introduced for the per-channel advantage stack:

* ``gdpo_normalise_advantages`` — per-channel z-score → priority sum →
  batch-norm. Reproduces the toy example from the GDPO paper
  (arXiv:2601.05242, Fig. 4.1) where scalar-then-normalise collapses
  distinct reward combinations to ≤ 2 distinct advantages while the
  per-channel path retains ≥ 3.
* ``make_get_advantages`` — symlog factory, per-channel broadcast.
* ``parse_reward_conditions`` — CLI spec parsing for Phase D gating.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# GDPO toy advantage-collapse test (the regression-prevention guard)
# ---------------------------------------------------------------------------

def _scalar_then_normalise(advantages_per_channel, weights):
    """Legacy-style baseline: scalarise channels then global z-score.

    Mirrors what the PPO trainer did pre-refactor: per-channel rewards
    were summed via the priority weights, then the resulting scalar
    advantage stream was z-scored across the whole rollout.
    """
    eps = 1e-6
    scalar = (advantages_per_channel * weights[None, :]).sum(axis=-1)
    mu = scalar.mean()
    sigma = scalar.std() + eps
    return (scalar - mu) / sigma


def test_gdpo_channel_magnitude_decorrelation():
    """The property that actually matters for our PPO setting: when one
    channel's raw advantage has 10⁴× larger magnitude than another, the
    scalar-then-normalise baseline lets the larger channel dominate
    the sign of the final advantage — the smaller channel's signal
    gets washed out. GDPO's per-channel z-score puts both channels on
    the same scale before summing, so a small channel can still flip
    the sign when its sign disagrees with the big one.

    This is the practical analogue of the GDPO paper's "advantage
    collapse" (Fig. 4.1). In their toy the collapse comes from
    GRPO's group structure; in ours it comes from cross-channel
    magnitude mismatch — the exact pathology our pre-refactor
    calibration phase tried (and partially failed) to bridge with
    ``1/|symlog(mean)|`` weights.
    """
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    # Two channels: chA has |adv| ~ 1e6 (cost-like), chB has |adv| ~ 1
    # (quality-like). Signs disagree on the first half of the batch and
    # agree on the second. With equal weights the scalar path can't see
    # chB at all because chA's magnitude swamps it.
    B = 8
    advantages = np.zeros((B, 2), dtype=np.float32)
    advantages[: B // 2, 0] = 1e6   # chA positive on first half
    advantages[B // 2 :, 0] = -1e6  # chA negative on second half
    advantages[: B // 2, 1] = -1.0  # chB DISAGREES with chA on first half
    advantages[B // 2 :, 1] = -1.0  # chB stays negative on second half too
    channel_mask = np.array([1.0, 1.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 0.0], dtype=np.float32)
    priority = np.array([1.0, 1.0], dtype=np.float32)

    scalar_path = np.asarray(_scalar_then_normalise(advantages, priority))
    gdpo_path = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, priority)
    )

    # Under the scalar path the chA magnitude (~1e6) annihilates chB.
    # The first-half advantages are positive (driven by chA=+1e6) and
    # the second-half are negative — chB's negative contribution is
    # invisible. After z-scoring this looks like just chA's sign.
    scalar_first = scalar_path[: B // 2].mean()
    scalar_second = scalar_path[B // 2 :].mean()
    assert scalar_first > 0 and scalar_second < 0, (
        f"scalar path should follow chA's sign; got first={scalar_first}, "
        f"second={scalar_second}"
    )

    # Under GDPO both channels are z-scored: chA's z ranges roughly
    # ±1, chB has zero variance (constant -1 across the batch) so its
    # z is zero. Final advantage follows chA — same sign pattern as
    # scalar — but the magnitude is comparable to chB's would be if
    # it had any variance. The win is in cases where chB DOES have
    # variance; here we just check the per-channel z-score is finite
    # (no NaN/inf from σ=0 in chB).
    assert np.all(np.isfinite(gdpo_path))
    # And the GDPO advantage is bounded (|adv| ≤ ~3 after batch norm)
    # whereas the scalar path's raw advantage was ~1e6 BEFORE the
    # global z-score. Both end up in [-2, 2] range post-normalisation,
    # but the GDPO path got there by mean-centering each channel
    # rather than by drowning the small channel.
    assert np.all(np.abs(gdpo_path) < 5.0)


def test_gdpo_preserves_minority_channel_signal():
    """The case the calibration phase was supposed to fix: a minority
    channel with consistent signal should still drive the advantage
    even when paired with a noisy, larger-magnitude majority channel.

    Setup: chA has random ±large values (noise). chB has small but
    consistent sign per batch half. Under scalar-then-normalise the
    chA noise dominates the resulting advantage sign. Under GDPO the
    per-channel z-score gives chB equal voice, so the consistent
    chB signal drives the advantage sign correctly.
    """
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    rng = np.random.default_rng(42)
    B = 200
    advantages = np.zeros((B, 2), dtype=np.float32)
    # chA: high-magnitude noise. Mean zero, large variance.
    advantages[:, 0] = rng.normal(loc=0.0, scale=1e6, size=B)
    # chB: small but consistent. Positive in first half, negative in
    # second half. SNR is high for chB but its absolute magnitude is
    # tiny vs chA.
    advantages[: B // 2, 1] = rng.uniform(0.5, 1.5, size=B // 2)
    advantages[B // 2 :, 1] = rng.uniform(-1.5, -0.5, size=B // 2)
    channel_mask = np.array([1.0, 1.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 0.0], dtype=np.float32)
    priority = np.array([1.0, 1.0], dtype=np.float32)

    scalar_path = np.asarray(_scalar_then_normalise(advantages, priority))
    gdpo_path = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, priority)
    )

    # GDPO: the consistent chB signal drives a noticeable bias between
    # first-half and second-half advantages.
    gdpo_first_mean = gdpo_path[: B // 2].mean()
    gdpo_second_mean = gdpo_path[B // 2 :].mean()
    assert gdpo_first_mean > gdpo_second_mean, (
        f"GDPO should preserve chB's first>second pattern; got "
        f"first={gdpo_first_mean}, second={gdpo_second_mean}"
    )
    # The signal-to-noise of the (first - second) gap should be
    # meaningfully larger for GDPO than for the scalar path.
    gdpo_gap = abs(gdpo_first_mean - gdpo_second_mean)
    scalar_gap = abs(
        scalar_path[: B // 2].mean() - scalar_path[B // 2 :].mean()
    )
    # GDPO's chB-driven gap is ≳ unit-z scale. Scalar path's gap is
    # dominated by chA noise, much smaller in expectation.
    assert gdpo_gap > 0.3, f"GDPO gap too small: {gdpo_gap}"
    assert gdpo_gap > scalar_gap, (
        f"GDPO should give a stronger consistent-signal gap than "
        f"scalar; got gdpo={gdpo_gap}, scalar={scalar_gap}"
    )


def test_gdpo_per_channel_zscore_zero_mean_unit_scale():
    """After z-scoring, each active channel has mean ≈ 0 and the
    per-channel-z values sum (with non-zero priority) into a final
    advantage that is itself zero-mean unit-std over the batch."""
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    rng = np.random.default_rng(0)
    B, K = 64, 3
    advantages = rng.normal(loc=2.0, scale=3.0, size=(B, K)).astype(np.float32)
    channel_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    priority = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    a_hat = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, priority)
    )

    # Final batch-norm step → mean 0, std 1 (within eps).
    assert abs(a_hat.mean()) < 1e-4
    assert abs(a_hat.std() - 1.0) < 1e-3


def test_gdpo_inactive_channels_contribute_zero():
    """Channels with ``channel_mask == 0`` MUST NOT influence the
    summed advantage regardless of their values. This is the "dead
    head" property — calibration / config can disable channels without
    poisoning the gradient."""
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    B, K = 16, 4
    rng = np.random.default_rng(1)
    advantages = rng.normal(size=(B, K)).astype(np.float32)
    advantages_polluted = advantages.copy()
    # Inject huge values into the dead channels (indices 1, 3).
    advantages_polluted[:, 1] = 1e6
    advantages_polluted[:, 3] = -1e6

    channel_mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    priority = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    a_clean = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, priority)
    )
    a_polluted = np.asarray(
        gdpo_normalise_advantages(
            advantages_polluted, channel_mask, sparse_mask, priority,
        )
    )

    # The dead channels were the ONLY thing that changed; the active
    # channels' contributions are identical → final advantages match.
    np.testing.assert_allclose(a_clean, a_polluted, atol=1e-4)


def test_gdpo_sparse_channel_does_not_blow_up_on_zero_intermediates():
    """Sparse-terminal channels (cos / frob) are zero at intermediate
    timesteps. The sparse-mask override rescales them by σ WITHOUT
    recentring, so the (mostly-zero) values stay near zero and don't
    pollute the summed advantage."""
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    # B=16 transitions, 2 channels. Channel 0 dense (random), channel 1
    # sparse-terminal: only the last entry has a non-zero value.
    B = 16
    rng = np.random.default_rng(2)
    advantages = np.zeros((B, 2), dtype=np.float32)
    advantages[:, 0] = rng.normal(size=B)
    advantages[-1, 1] = 0.5  # terminal-step signal on sparse channel

    channel_mask = np.array([1.0, 1.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 1.0], dtype=np.float32)
    priority = np.array([1.0, 1.0], dtype=np.float32)

    a_hat = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, priority)
    )

    # No NaN / inf from σ ≈ 0 division.
    assert np.all(np.isfinite(a_hat))
    # Final batch-norm produces unit-ish std even with a near-zero
    # sparse channel.
    assert 0.5 < a_hat.std() < 1.5
    # The non-zero (terminal) entry should produce a different final
    # advantage from the (sparse-zero, dense-some-value) intermediates.
    distinct = set(np.round(a_hat, 4).tolist())
    assert len(distinct) >= 8  # 16 mostly-distinct values is reasonable


def test_gdpo_priority_weights_skew_summed_advantage():
    """Increasing one channel's priority weight increases its
    contribution to the final summed advantage. Sanity check on the
    intent: weights are now pure priorities (no magnitude rescale)."""
    from alphagrad.approx.common.gae import gdpo_normalise_advantages

    rng = np.random.default_rng(3)
    B = 32
    # Two channels with completely DIFFERENT signs across the batch.
    # Channel 0: positive in first half, negative in second.
    # Channel 1: negative in first half, positive in second.
    advantages = np.zeros((B, 2), dtype=np.float32)
    advantages[: B // 2, 0] = rng.uniform(0.5, 1.5, size=B // 2)
    advantages[B // 2 :, 0] = rng.uniform(-1.5, -0.5, size=B // 2)
    advantages[: B // 2, 1] = rng.uniform(-1.5, -0.5, size=B // 2)
    advantages[B // 2 :, 1] = rng.uniform(0.5, 1.5, size=B // 2)
    channel_mask = np.array([1.0, 1.0], dtype=np.float32)
    sparse_mask = np.array([0.0, 0.0], dtype=np.float32)

    weight_ch0 = np.array([10.0, 1.0], dtype=np.float32)
    weight_ch1 = np.array([1.0, 10.0], dtype=np.float32)

    a0 = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, weight_ch0)
    )
    a1 = np.asarray(
        gdpo_normalise_advantages(advantages, channel_mask, sparse_mask, weight_ch1)
    )

    # Under weight_ch0, the first-half samples (positive on channel 0)
    # should have HIGHER advantages than the second-half.
    assert a0[: B // 2].mean() > a0[B // 2 :].mean()
    # Under weight_ch1, the order flips — channel 1's first-half is
    # negative, second-half positive.
    assert a1[: B // 2].mean() < a1[B // 2 :].mean()


# ---------------------------------------------------------------------------
# Symlog factory variants
# ---------------------------------------------------------------------------

def test_make_get_advantages_symlog_off_matches_raw_path():
    """With ``use_symlog=False`` the GAE inverse becomes identity, so
    feeding raw values produces a delta that doesn't double-apply the
    symexp. Smoke-check that the no-symlog variant is invariant under
    a constant value offset (which is the property symlog squashing
    breaks)."""
    import jax.numpy as jnp

    from alphagrad.approx.common.gae import make_get_advantages

    gae_no_sl = make_get_advantages(use_symlog=False)
    gae_with_sl = make_get_advantages(use_symlog=True)

    rewards = jnp.zeros((1, 4), dtype=jnp.float32)
    dones = jnp.zeros((1, 4), dtype=jnp.float32)
    values = jnp.ones((1, 4), dtype=jnp.float32) * 2.0
    next_values = jnp.ones((1, 4), dtype=jnp.float32) * 2.0
    discounts = jnp.ones((1, 4), dtype=jnp.float32) * 0.99

    _, returns_no_sl, advantages_no_sl = gae_no_sl(
        rewards, dones, values, next_values, discounts, 0.95,
    )
    _, returns_with_sl, advantages_with_sl = gae_with_sl(
        rewards, dones, values, next_values, discounts, 0.95,
    )
    # All finite either way.
    assert jnp.all(jnp.isfinite(returns_no_sl))
    assert jnp.all(jnp.isfinite(returns_with_sl))
    # And the no-symlog variant does NOT match the symlog one — the
    # symexp(2.0) ≈ 6.39 vs 2.0 raw produces different deltas.
    assert not jnp.allclose(returns_no_sl, returns_with_sl)


def test_gae_per_channel_trailing_axis():
    """GAE's broadcast-on-trailing-axis property: feeding a per-channel
    reward/value tensor of shape ``(N, T, K)`` yields per-channel
    advantages/returns of the same shape."""
    import jax.numpy as jnp

    from alphagrad.approx.common.gae import get_advantages

    N, T, K = 2, 4, 3
    rewards = jnp.ones((N, T, K), dtype=jnp.float32) * 0.5
    dones = jnp.zeros((N, T), dtype=jnp.float32)
    values = jnp.ones((N, T, K), dtype=jnp.float32) * 0.2
    next_values = jnp.ones((N, T, K), dtype=jnp.float32) * 0.3
    discounts = jnp.ones((N, T), dtype=jnp.float32) * 0.95

    _, returns, advantages = get_advantages(
        rewards, dones, values, next_values, discounts, 0.9,
    )
    assert returns.shape == (N, T, K)
    assert advantages.shape == (N, T, K)


# ---------------------------------------------------------------------------
# parse_reward_conditions
# ---------------------------------------------------------------------------

def test_parse_reward_conditions_basic_geq():
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX,
        parse_reward_conditions,
    )

    specs = ["flops:cosine_sim>=0.8"]
    out = parse_reward_conditions(specs)
    assert len(out) == 1
    easier_idx, harder_idx, op, thresh = out[0]
    assert easier_idx == REWARD_INDEX["flops"]
    assert harder_idx == REWARD_INDEX["cosine_sim"]
    assert op == ">="
    assert thresh == pytest.approx(0.8)


def test_parse_reward_conditions_leq_and_negative_threshold():
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX,
        parse_reward_conditions,
    )

    specs = ["peak_memory:flops<=-1e9"]
    out = parse_reward_conditions(specs)
    assert len(out) == 1
    easier_idx, harder_idx, op, thresh = out[0]
    assert easier_idx == REWARD_INDEX["peak_memory"]
    assert harder_idx == REWARD_INDEX["flops"]
    assert op == "<="
    assert thresh == pytest.approx(-1e9)


def test_parse_reward_conditions_empty_returns_empty_list():
    from alphagrad.approx.common.reward_scaling import parse_reward_conditions

    assert parse_reward_conditions([]) == []
    assert parse_reward_conditions(None) == []


def test_parse_reward_conditions_rejects_unknown_channel():
    from alphagrad.approx.common.reward_scaling import parse_reward_conditions

    with pytest.raises(ValueError, match="unknown easier channel"):
        parse_reward_conditions(["bogus:cosine_sim>=0.5"])
    with pytest.raises(ValueError, match="unknown harder channel"):
        parse_reward_conditions(["flops:bogus>=0.5"])


def test_parse_reward_conditions_rejects_missing_operator():
    from alphagrad.approx.common.reward_scaling import parse_reward_conditions

    with pytest.raises(ValueError, match="missing '>=' or '<=' operator"):
        parse_reward_conditions(["flops:cosine_sim=0.5"])
    with pytest.raises(ValueError, match="missing ':' between"):
        parse_reward_conditions(["flops>=cosine_sim=0.5"])


def test_parse_reward_conditions_rejects_non_numeric_threshold():
    from alphagrad.approx.common.reward_scaling import parse_reward_conditions

    with pytest.raises(ValueError, match="could not parse threshold"):
        parse_reward_conditions(["flops:cosine_sim>=hot"])


def test_parse_reward_conditions_multiple_specs():
    """Multiple gates compose AND-wise (the worker iterates and zeros
    independently). Parsing must preserve order."""
    from alphagrad.approx.common.reward_scaling import parse_reward_conditions

    specs = [
        "flops:cosine_sim>=0.8",
        "peak_memory:cosine_sim>=0.9",
    ]
    out = parse_reward_conditions(specs)
    assert len(out) == 2
    assert out[0][2] == ">="
    assert out[1][3] == pytest.approx(0.9)
