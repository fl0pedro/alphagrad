"""Tests for the post-IQR calibration scaling helper.

``compute_calibration_scaling(stats, statistic=...)`` consumes the
extended stats dict produced by the actor's ``reward_vec_means`` and
returns a per-channel rescale to multiply ``build_reward_weights(args)``
by. Three options are supported:

* ``iqr`` (default, recommended) — symlog-space IQR / 1.349 (σ-equiv,
  outlier-robust)
* ``mean_abs`` — legacy ``|symlog(mean)|`` (bias proxy)
* ``std`` — symlog-space std via the IQR proxy

Tests cover boundary cases (cosine_sim hard-pinned to 1.0; degenerate
zero-dispersion → floor-capped scale; sentinel-contaminated mean
handled defensively).
"""

from __future__ import annotations

import numpy as np
import pytest


def _stats_dict(samples_per_channel: dict[int, np.ndarray], n_channels: int = 8) -> dict:
    """Build the stats dict the calibration helper consumes from a
    per-channel sample array. Channels with no entry get zeros."""
    from alphagrad.approx.common.reward_scaling import symlog_np

    mean = np.zeros((n_channels,), dtype=np.float32)
    q25 = np.zeros_like(mean)
    q75 = np.zeros_like(mean)
    median = np.zeros_like(mean)
    q25_sl = np.zeros_like(mean)
    q75_sl = np.zeros_like(mean)
    median_sl = np.zeros_like(mean)
    count = 0
    for ch, samples in samples_per_channel.items():
        samples = np.asarray(samples, dtype=np.float32)
        mean[ch] = samples.mean()
        median[ch] = np.median(samples)
        q25[ch] = np.quantile(samples, 0.25)
        q75[ch] = np.quantile(samples, 0.75)
        sl = symlog_np(samples).astype(np.float32)
        median_sl[ch] = np.median(sl)
        q25_sl[ch] = np.quantile(sl, 0.25)
        q75_sl[ch] = np.quantile(sl, 0.75)
        count = max(count, samples.shape[0])
    return {
        "mean": mean, "median": median, "q25": q25, "q75": q75,
        "median_symlog": median_sl, "q25_symlog": q25_sl, "q75_symlog": q75_sl,
        "count": count,
    }


def test_iqr_default_returns_dispersion_inverse_per_channel():
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import (
        COSINE_SIM_IDX, REWARD_INDEX, symlog_np,
    )

    # Channel 1 (flops) ranges ~1e9 → symlog(1e9) ≈ 20.7. IQR ≈ 0
    # if all samples are equal, so the floor kicks in.
    rng = np.random.default_rng(0)
    flops_samples = rng.uniform(low=-2e9, high=-1e8, size=512).astype(np.float32)
    peak_samples = rng.uniform(low=-5e7, high=-1e6, size=512).astype(np.float32)

    stats = _stats_dict({
        REWARD_INDEX["flops"]: flops_samples,
        REWARD_INDEX["peak_memory"]: peak_samples,
    })

    scaling = compute_calibration_scaling(stats, statistic="iqr")
    # Wider channels get smaller scales (more compression). flops has
    # bigger symlog spread than peak_memory → flops gets a smaller scale
    # than peak_memory.
    assert scaling.shape == (8,)
    assert scaling[REWARD_INDEX["flops"]] < scaling[REWARD_INDEX["peak_memory"]]
    # Cosine_sim is hard-pinned to 1.0.
    assert scaling[COSINE_SIM_IDX] == pytest.approx(1.0)


def test_mean_abs_recovers_legacy_path():
    """`mean_abs` mode = ``1/|symlog(mean)|``, the pre-IQR behaviour."""
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import (
        COSINE_SIM_IDX, REWARD_INDEX, symlog_np,
    )

    # Single-valued channel so mean and IQR both produce well-defined
    # numbers but with different magnitudes — the legacy mean_abs path
    # is what was shipped before our refactor.
    samples_const = np.full((128,), -1e6, dtype=np.float32)
    stats = _stats_dict({REWARD_INDEX["peak_memory"]: samples_const})

    scaling = compute_calibration_scaling(stats, statistic="mean_abs")
    expected = 1.0 / np.abs(symlog_np(np.float32(-1e6)))
    assert scaling[REWARD_INDEX["peak_memory"]] == pytest.approx(
        float(expected), rel=1e-3,
    )
    # Cosine_sim still pinned at 1.0.
    assert scaling[COSINE_SIM_IDX] == pytest.approx(1.0)


def test_iqr_robust_to_outliers_unlike_mean_abs():
    """The point of switching to IQR: a single sentinel-like outlier in
    the sampled stream should NOT swing the calibration scale much.
    Mean-based stats are sensitive to outliers by construction; IQR
    truncates."""
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import REWARD_INDEX

    rng = np.random.default_rng(1)
    clean = rng.uniform(-1e7, -1e6, size=256).astype(np.float32)
    polluted = np.concatenate([clean, np.array([-1e10], dtype=np.float32)])

    s_clean = _stats_dict({REWARD_INDEX["flops"]: clean})
    s_polluted = _stats_dict({REWARD_INDEX["flops"]: polluted})

    scale_iqr_clean = compute_calibration_scaling(s_clean, statistic="iqr")[
        REWARD_INDEX["flops"]
    ]
    scale_iqr_pol = compute_calibration_scaling(s_polluted, statistic="iqr")[
        REWARD_INDEX["flops"]
    ]
    scale_mean_clean = compute_calibration_scaling(s_clean, statistic="mean_abs")[
        REWARD_INDEX["flops"]
    ]
    scale_mean_pol = compute_calibration_scaling(s_polluted, statistic="mean_abs")[
        REWARD_INDEX["flops"]
    ]

    # IQR ratio: outlier should barely move the IQR (it's the 75%-25%
    # range of symlog'd samples, and one outlier among ~250 doesn't
    # shift quartiles much).
    iqr_drift = abs(scale_iqr_pol - scale_iqr_clean) / abs(scale_iqr_clean)
    mean_drift = abs(scale_mean_pol - scale_mean_clean) / abs(scale_mean_clean)
    # The IQR-based scale should be more stable than the mean-based
    # scale under a single outlier injection.
    assert iqr_drift < mean_drift, (
        f"IQR should be more outlier-robust than mean_abs; got "
        f"iqr_drift={iqr_drift:.4g} >= mean_drift={mean_drift:.4g}"
    )


def test_zero_dispersion_channel_capped_by_floor():
    """A channel with constant samples (or no samples) has IQR=0 →
    division by zero. The floor caps the scaling at ``1/floor``."""
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import REWARD_INDEX

    samples_zero = np.zeros((64,), dtype=np.float32)
    stats = _stats_dict({REWARD_INDEX["flops"]: samples_zero})
    scaling = compute_calibration_scaling(stats, statistic="iqr", floor=1e-3)
    # 1.0 / 1e-3 = 1000 — the upper cap on scaling for zero-dispersion
    # channels.
    assert scaling[REWARD_INDEX["flops"]] == pytest.approx(1.0 / 1e-3, rel=1e-3)


def test_invalid_statistic_raises():
    from alphagrad.approx.common.calibration import compute_calibration_scaling

    stats = _stats_dict({})
    with pytest.raises(ValueError, match="unknown --calibration-statistic"):
        compute_calibration_scaling(stats, statistic="median")


def test_std_statistic_matches_iqr_proxy():
    """`std` mode uses the IQR→σ conversion since we only ship
    quartiles through Ray. Verify it returns the same shape as iqr but
    is otherwise computed the same."""
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import REWARD_INDEX

    rng = np.random.default_rng(2)
    samples = rng.uniform(-1e7, -1e6, size=256).astype(np.float32)
    stats = _stats_dict({REWARD_INDEX["flops"]: samples})

    s_iqr = compute_calibration_scaling(stats, statistic="iqr")
    s_std = compute_calibration_scaling(stats, statistic="std")
    # The current `std` implementation uses the same IQR / 1.349 proxy
    # so the two results match exactly. If a future commit ships full
    # samples (true std), this test asserts the proxy stays the
    # default contract.
    np.testing.assert_allclose(s_iqr, s_std, rtol=1e-5)


def test_sentinel_contaminated_mean_handled_in_mean_abs_mode():
    """A sentinel value (-1e10) in ``stats["mean"]`` would historically
    produce ``|symlog(-1e10)| ≈ 23``, then ``1/23 ≈ 0.04`` — a
    spuriously-small weight that crashed the value head. The fix:
    replace sentinels with 0 before computing the bias."""
    from alphagrad.approx.common.calibration import compute_calibration_scaling
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX, symlog_np,
    )
    from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE

    stats = _stats_dict({})
    # Inject the sentinel directly into the mean slot.
    stats["mean"][REWARD_INDEX["flops"]] = float(SENTINEL_REWARD_VALUE)

    scaling = compute_calibration_scaling(stats, statistic="mean_abs")
    # Sentinel → cleared to 0 → mean_abs = 0 → scaling = 1/floor.
    assert scaling[REWARD_INDEX["flops"]] == pytest.approx(1.0 / 1e-3, rel=1e-3)
