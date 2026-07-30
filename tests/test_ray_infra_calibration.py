"""Unit tests for ``alphagrad.approx.common.calibration.run_calibration``.

The helper takes a duck-typed actor with two methods
(``reward_vec_means.remote`` and ``set_reward_weights.remote``) and an
args object, runs N zero-pref rollouts, and pushes a per-channel
symlog-rescaled weight vector back to the actor.

These tests exercise the helper with a fake actor that exposes those
two methods (no real Ray cluster required). Ray itself is stubbed
just enough so ``ray.get(future)`` resolves to the future's ``.value``
attribute.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def _ray_stub(monkeypatch):
    """Minimal Ray stand-in for the calibration helper."""

    class _Fut:
        def __init__(self, value):
            self.value = value

    def _get(fut, timeout=None):
        return fut.value

    fake_ray = types.SimpleNamespace(
        get=_get,
        put=lambda x: _Fut(x),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    yield fake_ray


class _Remote:
    """Wraps a function so ``.remote(*a, **k)`` returns a fake future
    holding the synchronous call's return value."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        class _F:
            value = None
        f = _F()
        f.value = self._fn(*a, **k)
        return f


def _stats_dict_from_means(mean_vec: np.ndarray) -> dict:
    """Build the extended stats dict the post-IQR calibration helper
    expects, faking the median/quartile entries by treating each
    channel as a delta distribution at its mean."""
    from alphagrad.approx.common.reward_scaling import symlog_np

    mean_vec = np.asarray(mean_vec, dtype=np.float32)
    sl = symlog_np(mean_vec).astype(np.float32)
    return {
        "mean": mean_vec,
        "median": mean_vec.copy(),
        "q25": mean_vec.copy(),
        "q75": mean_vec.copy(),
        "median_symlog": sl,
        "q25_symlog": sl,
        "q75_symlog": sl,
        "count": 1,
    }


class _FakeActor:
    def __init__(self, mean_vec: np.ndarray):
        self._stats = _stats_dict_from_means(mean_vec)
        self.last_weights = None

        def _reward_vec_means(rng_seed, num_rollouts):
            return self._stats

        def _set_reward_weights(weights):
            self.last_weights = np.asarray(weights, dtype=np.float32)

        self.reward_vec_means = _Remote(_reward_vec_means)
        self.set_reward_weights = _Remote(_set_reward_weights)


def _args(*, calibration_statistic: str = "mean_abs"):
    """Default fixture-args use the legacy ``mean_abs`` statistic so the
    pre-IQR structural assertions (``w_flops < w_peak`` purely from
    channel magnitudes) hold unchanged. New IQR-specific behaviour is
    covered in ``test_calibration_statistic.py``."""
    return SimpleNamespace(
        seed=42,
        rewards=["cmp", "mem", "acc"],
        cmp_type="flops",
        mem_type="peak_memory",
        lambda_cmp=1.0,
        lambda_mem=1.0,
        lambda_frob=0.0,
        calibration_statistic=calibration_statistic,
    )


def test_calibration_rescales_per_channel(_ray_stub):
    """Channels with wider magnitude get smaller weights (1/mean_abs)."""
    from alphagrad.approx.common.calibration import run_calibration
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX,
    )

    mean_vec = np.zeros((NUM_REWARDS,), dtype=np.float32)
    mean_vec[REWARD_INDEX["flops"]] = -1e9
    mean_vec[REWARD_INDEX["peak_memory"]] = -1e6
    mean_vec[REWARD_INDEX["cosine_sim"]] = 0.5

    actor = _FakeActor(mean_vec)
    abs_weights = run_calibration(actor, _args(), num_rollouts=4, after_warmup=True)
    assert abs_weights.shape == (NUM_REWARDS,)

    # flops weight should be much smaller than peak_memory weight (its
    # raw magnitude is 1000x larger, so 1/|symlog| is roughly half).
    w_flops = abs_weights[REWARD_INDEX["flops"]]
    w_peak = abs_weights[REWARD_INDEX["peak_memory"]]
    assert 0 < w_flops < w_peak

    # cosine_sim is excluded from symlog — its weight is the raw lambda
    # (1.0), no scaling.
    w_cos = abs_weights[REWARD_INDEX["cosine_sim"]]
    assert w_cos == pytest.approx(1.0, rel=1e-3)

    # The actor received the rescaled weights.
    np.testing.assert_allclose(actor.last_weights, abs_weights)


def test_calibration_requires_after_warmup_true():
    """The kwarg is a guard against the pre-refactor calibration-before-
    warmup ordering that produced the MuZero NaN."""
    from alphagrad.approx.common.calibration import run_calibration
    actor = _FakeActor(np.zeros((8,), dtype=np.float32))
    with pytest.raises(AssertionError, match="AFTER the CPU-approx pool"):
        run_calibration(actor, _args(), num_rollouts=4, after_warmup=False)


def test_calibration_zero_rollouts_returns_zeros(_ray_stub):
    """``--calibrate-steps 0`` disables calibration cleanly."""
    from alphagrad.approx.common.calibration import run_calibration
    from alphagrad.approx.common.reward_scaling import NUM_REWARDS

    actor = _FakeActor(np.zeros((NUM_REWARDS,), dtype=np.float32))
    out = run_calibration(actor, _args(), num_rollouts=0, after_warmup=True)
    assert out.shape == (NUM_REWARDS,)
    assert np.all(out == 0.0)
    # set_reward_weights should NOT have been called.
    assert actor.last_weights is None
