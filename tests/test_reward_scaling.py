"""Unit tests for ``alphagrad.approx.common.reward_scaling``.

Covers the numpy-only helpers that both PPO-ray and MuZero-ray
trainers consume:

* ``build_reward_weights`` — args → weight vector
* ``symlog_np`` — signed log compression
* ``filter_sentinel_mask`` — sentinel detection
* ``aggregate_per_channel_stats`` — driver-facing per-channel report
* ``update_running_bests`` / ``format_milestone_line`` /
  ``build_wandb_log_dict`` — driver state + logging glue
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# build_reward_weights
# ---------------------------------------------------------------------------

def _make_args(**kw):
    base = dict(
        rewards=["cmp", "mem", "acc"],
        cmp_type="flops",
        mem_type="peak_memory",
        lambda_cmp=1.0,
        lambda_mem=1.0,
        lambda_frob=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_reward_weights_default_three_channels():
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX, build_reward_weights,
    )

    w = build_reward_weights(_make_args())
    assert w.shape == (8,)
    assert w[REWARD_INDEX["flops"]] == 1.0
    assert w[REWARD_INDEX["peak_memory"]] == 1.0
    assert w[REWARD_INDEX["cosine_sim"]] == 1.0
    assert w[REWARD_INDEX["muls_adds_fmas"]] == 0.0
    assert w[REWARD_INDEX["frob_residual"]] == 0.0


def test_build_reward_weights_empty_falls_back_to_mul_add_fma():
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX, build_reward_weights,
    )

    w = build_reward_weights(_make_args(rewards=[]))
    assert w[REWARD_INDEX["muls_adds_fmas"]] == 1.0
    assert (w[REWARD_INDEX["muls_adds_fmas"]] > 0)
    # other channels stay zero
    for k in ("flops", "peak_memory", "cosine_sim", "frob_residual"):
        assert w[REWARD_INDEX[k]] == 0.0


def test_build_reward_weights_with_lambdas():
    from alphagrad.approx.common.reward_scaling import (
        REWARD_INDEX, build_reward_weights,
    )

    args = _make_args(lambda_cmp=3.5, lambda_mem=0.5, lambda_frob=0.25)
    w = build_reward_weights(args)
    assert w[REWARD_INDEX["flops"]] == pytest.approx(3.5)
    assert w[REWARD_INDEX["peak_memory"]] == pytest.approx(0.5)
    assert w[REWARD_INDEX["frob_residual"]] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# symlog_np
# ---------------------------------------------------------------------------

def test_symlog_sign_and_magnitude():
    from alphagrad.approx.common.reward_scaling import symlog_np

    assert symlog_np(np.array(0.0)) == 0.0
    assert symlog_np(np.array(1.0)) == pytest.approx(np.log1p(1.0))
    assert symlog_np(np.array(-1.0)) == pytest.approx(-np.log1p(1.0))
    # Large sentinel value stays bounded.
    big = symlog_np(np.array(1e10))
    assert 22 < big < 24  # log1p(1e10) ≈ 23.03


# ---------------------------------------------------------------------------
# filter_sentinel_mask
# ---------------------------------------------------------------------------

def test_filter_sentinel_mask_detects_any_cost_channel():
    from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX, filter_sentinel_mask,
    )

    rv = np.zeros((3, 4, NUM_REWARDS), dtype=np.float32)
    rv[0, 0, REWARD_INDEX["flops"]] = SENTINEL_REWARD_VALUE
    rv[1, 2, REWARD_INDEX["peak_memory"]] = SENTINEL_REWARD_VALUE
    # cosine_sim is a quality channel — NOT considered for sentinel
    # detection (otherwise normal zero-cos values would be flagged).
    rv[2, 3, REWARD_INDEX["cosine_sim"]] = SENTINEL_REWARD_VALUE

    mask = filter_sentinel_mask(rv, SENTINEL_REWARD_VALUE)
    assert mask.shape == (3, 4)
    assert mask[0, 0] == False  # sentinel on cost channel
    assert mask[1, 2] == False  # sentinel on cost channel
    assert mask[2, 3] == True   # cosine_sim sentinel doesn't disqualify
    # All other cells are valid.
    assert mask.sum() == 3 * 4 - 2


# ---------------------------------------------------------------------------
# aggregate_per_channel_stats
# ---------------------------------------------------------------------------

def test_aggregate_per_channel_stats_shape_and_keys():
    from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX, REWARD_NAMES,
        aggregate_per_channel_stats,
    )

    T, N = 4, 3
    rv = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    # Vary so per-channel best argmax isn't trivially env 0.
    rv[:, 0, REWARD_INDEX["flops"]] = -1e6
    rv[:, 1, REWARD_INDEX["flops"]] = -1e9  # worst
    rv[:, 2, REWARD_INDEX["flops"]] = -1e3  # best (least negative)
    rv[:, 1, REWARD_INDEX["peak_memory"]] = -1e5  # only env 1 has non-zero peak
    rv[:, 0, REWARD_INDEX["cosine_sim"]] = 0.7

    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[REWARD_INDEX["flops"]] = 1.0
    weights[REWARD_INDEX["peak_memory"]] = 1.0
    weights[REWARD_INDEX["cosine_sim"]] = 1.0

    stats = aggregate_per_channel_stats(
        rv, weights, sentinel=SENTINEL_REWARD_VALUE,
    )
    assert set(stats.keys()) == {
        "per_reward_means", "best_per_reward",
        "best_overall_rewards", "best_overall_weighted",
        "best_overall_env", "best_overall_weighted_total",
        # Added with the best-sequence JSON dump: the overall winner's
        # action sequence (empty list when ``action_seq=None``).
        "best_overall_seq",
    }
    # per_reward_means covers every channel.
    assert set(stats["per_reward_means"].keys()) == set(REWARD_NAMES)
    # best_per_reward only covers non-zero-weight channels.
    assert "flops" in stats["best_per_reward"]
    assert "peak_memory" in stats["best_per_reward"]
    assert "muls_adds_fmas" not in stats["best_per_reward"]
    # Env 2 has the best flops; verify.
    assert stats["best_per_reward"]["flops"]["env_idx"] == 2


def test_aggregate_filters_sentinels_from_per_channel_means():
    from alphagrad.approx.common.cache import SENTINEL_REWARD_VALUE
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX, aggregate_per_channel_stats,
    )

    T, N = 2, 2
    rv = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    rv[:, :, REWARD_INDEX["flops"]] = -1e6  # baseline
    # Single sentinel transition.
    rv[0, 0, :] = SENTINEL_REWARD_VALUE
    rv[0, 0, REWARD_INDEX["cosine_sim"]] = 0.0
    rv[0, 0, REWARD_INDEX["frob_residual"]] = SENTINEL_REWARD_VALUE

    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[REWARD_INDEX["flops"]] = 1.0
    stats = aggregate_per_channel_stats(
        rv, weights, sentinel=SENTINEL_REWARD_VALUE,
    )
    # Without the sentinel-mask we'd average in -1e10; with it the
    # mean over the 3 non-sentinel transitions stays at -1e6.
    assert stats["per_reward_means"]["flops"] == pytest.approx(-1e6, rel=1e-3)


# ---------------------------------------------------------------------------
# update_running_bests / format_milestone_line / build_wandb_log_dict
# ---------------------------------------------------------------------------

def test_running_bests_records_improvement():
    from alphagrad.approx.common.reward_scaling import (
        init_running_bests, update_running_bests,
    )

    state = init_running_bests()
    update_running_bests(state, {
        "best_return": -100.0,
        "best_seq": ["a"],
        "best_overall_rewards": {"flops": -100.0},
        "best_overall_weighted": {"flops": -100.0},
        "best_per_reward": {"flops": {"raw_value": -100.0}},
    }, ep=0)
    assert state["best_global_return"] == -100.0
    assert state["best_global_ep"] == 0
    assert state["best_per_reward"]["flops"]["raw_value"] == -100.0

    # No improvement — best stays at ep 0.
    update_running_bests(state, {"best_return": -200.0}, ep=1)
    assert state["best_global_ep"] == 0

    # Improvement — best moves to ep 2.
    update_running_bests(state, {
        "best_return": -50.0,
        "best_overall_rewards": {"flops": -50.0},
        "best_overall_weighted": {"flops": -50.0},
        "best_per_reward": {"flops": {"raw_value": -50.0}},
    }, ep=2)
    assert state["best_global_return"] == -50.0
    assert state["best_global_ep"] == 2
    assert state["best_per_reward"]["flops"]["raw_value"] == -50.0


def test_format_milestone_line_includes_per_channel_blocks():
    from alphagrad.approx.common.reward_scaling import format_milestone_line

    state = {
        "best_global_return": -1234.5,
        "best_global_ep": 2,
        "best_global_rewards": {"flops": -1e6},
        "best_global_weighted_split": {},
        "best_per_reward": {"flops": {"raw_value": -1e6, "weighted_total": -1.0}},
    }
    stats = {
        "mean_return": -2000.0,
        "entropy_mean": 0.123,
        "per_reward_means": {"flops": -1.5e6},
        "buffer_size": 10,
        "train_step": 25,
    }
    line = format_milestone_line("foo", 5, 100, stats, state)
    assert "[foo] ep=   6/100" in line
    assert "best-overall-traj-rewards" in line
    assert "per-channel-best" in line
    assert "mean-per-channel" in line
