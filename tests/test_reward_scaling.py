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
    assert w.shape == (10,)
    assert w[REWARD_INDEX["flops"]] == 1.0
    assert w[REWARD_INDEX["peak_memory"]] == 1.0
    assert w[REWARD_INDEX["quality"]] == 1.0
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
    for k in ("flops", "peak_memory", "quality", "frob_residual"):
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
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX, filter_sentinel_mask,
    )

    rv = np.zeros((3, 4, NUM_REWARDS), dtype=np.float32)
    rv[0, 0, REWARD_INDEX["flops"]] = SENTINEL_REWARD_VALUE
    rv[1, 2, REWARD_INDEX["peak_memory"]] = SENTINEL_REWARD_VALUE
    # quality is a quality channel — NOT considered for sentinel
    # detection (otherwise normal zero-cos values would be flagged).
    rv[2, 3, REWARD_INDEX["quality"]] = SENTINEL_REWARD_VALUE

    mask = filter_sentinel_mask(rv, SENTINEL_REWARD_VALUE)
    assert mask.shape == (3, 4)
    assert mask[0, 0] == False  # sentinel on cost channel
    assert mask[1, 2] == False  # sentinel on cost channel
    assert mask[2, 3] == True   # quality sentinel doesn't disqualify
    # All other cells are valid.
    assert mask.sum() == 3 * 4 - 2


# ---------------------------------------------------------------------------
# aggregate_per_channel_stats
# ---------------------------------------------------------------------------

def test_aggregate_per_channel_stats_shape_and_keys():
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
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
    rv[:, 0, REWARD_INDEX["quality"]] = 0.7

    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[REWARD_INDEX["flops"]] = 1.0
    weights[REWARD_INDEX["peak_memory"]] = 1.0
    weights[REWARD_INDEX["quality"]] = 1.0

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
        # Added with Infra 1 (unified per-episode logging). Empty
        # dicts when ``dones_mask=None`` (this call site).
        "terminal_means", "best_terminal",
        # Per-channel order statistics over the episode's rollout envs plus
        # the representative sequence at each quantile — lets analysis pull
        # e.g. the median-cosine order instead of only the raw-return best.
        "reward_quantiles", "quantile_sequences",
    }
    # per_reward_means covers every channel.
    assert set(stats["per_reward_means"].keys()) == set(REWARD_NAMES)
    # best_per_reward only covers non-zero-weight channels.
    assert "flops" in stats["best_per_reward"]
    assert "peak_memory" in stats["best_per_reward"]
    assert "muls_adds_fmas" not in stats["best_per_reward"]
    # Env 2 has the best flops; verify.
    assert stats["best_per_reward"]["flops"]["env_idx"] == 2
    # Without dones_mask, terminal_means / best_terminal stay empty.
    assert stats["terminal_means"] == {}
    assert stats["best_terminal"] == {}


def test_aggregate_with_dones_mask_distinguishes_per_step_from_terminal():
    """Sparse-terminal channels (cossim) — per_step mean is heavily diluted by
    intermediate zero-reward steps, terminal mean is the honest signal."""
    import numpy as np
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        aggregate_per_channel_stats,
    )
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE

    T, N = 5, 4
    rv = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    cs_idx = REWARD_INDEX["quality"]
    rv[-1, :, cs_idx] = np.array([0.95, 0.50, 1.00, 0.85], dtype=np.float32)
    dones = np.zeros((T, N), dtype=bool)
    dones[-1, :] = True
    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[cs_idx] = 1.0
    stats = aggregate_per_channel_stats(
        rv, weights, sentinel=SENTINEL_REWARD_VALUE, dones_mask=dones,
    )
    # quality is a SPARSE-TERMINAL channel, so per_reward_means is
    # OVERWRITTEN with the terminal-only mean (reward_scaling.py:512-514):
    # the diluted 3.30/20 = 0.165 per-step figure was exactly the misleading
    # number that overwrite exists to remove.
    assert abs(stats["per_reward_means"]["quality"] - 0.825) < 1e-5
    # A dense cost channel is still a plain per-step mean over all T*N.
    assert abs(stats["per_reward_means"]["flops"]) < 1e-5
    # Terminal: just the 4 terminal entries. Mean = (0.95+0.50+1.0+0.85)/4 = 0.825.
    assert abs(stats["terminal_means"]["quality"] - 0.825) < 1e-5
    # Best terminal: max over the 4 terminal entries = 1.0.
    assert stats["best_terminal"]["quality"] == 1.0


def test_build_unified_reward_log_dict_emits_grouped_keys():
    import numpy as np
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        aggregate_per_channel_stats,
        build_unified_reward_log_dict,
    )
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE

    T, N = 3, 2
    rv = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    rv[-1, 0, REWARD_INDEX["quality"]] = 0.95
    rv[-1, 1, REWARD_INDEX["quality"]] = 0.70
    rv[-1, :, REWARD_INDEX["flops"]] = np.array([-1e6, -2e6], dtype=np.float32)
    dones = np.zeros((T, N), dtype=bool)
    dones[-1, :] = True
    weights = np.ones((NUM_REWARDS,), dtype=np.float32)
    stats = aggregate_per_channel_stats(
        rv, weights, sentinel=SENTINEL_REWARD_VALUE, dones_mask=dones,
    )
    log = build_unified_reward_log_dict(
        stats,
        corridor_low=0.8,
        corridor_high=0.9,
        terminal_cossims=rv[-1, :, REWARD_INDEX["quality"]],
    )
    # Grouping: flops is cost, quality is quality.
    assert "reward/cost/per_step/flops" in log
    assert "reward/cost/terminal/flops" in log
    assert "reward/cost/best_terminal/flops" in log
    assert "reward/quality/per_step/quality" in log
    assert "reward/quality/terminal/quality" in log
    assert "reward/quality/best_terminal/quality" in log
    # Symlog flags surfaced.
    assert log["reward/cost/symlog_applied"] is True
    assert log["reward/quality/symlog_applied"] is False
    # Corridor [0.8, 0.9]: env 0 (0.95) is above, env 1 (0.70) is below.
    assert log["corridor/in_band_fraction"] == 0.0
    assert log["corridor/below_band_fraction"] == 0.5
    assert log["corridor/above_band_fraction"] == 0.5


def test_build_unified_reward_log_dict_ceiling_only_corridor():
    """A ceiling-only corridor passes corridor_low=None."""
    import numpy as np
    from alphagrad.approx.common.reward_scaling import (
        REWARD_NAMES,
        build_unified_reward_log_dict,
    )

    stats = {
        "per_reward_means": {n: 0.0 for n in REWARD_NAMES},
        "terminal_means": {n: 0.0 for n in REWARD_NAMES},
        "best_terminal": {n: 0.0 for n in REWARD_NAMES},
    }
    cossims = np.array([0.85, 0.98, 1.0, 0.5], dtype=np.float32)
    log = build_unified_reward_log_dict(
        stats, corridor_low=None, corridor_high=0.99, terminal_cossims=cossims,
    )
    # below_mask uses -inf as floor → no env is below, 1 env (1.0) is above.
    assert log["corridor/below_band_fraction"] == 0.0
    assert log["corridor/above_band_fraction"] == 0.25
    assert log["corridor/in_band_fraction"] == 0.75
    # corridor/low key omitted when no floor is set.
    assert "corridor/low" not in log
    assert log["corridor/high"] == 0.99


def test_aggregate_filters_sentinels_from_per_channel_means():
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS, REWARD_INDEX, aggregate_per_channel_stats,
    )

    T, N = 2, 2
    rv = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    rv[:, :, REWARD_INDEX["flops"]] = -1e6  # baseline
    # Single sentinel transition.
    rv[0, 0, :] = SENTINEL_REWARD_VALUE
    rv[0, 0, REWARD_INDEX["quality"]] = 0.0
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
