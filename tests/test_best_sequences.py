"""Tests for the per-channel + overall best-sequence logging.

Covers the post-refactor additions to
``alphagrad.approx.common.reward_scaling``:

* ``aggregate_per_channel_stats`` now records the full 10-channel
  ``(a_i, b_i, c_i, r_i)`` tuple for every per-channel best (via
  ``all_raw`` / ``all_weighted``) plus the overall winning env's seq.
* ``update_running_bests`` preserves those tuples across episodes.
* ``best_sequences_snapshot`` / ``dump_best_sequences_json`` /
  ``build_best_sequences_wandb_payload`` produce the user-facing JSON
  + wandb log payload.

The tests construct synthetic ``buf_reward_vec`` tensors where the
per-channel argmax env is hand-controlled, then check the recorded
tuples + sequences match.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# aggregate_per_channel_stats: cross-channel tuples + sequences
# ---------------------------------------------------------------------------

def test_aggregate_records_full_reward_tuple_per_channel():
    """The env that wins each channel must have its FULL 10-channel
    raw + weighted vector recorded under ``all_raw`` / ``all_weighted``
    so the JSON dump can report ``(a_i, b_i, c_i, r_i)`` for every
    per-channel best — not just the winning channel's value."""
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        REWARD_NAMES,
        aggregate_per_channel_stats,
    )

    T, N = 3, 4
    buf = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    flops_idx = REWARD_INDEX["flops"]
    peak_idx = REWARD_INDEX["peak_memory"]
    cos_idx = REWARD_INDEX["cosine_sim"]

    # Env 0 wins on flops; env 2 wins on peak_memory; env 3 wins on cosine_sim.
    # The "tuple" we want recorded for the per-channel best is the full vector
    # the winning env carried at the time. Hand-pick values so they're all
    # distinguishable.
    buf[:, 0, flops_idx] = 5.0       # env 0: flops sum = 15
    buf[:, 0, peak_idx] = 1.0        # env 0: peak sum = 3
    buf[:, 0, cos_idx] = 0.1         # env 0: cos sum = 0.3

    buf[:, 2, flops_idx] = 1.0       # env 2: flops sum = 3
    buf[:, 2, peak_idx] = 4.0        # env 2: peak sum = 12 (WIN)
    buf[:, 2, cos_idx] = 0.2         # env 2: cos sum = 0.6

    buf[:, 3, flops_idx] = 0.5       # env 3: flops sum = 1.5
    buf[:, 3, peak_idx] = 0.5        # env 3: peak sum = 1.5
    buf[:, 3, cos_idx] = 0.9         # env 3: cos sum = 2.7  (WIN cos)

    # weights — only flops/peak/cos are tuned so the others get dropped
    # from best_per_reward (matches the live ppo args).
    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[flops_idx] = 1.0
    weights[peak_idx] = 1.0
    weights[cos_idx] = 1.0

    seqs = [[("env0", t) for t in range(T)] for _ in range(N)]
    seqs[0] = [("env0", t) for t in range(T)]
    seqs[2] = [("env2", t) for t in range(T)]
    seqs[3] = [("env3", t) for t in range(T)]

    stats = aggregate_per_channel_stats(
        buf, weights, sentinel=-1e10, action_seq=seqs,
    )

    bpr = stats["best_per_reward"]
    # flops winner is env 0 → its all_raw["peak_memory"] must be its
    # peak sum (3.0), not env 2's 12.0.
    assert bpr["flops"]["env_idx"] == 0
    assert bpr["flops"]["all_raw"]["flops"] == 15.0
    assert bpr["flops"]["all_raw"]["peak_memory"] == 3.0
    assert bpr["flops"]["all_raw"]["cosine_sim"] == pytest.approx(0.3, rel=1e-5)
    assert bpr["flops"]["seq"] == seqs[0]

    # peak_memory winner is env 2 → its all_raw must reflect ENV 2's values.
    assert bpr["peak_memory"]["env_idx"] == 2
    assert bpr["peak_memory"]["all_raw"]["peak_memory"] == 12.0
    assert bpr["peak_memory"]["all_raw"]["flops"] == 3.0
    assert bpr["peak_memory"]["all_raw"]["cosine_sim"] == pytest.approx(0.6, rel=1e-5)
    assert bpr["peak_memory"]["seq"] == seqs[2]

    # cosine_sim winner is env 3 → tuple should be env 3's full row.
    assert bpr["cosine_sim"]["env_idx"] == 3
    assert bpr["cosine_sim"]["all_raw"]["cosine_sim"] == pytest.approx(2.7, rel=1e-5)
    assert bpr["cosine_sim"]["all_raw"]["flops"] == pytest.approx(1.5, rel=1e-5)
    assert bpr["cosine_sim"]["all_raw"]["peak_memory"] == pytest.approx(1.5, rel=1e-5)
    assert bpr["cosine_sim"]["seq"] == seqs[3]

    # Untuned channels (e.g. muls_adds_fmas) must NOT appear — they had
    # zero weight, the live trainers don't want noise in the JSON.
    for ch in REWARD_NAMES:
        if ch not in {"flops", "peak_memory", "cosine_sim"}:
            assert ch not in bpr


def test_aggregate_records_best_overall_seq():
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        aggregate_per_channel_stats,
    )

    T, N = 2, 3
    buf = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    flops_idx = REWARD_INDEX["flops"]
    buf[:, 0, flops_idx] = 1.0
    buf[:, 1, flops_idx] = 5.0   # WIN overall
    buf[:, 2, flops_idx] = 2.0

    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[flops_idx] = 1.0

    seqs = [
        [("env0", t) for t in range(T)],
        [("env1", t) for t in range(T)],
        [("env2", t) for t in range(T)],
    ]
    stats = aggregate_per_channel_stats(
        buf, weights, sentinel=-1e10, action_seq=seqs,
    )
    assert stats["best_overall_env"] == 1
    assert stats["best_overall_seq"] == seqs[1]


# ---------------------------------------------------------------------------
# update_running_bests preserves full tuples + seqs
# ---------------------------------------------------------------------------

def test_running_bests_keeps_per_channel_seqs_across_episodes():
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        REWARD_NAMES,
        aggregate_per_channel_stats,
        init_running_bests,
        update_running_bests,
    )

    flops_idx = REWARD_INDEX["flops"]
    cos_idx = REWARD_INDEX["cosine_sim"]

    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[flops_idx] = 1.0
    weights[cos_idx] = 1.0

    def _stats_for(flops_winner_value, cos_winner_value, ep_seq_label):
        T, N = 2, 2
        buf = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        # env 0 wins flops, env 1 wins cosine_sim
        buf[:, 0, flops_idx] = flops_winner_value / T
        buf[:, 1, cos_idx] = cos_winner_value / T
        seqs = [
            [(ep_seq_label, "env0", t) for t in range(T)],
            [(ep_seq_label, "env1", t) for t in range(T)],
        ]
        return aggregate_per_channel_stats(
            buf, weights, sentinel=-1e10, action_seq=seqs,
        )

    state = init_running_bests()

    # Ep 0: flops=3, cos=0.5
    s0 = _stats_for(3.0, 0.5, "ep0")
    s0["best_return"] = s0["best_overall_weighted_total"]
    s0["best_seq"] = s0["best_overall_seq"]
    update_running_bests(state, s0, 0)
    assert state["best_per_reward"]["flops"]["raw_value"] == 3.0
    assert state["best_per_reward"]["flops"]["seq"] == [
        ("ep0", "env0", t) for t in range(2)
    ]
    # cosine_sim winner from ep0
    assert state["best_per_reward"]["cosine_sim"]["raw_value"] == pytest.approx(0.5)

    # Ep 1: flops=10 (NEW BEST), cos=0.2 (worse, ignored)
    s1 = _stats_for(10.0, 0.2, "ep1")
    s1["best_return"] = s1["best_overall_weighted_total"]
    s1["best_seq"] = s1["best_overall_seq"]
    update_running_bests(state, s1, 1)
    assert state["best_per_reward"]["flops"]["raw_value"] == 10.0
    assert state["best_per_reward"]["flops"]["seq"] == [
        ("ep1", "env0", t) for t in range(2)
    ]
    # cosine_sim should still be the ep0 winner
    assert state["best_per_reward"]["cosine_sim"]["raw_value"] == pytest.approx(0.5)
    assert state["best_per_reward"]["cosine_sim"]["seq"] == [
        ("ep0", "env1", t) for t in range(2)
    ]
    # ep field is set
    assert state["best_per_reward"]["flops"]["ep"] == 1
    assert state["best_per_reward"]["cosine_sim"]["ep"] == 0


# ---------------------------------------------------------------------------
# JSON dump round-trip
# ---------------------------------------------------------------------------

def test_dump_and_reload_best_sequences_json_preserves_tuples():
    from alphagrad.approx.common.reward_scaling import (
        NUM_REWARDS,
        REWARD_INDEX,
        REWARD_NAMES,
        aggregate_per_channel_stats,
        best_sequences_snapshot,
        dump_best_sequences_json,
        init_running_bests,
        update_running_bests,
    )

    flops_idx = REWARD_INDEX["flops"]
    cos_idx = REWARD_INDEX["cosine_sim"]
    weights = np.zeros((NUM_REWARDS,), dtype=np.float32)
    weights[flops_idx] = 1.0
    weights[cos_idx] = 2.0

    T, N = 2, 3
    buf = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
    buf[:, 0, flops_idx] = 4.0
    buf[:, 1, flops_idx] = 1.0
    buf[:, 1, cos_idx] = 0.5
    buf[:, 2, flops_idx] = 2.0
    buf[:, 2, cos_idx] = 0.1
    seqs = [[("e0", t) for t in range(T)], [("e1", t) for t in range(T)], [("e2", t) for t in range(T)]]

    stats = aggregate_per_channel_stats(
        buf, weights, sentinel=-1e10, action_seq=seqs,
    )
    stats["best_return"] = stats["best_overall_weighted_total"]
    stats["best_seq"] = stats["best_overall_seq"]
    state = init_running_bests()
    update_running_bests(state, stats, 7)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "subdir", "best_sequences.json")
        written = dump_best_sequences_json(state, path)
        assert written == path
        with open(path, "r") as f:
            blob = json.load(f)

    # Top-level structure
    assert "best_overall" in blob
    assert "best_per_channel" in blob
    # The flops winner is env 0 — its per-channel cosine_sim must be 0
    # (env 0 had 0 cosine_sim entries) and must round-trip exactly.
    flops_entry = blob["best_per_channel"]["flops"]
    assert flops_entry["ep"] == 7
    assert flops_entry["raw_value"] == pytest.approx(8.0)
    assert flops_entry["all_raw"]["flops"] == pytest.approx(8.0)
    assert flops_entry["all_raw"]["cosine_sim"] == pytest.approx(0.0)
    assert flops_entry["seq"] == [["e0", t] for t in range(T)]

    # The cosine_sim winner is env 1 — record its flops contribution too.
    cos_entry = blob["best_per_channel"]["cosine_sim"]
    assert cos_entry["all_raw"]["cosine_sim"] == pytest.approx(1.0)
    assert cos_entry["all_raw"]["flops"] == pytest.approx(2.0)
    assert cos_entry["seq"] == [["e1", t] for t in range(T)]

    # Overall best is whoever maxes flops*1 + cos*2; env 1: 2 + 2 = 4,
    # env 0: 8 + 0 = 8, env 2: 4 + 0.4 = 4.4 → env 0 wins.
    assert blob["best_overall"]["rewards_raw"]["flops"] == pytest.approx(8.0)
    assert blob["best_overall"]["seq"] == [["e0", t] for t in range(T)]


# ---------------------------------------------------------------------------
# wandb-loggable payload — flat keys + correctly namespaced
# ---------------------------------------------------------------------------

def test_wandb_payload_emits_per_channel_namespace():
    from alphagrad.approx.common.reward_scaling import (
        build_best_sequences_wandb_payload,
        init_running_bests,
    )

    state = init_running_bests()
    state["best_global_return"] = 42.0
    state["best_global_ep"] = 3
    state["best_global_seq"] = [0, 1, 2]
    state["best_global_rewards"] = {"flops": 4.0, "cosine_sim": 0.9}
    state["best_global_weighted_split"] = {"flops": 4.0, "cosine_sim": 1.8}
    state["best_per_reward"] = {
        "flops": {
            "raw_value": 9.0,
            "weighted_value": 9.0,
            "weighted_total": 10.0,
            "env_idx": 1,
            "ep": 2,
            "seq": [9, 8, 7],
            "all_raw": {"flops": 9.0, "cosine_sim": 0.1},
            "all_weighted": {"flops": 9.0, "cosine_sim": 0.2},
        },
    }

    payload = build_best_sequences_wandb_payload(state, ep=3)
    assert payload["best_sequences/overall_return"] == 42.0
    assert payload["best_sequences/overall_ep"] == 3
    assert payload["best_sequences/overall/rewards_raw/flops"] == 4.0
    assert payload["best_sequences/overall/seq_len"] == 3
    assert payload["best_sequences/per_channel/flops/raw_value"] == 9.0
    assert payload["best_sequences/per_channel/flops/weighted_total"] == 10.0
    assert payload["best_sequences/per_channel/flops/ep"] == 2
    assert payload["best_sequences/per_channel/flops/seq_len"] == 3
    assert payload["best_sequences/per_channel/flops/rewards_raw/flops"] == 9.0
    assert payload["best_sequences/per_channel/flops/rewards_raw/cosine_sim"] == 0.1


def test_dump_handles_empty_state():
    """Run-with-no-progress (no episodes yet) must NOT crash the dump."""
    from alphagrad.approx.common.reward_scaling import (
        dump_best_sequences_json,
        init_running_bests,
    )

    state = init_running_bests()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty.json")
        written = dump_best_sequences_json(state, path)
        assert written == path
        with open(path, "r") as f:
            blob = json.load(f)
    assert blob["best_overall"]["ep"] == -1
    assert blob["best_per_channel"] == {}
