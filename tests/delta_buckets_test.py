# -*- coding: utf-8 -*-
"""Properties the bucket plan must hold before it touches the training loop."""
import numpy as np
import pytest

from alphagrad.approx.common.delta_buckets import (
    DEFAULT_LADDER, BucketPlan, flat_window_cost, plan_buckets, rung_for,
    window_ladder, size_ladder)


def test_rung_is_the_smallest_that_fits():
    assert rung_for(0, DEFAULT_LADDER) == 128
    assert rung_for(128, DEFAULT_LADDER) == 128
    assert rung_for(129, DEFAULT_LADDER) == 512
    assert rung_for(19657, DEFAULT_LADDER) == 32768


def test_over_ladder_raises_rather_than_clipping():
    """A clipped delta desyncs the recurrence -- it must never be silent."""
    with pytest.raises(ValueError, match="exceeds the top ladder rung"):
        rung_for(DEFAULT_LADDER[-1] + 1, DEFAULT_LADDER)


def test_every_sample_appears_exactly_once_with_weight_one():
    """The property that keeps the summed loss equal to the unbucketed one."""
    rng = np.random.RandomState(0)
    counts = rng.randint(0, 3000, size=257)
    plan = plan_buckets(counts)
    seen = np.zeros(counts.shape[0], dtype=np.float64)
    for _w, _s, idx, wt in plan.groups:
        np.add.at(seen, idx, wt)
    assert np.all(seen == 1.0), "a sample was dropped or double-counted"


def test_each_sample_lands_in_a_rung_that_fits_it():
    rng = np.random.RandomState(1)
    counts = rng.randint(0, 30000, size=97)
    for w, _s, idx, wt in plan_buckets(counts).groups:
        real = idx[wt > 0]
        assert counts[real].max() <= w
        # and not needlessly oversized: some sample in the group needs this rung
        lower = [r for r in window_ladder() if r < w]
        if lower:
            assert counts[real].max() > lower[-1]


def test_padding_lanes_repeat_a_real_index_not_a_sentinel():
    """Padded lanes must be VALID samples with zero weight.

    A sentinel index (-1, or an out-of-range row) would be gathered anyway
    under vmap and could produce a non-finite value that the zero weight then
    multiplies -- 0 * NaN = NaN. Same class as the jnp.where gradient trap.
    """
    plan = plan_buckets([5])
    for _w, s, idx, wt in plan.groups:
        assert idx.shape[0] == s
        assert idx.min() >= 0
        assert wt.sum() == 1.0
        assert np.all(idx[wt == 0] == idx[0])


def test_group_sizes_are_quantised_so_compiles_stay_bounded():
    rng = np.random.RandomState(2)
    sizes = set()
    for trial in range(40):
        counts = rng.randint(0, 40000, size=rng.randint(1, 400))
        for w, s, _idx, _wt in plan_buckets(counts).groups:
            sizes.add((w, s))
    assert sizes <= {(w, s) for w in window_ladder() for s in size_ladder()}
    assert len(sizes) <= len(window_ladder()) * len(size_ladder())


def test_bucketing_beats_the_flat_window_on_a_realistic_distribution():
    """Measured TLM shape: mostly tiny, a long tail, rare deep-fill spikes."""
    rng = np.random.RandomState(3)
    counts = np.concatenate([
        np.zeros(500, dtype=int),                      # median 0
        rng.randint(1, 700, size=400),                 # bulk
        rng.randint(700, 3000, size=90),               # p99 tail
        rng.randint(4000, 20000, size=10),             # deep-fill spikes
    ])
    plan = plan_buckets(counts)
    flat = flat_window_cost(counts, 32768)             # ladder-safe global bound
    assert plan.total_lane_work() < flat / 4, (
        "bucketed=%d flat=%d" % (plan.total_lane_work(), flat))


def test_single_rung_ladder_reproduces_the_flat_layout():
    """The gate configuration: one rung == today's behaviour, one shape."""
    counts = np.array([0, 5, 100, 4000])
    plan = plan_buckets(counts, ladder=(8192,), sizes=(4,))
    assert plan.shapes() == [(8192, 4)]
    assert len(plan.groups) == 1
    _w, _s, idx, wt = plan.groups[0]
    assert list(idx) == [0, 1, 2, 3] and list(wt) == [1, 1, 1, 1]


def test_empty_minibatch_is_not_a_silent_nan():
    plan = plan_buckets([])
    assert plan.groups == [] and plan.n_samples == 0
