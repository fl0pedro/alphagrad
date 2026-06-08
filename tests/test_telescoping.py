"""Tests for the RQ8 max-over-vertices reducer.

Pins both reduction modes:
* ``telescope_increments`` — per-step increments that cumsum to the
  rollout-wide max (GAE-compatible).
* ``buffer_reduce_max`` — direct per-channel max reduction over the time
  axis.

Both must preserve the sign convention (env emits negative cost values).
"""
from __future__ import annotations

import numpy as np

from alphagrad.approx.common.telescoping import (
    buffer_reduce_max,
    telescope_increments,
)


def test_telescope_increments_cumsum_equals_rollout_max():
    """Cumulative sum of telescoped increments must equal the running
    max of the absolute per-step values (with sign restored)."""
    print("\n[telescope] cumsum of increments == rollout-wide max")
    per_step = np.array([-1e8, -3e8, -2e8, -5e8, -4e8], dtype=np.float64)
    inc = telescope_increments(per_step)
    cumsum_abs = np.abs(np.cumsum(inc))
    expected_running_max = np.maximum.accumulate(np.abs(per_step))
    print(f"  per_step={per_step}")
    print(f"  inc=     {inc}")
    print(f"  cumsum(|inc|)={cumsum_abs}")
    print(f"  expected runmax={expected_running_max}")
    np.testing.assert_allclose(cumsum_abs, expected_running_max, rtol=1e-8)


def test_telescope_increments_preserves_sign():
    """The env emits negative cost values; the telescoped stream must
    follow the same sign convention (cumsum is negative)."""
    print("\n[telescope] sign convention preserved")
    per_step_neg = np.array([-1.0, -3.0, -2.0, -5.0])
    per_step_pos = np.array([1.0, 3.0, 2.0, 5.0])
    inc_neg = telescope_increments(per_step_neg)
    inc_pos = telescope_increments(per_step_pos)
    assert (inc_neg <= 0).all(), f"negative-input increments should be ≤ 0: {inc_neg}"
    assert (inc_pos >= 0).all(), f"positive-input increments should be ≥ 0: {inc_pos}"
    print(f"  neg input → {inc_neg}, pos input → {inc_pos}")


def test_telescope_increments_works_on_2d_per_env():
    """Per-step shape (T, N) — telescoping must be applied per env-slot
    independently (axis=0), so two parallel envs with different peak
    schedules don't contaminate each other."""
    print("\n[telescope] (T, N) per-env-independent")
    per_step = np.array([
        [-1.0, -5.0],  # env 0 hits its peak at step 0; env 1 hits at step 0
        [-2.0, -3.0],  # env 0 grows; env 1 dropped from peak (still 5.0)
        [-1.5, -4.0],  # env 0 dropped (peak 2.0); env 1 dropped (peak 5.0)
    ])
    inc = telescope_increments(per_step)
    # Env 0: running max sequence 1, 2, 2 → increments -1, -1, 0
    # Env 1: running max sequence 5, 5, 5 → increments -5, 0, 0
    expected = np.array([
        [-1.0, -5.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
    ])
    print(f"  per_step=\n{per_step}\n  expected=\n{expected}\n  got=\n{inc}")
    np.testing.assert_allclose(inc, expected, rtol=1e-8)


def test_buffer_reduce_max_picks_correct_channels():
    """``buffer_reduce_max`` sums non-max channels and max-reduces the
    designated ones — pins the mixed-reduction semantics."""
    print("\n[telescope] buffer_reduce_max — sum for non-max, max for selected")
    T, N, C = 4, 2, 6
    buf = np.zeros((T, N, C))
    # Channel 0 (sum): all +1 per step → expect 4.0 per env.
    buf[:, :, 0] = 1.0
    # Channel 5 (peak_memory, max): peaks at step 2 with -3, otherwise -1.
    buf[:, :, 5] = -1.0
    buf[2, :, 5] = -3.0  # peak step
    out = buffer_reduce_max(buf, max_channel_indices=[5])
    print(f"  out shape = {out.shape}; channel 0 (sum) = {out[:, 0]}; "
          f"channel 5 (max) = {out[:, 5]}")
    np.testing.assert_allclose(out[:, 0], 4.0)  # sum reduction
    np.testing.assert_allclose(out[:, 5], -3.0)  # max-by-magnitude with sign


def main():
    print("=== RQ8 telescoping reducer tests ===")
    test_telescope_increments_cumsum_equals_rollout_max()
    test_telescope_increments_preserves_sign()
    test_telescope_increments_works_on_2d_per_env()
    test_buffer_reduce_max_picks_correct_channels()
    print("\nALL TELESCOPING TESTS OK")


if __name__ == "__main__":
    main()
