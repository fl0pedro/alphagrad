"""Max-over-vertices reducer for ``peak_memory`` / ``max_io_sum``.

The env emits these as per-step values (peak HBM during that step's
elimination exec, graphax memory accumulator for that step). The
current buffer aggregates with ``sum`` along the rollout — but the
quantity the *downstream training run* actually pays for is the
single worst peak, not the running total. See Pitch A §"Telescoping
increments for max-type channels" in
``docs/experiments/reward_pipeline_pca.md`` for the design.

Two reduction modes are useful, distinguished here so the caller can
pick:

* :func:`telescope_increments` — convert a per-step value series into
  telescoping increments ``max(0, peak_t - running_max_{t-1})`` so the
  cumulative sum equals the rollout-wide max. GAE-compatible: each
  per-step "reward" is non-negative and additive.

* :func:`buffer_reduce_max` — direct per-channel max reduction across
  the rollout, replacing the default ``sum`` reduction in the
  per-episode aggregation. Simpler; same final value as
  ``cumsum(telescope_increments)[-1]``.

Both operate on numpy arrays for symmetry with the rest of
``common/reward_scaling.py`` (the workers consume numpy before GAE).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def telescope_increments(per_step: np.ndarray) -> np.ndarray:
    """Convert a per-step "peak" series to running-max increments.

    Args:
        per_step: shape ``(T,)`` or ``(T, N)``. Per-step peak values
            from the env (assumed non-negative cost magnitudes; the
            sign convention is the caller's, this function works on
            absolute values and preserves sign at the end).

    Returns:
        Same shape as input. Element ``t`` is
        ``max(0, |per_step[t]| - max(|per_step[:t+1]|))`` × sign,
        i.e. how much THIS step contributes to the running max. The
        cumulative sum equals the rollout-wide max — GAE-compatible.

    The sign-preservation handles the env's convention that cost
    channels are negative-valued: a per-step ``peak_memory_t = -1.5e8``
    contributes nothing if a prior step recorded -2.0e8 ("higher peak
    already seen"), or contributes -0.5e8 if it's the new peak.
    """
    arr = np.asarray(per_step, dtype=np.float64)
    if arr.size == 0:
        return arr
    # Work in magnitudes so the running-max semantics are clear, then
    # restore the original sign at the end (per-channel via the sign
    # of the largest-magnitude entry).
    mags = np.abs(arr)
    if mags.ndim == 1:
        running_max = np.maximum.accumulate(mags)
        # Increment at step t is the jump in the running max.
        prev = np.concatenate([[0.0], running_max[:-1]])
        increments = np.maximum(running_max - prev, 0.0)
        # Restore the original cost convention (negative if input was
        # mostly negative).
        sign = -1.0 if np.median(arr) < 0 else 1.0
        return sign * increments.astype(arr.dtype)
    # (T, N): apply per env-slot (axis=0).
    out = np.zeros_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = telescope_increments(arr[:, j])
    return out.astype(arr.dtype)


def buffer_reduce_max(
    per_step_per_channel: np.ndarray,
    max_channel_indices: Iterable[int],
) -> np.ndarray:
    """Reduce per-step-per-channel buffer with ``max`` for the
    designated channels, ``sum`` for the rest.

    Args:
        per_step_per_channel: shape ``(T, N_envs, C)`` cost-magnitude
            buffer as in :data:`ppo_ray_worker.PPORayWorker.buf_reward_vec`.
        max_channel_indices: which channel indices should reduce via
            ``max`` along the time axis instead of ``sum``. Typical:
            indices of ``peak_memory`` and ``max_io_sum`` (channels 5
            and 3 in the canonical
            :data:`alphagrad.approx.env.REWARD_NAMES` ordering).

    Returns:
        Shape ``(N_envs, C)`` per-episode-per-env reduction. Sum reduction
        for channels not in ``max_channel_indices``; ``max(|.|)`` for the
        ones that are (with sign restored from the most-extreme step).

    Pure numpy — call from the worker's per-episode log emission, NOT
    inside the GAE pass (GAE assumes additivity; if you want both
    GAE-correctness and running-max bookkeeping, use
    :func:`telescope_increments` on the buffer BEFORE GAE).
    """
    buf = np.asarray(per_step_per_channel, dtype=np.float64)
    assert buf.ndim == 3, f"expected (T, N, C), got {buf.shape}"
    T, N, C = buf.shape
    max_set = set(int(i) for i in max_channel_indices)
    # Sum reduction along time for everyone, then overwrite the
    # max-reduced columns with the sign-preserving max.
    out = buf.sum(axis=0)  # (N, C)
    for c in max_set:
        if not (0 <= c < C):
            continue
        col = buf[:, :, c]                                  # (T, N)
        mags = np.abs(col)
        max_vals = mags.max(axis=0)                          # (N,)
        # Restore sign of the largest-magnitude entry per env.
        arg = mags.argmax(axis=0)                            # (N,)
        signs = np.sign(col[arg, np.arange(N)])
        # If every value in the column is 0, sign is 0 (preserved as zero).
        out[:, c] = signs * max_vals
    return out
