# -*- coding: utf-8 -*-
"""Length-bucketed delta windows for the PPO update.

WHY THIS EXISTS
---------------
``MAX_DELTA_TOKENS`` is one GLOBAL bound and it is a MEMORY FLOOR, not a
padding bound: every reverse-differentiated ``encode_extend`` materialises a
``(window, E)`` row block per sample per grad-window step whatever the real
delta length is. That is why the ``ALPHAGRAD_EXTEND_CHUNK`` sweep found peak
memory FLAT at 2051 MB across every chunk size -- blocking cannot shrink a
fixed window, only the window can.

So the bound is stuck between two failures:

  * too LOW and the recurrence desyncs. MEASURED on TransformerLM at
    ``MAX_DELTA_TOKENS=4096``: deltas of 4115, 7098, 16345, 19657 tokens, each
    one a dropped measurement. The 4096 default was fitted to a distribution
    (median 0, mean 75-114, p95 537-642, max 2833 over 380/540 steps) that
    never reached the deep-fill orders where this happens.
  * too HIGH and every step pays the ceiling. 32768 -> 4096 was worth
    -45/-62/-66% peak on the TLM at K=1/4/16.

Bucketing DECOUPLES THE CEILING FROM THE COST. A 65536 rung is instantiated
only on the handful of steps that need it; the median step runs at 128. That
is what makes a deliberately over-tall ladder cheap: an unused rung costs a
compile slot, not memory.

WHY BATCH-MAX IS NOT ENOUGH
---------------------------
Rounding the whole batch up to its max pays ~33x the p95 for every sample,
because one deep-fill step drags the entire minibatch to its width. The
grouping below is per-SAMPLE-GROUP: sort a minibatch by delta length,
partition into rungs, evaluate each rung at its own width.

THE PROPERTY THAT KEEPS PPO INTACT
----------------------------------
Per-rung gradients are summed with SAMPLE-COUNT weights and a single optimizer
step is taken per minibatch, so the update is arithmetically what the
unbucketed path computes -- bucketing changes the order of evaluation, not the
objective. Minibatch composition is untouched: a minibatch is still the same
set of samples, just evaluated in length-sorted pieces.

COMPILE COST IS THE THING TO WATCH
----------------------------------
Each distinct ``(window_rung, size_rung)`` pair is an executable, and compile
is already the dominant fixed cost of a run. Both axes are therefore quantised
to ladders, and ``BucketPlan.shapes()`` reports exactly which pairs a plan
will instantiate so a caller can log it rather than discover it.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# Window rungs. Geometric, so a sample wastes at most ~4x its own length, and
# deliberately taller than any delta measured so far -- see module docstring.
DEFAULT_LADDER = (128, 512, 2048, 8192, 32768, 65536)

# Sample-axis rungs. Without these, every distinct group SIZE is a separate
# shape and the compile count explodes; with them the count is bounded by
# len(ladder) * len(size ladder).
DEFAULT_SIZE_LADDER = (8, 32, 128, 512, 2048)


def _env_ladder(name: str, default: Sequence[int]) -> tuple:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return tuple(default)
    try:
        vals = tuple(sorted({int(x) for x in raw.replace(",", " ").split()}))
    except ValueError:
        raise ValueError("%s must be a comma/space separated integer list, "
                         "got %r" % (name, raw))
    if not vals or vals[0] <= 0:
        raise ValueError("%s must be positive integers, got %r" % (name, raw))
    return vals


def window_ladder() -> tuple:
    return _env_ladder("ALPHAGRAD_DELTA_LADDER", DEFAULT_LADDER)


def size_ladder() -> tuple:
    return _env_ladder("ALPHAGRAD_DELTA_SIZE_LADDER", DEFAULT_SIZE_LADDER)


def rung_for(n: int, ladder: Sequence[int]) -> int:
    """Smallest rung >= n.

    Raises rather than clipping if n exceeds the top rung: a silently clipped
    delta desyncs the recurrence for the rest of the episode, which is the
    exact failure mode ``ALPHAGRAD_DELTA_OVERFLOW`` was made loud to prevent.
    """
    for r in ladder:
        if n <= r:
            return r
    raise ValueError(
        "delta length %d exceeds the top ladder rung %d. Raise "
        "ALPHAGRAD_DELTA_LADDER; do NOT clip -- dropped delta tokens are not "
        "re-read at the next step, so the encoder desyncs from the stream."
        % (n, ladder[-1]))


class BucketPlan:
    """Which samples are evaluated at which (window, size) shape.

    ``groups`` is a list of ``(window, size, idx)`` where ``idx`` is an int
    array of positions into the minibatch, already padded up to ``size`` by
    REPEATING the first index. Padding by repetition rather than by a sentinel
    keeps every lane a valid sample, so no lane can produce a NaN that a mask
    would then have to multiply by zero -- the ``jnp.where`` gradient trap in
    another guise. ``weights`` carries 1.0 for real lanes and 0.0 for padded
    ones, so padded lanes contribute nothing to the loss or its gradient.
    """

    __slots__ = ("groups", "n_samples")

    def __init__(self, groups, n_samples):
        self.groups = groups
        self.n_samples = n_samples

    def shapes(self):
        return sorted({(w, s) for w, s, _, _ in self.groups})

    def total_lane_work(self):
        """sum(size * window) -- the quantity bucketing exists to minimise."""
        return sum(s * w for w, s, _, _ in self.groups)

    def describe(self):
        return "; ".join(
            "%d@w%d(s%d)" % (int(wt.sum()), w, s) for w, s, _, wt in self.groups)


def plan_buckets(counts, ladder=None, sizes=None) -> BucketPlan:
    """Partition minibatch sample indices by delta length.

    ``counts`` is the per-sample delta length -- for a K-step gradient window
    the caller passes the per-sample MAX over K, because one call encodes all K
    deltas at a single width.
    """
    ladder = tuple(ladder) if ladder is not None else window_ladder()
    sizes = tuple(sizes) if sizes is not None else size_ladder()
    counts = np.asarray(counts).reshape(-1)
    n = int(counts.shape[0])

    want = np.array([rung_for(int(c), ladder) for c in counts], dtype=np.int64)
    groups = []
    for w in sorted(set(want.tolist())):
        idx = np.nonzero(want == w)[0].astype(np.int64)
        # Split a large group across size rungs largest-first so the tail is
        # one small padded call rather than one huge one.
        pos = 0
        while pos < idx.shape[0]:
            remaining = idx.shape[0] - pos
            s = next((z for z in sizes if z >= remaining), sizes[-1])
            take = min(remaining, s)
            chunk = idx[pos:pos + take]
            pad = s - chunk.shape[0]
            if pad:
                chunk = np.concatenate([chunk, np.repeat(chunk[:1], pad)])
            wt = np.zeros((s,), dtype=np.float32)
            wt[:take] = 1.0
            groups.append((int(w), int(s), chunk, wt))
            pos += take
    return BucketPlan(groups, n)


def flat_window_cost(counts, window: int) -> int:
    """What the UNBUCKETED path costs: every sample at the global window."""
    return int(np.asarray(counts).reshape(-1).shape[0]) * int(window)
