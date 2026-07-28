# -*- coding: utf-8 -*-
"""The ep-39 cliff: degeneracy detection must identify the SENTINEL VECTOR,
not merely "some channel is a big negative number".

The old predicate was ``any(reward <= SENTINEL_COST*0.5)`` over all eight
channels. Channels 0 (-muls_adds_fmas) and 3 (-max_io_sum) are raw symbolic
counts, legitimately ~1e12 on nn256 — so every real plan tripped it and had
its advantage zeroed, leaving only near-zero-work plans with a policy
gradient. These pin the corrected predicate.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from alphagrad.approx.env import (
    COMPUTE_REWARD_INDICES, NUM_REWARDS, REWARD_INDEX, SENTINEL_COST,
    _SENTINEL_BAD_REWARD,
)

_SENT_CH = jnp.asarray(COMPUTE_REWARD_INDICES, dtype=jnp.int32)


def _is_degen(reward):
    """The corrected predicate, mirrored from ppo.train_episode."""
    return jnp.all(reward[..., _SENT_CH] <= (SENTINEL_COST * 0.99), axis=-1)


def _healthy(muls=4.2e12, io=2.8e8, lat_ns=7.4e4, peak=1.33e8,
             cos=0.79, frob=0.62):
    r = np.zeros(NUM_REWARDS, np.float32)
    r[REWARD_INDEX["muls_adds_fmas"]] = -muls
    r[REWARD_INDEX["flops"]] = 0.0
    r[REWARD_INDEX["latency_ns"]] = -lat_ns
    r[REWARD_INDEX["max_io_sum"]] = -io
    r[REWARD_INDEX["bytes_accessed"]] = 0.0
    r[REWARD_INDEX["peak_memory"]] = -peak
    r[REWARD_INDEX["cosine_sim"]] = cos
    r[REWARD_INDEX["frob_residual"]] = -frob
    return jnp.asarray(r)


def test_real_plan_is_not_degenerate():
    """The regression: 4.2e12 muls must NOT read as degenerate."""
    assert not bool(_is_degen(_healthy()))
    # The old predicate DID flag it — proof the bug was real, not theoretical.
    old = bool(jnp.any(_healthy() <= (SENTINEL_COST * 0.5)))
    assert old, "old predicate should have (wrongly) flagged a healthy plan"


def test_expensive_plan_is_not_degenerate():
    """A genuinely slow, memory-hungry, huge-op plan is still trainable."""
    r = _healthy(muls=9e13, io=9e12, lat_ns=8e9, peak=9e9, cos=0.1, frob=0.95)
    assert not bool(_is_degen(r))


def test_sentinel_is_degenerate():
    assert bool(_is_degen(_SENTINEL_BAD_REWARD))


def test_soft_degen_reward_is_trainable():
    """The soft-worst vector (ALPHAGRAD_DEGEN_SOFT=1) must NOT be classified
    as a sentinel — it is meant to carry an ordinary negative advantage."""
    r = np.zeros(NUM_REWARDS, np.float32)
    r[REWARD_INDEX["muls_adds_fmas"]] = -0.0
    r[REWARD_INDEX["latency_ns"]] = -1e8      # 100 ms
    r[REWARD_INDEX["peak_memory"]] = -2e9     # 2 GB
    r[REWARD_INDEX["frob_residual"]] = -1.0
    assert not bool(_is_degen(jnp.asarray(r)))


def test_batch_shape():
    batch = jnp.stack([
        jnp.stack([_healthy(), _SENTINEL_BAD_REWARD]),
        jnp.stack([_healthy(muls=1e13), _healthy()]),
    ])                                        # (E=2, T=2, R)
    d = _is_degen(batch)
    assert d.shape == (2, 2)
    assert np.array_equal(np.asarray(d), np.array([[False, True],
                                                   [False, False]]))


if __name__ == "__main__":
    test_real_plan_is_not_degenerate()
    test_expensive_plan_is_not_degenerate()
    test_sentinel_is_degenerate()
    test_soft_degen_reward_is_trainable()
    test_batch_shape()
    print("ALL PASS")
