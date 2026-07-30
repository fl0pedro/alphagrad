"""Resource-limit truncation vs zero-work — the two must be OPPOSITE.

"Time Limits in Reinforcement Learning" (Pardo et al. 2018): an episode ended
for reasons OUTSIDE the MDP (a clock, a memory limit) must not be treated as a
real terminal. We apply that to resource limits:

  * TRUNCATED (op-count cap, OOM) -> the apparatus gave up; we never measured
    this plan. Emit the exact sentinel, which the trainer recognises and
    EXCLUDES from the gradient (advantage 0, value target = critic's own).
    Scoring it either way teaches something we did not observe.

  * ZERO-WORK (plan computed nothing) -> a real, measurable MDP outcome. KEEP
    it: frob_residual is exactly 1.0 for an all-zero Jacobian, and under PopArt
    that punishment is commensurable with the cost channels. Refusing it would
    be a flat signal (what froze v17 for 22 episodes).
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


def _is_excluded(reward):
    """The trainer's predicate (ppo.train_episode) — mirrored."""
    return bool(jnp.all(reward[..., _SENT_CH] <= (SENTINEL_COST * 0.99)))


def _zero_work_reward(lat_ns=5.6e4, peak=1.32e8):
    """What a zero-work plan ACTUALLY measures now: fake-fast costs, frob=1."""
    r = np.zeros(NUM_REWARDS, np.float32)
    r[REWARD_INDEX["muls_adds_fmas"]] = -0.0
    r[REWARD_INDEX["latency_ns"]] = -lat_ns
    r[REWARD_INDEX["peak_memory"]] = -peak
    r[REWARD_INDEX["cosine_sim"]] = 0.0
    r[REWARD_INDEX["frob_residual"]] = -1.0      # worst attainable
    return jnp.asarray(r)


def test_truncated_is_excluded_from_gradient():
    """A resource-limit refusal must be recognised and dropped."""
    assert _is_excluded(_SENTINEL_BAD_REWARD)


def test_zero_work_is_kept():
    """A zero-work plan must NOT be excluded — frob has to punish it."""
    assert not _is_excluded(_zero_work_reward())


def test_zero_work_is_worst_on_quality():
    """frob = 1.0 is the worst attainable, so quality can rank it last."""
    r = _zero_work_reward()
    assert float(r[REWARD_INDEX["frob_residual"]]) == -1.0


def test_real_plan_is_kept():
    r = np.zeros(NUM_REWARDS, np.float32)
    r[REWARD_INDEX["muls_adds_fmas"]] = -4.2e12
    r[REWARD_INDEX["latency_ns"]] = -7.4e4
    r[REWARD_INDEX["peak_memory"]] = -1.33e8
    r[REWARD_INDEX["frob_residual"]] = -0.62
    assert not _is_excluded(jnp.asarray(r))


def test_expensive_plan_is_not_mistaken_for_truncation():
    """A genuinely huge/slow plan must still train (the ep-39 cliff bug)."""
    r = np.zeros(NUM_REWARDS, np.float32)
    r[REWARD_INDEX["muls_adds_fmas"]] = -9e13
    r[REWARD_INDEX["max_io_sum"]] = -9e12
    r[REWARD_INDEX["latency_ns"]] = -8e9
    r[REWARD_INDEX["peak_memory"]] = -9e9
    r[REWARD_INDEX["frob_residual"]] = -0.95
    assert not _is_excluded(jnp.asarray(r))


def test_oom_detector_matches_xla_texts():
    """_is_oom must catch the strings jaxlib actually raises."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "alphagrad", "approx", "env.py")).read()
    body = src[src.index("def _is_oom("):]
    body = body[: body.index("def _oom_truncate(")]
    for needle in ("RESOURCE_EXHAUSTED", "OUT OF MEMORY",
                   "CUDA_ERROR_OUT_OF_MEMORY", "MemoryError"):
        assert needle in body, f"_is_oom does not mention {needle}"


def test_zero_work_no_longer_returns_sentinel():
    """Regression: the zero-work path must not emit a refusal any more."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "src",
                            "alphagrad", "approx", "env.py")).read()
    assert "ZERO-WORK PLANS ARE KEPT" in src
    blk = src[src.index("ZERO-WORK PLANS ARE KEPT"):]
    blk = blk[: blk.index("return tokens, eqn_ids, rewards")]
    assert "_truncated_reward()" not in blk
    assert "_SENTINEL_BAD_REWARD" not in blk
