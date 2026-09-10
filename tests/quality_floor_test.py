"""--quality-floor TAU and the trainer-side halves of ticket dsnn-3qm.9.

1. THE HINGE. With the flag set, reward slot 6 = -max(0, tau - q) on the
   terminal step: q above tau gives EXACTLY 0, q below gives the linear
   shortfall, no clip; every other slot and every non-terminal step is
   untouched, so it composes with any preference over the heads
   (--preference-conditioned, arm P1) and with --cost-form.
2. Arm L: --reward-mode lagrangian composes with --preference-conditioned
   (`_lag_preferences`): the Dirichlet sample is restricted to the two cost
   heads and lambda takes the quality slot; the unconditioned path is
   bit-identical to before.
3. Symlog: the hinge slot is exempt under --quality-floor, slots 2 and 5 are
   exempt under --cost-form paired-log, and the flag-off mask is bit-identical
   to the module import state.
4. The flags exist, with the documented defaults (paired-log; floor off).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.ppo import (                              # noqa: E402
    HEAD_NAMES,
    NUM_VALUE_HEADS,
    _apply_lagrangian_channels,
    _apply_quality_floor,
    _lag_preferences,
    _set_no_symlog_indices,
    _symlog_rewards,
    configure_symlog,
    make_argparser,
)
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

Q = int(REWARD_INDEX["cosine_sim"])
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])
QHEAD = HEAD_NAMES.index("quality")


def _rewards(q_terminal, T=5, seed=0):
    q_terminal = np.asarray(q_terminal, dtype=np.float32)
    E = q_terminal.shape[0]
    rng = np.random.default_rng(seed)
    r = np.zeros((E, T, NUM_REWARDS), dtype=np.float32)
    r[..., LAT] = -rng.uniform(1e4, 3e5, (E, T))
    r[..., MEM] = -rng.uniform(1e2, 9e7, (E, T))
    r[:, -1, Q] = q_terminal
    return jnp.asarray(r)


# ------------------------------------------------------------------ 1. hinge

def test_hinge_is_exactly_zero_above_tau_and_linear_below():
    tau = 0.9
    qs = np.array([1.0, 0.95, 0.9, 0.85, 0.5, 0.0, -1.0], dtype=np.float32)
    r = _rewards(qs)
    out = np.asarray(_apply_quality_floor(r, tau))
    term = out[:, -1, Q]
    assert term[0] == 0.0 and term[1] == 0.0        # exactly, not approx
    assert term[2] == 0.0                           # at the floor: feasible
    assert term[3] == pytest.approx(-(0.9 - 0.85), abs=1e-7)
    assert term[4] == pytest.approx(-0.4, abs=1e-7)
    assert term[5] == pytest.approx(-0.9, abs=1e-7)
    assert term[6] == pytest.approx(-1.9, abs=1e-7)  # no clip: linear all the way
    # slope -1 everywhere below tau
    d = np.diff(term[3:])
    dq = np.diff(qs[3:])
    np.testing.assert_allclose(d / dq, 1.0, rtol=1e-5)


def test_hinge_touches_nothing_else():
    r = _rewards([0.3, 0.99])
    out = np.asarray(_apply_quality_floor(r, 0.9))
    base = np.asarray(r)
    keep = [i for i in range(NUM_REWARDS) if i != Q]
    np.testing.assert_array_equal(out[..., keep], base[..., keep])
    # non-terminal steps of the quality slot stay 0
    np.testing.assert_array_equal(out[:, :-1, Q], 0.0)


def test_hinge_agrees_with_the_lagrangian_channel_inside_the_clip_range():
    """One floor, two code paths: for q in [-0.5, 1] the P1 hinge and the L
    arm's violation channel are the same number. They differ only below
    q_eff = -0.5, where the lagrangian path keeps its loss_drop diverged-
    sentinel clip."""
    qs = np.linspace(-0.5, 1.0, 16).astype(np.float32)
    r = _rewards(qs)
    a = np.asarray(_apply_quality_floor(r, 0.75))
    b = np.asarray(_apply_lagrangian_channels(r, 0.75))
    np.testing.assert_allclose(a, b, atol=1e-7)


# ------------------------------------------------------------------ 2. arm L

def test_lagrangian_preferences_unconditioned_are_bit_identical_to_before():
    hw = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    static = jnp.broadcast_to(jnp.asarray(hw), (4, NUM_VALUE_HEADS))
    out = np.asarray(_lag_preferences(static, hw, 3.5, conditioned=False))
    want = np.array(hw, copy=True)
    want[QHEAD] = 3.5
    np.testing.assert_array_equal(out, np.broadcast_to(want, (4, NUM_VALUE_HEADS)))


def test_lagrangian_preferences_conditioned_are_a_cost_simplex_plus_lambda():
    rng = np.random.default_rng(0)
    sampled = jnp.asarray(rng.dirichlet(np.full(NUM_VALUE_HEADS, 0.5), size=64)
                          .astype(np.float32))
    hw = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    out = np.asarray(_lag_preferences(sampled, hw, 2.25, conditioned=True))
    cost = [i for i in range(NUM_VALUE_HEADS) if i != QHEAD]
    np.testing.assert_allclose(out[:, cost].sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(out[:, QHEAD], np.float32(2.25))
    # (x_lat, x_mem) / (x_lat + x_mem): the sample's own ratio survives
    s = np.asarray(sampled)
    np.testing.assert_allclose(
        out[:, cost[0]] / out[:, cost[1]], s[:, cost[0]] / s[:, cost[1]],
        rtol=1e-4)


# ------------------------------------------------------------------ 3. symlog

def _ns(**kw):
    base = dict(symlog_channels="all", no_symlog=False, reward_mode="additive",
                advantage_norm="none", lambda_cmp=1.0, lambda_mem=1.0,
                lambda_acc=16.0)
    base.update(kw)
    return argparse.Namespace(**base)


def _reset():
    _set_no_symlog_indices(())
    ppo._NO_SYMLOG_ALL[0] = False


def test_symlog_exemptions_follow_the_flags():
    try:
        assert configure_symlog(_ns()) == "all"
        assert ppo._NO_SYMLOG_REWARD_INDICES == ()          # flag-off: HEAD
        configure_symlog(_ns(cost_form="absolute", quality_floor=None))
        assert ppo._NO_SYMLOG_REWARD_INDICES == ()
        configure_symlog(_ns(quality_floor=0.9))
        assert ppo._NO_SYMLOG_REWARD_INDICES == (Q,)
        configure_symlog(_ns(cost_form="paired-log"))
        assert ppo._NO_SYMLOG_REWARD_INDICES == (LAT, MEM)
        configure_symlog(_ns(cost_form="paired-log", quality_floor=0.9))
        assert set(ppo._NO_SYMLOG_REWARD_INDICES) == {Q, LAT, MEM}
        # lagrangian + floor: the quality slot is exempt exactly once
        configure_symlog(_ns(reward_mode="lagrangian", quality_floor=0.9))
        assert ppo._NO_SYMLOG_REWARD_INDICES == (Q,)
        r = _rewards([0.5, 0.9])
        r = r.at[..., LAT].set(-0.3).at[..., MEM].set(2.1)
        configure_symlog(_ns(cost_form="paired-log", quality_floor=0.9))
        out = np.asarray(_symlog_rewards(_apply_quality_floor(r, 0.9)))
        # raw log-differences pass through bitwise (float32 in, float32 out)
        np.testing.assert_array_equal(out[..., LAT], np.float32(-0.3))
        np.testing.assert_array_equal(out[..., MEM], np.float32(2.1))
        assert out[0, -1, Q] == pytest.approx(-0.4, abs=1e-7)  # raw hinge
        assert out[1, -1, Q] == 0.0
    finally:
        _reset()


# ------------------------------------------------------------------ 4. flags

def test_flags_and_defaults():
    p = make_argparser()
    a = p.parse_args([])
    assert a.cost_form == "paired-log"                  # the campaign default
    assert a.quality_floor is None                      # P0: raw quality
    a = p.parse_args(["--cost-form", "absolute", "--quality-floor", "0.9"])
    assert a.cost_form == "absolute" and a.quality_floor == 0.9
    with pytest.raises(SystemExit):
        p.parse_args(["--cost-form", "ratio"])
    # The env-var knob is gone from the trainer's surface.
    src = open(ppo.__file__).read()
    assert "ALPHAGRAD_QUALITY_GATE_MIN" not in src.replace(
        "the deleted cost clamp ALPHAGRAD_QUALITY_GATE_MIN", "")
