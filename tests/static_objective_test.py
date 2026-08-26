"""sec 12.10 static-objective mode (docs/QUALITY_COLLAPSE_INVESTIGATION.md).

Pins the three mechanisms of the fully STATIC lagrangian objective
(--advantage-norm none, symlog on the cost channels, RAW bounded violation
channel, frozen lambda via --lag-eta 0):

1. The per-channel symlog exemption (_set_no_symlog_indices) hits EXACTLY
   the violation slot (REWARD_INDEX["cosine_sim"], the lagrangian-rewritten
   quality channel): that slot passes through _symlog_rewards bitwise raw,
   the latency/memory cost slots are symlog'd, and resetting to () restores
   the all-symlog default bitwise (flag-off regression at the unit level).
2. --lag-eta 0 makes _lag_dual_ascent a strict no-op: lambda is returned
   BITWISE unchanged for any violation level, any target, any bound config
   that contains it -- the [lagrangian] stdout/wandb telemetry keeps
   running off the same frozen float.
3. _per_channel_value_loss: total is BITWISE the pre-existing
   mean(sum(sq, -1)) critic loss; the per-channel vector behind the
   value_loss/<channel> wandb keys is the per-head MSE and sums back to the
   total exactly (mean-sum == sum-mean).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.ppo import (                              # noqa: E402
    NUM_VALUE_HEADS,
    _apply_lagrangian_channels,
    _lag_dual_ascent,
    _per_channel_value_loss,
    _set_no_symlog_indices,
    _symlog_rewards,
)

REWARD_INDEX = ppo.REWARD_INDEX
NUM_REWARDS = int(ppo.NUM_REWARDS)
QSLOT = int(REWARD_INDEX["cosine_sim"])
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])


def _mk_rewards(seed=0, E=3, T=5):
    """A full (E, T, NUM_REWARDS) reward tensor with cost-scale channels and
    a bounded quality slot, all strictly nonzero so symlog != identity."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.0, 3.0, (E, T, NUM_REWARDS)).astype(np.float32)
    r[..., LAT] = -rng.uniform(1e4, 3e5, (E, T))       # negated cost
    r[..., MEM] = -rng.uniform(1e6, 9e7, (E, T))       # negated cost
    r[..., QSLOT] = rng.uniform(-1.25, -0.01, (E, T))  # -violation range
    return jnp.asarray(r)


def _reset():
    """Restore the module-import symlog state."""
    _set_no_symlog_indices(())
    ppo._NO_SYMLOG_ALL[0] = False


# ---------------------------------------------------------------------------
# 1. per-channel symlog exemption
# ---------------------------------------------------------------------------

def test_exemption_hits_only_violation_slot():
    r = _mk_rewards()
    try:
        _reset()
        full = np.asarray(_symlog_rewards(r))          # all-symlog baseline
        _set_no_symlog_indices((QSLOT,))
        out = np.asarray(_symlog_rewards(r))
        # violation slot: BITWISE raw.
        np.testing.assert_array_equal(out[..., QSLOT], np.asarray(r)[..., QSLOT])
        # and raw != symlog there (the exemption is doing real work).
        assert np.all(out[..., QSLOT] != full[..., QSLOT])
        # every other slot -- latency and memory in particular -- is
        # BITWISE the symlog'd baseline.
        other = [j for j in range(NUM_REWARDS) if j != QSLOT]
        np.testing.assert_array_equal(out[..., other], full[..., other])
        assert LAT in other and MEM in other
        # and symlog actually compressed the cost channels.
        assert np.abs(out[..., LAT]).max() < 30.0
        assert np.abs(out[..., MEM]).max() < 30.0
    finally:
        _reset()


def test_reset_restores_all_symlog_bitwise():
    r = _mk_rewards(seed=1)
    try:
        _reset()
        before = np.asarray(_symlog_rewards(r))
        _set_no_symlog_indices((QSLOT,))
        _reset()
        after = np.asarray(_symlog_rewards(r))
        np.testing.assert_array_equal(before, after)   # flag-off regression
    finally:
        _reset()


def test_no_symlog_all_overrides_mask():
    """--no-symlog (the v65 control arm) is a full identity regardless of
    the mask state -- the popart arms are untouched by the exemption."""
    r = _mk_rewards(seed=2)
    try:
        _set_no_symlog_indices((QSLOT,))
        ppo._NO_SYMLOG_ALL[0] = True
        np.testing.assert_array_equal(
            np.asarray(_symlog_rewards(r)), np.asarray(r))
    finally:
        _reset()


def test_lagrangian_channel_end_to_end_raw():
    """_apply_lagrangian_channels -> _symlog_rewards with the exemption:
    the terminal quality slot carries EXACTLY -violation (raw, bounded by
    tau + 0.5), non-terminal steps exactly 0, and the cost slots are
    symlog'd -- the full static-arm reward pipeline."""
    tau = 0.75
    r = _mk_rewards(seed=3)
    r = r.at[..., QSLOT].set(
        jnp.asarray(np.random.default_rng(4).uniform(-1.0, 1.0, r.shape[:2]),
                    jnp.float32))                      # raw quality readings
    try:
        _reset()
        lag = _apply_lagrangian_channels(r, tau)
        _set_no_symlog_indices((QSLOT,))
        out = np.asarray(_symlog_rewards(lag))
        q_raw = np.asarray(r)[..., QSLOT]
        viol = np.maximum(0.0, tau - np.clip(q_raw, -0.5, 1.0))
        np.testing.assert_array_equal(out[:, -1, QSLOT], -viol[:, -1])
        np.testing.assert_array_equal(out[:, :-1, QSLOT], 0.0)
        assert out[..., QSLOT].min() >= -(tau + 0.5)
        # cost channels went through symlog exactly as without the
        # lagrangian rewrite (they are untouched by it).
        _reset()
        base = np.asarray(_symlog_rewards(r))
        np.testing.assert_array_equal(out[..., LAT], base[..., LAT])
        np.testing.assert_array_equal(out[..., MEM], base[..., MEM])
    finally:
        _reset()


# ---------------------------------------------------------------------------
# 2. --lag-eta 0 freezes lambda bitwise
# ---------------------------------------------------------------------------

def test_eta_zero_is_strict_noop():
    for lam in (10.0, 13.0, 16.0, 2.0, 20.0):
        for viol in (0.0, 0.02, 0.31, 0.75, 1.25):
            for tgt in (0.0, 0.02):
                out = _lag_dual_ascent(lam, viol, 0.0, 2.0, 20.0,
                                       violation_target=tgt)
                assert out == lam                      # BITWISE (float ==)
                assert isinstance(out, float)


def test_eta_positive_moves_lambda_control():
    # control: the same call with the campaign eta moves lambda, so the
    # freeze above is eta doing the freezing, not a broken updater.
    assert _lag_dual_ascent(10.0, 0.75, 0.05, 2.0, 20.0) > 10.0
    assert _lag_dual_ascent(10.0, 0.0, 0.05, 2.0, 20.0,
                            violation_target=0.02) < 10.0


# ---------------------------------------------------------------------------
# 3. per-channel value loss
# ---------------------------------------------------------------------------

def test_per_channel_value_loss_total_bitwise_and_decomposition():
    rng = np.random.default_rng(5)
    B = 64
    values = jnp.asarray(
        rng.normal(size=(B, NUM_VALUE_HEADS)).astype(np.float32))
    targets = jnp.asarray(
        (rng.normal(size=(B, NUM_VALUE_HEADS)) * 3.0).astype(np.float32))
    total, per_ch = _per_channel_value_loss(values, targets)
    # total is BITWISE the pre-existing summed loss formula.
    want = jnp.mean(jnp.sum(
        (values - ppo._value_target(targets)) ** 2, axis=-1))
    np.testing.assert_array_equal(np.asarray(total), np.asarray(want))
    assert np.asarray(per_ch).shape == (NUM_VALUE_HEADS,)
    # decomposition: sum of the per-channel means == the total.
    np.testing.assert_allclose(
        float(jnp.sum(per_ch)), float(total), rtol=1e-6)
    # per-channel correctness against numpy.
    sq = (np.asarray(values)
          - np.asarray(ppo._value_target(targets))) ** 2
    np.testing.assert_allclose(
        np.asarray(per_ch), sq.mean(axis=0), rtol=1e-6)


def test_per_channel_value_loss_respects_no_symlog():
    """_value_target is identity under --no-symlog; the helper must follow
    the same switch the summed loss uses (3-sites-agree discipline)."""
    rng = np.random.default_rng(6)
    values = jnp.asarray(
        rng.normal(size=(8, NUM_VALUE_HEADS)).astype(np.float32))
    targets = jnp.asarray(
        rng.normal(size=(8, NUM_VALUE_HEADS)).astype(np.float32) * 5.0)
    try:
        ppo._NO_SYMLOG_ALL[0] = True
        total, per_ch = _per_channel_value_loss(values, targets)
        sq = (np.asarray(values) - np.asarray(targets)) ** 2
        np.testing.assert_allclose(
            np.asarray(per_ch), sq.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            float(total), sq.sum(axis=-1).mean(), rtol=1e-6)
    finally:
        _reset()
