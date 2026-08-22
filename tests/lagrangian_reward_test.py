"""--reward-mode lagrangian: THE VALUE NET MUST BE SAFEGUARDED. This pins it.

Contracts of the RCPO-style quality-constrained reward
(docs/QUALITY_COLLAPSE_INVESTIGATION.md section 10, option 1):

1. Every channel's value target is a STATIONARY function of the measured
   outcome -- bitwise invariant to lambda. lambda enters the update at
   exactly ONE site: the quality slot of the advantage-scalarization
   preference (``norm_adv = sum(norm_adv_components * traj.preference)``).
2. Violation channel: ``v = max(0, tau - clip(q, -0.5, 1))``, stored
   NEGATED (higher is better) on the TERMINAL step only. The env's
   DIVERGED sentinel (-1.0) maps to q_eff = -0.5, so diverged carries
   tau + 0.5 violation -- strictly worse than any zero-work plan (tau) --
   killing the mult path's [0, 1]-clip conflation.
3. Dual ascent ``lam <- clip(lam + eta*mean_violation, lam_min, lam_max)``,
   once per episode, host-side, after the PPO update.
4. Basin freeze: when >50% of the batch sits in the basin (terminal
   q_eff <= 0.05 -- the dossier's section-2 occupancy measure, covering
   both the historical q = 0.0 zero-work mode and diverged plans), the
   quality channel's PopArt accumulators (m1, m2, w -- all per-channel)
   hold for the episode, so the debiased (mu, sigma) are bitwise unchanged
   and the ART rescale is a no-op.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from alphagrad.approx.ppo import (                              # noqa: E402
    HEAD_NAMES,
    _face_entropy_floor_penalty,
    _split_entropy_bonus,
    HEAD_REWARD_INDICES,
    NUM_VALUE_HEADS,
    _apply_lagrangian_channels,
    _lag_basin_freeze,
    _lag_dual_ascent,
    _lag_violation,
    _popart_derive,
    _popart_update,
    _symlog_rewards,
)
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

TAU = 0.75
QIDX = REWARD_INDEX["cosine_sim"]
QHEAD = HEAD_NAMES.index("quality")


def _mk_rewards(q_terminal, E=None, T=6, seed=0):
    """Raw (E, T, NUM_REWARDS) env-style reward tensor with sparse-terminal
    quality and negated costs on every step."""
    q_terminal = np.asarray(q_terminal, dtype=np.float32)
    E = E or q_terminal.shape[0]
    rng = np.random.default_rng(seed)
    r = np.zeros((E, T, NUM_REWARDS), dtype=np.float32)
    # negated costs, latency ~1e5, mem ~5e7 (the TLM scales)
    r[..., REWARD_INDEX["latency_ns"]] = -rng.uniform(1e5, 2e5, (E, T))
    r[..., REWARD_INDEX["peak_memory"]] = -rng.uniform(4e7, 6e7, (E, T))
    r[:, -1, QIDX] = q_terminal
    return jnp.asarray(r)


# ---------------------------------------------------------------------------
# 2. violation channel values
# ---------------------------------------------------------------------------

def test_violation_values():
    q = jnp.asarray([0.9, 0.4, -1.0, 0.0, TAU])
    v = np.asarray(_lag_violation(q, TAU))
    assert v[0] == 0.0                         # clean, constraint satisfied
    np.testing.assert_allclose(v[1], TAU - 0.4, rtol=1e-6)   # degraded: 0.35
    np.testing.assert_allclose(v[2], TAU + 0.5, rtol=1e-6)   # DIVERGED: 1.25
    np.testing.assert_allclose(v[3], TAU, rtol=1e-6)         # zero-work: 0.75
    assert v[4] == 0.0                         # exactly at tau: no violation
    # The conflation killer: diverged is STRICTLY worse than zero-work.
    assert v[2] > v[3]


def test_violation_monotone_no_flat_basin():
    # Strict slope -1 everywhere on (-0.5, tau): the dossier's 0.45-wide
    # flat plateau and the +1.9 discontinuity must not exist here.
    q = jnp.linspace(-0.49, TAU, 1000)
    v = np.asarray(_lag_violation(q, TAU))
    dv = np.diff(v) / np.diff(np.asarray(q))
    np.testing.assert_allclose(dv, -1.0, atol=1e-3)


def test_channel_placement_and_cost_passthrough():
    q_term = [0.9, 0.4, -1.0]
    r = _mk_rewards(q_term)
    out = _apply_lagrangian_channels(r, TAU)
    # ADDITIVE composition: every non-quality slot is bitwise untouched.
    mask = np.ones(NUM_REWARDS, dtype=bool)
    mask[QIDX] = False
    np.testing.assert_array_equal(np.asarray(out)[..., mask],
                                  np.asarray(r)[..., mask])
    # Quality slot: 0 on non-terminal steps, -violation at the terminal.
    np.testing.assert_array_equal(np.asarray(out)[:, :-1, QIDX], 0.0)
    want = -np.asarray(_lag_violation(jnp.asarray(q_term), TAU))
    np.testing.assert_allclose(np.asarray(out)[:, -1, QIDX], want, rtol=1e-6)


# ---------------------------------------------------------------------------
# 1. lambda-invariance of value targets
# ---------------------------------------------------------------------------

def _value_target_chain(r, lam):
    """The exact loss-path chain up to the value targets, with lambda handed
    in. lambda MUST be dead code here -- _apply_lagrangian_channels does not
    even accept it -- and the test proves the chain output ignores it while
    the one legitimate consumer (preference scalarization) does not."""
    del lam  # the point: no lambda-shaped hole exists anywhere in this chain
    out = _apply_lagrangian_channels(r, TAU)
    sl = _symlog_rewards(out)
    hr = np.asarray(sl)[..., list(HEAD_REWARD_INDICES)]         # (E, T, K)
    # Monte-Carlo return, gamma=0.99 -- the lambda=1 GAE limit the critic
    # regresses (same recursion the PopArt warm start uses).
    g = np.zeros_like(hr)
    run = np.zeros((hr.shape[0], hr.shape[2]))
    for t in range(hr.shape[1] - 1, -1, -1):
        run = hr[:, t, :] + 0.99 * run
        g[:, t, :] = run
    return g


def test_value_targets_bitwise_invariant_to_lambda():
    r = _mk_rewards([0.9, 0.4, -1.0, 0.0], seed=3)
    g_lo = _value_target_chain(r, lam=0.1)
    g_hi = _value_target_chain(r, lam=10.0)
    np.testing.assert_array_equal(g_lo, g_hi)   # bitwise
    # ... and the PopArt stats fed from those targets are equally invariant.
    m0 = jnp.zeros((NUM_VALUE_HEADS,), dtype=jnp.float32)
    up_lo = _popart_update(m0, m0, m0, jnp.asarray(g_lo, jnp.float32),
                           0.1, 0.1, 1e12, 5.0)
    up_hi = _popart_update(m0, m0, m0, jnp.asarray(g_hi, jnp.float32),
                           0.1, 0.1, 1e12, 5.0)
    for a, b in zip(up_lo, up_hi):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The ONE lambda consumer: preference scalarization of the (normalized)
    # advantages. Same advantages, different lambda => different scalar.
    adv = np.asarray(g_lo)  # stand-in for norm_adv_components: shape (E,T,K)
    pref_lo = np.array([1.0, 1.0, 0.1], dtype=np.float32)
    pref_hi = np.array([1.0, 1.0, 10.0], dtype=np.float32)
    s_lo = (adv * pref_lo).sum(-1)
    s_hi = (adv * pref_hi).sum(-1)
    assert not np.allclose(s_lo, s_hi)


# ---------------------------------------------------------------------------
# 3. dual ascent
# ---------------------------------------------------------------------------

def test_dual_ascent_clip_behavior():
    # zero violation, violation_target=0: lambda holds exactly.
    assert _lag_dual_ascent(1.0, 0.0, 0.05, 0.1, 10.0) == 1.0
    # violation raises lambda by eta * v.
    np.testing.assert_allclose(
        _lag_dual_ascent(1.0, 0.5, 0.05, 0.1, 10.0), 1.025, rtol=1e-9)
    # cap at lam_max.
    assert _lag_dual_ascent(9.99, 1.25, 1.0, 0.1, 10.0) == 10.0
    # floor at lam_min (never zero: quality must keep advantage weight).
    assert _lag_dual_ascent(0.05, 0.0, 0.05, 0.1, 10.0) == 0.1
    assert _lag_dual_ascent(0.1, -5.0, 1.0, 0.1, 10.0, violation_target=0.0) == 0.1


def test_dual_ascent_target_binding():
    """v62 --lag-target: lambda RISES above target, FALLS below it, holds
    exactly at it, and clips at both bounds (the start-high-decay
    schedule: --lag-init 10 --lag-min 2 --lag-max 20 --lag-target 0.02)."""
    # rises when mean_violation > target
    np.testing.assert_allclose(
        _lag_dual_ascent(10.0, 0.50, 0.05, 2.0, 20.0,
                         violation_target=0.02),
        10.0 + 0.05 * (0.50 - 0.02), rtol=1e-9)
    # FALLS when the constraint is essentially satisfied (v < target)
    np.testing.assert_allclose(
        _lag_dual_ascent(10.0, 0.0, 0.05, 2.0, 20.0,
                         violation_target=0.02),
        10.0 - 0.05 * 0.02, rtol=1e-9)
    # holds exactly at the target
    assert _lag_dual_ascent(5.0, 0.02, 0.05, 2.0, 20.0,
                            violation_target=0.02) == 5.0
    # decay clips at lam_min (quality stays binding after full decay)
    assert _lag_dual_ascent(2.0, 0.0, 1000.0, 2.0, 20.0,
                            violation_target=0.02) == 2.0
    # ascent clips at lam_max
    assert _lag_dual_ascent(20.0, 5.0, 1000.0, 2.0, 20.0,
                            violation_target=0.02) == 20.0


# ---------------------------------------------------------------------------
# 4. basin freeze
# ---------------------------------------------------------------------------

def _stats(seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.uniform(0.2, 2.0, NUM_VALUE_HEADS),
                             jnp.float32) for _ in range(3))


def test_basin_freeze_triggers_on_majority_basin_occupancy():
    old = _stats(1)
    new = _stats(2)
    # THE HISTORICAL FAILURE MODE (v58-v60): 9 of 16 envs at q = 0.0
    # exactly (zero-work plans) -> occupancy 0.5625 > 0.5 MUST freeze.
    # The first-cut predicate (violation > 0.9*(tau+0.5)) required
    # q < -0.375 and could never fire on this batch.
    q = jnp.asarray([0.0] * 9 + [0.885] * 7)
    m1, m2, w, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert bool(frozen)
    for got, o, n in zip((m1, m2, w), old, new):
        got = np.asarray(got)
        np.testing.assert_array_equal(got[QHEAD], np.asarray(o)[QHEAD])
        keep = np.arange(NUM_VALUE_HEADS) != QHEAD
        np.testing.assert_array_equal(got[keep], np.asarray(n)[keep])
    # (mu, sigma) of the frozen channel are bitwise those of the OLD stats.
    mu_f, sg_f = _popart_derive(m1, m2, w, 0.1, 1e12)
    mu_o, sg_o = _popart_derive(*old, 0.1, 1e12)
    np.testing.assert_array_equal(np.asarray(mu_f)[QHEAD],
                                  np.asarray(mu_o)[QHEAD])
    np.testing.assert_array_equal(np.asarray(sg_f)[QHEAD],
                                  np.asarray(sg_o)[QHEAD])


def test_basin_freeze_diverged_batch_also_freezes():
    # Diverged plans (q = -1 -> q_eff = -0.5 <= 0.05) count as basin
    # occupants too: 9 of 16 diverged freezes.
    old = _stats(1)
    new = _stats(2)
    q = jnp.asarray([-1.0] * 9 + [0.885] * 7)
    *_, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert bool(frozen)


def test_basin_freeze_does_not_trigger_at_half_or_below():
    old = _stats(3)
    new = _stats(4)
    # A HEALTHY batch (all plans at the identity quality) must never freeze.
    q = jnp.asarray([0.885] * 16)
    m1, m2, w, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)
    for got, n in zip((m1, m2, w), new):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(n))
    # 40% occupancy (8 of 20 at q = 0) -> below the majority bar, no freeze.
    q = jnp.asarray([0.0] * 8 + [0.885] * 12)
    *_, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)
    # Exactly half (8 of 16): frac 0.5 is NOT > 0.5 -> no freeze.
    q = jnp.asarray([0.0] * 8 + [0.885] * 8)
    *_, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)
    # Marginal-but-alive plans (q = 0.06 > 0.05) are NOT basin occupants.
    q = jnp.asarray([0.06] * 16)
    *_, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# --face-entropy-weight: face/vertex entropy-bonus split (v63)
# ---------------------------------------------------------------------------

def test_split_entropy_bonus_default_none_bit_identical():
    """None (the default) must be BITWISE the pre-change expression
    ``entropy_weight * entropy_loss`` -- every prior config unchanged."""
    rng = np.random.default_rng(3)
    ents = jnp.asarray(rng.uniform(0.0, 2.0, 64), jnp.float32)
    face = jnp.asarray(rng.uniform(0.0, 1.0, 64), jnp.float32)
    el, fm = jnp.mean(ents), jnp.mean(face)
    got = _split_entropy_bonus(el, fm, 0.05, None)
    ref = 0.05 * el
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    # and equal to the flag explicitly set to the global weight
    np.testing.assert_allclose(
        float(_split_entropy_bonus(el, fm, 0.05, 0.05)), float(ref),
        rtol=1e-6)


def test_split_entropy_bonus_face_scales_vertex_keeps_global():
    """Face term scales with the flag; vertex term keeps the global
    weight. The joint entropy folds the face entropy in per sample, so
    the effective gradient on the face entropy is EXACTLY the flag."""
    v, h, g = 0.9, 0.4, 0.05  # vertex part, face mean, global weight
    el = v + h
    for few in (0.0, 0.005, 0.05, 0.5):
        np.testing.assert_allclose(
            float(_split_entropy_bonus(el, h, g, few)),
            g * v + few * h, rtol=1e-6)
    # linear in the flag through the face mean only
    b1 = float(_split_entropy_bonus(el, h, g, 0.005))
    b2 = float(_split_entropy_bonus(el, h, g, 0.010))
    np.testing.assert_allclose(b2 - b1, 0.005 * h, rtol=1e-5)

    # gradient wrt the FACE entropy is the flag; wrt the VERTEX entropy
    # it stays the global weight, whatever the flag says.
    def bonus(hh, vv, few):
        return _split_entropy_bonus(vv + hh, hh, g, few)

    for few in (0.0, 0.005, 0.5):
        np.testing.assert_allclose(
            float(jax.grad(bonus, argnums=0)(h, v, few)), few, atol=1e-7)
        np.testing.assert_allclose(
            float(jax.grad(bonus, argnums=1)(h, v, few)), g, atol=1e-7)
    # None falls back to the global weight on both components
    np.testing.assert_allclose(
        float(jax.grad(lambda hh: bonus(hh, v, None))(h)), g, atol=1e-7)


def test_floor_hinge_unaffected_by_face_entropy_weight():
    """--face-entropy-floor keeps its value and its restoring gradient
    no matter what --face-entropy-weight says (including 0)."""
    H, FLOORV, W, g = 0.1, 0.3, 10.0, 0.05
    ref = float(_face_entropy_floor_penalty(H, FLOORV, W))
    hinge_grad = -2.0 * W * (FLOORV - H)  # d hinge / dH below the floor
    for few in (None, 0.0, 0.005, 1.0):
        def loss(hh):
            b = _split_entropy_bonus(0.9 + hh, hh, g, few)
            return -b + _face_entropy_floor_penalty(hh, FLOORV, W)
        b = float(_split_entropy_bonus(0.9 + H, H, g, few))
        # hinge contribution identical for every flag value
        np.testing.assert_allclose(float(loss(H)) + b, ref, rtol=1e-6)
        few_eff = g if few is None else few
        grad = float(jax.grad(loss)(H))
        np.testing.assert_allclose(grad + few_eff, hinge_grad, rtol=1e-5)
        # below the floor the loss still pushes H UP even at flag 0
        assert grad < 0.0, (few, grad)
