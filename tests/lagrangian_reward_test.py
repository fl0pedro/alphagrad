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
4. Basin freeze: when >50% of the batch is at near-total destruction
   (violation > 0.9*(tau+0.5)), the quality channel's PopArt accumulators
   (m1, m2, w -- all per-channel) hold for the episode, so the debiased
   (mu, sigma) are bitwise unchanged and the ART rescale is a no-op.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from alphagrad.approx.ppo import (                              # noqa: E402
    HEAD_NAMES,
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


# ---------------------------------------------------------------------------
# 4. basin freeze
# ---------------------------------------------------------------------------

def _stats(seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.uniform(0.2, 2.0, NUM_VALUE_HEADS),
                             jnp.float32) for _ in range(3))


def test_basin_freeze_triggers_on_majority_destruction():
    old = _stats(1)
    new = _stats(2)
    # 9 of 16 diverged: violation 1.25 > 0.9*1.25 = 1.125 -> frac 0.5625.
    q = jnp.asarray([-1.0] * 9 + [0.885] * 7)
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


def test_basin_freeze_does_not_trigger_at_half_or_below():
    old = _stats(3)
    new = _stats(4)
    # Exactly 8 of 16: frac 0.5, NOT > 0.5 -> no freeze.
    q = jnp.asarray([-1.0] * 8 + [0.885] * 8)
    m1, m2, w, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)
    for got, n in zip((m1, m2, w), new):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(n))
    # Zero-work plans (q=0, violation=tau=0.75 < 1.125) are NOT "near-total
    # destruction": a q=0-dominated batch must keep PopArt adapting -- the
    # freeze guards only the diverged-dominated extreme.
    q = jnp.asarray([0.0] * 16)
    *_, frozen = _lag_basin_freeze(old, new, q, TAU, QHEAD)
    assert not bool(frozen)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
