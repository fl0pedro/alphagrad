"""Pins for the owner batch: mult-gate PPO/AZ parity + face-head identity
init (`ALPHAGRAD_FACE_NONE_BIAS`)."""
import numpy as np
import jax.numpy as jnp
import pytest

from alphagrad.approx.common.sampled_az import mult_gate_scalar


def _ppo_gate_on_raw4(lat, peak, flops, cos, tau, W, P_pen, tau_d):
    """Run ppo's in-jit _apply_mult_gate on az's [lat,peak,flops,cos] by
    building the env-layout (E=1,T=1,R) reward row: costs NEGATED, cosine in
    its REWARD_INDEX slot, cost weights 1/1 on latency/peak."""
    from alphagrad.approx.ppo import _apply_mult_gate
    from alphagrad.approx.env import REWARD_INDEX, NUM_REWARDS
    r = np.zeros((1, 1, NUM_REWARDS), dtype=np.float32)
    r[0, 0, REWARD_INDEX["latency_ns"]] = -lat
    r[0, 0, REWARD_INDEX["peak_memory"]] = -peak
    r[0, 0, REWARD_INDEX["cosine_sim"]] = cos
    w = np.zeros((NUM_REWARDS,), dtype=np.float32)
    w[REWARD_INDEX["latency_ns"]] = 1.0
    w[REWARD_INDEX["peak_memory"]] = 1.0
    out = _apply_mult_gate(jnp.asarray(r), jnp.asarray(w), tau, W,
                           P_pen, tau_d)
    return float(out[0, 0, REWARD_INDEX["cosine_sim"]])


@pytest.mark.parametrize("lat,peak,cos", [
    (132e3, 4.0e9, 0.9),     # honest good plan above the gate
    (132e3, 4.0e9, 0.5),     # exactly at tau -> g = 0
    (37e3, 1e6, 0.0),        # destroyed: anti-degen penalty branch
    (37e3, 1e6, 0.03),       # destroyed but nonzero cos: sloped penalty
    (5e6, 9e9, 0.99),        # slow but faithful
])
def test_mult_gate_ppo_az_parity(lat, peak, cos):
    tau, W, P_pen, tau_d = 0.5, 40.0, 2.0, 0.05
    az = mult_gate_scalar([lat, peak, 0.0, cos], tau, W, P_pen, tau_d)
    ppo = _ppo_gate_on_raw4(lat, peak, 0.0, cos, tau, W, P_pen, tau_d)
    assert az == pytest.approx(ppo, rel=1e-5, abs=1e-6), (az, ppo)


def test_mult_gate_destroyed_below_every_valid_reward():
    tau, W, P_pen, tau_d = 0.5, 40.0, 2.0, 0.05
    destroyed = mult_gate_scalar([37e3, 1e6, 0.0, 0.0], tau, W, P_pen, tau_d)
    assert destroyed < 0.0
    honest = mult_gate_scalar([132e3, 4.0e9, 0.0, 0.9], tau, W, P_pen, tau_d)
    assert honest >= 0.0 and honest > destroyed


def test_face_none_bias_identity_init(monkeypatch):
    """factory applies +B to each slot's OP_NONE logit and -B to SKIP; the
    resulting per-face approx probability is small; default (unset) changes
    nothing."""
    monkeypatch.setenv("ALPHAGRAD_FACE_NONE_BIAS", "6")
    import importlib
    import test_ppo_az_parity as par
    agent = par._build({"face_actions": True, "unified_face_head": True,
                        "live_faces": True})
    monkeypatch.delenv("ALPHAGRAD_FACE_NONE_BIAS")
    agent0 = par._build({"face_actions": True, "unified_face_head": True,
                         "live_faces": True})
    from alphagrad.approx.unified_face_head import (
        FACE_SLOTS, OP_NONE, O_SKIP, S_OP, slot_base)
    b = np.asarray(agent.face_path_policy.head.proj.layers[-1].bias)
    b0 = np.asarray(agent0.face_path_policy.head.proj.layers[-1].bias)
    for s in range(FACE_SLOTS):
        idx = slot_base(s) + S_OP + OP_NONE
        assert b[idx] - b0[idx] == pytest.approx(6.0)
    assert b[O_SKIP] - b0[O_SKIP] == pytest.approx(-6.0)
    # everything else untouched
    touched = {O_SKIP} | {slot_base(s) + S_OP + OP_NONE
                          for s in range(FACE_SLOTS)}
    for i in range(len(b)):
        if i not in touched:
            assert b[i] == pytest.approx(b0[i])
    # softmax over one slot's 4 op logits: NONE dominates at init
    s0 = slot_base(0) + S_OP
    ops = b[s0:s0 + 4]
    p = np.exp(ops - ops.max()); p /= p.sum()
    assert p[OP_NONE] > 0.95
