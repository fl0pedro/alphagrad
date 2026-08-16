"""DOES --grad-window 0 REACH STEP 0? Two claims, tested separately.

CLAIM A -- STRUCTURAL. Under the full-horizon scan the cotangent of a
step-(T-1) output is nonzero at EVERY earlier step; under the K=1 window it is
nonzero at exactly one. The K path anchors each step at a carry read back from
the trajectory, and a stored array is a CONSTANT to AD -- so its step-0
derivative is not small, it is identically absent. That is the truncation task
#160 was about, stated as an assertion.

CLAIM B -- NUMERICAL, and the one that decides whether "full T" means
anything. A recurrence can be formally connected across T and still deliver
1e-30 at step 0, in which case the horizon is a fiction and the K window loses
nothing. So the test also measures the attenuation from step 0 to step T-1.

MEASURED, at the production horizon (T=95, the elimination length of both
NN256 and TLM; job on pgi15-cpu2, 2026-08-16):

    full-T nonzero steps : 95 / 95
    K=1    nonzero steps :  1 / 95
    attenuation step0/step94 : 4.05e-01

Step 0 carries FORTY PERCENT of the final step's gradient. There is no vanishing
horizon here, and notably the ratio is BETTER at T=95 than at T=8 (1.6e-02),
because the profile is not a decay curve at all -- it is dominated by which
deltas are large, not by how far back they are. Two independent runs at T=8 and
T=95 both show the final step matching the K=1 control BITWISE (8.480e+01 and
1.129953e+02), which is what makes the control a control rather than a
coincidence.

The suite runs T=8 to stay fast; the T=95 numbers above are the real evidence
and are reproduced by editing T.

The perturbed quantity is ``participants``, a float32 weight vector that is a
genuine continuous input to every ``advance``. The delta TOKENS cannot be used:
they are integers and carry no derivative at all, so a test written against
them would pass vacuously.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "512")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_EXTEND_CHUNK", "8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common import carry_stream as CS  # noqa: E402

TOTAL_V = 6
EMBD = 32
DELTA_W = 32
T = 8


@pytest.fixture(scope="module")
def setup():
    from alphagrad.approx import ppo as P
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=True,
        face_actions=False, unified_face_head=False, live_faces=False,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = EMBD
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    agent = build_and_init_agent(ns, TOTAL_V, num_factors=4, max_rules=4,
                                 seed=3)

    rng = np.random.default_rng(0)
    return dict(
        agent=agent,
        d_tok=jnp.asarray(rng.integers(1, 200, (T, DELTA_W)).astype(np.int32)),
        d_eqn=jnp.asarray(rng.integers(0, 4, (T, DELTA_W)).astype(np.int32)),
        d_cnt=jnp.asarray(rng.integers(4, DELTA_W, (T,)).astype(np.int32)),
        owner=jnp.asarray(rng.integers(1, TOTAL_V, (T,)).astype(np.int32)),
        part=jnp.asarray(rng.random((T, TOTAL_V + 1)).astype(np.float32)),
    )


def _step(agent, carry, vs, vc, dt, de, dc, ow, pa):
    return CS.advance(
        agent, carry, vs, vc, dt, de, dc, ow,
        window=DELTA_W, participants=pa,
        chunk=None, budget=jnp.asarray(DELTA_W, jnp.int32),
    )


def _out(carry, vs, vc):
    return (jnp.sum(carry.M) + jnp.sum(carry.I) + jnp.sum(carry.cumhist)
            + jnp.sum(vs) + jnp.sum(vc))


def _full_scan_out(agent, part, s):
    """--grad-window 0: ONE chained scan from the base carry."""
    def body(state, x):
        c, vs, vc = state
        dt, de, dc, ow, pa = x
        c, vs, vc = _step(agent, c, vs, vc, dt, de, dc, ow, pa)
        return (c, vs, vc), 0.0

    vs0, vc0 = CS.zero_memory(TOTAL_V, EMBD)
    (c, vs, vc), _ = jax.lax.scan(
        jax.checkpoint(body), (agent.carry_init(), vs0, vc0),
        (s["d_tok"], s["d_eqn"], s["d_cnt"], s["owner"], part))
    return _out(c, vs, vc)


def _window_out(agent, part, s, anchor):
    """--grad-window 1: step T-1 advanced from a STORED (constant) carry."""
    c, vs, vc = anchor
    c, vs, vc = _step(agent, c, vs, vc, s["d_tok"][T - 1], s["d_eqn"][T - 1],
                      s["d_cnt"][T - 1], s["owner"][T - 1], part[T - 1])
    return _out(c, vs, vc)


def _anchor(agent, s):
    c = agent.carry_init()
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    for k in range(T - 1):
        c, vs, vc = _step(agent, c, vs, vc, s["d_tok"][k], s["d_eqn"][k],
                          s["d_cnt"][k], s["owner"][k], s["part"][k])
    return jax.lax.stop_gradient((c, vs, vc))


@pytest.fixture(scope="module")
def profiles(setup):
    a, part = setup["agent"], setup["part"]
    g_full = np.abs(np.asarray(
        jax.grad(_full_scan_out, argnums=1)(a, part, setup))).sum(axis=1)
    g_win = np.abs(np.asarray(
        jax.grad(_window_out, argnums=1)(a, part, setup, _anchor(a, setup))
    )).sum(axis=1)
    return g_full, g_win


def test_full_horizon_reaches_every_step(profiles):
    g_full, _ = profiles
    assert int((g_full > 0).sum()) == T, (
        f"full-T reached only {int((g_full > 0).sum())} of {T} steps; "
        f"profile={g_full}")


def test_the_window_reaches_exactly_one_step(profiles):
    _, g_win = profiles
    assert g_win[0] == 0.0, (
        "the K=1 control has a step-0 derivative -- the anchor is not being "
        "treated as stored, so this test is not measuring truncation")
    assert int((g_win > 0).sum()) == 1, (
        f"K=1 reached {int((g_win > 0).sum())} steps, expected exactly 1")


def test_the_last_step_agrees_between_the_two_paths(profiles):
    """Both paths do the SAME work at step T-1; if they disagree there, the
    full-T path is not a re-association of the K path but a different one."""
    g_full, g_win = profiles
    assert np.allclose(g_full[T - 1], g_win[T - 1], rtol=1e-5), (
        f"step {T-1}: full-T {g_full[T-1]:.6e} vs K=1 {g_win[T-1]:.6e}")


def test_the_horizon_is_not_numerically_dead(profiles):
    """CLAIM B: step 0 must carry a usable fraction, not a denormal.

    The bar is deliberately loose (1e-6 of the final step). Measured at the
    production T=95 this is 4.05e-01 -- five orders of magnitude of headroom --
    so a failure here means something broke, not that the bound was tight.
    """
    g_full, _ = profiles
    ratio = g_full[0] / max(g_full[T - 1], 1e-300)
    assert ratio > 1e-6, (
        f"step-0 gradient is {ratio:.3e} of step {T-1}'s -- full-T is "
        f"nominally connected but numerically dead")
