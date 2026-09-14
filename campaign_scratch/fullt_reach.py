"""DOES --grad-window 0 ACTUALLY REACH STEP 0? Two separate claims.

CLAIM A (structural): the cotangent of a step-(T-1) output is nonzero at
step 0 under the full-horizon scan, and EXACTLY zero under the K=1 window.
The K path anchors each step at a carry read back from the trajectory, and a
stored array is a constant to AD -- so its step-0 derivative is not "small",
it is identically absent. That is the truncation, stated as a test.

CLAIM B (numerical, and the one that actually matters): the gradient that
arrives at step 0 is not so attenuated that "full T" is nominal. A recurrence
can be formally connected across T and still deliver 1e-30 there, in which
case the horizon is a fiction. So this prints the whole per-step profile
|d out_{T-1} / d part_k| for k = 0..T-1 rather than asserting only at k=0.

The perturbed quantity is `participants`, a float32 weight vector that is a
genuine continuous input to each step's `advance` -- unlike the delta TOKENS,
which are integers and carry no derivative at all.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "256")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_EXTEND_CHUNK", "8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphagrad.approx.common import carry_stream as CS  # noqa: E402

TOTAL_V = 6
EMBD = 32
DELTA_W = 32
T = 95


def _build():
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
    d_tok = jnp.asarray(rng.integers(1, 200, (T, DELTA_W)).astype(np.int32))
    d_eqn = jnp.asarray(rng.integers(0, 4, (T, DELTA_W)).astype(np.int32))
    d_cnt = jnp.asarray(rng.integers(4, DELTA_W, (T,)).astype(np.int32))
    owner = jnp.asarray(rng.integers(1, TOTAL_V, (T,)).astype(np.int32))
    part = jnp.asarray(
        rng.random((T, TOTAL_V + 1)).astype(np.float32))
    return agent, d_tok, d_eqn, d_cnt, owner, part


def _step(agent, carry, vs, vc, dt, de, dc, ow, pa):
    return CS.advance(
        agent, carry, vs, vc, dt, de, dc, ow,
        window=DELTA_W, participants=pa,
        chunk=None, budget=jnp.asarray(DELTA_W, jnp.int32),
    )


def _out(carry, vs, vc):
    """A scalar standing in for step T-1's heads -- it reads the carry."""
    return (jnp.sum(carry.M) + jnp.sum(carry.I) + jnp.sum(carry.cumhist)
            + jnp.sum(vs) + jnp.sum(vc))


def full_scan_out(agent, part, d_tok, d_eqn, d_cnt, owner):
    """--grad-window 0: ONE chained scan from the base carry."""
    def body(state, x):
        c, s, n = state
        dt, de, dc, ow, pa = x
        c, s, n = _step(agent, c, s, n, dt, de, dc, ow, pa)
        return (c, s, n), 0.0

    c0 = agent.carry_init()
    vs0, vc0 = CS.zero_memory(TOTAL_V, EMBD)
    (c, s, n), _ = jax.lax.scan(
        jax.checkpoint(body), (c0, vs0, vc0),
        (d_tok, d_eqn, d_cnt, owner, part))
    return _out(c, s, n)


def window_out(agent, part, d_tok, d_eqn, d_cnt, owner, anchor):
    """--grad-window 1: step T-1 advanced from a STORED (constant) carry.

    `anchor` stands for the trajectory's recorded encoder state. It enters as
    a plain value with no history, which is exactly what makes the K path's
    step-0 derivative vanish -- not attenuation, absence.
    """
    c, s, n = anchor
    c, s, n = _step(agent, c, s, n, d_tok[T - 1], d_eqn[T - 1], d_cnt[T - 1],
                    owner[T - 1], part[T - 1])
    return _out(c, s, n)


def main():
    agent, d_tok, d_eqn, d_cnt, owner, part = _build()

    g_full = jax.grad(full_scan_out, argnums=1)(
        agent, part, d_tok, d_eqn, d_cnt, owner)
    g_full = np.asarray(g_full)

    # Build the anchor the K path would have read back: the carry AFTER T-1
    # steps, detached exactly as storing it in the trajectory detaches it.
    def _prefix(agent, part):
        c = agent.carry_init()
        s, n = CS.zero_memory(TOTAL_V, EMBD)
        for k in range(T - 1):
            c, s, n = _step(agent, c, s, n, d_tok[k], d_eqn[k], d_cnt[k],
                            owner[k], part[k])
        return c, s, n

    anchor = jax.lax.stop_gradient(_prefix(agent, part))
    g_win = np.asarray(jax.grad(window_out, argnums=1)(
        agent, part, d_tok, d_eqn, d_cnt, owner, anchor))

    per_step_full = np.abs(g_full).sum(axis=1)
    per_step_win = np.abs(g_win).sum(axis=1)

    print("step |  full-T  sum|d out_{T-1}/d part_k| |   K=1")
    print("-----+--------------------------------------+---------")
    for k in range(T):
        print(f"{k:4d} | {per_step_full[k]:36.6e} | {per_step_win[k]:.3e}")

    print()
    print(f"full-T nonzero steps : {int((per_step_full > 0).sum())} / {T}")
    print(f"K=1    nonzero steps : {int((per_step_win > 0).sum())} / {T}")
    reach = per_step_full[0] / max(per_step_full[T - 1], 1e-300)
    print(f"attenuation step0/step{T-1} : {reach:.4e}")

    assert per_step_full[0] > 0.0, (
        "FULL-T IS A FICTION: step 0 has zero cotangent from step T-1.")
    assert per_step_win[0] == 0.0, (
        "the K=1 control is wrong -- it should have NO step-0 derivative.")
    assert int((per_step_full > 0).sum()) == T, (
        "full-T does not reach every step")
    print("\nPASS: full-T reaches all T steps; K=1 reaches exactly 1.")


if __name__ == "__main__":
    main()
