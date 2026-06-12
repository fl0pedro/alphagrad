"""Round-trip parity tests for `CpuApproximationServer` vs `env._callback`.

The `step_external_jax_part` + `CpuApproximationServer.evaluate` +
`assemble_step_result` pipeline is supposed to produce a bit-exact
:class:`EnvState` to what `env.step()`'s `io_callback` path returns for
the same inputs. These tests pin that contract — if either side ever
drifts (e.g. someone changes `_callback` and forgets to update the
server's pre-/post-processing), the test catches it before the change
lands in a production rollout.

Scoped to small jaxprs (`Helmholtz`, `(x @ W).sum()`) so they finish in a
few seconds without GPU / Ray / the production reward harness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    EnvConfig,
    StepAction,
    VertexEliminationEnv,
)


def _tiny_env():
    """Build a minimal VertexEliminationEnv around `(x @ W).sum()`.

    Mirrors `tests/test_env_callback.py::_tiny_jaxpr` but constructs
    a full `VertexEliminationEnv` so we can exercise the public
    `reset` / `step` / `step_external_jax_part` surface.
    """

    def fn(x, W):
        return jnp.sum(x @ W)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    W = jnp.ones((4, 4), dtype=jnp.float32)
    closed = jax.make_jaxpr(fn)(x, W)
    return VertexEliminationEnv.from_jaxpr(
        closed,
        args=(x, W),
        argnums=(0, 1),
        num_envs=0,
        cmp_type="graphax",
        mem_type="graphax",
        # target_fun=None → `_callback` skips the compile/exec branch
        # and reports the graphax-only reward subset. Keeps the test
        # fast and removes the cost_analysis non-determinism.
        target_fun=None,
        terminal_rewards_only=False,
    )


# ---------------------------------------------------------------------------
# Reset: external + assemble == reset()
# ---------------------------------------------------------------------------


def test_reset_external_matches_reset():
    """`reset_external_jax_part + server.evaluate(init=True) +
    assemble_reset_result` produces an identical EnvState to the
    `io_callback` path in `reset()` (for num_envs=None — a single env).
    """
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    env = _tiny_env()
    state_a = env.reset()  # baseline: io_callback path

    server = CpuApproximationServer.from_env(env)
    partial, order, specs, _step = env.reset_external_jax_part()
    tokens, eqn_ids, _reward = server.evaluate(order, specs, 0, init=True)
    state_b = env.assemble_reset_result(partial, tokens, eqn_ids)

    # Every leaf in EnvState should match. axis_state / axis_valid_mask
    # are static; tokens / eqn_ids come from the tokenizer and are the
    # only fields the external path touched on top of partial.
    for name in state_a._fields:
        a = np.asarray(getattr(state_a, name))
        b = np.asarray(getattr(state_b, name))
        np.testing.assert_array_equal(
            a, b, err_msg=f"reset mismatch on EnvState.{name}"
        )


# ---------------------------------------------------------------------------
# Step: external + assemble == step()
# ---------------------------------------------------------------------------


def _seed_action(env, rng_seed: int) -> StepAction:
    """Sample a deterministic StepAction (vertex + no-rule specs).

    We only need this to drive `step()` — the test cares about parity
    between the io_callback path and the external path, not about
    whether the chosen vertex is a good move. A vertex with no rules
    forces `_callback` down the "transforms = []" branch, which is
    well-behaved for tiny jaxprs.
    """
    rng = np.random.default_rng(rng_seed)
    vertex = int(env.valid_vertices[rng.integers(0, len(env.valid_vertices))])
    empty_specs = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    empty_specs[:, 2] = 0
    return StepAction(
        target_vertex=jnp.int32(vertex),
        rule_specs=jnp.asarray(empty_specs, dtype=jnp.int32),
    )


@pytest.mark.parametrize("rng_seed", [0, 1, 7])
def test_step_external_matches_step(rng_seed):
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    env = _tiny_env()
    state0 = env.reset()
    action = _seed_action(env, rng_seed)

    # Baseline: io_callback path
    out_a = env.step(state0, action)

    # External path: JIT part + server eval + assemble
    server = CpuApproximationServer.from_env(env)
    partial, order, specs, step = env.step_external_jax_part(state0, action)
    tokens, eqn_ids, reward = server.evaluate(order, specs, int(step))
    out_b = env.assemble_step_result(state0, partial, tokens, eqn_ids, reward)

    # State parity — every leaf must match.
    for name in out_a.state._fields:
        a = np.asarray(getattr(out_a.state, name))
        b = np.asarray(getattr(out_b.state, name))
        np.testing.assert_array_equal(
            a, b, err_msg=f"step mismatch on EnvState.{name} (seed={rng_seed})"
        )
    # Reward / terminated parity (also covered by the state field check
    # but spelled out so a failure here is unambiguous).
    np.testing.assert_array_equal(
        np.asarray(out_a.reward), np.asarray(out_b.reward),
        err_msg=f"step reward mismatch (seed={rng_seed})",
    )
    assert bool(out_a.terminated) == bool(out_b.terminated)


# ---------------------------------------------------------------------------
# Terminated-state guard: assemble must short-circuit
# ---------------------------------------------------------------------------


def test_assemble_step_result_respects_terminated_guard():
    """If `state_before.terminated` is True, `assemble_step_result`
    must return the input state with zero reward — mirroring the
    `lax.cond(state.terminated, _step_done, _step_process)` branch in
    `step()`. Otherwise the external path would silently keep
    advancing a finished env, off-spec for any rollout consumer.
    """
    env = _tiny_env()
    state0 = env.reset()
    terminated_state = state0._replace(terminated=jnp.array(True, dtype=jnp.bool_))

    # `partial_state` and the tokenizer outputs are real (we sourced
    # them from a normal step), but the assemble side should ignore
    # them because the input state was already terminated.
    partial, *_ = env.step_external_jax_part(state0, _seed_action(env, 0))
    out = env.assemble_step_result(
        terminated_state,
        partial,
        tokens=jnp.zeros_like(state0.tokens),
        eqn_ids=jnp.zeros_like(state0.eqn_ids),
        reward=jnp.ones(NUM_REWARDS, dtype=jnp.float32),
    )
    assert bool(out.terminated) is True
    np.testing.assert_array_equal(np.asarray(out.reward), np.zeros(NUM_REWARDS))
    # State is the input state unchanged — order / specs match terminated_state.
    np.testing.assert_array_equal(
        np.asarray(out.state.order), np.asarray(terminated_state.order),
    )


# ---------------------------------------------------------------------------
# Multi-step rollout parity (small but exercises the LRU compile cache)
# ---------------------------------------------------------------------------


def test_multi_step_rollout_matches():
    """Walk a 3-step trajectory through both paths and confirm
    state-leaf equality at every step. This catches drift that only
    surfaces when state accumulates across steps (e.g. an off-by-one
    in `step_count` or in how the order shift composes).
    """
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    env = _tiny_env()
    server = CpuApproximationServer.from_env(env)

    state_a = env.reset()
    state_b = env.reset()
    for t in range(3):
        action = _seed_action(env, rng_seed=10 + t)

        out_a = env.step(state_a, action)
        state_a = out_a.state

        partial, order, specs, step = env.step_external_jax_part(state_b, action)
        tokens, eqn_ids, reward = server.evaluate(order, specs, int(step))
        out_b = env.assemble_step_result(state_b, partial, tokens, eqn_ids, reward)
        state_b = out_b.state

        for name in state_a._fields:
            a = np.asarray(getattr(state_a, name))
            b = np.asarray(getattr(state_b, name))
            np.testing.assert_array_equal(
                a, b, err_msg=f"step {t} mismatch on EnvState.{name}"
            )
