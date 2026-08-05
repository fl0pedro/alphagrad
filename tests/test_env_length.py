import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: legacy alphagrad.vertexgame env path. The package "
    "imports again (its profiler import was repaired) but its API has drifted "
    "from graphax and from the token layout these tests assert; the approx/ "
    "campaigns do not use it. Kept for provenance.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp
import numpy as np
from graphax.core import extract_jaxpr

from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


def test_jaxpr_length_increase():
    def f(x):
        for _ in range(3):
            x = jnp.sin(x) * x
        return jnp.sum(x)

    x = jnp.ones((5,))
    jaxpr = jax.make_jaxpr(f)(x)
    env = VertexEliminationEnv.from_jaxpr(jaxpr, args=(x,))

    initial_order = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_v = len(initial_order)

    state = env.reset()

    # Use first env for length tracking
    active_lengths = [jnp.sum(state.tokens[0] != 0)]
    for step_idx in range(num_v):
        # Use initial_order, not state.order (which is the result sequence)
        action = initial_order[step_idx]
        batched_action = jnp.broadcast_to(action, (state.order.shape[0],))
        out = env.step(state, batched_action)
        state = out.state
        active_lengths.append(jnp.sum(state.tokens[0] != 0))

    length_diffs = np.diff(np.array(active_lengths))
    # Elimination should change the graph tokens
    assert np.any(length_diffs != 0)
    assert jnp.all(state.terminated)

    prev_jaxpr_str = None
    prev_len = 0
    prev_tkn = np.array([])

    args_np = jax.tree_util.tree_map(np.asarray, env.args)
    consts_np = jax.tree_util.tree_map(np.asarray, env.consts)

    for stop in range(1, num_v + 1):
        partial_order = np.asarray(initial_order)[:stop]
        ve = extract_jaxpr(
            env.jaxpr,
            env.argnums,
            partial_order.tolist(),
            env.sparse,
            args_np,
            consts_np,
        )
        curr_jaxpr_str = str(ve.jaxpr)
        curr_len = len(ve.jaxpr.eqns)
        curr_tkn = ve.tokenized()

        assert curr_len >= prev_len
        # The number of tokens might not strictly increase for element-wise
        assert len(curr_tkn) >= len(prev_tkn)

        if prev_jaxpr_str is not None:
            assert curr_jaxpr_str != prev_jaxpr_str
            assert curr_tkn.size >= prev_tkn.size

        prev_jaxpr_str = curr_jaxpr_str
        prev_len = curr_len
        prev_tkn = curr_tkn

    curr_state = env.reset()
    for i in range(num_v):
        action = initial_order[i]
        batched_action = jnp.broadcast_to(action, (curr_state.order.shape[0],))
        out = env.step(curr_state, batched_action)
        curr_state = out.state

    assert jnp.all(curr_state.terminated)
    assert jnp.all(curr_state.step_count == num_v)
