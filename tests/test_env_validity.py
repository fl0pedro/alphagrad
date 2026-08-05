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
import pytest
from graphax.core import extract_jaxpr

from alphagrad.vertexgame.vertex_game_w_tokens import MAX_TOKENS, VertexEliminationEnv


def verify_tokenization(env, state):
    # state is batched, take first env for verification
    if state.order.ndim > 1:
        state = jax.tree_util.tree_map(lambda x: x[0], state)
    order_np = np.array(state.order)
    step_count = int(state.step_count)

    if step_count < len(order_np):
        partial_order = order_np[:step_count]
    else:
        partial_order = order_np

    ve = extract_jaxpr(
        env.jaxpr,
        env.argnums,
        partial_order.tolist(),
        env.sparse,
        jax.tree_util.tree_map(np.asarray, env.args),
        jax.tree_util.tree_map(np.asarray, env.consts),
    )

    manual_tokens = np.zeros(MAX_TOKENS, dtype=np.int32)
    tokens = ve.tokenized()
    n = min(len(tokens), MAX_TOKENS)
    manual_tokens[:n] = tokens[:n]

    env_tokens = np.array(state.tokens)
    np.testing.assert_array_equal(env_tokens, manual_tokens)


@pytest.mark.parametrize("func_type", ["element-wise", "matrix-multiply"])
def test_rollout_validity(func_type):
    size = 5
    x = jnp.ones((size,))

    if func_type == "element-wise":

        def f(x):
            for _ in range(3):
                x = jnp.sin(x) * x + jnp.cos(x)
            return jnp.sum(x)

        expect_zero = False
    else:

        def f(x):
            W = jnp.ones((size, size)) * 0.1
            y = W @ x
            return jnp.sum(jnp.sin(y) * y)

        expect_zero = False

    closed_jaxpr = jax.make_jaxpr(f)(x)
    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=(x,))
    num_v = len(closed_jaxpr.jaxpr.eqns)
    initial_order = jnp.arange(1, num_v + 1, dtype=jnp.int32)

    state = env.reset()
    verify_tokenization(env, state)

    curr_state = state
    total_reward = 0.0
    for i in range(num_v):
        action = initial_order[i]
        batched_action = jnp.broadcast_to(action, (curr_state.order.shape[0],))
        out = env.step(curr_state, batched_action)
        curr_state = out.state
        verify_tokenization(env, curr_state)
        total_reward += out.reward[0]

    assert jnp.all(curr_state.terminated)
    if expect_zero:
        assert total_reward == 0.0
    else:
        assert total_reward < 0.0
