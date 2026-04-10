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
    np.testing.assert_array_equal(np.array(state.tokens), manual_tokens)


def get_matmul_task(size=5):
    def g(x):
        W = jnp.ones((size, size)) * 0.1
        y = W @ x
        return jnp.sum(jnp.sin(y) * y)

    x = jnp.ones((size,))
    jaxpr = jax.make_jaxpr(g)(x)
    env = VertexEliminationEnv.from_jaxpr(jaxpr, args=(x,))
    return env, len(jaxpr.jaxpr.eqns)


def test_reverse_order():
    env, num_v = get_matmul_task()
    order = jnp.arange(num_v, 0, -1, dtype=jnp.int32)
    state = env.reset()
    curr_state = state
    for i in range(num_v):
        action = order[i]
        batched_action = jnp.broadcast_to(action, (curr_state.order.shape[0],))
        out = env.step(curr_state, batched_action)
        curr_state = out.state
        verify_tokenization(env, curr_state)
    assert jnp.all(curr_state.terminated)


def test_shuffled_order():
    env, num_v = get_matmul_task()
    np.random.seed(42)
    order = np.arange(1, num_v + 1, dtype=np.int32)
    np.random.shuffle(order)
    order = jnp.asarray(order)
    state = env.reset()
    curr_state = state
    for i in range(num_v):
        action = order[i]
        batched_action = jnp.broadcast_to(action, (curr_state.order.shape[0],))
        out = env.step(curr_state, batched_action)
        curr_state = out.state
        verify_tokenization(env, curr_state)
    assert jnp.all(curr_state.terminated)


def test_invalid_actions():
    env, num_v = get_matmul_task()
    order = jnp.arange(1, num_v + 1, dtype=jnp.int32)
    state = env.reset()

    # OOB
    batched_action_oob = jnp.broadcast_to(999, (state.order.shape[0],))
    out = env.step(state, batched_action_oob)
    assert jnp.all(out.reward == 0.0)
    assert jnp.all(~out.terminated)

    # Redundant
    batched_action_1 = jnp.broadcast_to(1, (state.order.shape[0],))
    out1 = env.step(state, batched_action_1)
    out2 = env.step(out1.state, batched_action_1)
    assert jnp.all(out2.reward == 0.0)
    assert jnp.all(~out2.terminated)
