import jax
import jax.numpy as jnp
from jax import jit, vmap

from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


def test_reset_step_jit_vmap():
    def f(x):
        return jnp.sum(jnp.sin(x) * x)

    x = jnp.ones((5,))
    jaxpr = jax.make_jaxpr(f)(x)
    env = VertexEliminationEnv.from_jaxpr(jaxpr, args=(x,))

    num_v = len(jaxpr.jaxpr.eqns)
    initial_order = jnp.arange(1, num_v + 1, dtype=jnp.int32)

    jit_reset = jit(env.reset)
    jit_step = jit(env.step)

    state = jit_reset()
    # action needs to be batched
    batched_action_0 = jnp.broadcast_to(initial_order[0], (state.order.shape[0],))
    out = jit_step(state, batched_action_0)

    assert jnp.all(state.step_count == 0)
    assert jnp.all(out.state.step_count == 1)

    batch_size = 4
    v_states = env.reset(num_envs=batch_size)
    # v_states is (batch_size, ...)
    assert v_states.step_count.shape == (batch_size,)

    v_step = vmap(env.step)
    # env.step is already batched, but we test direct call here
    v_actions = jnp.broadcast_to(initial_order[0], (batch_size,))
    v_outs = env.step(v_states, v_actions)

    assert v_outs.state.step_count.shape == (batch_size,)
    assert jnp.all(v_outs.state.step_count == 1)
