import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


def test_parallel_consistency():
    n = 10
    compl = 5

    def f(x):
        for _ in range(compl):
            x = jnp.sin(jnp.cos(x)) * x
        return jnp.sum(x)

    args_in = (jnp.ones((n,)),)
    jaxpr = jax.make_jaxpr(f)(*args_in)
    batch_size = 4

    env = VertexEliminationEnv.from_jaxpr(jaxpr, args=args_in)

    state_seq = env.reset(num_envs=1)
    states_par = env.reset(num_envs=batch_size)

    # state_seq.tokens is (1, MAX_TOKENS)
    # states_par.tokens is (batch_size, MAX_TOKENS)
    np.testing.assert_array_equal(state_seq.tokens[0], states_par.tokens[0])

    action = 1
    # action needs to be batched
    batched_action_seq = jnp.broadcast_to(action, (state_seq.order.shape[0],))
    out_seq = env.step(state_seq, batched_action_seq)

    # actions for parallel need to be batched
    batched_actions_par = jnp.broadcast_to(action, (states_par.order.shape[0],))
    outs_par = env.step(states_par, batched_actions_par)

    np.testing.assert_array_equal(out_seq.state.tokens[0], outs_par.state.tokens[0])
    np.testing.assert_allclose(out_seq.reward[0], outs_par.reward[0])
