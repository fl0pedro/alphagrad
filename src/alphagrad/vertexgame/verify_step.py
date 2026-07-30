import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


def f(x):
    return jnp.sum(jnp.sin(jnp.cos(x)) * x)


args = (jnp.ones((10,)),)
graph = make_graph(f, *args)
closed = jax.make_jaxpr(f)(*args)
env = VertexEliminationEnv.from_jaxpr(closed, args=args, target_fun=f)

num_v = int(graph.at[0, 0, 1].get())

print("unbatched")
order = jnp.arange(1, num_v + 1, dtype=jnp.int32)
state = env.reset()
print(state.tokens.shape)
assert state.tokens.shape == (1024,)

for i in range(min(5, num_v)):
    out = env.step(state, order[i])
    state = out.state
    print(i + 1, out.reward)

print("batched")
order2 = order[::-1]
batch_order = jnp.stack([order, order2, order, order2])
batch_graph = jnp.broadcast_to(graph, (4, *graph.shape))
state = env.reset(batch_order)
print(state.tokens.shape)
assert state.tokens.shape == (4, 1024)

for i in range(min(5, num_v)):
    # batch_order is (4, 4), so batch_order[:, i] gets the i-th action for each batch element
    out = jax.vmap(env.step)(state, batch_order[:, i])
    state = out.state
    print(i + 1, out.reward)

print(jnp.count_nonzero(state.tokens[0]))

# we want to approximate the gradient
# we need jaxpr to
# function types + arguments
# leverage the group meeting for the
# gauge how mucht he people who don't care
# if you make a measurement and make a decision on the measurement, KEEP THAT MEASUREMENT
# keep the goal always in mind, and only optimize if it is a
# get feedback f
