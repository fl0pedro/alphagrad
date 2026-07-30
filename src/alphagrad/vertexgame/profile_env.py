print("Starting script")
import jax
import jax.numpy as jnp
import numpy as np
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv
print("Finished imports")

def target_f(x):
    for _ in range(20):
        x = jnp.sin(jnp.cos(x)) * x
    return jnp.sum(x)

print("def target_f")

# Setup
size = 100
args = (jnp.ones((size,)),)
closed_jaxpr = jax.make_jaxpr(target_f)(*args)
env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=args)#, prefetch_mode="diagonal")
print("Initialized Env")

# Initial state
num_v = len(env.valid_vertices)
initial_order = jnp.array(list(env.valid_vertices), dtype=jnp.int32)
state = env.reset()
print("Started Env")

# Warmup JIT
print("Warming up JIT...")
action = state.order[0]
_ = env.step(state, action)
jax.block_until_ready(_)

# Profile
print("Starting profile...")

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    curr_state = state
    for i in range(min(50, num_v)):
        # Simple strategy: just pick the first available vertex
        action = curr_state.order[curr_state.step_count]
        out = env.step(curr_state, action)
        curr_state = out.state
        if out.terminated:
            break

    jax.block_until_ready(curr_state)

print("Profile saved to /tmp/jax_trace")