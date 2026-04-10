import time
import os
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv, MAX_TOKENS
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

def f(x):
    return jnp.sin(jnp.cos(x)) * x

def benchmark_rollout():
    print("Initializing environment...")
    jaxpr = jax.make_jaxpr(f)(jnp.ones(10))
    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(10))
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr, 
        args=[jnp.ones(10)], sparse=True,
        target_fun=f
    )
    
    # Create batch
    batch_size = 8
    # Use closed_jaxpr directly as make_graph expects it
    graph = make_graph(closed_jaxpr, env.args)
    num_v = graph.shape[2]
    print(f"Num vertices: {num_v}")

    # Initial Reset
    key = jax.random.PRNGKey(0)
    initial_order = jnp.array([list(range(1, num_v + 1))] * batch_size, dtype=jnp.int32)
    # initial_edges = jnp.broadcast_to(graph, (batch_size, *graph.shape)) # VertexEliminationEnv doesn't take edges
    
    print("Warmup step...")
    state = env.reset(num_envs=batch_size)
    state.tokens.block_until_ready()
    
    print("Starting benchmark loop...")
    start_time = time.time()
    
    current_state = state
    for i in range(num_v): # Do exactly num_v steps
        step_idx = current_state.step_count[0]
        actions = current_state.order[:, step_idx] 
        
        # Pass dummy preferred vertices (next in order)
        preferred = current_state.order[:, step_idx+1:step_idx+4] if step_idx+4 < num_v else None
        
        env_out = env.step(current_state, actions, prefetch_order=preferred)
        current_state = env_out.state
        current_state.tokens.block_until_ready()
        print(f"Step {i} done")

    end_time = time.time()
    print(f"Total time for 5 steps (batch={batch_size}): {end_time - start_time:.4f}s")
    print("Benchmark complete.")

if __name__ == "__main__":
    benchmark_rollout()
