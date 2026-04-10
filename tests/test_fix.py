
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import multiprocessing
from alphagrad.vertexgame.vertex_game_w_tokens_2 import VertexEliminationEnv, _CONTEXT_REGISTRY

def main():
    print("Testing VertexEliminationEnv from vertex_game_w_tokens_2.py")
    print(f"Start method: {multiprocessing.get_start_method()}")
    
    try:
        print(f"/dev/shm content: {os.listdir('/dev/shm')}")
    except Exception as e:
        print(f"Cannot list /dev/shm: {e}")

    # Simple dummy graph
    def target(x):
        return jnp.sum(x)
        
    args = (jnp.ones((10,)),)
    consts = ()
    jaxpr = jax.make_jaxpr(target)(*args)
    
    env = VertexEliminationEnv(jaxpr.jaxpr, [0], args, consts, target_fun=target)
    print("Env created.")
    print(f"Context registry keys: {list(_CONTEXT_REGISTRY.keys())}")
    
    # Dummy step
    N = 10
    edges = jnp.zeros((N, N), dtype=jnp.int32)
    order = jnp.arange(N, dtype=jnp.int32)
    
    print("Resetting env...")
    start_state = env.reset(, edges)
    # block
    start_state.tokens.block_until_ready()
    print("Reset done. Tokens shape:", start_state.tokens.shape)
    
    # Step
    print("Stepping...")
    out = env.step(start_state, 0)
    out.state.tokens.block_until_ready()
    print("Step done.")

if __name__ == "__main__":
    main()
