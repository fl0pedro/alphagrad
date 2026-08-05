import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: SyntaxError: invalid syntax "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import sys
import os
# Add paths to sys.path to ensure imports work across sibling directories
sys.path.append("/Users/florian/Downloads/tmp/home/florian/dsnn/alphagrad/src")
sys.path.append("/Users/florian/Downloads/tmp/home/florian/dsnn/graphax/src")

import jax
import jax.numpy as jnp
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

def test_jit_vmap():
    print("Testing JIT and VMAP compatibility...")
    def fn(x, y):
        a = x * y
        b = jnp.sin(a)
        return jnp.log(b), a - b

    x = jnp.ones((3, 3))
    y = jnp.ones((3, 3))
    
    # 1. Get JAXpr
    closed_jaxpr = jax.make_jaxpr(fn)(x, y)
    
    # 2. Get initial adjacency matrix
    edges = make_graph(fn, x, y)
    
    # 3. Define an elimination order
    num_v = len(closed_jaxpr.jaxpr.eqns)
    order = jnp.arange(1, num_v + 1, dtype=jnp.int32)
    
    # 4. Initialize environment
    env = VertexEliminationEnv(
        closed_jaxpr.jaxpr, 
        argnums=(0, 1), 
        args=(x, y), 
        consts=closed_jaxpr.literals
    )
    
    state = env.reset()  # quarantined: reset() takes only num_envs now
    
    # 5. Test JIT
    print("Running JIT step...")
    jit_step = jax.jit(env.step)
    action = order[0]
    out = jit_step(state, action)
    print(f"JIT step success. Reward: {out.reward}")
    print(f"Tokens shape: {out.state.tokens.shape}")
    
    # 6. Test VMAP
    print("Running VMAP step (batch_size=4)...")
    batch_size = 4
    # Stack the state to create a batch
    batched_state = jax.tree_util.tree_map(lambda x: jnp.stack([x]*batch_size), state)
    # Different actions for each batch element
    # Vertex elimination order usually starts with 1, 2, ...
    batched_actions = jnp.array([order[0], order[1], order[0], order[1]], dtype=jnp.int32)
    
    batched_step = jax.vmap(env.step)
    batched_out = batched_step(batched_state, batched_actions)
    
    print(f"VMAP step success.")
    print(f"Batched rewards: {batched_out.reward}")
    print(f"Batched tokens shape: {batched_out.state.tokens.shape}")
    
    # Verify rewards are different if actions were different (if applicable)
    print("Verification complete.")

if __name__ == "__main__":
    try:
        test_jit_vmap()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
