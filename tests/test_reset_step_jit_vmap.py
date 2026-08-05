import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: legacy alphagrad.vertexgame env path. The package "
    "imports again (its profiler import was repaired) but its API has drifted "
    "from graphax and from the token layout these tests assert; the approx/ "
    "campaigns do not use it. Kept for provenance.",
    allow_module_level=True,
)

import sys
import os

# Add paths to sys.path to ensure imports work across sibling directories
# We use insert(0, ...) to override any installed versions.
# MUST DO THIS BEFORE IMPORTING ALPHAGRAD
sys.path.insert(0, "/Users/florian/Downloads/tmp/home/florian/dsnn/alphagrad/src")
sys.path.insert(0, "/Users/florian/Downloads/tmp/home/florian/dsnn/graphax/src")

import jax
import jax.numpy as jnp
import numpy as np
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

print(f"DEBUG: VertexEliminationEnv source: {sys.modules['alphagrad.vertexgame.vertex_game_w_tokens'].__file__}")

def test_reset_step_jit_vmap():
    print("Testing JIT and VMAP compatibility for reset and step...")
    
    def f(x):
        y = x
        for _ in range(5):
            y = jnp.sin(jnp.cos(y)) * y
        return jnp.sum(y)

    x = jnp.ones((10,))
    
    # 1. Get JAXpr
    closed_jaxpr = jax.make_jaxpr(f)(x)
    
    # 2. Get initial adjacency matrix (not strictly needed for reset in new version, but let's have it)
    edges = make_graph(f, x)
    
    # 3. Define an elimination order
    num_v = len(closed_jaxpr.jaxpr.eqns)
    order = jnp.arange(1, num_v + 1, dtype=jnp.int32)
    
    # 4. Initialize environment
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr, 
        args=(x,)
    )
    
    # --- Test Reset ---
    print("\n[RESET TESTS]")
    
    print("Testing jax.jit(env.reset)...")
    jit_reset = jax.jit(env.reset)
    state = jit_reset(order)
    print(f"JIT reset success. Tokens shape: {state.tokens.shape}")
    
    print("Testing jax.vmap(env.reset)...")
    batch_size = 4
    batched_orders = jnp.tile(order, (batch_size, 1))
    vmap_reset = jax.vmap(env.reset)
    batched_state = vmap_reset(batched_orders)
    print(f"VMAP reset success. Batched tokens shape: {batched_state.tokens.shape}")

    # --- Test Step ---
    print("\n[STEP TESTS]")
    
    print("Testing jax.jit(env.step)...")
    jit_step = jax.jit(env.step)
    action = order[0]
    out = jit_step(state, action)
    print(f"JIT step success. Reward: {out.reward}")
    
    print("Testing jax.vmap(env.step)...")
    batched_actions = jnp.array([order[0], order[1], order[0], order[1]], dtype=jnp.int32)
    vmap_step = jax.vmap(env.step)
    batched_out = vmap_step(batched_state, batched_actions)
    print(f"VMAP step success. Batched rewards: {batched_out.reward}")
    
    print("Testing jax.jit(jax.vmap(env.step))...")
    jit_vmap_step = jax.jit(jax.vmap(env.step))
    jit_vmap_out = jit_vmap_step(batched_state, batched_actions)
    print(f"JIT(VMAP) step success. Batched rewards: {jit_vmap_out.reward}")

    print("\nVerification complete - All JIT/VMAP tests for reset and step passed!")

if __name__ == "__main__":
    try:
        test_reset_step_jit_vmap()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
