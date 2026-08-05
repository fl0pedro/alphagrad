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

import jax.numpy as jnp
import jax
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

def test_env():
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
    
    # 3. Define an elimination order (e.g. forward)
    num_v = len(closed_jaxpr.jaxpr.eqns)
    order = list(range(1, num_v + 1))
    
    # 4. Reset environment
    # 4. Initialize environment
    env = VertexEliminationEnv(
        closed_jaxpr.jaxpr, 
        argnums=(0, 1), 
        args=(x, y), 
        consts=closed_jaxpr.literals
    )
    
    state = env.reset(=order, edges=edges)
    
    print(f"Initial tokens length: {len(state.tokens)}")
    print(f"Initial order: {state.order}")
    print(f"Step count: {state.step_count}")
    
    # 5. Take a step
    # Let's eliminate the first vertex in the order
    action = order[0] 
    out = env.step(state, action)
    
    print("\n--- After Step 1 ---")
    print(f"New tokens length: {len(out.state.tokens)}")
    print(f"New order: {out.state.order}")
    print(f"New step count: {out.state.step_count}")
    print(f"Reward: {out.reward}")
    print(f"Terminated: {out.terminated}")

    # 6. Take another step
    action = order[1]
    out2 = env.step(out.state, action)
    
    print("\n--- After Step 2 ---")
    print(f"New tokens length: {len(out2.state.tokens)}")
    print(f"New step count: {out2.state.step_count}")
    print(f"Reward: {out2.reward}")
    print(f"Terminated: {out2.terminated}")

if __name__ == "__main__":
    test_env()
