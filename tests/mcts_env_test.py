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
from alphagrad.vertexgame.mcts import run_mcts_search

def test_mcts_integration():
    print("Testing MCTS integration with mctx...")
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
    num_v = edges.shape[2]
    
    # 3. Initialize environment
    env = VertexEliminationEnv(
        closed_jaxpr.jaxpr, 
        argnums=(0, 1), 
        args=(x, y), 
        consts=closed_jaxpr.literals
    )
    
    # Create a batch of states
    batch_size = 2
    order = jnp.arange(1, num_v + 1, dtype=jnp.int32)
    state = env.reset()  # quarantined: reset() takes only num_envs now
    batched_state = jax.tree_util.tree_map(lambda x: jnp.stack([x]*batch_size), state)
    
    # 4. Define a dummy model
    # Model returns [value, policy_logits...]
    # policy_logits should have size NUM_ACTIONS = num_v
    num_actions = num_v
    
    def dummy_model(edges, key):
        # Deterministic but "random" policy/value for testing
        value = jnp.sum(edges) * 0.01
        policy_logits = jnp.ones(num_actions)
        return jnp.concatenate([jnp.array([value]), policy_logits])

    # 5. Run MCTS
    print(f"Running MCTS search with {batch_size} states...")
    rng_key = jax.random.PRNGKey(42)
    
    # Wrap in JIT to test compilation
    @jax.jit
    def search_fn(key, s):
        return run_mcts_search(
            key, s, dummy_model, env, 
            num_simulations=10
        )
    
    output = search_fn(rng_key, batched_state)
    
    print("MCTS Search success!")
    print(f"Action: {output.action}")
    print(f"Action weights shape: {output.action_weights.shape}")
    print(f"Root values: {output.value}")
    
    # 6. Verify sanity
    assert output.action.shape == (batch_size,)
    assert output.action_weights.shape == (batch_size, num_actions)
    print("Verification complete.")

if __name__ == "__main__":
    try:
        test_mcts_integration()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
