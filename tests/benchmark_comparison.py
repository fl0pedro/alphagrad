import time
import os

# Prevent JAX from pre-allocating all GPU memory, 
# which causes OOM when multiple processes are spawned.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import importlib.util
import sys

# Import current version
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv as NewEnv
import graphax.examples as examples
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

# Import old version from vertex_game_w_tokens_2.py
def import_old_env():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/alphagrad/vertexgame/vertex_game_w_tokens_2.py")
    if not os.path.exists(path):
        path = "src/alphagrad/vertexgame/vertex_game_w_tokens_2.py"
        
    spec = importlib.util.spec_from_file_location("vertex_game_w_tokens_2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vertex_game_w_tokens_2"] = module
    spec.loader.exec_module(module)
    return module.VertexEliminationEnv

OldEnv = import_old_env()

def setup_benchmark(EnvClass, batch_size=8):
    key = jrand.PRNGKey(0)
    a = jrand.uniform(key, (4,))
    b = jrand.uniform(key, (2, 3))
    c = jrand.uniform(key, (4, 4))
    d = jrand.uniform(key, (4, 1))
    xs = (a, b, c, d)
    fn = examples.f

    closed_jaxpr = jax.make_jaxpr(fn)(*xs)
    env = EnvClass.from_jaxpr(closed_jaxpr, args=xs, target_fun=fn)
    num_v = len(closed_jaxpr.jaxpr.eqns)
    initial_order = jnp.tile(jnp.arange(1, num_v + 1), (batch_size, 1))
    graph = make_graph(fn, *xs)
    initial_edges = jnp.broadcast_to(graph, (batch_size, *graph.shape))
    
    return env, initial_order, initial_edges

def run_rollout(env, initial_order, initial_edges, steps=10):
    state = env.reset(, initial_edges)
    state.tokens.block_until_ready()
    
    current_state = state
    step_fn = jax.jit(jax.vmap(env.step))
    
    # Warmup one step
    step_idx = current_state.step_count[0]
    actions = current_state.order[:, step_idx]
    out = step_fn(current_state, actions)
    out.state.tokens.block_until_ready()
    
    current_state = out.state
    
    start_time = time.perf_counter()
    for i in range(steps):
        step_idx = current_state.step_count[0]
        if step_idx >= current_state.order.shape[1]:
            break
        actions = current_state.order[:, step_idx]
        env_out = step_fn(current_state, actions)
        current_state = env_out.state
        current_state.tokens.block_until_ready()
    
    return time.perf_counter() - start_time

def main():
    batch_sizes = [1, 16, 32, 64]
    steps = 5
    print(f"Benchmarking with steps={steps}")
    
    results = []

    for b_size in batch_sizes:
        print(f"\n>>> Batch Size: {b_size} <<<")
        
        # Test New
        new_env, new_order, new_edges = setup_benchmark(NewEnv, b_size)
        new_time = run_rollout(new_env, new_order, new_edges, steps)
        new_per_step = new_time / steps
        print(f"NEW implementation time: {new_time:.4f}s ({new_per_step:.4f}s/step)")

        # Test Old
        # Ensure SHM Multiprocessing is ENABLED for the old implementation
        os.environ["FORCE_SEQUENTIAL_TOKENIZATION"] = "0"
        os.environ["DISABLE_SHM_TOKENIZATION"] = "0"
        os.environ["DISABLE_TOKEN_PREFETCH"] = "0"
        os.environ["EAGER_TOKEN_POOL"] = "1"
        
        old_env, old_order, old_edges = setup_benchmark(OldEnv, b_size)
        try:
            old_time = run_rollout(old_env, old_order, old_edges, steps)
            old_per_step = old_time / steps
            print(f"OLD implementation time: {old_time:.4f}s ({old_per_step:.4f}s/step)")
            speedup = old_time / new_time
            print(f"Speedup: {speedup:.2f}x")
            results.append((b_size, new_per_step, old_per_step, speedup))
        except Exception as e:
            print(f"OLD implementation failed for batch {b_size}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n=== SUMMARY ===")
    print(f"{'Batch Size':<10} | {'New (s/step)':<15} | {'Old (s/step)':<15} | {'Speedup':<10}")
    print("-" * 60)
    for b_size, new_s, old_s, speedup in results:
        print(f"{b_size:<10} | {new_s:<15.4f} | {old_s:<15.4f} | {speedup:<10.2f}")

if __name__ == "__main__":
    main()
