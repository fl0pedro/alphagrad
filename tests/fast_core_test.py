import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: legacy alphagrad.vertexgame env path. The package "
    "imports again (its profiler import was repaired) but its API has drifted "
    "from graphax and from the token layout these tests assert; the approx/ "
    "campaigns do not use it. Kept for provenance.",
    allow_module_level=True,
)


import jax
import jax.numpy as jnp
import pytest
import time
from alphagrad.vertexgame import core, fast_core

# Use a fixed seed for reproducibility
key = jax.random.PRNGKey(0)

def generate_random_graph(num_i=5, num_v=5, num_o=3):
    total_dim = num_i + num_v + num_o
    # Random sparsity
    sparsity = jax.random.choice(key, jnp.arange(-10, 12), shape=(total_dim, num_v))
    # Random shapes (1-d to keep it simpleish, or small dimensions)
    outs = jax.random.randint(key, (2, total_dim, num_v), 1, 5)
    inps = jax.random.randint(key, (2, total_dim, num_v), 1, 5)
    
    edges = jnp.zeros((5, total_dim, num_v), dtype=jnp.int32)
    edges = edges.at[0].set(sparsity)
    edges = edges.at[1:3].set(outs)
    edges = edges.at[3:].set(inps)
    
    return edges

def test_vertex_eliminate_correctness():
    # Setup
    num_v = 10
    num_rows = 20 # num_i + num_v
    edges = jnp.zeros((5, num_rows, num_v), dtype=jnp.int32)
    
    # Fill with random data
    k = jax.random.PRNGKey(42)
    edges = jax.random.randint(k, (5, num_rows, num_v), 0, 10).astype(jnp.int32)
    # Ensure sparsity is within valid range (-10 to 11), offset is handled in functions
    edges = edges.at[0].set(jax.random.randint(k, (num_rows, num_v), -10, 12))
    
    # Vertex to eliminate (1-based index)
    vertex = 3
    
    # Run core implementation
    res_core, fmas_core = core.vertex_eliminate(vertex, edges)
    
    # Run fast implementation
    res_fast, fmas_fast = fast_core.vertex_eliminate(vertex, edges)
    
    # Compare
    assert jnp.array_equal(res_core, res_fast)
    
    # Check fmas
    assert jnp.isclose(fmas_core, fmas_fast)

def test_benchmark_vertex_eliminate():
    # Setup larger graph for benchmarking
    num_v = 100
    num_rows = 200
    k = jax.random.PRNGKey(42)
    edges = jax.random.randint(k, (5, num_rows, num_v), 0, 10).astype(jnp.int32)
    edges = edges.at[0].set(jax.random.randint(k, (num_rows, num_v), -10, 12))
    vertex = 10
    
    # JIT compile both functions
    jit_core = jax.jit(core.vertex_eliminate)
    jit_fast = jax.jit(fast_core.vertex_eliminate)
    
    # Warmup
    jit_core(vertex, edges)
    jit_fast(vertex, edges)
    
    # Benchmark Core
    start_time = time.time()
    for _ in range(100):
        jit_core(vertex, edges)[0].block_until_ready()
    core_time = time.time() - start_time
    
    # Benchmark Fast
    start_time = time.time()
    for _ in range(100):
        jit_fast(vertex, edges)[0].block_until_ready()
    fast_time = time.time() - start_time
    
    print(f"\nCore time: {core_time:.4f}s")
    print(f"Fast time: {fast_time:.4f}s")
    print(f"Speedup: {core_time / fast_time:.2f}x")
    
    # Assert improvement
    assert fast_time < core_time

if __name__ == "__main__":
    test_vertex_eliminate_correctness()
    test_benchmark_vertex_eliminate()
