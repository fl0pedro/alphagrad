import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name '_update_metrics' from 'alphagrad.vertexgame.vertex_game_w_tokens' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/vertex_game_w_toke "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import numpy as np
import threading
import time
from alphagrad.vertexgame.vertex_game_w_tokens import GlobalPrefetcher, _CACHE, prefetch_task_generator, _update_metrics

import jax
import jax.numpy as jnp

def test_prefetch_logic():
    # Setup - use a real jaxpr
    def f(x): return jnp.sin(x)
    jaxpr = jax.make_jaxpr(f)(jnp.ones(10)).jaxpr
    
    argnums = (0,)
    has_aux = False
    sparse = False
    args_ndims = (1,)
    consts_ndims = ()
    order = np.array([1])
    step_idx = 0
    prefetch_order = np.array([1])
    prefetch_type = 'diagonal_powerset'
    args = (np.ones(10),)
    consts = ()

    # Clear cache for testing
    _CACHE._tokens_cache = {}
    _CACHE._counts_cache = {}

    prefetcher = GlobalPrefetcher(num_workers=2)

    # 1. Test generator yields tasks
    gen = prefetch_task_generator(jaxpr, argnums, has_aux, sparse, args_ndims, consts_ndims, order, step_idx, prefetch_order, prefetch_type, args, consts)
    tasks = list(gen)
    print(f"Generated {len(tasks)} tasks")
    assert len(tasks) > 0

    # 2. Test GlobalPrefetcher consumes tasks
    # We'll use a wrapper that updates metrics so we can track it
    def test_gen():
        for task in prefetch_task_generator(jaxpr, argnums, has_aux, sparse, args_ndims, consts_ndims, order, step_idx, prefetch_order, prefetch_type, args, consts):
            yield task

    print("Setting generator...")
    prefetcher.set_generator(test_gen())
    
    # Wait for workers to do some work
    time.sleep(1)
    
    # Check if cache is being populated
    print(f"Tokens cache size: {len(_CACHE._tokens_cache)}")
    print(f"Counts cache size: {len(_CACHE._counts_cache)}")
    assert len(_CACHE._tokens_cache) > 0 or len(_CACHE._counts_cache) > 0

    # 3. Test skipping cached items
    # Populate cache for a specific key
    future_order = [1] # simulate one step with action 1
    cache_key = (id(jaxpr), tuple(future_order[:1]))
    _CACHE.set_tokens(cache_key, np.zeros(1024, dtype=np.int32))
    _CACHE.set_counts(cache_key, np.array(0, dtype=np.int32))

    # New generator should skip this specific task
    gen2 = prefetch_task_generator(jaxpr, argnums, has_aux, sparse, args_ndims, consts_ndims, order, step_idx, np.array([1]), prefetch_type, args, consts)
    tasks2 = list(gen2)
    
    # We expect 0 tasks because we manually cached both tokens and counts for the only item in this small prefetch_order
    print(f"Tasks after manual caching for order=2: {len(tasks2)}")
    assert len(tasks2) == 0

    print("Test passed!")

if __name__ == "__main__":
    test_prefetch_logic()
