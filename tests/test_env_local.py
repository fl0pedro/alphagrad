import os
import sys
import time
import tracemalloc
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

print("Importing project...", flush=True)
import graphax.examples as examples

from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


def setup_env(batch_size=8):
    # Use examples.f (larger graph)
    key = jrand.PRNGKey(0)
    # Inputs for examples.f
    a = jrand.uniform(key, (4,))
    b = jrand.uniform(key, (2, 3))
    c = jrand.uniform(key, (4, 4))
    d = jrand.uniform(key, (4, 1))
    xs = (a, b, c, d)
    fn = examples.f

    print("Tracing function...", flush=True)
    closed_jaxpr = jax.make_jaxpr(fn)(*xs)

    print("Creating Env...", flush=True)
    # We pass target_fun=fn to use the new efficient path
    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=xs, target_fun=fn)

    num_v = len(closed_jaxpr.jaxpr.eqns)

    # Batch setup
    initial_order = jnp.tile(jnp.arange(1, num_v + 1), (batch_size, 1))

    print("Making graph...", flush=True)
    graph = make_graph(fn, *xs)
    initial_edges = jnp.broadcast_to(graph, (batch_size, *graph.shape))

    return env, initial_order, initial_edges, num_v


def run_benchmark(name, batch_size=8, steps=50):
    print(f"\n--- Running Benchmark: {name} ---")

    # Setup
    env, initial_order, initial_edges, num_v = setup_env(batch_size)

    # Warmup / Reset
    print("Resetting...", flush=True)
    state = env.reset(, initial_edges)
    state.tokens.block_until_ready()

    print(f"Starting rollout for {steps} steps...", flush=True)

    tracemalloc.start()
    start_time = time.perf_counter()

    current_state = state

    # JIT step function
    step_fn = jax.jit(jax.vmap(env.step))

    # We rely on JIT compilation happening during the first step or we can warmup.
    # Benchmarking typically includes first-run overhead unless explicitly warm-started.
    # But usually we care about steady state or total process time.
    # Let's do one warmup step outside timer?
    # No, let's include it but run enough steps.

    # Actually, let's warm up one step to avoid JIT noise in the timing if we want pure runtime.
    # But tokenization backend isn't JITted, so it doesn't matter much.
    # JAX dispatch overhead will be JITted.

    # Warmup
    print("Warming up JIT...", flush=True)
    warmup_actions = current_state.order[:, 0]
    out = step_fn(current_state, warmup_actions)
    out.state.tokens.block_until_ready()
    print("Warmup done. Measuring...", flush=True)

    # Reset timer after warmup
    tracemalloc.clear_traces()
    start_time = time.perf_counter()

    from tqdm import tqdm

    for i in tqdm(range(steps), desc=name):
        # Pick action: node at current step pointer
        # (This simulates following the given order)
        step_idx = current_state.step_count[0]
        actions = current_state.order[:, step_idx]

        env_out = step_fn(current_state, actions)
        current_state = env_out.state
        current_state.tokens.block_until_ready()

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration = end_time - start_time
    print(f"Benchmark {name} Done.")
    print(f"Time: {duration:.4f}s")
    print(f"Peak Memory: {peak_mem / 1024 / 1024:.2f} MB")

    return duration, peak_mem


def main():
    batch_sizes = [1]  # [32, 64, 128, 256, 512]  # [8, 32, 64]

    all_results = {}  # batch_size -> list of (name, dur, mem)

    try:
        for b_size in batch_sizes:
            print(f"\n\n>>> Benchmarking Batch Size: {b_size} <<<")
            results = []
            name = "ThreadPool"
            try:
                n_steps = 100

                dur, mem = run_benchmark(name, batch_size=b_size, steps=n_steps)
                results.append((name, dur, mem, n_steps))
                print(results[-1])
            except Exception as e:
                print(f"Benchmark {name} failed: {e}")
                import traceback

                traceback.print_exc()

            all_results[b_size] = results

    except KeyboardInterrupt:
        print("Interrupted.")

    # # Print Table
    # print("\n\n=== BENCHMARK RESULTS SUMMARY ===")
    # for b_size, results in all_results.items():
    #     print(f"\nBatch Size: {b_size}")
    #     print(
    #         f"{'Configuration':<35} | {'Steps':<5} | {'Total Time':<10} | {'Time/Step':<10} | {'Peak Mem (MB)':<15}"
    #     )
    #     print("-" * 85)
    #     for name, dur, mem, n_steps in results:
    #         time_per_step = dur / n_steps
    #         print(
    #             f"{name:<35} | {n_steps:<5} | {dur:<10.4f} | {time_per_step:<10.4f} | {mem / 1024 / 1024:<15.2f}"
    #         )


if __name__ == "__main__":
    import cProfile

    cProfile.runctx("main()", globals(), locals())
