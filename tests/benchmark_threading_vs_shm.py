
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv as ThreadingEnv
import alphagrad.vertexgame.vertex_game_w_tokens_2 as shm_impl

# Setup SHM Env pointing to correct implementation
SHMEnv = shm_impl.VertexEliminationEnv

def run_experiment(batch_size: int, n_ops: int, n_steps_rollout: int):
    print(f"\n=== Experiment: Batch={batch_size}, N_ops={n_ops}, Rollout={n_steps_rollout} ===")
    
    # Define target function dynamically based on n_ops
    def target(y):
        # We use a loop to generate O(n_ops) equations
        val = y
        for _ in range(n_ops):
            val = jnp.sin(val) + jnp.cos(val)
        return jnp.sum(val)

    N_in = 100
    args = (jnp.ones((N_in,)),)
    consts = ()
    
    # Trace once
    jaxpr = jax.make_jaxpr(target)(*args)
    N_eqns = len(jaxpr.jaxpr.eqns)
    N_invars = len(jaxpr.jaxpr.invars)
    # print(f"Graph Size: {N_eqns} equations")

    # Common Setup
    edges_shape = (batch_size, 5, N_invars + N_eqns + 1, N_eqns)
    edges = jnp.zeros(edges_shape, dtype=jnp.int32)
    order = jnp.tile(jnp.arange(N_invars, N_invars + N_eqns, dtype=jnp.int32), (batch_size, 1))

    # Helper for running env
    def benchmark_env(env_cls, name):
        print(f"--- Benchmarking {name} ---", flush=True)
        try:
            env = env_cls(jaxpr.jaxpr, [0], args, consts, target_fun=target)

            @jax.jit
            def step_batch(s, a):
                return jax.vmap(env.step, in_axes=(0, 0))(s, a)

            # Reset
            start_reset = time.perf_counter()
            state = env.reset(, edges)
            state.tokens.block_until_ready()
            reset_time = time.perf_counter() - start_reset
            
            curr_state = state
            start_steps = time.perf_counter()
            for i in range(n_steps_rollout):
                current_actions = order[:, i] # Valid elimination
                out = step_batch(curr_state, current_actions)
                curr_state = out.state
                curr_state.tokens.block_until_ready()
            steps_time = time.perf_counter() - start_steps
            
            return steps_time
        except Exception as e:
            print(f"FAILED {name}: {e}", flush=True)
            return float('nan')

    # Run benchmarks
    t_thread = benchmark_env(ThreadingEnv, "Threading")
    t_shm = benchmark_env(SHMEnv, "SHM/ProcessPool")
    
    return t_thread, t_shm

def main():
    # Adjusted configs for speed/reliability
    # BATCH_SIZES = [32, 128, 512]
    # N_OPS = [50, 200]
    
    configs = [
        (32, 50),
        (128, 50),
        (512, 50),
        (32, 200),
        (128, 200),
        # (512, 200) # Skip heavy load
    ]
    
    ROLLOUT_STEPS = 5
    
    results = []
    
    print(f"{'Batch':<8} {'N_ops':<8} {'Thread(s)':<12} {'SHM(s)':<12} {'Speedup':<8}", flush=True)
    print("-" * 60, flush=True)
    
    for b, n in configs:
        t_thread, t_shm = run_experiment(b, n, ROLLOUT_STEPS)
        speedup = t_thread / t_shm if t_shm > 0 else 0.0
        print(f"{b:<8} {n:<8} {t_thread:<12.4f} {t_shm:<12.4f} {speedup:<8.2f}x", flush=True)
        results.append((b, n, t_thread, t_shm, speedup))

    print("\n\nMarkdown Table:", flush=True)
    print("| Batch | Ops (Graph Size) | Threading (s) | SHM/Pool (s) | Speedup |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r[0]} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.2f}x |")

if __name__ == "__main__":
    # Ensure SHM implementation uses the fallback if needed (handled internally)
    main()
