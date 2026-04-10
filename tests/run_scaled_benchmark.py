
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from functools import partial

# Import both implementations
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv as ThreadingEnv
import alphagrad.vertexgame.vertex_game_w_tokens_2 as shm_impl
SHMEnv = shm_impl.VertexEliminationEnv

from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph
from graphax.core import extract_jaxpr
import alphagrad.vertexgame.vertex_game_w_tokens as threading_module
from concurrent.futures import ThreadPoolExecutor

def host_tokenize_fn(jaxpr, argnums, has_aux, order, step_count, sparse, args, consts):
    ve_jaxpr = extract_jaxpr(
        jaxpr, argnums, order, sparse, args, consts
    )
    tokens = ve_jaxpr.tokenized()
    res = np.zeros(1024, dtype=np.int32)
    n = min(len(tokens), 1024)
    res[:n] = tokens[:n]
    return res

def run_benchmark_orchestrated(env_cls, batch_size, input_size, complexity, steps=5):
    # Use a slightly more complex function to make tokenization meaningful
    def f(x):
        y = x
        for _ in range(complexity): 
            y = jnp.sin(jnp.cos(y)) * y
        return jnp.sum(y)

    # Setup
    args = [jnp.ones((input_size,))]
    closed_jaxpr = jax.make_jaxpr(f)(*args)
    jaxpr = closed_jaxpr.jaxpr
    argnums = [0]
    has_aux = False
    sparse = False
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, closed_jaxpr.consts)
    
    # Create Env
    env = env_cls.from_jaxpr(
        closed_jaxpr,
        args=args
    )
    
    graph = make_graph(f, *args)
    num_v = int(graph.at[0, 0, 1].get())
    initial_order = jnp.tile(jnp.arange(1, num_v + 1, dtype=jnp.int32), (batch_size, 1))
    initial_edges = jnp.broadcast_to(graph, (batch_size, *graph.shape))
    
    # Warmup
    state = env.reset(, initial_edges)
    state.tokens.block_until_ready()
    
    manager = threading_module._PREFETCH_MANAGER
    executor = threading_module._TOKEN_EXECUTOR if threading_module._TOKEN_EXECUTOR else ThreadPoolExecutor()
    
    start_time = time.perf_counter()
    current_state = state
    
    # Rollout loop
    run_steps = min(steps, num_v)
    for i in range(run_steps):
        actions = initial_order[:, i]
        
        # Forecast the order after this action.
        # In this benchmark, actions[b] is already at position initial_order[b, i],
        # so the order will not change logically.
        next_step_counts = np.asarray(current_state.step_count) + 1
        order_nps = np.asarray(current_state.order)
        
        token_futures = []
        for b in range(batch_size):
            # We need tokens for the NEXT step (next_step_counts[b])
            remaining = tuple(order_nps[b, int(next_step_counts[b]):].tolist())
            f = manager.get_or_submit(
                executor,
                remaining,
                host_tokenize_fn,
                (jaxpr, argnums, has_aux, order_nps[b], next_step_counts[b], sparse, args_np, consts_np)
            )
            token_futures.append(f)
            
        # Wait for tokens
        tokens_list = [f.result() for f in token_futures]
        tokens_batch = jnp.stack([jnp.asarray(t) for t in tokens_list])
        
        # Step with provided tokens. These tokens will be stored in next_state.
        out = jax.vmap(env.step, in_axes=(0, 0, None, 0))(current_state, actions, None, tokens_batch)
        current_state = out.state
        
    duration = time.perf_counter() - start_time
    return duration / run_steps if run_steps > 0 else 0

def run_benchmark_config(env_cls, batch_size, input_size, complexity, steps=5):
    # Use a slightly more complex function to make tokenization meaningful
    def f(x):
        y = x
        for _ in range(complexity): 
            y = jnp.sin(jnp.cos(y)) * y
        return jnp.sum(y)

    # Setup
    args = [jnp.ones((input_size,))]
    closed_jaxpr = jax.make_jaxpr(f)(*args)
    
    # Create Env
    env = env_cls.from_jaxpr(
        closed_jaxpr, 
        args=args, 
        sparse=True,
        target_fun=f
    )
    
    # Create Graph & Initial State
    graph = make_graph(closed_jaxpr, args)
    num_v = graph.shape[2]
    
    initial_order = jnp.tile(jnp.arange(1, num_v + 1, dtype=jnp.int32), (batch_size, 1))
    initial_edges = jnp.broadcast_to(graph, (batch_size, *graph.shape))
    
    # Warmup
    state = env.reset(, initial_edges)
    state.tokens.block_until_ready()
    
    # Step function
    @jax.jit
    def step_batch(s, a):
        return jax.vmap(env.step, in_axes=(0, 0))(s, a)
        
    start_time = time.perf_counter()
    current_state = state
    
    # We rollout for 'steps' or num_v, whichever is smaller
    run_steps = min(steps, num_v)
    for i in range(run_steps):
        actions = initial_order[:, i]
        # For prefetch, we pass the next few vertices as hints
        preferred = initial_order[:, i+1:i+4] if i+4 < num_v else None
        
        # We need to handle step signature difference if possible, 
        # or just assume the env.step handles preferred_vertices
        try:
             out = jax.vmap(env.step)(current_state, actions, preferred)
        except TypeError:
             # Fallback if preferred_vertices not supported (should not happen with my changes)
             out = jax.vmap(env.step)(current_state, actions)
             
        current_state = out.state
        current_state.tokens.block_until_ready()
        
    duration = time.perf_counter() - start_time
    return duration / run_steps if run_steps > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    args_cli = parser.parse_args()

    # Size configurations
    sizes = {
        "small": {"N": 10, "complexity": 5, "steps": 5},
        "medium": {"N": 100, "complexity": 20, "steps": 10},
        "large": {"N": 500, "complexity": 50, "steps": 20}
    }
    
    config_params = sizes[args_cli.size]
    print(f"Running Scaled Benchmark: Size={args_cli.size.upper()} (N={config_params['N']}, Complexity={config_params['complexity']})")

    # Configurations to test
    # (Name, EnvClass, EnvVars)
    configs = [
        ("Threading (Baseline)", ThreadingEnv, {"DISABLE_TOKEN_PREFETCH": "1"}),
        ("Threading + Prefetch", ThreadingEnv, {}),
        ("Pickle (No Prefetch)", SHMEnv, {"DISABLE_SHM_TOKENIZATION": "1", "DISABLE_TOKEN_PREFETCH": "1"}),
        ("Pickle + Prefetch", SHMEnv, {"DISABLE_SHM_TOKENIZATION": "1"}),
        ("SHM (No Prefetch)", SHMEnv, {"DISABLE_TOKEN_PREFETCH": "1"}),
        ("SHM + Prefetch", SHMEnv, {}),
        ("Orchestrated Rollout", ThreadingEnv, {"ORCHESTRATED": "1"}),
    ]
    
    batch_sizes = [8, 16, 32, 64, 128, 256]
    
    header = f"{'Configuration':<25} | " + " | ".join([f"B={b:<7}" for b in batch_sizes])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    
    for name, env_cls, env_vars in configs:
        # Apply Env Vars
        original_vars = {k: os.environ.get(k) for k in env_vars}
        for k, v in env_vars.items():
            os.environ[k] = v
            
        row_res = []
        for b in batch_sizes:
            try:
                if "ORCHESTRATED" in env_vars:
                    t = run_benchmark_orchestrated(
                        env_cls, 
                        b, 
                        config_params["N"], 
                        config_params["complexity"],
                        config_params["steps"]
                    )
                else:
                    t = run_benchmark_config(
                        env_cls, 
                        b, 
                        config_params["N"], 
                        config_params["complexity"],
                        config_params["steps"]
                    )
                row_res.append(f"{t:.4f}s")
            except Exception as e:
                row_res.append(f"Err")
                print(f"Error {name} Batch {b}: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"{name:<25} | " + " | ".join([f"{r:<9}" for r in row_res]), flush=True)
        
        # Restore env vars
        for k, v in original_vars.items():
            if v is None:
                if k in os.environ: del os.environ[k]
            else:
                os.environ[k] = v

if __name__ == "__main__":
    main()
