import time
import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple
import numpy as np
from jax.experimental.compilation_cache import compilation_cache as cc

# Set up compilation cache if needed
cc.set_cache_dir("~/.jax_cache")

def compute_cosine_similarity(exact: jax.Array, approx: jax.Array) -> jax.Array:
    exact_flat = exact.flatten()
    approx_flat = approx.flatten()
    num = jnp.dot(exact_flat, approx_flat)
    den = jnp.linalg.norm(exact_flat) * jnp.linalg.norm(approx_flat)
    return num / (den + 1e-8)

def compute_frob_norm(exact: jax.Array, approx: jax.Array) -> jax.Array:
    return jnp.linalg.norm(exact.flatten() - approx.flatten())

def winsorized_mean(data: np.ndarray, limits: Tuple[float, float] = (0.1, 0.1)) -> float:
    """
    Compute the winsorized mean of a 1D array.
    """
    from scipy.stats.mstats import winsorize
    win_data = winsorize(data, limits=limits)
    return float(np.mean(win_data))

class PopArt:
    """
    PopArt normalizer to scale vastly different metrics (memory, latency, cossim)
    to a stable reward signal.
    """
    def __init__(self, num_metrics: int, beta: float = 0.01):
        self.num_metrics = num_metrics
        self.beta = beta
        self.mu = np.zeros(num_metrics, dtype=np.float32)
        self.var = np.ones(num_metrics, dtype=np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        sigma = np.sqrt(self.var)
        return (x - self.mu) / (sigma + 1e-8)

    def update(self, x: np.ndarray):
        """
        Updates the running mean and variance using the new sample.
        """
        self.mu = (1 - self.beta) * self.mu + self.beta * x
        self.var = (1 - self.beta) * self.var + self.beta * np.square(x - self.mu)
        
def evaluate_metrics(
    approx_fn: Callable, 
    exact_fn: Callable, 
    key: jax.Array, 
    num_samples: int = 5, 
    runs_per_sample: int = 4, 
    loops: int = 50
) -> Tuple[float, float, float]:
    """
    Measures latency, peak memory, and cosine similarity for the approx_fn.
    Uses 5 random samples * 4 runs, and loops for 50 times to better accumulate.
    """
    # XLA/HLO Fusion via jit
    jitted_approx = jax.jit(approx_fn)
    jitted_exact = jax.jit(exact_fn)
    
    # Generate dummy data for samples
    # Assuming signature: fn(params, x) -> loss
    
    all_latencies = []
    all_memories = []
    all_cossims = []
    
    device = jax.devices()[0]

    for sample_idx in range(num_samples):
        # Randomly reinitialize weights and inputs
        k1, k2 = jax.random.split(key, 2)
        params = jax.random.normal(k1, (10, 10))
        x = jax.random.normal(k2, (10,))
        
        # Warmup and exact jacobian
        exact_jac = jax.jacrev(jitted_exact, argnums=0)(params, x)
        
        for run_idx in range(runs_per_sample):
            # JAX Memory Stats requires running under local sync
            try:
                device.clear_memory_stats()
            except Exception:
                pass # Not supported on CPU/some backends
            
            # Measure Latency over loops
            start_t = time.perf_counter()
            for _ in range(loops):
                # The approx_fn computes gradients from Jacobian
                approx_jac = jax.jacrev(jitted_approx, argnums=0)(params, x)
                # block until ready
                jax.block_until_ready(approx_jac)
            end_t = time.perf_counter()
            
            mem_stats = device.memory_stats()
            if mem_stats is not None:
                peak_mem = mem_stats.get('peak_bytes_in_use', 0)
            else:
                peak_mem = 0
            
            latency = (end_t - start_t) / loops
            cossim = compute_cosine_similarity(exact_jac, approx_jac)
            
            all_latencies.append(latency)
            all_memories.append(peak_mem)
            all_cossims.append(cossim)
            
    # Winsorized mean of 20 runs
    lat_w = winsorized_mean(np.array(all_latencies))
    mem_w = winsorized_mean(np.array(all_memories))
    cossim_w = winsorized_mean(np.array(all_cossims))
    
    return lat_w, mem_w, cossim_w
