import jax
import jax.numpy as jnp
import time
import numpy as np

def measure_compiled_function(func, params, inputs, labels, device_idx=0):
    """
    Measure memory and latency of the compiled function.
    """
    device = jax.devices()[device_idx]
    
    # Compile
    jitted_func = jax.jit(func, device=device)
    
    # Warmup
    _ = jitted_func(params, inputs, labels)
    
    # Clear memory stats
    device.clear_memory_stats()
    
    # Measure latency
    start = time.perf_counter()
    loss, grads = jitted_func(params, inputs, labels)
    # block until done
    jax.block_until_ready(grads)
    latency = time.perf_counter() - start
    
    # Measure memory
    stats = device.memory_stats()
    peak_mem = stats.get('peak_bytes_in_use', 0)
    
    return loss, grads, latency, peak_mem

def popart_normalize(metrics_dict):
    """
    Simple PopArt normalization for different metrics to sum them into a single reward.
    """
    normalized = {}
    for k, v in metrics_dict.items():
        # In a real PopArt, we maintain running mean/std. Here we just mock it for minimalism.
        mean = jnp.mean(v)
        std = jnp.std(v) + 1e-5
        normalized[k] = (v - mean) / std
    
    reward = sum(normalized.values())
    return reward

class ApproxEnv:
    def __init__(self, num_samples=5, num_runs=4):
        self.num_samples = num_samples
        self.num_runs = num_runs
        
    def step(self, action):
        """
        Executes one step in the environment.
        For minimalism, this is a mock representation of the environment step.
        """
        pass
        
    def evaluate_policy_trajectory(self, policy, func, dataloader):
        """
        Evaluates the chosen path approximations over the function.
        Returns the reward and metrics.
        """
        metrics = {'latency': [], 'peak_mem': [], 'cossim': [], 'flops': [], 'bytes_accessed': []}
        
        # Load 5 different samples
        samples = next(iter(dataloader))
        inputs, labels = samples['inputs'][:self.num_samples], samples['labels'][:self.num_samples]
        
        for _ in range(self.num_runs):
            # Randomly reinitialize weights
            params = {"w": jax.random.normal(jax.random.PRNGKey(np.random.randint(0, 1000)), (10, 10))}
            
            # Apply approximations derived from policy
            # ... (mocked) ...
            
            loss, grads, latency, peak_mem = measure_compiled_function(func, params, inputs, labels)
            
            metrics['latency'].append(latency)
            metrics['peak_mem'].append(peak_mem)
            metrics['cossim'].append(1.0) # mock
            metrics['flops'].append(1000) # mock
            metrics['bytes_accessed'].append(1000) # mock
            
        # Winsorized mean over the runs (mocking by just taking mean for minimalism)
        aggregated_metrics = {k: jnp.mean(jnp.array(v)) for k, v in metrics.items()}
        
        # PopArt reward
        reward = popart_normalize(aggregated_metrics)
        return reward, aggregated_metrics
