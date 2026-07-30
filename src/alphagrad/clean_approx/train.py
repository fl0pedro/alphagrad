import jax
import jax.numpy as jnp
import time

from alphagrad.clean_approx.env import ApproxEnv
from alphagrad.clean_approx.models import NNPolicy
from alphagrad.clean_approx.agents import PPOAgent
from alphagrad.clean_approx.reward import PopArt, evaluate_metrics

def main():
    print("Initializing Clean Approx RL Pipeline...")
    key = jax.random.PRNGKey(0)
    
    # 1. Environment Setup
    num_vertices = 10
    env = ApproxEnv(num_vertices=num_vertices, num_dynamic_steps=1)
    
    # 2. Model Setup
    vocab_size = 100
    embed_dim = 64
    model = NNPolicy(vocab_size=vocab_size, embed_dim=embed_dim, num_vertices=num_vertices)
    
    dummy_state = env.reset(key)
    # Initialize model weights with a batch dimension
    k1, key = jax.random.split(key)
    
    batched_tokens = jnp.expand_dims(dummy_state.tokens, 0)
    batched_avail = jnp.expand_dims(dummy_state.available_vertices, 0)
    batched_step = jnp.expand_dims(dummy_state.step_type, 0)
    
    params = model.init(k1, batched_tokens, batched_avail, batched_step)
    
    # 3. Agent Setup
    agent = PPOAgent(model_apply_fn=model.apply, learning_rate=3e-4)
    opt_state = None # opt_state is None for simple SGD
    
    # 4. PopArt Normalizer Setup
    # Metrics: Latency, Peak Memory, Cosine Similarity
    popart = PopArt(num_metrics=3, beta=0.01)
    
    # Dummy Target Functions for Evaluation
    def dummy_approx_fn(p, x):
        return jnp.sum(p @ x)
        
    def dummy_exact_fn(p, x):
        return jnp.sum(p @ x)
        
    # 5. Training Loop (Minimal)
    num_iterations = 10
    
    print("Starting Training Loop...")
    for i in range(num_iterations):
        start_t = time.time()
        
        # Rollout Phase (Mocked)
        # We would collect a batch of transitions from env interactions.
        
        # Reward Evaluation (Simulated Terminal Step)
        k_eval, key = jax.random.split(key)
        lat, mem, cossim = evaluate_metrics(
            dummy_approx_fn, dummy_exact_fn, k_eval, 
            num_samples=5, runs_per_sample=4, loops=50
        )
        
        raw_metrics = jnp.array([lat, mem, cossim], dtype=jnp.float32)
        
        # PopArt Normalization
        popart.update(raw_metrics)
        normalized_reward = popart.normalize(raw_metrics)
        total_scalar_reward = jnp.sum(normalized_reward)
        
        # Mock Batch
        batch_size = 32
        mock_batch = (
            jnp.zeros((batch_size, 1024), dtype=jnp.int32), # obs tokens
            jnp.zeros((batch_size,), dtype=jnp.int32),      # actions
            jnp.zeros((batch_size,)),                       # log_probs_old
            jnp.full((batch_size,), total_scalar_reward),   # returns
            jnp.ones((batch_size,))                         # advantages
        )
        
        # Update Agent
        params, opt_state, loss = agent.update(params, opt_state, mock_batch)
        
        duration = time.time() - start_t
        print(f"Iteration {i+1}/{num_iterations} | Loss: {loss:.4f} | "
              f"Lat: {lat:.6f}, Mem: {mem:.0f}, CosSim: {cossim:.4f} | "
              f"Reward: {total_scalar_reward:.4f} | Time: {duration:.2f}s")
              
    print("Training Complete.")

if __name__ == "__main__":
    main()
