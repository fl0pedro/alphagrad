import jax
import jax.numpy as jnp
from graphax.core import extract_jaxpr
from alphagrad.approx.env import VertexEliminationEnv, EnvState as State
from alphagrad.approx.policy import Policy
from graphax.examples.models.transformer import Transformer

def test_random_rollout_transformer():
    print("Initializing Transformer model...")
    rng = jax.random.PRNGKey(42)
    B, T, D = 2, 8, 32
    x = jnp.ones((B, T, D))
    model = Transformer(num_layers=1, num_heads=2, qkv_dim=16, mlp_dim=64, out_dim=D)
    
    # Init params
    variables = model.init(rng, x)
    
    def loss_fn(params, inputs):
        out = model.apply(params, inputs)
        return jnp.sum(out ** 2)

    print("Extracting JAXPR...")
    closed_jaxpr = extract_jaxpr(loss_fn, variables, x)
    
    print("Initializing ApproxEnv...")
    env = VertexEliminationEnv(closed_jaxpr.jaxpr)
    
    print("Setting up JIT step...")
    policy = Policy(hidden_dim=64)
    policy_params = policy.init(rng, env.initial_state())
    
    @jax.jit
    def step_random(state, key):
        # We sample random actions from the policy, which might include Compress and Diag
        # with mismatched shapes.
        actions, log_probs, values = policy.apply(policy_params, state, key)
        next_state, reward, done = env.step(state, actions)
        return next_state, reward, done

    state = env.initial_state()
    print("Starting rollout...")
    for i in range(10):
        rng, step_key = jax.random.split(rng)
        state, reward, done = step_random(state, step_key)
        print(f"Step {i}, Reward: {reward}, Done: {done}")
        if done:
            break
            
    print("Rollout complete without crashing!")

if __name__ == "__main__":
    test_random_rollout_transformer()
