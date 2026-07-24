import jax
import jax.numpy as jnp
import jax.lax as lax
import optax
import equinox as eqx

from alphagrad.approx.env import VertexEliminationEnv, StepAction
from alphagrad.approx.policy import ApproxAgent
from alphagrad.approx.heads import precompute_factor_tables
from graphax.sparse.micro_actions import verify_hardware_compat

def simple_mlp(params, x, y):
    h = jnp.dot(x, params['w1'])
    h = jnp.maximum(0.0, h)
    out = jnp.dot(h, params['w2'])
    return jnp.mean((out - y)**2), out

def get_returns(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    # GAE
    def loop_fn(gae_and_return, transition):
        r, v, nv, d = transition
        gae, ret = gae_and_return
        delta = r + gamma * nv * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        ret = gae + v
        return (gae, ret), (gae, ret)
    
    transitions = (rewards, values, jnp.append(values[1:], next_value), dones)
    _, (gaes, returns) = jax.lax.scan(loop_fn, (0.0, 0.0), transitions, reverse=True)
    return gaes, returns

def main():
    print("Verifying hardware quantization compatibility...")
    avail_mask, compat_matrix = verify_hardware_compat()
    print(f"Hardware compat check complete. Found {int(jnp.sum(avail_mask))} available quantization dtypes.")

    rng = jax.random.PRNGKey(42)
    
    # Create Jaxpr
    x = jnp.zeros((32, 64))
    y = jnp.zeros((32, 10))
    params = {'w1': jnp.zeros((64, 128)), 'w2': jnp.zeros((128, 10))}
    jaxpr = jax.make_jaxpr(simple_mlp)(params, x, y)
    flat_args, _ = jax.tree_util.tree_flatten((params, x, y))
    
    # Init Env
    env = VertexEliminationEnv.from_jaxpr(
        jaxpr=jaxpr,
        args=flat_args,
        num_envs=1
    )
    
    # Init Agent
    rng, agent_rng = jax.random.split(rng)
    tables = precompute_factor_tables(max_axis_size=256)
    max_vertices = len(env.valid_vertices)
    agent = ApproxAgent(embd_dim=64, num_heads=4, max_substeps=8, max_vertices=max_vertices, key=agent_rng)
    
    # Optax
    optim = optax.adam(learning_rate=3e-4)
    opt_state = optim.init(eqx.filter(agent, eqx.is_array))
    
    print("Environment and Agent initialized successfully!")

if __name__ == "__main__":
    main()
