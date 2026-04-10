import jax
import jax.numpy as jnp
import jax.random as jrand
from graphax import jacve

hidden_dim = 32
key = jrand.PRNGKey(42)
keys = jrand.split(key, 3)
key = keys[0]

params = (
    jrand.normal(keys[1], (hidden_dim, 4)) * jnp.sqrt(2.0 / 4),
    jnp.zeros(hidden_dim),
    jrand.normal(keys[2], (4, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim),
    jnp.zeros(4)
)

def _neural_network(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(W1 @ x + b1)
    return 0.5 * (jnp.tanh(W2 @ a1 + b2) - y) ** 2

@jax.jit
def fn(keys):
    r1 = jrand.uniform(keys[0])
    th1 = jrand.uniform(keys[1], minval=-jnp.pi, maxval=jnp.pi)
    r2 = jrand.uniform(keys[2])
    th2 = jrand.uniform(keys[3], minval=-jnp.pi, maxval=jnp.pi)

    x = jnp.stack([r1, th1 / jnp.pi, r2, th2 / jnp.pi], axis=-1)

    y = jnp.stack(
        [
            r1 * jnp.cos(th1),
            r1 * jnp.sin(th1),
            r2 * jnp.cos(th2),
            r2 * jnp.sin(th2),
        ],
        axis=-1,
    )

    y += 0.05 * jrand.normal(keys[4], y.shape)

    return x, y

def get_sparsity_map(fun, args, mapping_list):
    jaxpr = jax.make_jaxpr(fun)(*args).jaxpr
    sp_type_to_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    
    sparsity_map = []
    for v, sp in mapping_list:
        if sp > 0 and sp in sp_type_to_map:
            eqn = jaxpr.eqns[v - 1]
            if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
                continue
            out_len = len(eqn.outvars[0].aval.shape)
            idx1 = sp_type_to_map[sp][0]
            idx2 = out_len + sp_type_to_map[sp][1]
            rule = ((idx1, idx2, -1),)
            sparsity_map.append((v, rule))
    return sparsity_map

hidden_dim = 32
master_key = jrand.PRNGKey(42)

def init_params(key):
    k1, k2 = jrand.split(key, 2)
    return (
        jrand.normal(k1, (hidden_dim, 4)) * jnp.sqrt(2.0 / 4),
        jnp.zeros(hidden_dim),
        jrand.normal(k2, (4, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim),
        jnp.zeros(4)
    )

dummy_args = (jnp.zeros(4), jnp.zeros(4), *init_params(master_key))
custom_map_list = [(8, 1), (7, 1), (6, 1), (5, 1), (3, 1), (4, 1), (2, 1), (1, 2)]
custom_smap = get_sparsity_map(_neural_network, dummy_args, custom_map_list)
custom_order = [v for v, sp in custom_map_list]

configs = {
    "fwd": {"order": "fwd", "sparsity_map": None},
    "rev": {"order": "rev", "sparsity_map": None},
    "mapped": {"order": custom_order, "sparsity_map": custom_smap}
}

epochs = 1000
results = []

for name, kwargs in configs.items():
    jac_fn = jacve(_neural_network, argnums=(2, 3, 4, 5), **kwargs)
    
    @jax.jit
    def step(params, x, y, lr=0.01):
        out_vals = _neural_network(x, y, *params)
        loss = jnp.sum(out_vals)
        
        J = jac_fn(x, y, *params)
        
        J_unwrapped = J[0] if isinstance(J, list) else J 
        
        ones = jnp.ones_like(y)
        grads = jax.tree.map(lambda j_i: jnp.tensordot(ones, j_i, axes=1), J_unwrapped)
        
        updated_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        
        W1, b1, W2, b2 = updated_params
        pred = jnp.tanh(W2 @ jnp.tanh(W1 @ x + b1) + b2)
        acc = jnp.array(jnp.linalg.norm(pred - y) < 0.25, dtype=jnp.float32)
        
        return updated_params, loss, acc

    params = init_params(master_key)
    key = master_key
    csv_lines = [f"Epoch,Loss,Accuracy ({name})"]
    
    for ep in range(epochs):
        keys = jrand.split(key, 6)
        key = keys[0]
        x_single, y_single = fn(keys[1:])
        
        params, loss, acc = step(params, x_single, y_single)
        
        # if ep % 100 == 0 or ep == epochs - 1:
        csv_lines.append(f"{ep},{loss:.4f},{acc:.4f}")
            
    # Generate a batch of 100 test samples
    test_keys = jrand.split(key, 500).reshape(100, 5, 2)
    x_test, y_test = jax.vmap(fn)(test_keys)

    # Batched inference
    W1, b1, W2, b2 = params
    pred_test = jax.vmap(lambda x: jnp.tanh(W2 @ jnp.tanh(W1 @ x + b1) + b2))(x_test)

    test_acc = jnp.mean(jnp.linalg.norm(pred_test - y_test, axis=-1) < 0.25)
    test_loss = jnp.mean(jnp.sum(0.5 * (pred_test - y_test) ** 2, axis=-1))

    csv_lines.append(f"Test,{test_loss:.4f},{test_acc:.4f}\n")
    results.append("\n".join(csv_lines))

print("\n".join(results))