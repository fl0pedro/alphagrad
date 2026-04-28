import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import pandas as pd
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
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2

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

hidden_dim = 640
batch_size = 16
master_key = jrand.PRNGKey(42)

def init_params(key):
    k1, k2 = jrand.split(key, 2)
    return (
        jrand.normal(k1, (hidden_dim, 4)) * jnp.sqrt(2.0 / 4),
        jnp.zeros(hidden_dim),
        jrand.normal(k2, (4, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim),
        jnp.zeros(4)
    )

v_neural_network = jax.vmap(_neural_network, in_axes=(0, 0, None, None, None, None))
dummy_args = (jnp.zeros((batch_size, 4)), jnp.zeros((batch_size, 4)), *init_params(master_key))

sequences = {
    0.2875: [(12, 0), (9, 4), (11, 3), (3, 0), (6, 1), (8, 0), (1, 1), (10, 0), (5, 4), (2, 1), (7, 4), (4, 0)],
    0.289: [(8, 0), (1, 1), (6, 1), (11, 4), (10, 0), (12, 4), (3, 3), (9, 4), (2, 4), (7, 4), (5, 3), (4, 4)],
    0.2914: [(2, 0), (7, 4), (10, 1), (9, 1), (11, 2), (12, 2), (3, 3), (6, 3), (1, 1), (4, 0), (8, 3), (5, 3)],
    0.2922: [(3, 1), (6, 1), (4, 1), (12, 0), (7, 4), (8, 0), (11, 4), (10, 0), (2, 1), (1, 1), (5, 3), (9, 3)],
    0.4852: [(7, 4), (5, 1), (4, 3), (1, 1), (3, 3), (8, 1), (10, 4), (12, 1), (6, 3), (2, 0), (11, 3), (9, 3)],
    0.7219: [(11, 3), (12, 4), (10, 3), (6, 2), (9, 0), (8, 3), (7, 0), (4, 0), (3, 3), (1, 0), (5, 0), (2, 4)],
    0.7257: [(8, 3), (3, 1), (6, 3), (12, 2), (10, 4), (9, 2), (11, 1), (1, 1), (2, 1), (4, 3), (7, 3), (5, 3)],
    0.7592: [(11, 3), (12, 3), (8, 0), (6, 4), (1, 2), (9, 4), (10, 0), (3, 1), (7, 2), (4, 0), (2, 4), (5, 1)],
    0.7598: [(11, 1), (12, 3), (1, 1), (8, 0), (6, 2), (10, 3), (9, 3), (3, 0), (4, 1), (7, 3), (5, 0), (2, 4)],
    0.7795: [(3, 0), (10, 0), (2, 2), (4, 2), (6, 2), (8, 1), (11, 3), (9, 1), (12, 1), (5, 0), (1, 2), (7, 0)],
    0.8014: [(6, 3), (12, 4), (5, 3), (1, 4), (7, 0), (2, 0), (11, 1), (10, 1), (9, 1), (3, 0), (4, 0), (8, 0)],
    0.8844: [(8, 0), (6, 2), (9, 2), (1, 2), (11, 2), (10, 4), (12, 1), (4, 1), (2, 3), (5, 3), (3, 0), (7, 0)],
    1.0: [(6, 2), (3, 0), (7, 2), (8, 1), (2, 3), (1, 0), (10, 0), (5, 3), (11, 2), (12, 2), (4, 0), (9, 0)]
}

configs = {
    "fwd": {"order": "fwd", "sparsity_map": None},
    "rev": {"order": "rev", "sparsity_map": None},
    "jax_fwd": "jax_fwd",
    "jax_rev": "jax_rev",
}

for k, seq in sequences.items():
    smap = get_sparsity_map(v_neural_network, dummy_args, seq)
    order = [v for v, _ in seq]
    configs[f"{k}"] = {"order": order, "sparsity_map": smap}

import time

epochs = 10000
time_limit_s = 10
lr_schedule = optax.exponential_decay(init_value=0.01, transition_steps=1000, decay_rate=0.9)
optimizer = optax.adam(learning_rate=lr_schedule)

results = []

val_key = jrand.PRNGKey(123)
val_keys = jrand.split(val_key, 500).reshape(100, 5, 2)
x_val, y_val = jax.vmap(fn)(val_keys)

for name, kwargs in configs.items():
    if name == "jax_fwd":
        jac_fn = jax.jacfwd(v_neural_network, argnums=(2, 3, 4, 5))
    elif name == "jax_rev":
        jac_fn = jax.jacrev(v_neural_network, argnums=(2, 3, 4, 5))
    else:
        jac_fn = jacve(v_neural_network, argnums=(2, 3, 4, 5), **kwargs)
    
    # Cost analysis
    lowered = jax.jit(jac_fn).lower(*dummy_args)
    compiled = lowered.compile()
    cost = compiled.cost_analysis()
    estimated_flops = cost.get("flops", 0)
    estimated_mem = cost.get("bytes accessed", 0)
    print(f"Config {name}: Estimated FLOPs {estimated_flops}, Memory {estimated_mem} bytes")
    
    @jax.jit
    def step(params, opt_state, x, y):
        out_vals = v_neural_network(x, y, *params)
        loss = jnp.sum(out_vals)
        
        J = jac_fn(x, y, *params)
        J_unwrapped = J[0] if isinstance(J, list) else J 
        
        ones = jnp.ones_like(y)
        raw_grads = jax.tree.map(lambda j_i: jnp.tensordot(ones, j_i, axes=([0, 1], [0, 1])), J_unwrapped)
        
        if name.startswith("jax_"):
            grads = tuple(jnp.reshape(raw_grads[i], params[i].shape) for i in range(4))
        else:
            grads = (
                jnp.reshape(raw_grads[0], params[0].shape[::-1]).T,
                jnp.reshape(raw_grads[1], params[1].shape),
                jnp.reshape(raw_grads[2], params[2].shape[::-1]).T,
                jnp.reshape(raw_grads[3], params[3].shape)
            )
        
        # Exact gradients for analysis
        def loss_fn(p):
            return jnp.sum(v_neural_network(x, y, *p))
        exact_grads = jax.grad(loss_fn)(params)
        
        # Per-variable comparison metrics
        param_names = ["W1", "b1", "W2", "b2"]
        per_var_metrics = {}
        for i, (g_approx, g_exact, p_name) in enumerate(zip(grads, exact_grads, param_names)):
            diff = g_approx - g_exact
            mse = jnp.mean(diff**2)
            abs_sq_err = jnp.sum(diff**2)
            fro_norm = jnp.linalg.norm(diff)
            
            norm_approx = jnp.linalg.norm(g_approx)
            norm_exact = jnp.linalg.norm(g_exact)
            cos_sim = jnp.dot(g_approx.ravel(), g_exact.ravel()) / (norm_approx * norm_exact + 1e-8)
            
            per_var_metrics[f"MSE_{p_name}"] = mse
            per_var_metrics[f"AbsSqErr_{p_name}"] = abs_sq_err
            per_var_metrics[f"Frobenius_{p_name}"] = fro_norm
            per_var_metrics[f"CosSim_{p_name}"] = cos_sim
        
        # Overall comparison metrics
        flat_approx = jnp.concatenate([g.ravel() for g in grads])
        flat_exact = jnp.concatenate([g.ravel() for g in exact_grads])
        
        diff = flat_approx - flat_exact
        mse = jnp.mean(diff**2)
        abs_sq_err = jnp.sum(diff**2)
        fro_norm = jnp.linalg.norm(diff)
        
        norm_approx = jnp.linalg.norm(flat_approx)
        norm_exact = jnp.linalg.norm(flat_exact)
        cos_sim = jnp.dot(flat_approx, flat_exact) / (norm_approx * norm_exact + 1e-8)
        
        # Accumulation methods
        agg_metrics = {}
        for m_name in ["MSE", "AbsSqErr", "Frobenius", "CosSim"]:
            m_vals = jnp.array([per_var_metrics[f"{m_name}_{p_name}"] for p_name in param_names])
            
            # 1. Unweighted Average
            agg_metrics[f"Unweighted_{m_name}"] = jnp.mean(m_vals)
            
            # 2. Weighted (True Grad Norm)
            g_norms = jnp.array([jnp.linalg.norm(g) for g in exact_grads])
            agg_metrics[f"WeightedNorm_{m_name}"] = jnp.sum(m_vals * g_norms) / (jnp.sum(g_norms) + 1e-8)
            
            # 3. Weighted (Volume)
            vols = jnp.array([g.size for g in exact_grads])
            agg_metrics[f"WeightedVol_{m_name}"] = jnp.sum(m_vals * vols) / (jnp.sum(vols) + 1e-8)

        metrics = {
            "Flattened_MSE": mse,
            "Flattened_AbsSqErr": abs_sq_err,
            "Flattened_Frobenius": fro_norm,
            "Flattened_CosSim": cos_sim,
            **per_var_metrics,
            **agg_metrics
        }
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        W1, b1, W2, b2 = updated_params
        pred = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x)
        acc = jnp.mean(jnp.linalg.norm(pred - y, axis=-1) < 0.25)
        
        return updated_params, opt_state, loss, acc, metrics
    
    @jax.jit
    def efficient_step(params, opt_state, x, y):
        out_vals = v_neural_network(x, y, *params)
        loss = jnp.sum(out_vals)
        
        J = jac_fn(x, y, *params)
        J_unwrapped = J[0] if isinstance(J, list) else J 
        
        ones = jnp.ones_like(y)
        raw_grads = jax.tree.map(lambda j_i: jnp.tensordot(ones, j_i, axes=([0, 1], [0, 1])), J_unwrapped)
        
        if name.startswith("jax_"):
            grads = tuple(jnp.reshape(raw_grads[i], params[i].shape) for i in range(4))
        else:
            grads = (
                jnp.reshape(raw_grads[0], params[0].shape[::-1]).T,
                jnp.reshape(raw_grads[1], params[1].shape),
                jnp.reshape(raw_grads[2], params[2].shape[::-1]).T,
                jnp.reshape(raw_grads[3], params[3].shape)
            )
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        
        W1, b1, W2, b2 = updated_params
        pred = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x)
        acc = jnp.mean(jnp.linalg.norm(pred - y, axis=-1) < 0.25)
        
        return updated_params, opt_state, loss, acc

    for train_mode in ["epochs", "time", "efficient"]:
        print(f"Training Config {name} in {train_mode} mode...")
        params = init_params(master_key)
        opt_state = optimizer.init(params)
        key = master_key
        total_latency_ms = 0.0
        
        # warmup
        warmup_keys = jrand.split(key, batch_size * 5 + 1)
        x_w, y_w = jax.vmap(fn)(warmup_keys[1:].reshape(batch_size, 5, 2))
        params, opt_state, _, _, _ = step(params, opt_state, x_w, y_w)
        params, opt_state, _, _ = efficient_step(params, opt_state, x_w, y_w)
        jax.block_until_ready(params)

        ep = 0
        while not train_mode == "efficient":
            if train_mode == "epochs" and ep >= epochs:
                break
            if train_mode == "time" and total_latency_ms >= time_limit_s * 1000:
                break
                
            keys = jrand.split(key, batch_size * 5 + 1)
            key = keys[0]
            batch_keys = keys[1:].reshape(batch_size, 5, 2)
            x_batch, y_batch = jax.vmap(fn)(batch_keys)
            
            start_time = time.perf_counter()
            if train_mode == "epochs":
                params, opt_state, loss, acc, metrics = step(params, opt_state, x_batch, y_batch)
            else:
                params, opt_state, loss, acc = efficient_step(params, opt_state, x_batch, y_batch)
            jax.block_until_ready(params)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            total_latency_ms += latency_ms
            
            if ep % 100 == 0:
                W1, b1, W2, b2 = params
                pred_val = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x_val)
                val_acc = jnp.mean(jnp.linalg.norm(pred_val - y_val, axis=-1) < 0.25)
                val_loss = jnp.mean(jnp.sum(0.5 * (pred_val - y_val) ** 2, axis=-1))
                
                results_dict = {
                    "Config": name,
                    "Mode": train_mode,
                    "Epoch": ep,
                    "Loss": float(loss),
                    "Accuracy": float(acc),
                    "Latency": float(latency_ms),
                    "TotalLatency": float(total_latency_ms),
                    "ValLoss": float(val_loss),
                    "ValAcc": float(val_acc),
                }
                if train_mode == "epochs":
                    for k, v in metrics.items():
                        results_dict[k] = float(v)
                results.append(results_dict)
            ep += 1
        
        if train_mode == "efficient":
            large_run_size = 100_000
            chunk_size = 1000
            num_chunks = large_run_size // chunk_size
            
            all_keys = jrand.split(key, large_run_size * batch_size * 5 + 1)
            key = all_keys[0]
            batch_keys = all_keys[1:].reshape(large_run_size, batch_size, 5, 2)
            x_train_all, y_train_all = jax.vmap(jax.vmap(jax.vmap(fn)))(batch_keys.reshape(num_chunks, chunk_size, batch_size, 5, 2))
            
            @jax.jit
            def train_chunk(params, opt_state, x_chunk, y_chunk):
                def body(carry, i):
                    p, s = carry
                    p, s, l, a = efficient_step(p, s, x_chunk[i], y_chunk[i])
                    return (p, s), (l, a)
                (params, opt_state), (losses, accs) = jax.lax.scan(body, (params, opt_state), jnp.arange(chunk_size))
                return params, opt_state, jnp.mean(losses), jnp.mean(accs)

            for c in range(num_chunks):
                start_time = time.perf_counter()
                params, opt_state, loss, acc = train_chunk(params, opt_state, x_train_all[c], y_train_all[c])
                jax.block_until_ready(params)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                total_latency_ms += latency_ms
                
                W1, b1, W2, b2 = params
                pred_val = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x_val)
                val_acc = jnp.mean(jnp.linalg.norm(pred_val - y_val, axis=-1) < 0.25)
                val_loss = jnp.mean(jnp.sum(0.5 * (pred_val - y_val) ** 2, axis=-1))
                
                results_dict = {
                    "Config": name,
                    "Mode": train_mode,
                    "Epoch": ep,
                    "Loss": float(loss),
                    "Accuracy": float(acc),
                    "Latency": float(latency_ms),
                    "TotalLatency": float(total_latency_ms),
                    "ValLoss": float(val_loss),
                    "ValAcc": float(val_acc),
                }
                results.append(results_dict)


        W1, b1, W2, b2 = params
        test_keys = jrand.split(key, 500).reshape(100, 5, 2)
        x_test, y_test = jax.vmap(fn)(test_keys)
        pred_test = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x_test)
        test_acc = jnp.mean(jnp.linalg.norm(pred_test - y_test, axis=-1) < 0.25)
        test_loss = jnp.mean(jnp.sum(0.5 * (pred_test - y_test) ** 2, axis=-1))
        print(f"Config {name} ({train_mode}): Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}")
        pd.DataFrame(results).to_csv("nn_results_vmapped.csv", index=False)

import pandas as pd
df = pd.DataFrame(results)
df.to_csv("nn_results_vmapped.csv", index=False)
print("Results saved to nn_results_vmapped.csv")