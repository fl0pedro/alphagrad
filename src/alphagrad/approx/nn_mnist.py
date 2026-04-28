import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import pandas as pd
from graphax import jacve
import time
from functools import partial
import os
import urllib.request
import gzip
import numpy as np
import grain.python as grain
import psutil

def print_mem():
    process = psutil.Process(os.getpid())
    print(f"  [RAM Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB]", end="")

# --- MNIST Data Loading with Grain ---

class MNISTDataSource(grain.RandomAccessDataSource):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __getitem__(self, i):
        return {"image": self.x[i], "label": self.y[i]}
    def __len__(self):
        return len(self.x)

def load_mnist_raw(data_dir="./.data/mnist"):
    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return jnp.array(data.reshape(-1, 28 * 28).astype(np.float32) / 255.0)

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return jnp.array(data)

    try:
        train_x = read_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
        train_y = read_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
        test_x = read_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
        test_y = read_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
        return (train_x, train_y), (test_x, test_y)
    except Exception as e:
        print(f"Error reading raw MNIST from {data_dir}: {e}")
        return None

class OneHotTransform(grain.MapTransform):
    def map(self, element):
        element["label"] = jax.nn.one_hot(element["label"], 10)
        return element

def get_dataloader(x, y, batch_size, shuffle=True, seed=42):
    source = MNISTDataSource(x, y)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )
    
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[OneHotTransform(), grain.Batch(batch_size)],
        worker_count=0,
    )

# --- Model Definition ---
def _neural_network(x, y, W1, b1, W2, b2):
    """Two-layer MLP with tanh activation and MSE loss."""
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2

v_neural_network = jax.vmap(_neural_network, in_axes=(0, 0, None, None, None, None))

# --- Hyperparameters ---
hidden_dim = 512
batch_size = 16
master_key = jrand.PRNGKey(42)

def init_params(key):
    k1, k2 = jrand.split(key, 2)
    return (
        jrand.normal(k1, (hidden_dim, 784)) * jnp.sqrt(2.0 / 784),
        jnp.zeros(hidden_dim),
        jrand.normal(k2, (10, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim),
        jnp.zeros(10)
    )

# --- Sparsity / Graphax Setup ---
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

dummy_args = (jnp.zeros((batch_size, 784)), jnp.zeros((batch_size, 10)), *init_params(master_key))

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

# --- Training Config ---
epochs = 20
lr_schedule = optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.9)
optimizer = optax.adam(learning_rate=lr_schedule)

# --- Debug/Analysis Options ---
analysis_frequency = 100  # Set to 1 to analyze every batch, or higher to speed up

mnist_data = load_mnist_raw()
if mnist_data:
    (x_train, y_train), (x_test, y_test) = mnist_data
    y_test_all = jax.nn.one_hot(y_test, 10)
    x_test_all = x_test
else:
    print("Warning: Raw MNIST not found. Using random data for structural check.")
    key = jrand.PRNGKey(0)
    x_train = jrand.normal(key, (1000, 784))
    y_train = jrand.randint(key, (1000,), 0, 10)
    x_test = jrand.normal(key, (100, 784))
    y_test = jrand.randint(key, (1000,), 0, 10) # fixed typo in range
    x_test_all = x_test
    y_test_all = jax.nn.one_hot(y_test, 10)

results = []

try:
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
            
            loss = jnp.sum(v_neural_network(x, y, *params))
            updates, opt_state = optimizer.update(grads, opt_state, params)
            updated_params = optax.apply_updates(params, updates)
            
            W1, b1, W2, b2 = updated_params
            pred = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x)
            acc = jnp.mean(jnp.argmax(pred, axis=-1) == jnp.argmax(y, axis=-1))
            
            return updated_params, opt_state, loss, acc
    
        @jax.jit
        def compute_metrics(params, x, y):
            if name == "jax_fwd":
                jac_fn = jax.jacfwd(v_neural_network, argnums=(2, 3, 4, 5))
            elif name == "jax_rev":
                jac_fn = jax.jacrev(v_neural_network, argnums=(2, 3, 4, 5))
            else:
                jac_fn = jacve(v_neural_network, argnums=(2, 3, 4, 5), **kwargs)
    
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
            
            def loss_fn(p):
                return jnp.sum(v_neural_network(x, y, *p))
            exact_grads = jax.grad(loss_fn)(params)
            
            param_names = ["W1", "b1", "W2", "b2"]
            metrics = {}
            for i, (g_approx, g_exact, p_name) in enumerate(zip(grads, exact_grads, param_names)):
                cos_sim = jnp.dot(g_approx.ravel(), g_exact.ravel()) / (jnp.linalg.norm(g_approx) * jnp.linalg.norm(g_exact) + 1e-8)
                metrics[f"CosSim_{p_name}"] = cos_sim
            return metrics
        
        @jax.jit
        def efficient_step(params, opt_state, x, y):
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
            return updated_params, opt_state
    
        print(f"Training Config {name}...")
        params = init_params(master_key)
        opt_state = optimizer.init(params)
        key = master_key
        total_latency_ms = 0.0
        
        # Warmup
        warmup_loader = get_dataloader(x_train[:batch_size], y_train[:batch_size], batch_size, shuffle=False)
        warmup_batch = next(iter(warmup_loader))
        params, opt_state, _, _ = step(params, opt_state, warmup_batch["image"], warmup_batch["label"])
        jax.block_until_ready(params)
    
        for ep in range(epochs):
            dataloader = get_dataloader(x_train, y_train, batch_size, shuffle=True, seed=int(jrand.randint(key, (), 0, 1000)))
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            last_metrics = {}
            
            start_time = time.perf_counter()
            for batch in dataloader:
                x_batch, y_batch = batch["image"], batch["label"]
                params, opt_state, loss, acc = step(params, opt_state, x_batch, y_batch)
                
                if num_batches % analysis_frequency == 0:
                    print_mem()
                    last_metrics = compute_metrics(params, x_batch, y_batch)
                    print(f"  [Epoch {ep} Batch {num_batches}] Loss: {loss:.4f}, Acc: {acc:.4f}, CosSim_W1: {last_metrics.get('CosSim_W1', 0):.4f}")
                
                epoch_loss += loss
                epoch_acc += acc
                num_batches += 1
            
            jax.block_until_ready(params)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            total_latency_ms += latency_ms
            
            # Validation
            W1, b1, W2, b2 = params
            val_pred = jax.vmap(lambda x: jnp.tanh(jnp.tanh(x @ W1.T + b1) @ W2.T + b2))(x_test_all)
            val_acc = jnp.mean(jnp.argmax(val_pred, axis=-1) == jnp.argmax(y_test_all, axis=-1))
            
            print(f"Epoch {ep}: Loss {epoch_loss/num_batches:.4f}, Acc {epoch_acc/num_batches:.4f}, Val Acc {val_acc:.4f}, Time {latency_ms:.2f}ms")
            
            results.append({
                "Config": name,
                "Epoch": ep,
                "Loss": float(epoch_loss/num_batches),
                "Accuracy": float(epoch_acc/num_batches),
                "ValAcc": float(val_acc),
                "Latency": latency_ms,
                "TotalLatency": total_latency_ms,
                **{k: float(v) for k, v in last_metrics.items()}
            })
    
        pd.DataFrame(results).to_csv("nn_mnist_results.csv", index=False)
    
    # Clear caches and free memory between configs
    jax.clear_caches()
    print(f"Finished Config {name}. Cache cleared.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Results saved to nn_mnist_results.csv")
