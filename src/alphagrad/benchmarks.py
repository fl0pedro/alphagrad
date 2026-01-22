import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from alphagrad.transformer.models import TransformerClassifier, SSMClassifier
from alphagrad.memory_monitor import PeakMemoryMonitor

def benchmark_run(model_name, model_class, batch_size, seq_len, vocab_size, embd_dim, num_layers, num_heads, key):
    # Initialize model
    model_key, data_key = jr.split(key)
    
    # Handle Transformer specialized args
    kwargs = {}
    if model_name == "Transformer":
        kwargs["num_heads"] = num_heads
    
    model = model_class(
        vocab_size=vocab_size,
        embd_dim=embd_dim,
        hidden_dim=embd_dim,
        num_layers=num_layers,
        key=model_key,
        **kwargs
    )

    # Make dummy data
    # (batch_size, seq_len)
    x = jr.randint(data_key, (batch_size, seq_len), 0, vocab_size)
    # Target for dummy loss (batch_size, 2) - let's say 2 classes
    y_target = jnp.zeros((batch_size, 2))
    y_target = y_target.at[:, 0].set(1.0) 

    @eqx.filter_jit
    def forward(m, x):
        # Map over batch dimension of x, reusing m (broadcasting model)
        return jax.vmap(lambda x_i: m(x_i))(x)

    @eqx.filter_jit
    def backward(m, x, y):
        def loss_fn(model, x, y):
            # Map over batch dimension
            pred = jax.vmap(lambda x_i: model(x_i))(x)
            return jnp.mean((pred - y) ** 2)
        return eqx.filter_grad(loss_fn)(m, x, y)

    # --- Forward Benchmark ---
    # Compile
    start = time.time()
    _ = forward(model, x).block_until_ready()
    end = time.time()
    fwd_compile_time = end - start

    # Run
    # Warmup? The compile run effectively warms it up, but let's run clean for timing
    start = time.time()
    with PeakMemoryMonitor(interval=0.01) as mem:
        _ = forward(model, x).block_until_ready()
    end = time.time()
    fwd_run_time = end - start
    fwd_peak_mem = mem.peak

    # --- Backward Benchmark ---
    # Compile
    start = time.time()
    jax.tree_util.tree_map(lambda l: l.block_until_ready(), backward(model, x, y_target))
    end = time.time()
    bwd_compile_time = end - start

    # Run
    start = time.time()
    with PeakMemoryMonitor(interval=0.01) as mem:
        grads = backward(model, x, y_target)
        jax.tree_util.tree_map(lambda l: l.block_until_ready(), grads)
    end = time.time()
    bwd_run_time = end - start
    bwd_peak_mem = mem.peak
    
    return {
        "Model": model_name,
        "Batch": batch_size,
        "SeqLen": seq_len,
        "Vocab": vocab_size,
        "Dim": embd_dim,
        "Layers": num_layers,
        "Heads": num_heads if model_name == "Transformer" else "N/A",
        "FwdCompile(s)": f"{fwd_compile_time:.4f}",
        "FwdRun(s)": f"{fwd_run_time:.4f}",
        "FwdMem(MB)": f"{fwd_peak_mem / 1e6:.2f}",
        "BwdCompile(s)": f"{bwd_compile_time:.4f}",
        "BwdRun(s)": f"{bwd_run_time:.4f}",
        "BwdMem(MB)": f"{bwd_peak_mem / 1e6:.2f}",
    }

def main():
    # Print Header
    header = ["Model", "Batch", "SeqLen", "Vocab", "Dim", "Layers", "Heads", 
              "FwdCompile(s)", "FwdRun(s)", "FwdMem(MB)", 
              "BwdCompile(s)", "BwdRun(s)", "BwdMem(MB)"]
    print("\t".join(header))

    key = jr.PRNGKey(42)
    
    batch_sizes = [8,16,32,64,128]
    seq_lens = [526, 1024, 2048, 4096, 8192, 16384] 
    vocab_sizes = [128, 256]
    embd_dims = [128, 256] 
    layer_counts = [2,4,6]
    head_counts = [2,4,6]

    configs = []
    
    # Transformers
    for b in batch_sizes:
        for s in seq_lens:
            for v in vocab_sizes:
                for d in embd_dims:
                    for l in layer_counts:
                        for h in head_counts:
                            key, subkey = jr.split(key)
                            stats = benchmark_run("Transformer", TransformerClassifier, b, s, v, d, l, h, subkey)
                            row = [str(stats[col]) for col in header]
                            print("\t".join(row))

    # SSMs
    for b in batch_sizes:
        for s in seq_lens:
            for v in vocab_sizes:
                for d in embd_dims:
                    for l in layer_counts:
                         # SSM doesn't use heads
                        key, subkey = jr.split(key)
                        stats = benchmark_run("SSM", SSMClassifier, b, s, v, d, l, 0, subkey)
                        row = [str(stats[col]) for col in header]
                        print("\t".join(row))

if __name__ == "__main__":
    main()
