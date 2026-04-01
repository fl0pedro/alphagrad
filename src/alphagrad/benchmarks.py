import time
from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from alphagrad.memory_monitor import PeakMemoryMonitor
from alphagrad.transformer.models import SSMClassifier, TransformerClassifier


def benchmark_run(
    model_name,
    model_class,
    batch_size,
    seq_len,
    vocab_size,
    embd_dim,
    num_layers,
    num_heads,
    key,
):
    model_key, data_key = jr.split(key)

    kwargs = {}
    if model_name == "Transformer":
        kwargs["num_heads"] = num_heads

    model = model_class(
        vocab_size=vocab_size,
        embd_dim=embd_dim,
        hidden_dim=embd_dim,
        num_layers=num_layers,
        key=model_key,
        **kwargs,
    )

    x = jr.randint(data_key, (batch_size, seq_len), 0, vocab_size)
    y_target = jnp.zeros((batch_size, 2))
    y_target = y_target.at[:, 0].set(1.0)

    @eqx.filter_jit
    def forward(m, x):
        return jax.vmap(lambda x_i: m(x_i))(x)

    @eqx.filter_jit
    def backward(m, x, y):
        def loss_fn(model, x, y):
            pred = jax.vmap(lambda x_i: model(x_i))(x)
            return jnp.mean((pred - y) ** 2)

        return eqx.filter_grad(loss_fn)(m, x, y)

    start = time.time()
    _ = forward(model, x).block_until_ready()
    end = time.time()
    fwd_compile_time = end - start

    start = time.time()
    with PeakMemoryMonitor(interval=0.01) as mem:
        _ = forward(model, x).block_until_ready()
    end = time.time()
    fwd_run_time = end - start
    fwd_peak_mem = mem.peak

    start = time.time()
    jax.tree_util.tree_map(
        lambda l: l.block_until_ready(), backward(model, x, y_target)
    )
    end = time.time()
    bwd_compile_time = end - start

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
    header = [
        "Model",
        "Batch",
        "SeqLen",
        "Vocab",
        "Dim",
        "Layers",
        "Heads",
        "FwdCompile(s)",
        "FwdRun(s)",
        "FwdMem(MB)",
        "BwdCompile(s)",
        "BwdRun(s)",
        "BwdMem(MB)",
    ]
    print("\t".join(header))

    key = jr.PRNGKey(42)

    batch_sizes = [16, 32, 64, 128]
    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    vocab_sizes = [128, 256]
    embed_dims = [64, 128, 256]
    layer_counts = [2, 4, 6]
    head_counts = [2, 4, 6, 8, 10, 12]

    for b, s, v, d, l, h in product(
        batch_sizes, seq_lens, vocab_sizes, embed_dims, layer_counts, head_counts
    ):
        key, subkey = jr.split(key)
        stats = benchmark_run(
            "Transformer", TransformerClassifier, b, s, v, d, l, h, subkey
        )
        row = [str(stats[col]) for col in header]
        print("\t".join(row))

        key, subkey = jr.split(key)
        stats = benchmark_run("SSM", SSMClassifier, b, s, v, d, l, 0, subkey)
        row = [str(stats[col]) for col in header]
        print("\t".join(row))


if __name__ == "__main__":
    main()
