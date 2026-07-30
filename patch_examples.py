import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/common/examples.py")
s = p.read_text()
if "_lif_shd_args" in s:
    print("examples ALREADY PATCHED"); sys.exit(0)
# 1) args builder — insert right before _BASIC_ARGS
builder = '''def _lif_shd_args():
    """SHD-shaped temporal LIF args: n_in=700 channels, hidden=128, n_out=20
    classes, T=100 timesteps. S_in_seq is a sparse Poisson spike train (SHD is
    an event/spike dataset; values are structurally identical to real SHD for
    the elimination-order search — real-data swap is a loader change). Weights
    W1/W2/W3 at argnums 8,9,10."""
    n_in, h, n_out, T = 700, 128, 20, 100
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    S_in_seq = jax.random.bernoulli(k[0], 0.05, (T, n_in)).astype(jnp.float32)
    S_target = jax.nn.one_hot(jax.random.randint(k[1], (), 0, n_out), n_out).astype(jnp.float32)
    U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
    I1 = jnp.zeros((h,)); I2 = jnp.zeros((h,)); I3 = jnp.zeros((n_out,))
    W1 = jax.random.normal(k[2], (h, n_in)) * 0.1
    W2 = jax.random.normal(k[3], (h, h)) * 0.1
    W3 = jax.random.normal(k[4], (n_out, h)) * 0.1
    alpha = jnp.array(0.9); beta = jnp.array(0.8); thresh = jnp.array(1.0)
    return (S_in_seq, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh)


'''
s = s.replace("_BASIC_ARGS = {", builder + "_BASIC_ARGS = {", 1)
# 2) registry entry
s = s.replace('    "LIF_SNN": _lif_snn_args(),',
              '    "LIF_SNN": _lif_snn_args(),\n    "LIF_SNN_SHD": _lif_shd_args(),', 1)
# 3) infer_argnums: SNN weights at 8,9,10
s = s.replace('def infer_argnums(fn_str: str) -> tuple[int, ...]:\n    """Default `argnums` (which input slots are differentiated through) per example name."""\n',
              'def infer_argnums(fn_str: str) -> tuple[int, ...]:\n    """Default `argnums` (which input slots are differentiated through) per example name."""\n    if fn_str in ("LIF_SNN", "LIF_SNN_SHD"):\n        return (8, 9, 10)\n', 1)
p.write_text(s)
print("examples.py: _lif_shd_args + registry + infer_argnums(8,9,10)")
