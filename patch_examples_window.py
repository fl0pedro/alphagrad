import pathlib
p = pathlib.Path("src/alphagrad/approx/common/examples.py")
s = p.read_text()
a = s.index("def _lif_shd_args():")
b = s.index("_BASIC_ARGS = {")
new = '''def _lif_shd_args():
    """SHD-shaped temporal LIF args with REVERSE-mode truncation baked into the
    graph. n_in=700, hidden=128, n_out=20, T=100 Poisson spike train. The full
    T-step forward runs here (detached) to produce the recurrent carry entering
    the truncation window; only the last N steps (N from ALPHAGRAD_SNN_TRUNC:
    unset=T full BPTT, k>0=window k, 0=online=1) are returned as the differentiable
    window, so the elimination graph = constant base + N*per-step. Weights 8,9,10."""
    import os as _os
    from graphax.examples.neuromorphic import lif as _lif
    n_in, h, n_out, T = 700, 128, 20, 100
    k = jax.random.split(jax.random.PRNGKey(1), 8)
    full_seq = jax.random.bernoulli(k[0], 0.05, (T, n_in)).astype(jnp.float32)
    S_target = jax.nn.one_hot(jax.random.randint(k[1], (), 0, n_out), n_out).astype(jnp.float32)
    U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
    I1 = jnp.zeros((h,)); I2 = jnp.zeros((h,)); I3 = jnp.zeros((n_out,))
    W1 = jax.random.normal(k[2], (h, n_in)) * 0.1
    W2 = jax.random.normal(k[3], (h, h)) * 0.1
    W3 = jax.random.normal(k[4], (n_out, h)) * 0.1
    alpha = jnp.array(0.9); beta = jnp.array(0.8); thresh = jnp.array(1.0)
    v = _os.environ.get("ALPHAGRAD_SNN_TRUNC", None)
    if v is None or v == "":
        N = T
    elif int(v) <= 0:
        N = 1               # online
    else:
        N = min(int(v), T)
    for t in range(T - N):  # FULL detached pre-window forward (activations only)
        i1 = W1 @ full_seq[t]; U1, I1, s1 = _lif(U1, I1, i1, alpha, beta, thresh)
        i2 = W2 @ s1;          U2, I2, s2 = _lif(U2, I2, i2, alpha, beta, thresh)
        i3 = W3 @ s2;          U3, I3, s3 = _lif(U3, I3, i3, alpha, beta, thresh)
    sg = jax.lax.stop_gradient
    U1, U2, U3 = sg(U1), sg(U2), sg(U3)
    I1, I2, I3 = sg(I1), sg(I2), sg(I3)
    window = full_seq[T - N:]
    return (window, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh)


'''
s = s[:a] + new + s[b:]
p.write_text(s)
print("_lif_shd_args = full-forward precompute + N-step window")
