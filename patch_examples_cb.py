import pathlib
p = pathlib.Path("src/alphagrad/approx/common/examples.py")
s = p.read_text()
s = s.replace("from graphax.examples.neuromorphic import lif as _lif",
              "from graphax.examples.neuromorphic import lif_cb as _lif", 1)
# tuned spiking params: rate 0.1, gain 6/sqrt(fan), thresh 0.3
s = s.replace("full_seq = jax.random.bernoulli(k[0], 0.05, (T, n_in))",
              "full_seq = jax.random.bernoulli(k[0], 0.1, (T, n_in))", 1)
s = s.replace("W1 = jax.random.normal(k[2], (h, n_in)) * 0.1",
              "W1 = jax.random.normal(k[2], (h, n_in)) * (6.0 / (n_in ** 0.5))", 1)
s = s.replace("W2 = jax.random.normal(k[3], (h, h)) * 0.1",
              "W2 = jax.random.normal(k[3], (h, h)) * (6.0 / (h ** 0.5))", 1)
s = s.replace("W3 = jax.random.normal(k[4], (n_out, h)) * 0.1",
              "W3 = jax.random.normal(k[4], (n_out, h)) * (6.0 / (h ** 0.5))", 1)
s = s.replace("thresh = jnp.array(1.0)", "thresh = jnp.array(0.3)", 1)
p.write_text(s)
print("examples: lif_cb + tuned spiking params (rate .1, gain 6/sqrt, th .3)")
