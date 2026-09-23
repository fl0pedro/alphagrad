import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.common import datasets as ds
from alphagrad.approx.common import examples as ex

B, S, D, V = 4, 8, 16, 32


@pytest.fixture
def small_tlm(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_TLM_SEQ", str(S))
    monkeypatch.setenv("ALPHAGRAD_TLM_DMODEL", str(D))
    monkeypatch.setenv("ALPHAGRAD_TLM_VOCAB", str(V))
    monkeypatch.setattr(ex, "NN_VMAP_BATCH", B)
    corpus = np.asarray(
        jax.random.randint(jax.random.PRNGKey(7), (4096,), 0, V), np.int32)
    monkeypatch.setattr(ds, "load_wikitext2", lambda vocab, subset="train": corpus)


def test_vmapped_tlm_args_batch_x_and_y(small_tlm):
    xs = ex.get_args("VmappedTransformerLM", jax.random.PRNGKey(0))
    ref = ex.get_args("TransformerLM", jax.random.PRNGKey(0))
    assert xs[0].shape == (B, S, D)
    assert xs[1].shape == (B, S, V)
    assert ref[0].shape == (S, D) and ref[1].shape == (S, V)
    assert len(xs) == len(ref)
    for a, b in zip(xs[2:], ref[2:]):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    gen = ex.data_gen("VmappedTransformerLM", dataset="wikitext2")
    x, y = gen(jax.random.split(jax.random.PRNGKey(3), 2))
    assert x.shape == (B, S, D) and y.shape == (B, S, V)
    assert (ex.infer_argnums("VmappedTransformerLM")
            == ex.infer_argnums("TransformerLM"))


def test_vmapped_tlm_is_mean_of_row_losses(small_tlm):
    xs = ex.get_args("VmappedTransformerLM", jax.random.PRNGKey(0))
    x, y, ws = xs[0], xs[1], xs[2:]
    batched = ex.get_fn("VmappedTransformerLM")
    single = ex.get_fn("TransformerLM")
    loss_b = batched(x, y, *ws)
    rows = [single(x[i], y[i], *ws) for i in range(B)]
    assert loss_b.shape == ()
    np.testing.assert_allclose(float(loss_b), float(jnp.mean(jnp.stack(rows))),
                               rtol=1e-6)

    argnums = ex.infer_argnums("VmappedTransformerLM")
    g_b = jax.grad(batched, argnums=argnums)(x, y, *ws)
    g_rows = [jax.grad(single, argnums=argnums)(x[i], y[i], *ws)
              for i in range(B)]
    for k in range(len(argnums)):
        want = jnp.mean(jnp.stack([g[k] for g in g_rows]), axis=0)
        np.testing.assert_allclose(np.asarray(g_b[k]), np.asarray(want),
                                   rtol=1e-5, atol=1e-6)
