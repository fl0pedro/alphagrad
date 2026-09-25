import jax
import jax.numpy as jnp
import numpy as np
import pytest
from graphax import jacve

from alphagrad.approx.common import rsnn_shd as R

H, N_IN, N_OUT = 6, 5, 3
SPIKES = (1.0, 0.0, 1.0, 1.0, 0.0, 1.0)


def _step(seed):
    rng = np.random.default_rng(seed)

    def draw(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    x = jnp.asarray(rng.random(N_IN) < 0.5, jnp.float32)
    S = jnp.asarray(SPIKES, jnp.float32)
    return [x, S, draw(H), draw(H), draw(H), draw(N_OUT),
            draw(H, N_IN), draw(H, H), draw(N_OUT, H), *R._consts()]


def _current(cell, args, g):
    def f(V, S):
        a = list(args)
        a[1], a[7] = S, V
        return jnp.dot(g, cell(*a)[1])
    return f


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_eprop_cell_keeps_the_direct_gradient_of_v(seed):
    # dsnn-eaa: the Diag is on the state edge of V @ S only; the direct V term stays
    args = _step(seed)
    g = jnp.asarray(np.random.default_rng(seed + 100).normal(size=H),
                    jnp.float32)
    bd = R._blockdiag_cell("eprop")
    ex = R._cell()
    for a, b in zip(bd(*args), ex(*args)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-6, atol=1e-6)
    dV, dS = jax.grad(_current(bd, args, g), argnums=(0, 1))(args[7], args[1])
    dV_ex, dS_ex = jax.grad(_current(ex, args, g),
                            argnums=(0, 1))(args[7], args[1])
    np.testing.assert_allclose(np.asarray(dV), np.asarray(dV_ex),
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(dS),
                               np.asarray(jnp.diagonal(args[7]) * g),
                               rtol=1e-6, atol=1e-7)
    assert not np.allclose(np.asarray(dS), np.asarray(dS_ex))


def test_jacve_traces_the_eprop_cell_and_agrees_with_jacrev():
    args = _step(0)
    bd = R._blockdiag_cell("eprop")
    rows = jacve(bd, "rev", argnums=(7,))(*args)
    got = rows[1][0] if isinstance(rows[1], (tuple, list)) else rows[1]
    want = jax.jacrev(lambda V: bd(*args[:7], V, *args[8:])[1])(args[7])
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=1e-5, atol=1e-6)
