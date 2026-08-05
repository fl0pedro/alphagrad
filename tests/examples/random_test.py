import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'make_random_code' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx
from alphagrad.vertexgame import make_random_code


info = [15, 105, 20]
key = jrand.PRNGKey(123)
code, jaxpr = make_random_code(key, info, primal_p=[1, 0, 0], primitive_p=[.2, .8, .0, .0, .0])
print(code)
print(jaxpr)

edges = gx.make_graph(jaxpr)
print(edges)

edges = gx.make_graph(jaxpr)
# print(edges)
# edges = gx.safe_preeliminations(edges)
# edges = gx.compress(edges)

_, fops = gx.forward(edges)
_, rops = gx.reverse(edges)
order = gx.minimal_markowitz(edges)
_, ccops = gx.cross_country(order, edges)
print(fops, rops, ccops, f"gain: {100.*(1.-ccops/min(fops, rops)):.2f}%")


