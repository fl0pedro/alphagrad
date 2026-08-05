import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: alphagrad.make_random_code was removed. "
    "Kept for provenance; delete once the subsystem is confirmed gone for good.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_Helmholtz
from graphax.examples import make_random_code
from graphax.transforms.clean import clean, connectivity_checker

import sys
jnp.set_printoptions(threshold=sys.maxsize)

edges = make_Helmholtz()
print(connectivity_checker(edges))

key = jrand.PRNGKey(42)
info = [5, 15, 5]
code, jaxpr = make_random_code(key, info)
print(jaxpr)
edges = make_graph(jaxpr)
print(edges)
edges = clean(edges)

