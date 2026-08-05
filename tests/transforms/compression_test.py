import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: alphagrad.make_random_code was removed. "
    "Kept for provenance; delete once the subsystem is confirmed gone for good.",
    allow_module_level=True,
)

import jax
import jax.random as jrand

from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_random_code
from graphax.transforms.clean import clean
from graphax.transforms.compression import compress


key = jrand.PRNGKey(42)
info = [10, 20, 10]
code, jaxpr = make_random_code(key, info)
edges = make_graph(jaxpr)
print(edges)
edges = clean(edges)
print(edges.shape)

edges = compress(edges)
print(edges)
print(edges.shape)

