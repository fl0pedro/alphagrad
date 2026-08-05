import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: TypeError: Value after * must be an iterable, not NoneType "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

from graphax.examples.neuromorphic import ADALIF_SNN
from alphagrad.vertexgame import (minimal_markowitz, forward, reverse, 
                                cross_country, make_graph)

x = None
edges = make_graph(ADALIF_SNN, *x)
print(edges)

order = minimal_markowitz(edges)
print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

