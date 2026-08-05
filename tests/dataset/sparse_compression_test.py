import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'sparsify' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

from graphax.examples import Helmholtz
from alphagrad.vertexgame import sparsify, densify, make_graph

edges = make_graph(Helmholtz, jnp.ones(4))


header, sparse_edges = sparsify(edges)
print(sparse_edges)

dense_edges = densify(header, sparse_edges)
print(dense_edges)

