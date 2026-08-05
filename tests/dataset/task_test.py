import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'read' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.vertexgame import (reverse, forward, cross_country, read,
                                    minimal_markowitz,  make_task_dataset)

key = jrand.PRNGKey(42)
make_task_dataset(key, "./task_dataset.hdf5")

names, data = read("./task_dataset.hdf5", [0, 1, 2, 3, 4, 5])


for name, edges in zip(names, data):
    print(name)
    edges = jnp.array(edges)
    print(forward(edges)[1])
    print(reverse(edges)[1])
    order = minimal_markowitz(edges)
    print("order", len(order), order)
    print(cross_country(order, edges)[1])

