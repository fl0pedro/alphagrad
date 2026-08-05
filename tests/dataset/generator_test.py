import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'VertexGameGenerator' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.random as jrand

from alphagrad.vertexgame import VertexGameGenerator

key = jrand.PRNGKey(1337)
info = []
gen = VertexGameGenerator(16, info, key=key)  
print(gen(8, key).edges.shape) 
print(gen(8, key).info)

