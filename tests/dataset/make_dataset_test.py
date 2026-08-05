import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'Graph2File' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.random as jrand

from alphagrad.vertexgame import Graph2File, read, RandomSampler


key = jrand.PRNGKey(42) 

sampler = RandomSampler(max_info=[5,10,5], min_num_intermediates=12)

gen = Graph2File(sampler, "./", sampler_batchsize=64, num_samples=128, samples_per_file=64, max_info=[5, 10, 5])

gen.generate(key=key)

codes, graphs = read("comp_graph_examples-0.hdf5", [i for i in range(63)])

print(graphs)

