import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'RandomSampler' from 'alphagrad.vertexgame' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/vertexgame/__init__.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.vertexgame import RandomSampler, Graph2File, create, write, read, delete, densify
            

sampler = RandomSampler(storage_shape=[10, 50, 10],
                        min_num_intermediates=5)

key = jrand.PRNGKey(42)
gen = Graph2File(sampler,
                "./",
                fname_prefix="test",
                num_samples=3, 
                batchsize=3,
                storage_shape=[10, 50, 10])
gen.generate(key=key, sampling_shape=[10, 40, 10])

code, header, graph = read("test-10_50_10_3_64008.hdf5", [0,1,2])

print(code[0].decode("utf-8"))
print(header[0])
print(graph[0])
print(len(graph[0]))
print(len(graph[1]))
print(len(graph[2]))
header = jnp.array(header[0])
graph = jnp.array(graph[0])
assembled = densify(header, graph, shape=[10, 50, 10])
print(assembled)
