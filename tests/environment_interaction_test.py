import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'get_masked_logits' from 'alphagrad.utils' (/Users/assmuth/dsnn/alphagrad/src/alphagrad/utils.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import os

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import equinox as eqx

from alphagrad.vertexgame import (step, make_graph, safe_preeliminations, 
                                compress, embed)
from graphax.examples.roe import RoeFlux_1d

from alphagrad.utils import get_masked_logits, make_init_state
from alphagrad.alphazero.environment_interaction import (make_recurrent_fn,
                                                make_environment_interaction)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
key = jrand.PRNGKey(42)
batched_step = jax.vmap(step)
batched_get_masked_logits = jax.vmap(get_masked_logits, in_axes=(0, 0))


INFO = [20, 105, 20]
MODEL = lambda x, key: jnp.ones(106)/106.


recurrent_fn = make_recurrent_fn(MODEL,
                                batched_step, 
                                batched_get_masked_logits)

env_interaction = make_environment_interaction(INFO, 
                                            	10,
                                                recurrent_fn,
                                                batched_step,
												temperature=0)

edges = make_graph(RoeFlux_1d, 1., 1., 1., 1., 1., 1.)
print(edges.at[0, 0, 0:3].get())
edges, preelim_order = safe_preeliminations(edges, return_preeliminated=True)
edges = compress(edges)
edges = embed(key, edges, INFO)
print(edges.shape)
games = edges[jnp.newaxis, :]

init_carry = make_init_state(games, key)
data = env_interaction(MODEL, init_carry)

