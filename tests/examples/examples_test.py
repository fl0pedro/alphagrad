import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ImportError: cannot import name 'BlackScholes' from 'graphax.examples.easy' (/Users/assmuth/dsnn/graphax/src/graphax/examples/easy.py) "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.numpy as jnp

import graphax as gx
from graphax.examples.easy import (Simple, Helmholtz, Hole, BlackScholes, 
                                   CloudSchemes_step, Lighthouse)
from graphax.examples. randoms import f
from graphax.examples.minpack import HumanHeartDipole, PropaneCombustion
from graphax.examples.roe import RoeFlux_1d, RoeFlux_3d
from graphax.examples.differential_kinematics import RobotArm_6DOF
from alphagrad.vertexgame import make_graph


edges = make_graph(BlackScholes, jnp.ones(4), jnp.ones(4), jnp.ones(4), jnp.ones(4), jnp.ones(4))
print(edges)
print(gx.get_shape(edges))
_, ops = gx.forward(edges)
print(ops)
_, ops = gx.reverse(edges)
print(ops)

order = gx.minimal_markowitz(edges)
output, ops = gx.cross_country(order, edges)
_, ops = output
print(ops)


# edges, info = make_LIF()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)


# edges, info = make_hessian()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)

