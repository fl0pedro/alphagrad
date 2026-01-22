from typing import Any, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from .core import get_elimination_order, get_shape, get_vertex_mask, vertex_eliminate

Array = jax.Array

EnvOut = Tuple[Array, float, bool, Any]


def step(edges: Array, action: int) -> EnvOut:
    vertex = action + 1
    t = jnp.where(get_elimination_order(edges) > 0, 1, 0).sum()
    new_edges, nops = vertex_eliminate(vertex, edges)
    new_edges = new_edges.at[3, 0, t].set(vertex)

    # Reward is the negative of the multiplication count
    reward = -nops
    num_eliminated_vertices = get_vertex_mask(new_edges).sum()
    num_intermediates = get_shape(new_edges)[1]
    terminated = lax.select(num_eliminated_vertices == num_intermediates, True, False)

    return new_edges, reward, terminated
