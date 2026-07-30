from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit

from .core import (
    ADD_SPARSITY_MAP,
    CONTRACTION_MAP,
    MUL_SPARSITY_MAP,
    OFFSET,
    get_shape,
)


@partial(jax.vmap, in_axes=(1, None))
def sparsity_fmas_map(in_edge, out_edge):
    i = in_edge[0].astype(jnp.int32) + OFFSET
    j = out_edge[0].astype(jnp.int32) + OFFSET

    new_sparsity_type = MUL_SPARSITY_MAP[i, j]
    contraction_map = CONTRACTION_MAP[:, i, j]

    factors = jnp.concatenate((out_edge[1:3], jnp.abs(out_edge[3:]), in_edge[3:]))

    def true_branch(factors):
        return jnp.where(contraction_map > 0, factors, 1)

    def false_branch(factors):
        return jnp.zeros_like(factors, dtype=jnp.int32)

    masked_factors = lax.cond(
        jnp.sum(contraction_map) > 0, true_branch, false_branch, factors
    )

    masked_factors = jnp.where(masked_factors >= 0, masked_factors, 1)

    fmas = jnp.prod(masked_factors)
    fmas = lax.select(
        jnp.logical_and(jnp.abs(i) == 10 + OFFSET, jnp.abs(j) == 10 + OFFSET), 1, fmas
    )

    return new_sparsity_type, fmas


@partial(jax.vmap, in_axes=(0, 0))
def sparsity_where(in_edge, out_edge):
    i = in_edge.astype(jnp.int32) + OFFSET
    j = out_edge.astype(jnp.int32) + OFFSET
    return ADD_SPARSITY_MAP[i, j]


@jit
def vertex_eliminate(vertex: int, graph: Array):
    num_i, num_v = get_shape(graph)
    edges = graph[:, 1:, :]

    in_edges = edges[:, :, vertex - 1]

    vertex_row_idx = num_i + vertex - 1
    from_vertex = edges[:, vertex_row_idx, :]

    vectorized_map = jax.vmap(sparsity_fmas_map, in_axes=(None, 1))
    path_sparsity, path_fmas = vectorized_map(in_edges, from_vertex)
    path_sparsity = path_sparsity.T
    path_fmas = path_fmas.T

    old_sparsity = edges[0]
    new_sparsity = sparsity_where(old_sparsity, path_sparsity)
    new_sparsity = new_sparsity[None, :, :]

    in_edges_primals = in_edges[3:, :, None]
    out_edges_primals = edges[3:, :, :]

    cond_ins = in_edges_primals[1] != 0
    new_edges_ins = jnp.where(cond_ins, in_edges_primals, out_edges_primals)

    mid_edge_outs = from_vertex[1:3, None, :]
    in_edges_outs = in_edges[1:3, :, None]
    out_edges_outs = edges[1:3, :, :]

    cond_outs = in_edges_outs[1] != 0
    new_edges_outs = jnp.where(cond_outs, mid_edge_outs, out_edges_outs)

    updated_edges = jnp.concatenate(
        [new_sparsity, new_edges_outs, new_edges_ins], axis=0
    )
    updated_edges = updated_edges.at[:, vertex_row_idx, :].set(0)
    updated_edges = updated_edges.at[:, :, vertex - 1].set(0)

    new_graph = graph.at[1, 0, vertex - 1].set(1)
    new_graph = new_graph.at[:, 1:, :].set(updated_edges)

    return new_graph, jnp.sum(path_fmas)
