"""The observation carry: bootstrap from the base stream, advance by a delta.

Stage 2 retired the growing token stream -- the env emits each step's DELTA
with its own exact count, and the base stream is a host-side constant. What
remains is a two-move protocol every trainer runs identically:

``init_carry``  consume the base stream ONCE, fold its rows into the
                per-vertex memory (base eqn ids map to vertex slots
                positionally; structural/overflow -> the global slot).
``advance``     extend the carry by ONE step's delta and fold those rows in
                under the delta's owning vertex.
``heads``       run the pointer / value block off the memory.

Pure ``jnp``: PPO's jitted scan and its vmapped loss call these unchanged, and
AZ can call them eagerly on a single env. Nothing here closes over PPO state.
"""
from __future__ import annotations

import jax.numpy as jnp

from alphagrad.approx import vertex_memory as _vmem

__all__ = ["init_carry", "advance", "heads"]


def init_carry(agent, base_tokens, base_eqns, base_count, *, window,
               total_v, embd_dim):
    """``(enc_carry, vmem_sums, vmem_counts)`` after the base stream.

    ``window`` is the base stream's OWN length (it is a constant of the
    jaxpr, not of the elimination order), so the base encode scan is exactly
    as long as the base is.
    """
    enc0 = agent.carry_init()
    enc1, rows0, valid0, eqns0 = agent.encode_extend(
        enc0, base_tokens, base_eqns, base_count, window=window, start=0,
    )
    base_ids = jnp.where(
        (eqns0 >= 0) & (eqns0 < total_v), eqns0, -1
    )
    vs0 = jnp.zeros((total_v + 1, embd_dim), jnp.float32)
    vc0 = jnp.zeros((total_v + 1,), jnp.float32)
    vs0, vc0 = _vmem.update_ids(vs0, vc0, rows0, base_ids, valid0)
    return enc1, vs0, vc0


def advance(agent, enc_carry, vmem_sums, vmem_counts,
            delta_tokens, delta_eqns, delta_count, owner, *, window):
    """Extend the carry by one step's delta; returns the new
    ``(enc_carry, vmem_sums, vmem_counts)``.

    ``owner`` is the vertex whose elimination emitted this delta -- every row
    of it belongs to that vertex's memory slot.
    """
    carry2, rows, valid, eqns = agent.encode_extend(
        enc_carry, delta_tokens, delta_eqns, delta_count,
        window=window, start=0,
    )
    ids = jnp.where(eqns >= 0, owner, -1)
    sums2, counts2 = _vmem.update_ids(
        vmem_sums, vmem_counts, rows, ids, valid
    )
    return carry2, sums2, counts2


def heads(agent, vmem_sums, vmem_counts, *, vertex_features=None,
          residual_state=None, preference=None):
    """``(vertex_logits, vertex_contexts, value)`` off the vertex memory."""
    return agent.heads_from_memory(
        vmem_sums, vmem_counts,
        vertex_features=vertex_features,
        residual_state=residual_state,
        preference=preference,
    )
