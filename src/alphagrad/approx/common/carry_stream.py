"""The observation carry: bootstrap from the base stream, advance by a delta.

Stage 2 retired the growing token stream -- the env emits each step's DELTA
with its own exact count, and the base stream is a host-side constant. What
remains is a two-move protocol every trainer runs identically:

``init_carry``  consume the base stream ONCE, folding each row into the
                slot of the VERTEX that produced it (``base_owners``, from
                the tokenizer); headers and inputs, and the whole block if
                owners are unavailable, go to the global slot.
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
               total_v, embd_dim, base_owners=None):
    """``(enc_carry, vmem_sums, vmem_counts)`` after the base stream.

    ``base_owners`` is the tokenizer's per-token owning VERTEX (1-based,
    0 = none). Its SEGMENT ids carry no vertex information and must not be
    used for this. Omit it and the whole block goes to the global slot.

    ``window`` is the base stream's OWN length (it is a constant of the
    jaxpr, not of the elimination order), so the base encode scan is exactly
    as long as the base is.
    """
    enc0 = agent.carry_init()
    # chunk=0: the base window IS the base length (count == window), so a
    # dynamic trip count has nothing to skip -- and this runs once per
    # episode, where the flat scan is the cheaper thing to compile.
    enc1, rows0, valid0, eqns0 = agent.encode_extend(
        enc0, base_tokens, base_eqns, base_count, window=window, start=0,
        chunk=0,
    )
    # BASE ATTRIBUTION. `eqns0` are SEGMENT ids from a stream-global running
    # counter -- graphax.jaxpr.last_eqn_ids is explicit that they are "built
    # for relational consumers that only compare ids", NOT vertex indices,
    # and base_tokens emits the whole base block in ONE _emit_eqns call so
    # every base equation token shares id 0. Reading them as vertex indices
    # credited the entire base stream to vertex 1 (#92).
    #
    # The TRUE owners come from the tokenizer instead:
    # `IncrementalPathTokenizer.last_owner_ids()` gives the 1-based vertex
    # that produced each base token (0 = no owner: headers, the input list),
    # recorded because `_build_graph` walks `jaxpr.eqns` and every traced
    # base equation therefore belongs to exactly one original equation.
    # With them, every vertex starts with palimpsa content describing its
    # own primal op and elemental partials -- without them the pointer has
    # only static vertex_features to tell candidates apart at the root.
    #
    # Fallback is the #92 behaviour (everything to the GLOBAL slot), which
    # is always safe. Never fall back to reading `eqns0` as vertex ids.
    if base_owners is None:
        base_ids = jnp.full(eqns0.shape, -1, jnp.int32)
    else:
        _own = jnp.asarray(base_owners, jnp.int32)
        _own = jnp.concatenate(
            [_own, jnp.zeros(eqns0.shape, jnp.int32)])[:eqns0.shape[0]]
        # owner is 1-based; slot index is owner-1. 0 (no owner) -> -1 ->
        # the global slot, same destination the fallback uses.
        base_ids = jnp.where(_own > 0, _own - 1, -1)
    vs0 = jnp.zeros((total_v + 1, embd_dim), jnp.float32)
    vc0 = jnp.zeros((total_v + 1,), jnp.float32)
    vs0, vc0 = _vmem.update_ids(vs0, vc0, rows0, base_ids, valid0)
    return enc1, vs0, vc0


def advance(agent, enc_carry, vmem_sums, vmem_counts,
            delta_tokens, delta_eqns, delta_count, owner, *, window,
            chunk=None, budget=None):
    """Extend the carry by one step's delta; returns the new
    ``(enc_carry, vmem_sums, vmem_counts)``.

    ``owner`` is the vertex whose elimination emitted this delta -- every row
    of it belongs to that vertex's memory slot.

    ``chunk`` is forwarded to :meth:`Agent.encode_extend`: it bounds how much
    of the (mostly empty) delta window is actually scanned. ``None`` takes the
    ``ALPHAGRAD_EXTEND_CHUNK`` default. REVERSE-DIFFERENTIATED callers add
    ``budget`` -- an unbatched batch-wide bound on ``delta_count`` -- which
    swaps the ``lax.while_loop`` for a transposable ``scan``/``cond`` pair;
    without it they must pass ``chunk=0``.
    """
    carry2, rows, valid, eqns = agent.encode_extend(
        enc_carry, delta_tokens, delta_eqns, delta_count,
        window=window, start=0, chunk=chunk, budget=budget,
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
