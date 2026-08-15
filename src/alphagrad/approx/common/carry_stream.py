"""The observation carry: the BASE memory, plus a delta per step.

THE GOVERNING RULE OF THIS MODULE (2026-08-15)
----------------------------------------------
EVERY LEARNED PARAMETER SITS DOWNSTREAM OF PALIMPSA, AND EVERY PATH FROM
PALIMPSA TO THE LOSS IS DIFFERENTIATED. Nothing the encoder produces may be
stored, detached, or precomputed outside the gradient.

That rule is why the base stream is no longer consumed into a value that the
loss then treats as a constant. It used to be, twice over:

  * ``init_carry`` folded the base rows into a per-vertex memory which the
    rollout STORED per step and the loss read back -- so the encode that
    produced them was outside every gradient;
  * ``base_identity_stream`` ran a SECOND encoder pass per episode, outside
    the loss, and handed the resulting rows to a ``VertexIdentityPool``. The
    pool's own weights took gradient; the palimpsa pass that wrote its input
    took none. The encoder that produces vertex identity was trained only
    through the single step-delta encode of ``--grad-window`` K=1.

Now there is one function, :func:`init_carry`, which returns the BASE MEMORY
as its own object, and callers are required to keep it separate from the
dynamic accumulation:

    base_mem = (base_sums, base_counts)     <- RECOMPUTED under gradient
    dyn_mem  = zero_memory(...) then advance(), advance(), ...   <- stored
    slots    = read(base_sums + dyn_sums, base_counts + dyn_counts)

The split is exact because the memory is (sum, count) pairs and the readout
is a mean: adding the two memories IS pooling the union of their rows. So the
rollout can store the dynamic half (cheap, and it must, since it is a
trajectory) while the loss re-derives the base half from the tokens every
time it differentiates -- which is what puts the base encode back inside the
gradient, over the WHOLE base stream, not a K-step window of it.

The two-move protocol every trainer runs is otherwise unchanged:

``init_carry``  consume the base stream ONCE -> the palimpsa carry and the
                base rows SCATTERED into the slot of the vertex that produced
                each one (``base_owners``, from the tokenizer). Headers and
                inputs, and the whole block if owners are unavailable, go to
                the global slot.
``advance``     extend the carry by ONE step's delta and scatter those rows
                over the slots the delta PARTICIPATES in.
``heads``       run the pointer / value block off ``dyn_mem + base_mem``.

Pure ``jnp``: PPO's jitted scan and its vmapped loss call these unchanged, and
AZ can call them eagerly on a single env. Nothing here closes over PPO state.
"""
from __future__ import annotations

import jax.numpy as jnp

from alphagrad.approx import vertex_memory as _vmem

__all__ = ["init_carry", "base_memory", "zero_memory", "advance", "heads"]


def zero_memory(total_v, embd_dim):
    """The EMPTY dynamic memory: ``(sums, counts)``.

    LAYOUT (total_v + 2): 0..V-1 the vertices, V the GLOBAL slot (structural
    tokens), V+1 the SUMMARY slot -- every row credited exactly ONCE. The
    summary slot exists because a row lands in EVERY vertex it touches (see
    :func:`advance`), so the per-slot sums no longer re-add to the token
    total and the value head would otherwise get a fan-out-weighted mean
    instead of the plain one.
    """
    return (jnp.zeros((int(total_v) + 2, int(embd_dim)), jnp.float32),
            jnp.zeros((int(total_v) + 2,), jnp.float32))


def init_carry(agent, base_tokens, base_eqns, base_count, *, window,
               total_v, embd_dim, base_owners=None):
    """``(enc_carry, base_sums, base_counts)`` -- the BASE MEMORY.

    The returned memory is NOT an accumulator to advance into: it is the base
    stream's own contribution, kept separate so it can be recomputed inside
    the loss (see this module's docstring). Start the dynamic accumulation
    from :func:`zero_memory` and hand this to :func:`heads` as ``base_mem``.

    A VERTEX'S IDENTITY IS ITS OWN ROWS, ARRIVING THROUGH THE SAME SCATTER.
    There is no identity pool and no ``[identity || dynamic]`` concatenation:
    the rows palimpsa emits for a vertex's own equation are scattered into
    that vertex's slot by ``_vmem.scatter``, exactly as a step delta's rows
    are scattered into the slots it participates in. One mechanism, no
    weights, and the slots stay E wide.

    ``base_owners`` is the tokenizer's per-token owning VERTEX (1-based,
    0 = none). Its SEGMENT ids carry no vertex information and must not be
    used for this. Omit it and the whole block goes to the global slot.

    ``window`` is the base stream's OWN length (it is a constant of the
    jaxpr, not of the elimination order), so the base encode scan is exactly
    as long as the base is.
    """
    enc0 = agent.carry_init()
    # chunk=0: the base window IS the base length (count == window), so a
    # dynamic trip count has nothing to skip -- and the flat scan is the form
    # reverse-mode AD can transpose, which this call now needs.
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
    vs0, vc0 = zero_memory(total_v, embd_dim)
    vs0, vc0 = _vmem.update_ids(vs0, vc0, rows0, base_ids, valid0,
                                global_slot=total_v)
    return (enc1,) + _credit_summary(vs0, vc0, rows0, valid0)


def base_memory(agent, base_tokens, base_eqns, base_count, *, window,
                total_v, embd_dim, base_owners=None):
    """:func:`init_carry` without the carry -- ``(base_sums, base_counts)``.

    THIS IS THE CALL THE LOSS MAKES, inside the differentiated region, once
    per loss evaluation and OUTSIDE the per-sample vmap (the base stream is a
    constant of the graph, so the result is the same for every sample in the
    minibatch and a closed-over unbatched tracer is what vmap wants). It is
    the entire reason palimpsa's base encode now has a cotangent.
    """
    return init_carry(agent, base_tokens, base_eqns, base_count,
                      window=window, total_v=total_v, embd_dim=embd_dim,
                      base_owners=base_owners)[1:]


def _credit_summary(sums, counts, rows, valid):
    """Credit the SUMMARY slot (the last one) with every valid row, once."""
    w = jnp.asarray(valid, jnp.float32)
    return (sums.at[-1].add(jnp.sum(rows * w[:, None], axis=0)),
            counts.at[-1].add(jnp.sum(w)))


def advance(agent, enc_carry, vmem_sums, vmem_counts,
            delta_tokens, delta_eqns, delta_count, owner, *, window,
            chunk=None, budget=None, participants=None):
    """Extend the carry by one step's delta; returns the new
    ``(enc_carry, vmem_sums, vmem_counts)``.

    PARTICIPATION, NOT AUTHORSHIP. ``owner`` is the vertex whose elimination
    emitted this delta, and crediting the rows to that slot alone was the
    whole dynamic channel: an un-eliminated CANDIDATE's slot never changed,
    however much the elimination rewired the graph around it, so the pointer
    was choosing between vertices whose state had not moved since the base
    stream. ``participants`` is the set of slots the delta TOUCHES -- the
    eliminated vertex plus the endpoints of every face it contracted through
    -- as a ``(total_v + 1,)`` 0/1 mask (the trailing entry is the global
    slot). Every equation row of the delta is credited to every one of them.

    There is NO fan-out cap and nothing to overflow: the set is a mask over
    the slots, not a K-vector of ids, so it costs O(V) and cannot truncate.
    It is the same scatter :func:`init_carry` uses, written in its mask form
    because every row of one delta shares one participation set -- an outer
    product instead of a (rows x slots) key.

    WHAT ``vmem_counts`` MEANS under fan-out: "how many rows TOUCHED this
    slot", so ``read`` stays the mean over the rows that touched it. It is no
    longer the token count, which is why the value head reads the dedicated
    SUMMARY slot (credited exactly once per row) instead of re-adding the
    per-slot sums.

    ``participants=None`` keeps the old authorship crediting, which is what
    the unit tests and any caller without a face enumeration get.

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
    n_slots = vmem_sums.shape[0]
    gid = n_slots - 2                      # the GLOBAL slot (see zero_memory)
    w = jnp.asarray(valid, jnp.float32)
    if participants is None:
        ids = jnp.where(eqns >= 0, owner, -1)
        sums2, counts2 = _vmem.update_ids(
            vmem_sums, vmem_counts, rows, ids, valid, global_slot=gid,
        )
        return (carry2,) + _credit_summary(sums2, counts2, rows, valid)
    # Every row of a delta carries the SAME participation set, so the fan-out
    # is one outer product, not a (rows x slots) scatter.
    w_eqn = w * (eqns >= 0).astype(jnp.float32)
    w_str = w - w_eqn                       # structural rows -> global slot
    tot_eqn = jnp.sum(rows * w_eqn[:, None], axis=0)
    n_eqn = jnp.sum(w_eqn)
    part = jnp.asarray(participants, jnp.float32)
    # A delta with no participants at all (the pre-scan bootstrap, whose
    # "owner" is -1) goes to the global slot -- the destination authorship
    # gave it, so nothing is silently dropped.
    part = jnp.where(jnp.sum(part) > 0, part,
                     jnp.eye(n_slots - 1, dtype=jnp.float32)[gid])
    sums2 = vmem_sums.at[: n_slots - 1].add(part[:, None] * tot_eqn[None, :])
    counts2 = vmem_counts.at[: n_slots - 1].add(part * n_eqn)
    sums2 = sums2.at[gid].add(jnp.sum(rows * w_str[:, None], axis=0))
    counts2 = counts2.at[gid].add(jnp.sum(w_str))
    return (carry2,) + _credit_summary(sums2, counts2, rows, valid)


def heads(agent, vmem_sums, vmem_counts, *, base_mem=None, preference=None):
    """``(vertex_logits, vertex_contexts, value)`` off the vertex memory.

    ``base_mem`` is :func:`base_memory`'s ``(sums, counts)``, ADDED to the
    dynamic memory before the readout. It is a separate argument and not a
    stored part of ``vmem_*`` for one reason: the caller has to be able to
    recompute it inside the gradient.
    """
    return agent.heads_from_memory(
        vmem_sums, vmem_counts,
        base_mem=base_mem,
        preference=preference,
    )
