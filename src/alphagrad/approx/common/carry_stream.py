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

__all__ = ["init_carry", "advance", "heads", "base_identity_stream"]


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
    # LAYOUT (total_v + 2): 0..V-1 the vertices, V the GLOBAL slot
    # (structural tokens), V+1 the SUMMARY slot -- every row credited exactly
    # ONCE. The summary slot exists because a row now lands in EVERY vertex it
    # touches (see `advance`), so the per-slot sums no longer re-add to the
    # token total and the value head would otherwise get a fan-out-weighted
    # mean instead of the plain one.
    vs0 = jnp.zeros((total_v + 2, embd_dim), jnp.float32)
    vc0 = jnp.zeros((total_v + 2,), jnp.float32)
    # THE VERTEX SLOTS ARE THE DYNAMIC CHANNEL, so the base stream does NOT
    # go into them: a vertex's own base tokens are its IDENTITY, and the
    # identity is a separate half of the representation
    # (`VertexIdentityPool`, concatenated in `heads_from_memory`). Folding
    # them in here as well would put identity back into the dynamic address
    # -- the exact sharing the split exists to end -- and would double-count
    # them. What still lands:
    #   * UNOWNED base rows (headers, the input list) -> the GLOBAL slot,
    #     which is the pointer's learned summary of the graph's syntax and
    #     has no identity half to move to;
    #   * every base row -> the SUMMARY slot, so the value head's mean is
    #     still over the whole stream.
    _unowned = jnp.asarray(valid0, jnp.float32) * (base_ids < 0)
    vs0, vc0 = _vmem.update_ids(vs0, vc0, rows0, base_ids, _unowned,
                                global_slot=total_v)
    return (enc1,) + _credit_summary(vs0, vc0, rows0, valid0)


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
    gid = n_slots - 2                      # the GLOBAL slot (see init_carry)
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


def heads(agent, vmem_sums, vmem_counts, *, identity_stream=None,
          preference=None):
    """``(vertex_logits, vertex_contexts, value)`` off the vertex memory."""
    return agent.heads_from_memory(
        vmem_sums, vmem_counts,
        identity_stream=identity_stream,
        preference=preference,
    )


def base_identity_stream(agent, base_tokens, base_eqns, base_count, *,
                         window, total_v, base_owners=None):
    """``(rows (T, E), slot ids (T,), valid (T,))`` for the IDENTITY pool.

    One pass of the encoder over the BASE stream -- the same pass
    :func:`init_carry` makes, kept as ROWS instead of pooled sums so the
    identity is an attention pool the head runs (and trains) rather than a
    stored constant. The base stream is a constant of the graph, so this is
    computed ONCE per episode, outside the rollout's vmap; it is
    params-dependent, so it cannot outlive an update.

    ``base_owners`` is the tokenizer's 1-based owning vertex per base token
    (0 = none), read exactly as :func:`init_carry` reads it. Without it every
    row is unowned and the identity is empty for every vertex -- safe, and
    the pre-identity behaviour.
    """
    enc0 = agent.carry_init()
    _enc1, rows, valid, eqns = agent.encode_extend(
        enc0, base_tokens, base_eqns, base_count, window=window, start=0,
        chunk=0,
    )
    if base_owners is None:
        ids = jnp.full(eqns.shape, -1, jnp.int32)
    else:
        _own = jnp.asarray(base_owners, jnp.int32)
        _own = jnp.concatenate(
            [_own, jnp.zeros(eqns.shape, jnp.int32)])[:eqns.shape[0]]
        ids = jnp.where(_own > 0, _own - 1, -1)
    return rows, ids, valid.astype(jnp.float32)
