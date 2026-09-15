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

import os

import jax.numpy as jnp
from jax import lax

from alphagrad.approx import vertex_memory as _vmem
from alphagrad.approx.common import delta_fold as _fold

__all__ = ["init_carry", "base_memory", "zero_memory", "zero_edge_memory",
           "advance", "heads"]


def zero_edge_memory(n_slots, embd_dim):
    """The EMPTY edge-keyed memory (--face-edge-mem): ``(sums, counts)``.

    ``(K, E)`` sums + ``(K,)`` counts, K = the face bound (each face writes
    exactly one res edge, so distinct keys <= faces eliminated; the host
    slot table evicts-oldest past K). No global and no summary slot: an
    unowned row is simply dropped by ``_vmem.scatter``'s trash segment --
    an edge event either has a slot or it does not exist.
    """
    return (jnp.zeros((int(n_slots), int(embd_dim)), jnp.float32),
            jnp.zeros((int(n_slots),), jnp.float32))


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


def init_carry(agent, base_tokens, base_count, *, window,
               total_v, embd_dim, base_owners=None, path):
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

    ``path`` is ``"rollout"`` or ``"loss"`` and is REQUIRED. BOTH sides call
    this function on the same base stream -- the rollout once per episode,
    the loss again inside ``filter_grad`` -- and since owner ruling
    2026-09-15 the two may read palimpsa with different operators. Nothing
    else in the call distinguishes them, so the caller has to say.
    """
    enc0 = agent.carry_init()
    # chunk=0: the base window IS the base length (count == window), so a
    # dynamic trip count has nothing to skip -- and the flat scan is the form
    # reverse-mode AD can transpose, which this call now needs.
    # The base stream is the LARGEST row block in the system -- mean 43,678
    # tokens on the TLM -- and d815349 made it full-horizon inside the
    # gradient, so its (window, E) rows are materialised and held for the
    # backward on every loss call. Folding removes that array outright; the
    # per-chunk reduction reuses the SAME `_vmem.update_ids` +
    # `_credit_summary` primitives, so the attribution semantics are
    # unchanged and scatter-add's associativity is what makes the split
    # exact. ALPHAGRAD_FOLD_DELTA=0 restores the full-width form.
    if _FOLD:
        return _init_carry_folded(
            agent, enc0, base_tokens, base_count,
            window=window, total_v=total_v, embd_dim=embd_dim,
            base_owners=base_owners, path=path)
    enc1, rows0, valid0 = agent.encode_extend(
        enc0, base_tokens, base_count, window=window, start=0, chunk=0,
        path=path,
    )
    # BASE ATTRIBUTION. The owners come from the tokenizer:
    # `IncrementalPathTokenizer.last_owner_ids()` gives the 1-based vertex
    # that produced each base token (0 = no owner: headers, the input list),
    # recorded because `_build_graph` walks `jaxpr.eqns` and every traced
    # base equation therefore belongs to exactly one original equation.
    #
    # (The stream-global SEGMENT ids never belonged here and are gone now:
    # base_tokens emitted the whole base block in ONE _emit_eqns call, so
    # every base equation token shared id 0 and reading them as vertex
    # indices credited the entire base stream to vertex 1, bug #92.)
    #
    # Fallback is the #92 behaviour (everything to the GLOBAL slot), which
    # is always safe.
    base_ids = _base_ids(base_owners, rows0.shape[0])
    vs0, vc0 = zero_memory(total_v, embd_dim)
    vs0, vc0 = _vmem.update_ids(vs0, vc0, rows0, base_ids, valid0,
                                global_slot=total_v)
    return (enc1,) + _credit_summary(vs0, vc0, rows0, valid0)


def _base_ids(base_owners, n):
    """Per-token owning vertex SLOT, or -1 for the global slot.

    Extracted verbatim from `init_carry` so both paths derive the ids the
    same way.
    """
    if base_owners is None:
        return jnp.full((n,), -1, jnp.int32)
    _own = jnp.asarray(base_owners, jnp.int32)
    _own = jnp.concatenate([_own, jnp.zeros((n,), jnp.int32)])[:n]
    return jnp.where(_own > 0, _own - 1, -1)


def _init_carry_folded(agent, enc0, base_tokens, base_count, *,
                       window, total_v, embd_dim, base_owners=None, path):
    """`init_carry` with the (window, E) rows folded away chunk by chunk."""
    # Pad the ids to the fold's PADDED length, not the window: the fold pads
    # its token buffers to nb * C and slices side arrays at the same offsets,
    # so a window-length ids array overruns the final chunk. -1 sends the pad
    # to the global slot, and those lanes are invalid anyway so they carry
    # zero weight.
    C, _nb, padded = _fold.plan_chunks(window, path=path)
    base_ids = _base_ids(base_owners, padded)
    vs0, vc0 = zero_memory(total_v, embd_dim)

    def fold(acc, rows, valid, off):
        sums, counts = acc
        ids_c = lax.dynamic_slice(base_ids, (off,), (C,))
        s2, c2 = _vmem.update_ids(sums, counts, rows, ids_c, valid,
                                  global_slot=total_v)
        return _credit_summary(s2, c2, rows, valid)

    enc1, (vs1, vc1) = _fold.extend_fold(
        agent, enc0, base_tokens, base_count,
        window=window, init_acc=(vs0, vc0), fold_fn=fold, path=path)
    return enc1, vs1, vc1


def base_memory(agent, base_tokens, base_count, *, window,
                total_v, embd_dim, base_owners=None, path):
    """:func:`init_carry` without the carry -- ``(base_sums, base_counts)``.

    THIS IS THE CALL THE LOSS MAKES, inside the differentiated region, once
    per loss evaluation and OUTSIDE the per-sample vmap (the base stream is a
    constant of the graph, so the result is the same for every sample in the
    minibatch and a closed-over unbatched tracer is what vmap wants). It is
    the entire reason palimpsa's base encode now has a cotangent.

    The ROLLOUT calls it too, for its own base scatter, and passes
    ``path="rollout"``. Behaviour policy and target policy agree at epoch 0
    only while the two paths read the same operator.
    """
    return init_carry(agent, base_tokens, base_count,
                      window=window, total_v=total_v, embd_dim=embd_dim,
                      base_owners=base_owners, path=path)[1:]


_FOLD = os.environ.get("ALPHAGRAD_FOLD_DELTA", "1") != "0"


def _credit_summary(sums, counts, rows, valid):
    """Credit the SUMMARY slot (the last one) with every valid row, once."""
    w = jnp.asarray(valid, jnp.float32)
    return (sums.at[-1].add(jnp.sum(rows * w[:, None], axis=0)),
            counts.at[-1].add(jnp.sum(w)))


def _stream_window(stream, start, row, window):
    """Cut one step's ``window`` out of an episode stream.

    Only the callers that cannot read chunk by chunk use this -- it is the
    materialised window the episode stream exists to avoid.
    """
    s = jnp.asarray(start, jnp.int32)
    if row is None:
        return lax.dynamic_slice(stream, (s,), (int(window),))
    r = jnp.asarray(row, jnp.int32)
    return lax.dynamic_slice(stream, (r, s), (1, int(window))).reshape(
        int(window))


def advance(agent, enc_carry, vmem_sums, vmem_counts,
            delta_tokens, delta_count, owner, *, window,
            chunk=None, budget=None, participants=None,
            edge_mem=None, edge_ids=None, start=None, row=None, path):
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

    EVERY VALID ROW IS AN EQUATION ROW NOW, AND THIS IS A CHANGE. The split
    between "equation" rows (credited to the participants / the owner) and
    "structural" rows (credited to the GLOBAL slot) was made by
    ``delta_eqns >= 0``, and the equation-id stream was removed on
    2026-09-13. There is no other per-token signal on the delta path, and
    inventing one would be a guess. The collapse is the honest reading of
    what a delta is: one elimination emits one block, punctuation included,
    and the whole block describes that elimination -- so the whole block goes
    to the slots the elimination touched. The global slot still receives the
    pre-scan bootstrap delta, whose participation set is empty (see the
    ``sum(part) > 0`` fallback below), so nothing is left without a
    destination.

    ``chunk`` is forwarded to :meth:`Agent.encode_extend`: it bounds how much
    of the (mostly empty) delta window is actually scanned. ``None`` takes the
    ``ALPHAGRAD_EXTEND_CHUNK`` default. REVERSE-DIFFERENTIATED callers add
    ``budget`` -- an unbatched batch-wide bound on ``delta_count`` -- which
    swaps the ``lax.while_loop`` for a transposable ``scan``/``cond`` pair;
    without it they must pass ``chunk=0``.

    EDGE-KEYED MEMORY (--face-edge-mem, docs/FACE_LATENT_INFO_LOSS.md
    section 8): pass ``edge_mem=(emem_sums (K, E), emem_counts (K,))`` and
    ``edge_ids`` -- the per-token EDGE SLOT of this delta (``(window,)``
    int32, -1 = no slot -> dropped). The SAME rows this call already
    encodes are additionally scattered by those ids (``_vmem.scatter`` --
    the identical primitive, a third keying), inside the SAME fold, so the
    edge write costs no second encode. The return then appends the updated
    ``(emem_sums, emem_counts)``; with ``edge_mem=None`` the signature,
    the arithmetic and the trace are exactly the pre-flag ones.

    THE EPISODE STREAM. ``start`` (and ``row`` for an ``(E, L)`` stream, the
    shape the loss holds) say that ``delta_tokens`` is the environment's
    whole episode stream and this delta begins at ``start``. The FOLDED
    branch reads chunk ``j`` straight out of it with a ``dynamic_slice``;
    the unfolded legacy branch cuts the ``window`` out first, because
    ``encode_extend`` takes a standalone buffer. See
    ``common.episode_stream``.

    ``path`` is ``"rollout"`` or ``"loss"`` and is REQUIRED, for the reason
    :func:`init_carry` gives: both sides advance the same carry over the same
    delta and only the caller knows which side it is.
    """
    if edge_mem is not None and edge_ids is None:
        raise ValueError("advance: edge_mem given without edge_ids")
    if row is not None and start is None:
        raise ValueError("advance: row= given without start=")
    if _FOLD and participants is not None:
        # Only the PARTICIPATION branch folds. The authorship branch below is
        # the legacy path for callers without a face enumeration (unit tests),
        # so leaving it full-width keeps this patch off code production does
        # not run.
        return _advance_folded(
            agent, enc_carry, vmem_sums, vmem_counts, delta_tokens,
            delta_count, window=window, chunk=chunk,
            budget=budget, participants=participants,
            edge_mem=edge_mem, edge_ids=edge_ids, start=start, row=row,
            path=path)
    if start is not None:
        # LEGACY, UNFOLDED BRANCH. `encode_extend` reads a standalone
        # buffer, so the delta's window is cut out of the stream first --
        # the one place the episode stream still materialises a window, and
        # it is the branch production does not run (`_FOLD` is on by
        # default and the trainer always passes `participants`).
        delta_tokens = _stream_window(delta_tokens, start, row, window)
    carry2, rows, valid = agent.encode_extend(
        enc_carry, delta_tokens, delta_count,
        window=window, start=0, chunk=chunk, budget=budget, path=path,
    )
    _edge_out = ()
    if edge_mem is not None:
        # The SAME rows, scattered a third way: by the host-assigned edge
        # slot of the token's owning face's res edge. -1 rides the trash
        # segment, so an unattributed token lands nowhere.
        _eids = jnp.asarray(edge_ids, jnp.int32)[: rows.shape[0]]
        _es, _ec = _vmem.scatter(rows, _eids, valid, edge_mem[0].shape[0])
        _edge_out = (edge_mem[0] + _es, edge_mem[1] + _ec)
    n_slots = vmem_sums.shape[0]
    gid = n_slots - 2                      # the GLOBAL slot (see zero_memory)
    w = jnp.asarray(valid, jnp.float32)
    if participants is None:
        # AUTHORSHIP. Every valid row goes to the owner's slot; the
        # ``eqns >= 0`` test that used to send structural rows to the global
        # slot went with the equation ids (see the docstring).
        ids = jnp.full(rows.shape[:1], jnp.asarray(owner, jnp.int32),
                       jnp.int32)
        sums2, counts2 = _vmem.update_ids(
            vmem_sums, vmem_counts, rows, ids, valid, global_slot=gid,
        )
        return ((carry2,) + _credit_summary(sums2, counts2, rows, valid)
                + _edge_out)
    # Every row of a delta carries the SAME participation set, so the fan-out
    # is one outer product, not a (rows x slots) scatter.
    tot_rows = jnp.sum(rows * w[:, None], axis=0)
    n_rows = jnp.sum(w)
    part = jnp.asarray(participants, jnp.float32)
    # A delta with no participants at all (the pre-scan bootstrap, whose
    # "owner" is -1) goes to the global slot -- the destination authorship
    # gave it, so nothing is silently dropped.
    part = jnp.where(jnp.sum(part) > 0, part,
                     jnp.eye(n_slots - 1, dtype=jnp.float32)[gid])
    sums2 = vmem_sums.at[: n_slots - 1].add(part[:, None] * tot_rows[None, :])
    counts2 = vmem_counts.at[: n_slots - 1].add(part * n_rows)
    return ((carry2,) + _credit_summary(sums2, counts2, rows, valid)
            + _edge_out)


def _advance_folded(agent, enc_carry, vmem_sums, vmem_counts, delta_tokens,
                    delta_count, *, window, chunk=None,
                    budget=None, participants=None,
                    edge_mem=None, edge_ids=None, start=None, row=None,
                    path):
    """`advance`'s participation branch with the rows folded away.

    Everything the branch does with `rows` is LINEAR in two running
    quantities -- the row sum and the row count -- so the chunks accumulate
    those and the outer product with `part` is applied ONCE at the end,
    exactly as the full-width form applies it once. Fewer scatters than
    folding the scatter itself, and identical arithmetic up to the order of
    the additions. (It used to be four quantities: the eqn-owned and
    structural halves were accumulated separately. There is one half now --
    see :func:`advance`.)

    ``edge_mem``/``edge_ids`` (--face-edge-mem) add a (K, E)/(K,) scatter
    accumulator to the SAME fold -- `_vmem.scatter` per chunk with the ids
    sliced at the chunk offset, exactly the `_init_carry_folded` pattern --
    so the edge write shares the one encode this call already pays for.
    """
    n_slots = vmem_sums.shape[0]
    gid = n_slots - 2
    E = vmem_sums.shape[1]
    init = (jnp.zeros((E,), jnp.float32), jnp.zeros((), jnp.float32))
    if edge_mem is not None:
        K = edge_mem[0].shape[0]
        C, _nb, padded = _fold.plan_chunks(window, chunk, path=path)
        # Pad the ids to the fold's PADDED length (see plan_chunks): -1
        # sends the pad lanes to the trash segment, and they are invalid
        # anyway so they carry zero weight.
        _eids = jnp.asarray(edge_ids, jnp.int32)
        _eids = jnp.concatenate(
            [_eids, -jnp.ones((padded,), jnp.int32)])[:padded]
        init = init + (jnp.zeros((K, E), jnp.float32),
                       jnp.zeros((K,), jnp.float32))

    def fold(acc, rows, valid, _off):
        tot_r, n_r = acc[:2]
        w = jnp.asarray(valid, jnp.float32)
        out = (tot_r + jnp.sum(rows * w[:, None], axis=0), n_r + jnp.sum(w))
        if edge_mem is not None:
            es_a, ec_a = acc[2], acc[3]
            ids_c = lax.dynamic_slice(_eids, (_off,), (rows.shape[0],))
            e_s, e_c = _vmem.scatter(rows, ids_c, valid, K)
            out = out + (es_a + e_s, ec_a + e_c)
        return out

    carry2, _acc = _fold.extend_fold(
        agent, enc_carry, delta_tokens, delta_count,
        window=window, chunk=chunk, budget=budget,
        init_acc=init, fold_fn=fold, start=start, row=row, path=path)
    tot_rows, n_rows = _acc[:2]

    part = jnp.asarray(participants, jnp.float32)
    part = jnp.where(jnp.sum(part) > 0, part,
                     jnp.eye(n_slots - 1, dtype=jnp.float32)[gid])
    sums2 = vmem_sums.at[: n_slots - 1].add(part[:, None] * tot_rows[None, :])
    counts2 = vmem_counts.at[: n_slots - 1].add(part * n_rows)
    # The summary slot takes EVERY valid row exactly once.
    sums2 = sums2.at[-1].add(tot_rows)
    counts2 = counts2.at[-1].add(n_rows)
    if edge_mem is not None:
        return (carry2, sums2, counts2,
                edge_mem[0] + _acc[2], edge_mem[1] + _acc[3])
    return carry2, sums2, counts2


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
