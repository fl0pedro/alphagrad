"""Per-vertex MEMORY: a constant-size stand-in for the whole token stream.

``PointerVertexPolicy`` cross-attends learned per-vertex queries over the full
``enc_x`` (S, E). That is the last thing forcing an S-sized buffer once the
positional encoding is gone: even with an O(delta) encoder you would still have
to keep every row around so the pointer head could look at it.

But the pointer head does not need the rows — it needs one representation per
vertex. Every token already carries an ``eqn_id`` (which equation, i.e. which
vertex, emitted it), so the rows can be folded into per-vertex slots AS THEY
STREAM and the raw sequence dropped. Memory is (V+1, E) regardless of how long
the stream gets.

WHY MEAN-POOLING AND NOT SOMETHING FANCIER: the fold has to be exactly
incremental. Sum and count are associative, so folding a delta into the memory
gives BITWISE the same result as folding the whole stream at once, in any
chunking. The PPO ratio-1 invariant needs the rollout encode and the loss-time
re-encode to agree exactly; a gated/recurrent pooling would make the result
depend on chunk boundaries and quietly break that. Order-independence is the
feature here, not a limitation.

SLOT V IS THE GLOBAL SLOT. Tokens with ``eqn_id < 0`` are structural (delimiters,
shape atoms) and belong to no vertex, but they are not noise — they carry the
graph's syntax. They fold into slot V, which the pointer head may ATTEND to but
never SELECT (queries are the V real vertices). It acts as a learned summary of
everything structural.

The value-head summary comes out of the same state for free: the token-count-
weighted mean over slots is exactly the mean over all tokens, which is what
``Agent.encode`` computes from ``enc_x``. So the memory replaces the (S, E)
buffer for both consumers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init(num_vertices: int, embd_dim: int):
    """Empty memory: ``(sums, counts)`` with a trailing global slot."""
    return (jnp.zeros((num_vertices + 1, embd_dim), jnp.float32),
            jnp.zeros((num_vertices + 1,), jnp.float32))


def update(sums, counts, rows, eqn_ids, valid=None):
    """Fold ``rows`` (D, E) into the memory by ``eqn_ids`` (D,).

    ``eqn_ids < 0`` (structural tokens) land in the global slot. ``valid``
    (D,) masks padding; omit it when every row is real.

    Associative in ``rows``: update(update(m, a), b) == update(m, concat(a,b)),
    bitwise up to float addition order within a single segment_sum, which is
    what makes the append-only path safe for ratio-1.
    """
    n_slots = sums.shape[0]
    gid = n_slots - 1
    ids = jnp.where(eqn_ids < 0, gid, jnp.minimum(eqn_ids, gid - 1))
    ids = ids.astype(jnp.int32)

    if valid is not None:
        w = valid.astype(jnp.float32)
        rows = rows * w[:, None]
    else:
        w = jnp.ones(rows.shape[0], jnp.float32)

    sums = sums + jax.ops.segment_sum(rows, ids, num_segments=n_slots)
    counts = counts + jax.ops.segment_sum(w, ids, num_segments=n_slots)
    return sums, counts


def scatter(rows, ids, valid, n_segments, *, spill=None):
    """THE SCATTER. ``(sums (n, E), counts (n,))`` -- and NOTHING ELSE.

    One ``segment_sum`` of the rows and one of their weights, keyed by
    ``ids``. It has NO PARAMETERS and it is the ONLY pooling primitive in the
    policy: the vertex slots are this scatter keyed by the owning VERTEX, and
    a face's latent is this scatter keyed by the FACE. Two keyings of one
    operation, not two mechanisms -- which is the entire point. If a weight
    ever appears in here, the two readouts have stopped being the same object.

    ``ids < 0`` (or an invalid row) goes to ``spill`` when a spill segment is
    named, and is DROPPED otherwise -- dropped via a trash segment that is
    sliced off, never by folding into segment 0. A row with no owner must land
    nowhere, not on whichever vertex happens to be first.

    Associative in ``rows`` up to float addition order within one segment,
    which is what lets the delta path fold incrementally and still agree with
    a single pass over the whole stream (the PPO ratio-1 invariant).
    """
    n = int(n_segments)
    w = jnp.ones(rows.shape[0], jnp.float32) if valid is None \
        else jnp.asarray(valid, jnp.float32)
    ids = jnp.asarray(ids, jnp.int32)
    if spill is None:
        seg = jnp.where(ids >= 0, jnp.clip(ids, 0, n - 1), n)
    else:
        seg = jnp.where(ids >= 0, jnp.clip(ids, 0, n - 1), int(spill))
    live = (w > 0.0).astype(jnp.float32)
    seg = jnp.where(live > 0.5, seg, n).astype(jnp.int32)
    w = w * live
    sums = jax.ops.segment_sum(rows * w[:, None], seg, num_segments=n + 1)
    counts = jax.ops.segment_sum(w, seg, num_segments=n + 1)
    return sums[:n], counts[:n]


def scatter_mean(rows, ids, valid, n_segments, *, spill=None):
    """:func:`scatter` read as a per-segment MEAN; empty segments read 0."""
    s, c = scatter(rows, ids, valid, n_segments, spill=spill)
    return s / jnp.maximum(c, 1.0)[:, None]


def update_ids(sums, counts, rows, ids, valid=None, *, global_slot=None):
    """Fold ``rows`` (D, E) into the memory by EXPLICIT slot ids (D,).

    Unlike :func:`update`, ``ids`` are already per-VERTEX slot indices
    (``-1`` ⇒ the global slot), not per-token eqn ids. The incremental
    encoder uses this: base-stream tokens map eqn→vertex positionally,
    while a delta block's tokens all belong to the vertex whose
    elimination emitted them — a mapping only the caller knows.

    This is :func:`scatter` with the unowned rows spilled to the global slot,
    accumulated into an existing memory.
    """
    n_slots = sums.shape[0]
    # The unowned/global slot must be named EXPLICITLY. It used to be
    # `n_slots - 1`, which silently follows the array width -- and once a
    # trailing SUMMARY row exists (see `summary`), that would route every
    # header and input token into the value head's accumulator.
    gid = n_slots - 1 if global_slot is None else int(global_slot)
    s, c = scatter(rows, jnp.minimum(jnp.asarray(ids, jnp.int32), gid - 1),
                   valid, n_slots, spill=gid)
    return sums + s, counts + c


def read(sums, counts):
    """Per-slot mean, ``(V+1, E)``. Empty slots read as zeros."""
    return sums / jnp.maximum(counts, 1.0)[:, None]


def occupancy(counts):
    """``(V+1,)`` bool — which slots have seen at least one token.

    This is the pointer head's attention mask: attending to a slot no token
    ever landed in would mix a zero vector into the representation.
    """
    return counts > 0.0


def summary(sums, counts, *, summary_slot=None):
    """Mean over ALL tokens, ``(E,)``.

    With ``summary_slot`` given, reads THAT slot alone -- the accumulator
    every token is credited to EXACTLY ONCE. Under multi-credit (a face's
    tokens landing in all three of its i/v/j slots) the per-slot sums no
    longer re-add to the token total, so summing them would hand the value
    head an arity-weighted mean instead of the plain one. Without it the
    historical all-slot behaviour is kept, which is correct while every
    token has exactly one owner.

    Identical to ``sum(enc_x * mask) / sum(mask)`` in ``Agent.encode`` — the
    per-slot sums re-add to the same total, so the value heads see exactly what
    they saw under the full-sequence path.
    """
    if summary_slot is not None:
        k = int(summary_slot)
        return sums[k] / jnp.maximum(counts[k], 1e-9)
    return jnp.sum(sums, axis=0) / jnp.maximum(jnp.sum(counts), 1e-9)
