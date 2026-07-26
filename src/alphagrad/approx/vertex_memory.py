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


def read(sums, counts):
    """Per-slot mean, ``(V+1, E)``. Empty slots read as zeros."""
    return sums / jnp.maximum(counts, 1.0)[:, None]


def occupancy(counts):
    """``(V+1,)`` bool — which slots have seen at least one token.

    This is the pointer head's attention mask: attending to a slot no token
    ever landed in would mix a zero vector into the representation.
    """
    return counts > 0.0


def summary(sums, counts):
    """Mean over ALL tokens, ``(E,)``.

    Identical to ``sum(enc_x * mask) / sum(mask)`` in ``Agent.encode`` — the
    per-slot sums re-add to the same total, so the value heads see exactly what
    they saw under the full-sequence path.
    """
    return jnp.sum(sums, axis=0) / jnp.maximum(jnp.sum(counts), 1e-9)
