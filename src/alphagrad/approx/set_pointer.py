"""Content-based vertex pointer over the segment-pooled vertex memory.

WHY THIS REPLACES ``PointerVertexPolicy``
-----------------------------------------
The old head built its queries from ``eqx.nn.Embedding(num_vertices, embd_dim)``
-- a FIXED-SIZE LEARNED TABLE INDEXED BY VERTEX ID. Vertex 5 always used
embedding row 5, so:

  * vertex identity was a learned slot rather than graph CONTENT;
  * the policy was hard-tied to one ``num_vertices`` (no transfer between
    targets, and a new graph size means a new table);
  * scoring against the raw (S, E) stream cost O(V*S*E).

A pointer network in the Vinyals sense scores candidates by their content. Here
the content already exists: ``vertex_memory`` segment-pools the token stream
into (V+1, E) slots by ``eqn_id``. So keys AND the query are derived from those
pooled rows, and nothing depends on V.

SHAPE / COST
------------
    vmem (V+1, E) --[L perm-equivariant self-attention blocks]--> h (V+1, E)
    q = W_q(masked_mean(h))            (E,)
    k = vmap(W_k)(h)                   (V+1, E)
    logits = (k @ q) / sqrt(E)         (V+1,)   masked to legal, then [:V]

Fully parallel: no per-vertex python loop, no slicing, no gather. O(V^2 E) for
the self-attention blocks and O(V E) for the scoring, against O(V S E) before
(V ~ 13-83, S ~ 7000).

The trailing slot (index V) holds structural tokens with no owning equation. It
may be ATTENDED (it carries global context) but never SELECTED, so the returned
logits are sliced to the first V entries -- matching the old head's contract.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand


class SetBlock(eqx.Module):
    """One permutation-equivariant Set-Transformer block: masked MHSA + MLP,
    both pre-norm with residuals. Permutation equivariance is what makes this
    valid on an unordered vertex set -- there is no positional signal, so
    relabelling the vertices permutes the output identically."""

    norm1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, embd_dim: int, num_heads: int, *, key):
        k1, k2 = jrand.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(embd_dim)
        self.attn = eqx.nn.MultiheadAttention(num_heads, embd_dim, key=k1)
        self.norm2 = eqx.nn.LayerNorm(embd_dim)
        self.mlp = eqx.nn.MLP(embd_dim, embd_dim, embd_dim, depth=1, key=k2)

    def __call__(self, h: jax.Array, mask: jax.Array) -> jax.Array:
        n = jax.vmap(self.norm1)(h)
        # (n_q, n_kv) boolean mask: every query may attend every OCCUPIED slot.
        attn_mask = jnp.broadcast_to(mask[None, :] > 0.5, (h.shape[0], h.shape[0]))
        h = h + self.attn(n, n, n, mask=attn_mask)
        h = h + jax.vmap(self.mlp)(jax.vmap(self.norm2)(h))
        return h


class SetPointerVertexPolicy(eqx.Module):
    """Size-agnostic content-based pointer over pooled vertex slots."""

    blocks: tuple
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    out_norm: eqx.nn.LayerNorm
    embd_dim: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)

    def __init__(self, *, num_vertices, embd_dim, num_heads, num_blocks=2, key):
        keys = jrand.split(key, num_blocks + 3)
        self.embd_dim = embd_dim
        # Kept ONLY to slice the selectable prefix and to satisfy the existing
        # call sites; no parameter depends on it, so a different V needs no
        # new weights.
        self.num_vertices = num_vertices
        self.blocks = tuple(
            SetBlock(embd_dim, num_heads, key=keys[i]) for i in range(num_blocks)
        )
        self.q_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[num_blocks])
        self.k_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[num_blocks + 1])
        self.out_norm = eqx.nn.LayerNorm(embd_dim)

    def _score(self, vmem: jax.Array, vmask: jax.Array):
        h = vmem
        for blk in self.blocks:
            h = blk(h, vmask)
        h = jax.vmap(self.out_norm)(h)
        # Query = occupancy-weighted mean over slots (permutation invariant).
        w = (vmask > 0.5).astype(h.dtype)[:, None]
        summary = jnp.sum(h * w, axis=0) / jnp.maximum(jnp.sum(w), 1.0)
        q = self.q_proj(summary)
        k = jax.vmap(self.k_proj)(h)
        logits = (k @ q) / jnp.sqrt(jnp.asarray(self.embd_dim, h.dtype))
        logits = jnp.where(vmask > 0.5, logits, -1e9)
        return logits, h

    def from_vertex_memory(self, vmem, vmask):
        """(V+1, E) pooled slots -> (V,) logits and (V, E) contexts."""
        logits, h = self._score(vmem, vmask)
        return logits[: self.num_vertices], h[: self.num_vertices]

    def __call__(self, enc_x, token_mask, eqn_ids=None):
        """Raw-stream fallback. With ``eqn_ids`` the stream is segment-pooled
        first, so this is the SAME computation as ``from_vertex_memory``.
        Without them there is no vertex assignment to pool by, so the tokens
        are treated as a single set and every vertex sees the same summary --
        which is only meaningful before any elimination has happened."""
        if eqn_ids is None:
            pooled = jnp.broadcast_to(
                jnp.mean(enc_x, axis=0), (self.num_vertices + 1, enc_x.shape[-1])
            )
            vmask = jnp.ones((self.num_vertices + 1,), dtype=enc_x.dtype)
            return self.from_vertex_memory(pooled, vmask)
        n_slots = self.num_vertices + 1
        w = (token_mask > 0.5).astype(enc_x.dtype)
        ids = jnp.clip(eqn_ids, 0, n_slots - 1)
        sums = jax.ops.segment_sum(enc_x * w[:, None], ids, num_segments=n_slots)
        cnts = jax.ops.segment_sum(w, ids, num_segments=n_slots)
        pooled = sums / jnp.maximum(cnts, 1.0)[:, None]
        return self.from_vertex_memory(pooled, (cnts > 0).astype(enc_x.dtype))
