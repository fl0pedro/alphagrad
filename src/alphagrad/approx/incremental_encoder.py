# -*- coding: utf-8 -*-
"""INCREMENTAL causal palimpsa encoder (loop-integration module).

Extracted verbatim from the PROVEN recurrence in incremental_encoder_test.py
(max_abs_diff 9.5e-7 vs agent.encode_tokens under
jax_default_matmul_precision="highest", eqn_ids=None). Carries per-layer
palimpsa (M, I) state so that APPENDING delta tokens to an append-only stream
(GRAPHAX_STATE_TOKENS=1) reproduces the FULL re-encode's ``enc_x`` (S, E)
bit-for-bit (within 1e-4).

The state is a lightweight snapshot (list of per-layer (M,I) + running enc rows
+ position counter) so the loop can (a) encode the STATIC prefix once, (b) copy
the snapshot per decision, and (c) extend by the delta tokens in O(delta), then
feed the resulting full ``enc_x`` into ``agent.vertex_policy``.

The vertex pointer head (PointerVertexPolicy) cross-attends over the WHOLE
``enc_x`` sequence (masked by token_mask), so it is NOT last-token-only; but
because the incremental path reproduces the identical ``enc_x`` tensor, the
pointer logits are identical too. Incrementality holds at the ENCODER level.
"""
import jax
import jax.numpy as jnp
import jax.nn as jnn


def _palimpsa_step(mixer, y_t, carry):
    """One-token palimpsa recurrence, MASK ALL-ONES (fully-valid prefix)."""
    H = mixer.num_heads
    d = mixer.head_dim
    scale = d ** -0.5

    q = mixer.query_proj(y_t).reshape(H, d)
    kk = mixer.key_proj(y_t).reshape(H, d)
    v = mixer.value_proj(y_t).reshape(H, d)
    b = jnn.softplus(mixer.bias_proj(y_t)).reshape(H, d)
    gt = jnn.softplus(mixer.gate_proj(y_t))          # (H,)
    g = jnn.softplus(mixer.g_raw)                     # (H,)
    Ip = jnn.softplus(mixer.Ip_raw)                   # (H,)

    M_prev, I_prev = carry                            # (H, DV, DK) each
    decay = jnp.exp(-gt[:, None, None] * g[:, None, None])   # (H,1,1)
    outer_M = v[:, :, None] * kk[:, None, :]          # (H, DV, DK)
    outer_I = b[:, :, None] * (kk[:, None, :] ** 2)
    M = outer_M + decay * M_prev
    I = outer_I + (1.0 - decay) * Ip[:, None, None] + decay * I_prev
    mu = M / I
    out_t = jnp.einsum('hdn,hn->hd', mu, q * scale)   # (H, d)
    out = out_t.reshape(H * d)
    attn_out = mixer.output_proj(out)                 # (E,)
    return attn_out, (M, I)


def _layer_step(layer, x_t, carry):
    """One EncoderLayer applied to one token x_t (E,) with palimpsa carry."""
    mixer = layer.attn_layer
    y = layer.attn_norm(x_t)
    attn_out, new_carry = _palimpsa_step(mixer, y, carry)
    x_t = x_t + attn_out
    y = layer.mlp_norm(x_t)
    y = layer.mlp(y)
    x_t = x_t + y
    return x_t, new_carry


def _init_carry(mixer):
    """M0 = 0, I0 = broadcast(Ip)  (matches palimpsa_ref)."""
    H = mixer.num_heads
    d = mixer.head_dim
    Ip = jnn.softplus(mixer.Ip_raw)                   # (H,)
    M0 = jnp.zeros((H, d, d), jnp.float32)
    I0 = jnp.broadcast_to(Ip[:, None, None], (H, d, d)).astype(jnp.float32)
    return (M0, I0)


# jit the single-layer step for speed / precision consistency with the
# proven test (the test jits _layer_step + relies on final_norm as-is).
_layer_step_j = jax.jit(_layer_step)


class IncrementalEncoderState:
    """Mutable-ish snapshot of the incremental encode.

    Fields:
      carries : list of per-layer (M, I) tuples (jax arrays)
      rows    : list of per-token enc rows (E,) already emitted (final_norm'd)
      pos     : next positional index to consume
    A ``copy()`` shares the (immutable) jax arrays but snapshots the python
    containers, so branching (extend a copy by different next tokens) is safe.
    """
    __slots__ = ("carries", "rows", "pos")

    def __init__(self, carries, rows, pos):
        self.carries = carries
        self.rows = rows
        self.pos = pos

    def copy(self):
        return IncrementalEncoderState(list(self.carries), list(self.rows), self.pos)


def init_state(agent):
    layers = agent.encoder.layers
    carries = [_init_carry(layer.attn_layer) for layer in layers]
    return IncrementalEncoderState(carries, [], 0)


def extend(agent, state, new_tokens):
    """Extend ``state`` IN PLACE by processing ``new_tokens`` one at a time.

    ``new_tokens`` = iterable of int token ids (real, non-pad, >0). Returns the
    same (mutated) state. Positions continue from ``state.pos``.
    """
    layers = agent.encoder.layers
    embedding = agent.embedding
    pe = agent.pos_enc.pe
    final_norm = agent.final_norm
    carries = state.carries
    pos = state.pos
    for tok in new_tokens:
        x_t = embedding(jnp.asarray(tok, dtype=jnp.int32)) + pe[pos, :]
        for li, layer in enumerate(layers):
            x_t, carries[li] = _layer_step_j(layer, x_t, carries[li])
        state.rows.append(final_norm(x_t))
        pos += 1
    state.pos = pos
    return state


def enc_x(state):
    """Return the full (S, E) enc_x for all real tokens processed so far."""
    return jnp.stack(state.rows, axis=0)
