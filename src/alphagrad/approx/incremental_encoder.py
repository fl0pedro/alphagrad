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
import equinox as eqx

from alphagrad.transformer.palimpsa_encoder import (
    palimpsa_beta as _pal_beta, palimpsa_qk_norm as _pal_qk_norm)


def _palimpsa_step(mixer, y_t, carry):
    """One-token palimpsa recurrence, MASK ALL-ONES (fully-valid prefix)."""
    H = mixer.num_heads
    d = mixer.head_dim
    scale = d ** -0.5

    q = _pal_qk_norm(mixer.query_proj(y_t).reshape(H, d))
    kk = _pal_qk_norm(mixer.key_proj(y_t).reshape(H, d))
    v = mixer.value_proj(y_t).reshape(H, d)
    b = _pal_beta(mixer.bias_proj(y_t).reshape(H, d), mixer.b_scale_raw)
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


# --- SCAN-based batched extend --------------------------------------------
# The per-token Python loop (embed -> L layer_steps -> final_norm, one tiny
# jitted call PER TOKEN PER LAYER) dispatched an unbounded storm of ~O(delta*L)
# micro-executables per decision. Across a proposer round (pool * decisions *
# candidates) that pegs the host CPU on JAX dispatch/tracing and the round never
# completes (the "hang" — process CPU-busy, not deadlocked). Folding the whole
# delta into ONE lax.scan makes ``extend`` dispatch a SINGLE program per call
# (compiled once per distinct delta-length; <= num_vertices distinct lengths,
# each cheap+cached). The scan body is byte-identical math to _layer_step +
# final_norm, so the proven equivalence is preserved.
def _extend_block(agent, carries, tokens, positions):
    """Process ``tokens`` (int32 (D,)) at ``positions`` (int32 (D,)) starting
    from ``carries`` (list of L (M,I) pairs) via a single lax.scan. Returns
    (new_carries, rows) where rows is (D, E) final_norm'd. Same math as the
    per-token loop, one XLA program."""
    layers = agent.encoder.layers
    embedding = agent.embedding
    pe = agent.pos_enc.pe if agent.pos_enc is not None else None
    # The trainer's Agent has NO final_norm and PalimpsaEncoder.__call__
    # applies none — it just stacks layers and returns. MicroPPOAgent (the
    # deprecated Ray line this module was written against) DOES have one, so
    # applying it unconditionally would add a normalization the full encode
    # never performs and silently break the equivalence this module exists to
    # guarantee. Identity when absent.
    final_norm = getattr(agent, "final_norm", None) or (lambda z: z)

    def _step(carry_list, tp):
        tok, pos = tp
        x_t = embedding(tok.astype(jnp.int32))
        if pe is not None:
            x_t = x_t + pe[pos, :]
        new_carry = []
        for li, layer in enumerate(layers):
            x_t, c = _layer_step(layer, x_t, carry_list[li])
            new_carry.append(c)
        return new_carry, final_norm(x_t)

    carry_in = [tuple(c) for c in carries]
    new_carry, rows = jax.lax.scan(_step, carry_in, (tokens, positions))
    return new_carry, rows


# eqx.filter_jit partitions the agent's static (non-array) leaves so the whole
# MicroPPOAgent can be passed through jit safely (plain jax.jit would choke on
# its int/str static fields).
_extend_block_j = eqx.filter_jit(_extend_block)


def _extend_block_masked(agent, carries, tokens, positions, valid):
    """Fixed-shape ``_extend_block``: pad steps FREEZE the carry.

    ``_extend_block`` scans a variable-length delta, so it recompiles per
    length and cannot live in a jitted rollout whose shapes must be static.
    This variant takes a fixed ``(D,)`` buffer plus a ``valid`` mask and is
    therefore compiled ONCE, which is what lets the palimpsa carry ride in a
    ``lax.scan`` carry alongside the env state.

    The valid prefix is BITWISE identical to ``_extend_block`` over the same
    tokens: an invalid step runs the same arithmetic but its result is
    discarded by a select, so no value that survives is computed differently.
    That matters more than it sounds — the PPO ratio-1 invariant needs the
    rollout encode and the loss-time re-encode to agree exactly, not closely.

    Returns ``(new_carries, rows)`` with ``rows`` zeroed on invalid steps.
    """
    layers = agent.encoder.layers
    embedding = agent.embedding
    pe = agent.pos_enc.pe if agent.pos_enc is not None else None
    # The trainer's Agent has NO final_norm and PalimpsaEncoder.__call__
    # applies none — it just stacks layers and returns. MicroPPOAgent (the
    # deprecated Ray line this module was written against) DOES have one, so
    # applying it unconditionally would add a normalization the full encode
    # never performs and silently break the equivalence this module exists to
    # guarantee. Identity when absent.
    final_norm = getattr(agent, "final_norm", None) or (lambda z: z)
    n_pos = pe.shape[0] if pe is not None else 0

    def _step(carry_list, tpv):
        tok, pos, ok = tpv
        x_t = embedding(tok.astype(jnp.int32))
        if pe is not None:
            # Clamp: an invalid step's position is never used, but it must not
            # index out of the table before the select discards it.
            x_t = x_t + pe[jnp.clip(pos, 0, n_pos - 1), :]
        new_carry = []
        for li, layer in enumerate(layers):
            x_t, c = _layer_step(layer, x_t, carry_list[li])
            old = carry_list[li]
            new_carry.append(tuple(
                jnp.where(ok, new_leaf, old_leaf)
                for new_leaf, old_leaf in zip(c, old)))
        row = final_norm(x_t)
        return new_carry, jnp.where(ok, row, jnp.zeros_like(row))

    carry_in = [tuple(c) for c in carries]
    new_carry, rows = jax.lax.scan(
        _step, carry_in, (tokens, positions, valid))
    return new_carry, rows


_extend_block_masked_j = eqx.filter_jit(_extend_block_masked)


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
    import os as _os, time as _t
    _dbg = _os.environ.get("ALPHAGRAD_IE_DEBUG", "0") == "1"
    toks = list(new_tokens)
    pos0 = state.pos
    n = len(toks)
    if _dbg:
        print(f"[ie.extend] START ntokens={n} pos0={pos0}", flush=True)
    if n == 0:
        return state
    _ts = _t.time()
    tok_arr = jnp.asarray(toks, dtype=jnp.int32)
    pos_arr = jnp.arange(pos0, pos0 + n, dtype=jnp.int32)
    new_carry, rows = _extend_block_j(agent, state.carries, tok_arr, pos_arr)
    # carry: list of tuples -> keep as list-of-tuples (matches state.carries).
    state.carries = [tuple(c) for c in new_carry]
    # rows is (n, E); split into per-row (E,) to preserve the append-only API.
    for i in range(n):
        state.rows.append(rows[i])
    state.pos = pos0 + n
    if _dbg:
        jax.block_until_ready(rows)
        print(f"[ie.extend] DONE pos={state.pos} ({n} toks in {_t.time()-_ts:.3f}s)", flush=True)
    return state


def enc_x(state):
    """Return the full (S, E) enc_x for all real tokens processed so far."""
    return jnp.stack(state.rows, axis=0)
