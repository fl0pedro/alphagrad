"""decode3_arch: the 2026-08-14 architecture's vertex/face assembly, rebuilt
from a stored token stream.

ONE place defines how a per-vertex and a per-face representation are built, so
the two trainers and the equivalence check cannot drift.  Every line here
mirrors a named piece of the live path:

  identity      Agent.identity_pool over the BASE rows, keyed by the
                tokenizer's per-token owning vertex (carry_stream
                .base_identity_stream).
  vmem          (V+2) layout: 0..V-1 vertices, V global, V+1 summary.  Base
                contributes ONLY unowned rows (-> global) and every row
                (-> summary); vertex slots are the DYNAMIC channel and start
                empty (carry_stream.init_carry).
  advance       a delta's EQUATION rows are credited to every PARTICIPANT
                slot (the (V+1) mask), its STRUCTURAL rows to the global slot,
                and all of them once to the summary slot (carry_stream
                .advance, participants=...).
  heads         [identity || dynamic] concatenated, SetPointer over V+2 slots
                with an all-ones mask, contexts projected back to E by
                ctx_proj (Agent.heads_from_memory).
  face          [ctx_i || ctx_j || face_latent], endpoints gathered from the
                vertex contexts, latent = mean palimpsa row over the face's
                own token chunk (UnifiedFacePolicy._repr / Agent._face_pool).

Nothing here reads a hand-written vertex feature.  Arms that do take it as an
explicit extra input, so the cost of the deleted features is measurable.
"""
import os
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_CHUNKED_EXTEND", "1")
os.environ.setdefault("ALPHAGRAD_CHUNK_BLOCK", "512")

import jax
import jax.numpy as jnp
from types import SimpleNamespace

from alphagrad.approx.ppo import _build_agent


def agent_args(embd_dim, num_layers, num_heads, pointer_blocks, vocab=512):
    return SimpleNamespace(
        vocab_size=vocab, embd_dim=embd_dim, op_embd_dim=8,
        num_layers=num_layers, num_heads=num_heads, hidden_dim=64,
        value_dims="64,32", set_pointer=True,
        set_pointer_blocks=pointer_blocks, dynamic_substeps=False,
        no_approx_head=True, live_faces=False, face_actions=False,
        unified_head=False, unified_face_head=False, max_substeps=1)


def build(embd_dim, num_layers, num_heads, pointer_blocks, NV, key,
          vocab=512):
    return _build_agent(
        agent_args(embd_dim, num_layers, num_heads, pointer_blocks, vocab),
        NV, 1, 1, key)


def encode(agent, tok, eqn, ntok, W):
    """ONE causal encode over the concatenated stream -> (rows, weights)."""
    c0 = agent.carry_init()
    _, rows, valid, _ = agent.encode_extend(c0, tok, eqn, ntok, window=W,
                                            start=0, chunk=0)
    return rows, valid.astype(jnp.float32)


def identity(agent, rows, w, own, did, NV):
    """``(V+2, E)`` per-vertex identity: the learned attention pool over each
    vertex's OWN span of the BASE stream.  Delta rows are excluded by weight,
    which is what makes this an IDENTITY and not a state."""
    is_base = (did < 0)
    ids = jnp.where(is_base, own, -1).astype(jnp.int32)
    val = w * is_base.astype(jnp.float32)
    return agent.identity_pool(rows, ids, val, NV + 2)


def memory_tables(rows, w, own, eqn, did, part, NV, NSTEP):
    """Everything needed to read the vertex memory at ANY step, in one pass.

    Returns ``(S_base, C_base, cS, cC, gS, gC, sS, sC)`` where the ``c*`` are
    INCLUSIVE cumulative sums over elimination steps, so the memory before
    step ``t`` reads index ``t-1``.
    """
    E = rows.shape[-1]
    is_base = (did < 0).astype(jnp.float32)
    bw = w * is_base
    unowned = bw * (own < 0).astype(jnp.float32)
    S_base = (jnp.zeros((NV + 2, E), jnp.float32)
              .at[NV].add(jnp.sum(rows * unowned[:, None], 0))
              .at[NV + 1].add(jnp.sum(rows * bw[:, None], 0)))
    C_base = (jnp.zeros((NV + 2,), jnp.float32)
              .at[NV].add(jnp.sum(unowned))
              .at[NV + 1].add(jnp.sum(bw)))

    d_id = jnp.where(did < 0, 0, did).astype(jnp.int32)
    dw = w * (1.0 - is_base)
    w_eqn = dw * (eqn >= 0).astype(jnp.float32)
    w_str = dw - w_eqn
    seg = lambda a: jax.ops.segment_sum(a, d_id, num_segments=NSTEP)
    tot_eqn = seg(rows * w_eqn[:, None]); n_eqn = seg(w_eqn)
    tot_str = seg(rows * w_str[:, None]); n_str = seg(w_str)
    tot_all = tot_eqn + tot_str; n_all = n_eqn + n_str

    P = part.astype(jnp.float32)                       # (NSTEP, NV+1)
    cS = jnp.cumsum(P[:, :, None] * tot_eqn[:, None, :], axis=0)
    cC = jnp.cumsum(P * n_eqn[:, None], axis=0)
    gS = jnp.cumsum(tot_str, axis=0); gC = jnp.cumsum(n_str, axis=0)
    sS = jnp.cumsum(tot_all, axis=0); sC = jnp.cumsum(n_all, axis=0)
    return S_base, C_base, cS, cC, gS, gC, sS, sC


def mem_at(tab, t, NV, E, NSTEP):
    """``(S, C)`` of the (V+2) memory covering every delta STRICTLY BEFORE
    step ``t``.  ``t=0`` is the base stream alone."""
    S_base, C_base, cS, cC, gS, gC, sS, sC = tab
    i = jnp.clip(t - 1, 0, NSTEP - 1)
    on = (t > 0).astype(jnp.float32)
    S = jnp.zeros((NV + 2, E), jnp.float32).at[:NV + 1].set(cS[i] * on)
    S = S.at[NV].add(gS[i] * on).at[NV + 1].add(sS[i] * on)
    C = jnp.zeros((NV + 2,), jnp.float32).at[:NV + 1].set(cC[i] * on)
    C = C.at[NV].add(gC[i] * on).at[NV + 1].add(sC[i] * on)
    return S + S_base, C + C_base


def heads(agent, S, C, ident):
    """``(vertex_contexts (V, E), vmem_rows (V+2, E))`` -- the live
    ``heads_from_memory`` pointer block, minus the value head."""
    vrows = S / jnp.maximum(C, 1.0)[:, None]
    slots = jnp.concatenate([ident, vrows], axis=-1)
    _logits, ctx2 = agent.vertex_policy.from_vertex_memory(
        slots, jnp.ones(slots.shape[0], jnp.float32))
    return jax.vmap(agent.ctx_proj)(ctx2), vrows


def face_latent(rows, w, start, split, cap):
    """Mean palimpsa row over ``[start, split)`` -- ``Agent._face_pool``."""
    ar = jnp.arange(cap, dtype=jnp.int32)
    idx = jnp.minimum(start + ar, rows.shape[0] - 1)
    m = ((start + ar) < split).astype(jnp.float32) * w[idx]
    return jnp.sum(rows[idx] * m[:, None], 0) / jnp.maximum(m.sum(), 1.0)
