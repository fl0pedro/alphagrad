# -*- coding: utf-8 -*-
"""
Palimpsa (Bayesian-metaplastic gated linear attention) ported to JAX Pallas-Triton.

Milestone 1 deliverable: a VERIFIED Pallas-Triton kernel matching a pure-JAX
reference. NOT yet integrated into the PPO policy.

Ported from the original Triton source:
  chunk_palimpsa_forgetting_scalar.py  (chunk fwd + bwd autograd.Function)
  fused_recurrent_palimpsa.py          (forward-only recurrent, used to cross-check math)

Core per-token recurrence (ground truth; matches the torch `palimpsa_ref` in the source):
    decay_t = exp(-gt_t * g)                              # scalar forget gate
    M_t     = v_t (x) k_t   + decay_t * M_{t-1}           # KV numerator state   [D_V, D_K]
    I_t     = b_t (x) k_t^2 + (1-decay_t)*Ip + decay_t*I_{t-1}   # precision state [D_V, D_K]
    mu_t    = M_t / I_t
    out_t   = sum_k (mu_t * q_t * scale)

(In the source's I_bar formulation, I_t = I_bar_t + Ip with
 I_bar_t = b_t (x) k_t^2 + decay_t*I_bar_{t-1}; algebraically identical to the above.)

API
---
    out = palimpsa_attention(q, k, v, b, gt, g, Ip, scale=None, chunk_size=16)
      q,k : [B,T,H,D_K]   v,b : [B,T,H,D_V]   gt : [B,T,H]   g,Ip : [H]
      -> out : [B,T,H,D_V]
    Wrapped in jax.custom_vjp: fwd = Pallas chunked forward kernel (saves chunk-boundary
    (mu,I) states as residuals); bwd = Pallas chunked backward kernel (reverse adjoint
    recurrence, recomputing intra-chunk state from the saved boundaries).

Pallas-mapping caveats (also see report):
  - Triton's `tl.associative_scan` over the chunk axis is NOT reproduced literally.
    We replace the chunk-parallel scan with the *exact* serial per-token recurrence
    (`lax.fori_loop`) inside the kernel. Same math, same linear complexity. The grid
    still parallelizes over (batch*head); within a (b,h) program the recurrence is
    serial. Chosen for guaranteed correctness over literal kernel fidelity, as
    instructed. (The chunk_size only affects how often boundary states are saved; it
    has no effect on the numerics, which is exactly what the chunk-size sweep verifies.)
  - The original kernel splits D_V into BV blocks (BV=min(pow2,8)) on grid axis 0 and
    accumulates dv/db across NK key-blocks. We do NOT split D_V/D_K (one program holds
    the full [D_V,D_K] state); the BV-bites-at-D_V=16 case is therefore handled by a
    single program and is automatically correct. No atomics / split-accumulate needed.
  - Backward dg/dIp are reduced per (batch*head) inside the kernel and summed over batch
    on the host to yield per-head [H] gradients.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

# Force the Triton GPU backend (the Mosaic-GPU backend rejects sub-warpgroup
# scalar transfers like the per-head g/Ip scalar loads).
_TRITON = plt.CompilerParams(num_warps=4, num_stages=2)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _pad_last(x, target):
    """Zero-pad the last axis of x up to `target` (no-op if already >=)."""
    d = x.shape[-1]
    if d == target:
        return x
    pad = [(0, 0)] * (x.ndim - 1) + [(0, target - d)]
    return jnp.pad(x, pad)


# =============================================================================
# Pure-JAX reference (ground truth) -- lax.scan per-token recurrence
# =============================================================================
def palimpsa_ref(q, k, v, b, gt, g, Ip, scale=None):
    """Pure-JAX reference. q,k:[B,T,H,DK] v,b:[B,T,H,DV] gt:[B,T,H] g,Ip:[H] -> [B,T,H,DV]."""
    q = q.astype(jnp.float32); k = k.astype(jnp.float32)
    v = v.astype(jnp.float32); b = b.astype(jnp.float32)
    gt = gt.astype(jnp.float32); g = g.astype(jnp.float32); Ip = Ip.astype(jnp.float32)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    if scale is None:
        scale = DK ** -0.5

    qT = jnp.transpose(q, (1, 0, 2, 3))
    kT = jnp.transpose(k, (1, 0, 2, 3))
    vT = jnp.transpose(v, (1, 0, 2, 3))
    bT = jnp.transpose(b, (1, 0, 2, 3))
    gtT = jnp.transpose(gt, (1, 0, 2))               # [T,B,H]

    M0 = jnp.zeros((B, H, DV, DK), jnp.float32)
    I0 = jnp.broadcast_to(Ip.reshape(1, H, 1, 1), (B, H, DV, DK)).astype(jnp.float32)
    gh = g.reshape(1, H, 1, 1)
    Iph = Ip.reshape(1, H, 1, 1)

    def step(carry, inp):
        M_prev, I_prev = carry
        q_t, k_t, v_t, b_t, gt_t = inp
        decay = jnp.exp(-gt_t[:, :, None, None] * gh)             # [B,H,1,1]
        outer_M = v_t[:, :, :, None] * k_t[:, :, None, :]         # [B,H,DV,DK]
        outer_I = b_t[:, :, :, None] * (k_t[:, :, None, :] ** 2)
        M = outer_M + decay * M_prev
        I = outer_I + (1.0 - decay) * Iph + decay * I_prev
        mu = M / I
        out_t = jnp.einsum('bhdn,bhn->bhd', mu, q_t * scale)
        return (M, I), out_t

    inps = (qT, kT, vT, bT, gtT)
    (_, _), outs = lax.scan(step, (M0, I0), inps)
    return jnp.transpose(outs, (1, 0, 2, 3))


# =============================================================================
# Pallas FORWARD kernel.  Grid = (B*H,).  One program -> full [D_V,D_K] state.
# Serial token loop over T; save (mu,I) at each chunk boundary as a residual.
# =============================================================================
def _fwd_kernel(q_ref, k_ref, v_ref, b_ref, gt_ref, g_ref, Ip_ref,
                o_ref, state_mu_ref, state_I_ref,
                *, T, DK, DV, scale, chunk_size, n_chunks):
    # refs carry a leading singleton block dim from BlockSpec; index with [0,...].
    g_s = g_ref[0]
    Ip_s = Ip_ref[0]

    def body(t, carry):
        mu_p, I_p = carry                          # [DV,DK]
        # at a chunk boundary, save the entry state for that chunk
        c = t // chunk_size
        is_boundary = (t - c * chunk_size) == 0

        @pl.when(is_boundary)
        def _save():
            state_mu_ref[0, c, :, :] = mu_p
            state_I_ref[0, c, :, :] = I_p

        q_t = q_ref[0, t, :]
        k_t = k_ref[0, t, :]
        v_t = v_ref[0, t, :]
        b_t = b_ref[0, t, :]
        gt_t = gt_ref[0, t]
        decay = jnp.exp(-gt_t * g_s)
        M_prev = mu_p * I_p
        M = v_t[:, None] * k_t[None, :] + decay * M_prev
        I = b_t[:, None] * (k_t[None, :] ** 2) + (1.0 - decay) * Ip_s + decay * I_p
        mu = M / I
        out_t = jnp.sum(mu * (q_t * scale)[None, :], axis=1)     # [DV]
        o_ref[0, t, :] = out_t
        return (mu, I)

    mu0 = jnp.zeros((DV, DK), jnp.float32)
    I0 = jnp.full((DV, DK), Ip_s, jnp.float32)
    lax.fori_loop(0, T, body, (mu0, I0))


def _palimpsa_fwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size):
    B, T, H, DK0 = q.shape
    DV0 = v.shape[-1]
    n_chunks = (T + chunk_size - 1) // chunk_size
    BH = B * H
    # Triton-Pallas requires power-of-2 array dims for in-kernel ops; pad D_K/D_V.
    DK = _next_pow2(DK0)
    DV = _next_pow2(DV0)

    qf = _pad_last(jnp.transpose(q, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    kf = _pad_last(jnp.transpose(k, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    vf = _pad_last(jnp.transpose(v, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    bf = _pad_last(jnp.transpose(b, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    gtf = jnp.transpose(gt, (0, 2, 1)).reshape(BH, T)
    gf = jnp.broadcast_to(g.reshape(1, H), (B, H)).reshape(BH)
    Ipf = jnp.broadcast_to(Ip.reshape(1, H), (B, H)).reshape(BH)

    kernel = functools.partial(_fwd_kernel, T=T, DK=DK, DV=DV,
                               scale=float(scale), chunk_size=chunk_size, n_chunks=n_chunks)
    out_shapes = (
        jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
        jax.ShapeDtypeStruct((BH, n_chunks, DV, DK), jnp.float32),
        jax.ShapeDtypeStruct((BH, n_chunks, DV, DK), jnp.float32),
    )
    o, state_mu, state_I = pl.pallas_call(
        kernel,
        grid=(BH,),
        in_specs=[
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ],
        out_specs=[
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, n_chunks, DV, DK), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, n_chunks, DV, DK), lambda i: (i, 0, 0, 0)),
        ],
        out_shape=out_shapes,
        compiler_params=_TRITON,
    )(qf, kf, vf, bf, gtf, gf, Ipf)

    out = o.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    return out, (state_mu, state_I)


# =============================================================================
# Pallas BACKWARD kernel.  Grid = (B*H,).  Reverse adjoint recurrence.
# Carries (dM_next, dI_next) -- adjoints wrt the (M,I) state arriving from t+1.
# Recomputes per-token M_prev,I_prev from the saved chunk-boundary states.
# =============================================================================
def _bwd_kernel(do_ref, q_ref, k_ref, v_ref, b_ref, gt_ref, g_ref, Ip_ref,
                state_mu_ref, state_I_ref,
                dq_ref, dk_ref, dv_ref, db_ref, dgt_ref, dg_ref, dIp_ref,
                *, T, DK, DV, scale, chunk_size, n_chunks):
    g_s = g_ref[0]
    Ip_s = Ip_ref[0]

    def replay_one(jr, st, base):
        # advance forward one token tr=base+jr, returns updated (mu,I)
        mu_p, I_p = st
        tr = base + jr
        k_r = k_ref[0, tr, :]; v_r = v_ref[0, tr, :]; b_r = b_ref[0, tr, :]; gt_r = gt_ref[0, tr]
        dec = jnp.exp(-gt_r * g_s)
        M = v_r[:, None] * k_r[None, :] + dec * (mu_p * I_p)
        I = b_r[:, None] * (k_r[None, :] ** 2) + (1.0 - dec) * Ip_s + dec * I_p
        return (M / I, I)

    def body(rt, carry):
        dM_n, dI_n, dg_a, dIp_a = carry
        t = T - 1 - rt
        c = t // chunk_size
        base = c * chunk_size
        j = t - base                                   # position within chunk
        mu_entry = state_mu_ref[0, c, :, :]
        I_entry = state_I_ref[0, c, :, :]
        # state just BEFORE token t (after t-1): replay j tokens from chunk entry
        mu_bef, I_bef = lax.fori_loop(0, j, lambda jr, st: replay_one(jr, st, base),
                                      (mu_entry, I_entry))
        M_prev = mu_bef * I_bef
        I_prev = I_bef

        q_t = q_ref[0, t, :]; k_t = k_ref[0, t, :]; v_t = v_ref[0, t, :]; b_t = b_ref[0, t, :]
        gt_t = gt_ref[0, t]; do_t = do_ref[0, t, :]
        decay = jnp.exp(-gt_t * g_s)
        M = v_t[:, None] * k_t[None, :] + decay * M_prev
        I = b_t[:, None] * (k_t[None, :] ** 2) + (1.0 - decay) * Ip_s + decay * I_prev
        mu = M / I

        # out_t = sum_k mu * q_t * scale
        dmu_out = do_t[:, None] * (q_t * scale)[None, :]    # [DV,DK]
        dq_t = jnp.sum(do_t[:, None] * mu * scale, axis=0)  # [DK]

        dM_here = dM_n + dmu_out / I
        dI_here = dI_n + (-dmu_out * M / (I * I))

        d_outer_M = dM_here
        d_outer_I = dI_here
        d_decay = jnp.sum(dM_here * M_prev) + jnp.sum(dI_here * (I_prev - Ip_s))
        dIp_here = jnp.sum(dI_here * (1.0 - decay))

        dM_prev = dM_here * decay
        dI_prev = dI_here * decay

        dv_t = jnp.sum(d_outer_M * k_t[None, :], axis=1)              # [DV]
        db_t = jnp.sum(d_outer_I * (k_t[None, :] ** 2), axis=1)       # [DV]
        dk_t = jnp.sum(d_outer_M * v_t[:, None], axis=0) \
               + jnp.sum(d_outer_I * b_t[:, None], axis=0) * 2.0 * k_t  # [DK]

        ddecay_dgt = -g_s * decay
        ddecay_dg = -gt_t * decay
        dgt_t = d_decay * ddecay_dgt
        dg_inc = d_decay * ddecay_dg

        dq_ref[0, t, :] = dq_t
        dk_ref[0, t, :] = dk_t
        dv_ref[0, t, :] = dv_t
        db_ref[0, t, :] = db_t
        dgt_ref[0, t] = dgt_t
        return (dM_prev, dI_prev, dg_a + dg_inc, dIp_a + dIp_here)

    dM0 = jnp.zeros((DV, DK), jnp.float32)
    dI0 = jnp.zeros((DV, DK), jnp.float32)
    dM_fin, dI_fin, dg_acc, dIp_acc = lax.fori_loop(0, T, body, (dM0, dI0, 0.0, 0.0))
    # I_0 = Ip (broadcast over [DV,DK]); the adjoint wrt I_0 (== dI_fin after the
    # reverse loop passes token 0) is an additional contribution to dIp.
    # M_0 = 0 (independent of Ip), so dM_fin does not contribute.
    dIp_acc = dIp_acc + jnp.sum(dI_fin)
    dg_ref[0] = dg_acc
    dIp_ref[0] = dIp_acc


def _palimpsa_bwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size, state_mu, state_I, do):
    B, T, H, DK0 = q.shape
    DV0 = v.shape[-1]
    n_chunks = (T + chunk_size - 1) // chunk_size
    BH = B * H
    DK = _next_pow2(DK0)
    DV = _next_pow2(DV0)

    qf = _pad_last(jnp.transpose(q, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    kf = _pad_last(jnp.transpose(k, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    vf = _pad_last(jnp.transpose(v, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    bf = _pad_last(jnp.transpose(b, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    gtf = jnp.transpose(gt, (0, 2, 1)).reshape(BH, T)
    gf = jnp.broadcast_to(g.reshape(1, H), (B, H)).reshape(BH)
    Ipf = jnp.broadcast_to(Ip.reshape(1, H), (B, H)).reshape(BH)
    dof = _pad_last(jnp.transpose(do, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)

    kernel = functools.partial(_bwd_kernel, T=T, DK=DK, DV=DV,
                               scale=float(scale), chunk_size=chunk_size, n_chunks=n_chunks)
    out_shapes = (
        jax.ShapeDtypeStruct((BH, T, DK), jnp.float32),
        jax.ShapeDtypeStruct((BH, T, DK), jnp.float32),
        jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
        jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
        jax.ShapeDtypeStruct((BH, T), jnp.float32),
        jax.ShapeDtypeStruct((BH,), jnp.float32),
        jax.ShapeDtypeStruct((BH,), jnp.float32),
    )
    dq, dk, dv, db, dgt, dg_bh, dIp_bh = pl.pallas_call(
        kernel,
        grid=(BH,),
        in_specs=[
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1, n_chunks, DV, DK), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, n_chunks, DV, DK), lambda i: (i, 0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
        ],
        out_shape=out_shapes,
        compiler_params=_TRITON,
    )(dof, qf, kf, vf, bf, gtf, gf, Ipf, state_mu, state_I)

    dq = dq.reshape(B, H, T, DK).transpose(0, 2, 1, 3)[..., :DK0]
    dk = dk.reshape(B, H, T, DK).transpose(0, 2, 1, 3)[..., :DK0]
    dv = dv.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    db = db.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    dgt = dgt.reshape(B, H, T).transpose(0, 2, 1)
    dg = dg_bh.reshape(B, H).sum(axis=0)
    dIp = dIp_bh.reshape(B, H).sum(axis=0)
    return dq, dk, dv, db, dgt, dg, dIp


# =============================================================================
# custom_vjp public API
# =============================================================================
@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def palimpsa_attention(q, k, v, b, gt, g, Ip, scale=None, chunk_size=16):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, _ = _palimpsa_fwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size)
    return out


def _fwd(q, k, v, b, gt, g, Ip, scale, chunk_size):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, (state_mu, state_I) = _palimpsa_fwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size)
    return out, (q, k, v, b, gt, g, Ip, state_mu, state_I, scale)


def _bwd(scale, chunk_size, res, do):
    q, k, v, b, gt, g, Ip, state_mu, state_I, scale_eff = res
    return _palimpsa_bwd_pallas(q, k, v, b, gt, g, Ip, scale_eff, chunk_size, state_mu, state_I, do)


palimpsa_attention.defvjp(_fwd, _bwd)
