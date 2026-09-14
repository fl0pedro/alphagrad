# -*- coding: utf-8 -*-
"""Fast Palimpsa (chunked, isotropic-in-D_V read) ported to JAX Pallas-Triton.

PROVENANCE
----------
Ported from the PyTorch + Triton reference implementation in

    github.com/djo1996/Palimpsa, branch ``main``, downloaded 2026-09-14,
    file ``ops/fast_palimpsa/chunk_fast_palimpsa.py``
      (``fast_palimpsa_ref`` -- the differentiable contract,
       ``fast_palimpsa_vec`` -- the closed-form chunk-parallel form,
       ``_fp_state_kernel`` et al. -- the Triton kernels, ``CHUNK_C = 32``),
    file ``ops/fast_palimpsa/fused_recurrent_fast_palimpsa.py``
      (the per-token decode path, which is exact Palimpsa's own recurrence),
    file ``layers/palimpsa.py`` (``kernel="fast"``).

The upstream project is MIT licensed (see its README badge and LICENSE). This
port keeps the same algorithm and the same chunk size; the code is rewritten
for JAX/Pallas.

WHAT THE APPROXIMATION IS
-------------------------
Exact Palimpsa carries a full ``[D_V, D_K]`` numerator ``M`` and precision
``I`` and reads ``mu_t = M_t / I_t`` at every token. That read costs
``O(D_V * D_K)`` per token and it is a division, so it cannot use tensor
cores.

Fast Palimpsa keeps the state carried ACROSS chunk boundaries EXACT -- the
boundary update below is the same closed form the exact recurrence arrives at
-- and only approximates the read INSIDE a chunk of ``C = 32`` tokens:

  * the history (everything before the chunk) is read against the FROZEN
    boundary posterior ``mu_c = M_c / I_c``, with a per-key isotropic
    correction ``Ibar_c / Ibar_t`` for the precision that accumulated inside
    the chunk;
  * the local contribution (tokens inside the chunk) is read against a single
    isotropic precision ``Ibar_t in R^{D_K}``, obtained by collapsing ``I``
    over ``D_V`` (``Ibar_c = mean_v I_c``) and evolving it with the same
    dynamics using a ``D_V``-collapsed beta (``betabar_t = mean_v b_t``).

Everything in the chunk then becomes a matmul.

THE CHUNK, WRITTEN OUT
----------------------
With ``qs = q * scale``, ``lf_t = -gt_t * g``, ``clf = cumsum(lf)``::

    Dm[t,s]   = exp(clf_t - clf_s)  for s <= t, else 0
    cd_t      = exp(clf_t)
    Ibar_c    = mean_v I_c                                   [D_K]
    abar      = Ibar_c - Ip                                  [D_K]
    betabar_t = mean_v b_t                                   [C]
    a_t       = cd_t * abar + sum_{s<=t} Dm[t,s] betabar_s k_s^2
    Ibar_t    = Ip + a_t                                     [C, D_K]

    y_local   = ((qs / Ibar) k^T * Dm) v                     [C, D_V]
    mu_c      = M_c / I_c                                    [D_V, D_K]
    y_carry_t = cd_t * (mu_c @ (qs_t * Ibar_c / Ibar_t))      [C, D_V]
    y         = y_local + y_carry

    prodF     = cd_{C-1};   w_t = exp(clf_{C-1} - clf_t)
    M_{c+1}   = prodF M_c + (w v)^T k
    I_{c+1}   = prodF I_c + (1 - prodF) Ip + (w b)^T k^2

The last two lines are EXACT: they are what the token-by-token recurrence
``M_t = f_t M_{t-1} + v_t k_t^T``, ``I_t = f_t I_{t-1} + (1-f_t) Ip + b_t k_t^2``
sums to over the chunk.

API
---
    out = fast_palimpsa(q, k, v, b, gt, g, Ip, scale=None, chunk_size=32,
                        initial_M=None, initial_I=None, output_final_state=False)
      q,k : [B,T,H,D_K]   v,b : [B,T,H,D_V]   gt : [B,T,H]   g,Ip : [H]
      initial_M, initial_I : [B,H,D_V,D_K]  (None => M0 = 0, I0 = Ip)
      -> out : [B,T,H,D_V]   (plus final (M, I) when output_final_state)

The carry is (M, I), NOT upstream's (mu, I): ``EncCarry`` in
``alphagrad.approx.ppo`` holds M and I, and ``mu = M / I`` is one division
away, so passing mu would only add a round trip. Upstream's
``initial_mu_state``/``initial_I_state`` pair maps onto this as
``M0 = mu0 * I0``.

CARRY ROUND TRIP. Because the boundary state is exact and the chunk read
depends on nothing but (chunk inputs, boundary state), splitting a call at a
multiple of ``C`` and threading (M, I) gives EXACTLY the same outputs as one
call over the concatenation. ``tests/fast_palimpsa_kernel_test.py`` pins that
at tolerance zero.

T NOT A MULTIPLE OF C. The tail is zero padded, exactly as upstream's
``chunk_fast_palimpsa`` does: a padded step has ``gt = 0`` (so ``f = 1``, the
state passes through untouched) and ``k = v = b = 0`` (so it contributes
nothing), and the read is causal, so the padding cannot reach any real
position's output or any state a real position reads. It is a no-op, not an
approximation.

DIFFERENCES FROM THE UPSTREAM TRITON KERNEL, DELIBERATE
-------------------------------------------------------
  * Upstream tiles the ``[D_V, D_K]`` state by (BK, BV) over a 3-D grid and
    autotunes. One program here holds the whole ``[D_V, D_K]`` state, exactly
    as our exact port ``palimpsa_pallas`` already does, and the grid is
    ``(B*H,)``. Our head dims are small (``head_dim = embd_dim / num_heads``),
    so there is nothing to tile, and no cross-block accumulation is needed.
  * Upstream computes ``logf = log(f.clamp_min(1e-30))`` in its PyTorch
    reference but ``logf = -gt * g`` directly in its Triton kernel. We use the
    direct form everywhere -- it is the kernel being ported, and our
    ``gt = softplus(.) >= 0`` with ``g = softplus(.) ~ 0.05`` never gets near
    the clamp.
  * Upstream checkpoints the boundary trajectory in bf16 for M and fp32 for I.
    We keep both in fp32: our chunk counts are two orders of magnitude smaller
    (a delta is ~3000 tokens, not a 4096-token batch times 16 heads), so the
    memory is not the binding constraint and fp32 removes a question from the
    gradient comparison.
  * ``output_uncertainty`` (the ``yvar`` second return) is not ported. No
    caller in this repo reads it.
"""
from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

# Force the Triton GPU backend, for the same reason palimpsa_pallas does (the
# Mosaic-GPU backend rejects the per-head scalar g/Ip loads).
_TRITON = plt.CompilerParams(num_warps=4, num_stages=2)

#: Upstream's ``CHUNK_C``. The chunk size is part of the OPERATOR, not a
#: tuning knob: a different C is a different function, so a policy trained at
#: C=32 must be evaluated at C=32.
CHUNK_C = 32

#: Matmul operands below this width are not worth a tensor core and Triton
#: rejects some of them outright, so every padded dim is at least this wide.
_MIN_DIM = 16


def _next_pow2(n, floor=1):
    p = max(1, int(floor))
    while p < n:
        p *= 2
    return p


def _pad_last(x, target):
    d = x.shape[-1]
    if d == target:
        return x
    pad = [(0, 0)] * (x.ndim - 1) + [(0, target - d)]
    return jnp.pad(x, pad)


def _pad_state(M, I, Ip_bh, DV, DK):
    """Pad a ``[BH, DV0, DK0]`` (M, I) pair up to ``[BH, DV, DK]``.

    M pads with 0 and I pads with Ip, which is what the recurrence itself puts
    there: a padded key column has k = 0 so it never receives a b k^2 update,
    and ``I = f I + (1-f) Ip`` has Ip as its fixed point. Padding I with zeros
    instead would put a division by zero in ``mu = M / I`` and in
    ``Qtil = qs / Ibar``.
    """
    BH, DV0, DK0 = M.shape
    if (DV0, DK0) == (DV, DK):
        return M, I
    Mp = jnp.pad(M, ((0, 0), (0, DV - DV0), (0, DK - DK0)))
    Ib = Ip_bh.reshape(BH, 1, 1)
    Ip_full = jnp.broadcast_to(Ib, (BH, DV, DK))
    Ipad = jnp.pad(I, ((0, 0), (0, DV - DV0), (0, DK - DK0)))
    keep = ((jnp.arange(DV)[:, None] < DV0)
            & (jnp.arange(DK)[None, :] < DK0))
    return Mp, jnp.where(keep[None], Ipad, Ip_full)


# =============================================================================
# 1. Pure-JAX oracle.  Mirrors upstream ``fast_palimpsa_ref`` line for line,
#    with the carry interface added.  TEST ORACLE, not a live path.
# =============================================================================
def fast_palimpsa_ref(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                      initial_M=None, initial_I=None, output_final_state=False):
    """Pure-JAX Fast Palimpsa. The contract the Pallas kernel is pinned to.

    Written with a PYTHON loop over chunks and the closed-form intra-chunk
    algebra of upstream ``fast_palimpsa_vec`` (which upstream verified against
    its own token-loop ``fast_palimpsa_ref`` to machine precision in fp64).
    A python loop, not a ``lax.scan``, so the oracle cannot share a bug with
    the scan the kernel replaces.
    """
    q = jnp.asarray(q, jnp.float32); k = jnp.asarray(k, jnp.float32)
    v = jnp.asarray(v, jnp.float32); b = jnp.asarray(b, jnp.float32)
    gt = jnp.asarray(gt, jnp.float32)
    g = jnp.asarray(g, jnp.float32); Ip = jnp.asarray(Ip, jnp.float32)

    B, T, H, DK = q.shape
    DV = v.shape[-1]
    C = int(chunk_size)
    if T % C != 0:
        raise ValueError(
            f"fast_palimpsa_ref: T={T} is not a multiple of chunk_size={C}. "
            "The oracle takes whole chunks only; pad the tail with zeros "
            "(gt=0, k=v=b=0) the way fast_palimpsa() does before calling it.")
    nc = T // C
    if scale is None:
        scale = DK ** -0.5

    gh = g.reshape(1, H, 1, 1)
    Iph = Ip.reshape(1, H, 1, 1)
    M_c = (jnp.zeros((B, H, DV, DK), jnp.float32) if initial_M is None
           else jnp.asarray(initial_M, jnp.float32))
    I_c = (jnp.broadcast_to(Iph, (B, H, DV, DK)).astype(jnp.float32)
           if initial_I is None else jnp.asarray(initial_I, jnp.float32))

    tri = jnp.tril(jnp.ones((C, C), jnp.float32))           # [C,C] s <= t
    outs = []
    for c in range(nc):
        sl = slice(c * C, (c + 1) * C)
        qc = q[:, sl] * scale                                # [B,C,H,DK]
        kc = k[:, sl]; vc = v[:, sl]; bc = b[:, sl]
        lf = -(gt[:, sl] * g.reshape(1, 1, H))               # [B,C,H]
        clf = jnp.cumsum(lf, axis=1)
        cd = jnp.exp(clf)                                    # [B,C,H]
        # Dm[t,s] = exp(clf_t - clf_s), s <= t.  The masked-out half is
        # exp(positive) and can overflow, so mask the EXPONENT, not the
        # result: 0 * inf is nan, exp(-huge) is 0.
        dexp = clf[:, :, None, :] - clf[:, None, :, :]       # [B,C,C,H]
        dexp = jnp.where(tri[None, :, :, None] > 0, dexp, -1e30)
        Dm = jnp.exp(dexp)                                   # [B,C,C,H]

        Ibar_c = jnp.mean(I_c, axis=2)                       # [B,H,DK]
        abar = Ibar_c - Ip.reshape(1, H, 1)                  # [B,H,DK]
        betabar = jnp.mean(bc, axis=-1)                      # [B,C,H]
        ksq = kc * kc

        src = betabar[..., None] * ksq                       # [B,C,H,DK]
        a = (cd[..., None] * abar[:, None]
             + jnp.einsum('btsh,bshd->bthd', Dm, src))
        Ibar = Ip.reshape(1, 1, H, 1) + a                    # [B,C,H,DK]

        Qtil = qc / Ibar
        score = jnp.einsum('bthd,bshd->btsh', Qtil, kc) * Dm
        y_local = jnp.einsum('btsh,bshv->bthv', score, vc)

        mu_c = M_c / I_c                                     # [B,H,DV,DK]
        Qc = qc * (Ibar_c[:, None] / Ibar)
        base = jnp.einsum('bhvd,bthd->bthv', mu_c, Qc)
        outs.append(y_local + base * cd[..., None])

        prodF = cd[:, -1]                                    # [B,H]
        w = jnp.exp(clf[:, -1:] - clf)                       # [B,C,H]
        pf = prodF[:, :, None, None]
        M_c = pf * M_c + jnp.einsum('bthv,bthd->bhvd', w[..., None] * vc, kc)
        I_c = (pf * I_c + (1.0 - pf) * Iph
               + jnp.einsum('bthv,bthd->bhvd', w[..., None] * bc, ksq))

    out = jnp.concatenate(outs, axis=1)
    if output_final_state:
        return out, M_c, I_c
    return out


# =============================================================================
# 2. The chunk, once, in the vocabulary both Pallas kernels use.
#    Everything below operates on ONE (batch, head) program's tiles:
#      qs, k : [C, DK]   v, b : [C, DV]   gt : [C]   M, I : [DV, DK]
# =============================================================================
def _chunk_decays(gt_c, g_s, C):
    """``(clf, cd, Dm, w, prodF, tri)`` for one chunk.

    ``cumsum`` is spelled as a triangular mask-and-reduce rather than
    ``lax.cumsum``: Pallas-Triton has no lowering for the cumulative-sum
    primitive, and at C = 32 the [C,C] reduce is free.

    ``Dm`` masks the EXPONENT, not the product. ``exp(clf_t - clf_s)`` above
    the diagonal is ``exp(positive)`` and overflows to inf in fp32 as soon as
    a chunk decays hard; ``0 * inf`` is nan, so the strictly-upper half would
    poison the whole tile. Zeroing the exponent first makes it ``1 * 0``.
    """
    ar = jnp.arange(C)
    tri = (ar[:, None] >= ar[None, :])                       # [C,C] s <= t
    trif = tri.astype(jnp.float32)
    lf = -gt_c * g_s                                         # [C]
    clf = jnp.sum(trif * lf[None, :], axis=1)                # [C] == cumsum(lf)
    cd = jnp.exp(clf)
    dexp = clf[:, None] - clf[None, :]
    Dm = trif * jnp.exp(dexp * trif)                         # [C,C]
    clf_last = jnp.sum(lf)                                   # == clf[C-1]
    prodF = jnp.exp(clf_last)
    w = jnp.exp(clf_last - clf)                              # [C]
    return clf, cd, Dm, w, prodF, trif


def _chunk_forward(qs, kc, vc, bc, gt_c, g_s, Ip_s, M, I, vmaskf, DV0, C):
    """One chunk's forward. Returns the output tile and the next state.

    ``vmaskf`` is the real-vs-padded ``D_V`` row mask as floats and ``DV0``
    the real ``D_V``: the two ``D_V`` means (``Ibar_c`` and ``betabar``) must
    average over the REAL rows, or the pow2 padding would change the operator
    (the padded rows of ``I`` sit at ``Ip``, not at zero).
    """
    _clf, cd, Dm, w, prodF, _trif = _chunk_decays(gt_c, g_s, C)

    Ibar_c = jnp.sum(I * vmaskf[:, None], axis=0) / DV0       # [DK]
    abar = Ibar_c - Ip_s
    betabar = jnp.sum(bc, axis=1) / DV0                       # [C] (b pads are 0)
    ksq = kc * kc
    src = betabar[:, None] * ksq                              # [C,DK]
    a = cd[:, None] * abar[None, :] + jnp.dot(Dm, src)
    Ibar = Ip_s + a                                           # [C,DK]

    Qtil = qs / Ibar
    P = jnp.dot(Qtil, kc.T)                                   # [C,C]
    S = P * Dm
    y_local = jnp.dot(S, vc)                                  # [C,DV]

    mu = M / I
    Qc = qs * (Ibar_c[None, :] / Ibar)
    base = jnp.dot(Qc, mu.T)                                  # [C,DV]
    y = y_local + base * cd[:, None]

    M2 = prodF * M + jnp.dot((w[:, None] * vc).T, kc)
    I2 = prodF * I + (1.0 - prodF) * Ip_s + jnp.dot((w[:, None] * bc).T, ksq)
    return y, M2, I2


def _chunk_backward(qs, kc, vc, bc, gt_c, g_s, Ip_s, M, I, vmaskf, kvmaskf,
                    DV0, C, dy, dMn, dIn):
    """One chunk's VJP, derived term by term from :func:`_chunk_forward`.

    ``dMn``/``dIn`` are the cotangents of the state LEAVING the chunk;
    ``dM``/``dI`` returned are the cotangents of the state ENTERING it, which
    is what the reverse chunk sweep carries. ``dg_inc``/``dIp_inc`` are this
    chunk's contribution to the two per-head scalars.

    The forward quantities are RECOMPUTED here from the saved chunk-entry
    (M, I) rather than stored: one extra chunk forward buys back the whole
    per-chunk residual set, and the chunk forward is the cheap half.
    """
    _clf, cd, Dm, w, prodF, trif = _chunk_decays(gt_c, g_s, C)

    # ---- recompute the forward -------------------------------------------
    Ibar_c = jnp.sum(I * vmaskf[:, None], axis=0) / DV0
    abar = Ibar_c - Ip_s
    betabar = jnp.sum(bc, axis=1) / DV0
    ksq = kc * kc
    src = betabar[:, None] * ksq
    a = cd[:, None] * abar[None, :] + jnp.dot(Dm, src)
    Ibar = Ip_s + a
    Qtil = qs / Ibar
    P = jnp.dot(Qtil, kc.T)
    S = P * Dm
    mu = M / I
    Qc = qs * (Ibar_c[None, :] / Ibar)
    base = jnp.dot(Qc, mu.T)

    dMn = dMn * kvmaskf
    dIn = dIn * kvmaskf

    # ---- boundary update  M2 = prodF M + (w v)^T k,
    #                       I2 = prodF I + (1-prodF) Ip + (w b)^T k^2 -------
    dM = prodF * dMn
    dI = prodF * dIn
    A = jnp.dot(kc, dMn.T)                        # [C,DV] sum_d k[t,d] dMn[v,d]
    A2 = jnp.dot(ksq, dIn.T)                      # [C,DV]
    dv = w[:, None] * A
    db = w[:, None] * A2
    dk = w[:, None] * jnp.dot(vc, dMn)            # [C,DK]
    dksq = w[:, None] * jnp.dot(bc, dIn)          # [C,DK]
    dprodF = jnp.sum(dMn * M) + jnp.sum(dIn * (I - Ip_s))
    dIp_inc = (1.0 - prodF) * jnp.sum(dIn)
    dw = jnp.sum(vc * A, axis=1) + jnp.sum(bc * A2, axis=1)   # [C]

    # w_t = exp(clf_last - clf_t), prodF = exp(clf_last)
    dclf = -w * dw
    dclf_last = jnp.sum(w * dw) + prodF * dprodF
    dclf = dclf + jnp.where(jnp.arange(C) == C - 1, dclf_last, 0.0)

    # ---- output  y = y_local + base * cd ---------------------------------
    dbase = dy * cd[:, None]
    dcd = jnp.sum(dy * base, axis=1)                          # [C]

    # carry read: base = Qc @ mu^T
    dmu = jnp.dot(dbase.T, Qc)                                # [DV,DK]
    dQc = jnp.dot(dbase, mu)                                  # [C,DK]

    # Qc = qs * Ibar_c / Ibar
    dqs = dQc * (Ibar_c[None, :] / Ibar)
    dIbar_c = jnp.sum(dQc * qs / Ibar, axis=0)                # [DK]
    dIbar = -dQc * Qc / Ibar

    # mu = M / I
    dM = dM + dmu / I
    dI = dI - dmu * M / (I * I)

    # local read: y_local = S @ v, S = P * Dm, P = Qtil @ k^T
    dS = jnp.dot(dy, vc.T)                                    # [C,C]
    dv = dv + jnp.dot(S.T, dy)
    dP = dS * Dm
    dDm = dS * P
    dQtil = jnp.dot(dP, kc)                                   # [C,DK]
    dk = dk + jnp.dot(dP.T, Qtil)

    # Qtil = qs / Ibar
    dqs = dqs + dQtil / Ibar
    dIbar = dIbar - dQtil * Qtil / Ibar

    # Ibar = Ip + a
    da = dIbar
    dIp_inc = dIp_inc + jnp.sum(dIbar)

    # a = cd * abar + Dm @ (betabar * k^2)
    dcd = dcd + jnp.sum(da * abar[None, :], axis=1)
    dabar = jnp.sum(da * cd[:, None], axis=0)                 # [DK]
    dDm = dDm + jnp.dot(da, src.T)                            # [C,C]
    dsrc = jnp.dot(Dm.T, da)                                  # [C,DK]
    dbetabar = jnp.sum(dsrc * ksq, axis=1)                    # [C]
    dksq = dksq + betabar[:, None] * dsrc

    dk = dk + 2.0 * kc * dksq
    db = db + (dbetabar[:, None] / DV0) * vmaskf[None, :]

    # abar = Ibar_c - Ip ; Ibar_c = mean over the real D_V rows of I
    dIbar_c = dIbar_c + dabar
    dIp_inc = dIp_inc - jnp.sum(dabar)
    dI = dI + (dIbar_c[None, :] / DV0) * vmaskf[:, None]

    # Dm[t,s] = exp(clf_t - clf_s) on the lower triangle
    DD = dDm * Dm
    dclf = dclf + jnp.sum(DD, axis=1) - jnp.sum(DD, axis=0)
    dclf = dclf + cd * dcd

    # clf = cumsum(lf)  ->  dlf_t = sum_{u >= t} dclf_u
    dlf = jnp.sum(trif.T * dclf[None, :], axis=1)
    dgt = -g_s * dlf
    dg_inc = -jnp.sum(gt_c * dlf)

    dM = dM * kvmaskf
    dI = dI * kvmaskf
    return dqs, dk, dv, db, dgt, dM, dI, dg_inc, dIp_inc


# =============================================================================
# 3. Backend resolution (same three-way contract as palimpsa_pallas).
# =============================================================================
_BACKEND_LOGGED = [False]


def _resolve_backend() -> str:
    if os.environ.get("GRAPHAX_FAST_PALIMPSA_REF", "0") == "1":
        mode = "ref"
    elif jax.default_backend() == "cpu":
        mode = "interpret"
    else:
        mode = "kernel"
    if not _BACKEND_LOGGED[0]:
        _BACKEND_LOGGED[0] = True
        print("[fast-palimpsa] backend=%s jax_default_backend=%s"
              % (mode, jax.default_backend()), flush=True)
    return mode


def _interpret() -> bool:
    return _resolve_backend() == "interpret"


# =============================================================================
# 4. Pallas FORWARD kernel.  Grid = (B*H,).  One program holds [D_V, D_K].
# =============================================================================
def _fwd_kernel(q_ref, k_ref, v_ref, b_ref, gt_ref, g_ref, Ip_ref,
                M0_ref, I0_ref,
                o_ref, Mb_ref, Ib_ref, Mf_ref, If_ref,
                *, DK, DV, DK0, DV0, scale, C, nc):
    g_s = g_ref[0]
    Ip_s = Ip_ref[0]
    vmaskf = (jnp.arange(DV) < DV0).astype(jnp.float32)

    def body(c, carry):
        M, I = carry
        Mb_ref[0, c, :, :] = M
        Ib_ref[0, c, :, :] = I
        off = c * C
        qs = q_ref[0, pl.ds(off, C), :] * scale
        kc = k_ref[0, pl.ds(off, C), :]
        vc = v_ref[0, pl.ds(off, C), :]
        bc = b_ref[0, pl.ds(off, C), :]
        gt_c = gt_ref[0, pl.ds(off, C)]
        y, M2, I2 = _chunk_forward(qs, kc, vc, bc, gt_c, g_s, Ip_s,
                                   M, I, vmaskf, DV0, C)
        o_ref[0, pl.ds(off, C), :] = y
        return (M2, I2)

    M_fin, I_fin = lax.fori_loop(0, nc, body, (M0_ref[0], I0_ref[0]))
    Mf_ref[0, :, :] = M_fin
    If_ref[0, :, :] = I_fin


# =============================================================================
# 5. Pallas BACKWARD kernel.  Reverse chunk sweep from the saved boundaries.
# =============================================================================
def _bwd_kernel(do_ref, dMf_ref, dIf_ref,
                q_ref, k_ref, v_ref, b_ref, gt_ref, g_ref, Ip_ref,
                Mb_ref, Ib_ref,
                dq_ref, dk_ref, dv_ref, db_ref, dgt_ref, dg_ref, dIp_ref,
                dM0_ref, dI0_ref,
                *, DK, DV, DK0, DV0, scale, C, nc):
    g_s = g_ref[0]
    Ip_s = Ip_ref[0]
    vmaskf = (jnp.arange(DV) < DV0).astype(jnp.float32)
    kmaskf = (jnp.arange(DK) < DK0).astype(jnp.float32)
    kvmaskf = vmaskf[:, None] * kmaskf[None, :]

    def body(rc, carry):
        dMn, dIn, dg_a, dIp_a = carry
        c = nc - 1 - rc
        off = c * C
        M = Mb_ref[0, c, :, :]
        I = Ib_ref[0, c, :, :]
        qs = q_ref[0, pl.ds(off, C), :] * scale
        kc = k_ref[0, pl.ds(off, C), :]
        vc = v_ref[0, pl.ds(off, C), :]
        bc = b_ref[0, pl.ds(off, C), :]
        gt_c = gt_ref[0, pl.ds(off, C)]
        dy = do_ref[0, pl.ds(off, C), :]
        (dqs, dk, dv, db, dgt, dM, dI,
         dg_inc, dIp_inc) = _chunk_backward(
            qs, kc, vc, bc, gt_c, g_s, Ip_s, M, I, vmaskf, kvmaskf,
            DV0, C, dy, dMn, dIn)
        dq_ref[0, pl.ds(off, C), :] = dqs * scale
        dk_ref[0, pl.ds(off, C), :] = dk
        dv_ref[0, pl.ds(off, C), :] = dv
        db_ref[0, pl.ds(off, C), :] = db
        dgt_ref[0, pl.ds(off, C)] = dgt
        return (dM, dI, dg_a + dg_inc, dIp_a + dIp_inc)

    dM0, dI0, dg_acc, dIp_acc = lax.fori_loop(
        0, nc, body, (dMf_ref[0], dIf_ref[0], 0.0, 0.0))
    dM0_ref[0, :, :] = dM0
    dI0_ref[0, :, :] = dI0
    dg_ref[0] = dg_acc
    dIp_ref[0] = dIp_acc


# =============================================================================
# 6. Host wrappers
# =============================================================================
def _flatten(q, k, v, b, gt, g, Ip, M0, I0, DK, DV):
    B, T, H = q.shape[0], q.shape[1], q.shape[2]
    DK0 = q.shape[-1]
    DV0 = v.shape[-1]
    BH = B * H
    qf = _pad_last(jnp.transpose(q, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    kf = _pad_last(jnp.transpose(k, (0, 2, 1, 3)).reshape(BH, T, DK0), DK)
    vf = _pad_last(jnp.transpose(v, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    bf = _pad_last(jnp.transpose(b, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)
    gtf = jnp.transpose(gt, (0, 2, 1)).reshape(BH, T)
    gf = jnp.broadcast_to(g.reshape(1, H), (B, H)).reshape(BH)
    Ipf = jnp.broadcast_to(Ip.reshape(1, H), (B, H)).reshape(BH)
    if M0 is None:
        M0f = jnp.zeros((BH, DV, DK), jnp.float32)
        I0f = jnp.broadcast_to(Ipf.reshape(BH, 1, 1),
                               (BH, DV, DK)).astype(jnp.float32)
    else:
        M0f, I0f = _pad_state(M0.reshape(BH, DV0, DK0),
                              I0.reshape(BH, DV0, DK0), Ipf, DV, DK)
    return qf, kf, vf, bf, gtf, gf, Ipf, M0f, I0f


def _fast_fwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size, M0, I0):
    B, T, H, DK0 = q.shape
    DV0 = v.shape[-1]
    C = int(chunk_size)
    if T % C != 0:
        raise ValueError(
            f"_fast_fwd_pallas: T={T} must be a multiple of C={C}; "
            "fast_palimpsa() pads the tail before calling this.")
    nc = T // C
    BH = B * H
    DK = _next_pow2(DK0, _MIN_DIM)
    DV = _next_pow2(DV0, _MIN_DIM)

    qf, kf, vf, bf, gtf, gf, Ipf, M0f, I0f = _flatten(
        q, k, v, b, gt, g, Ip, M0, I0, DK, DV)

    kernel = functools.partial(_fwd_kernel, DK=DK, DV=DV, DK0=DK0,
                               DV0=DV0, scale=float(scale), C=C, nc=nc)
    o, Mb, Ib, Mfin, Ifin = pl.pallas_call(
        kernel,
        interpret=_interpret(),
        grid=(BH,),
        in_specs=[
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1, DV, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, DV, DK), lambda i: (i, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, nc, DV, DK), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, nc, DV, DK), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, DV, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, DV, DK), lambda i: (i, 0, 0)),
        ],
        out_shape=(
            jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
            jax.ShapeDtypeStruct((BH, nc, DV, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, nc, DV, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, DV, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, DV, DK), jnp.float32),
        ),
        compiler_params=_TRITON,
    )(qf, kf, vf, bf, gtf, gf, Ipf, M0f, I0f)

    out = o.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    Mf = Mfin.reshape(B, H, DV, DK)[..., :DV0, :DK0]
    If = Ifin.reshape(B, H, DV, DK)[..., :DV0, :DK0]
    return out, Mf, If, (Mb, Ib)


def _fast_bwd_pallas(q, k, v, b, gt, g, Ip, scale, chunk_size,
                     Mb, Ib, do, dMf, dIf):
    B, T, H, DK0 = q.shape
    DV0 = v.shape[-1]
    C = int(chunk_size)
    nc = T // C
    BH = B * H
    DK = _next_pow2(DK0, _MIN_DIM)
    DV = _next_pow2(DV0, _MIN_DIM)

    qf, kf, vf, bf, gtf, gf, Ipf, _m, _i = _flatten(
        q, k, v, b, gt, g, Ip, None, None, DK, DV)
    dof = _pad_last(jnp.transpose(do, (0, 2, 1, 3)).reshape(BH, T, DV0), DV)

    def _pad_kv(x):
        # A state cotangent pads with ZERO on both axes (unlike the state
        # itself, whose padded precision rows sit at Ip): the padded entries
        # are not outputs, so nothing may flow back through them.
        if x is None:
            return jnp.zeros((BH, DV, DK), jnp.float32)
        return jnp.pad(x.reshape(BH, DV0, DK0),
                       ((0, 0), (0, DV - DV0), (0, DK - DK0)))

    dMff = _pad_kv(dMf)
    dIff = _pad_kv(dIf)

    kernel = functools.partial(_bwd_kernel, DK=DK, DV=DV, DK0=DK0,
                               DV0=DV0, scale=float(scale), C=C, nc=nc)
    kv = pl.BlockSpec((1, DV, DK), lambda i: (i, 0, 0))
    outs = pl.pallas_call(
        kernel,
        interpret=_interpret(),
        grid=(BH,),
        in_specs=[
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),   # do
            kv, kv,                                          # dMf, dIf
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),   # q
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),   # k
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),   # v
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),   # b
            pl.BlockSpec((1, T), lambda i: (i, 0)),          # gt
            pl.BlockSpec((1,), lambda i: (i,)),              # g
            pl.BlockSpec((1,), lambda i: (i,)),              # Ip
            pl.BlockSpec((1, nc, DV, DK), lambda i: (i, 0, 0, 0)),
            pl.BlockSpec((1, nc, DV, DK), lambda i: (i, 0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DK), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T, DV), lambda i: (i, 0, 0)),
            pl.BlockSpec((1, T), lambda i: (i, 0)),
            pl.BlockSpec((1,), lambda i: (i,)),
            pl.BlockSpec((1,), lambda i: (i,)),
            kv, kv,
        ],
        out_shape=(
            jax.ShapeDtypeStruct((BH, T, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, T, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
            jax.ShapeDtypeStruct((BH, T, DV), jnp.float32),
            jax.ShapeDtypeStruct((BH, T), jnp.float32),
            jax.ShapeDtypeStruct((BH,), jnp.float32),
            jax.ShapeDtypeStruct((BH,), jnp.float32),
            jax.ShapeDtypeStruct((BH, DV, DK), jnp.float32),
            jax.ShapeDtypeStruct((BH, DV, DK), jnp.float32),
        ),
        compiler_params=_TRITON,
    )(dof, dMff, dIff, qf, kf, vf, bf, gtf, gf, Ipf, Mb, Ib)
    dq, dk, dv, db, dgt, dg_bh, dIp_bh, dM0, dI0 = outs

    dq = dq.reshape(B, H, T, DK).transpose(0, 2, 1, 3)[..., :DK0]
    dk = dk.reshape(B, H, T, DK).transpose(0, 2, 1, 3)[..., :DK0]
    dv = dv.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    db = db.reshape(B, H, T, DV).transpose(0, 2, 1, 3)[..., :DV0]
    dgt = dgt.reshape(B, H, T).transpose(0, 2, 1)
    dg = dg_bh.reshape(B, H).sum(axis=0)
    dIp = dIp_bh.reshape(B, H).sum(axis=0)
    dM0 = dM0.reshape(B, H, DV, DK)[..., :DV0, :DK0]
    dI0 = dI0.reshape(B, H, DV, DK)[..., :DV0, :DK0]
    return dq, dk, dv, db, dgt, dg, dIp, dM0, dI0


# =============================================================================
# 7. custom_vjp public API
# =============================================================================
@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def fast_palimpsa_attention(q, k, v, b, gt, g, Ip, M0, I0,
                            scale=None, chunk_size=CHUNK_C):
    """Fast Palimpsa over WHOLE chunks, with an explicit (M, I) carry.

    ``M0``/``I0`` are REQUIRED here (pass the ``Ip`` broadcast for a fresh
    state); ``fast_palimpsa`` below is the argument-defaulting front door.
    Returns ``(out, M_final, I_final)``.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, Mf, If, _res = _fast_fwd_pallas(q, k, v, b, gt, g, Ip, scale,
                                         chunk_size, M0, I0)
    return out, Mf, If


def _fp_fwd(q, k, v, b, gt, g, Ip, M0, I0, scale, chunk_size):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, Mf, If, (Mb, Ib) = _fast_fwd_pallas(q, k, v, b, gt, g, Ip, scale,
                                             chunk_size, M0, I0)
    return (out, Mf, If), (q, k, v, b, gt, g, Ip, Mb, Ib, scale)


def _fp_bwd(scale, chunk_size, res, cts):
    q, k, v, b, gt, g, Ip, Mb, Ib, scale_eff = res
    do, dMf, dIf = cts
    (dq, dk, dv, db, dgt, dg, dIp, dM0, dI0) = _fast_bwd_pallas(
        q, k, v, b, gt, g, Ip, scale_eff, chunk_size, Mb, Ib, do, dMf, dIf)
    return dq, dk, dv, db, dgt, dg, dIp, dM0, dI0


fast_palimpsa_attention.defvjp(_fp_fwd, _fp_bwd)


def fast_palimpsa(q, k, v, b, gt, g, Ip, scale=None, chunk_size=CHUNK_C,
                  initial_M=None, initial_I=None, output_final_state=False):
    """Fast Palimpsa front door. Pads the tail, defaults the carry, dispatches.

    ``GRAPHAX_FAST_PALIMPSA_REF=1`` forces the pure-JAX oracle. It is a
    cross-check and an escape hatch if a jaxlib/Triton skew breaks Pallas, not
    a training path. The resolved backend is logged once per process.
    """
    B, T, H, DK0 = q.shape
    DV0 = v.shape[-1]
    C = int(chunk_size)
    if C <= 0:
        raise ValueError(f"fast_palimpsa: chunk_size must be > 0, got {C}")
    if scale is None:
        scale = DK0 ** -0.5

    pad = (-T) % C
    if pad:
        def _p(x, n=pad):
            z = [(0, 0)] * x.ndim
            z[1] = (0, n)
            return jnp.pad(x, z)
        q, k, v, b, gt = (_p(x) for x in (q, k, v, b, gt))

    if initial_M is None:
        initial_M = jnp.zeros((B, H, DV0, DK0), jnp.float32)
    if initial_I is None:
        initial_I = jnp.broadcast_to(
            jnp.asarray(Ip, jnp.float32).reshape(1, H, 1, 1),
            (B, H, DV0, DK0)).astype(jnp.float32)

    if _resolve_backend() == "ref":
        out, Mf, If = fast_palimpsa_ref(
            q, k, v, b, gt, g, Ip, scale=scale, chunk_size=C,
            initial_M=initial_M, initial_I=initial_I, output_final_state=True)
    else:
        out, Mf, If = fast_palimpsa_attention(
            q, k, v, b, gt, g, Ip, initial_M, initial_I, scale, C)

    if pad:
        out = out[:, :T]
    if output_final_state:
        return out, Mf, If
    return out
