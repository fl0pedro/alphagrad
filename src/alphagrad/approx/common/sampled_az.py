"""Sampled AlphaZero (Hubert et al. 2021) arithmetic for the composite
(vertex, face-sequence) action, shared between `az_gumbel.py` and its tests.

`az_gumbel.py` is a SCRIPT (it builds its env at import time), so everything
here is deliberately import-light and env-free: the numpy target helpers are
pure functions of Q estimates, and the one jax function (`face_ce_term`)
takes every dependency -- including the replay callable -- as an argument.
Tests exercise THESE functions, i.e. the exact code the trainer runs, instead
of mirrored copies that can drift (the historical failure mode of
test_gumbel_search.py's local re-implementations).

Design (agreed with the owner, 2026-08):
  * macro action a = (v, F): vertex v + the face-decision sequence F,
    pi_theta(a|s) = pi_ve(v|s) * beta_theta(F|s, v), log beta = sum_f lp_f.
  * per-vertex Q from K i.i.d. face-sequence draws F_k ~ beta:
        q(v) = sum_k w_k q_k / sum_k w_k,   w_k = rho_k * exp(sigma(q_k)),
    with sigma the usual Gumbel-AZ Q-scaling over the vertex's own draws and
    rho_k = pi/beta == 1 while we sample from the head we train
    (`rho_from_beta_temp` asserts that; a future proposal temperature must
    implement the importance correction before it may be != 1).
  * loss (replaces the old unclipped off-policy REINFORCE face term):
        L = - sum_v pi'_ve(v) log softmax(vertex_logits)_v
            - lambda_f * sum_v pi'_ve(v) sum_k w_hat_{v,k} log beta(F_{v,k})
    with pi'_ve the Gumbel completed-Q improved policy and w_hat the
    normalized draw weights (`face_ce_term`).
  * #93: every Q estimate entering ONE target must come from the SAME rollout
    depth (`assert_same_depth`); the halving budget widens (more draws per
    surviving vertex) instead of deepening unless ALPHAGRAD_GAZ_DEEPEN=1
    (`phase_plan`).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "assert_same_depth", "draw_weights", "face_ce_term", "improved_policy",
    "phase_plan", "qscale_sigma", "rho_from_beta_temp", "weighted_q",
]


def qscale_sigma(q, max_n=1, cvisit=50.0, cscale=0.1):
    """Danihelka et al. 2022 monotone Q-transform (mctx qtransform form):
    min-max-normalize Q over the given set, scaled by
    (c_visit + max_N) * c_scale. THE one sigma -- `az_gumbel.sigma` delegates
    here, and the draw weights below use the same transform, so the search
    and the loss cannot disagree on the Q-scaling."""
    q = np.asarray(q, dtype=np.float64)
    lo, hi = float(q.min()), float(q.max())
    qn = (q - lo) / max(hi - lo, 1e-8)
    return (float(cvisit) + float(max_n)) * float(cscale) * qn


def rho_from_beta_temp(beta_temp):
    """rho = pi/beta, the importance ratio between the trained face head and
    the proposal the draws were sampled from. We sample from the head we
    train, so rho == 1 identically -- and this function HARD-FAILS on any
    other temperature: a future proposal temperature (beta != pi) must
    implement the per-draw importance correction before it may bias the
    target. Plumbed via ALPHAGRAD_GAZ_BETA_TEMP (default 1.0)."""
    bt = float(beta_temp)
    if bt != 1.0:
        raise AssertionError(
            f"ALPHAGRAD_GAZ_BETA_TEMP={bt} != 1.0: rho = pi/beta would not be "
            "1 and the weighted-Q target would be silently biased. Sampling "
            "from a tempered proposal requires implementing the per-draw "
            "importance ratio rho_k first.")
    return 1.0


def draw_weights(qs, rho=1.0, cvisit=50.0, cscale=0.1):
    """Per-draw weights over ONE vertex's K face-sequence draws.

    w_k = rho_k * exp(sigma(q_k)) with sigma over the vertex's own draw set
    (max_N = K). Returns ``(w, w_hat)`` with ``w_hat`` normalized to sum 1.
    All sigma(q) equal (e.g. K == 1, or identical q) => uniform w_hat, so the
    weighted Q reduces to the plain mean.
    """
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    if qs.size == 0:
        raise ValueError("draw_weights needs at least one draw")
    w = float(rho) * np.exp(qscale_sigma(qs, max_n=qs.size,
                                         cvisit=cvisit, cscale=cscale))
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise FloatingPointError(f"degenerate draw weights: {w}")
    return w, w / s


def weighted_q(qs, rho=1.0, cvisit=50.0, cscale=0.1):
    """q(v) = sum_k w_k q_k / sum_k w_k over the vertex's draws."""
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    _w, w_hat = draw_weights(qs, rho=rho, cvisit=cvisit, cscale=cscale)
    return float(np.sum(w_hat * qs))


def assert_same_depth(depths, context=""):
    """#93: Q estimates from different rollout depths must never mix inside
    one target (a 200:1 spread in the completed-Q target was attributable to
    depth alone). Call on the depths of every estimate entering a target."""
    ds = {int(d) for d in depths}
    if len(ds) > 1:
        raise AssertionError(
            f"depth-mixed target{' (' + context + ')' if context else ''}: "
            f"Q estimates from rollout depths {sorted(ds)} entered one "
            f"target; all estimates in a target must share one depth (#93)")


def improved_policy(logits, comp_q, max_n=1, cvisit=50.0, cscale=0.1):
    """Gumbel-AZ improved policy: softmax(logits + sigma(completed_q)) over
    the legal set. ``comp_q`` must already be completed (v_mix on unvisited
    actions) and depth-consistent (`assert_same_depth`)."""
    logits = np.asarray(logits, dtype=np.float64)
    pi = logits + qscale_sigma(comp_q, max_n=max_n, cvisit=cvisit,
                               cscale=cscale)
    pi = np.exp(pi - pi.max())
    return pi / pi.sum()


def phase_plan(m, deepen=False, rollout_depth=0):
    """Sequential-halving schedule: ``[(survivors, new_draws_per_survivor,
    rollout_depth)]`` per evaluation round, mirroring the search loop's
    ceil-halving (Karnin 2013; strictly shrinking).

    Default (#93 fix): every round is depth 0 (pure value bootstrap) and the
    halving budget WIDENS -- round p >= 1 gives each survivor 2**p new
    face-sequence draws, the widening analogue of the old 2x deepening.
    ``deepen=True`` (ALPHAGRAD_GAZ_DEEPEN=1) restores depth-doubling from
    ``rollout_depth`` with ONE evaluation per round; its targets must then be
    built from a single common depth (round 0), never mixed.
    """
    out = []
    n = max(int(m), 1)
    p = 0
    while True:
        if deepen:
            draws, depth = 1, int(rollout_depth) * (2 ** p)
        else:
            draws, depth = (1 if p == 0 else 2 ** p), 0
        out.append((n, draws, depth))
        if n <= 1:
            break
        keep = max(1, -(-n // 2))          # ceil(n/2)
        if keep >= n:                      # guard: must strictly shrink
            keep = n - 1
        n = keep
        if n <= 1:
            break
        p += 1
    return out


def face_ce_term(face_replay_fn, ctx, enc_carry, axis_state, axis_valid,
                 fact_tables, op_override, axis_feats_fn, pi_pad,
                 sd_li, sd_vidx, sd_w, sd_fpair, sd_fcomp, sd_fvalid,
                 sd_cnt, sd_dt, sd_de, sd_fa, sd_fends):
    """The Sampled-AZ face cross-entropy for ONE decision:

        CE = - sum_v pi'_ve(v) sum_k w_hat_{v,k} log beta_theta(F_{v,k})

    over the search's stored draws, flattened to D slots (padding has
    ``sd_li == -1`` / ``sd_w == 0`` and contributes exactly 0). ``log beta``
    is recomputed with gradient through ``face_replay_fn`` (the caller passes
    ``agent._face_replay`` -- "Gradient reaches palimpsa through this scan"),
    scored off each draw's stored emission window against the REPLAYED
    encoding carry ``enc_carry``, exactly as the draws were sampled.

    Returns ``(ce, mean_entropy)`` with the entropy arity-normalised per draw
    and averaged over the real (non-padding) draws, for the entropy bonus and
    the entropy/approx_head telemetry.
    """
    import jax
    import jax.numpy as jnp

    nv = ctx.shape[0]

    def _one(vidx, fp, fc, fv, cnt, dt, de, fa_k, fend):
        vi = jnp.clip(vidx, 0, nv - 1)
        features = axis_feats_fn(axis_state[vi], axis_valid[vi])
        # The FULL per-vertex contexts, not the central vertex's row: the
        # face head reads its own two ENDPOINT vertices' contexts, gathered
        # inside `_face_replay` from the draw's stored endpoint ids.
        lp, ent, ar = face_replay_fn(
            ctx, features, fact_tables, fa_k, fp, fc, fv,
            enc_carry, (cnt, dt, de), op_override, face_ends=fend)
        return lp, ent / jnp.maximum(ar, 1.0)

    lp, ent_n = jax.vmap(_one)(
        sd_vidx, sd_fpair, sd_fcomp, sd_fvalid, sd_cnt, sd_dt, sd_de, sd_fa,
        sd_fends)
    li = jnp.clip(sd_li, 0, pi_pad.shape[0] - 1)
    valid = (sd_li >= 0) & (sd_w > 0)
    w = jnp.where(valid, sd_w * pi_pad[li], 0.0)
    ce = -jnp.sum(jnp.where(valid, w * lp, 0.0))
    n_real = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    mean_ent = jnp.sum(jnp.where(valid, ent_n, 0.0)) / n_real
    return ce, mean_ent


def mult_gate_scalar(raw4, gate_tau, gate_w, anti_degen_penalty,
                     anti_degen_tau, cost_weights=(1.0, 1.0, 0.0)):
    """PPO ``_apply_mult_gate`` parity for az's raw [lat, peak, flops, cos].

    g(cos) = clip((cos - tau)/(1 - tau), 0, 1); cheapness =
    max(0, W - sum_c w_c * log1p(cost_c)); scalar = g * cheapness. Below
    ``anti_degen_tau`` the shaped penalty -(P - cos*P) replaces the flat 0 so
    the destroyed basin keeps a positive slope in cos. Costs arrive RAW
    POSITIVE (native units); PPO's env layout stores them negated, which is
    the only difference the parity test bridges.
    """
    import numpy as _np
    r = _np.asarray(raw4, dtype=_np.float64)
    cos = float(_np.clip(r[3], 0.0, 1.0))
    g = float(_np.clip((cos - gate_tau) / max(1.0 - gate_tau, 1e-6), 0.0, 1.0))
    w = _np.asarray(cost_weights, dtype=_np.float64)
    cheap = max(0.0, gate_w - float(_np.sum(w * _np.log1p(_np.abs(r[:3])))))
    scal = g * cheap
    if cos < anti_degen_tau:
        fid_basin = min(max(cos, 0.0), anti_degen_tau)
        scal = -(anti_degen_penalty - fid_basin * anti_degen_penalty)
    return float(scal)
