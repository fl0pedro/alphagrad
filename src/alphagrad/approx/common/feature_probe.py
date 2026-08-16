"""Read-only feature probes on the live PPO representation.

WHAT THIS IS FOR. The offline decodability campaign (``decode3..6_*.py``,
``CAMPAIGN_STATE.md``) asked whether the face representation the approximation
head reads -- ``[ctx_i || ctx_j || face_latent]`` -- carries the properties of
the block it is approximating. It answered NO for today's representation
(within-step R2 -0.13..+0.12 on all five targets) and YES once explicit axis
extents are concatenated (+0.513, the single largest main effect) with message
passing second (+0.192).

That campaign trained the WHOLE agent, palimpsa included, on the probe loss.
This module asks the different question the owner asked for: what does the
representation carry when palimpsa is trained by PPO's reward ALONE and the
probe is a passive decoder bolted on top? The two can disagree in either
direction -- a reward-trained encoder is under no pressure to keep anything the
probe wants, but it is also not being handed the answer.

THE ONE INVARIANT THIS FILE EXISTS TO ENFORCE: the probe must not train the
thing it is measuring. Every head here takes its input through
``jax.lax.stop_gradient``, so the probe loss has NO path to palimpsa, to the
vertex policy, or to the face head -- its cotangent reaches probe parameters
and nothing else. Add the probe loss to the total loss and PPO's numbers are
unchanged. That is checked directly in ``feature_probe_test.py`` by taking the
gradient of the probe loss with respect to the agent and asserting it is
exactly zero, which is a stronger statement than "the weight is small".

WHY R2 AND NOT LOSS. A raw MSE is unreadable across targets with different
scales (``ln_stored`` spans ~20 nats, ``n_diag`` is a small count). The offline
campaign reported WITHIN-STEP R2 -- variance explained after removing each
elimination step's mean -- because the per-step mean is trivially predictable
from the step index alone and would otherwise flatter every arm. The bars in
``BARS`` below are that same statistic, so an online number is directly
comparable to the offline one.
"""

from __future__ import annotations

import math
import os

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------- targets ---
# The five real targets, in a FIXED order -- the index is the wire format
# between the host callback and the device head, so appending is safe and
# reordering is not.
FACE_TARGETS = ("ln_factor", "n_diag", "n_comp", "ln_stored", "n_paired")

# Two STATIC controls. They depend only on the face's endpoint vertices, which
# the head is handed directly, so a working harness must score ~0.97 on them.
# The offline campaign got 0.96-0.98 in every arm; an online number far below
# that means the probe plumbing is broken, NOT that the representation is poor.
# Without these a 0.0 on the real targets is unfalsifiable.
FACE_CONTROLS = ("stat_ln_i", "stat_ln_j")

FACE_NAMES = FACE_TARGETS + FACE_CONTROLS
NFT = len(FACE_NAMES)

# Within-step test R2 of the offline `today_sizes` arm at step 3000
# (decode2_out/face_today_sizes.log; hardcoded identically in
# decode5_summary.py:18-25 and decode6_summary.py:32-33). This is the bar a
# representation must clear to be called decodable.
BARS = {
    "ln_factor": 0.600,
    "n_diag": 0.477,
    "n_comp": 0.557,
    "ln_stored": 0.671,
    "n_paired": 0.498,
}

# What today's PPO representation scored offline WITHOUT extents or message
# passing -- i.e. the null this online probe is expected to reproduce.
OFFLINE_NULL = {
    "ln_factor": -0.131,
    "n_diag": -0.137,
    "n_comp": 0.121,
    "ln_stored": 0.110,
    "n_paired": 0.066,
}


def _dims_of(st):
    return (list(getattr(st, "out_dims", ()) or ()),
            list(getattr(st, "primal_dims", ()) or ()))


def _logical(d):
    return max(int(getattr(d, "logical_size", 1) or 1), 1)


def _best_gnom(jaxpr, vertex, pair_k, row2pair, max_axes):
    """``ln_factor``'s pre-image: the largest legal DIAG blocking factor.

    Transcribed from ``decode4_data.best_gnom``. The gcd is over the equation's
    primal shapes at the paired axis, crossed with the per-face DIAG legality
    mask -- so this is the one target that depends on BOTH the shapes and the
    mask, which is why it was the hardest of the five offline.
    """
    eqn = jaxpr.eqns[vertex - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return 1
    out_shape = tuple(eqn.outvars[0].aval.shape)
    primal_shapes = [tuple(iv.aval.shape) for iv in eqn.invars
                     if hasattr(iv, "aval")]
    if not primal_shapes:
        return 1
    n_primal = min(len(ps) for ps in primal_shapes)
    best = 1
    for bi1 in range(len(out_shape)):
        n1 = int(out_shape[bi1])
        for bi2 in range(n_primal):
            try:
                i, j = row2pair(jaxpr, vertex, bi1, bi2)
            except Exception:
                continue
            if i == j or i >= max_axes or j >= max_axes:
                continue
            if not pair_k[i, j]:
                continue
            g = n1
            for ps in primal_shapes:
                if bi2 < len(ps):
                    g = math.gcd(g, int(ps[bi2]))
            best = max(best, g)
    return best


def face_targets_host(oracle, jaxpr, vertex, max_faces, max_axes,
                      ln_of_vidx=None, endpoints=None):
    """``(targets (F, NFT), extents (F, max_axes), n_faces)`` for one vertex.

    HOST ONLY, and unavoidably so: every target is read off a live graphax
    ``SparseTensor``, which ``probe_faces`` produces by re-tracing
    ``_eliminate_vertex`` under a Python recorder. There is no device form of
    this -- it is a property of the traced graph, not of any array.

    COST. This runs ONE ``probe_faces`` on top of the ``face_masks`` the
    callback already does, so it is +1 probe per step against the 4 the
    single-vertex oracle path already pays -- about +25% of the oracle's own
    time, which is why the caller keeps it behind a default-off flag. It
    deliberately does NOT reuse ``face_masks``' internal probe: that method is
    on the live training path and the plan's own rule for it is that it "must
    not be perturbed".
    """
    from alphagrad.approx.env import diag_row_to_pair as _row2pair

    F, N = int(max_faces), int(max_axes)
    tgt = np.zeros((F, NFT), np.float32)
    ext = np.zeros((F, N), np.float32)
    vertex = int(vertex)
    try:
        pair, comp, n_faces = oracle.face_masks(vertex, F)
        faces = oracle.probe_faces(vertex, approx=True)
    except Exception:
        return tgt, ext, 0
    if not faces:
        return tgt, ext, 0
    n_faces = min(int(n_faces), len(faces), F)

    for k in range(n_faces):
        st = faces[k]
        od, pd = _dims_of(st)
        dims = od + pd
        val = getattr(st, "val", None)
        shp = tuple(getattr(val, "shape", ()) or ())
        stored = 1
        for s in shp:
            stored *= int(s)
        n_paired = sum(1 for d in dims
                       if getattr(d, "other_id", None) is not None)
        gnom = _best_gnom(jaxpr, vertex, pair[k], _row2pair, N)

        tgt[k, 0] = math.log2(max(gnom, 1))            # ln_factor
        tgt[k, 1] = float(pair[k].sum() * 0.5)         # n_diag
        tgt[k, 2] = float(comp[k].sum())               # n_comp
        tgt[k, 3] = math.log2(max(stored, 1))          # ln_stored
        tgt[k, 4] = float(n_paired)                    # n_paired
        if ln_of_vidx is not None and endpoints is not None:
            ki, kj = int(endpoints[k][0]), int(endpoints[k][1])
            n_ln = len(ln_of_vidx)
            tgt[k, 5] = ln_of_vidx[ki] if 0 <= ki < n_ln else 0.0
            tgt[k, 6] = ln_of_vidx[kj] if 0 <= kj < n_ln else 0.0
        for a, d in enumerate(dims[:N]):
            ext[k, a] = math.log2(_logical(d))
    return tgt, ext, n_faces


# ------------------------------------------------------------------ heads ---
class Probe(eqx.Module):
    """One MLP decoder. Its input arrives already ``stop_gradient``-ed.

    Depth 2 and the same width as the offline ``Readout`` so an online number
    is comparable to the offline one rather than confounded by capacity.
    """

    mlp: eqx.nn.MLP
    n_out: int = eqx.field(static=True)

    def __init__(self, n_in, n_out, width, key):
        self.mlp = eqx.nn.MLP(n_in, n_out, width, depth=2, key=key)
        self.n_out = int(n_out)

    def __call__(self, x):
        return self.mlp(x)


class FeatureProbes(eqx.Module):
    """The probe bundle: one face head, one vertex head.

    Kept in its OWN module rather than as fields on ``Agent`` so that the PPO
    optimiser's parameter tree is untouched when probes are off, and so that
    "which parameters does the probe loss own" is answerable by pointing at a
    subtree instead of by reading the loss.
    """

    face: Probe
    vertex: Probe
    use_extents: bool = eqx.field(static=True)

    def __init__(self, embd_dim, max_axes, n_vertex_out, width, key,
                 use_extents=False):
        kf, kv = jax.random.split(key)
        # [ctx_i || ctx_j || face_latent] -- exactly the offline probe's input,
        # and exactly what the face head itself reads today.
        din = 3 * int(embd_dim) + (int(max_axes) if use_extents else 0)
        self.face = Probe(din, NFT, width, kf)
        self.vertex = Probe(int(embd_dim), int(n_vertex_out), width, kv)
        self.use_extents = bool(use_extents)

    def face_predict(self, ctx_i, ctx_j, latent, extents=None):
        parts = [ctx_i, ctx_j, latent]
        if self.use_extents:
            if extents is None:
                raise ValueError(
                    "FeatureProbes was built with use_extents=True but no "
                    "extents were passed; the head's input width would not "
                    "match. Pass the per-face axis sizes or rebuild with "
                    "use_extents=False.")
            parts.append(extents)
        return self.face(jax.lax.stop_gradient(jnp.concatenate(parts)))

    def vertex_predict(self, ctx):
        return self.vertex(jax.lax.stop_gradient(ctx))


# ------------------------------------------------------------------- loss ---
def masked_mse(pred, target, valid):
    """Mean squared error over VALID rows only, per output column.

    The denominator is the valid count, not the padded row count -- the same
    mistake that made ``approx_prob`` read 99.2% when it was measuring the
    padding ratio (``project_face_telemetry_padding_denominator``). Returns
    zero, not NaN, when nothing is valid.
    """
    w = jnp.asarray(valid, jnp.float32)[:, None]
    se = w * (pred - target) ** 2
    denom = jnp.maximum(w.sum(), 1.0)
    return se.sum(0) / denom


def within_step_r2(pred, target, valid, step_id, n_steps):
    """Within-elimination-step R2, per output column -- the offline statistic.

    Removing each step's mean is what makes this a fair number: the raw R2
    would credit the probe for predicting the step-to-step drift, which the
    step index alone gives away. Per-vertex signal is CONSTANT inside a step
    group and so contributes exactly 0 here, by construction.
    """
    w = jnp.asarray(valid, jnp.float32)[:, None]
    oh = jnn.one_hot(step_id, n_steps, dtype=jnp.float32) * w   # (n, S)
    cnt = jnp.maximum(oh.sum(0), 1.0)                           # (S,)

    def _center(z):
        mu = (oh.T @ (z * w)) / cnt[:, None]                    # (S, C)
        return (z - oh @ mu) * w

    zt, zp = _center(target), _center(pred)
    ss_res = ((zt - zp) ** 2).sum(0)
    ss_tot = (zt ** 2).sum(0)
    # A column with no within-step variance is undefined, not 1.0: report 0 so
    # a degenerate target cannot masquerade as a perfect decode.
    return jnp.where(ss_tot > 1e-8, 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8),
                     0.0)


def normalise(target, mean, std):
    """Z-score with a floored std, so a constant column cannot blow up."""
    return (target - mean) / jnp.maximum(std, 1e-3)


PROBE_ON = os.environ.get("ALPHAGRAD_FEATURE_PROBE", "0") != "0"
PROBE_EXTENTS = os.environ.get("ALPHAGRAD_FEATURE_PROBE_EXTENTS", "0") != "0"
PROBE_WIDTH = int(os.environ.get("ALPHAGRAD_FEATURE_PROBE_WIDTH", "128"))
