"""Read-only feature probes on the LEAN policy representation.

THE QUESTION, in wip.dot's words: "Can this representation still READ vertex
and face information?" Nothing currently in flight measures it, and the prior
decodability numbers DO NOT CARRY OVER:

  * the vertex R2 of 0.93 was measured on the SPLIT 64-wide representation;
    the lean pointer collapsed slots 64 -> 32 (identity||dynamic concat and
    ctx_proj deleted), so it is a different input;
  * every face arm ever tested was fed ctx_i||ctx_j||latent (96 wide). The
    lean head is MLP(32 -> 94) on the face-keyed scatter ALONE -- verified in
    source: UnifiedFacePolicy builds UnifiedFaceHead(embd_dim, in_dim=embd_dim)
    and `_repr` returns the face latent unchanged, E wide;
  * so the lean face input is STRICTLY LESS than any arm previously measured.
    The nearest tested configuration is tokens-only, which scored about zero.

That last point is why this is worth running rather than assuming. A drop is
expected; its SIZE is the number that decides whether the lean face head can
see what it is approximating at all.

DEFAULT ARM = WHAT THE POLICY ACTUALLY READS. `FaceProbeArm.LEAN` feeds the
probe exactly the head's own input and nothing else. The richer arms exist
only as PAIRED REFERENCE ROWS, so a lean number is read against the same
targets and the same split rather than against a remembered figure from a
different architecture:

  LEAN      face latent                       (32)  <- the live head's input
  ENDPOINTS ctx_i || ctx_j || face latent     (96)  <- every prior face arm
  EXTENTS   LEAN + log2 axis sizes            (32+N) <- oracle produces these
                                                       but they are NOT fed
                                                       ("remove for now")

THE ONE INVARIANT THIS FILE ENFORCES: the probe must not train the thing it
measures. Every head takes its input through `jax.lax.stop_gradient`, so the
probe loss has NO path to palimpsa, the pointer, or the face head -- its
cotangent reaches probe parameters and nothing else. That is asserted
directly in feature_probe_test.py by differentiating the probe loss w.r.t.
the representation and requiring EXACTLY zero, which is stronger than "the
weight is small".

WHY THE SCATTER MAKES THIS CHEAP. The participation scatter is
parameter-free -- segment_mean(rows, ids) -> (V+2, E) keyed by vertex, and
the SAME op keyed by face gives the face latent. So both probes read two
keyings of one array; there is no separate machinery to build for the face
side.
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
FACE_TARGETS = ("ln_factor", "n_diag", "n_comp", "ln_stored", "n_paired")

# Two STATIC controls, recoverable from the endpoint vertex ids alone. A
# working harness scores ~0.97 on them (offline: 0.96-0.98 in EVERY arm).
# Without them a 0.0 on the real targets is unfalsifiable -- it could equally
# mean the plumbing is broken.
FACE_CONTROLS = ("stat_ln_i", "stat_ln_j")

FACE_NAMES = FACE_TARGETS + FACE_CONTROLS
NFT = len(FACE_NAMES)

# Within-step test R2 of the offline `today_sizes` arm at step 3000
# (decode2_out/face_today_sizes.log; same numbers in decode5_summary.py:18-25
# and decode6_summary.py:32-33). NOTE these were measured on the ENDPOINTS
# arm plus extents, NOT on the lean input -- they are a reference line, not a
# target the lean arm is expected to reach.
BARS = {
    "ln_factor": 0.600, "n_diag": 0.477, "n_comp": 0.557,
    "ln_stored": 0.671, "n_paired": 0.498,
}

# What the ENDPOINTS arm scored offline WITHOUT extents or message passing.
# The lean arm removes ctx_i/ctx_j on top of this, so it should land at or
# below these.
OFFLINE_ENDPOINTS_NULL = {
    "ln_factor": -0.131, "n_diag": -0.137, "n_comp": 0.121,
    "ln_stored": 0.110, "n_paired": 0.066,
}


class FaceProbeArm:
    LEAN = "lean"            # face latent alone -- the live head's input
    ENDPOINTS = "endpoints"  # ctx_i || ctx_j || latent -- every prior arm
    EXTENTS = "extents"      # lean + explicit log2 axis sizes
    ALL = (LEAN, ENDPOINTS, EXTENTS)


def face_input_dim(arm, embd_dim, max_axes):
    if arm == FaceProbeArm.LEAN:
        return int(embd_dim)
    if arm == FaceProbeArm.ENDPOINTS:
        return 3 * int(embd_dim)
    if arm == FaceProbeArm.EXTENTS:
        return int(embd_dim) + int(max_axes)
    raise ValueError("unknown face probe arm %r (expected one of %r)"
                     % (arm, FaceProbeArm.ALL))


def _dims_of(st):
    return (list(getattr(st, "out_dims", ()) or ()),
            list(getattr(st, "primal_dims", ()) or ()))


def _logical(d):
    return max(int(getattr(d, "logical_size", 1) or 1), 1)


def _best_gnom(jaxpr, vertex, pair_k, row2pair, max_axes):
    """``ln_factor``'s pre-image: the largest legal DIAG blocking factor.

    Transcribed from decode4_data.best_gnom. The gcd runs over the equation's
    primal shapes at the paired axis, crossed with the per-face DIAG legality
    mask -- so it depends on BOTH the shapes and the mask, which is why it was
    the hardest of the five offline.
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
            if i == j or i >= max_axes or j >= max_axes or not pair_k[i, j]:
                continue
            g = n1
            for ps in primal_shapes:
                if bi2 < len(ps):
                    g = math.gcd(g, int(ps[bi2]))
            best = max(best, g)
    return best


# Host-side failure telemetry for `face_targets_host`. A failing oracle
# used to be swallowed into "return 0 faces", which is indistinguishable from
# a vertex that genuinely has none -- a diagnostic that converts errors into
# zeros is worse than no diagnostic. The caller prints the count once per
# episode; ALPHAGRAD_FEATURE_PROBE_DEBUG=1 re-raises instead.
TARGET_FAILS = [0]
LAST_TARGET_ERR = [""]
PROBE_DEBUG = os.environ.get("ALPHAGRAD_FEATURE_PROBE_DEBUG", "0") != "0"


def face_targets_host(oracle, jaxpr, vertex, max_faces, max_axes,
                      ln_of_vidx=None, endpoints=None):
    """``(targets (F, NFT), extents (F, max_axes), n_faces)`` for one vertex.

    HOST ONLY, unavoidably: every target is read off a live graphax
    SparseTensor, which `probe_faces` produces by RE-TRACING _eliminate_vertex
    under a Python recorder. There is no device form -- it is a property of
    the traced graph, not of any array.

    COST: one `probe_faces` on top of the `face_masks` the oracle callback
    already runs, i.e. +1 probe per step against the 4 the single-vertex path
    already pays. That is why the caller keeps this behind a default-off flag.
    It deliberately does NOT reuse face_masks' internal probe: that method is
    on the live training path and must not be perturbed.

    `extents` is returned even though the policy does not consume them -- the
    oracle already computes them ("remove for now" was about FEEDING them, not
    about computing them), so the EXTENTS reference arm costs nothing extra.
    """
    from alphagrad.approx.env import diag_row_to_pair as _row2pair

    F, N = int(max_faces), int(max_axes)
    tgt = np.zeros((F, NFT), np.float32)
    ext = np.zeros((F, N), np.float32)
    vertex = int(vertex)
    try:
        pair, comp, n_faces = oracle.face_masks(vertex, F)
        faces = oracle.probe_faces(vertex, approx=True)
    except Exception as _exc:
        if PROBE_DEBUG:
            raise
        TARGET_FAILS[0] += 1
        LAST_TARGET_ERR[0] = "%s: %s" % (type(_exc).__name__, str(_exc)[:160])
        return tgt, ext, 0
    if not faces:
        return tgt, ext, 0
    n_faces = min(int(n_faces), len(faces), F)

    for k in range(n_faces):
        st = faces[k]
        od, pd = _dims_of(st)
        dims = od + pd
        shp = tuple(getattr(getattr(st, "val", None), "shape", ()) or ())
        stored = 1
        for s in shp:
            stored *= int(s)
        n_paired = sum(1 for d in dims
                       if getattr(d, "other_id", None) is not None)
        gnom = _best_gnom(jaxpr, vertex, pair[k], _row2pair, N)

        tgt[k, 0] = math.log2(max(gnom, 1))        # ln_factor
        tgt[k, 1] = float(pair[k].sum() * 0.5)     # n_diag
        tgt[k, 2] = float(comp[k].sum())           # n_comp
        tgt[k, 3] = math.log2(max(stored, 1))      # ln_stored
        tgt[k, 4] = float(n_paired)                # n_paired
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
    """One MLP decoder. Its input arrives already stop_gradient-ed.

    Depth 2 and the offline Readout's width, so a lean number is comparable to
    the reference rows rather than confounded by capacity.
    """

    mlp: eqx.nn.MLP

    def __init__(self, n_in, n_out, width, key):
        self.mlp = eqx.nn.MLP(n_in, n_out, width, depth=2, key=key)

    def __call__(self, x):
        return self.mlp(jax.lax.stop_gradient(x))


class FeatureProbes(eqx.Module):
    """Face probe + vertex probe, in their own module.

    Separate from `Agent` on purpose: the PPO optimiser's parameter tree stays
    untouched when probes are off, and "which parameters does the probe loss
    own" is answerable by pointing at a subtree instead of by reading a loss.
    """

    face: Probe
    vertex: Probe
    arm: str = eqx.field(static=True)

    def __init__(self, embd_dim, max_axes, n_vertex_out, width, key,
                 arm=FaceProbeArm.LEAN):
        if arm not in FaceProbeArm.ALL:
            raise ValueError("unknown face probe arm %r (expected one of %r)"
                             % (arm, FaceProbeArm.ALL))
        kf, kv = jax.random.split(key)
        self.face = Probe(face_input_dim(arm, embd_dim, max_axes), NFT,
                          width, kf)
        # The vertex probe reads ONE pointer slot row, E wide -- the lean
        # collapse (64 -> 32) is exactly why the old 0.93 does not carry.
        self.vertex = Probe(int(embd_dim), int(n_vertex_out), width, kv)
        self.arm = arm

    def face_predict(self, latent, ctx_i=None, ctx_j=None, extents=None):
        """`latent` is the face-keyed scatter row -- the live head's input."""
        if self.arm == FaceProbeArm.LEAN:
            x = latent
        elif self.arm == FaceProbeArm.ENDPOINTS:
            if ctx_i is None or ctx_j is None:
                raise ValueError(
                    "the ENDPOINTS reference arm needs ctx_i and ctx_j; the "
                    "LEAN arm (the live head's input) takes the latent alone")
            x = jnp.concatenate([ctx_i, ctx_j, latent])
        else:
            if extents is None:
                raise ValueError(
                    "the EXTENTS reference arm needs per-face axis sizes")
            x = jnp.concatenate([latent, extents])
        return self.face(x)

    def vertex_predict(self, slot_row):
        return self.vertex(slot_row)


# ------------------------------------------------------------------- loss ---
def masked_mse(pred, target, valid):
    """MSE over VALID rows only, per output column.

    The denominator is the valid count, not the padded row count -- averaging
    over padded slots is how approx_prob came to report 99.2% when it was
    reporting a padding ratio. Returns zero, not NaN, when nothing is valid.
    """
    w = jnp.asarray(valid, jnp.float32)[:, None]
    se = w * (pred - target) ** 2
    return se.sum(0) / jnp.maximum(w.sum(), 1.0)


def within_step_r2(pred, target, valid, step_id, n_steps):
    """Within-elimination-step R2 per column -- the offline statistic.

    Removing each step's mean is what makes this fair: the raw R2 would credit
    the probe for predicting step-to-step drift, which the step index alone
    gives away. Per-vertex signal is CONSTANT inside a step group and so
    contributes exactly 0, by construction.
    """
    w = jnp.asarray(valid, jnp.float32)[:, None]
    oh = jnn.one_hot(step_id, n_steps, dtype=jnp.float32) * w
    cnt = jnp.maximum(oh.sum(0), 1.0)

    def _center(z):
        mu = (oh.T @ (z * w)) / cnt[:, None]
        return (z - oh @ mu) * w

    zt, zp = _center(target), _center(pred)
    ss_res = ((zt - zp) ** 2).sum(0)
    ss_tot = (zt ** 2).sum(0)
    # No within-step variance => undefined, report 0. A degenerate target must
    # never read as a perfect decode.
    return jnp.where(ss_tot > 1e-8,
                     1.0 - ss_res / jnp.maximum(ss_tot, 1e-8), 0.0)


def steps_to_threshold(r2_history, threshold):
    """First index whose R2 >= threshold, else -1.

    wip.dot asks for this ALONGSIDE final R2: two arms can converge to the
    same value at very different speeds, and for a probe that is the
    difference between "the information is there" and "the information is
    there but buried".
    """
    hist = np.asarray(r2_history, dtype=np.float64)
    hit = np.nonzero(hist >= float(threshold))[0]
    return int(hit[0]) if hit.size else -1


PROBE_ON = os.environ.get("ALPHAGRAD_FEATURE_PROBE", "0") != "0"
PROBE_ARM = os.environ.get("ALPHAGRAD_FEATURE_PROBE_ARM", FaceProbeArm.LEAN)
PROBE_WIDTH = int(os.environ.get("ALPHAGRAD_FEATURE_PROBE_WIDTH", "128"))
