"""Gradient-isolated online probes for BASIC variable-level structure.

THE QUESTION (owner's words): "I want three heads for each variable in the
equation, and I want to know the ndim, size, a small GRU for the shape
(maybe) and dtype." Before spending hundreds of episodes on representation
fixes (QUALITY_COLLAPSE_INVESTIGATION.md, H3: the 32-wide face latent decodes
quality-relevant structure at only R2 0.03-0.18), this measures whether the
latents carry the MOST BASIC variable-level facts at all.

Every face in the two-op form carries three variables -- the env emits
``((lhs, rhs, new), (jl, jr, jres))`` wires, and graphax's local path is
``res = op(lhs, rhs)`` (core.py `_unpack_face_slots`). So each probe LEVEL
gets three head-groups, one per variable slot {lhs, rhs, res}, each
predicting from the latent ALONE:

  (a) ndim      -- softmax CE over 0..MAX_NDIM (clamped);
  (b) log-size  -- MSE on log10(prod(shape) + 1); sizes span orders of
                   magnitude, raw regression is unreadable;
  (c) dtype     -- softmax CE over a small fixed vocab built from what the
                   pipeline actually materializes (float32 graphs + the
                   QUANT_DTYPES ladder: float8_*, float4/6, int4/2, uints);
  (d) shape     -- a SMALL GRU decoder (hidden 32) unrolled MAX_NDIM steps,
                   teacher-forced, each step classifying the dim into
                   floor-log2 BUCKETS (exact regression on raw dims is
                   hopeless; buckets give a readable accuracy) plus a
                   per-step STOP flag. Dim CE is masked past ndim, and the
                   teacher inputs past ndim are forced to the PAD id so a
                   padded target can never leak into a scored step.

TWO LEVELS, one module:
  * FACE level: input is the live 32-wide face-keyed scatter latent --
    EXACTLY the tensor the UnifiedFaceHead consumes (same point in the
    pipeline as feature_probe's LEAN arm). Targets are the face's own
    (lhs, rhs, new) tensors, read HOST-side off the live graphax
    SparseTensors via `face_var_targets_host` (recording face_transforms
    hooks -- the same re-trace probe_faces runs).
  * VERTEX level: input is the pointer slot row the VE head scores.
    Targets are the ELIMINATED vertex's equation variables from the
    ORIGINAL jaxpr (invars[0] -> lhs, invars[1] -> rhs, outvars[0] -> res;
    missing slots masked) -- STATIC per vertex, precomputed once by
    `vertex_var_table` and gathered by the sampled vertex id. No callback.

THE ONE INVARIANT (owner: "we don't want to pass the gradient past these
probes"): every head takes its input through ``jax.lax.stop_gradient``, so
the probe loss has NO path back into palimpsa, the pointer or the face head.
tests/var_probe_test.py differentiates the loss w.r.t. the latents and
asserts EXACTLY zero -- an algebraic property of stop_gradient, not a small
coefficient.

METRIC READABILITY: every classification metric is logged NEXT TO its
majority-class baseline computed from the same episode's targets
(`episode_metrics`); the last probe round's lesson is that raw accuracy
without a baseline bar is unreadable.
"""

from __future__ import annotations

import math
import os

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jax import lax

# ---------------------------------------------------------------- layout ---
MAX_NDIM = 6                      # ndim classes 0..6 (clamped); "6 to be safe"
N_NDIM_CLASSES = MAX_NDIM + 1
N_DIM_BUCKETS = 16                # bucket = min(floor(log2(max(d,1))), 15)
PAD_BUCKET = N_DIM_BUCKETS        # BOS / pad id for the teacher-forced GRU
VAR_SLOTS = ("lhs", "rhs", "res")
N_SLOTS = len(VAR_SLOTS)

# Small fixed dtype vocab. Base graphs are float32 (+int32 index plumbing);
# QUANT introduces graphax.sparse.micro_actions.QUANT_DTYPES: the standard
# float/int ladder plus float8_* / float6/float4 / int4/uint4/int2/uint2.
# Families are bucketed so the vocab stays small and stable across jax
# versions; "abstract" is an edge whose SparseTensor has no materialized
# ``val`` (identity-seed Jacobian), which is a real and frequent face state.
DTYPE_VOCAB = (
    "float32", "float64", "bfloat16", "float16",
    "float8",    # any float8_* variant
    "fsub8",     # float4* / float6* families
    "int32", "int16", "int8",
    "isub8",     # int4 / int2 (signed sub-byte)
    "uint",      # any unsigned int width
    "bool",
    "abstract",  # no materialized val on the edge
    "other",
)
N_DTYPES = len(DTYPE_VOCAB)
_DTYPE_INDEX = {n: i for i, n in enumerate(DTYPE_VOCAB)}

# Target row layout, one row per variable slot:
#   [ndim, dtype_code, log10(size+1), dim_bucket_0 .. dim_bucket_{MAX_NDIM-1}]
COL_NDIM, COL_DTYPE, COL_LOGSIZE, COL_DIMS = 0, 1, 2, 3
TGT_COLS = 3 + MAX_NDIM


def dtype_code(name) -> int:
    """Vocab code for a dtype NAME (exact match first, then family rules)."""
    name = str(name)
    if name in _DTYPE_INDEX:
        return _DTYPE_INDEX[name]
    if name.startswith("float8"):
        return _DTYPE_INDEX["float8"]
    if name.startswith(("float4", "float6")):
        return _DTYPE_INDEX["fsub8"]
    if name.startswith(("int4", "int2")):
        return _DTYPE_INDEX["isub8"]
    if name.startswith("uint"):
        return _DTYPE_INDEX["uint"]
    return _DTYPE_INDEX["other"]


def dim_bucket(d) -> int:
    """floor-log2 bucket of one dim: 1->0, 2..3->1, 4..7->2, ..., >=2^15->15."""
    d = max(int(d), 1)
    return min(int(math.floor(math.log2(d))), N_DIM_BUCKETS - 1)


def encode_var(shape, dtype_name) -> np.ndarray:
    """One (TGT_COLS,) float32 target row from a shape tuple + dtype name."""
    row = np.zeros((TGT_COLS,), np.float32)
    shape = tuple(int(s) for s in (shape or ()))
    row[COL_NDIM] = float(min(len(shape), MAX_NDIM))
    row[COL_DTYPE] = float(dtype_code(dtype_name))
    size = 1
    for s in shape:
        size *= int(s)
    row[COL_LOGSIZE] = math.log10(max(size, 0) + 1)
    for t, d in enumerate(shape[:MAX_NDIM]):
        row[COL_DIMS + t] = float(dim_bucket(d))
    return row


# ------------------------------------------------------- host-side targets ---
# Same failure telemetry contract as feature_probe: a failing oracle must be
# COUNTED, never silently converted into "0 faces".
TARGET_FAILS = [0]
LAST_TARGET_ERR = [""]
PROBE_DEBUG = os.environ.get("ALPHAGRAD_VAR_PROBE_DEBUG", "0") != "0"


def probe_on() -> bool:
    """Read at call time (NOT import time) so ``--var-probe`` can set the env
    var after this module is already imported."""
    return os.environ.get("ALPHAGRAD_VAR_PROBE", "0") != "0"


def meta_of_sparse(st):
    """``(shape tuple, dtype name)`` of one live tensor.

    A SparseTensor with a materialized ``val`` reports the STORED form
    (val.shape / val.dtype) -- the honest "what is this tensor" answer, and
    the same convention feature_probe's ln_stored uses. An identity-seed edge
    (``val is None``) falls back to the LOGICAL sizes of its dims with dtype
    "abstract". A bare array (a hook handed a raw ndarray) reports itself.
    """
    if hasattr(st, "out_dims") or hasattr(st, "primal_dims"):
        val = getattr(st, "val", None)
        if val is not None:
            shp = tuple(int(s) for s in (getattr(val, "shape", ()) or ()))
            dt = getattr(getattr(val, "dtype", None), "name", "other")
            return shp, dt
        od = list(getattr(st, "out_dims", ()) or ())
        pd = list(getattr(st, "primal_dims", ()) or ())
        shp = tuple(max(int(getattr(d, "logical_size", 1) or 1), 1)
                    for d in od + pd)
        return shp, "abstract"
    shp = tuple(int(s) for s in (getattr(st, "shape", ()) or ()))
    dt = getattr(getattr(st, "dtype", None), "name", "other")
    return shp, dt


def face_var_targets_host(oracle, vertex, max_faces):
    """``(targets (F, 3, TGT_COLS), valid (F, 3), n_faces)`` for one vertex.

    HOST ONLY: the three variables of a face are live graphax SparseTensors,
    produced by RE-TRACING ``_eliminate_vertex`` (probe_faces' technique) with
    per-face RECORDING hooks in the two-op form
    ``((lhs, rhs, new), (None, None, None))`` -- so ``lhs`` records pre_val,
    ``rhs`` records post_val, and ``new`` records the fresh contraction
    result pre-join, exactly the three wires the env emits. The hooks return
    their tensor unchanged, and the probe copy of the graph is dropped, so
    the oracle's own state is untouched.

    Hook-call order inside one face is lhs -> rhs -> res (core.py applies the
    operand slots before the contraction and ``new`` at the result site), and
    faces are visited sequentially, so the flat event stream parses back into
    faces by starting a new face at each ``lhs`` event.
    """
    from jax._src import core as _jcore
    from graphax.core import _eliminate_vertex, faces_of
    from alphagrad.approx.common.masks import _shallow_copy_graph

    F = int(max_faces)
    tgt = np.zeros((F, N_SLOTS, TGT_COLS), np.float32)
    val = np.zeros((F, N_SLOTS), np.float32)
    vertex = int(vertex)
    events: list = []

    def _rec(key, slot):
        def _hook(st):
            events.append((key, slot, meta_of_sparse(st)))
            return st
        return _hook

    try:
        from graphax.sparse.elemental.dispatch import (
            approx_active, set_approx_active,
        )
        incr = oracle._incrs[True]
        graph = _shallow_copy_graph(incr.graph)
        tgraph = _shallow_copy_graph(incr.tgraph)
        keys = faces_of(graph, tgraph, vertex, incr.jaxpr)
        if not keys:
            return tgt, val, 0
        ft = {k: ((_rec(k, "lhs"), _rec(k, "rhs"), _rec(k, "res")),
                  (None, None, None))
              for k in set(keys)}
        n_eqns0 = len(incr.trace.frame.tracing_eqns)
        prev = approx_active()
        set_approx_active(True)
        try:
            with _jcore.set_current_trace(incr.trace):
                _eliminate_vertex(
                    vertex, incr.jaxpr, graph, tgraph, incr.vo, False,
                    transforms=(), face_transforms=ft,
                )
        finally:
            set_approx_active(prev)
            # Drop the equations the probe traced (probe_faces' hygiene: the
            # list would grow without bound over an episode otherwise).
            del incr.trace.frame.tracing_eqns[n_eqns0:]
    except Exception as _exc:
        if PROBE_DEBUG:
            raise
        TARGET_FAILS[0] += 1
        LAST_TARGET_ERR[0] = "%s: %s" % (type(_exc).__name__,
                                         str(_exc)[:160])
        if not events:
            return tgt, val, 0
        # A mid-elimination failure still yields the faces recorded so far.

    # KEYED parse (hygiene (i), docs/FACE_LATENT_INFO_LOSS.md section 2
    # H-A): rows follow the ENUMERATION order of ``keys`` -- the same order
    # the face loop pools latents in -- and each row is filled only from
    # events recorded UNDER THAT FACE'S KEY, the way live_faces keys
    # chunks. The old positional parse ("new face at each lhs") misassigned
    # every row after a mid-elimination failure and shifted neighbours when
    # a face emitted nothing (SKIP_FACE prefix replays); such faces are now
    # masked rows in place. First event per (key, slot) wins: it is the
    # operand the head decided over (a two-op face form fires later hooks
    # on derived tensors).
    by_key: dict = {}
    for k, slot, meta in events:
        by_key.setdefault(k, {}).setdefault(slot, meta)
    n_faces = min(len(keys), F)
    for r in range(n_faces):
        m = by_key.get(keys[r])
        if not m:
            continue  # skipped / failed face: every slot stays masked
        for s, sname in enumerate(VAR_SLOTS):
            meta = m.get(sname)
            if meta is None:
                continue  # truncated face (mid-face failure): slot masked
            tgt[r, s] = encode_var(*meta)
            val[r, s] = 1.0
    return tgt, val, n_faces


def vertex_var_table(jaxpr, total_v):
    """STATIC per-vertex targets: ``(tgt (V+2, 3, TGT_COLS), valid (V+2, 3))``.

    The eliminated vertex's equation variables come from the ORIGINAL jaxpr
    and never change, so the whole table is built once and the per-step
    "plumbing" is a device-side gather by the sampled vertex id (1-based,
    row 0 = padding = all-masked). ``lhs``/``rhs`` are the first two invars
    that carry an aval; a unary equation masks the ``rhs`` slot (validity
    mask per variable slot, as specified). ``res`` is the primary output var.
    """
    V = int(total_v)
    tgt = np.zeros((V + 2, N_SLOTS, TGT_COLS), np.float32)
    val = np.zeros((V + 2, N_SLOTS), np.float32)
    for i, eqn in enumerate(jaxpr.eqns[:V], start=1):
        ins = [v for v in eqn.invars if hasattr(v, "aval")]
        res = (eqn.outvars[0]
               if eqn.outvars and hasattr(eqn.outvars[0], "aval") else None)
        slots = (ins[0] if len(ins) >= 1 else None,
                 ins[1] if len(ins) >= 2 else None,
                 res)
        for s, var in enumerate(slots):
            if var is None:
                continue
            aval = var.aval
            shape = tuple(int(x) for x in (getattr(aval, "shape", ()) or ()))
            dt = getattr(getattr(aval, "dtype", None), "name", "other")
            tgt[i, s] = encode_var(shape, dt)
            val[i, s] = 1.0
    return tgt, val


# ------------------------------------------------------------------ heads ---
class VarHead(eqx.Module):
    """One variable slot's head-group: ndim + log-size + dtype + shape-GRU.

    The input arrives through ``lax.stop_gradient`` HERE, inside the module,
    so no caller can accidentally construct a leaking probe.
    """

    trunk: eqx.nn.MLP
    ndim_head: eqx.nn.Linear
    size_head: eqx.nn.Linear
    dtype_head: eqx.nn.Linear
    h0: eqx.nn.Linear
    embed: eqx.nn.Embedding
    gru: eqx.nn.GRUCell
    dim_head: eqx.nn.Linear
    stop_head: eqx.nn.Linear

    def __init__(self, n_in, width, gru_hidden, embed_dim, key):
        ks = jax.random.split(key, 9)
        self.trunk = eqx.nn.MLP(n_in, width, width, depth=1, key=ks[0])
        self.ndim_head = eqx.nn.Linear(width, N_NDIM_CLASSES, key=ks[1])
        self.size_head = eqx.nn.Linear(width, 1, key=ks[2])
        self.dtype_head = eqx.nn.Linear(width, N_DTYPES, key=ks[3])
        self.h0 = eqx.nn.Linear(width, gru_hidden, key=ks[4])
        self.embed = eqx.nn.Embedding(N_DIM_BUCKETS + 1, embed_dim, key=ks[5])
        self.gru = eqx.nn.GRUCell(embed_dim, gru_hidden, key=ks[6])
        self.dim_head = eqx.nn.Linear(gru_hidden, N_DIM_BUCKETS, key=ks[7])
        self.stop_head = eqx.nn.Linear(gru_hidden, 1, key=ks[8])

    def __call__(self, latent, tf_dims, ndim):
        """latent (E,), tf_dims (MAX_NDIM,) int32 buckets, ndim () int32.

        Returns (ndim_logits, size_pred, dtype_logits,
                 dim_logits (MAX_NDIM, NB), stop_logits (MAX_NDIM,)).
        """
        x = self.trunk(lax.stop_gradient(latent))
        ndim_logits = self.ndim_head(x)
        size_pred = self.size_head(x)[0]
        dtype_logits = self.dtype_head(x)
        h = jnn.tanh(self.h0(x))
        # Teacher forcing: step 0 reads BOS(=PAD id); step t reads the TRUE
        # bucket of dim t-1. Steps past the true ndim read PAD, never the
        # zero-padded target -- a padded value must not leak into any input.
        steps = jnp.arange(MAX_NDIM)
        prev = jnp.concatenate(
            [jnp.array([PAD_BUCKET], jnp.int32), tf_dims[:-1]])
        prev = jnp.where((steps >= 1) & (steps - 1 >= ndim),
                         PAD_BUCKET, prev)

        def _step(hh, i):
            hh = self.gru(self.embed(i), hh)
            return hh, (self.dim_head(hh), self.stop_head(hh)[0])

        _, (dim_logits, stop_logits) = lax.scan(_step, h, prev)
        return ndim_logits, size_pred, dtype_logits, dim_logits, stop_logits


class VarProbes(eqx.Module):
    """3 face head-groups + 3 vertex head-groups, one per variable slot.

    A separate parameter tree from Agent AND from FeatureProbes: the PPO
    optimiser is initialised from the agent alone, so it provably cannot
    reach a probe weight, and the probe optimiser owns exactly this subtree.
    """

    face: tuple
    vertex: tuple

    def __init__(self, embd_dim, key, width=64, gru_hidden=32, embed_dim=16):
        ks = jax.random.split(key, 2 * N_SLOTS)
        self.face = tuple(
            VarHead(int(embd_dim), width, gru_hidden, embed_dim, ks[s])
            for s in range(N_SLOTS))
        self.vertex = tuple(
            VarHead(int(embd_dim), width, gru_hidden, embed_dim,
                    ks[N_SLOTS + s])
            for s in range(N_SLOTS))


# ------------------------------------------------------------------- loss ---
def _ce(logits, label):
    return -jnn.log_softmax(logits)[label]


def _head_loss(head, latent, row, valid):
    """Loss + preds for ONE sample x ONE variable slot. All targets clamped
    into their class ranges so a malformed row can never index out."""
    ndim_t = jnp.clip(row[COL_NDIM].astype(jnp.int32), 0, MAX_NDIM)
    dtype_t = jnp.clip(row[COL_DTYPE].astype(jnp.int32), 0, N_DTYPES - 1)
    size_t = row[COL_LOGSIZE]
    dims_t = jnp.clip(row[COL_DIMS:].astype(jnp.int32), 0, N_DIM_BUCKETS - 1)

    nl, sp, dl, diml, stopl = head(latent, dims_t, ndim_t)

    ce_ndim = _ce(nl, ndim_t)
    ce_dtype = _ce(dl, dtype_t)
    mse_size = (sp - size_t) ** 2
    steps = jnp.arange(MAX_NDIM)
    dim_mask = (steps < ndim_t).astype(jnp.float32)
    ce_dims = -jnp.take_along_axis(
        jnn.log_softmax(diml, axis=-1), dims_t[:, None], axis=-1)[:, 0]
    ce_dim = (ce_dims * dim_mask).sum() / jnp.maximum(dim_mask.sum(), 1.0)
    # Per-step STOP flag: target 1 from step ndim on. Trained at EVERY step
    # (it is what terminates the decode), numerically stable BCE.
    stop_t = (steps >= ndim_t).astype(jnp.float32)
    bce = jnp.mean(jnp.maximum(stopl, 0.0) - stopl * stop_t
                   + jnp.log1p(jnp.exp(-jnp.abs(stopl))))

    loss = valid * (ce_ndim + ce_dtype + mse_size + ce_dim + bce)
    preds = (jnp.argmax(nl).astype(jnp.int32),
             jnp.argmax(dl).astype(jnp.int32),
             sp,
             jnp.argmax(diml, axis=-1).astype(jnp.int32),
             (stopl > 0.0).astype(jnp.int32))
    return loss, preds


def _group_loss(heads, latent, tgt, valid):
    """ONE sample through all three slot heads. tgt (3, TGT_COLS), valid (3,).
    Returns (summed loss, preds stacked to (3, ...) per field)."""
    losses, preds = [], []
    for s in range(N_SLOTS):
        l, p = _head_loss(heads[s], latent, tgt[s], valid[s])
        losses.append(l)
        preds.append(p)
    stack = tuple(jnp.stack([p[i] for p in preds]) for i in range(5))
    return sum(losses), stack


def level_loss(heads, lats, tgts, valids):
    """Batched masked loss for one level. lats (n, E), tgts (n, 3, TGT_COLS),
    valids (n, 3). Returns (scalar mean-over-valid loss, preds tuple with
    leading (n, 3) axes). The denominator is the VALID count, never the
    padded row count (the approx_prob padding-ratio lesson)."""
    losses, preds = jax.vmap(
        lambda l, t, v: _group_loss(heads, l, t, v))(lats, tgts, valids)
    total = losses.sum() / jnp.maximum(valids.sum(), 1.0)
    return total, preds


def var_probe_loss(probes, face_lat, face_tgt, face_val, vctx, v_tgt, v_val):
    """The full probe objective, a function of the PROBES ALONE.

    face_lat (m, E) flattened (rows x faces), face_tgt (m, 3, TGT_COLS),
    face_val (m, 3); vctx (n, E), v_tgt (n, 3, TGT_COLS), v_val (n, 3).
    Returns (loss, aux) where aux carries preds+targets+valids for the
    episode-level metric pass (predictions ride out through has_aux; nothing
    here ever enters the PPO total_loss).
    """
    l_face, f_preds = level_loss(probes.face, face_lat, face_tgt, face_val)
    l_vertex, v_preds = level_loss(probes.vertex, vctx, v_tgt, v_val)
    aux = (f_preds, face_tgt, face_val, v_preds, v_tgt, v_val,
           l_face, l_vertex)
    return l_face + l_vertex, aux


# ---------------------------------------------------------------- metrics ---
METRIC_KEYS = ("ndim_acc", "ndim_base", "dtype_acc", "dtype_base",
               "size_r2", "shape_dim_acc", "shape_dim_base", "shape_exact",
               "n")


def episode_metrics(preds, tgt, val, level, prefix="probe"):
    """Numpy episode metrics for one level, WITH majority-class baselines.

    preds = (ndim_p (n,3), dtype_p (n,3), size_p (n,3), dim_p (n,3,MAX_NDIM),
    stop_p (n,3,MAX_NDIM)); tgt (n,3,TGT_COLS); val (n,3). Keys:
    ``{prefix}/{level}/{slot}/{metric}``. Every classification metric is
    logged next to the majority-class baseline of the SAME episode's targets
    -- raw accuracy without the baseline bar is unreadable.
    """
    ndim_p, dtype_p, size_p, dim_p, stop_p = [np.asarray(x) for x in preds]
    tgt = np.asarray(tgt)
    val = np.asarray(val)
    out = {}
    for s, name in enumerate(VAR_SLOTS):
        pre = "%s/%s/%s/" % (prefix, level, name)
        m = val[:, s] > 0
        n = int(m.sum())
        if n == 0:
            for k in METRIC_KEYS:
                out[pre + k] = 0.0
            continue
        ndim_t = np.clip(tgt[m, s, COL_NDIM].astype(np.int64), 0, MAX_NDIM)
        dtype_t = np.clip(tgt[m, s, COL_DTYPE].astype(np.int64), 0,
                          N_DTYPES - 1)
        size_t = tgt[m, s, COL_LOGSIZE]
        dims_t = tgt[m, s, COL_DIMS:].astype(np.int64)

        out[pre + "ndim_acc"] = float((ndim_p[m, s] == ndim_t).mean())
        out[pre + "ndim_base"] = float(np.bincount(ndim_t).max() / n)
        out[pre + "dtype_acc"] = float((dtype_p[m, s] == dtype_t).mean())
        out[pre + "dtype_base"] = float(np.bincount(dtype_t).max() / n)

        ss_tot = float(((size_t - size_t.mean()) ** 2).sum())
        ss_res = float(((size_p[m, s] - size_t) ** 2).sum())
        # A constant target must read 0, never 1 (feature_probe's rule).
        out[pre + "size_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-8 \
            else 0.0

        steps = np.arange(MAX_NDIM)[None, :]
        dmask = steps < ndim_t[:, None]
        nd = int(dmask.sum())
        dim_ok = (dim_p[m, s] == dims_t) & dmask
        out[pre + "shape_dim_acc"] = float(dim_ok.sum() / nd) if nd else 0.0
        if nd:
            bc = np.bincount(dims_t[dmask])
            out[pre + "shape_dim_base"] = float(bc.max() / nd)
        else:
            out[pre + "shape_dim_base"] = 0.0
        stop_t = (steps >= ndim_t[:, None])
        stop_ok = (stop_p[m, s].astype(bool) == stop_t).all(axis=1)
        exact = (dim_ok.sum(axis=1) == dmask.sum(axis=1)) & stop_ok
        out[pre + "shape_exact"] = float(exact.mean())
        out[pre + "n"] = float(n)
    return out
