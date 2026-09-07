#!/usr/bin/env python
"""t284 deliverable (2): what still emits a growing broadcast, by case.

    python t284_broadcast_census.py OUT_DIR

Wraps ``matmul._as_shape`` (and the verbatim name inside
``matmul_legacy_tiled``) to count every call whose broadcast mode GROWS the
buffer, and wraps ``_execute_block_sparse_contraction`` so each growth is
attributed to the contraction that caused it and to the CASE of that
contraction's pairs.

The cases, in CONTEXT.md's words:

  no implicit axis            the extent is stored by both operands
  single implicit sparse      the meta axis of a diagonal pair, stored by
                              exactly one operand, entering as a batch axis
  single implicit dense       a dense axis or the block axis of a diagonal
                              pair, stored by exactly one operand, entering as
                              a contracted or carried axis
  double implicit contracted  a contracted extent stored by neither operand
  double implicit batch       a meta extent stored by neither operand
  uniform operand             val is None on every dim of one operand
  lcm grid                    the two outer lens are both above 1 and unequal
  spatial_sparse pairing      pairing_type starts with spatial_sparse
  partially stored extent     the physical slot is neither 1 nor the logical
                              extent

Modes: the incumbent executor (GRAPHAX_TILED_LEGACY=1), the lazy frame under
GRAPHAX_TILED_LAZY in {off, nodemote, full}, and the planner as the reference
(its LOWER_STATS rule counts, including out:implicit_kept, out:val_none and
fold_scale).
"""
from __future__ import annotations

import collections
import importlib
import json
import math
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.pop("ALPHAGRAD_FORCE_REV_ORDER", None)

import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402
from graphax import jacve                                      # noqa: E402

_mm = importlib.import_module("graphax.sparse.ops.matmul")
_mm_legacy = importlib.import_module("graphax.sparse.ops.matmul_legacy_tiled")
_lower_mm = importlib.import_module("graphax.sparse.lower.matmul")

JSONL = os.path.join(OUT, "t284_census.jsonl")

MODES = {
    "incumbent": {"GRAPHAX_TILED_LEGACY": "1", "GRAPHAX_TILED_LAZY": "nodemote",
                  "GRAPHAX_TILED_MULREDUCE": "0",
                  "GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "lazy_off": {"GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_LAZY": "off",
                 "GRAPHAX_TILED_MULREDUCE": "0",
                 "GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "lazy_nodemote": {"GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_LAZY": "nodemote",
                      "GRAPHAX_TILED_MULREDUCE": "0",
                      "GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "lazy_full": {"GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_LAZY": "full",
                  "GRAPHAX_TILED_MULREDUCE": "0",
                  "GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "mul_then_reduce": {"GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_LAZY": "nodemote",
                        "GRAPHAX_TILED_MULREDUCE": "1",
                        "GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "planner": {"GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_LAZY": "nodemote",
                "GRAPHAX_TILED_MULREDUCE": "0",
                "GRAPHAX_EINSUM_GENERAL": "1", "GRAPHAX_PLANNER_EXACT": "1"},
}
ALL_KEYS = ("GRAPHAX_TILED_LEGACY", "GRAPHAX_TILED_LAZY",
            "GRAPHAX_TILED_MULREDUCE", "GRAPHAX_EINSUM_GENERAL",
            "GRAPHAX_PLANNER_EXACT")


def emit(rec, short=None):
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print("[t284c] " + (short if short is not None
                        else json.dumps(rec, default=str)), flush=True)


# ------------------------------------------------------------------ cases ---
# ``_prepare_contraction_views`` broadcasts exactly twice, lhs then rhs, on the
# "unmerged" shapes. Their axis layout is fixed:
#   lhs: [(outer_len_i, total_i // outer_len_i)] * N, [block_len_i] * N,
#        [split_l_i] * N, leftover
#   rhs: [(outer_len_i, total_i // outer_len_i)] * N, [split_r_i] * N,
#        [shared_block_len_i] * N, leftover
# So the axis that grew names the extent that grew, and the pair's storage
# status names the case.
def axis_labels(N, side):
    lab = []
    for i in range(N):
        lab.append(("own_outer", i))
        lab.append(("meta", i))
    if side == "lhs":
        lab += [("block", i) for i in range(N)] + [("split", i) for i in range(N)]
    else:
        lab += [("split", i) for i in range(N)] + [("shared_block", i) for i in range(N)]
    return lab


def case_of(kind, i, lhs_val, rhs_val, pairs, total, split):
    """CONTEXT.md's name for the extent that grew."""
    p = pairs[i]
    lo, lb, ls = (int(lhs_val.shape[3 * i]), int(lhs_val.shape[3 * i + 1]),
                  int(lhs_val.shape[3 * i + 2]))
    ro, rb, rs = (int(rhs_val.shape[3 * i]), int(rhs_val.shape[3 * i + 1]),
                  int(rhs_val.shape[3 * i + 2]))
    l, r = p.lhs, p.rhs
    ol, orr = int(l.outer_len), int(r.outer_len)
    T, G = math.lcm(ol, orr), math.gcd(ol, orr)
    pt = str(p.pairing_type)
    if pt.startswith("spatial_sparse"):
        return "spatial_sparse pairing"
    if kind in ("own_outer", "meta"):
        if T > 1 and not (T == G or ol == 1 or orr == 1):
            return "lcm grid (meta)"
        m_l = lo * ((T // ol) if ls != 1 else 1)
        m_r = ro * ((T // orr) if rb != 1 else 1)
        if m_l == T and m_r == T:
            return "no implicit axis (meta)"
        if m_l == 1 and m_r == 1:
            return "double implicit batch axis"
        if (m_l == T) != (m_r == T):
            return "single implicit sparse axis"
        return "partially stored extent (meta)"
    if kind == "split":
        if pt != "contract":
            return "single implicit dense axis (carried)"
        st_l, st_r = ls != 1, rb != 1
        if st_l and st_r:
            return "no implicit axis (contracted)"
        if st_l != st_r:
            return "single implicit dense axis (contracted)"
        return "double implicit contracted axis"
    if kind in ("block", "shared_block"):
        return "single implicit dense axis (block)"
    return f"other ({kind})"


# ------------------------------------------------------------------ probe ---
STATE = {"ctx": None, "bcast_seen": 0}


def install(mode):
    orig_as_shape = _mm._as_shape
    orig_prep = _mm._prepare_contraction_views
    orig_prep_legacy = _mm_legacy._prepare_contraction_views
    orig_exec = _mm._execute_block_sparse_contraction
    orig_exec_legacy = _mm_legacy._execute_block_sparse_contraction
    tally = {"calls": 0, "elems": 0,
             "by_case_calls": collections.Counter(),
             "by_case_elems": collections.Counter(),
             "unattributed_calls": 0, "unattributed_elems": 0,
             "contractions": 0, "uniform_operand_contractions": 0}

    def _as_shape(view, target_shape, *, mode):
        out = orig_as_shape(view, target_shape, mode=mode)
        if mode != "broadcast":
            return out
        in_shape = tuple(view.shape)
        tgt = tuple(target_shape)
        in_n = math.prod(in_shape) if in_shape else 1
        out_n = math.prod(tgt) if tgt else 1
        if out_n <= in_n:
            return out
        grown = out_n - in_n
        tally["calls"] += 1
        tally["elems"] += grown
        ctx = STATE["ctx"]
        side = "lhs" if STATE["bcast_seen"] == 0 else "rhs"
        STATE["bcast_seen"] += 1
        if ctx is None or len(in_shape) != len(tgt):
            tally["unattributed_calls"] += 1
            tally["unattributed_elems"] += grown
            return out
        N, lhs_val, rhs_val, pairs, total, split = ctx
        labels = axis_labels(N, side)
        # the growth factor of each axis, and the case it belongs to
        factors = {}
        for a, (kind, i) in enumerate(labels):
            if a < len(tgt) and tgt[a] > in_shape[a]:
                try:
                    c = case_of(kind, i, lhs_val, rhs_val, pairs, total, split)
                except Exception:
                    c = "<unclassified>"
                factors[c] = factors.get(c, 1) * (tgt[a] // max(in_shape[a], 1))
        for a in range(len(labels), len(tgt)):
            if tgt[a] > in_shape[a]:
                factors["leftover axis"] = factors.get("leftover axis", 1) * (
                    tgt[a] // max(in_shape[a], 1))
        if not factors:
            tally["unattributed_calls"] += 1
            tally["unattributed_elems"] += grown
            return out
        # split the grown elements between the cases in proportion to their
        # growth factor exponent, so the total is conserved
        tot_log = sum(math.log(f) for f in factors.values())
        for c, f in factors.items():
            tally["by_case_calls"][c] += 1
            share = (math.log(f) / tot_log) if tot_log > 0 else 1.0 / len(factors)
            tally["by_case_elems"][c] += int(round(grown * share))
        return out

    def _mk_prep(orig):
        def _prep(lhs_val, rhs_val, pairs, shared, total, split, **kw):
            STATE["ctx"] = (len(pairs), lhs_val, rhs_val, pairs, total, split)
            STATE["bcast_seen"] = 0
            try:
                return orig(lhs_val, rhs_val, pairs, shared, total, split, **kw)
            finally:
                STATE["ctx"] = None
        return _prep

    def _mk_exec(orig):
        def _exec(lhs_val, rhs_val, pairs, ctx):
            tally["contractions"] += 1
            if math.prod(lhs_val.shape) == 1 or math.prod(rhs_val.shape) == 1:
                tally["uniform_operand_contractions"] += 1
            return orig(lhs_val, rhs_val, pairs, ctx)
        return _exec

    _mm._as_shape = _as_shape
    _mm_legacy._as_shape = _as_shape
    _mm._prepare_contraction_views = _mk_prep(orig_prep)
    _mm_legacy._prepare_contraction_views = _mk_prep(orig_prep_legacy)
    _mm._execute_block_sparse_contraction = _mk_exec(orig_exec)
    _mm_legacy._execute_block_sparse_contraction = _mk_exec(orig_exec_legacy)

    def restore():
        _mm._as_shape = orig_as_shape
        _mm_legacy._as_shape = orig_as_shape
        _mm._prepare_contraction_views = orig_prep
        _mm_legacy._prepare_contraction_views = orig_prep_legacy
        _mm._execute_block_sparse_contraction = orig_exec
        _mm_legacy._execute_block_sparse_contraction = orig_exec_legacy

    return tally, restore



def jaxpr_growing_broadcasts(fn, args):
    """Growing ``broadcast_in_dim`` equations of the traced jaxpr. This is the
    census ticket dsnn-3qm.28.1 used; it sees the OUTPUT construction too,
    where the _as_shape census only sees the operand views."""
    try:
        cj = jax.make_jaxpr(fn)(*args)
    except Exception:
        return None, None
    calls, elems = 0, 0
    def walk(jaxpr):
        nonlocal calls, elems
        for e in jaxpr.eqns:
            if e.primitive.name == "broadcast_in_dim":
                i = math.prod(e.invars[0].aval.shape) if hasattr(e.invars[0], "aval") else 1
                o = math.prod(e.outvars[0].aval.shape)
                if o > i:
                    calls += 1
                    elems += o - i
            for p in ("jaxpr", "call_jaxpr", "branches"):
                sub = e.params.get(p)
                if sub is None:
                    continue
                for j in (sub if isinstance(sub, (list, tuple)) else [sub]):
                    walk(getattr(j, "jaxpr", j))
    walk(cj.jaxpr)
    return calls, elems


def run_direct(label, build):
    """A hand-built SparseTensor pair, contracted directly."""
    from graphax.sparse.tensor import SparseTensor  # noqa: F401
    for mode, envv in MODES.items():
        saved = {k: os.environ.get(k) for k in ALL_KEYS}
        os.environ.update(envv)
        _lower_mm.reset_stats()
        tally, restore = install(mode)
        err = None
        jb_calls = jb_elems = None
        from graphax.sparse.ops.matmul import matmul as gx_matmul
        try:
            a, b = build()
            jax.eval_shape(lambda x, y: gx_matmul(x, y), a, b)
        except Exception as exc:
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            restore()
        try:
            def _traced(x, y):
                r = gx_matmul(x, y)
                v = getattr(r, "val", None)
                return v if v is not None else jnp.zeros(())
            jb_calls, jb_elems = jaxpr_growing_broadcasts(_traced, build())
        except Exception:
            pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        rec = {"phase": "census", "target": label, "mode": mode,
               "contractions": tally["contractions"],
               "growing_calls": tally["calls"], "growing_elems": tally["elems"],
               "jaxpr_growing_calls": jb_calls, "jaxpr_growing_elems": jb_elems,
               "by_case_calls": dict(tally["by_case_calls"]),
               "by_case_elems": dict(tally["by_case_elems"]),
               "unattributed_calls": tally["unattributed_calls"],
               "unattributed_elems": tally["unattributed_elems"],
               "uniform_operand_contractions": tally["uniform_operand_contractions"],
               "lower_stats": dict(_lower_mm.LOWER_STATS), "error": err}
        emit(rec, f"{label:28s} {mode:16s} contractions {tally['contractions']:4d} "
                  f"growing {tally['calls']:4d} calls / {tally['elems']:9d} elems"
                  f" | jaxpr growing {jb_calls} / {jb_elems}"
                  + (f"  ERROR {err}" if err else ""))
        for c, n in sorted(tally["by_case_calls"].items(), key=lambda kv: -kv[1]):
            print(f"[t284c]     {c:36s} {n:4d} calls  "
                  f"{tally['by_case_elems'][c]:9d} elems", flush=True)
        if mode == "planner" and _lower_mm.LOWER_STATS:
            print(f"[t284c]     planner rules: {dict(_lower_mm.LOWER_STATS)}", flush=True)


def run(label, target_fun, args, argnums, order, has_aux=False):
    for mode, env in MODES.items():
        saved = {k: os.environ.get(k) for k in ALL_KEYS}
        os.environ.update(env)
        _lower_mm.reset_stats()
        tally, restore = install(mode)
        err = None
        try:
            fn = jacve(target_fun, list(order), argnums=argnums, has_aux=has_aux,
                       sparse_representation=True, transforms=[],
                       face_transforms=None)
            jax.eval_shape(fn, *args)
        except Exception as exc:
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            restore()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        rec = {"phase": "census", "target": label, "mode": mode,
               "contractions": tally["contractions"],
               "growing_calls": tally["calls"], "growing_elems": tally["elems"],
               "by_case_calls": dict(tally["by_case_calls"]),
               "by_case_elems": dict(tally["by_case_elems"]),
               "unattributed_calls": tally["unattributed_calls"],
               "unattributed_elems": tally["unattributed_elems"],
               "uniform_operand_contractions": tally["uniform_operand_contractions"],
               "lower_stats": dict(_lower_mm.LOWER_STATS), "error": err}
        emit(rec, f"{label:28s} {mode:16s} contractions {tally['contractions']:4d} "
                  f"growing {tally['calls']:4d} calls / {tally['elems']:9d} elems"
                  f"  uniform-operand contractions "
                  f"{tally['uniform_operand_contractions']}"
                  + (f"  ERROR {err}" if err else ""))
        for c, n in sorted(tally["by_case_calls"].items(), key=lambda kv: -kv[1]):
            print(f"[t284c]     {c:36s} {n:4d} calls  "
                  f"{tally['by_case_elems'][c]:9d} elems", flush=True)
        if mode == "planner" and _lower_mm.LOWER_STATS:
            keys = {k: v for k, v in _lower_mm.LOWER_STATS.items()
                    if "implicit" in k or "val_none" in k or "fold_scale" in k
                    or "broadcast" in k}
            print(f"[t284c]     planner rules: {keys}", flush=True)


# ------------------------------------------------------------- target one ---
B, DIN, H, V = 4, 8, 8, 16
KEY = jax.random.PRNGKey(0)
KS = jax.random.split(KEY, 6)
MLP_ARGS = (jax.random.normal(KS[0], (B, DIN)),
            jax.random.normal(KS[1], (B, V)),
            jax.random.normal(KS[2], (DIN, H)) * 0.3,
            jax.random.normal(KS[3], (H,)) * 0.1,
            jax.random.normal(KS[4], (H, V)) * 0.3)


def mlp_loss(x, y, w1, b1, wout):
    h = jnp.tanh(x @ w1 + b1)
    return jnp.mean((h @ wout - y) ** 2)


def mlp_order():
    cj = jax.make_jaxpr(mlp_loss)(*MLP_ARGS)
    outvars = set(map(id, cj.jaxpr.outvars))
    valid = [i + 1 for i, e in enumerate(cj.jaxpr.eqns)
             if not any(id(o) in outvars for o in e.outvars)]
    return list(reversed(valid))


emit({"phase": "start", "jax": jax.__version__,
      "devices": [str(d) for d in jax.devices()], "host": os.uname().nodename},
     "start")


# ------------------------------------------------- target zero: test_14 ----
# The uniform operand: ``a`` has val None on every dim. This is the growing
# broadcast that ticket dsnn-3qm.28.1 found survives under EVERY mode.
def _test14():
    import numpy as np
    from graphax.sparse.tensor import (SparseTensor, DiagonalIndex,
                                       DenseIndex)
    s1, s2, s3 = 5, 4, 2
    a = SparseTensor(
        (DiagonalIndex(0, s1, axis=None, other_id=2), DenseIndex(1, s2, None)),
        (DiagonalIndex(2, s1, axis=None, other_id=0),), val=None)
    b = SparseTensor(
        (DiagonalIndex(0, s1, axis=0, other_id=1),),
        (DiagonalIndex(1, s1, axis=0, other_id=0), DenseIndex(2, s3, 1)),
        jnp.asarray(np.arange(s1 * s3, dtype=np.float32).reshape(s1, s3)))
    return a, b


run_direct("test_14 uniform operand", _test14)
run("MLP toy, reverse, exact", mlp_loss, MLP_ARGS, (2, 3, 4), mlp_order())

# ------------------------------------------------------------- target two ---
try:
    import alphagrad.approx.tools.landscape_map as lm
    CLI = ["--example", "NeuralNetwork", "--dataset", "mnist", "--seed", "250197",
           "--latency-inner-reps", "1", "--num-data-points", "1",
           "--reps-per-point", "1", "--quality-metric", "grad_cosine",
           "--approx-old", "same", "--out-dir", OUT, "--dry-run",
           "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2"]
    lm.ARGS = lm.make_argparser().parse_args(CLI)
    env, _s, _c = lm.build_env(lm.ARGS)
    cfg = env.config
    run("NeuralNetwork, reverse, exact", cfg.target_fun, env.args, cfg.argnums,
        [int(v) for v in lm.rev_order(env)], has_aux=cfg.has_aux)
    run("NeuralNetwork, markowitz, exact", cfg.target_fun, env.args, cfg.argnums,
        [int(v) for v in lm.markowitz_order(env)], has_aux=cfg.has_aux)
except Exception as exc:
    emit({"phase": "skip", "target": "NeuralNetwork",
          "error": f"{type(exc).__name__}: {str(exc)[:300]}"})

emit({"phase": "done"}, "done")
