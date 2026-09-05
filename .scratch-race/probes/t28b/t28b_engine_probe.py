#!/usr/bin/env python
"""t28b: the landing test of the .28 race, lane B (ticket dsnn-3qm.67).

A copy of the t65 engine probe of finding 61 with ONE change of substance: the
ENGINES dict. The candidate is this tree's LAZY tiled executor, the incumbent
is the tiled executor of graphax 1f3d404 reached in the same process under
GRAPHAX_TILED_LEGACY=1, and the planner is the untouched second engine kept as
a third reference point. Every latency ratio is paired against the INCUMBENT
rev-exact arm measured immediately before it, so the headline number reads
candidate over incumbent.

Original header follows.

t65: tiled versus planner, per contraction class, ONE process, ONE device.

    python t65_engine_probe.py OUT_DIR EXAMPLE DATASET

EXAMPLE/DATASET: ``TransformerLM wikitext2`` (campaign shape from the job's
ALPHAGRAD_TLM_* variables) or ``NeuralNetwork mnist``.  GPU when the job
exports ALPHAGRAD_MEASURE_ACTOR=1 and a GPU is visible, else CPU.

This probe is the ONE sanctioned measurement while every other measurement is
blocked (owner Q22, 2026-09-05): it measures the two engines against each
other so that the single engine can be chosen.

Engines (env knobs read per call inside graphax, toggled around ``.lower()``):

  tiled     GRAPHAX_EINSUM_GENERAL=0                      the incumbent
  planner   GRAPHAX_EINSUM_GENERAL=1 GRAPHAX_PLANNER_EXACT=1  the einsum planner,
            also for exact contractions (what the campaign launchers export)

Plans (per elimination order: reverse and static minimum Markowitz degree):

  exact           no face transforms (dense x dense and intrinsic diagonal pairs)
  quant_all       Quant bf16 on slots lhs, rhs, new of every live face
  diag_all        Diag (joint gcd) on slot new of every live face where legal
  compress_all    Reduce mean on slot new of every live face where legal
  skip_first      SKIP the first live face of step 0
  skip_last       SKIP the first live face of the last step that has one

For every (order, plan, engine): the sparse executable (sparse_representation
True, what the reward path measures) and the dense executable (False, the
dense oracle of owner Q12).  Recorded per executable: XLA static temp,
argument and output bytes, fusion count, HLO md5, the graphax matmul path
census (track_paths), LOWER_STATS and DISPATCH_STATS deltas, the layout of
every returned parameter gradient (Index axes, val shape, parameter layout
yes/no), and the dense value.  Numerical agreement: exact plans against
jax.grad; every plan sparse against dense of the SAME engine; tiled against
planner.  Latency: paired against rev-exact tiled measured immediately before,
ROUNDS rounds, plus the drift floor (rev-exact tiled compiled twice).  On GPU
every executable is also compiled once with Triton GEMM on, for the static
temp reading only (the campaign flags put a 32 MiB cuBLAS workspace in every
temp, finding 58).

Read-only: no library file is edited, nothing is monkeypatched.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import statistics
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
EXAMPLE = sys.argv[2] if len(sys.argv) > 2 else "TransformerLM"
DATASET = sys.argv[3] if len(sys.argv) > 3 else "wikitext2"
ROUNDS = int(os.environ.get("T65_ROUNDS", "5"))
ORDERS = os.environ.get("T65_ORDERS", "reverse,markowitz").split(",")
PLANS = os.environ.get(
    "T65_PLANS",
    "exact,quant_all,diag_all,compress_all,skip_first,skip_last").split(",")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "t65_probe.jsonl")

GPU = os.environ.get("ALPHAGRAD_MEASURE_ACTOR", "0") == "1"
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_DIRECT_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_MEASURE_WARMUP", "1")
# The probe owns the engine knobs; a value inherited from the job would make
# the "candidate" arm silently an incumbent arm.
for _k in ("GRAPHAX_PLANNER_EXACT", "GRAPHAX_EINSUM_GENERAL",
           "GRAPHAX_TILED_LEGACY"):
    os.environ.pop(_k, None)

# The .28 race, lane B (ticket dsnn-3qm.67). The candidate is the LAZY tiled
# executor of this tree; the incumbent is the tiled executor of graphax
# 1f3d404, kept reachable in the same process under GRAPHAX_TILED_LEGACY=1;
# the planner is the untouched second engine, a third reference point.
ENGINES = {
    "candidate": {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0",
                  "GRAPHAX_TILED_LEGACY": "0"},
    "incumbent": {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0",
                  "GRAPHAX_TILED_LEGACY": "1"},
    "planner": {"GRAPHAX_EINSUM_GENERAL": "1", "GRAPHAX_PLANNER_EXACT": "1",
                "GRAPHAX_TILED_LEGACY": "0"},
}


def emit(rec: dict) -> None:
    rec = dict(rec, t=time.time())
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print("[t65] " + json.dumps(rec, default=str), flush=True)


t_import = time.perf_counter()
import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import alphagrad.approx.tools.landscape_map as lm             # noqa: E402
from graphax import jacve                                     # noqa: E402
from graphax.sparse.tensor import SparseTensor                # noqa: E402
from graphax.sparse.ops._path_tracking import track_paths     # noqa: E402
from graphax.sparse.lower import matmul as lower_mm           # noqa: E402
from graphax.sparse.elemental import dispatch as gx_dispatch  # noqa: E402

envmod = lm.envmod
# The campaign protocol (inner 50, 5 points x 4 reps) unless the job shrinks
# it for a slow target (T65_INNER / T65_POINTS / T65_REPS; recorded below).
CLI = [
    "--example", EXAMPLE, "--dataset", DATASET, "--seed", "250197",
    "--latency-inner-reps", os.environ.get("T65_INNER", "50"),
    "--num-data-points", os.environ.get("T65_POINTS", "5"),
    "--reps-per-point", os.environ.get("T65_REPS", "4"),
    "--quality-metric", "grad_cosine",
    "--approx-old", "same", "--out-dir", OUT, "--dry-run",
    "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2",
]
if GPU:
    CLI.append("--exec-on-gpu")
lm.ARGS = lm.make_argparser().parse_args(CLI)
emit({"phase": "import", "s": round(time.perf_counter() - t_import, 1),
      "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
      "host": os.uname().nodename, "example": EXAMPLE, "dataset": DATASET,
      "gpu": GPU, "rounds": ROUNDS, "orders": ORDERS, "plans": PLANS,
      "env": {k: os.environ.get(k) for k in (
          "GRAPHAX_PLANNER_EXACT", "GRAPHAX_EINSUM_GENERAL",
          "GRAPHAX_DEMAND_EMIT", "GRAPHAX_ELEMENTAL", "GRAPHAX_STRUCT_LOWER",
          "GRAPHAX_COMPACT_FRAME", "XLA_FLAGS", "JAX_PLATFORMS",
          "ALPHAGRAD_TLM_SEQ", "ALPHAGRAD_TLM_DMODEL", "ALPHAGRAD_TLM_VOCAB",
          "ALPHAGRAD_MAX_FACES", "ALPHAGRAD_SKIP_COUNT_OPS",
          "ALPHAGRAD_DIRECT_MEASURE", "JAX_COMPILATION_CACHE_DIR",
          "CUDA_VISIBLE_DEVICES")}})

# ---------------------------------------------------------------- target ---
t0 = time.perf_counter()
env, eval_samples, _cj = lm.build_env(lm.ARGS)
cfg = env.config
dev = jax.devices("gpu")[0] if GPU else jax.devices()[0]
args_l = jax.device_put(env.args, dev)
n_points = int(cfg.num_data_points)
n_reps = int(cfg.reps_per_point)
inner = max(1, int(cfg.latency_inner_reps))
warmup = int(envmod._resolve_warmup(cfg))
eval_args_all = [[jax.device_put(a[i], dev) for a in eval_samples]
                 for i in range(n_points)]
unique_devices = [dev]
ORDER_OF = {}
ORDER_OF["reverse"] = [int(v) for v in lm.rev_order(env)]   # the reference order, always
if "markowitz" in ORDERS:
    ORDER_OF["markowitz"] = [int(v) for v in lm.markowitz_order(env)]
emit({"phase": "target", "s": round(time.perf_counter() - t0, 1),
      "n_vertices": len(ORDER_OF["reverse"]), "sparse": bool(cfg.sparse),
      "has_aux": bool(cfg.has_aux), "argnums": list(cfg.argnums),
      "device": str(dev), "n_points": n_points, "n_reps": n_reps,
      "inner": inner, "warmup": warmup, "max_faces": int(envmod.MAX_FACES),
      "orders": {k: v for k, v in ORDER_OF.items()}})


def _jac(order, sparse: bool, **kw):
    return jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
                 has_aux=cfg.has_aux, sparse_representation=sparse, **kw)


def face_transforms_of(order, plan):
    """The measurement's own wire decode (env._face_transforms_for_order)."""
    specs, face_specs, face_skips = lm.get_plan_arrays(plan, len(order))
    ft = envmod._face_transforms_for_order(
        cfg, env.consts, env.args, list(order), list(np.asarray(specs)),
        list(np.asarray(face_specs)), list(np.asarray(face_skips)))
    return ft


def build_plans(order_name, order):
    inv = lm.face_inventory(env, order)
    steps_with_faces = sorted({int(e["k"]) for e in inv})
    out = {}
    for name in PLANS:
        t0 = time.perf_counter()
        if name == "exact":
            out[name] = (None, {"n_faces_approx": 0, "n_slot_rows": 0,
                                "total_live_faces": len(inv)})
            continue
        if name == "skip_first":
            k = steps_with_faces[0]
            plan = lm.build_skip_only_plan(env, order, [(k, 0)], inventory=inv)
        elif name == "skip_last":
            k = steps_with_faces[-1]
            plan = lm.build_skip_only_plan(env, order, [(k, 0)], inventory=inv)
        else:
            op = {"quant_all": "quant", "diag_all": "diag",
                  "compress_all": "compress"}[name]
            plan = lm.build_ladder_plan(env, order, op, "all", lm.ARGS)
        ft = face_transforms_of(order, plan)
        out[name] = (ft, {k: plan.get(k) for k in (
            "n_faces_approx", "n_slot_rows", "total_live_faces")})
        emit({"phase": "plan", "order": order_name, "plan": name,
              "s": round(time.perf_counter() - t0, 1),
              "vertices_with_hooks": len(ft),
              "faces_with_hooks": sum(len(d) for d in ft.values()),
              **out[name][1]})
    emit({"phase": "faces", "order": order_name, "live_faces": len(inv),
          "steps_with_faces": len(steps_with_faces)})
    return out


# --------------------------------------------------------------- compile ---
def _set_engine(engine):
    saved = {k: os.environ.get(k) for k in ENGINES[engine]}
    os.environ.update(ENGINES[engine])
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _triton_options():
    return {"xla_gpu_autotune_level": 0, "xla_gpu_enable_triton_gemm": True,
            "xla_gpu_enable_llvm_module_compilation_parallelism": True}


def compile_arm(tag, order, ft, engine, sparse):
    """Lower under ``engine``, compile with the campaign options, census."""
    t0 = time.perf_counter()
    kw = {"transforms": [], "face_transforms": ft}
    fn = _jac(order, sparse, **kw)
    saved = _set_engine(engine)
    lower_mm.reset_stats()
    d0 = dict(gx_dispatch.DISPATCH_STATS)
    try:
        with track_paths() as paths:
            lowered = jax.jit(fn, keep_unused=True).lower(*args_l)
    finally:
        _restore(saved)
    lower_stats = dict(lower_mm.LOWER_STATS)
    d1 = gx_dispatch.DISPATCH_STATS
    dispatch_delta = {k: d1[k] - d0.get(k, 0) for k in d1 if d1[k] != d0.get(k, 0)}
    t_low = time.perf_counter() - t0
    ex = envmod._compile_measure(lowered)
    t_all = time.perf_counter() - t0
    ma = ex.memory_analysis()
    try:
        hlo = ex.as_text()
    except Exception as exc:                      # pragma: no cover
        hlo = f"<as_text failed: {exc!r}>"
    rec = {
        "phase": "compile", "tag": tag, "engine": engine, "sparse": sparse,
        "lower_s": round(t_low, 2), "compile_s": round(t_all - t_low, 2),
        "temp_bytes": int(getattr(ma, "temp_size_in_bytes", -1)),
        "argument_bytes": int(getattr(ma, "argument_size_in_bytes", -1)),
        "output_bytes": int(getattr(ma, "output_size_in_bytes", -1)),
        "generated_code_bytes": int(getattr(ma, "generated_code_size_in_bytes", -1)),
        "hlo_md5": hashlib.md5(hlo.encode()).hexdigest(),
        "hlo_fusions": hlo.count(" fusion("), "hlo_lines": hlo.count("\n"),
        "paths": dict(collections.Counter(paths)),
        "lower_stats": lower_stats, "dispatch_delta": dispatch_delta,
        "compile_fallbacks": int(envmod._MEASURE_COMPILE_FALLBACKS["n"]),
    }
    if GPU:
        try:
            ex_t = lowered.compile(compiler_options=_triton_options())
            ma_t = ex_t.memory_analysis()
            rec["temp_triton_bytes"] = int(getattr(ma_t, "temp_size_in_bytes", -1))
            rec["hlo_fusions_triton"] = ex_t.as_text().count(" fusion(")
            del ex_t
        except Exception as exc:
            rec["temp_triton_bytes"] = f"failed: {exc!r}"[:200]
    emit(rec)
    with open(os.path.join(OUT, f"hlo_{tag}_{engine}_{'sp' if sparse else 'dn'}.txt"), "w") as fh:
        fh.write(hlo)
    return ex, rec


# ---------------------------------------------------------------- values ---
def _jac_of(out):
    return out[1] if cfg.has_aux else out


def leaf_layouts(out):
    """Per returned parameter gradient: Index axes, val shape, dense shape,
    parameter layout (axis == position over the dims that own a val axis)."""
    jac = _jac_of(out)
    leaves = jax.tree_util.tree_leaves(
        jac, is_leaf=lambda x: isinstance(x, SparseTensor))
    rows = []
    for t in leaves:
        if isinstance(t, SparseTensor):
            dims = tuple(t.out_dims) + tuple(t.primal_dims)
            axes = [None if d.axis is None else int(d.axis) for d in dims]
            sizes = [int(d.size) for d in dims]
            owned = [(i, a) for i, a in enumerate(axes) if a is not None]
            param_layout = all(a == i for i, a in owned)
            rows.append({"kind": "sparse", "axes": axes, "sizes": sizes,
                         "out_ndim": len(t.out_dims),
                         "val_shape": None if t.val is None else [int(s) for s in t.val.shape],
                         "dtype": str(t.dtype), "param_layout": param_layout,
                         "val_matches_sizes": (t.val is not None and
                                               [int(s) for s in t.val.shape] == sizes)})
        elif hasattr(t, "shape"):
            rows.append({"kind": "array", "shape": [int(s) for s in t.shape],
                         "dtype": str(t.dtype), "param_layout": True})
        else:
            rows.append({"kind": type(t).__name__})
    return rows


def dense_vector(out):
    jac = _jac_of(out)
    leaves = jax.tree_util.tree_leaves(
        jac, is_leaf=lambda x: isinstance(x, SparseTensor))
    parts = []
    shapes = []
    for t in leaves:
        if isinstance(t, SparseTensor):
            a = np.asarray(t.dense(), dtype=np.float64)
        elif hasattr(t, "shape"):
            a = np.asarray(t, dtype=np.float64)
        else:
            continue
        shapes.append(list(a.shape))
        parts.append(a.ravel())
    return (np.concatenate(parts) if parts else np.zeros(0)), shapes


def agree(a, b):
    if a.shape != b.shape:
        return {"shape_mismatch": [int(a.size), int(b.size)]}
    if a.size == 0:
        return {"empty": True}
    diff = a - b
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    return {"max_abs": float(np.max(np.abs(diff))),
            "rel_l2": float(np.linalg.norm(diff) / nb) if nb > 0 else None,
            "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None,
            "bit_identical": bool(np.array_equal(a, b)),
            "n_nan": int(np.isnan(diff).sum())}


# The VALUE comparisons run on a REAL probe batch, as env._grad_cosine_quality
# does: the calibration eval samples are N(0, 1) draws of every argument and
# on TLM (integer token ids) they produce NaN gradients (epic note on
# common/eval_samples.py). The latency loop keeps the eval samples: that is
# the campaign's instrument.
VAL_ARGS = list(args_l)
_pb = envmod._probe_batch(cfg, list(env.args), role="train", index=0)
if _pb is not None:
    for _slot in range(min(2, len(_pb))):
        VAL_ARGS[_slot] = jax.device_put(jnp.asarray(_pb[_slot]), dev)
    VAL_SRC = "probe_batch"
else:
    VAL_SRC = "eval_samples"

# Oracle A: jax.grad of the target, once per process.
t0 = time.perf_counter()
_grad_fn = jax.jit(jax.grad(cfg.target_fun, argnums=cfg.argnums,
                            has_aux=cfg.has_aux))
_g = _grad_fn(*VAL_ARGS)
_g = _g[0] if cfg.has_aux else _g
JAXGRAD, JAXGRAD_SHAPES = dense_vector(_g)
emit({"phase": "oracle_jax_grad", "s": round(time.perf_counter() - t0, 1),
      "value_args": VAL_SRC,
      "shapes": JAXGRAD_SHAPES, "norm": float(np.linalg.norm(JAXGRAD))})


def measure(ex):
    lat, peak = envmod._campaign_measure_cost(
        ex, eval_args_all, unique_devices, inner, warmup, n_reps)
    return float(lat), float(peak)


# ------------------------------------------------------------ reference ---
REV = ORDER_OF["reverse"]
REF, REF_META = compile_arm("ref_rev_exact", REV, None, "incumbent", True)
REF2, REF2_META = compile_arm("ref2_rev_exact", REV, None, "incumbent", True)
REF_PL, REF_PL_META = compile_arm("refpl_rev_exact", REV, None, "planner", True)
for name, ex in (("ref", REF), ("ref2", REF2), ("ref_planner", REF_PL)):
    _out = ex(*VAL_ARGS)
    v, _ = dense_vector(_out)
    emit({"phase": "value", "tag": name, "vs_jax_grad": agree(v, JAXGRAD),
          "layout": leaf_layouts(_out)})
if ROUNDS > 0:
    measure(REF); measure(REF2); measure(REF_PL)
FLOOR = {"lat": [], "wm": []}
PLREF = {"lat": [], "wm": []}
for r in range(ROUNDS):
    lr, wr = measure(REF)
    l2, w2 = measure(REF2)
    lr2, wr2 = measure(REF)
    lp, wp = measure(REF_PL)
    FLOOR["lat"].append(l2 / lr); FLOOR["wm"].append(w2 / wr if wr else float("nan"))
    PLREF["lat"].append(lp / lr2); PLREF["wm"].append(wp / wr2 if wr2 else float("nan"))
    emit({"phase": "pair", "tag": "floor", "round": r, "ref_lat_ns": lr,
          "arm_lat_ns": l2, "lat_ratio": l2 / lr})
    emit({"phase": "pair", "tag": "ref_planner_vs_tiled", "round": r,
          "ref_lat_ns": lr2, "arm_lat_ns": lp, "lat_ratio": lp / lr2})
def _med(x):
    return statistics.median(x) if x else None


def _mn(x):
    return min(x) if x else None


def _mx(x):
    return max(x) if x else None


emit({"phase": "floor", "lat_median": _med(FLOOR["lat"]),
      "lat_min": _mn(FLOOR["lat"]), "lat_max": _mx(FLOOR["lat"]),
      "wm_median": _med(FLOOR["wm"]),
      "planner_ref_lat_median": _med(PLREF["lat"]),
      "planner_ref_lat_min": _mn(PLREF["lat"]),
      "planner_ref_lat_max": _mx(PLREF["lat"]),
      "planner_ref_temp_ratio": REF_PL_META["temp_bytes"] / REF_META["temp_bytes"],
      "planner_ref_temp_triton_ratio": (
          REF_PL_META["temp_triton_bytes"] / REF_META["temp_triton_bytes"]
          if GPU and isinstance(REF_PL_META.get("temp_triton_bytes"), int)
          and REF_META.get("temp_triton_bytes") else None),
      "planner_ref_fusions": [REF_META["hlo_fusions"], REF_PL_META["hlo_fusions"]],
      "planner_ref_hlo_same": REF_PL_META["hlo_md5"] == REF_META["hlo_md5"]})

# ---------------------------------------------------------------- arms ---
SUMMARY = []
for order_name in ORDERS:
    order = ORDER_OF[order_name]
    plans = build_plans(order_name, order)
    for plan_name, (ft, pmeta) in plans.items():
        tag = f"{order_name}:{plan_name}"
        EX = {}
        META = {}
        VEC = {}
        LAY = {}
        ok = True
        for engine in ENGINES:
            for sparse in (True, False):
                key = (engine, sparse)
                try:
                    EX[key], META[key] = compile_arm(tag, order, ft, engine, sparse)
                    out = EX[key](*VAL_ARGS)
                    VEC[key], _ = dense_vector(out)
                    LAY[key] = leaf_layouts(out)
                except Exception as exc:
                    emit({"phase": "compile", "tag": tag, "engine": engine,
                          "sparse": sparse, "error": repr(exc)[:800]})
                    ok = False
        # values
        vals = {"tag": tag, "phase": "values", **pmeta}
        for engine in ENGINES:
            ks, kd = (engine, True), (engine, False)
            if ks in VEC and kd in VEC:
                vals[f"{engine}:sparse_vs_dense"] = agree(VEC[ks], VEC[kd])
            if ks in VEC:
                vals[f"{engine}:sparse_vs_jax_grad"] = agree(VEC[ks], JAXGRAD)
                vals[f"{engine}:layout_param"] = all(
                    r.get("param_layout", False) for r in LAY[ks])
                vals[f"{engine}:layout"] = LAY[ks]
            if kd in VEC:
                vals[f"{engine}:dense_vs_jax_grad"] = agree(VEC[kd], JAXGRAD)
        for _other in ("incumbent", "planner"):
            if ("candidate", True) in VEC and (_other, True) in VEC:
                vals[f"candidate_vs_{_other}:sparse"] = agree(
                    VEC[("candidate", True)], VEC[(_other, True)])
            if ("candidate", False) in VEC and (_other, False) in VEC:
                vals[f"candidate_vs_{_other}:dense"] = agree(
                    VEC[("candidate", False)], VEC[(_other, False)])
        emit(vals)
        # latency, paired: ref tiled immediately before every arm
        arms = [k for k in (("candidate", True), ("incumbent", True),
                            ("planner", True)) if k in EX]
        for k in arms:
            if ROUNDS > 0:
                measure(EX[k])
        R = {k: {"lat": [], "wm": []} for k in arms}
        PT = []
        for r in range(ROUNDS):
            lats = {}
            for k in arms:
                lr, wr = measure(REF)
                lc, wc = measure(EX[k])
                R[k]["lat"].append(lc / lr)
                R[k]["wm"].append(wc / wr if wr else float("nan"))
                lats[k] = lc
                emit({"phase": "pair", "tag": tag, "engine": k[0], "round": r,
                      "ref_lat_ns": lr, "arm_lat_ns": lc, "lat_ratio": lc / lr,
                      "ref_wm_b": wr, "arm_wm_b": wc})
            if ("candidate", True) in lats and ("incumbent", True) in lats:
                PT.append(lats[("candidate", True)] / lats[("incumbent", True)])
        row = {"phase": "summary", "tag": tag, "order": order_name,
               "plan": plan_name, **pmeta, "n_pairs": ROUNDS}
        for k in arms:
            e = k[0]
            m = META[k]
            row[f"{e}:lat_ratio_median"] = _med(R[k]["lat"])
            row[f"{e}:lat_ratio_min"] = _mn(R[k]["lat"])
            row[f"{e}:lat_ratio_max"] = _mx(R[k]["lat"])
            row[f"{e}:wm_ratio_median"] = _med(R[k]["wm"])
            row[f"{e}:temp_ratio"] = m["temp_bytes"] / REF_META["temp_bytes"]
            row[f"{e}:temp_bytes"] = m["temp_bytes"]
            row[f"{e}:output_bytes"] = m["output_bytes"]
            row[f"{e}:fusions"] = m["hlo_fusions"]
            row[f"{e}:paths"] = m["paths"]
            if GPU and isinstance(m.get("temp_triton_bytes"), int):
                row[f"{e}:temp_triton_bytes"] = m["temp_triton_bytes"]
        if PT:
            row["candidate_over_incumbent:lat_median"] = statistics.median(PT)
            row["candidate_over_incumbent:lat_min"] = min(PT)
            row["candidate_over_incumbent:lat_max"] = max(PT)
            _ci = META[("incumbent", True)]["temp_bytes"]
            row["candidate_over_incumbent:temp_ratio"] = (
                META[("candidate", True)]["temp_bytes"] / _ci if _ci else None)
            if GPU and all(isinstance(META[k].get("temp_triton_bytes"), int)
                           for k in (("candidate", True), ("incumbent", True))):
                tt = META[("incumbent", True)]["temp_triton_bytes"]
                row["candidate_over_incumbent:temp_triton_ratio"] = (
                    META[("candidate", True)]["temp_triton_bytes"] / tt if tt else None)
            row["candidate_over_incumbent:hlo_same"] = (
                META[("candidate", True)]["hlo_md5"]
                == META[("incumbent", True)]["hlo_md5"])
        emit(row)
        SUMMARY.append(row)
        del EX, VEC
        jax.clear_caches()

print("\n[t65] ===== paired ratios vs rev-exact INCUMBENT tiled (ref), "
      f"{EXAMPLE} on {dev}, {ROUNDS} pairs each, one seed =====")
print(f"[t65] drift floor lat {_med(FLOOR['lat'])} "
      f"({_mn(FLOOR['lat'])}..{_mx(FLOOR['lat'])}); planner ref / tiled ref "
      f"lat {_med(PLREF['lat'])} temp "
      f"{REF_PL_META['temp_bytes'] / REF_META['temp_bytes']:.4f} fusions "
      f"{REF_META['hlo_fusions']}->{REF_PL_META['hlo_fusions']}")
print(f"[t65] {'order:plan':26s} {'cand lat':>9s} {'incu lat':>9s} {'plan lat':>9s} "
      f"{'ca/in lat':>9s} {'ca temp':>8s} {'in temp':>8s} {'pl temp':>8s} "
      f"{'fus ca':>6s} {'fus in':>6s} {'fus pl':>6s}")
for row in SUMMARY:
    def f(x, w=9):
        return " " * (w - 3) + "n/a" if x is None else f"{x:{w}.4f}"
    print(f"[t65] {row['tag']:26s} {f(row.get('candidate:lat_ratio_median'))} "
          f"{f(row.get('incumbent:lat_ratio_median'))} "
          f"{f(row.get('planner:lat_ratio_median'))} "
          f"{f(row.get('candidate_over_incumbent:lat_median'))} "
          f"{f(row.get('candidate:temp_ratio'), 8)} "
          f"{f(row.get('incumbent:temp_ratio'), 8)} "
          f"{f(row.get('planner:temp_ratio'), 8)} "
          f"{row.get('candidate:fusions', -1):6d} "
          f"{row.get('incumbent:fusions', -1):6d} "
          f"{row.get('planner:fusions', -1):6d}")
emit({"phase": "done"})
