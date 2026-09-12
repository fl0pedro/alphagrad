#!/usr/bin/env python
"""t284: the three emissions of a single implicit sparse axis, inside the real
engine, on real plans. ONE process, ONE device (ticket dsnn-3qm.28.4).

    python t284_emission_probe.py OUT_DIR EXAMPLE DATASET

Adapted from ``probes/t65/t65_engine_probe.py`` (finding 61). The arms are no
longer two engines; they are three emissions of ONE engine, lane B's lazy
tiled frame:

  bcast_in_dot          GRAPHAX_TILED_LAZY=nodemote
                        the implicit side is broadcast inside the contraction
                        and the axis stays in the dot's batch list. The
                        incumbent form for this axis.
  kept_on_storing_side  GRAPHAX_TILED_LAZY=full
                        the axis rides as a free axis of the storing operand:
                        a clean 2-D dot, which is a cuBLAS call on GPU.
  mul_then_reduce       GRAPHAX_TILED_LAZY=nodemote GRAPHAX_TILED_MULREDUCE=1
                        no dot at all: sum(lhs * rhs, axis=contracted).
  legacy                GRAPHAX_TILED_LEGACY=1 (optional 4th arm, T284_ARMS)
                        the incumbent executor of finding 61, for continuity.

Everything is paired in one process against the reverse-order exact plan of
the first arm, measured immediately before every arm, with a drift floor from
a second executable of that same reference. Ratios only.

Recorded per arm: XLA static temp, the runtime watermark from the campaign
instrument, the HLO kernel census (fusions, cuBLAS custom calls,
wrapped_slice / wrapped_concatenate copies, top-level broadcasts and reduces),
the stored element count of the returned sparse gradients, the parameter
layout, and the value agreement against jax.grad and between the arms.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
EXAMPLE = sys.argv[2] if len(sys.argv) > 2 else "TransformerLM"
DATASET = sys.argv[3] if len(sys.argv) > 3 else "wikitext2"
ROUNDS = int(os.environ.get("T284_ROUNDS", "5"))
ORDERS = os.environ.get("T284_ORDERS", "reverse,markowitz").split(",")
PLANS = os.environ.get("T284_PLANS", "exact,quant_all,compress_all").split(",")
ARMS = os.environ.get(
    "T284_ARMS", "bcast_in_dot,kept_on_storing_side,mul_then_reduce").split(",")
os.makedirs(OUT, exist_ok=True)
JSONL = os.path.join(OUT, "t284_probe.jsonl")

GPU = os.environ.get("ALPHAGRAD_MEASURE_ACTOR", "0") == "1"
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_DIRECT_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_MEASURE_WARMUP", "1")
# The probe owns every engine knob; an inherited value would make an arm
# silently a different arm.
for _k in ("GRAPHAX_PLANNER_EXACT", "GRAPHAX_EINSUM_GENERAL",
           "GRAPHAX_TILED_LAZY", "GRAPHAX_TILED_LEGACY",
           "GRAPHAX_TILED_MULREDUCE"):
    os.environ.pop(_k, None)
# The campaign SHARED_ENV still exports ALPHAGRAD_FORCE_REV_ORDER, which
# ticket dsnn-3qm.64 removed from the code: masks.py raises at import if a
# process still sets it. The probe supplies both elimination orders itself, so
# the variable has nothing to pin here. Drop it before the import.
os.environ.pop("ALPHAGRAD_FORCE_REV_ORDER", None)

_BASE = {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0",
         "GRAPHAX_TILED_LEGACY": "0", "GRAPHAX_TILED_MULREDUCE": "0",
         "GRAPHAX_TILED_LAZY": "nodemote"}
EMISSIONS = {
    "bcast_in_dot": dict(_BASE),
    "kept_on_storing_side": dict(_BASE, GRAPHAX_TILED_LAZY="full"),
    "mul_then_reduce": dict(_BASE, GRAPHAX_TILED_MULREDUCE="1"),
    "legacy": dict(_BASE, GRAPHAX_TILED_LEGACY="1"),
}
ARMS = [a for a in ARMS if a in EMISSIONS]
REF_ARM = ARMS[0]


def emit(rec: dict) -> None:
    rec = dict(rec, t=time.time())
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print("[t284] " + json.dumps(rec, default=str)[:1400], flush=True)


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
CLI = [
    "--example", EXAMPLE, "--dataset", DATASET, "--seed", "250197",
    "--latency-inner-reps", os.environ.get("T284_INNER", "50"),
    "--num-data-points", os.environ.get("T284_POINTS", "5"),
    "--reps-per-point", os.environ.get("T284_REPS", "4"),
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
      "arms": ARMS, "ref_arm": REF_ARM})

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
ORDER_OF = {"reverse": [int(v) for v in lm.rev_order(env)]}
if "markowitz" in ORDERS:
    ORDER_OF["markowitz"] = [int(v) for v in lm.markowitz_order(env)]
emit({"phase": "target", "s": round(time.perf_counter() - t0, 1),
      "n_vertices": len(ORDER_OF["reverse"]), "device": str(dev),
      "n_points": n_points, "n_reps": n_reps, "inner": inner,
      "warmup": warmup})


def _jac(order, sparse: bool, **kw):
    return jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
                 has_aux=cfg.has_aux, sparse_representation=sparse, **kw)


def face_transforms_of(order, plan):
    specs, face_specs, face_skips = lm.get_plan_arrays(plan, len(order))
    return envmod._face_transforms_for_order(
        cfg, env.consts, env.args, list(order), list(np.asarray(specs)),
        list(np.asarray(face_specs)), list(np.asarray(face_skips)))


def build_plans(order_name, order):
    inv = lm.face_inventory(env, order)
    out = {}
    for name in PLANS:
        if name == "exact":
            out[name] = (None, {"n_faces_approx": 0, "total_live_faces": len(inv)})
            continue
        op = {"quant_all": "quant", "diag_all": "diag",
              "compress_all": "compress"}[name]
        plan = lm.build_ladder_plan(env, order, op, "all", lm.ARGS)
        out[name] = (face_transforms_of(order, plan),
                     {k: plan.get(k) for k in ("n_faces_approx", "total_live_faces")})
        emit({"phase": "plan", "order": order_name, "plan": name,
              **out[name][1]})
    return out


def _set_arm(arm):
    saved = {k: os.environ.get(k) for k in EMISSIONS[arm]}
    os.environ.update(EMISSIONS[arm])
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


_CUSTOM = re.compile(r"custom-call\(")


def hlo_census(hlo: str) -> dict:
    entry = hlo.split("ENTRY ")[-1] if "ENTRY " in hlo else hlo
    return {
        "fusions": hlo.count(" fusion("),
        "cublas_calls": hlo.count("__cublas$"),
        "custom_calls": len(_CUSTOM.findall(hlo)),
        "wrapped_slice": hlo.count("wrapped_slice"),
        "wrapped_concatenate": hlo.count("wrapped_concatenate"),
        "top_broadcasts": len(re.findall(r"= \S+ broadcast\(", entry)),
        "top_reduces": len(re.findall(r"= \S+ reduce\(", entry)),
        "top_dots": len(re.findall(r"= \S+ dot\(", entry)),
        "top_copies": len(re.findall(r"= \S+ copy\(", entry)),
    }


def compile_arm(tag, order, ft, arm):
    t0 = time.perf_counter()
    fn = _jac(order, True, transforms=[], face_transforms=ft)
    saved = _set_arm(arm)
    lower_mm.reset_stats()
    d0 = dict(gx_dispatch.DISPATCH_STATS)
    try:
        with track_paths() as paths:
            lowered = jax.jit(fn, keep_unused=True).lower(*args_l)
    finally:
        _restore(saved)
    d1 = gx_dispatch.DISPATCH_STATS
    dispatch_delta = {k: d1[k] - d0.get(k, 0) for k in d1 if d1[k] != d0.get(k, 0)}
    t_low = time.perf_counter() - t0
    ex = envmod._compile_measure(lowered)
    ma = ex.memory_analysis()
    try:
        hlo = ex.as_text()
    except Exception as exc:                      # pragma: no cover
        hlo = f"<as_text failed: {exc!r}>"
    rec = {
        "phase": "compile", "tag": tag, "arm": arm,
        "lower_s": round(t_low, 2),
        "compile_s": round(time.perf_counter() - t0 - t_low, 2),
        "temp_bytes": int(getattr(ma, "temp_size_in_bytes", -1)),
        "argument_bytes": int(getattr(ma, "argument_size_in_bytes", -1)),
        "output_bytes": int(getattr(ma, "output_size_in_bytes", -1)),
        "hlo_md5": hashlib.md5(hlo.encode()).hexdigest(),
        "hlo_lines": hlo.count("\n"),
        **hlo_census(hlo),
        "paths": dict(collections.Counter(paths)),
        "lower_stats": dict(lower_mm.LOWER_STATS),
        "dispatch_delta": dispatch_delta,
    }
    if GPU:
        try:
            ex_t = lowered.compile(compiler_options=_triton_options())
            rec["temp_triton_bytes"] = int(
                getattr(ex_t.memory_analysis(), "temp_size_in_bytes", -1))
            del ex_t
        except Exception as exc:
            rec["temp_triton_bytes"] = f"failed: {exc!r}"[:200]
    emit(rec)
    with open(os.path.join(OUT, f"hlo_{tag.replace(':', '_')}_{arm}.txt"), "w") as fh:
        fh.write(hlo)
    return ex, rec


def _jac_of(out):
    return out[1] if cfg.has_aux else out


def leaf_report(out):
    """Layout and stored element count of every returned parameter gradient."""
    leaves = jax.tree_util.tree_leaves(
        _jac_of(out), is_leaf=lambda x: isinstance(x, SparseTensor))
    rows, stored = [], 0
    for t in leaves:
        if isinstance(t, SparseTensor):
            dims = tuple(t.out_dims) + tuple(t.primal_dims)
            axes = [None if d.axis is None else int(d.axis) for d in dims]
            owned = [(i, a) for i, a in enumerate(axes) if a is not None]
            n = 0 if t.val is None else int(np.prod(t.val.shape))
            stored += n
            rows.append({"axes": axes, "sizes": [int(d.size) for d in dims],
                         "val_shape": None if t.val is None else
                         [int(s) for s in t.val.shape],
                         "stored": n, "dtype": str(t.dtype),
                         "param_layout": all(a == i for i, a in owned)})
        elif hasattr(t, "shape"):
            n = int(np.prod(t.shape))
            stored += n
            rows.append({"dense_shape": [int(s) for s in t.shape], "stored": n,
                         "param_layout": True})
    return rows, stored


def dense_vector(out):
    parts = []
    for t in jax.tree_util.tree_leaves(
            _jac_of(out), is_leaf=lambda x: isinstance(x, SparseTensor)):
        if isinstance(t, SparseTensor):
            parts.append(np.asarray(t.dense(), dtype=np.float64).ravel())
        elif hasattr(t, "shape"):
            parts.append(np.asarray(t, dtype=np.float64).ravel())
    return np.concatenate(parts) if parts else np.zeros(0)


def agree(a, b):
    if a.shape != b.shape:
        return {"shape_mismatch": [int(a.size), int(b.size)]}
    if a.size == 0:
        return {"empty": True}
    d = a - b
    nb, na = float(np.linalg.norm(b)), float(np.linalg.norm(a))
    return {"rel_l2": float(np.linalg.norm(d) / nb) if nb > 0 else None,
            "cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else None,
            "bit_identical": bool(np.array_equal(a, b)),
            "n_nan": int(np.isnan(d).sum())}


VAL_ARGS = list(args_l)
_pb = envmod._probe_batch(cfg, list(env.args), role="train", index=0)
if _pb is not None:
    for _slot in range(min(2, len(_pb))):
        VAL_ARGS[_slot] = jax.device_put(jnp.asarray(_pb[_slot]), dev)
    VAL_SRC = "probe_batch"
else:
    VAL_SRC = "eval_samples"

t0 = time.perf_counter()
_grad_fn = jax.jit(jax.grad(cfg.target_fun, argnums=cfg.argnums,
                            has_aux=cfg.has_aux))
_g = _grad_fn(*VAL_ARGS)
JAXGRAD = dense_vector(_g[0] if cfg.has_aux else _g)
emit({"phase": "oracle_jax_grad", "s": round(time.perf_counter() - t0, 1),
      "value_args": VAL_SRC, "norm": float(np.linalg.norm(JAXGRAD))})


def measure(ex):
    lat, peak = envmod._campaign_measure_cost(
        ex, eval_args_all, unique_devices, inner, warmup, n_reps)
    return float(lat), float(peak)


# ------------------------------------------------------------- reference ---
REV = ORDER_OF["reverse"]
REF, REF_META = compile_arm("ref_rev_exact", REV, None, REF_ARM)
REF2, REF2_META = compile_arm("ref2_rev_exact", REV, None, REF_ARM)
_out = REF(*VAL_ARGS)
emit({"phase": "value", "tag": "ref", "arm": REF_ARM,
      "vs_jax_grad": agree(dense_vector(_out), JAXGRAD)})
if ROUNDS > 0:
    measure(REF); measure(REF2)
FLOOR = {"lat": [], "wm": []}
for r in range(ROUNDS):
    lr, wr = measure(REF)
    l2, w2 = measure(REF2)
    FLOOR["lat"].append(l2 / lr)
    FLOOR["wm"].append(w2 / wr if wr else float("nan"))


def _med(x):
    return statistics.median(x) if x else None


emit({"phase": "floor", "lat_median": _med(FLOOR["lat"]),
      "lat_min": min(FLOOR["lat"]) if FLOOR["lat"] else None,
      "lat_max": max(FLOOR["lat"]) if FLOOR["lat"] else None,
      "wm_median": _med(FLOOR["wm"]),
      "ref_temp_bytes": REF_META["temp_bytes"],
      "ref_fusions": REF_META["fusions"],
      "ref_cublas": REF_META["cublas_calls"]})

# ------------------------------------------------------------------ arms ---
SUMMARY = []
for order_name in ORDERS:
    order = ORDER_OF[order_name]
    plans = build_plans(order_name, order)
    for plan_name, (ft, pmeta) in plans.items():
        tag = f"{order_name}:{plan_name}"
        EX, META, VEC, STORED, LAY = {}, {}, {}, {}, {}
        for arm in ARMS:
            try:
                EX[arm], META[arm] = compile_arm(tag, order, ft, arm)
                out = EX[arm](*VAL_ARGS)
                VEC[arm] = dense_vector(out)
                LAY[arm], STORED[arm] = leaf_report(out)
            except Exception as exc:
                emit({"phase": "compile", "tag": tag, "arm": arm,
                      "error": repr(exc)[:800]})
        vals = {"tag": tag, "phase": "values", **pmeta}
        for arm in VEC:
            vals[f"{arm}:vs_jax_grad"] = agree(VEC[arm], JAXGRAD)
            vals[f"{arm}:stored_elems"] = STORED[arm]
            vals[f"{arm}:param_layout_all"] = all(
                r.get("param_layout", False) for r in LAY[arm])
            vals[f"{arm}:layout"] = LAY[arm]
            if arm != REF_ARM and REF_ARM in VEC:
                vals[f"{arm}:vs_{REF_ARM}"] = agree(VEC[arm], VEC[REF_ARM])
        emit(vals)

        arms = [a for a in ARMS if a in EX]
        for a in arms:
            if ROUNDS > 0:
                measure(EX[a])
        R = {a: {"lat": [], "wm": []} for a in arms}
        for r in range(ROUNDS):
            for a in arms:
                lr, wr = measure(REF)
                lc, wc = measure(EX[a])
                R[a]["lat"].append(lc / lr)
                R[a]["wm"].append(wc / wr if wr else float("nan"))
        row = {"phase": "summary", "tag": tag, "order": order_name,
               "plan": plan_name, **pmeta, "n_pairs": ROUNDS}
        for a in arms:
            m = META[a]
            row[f"{a}:lat_ratio_median"] = _med(R[a]["lat"])
            row[f"{a}:lat_ratio_min"] = min(R[a]["lat"])
            row[f"{a}:lat_ratio_max"] = max(R[a]["lat"])
            row[f"{a}:wm_ratio_median"] = _med(R[a]["wm"])
            row[f"{a}:temp_ratio"] = (m["temp_bytes"] / REF_META["temp_bytes"]
                                      if REF_META["temp_bytes"] else None)
            row[f"{a}:temp_bytes"] = m["temp_bytes"]
            row[f"{a}:stored_elems"] = STORED.get(a)
            for c in ("fusions", "cublas_calls", "wrapped_slice",
                      "wrapped_concatenate", "top_broadcasts", "top_reduces",
                      "top_dots", "top_copies"):
                row[f"{a}:{c}"] = m[c]
            if GPU and isinstance(m.get("temp_triton_bytes"), int):
                row[f"{a}:temp_triton_bytes"] = m["temp_triton_bytes"]
        # every arm over the reference arm, same rounds
        base = row.get(f"{REF_ARM}:lat_ratio_median")
        for a in arms:
            if base:
                row[f"{a}:over_ref_lat"] = row[f"{a}:lat_ratio_median"] / base
            tb = META.get(REF_ARM, {}).get("temp_bytes")
            if tb:
                row[f"{a}:over_ref_temp"] = META[a]["temp_bytes"] / tb
        emit(row)
        SUMMARY.append(row)
        del EX, VEC
        jax.clear_caches()

print(f"\n[t284] ===== {EXAMPLE} on {dev}, {ROUNDS} pairs, ratios against "
      f"the rev-exact {REF_ARM} executable =====")
print(f"[t284] drift floor lat {_med(FLOOR['lat'])} "
      f"({min(FLOOR['lat']) if FLOOR['lat'] else None}.."
      f"{max(FLOOR['lat']) if FLOOR['lat'] else None})")
hdr = f"[t284] {'order:plan':22s}"
for a in ARMS:
    hdr += f" {a[:14]:>14s}"
print(hdr + "   (latency ratio; then temp bytes; then cuBLAS/fusions)")
for row in SUMMARY:
    line = f"[t284] {row['tag']:22s}"
    for a in ARMS:
        v = row.get(f"{a}:lat_ratio_median")
        line += "            n/a" if v is None else f" {v:14.4f}"
    print(line)
    line = f"[t284] {'  temp':22s}"
    for a in ARMS:
        v = row.get(f"{a}:temp_bytes")
        line += "            n/a" if v is None else f" {v:14d}"
    print(line)
    line = f"[t284] {'  cublas/fusions':22s}"
    for a in ARMS:
        line += f" {str(row.get(a + ':cublas_calls')) + '/' + str(row.get(a + ':fusions')):>14s}"
    print(line)
emit({"phase": "done"})
