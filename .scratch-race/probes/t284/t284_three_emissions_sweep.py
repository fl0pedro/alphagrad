#!/usr/bin/env python
"""dsnn-3qm.28.4 deliverable (a): the three emissions of a single implicit
sparse axis, swept over shape, on one device.

Extends finding 62's 80-cell grid (probes/t281/t281_size_rule_sweep.py) by the
third emission. One contraction: meta extent M (the implicit sparse axis, the
meta axis of a diagonal pair, stored by exactly one side), block extent P,
contracted extent K, output extent Q.

  (a) BROADCAST-IN-DOT      broadcast the implicit side up to M, then one
                            dot_general with M in the batch list. The
                            incumbent tiled form.
  (b) KEPT-ON-STORING-SIDE  the axis rides as a free axis of the storing
                            operand: a clean 2-D dot. The planner's form and
                            GRAPHAX_TILED_LAZY=full.
  (c) MULTIPLY-THEN-REDUCE  no dot at all: sum(l * r, axis=K) over the
                            broadcast shapes.

Per cell it records: XLA static temp, the runtime watermark where the device
reports one, the HLO kernel census (fusions, cuBLAS custom calls,
wrapped_slice / wrapped_concatenate copies, top-level broadcasts, reduces),
the stored element count, paired latency in one process with a drift floor,
and the value agreement between the three emissions.

Usage: t284_three_emissions_sweep.py OUT_DIR [cpu|gpu]
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MODE = (sys.argv[2] if len(sys.argv) > 2 else "cpu").lower()
if MODE == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.makedirs(OUT, exist_ok=True)

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

Q = int(os.environ.get("T284_Q", "32"))
M_VALUES = [int(x) for x in os.environ.get("T284_M", "4,16,64,256,1024").split(",")]
P_VALUES = [int(x) for x in os.environ.get("T284_P", "1,4,16,64").split(",")]
K_VALUES = [int(x) for x in os.environ.get("T284_K", "4,16,64,256").split(",")]
ROUNDS = int(os.environ.get("T284_ROUNDS", "3"))
JSONL = os.path.join(OUT, "t284_sweep.jsonl")
DEV = jax.devices()[0]


def emit(rec, short=None):
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print("[t284] " + (short if short is not None
                       else json.dumps(rec, default=str)), flush=True)


def short_line(row):
    def g(k, d="n/a"):
        v = row.get(k)
        return d if v is None else v
    return (f"M={row['M']:5d} P={row['P']:3d} K={row['K']:4d} | "
            f"lat a/b {g('a_bcast_in_dot:lat_over_b', 0):6.3f} "
            f"c/b {g('c_mul_then_reduce:lat_over_b', 0):7.3f} "
            f"floor {g('floor_b2:lat_over_b', 0):6.3f} | "
            f"temp a {g('a_bcast_in_dot:temp_bytes', -1):10d} "
            f"b {g('b_kept_on_storing_side:temp_bytes', -1):10d} "
            f"c {g('c_mul_then_reduce:temp_bytes', -1):10d} | "
            f"gemm a/b/c {g('a_bcast_in_dot:cublas_calls', -1)}/"
            f"{g('b_kept_on_storing_side:cublas_calls', -1)}/"
            f"{g('c_mul_then_reduce:cublas_calls', -1)} "
            f"fus {g('a_bcast_in_dot:fusions', -1)}/"
            f"{g('b_kept_on_storing_side:fusions', -1)}/"
            f"{g('c_mul_then_reduce:fusions', -1)}")


# ------------------------------------------------------------- emissions ---
def emission_a(l, r, M, K, Q):
    """broadcast the implicit side inside the contraction (incumbent tiled)."""
    rb = jnp.broadcast_to(r, (M, K, Q))
    return jax.lax.dot_general(l, rb, (((2,), (1,)), ((0,), (0,))))


def emission_b(l, r, M, P, K, Q):
    """keep the axis on the storing operand (planner, TILED_LAZY=full)."""
    d = jax.lax.dot_general(jnp.reshape(l, (M * P, K)), r, (((1,), (0,)), ((), ())))
    return jnp.reshape(d, (M, P, Q))


def emission_c(l, r, M, P, K, Q):
    """multiply then reduce: never emit a dot."""
    return jnp.sum(l[:, :, :, None] * r[None, None, :, :], axis=2)


# ----------------------------------------------------------------- census ---
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
        "lines": hlo.count("\n"),
    }


def build(fn, args):
    jitted = jax.jit(fn)
    compiled = jitted.lower(*args).compile()
    ma = compiled.memory_analysis()
    hlo = compiled.as_text()
    meta = {
        "temp_bytes": int(getattr(ma, "temp_size_in_bytes", -1)),
        "argument_bytes": int(getattr(ma, "argument_size_in_bytes", -1)),
        "output_bytes": int(getattr(ma, "output_size_in_bytes", -1)),
        **hlo_census(hlo),
    }
    return compiled, meta


def watermark():
    """Runtime high-water mark where the device reports one (GPU only)."""
    try:
        st = DEV.memory_stats() or {}
    except Exception:
        return None
    for k in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
        if k in st:
            return int(st[k])
    return None


def timeit(compiled, args, n):
    out = compiled(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = compiled(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e6


# -------------------------------------------------------------------- run ---
def main():
    emit({"phase": "start", "device": str(DEV), "platform": DEV.platform,
          "mode": MODE, "jax": jax.__version__, "Q": Q, "rounds": ROUNDS,
          "M": M_VALUES, "P": P_VALUES, "K": K_VALUES,
          "host": os.uname().nodename})
    rows = []
    for M, P, K in itertools.product(M_VALUES, P_VALUES, K_VALUES):
        grid = M * P * K * Q
        n = max(20, min(200, int(4e8 // max(grid, 1))))
        L = jax.device_put(jnp.asarray(
            np.random.default_rng(0).standard_normal((M, P, K)), jnp.float32), DEV)
        R3 = jax.device_put(jnp.asarray(
            np.random.default_rng(1).standard_normal((1, K, Q)), jnp.float32), DEV)
        R2 = jnp.reshape(R3, (K, Q))

        specs = {
            "a_bcast_in_dot": (lambda l, r: emission_a(l, r, M, K, Q), (L, R3)),
            "b_kept_on_storing_side": (lambda l, r: emission_b(l, r, M, P, K, Q), (L, R2)),
            "c_mul_then_reduce": (lambda l, r: emission_c(l, r, M, P, K, Q), (L, R2)),
        }
        # A second build of (b) is the drift floor: same emission, same shape,
        # a separate executable, measured in the same rounds as the arms.
        specs["floor_b2"] = (lambda l, r: emission_b(l, r, M, P, K, Q), (L, R2))

        EX, META, VAL, WM = {}, {}, {}, {}
        for name, (fn, args) in specs.items():
            try:
                wm0 = watermark()
                EX[name], META[name] = build(fn, args)
                out = EX[name](*args)
                jax.block_until_ready(out)
                WM[name] = None if wm0 is None else (watermark() or 0) - wm0
                VAL[name] = np.asarray(out, dtype=np.float64)
            except Exception as ex:
                emit({"phase": "build_error", "M": M, "P": P, "K": K,
                      "emission": name, "error": f"{type(ex).__name__}: {str(ex)[:200]}"})

        # paired latency, all arms back to back, ROUNDS times
        lat = {k: [] for k in EX}
        for _ in range(ROUNDS):
            for name in specs:
                if name in EX:
                    lat[name].append(timeit(EX[name], specs[name][1], n))
        med = {k: float(np.median(v)) for k, v in lat.items() if v}

        base = med.get("b_kept_on_storing_side")
        row = {
            "phase": "cell", "M": M, "P": P, "K": K, "Q": Q, "n_reps": n,
            "stored_elems_lhs": M * P * K,
            "stored_elems_rhs": K * Q,
            "broadcast_grid_elems": M * K * Q,
            "mul_grid_elems": grid,
            "macs_per_output_elem": K,
            "output_elems": M * P * Q,
            "bcast_share": (M * K * Q) / (M * P * K + K * Q),
        }
        for name in ("a_bcast_in_dot", "b_kept_on_storing_side",
                     "c_mul_then_reduce", "floor_b2"):
            m = META.get(name)
            if m is None:
                continue
            row[f"{name}:lat_us"] = med.get(name)
            row[f"{name}:lat_over_b"] = (med[name] / base) if base else None
            row[f"{name}:temp_bytes"] = m["temp_bytes"]
            row[f"{name}:wm_delta_bytes"] = WM.get(name)
            for c in ("fusions", "cublas_calls", "custom_calls", "wrapped_slice",
                      "wrapped_concatenate", "top_broadcasts", "top_reduces",
                      "top_dots", "top_copies"):
                row[f"{name}:{c}"] = m[c]
        # value agreement between the three emissions
        ref = VAL.get("b_kept_on_storing_side")
        if ref is not None:
            nb = float(np.linalg.norm(ref))
            for name in ("a_bcast_in_dot", "c_mul_then_reduce"):
                v = VAL.get(name)
                if v is None or v.shape != ref.shape:
                    continue
                row[f"{name}:rel_l2_vs_b"] = (
                    float(np.linalg.norm(v - ref) / nb) if nb > 0 else None)
                row[f"{name}:bit_identical_vs_b"] = bool(np.array_equal(v, ref))
        rows.append(row)
        emit(row, short_line(row))
        del EX, VAL
        jax.clear_caches()

    with open(os.path.join(OUT, "t284_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    emit({"phase": "done", "cells": len(rows)})


if __name__ == "__main__":
    main()
