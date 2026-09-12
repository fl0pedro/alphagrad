#!/usr/bin/env python
"""t284 deliverable (1): what the GPU 10 percent is made of.

    python t284_copy_attribution.py OUT_DIR [cpu|gpu]

Hypothesis (owner D12): keeping the axis on the storing operand costs about 10
percent on GPU because a cuBLAS operand must be a contiguous buffer, so every
slice at the dot's edge becomes its own copy kernel, not because the
contraction itself is slower.

The real numbers this replicates come from the HLO of job 63830, TLM at the
campaign shape, reverse order, exact plan. Emission (2) there launches 12
standalone `wrapped_slice` copy kernels that emissions (1) and (3) do not: six
`f32[32,384] -> f32[32,128]` and six `f32[128,384] -> f32[128,128]`, three
offsets each from four distinct source buffers. Those are the three pieces of
the target's own QKV split, so they are twelve DIFFERENT pieces and there is
nothing to common up.

Arms, all at those exact shapes, all paired in one process:

  sliced      the operand is a slice of the wide buffer, then a 2-D dot.
              This is emission (2)'s kernel shape: 12 copies + 12 GEMMs.
  presliced   the operand is already its own contiguous buffer, then the same
              2-D dot. This is the fold: 0 copies + 12 GEMMs.
  mulreduce   the same contractions written as sum(a * b, axis=k). No GEMM,
              no copy: what emissions (1) and (3) compile to.
  floor       a second executable of `sliced`, the drift floor.

`sliced / presliced` is the price of the copy kernels alone.
`presliced / mulreduce` is the price of the GEMM as a fusion barrier alone.
Their product is the whole gap.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MODE = (sys.argv[2] if len(sys.argv) > 2 else "gpu").lower()
if MODE == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.makedirs(OUT, exist_ok=True)

import numpy as np                                             # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

DEV = jax.devices()[0]
ROUNDS = int(os.environ.get("T284_ROUNDS", "5"))
JSONL = os.path.join(OUT, "t284_copy.jsonl")

# (source rows, wide cols, piece cols, number of source buffers, pieces each)
SEQ, DMODEL, NPIECE = 32, 128, 3
WIDE = DMODEL * NPIECE          # 384
SHAPES = [(SEQ, WIDE), (SEQ, WIDE), (DMODEL, WIDE), (DMODEL, WIDE)]
RNG = np.random.default_rng(0)


def emit(rec, short=None):
    with open(JSONL, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print("[t284k] " + (short if short is not None
                        else json.dumps(rec, default=str)), flush=True)


def _weights():
    return [jnp.asarray(RNG.standard_normal((DMODEL, DMODEL)), jnp.float32)
            for _ in range(len(SHAPES) * NPIECE)]


def f_sliced(wides, ws):
    """The operand is a slice of the wide buffer: emission (2)'s kernel shape."""
    outs = []
    k = 0
    for w in wides:
        for j in range(NPIECE):
            piece = jax.lax.slice(w, (0, j * DMODEL),
                                  (w.shape[0], (j + 1) * DMODEL))
            outs.append(jax.lax.dot_general(
                piece, ws[k], (((1,), (0,)), ((), ()))))
            k += 1
    return sum(jnp.sum(o) for o in outs)


def f_presliced(pieces, ws):
    """The operand is already its own buffer: the fold."""
    return sum(jnp.sum(jax.lax.dot_general(p, ws[k], (((1,), (0,)), ((), ()))))
               for k, p in enumerate(pieces))


def f_mulreduce(pieces, ws):
    """No dot at all, the form emissions (1) and (3) compile to."""
    return sum(jnp.sum(jnp.sum(p[:, :, None] * ws[k][None, :, :], axis=1))
               for k, p in enumerate(pieces))


_CT = re.compile(r'custom_call_target="([^"]+)"')


def census(hlo):
    entry = hlo[hlo.index("\nENTRY "):] if "\nENTRY " in hlo else hlo
    return {"cublas": entry.count("__cublas$"),
            "wrapped_slice": len(re.findall(r"%wrapped_slice[.\d]* =", entry)),
            "fusions": len(re.findall(r"= \S+ fusion\(", entry)),
            "kernels": len(re.findall(r"= .* (?:fusion|custom-call)\(", entry))}


def build(fn, args):
    ex = jax.jit(fn).lower(*args).compile()
    hlo = ex.as_text()
    ma = ex.memory_analysis()
    return ex, {"temp_bytes": int(getattr(ma, "temp_size_in_bytes", -1)),
                **census(hlo)}


def timeit(ex, args, n=200):
    o = ex(*args)
    jax.block_until_ready(o)
    t0 = time.perf_counter()
    for _ in range(n):
        o = ex(*args)
    jax.block_until_ready(o)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    emit({"phase": "start", "device": str(DEV), "platform": DEV.platform,
          "seq": SEQ, "dmodel": DMODEL, "npiece": NPIECE, "rounds": ROUNDS,
          "host": os.uname().nodename}, f"start on {DEV}")
    wides = [jax.device_put(jnp.asarray(RNG.standard_normal(s), jnp.float32), DEV)
             for s in SHAPES]
    pieces = [jax.device_put(w[:, j * DMODEL:(j + 1) * DMODEL], DEV)
              for w in wides for j in range(NPIECE)]
    ws = [jax.device_put(w, DEV) for w in _weights()]

    arms = {
        "sliced": (f_sliced, (wides, ws)),
        "presliced": (f_presliced, (pieces, ws)),
        "mulreduce": (f_mulreduce, (pieces, ws)),
        "floor": (f_sliced, (wides, ws)),
    }
    EX, META, VAL = {}, {}, {}
    for name, (fn, args) in arms.items():
        EX[name], META[name] = build(fn, args)
        VAL[name] = float(np.asarray(EX[name](*args)))
        emit({"phase": "compile", "arm": name, **META[name],
              "value": VAL[name]},
             f"{name:10s} temp {META[name]['temp_bytes']:>10d} B  "
             f"cuBLAS {META[name]['cublas']:3d}  wrapped_slice "
             f"{META[name]['wrapped_slice']:3d}  fusions "
             f"{META[name]['fusions']:3d}  kernels {META[name]['kernels']:3d}")

    lat = {k: [] for k in arms}
    for r in range(ROUNDS):
        for name, (fn, args) in arms.items():
            lat[name].append(timeit(EX[name], args))
    med = {k: float(np.median(v)) for k, v in lat.items()}
    base = med["presliced"]
    row = {"phase": "summary",
           "lat_us": med,
           "floor_over_sliced": med["floor"] / med["sliced"],
           "sliced_over_presliced": med["sliced"] / base,
           "mulreduce_over_presliced": med["mulreduce"] / base,
           "sliced_over_mulreduce": med["sliced"] / med["mulreduce"],
           "census": META,
           "value_max_rel_diff": max(
               abs(VAL[k] - VAL["presliced"]) / max(abs(VAL["presliced"]), 1e-9)
               for k in VAL)}
    emit(row, "")
    print(f"[t284k] drift floor (a second executable of sliced) "
          f"{row['floor_over_sliced']:.4f}")
    print(f"[t284k] sliced / presliced   = {row['sliced_over_presliced']:.4f}"
          f"   <- the price of the 12 copy kernels alone")
    print(f"[t284k] sliced / mulreduce   = {row['sliced_over_mulreduce']:.4f}"
          f"   <- the whole gap, GEMM plus copies")
    print(f"[t284k] presliced / mulreduce= {row['mulreduce_over_presliced']:.4f}"
          f"   (inverted: mulreduce over presliced)")
    print(f"[t284k] value agreement, max relative difference "
          f"{row['value_max_rel_diff']:.3e}")
    emit({"phase": "done"}, "done")


if __name__ == "__main__":
    main()
