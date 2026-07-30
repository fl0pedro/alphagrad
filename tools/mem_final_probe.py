#!/usr/bin/env python3
"""Final checks for the peak-memory reward channel:
  (1) per-call cost of clear_memory_stats() / memory_stats()
  (2) the recommended env.py snippet, end-to-end
  (3) contamination by a REAL OOM (the scenario that once flattened 4 variants)
  (4) drift of the recommended metric over 200 repeated measurements (leak check)
"""
from __future__ import annotations
import gc, os, statistics, time
import jax, jax.numpy as jnp, numpy as np

MB = 2.0 ** 20


# ---- the RECOMMENDED snippet ----------------------------------------------
def measure_peak_delta(compiled, args, devices):
    """Above-baseline peak device bytes for one execution of `compiled`."""
    for d in devices:
        if hasattr(d, "clear_memory_stats"):
            d.clear_memory_stats()
    jax.effects_barrier()
    base = 0.0
    for d in devices:
        base += float((d.memory_stats() or {}).get("bytes_in_use", 0.0))
    out = compiled(*args)
    jax.block_until_ready(out)
    peak = 0.0
    for d in devices:
        peak += float((d.memory_stats() or {}).get("peak_bytes_in_use", 0.0))
    return max(0.0, peak - base), out


def main():
    from graphax.core import jacve
    from graphax.sparse.micro_actions import Quant
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup
    from alphagrad.approx.common.masks import make_live_masked_hook

    dev = jax.devices()[0]
    devs = [dev]
    ex = "VmappedNeuralNetwork"

    # ---- (1) probe cost -------------------------------------------------
    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        dev.memory_stats()
    t_stats = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        dev.clear_memory_stats()
    t_clear = (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(200):
        jax.effects_barrier()
    t_bar = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(20):
        sum(a.nbytes for a in jax.live_arrays())
    t_live = (time.perf_counter() - t0) / 20
    print(f"(1) COST/call: memory_stats={t_stats*1e6:7.2f} us  "
          f"clear_memory_stats={t_clear*1e6:7.2f} us  "
          f"effects_barrier={t_bar*1e6:7.2f} us  "
          f"live_arrays={t_live*1e6:9.2f} us")

    class _A:
        measure_grad = False
        seed_vertices = False
    fn, xs, argnums = grad_target_setup(
        _A(), get_fn(ex), get_args(ex, jax.random.PRNGKey(0), dataset="mnist"), ex)
    n = len(jax.make_jaxpr(fn)(*xs).jaxpr.eqns)
    rev = list(range(n, 0, -1))
    variants = {
        "exact": None,
        "bf16": [(v, (make_live_masked_hook((Quant(dtype="bfloat16"),)),))
                 for v in range(1, n + 1)],
        "f8": [(v, (make_live_masked_hook((Quant(dtype="float8_e4m3fn"),)),))
               for v in range(1, n + 1)],
    }
    comp = {}
    for k, tf in variants.items():
        comp[k] = jax.jit(jacve(fn, rev, argnums=argnums, transforms=tf)).lower(*xs).compile()
        o = comp[k](*xs); jax.block_until_ready(o); del o
    gc.collect()

    def sweep(tag):
        row = {}
        for k in comp:
            v, out = measure_peak_delta(comp[k], xs, devs)
            del out; gc.collect()
            row[k] = v
        abs_row = {}
        for k in comp:
            dev.clear_memory_stats(); jax.effects_barrier()
            o = comp[k](*xs); jax.block_until_ready(o)
            abs_row[k] = float(dev.memory_stats()["peak_bytes_in_use"])
            del o; gc.collect()
        print(f"  {tag:<28} DELTA " +
              " ".join(f"{k}={v/MB:8.3f}" for k, v in row.items()) +
              "   |  ABS " + " ".join(f"{k}={v/MB:9.2f}" for k, v in abs_row.items()))
        return row

    print("\n(3) CONTAMINATION")
    clean = sweep("clean")

    # dirty A: alloc + free 3 GB
    b = jnp.zeros((768, 1024, 1024), jnp.float32); jax.block_until_ready(b); del b
    gc.collect()
    sweep("after 3GB alloc+free")

    # dirty B: a REAL OOM
    try:
        h = jnp.zeros((1024, 1024, 1024, 16), jnp.float32); jax.block_until_ready(h)
        print("  (OOM attempt did not OOM)")
        del h
    except Exception as e:
        print(f"  (OOM raised {type(e).__name__})")
    gc.collect()
    sweep("after real OOM")

    # dirty C: 2 GB held resident
    held = jnp.zeros((512, 1024, 1024), jnp.float32); jax.block_until_ready(held)
    dirty = sweep("with 2GB resident held")
    del held; gc.collect()

    ok = all(abs(clean[k] - dirty[k]) < 1024 for k in clean)
    print(f"  -> DELTA contamination-resistant: {ok} "
          f"(max drift {max(abs(clean[k]-dirty[k]) for k in clean):.0f} bytes)")

    # ---- (4) drift over many repeats -----------------------------------
    print("\n(4) DRIFT over 200 repeated measurements of 'exact' (leak check)")
    vals = []
    t0 = time.perf_counter()
    for i in range(200):
        v, out = measure_peak_delta(comp["exact"], xs, devs)
        del out
        if i % 20 == 0:
            gc.collect()
        vals.append(v)
    dt = time.perf_counter() - t0
    print(f"  first10={[round(v/MB,3) for v in vals[:10]]}")
    print(f"  last10 ={[round(v/MB,3) for v in vals[-10:]]}")
    print(f"  mean={statistics.fmean(vals)/MB:.4f} MB  "
          f"std={statistics.pstdev(vals)/MB:.6f} MB  "
          f"cv={statistics.pstdev(vals)/statistics.fmean(vals)*100:.4f}%  "
          f"min={min(vals)/MB:.4f} max={max(vals)/MB:.4f}")
    print(f"  total {dt:.2f}s for 200 measurements = {dt/200*1e3:.3f} ms each "
          f"(incl. execution)")

    # ---- (2b) same for ResourceMonitor, 200x, leak check ---------------
    from jax_memory_monitor import ResourceMonitor
    print("\n(4b) DRIFT of ResourceMonitor -- FRESH object per call (the documented leak)")
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    vals = []
    t0 = time.perf_counter()
    for i in range(200):
        with ResourceMonitor(devices=devs) as m:
            o = comp["exact"](*xs)
        vals.append(float(m.stats["memory"]))
        del o, m
        if i % 20 == 0:
            gc.collect()
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  first5={[round(v/MB,3) for v in vals[:5]]} "
          f"last5={[round(v/MB,3) for v in vals[-5:]]}  "
          f"cv={statistics.pstdev(vals)/statistics.fmean(vals)*100:.4f}%")
    print(f"  {dt/200*1e3:.3f} ms each; host maxRSS {rss0/1024:.0f} -> {rss1/1024:.0f} MB "
          f"(+{(rss1-rss0)/1024:.1f} MB over 200 fresh monitors)")


if __name__ == "__main__":
    main()
