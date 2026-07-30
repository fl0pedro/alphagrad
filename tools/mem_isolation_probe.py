#!/usr/bin/env python3
"""Is the peak-delta channel poisoned by CONCURRENT device activity?

peak_bytes_in_use is a DEVICE-WIDE high-water mark. env.py runs 16 envs; if
several measure on the same GPU at once, one env's transient enters another
env's reading. Also isolates the additive offset seen between scripts:
measure `exact` alone vs with sibling executables resident.
"""
from __future__ import annotations
import gc, sys, threading, time, statistics
import jax, jax.numpy as jnp

MB = 2.0 ** 20


def measure_peak_delta(compiled, args, devices):
    for d in devices:
        if hasattr(d, "clear_memory_stats"):
            d.clear_memory_stats()
    jax.effects_barrier()
    base = sum(float((d.memory_stats() or {}).get("bytes_in_use", 0.0)) for d in devices)
    out = compiled(*args)
    jax.block_until_ready(out)
    peak = sum(float((d.memory_stats() or {}).get("peak_bytes_in_use", 0.0)) for d in devices)
    return max(0.0, peak - base), out


def main():
    from graphax.core import jacve
    from graphax.sparse.micro_actions import Quant
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup
    from alphagrad.approx.common.masks import make_live_masked_hook

    dev = jax.devices()[0]; devs = [dev]
    ex = "VmappedNeuralNetwork"

    class _A:
        measure_grad = False
        seed_vertices = False
    fn, xs, argnums = grad_target_setup(
        _A(), get_fn(ex), get_args(ex, jax.random.PRNGKey(0), dataset="mnist"), ex)
    n = len(jax.make_jaxpr(fn)(*xs).jaxpr.eqns)
    rev = list(range(n, 0, -1))

    mode = sys.argv[1] if len(sys.argv) > 1 else "solo"

    def build(tf):
        c = jax.jit(jacve(fn, rev, argnums=argnums, transforms=tf)).lower(*xs).compile()
        o = c(*xs); jax.block_until_ready(o); del o; gc.collect()
        return c

    if mode == "solo":
        c = build(None)
        vals = []
        for _ in range(5):
            v, o = measure_peak_delta(c, xs, devs); del o; gc.collect(); vals.append(v)
        print(f"SOLO   exact-only process: {statistics.fmean(vals)/MB:.4f} MB  "
              f"(all: {[round(v/MB,4) for v in vals]})")
        return

    # siblings resident
    cs = {"exact": build(None)}
    for lbl, dt in (("bf16", "bfloat16"), ("f8", "float8_e4m3fn")):
        cs[lbl] = build([(v, (make_live_masked_hook((Quant(dtype=dt),)),))
                         for v in range(1, n + 1)])
    vals = []
    for _ in range(5):
        v, o = measure_peak_delta(cs["exact"], xs, devs); del o; gc.collect(); vals.append(v)
    print(f"SIBLINGS exact w/ 2 other executables resident: "
          f"{statistics.fmean(vals)/MB:.4f} MB  {[round(v/MB,4) for v in vals]}")

    # --- CONCURRENCY: a second thread allocating on the same device --------
    stop = threading.Event()
    err = []

    def noisy():
        try:
            while not stop.is_set():
                a = jnp.zeros((64, 1024, 1024), jnp.float32)   # 256 MB
                jax.block_until_ready(a)
                del a
        except Exception as e:
            err.append(str(e)[:100])

    th = threading.Thread(target=noisy, daemon=True)
    th.start()
    time.sleep(0.5)
    vals2 = []
    for _ in range(8):
        v, o = measure_peak_delta(cs["exact"], xs, devs); del o; gc.collect(); vals2.append(v)
    stop.set(); th.join(timeout=5)
    gc.collect()
    print(f"CONCURRENT (a 2nd thread churning 256 MB on the same GPU): "
          f"mean={statistics.fmean(vals2)/MB:.2f} MB "
          f"min={min(vals2)/MB:.2f} max={max(vals2)/MB:.2f} "
          f"cv={statistics.pstdev(vals2)/statistics.fmean(vals2)*100:.1f}%")
    print(f"  values: {[round(v/MB,2) for v in vals2]}   thread_err={err}")

    vals3 = []
    for _ in range(5):
        v, o = measure_peak_delta(cs["exact"], xs, devs); del o; gc.collect(); vals3.append(v)
    print(f"RECOVERED after noise stops: {statistics.fmean(vals3)/MB:.4f} MB "
          f"{[round(v/MB,4) for v in vals3]}")


if __name__ == "__main__":
    main()
