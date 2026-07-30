#!/usr/bin/env python3
"""Rank candidate peak-memory measurement methods for the RL reward channel.

Runs a matrix of METHOD x VARIANT x TARGET and reports discrimination,
reproducibility (in-process N reps + rotated variant order), accuracy vs the
analytically-known output size, contamination resistance and cost.

Autotuning must be OFF (XLA_FLAGS=--xla_gpu_autotune_level=0) and
XLA_PYTHON_CLIENT_PREALLOCATE=false, else nothing here is meaningful.

Usage:
  uv run --no-sync python tools/mem_method_bench.py [--reps 7] [--json out.json]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

STAT_KEYS = [
    "bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "peak_bytes_reserved",
    "pool_bytes", "peak_pool_bytes", "largest_alloc_size", "num_allocs",
    "largest_free_block_bytes", "bytes_limit",
]

MB = float(2 ** 20)


def read_stats(dev):
    try:
        s = dev.memory_stats() or {}
    except Exception:
        s = {}
    return {k: float(s.get(k, float("nan"))) for k in STAT_KEYS}


def live_bytes():
    try:
        return float(sum(a.nbytes for a in jax.live_arrays()))
    except Exception:
        return float("nan")


def profile_bytes():
    """jax.profiler.device_memory_profile() total, decoded via pprof proto."""
    try:
        from jax_memory_monitor.jax_peak_memory_monitor import device_memory
        return float(device_memory())
    except Exception:
        pass
    try:
        import gzip
        from jax_memory_monitor import profile_pb2
        raw = jax.profiler.device_memory_profile()
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
        p = profile_pb2.Profile()
        p.ParseFromString(raw)
        s = set((tuple(x.location_id), x.value[1]) for x in p.sample)
        return float(sum(v for _, v in s))
    except Exception:
        return float("nan")


def out_bytes(out):
    return float(sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(out)))


# ---------------------------------------------------------------------------
# Targets / variants
# ---------------------------------------------------------------------------
EXAMPLE = os.environ.get("BENCH_EXAMPLE", "VmappedNeuralNetwork")


def build_target(kind):
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup

    class _A:
        pass
    a = _A()
    if kind == "jac":
        a.measure_grad = False
        a.seed_vertices = False
    elif kind == "gradseed":
        a.measure_grad = True
        a.seed_vertices = True
    elif kind == "gradscalar":
        a.measure_grad = True
        a.seed_vertices = False
    else:
        raise ValueError(kind)
    base = get_fn(EXAMPLE)
    xs0 = get_args(EXAMPLE, jax.random.PRNGKey(0), dataset="mnist")
    fn, xs, argnums = grad_target_setup(a, base, xs0, EXAMPLE)
    cj = jax.make_jaxpr(fn)(*xs)
    n = len(cj.jaxpr.eqns)
    return fn, tuple(xs), tuple(argnums), n, cj


def build_order_variants(n, k=6):
    """EXACT AD under different elimination ORDERS -- the other half of the RL
    action space. All are numerically identical (exact AD is order-invariant),
    so any memory difference is pure schedule cost: the cleanest possible
    discrimination test for a memory channel."""
    import random
    rev = list(range(n, 0, -1))
    fwd = list(range(1, n + 1))
    outs = [("order_reverse", rev), ("order_forward", fwd)]
    rng = random.Random(20260728)
    for i in range(k):
        o = fwd[:]
        rng.shuffle(o)
        outs.append((f"order_rand{i}", o))
    return outs


def build_variants(n):
    from graphax.sparse.micro_actions import Compress, Diag, Quant
    from alphagrad.approx.common.masks import make_live_masked_hook

    def allv(rules, stats=None):
        return [(v, (make_live_masked_hook(rules, stats=stats),))
                for v in range(1, n + 1)]

    out = [
        ("exact", None),
        ("bfloat16_all", allv((Quant(dtype="bfloat16"),))),
        ("float16_all", allv((Quant(dtype="float16"),))),
        ("float8_e4m3_all", allv((Quant(dtype="float8_e4m3fn"),))),
        ("diag01_all", allv((Diag(i=0, j=1, factor=1),))),
        ("compress_ax0_last3", [(v, (make_live_masked_hook((Compress(axes=(0,)),)),))
                                for v in range(max(1, n - 2), n + 1)]),
        ("compress_ax0_all", allv((Compress(axes=(0,)),))),
    ]
    return out


def hlo_dtype_census(comp):
    """Count dtype tokens in the optimized HLO -- proves an approximation
    actually reached XLA rather than being silently masked away."""
    try:
        txt = comp.as_text()
    except Exception:
        return {}
    import re
    cens = {}
    for tok in ("bf16", "f16", "f8e4m3fn", "f8e5m2", "s8", "f32", "f64"):
        cens[tok] = len(re.findall(r"\b" + tok + r"\[", txt))
    cens["_hlo_chars"] = len(txt)
    return cens


# ---------------------------------------------------------------------------
# One measurement round
# ---------------------------------------------------------------------------

def measure_once(dev, f, xs, want_monitor, want_profile):
    """Return dict of every runtime candidate for one execution of `f`."""
    rec = {}
    gc.collect()
    if hasattr(dev, "clear_memory_stats"):
        dev.clear_memory_stats()
    jax.effects_barrier()

    t_probe0 = time.perf_counter()
    before = read_stats(dev)
    t_probe1 = time.perf_counter()
    lb_before = live_bytes()
    t_probe1b = time.perf_counter()

    t0 = time.perf_counter()
    out = f(*xs)
    jax.block_until_ready(out)
    t1 = time.perf_counter()

    t_probe2 = time.perf_counter()
    after = read_stats(dev)
    t_probe3 = time.perf_counter()
    lb_after = live_bytes()
    t_probe3b = time.perf_counter()

    rec["exec_s"] = t1 - t0
    rec["probe_stats_s"] = (t_probe1 - t_probe0) + (t_probe3 - t_probe2)
    rec["probe_live_s"] = (t_probe1b - t_probe1) + (t_probe3b - t_probe3)
    rec["probe_s"] = rec["probe_stats_s"] + rec["probe_live_s"]
    rec["out_bytes"] = out_bytes(out)

    # --- method family 1/2/3/4 : memory_stats
    rec["M1_abs_peak"] = after["peak_bytes_in_use"]
    rec["M2_delta_peak"] = after["peak_bytes_in_use"] - before["bytes_in_use"]
    rec["M3_steady_delta"] = after["bytes_in_use"] - before["bytes_in_use"]
    for k in STAT_KEYS:
        rec["S_before_" + k] = before[k]
        rec["S_after_" + k] = after[k]
    rec["M4a_peak_reserved"] = after["peak_bytes_reserved"]
    rec["M4b_peak_pool"] = after["peak_pool_bytes"]
    rec["M4c_largest_alloc"] = after["largest_alloc_size"]
    rec["M4d_num_allocs_delta"] = after["num_allocs"] - before["num_allocs"]
    rec["M4e_pool_delta"] = after["pool_bytes"] - before["pool_bytes"]
    # peak-minus-output: strip the (retained) result so only the transient shows
    rec["M2b_delta_peak_minus_out"] = rec["M2_delta_peak"] - rec["out_bytes"]

    # --- method 7 : live_arrays
    rec["M7_live_delta"] = lb_after - lb_before

    # --- method 8 : device_memory_profile
    if want_profile:
        tp0 = time.perf_counter()
        rec["M8_profile_after"] = profile_bytes()
        rec["M8_cost_s"] = time.perf_counter() - tp0
    del out
    gc.collect()
    return rec


def measure_monitor(devs, f, xs):
    """Method 6: project ResourceMonitor (above-baseline peak, C++ tracker)."""
    from jax_memory_monitor import ResourceMonitor
    t0 = time.perf_counter()
    with ResourceMonitor(devices=devs) as mon:
        out = f(*xs)
    peak = float(mon.stats.get("memory", 0.0))
    dt = time.perf_counter() - t0
    del out
    gc.collect()
    return peak, dt


def measure_monitor_reused(mon, f, xs):
    with mon:
        out = f(*xs)
    peak = float(mon.stats.get("memory", 0.0))
    del out
    gc.collect()
    return peak


# ---------------------------------------------------------------------------

def agg(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return dict(mean=float("nan"), std=float("nan"), cv=float("nan"),
                    mn=float("nan"), mx=float("nan"), n=0)
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return dict(mean=m, std=s, cv=(s / abs(m) if m else float("nan")),
                mn=min(vals), mx=max(vals), n=len(vals))


def run_control(dev, reps, results):
    """Synthetic control: pure-JAX functions with ANALYTICALLY KNOWN peak
    memory that genuinely differ. Validates that the measurement methods can
    detect a real difference at all -- decoupling 'method is blind' from
    'the graphax variants happen not to differ'."""
    devs = [dev]
    N = 2048
    x = jax.random.normal(jax.random.PRNGKey(0), (N, N), jnp.float32)

    def mk(dtype, chain):
        def f(a):
            b = a.astype(dtype)
            acc = b
            keep = []
            for i in range(chain):
                acc = jnp.tanh(acc) + b * (i + 1)
                keep.append(acc)          # force all intermediates live
            return jnp.stack(keep).astype(jnp.float32)
        return f

    cases = [
        ("f32_chain2", jnp.float32, 2),
        ("f32_chain4", jnp.float32, 4),
        ("f32_chain8", jnp.float32, 8),
        ("bf16_chain8", jnp.bfloat16, 8),
    ]
    comp, static = {}, {}
    print(f"\n{'='*100}\nTARGET=control (synthetic, analytic ground truth) "
          f"N={N} base={N*N*4/MB:.1f} MB/f32 array\n{'='*100}")
    for label, dt, ch in cases:
        c = jax.jit(mk(dt, ch)).lower(x).compile()
        o = c(x); jax.block_until_ready(o); ob = out_bytes(o); del o; gc.collect()
        ma = c.memory_analysis()
        d = {k: float(getattr(ma, k, float("nan"))) for k in (
            "temp_size_in_bytes", "argument_size_in_bytes",
            "output_size_in_bytes", "alias_size_in_bytes",
            "generated_code_size_in_bytes")}
        comp[label] = c
        # analytic: output = chain * N*N*4 bytes (stacked, cast to f32)
        analytic_out = ch * N * N * 4
        static[label] = dict(memory_analysis=d, ma_cost_s=0.0, compile_s=0.0,
                             out_bytes=ob, analytic_out=analytic_out)
        print(f"  {label:<14} out={ob/MB:8.2f} MB (analytic {analytic_out/MB:8.2f}) "
              f"temp={d['temp_size_in_bytes']/MB:8.2f} MB")

    labels = list(comp)
    raw = {l: [] for l in labels}
    mon_raw = {l: [] for l in labels}
    from jax_memory_monitor import ResourceMonitor
    shared_mon = ResourceMonitor(devices=devs)
    for rep in range(reps):
        k = rep % len(labels)
        order = labels[k:] + labels[:k]
        if rep % 2 == 1:
            order = list(reversed(order))
        for l in order:
            raw[l].append(measure_once(dev, comp[l], (x,), False, False))
            mon_raw[l].append(measure_monitor_reused(shared_mon, comp[l], (x,)))

    print(f"\n--- control: CLEAN (mean MB / CV%) ---")
    hdr = f"{'method':<32}" + "".join(f"{l[:15]:>17}" for l in labels)
    print(hdr); print("-" * len(hdr))
    for key, name in METHODS:
        cells, vals = "", []
        for l in labels:
            a = agg([r[key] for r in raw[l]])
            vals.append(a["mean"])
            u = a["mean"] if key == "M4d_num_allocs_delta" else a["mean"] / MB
            cells += f"{u:>11.2f}/{a['cv']*100:>4.1f}"
        sp = (max(vals) - min(vals)) / abs(max(vals)) if max(vals) else 0.0
        print(f"{name:<32}{cells}   spread={sp*100:5.1f}%")
    cells = ""
    for l in labels:
        a = agg(mon_raw[l]); cells += f"{a['mean']/MB:>11.2f}/{a['cv']*100:>4.1f}"
    print(f"{'ResourceMonitor(reused)':<32}{cells}")
    cells = ""
    for l in labels:
        ma = static[l]["memory_analysis"]
        t = ma["temp_size_in_bytes"] + ma["output_size_in_bytes"] + ma["argument_size_in_bytes"]
        cells += f"{t/MB:>11.2f}/ 0.0"
    print(f"{'XLA memory_analysis t+o+a':<32}{cells}")
    results["control"] = {"static": static, "clean": raw, "monitor": mon_raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--targets", default="jac,gradseed")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--example", default=None)
    ap.add_argument("--orders", action="store_true",
                    help="sweep elimination ORDERS (exact AD) instead of approx variants")
    ap.add_argument("--profile", action="store_true",
                    help="also run jax.profiler.device_memory_profile (slow)")
    args = ap.parse_args()

    global EXAMPLE
    if args.example:
        EXAMPLE = args.example
    dev = jax.devices()[0]
    devs = [dev]
    print(f"# example={EXAMPLE}")
    print(f"# tag={args.tag} jax={jax.__version__} dev={dev} "
          f"backend={jax.default_backend()} pid={os.getpid()}")
    print(f"# XLA_FLAGS={os.environ.get('XLA_FLAGS')}")
    print(f"# PREALLOC={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}")
    print(f"# memory_stats keys: {sorted((dev.memory_stats() or {}).keys())}")

    from graphax.core import jacve

    results = {"tag": args.tag, "pid": os.getpid(), "targets": {}}

    tkinds = [t for t in args.targets.split(",") if t]
    if "control" in tkinds:
        run_control(dev, args.reps, results)
        tkinds = [t for t in tkinds if t != "control"]

    for tkind in tkinds:
        fn, xs, argnums, n, cj = build_target(tkind)
        rev = list(range(n, 0, -1))
        inp_b = float(sum(np.asarray(a).nbytes for a in xs))
        print(f"\n{'='*100}\nTARGET={tkind}  vertices={n}  "
              f"argnums={argnums}  input_bytes={inp_b/MB:.2f} MB\n{'='*100}")

        if args.orders:
            variants = [(l, None, o) for l, o in build_order_variants(n)]
        else:
            variants = [(l, tf, rev) for l, tf in build_variants(n)]
        compiled = {}
        static = {}
        for label, tf, order in variants:
            try:
                t0 = time.perf_counter()
                jf = jacve(fn, order, argnums=argnums, transforms=tf)
                lowered = jax.jit(jf).lower(*xs)
                comp = lowered.compile()
                t_comp = time.perf_counter() - t0
                # warm run
                o = comp(*xs); jax.block_until_ready(o)
                ob = out_bytes(o); del o; gc.collect()
                ma = None
                t_ma0 = time.perf_counter()
                try:
                    m = comp.memory_analysis()
                    ma = {k: float(getattr(m, k, float("nan"))) for k in (
                        "temp_size_in_bytes", "argument_size_in_bytes",
                        "output_size_in_bytes", "alias_size_in_bytes",
                        "generated_code_size_in_bytes", "host_temp_size_in_bytes")}
                except Exception as e:
                    ma = {"err": str(e)[:120]}
                t_ma = time.perf_counter() - t_ma0
                compiled[label] = comp
                cens = hlo_dtype_census(comp)
                static[label] = dict(memory_analysis=ma, ma_cost_s=t_ma,
                                     compile_s=t_comp, out_bytes=ob,
                                     hlo=cens)
                print(f"  compiled {label:<20} out={ob/MB:9.2f} MB  "
                      f"compile={t_comp:6.2f}s  ma_cost={t_ma*1e3:7.2f}ms  "
                      f"hlo bf16={cens.get('bf16')} f16={cens.get('f16')} "
                      f"f8={cens.get('f8e4m3fn')} f32={cens.get('f32')}")
            except Exception as e:
                print(f"  FAILED   {label:<20} {type(e).__name__}: {str(e)[:200]}")
                static[label] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

        labels = [l for l in compiled]

        # ---- fidelity: is the approximation REAL? --------------------------
        # Cosine + relative Frobenius vs the exact executable. cos==1.0 and
        # rel==0.0 means the transform was silently masked away and the
        # "variant" is not a variant at all.
        fid = {}
        if "exact" in compiled:
            ref = compiled["exact"](*xs)
            jax.block_until_ready(ref)
            rf = jnp.concatenate([jnp.ravel(x).astype(jnp.float32)
                                  for x in jax.tree.leaves(ref)])
            rn = float(jnp.linalg.norm(rf))
            for l in labels:
                try:
                    o = compiled[l](*xs)
                    of = jnp.concatenate([jnp.ravel(x).astype(jnp.float32)
                                          for x in jax.tree.leaves(o)])
                    if of.shape != rf.shape:
                        fid[l] = {"shape_mismatch": [list(of.shape), list(rf.shape)]}
                    else:
                        cos = float(jnp.vdot(rf, of) /
                                    (jnp.linalg.norm(rf) * jnp.linalg.norm(of) + 1e-30))
                        rel = float(jnp.linalg.norm(of - rf) / (rn + 1e-30))
                        fid[l] = {"cos": cos, "rel_frob": rel}
                    del o, of
                except Exception as e:
                    fid[l] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            del ref, rf
            gc.collect()
        print("\n  fidelity vs exact (cos / rel_frob) -- 1.0/0.0 = transform did NOTHING:")
        for l in labels:
            print(f"    {l:<22} {fid.get(l)}")

        # ---- graphax symbolic mem (method 9) -------------------------------
        sym = {}
        from graphax.core import jacve as _jacve
        for label, tf, order in variants:
            if label not in compiled:
                continue
            try:
                t0 = time.perf_counter()
                r = _jacve(fn, order, argnums=argnums, count_ops=True,
                           transforms=tf)(*xs)
                jax.block_until_ready(r)
                aux = r[-1] if isinstance(r, (tuple, list)) else None
                d = {}
                if isinstance(aux, dict):
                    for k, v in aux.items():
                        try:
                            a = np.asarray(v)
                            if a.size == 1:
                                d[k] = float(a)
                        except Exception:
                            pass
                sym[label] = {"aux": d, "cost_s": time.perf_counter() - t0}
                del r
                gc.collect()
            except Exception as e:
                sym[label] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

        # ---- runtime rounds, rotated variant order -------------------------
        raw = {l: [] for l in labels}
        mon_raw = {l: [] for l in labels}
        order_log = []
        from jax_memory_monitor import ResourceMonitor
        shared_mon = ResourceMonitor(devices=devs)

        for rep in range(args.reps):
            k = rep % max(len(labels), 1)
            order = labels[k:] + labels[:k]
            if rep % 2 == 1:
                order = list(reversed(order))
            order_log.append(order)
            for label in order:
                rec = measure_once(dev, compiled[label], xs, False, args.profile)
                rec["rep"] = rep
                raw[label].append(rec)
                p = measure_monitor_reused(shared_mon, compiled[label], xs)
                mon_raw[label].append(p)

        # ---- fresh-monitor-per-call (leak check) ---------------------------
        mon_fresh = {}
        mon_fresh_cost = {}
        for label in labels:
            vals, costs = [], []
            for _ in range(min(args.reps, 5)):
                p, dt = measure_monitor(devs, compiled[label], xs)
                vals.append(p); costs.append(dt)
            mon_fresh[label] = vals
            mon_fresh_cost[label] = statistics.fmean(costs)

        # ---- contamination -------------------------------------------------
        print("\n  -- contaminating allocator (2 GB alloc/free + a failed big alloc) --")
        contam_note = []
        try:
            big = jnp.zeros((512, 1024, 1024), dtype=jnp.float32)  # 2 GB
            jax.block_until_ready(big)
            contam_note.append("alloc2GB ok")
            del big
        except Exception as e:
            contam_note.append(f"alloc2GB {type(e).__name__}")
        gc.collect()
        try:  # deliberate OOM
            huge = jnp.zeros((1024, 1024, 1024, 8), dtype=jnp.float32)  # 32 GB
            jax.block_until_ready(huge)
            del huge
            contam_note.append("OOM-attempt SUCCEEDED (no OOM)")
        except Exception as e:
            contam_note.append(f"OOM-attempt raised {type(e).__name__}")
        gc.collect()
        # keep a resident 1 GB block alive to shift the baseline
        resident = jnp.zeros((256, 1024, 1024), dtype=jnp.float32)
        jax.block_until_ready(resident)
        contam_note.append("1GB resident held")
        print("     " + "; ".join(contam_note))

        dirty = {l: [] for l in labels}
        dirty_mon = {l: [] for l in labels}
        for rep in range(3):
            k = rep % max(len(labels), 1)
            order = labels[k:] + labels[:k]
            for label in order:
                dirty[label].append(measure_once(dev, compiled[label], xs, False, False))
                dirty_mon[label].append(measure_monitor_reused(shared_mon, compiled[label], xs))
        del resident
        gc.collect()

        results["targets"][tkind] = {
            "n_vertices": n, "argnums": list(argnums), "input_bytes": inp_b,
            "static": static, "symbolic": sym,
            "clean": {l: raw[l] for l in labels},
            "clean_monitor_reused": mon_raw,
            "monitor_fresh": mon_fresh, "monitor_fresh_cost_s": mon_fresh_cost,
            "dirty": {l: dirty[l] for l in labels},
            "dirty_monitor_reused": dirty_mon,
            "order_log": order_log, "contam_note": contam_note,
        }

        # ---- report --------------------------------------------------------
        report(tkind, labels, static, sym, raw, mon_raw, mon_fresh,
               mon_fresh_cost, dirty, dirty_mon)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        print(f"\n# wrote {args.json}")


METHODS = [
    ("M1_abs_peak", "peak_bytes_in_use ABS"),
    ("M2_delta_peak", "peak - bytes_in_use(before)"),
    ("M2b_delta_peak_minus_out", "  ^ minus output bytes"),
    ("M3_steady_delta", "bytes_in_use after-before"),
    ("M4a_peak_reserved", "peak_bytes_reserved"),
    ("M4b_peak_pool", "peak_pool_bytes"),
    ("M4c_largest_alloc", "largest_alloc_size"),
    ("M4d_num_allocs_delta", "num_allocs delta"),
    ("M4e_pool_delta", "pool_bytes delta"),
    ("M7_live_delta", "live_arrays delta"),
    ("out_bytes", "[truth] output bytes"),
]


def report(tkind, labels, static, sym, raw, mon_raw, mon_fresh, mon_fresh_cost,
           dirty, dirty_mon):
    print(f"\n--- TARGET {tkind}: CLEAN, per method (mean MB / CV%) ---")
    hdr = f"{'method':<32}" + "".join(f"{l[:15]:>17}" for l in labels)
    print(hdr)
    print("-" * len(hdr))
    for key, name in METHODS:
        cells = ""
        vals = []
        for l in labels:
            a = agg([r[key] for r in raw[l]])
            vals.append(a["mean"])
            unit = a["mean"] if key == "M4d_num_allocs_delta" else a["mean"] / MB
            cells += f"{unit:>11.2f}/{a['cv']*100:>4.1f}"
        spread = (max(vals) - min(vals)) / abs(max(vals)) if max(vals) else 0.0
        print(f"{name:<32}{cells}   spread={spread*100:5.1f}%")

    # ResourceMonitor
    for nm, d in (("ResourceMonitor(reused)", mon_raw), ("ResourceMonitor(fresh)", mon_fresh)):
        cells = ""
        for l in labels:
            a = agg(d[l])
            cells += f"{a['mean']/MB:>11.2f}/{a['cv']*100:>4.1f}"
        print(f"{nm:<32}{cells}")

    # ResourceMonitor drift over reps (leak check)
    print(f"\n  ResourceMonitor(reused) per-rep series (MB), leak check:")
    for l in labels:
        print(f"    {l:<20} " + " ".join(f"{v/MB:8.2f}" for v in mon_raw[l]))
    print(f"  ResourceMonitor(fresh) per-rep series (MB):")
    for l in labels:
        print(f"    {l:<20} " + " ".join(f"{v/MB:8.2f}" for v in mon_fresh[l]))

    print(f"\n--- TARGET {tkind}: STATIC XLA memory_analysis (MB, deterministic) ---")
    keys = ["temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes",
            "alias_size_in_bytes", "generated_code_size_in_bytes"]
    print(f"{'variant':<20}" + "".join(f"{k.replace('_size_in_bytes',''):>14}" for k in keys)
          + f"{'temp+out+arg':>15}{'temp+out':>12}{'ma_ms':>9}")
    for l in labels:
        ma = static[l]["memory_analysis"]
        row = "".join(f"{ma.get(k, float('nan'))/MB:>14.2f}" for k in keys)
        tot = (ma.get("temp_size_in_bytes", 0) + ma.get("output_size_in_bytes", 0)
               + ma.get("argument_size_in_bytes", 0))
        to = ma.get("temp_size_in_bytes", 0) + ma.get("output_size_in_bytes", 0)
        print(f"{l:<20}{row}{tot/MB:>15.2f}{to/MB:>12.2f}"
              f"{static[l]['ma_cost_s']*1e3:>9.2f}")

    print(f"\n--- TARGET {tkind}: graphax symbolic aux (count_ops) ---")
    for l in labels:
        s = sym.get(l, {})
        if "aux" in s:
            print(f"  {l:<20} {s['aux']}  cost={s['cost_s']:.2f}s")
        else:
            print(f"  {l:<20} {s}")

    print(f"\n--- TARGET {tkind}: DIRTY allocator (1 GB resident held) ---")
    hdr = f"{'method':<32}" + "".join(f"{l[:15]:>17}" for l in labels)
    print(hdr)
    for key, name in METHODS:
        cells = ""
        for l in labels:
            a = agg([r[key] for r in dirty[l]])
            unit = a["mean"] if key == "M4d_num_allocs_delta" else a["mean"] / MB
            cells += f"{unit:>11.2f}/{a['cv']*100:>4.1f}"
        print(f"{name:<32}{cells}")
    cells = ""
    for l in labels:
        a = agg(dirty_mon[l])
        cells += f"{a['mean']/MB:>11.2f}/{a['cv']*100:>4.1f}"
    print(f"{'ResourceMonitor(reused)':<32}{cells}")

    print(f"\n--- TARGET {tkind}: COST (mean seconds) ---")
    for l in labels[:1]:
        a_exec = agg([r["exec_s"] for r in raw[l]])
        a_st = agg([r["probe_stats_s"] for r in raw[l]])
        a_lv = agg([r["probe_live_s"] for r in raw[l]])
        print(f"  exec={a_exec['mean']*1e3:.3f} ms | memory_stats(2 reads, 10 keys)="
              f"{a_st['mean']*1e6:.1f} us | live_arrays(2 reads)="
              f"{a_lv['mean']*1e6:.1f} us | memory_analysis="
              f"{static[l]['ma_cost_s']*1e3:.2f} ms | "
              f"ResourceMonitor(fresh, incl. exec)={mon_fresh_cost[l]*1e3:.2f} ms")
    print(f"\n--- TARGET {tkind}: HLO dtype census (proof transform reached XLA) ---")
    for l in labels:
        print(f"  {l:<22} {static[l].get('hlo')}")
    if any("M8_profile_after" in r for r in raw[labels[0]]):
        a = agg([r.get("M8_cost_s", float('nan')) for r in raw[labels[0]]])
        print(f"  device_memory_profile cost={a['mean']*1e3:.2f} ms")


if __name__ == "__main__":
    main()
