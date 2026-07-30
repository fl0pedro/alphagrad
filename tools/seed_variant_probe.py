#!/usr/bin/env python3
"""Seeded (N+2) grad target: sparsity + measured cost of concrete approximations.

Part 1 — SPARSITY under random orders x random approximations, re-run against
         the TIGHTENED seed (13 -> 15, no scaffolding). The earlier numbers were
         polluted by 21 scaffolding vertices incl. identity ops on the two
         largest tensors, so they are stale.

Part 2 — REVERSE order, four approximation settings, each measured with the
         spec protocol (4 random points x 5 repeats = 20, winsorized):
           A  rev, NO approximation            (the exact reference)
           B  rev, random approximations
           C  rev, bfloat16 QUANT on every vertex
           D  rev, exactly one Diag + one Compress
         Reported: latency, peak memory, and FROB residual vs A
           frob = ||g_approx - g_exact||_F / ||g_exact||_F
"""
from __future__ import annotations

import random
import time

import jax
import jax.numpy as jnp
import numpy as np

N_POINTS, N_REPS = 4, 5
import os as _osenv
INNER = int(_osenv.environ.get('PROBE_INNER_REPS', '1'))
SEED = 0


def _winsor(xs, frac=0.2):
    a = np.sort(np.asarray(xs, dtype=np.float64))
    k = int(len(a) * frac / 2)
    if k:
        a = a[k:len(a) - k]
    return float(a.mean())


def _peak(dev):
    try:
        return float(dev.memory_stats().get("peak_bytes_in_use", 0.0))
    except Exception:
        return 0.0


def _reset(dev):
    try:
        dev.clear_memory_stats()
    except Exception:
        pass


def _flat(tree):
    return jnp.concatenate([jnp.ravel(l) for l in jax.tree.leaves(tree)])


def _frob(g, ref):
    a, b = _flat(g).astype(jnp.float64), _flat(ref).astype(jnp.float64)
    d = float(jnp.linalg.norm(a - b))
    n = float(jnp.linalg.norm(b))
    return d / n if n > 0 else float("nan")


def _cos(g, ref):
    a, b = _flat(g).astype(jnp.float64), _flat(ref).astype(jnp.float64)
    na, nb = float(jnp.linalg.norm(a)), float(jnp.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(jnp.dot(a, b) / (na * nb))


def build():
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup

    class _A:
        measure_grad = True
        seed_vertices = True

    ex = "VmappedNeuralNetwork"
    fn, xs, argnums = grad_target_setup(
        _A(), get_fn(ex), get_args(ex, jax.random.PRNGKey(0), dataset="mnist"), ex)
    cj = jax.make_jaxpr(fn)(*xs)
    pts = [
        grad_target_setup(_A(), get_fn(ex),
                          get_args(ex, jax.random.PRNGKey(100 + p),
                                   dataset="mnist"), ex)[1]
        for p in range(N_POINTS)
    ]
    return fn, xs, argnums, cj, pts


def _inuse(dev):
    try:
        return float(dev.memory_stats().get("bytes_in_use", 0.0))
    except Exception:
        return 0.0


def measure(fn_c, pts, dev, label):
    # WARM UP FIRST: this is where XLA autotuning happens. Autotuning trials
    # allocate aggressively (observed: repeated 150 GB attempts) and a failed
    # trial leaves the allocator holding blocks. Doing it before the timed
    # loop keeps that cost and that churn out of the measurements.
    out = fn_c(*pts[0])
    jax.block_until_ready(out)

    lat, dmem = [], []
    for p in range(N_POINTS):
        for _ in range(N_REPS):
            _reset(dev)
            jax.effects_barrier()
            base = _inuse(dev)              # resident BEFORE this call
            # INNER REPS: at this problem size a single call is dominated by
            # dispatch/launch overhead (~600us wall for a 1.6 MB, 16x784
            # gradient — real compute is tens of us). One call per timing made
            # the RANKING of four variants irreproducible across runs while the
            # exact baseline stayed stable within 3%. Looping inside the timer
            # amortises the fixed overhead so the graph's cost is what varies.
            t0 = time.perf_counter()
            for _i in range(INNER):
                r = fn_c(*pts[p])
            jax.block_until_ready(r)
            lat.append((time.perf_counter() - t0) * 1e9 / INNER)
            # DELTA, not the device-wide high-water mark: peak_bytes_in_use is
            # absolute and clear_memory_stats() resets the COUNTER, not the
            # resident baseline. Reporting the absolute made four very
            # different approximations look identical at ~281 MB because a
            # dirty 278 MB baseline dominated them.
            dmem.append(max(_peak(dev) - base, 0.0))
    return {"label": label, "lat": _winsor(lat), "peak": _winsor(dmem),
            "out": out}


def main():
    from graphax.core import jacve, vertex_elimination_jaxpr
    from graphax.sparse.micro_actions import Compress, Diag, Quant
    from alphagrad.approx.common.masks import make_live_masked_hook

    dev = jax.devices()[0]
    fn, xs, argnums, cj, pts = build()
    n = len(cj.jaxpr.eqns)
    rev = list(range(n, 0, -1))
    print(f"device={dev} vertices={n} argnums={argnums}  (seeded N+2)"
          f"  inner_reps={INNER}")

    # ---------------- Part 1: random orders x random approximations --------
    def rand_rules(rng):
        r = rng.random()
        if r < 0.25:
            return ()
        if r < 0.50:
            return (Diag(i=0, j=1, factor=rng.choice([2, 4])),)
        if r < 0.75:
            return (Compress(axes=(0,), kind="mean"),)
        return (Quant(dtype="bfloat16"),)

    import os as _os
    if _os.environ.get('PROBE_SKIP_PART1', '0') == '1':
        print('\n--- Part 1 SKIPPED (PROBE_SKIP_PART1=1) ---')
        mios = []
    else:
      print("\n--- Part 1: random order x random approx (max_io) ---")
      rng = random.Random(SEED)
      mios = []
      for k in range(6):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        tf = [(int(v), (make_live_masked_hook(rand_rules(rng)),))
              for v in order if rand_rules(random.Random(k * 100 + v))]
        try:
            _, aux = vertex_elimination_jaxpr(
                cj.jaxpr, order, cj.literals, *xs, argnums=argnums,
                count_ops=True, sparse_representation=True, transforms=tf)
            mio = float(aux["mem"])
            mios.append(mio)
            print(f"  plan {k}: ops={float(aux['adds']+aux['muls']+aux['fmas']):.3e}"
                  f"  max_io={mio:.3e}")
        except Exception as e:
            print(f"  plan {k}: FAILED {type(e).__name__}: {str(e)[:90]}")
    if mios:
        print(f"  --> max_io median={np.median(mios):.3e} min={min(mios):.3e}"
              f" max={max(mios):.3e}")

    # ---------------- Part 2: reverse order, four settings -----------------
    every = list(range(1, n + 1))
    variants = [
        ("A rev  exact (no approx)", []),
        ("B rev  random approx",
         [(v, (make_live_masked_hook(rand_rules(random.Random(v))),))
          for v in every]),
        ("C rev  bfloat16 everywhere",
         [(v, (make_live_masked_hook((Quant(dtype="bfloat16"),)),))
          for v in every]),
        ("D rev  one Diag + one Compress",
         [(every[len(every) // 3], (make_live_masked_hook((Diag(i=0, j=1, factor=2),)),)),
          (every[2 * len(every) // 3], (make_live_masked_hook((Compress(axes=(0,), kind="mean"),)),))]),
    ]

    print("\n--- Part 2: REVERSE order, 4x5=20 measurements each ---")
    rows, ref = [], None
    for label, tf in variants:
        try:
            f = jax.jit(jacve(fn, rev, argnums=argnums,
                              transforms=tf) if tf else
                        jacve(fn, rev, argnums=argnums))
            r = measure(f, pts, dev, label)
            if ref is None:
                ref = r["out"]
                r["frob"], r["cos"] = 0.0, 1.0
            else:
                r["frob"], r["cos"] = _frob(r["out"], ref), _cos(r["out"], ref)
            rows.append(r)
        except Exception as e:
            print(f"  {label}: FAILED {type(e).__name__}: {str(e)[:140]}")

    print("\n" + "=" * 92)
    print(f"{'variant':<34}{'latency (us)':>14}{'peak (MB)':>12}"
          f"{'frob':>12}{'cos':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['label']:<34}{r['lat']/1e3:>14.1f}{r['peak']/2**20:>12.1f}"
              f"{r['frob']:>12.4f}{r['cos']:>10.4f}")
    print("=" * 92)
    if len(rows) > 1:
        a = rows[0]
        for r in rows[1:]:
            print(f"  {r['label'][:28]:<30} speedup={a['lat']/max(r['lat'],1):.2f}x"
                  f"  mem={r['peak']/max(a['peak'],1):.2f}x  frob={r['frob']:.4f}")


if __name__ == "__main__":
    main()
