#!/usr/bin/env python3
"""Does wrapping a JACOBIAN-producing fn in a GRADIENT-producing fn let XLA
fuse away the Jacobian materialisation?

HYPOTHESIS (user, 2026-07-28): keep the ACTION SPACE on the Jacobian graph,
but measure a function that ends in the gradient contraction, so XLA fuses the
reduction into the elimination and we time something realistic instead of a
materialised Jacobian. If true, the action space is unchanged and only the
reward landscape moves.

Compares, on identical inputs and the same measurement protocol:

  A  jit( mean_over_outputs( jacrev(f)(x) ) )   "Jacobian fn wrapped in a grad fn"
  B  jit( grad( mean(f) )(x) )                  the direct gradient
  C  jit( jacrev(f)(x) )                        the raw Jacobian (what we
                                                 have actually been timing)

Protocol matches the spec: 4 random points x 5 seeds = 20 measurements per
variant, winsorized mean; latency via time.perf_counter around a
block_until_ready, peak memory via device memory_stats()['peak_bytes_in_use']
with clear_memory_stats() before each.

If A ~= B, the hypothesis holds: measure the wrapped form and keep the
Jacobian action space. If A ~= C, XLA did NOT fuse and the Jacobian is really
being built.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

N_POINTS = 4
N_SEEDS = 5


def _winsor_mean(xs, frac=0.2):
    a = np.sort(np.asarray(xs, dtype=np.float64))
    k = int(len(a) * frac / 2)
    if k:
        a = a[k:len(a) - k]
    return float(a.mean())


def _peak_bytes(dev):
    try:
        return float(dev.memory_stats().get("peak_bytes_in_use", 0.0))
    except Exception:
        return 0.0


def _reset_peak(dev):
    try:
        dev.clear_memory_stats()
    except Exception:
        pass


def measure(fn, args_list, label, dev):
    """20 timings: N_POINTS distinct inputs x N_SEEDS repeats."""
    out = fn(*args_list[0])
    jax.block_until_ready(out)          # compile once, excluded from timing

    lats, peaks = [], []
    for p in range(N_POINTS):
        a = args_list[p % len(args_list)]
        for _ in range(N_SEEDS):
            _reset_peak(dev)
            jax.effects_barrier()
            t0 = time.perf_counter()
            r = fn(*a)
            jax.block_until_ready(r)
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1e9)
            peaks.append(_peak_bytes(dev))
    return {
        "label": label,
        "latency_ns": _winsor_mean(lats),
        "peak_bytes": _winsor_mean(peaks),
        "out_bytes": float(sum(x.size * x.dtype.itemsize
                               for x in jax.tree.leaves(out))),
    }


def main():
    dev = jax.devices()[0]
    print(f"device: {dev}  platform={jax.default_backend()}")

    from alphagrad.approx.common import get_args, get_fn, infer_argnums

    example = "VmappedNeuralNetwork"
    f = get_fn(example)
    argnums = infer_argnums(example)
    print(f"example={example} argnums={argnums}")

    base = get_args(example, jax.random.PRNGKey(0), dataset="mnist")
    shapes = [getattr(a, "shape", None) for a in base]
    print(f"arg shapes: {shapes}")

    # N distinct random points (fresh parameters, as the spec prescribes).
    args_list = [
        get_args(example, jax.random.PRNGKey(1234 + p), dataset="mnist")
        for p in range(N_POINTS)
    ]

    out0 = f(*base)
    n_out = np.ndim(out0)
    print(f"f output shape: {getattr(out0, 'shape', None)} (ndim={n_out})")

    def loss(*a):
        return jnp.mean(f(*a))

    # A — the Jacobian-producing fn wrapped in a gradient-producing fn.
    #     d mean(f) / dx == mean over ALL output axes of the Jacobian, so the
    #     wrapper is exactly the contraction that turns J into the gradient.
    def grad_via_jac(*a):
        J = jax.jacrev(f, argnums=argnums)(*a)
        scale = 1.0 / float(np.prod(np.shape(out0)))
        return jax.tree.map(
            lambda j: j.reshape((-1,) + j.shape[n_out:]).sum(0) * scale, J)

    # B — the direct gradient.
    grad_direct = jax.grad(loss, argnums=argnums)

    # C — the raw Jacobian (what every run so far has actually timed).
    def jac_raw(*a):
        return jax.jacrev(f, argnums=argnums)(*a)

    variants = [
        ("A  grad_via_jac (wrapped)", jax.jit(grad_via_jac)),
        ("B  grad_direct  (jax.grad)", jax.jit(grad_direct)),
        ("C  jac_raw      (Jacobian)", jax.jit(jac_raw)),
    ]

    # Correctness first: A and B must agree, or the comparison is meaningless.
    ga = jax.jit(grad_via_jac)(*base)
    gb = jax.jit(grad_direct)(*base)
    la, lb = jax.tree.leaves(ga), jax.tree.leaves(gb)
    maxdiff = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(la, lb))
    rel = maxdiff / max(float(jnp.max(jnp.abs(jnp.stack(
        [jnp.max(jnp.abs(x)) for x in lb])))), 1e-30)
    print(f"\nA vs B agreement: max|diff|={maxdiff:.3e}  rel={rel:.3e}")
    if rel > 1e-4:
        print("  WARNING: A and B disagree — the wrapper is not the gradient!")

    rows = []
    for label, fn in variants:
        try:
            rows.append(measure(fn, args_list, label, dev))
        except Exception as exc:
            print(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:200]}")

    print("\n" + "=" * 88)
    print(f"{'variant':<30}{'latency (us)':>16}{'peak (MB)':>14}"
          f"{'output (MB)':>14}")
    print("-" * 88)
    for r in rows:
        print(f"{r['label']:<30}{r['latency_ns']/1e3:>16.1f}"
              f"{r['peak_bytes']/2**20:>14.1f}{r['out_bytes']/2**20:>14.2f}")
    print("=" * 88)

    if len(rows) == 3:
        a, b, c = rows
        print(f"\nA/B latency ratio  = {a['latency_ns']/max(b['latency_ns'],1):.2f}x"
              f"   (1.0 => XLA fused the contraction: hypothesis HOLDS)")
        print(f"A/C latency ratio  = {a['latency_ns']/max(c['latency_ns'],1):.2f}x"
              f"   (1.0 => no fusion, still building the Jacobian)")
        print(f"A/B peak ratio     = {a['peak_bytes']/max(b['peak_bytes'],1):.2f}x")
        print(f"A/C peak ratio     = {a['peak_bytes']/max(c['peak_bytes'],1):.2f}x")


if __name__ == "__main__":
    main()
