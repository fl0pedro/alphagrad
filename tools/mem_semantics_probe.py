#!/usr/bin/env python3
"""What does our peak_memory channel actually measure?

Three numbers have been reported for "the same" thing and they disagree by 4
orders of magnitude:
    50 KB    CPU smoke        (ResourceMonitor, JAX_PLATFORMS=cpu)
   132 MB    v18 GPU campaign (ResourceMonitor, Jacobian graph)
   280 MB    GPU probe        (memory_stats peak_bytes_in_use, seeded graph)

Two candidate explanations, and they have opposite consequences for the reward:

  (1) BACKEND — CPU has no device allocator, so ResourceMonitor reports
      something meaningless there. Then the 50 KB is simply not a memory
      measurement and CPU runs have a dead memory channel.

  (2) ABSOLUTE vs DELTA — `peak_bytes_in_use` is a device-wide HIGH-WATER MARK.
      `clear_memory_stats()` resets the peak but NOT the resident baseline, so
      peak = (already-resident arrays) + (this call's transient). If the
      baseline dominates, the variable part is a small fraction of the number
      we train on — which would explain why peak memory looked CONSTANT
      (280.8 / 281.6 / 282.8 / 283.7 MB) across an exact gradient, a destroyed
      one, full bfloat16 and structural approximation.

This reports bytes_in_use BEFORE the call, the peak AFTER, and the DELTA, so
the two can be told apart.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def stats(dev, key):
    try:
        return float(dev.memory_stats().get(key, 0.0))
    except Exception:
        return float("nan")


def main():
    from graphax.core import jacve
    from graphax.sparse.micro_actions import Quant
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup
    from alphagrad.approx.common.masks import make_live_masked_hook

    dev = jax.devices()[0]
    print(f"device={dev} platform={jax.default_backend()}")
    print(f"memory_stats keys: {sorted(dev.memory_stats().keys())[:12]}")

    class _A:
        measure_grad = True
        seed_vertices = True

    ex = "VmappedNeuralNetwork"
    fn, xs, argnums = grad_target_setup(
        _A(), get_fn(ex), get_args(ex, jax.random.PRNGKey(0), dataset="mnist"), ex)
    cj = jax.make_jaxpr(fn)(*xs)
    n = len(cj.jaxpr.eqns)
    rev = list(range(n, 0, -1))

    inp_bytes = sum(np.asarray(a).nbytes for a in xs)
    print(f"vertices={n}  input args total = {inp_bytes/2**20:.2f} MB")

    variants = [
        ("exact", None),
        ("bfloat16 everywhere",
         [(v, (make_live_masked_hook((Quant(dtype='bfloat16'),)),))
          for v in range(1, n + 1)]),
    ]

    print(f"\n{'variant':<24}{'resident before':>18}{'peak after':>14}"
          f"{'DELTA':>14}{'out bytes':>12}")
    print("-" * 84)
    for label, tf in variants:
        f = jax.jit(jacve(fn, rev, argnums=argnums, transforms=tf) if tf
                    else jacve(fn, rev, argnums=argnums))
        r = f(*xs)
        jax.block_until_ready(r)          # compile + warm

        dev.clear_memory_stats()
        jax.effects_barrier()
        before = stats(dev, "bytes_in_use")
        out = f(*xs)
        jax.block_until_ready(out)
        peak = stats(dev, "peak_bytes_in_use")
        outb = float(sum(x.size * x.dtype.itemsize
                         for x in jax.tree.leaves(out)))
        print(f"{label:<24}{before/2**20:>15.1f} MB{peak/2**20:>11.1f} MB"
              f"{(peak-before)/2**20:>11.1f} MB{outb/2**20:>9.2f} MB")
        del out

    print("\nIf DELTA is small and similar across variants while PEAK is large,")
    print("the channel is dominated by a constant resident baseline and the")
    print("approximation-dependent signal is buried in it.")


if __name__ == "__main__":
    main()
