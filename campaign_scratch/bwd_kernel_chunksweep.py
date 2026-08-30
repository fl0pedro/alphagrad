"""``PalimpsaMixer.chunk_size`` sweep: backward time and peak memory.

The kernel's ``custom_vjp`` saves ``(state_mu, state_I)`` every ``chunk_size``
tokens and recomputes the intra-chunk state in the backward, so chunk_size is
a pure memory/recompute dial with NO effect on the numerics (its own
docstring). Nothing in alphagrad has ever set it; this measures what it would
buy at the encoder's real shapes.

Reported per (T, chunk_size): forward ms, value-and-grad ms, and the
``peak_bytes_in_use`` delta over one value_and_grad execution.
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.transformer.palimpsa_pallas import palimpsa

H = int(os.environ.get("BWD_H", "4"))
D = int(os.environ.get("BWD_D", "32"))
TS = [int(x) for x in os.environ.get("BWD_TS", "4096,32768").split(",")]
CS = [int(x) for x in os.environ.get("BWD_CS", "8,16,32,64,128").split(",")]
REPS = int(os.environ.get("BWD_REPS", "5"))


def _mk(T, seed=0):
    r = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(r.normal(0, 0.5, s).astype(np.float32))
    return dict(q=f(1, T, H, D), k=f(1, T, H, D), v=f(1, T, H, D),
                b=f(1, T, H, D), gt=jnp.asarray(
                    -np.abs(r.normal(0, 0.1, (1, T, H))).astype(np.float32)),
                g=jnp.full((H,), 0.05, jnp.float32),
                Ip=jnp.full((H,), 0.69, jnp.float32))


def _peak_delta(fn, *a):
    dev = jax.devices()[0]
    try:
        dev.clear_memory_stats()
    except Exception:
        pass
    base = float((dev.memory_stats() or {}).get("peak_bytes_in_use", 0.0))
    out = jax.block_until_ready(fn(*a))
    peak = float((dev.memory_stats() or {}).get("peak_bytes_in_use", 0.0))
    return out, max(0.0, peak - base)


def main():
    print(f"[chunksweep] device={jax.devices()[0]} H={H} D={D}", flush=True)
    ref = {}
    for T in TS:
        arrs = _mk(T)
        for cs in CS:
            if cs > T:
                continue

            def loss(q, k, v, b, gt, g, Ip, _cs=cs):
                out = palimpsa(q, k, v, b, gt, g, Ip, scale=None,
                               chunk_size=_cs)
                return jnp.sum(out * out)

            fwd = jax.jit(lambda *a, _cs=cs: palimpsa(
                *a, scale=None, chunk_size=_cs))
            vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4)))
            args = (arrs["q"], arrs["k"], arrs["v"], arrs["b"], arrs["gt"],
                    arrs["g"], arrs["Ip"])
            try:
                o = jax.block_until_ready(fwd(*args))
                (val, grads), peak = _peak_delta(vg, *args)
            except Exception as e:  # pragma: no cover
                print(f"[chunksweep] T={T} cs={cs} FAILED: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            t0 = time.perf_counter()
            for _ in range(REPS):
                jax.block_until_ready(fwd(*args))
            t_f = (time.perf_counter() - t0) / REPS * 1e3
            t0 = time.perf_counter()
            for _ in range(REPS):
                jax.block_until_ready(vg(*args))
            t_b = (time.perf_counter() - t0) / REPS * 1e3
            gsum = float(sum(jnp.sum(jnp.abs(x)) for x in grads))
            key = T
            tag = ""
            if key not in ref:
                ref[key] = (float(val), gsum, np.asarray(o))
            else:
                v0, g0, o0 = ref[key]
                same_o = np.array_equal(np.asarray(o), o0)
                tag = (f"  numerics: out {'BIT-IDENTICAL' if same_o else 'DIFFER'}"
                       f" dval={abs(float(val) - v0):.3e}"
                       f" dgrad={abs(gsum - g0):.3e}")
            print(f"[chunksweep] T={T:6d} chunk={cs:4d} "
                  f"fwd={t_f:8.3f} ms  val+grad={t_b:9.3f} ms  "
                  f"peak={peak / 2**20:9.1f} MiB{tag}", flush=True)


if __name__ == "__main__":
    main()
