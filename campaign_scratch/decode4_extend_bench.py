"""decode4_extend_bench: is `Agent.encode_extend` PREFIX-PROPORTIONAL?

Part 1(d).  Raising ALPHAGRAD_MAX_DELTA_TOKENS 1024 -> 32768 costs BUFFER
MEMORY for free only if the scan's trip count follows `count`, not `window`.
This times the exact call the live path makes -- `window=MAX_DELTA_TOKENS`,
`start=0` -- at both budgets, for the rollout form (no `budget`) and the
reverse-differentiated loss form (`budget=`), and with the chunked trip count
both OFF (the shipped default `ALPHAGRAD_EXTEND_CHUNK=0`) and ON.

Reports median-of-N wall per call and the device peak.  A flat scan shows
time ~ window; a prefix-proportional one shows time ~ count.
"""
import os, sys, time, json
os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "0"      # live path = _extend_sequential
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand
import decode3_arch as ARCH

E, NV = 32, 64
REPS = int(os.environ.get("BENCH_REPS", "7"))
agent = ARCH.build(E, 3, 2, 2, NV, jrand.PRNGKey(0))
print("device", jax.local_devices()[0], flush=True)

# Measured TLM delta lengths: median-ish, nn256 worst, TLM worst.
COUNTS = [78, 5664, 25737]
WINDOWS = [1024, 32768]
CHUNKS = [0, 256]


def _peak():
    try:
        return jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)
    except Exception:
        return 0


def timeit(f, *a):
    o = f(*a); jax.block_until_ready(o)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


rows = []
for W in WINDOWS:
    rng = np.random.default_rng(0)
    toks = jnp.asarray(rng.integers(1, 512, size=(W,)), jnp.int32)
    eqns = jnp.asarray(rng.integers(-1, 64, size=(W,)), jnp.int32)
    c0 = agent.carry_init()
    for C in CHUNKS:
        for cnt in COUNTS:
            if cnt > W:
                continue
            n = jnp.asarray(cnt, jnp.int32)

            # ROLLOUT form: forward only, dynamic trip count via while_loop.
            fwd = jax.jit(lambda c, t, e, k: agent.encode_extend(
                c, t, e, k, window=W, start=0, chunk=C)[1].sum())
            # LOSS form: reverse-differentiated, trip count via scan+cond.
            def _l(p, c, t, e, k):
                ag = __import__("equinox").combine(p, st)
                return ag.encode_extend(c, t, e, k, window=W, start=0,
                                        chunk=C, budget=cnt)[1].sum()
            import equinox as eqx
            p, st = eqx.partition(agent, eqx.is_inexact_array)
            bwd = jax.jit(jax.grad(_l))

            t_f = timeit(fwd, c0, toks, eqns, n)
            pk0 = _peak()
            t_b = timeit(bwd, p, c0, toks, eqns, n)
            r = dict(window=W, chunk=C, count=cnt, fwd_ms=t_f, bwd_ms=t_b,
                     peak_mb=_peak() / 2**20)
            rows.append(r)
            print(f"window {W:6d} chunk {C:4d} count {cnt:6d}  "
                  f"fwd {t_f:9.2f} ms   bwd(grad) {t_b:9.2f} ms   "
                  f"peak {r['peak_mb']:8.1f} MB", flush=True)

print("\n=== PREFIX-PROPORTIONAL? (fwd ms, rollout form) ===", flush=True)
for C in CHUNKS:
    for cnt in COUNTS:
        a = [r for r in rows if r["chunk"] == C and r["count"] == cnt]
        if len(a) == 2:
            lo = [r for r in a if r["window"] == 1024][0]
            hi = [r for r in a if r["window"] == 32768][0]
            print(f"  chunk={C:4d} count={cnt:6d}: 1024 -> 32768 costs "
                  f"{hi['fwd_ms']/max(lo['fwd_ms'],1e-9):6.2f}x fwd, "
                  f"{hi['bwd_ms']/max(lo['bwd_ms'],1e-9):6.2f}x bwd", flush=True)
json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1
                     else "decode4_extend_bench.json", "w"))
