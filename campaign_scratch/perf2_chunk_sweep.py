"""perf2_chunk_sweep: tune ALPHAGRAD_EXTEND_CHUNK / ALPHAGRAD_LOSS_EXTEND_CHUNK
at the shipped delta budget (window = ALPHAGRAD_MAX_DELTA_TOKENS = 32768).

The chunk is a SEQUENCE-dimension blocking of `_extend_sequential`:
  * ROLLOUT form (no `budget`)  -> `while_loop` of ceil(count/C) C-long scans.
    Smaller C = strictly prefix-proportional, but more while trips.
  * LOSS form (`budget=`)       -> `scan` of nb_max = ceil(W/C) remat'd chunks
    with a `cond` predicate. nb_max does NOT depend on count, so smaller C
    costs MORE outer iterations for nothing. Its optimum is a LARGER C.

Sweeps C across a realistic delta-length distribution and reports median wall
(fwd, bwd) and device peak for each.

usage: perf2_chunk_sweep.py OUT.json
"""
import os, sys, time, json

os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "0"       # live path = sequential
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_EXTEND_UNROLL", "32")

import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand
import equinox as eqx
import decode3_arch as ARCH

OUT = sys.argv[1] if len(sys.argv) > 1 else "perf2_chunk_sweep.json"
E, NV = 32, 64
REPS = int(os.environ.get("BENCH_REPS", "5"))
W = int(os.environ.get("BENCH_WINDOW", "32768"))

# Realistic TLM delta-length distribution: measured median-ish 78, the bulk of
# a rollout's steps, and the worst single delta (25737) that sized the budget.
COUNTS = [int(x) for x in os.environ.get(
    "BENCH_COUNTS", "78,512,2048,5664,12000,25737").split(",")]
CHUNKS = [int(x) for x in os.environ.get(
    "BENCH_CHUNKS", "0,32,128,256,512,1024,2048,4096,8192,16384").split(",")]

agent = ARCH.build(E, 3, 2, 2, NV, jrand.PRNGKey(0))
p, st = eqx.partition(agent, eqx.is_inexact_array)
print("device", jax.local_devices()[0], "window", W, flush=True)


def _peak():
    try:
        d = jax.local_devices()[0]
        d.memory_stats()  # refresh
        return d.memory_stats().get("peak_bytes_in_use", 0)
    except Exception:
        return 0


def _reset_peak():
    try:
        jax.local_devices()[0].memory_stats()
    except Exception:
        pass


def timeit(f, *a):
    o = f(*a); jax.block_until_ready(o)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


rng = np.random.default_rng(0)
toks = jnp.asarray(rng.integers(1, 512, size=(W,)), jnp.int32)
eqns = jnp.asarray(rng.integers(-1, 64, size=(W,)), jnp.int32)
c0 = agent.carry_init()

rows = []
for C in CHUNKS:
    fwd = jax.jit(lambda c, t, e, k: agent.encode_extend(
        c, t, e, k, window=W, start=0, chunk=C)[1].sum())
    for cnt in COUNTS:
        n = jnp.asarray(cnt, jnp.int32)

        def _l(pp, c, t, e, k, _cnt=cnt, _C=C):
            ag = eqx.combine(pp, st)
            return ag.encode_extend(c, t, e, k, window=W, start=0,
                                    chunk=_C, budget=_cnt)[1].sum()
        bwd = jax.jit(jax.grad(_l))

        _reset_peak()
        t_f = timeit(fwd, c0, toks, eqns, n)
        pk_f = _peak()
        _reset_peak()
        t_b = timeit(bwd, p, c0, toks, eqns, n)
        pk_b = _peak()
        nb_fwd = 1 if C <= 0 or C >= W else -(-cnt // C)
        nb_bwd = 1 if C <= 0 or C >= W else -(-W // C)
        r = dict(window=W, chunk=C, count=cnt, fwd_ms=t_f, bwd_ms=t_b,
                 fwd_peak_mb=pk_f / 2**20, bwd_peak_mb=pk_b / 2**20,
                 fwd_trips=nb_fwd, bwd_trips=nb_bwd)
        rows.append(r)
        print(f"chunk {C:6d} count {cnt:6d}  fwd {t_f:9.2f} ms "
              f"(trips {nb_fwd:5d})  bwd {t_b:9.2f} ms (trips {nb_bwd:4d})  "
              f"peak fwd {r['fwd_peak_mb']:7.1f} / bwd {r['bwd_peak_mb']:7.1f} MB",
              flush=True)

json.dump(rows, open(OUT, "w"))

print("\n=== FORWARD (rollout while_loop) ms by chunk x count ===", flush=True)
hdr = "chunk  " + "".join(f"{c:>10d}" for c in COUNTS) + "     SUM"
print(hdr, flush=True)
for C in CHUNKS:
    a = [r for r in rows if r["chunk"] == C]
    d = {r["count"]: r["fwd_ms"] for r in a}
    s = sum(d.values())
    print(f"{C:6d}" + "".join(f"{d.get(c, float('nan')):10.2f}" for c in COUNTS)
          + f"{s:9.1f}", flush=True)

print("\n=== BACKWARD (loss scan+cond+remat) ms by chunk x count ===", flush=True)
print(hdr, flush=True)
for C in CHUNKS:
    a = [r for r in rows if r["chunk"] == C]
    d = {r["count"]: r["bwd_ms"] for r in a}
    s = sum(d.values())
    print(f"{C:6d}" + "".join(f"{d.get(c, float('nan')):10.2f}" for c in COUNTS)
          + f"{s:9.1f}", flush=True)

print("\n=== PEAK MEM (MB) fwd / bwd by chunk (max over counts) ===", flush=True)
for C in CHUNKS:
    a = [r for r in rows if r["chunk"] == C]
    print(f"{C:6d}  fwd {max(r['fwd_peak_mb'] for r in a):8.1f}   "
          f"bwd {max(r['bwd_peak_mb'] for r in a):8.1f}", flush=True)

# A rollout is dominated by SHORT deltas: weight the sum by the measured
# frequency instead of treating 25737 as if it happened every step.
WGT = {78: 0.70, 512: 0.12, 2048: 0.08, 5664: 0.05, 12000: 0.03, 25737: 0.02}
print("\n=== WEIGHTED (0.70/0.12/0.08/0.05/0.03/0.02) ms per call ===",
      flush=True)
for C in CHUNKS:
    a = {r["count"]: r for r in rows if r["chunk"] == C}
    wf = sum(WGT.get(c, 0) * a[c]["fwd_ms"] for c in COUNTS if c in a)
    wb = sum(WGT.get(c, 0) * a[c]["bwd_ms"] for c in COUNTS if c in a)
    print(f"{C:6d}  fwd {wf:9.3f} ms   bwd {wb:9.3f} ms", flush=True)
print("[perf2] wrote", OUT, flush=True)
