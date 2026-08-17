"""Compile pool + measure pool, pure JAX + threads. No Ray, no processes.

THE ARCHITECTURE UNDER TEST (the owner's design, adapted to what the
benchmarks proved):

    plan queue -> K compile threads -> ready queue -> 1 measure thread/GPU

  * COMPILE THREADS overlap because XLA's backend compile is C++ that
    releases the GIL -- bench 61445 measured the backend at 96% of the
    compile pipeline (median 14.5 s vs 0.53 s of Python trace+lower), so
    nearly all of each compile can run concurrently with the others.
  * The 3.5% Python part (graphax trace + MLIR lowering) runs under ONE
    GLOBAL LOCK. That serialisation costs ~0.5 s/plan and buys total
    freedom from JAX's tracing thread-safety rules ("It is not permitted
    to manipulate JAX trace values concurrently from multiple threads") --
    the cheapest correctness argument available.
  * Compiles bind to DEVICE 0 (production: the trainer GPU, never a
    measurement device), so executable loads NEVER touch a measure GPU.
    The binary is then serialize()d -- a pickle of the compiled PJRT
    executable, no recompilation -- and the measure thread
    deserialize_and_load()s it onto ITS device, outside any timed window.
    This is the same machinery compile_cache.py already uses in-process.
  * ONE measure thread per measure GPU, pulling from a single global
    ready queue (work-stealing across GPUs, FIFO within). The validity
    contract holds by construction: only that thread ever loads or runs
    anything on its device, and it does load -> warmup -> clear ->
    barrier -> timed reps -> block -> read peak, exactly env.py's order.

THE CONTROL. The same plans also run through a SEQUENTIAL arm -- lower,
compile, measure, one at a time, the current actor model in miniature.
Two numbers come out:
  1. wall(pool) vs wall(seq): the throughput claim.
  2. per-plan latency and peak, pool vs seq: the CLEANLINESS claim. If the
     pool's concurrency contaminates the timed windows, these diverge;
     the paired-comparison rule (drift moves 18-20%) applies to this
     benchmark as much as to any other.

KNOWN LIMIT, stated up front: threads do not isolate crashes. ~10% of TLM
plans deterministically segfault ptxas on Blackwell (#104); in this
prototype such a plan kills the whole process, where Ray would respawn an
actor. That is the one argument processes keep. Everything else Ray was
providing -- queueing, device pinning, compile overlap -- this file does
with ~200 lines of stdlib threading.
"""

import argparse
import queue
import threading
import time

import io as _io

import jax
import numpy as np
from jax.experimental import serialize_executable as sx

import bench_compile as bc


def load_on(blob, in_tree, out_tree, dev):
    """deserialize_and_load with the device REBOUND to `dev`.

    The stock loader resolves pickled Device refs strictly BY ID
    (`devices_by_id[pid[1]]`), so an executable compiled on device 0
    cannot load with execution_devices=[dev1] -- KeyError 0. The
    EXECUTABLE itself rebinds fine (deserialize_executable takes
    executable_devices); only these auxiliary refs need the remap: send
    every pickled id to the target device. Single-device executables
    only, which is all this pool ever makes.
    """
    u = sx._JaxPjrtUnpickler(_io.BytesIO(blob), dev.client, [dev])
    u.devices_by_id = {d.id: dev for d in dev.client.devices()}
    unloaded, args_info_flat, no_kwargs = u.load()
    args_info = in_tree.unflatten(args_info_flat)
    return jax.stages.Compiled(
        unloaded.load(), [], args_info, out_tree, no_kwargs=no_kwargs)


# ---------------------------------------------------------------- measure ---
def measure_on(dev, loaded, xs, reps):
    """env.py's window: put -> warmup -> clear -> timed reps -> peak delta."""
    xs_d = [jax.device_put(x, dev) for x in xs]
    out = loaded(*xs_d)                      # warmup (compile-free: AOT)
    jax.block_until_ready(out)
    jax.effects_barrier()
    base = float((dev.memory_stats() or {}).get("bytes_in_use", 0))
    dev.clear_memory_stats()   # DEVICE method (env.py _direct branch)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = loaded(*xs_d)
    jax.block_until_ready(out)
    lat = (time.perf_counter() - t0) / reps
    peak = float((dev.memory_stats() or {}).get("peak_bytes_in_use", 0))
    return lat, max(0.0, peak - base)


# ------------------------------------------------------------------- pool ---
def run_pool(plans, xs, argnums, fn, k_compile, measure_devs, reps,
             ship=True):
    """ship=True : compile on device 0, serialize, REBIND onto the measure
                 device -- loads never touch a measure GPU outside its own
                 measure thread. Depends on the CUDA client honouring
                 executable_devices (the CPU client does NOT).
    ship=False: compile DIRECTLY for a round-robin measure device; the
                 executable load lands on that device from the compile
                 thread, possibly during another plan's timed window. The
                 cleanliness check exists to price exactly that risk."""
    lower_lock = threading.Lock()       # serialises the 3.5% Python part
    plan_q = queue.Queue()
    ready_qs = ({d.id: queue.Queue() for d in measure_devs}
                if not ship else None)
    ready_q = queue.Queue() if ship else None
    results = {}
    errors = []
    # DIRTY-WINDOW DEFENSE (no-ship): an executable LOAD lands on the target
    # device at the END of compile(), possibly inside another plan's timed
    # window on that device -- measured: median latency ratio 1.047 but max
    # 3.587 without this. Compile threads log their load-completion times per
    # device; the measurer rejects any window a load overlapped and simply
    # re-times (the executable is already resident, a retry costs one window,
    # ~1s, against 14.5s compiles -- collisions are the exception).
    load_log = {d.id: [] for d in measure_devs}
    load_log_lock = threading.Lock()
    retries = [0]

    for i, p in enumerate(plans):
        p["dev"] = measure_devs[i % len(measure_devs)]
        plan_q.put(p)

    def compiler():
        while True:
            try:
                p = plan_q.get_nowait()
            except queue.Empty:
                return
            try:
                target = p["dev"]
                lower_xs = xs if ship else [jax.device_put(x, target)
                                            for x in xs]
                with lower_lock:
                    lowered = jax.jit(
                        bc.jacve(fn, p["order"], argnums=argnums,
                                 transforms=p["tf"] or None)
                    ).lower(*lower_xs)
                t0 = time.perf_counter()
                exe = lowered.compile(compiler_options=bc.PAR_OPTS)
                t_c = time.perf_counter() - t0
                if ship:
                    blob, in_tree, out_tree = sx.serialize(exe)
                    del exe, lowered             # free device-0 residency
                    ready_q.put((p["k"], blob, in_tree, out_tree, t_c))
                else:
                    with load_log_lock:
                        load_log[target.id].append(time.perf_counter())
                    ready_qs[target.id].put((p["k"], exe, t_c))
            except Exception as e:               # noqa: BLE001
                errors.append((p["k"], f"{type(e).__name__}: {e}"))

    def measurer(dev):
        q = ready_q if ship else ready_qs[dev.id]
        while len(results) + len(errors) < len(plans):
            try:
                item = q.get(timeout=2.0)
            except queue.Empty:
                continue
            try:
                if ship:
                    k, blob, in_tree, out_tree, t_c = item
                    loaded = load_on(blob, in_tree, out_tree, dev)
                else:
                    k, loaded, t_c = item
                for attempt in range(5):
                    w0 = time.perf_counter()
                    lat, peak = measure_on(dev, loaded, xs, reps)
                    w1 = time.perf_counter()
                    with load_log_lock:
                        # 0.5s of pre-window slack: the logged stamp is
                        # post-load, the allocation lands slightly earlier.
                        dirty = any(w0 - 0.5 <= t <= w1
                                    for t in load_log[dev.id])
                    if not dirty:
                        break
                    retries[0] += 1
                results[k] = dict(lat=lat, peak=peak, t_compile=t_c,
                                  dev=dev.id, attempts=attempt + 1)
                del loaded
            except Exception as e:               # noqa: BLE001
                errors.append((item[0], f"measure: {type(e).__name__}: {e}"))

    t0 = time.perf_counter()
    cs = [threading.Thread(target=compiler) for _ in range(k_compile)]
    ms = [threading.Thread(target=measurer, args=(d,))
          for d in measure_devs]
    for t in cs + ms:
        t.start()
    for t in cs + ms:
        t.join()
    if retries[0]:
        print(f"  [pool] dirty-window retries: {retries[0]}", flush=True)
    return results, errors, time.perf_counter() - t0


# -------------------------------------------------------------- sequential --
def run_seq(plans, xs, argnums, fn, dev, reps):
    """The current actor model in miniature: one plan at a time, compile on
    the measure device itself (loads land there, as they do in the actor)."""
    results = {}
    t0 = time.perf_counter()
    xs_d = [jax.device_put(x, dev) for x in xs]
    for p in plans:
        lowered = jax.jit(
            bc.jacve(fn, p["order"], argnums=argnums,
                     transforms=p["tf"] or None)).lower(*xs_d)
        t1 = time.perf_counter()
        exe = lowered.compile(compiler_options=bc.PAR_OPTS)
        t_c = time.perf_counter() - t1
        lat, peak = measure_on(dev, exe, xs, reps)
        results[p["k"]] = dict(lat=lat, peak=peak, t_compile=t_c, dev=dev.id)
        del exe, lowered
    return results, time.perf_counter() - t0


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="TransformerLM")
    ap.add_argument("--plans", type=int, default=18)
    ap.add_argument("--compile-threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-approx", type=int, default=8)
    ap.add_argument("--seed", type=int, default=250197)
    ap.add_argument("--no-ship", action="store_true",
                    help="compile directly for the measure device instead "
                         "of serialize+rebind (fallback if the CUDA client "
                         "does not honour executable_devices)")
    a = ap.parse_args()

    fn = bc.get_fn(a.example)
    xs = bc.get_args(a.example, jax.random.PRNGKey(a.seed))
    argnums = tuple(range(len(xs)))
    jaxpr = jax.make_jaxpr(fn)(*xs)
    n_v = len(jaxpr.eqns)
    rng = np.random.default_rng(a.seed)

    devs = jax.devices()
    compile_dev, measure_devs = devs[0], devs[1:]
    assert measure_devs, "need >= 2 GPUs: device 0 compiles, the rest measure"
    print(f"example={a.example} V={n_v} compile_dev=0 "
          f"measure_devs={[d.id for d in measure_devs]} "
          f"K={a.compile_threads} plans={a.plans}", flush=True)

    plans = []
    for k in range(a.plans):
        order = [int(v) for v in rng.permutation(np.arange(1, n_v + 1))]
        tf = (bc.random_transforms(jaxpr, order, rng, a.n_approx)
              if k % 2 == 1 else [])
        plans.append(dict(k=k, order=order, tf=tf))

    # SEQ first: it is the reference the pool's numbers are checked against,
    # and running it first means its windows see a cold, quiet device.
    seq, wall_seq = run_seq(plans, xs, argnums, fn, measure_devs[0], a.reps)
    print(f"\nwall(seq)  = {wall_seq:7.1f}s  ({len(seq)}/{len(plans)} plans)",
          flush=True)

    pool, errors, wall_pool = run_pool(
        plans, xs, argnums, fn, a.compile_threads, measure_devs, a.reps,
        ship=not a.no_ship)
    print(f"wall(pool) = {wall_pool:7.1f}s  ({len(pool)}/{len(plans)} plans, "
          f"{len(errors)} errors)", flush=True)
    for k, msg in errors:
        print(f"  plan {k} FAILED in pool: {msg}", flush=True)

    if pool:
        print(f"\nSPEEDUP pool/seq wall: "
              f"{wall_seq / max(wall_pool, 1e-9):5.2f}x", flush=True)
    else:
        print("\nNO SPEEDUP CLAIM: the pool produced zero results, so its "
              "wall time is meaningless.", flush=True)

    # Cleanliness: the pool's timed windows must reproduce seq's numbers.
    print("\nplan |    lat(seq)    lat(pool)  ratio |   peak(seq)MB "
          "peak(pool)MB  ratio | dev", flush=True)
    ratios_l, ratios_p = [], []
    for k in sorted(set(seq) & set(pool)):
        s, p = seq[k], pool[k]
        rl = p["lat"] / max(s["lat"], 1e-12)
        rp = p["peak"] / max(s["peak"], 1.0)
        ratios_l.append(rl)
        ratios_p.append(rp)
        print(f"{k:4d} | {s['lat']*1e3:10.3f}ms {p['lat']*1e3:10.3f}ms "
              f"{rl:6.3f} | {s['peak']/2**20:12.1f} {p['peak']/2**20:12.1f} "
              f"{rp:6.3f} | {p['dev']}", flush=True)
    if ratios_l:
        print(f"\nlatency ratio pool/seq: median "
              f"{np.median(ratios_l):5.3f}  max {np.max(ratios_l):5.3f}")
        print(f"peak    ratio pool/seq: median "
              f"{np.median(ratios_p):5.3f}  max {np.max(ratios_p):5.3f}")
        print("CLEAN if both medians ~1.00 and max within a few %; the "
              "49.7%-CV failure mode would show as large, one-sided "
              "latency inflation.")


if __name__ == "__main__":
    main()
