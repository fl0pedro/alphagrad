"""PROCESS pool with CUDA_VISIBLE_DEVICES pinning -- the production design.

Every link is now individually measured: compile is 96% GIL-free XLA
backend (61445), blobs ship across processes bitwise-identically with
same-id pinning (61478, 0.08s load vs 32.7s compile), and the thread
variant hit 4.36x but kept a deterministic 3.5x peak outlier that
processes remove BY CONSTRUCTION -- an executable load can only ever
happen inside the measure process that owns the device, between windows.

    K compile procs (CVD=0)  ->  blob files + atomic-rename claims  ->
    1 measure proc per remaining GPU (CVD=i, sees its card as id 0)

Three arms answer the owner's three questions:
  seq        one process, compile+measure serially   -> the reference
  pool       full pipeline, no quality signal        -> THROUGHPUT + CLEAN?
  pool+walk  each plan adds a 200-step Adam-style    -> WHAT DOES THE
             quality walk after its timed windows       WALK COST?

WALK COST MODEL, stated honestly: the real `_loss_drop_quality` dispatches
200 plan-executions + 200 tiny adam steps asynchronously and blocks twice.
Here the walk is 200 sequential executions of the SAME compiled plan,
blocked each step -- the same device work, ~10-20 ms more sync overhead
total, i.e. a slight UPPER bound on the real walk's wall cost. What it
prices is the owner's actual question: does adding ~200 executions per
plan to the measure processes move the pipeline bottleneck, given that
the walk never touches a timed window (it runs after them, sequentially,
in the same process -- the next plan's window starts only after it ends).

PLANS ARE REGENERATED FROM THE SEED in every role rather than serialized:
same rng -> same 18 orders/transforms everywhere; blob files carry only
the executables. Coordination is the filesystem: compile worker w takes
plans w::K (static shard, no contention); measurers claim blobs with
os.rename, which is atomic -- a 3-line global queue.
"""

import argparse
import json
import os
import subprocess
import sys
import time

WORKDIR = "/tmp/xpool"


# --------------------------------------------------------------- plan gen ---
def make_plans(n_plans, seed, n_approx):
    import jax
    import numpy as np
    import bench_compile as bc

    fn = bc.get_fn("TransformerLM")
    xs = bc.get_args("TransformerLM", jax.random.PRNGKey(seed))
    argnums = tuple(range(len(xs)))
    jaxpr = jax.make_jaxpr(fn)(*xs)
    rng = np.random.default_rng(seed)
    plans = []
    for k in range(n_plans):
        order = [int(v) for v in
                 rng.permutation(np.arange(1, len(jaxpr.eqns) + 1))]
        tf = (bc.random_transforms(jaxpr, order, rng, n_approx)
              if k % 2 == 1 else [])
        plans.append(dict(k=k, order=order, tf=tf))
    return fn, xs, argnums, plans


def compile_one(fn, xs, argnums, plan):
    import jax
    import bench_compile as bc

    t0 = time.perf_counter()
    lowered = jax.jit(bc.jacve(fn, plan["order"], argnums=argnums,
                               transforms=plan["tf"] or None)).lower(*xs)
    exe = lowered.compile(compiler_options=bc.PAR_OPTS)
    return exe, time.perf_counter() - t0


def measure_one(exe, xs, reps):
    """clear -> timed reps -> peak delta; the contract's window."""
    import jax

    dev = jax.devices()[0]
    xs_d = [jax.device_put(x, dev) for x in xs]
    out = exe(*xs_d)
    jax.block_until_ready(out)
    jax.effects_barrier()
    base = float((dev.memory_stats() or {}).get("bytes_in_use", 0))
    dev.clear_memory_stats()   # DEVICE method (env.py _direct branch)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = exe(*xs_d)
    jax.block_until_ready(out)
    lat = (time.perf_counter() - t0) / reps
    peak = float((dev.memory_stats() or {}).get("peak_bytes_in_use", 0))
    return lat, max(0.0, peak - base)


def walk_one(exe, xs, steps):
    """200 sequential executions -- the quality walk's device cost.

    AFTER the timed windows, NEVER inside one (mirrors env.py:4186). Blocked
    per step, a slight upper bound on the async-dispatched real walk.
    """
    import jax

    t0 = time.perf_counter()
    for _ in range(steps):
        jax.block_until_ready(exe(*xs))
    return time.perf_counter() - t0


# ------------------------------------------------------------------ roles ---
def role_compile(a):
    fn, xs, argnums, plans = make_plans(a.plans, a.seed, a.n_approx)
    from jax.experimental import serialize_executable as sx
    import pickle

    for p in plans[a.worker::a.workers]:
        exe, t_c = compile_one(fn, xs, argnums, p)
        blob, in_tree, out_tree = sx.serialize(exe)
        del exe
        tmp = f"{WORKDIR}/p{p['k']}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump((blob, in_tree, out_tree, t_c), f)
        os.rename(tmp, f"{WORKDIR}/p{p['k']}.blob")   # atomic publish
        print(f"[compile w{a.worker}] plan {p['k']} {t_c:.1f}s", flush=True)


def role_measure(a):
    _, xs, _, plans = make_plans(a.plans, a.seed, a.n_approx)
    from jax.experimental import serialize_executable as sx
    import pickle

    pending = {p["k"] for p in plans}
    t_end = time.time() + 1500
    while pending and time.time() < t_end:
        claimed = None
        for k in sorted(pending):
            src = f"{WORKDIR}/p{k}.blob"
            dst = f"{WORKDIR}/p{k}.claim{a.dev}"
            try:
                os.rename(src, dst)          # atomic claim, one winner
                claimed = (k, dst)
                break
            except OSError:
                continue
        if claimed is None:
            done = {int(f[1:].split(".")[0]) for f in os.listdir(WORKDIR)
                    if ".claim" in f or f.endswith(".json")}
            pending -= done
            time.sleep(0.3)
            continue
        k, path = claimed
        pending.discard(k)
        with open(path, "rb") as f:
            blob, in_tree, out_tree, t_c = pickle.load(f)
        loaded = sx.deserialize_and_load(blob, in_tree, out_tree)
        lat, peak = measure_one(loaded, xs, a.reps)
        walk_s = walk_one(loaded, xs, a.walk) if a.walk else 0.0
        del loaded
        with open(f"{WORKDIR}/r{k}.json", "w") as f:
            json.dump(dict(k=k, lat=lat, peak=peak, t_compile=t_c,
                           walk_s=walk_s, dev=a.dev), f)
        print(f"[measure d{a.dev}] plan {k} lat={lat*1e3:.2f}ms "
              f"peak={peak/2**20:.0f}MB walk={walk_s:.1f}s", flush=True)


def role_seq(a):
    fn, xs, argnums, plans = make_plans(a.plans, a.seed, a.n_approx)
    for p in plans:
        exe, t_c = compile_one(fn, xs, argnums, p)
        lat, peak = measure_one(exe, xs, a.reps)
        walk_s = walk_one(exe, xs, a.walk) if a.walk else 0.0
        del exe
        with open(f"{WORKDIR}/r{p['k']}.json", "w") as f:
            json.dump(dict(k=p["k"], lat=lat, peak=peak, t_compile=t_c,
                           walk_s=walk_s, dev=1), f)
        print(f"[seq] plan {p['k']} compile={t_c:.1f}s "
              f"lat={lat*1e3:.2f}ms", flush=True)


# ----------------------------------------------------------- orchestrator ---
def spawn(role, cvd, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cvd)
    return subprocess.Popen(
        [sys.executable, "-u", __file__, "--role", role] + extra,
        env=env, stdout=sys.stdout, stderr=sys.stderr)

def run_arm(name, a, walk):
    import glob
    for f in glob.glob(f"{WORKDIR}/*"):
        os.remove(f)
    base = ["--plans", str(a.plans), "--seed", str(a.seed),
            "--reps", str(a.reps), "--n-approx", str(a.n_approx),
            "--walk", str(walk), "--workers", str(a.workers)]
    t0 = time.perf_counter()
    if name == "seq":
        procs = [spawn("seq", "1", base)]
    else:
        procs = ([spawn("compile", "0", base + ["--worker", str(w)])
                  for w in range(a.workers)]
                 + [spawn("measure", str(d), base + ["--dev", str(d)])
                    for d in (1, 2, 3)])
    rc = max(p.wait() for p in procs)
    wall = time.perf_counter() - t0
    res = {}
    for k in range(a.plans):
        try:
            with open(f"{WORKDIR}/r{k}.json") as f:
                res[k] = json.load(f)
        except FileNotFoundError:
            pass
    print(f"\n== {name}{' +walk' if walk else ''}: wall={wall:.1f}s "
          f"plans={len(res)}/{a.plans} rc={rc} ==", flush=True)
    return res, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="orchestrate")
    ap.add_argument("--plans", type=int, default=18)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--dev", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--walk", type=int, default=0)
    ap.add_argument("--walk-steps", type=int, default=200)
    ap.add_argument("--n-approx", type=int, default=8)
    ap.add_argument("--seed", type=int, default=250197)
    a = ap.parse_args()

    if a.role == "compile":
        return role_compile(a)
    if a.role == "measure":
        return role_measure(a)
    if a.role == "seq":
        return role_seq(a)

    os.makedirs(WORKDIR, exist_ok=True)
    import numpy as np

    seq, w_seq = run_arm("seq", a, walk=0)
    pool, w_pool = run_arm("pool", a, walk=0)
    poolw, w_poolw = run_arm("pool", a, walk=a.walk_steps)

    print(f"\n=========== VERDICT ({a.plans} plans, {a.workers} compile "
          f"procs, 3 measure GPUs) ===========")
    print(f"wall seq        : {w_seq:7.1f}s")
    print(f"wall pool       : {w_pool:7.1f}s   speedup {w_seq/w_pool:5.2f}x")
    print(f"wall pool+walk  : {w_poolw:7.1f}s   walk overhead "
          f"{(w_poolw-w_pool)/max(w_pool,1e-9)*100:+.0f}% of pool wall")
    ws = [r["walk_s"] for r in poolw.values() if r["walk_s"]]
    if ws:
        print(f"walk per plan   : median {np.median(ws):5.1f}s  "
              f"max {np.max(ws):5.1f}s  (vs median compile "
              f"{np.median([r['t_compile'] for r in poolw.values()]):.1f}s)")

    for tag, arm in (("pool", pool), ("pool+walk", poolw)):
        ks = sorted(set(seq) & set(arm))
        rl = [arm[k]["lat"] / max(seq[k]["lat"], 1e-12) for k in ks]
        rp = [arm[k]["peak"] / max(seq[k]["peak"], 1.0) for k in ks]
        print(f"{tag:10s} latency ratio vs seq: median {np.median(rl):5.3f} "
              f"max {np.max(rl):5.3f} | peak ratio: median "
              f"{np.median(rp):5.3f} max {np.max(rp):5.3f}")
    print("CLEAN = medians ~1.00, max within a few %. The walk arm's ratios "
          "are the whole point: the walk must not dirty the windows.")


if __name__ == "__main__":
    main()
