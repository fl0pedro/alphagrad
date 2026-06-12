"""Standalone search-space sampler for the graphax "full" variant.

Generates N uniformly-random, valid FULL-variant configurations (random
elimination order + random per-vertex DIAG/COMPRESS/QUANT transforms) for
VmappedNeuralNetwork+MNIST, then measures the 6 cost channels + quality
for each over a FIXED bank of 8 MNIST data points x 10 reps (raw, not
aggregated). Purpose: rough characterisation of the "full" search space.

Design notes
------------
* **Reproducible & dynamic**: config(idx) is a pure function of
  (--seed, idx) via ``np.random.SeedSequence([seed, idx])`` — INDEPENDENT
  of the total sample count. So sample idx is byte-identical whether you
  ask for 10k or 100k; bumping --num-samples just appends new configs and
  re-measures only the new tail.
* **Skip-existing**: results stream to ``results_seed{S}.jsonl`` (one JSON
  line per order, append-only / crash-safe). On restart, already-measured
  idx are skipped. Same for the frozen ``samples_seed{S}.npz`` config bank.
* **Faithful sampling**: micro-actions are sampled uniformly over the
  legal masked set and encoded by the SAME canonical encoder the PPO
  policy uses (``env.micro_actions_to_rule_specs_jax``); invalid combos
  fall back to the env's post-filter (sentinel), exactly as in training.
* **Raw, not aggregated**: each measurement task captures the per-point
  latency/peak/quality via env._callback(..., raw_sink=...) and stores them
  whole — one jsonl line per (config, pass). You aggregate in analysis.
* **Shuffled run order**: the 10x8 per config is split into ``num_passes``
  passes (default 10) of ``num_data_points`` points (default 8). EVERY
  (config, pass) task is shuffled into one global list (deterministic in
  --seed) and dispatched in that order, so a config's passes land at
  scattered times across the multi-day run — time-varying system noise
  (thermal, neighbour jobs, clock drift) is decorrelated from config
  identity and captured by the per-config 10x8 distribution.
* **No trimming**: flop-gate and slow-exec cutoff are OFF (per request) —
  expensive orders run to completion so the sampled distribution is
  unbiased. Use --max-exec-seconds to cap only if you must.
* **Max parallel**: one CpuApproximationServer per Ray actor, each pinned
  to a disjoint logical-core slice on the single node (cpu2). All actors
  share the same fixed 8 MNIST examples (same --seed); only the run ORDER
  is randomized, so the 8 examples stay consistent across configs.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

# Disable Ray's `uv run` runtime-env hook BEFORE any `import ray`. Under
# `uv run`, Ray auto-sets the runtime_env working_dir to the uv project root
# (~/dsnn, ~2.8GB) and uploads it, exceeding Ray's 512MB cap and killing
# ray.init. The sampler runs a purely LOCAL single-node cluster and needs no
# project shipping, so we turn the hook off (ray_constants documents this flag).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


# QUANT dtype indices that actually RUN in this graphax build. The other
# ~15 (sub-byte ints int2/int4/uint2/uint4, every float8 variant, float4,
# complex64/128) raise TypePromotionError in matmul-densify, so a config
# containing any of them is invalid — sampling uniformly over all 28 made
# ~100% of full-space configs fail. Probed 2026-06-06 against
# graphax.sparse.micro_actions.QUANT_DTYPES (28 entries):
#   [bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
#    bfloat16, float16, float32, float64]
# Override with --quant-dtype-indices if the graphax build changes.
WORKING_QUANT_DTYPE_IDX = (0, 3, 4, 5, 6, 9, 10, 11, 12, 22, 23, 24, 25)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num-workers", type=int, default=64,
                   help="Ray actors; each pinned to (total_cores//workers) cores")
    p.add_argument("--out-dir", default=os.path.expanduser("~/dsnn/search_space_full"))
    p.add_argument("--example", default="VmappedNeuralNetwork")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--dataset-size", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="override the network hidden width (NN_HIDDEN_DIM). "
                        "SAMPLER-ONLY runtime monkeypatch; the shared default "
                        "(128) and the matrix are left untouched. e.g. 256")
    p.add_argument("--num-data-points", type=int, default=8,
                   help="distinct MNIST examples per pass (fixed across all configs)")
    p.add_argument("--num-passes", type=int, default=10,
                   help="time-distributed measurement passes per config (the '10' in "
                        "10x8). ALL (config,pass) tasks are SHUFFLED together so each "
                        "config's passes land at scattered times across the run, "
                        "decorrelating time-varying system noise from config identity.")
    p.add_argument("--num-eval-samples", type=int, default=8,
                   help="size of the fixed MNIST bank (>= num-data-points)")
    p.add_argument("--max-exec-seconds", type=float, default=0.0,
                   help="0 = no cutoff (default; do NOT trim slow runs)")
    p.add_argument("--quant-dtype-indices", default=None,
                   help="comma-sep indices into QUANT_DTYPES to sample from; "
                        "default = the runnable subset WORKING_QUANT_DTYPE_IDX")
    p.add_argument("--generate-only", action="store_true",
                   help="only build+freeze the config bank, skip measurement")
    p.add_argument("--exec-on-gpu", action="store_true",
                   help="run the Jacobian measurement on GPU (exec_on_gpu=True)")
    p.add_argument("--actor-num-gpus", type=float, default=0.0,
                   help="GPUs reserved per measurement actor (1.0 with --exec-on-gpu)")
    p.add_argument("--ray-address", default=None)
    p.add_argument("--measure-grad", action="store_true",
                   help="measure value_and_grad of the scalar loss (the gradient "
                        "computation) instead of the full Jacobian; records the "
                        "grad-cosine quality + the deterministic xla_peak_memory "
                        "channel — matches the MORL trainers' measurement.")
    p.add_argument("--latency-timer", default="perf_counter",
                   choices=["perf_counter", "rm"],
                   help="latency timer: perf_counter (default) or rm (fixed "
                        "ResourceMonitor).")
    return p.parse_args()


def make_args_dict(a) -> dict:
    # Mirrors the keys CpuApproximationServer.from_args_dict reads. Gates
    # OFF (full fidelity), latency measurement ON, 8x10 measurement pool.
    return dict(
        example=a.example,
        hidden_dim=getattr(a, "hidden_dim", None),
        dataset=a.dataset,
        dataset_size=a.dataset_size,
        cmp_type="latency",
        mem_type="peak_memory",
        exec_on_gpu=bool(getattr(a, "exec_on_gpu", False)),
        measure_latency=True,
        latency_samples=1,
        num_data_points=int(a.num_data_points),
        reps_per_point=1,  # one rep per call; the passes are shuffled at task level
        percentile_keep=0.60,  # unused for raw, but the env expects it
        slow_exec_cutoff_seconds=float(a.max_exec_seconds),  # 0 => no trim
        flop_gate_threshold=0.0,  # no flop gate
        num_eval_samples=int(a.num_eval_samples),
        intermediate_rewards=False,  # terminal-only
        cost_pipeline_schedule="always_full",
        seed=int(a.seed),
        # Grad-mode measurement + deterministic xla_peak_memory channel — mirrors
        # the MORL trainers so the sampled space matches what they optimize.
        measure_grad=bool(getattr(a, "measure_grad", False)),
        latency_timer=str(getattr(a, "latency_timer", "perf_counter")),
    )


# ---------------------------------------------------------------------------
# Config generation (host-side, deterministic per idx)
# ---------------------------------------------------------------------------
def build_gen_context(args_dict, seed):
    """Build one server to extract the static graph facts needed to
    generate configs: valid vertices, per-vertex axis state/validity."""
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
    srv = CpuApproximationServer.from_args_dict(args_dict, variant="full", seed=seed)
    env = srv._env
    vv = np.array(env.valid_vertices, dtype=np.int32)
    axis_state = np.asarray(env.axis_state_static)      # (total_v, MAX_AXES, FEAT)
    axis_valid = np.asarray(env.axis_valid_static)      # (total_v, MAX_AXES)
    return srv, vv, axis_state, axis_valid


def make_encoder():
    """Return a host callable wrapping the canonical micro-action encoder."""
    import jax.numpy as jnp
    from alphagrad.approx.env import micro_actions_to_rule_specs_jax

    def enc(op, i, j, fac, axis_state_v, ck, qd):
        row = micro_actions_to_rule_specs_jax(
            jnp.asarray(op), jnp.asarray(i), jnp.asarray(j), jnp.asarray(fac),
            jnp.asarray(axis_state_v), compress_kinds=jnp.asarray(ck),
            quant_dtypes=jnp.asarray(qd),
        )
        return np.asarray(row, dtype=np.int32)

    return enc


def gen_one(idx, base_seed, vv, axis_state, axis_valid, enc,
            factors, quant_idx, nk, max_rules, max_axes):
    """Deterministic config for sample ``idx`` — depends ONLY on
    (base_seed, idx), so it is stable across changes to total N.

    QUANT dtypes are drawn from ``quant_idx`` (the runnable subset) so the
    config is overwhelmingly likely to be valid; the rare residual invalid
    (e.g. a DIAG factor that doesn't divide an axis) is caught + recorded
    by the actor rather than crashing."""
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(idx)]))
    N = len(vv)
    order = vv.copy()
    rng.shuffle(order)
    specs = np.full((N, max_rules, 3), -1, dtype=np.int32)
    specs[..., 2] = 0
    S = max_rules
    quant_idx = np.asarray(quant_idx, dtype=np.int32)
    for vidx in range(N):
        v = int(order[vidx])
        valid_axes = np.where(axis_valid[v - 1] > 0.5)[0]
        if valid_axes.size == 0:
            valid_axes = np.arange(max_axes)
        op = rng.integers(0, 4, size=S).astype(np.int32)            # DIAG/COMPRESS/QUANT/END
        i = rng.choice(valid_axes, size=S).astype(np.int32)
        j = rng.choice(valid_axes, size=S).astype(np.int32)
        fac = rng.choice(factors, size=S).astype(np.int32)
        ck = rng.integers(0, nk, size=S).astype(np.int32)
        qd = rng.choice(quant_idx, size=S).astype(np.int32)
        specs[vidx] = enc(op, i, j, fac, axis_state[v - 1], ck, qd)
    return order.astype(np.int32), specs


def ensure_samples(path, n, base_seed, vv, axis_state, axis_valid, quant_idx):
    """Generate-or-extend the frozen config bank at ``path`` to >= n
    samples. Returns (orders (n,V) int32, specs (n,V,R,3) int32)."""
    from alphagrad.approx.env import MAX_RULES_PER_VERTEX
    from graphax.sparse.micro_actions import QUANT_DTYPES, COMPRESS_KINDS

    enc = make_encoder()
    factors = np.array([-1, 2, 3, 4, 8, 16], dtype=np.int32)
    nk = len(COMPRESS_KINDS)
    max_rules = int(MAX_RULES_PER_VERTEX)
    max_axes = int(axis_valid.shape[1])
    V = len(vv)

    existing_orders = existing_specs = None
    have = 0
    if os.path.exists(path):
        z = np.load(path)
        existing_orders, existing_specs = z["orders"], z["specs"]
        have = existing_orders.shape[0]
        if have >= n and existing_orders.shape[1] == V:
            print(f"[gen] reuse {have} cached samples (>= {n})", flush=True)
            return existing_orders[:n], existing_specs[:n]
        print(f"[gen] extending cache {have} -> {n}", flush=True)

    orders = np.zeros((n, V), dtype=np.int32)
    specs = np.zeros((n, V, max_rules, 3), dtype=np.int32)
    if have:
        orders[:have] = existing_orders[:have]
        specs[:have] = existing_specs[:have]
    t0 = time.time()
    for idx in range(have, n):
        o, s = gen_one(idx, base_seed, vv, axis_state, axis_valid, enc,
                       factors, quant_idx, nk, max_rules, max_axes)
        orders[idx] = o
        specs[idx] = s
        if (idx + 1) % 1000 == 0:
            print(f"[gen] {idx + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, orders=orders, specs=specs)
    os.replace(tmp, path)
    print(f"[gen] froze {n} samples -> {path} ({time.time() - t0:.0f}s)", flush=True)
    return orders, specs


# ---------------------------------------------------------------------------
# Ray measurement actor
# ---------------------------------------------------------------------------
def _pin_affinity(actor_id, num_workers):
    try:
        avail = sorted(os.sched_getaffinity(0))
        n = len(avail)
        per = max(1, n // max(num_workers, 1))
        base = (actor_id * per) % n
        cores = {avail[(base + k) % n] for k in range(per)}
        os.sched_setaffinity(0, cores)
        return sorted(cores)
    except (AttributeError, OSError, ValueError):
        return None


def make_actor_cls():
    import ray

    @ray.remote
    class SamplerActor:
        def __init__(self, args_dict, actor_id, num_workers, seed):
            self._cores = _pin_affinity(actor_id, num_workers)
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            _hd = args_dict.get("hidden_dim")
            if _hd:
                import alphagrad.approx.common.examples as _ex
                _ex.NN_HIDDEN_DIM = int(_hd)
            from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
            from alphagrad.approx.env import _callback
            self._srv = CpuApproximationServer.from_args_dict(
                args_dict, variant="full", seed=seed,
            )
            self._cb = _callback
            self._id = actor_id

        def ready(self):
            info = {"id": self._id, "host": socket.gethostname(), "cores": self._cores,
                    "cuda": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
            try:
                import jax
                info["devices"] = str(jax.devices())
            except Exception:
                pass
            return info

        def measure(self, idx, pass_id, order, specs):
            # Calling _callback DIRECTLY (to capture raw_sink) bypasses
            # evaluate()'s try/except, so we replicate it here: a chunk of
            # the FULL action space is invalid (e.g. float8 QUANT that
            # breaks graphax matmul-densify, DIAG factors that don't divide
            # an axis, COMPRESS mid-order) and raises at compile/exec time.
            # The PPO policy hits the same and gets a -1e10 sentinel, so
            # recording these as invalid (not crashing the actor) is the
            # faithful behaviour. Filter on `invalid` in analysis.
            import jax.numpy as jnp
            srv = self._srv
            raw: dict = {}
            t0 = time.time()
            try:
                _, _, reward = self._cb(
                    srv._config, srv._args, srv._consts,
                    jnp.asarray(order, dtype=jnp.int32),
                    jnp.asarray(specs, dtype=jnp.int32),
                    int(len(order)), *srv._eval_samples,
                    init=False, point_idx=-1, raw_sink=raw,
                )
                raw["idx"] = int(idx)
                raw["pass"] = int(pass_id)
                raw["wall_s"] = round(time.time() - t0, 3)
                raw["reward_vec"] = [float(x) for x in np.asarray(reward)]
                raw["invalid"] = False
            except Exception as exc:
                # Keep failed sequences IN the database with all measurements
                # marked NaN (uniform schema with valid rows) + the error, so
                # failure modes can be analysed (group by `err`). A chunk of the
                # full action space is invalid (int4/float8 QUANT into a matmul,
                # DIAG factors that don't divide an axis, COMPRESS mid-order).
                from alphagrad.approx.env import NUM_REWARDS
                nan = float("nan")
                raw = {
                    "idx": int(idx), "pass": int(pass_id), "invalid": True,
                    "err": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "wall_s": round(time.time() - t0, 3),
                    "reward_vec": [nan] * NUM_REWARDS,
                    "muls_adds_fmas": nan, "max_io_sum": nan, "flops": nan,
                    "bytes_accessed": nan, "xla_peak_memory": nan,
                    "latency_ns_samples": [], "peak_memory_samples": [],
                    "cosine_sim_per_point": [], "frob_residual_per_point": [],
                }
            return raw

    return SamplerActor


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_done(results_path):
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    # A 'fatal' line is a transient actor-death record from the
                    # old (non-fault-tolerant) dispatch -> DON'T count it as
                    # done, so it gets retried. valid / invalid / crashed are
                    # final outcomes and count as done.
                    if o.get("fatal") or o.get("error"):
                        continue  # transient worker-death record -> retry it
                    done.add((int(o["idx"]), int(o["pass"])))
                except Exception:
                    pass
    return done


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args_dict = make_args_dict(a)
    if getattr(a, "hidden_dim", None):
        import alphagrad.approx.common.examples as _ex
        _ex.NN_HIDDEN_DIM = int(a.hidden_dim)
        print(f"[hidden-dim] NN_HIDDEN_DIM -> {int(a.hidden_dim)} "
              f"(sampler-only override; matrix/default untouched)", flush=True)

    samples_path = os.path.join(a.out_dir, f"samples_seed{a.seed}.npz")
    results_path = os.path.join(a.out_dir, f"results_seed{a.seed}.jsonl")

    # ---- Phase 1: generate / freeze the config bank ----
    print(f"== Phase 1: generate {a.num_samples} configs ==", flush=True)
    _srv, vv, axis_state, axis_valid = build_gen_context(args_dict, a.seed)
    del _srv  # driver only needed it for the static graph facts
    if a.quant_dtype_indices:
        quant_idx = tuple(int(x) for x in a.quant_dtype_indices.split(","))
    else:
        # Post-merge (graphax dtype-promotion fix) ALL quant dtypes run, so
        # sample the full action space. WORKING_QUANT_DTYPE_IDX is the
        # pre-merge runnable subset, kept only as a documented fallback.
        from graphax.sparse.micro_actions import NUM_QUANT_DTYPES
        quant_idx = tuple(range(NUM_QUANT_DTYPES))
    print(f"[gen] num_valid_vertices={len(vv)} max_axes={axis_valid.shape[1]} "
          f"quant_dtype_indices={quant_idx}", flush=True)
    orders, specs = ensure_samples(samples_path, a.num_samples, a.seed,
                                   vv, axis_state, axis_valid, quant_idx)
    if a.generate_only:
        print("generate-only: done.", flush=True)
        return

    # ---- Phase 2: measure (parallel, resumable, SHUFFLED run order) ----
    # Every (config, pass) is its own task; the full list is shuffled together
    # so a config's `num_passes` repeats land at scattered times across the run
    # (B1, A2, A0, C1, ...). Time-varying system noise (thermal, neighbour jobs,
    # clock drift) is thereby decorrelated from config identity and captured by
    # the per-config 10x8 distribution. Shuffle is deterministic in --seed, so
    # the run order is reproducible and resumable. Each task records its own
    # {idx, pass} jsonl line; analysis regroups the num_passes x num_data_points
    # raw measurements per config.
    done = load_done(results_path)
    tasks = [(i, p) for i in range(a.num_samples) for p in range(a.num_passes)]
    np.random.default_rng(np.random.SeedSequence([int(a.seed), 999])).shuffle(tasks)
    todo = [t for t in tasks if t not in done]
    print(f"== Phase 2: measure | {len(done)} done, {len(todo)} todo "
          f"({a.num_samples} configs x {a.num_passes} passes = {len(tasks)} "
          f"shuffled tasks) ==", flush=True)
    if not todo:
        print("nothing to measure.", flush=True)
        return

    import ray
    if a.ray_address:
        ray.init(address=a.ray_address)
    else:
        # Fresh LOCAL single-node cluster (the sampler runs entirely on one
        # node). address="local" ignores any ambient RAY_ADDRESS.
        ray.init(address="local",
                 num_cpus=a.num_workers + 4,
                 ignore_reinit_error=True,
                 include_dashboard=False)

    from collections import deque, defaultdict
    from ray.exceptions import RayActorError

    SamplerActor = make_actor_cls()
    MAX_RETRY = 1       # retry a task once on a fresh worker before marking crashed
    RECYCLE_EVERY = 25  # proactively ray.kill+respawn a worker every N tasks to
                        # bound the cost_analysis C++ leak (the env's reset is a
                        # no-op; only killing the process frees the C++ allocation
                        # to the OS). At 150 the node filled (~700GB/64 workers)
                        # before the first recycle and Ray's memory-monitor began
                        # killing workers -> mass failures. Mirrors PPO recycling.

    def spawn(slot):
        # Same slot id -> same disjoint core slice (stable affinity on respawn).
        return SamplerActor.options(
            num_cpus=1, num_gpus=float(getattr(a, "actor_num_gpus", 0.0) or 0.0),
        ).remote(args_dict, slot, a.num_workers, a.seed)

    actors = {slot: spawn(slot) for slot in range(a.num_workers)}
    infos = ray.get([actors[s].ready.remote() for s in actors])
    print(f"[ray] {len(actors)} actors up on {infos[0]['host']}; "
          f"cores/actor={len(infos[0]['cores']) if infos[0]['cores'] else '?'}; "
          f"MAX_RETRY={MAX_RETRY} RECYCLE_EVERY={RECYCLE_EVERY}", flush=True)

    # Fault-tolerant streaming dispatch. The FULL random action space contains
    # poison configs (memory blowups; gates are OFF by design) that OOM-kill the
    # worker PROCESS -- uncatchable by the in-actor try/except. On worker death
    # we respawn the slot (async: the next measure RPC queues behind the new
    # actor's init, so the driver never blocks) and retry the task up to
    # MAX_RETRY times; a task that keeps killing workers is recorded
    # {"crashed": true} (itself a data point) and skipped, so one poison config
    # can't cascade-kill the run. Periodic recycle bounds the slower leak.
    fout = open(results_path, "a", buffering=1)
    work = deque(todo)
    fut_meta = {}                  # future -> (slot, task)
    attempts = defaultdict(int)
    since_recycle = defaultdict(int)
    n_done = n_crashed = n_respawn = 0
    t_start = time.time()

    def launch(slot):
        if not work:
            return
        idx, p = work.popleft()
        fut = actors[slot].measure.remote(idx, p, orders[idx], specs[idx])
        fut_meta[fut] = (slot, (idx, p))

    for slot in list(actors):
        launch(slot)

    while fut_meta:
        done_futs, _ = ray.wait(list(fut_meta), num_returns=1)
        for fut in done_futs:
            slot, task = fut_meta.pop(fut)
            idx, p = task
            try:
                raw = ray.get(fut)
                fout.write(json.dumps(raw) + "\n")
                n_done += 1
                since_recycle[slot] += 1
                if since_recycle[slot] >= RECYCLE_EVERY:
                    ray.kill(actors[slot]); actors[slot] = spawn(slot)
                    since_recycle[slot] = 0; n_respawn += 1
                launch(slot)
            except Exception as exc:
                # ANY ray.get failure means the worker is gone: either a poison
                # config that crashed the process (RayActorError) OR Ray's
                # memory-monitor killing it because the node hit its RAM limit
                # (OutOfMemoryError -> the cost_analysis leak; the proactive
                # RECYCLE_EVERY above keeps the node from ever getting there).
                # Respawn the slot (async) and retry; record crashed once the
                # retry budget is spent so one bad task can't cascade.
                actors[slot] = spawn(slot); since_recycle[slot] = 0; n_respawn += 1
                attempts[task] += 1
                if attempts[task] <= MAX_RETRY:
                    work.appendleft(task)
                else:
                    fout.write(json.dumps(
                        {"idx": idx, "pass": p, "crashed": True,
                         "reason": str(exc)[:120]}) + "\n")
                    n_done += 1; n_crashed += 1
                launch(slot)
            if n_done % 100 == 0 and n_done:
                rate = n_done / max(time.time() - t_start, 1e-9)
                eta = (len(todo) - n_done) / max(rate, 1e-9)
                print(f"[measure] {n_done}/{len(todo)} ({rate * 60:.0f}/min, "
                      f"eta {eta / 3600:.1f}h; crashed={n_crashed} respawns={n_respawn})",
                      flush=True)

    fout.close()
    print(f"== done: {n_done} results ({n_crashed} crashed, {n_respawn} respawns) "
          f"in {(time.time() - t_start) / 3600:.2f}h ==", flush=True)


if __name__ == "__main__":
    sys.exit(main())
