"""Controlled experiment: how each latency-measurement technique improves
signal quality, using plain JAX matmuls of clearly different sizes as the
ground-truth ordering (bigger = slower). No graphax, no env.

Each "kernel" is a sequence of 2–4 matmuls chosen so the size ordering
produces a known wall-time ranking (rank 0 = fastest).  We measure every
kernel under every technique combination, repeat K=30 independent sessions
to gather statistics, then report for each technique arm:

  CV          – within-kernel coefficient of variation (noise)
  rank_err    – fraction of kernel pairs whose measured ranking inverts
  discrim     – between-kernel SD / within-kernel SD (signal-to-noise)
  spearman    – rank correlation with ground-truth (0=fastest)

Technique factors (varied independently + in combination):
  timer       : ResourceMonitor (old) | perf_counter (new)
  inner_reps  : 1 | 4 | 16
  warmup      : 0 | 3
  estimator   : min | p60 | mean | winsor20

Run on a pgi15 GPU node in CPU mode (single-core-pinned):
  numactl --physcpubind=+0 --membind=+0 \\
  JAX_PLATFORMS=cpu uv run --no-sync python \\
      src/alphagrad/approx/latency_technique_experiment.py

Output: JSON to stdout + a compact table.
"""
import os, json, sys, time, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# CPU_LABEL is injected by the launcher via env var so the two runs
# (single-core-pinned vs multi-core unpinned) write self-describing JSON.
CPU_LABEL = os.environ.get("LATEXP_CPU_LABEL", "unknown_cores")

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Kernels (JAX matmuls of different sizes, compiled once)
# ---------------------------------------------------------------------------
def _mm(n):   return ("mm", [(n, n), (n, n)])
def _axpy(n): return ("ax", [(n,), (n,), (n,)])  # z = a*x + y, compute-light mem-bound

# Mix of compute-bound (matmul) and memory-bound (axpy) at various sizes.
# Ordered roughly by wall-time (fastest first, ascending cost = ground truth).
KERNELS = [
    ("axpy_tiny",   _axpy(256)),
    ("axpy_small",  _axpy(4096)),
    ("axpy_med",    _axpy(65536)),
    ("mm_tiny",     _mm(32)),
    ("mm_small",    _mm(128)),
    ("mm_med",      _mm(512)),
    ("mm_large",    _mm(1024)),
]
# True rank: axpy_tiny < axpy_small < axpy_med < mm_tiny < mm_small < mm_med < mm_large
GROUND_TRUTH_RANK = {name: i for i, (name, _) in enumerate(KERNELS)}

rng = np.random.default_rng(0)


def make_kernel(kind_shapes: tuple):
    kind, shapes = kind_shapes
    if kind == "mm":
        # Chain of matmuls: compute-bound.
        arrays = [jnp.ones(s, dtype=jnp.float32) for s in shapes]
        def fn(*arrs):
            out = arrs[0]
            for a in arrs[1:]:
                out = out @ a
            return out
    else:
        # AXPY: z = alpha*x + y  (three 1-D arrays, memory-bandwidth bound).
        alpha = jnp.float32(2.0)
        arrays = [jnp.ones(shapes[0], jnp.float32),  # x
                  jnp.ones(shapes[1], jnp.float32),  # y
                  alpha]
        def fn(x, y, a):
            return a * x + y

    compiled = jax.jit(fn).lower(*arrays).compile()
    return compiled, arrays


print("Compiling kernels...", flush=True)
compiled_kernels = [(name, *make_kernel(kind_shapes)) for name, kind_shapes in KERNELS]
# Warmup the XLA executor
for _, fn, args in compiled_kernels:
    jax.block_until_ready(fn(*args))
print("Done.\n", flush=True)

# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------
try:
    from jax_memory_monitor import ResourceMonitor as _RM
    _RM_AVAIL = True
except ImportError:
    _RM_AVAIL = False
    class _RM:  # stub
        def __enter__(self): return self
        def __exit__(self, *a): pass
        stats = {"time": 0.0}


def measure_rm(fn, args, inner=1, warmup=0):
    """Old ResourceMonitor single-call per reading."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    with _RM() as mon:
        out = fn(*args)
    jax.block_until_ready(out)
    return float(mon.stats.get("time", 0.0)) * 1e9  # ns


def measure_pc(fn, args, inner=1, warmup=0):
    """perf_counter inner-loop (the new approach)."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(inner):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / inner * 1e9  # ns


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def agg_min(readings):   return float(np.min(readings))
def agg_p60(readings):   return float(np.percentile(readings, 60))
def agg_mean(readings):  return float(np.mean(readings))
def agg_winsor20(readings):
    a = np.sort(np.asarray(readings, dtype=np.float64))
    n = len(a); k = max(1, int(n * 0.20))
    if n - 2*k >= 1:
        a = a.copy(); a[:k] = a[k]; a[n-k:] = a[n-k-1]
    return float(a.mean())

ESTIMATORS = {
    "min":     agg_min,
    "p60":     agg_p60,
    "mean":    agg_mean,
    "winsor20": agg_winsor20,
}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
TIMER_MODES = ["rm", "pc"] if _RM_AVAIL else ["pc"]
INNER_REPS  = [1, 4, 16]
WARMUPS     = [0, 3]
POOL_SIZE   = 10   # readings per estimate (= num_data_points * reps_per_point)
SESSIONS    = 20   # independent repeated estimates per arm per kernel

arms = list(itertools.product(TIMER_MODES, INNER_REPS, WARMUPS, ESTIMATORS.keys()))
print(f"CPU config: {CPU_LABEL}")
print(f"Arms: {len(arms)}  ×  kernels: {len(KERNELS)}  ×  sessions: {SESSIONS}")
print(f"Total measurements: {len(arms)*len(KERNELS)*SESSIONS*POOL_SIZE:,}\n", flush=True)

# arm -> kernel -> list[SESSIONS estimates]
results: dict[tuple, dict[str, list[float]]] = {}

for arm_idx, (timer, inner, wu, est_name) in enumerate(arms):
    arm = (timer, inner, wu, est_name)
    results[arm] = {name: [] for name, *_ in compiled_kernels}
    print(
        f"[{arm_idx+1}/{len(arms)}] timer={timer} inner={inner} warmup={wu} est={est_name}",
        end="", flush=True
    )
    est_fn  = ESTIMATORS[est_name]
    meas_fn = measure_pc if timer == "pc" else measure_rm

    for _sess in range(SESSIONS):
        # Shuffle kernel order per session so any drift is uncorrelated
        perm = rng.permutation(len(compiled_kernels))
        for ki in perm:
            name, fn, args = compiled_kernels[ki]
            pool = [meas_fn(fn, args, inner=inner, warmup=(wu if _sess == 0 or True else 0))
                    for _ in range(POOL_SIZE)]
            pool = [x for x in pool if x > 0 and np.isfinite(x)]
            if pool:
                results[arm][name].append(est_fn(pool))
    print(f" ✓", flush=True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def spearman_rho(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 else float("nan")


def compute_metrics(arm):
    est_mat = np.array(
        [results[arm][name] for name, *_ in compiled_kernels]
    )  # (n_kernels, n_sessions)

    # Per-kernel CV
    cv_per = est_mat.std(axis=1) / (est_mat.mean(axis=1) + 1e-12)
    cv = float(np.mean(cv_per))

    # Discriminability: between-kernel SD / within-kernel SD
    between = float(np.std(est_mat.mean(axis=1)))
    within  = float(np.mean(est_mat.std(axis=1)))
    discrim = between / (within + 1e-12)

    # Spearman vs ground truth, per session then averaged
    gt_order = [GROUND_TRUTH_RANK[name] for name, *_ in compiled_kernels]
    spears = []
    for s in range(SESSIONS):
        meas_order = est_mat[:, s].tolist()
        spears.append(spearman_rho(gt_order, meas_order))
    spear = float(np.mean(spears))

    # Rank-inversion rate (fraction of pairs measured in wrong order)
    n = len(compiled_kernels)
    inversions = []
    for s in range(SESSIONS):
        inv = 0; total = 0
        for i in range(n):
            for j in range(i+1, n):
                total += 1
                if (est_mat[i, s] < est_mat[j, s]) != (gt_order[i] < gt_order[j]):
                    inv += 1
        inversions.append(inv / total)
    rank_err = float(np.mean(inversions))

    return {"cv": cv, "discrim": discrim, "spearman": spear, "rank_err": rank_err}


metrics = {arm: compute_metrics(arm) for arm in arms}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
header = (
    f"{'timer':4} {'inner':5} {'wu':3} {'est':9} | "
    f"{'CV':7} {'discrim':8} {'spearman':9} {'rank_err':9}"
)
sep = "-" * len(header)
print("\n" + sep)
print(header)
print(sep)

rows = []
for arm in arms:
    timer, inner, wu, est_name = arm
    m = metrics[arm]
    row = dict(timer=timer, inner_reps=inner, warmup=wu, estimator=est_name, **m)
    row["cpu_label"] = CPU_LABEL
    rows.append(row)
    print(
        f"{timer:4} {inner:5} {wu:3} {est_name:9} | "
        f"{m['cv']:7.4f} {m['discrim']:8.2f} {m['spearman']:9.4f} {m['rank_err']:9.4f}"
    )

print(sep)

# Sort by discriminability descending for a quick winner summary
rows_sorted = sorted(rows, key=lambda r: r["discrim"], reverse=True)
print("\nTop-5 arms by discriminability:")
for r in rows_sorted[:5]:
    print(
        f"  timer={r['timer']} inner={r['inner_reps']} wu={r['warmup']} "
        f"est={r['estimator']:9} | discrim={r['discrim']:.2f}  "
        f"spear={r['spearman']:.4f}  CV={r['cv']:.4f}"
    )

print("\nBottom-5 (worst) arms by discriminability:")
for r in rows_sorted[-5:]:
    print(
        f"  timer={r['timer']} inner={r['inner_reps']} wu={r['warmup']} "
        f"est={r['estimator']:9} | discrim={r['discrim']:.2f}  "
        f"spear={r['spearman']:.4f}  CV={r['cv']:.4f}"
    )

# Factor-level summary: average each metric across the OTHER factors
factors = {
    "timer":     lambda r: r["timer"],
    "inner_reps":lambda r: str(r["inner_reps"]),
    "warmup":    lambda r: str(r["warmup"]),
    "estimator": lambda r: r["estimator"],
}
print()
for fname, fget in factors.items():
    levels = sorted(set(fget(r) for r in rows))
    print(f"Factor '{fname}'  (mean across all other factors):")
    for lv in levels:
        sub = [r for r in rows if fget(r) == lv]
        print(
            f"  {lv:12} CV={np.mean([r['cv'] for r in sub]):.4f}  "
            f"discrim={np.mean([r['discrim'] for r in sub]):.2f}  "
            f"spear={np.mean([r['spearman'] for r in sub]):.4f}  "
            f"rank_err={np.mean([r['rank_err'] for r in sub]):.4f}"
        )
    print()

# JSON dump for further analysis
print("\n=== JSON ===")
print(json.dumps(rows, indent=2))
