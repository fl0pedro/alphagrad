"""Sigma calibration for grad-mode PopArt-OFF manual lambdas (bridge-cse).

Measures the RAW per-order value of the cost channels (flops, latency_ns,
peak_memory) on the --measure-grad executable over N random full-variant
grad orders and reports the raw std == the PopArt-equivalent per-channel
sigma. With PopArt OFF the reward weight per cost channel = 0.06 / sigma
replicates PopArt's per-channel (G_k - mu_k)/sigma_k * w_outer scaling.

Also sweeps --latency-inner-reps and reports the latency CV per setting so
the ViT run can drop the NN's 50 reps to the lowest stable value.

Adapted from src/alphagrad/approx/measure_edge.py (same env-build + _callback
raw_sink API). Run on ONE GPU via srun; exec-on-gpu measures on GPU exactly
like the training measure actors.
"""
import os, argparse, random
# grad-mode measurement; bkstep OFF (not needed for cost-channel sigmas).
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")  # need flops (idx1)
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_ABSOLUTE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn,
)
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

ap = argparse.ArgumentParser()
ap.add_argument("--example", default="VmappedNeuralNetwork")
ap.add_argument("--nrandom", type=int, default=24)
ap.add_argument("--ndata", type=int, default=5)
ap.add_argument("--reps-per-point", type=int, default=2)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--inner-reps-sweep", default="",
                help="comma list of inner-reps to sweep for latency CV, e.g. 5,10,20,50")
ap.add_argument("--quant-dtypes", default="int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16")
ap.add_argument("--vit-pow2", type=int, default=1)
ap.add_argument("--with-compress", type=int, default=0,
                help="include COMPRESS ops in random orders (can trigger an "
                     "uncatchable Triton mixed-dtype LLVM fatal on H100/Blackwell "
                     "-> whole run dies). Default 0 = DIAG-only (clean, and "
                     "representative: the mem-gate/prevalidate skips pathological "
                     "COMPRESS in training so those never enter PopArt's sigma).")
A = ap.parse_args()

if A.example == "VmappedViT" and A.vit_pow2:
    os.environ["ALPHAGRAD_VIT_POW2"] = "1"

LOSS = scalar_loss_fn(get_fn(A.example))
ARGN = infer_argnums(A.example)
FI = REWARD_INDEX["flops"]
LI = REWARD_INDEX["latency_ns"]
PI = REWARD_INDEX["peak_memory"]


def build_env(inner_reps):
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args(A.example, ak, dataset="mnist")
    gen = data_gen(A.example, dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory",
        exec_on_gpu=True, measure_latency=True,
        num_data_points=A.ndata, reps_per_point=A.reps_per_point,
        percentile_keep=0.60, slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0, measure_grad=True,
        latency_inner_reps=inner_reps, latency_timer="perf_counter",
    )
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


def measure_seq(env, ev, seq, label):
    try:
        order, specs, nr = build_order_specs(seq, env)
        rs = {}
        _callback(env.config, env.args, env.consts,
                  jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs)
        flops = float(rs.get("flops", float("nan")))
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        peak_s = [x for x in rs.get("peak_memory_samples", []) if x > 0 and np.isfinite(x)]
        lat = float(np.mean(lat_s)) if lat_s else float("nan")
        peak = float(np.mean(peak_s)) if peak_s else float("nan")
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
        # within-order latency CV (dispersion of the per-sample latency pool)
        lat_cv = float(np.std(lat_s) / np.mean(lat_s)) if len(lat_s) > 1 else float("nan")
        print(f"[cal] {label:16s} nrules={nr:2d} flops={flops:.4g} "
              f"lat_ns={lat:.4g} peak={peak:.4g} cos={cos:+.3f} lat_cv={lat_cv:.4f}",
              flush=True)
        return dict(flops=flops, lat=lat, peak=peak, cos=cos, lat_cv=lat_cv)
    except Exception as e:
        print(f"[cal] {label}: FAIL {type(e).__name__}: {e}", flush=True)
        return None


def make_random_seqs(nvtx, n, seed=0, with_quant=False, with_compress=False):
    # DIAG (plain elim) spans the cost range for sigma calibration via the
    # elimination ORDER (rand0 flops 3.7e8 .. rand1 1.8e10 = 50x on NN). Both
    # COMPRESS and QUANT default OFF: standalone build_order_specs bypasses the
    # policy masks + prevalidate, and a COMPRESS densify or a raw quant dtype can
    # trigger an UNCATCHABLE Triton mixed-dtype LLVM fatal (kills the process).
    # In training the mem-gate/prevalidate SKIPS those pathological orders, so
    # they never enter PopArt's per-channel sigma anyway -> DIAG-only sigma is
    # the PopArt-representative one.
    KINDS = ["min", "mean"]
    QD = ["bfloat16", "float16"]
    rng = random.Random(seed)
    out = []
    for r in range(n):
        # permute the elimination order to vary flops/latency/peak spread
        perm = list(range(nvtx))
        rng.shuffle(perm)
        seq = []
        for i in perm:
            c = rng.random()
            if with_compress and c < 0.25:
                seq.append((i, [f"compress('{rng.choice(KINDS)}', {rng.choice([0,1])})"]))
            elif with_quant and c < 0.40:
                seq.append((i, [f"quant('{rng.choice(QD)}')"]))
            else:
                seq.append((i, []))
        out.append((f"rand{r}", seq))
    return out


# ------------------------------------------------------------------ main sigma
env, ev = build_env(A.latency_inner_reps)
valid = np.asarray(env.valid_vertices, dtype=np.int32)
nvtx = len(valid)
print(f"[cal] example={A.example} nvtx={nvtx} inner_reps={A.latency_inner_reps} "
      f"ndata={A.ndata} reps_per_point={A.reps_per_point}", flush=True)

seqs = make_random_seqs(nvtx, A.nrandom)
rows = []
for name, seq in seqs:
    r = measure_seq(env, ev, seq, name)
    if r:
        rows.append(r)

flops = np.array([r["flops"] for r in rows if np.isfinite(r["flops"])])
lat = np.array([r["lat"] for r in rows if np.isfinite(r["lat"])])
peak = np.array([r["peak"] for r in rows if np.isfinite(r["peak"])])


def _report(name, arr):
    if len(arr) < 2:
        print(f"[SIGMA] {name}: INSUFFICIENT n={len(arr)}", flush=True)
        return
    print(f"[SIGMA] {name}: n={len(arr)} mean={arr.mean():.6g} std={arr.std(ddof=1):.6g} "
          f"median={np.median(arr):.6g} min={arr.min():.6g} max={arr.max():.6g} "
          f"iqr/1.349={(np.quantile(arr,0.75)-np.quantile(arr,0.25))/1.349:.6g}", flush=True)


print("\n[cal] ===== RAW PER-ORDER SIGMAS (PopArt-equivalent) =====", flush=True)
_report("flops", flops)
_report("latency_ns", lat)
_report("peak_memory", peak)
print("[cal] weight = 0.06 / std  (cost channels); cosine_sim:1.0", flush=True)
for nm, arr in (("flops", flops), ("latency_ns", lat), ("peak_memory", peak)):
    if len(arr) >= 2:
        s = arr.std(ddof=1)
        print(f"[WEIGHT] {nm}:{0.06/s:.4g}   (sigma={s:.6g})", flush=True)

# ------------------------------------------------------------- inner-reps sweep
if A.inner_reps_sweep:
    print("\n[cal] ===== LATENCY INNER-REPS CV SWEEP =====", flush=True)
    # fixed representative order (mostly-identity, high-fidelity grad) for a
    # clean cross-inner-reps CV comparison.
    ref_seq = [(i, []) for i in range(nvtx)]
    for reps in [int(x) for x in A.inner_reps_sweep.split(",") if x.strip()]:
        e2, ev2 = build_env(reps)
        cvs = []
        for trial in range(3):
            r = measure_seq(e2, ev2, ref_seq, f"ir{reps}_t{trial}")
            if r and np.isfinite(r["lat_cv"]):
                cvs.append(r["lat_cv"])
        cv = float(np.mean(cvs)) if cvs else float("nan")
        print(f"[INNER_REPS_CV] inner_reps={reps:3d} mean_lat_cv={cv:.4f} "
              f"(over {len(cvs)} trials)", flush=True)
