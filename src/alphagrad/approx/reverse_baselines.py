"""Reverse-mode GRAD baselines + ViT-front Blackwell re-measurement.

Reuses the env grad-measure harness EXACTLY (--measure-grad, --exec-on-gpu,
peak_bytes_in_use via ALPHAGRAD_PEAK_MEMORY_SYNC, prealloc=false) so the numbers
are directly comparable to the runs' pareto front latency_ns/peak_memory.

Configs measured (per --example):
  * jacve-reverse : graphax.value_and_grad of the scalar loss along the REVERSE
    elimination order (o_list = valid_vertices reversed), NO transforms -> the
    reverse-mode elimination grad. Measured via env._callback (same harness).
  * jax-grad      : native jax.value_and_grad(scalar_loss) (== jacrev of the
    scalar loss == grad), measured with the SAME winsorized/percentile latency +
    exact peak_bytes_in_use loop the env uses.
  * --front-json  : (ViT) replay each pareto seq through the SAME env._callback
    grad-measure path (0-based valid_vertices action idx -> resolve).

cosine=1.0 by construction for the reverse baselines (they reproduce the exact
grad); the front replays report the recorded cosine.
"""
import os, argparse, json, time
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_SYNC", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_ABSOLUTE", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import (
    VertexEliminationEnv, _callback, REWARD_INDEX,
    _winsorized_mean, _percentile_pool, _device_peak_in_use, _device_clear_peaks,
)
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn,
)
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

ap = argparse.ArgumentParser()
ap.add_argument("--example", default="VmappedNeuralNetwork")
ap.add_argument("--ndata", type=int, default=5)
ap.add_argument("--reps-per-point", type=int, default=2)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--percentile-keep", type=float, default=0.60)
ap.add_argument("--latency-winsor", type=float, default=0.0)  # runs use 0.0 (P60)
ap.add_argument("--vit-pow2", type=int, default=0)
ap.add_argument("--front-json", default="")  # if set, also replay these seqs
A = ap.parse_args()

if A.example == "VmappedViT" and A.vit_pow2:
    os.environ["ALPHAGRAD_VIT_POW2"] = "1"

LOSS = scalar_loss_fn(get_fn(A.example))
ARGN = infer_argnums(A.example)
LI = REWARD_INDEX["latency_ns"]; PI = REWARD_INDEX["peak_memory"]; CI = REWARD_INDEX["cosine_sim"]


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args(A.example, ak, dataset="mnist")
    gen = data_gen(A.example, dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory",
        exec_on_gpu=True, measure_latency=True,
        num_data_points=A.ndata, reps_per_point=A.reps_per_point,
        percentile_keep=A.percentile_keep, slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0, measure_grad=True,
        latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter",
        latency_winsor=A.latency_winsor,
    )
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


env, ev = build_env()
valid = np.asarray(env.valid_vertices, dtype=np.int32)
nvtx = len(valid)
devs = list(jax.devices())
node = os.environ.get("SLURMD_NODENAME", jax.devices()[0].platform)
print(f"[rev] example={A.example} nvtx={nvtx} node={node} device={jax.devices()[0]} "
      f"inner_reps={A.latency_inner_reps} pk={A.percentile_keep} winsor={A.latency_winsor}", flush=True)


def _measure_callback(order, specs, label):
    rs = {}
    _callback(env.config, env.args, env.consts,
              jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs)
    lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
    peak_s = [x for x in rs.get("peak_memory_samples", []) if x > 0 and np.isfinite(x)]
    if A.latency_winsor > 0:
        lat = _winsorized_mean(lat_s, A.latency_winsor) if lat_s else float("nan")
    else:
        lat = _percentile_pool(lat_s, A.percentile_keep) if lat_s else float("nan")
    peak = _percentile_pool(peak_s, A.percentile_keep) if peak_s else float("nan")
    cos_pp = rs.get("cosine_sim_per_point", [])
    cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
    print(f"[rev] {label:22s} lat_us={lat/1e3:.3f} peak_MB={peak/1e6:.2f} cos={cos:+.4f}",
          flush=True)
    return dict(lat_us=lat/1e3, peak_MB=peak/1e6, cos=cos)


# ---------------- jacve REVERSE mode (reverse elimination order, no transforms)
rev_order = list(reversed(range(nvtx)))          # 0-based action idx, reversed
rev_seq = [(v, []) for v in rev_order]
order, specs, _ = build_order_specs(rev_seq, env)
res = {}
res["jacve_reverse"] = _measure_callback(order, specs, "jacve-reverse(grad)")


# ---------------- native jax.value_and_grad (== jacrev-as-grad), same harness
def measure_native_grad():
    fn = jax.value_and_grad(LOSS, argnums=tuple(ARGN))
    # use the SAME eval sample points the env uses (ev is a tuple of stacked
    # per-arg sample batches, shape (ndata, ...)); measure per point like env.
    n_points = A.ndata
    lat_samples, peak_samples, cosines = [], [], []
    unique_devs = list({d for d in devs[:1]})  # single measure device
    for d in range(n_points):
        eval_args = [arg[d] for arg in ev]
        eval_args = [jax.device_put(a, devs[0]) for a in eval_args]
        g = jax.jit(fn, keep_unused=True)
        val, grads = g(*eval_args); jax.block_until_ready((val, grads))  # warm compile
        for _rep in range(A.reps_per_point):
            _device_clear_peaks(unique_devs)
            t0 = time.perf_counter()
            for _ in range(A.latency_inner_reps):
                val, grads = g(*eval_args)
            jax.block_until_ready((val, grads))
            dt = (time.perf_counter() - t0) / A.latency_inner_reps * 1e9
            peak = _device_peak_in_use(unique_devs)
            lat_samples.append(dt); peak_samples.append(peak)
        cosines.append(1.0)  # native grad == exact grad by construction
    if A.latency_winsor > 0:
        lat = _winsorized_mean(lat_samples, A.latency_winsor)
    else:
        lat = _percentile_pool(lat_samples, A.percentile_keep)
    peak = _percentile_pool([p for p in peak_samples if p > 0], A.percentile_keep)
    print(f"[rev] {'jax-grad(native)':22s} lat_us={lat/1e3:.3f} peak_MB={peak/1e6:.2f} cos=+1.0000",
          flush=True)
    return dict(lat_us=lat/1e3, peak_MB=peak/1e6, cos=1.0)


res["jax_grad"] = measure_native_grad()

# ---------------- optional: replay the pareto front seqs (ViT apples-to-apples)
front_rows = []
if A.front_json:
    fj = json.load(open(A.front_json))
    print(f"[rev] === FRONT REPLAY ({A.front_json}) {len(fj.get('front',[]))} pts ===", flush=True)
    for i, pt in enumerate(fj.get("front", [])):
        seq = pt.get("seq")
        if not seq:
            continue
        # The recorded pareto/best seq stores the 1-based jaxpr vertex_id
        # (vertex_action + 1 already applied), but build_order_specs expects the
        # 0-based ACTION INDEX (it re-resolves via valid_vertices). Detect the
        # 1-based case (max vid == nvtx, min == 1) and convert v -> v-1 so the
        # resolution round-trips to the correct elimination order.
        def _vid(s):
            return s[0] if isinstance(s, list) else s["vertex"]
        _vids = [int(_vid(s)) for s in seq]
        _one_based = (max(_vids) == nvtx and min(_vids) == 1)
        if _one_based:
            seq = [
                ([_vid(s) - 1, s[1]] if isinstance(s, list)
                 else {**s, "vertex": s["vertex"] - 1})
                for s in seq
            ]
        try:
            o, s, _ = build_order_specs(seq, env)
            r = _measure_callback(o, s, f"front[{i}]")
            r["recorded_cos"] = pt["obj"].get("cosine_sim")
            front_rows.append((i, r))
        except Exception as e:
            print(f"[rev] front[{i}]: FAIL {type(e).__name__}: {e}", flush=True)

print("\n[rev] ===== SUMMARY (node=%s, %s) =====" % (node, A.example), flush=True)
for k, v in res.items():
    print(f"[SUMMARY] {A.example} {k}: lat_us={v['lat_us']:.3f} peak_MB={v['peak_MB']:.2f} cos={v['cos']:+.4f}", flush=True)
for i, r in front_rows:
    print(f"[FRONT] {A.example} front[{i}]: lat_us={r['lat_us']:.3f} peak_MB={r['peak_MB']:.2f} "
          f"cos={r['cos']:+.4f} recorded_cos={r.get('recorded_cos')}", flush=True)
