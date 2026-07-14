"""Standalone GPU measurement worker for the parallel autoscheduler loop.

Pinned to ONE GPU (via CUDA_VISIBLE_DEVICES set by the parent), builds the env
once, measures a batch of orders (read from a JSON job file), writes the measured
4-tuples to an output JSON, then EXITS — so all its retained XLA executables (the
measure-GPU leak) are freed by process teardown. Called once per round per GPU.

Job JSON: {"orders": [[aidx...], ...], "micro_budget": 16, "seed": N}
Out JSON: {"results": [[lat,peak,flops,cos] or null, ...]}
"""
import os, sys, json
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_SYNC", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("ALPHAGRAD_QUANT_ALLOWED", "int8,int16,bfloat16,float16")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# leak/COMPRESS: mem-gate off (worker dies each round -> bounded), Safe monitor on
os.environ.setdefault("ALPHAGRAD_MEASURE_MEM_FLOOR_GIB", "0")
os.environ.setdefault("ALPHAGRAD_MAX_MEASURE_MEM_GIB", "200")
os.environ.setdefault("ALPHAGRAD_MEASURE_MEM_SAFETY", "1.0")

job_path, out_path, nn_hidden, ndata, inner_reps = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(nn_hidden)

import numpy as np, jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs
MAX_RULES = 16

job = json.load(open(job_path))
orders, micro_budget, seed = job["orders"], int(job["micro_budget"]), int(job["seed"])

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork")); ARGN = infer_argnums("VmappedNeuralNetwork")
k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True, measure_latency=True,
    num_data_points=int(ndata), reps_per_point=1, percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=True,
    latency_inner_reps=int(inner_reps), latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, int(ndata))
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)

QD = ["int8", "int16", "bfloat16", "float16"]
def seq_of(order_ids):
    if micro_budget <= 0:
        return [(int(v), []) for v in order_ids]
    dr = np.random.default_rng(abs(hash(tuple(int(x) for x in order_ids))) % (2 ** 31))
    return [(int(v), [f"quant('{QD[int(dr.integers(0, len(QD)))]}')" for _ in range(micro_budget)])
            for v in order_ids]

results = []
for order_ids in orders:
    try:
        order, specs, _ = build_order_specs(seq_of(order_ids), env)
        rs = {}
        _callback(env.config, env.args, env.consts, jnp.asarray(order),
                  jnp.asarray(specs), len(order), *ev, raw_sink=rs)
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        lat = float(np.mean(lat_s)) if lat_s else float("nan")
        peak = float(rs.get("xla_peak_memory", float("nan")))
        flops = float(rs.get("flops", float("nan")))
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
        r = [lat, peak, flops, cos]
        if not all(np.isfinite(x) for x in r):
            print(f"[mw-nan] lat={lat} peak={peak} flops={flops} cos={cos} "
                  f"lat_n={len(lat_s)} cos_n={len(cos_pp)} keys={sorted(rs.keys())}", file=sys.stderr, flush=True)
        results.append(r if all(np.isfinite(x) for x in r) else None)
    except Exception as e:
        import traceback; traceback.print_exc(file=sys.stderr); sys.stderr.flush()
        results.append(None)
json.dump({"results": results}, open(out_path, "w"))
print(f"[mw] measured {sum(1 for r in results if r)}/{len(orders)} on {os.environ.get('CUDA_VISIBLE_DEVICES','?')}", flush=True)
