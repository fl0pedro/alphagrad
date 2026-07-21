"""Persistent SUBPROCESS measure server (CUDA-poison isolation).

Builds the measurement env ONCE, then serves line-JSON requests on stdin:
    {"seq": [[action_idx, ["quant('int8')", ...]], ...]}
and answers one line-JSON on stdout:
    {"raw": [latency_ns, xla_peak_memory, flops, cosine]}  or  {"error": "..."}

A CUDA_ERROR_ILLEGAL_ADDRESS from a bad diag/compress kernel kills THIS process
only; the parent (measure_client.MeasureClient) restarts it and treats the
request as invalid (redraw). Env knobs: ALPHAGRAD_NN_HIDDEN, ALPHAGRAD_MS_NDATA,
ALPHAGRAD_MS_INNER_REPS.
"""
import os, sys, json

os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn)
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.order_specs import build_order_specs

NDATA = int(os.environ.get("ALPHAGRAD_MS_NDATA", "5"))
REPS = int(os.environ.get("ALPHAGRAD_MS_INNER_REPS", "50"))
TASK = os.environ.get("ALPHAGRAD_MS_TASK", "VmappedNeuralNetwork")
DSET = os.environ.get("ALPHAGRAD_MS_DATASET", "mnist")

LOSS = scalar_loss_fn(get_fn(TASK))
ARGN = infer_argnums(TASK)
k0 = jax.random.PRNGKey(0); ak, ek = jax.random.split(k0)
xs = get_args(TASK, ak, dataset=DSET)
gen = data_gen(TASK, dataset=DSET, dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    sparse=(os.environ.get("ALPHAGRAD_SPARSE", "0") == "1"),
    cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True, measure_latency=True,
    num_data_points=NDATA, reps_per_point=1, percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=True,
    latency_inner_reps=REPS, latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, NDATA)
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)

# startup diagnostics -> stderr (parent logs it): the ONE authoritative view of
# what this child actually sees (dev platform, CVD, JAX_PLATFORMS).
print(f"[ms-diag] devices={[str(d) for d in jax.devices()]} "
      f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r} "
      f"exe={sys.executable}", file=sys.stderr, flush=True)
print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        seq = [(int(a), [str(o) for o in ops]) for a, ops in req["seq"]]
        order, specs, _ = build_order_specs(seq, env)
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
        if all(np.isfinite(x) for x in r):
            out = {"raw": r}
        else:
            out = {"error": "non-finite channels"}
    except BaseException as e:  # noqa: BLE001 — anything survivable stays served
        try:
            jax.clear_caches()
        except Exception:
            pass
        out = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
    print(json.dumps(out), flush=True)
