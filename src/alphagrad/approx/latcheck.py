"""Measure approx-grad latency for a few rules under one single-core mechanism, so
we can compare the training-experiment mechanism (XLA eigen-off + OMP=1, unpinned)
to the MORL front mechanism (sched_setaffinity to 1 core).

  --mech thread   : XLA_FLAGS=--xla_cpu_multi_thread_eigen=false + OMP=1 (training)
  --mech affinity : os.sched_setaffinity(0,{0}) before jax init  (MORL front)
"""
import os, sys, json, time, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mech", choices=["thread", "affinity"], required=True)
ap.add_argument("--source", default="cmorl")
ap.add_argument("--idxs", default="3,9,28,5")
A = ap.parse_args()

os.environ["JAX_PLATFORMS"] = "cpu"
if A.mech == "thread":
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
else:  # affinity: pin to one core BEFORE jax/XLA init so XLA sizes its pool to 1
    try:
        os.sched_setaffinity(0, {0})
    except Exception as e:
        print("affinity set failed:", e)

import numpy as np
import alphagrad.approx.common.datasets as ds
ds.NN_VMAP_BATCH = 16
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback
import alphagrad.approx.common.compile_cache as ccmod
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

ARGN = infer_argnums("VmappedNeuralNetwork")
print(f"[mech={A.mech}] jax devices: {jax.devices()}  affinity: {os.sched_getaffinity(0)}")

k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
tfn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=16)
closed = jax.make_jaxpr(tfn)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=tfn,
    cmp_type="latency", mem_type="peak_memory", measure_latency=False,
    num_data_points=1, reps_per_point=1, percentile_keep=0.6,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
    measure_grad=True, latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, 1)
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)


def capture(order, specs):
    cap = {}; orig = ccmod.cached_compile
    def cc(key, fn):
        out = orig(key, fn)
        if isinstance(key, (bytes, bytearray)) and key.startswith(b"approx:"): cap["fn"] = out
        return out
    ccmod.cached_compile = cc
    try:
        _callback(env.config, env.args, env.consts, jnp.asarray(order), jnp.asarray(specs), len(order), *ev)
    finally:
        ccmod.cached_compile = orig
    return cap["fn"]


def init_w(ind=784, hid=256, out=10):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    return (jax.random.normal(k1, (hid, ind)) / np.sqrt(ind), jnp.zeros(hid),
            jax.random.normal(k2, (out, hid)) / np.sqrt(hid), jnp.zeros(out))


front = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{A.source}_front.json")))["front"]
x, y = ev[0][0], ev[1][0]
W = init_w()
items = A.idxs.split(",")
for it in items:
    if it == "rev":
        gfn = jax.jit(jax.value_and_grad(tfn, argnums=ARGN))
    else:
        order, specs, _ = build_order_specs(front[int(it)]["seq"], env)
        gfn = capture(order, specs)
    idx = it
    for _ in range(5):
        jax.block_until_ready(gfn(x, y, *W))
    ts = []
    for _ in range(40):
        t0 = time.perf_counter(); jax.block_until_ready(gfn(x, y, *W)); ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.array(ts); lo, hi = np.percentile(ts, [10, 90]); w = ts[(ts >= lo) & (ts <= hi)]
    print(f"  {A.source}#{idx}: lat={np.mean(w):8.1f}us  (median {np.median(ts):8.1f}, std {np.std(ts):6.1f})")
