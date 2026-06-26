"""Verify the dispersion measurement is faithful: (1) replayed seqs really carry
dynamic substeps (>1 micro-action on some vertices) and build_order_specs keeps them;
(2) the eval data points are DISTINCT MNIST batches (else within-config cosine is
artificially constant); (3) re-measured mean cosine matches the known-good
front_recos value. Prints per-config: #substeps, distinct-batch check, per-point
cosine vector, and stored-vs-remeasured cosine.
"""
import os, json
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX, MAX_RULES_PER_VERTEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
CI = REWARD_INDEX["cosine_sim"]
NDATA = 16


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=NDATA, reps_per_point=1, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, NDATA)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


env, ev = build_env()

# (2) are the NDATA eval data points distinct? hash the first array leaf of each sample
print("=== eval-sample diversity check ===")
print("ev type:", type(ev), "len:", len(ev) if hasattr(ev, "__len__") else "?")
try:
    leaf0 = np.asarray(ev[0])
    print("ev[0] leaf shape:", leaf0.shape)
    # ev is the tuple passed to _callback; the data points live along an axis — show variance across points
    print("ev[0] per-point means (first 8):", [round(float(x), 4) for x in np.asarray(ev[0]).reshape(NDATA, -1)[:, :].mean(1)[:8]] if np.asarray(ev[0]).size % NDATA == 0 else "n/a")
except Exception as e:
    print("diversity probe error:", e)

# load front_recos (known-good aligned cosine) for spot configs spanning the range
recos = []
for s in ("cmorl", "mogfn"):
    for p in json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{s}_front_recos.json")))["front"]:
        recos.append((s, p["seq"], float(p["obj"]["cosine_sim"])))
recos.sort(key=lambda t: t[2])
pick = [recos[int(i)] for i in np.linspace(0, len(recos) - 1, 10).round().astype(int)]


def nsubsteps(seq):
    return [(v, len(calls)) for v, calls in seq]


print("\n=== per-config: dynamic-substeps + per-point cosine + stored vs remeasured ===")
for s, seq, stored in pick:
    order, specs, n_rules = build_order_specs(seq, env)
    multi = sum(1 for _, c in seq if len(c) > 1)
    maxcalls = max((len(c) for _, c in seq), default=0)
    rs = {}
    _, _, reward = _callback(env.config, env.args, env.consts,
                             jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs)
    cpp = np.asarray(rs.get("cosine_sim_per_point", []), float)
    print(f"[{s}] stored_cos={stored:.3f} remeasured_mean={cpp.mean() if cpp.size else float('nan'):.3f} "
          f"std={cpp.std() if cpp.size else float('nan'):.3f} n_rules={n_rules} multi_call_vtx={multi} max_calls={maxcalls}")
    print(f"      per-point cos[:8]: {[round(float(x),3) for x in cpp[:8]]}")
