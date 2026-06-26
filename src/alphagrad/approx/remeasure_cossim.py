"""Re-measure cosine_sim for every dumped MORL candidate with the CURRENT env
implementation (layout-aligned value_and_grad cosine), on fresh deterministic MNIST
eval samples. Matches the MORL run config (--measure-grad, 5 data points,
percentile_keep 0.60) but skips latency timing (latency/memory are deterministic and
kept from the dump). Sharded for parallelism.
  uv run python src/alphagrad/approx/remeasure_cossim.py --src cmorl --shard 0 --nshards 48 --out /tmp/rm/cmorl_0.jsonl
"""
import os, json, argparse
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--ndata", type=int, default=5)
ap.add_argument("--out", required=True)
A = ap.parse_args()

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
CI = REWARD_INDEX["cosine_sim"]


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=A.ndata, reps_per_point=1, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


env, ev = build_env()


def measure(seq):
    order, specs, _ = build_order_specs(seq, env)
    _, _, reward = _callback(env.config, env.args, env.consts,
                             jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink={})
    return float(np.asarray(reward)[CI])


cands = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{A.src}_allcand.json")))["candidates"]
idxs = list(range(A.shard, len(cands), A.nshards))
os.makedirs(os.path.dirname(A.out), exist_ok=True)
# resume: skip candidates already measured in ANY prior shard file (global done
# set across all {src}_*.jsonl) — avoids redundant re-measurement when re-sharded
import glob as _glob
done = set()
for _f in _glob.glob(os.path.join(os.path.dirname(A.out), f"{A.src}_*.jsonl")):
    for line in open(_f):
        try:
            r = json.loads(line)
            if np.isfinite(r.get("cos_new", float("nan"))):
                done.add(r["i"])
        except Exception:
            pass
todo = [i for i in idxs if i not in done]
with open(A.out, "a") as f:
    for n, i in enumerate(todo):
        c = cands[i]
        try:
            cn = measure(c["seq"])
        except Exception:
            cn = float("nan")
        f.write(json.dumps(dict(i=i, cos_old=c["obj"]["cosine_sim"], cos_new=cn,
                                latency_ns=c["obj"]["latency_ns"],
                                xla_peak_memory=c["obj"]["xla_peak_memory"])) + "\n")
        f.flush()
        jax.clear_caches()   # release compiled XLA executables — bounds RSS
print(f"done shard {A.shard}/{A.nshards} src={A.src} measured={len(todo)} skipped={len(done)}")
