"""Validate the latency-measurement overhaul end-to-end: build the env with
OLD (single-call, P60) vs NEW (perf_counter inner-loop + winsorized mean)
settings, replay a FIXED elimination order K times, and compare the
reproducibility (CV) of the aggregated latency *reward* — the quantity the RL
agent actually sees."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import equinox as eqx
from alphagrad.approx.env import (
    VertexEliminationEnv, MAX_RULES_PER_VERTEX, REWARD_INDEX, _callback,
)
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums
from alphagrad.approx.common.eval_samples import generate_eval_samples

LI = REWARD_INDEX["latency_ns"]


def build_env(inner_reps, warmup, winsor, ndp=5, reps=2):
    key = jax.random.PRNGKey(0)
    ak, ek = jax.random.split(key)
    fn = get_fn("VmappedNeuralNetwork")
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(fn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=infer_argnums("VmappedNeuralNetwork"),
        num_envs=0, data_gen=gen, target_fun=fn,
        cmp_type="latency", mem_type="peak_memory", measure_latency=True,
        num_data_points=ndp, reps_per_point=reps, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        latency_inner_reps=inner_reps, latency_warmup=warmup, latency_winsor=winsor,
    )
    es = generate_eval_samples(env, ek, ndp)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, es)
    return env, es


def measure(env, es, order, K):
    specs = np.full((len(order), MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    vals = []
    for _ in range(K):
        _, _, r = _callback(
            env.config, env.args, env.consts,
            jax.numpy.asarray(order, dtype=jax.numpy.int32),
            jax.numpy.asarray(specs), len(order), *es,
        )
        vals.append(-float(np.asarray(r)[LI]))  # latency_ns (positive)
    return np.array(vals)


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    env0, es0 = build_env(inner_reps=1, warmup=0, winsor=0.0)
    order = list(env0.valid_vertices)  # fixed order, no sparsification
    print(f"order={order}  (no rules)  K={K} repeats of the full reward call\n")

    for label, (ir, wu, ws) in [
        ("OLD  (inner=1, P60)", (1, 0, 0.0)),
        ("NEW  (inner=8, winsor=0.2, warmup=3)", (8, 3, 0.2)),
    ]:
        env, es = build_env(inner_reps=ir, warmup=wu, winsor=ws)
        v = measure(env, es, order, K)
        cv = v.std() / v.mean() if v.mean() else float("nan")
        print(f"{label}")
        print(f"   aggregated latency reward (ns): median={np.median(v):.4g} "
              f"mean={v.mean():.4g}  min={v.min():.4g} max={v.max():.4g}")
        print(f"   reproducibility CV over {K} repeats = {cv:.4f}  "
              f"(max/min = {v.max()/v.min():.2f})\n")


if __name__ == "__main__":
    main()
