"""Replay a saved Pareto-front solution: rebuild the elimination order +
per-vertex sparsification rules, apply them through the SAME env path the
trainer uses, then dump the sparsified jaxpr, the compiled HLO, and the
measured latency. Used to sanity-check Pareto points (e.g. an alleged
"zero-latency" solution).

Run on a pgi15 CPU node:
    JAX_PLATFORMS=cpu uv run --no-sync \
      src/alphagrad/approx/verify_pareto_solution.py <pareto.json> [latency|frob]
"""
import os, sys, json

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import equinox as eqx

import alphagrad.approx.env as envmod
from alphagrad.approx.env import (
    VertexEliminationEnv, REWARD_NAMES, REWARD_INDEX, _callback,
)
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.order_specs import build_order_specs
import alphagrad.approx.common.compile_cache as ccmod


# ---------------------------------------------------------------- env (matches trainer)
def build_env():
    key = jax.random.PRNGKey(0)
    args_key, eval_key = jax.random.split(key)
    target_fn = get_fn("VmappedNeuralNetwork")
    xs = get_args("VmappedNeuralNetwork", args_key, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(target_fn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=infer_argnums("VmappedNeuralNetwork"),
        num_envs=0, data_gen=gen, target_fun=target_fn,
        cmp_type="latency", mem_type="peak_memory", measure_latency=True,
        latency_samples=1, num_data_points=5, reps_per_point=4,
        percentile_keep=0.60, slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0,
    )
    eval_samples = generate_eval_samples(env, eval_key, 5)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
    return env, eval_samples


# ---------------------------------------------------------------- replay one point
def replay(env, eval_samples, order, specs, label):
    captured = {}
    orig_jacve = envmod.jacve
    orig_cc = ccmod.cached_compile

    def jacve_cap(*a, **k):
        fn = orig_jacve(*a, **k)
        captured["fn"] = fn
        return fn

    def cc_cap(key, fn):
        out = orig_cc(key, fn)
        if isinstance(key, (bytes, bytearray)) and key.startswith(b"approx:"):
            captured["compiled"] = out
        return out

    envmod.jacve = jacve_cap
    ccmod.cached_compile = cc_cap
    rs = {}
    try:
        _, _, reward = _callback(
            env.config, env.args, env.consts,
            jax.numpy.asarray(order), jax.numpy.asarray(specs), len(order),
            *eval_samples, raw_sink=rs,
        )
    finally:
        envmod.jacve = orig_jacve
        ccmod.cached_compile = orig_cc

    reward = np.asarray(reward)
    print(f"\n{'='*70}\n  {label}\n{'='*70}")
    li = REWARD_INDEX["latency_ns"]
    lat_samples = rs.get("latency_ns_samples", [])
    print(f"  latency reward (env)      : {reward[li]:.6g}  "
          f"(=> latency_ns = {-reward[li]:.6g} ns = {-reward[li]/1e6:.4f} ms)")
    if lat_samples:
        a = np.array(lat_samples)
        print(f"  raw latency samples (ns)  : n={len(a)} min={a.min():.4g} "
              f"P60={np.percentile(a,60):.4g} max={a.max():.4g} "
              f"#zeros={(a==0).sum()}")
    print(f"  frob_residual reward      : {reward[REWARD_INDEX['frob_residual']]:.6g}")
    print(f"  cosine_sim reward         : {reward[REWARD_INDEX['cosine_sim']]:.6g}")
    print(f"  peak_memory reward        : {reward[REWARD_INDEX['peak_memory']]:.6g}")

    fn = captured.get("fn")
    if fn is not None:
        try:
            jaxpr = jax.make_jaxpr(fn)(*env.args)
            eqns = jaxpr.jaxpr.eqns
            prims = {}
            dtypes = set()
            for e in eqns:
                prims[e.primitive.name] = prims.get(e.primitive.name, 0) + 1
                for ov in e.outvars:
                    if hasattr(ov, "aval"):
                        dtypes.add(str(ov.aval.dtype))
            print(f"  --- sparsified jaxpr: {len(eqns)} eqns ---")
            top = sorted(prims.items(), key=lambda x: -x[1])[:12]
            print("    primitives:", ", ".join(f"{k}×{v}" for k, v in top))
            print("    dtypes    :", sorted(dtypes))
            conv = [e for e in eqns if e.primitive.name == "convert_element_type"]
            f8 = [d for d in dtypes if "float8" in d or "e5m2" in d or "e4m3" in d]
            print(f"    convert_element_type count: {len(conv)}  | float8 dtypes present: {f8}")
        except Exception as exc:
            print("  jaxpr build failed:", repr(exc))

    comp = captured.get("compiled")
    if comp is not None:
        try:
            hlo = comp.as_text()
            lines = hlo.splitlines()
            print(f"  --- compiled HLO: {len(lines)} lines, {len(hlo)} chars ---")
            fused = sum(1 for ln in lines if "fusion" in ln)
            convs = sum(ln.count("convert(") for ln in lines)
            print(f"    HLO fusion lines: {fused}  convert() occurrences: {convs}")
            try:
                ca = comp.cost_analysis()
                if isinstance(ca, list):
                    ca = ca[0] if ca else {}
                flops = ca.get("flops") if isinstance(ca, dict) else None
                print(f"    cost_analysis flops: {flops}")
            except Exception:
                pass
            head = "\n".join(lines[:18])
            print("    HLO head:\n" + "\n".join("      " + l for l in head.splitlines()))
        except Exception as exc:
            print("  HLO dump failed:", repr(exc))
    return reward


def main():
    path = sys.argv[1]
    pref = sys.argv[2] if len(sys.argv) > 2 else "latency"
    d = json.load(open(path))
    print(f"loaded {path}: {d['num_points']} points, objectives={d['objectives']}")
    front = d["front"]
    # obj values are REWARDS (= -cost, higher is better), so the best point on
    # any objective is the MAX of its reward — including frob_residual (max
    # reward = min residual = best quality).
    key = "latency_ns" if pref == "latency" else "frob_residual"
    pt = max(front, key=lambda p: p["obj"][key])
    print("selected point obj:", {k: round(v, 6) for k, v in pt["obj"].items()})

    env, eval_samples = build_env()
    print(f"env built: num_valid={len(env.valid_vertices)} "
          f"valid_vertices={list(env.valid_vertices)}")

    order, specs, n_rules = build_order_specs(pt["seq"], env)
    print(f"reconstructed order={order.tolist()} | total non-END rules={n_rules}")
    replay(env, eval_samples, order, specs,
           f"REPLAY selected point ({n_rules} sparsification rules)")

    # Baseline: same order, NO sparsification (empty specs) — isolates the
    # effect of the rules vs the bare elimination order.
    empty = np.full_like(specs, -1)
    empty[:, :, 2] = 0
    replay(env, eval_samples, order, empty,
           "BASELINE same order, ZERO rules (pure vertex elimination)")


if __name__ == "__main__":
    main()
