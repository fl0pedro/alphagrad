"""alphagrad.elimrl.encoder_bench -- M2 acceptance gate.

Walks a full REVERSE episode (rev_policy face actions) and a RANDOM episode
on the TransformerLM seq32/dm128/vocab1024 target, timing every step
end-to-end (feature extraction + GNN forward + candidate embeddings), for
both the full-recompute and the incremental encoder paths, and reports
median / p95 / max per-step ms plus bucket-compile counts.

GATE: incremental-path median per-step end-to-end < 50 ms on the TLM target.

The episode is walked on a SYMBOLIC env (graph dynamics only, no coefficient
math) -- that is the regime the RL rollout runs in; the encoder cost is
identical either way. Env-side costs (state()/apply()) are reported
separately and are NOT part of the encoder gate.

Run (pgi15-cpu2):
    JAX_PLATFORMS=cpu uv run --no-sync python -m alphagrad.elimrl.encoder_bench
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def _percentiles(xs):
    xs = np.asarray(xs, np.float64)
    if not len(xs):
        return {"n": 0}
    return {"n": int(len(xs)), "median": float(np.median(xs)),
            "p95": float(np.percentile(xs, 95)), "max": float(xs.max()),
            "mean": float(xs.mean())}


def run_episode(env, rt_full, rt_inc, policy, rng, check_every=25,
                max_steps=20000):
    """Walk one episode; returns per-step timing lists + env-side costs."""
    import jax.numpy as jnp
    from alphagrad.elimrl.env import rev_policy

    rows = {"full": [], "inc": [], "feat_full": [], "feat_inc": [],
            "fwd_full": [], "fwd_inc": [], "cand_full": [], "cand_inc": [],
            "env_state": [], "env_apply": [], "compile_steps": 0,
            "checks": 0, "max_check_diff": 0.0}
    rt_inc.reset()
    steps = 0
    while steps < max_steps:
        t0 = time.perf_counter()
        st = env.state()
        rows["env_state"].append((time.perf_counter() - t0) * 1e3)
        if st.done:
            break
        out_f = rt_full.encode_full(st)
        out_i = rt_inc.encode_incremental(st)
        compiled = out_f["new_compiles"] + out_i["new_compiles"]
        if compiled:
            rows["compile_steps"] += 1
        else:
            rows["full"].append(out_f["timings"]["total_ms"])
            rows["inc"].append(out_i["timings"]["total_ms"])
            rows["feat_full"].append(out_f["timings"]["feat_ms"])
            rows["feat_inc"].append(out_i["timings"]["feat_ms"])
            rows["fwd_full"].append(out_f["timings"]["forward_ms"])
            rows["fwd_inc"].append(out_i["timings"]["forward_ms"])
            rows["cand_full"].append(out_f["timings"]["cand_ms"])
            rows["cand_inc"].append(out_i["timings"]["cand_ms"])
        if steps % check_every == 0:      # incremental == full spot check
            n = out_f["n_rows"]
            d = float(jnp.max(jnp.abs(out_f["h"][:n] - out_i["h"][:n])))
            nf = out_f["n_faces"]
            if nf:
                d = max(d, float(jnp.max(jnp.abs(
                    out_f["face_emb"][:nf] - out_i["face_emb"][:nf]))))
            rows["checks"] += 1
            rows["max_check_diff"] = max(rows["max_check_diff"], d)
        if policy == "rev":
            action = rev_policy(env)
        else:
            faces = env.faces()
            action = ("F",) + faces[rng.integers(len(faces))]
        t0 = time.perf_counter()
        env.apply(action)
        rows["env_apply"].append((time.perf_counter() - t0) * 1e3)
        steps += 1
    rows["steps"] = steps
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="tlm", choices=("tlm", "tiny"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--gate-ms", type=float, default=50.0)
    a = p.parse_args(argv)

    from alphagrad.elimrl.baselines import tiny_target, tlm_target
    from alphagrad.elimrl.env import ElimEnv
    from alphagrad.elimrl.features import build_static
    from alphagrad.elimrl.encoder import EncoderRuntime, make_model

    fn, args, argnums = (tlm_target() if a.target == "tlm" else tiny_target())
    t0 = time.perf_counter()
    env = ElimEnv(fn, args, argnums=argnums, symbolic=True)
    static = build_static(env)
    model = make_model(static, seed=a.seed, hidden=a.hidden)
    rt_full = EncoderRuntime(model, static)
    rt_inc = EncoderRuntime(model, static)
    print(f"[bench] target={a.target} rows={static.n_rows} "
          f"(invars={static.n_invars} eqns={static.n_eqns}) "
          f"prims={len(static.prim_vocab)} setup={time.perf_counter()-t0:.1f}s",
          flush=True)

    rng = np.random.default_rng(a.seed)
    summary = {}
    for name, policy in (("reverse", "rev"), ("random", "rand")):
        t0 = time.perf_counter()
        rows = run_episode(env, rt_full, rt_inc, policy, rng)
        wall = time.perf_counter() - t0
        summary[name] = {
            "steps": rows["steps"],
            "compile_steps_excluded": rows["compile_steps"],
            "full_ms": _percentiles(rows["full"]),
            "inc_ms": _percentiles(rows["inc"]),
            "feat_full_ms": _percentiles(rows["feat_full"]),
            "feat_inc_ms": _percentiles(rows["feat_inc"]),
            "fwd_full_ms": _percentiles(rows["fwd_full"]),
            "fwd_inc_ms": _percentiles(rows["fwd_inc"]),
            "cand_full_ms": _percentiles(rows["cand_full"]),
            "cand_inc_ms": _percentiles(rows["cand_inc"]),
            "env_state_ms": _percentiles(rows["env_state"]),
            "env_apply_ms": _percentiles(rows["env_apply"]),
            "check_max_abs_diff_inc_vs_full": rows["max_check_diff"],
            "checks": rows["checks"],
        }
        s = summary[name]
        print(f"\n[{name}] steps={rows['steps']} wall={wall:.1f}s "
              f"compile-steps excluded={rows['compile_steps']}", flush=True)
        for k in ("full_ms", "inc_ms", "feat_inc_ms", "fwd_inc_ms",
                  "cand_inc_ms", "env_state_ms", "env_apply_ms"):
            v = s[k]
            if v.get("n"):
                print(f"  {k:14s} median={v['median']:8.2f} "
                      f"p95={v['p95']:8.2f} max={v['max']:8.2f}", flush=True)
        print(f"  inc==full max|diff| {s['check_max_abs_diff_inc_vs_full']:.2e} "
              f"over {s['checks']} checks", flush=True)
        env.reset()

    summary["compile_signatures"] = {"full_path": rt_full.compile_signatures,
                                     "inc_path": rt_inc.compile_signatures}
    med = summary["reverse"]["inc_ms"].get("median", float("inf"))
    med_r = summary["random"]["inc_ms"].get("median", float("inf"))
    gate = med < a.gate_ms and med_r < a.gate_ms
    summary["gate_ms"] = a.gate_ms
    summary["gate_pass"] = bool(gate)
    print(f"\ncompile signatures: full-path runtime="
          f"{rt_full.compile_signatures} inc-path runtime="
          f"{rt_inc.compile_signatures}", flush=True)
    print(f"GATE (<{a.gate_ms:.0f}ms incremental median, end-to-end): "
          f"reverse={med:.2f}ms random={med_r:.2f}ms -> "
          f"{'PASS' if gate else 'FAIL'}", flush=True)
    print("\nELIMRL_M2_BENCH " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
