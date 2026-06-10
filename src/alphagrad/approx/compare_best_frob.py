"""Compare the best-frob (highest-quality) learned solution from each Pareto
front against the four standard Jacobian baselines — jax.jacfwd, jax.jacrev,
graphax forward-mode, graphax reverse-mode — on REAL MNIST data.

The existing compare_baselines.py replays from best_sequences.json via a parser
built for the OLD wire format (flat 6-tuples / numeric DIAG triples); the new
micro-action sequences are ``[vertex, ["diag(...)","quant('...')",...]]`` (call
strings), which that parser mis-reads. So we reconstruct the learned fn with the
SAME new-format path used in verify_pareto_solution.py (parse → order +
rule_specs → env-built jacve), and measure every row identically.

Usage (single-core for clean latency):
  numactl --physcpubind=+0 --membind=+0 uv run --no-sync \
    python .../compare_best_frob.py <cmorl_pareto.json> <mogfn_pareto.json>
"""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from graphax import jacve

import alphagrad.approx.env as envmod
from alphagrad.approx.env import _callback
import alphagrad.approx.common.compile_cache as ccmod
# Package-qualified import — `from verify_pareto_solution import ...` only
# resolves when sys.path[0] happens to be this directory (direct `python
# path/to/script.py` invocation) and breaks under `python -m` or imports.
from alphagrad.approx.verify_pareto_solution import build_env, build_order_specs


def _flat(j):
    leaves = jax.tree_util.tree_leaves(j)
    return np.concatenate([np.asarray(l).reshape(-1) for l in leaves]).astype(np.float64)


def _cossim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _frob(exact, approx):
    n = np.linalg.norm(exact)
    return float(np.linalg.norm(exact - approx) / n) if n > 0 else float("nan")


def _measure(fn, args, inner=3, warmup=1, nrep=3):
    compiled = jax.jit(fn).lower(*args).compile()
    cost = compiled.cost_analysis() or {}
    flops = float(cost.get("flops", 0.0))
    bytes_a = float(cost.get("bytes accessed", 0.0))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    samples = []
    out = None
    for _ in range(nrep):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = compiled(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) / inner)
    a = np.sort(np.array(samples))
    k = int(len(a) * 0.2)
    lat_ns = float((a[k:len(a) - k] if len(a) - 2 * k >= 1 else a).mean()) * 1e9
    cv = float(a.std() / a.mean()) if a.mean() else float("nan")
    return {"flops": flops, "bytes": bytes_a, "lat_ns": lat_ns, "cv": cv,
            "jac": _flat(out)}


def _build_learned(env, es, seq):
    order, specs, n_rules = build_order_specs(seq, env)
    captured = {}
    oj = envmod.jacve
    def cap(*a, **k):
        fn = oj(*a, **k)
        captured["fn"] = fn
        return fn
    envmod.jacve = cap
    try:
        _callback(env.config, env.args, env.consts,
                  jnp.asarray(order, jnp.int32), jnp.asarray(specs), len(order), *es)
    finally:
        envmod.jacve = oj
    return captured["fn"], n_rules


def _best_frob_seq(path):
    d = json.load(open(path))
    # obj['frob_residual'] is the REWARD (= -residual); max reward = min residual.
    p = max(d["front"], key=lambda p: p["obj"]["frob_residual"])
    return p["seq"], p["obj"]


def main():
    cmorl_json = sys.argv[1]
    mogfn_json = sys.argv[2]
    env, es = build_env()
    target_fn = env.config.target_fun
    argnums = tuple(env.config.argnums)
    args = env.args

    rows = []
    # Exact reference (jax.jacrev) for cossim / frob.
    ref = _measure(jax.jacrev(target_fn, argnums=argnums), args)
    ref_jac = ref["jac"]

    def add(label, m, extra=""):
        rows.append((label, m["flops"], m["bytes"], m["lat_ns"], m["cv"],
                     _cossim(ref_jac, m["jac"]), _frob(ref_jac, m["jac"]), extra))

    add("jax_jacrev (reference)", ref)
    add("jax_jacfwd", _measure(jax.jacfwd(target_fn, argnums=argnums), args))
    add("graphax_fwd", _measure(jacve(target_fn, order="fwd", argnums=argnums), args))
    add("graphax_rev", _measure(jacve(target_fn, order="rev", argnums=argnums), args))

    # C-MORL best-frob (single reference row).
    try:
        seq, obj = _best_frob_seq(cmorl_json)
        fn, nr = _build_learned(env, es, seq)
        add("C-MORL best-frob", _measure(fn, args),
            extra=f"{nr} rules; rec frob={-obj['frob_residual']:.4g}")
    except Exception as e:
        print(f"  ! C-MORL best-frob failed: {type(e).__name__}: {e}")

    # ALL MOGFN front sequences (best-quality first).
    d = json.load(open(mogfn_json))
    pts = sorted(d["front"], key=lambda p: p["obj"]["frob_residual"], reverse=True)
    for idx, p in enumerate(pts):
        o = p["obj"]
        try:
            fn, nr = _build_learned(env, es, p["seq"])
            add(f"MOGFN[{idx}]", _measure(fn, args),
                extra=f"{nr} rules; rec frob={-o['frob_residual']:.4g} "
                      f"lat={-o['latency_ns']/1e6:.1f}ms")
        except Exception as e:
            print(f"  ! MOGFN[{idx}] failed: {type(e).__name__}: {e}")

    # Table.
    print(f"\n{'source':<26s} {'flops':>11s} {'bytes':>11s} {'latency_ms':>11s} "
          f"{'CV':>6s} {'cossim':>9s} {'frob':>9s}  notes")
    print("-" * 110)
    base_lat = next(r[3] for r in rows if r[0].startswith("jax_jacrev"))
    for (lbl, fl, by, lat, cv, cs, fr, ex) in rows:
        spd = base_lat / lat if lat else float("nan")
        print(f"{lbl:<26s} {fl:>11.4g} {by:>11.4g} {lat/1e6:>11.4f} {cv:>6.3f} "
              f"{cs:>9.5f} {fr:>9.4g}  {ex}  ({spd:.2f}x vs jacrev)")


if __name__ == "__main__":
    main()
