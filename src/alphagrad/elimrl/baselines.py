"""alphagrad.elimrl.baselines -- M0 baselines, ALL measured through the same
isolated worker: graphax jacve (rev/fwd), the face engine driven in reverse
("jacfe rev"), jax.grad and jax.jacrev.

Benchmark-only module: the TransformerLM/wikitext2 target builder imports
``alphagrad.approx.common.examples`` -- allowed in benchmarks/tests ONLY; the
env core (env.py) never touches it. Target builders are resolved by
"module:attr" strings INSIDE the worker subprocess, so this parent process
never imports jax (and never touches the GPU).

GPU acceptance gate (task #114 M0), on pgi15-gpu16 via sbatch:

    uv run --no-sync python -m alphagrad.elimrl.baselines --demo

which (1) measures the five baselines on TransformerLM seq32/dm128/vocab1024
(v53 reverse reference 149.5us +-3%), with jacfe-rev numerically cross-checked
against jacve-rev in-worker; (2) scores a deliberately huge plan infeasible
via the predicted-memory gate WITHOUT executing it and proves the run
continues; (3) crashes the worker on purpose and proves respawn + a correct
subsequent measurement.
"""

from __future__ import annotations

import argparse
import json
import os


# ---------------------------------------------------------------------------
# target builders (executed inside the worker subprocess)
# ---------------------------------------------------------------------------
def tiny_target():
    """Small MLP + softmax MSE loss -- CPU-fast worker/e2e tests."""
    import jax
    import jax.numpy as jnp
    import jax.nn as jnn

    def fn(x, y, W1, b1, W2, b2):
        return jnp.sum((jnn.softmax(W2 @ jnn.relu(W1 @ x + b1) + b2) - y) ** 2)

    k = jax.random.split(jax.random.PRNGKey(0), 3)
    args = [jax.random.normal(k[0], (12,)), jnn.one_hot(2, 5),
            jax.random.normal(k[1], (10, 12)) / 5, jnp.zeros(10),
            jax.random.normal(k[2], (5, 10)) / 5, jnp.zeros(5)]
    return fn, args, (2, 3, 4, 5)


def huge_target(n: int = 100_000):
    """Deliberately infeasible plan: jax.jacrev of an elementwise map is an
    (n, n) dense Jacobian -- n=100k => ~40 GB f32 output, so memory_analysis
    trips any sane budget BEFORE execution (synthetic-allocation variant of
    the acceptance gate's 'deliberately huge plan')."""
    import jax.numpy as jnp

    def fn(x):
        return jnp.tanh(x)

    return fn, [jnp.ones((n,), jnp.float32)], (0,)


def tlm_target(seq: int = 32, dmodel: int = 128, vocab: int = 1024,
               seed: int = 0):
    """TransformerLM / wikitext2 (the v53 measurement target)."""
    os.environ["ALPHAGRAD_TLM_SEQ"] = str(seq)
    os.environ["ALPHAGRAD_TLM_DMODEL"] = str(dmodel)
    os.environ["ALPHAGRAD_TLM_VOCAB"] = str(vocab)
    import jax
    from alphagrad.approx.common.examples import (
        get_fn, get_args, infer_argnums, scalar_loss_fn)
    fn = scalar_loss_fn(get_fn("TransformerLM"))
    argnums = infer_argnums("TransformerLM")
    args = get_args("TransformerLM", jax.random.PRNGKey(seed),
                    dataset="wikitext2")
    return fn, list(args), tuple(argnums)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
_SPECS = {
    "jacve_rev": dict(method="jacve", order="rev"),
    "jacve_fwd": dict(method="jacve", order="fwd"),
    "jacfe_rev": dict(method="jacfe", order="rev",
                      check_against={"method": "jacve", "order": "rev"}),
    "jax.grad": dict(method="jax.grad"),
    "jax.jacrev": dict(method="jax.jacrev"),
}


def _row(name: str, res: dict):
    lat = res.get("latency_ns")
    lat_s = f"{lat / 1e3:10.1f}us" if lat else " " * 12
    mem = res.get("mem_total_bytes")
    mem_s = f"{mem / 2**20:9.1f}MB" if mem is not None else " " * 11
    extra = ""
    if "check_maxdiff" in res:
        extra += f"  check_maxdiff={res['check_maxdiff']:.3e} cos={res.get('check_cos', 0):.6f}"
    if res.get("reason"):
        extra += f"  reason={res['reason']}"
    if res.get("peak_delta_bytes") is not None:
        extra += f"  peak_delta={res['peak_delta_bytes'] / 2**20:.1f}MB"
    print(f"  {name:28s} {res.get('status', '?'):10s} {lat_s} {mem_s}"
          f"  compile={res.get('compile_s', 0):7.1f}s{extra}", flush=True)


def _slim(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "latency_samples_ns"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="tlm", choices=("tlm", "tiny"))
    p.add_argument("--methods",
                   default="jacve_rev,jacfe_rev,jax.grad,jax.jacrev,jacve_fwd")
    p.add_argument("--budget-gb", type=float, default=24.0,
                   help="predicted-memory budget for the baseline runs")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--timeout", type=float, default=1500.0,
                   help="hard per-measurement timeout (kill + respawn)")
    p.add_argument("--demo", action="store_true",
                   help="run the infeasible-plan + crash-survival demos")
    a = p.parse_args(argv)

    from alphagrad.elimrl.measure_worker import MeasureClient

    builder = "tlm_target" if a.target == "tlm" else "tiny_target"
    target = {"builder": f"alphagrad.elimrl.baselines:{builder}"}
    budget = a.budget_gb * 2 ** 30
    common = dict(target=target, budget_bytes=budget,
                  reps=a.reps, inner=a.inner, timeout=a.timeout)

    client = MeasureClient()
    print(f"[baselines] target={a.target} worker backend={client.backend} "
          f"budget={a.budget_gb}GiB reps={a.reps} inner={a.inner}", flush=True)

    results = {}
    for name in [m.strip() for m in a.methods.split(",") if m.strip()]:
        results[name] = client.measure(**{**common, **_SPECS[name]})
        _row(name, results[name])

    if a.demo:
        print("\n[demo] deliberately huge plan -> predicted-memory gate "
              "(budget 8GiB, ~40GB predicted):", flush=True)
        res = client.measure(
            target={"builder": "alphagrad.elimrl.baselines:huge_target"},
            method="jax.jacrev", budget_bytes=8 * 2 ** 30,
            reps=2, inner=2, timeout=a.timeout)
        _row("huge_jacrev(40GB, 8GiB cap)", res)
        results["demo_infeasible"] = res
        assert res.get("status") == "infeasible" and not res.get("executed"), (
            "infeasible-plan demo FAILED", res)

        res2 = client.measure(**{**common, **_SPECS["jacve_rev"]})
        _row("jacve_rev(after-infeasible)", res2)
        results["after_infeasible"] = res2
        assert res2.get("status") == "ok", ("run did not continue", res2)

        print("\n[demo] hard worker crash -> respawn:", flush=True)
        r = client.request({"cmd": "crash"})
        print(f"  crash scored: {r}", flush=True)
        results["crash_score"] = r
        assert r.get("status") == "infeasible" and r.get("reason") == "worker_died", r

        res3 = client.measure(**{**common, **_SPECS["jacve_rev"]})
        _row("jacve_rev(after-crash)", res3)
        results["after_crash"] = res3
        assert res3.get("status") == "ok", ("respawn measurement FAILED", res3)
        print(f"  [demo] worker respawns so far: {client.respawns}", flush=True)

    print("\nELIMRL_GATE_SUMMARY "
          + json.dumps({k: _slim(v) for k, v in results.items()}), flush=True)
    client.close()
    return results


if __name__ == "__main__":
    main()
