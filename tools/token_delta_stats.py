"""Per-elimination token DELTA statistics — how big is one palimpsa call?

The append-only path feeds palimpsa one delta per elimination instead of the
whole stream. The batched kernel needs a STATIC buffer, so the question this
answers is: how many tokens does one elimination actually emit, and what is
the worst case across elimination orders?

Delta size depends on the order (eliminating a high-degree vertex early emits
more paths), so a single order tells you the average but not the ceiling.
This samples several orders and reports the max over all of them.

Usage:
    python -m alphagrad.tools.token_delta_stats --example VmappedNeuralNetwork \
        --dataset mnist --hidden-dim 256 --orders 24
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def _percentiles(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "mean": round(float(a.mean()), 1),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--example", default="VmappedNeuralNetwork")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--orders", type=int, default=24,
                   help="random elimination orders to sample (plus identity)")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import jax
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.common.examples import (
        get_fn, get_args, infer_argnums)

    key = jax.random.PRNGKey(a.seed)
    fn = get_fn(a.example)
    xs = get_args(a.example, key, dataset=a.dataset)
    argnums = infer_argnums(a.example)
    closed = jax.make_jaxpr(fn)(*xs)
    n_eqns = len(closed.jaxpr.eqns)
    print(f"[stats] example={a.example} eqns={n_eqns} argnums={argnums}",
          flush=True)

    def _run(order):
        tk = IncrementalPathTokenizer(closed.jaxpr, tuple(argnums),
                                      list(closed.literals), list(xs),
                                      vocab_size=a.vocab_size)
        base = len(list(tk.base_tokens()))
        deltas = []
        for v in order:
            try:
                deltas.append(len(list(tk.eliminate(int(v)))))
            except Exception as e:            # illegal for this prefix
                print(f"  [skip] vertex {v}: {type(e).__name__}", flush=True)
                break
        return base, deltas

    rng = np.random.default_rng(a.seed)
    base_len = None
    all_deltas: list[int] = []
    per_order_totals: list[int] = []
    per_order_max: list[int] = []

    orders = [list(range(1, n_eqns + 1))]
    for _ in range(a.orders):
        o = list(range(1, n_eqns + 1))
        rng.shuffle(o)
        orders.append(o)

    for oi, order in enumerate(orders):
        base, deltas = _run(order)
        if not deltas:
            continue
        base_len = base
        all_deltas.extend(deltas)
        per_order_totals.append(base + sum(deltas))
        per_order_max.append(max(deltas))
        if oi == 0:
            print(f"[stats] identity order: base={base} "
                  f"steps={len(deltas)} total={base + sum(deltas)}", flush=True)

    out = {
        "example": a.example,
        "n_eqns": n_eqns,
        "base_tokens": base_len,
        "delta": _percentiles(all_deltas),
        "orders_sampled": len(per_order_totals),
        "worst_delta_over_all_orders": int(max(per_order_max)),
        "total_stream": _percentiles(per_order_totals),
    }
    print("\n=== TOKEN DELTA STATS ===")
    print(json.dumps(out, indent=2))

    d, t = out["delta"], out["total_stream"]
    print("\n--- sizing ---")
    print(f"one palimpsa call (delta): mean={d['mean']} p99={d['p99']} "
          f"max={d['max']} (worst over all orders={out['worst_delta_over_all_orders']})")
    print(f"ALPHAGRAD_MAX_DELTA_TOKENS >= {out['worst_delta_over_all_orders']} "
          f"-> suggest {1 << int(np.ceil(np.log2(max(out['worst_delta_over_all_orders'], 1))))}")
    print(f"full stream (sizes the positional-encoding table): "
          f"max={t['max']} -> ALPHAGRAD_MAX_TOKENS >= {t['max']}, "
          f"suggest {1 << int(np.ceil(np.log2(max(t['max'], 1))))}")


if __name__ == "__main__":
    main()
