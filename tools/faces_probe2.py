#!/usr/bin/env python3
"""Why do some vertices have zero faces, and how many faces can a vertex have?

Three questions:
 1. Zero-face vertices -- is it no PREDECESSORS or no SUCCESSORS in the
    Jacobian graph, and does eliminating one actually change anything?
 2. What is the true max face count, over several orders -- MAX_FACES must
    cover it or the head silently never sees the tail faces.
 3. Emit the flat unrolling string.
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jrand


def build_simple():
    def f(x1, x2):
        y = x1 * x2
        s = jnp.sin(y)
        return jnp.log(s), s - y
    xs = (jnp.array(1.3), jnp.array(0.7))
    cj = jax.make_jaxpr(f)(*xs)
    return "simple", cj.jaxpr, (0, 1), cj.literals, xs


def build_nn():
    sys.path.insert(0, os.path.expanduser("~/dsnn/alphagrad/src"))
    from alphagrad.approx.common.examples import (
        get_fn, get_args, infer_argnums)
    name = "VmappedNeuralNetwork"
    fn = get_fn(name)
    xs = get_args(name, jrand.PRNGKey(0), dataset="mnist")
    cj = jax.make_jaxpr(fn)(*xs)
    return "nn256-xent", cj.jaxpr, infer_argnums(name), cj.literals, xs


def valid_of(jaxpr):
    outvars = {str(v) for v in jaxpr.outvars}
    out_ids = {i for i, e in enumerate(jaxpr.eqns, start=1)
               if any(str(ov) in outvars for ov in e.outvars)}
    return [i for i in range(1, len(jaxpr.eqns) + 1) if i not in out_ids]


def new_ij(jaxpr, argnums, consts, args):
    from graphax.incremental import IncrementalJaxpr
    return IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                            track_faces=True)


def degrees(ij, v):
    """(#predecessors, #successors) of v in the LIVE Jacobian graph."""
    succ = len(ij.graph.get(v, {}) or {})
    pred = len(ij.tgraph.get(v, {}) or {})
    return pred, succ


def q1(name, jaxpr, argnums, consts, args, order):
    print(f"\n=== {name}: why zero faces ===")
    ij = new_ij(jaxpr, argnums, consts, args)
    for v in order:
        n = len(list(ij.faces(int(v))))
        if n == 0:
            p, s = degrees(ij, int(v))
            prim = jaxpr.eqns[int(v) - 1].primitive.name
            n_eq0 = len(ij.trace.frame.tracing_eqns)
            g0 = {k: dict(x) for k, x in ij.graph.items()}
            ij.eliminate(int(v))
            n_eq1 = len(ij.trace.frame.tracing_eqns)
            changed = (ij.graph != g0)
            print(f"  v{v:<3} {prim:<16} preds={p} succs={s}  "
                  f"eqns_emitted={n_eq1 - n_eq0}  graph_changed={changed}")
        else:
            ij.eliminate(int(v))


def q2(name, jaxpr, argnums, consts, args, order):
    print(f"\n=== {name}: max faces over several orders ===")
    import random
    orders = {
        "ascending": list(order),
        "descending": list(reversed(order)),
    }
    for seed in range(int(os.environ.get("NORD", "5"))):
        r = random.Random(seed)
        o = list(order)
        r.shuffle(o)
        orders[f"random{seed}"] = o
    overall = 0
    for tag, o in orders.items():
        ij = new_ij(jaxpr, argnums, consts, args)
        counts = []
        for v in o:
            try:
                n = len(list(ij.faces(int(v))))
            except Exception:
                counts.append(-1)
                break
            counts.append(n)
            try:
                ij.eliminate(int(v))
            except Exception:
                break
        mx = max(counts) if counts else 0
        overall = max(overall, mx)
        if tag in ("ascending", "descending") or mx >= overall:
            print(f"  {tag:<11} steps={len(counts):<3} "
                  f"total={sum(c for c in counts if c>0):<5} max={mx}")
    print(f"  -> MAX_FACES must be >= {overall}")
    return overall


def flat(name, jaxpr, argnums, consts, args, order, skip_zero=True):
    ij = new_ij(jaxpr, argnums, consts, args)
    parts = ["base"]
    kept, skipped = 0, 0
    for k, v in enumerate(order, start=1):
        n = len(list(ij.faces(int(v))))
        if n == 0 and skip_zero:
            skipped += 1
            ij.eliminate(int(v))
            continue
        kept += 1
        i = kept if os.environ.get("STEP_IDX", "1") == "1" else int(v)
        parts.append(f"=> palimpsa -> VE head")
        if n == 0:
            ij.eliminate(int(v))
            continue
        parts.append(f"=> face {i},1 accumulation")
        for j in range(1, n + 1):
            parts.append("=> palimpsa -> approx head")
            if j < n:
                parts.append(
                    f"=> {i},{j} approximation & face {i},{j+1} accumulation")
            else:
                parts.append(f"=> {i},{j} approximation")
        ij.eliminate(int(v))
    parts.append("=> reward")
    line = " ".join(parts)
    print(f"\n=== {name}: FLAT ({kept} vertices kept, {skipped} zero-face "
          f"skipped, {len(line)} chars) ===")
    print(line)


if __name__ == "__main__":
    which = build_nn if "--nn" in sys.argv else build_simple
    name, jaxpr, argnums, consts, args = which()
    order = valid_of(jaxpr)
    q1(name, jaxpr, argnums, consts, args, order)
    q2(name, jaxpr, argnums, consts, args, order)
    flat(name, jaxpr, argnums, consts, args, order)
