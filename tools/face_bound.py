#!/usr/bin/env python3
"""The exact per-graph face bound, so no cap is ever needed.

A vertex's face count is |predecessors| x |successors| in the LIVE Jacobian
graph. Elimination only ever replaces a neighbour by ITS neighbours, so after
any sequence v's predecessors are a subset of v's ANCESTORS in the original
pruned graph and its successors a subset of its DESCENDANTS. Therefore

    faces(v) <= |ancestors(v)| * |descendants(v)|      for every order,

and max over v is a bound no elimination order can exceed. That makes the
static trip count a DERIVED property of the graph rather than a knob that
silently truncates.

Reported next to the max actually reached over sampled orders, because the
bound is loose (it ignores that an ancestor and a descendant cannot both be
live once the path between them is contracted) and the gap decides whether
the bound is usable as the array shape.
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


def bound(name, jaxpr, argnums, consts, args):
    from graphax.incremental import IncrementalJaxpr
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=True)
    g = {k: set(v.keys()) for k, v in ij.graph.items()}
    tg = {k: set(v.keys()) for k, v in ij.tgraph.items()}
    nodes = set(g) | set(tg)
    for d in (g, tg):
        for n in nodes:
            d.setdefault(n, set())

    def closure(adj, start):
        seen, stack = set(), [start]
        while stack:
            u = stack.pop()
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen

    print(f"\n=== {name}: exact face bound ===")
    # Graph nodes are the jaxpr VARS. Map each eliminable eqn (1-based index,
    # the id space the policy uses) to its outvar and take closures there.
    outvars = {str(x) for x in jaxpr.outvars}
    rows = []
    per_v = {}
    for i, e in enumerate(jaxpr.eqns, start=1):
        if any(str(ov) in outvars for ov in e.outvars):
            continue                      # output eqns are not eliminated
        ov = e.outvars[0]
        if ov not in nodes:
            continue                      # pruned off the Jacobian path
        anc = closure(tg, ov)
        des = closure(g, ov)
        b = len(anc) * len(des)
        per_v[i] = b
        rows.append((i, e.primitive.name, len(anc), len(des), b))
    rows.sort(key=lambda r: -r[4])
    for i, prim, a, d, b in rows[:10]:
        print(f"  v{i:<3} {prim:<16} anc={a:<3} desc={d:<3} bound={b}")
    B = max((r[4] for r in rows), default=0)
    S = sum(r[4] for r in rows)
    N = len(nodes)
    print(f"  per-vertex BOUND (any order) = {B}")
    print(f"  episode-total BOUND (any order, sum) = {S}")
    print(f"  graph nodes N = {N}; worst-case analytic (N-1)^2/4 = "
          f"{(N - 1) ** 2 // 4}")
    return B


if __name__ == "__main__":
    which = build_nn if "--nn" in sys.argv else build_simple
    name, jaxpr, argnums, consts, args = which()
    bound(name, jaxpr, argnums, consts, args)
