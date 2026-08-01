#!/usr/bin/env python3
"""How many faces each vertex has, ALONG an elimination order.

The unrolling's length is not a property of the graph alone: eliminating a
vertex rewires its neighbours, so vertex k's face count depends on the k-1
before it. Enumerate with `IncrementalJaxpr.faces` on the LIVE graph, stepping
the builder as we go -- the same call `_face_transforms_for_order` and
`LiveFaceStream` use.
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jrand


def walk(name, jaxpr, argnums, consts, args, order):
    from graphax.incremental import IncrementalJaxpr
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=True)
    rows = []
    for v in order:
        try:
            n = len(list(ij.faces(int(v))))
        except Exception as exc:
            rows.append((int(v), -1, type(exc).__name__))
            break
        rows.append((int(v), n, ""))
        try:
            ij.eliminate(int(v))
        except Exception as exc:
            rows.append((int(v), -1, "elim:" + type(exc).__name__))
            break
    print(f"\n### {name}: {len(rows)} vertices, "
          f"{sum(r[1] for r in rows if r[1] > 0)} faces total")
    for v, n, err in rows:
        print(f"  vertex {v:>3}  faces={n:>3} {err}")
    return rows


def simple():
    """f(x1, x2) = ( log(sin(x1*x2)), sin(x1*x2) - x1*x2 )"""
    def f(x1, x2):
        y = x1 * x2
        s = jnp.sin(y)
        return jnp.log(s), s - y

    xs = (jnp.array(1.3), jnp.array(0.7))
    cj = jax.make_jaxpr(f)(*xs)
    jaxpr = cj.jaxpr
    print("=== f(x1,x2) = (log(sin(x1*x2)), sin(x1*x2) - x1*x2) ===")
    for i, e in enumerate(jaxpr.eqns, start=1):
        print(f"  v{i}: {e.primitive.name:<10} {e.invars} -> {e.outvars}")
    from graphax.core import _build_graph  # noqa: F401  (validity via jacve)
    from graphax import jacve
    # valid vertices == the intermediate (non-output) eqns graphax will accept
    n_eqns = len(jaxpr.eqns)
    out_ids = set()
    outvars = {str(v) for v in jaxpr.outvars}
    for i, e in enumerate(jaxpr.eqns, start=1):
        if any(str(ov) in outvars for ov in e.outvars):
            out_ids.add(i)
    valid = [i for i in range(1, n_eqns + 1) if i not in out_ids]
    print(f"  eqns={n_eqns}  output-eqns={sorted(out_ids)}  "
          f"valid (eliminable) = {valid}")
    walk("f(x1,x2), order = ascending", jaxpr, (0, 1), cj.literals, xs, valid)
    return


def nn256():
    sys.path.insert(0, os.path.expanduser("~/dsnn/alphagrad/src"))
    from alphagrad.approx.common.examples import (
        get_fn, get_args, infer_argnums)
    key = jrand.PRNGKey(0)
    name = "VmappedNeuralNetwork"
    fn = get_fn(name)
    xs = get_args(name, key, dataset="mnist")
    argnums = infer_argnums(name)
    cj = jax.make_jaxpr(fn, static_argnums=())(*xs)
    jaxpr = cj.jaxpr
    n_eqns = len(jaxpr.eqns)
    outvars = {str(v) for v in jaxpr.outvars}
    out_ids = {i for i, e in enumerate(jaxpr.eqns, start=1)
               if any(str(ov) in outvars for ov in e.outvars)}
    valid = [i for i in range(1, n_eqns + 1) if i not in out_ids]
    print(f"\n=== {name} (xent) ===")
    print(f"  eqns={n_eqns}  valid={len(valid)}  argnums={argnums}")
    for i, e in enumerate(jaxpr.eqns, start=1):
        mark = " (out)" if i in out_ids else ""
        print(f"  v{i:>3}: {e.primitive.name}{mark}")
    walk(f"{name}, order = ascending", jaxpr, argnums, cj.literals, xs, valid)


if __name__ == "__main__":
    simple()
    if "--nn" in sys.argv:
        nn256()
