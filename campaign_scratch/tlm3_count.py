"""tlm3_: equation / eliminable-vertex count for TransformerLM (2 blk) vs
TransformerLM3 (3 blk), on the SAME graph the PPO env builds.

Mirrors ppo.py: get_fn -> get_args -> grad_target_setup -> _traced_inlined,
then reproduces env.VertexEliminationEnv's `valid_vertices` derivation exactly
(env.py ~4244) without paying for the full env construction.
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
from graphax.core import _build_graph  # noqa: E402
from alphagrad.approx.common.examples import (  # noqa: E402
    get_fn, get_args, infer_argnums, grad_target_setup, _tlm_dims,
)


def traced_inlined(target_fn, xs):
    from graphax import inline_call_primitives
    cj = jax.make_jaxpr(target_fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:
        from jax.extend.core import ClosedJaxpr
    except ImportError:
        from jax.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def report(name, measure_grad):
    key = jax.random.PRNGKey(0)
    target_fn = get_fn(name)
    xs = get_args(name, key, dataset="wikitext2")
    argnums = infer_argnums(name)
    fn, xs, argnums = grad_target_setup(
        SimpleNamespace(measure_grad=measure_grad, seed_vertices=False),
        target_fn, xs, name)
    cj = traced_inlined(fn, xs)
    jaxpr = cj.jaxpr
    _, _, _, vo_vertices = _build_graph(jaxpr, tuple(xs), cj.literals, argnums)
    valid = []
    for i, eqn in enumerate(jaxpr.eqns, 1):
        if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices:
            valid.append(i)
    print(f"[tlm3] {name:16s} measure_grad={int(measure_grad)} "
          f"n_args={len(xs):3d} argnums={len(argnums):3d} "
          f"eqns={len(jaxpr.eqns):4d} eliminable={len(valid):4d}",
          flush=True)
    return len(jaxpr.eqns), len(valid)


if __name__ == "__main__":
    print("[tlm3] dims (SEQ, DMODEL, VOCAB) =", _tlm_dims(), flush=True)
    out = {}
    for name in ("TransformerLM", "TransformerLM3"):
        for mg in (False, True):
            try:
                out[(name, mg)] = report(name, mg)
            except Exception as e:  # noqa: BLE001
                print(f"[tlm3] {name} measure_grad={int(mg)} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
    for mg in (False, True):
        a = out.get(("TransformerLM", mg))
        b = out.get(("TransformerLM3", mg))
        if a and b:
            print(f"[tlm3] SUMMARY measure_grad={int(mg)}: "
                  f"2blk eqns {a[0]} / elim {a[1]}  ->  "
                  f"3blk eqns {b[0]} / elim {b[1]}  "
                  f"(+{b[0]-a[0]} eqns, +{b[1]-a[1]} elim, "
                  f"V^4 ratio {(b[1]/a[1])**4:.2f}x)", flush=True)
    sys.stdout.flush()
