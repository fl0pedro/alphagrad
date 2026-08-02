#!/usr/bin/env python3
"""Does ONE face approximation reduce latency / memory? Controlled ablation.

The training data cannot answer this: latency and memory vary with the
ELIMINATION ORDER (26x mem spread on the xent graph) while cosine varies with
the APPROXIMATIONS, so across episodes the two are driven by nearly disjoint
action subspaces and their correlation says nothing about whether an
approximation pays for itself. Here the order is FIXED (ascending valid) and
exactly one face carries exactly one action; everything else is exact.

Per variant: dLatency (perf_counter median over reps, CPU -- relative deltas
only, GPU ranking may differ), dPeakMem (compiled memory_analysis: args +
outputs + temps, deterministic), cosine vs the exact Jacobian on the same
inputs, and APPLIED -- whether the hook actually fired or silently skipped the
rule (a skipped rule measures as exact and must not be read as "approximation
with no effect").
"""
from __future__ import annotations
import os
import time
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_NN_HIDDEN", "256")

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve
from graphax.incremental import IncrementalJaxpr
from graphax.sparse.micro_actions import Compress, Diag, Quant
from alphagrad.approx.common.examples import get_fn, get_args, infer_argnums
from alphagrad.approx.common.masks import (
    legal_compress_actions, legal_diag_actions, make_live_masked_hook)

REPS = int(os.environ.get("ABL_REPS", "30"))
MAX_FACES_PER_VERTEX = int(os.environ.get("ABL_FACES_PER_V", "2"))


def build():
    name = "VmappedNeuralNetwork"
    fn = get_fn(name)
    xs = get_args(name, jrand.PRNGKey(0), dataset="mnist")
    cj = jax.make_jaxpr(fn)(*xs)
    return fn, xs, cj, infer_argnums(name)


def valid_order(jaxpr):
    outvars = {str(v) for v in jaxpr.outvars}
    out_ids = {i for i, e in enumerate(jaxpr.eqns, start=1)
               if any(str(ov) in outvars for ov in e.outvars)}
    return [i for i in range(1, len(jaxpr.eqns) + 1) if i not in out_ids]


def compile_variant(fn, xs, order, argnums, ft):
    return (jax.jit(
        jacve(fn, list(order), argnums=argnums,
              transforms=None, face_transforms=ft or None),
        keep_unused=True).lower(*xs).compile())


def measure(compiled, xs):
    out = compiled(*xs)
    jax.block_until_ready(out)          # warmup + result
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = compiled(*xs)
        jax.block_until_ready(r)
        ts.append(time.perf_counter() - t0)
    ma = compiled.memory_analysis()
    mem = float(getattr(ma, "argument_size_in_bytes", 0)
                + getattr(ma, "output_size_in_bytes", 0)
                + getattr(ma, "temp_size_in_bytes", 0))
    return float(np.median(ts) * 1e3), mem, out   # ms, bytes, outputs


def flat(out):
    leaves = jax.tree_util.tree_leaves(out)
    return jnp.concatenate([jnp.ravel(x).astype(jnp.float32) for x in leaves])


def cosine(a, b):
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return float(jnp.dot(a, b) / jnp.maximum(na * nb, 1e-30))


def catalog(cj, xs, argnums, order):
    """(vertex, face_idx, face_key, action) candidates on the LIVE graph."""
    from jax._src import core as _jcore
    from graphax.core import _eliminate_vertex
    out = []
    ij = IncrementalJaxpr(cj.jaxpr, tuple(argnums), list(cj.literals),
                          list(xs), track_faces=False)
    for v in order:
        keys = list(ij.faces(int(v)))
        seen = []

        def _rec(st, _s=seen):
            _s.append(st)
            return st

        import copy
        from alphagrad.approx.common.masks import _shallow_copy_graph
        g0 = _shallow_copy_graph(ij.graph)
        tg0 = _shallow_copy_graph(ij.tgraph)
        n0 = len(ij.trace.frame.tracing_eqns)
        try:
            with _jcore.set_current_trace(ij.trace):
                _eliminate_vertex(int(v), ij.jaxpr, g0, tg0, ij.vo, False,
                                  transforms=(_rec,), face_transforms=None)
        except Exception:
            seen = []
        finally:
            del ij.trace.frame.tracing_eqns[n0:]
        for f, st in enumerate(seen[:MAX_FACES_PER_VERTEX]):
            if f >= len(keys):
                break
            acts = []
            d = legal_diag_actions(st, 8)
            c = legal_compress_actions(st, 8, ("mean",))
            if d:
                acts.append(("DIAG", d[0]))
            if c:
                acts.append(("COMPRESS", c[0]))
            acts.append(("QUANT_bf16", Quant(dtype="bfloat16")))
            for tag, a in acts:
                out.append((int(v), f, keys[f], tag, a))
        ij.eliminate(int(v))
    return out


def main():
    fn, xs, cj, argnums = build()
    order = valid_order(cj.jaxpr)
    print(f"graph: {len(cj.jaxpr.eqns)} eqns, order = ascending "
          f"({len(order)} vertices), reps={REPS}")

    base_c = compile_variant(fn, xs, order, argnums, None)
    lat0, mem0, out0 = measure(base_c, xs)
    ref = flat(out0)
    print(f"EXACT baseline: latency {lat0:.2f} ms   mem {mem0/1e6:.1f} MB\n")

    cands = catalog(cj, xs, argnums, order)
    print(f"{len(cands)} single-face candidates "
          f"(<= {MAX_FACES_PER_VERTEX} faces/vertex, 1 action/type)\n")
    rows = []
    for (v, f, key, tag, act) in cands:
        stats = {}
        hook = make_live_masked_hook((act,), stats=stats)
        ft = {v: {key: (hook, None, None)}}
        try:
            c = compile_variant(fn, xs, order, argnums, ft)
            lat, mem, out = measure(c, xs)
        except Exception as e:
            rows.append((v, f, tag, None, None, None, False,
                         type(e).__name__))
            continue
        applied = bool(stats.get("applied", 0))
        rows.append((v, f, tag, lat - lat0, mem - mem0,
                     cosine(ref, flat(out)), applied, ""))

    print(f"{'v':>3} {'f':>2} {'action':<11} {'dLat ms':>9} {'dLat %':>7} "
          f"{'dMem MB':>9} {'cos':>8} {'applied':>7}")
    ap = []
    for (v, f, tag, dl, dm, cs, applied, err) in rows:
        if err:
            print(f"{v:>3} {f:>2} {tag:<11} {'--':>9} {'--':>7} {'--':>9} "
                  f"{'--':>8} {'--':>7}  {err}")
            continue
        print(f"{v:>3} {f:>2} {tag:<11} {dl:>+9.3f} {dl/lat0*100:>+6.1f}% "
              f"{dm/1e6:>+9.2f} {cs:>8.4f} {str(applied):>7}")
        if applied:
            ap.append((tag, dl, dm, cs))

    if ap:
        arr_dl = np.array([r[1] for r in ap])
        arr_dm = np.array([r[2] for r in ap])
        arr_cs = np.array([r[3] for r in ap])
        n = len(ap)
        print(f"\n=== APPLIED approximations only (n={n}) ===")
        print(f"latency reduced : {int((arr_dl < 0).sum())}/{n} "
              f"(mean {arr_dl.mean():+.3f} ms, {arr_dl.mean()/lat0*100:+.1f}%)")
        print(f"memory  reduced : {int((arr_dm < 0).sum())}/{n} "
              f"(mean {arr_dm.mean()/1e6:+.2f} MB)")
        print(f"cosine  < 0.999 : {int((arr_cs < 0.999).sum())}/{n} "
              f"(min {arr_cs.min():.4f})")
        for tag in ("DIAG", "COMPRESS", "QUANT_bf16"):
            sel = [r for r in ap if r[0] == tag]
            if sel:
                dls = np.array([r[1] for r in sel])
                dms = np.array([r[2] for r in sel])
                css = np.array([r[3] for r in sel])
                print(f"  {tag:<11} n={len(sel):>3}  dLat {dls.mean():+.3f} ms"
                      f"  dMem {dms.mean()/1e6:+.2f} MB  cos {css.mean():.4f}")
        skipped = sum(1 for r in rows if not r[7] and not r[6] and r[3] is not None)
        print(f"\nsampled-but-SKIPPED at application: "
              f"{sum(1 for r in rows if r[3] is not None and not r[6])}"
              f"/{sum(1 for r in rows if r[3] is not None)}")


if __name__ == "__main__":
    main()
