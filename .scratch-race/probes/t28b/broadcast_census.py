#!/usr/bin/env python
"""Census of the tiled path's implicit-axis broadcast.

Wraps ``_as_shape`` and records every ``mode="broadcast"`` call that actually
grows the buffer: the element count before and after, and which frame slot
grew.  Run on a small target so it is a quick script.

    TR_EXAMPLE=NeuralNetwork TR_DATASET=mnist REPRO_PLAN=exact python broadcast_census.py
"""
from __future__ import annotations
import os, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_DEMAND_EMIT", "1")
os.environ["GRAPHAX_EINSUM_GENERAL"] = os.environ.get("TR_EG", "0")
os.environ["GRAPHAX_PLANNER_EXACT"] = os.environ.get("TR_PE", "0")

import numpy as np, jax, importlib
import alphagrad.approx.tools.landscape_map as lm
from graphax import jacve
from graphax.sparse.tensor import SparseTensor
mm = importlib.import_module("graphax.sparse.ops.matmul")

envmod = lm.envmod
PLAN = os.environ.get("REPRO_PLAN", "exact")
EX = os.environ.get("TR_EXAMPLE", "NeuralNetwork")
DS = os.environ.get("TR_DATASET", "mnist")
CLI = ["--example", EX, "--dataset", DS, "--seed", "250197",
       "--latency-inner-reps", "1", "--num-data-points", "1", "--reps-per-point", "1",
       "--quality-metric", "grad_cosine", "--approx-old", "same",
       "--out-dir", "/tmp/t28b-repro", "--dry-run",
       "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2"]
lm.ARGS = lm.make_argparser().parse_args(CLI)
env, _s, _ = lm.build_env(lm.ARGS)
cfg = env.config
order = [int(v) for v in lm.rev_order(env)]
if PLAN == "exact":
    ft = None
else:
    op = {"quant_all": "quant", "diag_all": "diag", "compress_all": "compress"}[PLAN]
    plan = lm.build_ladder_plan(env, order, op, "all", lm.ARGS)
    specs, face_specs, face_skips = lm.get_plan_arrays(plan, len(order))
    ft = envmod._face_transforms_for_order(
        cfg, env.consts, env.args, list(order), list(np.asarray(specs)),
        list(np.asarray(face_specs)), list(np.asarray(face_skips)))

_orig_as_shape = mm._as_shape
STATS = collections.Counter()
GROWTH = []


def _as_shape(view, target_shape, *, mode):
    out = _orig_as_shape(view, target_shape, mode=mode)
    if mode == "broadcast":
        a = int(np.prod(view.shape)) if view.shape else 1
        b = int(np.prod(tuple(target_shape))) if tuple(target_shape) else 1
        STATS["calls"] += 1
        if b > a:
            STATS["grew"] += 1
            STATS["elems_before"] += a
            STATS["elems_after"] += b
            GROWTH.append((tuple(view.shape), tuple(target_shape)))
    return out


mm._as_shape = _as_shape
fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
           has_aux=cfg.has_aux, sparse_representation=True,
           transforms=[], face_transforms=ft)
jax.eval_shape(fn, *env.args)
print(f"target={EX} plan={PLAN} engine_eg={os.environ['GRAPHAX_EINSUM_GENERAL']}")
print("stats", dict(STATS))
top = collections.Counter()
for a, b in GROWTH:
    top[(a, b)] += 1
for (a, b), n in top.most_common(15):
    ra = int(np.prod(a)) if a else 1
    rb = int(np.prod(b)) if b else 1
    print(f"  x{n:4d}  {a} -> {b}   {ra} -> {rb}  (x{rb/max(ra,1):.1f})")
