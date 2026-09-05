#!/usr/bin/env python
"""Local repro of finding 61 verdict 6: NN256, reverse order, Reduce(mean) on
slot ``new`` of every face; the tiled engine loses the logical extents of W1
and Wout.  Prints the returned dims per engine.  CPU only, small target."""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
for _k in ("GRAPHAX_PLANNER_EXACT", "GRAPHAX_EINSUM_GENERAL", "GRAPHAX_TILED_LEGACY"):
    os.environ.pop(_k, None)

import numpy as np
import jax
import alphagrad.approx.tools.landscape_map as lm
from graphax import jacve
from graphax.sparse.tensor import SparseTensor

envmod = lm.envmod
PLAN = os.environ.get("REPRO_PLAN", "compress_all")
ENGINES = {
    "tiled":   {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "planner": {"GRAPHAX_EINSUM_GENERAL": "1", "GRAPHAX_PLANNER_EXACT": "1"},
}
if os.environ.get("REPRO_LEGACY") == "1":
    ENGINES["legacy"] = {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0",
                         "GRAPHAX_TILED_LEGACY": "1"}

CLI = ["--example", "NeuralNetwork", "--dataset", "mnist", "--seed", "250197",
       "--latency-inner-reps", "1", "--num-data-points", "1", "--reps-per-point", "1",
       "--quality-metric", "grad_cosine", "--approx-old", "same",
       "--out-dir", "/tmp/t28b-repro", "--dry-run",
       "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2"]
lm.ARGS = lm.make_argparser().parse_args(CLI)
env, eval_samples, _ = lm.build_env(lm.ARGS)
cfg = env.config
order = [int(v) for v in lm.rev_order(env)]
inv = lm.face_inventory(env, order)
if PLAN == "exact":
    ft = None
else:
    op = {"quant_all": "quant", "diag_all": "diag", "compress_all": "compress"}[PLAN]
    plan = lm.build_ladder_plan(env, order, op, "all", lm.ARGS)
    specs, face_specs, face_skips = lm.get_plan_arrays(plan, len(order))
    ft = envmod._face_transforms_for_order(
        cfg, env.consts, env.args, list(order), list(np.asarray(specs)),
        list(np.asarray(face_specs)), list(np.asarray(face_skips)))
print(f"live faces {len(inv)}  plan {PLAN}")

DENSE = {}
for name, knobs in ENGINES.items():
    saved = {k: os.environ.get(k) for k in knobs}
    os.environ.update(knobs)
    try:
        fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
                   has_aux=cfg.has_aux, sparse_representation=True,
                   transforms=[], face_transforms=ft)
        out = jax.jit(fn, keep_unused=True).trace(*env.args).out_info if False else fn(*env.args)
        jac = out[1] if cfg.has_aux else out
        leaves = jax.tree_util.tree_leaves(
            jac, is_leaf=lambda x: isinstance(x, SparseTensor))
        print(f"--- {name}")
        import numpy as _np
        _parts=[]
        for t in leaves:
            _a = _np.asarray(t.dense() if isinstance(t, SparseTensor) else t, dtype=_np.float64)
            _parts.append(_a.ravel())
        _v = _np.concatenate(_parts)
        DENSE[name] = _v
        print("   dense n=%d  l2=%.6e" % (_v.size, _np.linalg.norm(_v)))
        for t in leaves:
            if isinstance(t, SparseTensor):
                dims = tuple(t.out_dims) + tuple(t.primal_dims)
                print("   sizes", [int(d.size) for d in dims],
                      "logical", [int(d.logical_size) for d in dims],
                      "axes", [None if d.axis is None else int(d.axis) for d in dims],
                      "val", None if t.val is None else list(t.val.shape))
            else:
                print("   array", getattr(t, "shape", None))
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

import numpy as _np
_keys=list(DENSE)
for i in range(len(_keys)):
    for j in range(i+1,len(_keys)):
        a,b=DENSE[_keys[i]],DENSE[_keys[j]]
        if a.shape!=b.shape:
            print("cmp %s vs %s SHAPE %d vs %d"%(_keys[i],_keys[j],a.size,b.size)); continue
        nb=_np.linalg.norm(b)
        print("cmp %s vs %s rel_l2=%.3e"%(_keys[i],_keys[j],_np.linalg.norm(a-b)/(nb if nb else 1.0)))
