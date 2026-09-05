#!/usr/bin/env python
"""Trace every tiled matmul of the NN256 reverse Reduce plan: print the operand
dims and the output dims, so the step that drops a logical extent is visible."""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_DEMAND_EMIT", "1")
os.environ["GRAPHAX_EINSUM_GENERAL"] = os.environ.get("TR_EG", "0")
os.environ["GRAPHAX_PLANNER_EXACT"] = os.environ.get("TR_PE", "0")

import numpy as np, jax
import alphagrad.approx.tools.landscape_map as lm
from graphax import jacve
from graphax.sparse.tensor import SparseTensor
import importlib as _il
mm = _il.import_module("graphax.sparse.ops.matmul")

envmod = lm.envmod
PLAN = os.environ.get("REPRO_PLAN", "compress_all")
CLI = ["--example", "NeuralNetwork", "--dataset", "mnist", "--seed", "250197",
       "--latency-inner-reps", "1", "--num-data-points", "1", "--reps-per-point", "1",
       "--quality-metric", "grad_cosine", "--approx-old", "same",
       "--out-dir", "/tmp/t28b-repro", "--dry-run",
       "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2"]
lm.ARGS = lm.make_argparser().parse_args(CLI)
env, _s, _ = lm.build_env(lm.ARGS)
cfg = env.config
order = [int(v) for v in lm.rev_order(env)]
op = {"quant_all": "quant", "diag_all": "diag", "compress_all": "compress"}[PLAN]
plan = lm.build_ladder_plan(env, order, op, "all", lm.ARGS)
specs, face_specs, face_skips = lm.get_plan_arrays(plan, len(order))
ft = envmod._face_transforms_for_order(
    cfg, env.consts, env.args, list(order), list(np.asarray(specs)),
    list(np.asarray(face_specs)), list(np.asarray(face_skips)))


def d2s(t):
    if not isinstance(t, SparseTensor):
        return f"arr{getattr(t,'shape',None)}"
    def one(d):
        return (f"{type(d).__name__[0]}{d.id}:log{d.logical_size}"
                f"(sz{d.size},blk{getattr(d,'block_size',None)},"
                f"ax{d.axis},bax{getattr(d,'block_axis',None)})")
    return ("out[" + ",".join(one(d) for d in t.out_dims) + "] pri["
            + ",".join(one(d) for d in t.primal_dims) + "] val"
            + (str(list(t.val.shape)) if t.val is not None else "None"))


_orig = mm.matmul
_n = [0]


def traced(lhs, rhs, count=False):
    res = _orig(lhs, rhs, count=count)
    out = res[0] if count else res
    _n[0] += 1
    print(f"[mm {_n[0]:04d}]")
    print(f"    L {d2s(lhs)}")
    print(f"    R {d2s(rhs)}")
    print(f"    O {d2s(out)}")
    return res


mm.matmul = traced
import graphax.core as gxcore
import graphax.sparse.tensor as gxt
gxcore.sparse_matmul = traced
gxt.matmul = traced
fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
           has_aux=cfg.has_aux, sparse_representation=True,
           transforms=[], face_transforms=ft)
out = fn(*env.args)
jac = out[1] if cfg.has_aux else out
for t in jax.tree_util.tree_leaves(jac, is_leaf=lambda x: isinstance(x, SparseTensor)):
    print("FINAL", d2s(t))
