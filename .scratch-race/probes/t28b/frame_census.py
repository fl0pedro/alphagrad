#!/usr/bin/env python
"""Per-pair census of the tiled frame: for every contraction, for every Pair,
report the pairing type, the logical grid extents and which of the six frame
slots the operands actually store.  Tells which laziness rule would fire."""
from __future__ import annotations
import os, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_DEMAND_EMIT", "1")
os.environ["GRAPHAX_EINSUM_GENERAL"] = os.environ.get("TR_EG", "0")
os.environ["GRAPHAX_PLANNER_EXACT"] = os.environ.get("TR_PE", "0")

import numpy as np, jax, importlib, math
import alphagrad.approx.tools.landscape_map as lm
from graphax import jacve
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

ROWS = collections.Counter()
BYTES = collections.Counter()
_orig = mm._execute_block_sparse_contraction


def wrapped(lhs_val, rhs_val, pairs, ctx):
    N = len(pairs)
    shared, total, split, scalar = mm._contraction_factors(pairs)
    for i, p in enumerate(pairs):
        lo, lb, ls = (int(lhs_val.shape[3 * i + k]) for k in range(3))
        ro, rb, rs = (int(rhs_val.shape[3 * i + k]) for k in range(3))
        key = (p.pairing_type,
               f"outer l{p.lhs.outer_len}/{lo} r{p.rhs.outer_len}/{ro}"
               f" tot{total[i]} shr{shared[i]} split{split[i]}"
               f" blk l{p.lhs.block_len}/{lb} r{p.rhs.shared_block_len}/{rs}")
        ROWS[key] += 1
        # grid growth attributable to this pair's meta axis
        if total[i] > 1:
            if lo == 1 and ro == 1:
                ROWS[("META", "both implicit")] += 1
            elif lo == 1 or ro == 1:
                ROWS[("META", "one implicit")] += 1
            else:
                ROWS[("META", "both stored")] += 1
    return _orig(lhs_val, rhs_val, pairs, ctx)


mm._execute_block_sparse_contraction = wrapped
fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums,
           has_aux=cfg.has_aux, sparse_representation=True,
           transforms=[], face_transforms=ft)
jax.eval_shape(fn, *env.args)
print(f"target={EX} plan={PLAN}")
for k, n in ROWS.most_common(40):
    print(f"  x{n:4d}  {k}")
