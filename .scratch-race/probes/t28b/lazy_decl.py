#!/usr/bin/env python
"""Report every frame slot the lazy analysis DECLINED to shrink."""
from __future__ import annotations
import os, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_DEMAND_EMIT", "1")
os.environ["GRAPHAX_EINSUM_GENERAL"] = "0"
os.environ["GRAPHAX_PLANNER_EXACT"] = "0"
import numpy as np, jax, importlib
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
    specs, fs, sk = lm.get_plan_arrays(plan, len(order))
    ft = envmod._face_transforms_for_order(cfg, env.consts, env.args, list(order),
        list(np.asarray(specs)), list(np.asarray(fs)), list(np.asarray(sk)))

_orig = mm._lazy_frame
DECL = collections.Counter()


def wrapped(lhs_val, rhs_val, pairs):
    eff, lazy, demote = _orig(lhs_val, rhs_val, pairs)
    for i, (p, e, z) in enumerate(zip(pairs, eff, lazy)):
        lo, lb, ls = mm._slot_phys(lhs_val, i)
        ro, rb, rs = mm._slot_phys(rhs_val, i)
        for name, tgt, phys, shrunk in (
            ("lhs.block", p.lhs.block_len, lb, e.lhs.block_len),
            ("lhs.shared", p.lhs.shared_block_len, ls, e.lhs.shared_block_len),
            ("rhs.block", p.rhs.block_len, rb, e.rhs.block_len),
            ("rhs.shared", p.rhs.shared_block_len, rs, e.rhs.shared_block_len),
        ):
            if tgt > 1 and phys == 1 and shrunk == tgt:
                DECL[(p.pairing_type, name, f"ol{p.lhs.outer_len} or{p.rhs.outer_len}",
                      f"n{len(pairs)}")] += 1
    return eff, lazy, demote


mm._lazy_frame = wrapped
fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums, has_aux=cfg.has_aux,
           sparse_representation=True, transforms=[], face_transforms=ft)
jax.eval_shape(fn, *env.args)
print(f"target={EX} plan={PLAN} DECLINED:")
for k, n in DECL.most_common(30):
    print(f"  x{n:4d} {k}")
