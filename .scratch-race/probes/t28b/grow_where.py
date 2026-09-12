#!/usr/bin/env python
"""Which side and which frame slot the tiled path still broadcasts."""
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

WHERE = collections.Counter()
_orig = mm._prepare_contraction_views


def wrapped(lhs_val, rhs_val, pairs, shared, total, split, keep_l=None, keep_r=None,
            keep_sl=None, keep_sr=None):
    N = len(pairs)
    pre_l = [int(lhs_val.shape[k]) for k in range(lhs_val.ndim)]
    pre_r = [int(rhs_val.shape[k]) for k in range(rhs_val.ndim)]
    lv, rv, lbc, rbc = _orig(lhs_val, rhs_val, pairs, shared, total, split, keep_l,
                             keep_r, keep_sl, keep_sr)
    for i, p in enumerate(pairs):
        lo, lb, ls = pre_l[3 * i], pre_l[3 * i + 1], pre_l[3 * i + 2]
        ro, rb, rs = pre_r[3 * i], pre_r[3 * i + 1], pre_r[3 * i + 2]
        m_l, b_l, s_l = int(lv.shape[i]), int(lv.shape[N + i]), int(lv.shape[2 * N + i])
        m_r, s_r, f_r = int(rv.shape[i]), int(rv.shape[N + i]), int(rv.shape[2 * N + i])
        if m_l > 1 and lo == 1 and ls == 1:
            WHERE[(p.pairing_type, "L.meta", m_l)] += 1
        if b_l > 1 and lb == 1:
            WHERE[(p.pairing_type, "L.block", b_l)] += 1
        if s_l > 1 and ls == 1:
            WHERE[(p.pairing_type, "L.split", s_l)] += 1
        if m_r > 1 and ro == 1 and rb == 1:
            WHERE[(p.pairing_type, "R.meta", m_r)] += 1
        if s_r > 1 and rb == 1:
            WHERE[(p.pairing_type, "R.split", s_r)] += 1
        if f_r > 1 and rs == 1:
            WHERE[(p.pairing_type, "R.sblk", f_r)] += 1
    return lv, rv, lbc, rbc


mm._prepare_contraction_views = wrapped
fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums, has_aux=cfg.has_aux,
           sparse_representation=True, transforms=[], face_transforms=ft)
jax.eval_shape(fn, *env.args)
print(f"target={EX} plan={PLAN} still-broadcast slots:")
for k, n in WHERE.most_common(30):
    print(f"  x{n:4d} {k}")
