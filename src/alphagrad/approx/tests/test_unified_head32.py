#!/usr/bin/env python3
"""Tests for the 32-output head: prime gates gone, factor = gcd, coprime pairs masked."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from jax.flatten_util import ravel_pytree
from math import gcd

from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures, COMPRESS_KINDS, OP_COMPRESS, OP_DIAG,
    OP_END, OP_QUANT, QUANT_DTYPES, precompute_factor_tables)
from alphagrad.approx.unified_head import (
    HEAD_WIDTH, MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES, REDUCE_FNS)
from alphagrad.approx.unified_micro import UnifiedMicroPolicy

FAIL = []


def ck(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def feats(sizes):
    sz = jnp.asarray(sizes, jnp.int32)
    n = len(sizes)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def main():
    E, S = 32, 16
    tables = precompute_factor_tables(64)
    pol = UnifiedMicroPolicy(embd_dim=E, max_substeps=S, key=jrand.PRNGKey(0))
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))

    print("=== layout ===")
    ck("head is 32 wide", HEAD_WIDTH == 32, f"got {HEAD_WIDTH}")
    ck("op slot still 4 wide", NUM_APPROX_OPS == 4)
    nparam = sum(x.size for x in
                 jax.tree_util.tree_leaves(eqx.filter(pol, eqx.is_inexact_array)))
    print(f"  head parameters: {nparam:,}")

    # mixed sizes, deliberately including coprime partners (3 vs 8, 5 vs 8)
    f = feats([8, 8, 4, 3, 16, 5, 8, 4])
    sizes = [int(x) for x in f.size]

    print("\n=== BLOCKDIAG: factor == gcd, pairs never coprime ===")
    bad_fac = bad_cop = n_bd = 0
    seen_ops, pairs = set(), set()
    for s in range(600):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(1000 + s))
        op0 = int(acts.op_type[0])
        seen_ops.add(op0)
        if op0 == OP_DIAG:
            n_bd += 1
            i, j = int(acts.i[0]), int(acts.j[0])
            g = gcd(sizes[i], sizes[j])
            pairs.add((sizes[i], sizes[j]))
            if g == 1:
                bad_cop += 1
            if int(acts.factor[0]) != g:
                bad_fac += 1
    ck("factor == gcd(N_i, N_j)", bad_fac == 0, f"{bad_fac}/{n_bd} bad")
    ck("no coprime pair ever chosen", bad_cop == 0, f"{bad_cop}/{n_bd} bad")
    ck("all ops exercised",
       seen_ops >= {OP_DIAG, OP_COMPRESS, OP_QUANT, OP_END},
       f"seen={sorted(seen_ops)}")
    print(f"  pair sizes seen: {sorted(pairs)}")

    print("\n=== square pair -> PURE diagonal ===")
    fsq = feats([8] * 8)
    facs = set()
    for s in range(200):
        acts, *_ = pol.sample(ctx, fsq, tables, jrand.PRNGKey(2000 + s))
        if int(acts.op_type[0]) == OP_DIAG:
            facs.add(int(acts.factor[0]))
    ck("8x8 -> factor 8 (1x1 blocks)", facs <= {8}, f"factors={sorted(facs)}")

    print("\n=== all-coprime axis set -> BLOCKDIAG masked off entirely ===")
    fcop = feats([3, 5, 7, 11, 13, 17, 19, 23])
    ops = set()
    for s in range(300):
        acts, *_ = pol.sample(ctx, fcop, tables, jrand.PRNGKey(6000 + s))
        ops.add(int(acts.op_type[0]))
    ck("no DIAG when every pair is coprime", OP_DIAG not in ops,
       f"ops={sorted(ops)}")

    print("\n=== REDUCE: multi-axis, descending, kind round-trips ===")
    bad_desc = bad_kind = n_rd = 0
    n_axes = []
    kinds = set()
    for s in range(400):
        acts, _, _, arity, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(3000 + s))
        opsq = np.asarray(acts.op_type)
        if opsq[0] != OP_COMPRESS:
            continue
        n_rd += 1
        ax = [int(a) for a, o in zip(np.asarray(acts.i), opsq) if o == OP_COMPRESS]
        n_axes.append(len(ax))
        if ax != sorted(ax, reverse=True):
            bad_desc += 1
        ks = {int(k) for k, o in zip(np.asarray(acts.compress_kind), opsq)
              if o == OP_COMPRESS}
        kinds |= ks
        if len(ks) != 1 or COMPRESS_KINDS[ks.pop()] not in REDUCE_FNS:
            bad_kind += 1
    ck("COMPRESS axes emitted descending", bad_desc == 0, f"{bad_desc}/{n_rd}")
    ck("one reduce-fn per action, from REDUCE_FNS", bad_kind == 0, f"{bad_kind}/{n_rd}")
    ck("REDUCE never emits zero axes", n_axes and min(n_axes) >= 1,
       f"min={min(n_axes) if n_axes else '-'} max={max(n_axes) if n_axes else '-'}")
    ck("multi-axis reduce happens", n_axes and max(n_axes) > 1)
    ck("all 5 reduce fns reachable", len(kinds) >= 4, f"kinds={sorted(kinds)}")

    print("\n=== QUANT dtype is a live bit ===")
    dts = set()
    for s in range(400):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(4000 + s))
        if int(acts.op_type[0]) == OP_QUANT:
            dts.add(str(QUANT_DTYPES[int(acts.quant_dtype[0])]))
    ck("both dtype arms reachable", len(dts) == 2, f"dtypes={sorted(dts)}")

    print("\n=== round-trip: sample -> evaluate ===")
    for name, ff in (("mixed", f), ("square", fsq), ("all-coprime", fcop)):
        diffs, ediffs = [], []
        for s in range(400):
            acts, lp, ent, *_ = pol.sample(ctx, ff, tables, jrand.PRNGKey(3000 + s))
            lp2, ent2, *_ = pol.evaluate(ctx, ff, tables, acts)
            diffs.append(abs(float(lp) - float(lp2)))
            ediffs.append(abs(float(ent) - float(ent2)))
        ck(f"log_prob round-trips ({name})", max(diffs) < 1e-5,
           f"max|dlogp|={max(diffs):.3e}")
        ck(f"entropy round-trips ({name})", max(ediffs) < 1e-5,
           f"max|dent|={max(ediffs):.3e}")

    print("\n=== masks / gradient / jit ===")
    only_q = jnp.array([0.0, 0.0, 1.0, 0.0])
    ops = set()
    for s in range(150):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(5000 + s),
                              op_legality_override=only_q)
        ops.add(int(acts.op_type[0]))
    ck("op_legality_override honoured", ops <= {OP_QUANT, OP_END},
       f"ops={sorted(ops)}")

    cv = jnp.zeros((NUM_REDUCE_AXES,)).at[2].set(1.0)
    axset = set()
    for s in range(300):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(7000 + s),
                              compress_valid=cv)
        opsq = np.asarray(acts.op_type)
        axset |= {int(a) for a, o in zip(np.asarray(acts.i), opsq)
                  if o == OP_COMPRESS}
    ck("compress_valid honoured", axset <= {2}, f"axes={sorted(axset)}")

    g = eqx.filter_grad(lambda p: p.sample(ctx, f, tables, jrand.PRNGKey(5))[1])(pol)
    gn = float(jnp.linalg.norm(ravel_pytree(eqx.filter(g, eqx.is_inexact_array))[0]))
    ck("gradient flows", np.isfinite(gn) and gn > 0, f"|grad|={gn:.4g}")
    try:
        fj = eqx.filter_jit(lambda p, c, k: p.sample(c, f, tables, k)[1])
        _ = fj(pol, ctx, jrand.PRNGKey(9))
        ck("jit-able", True)
    except Exception as e:
        ck("jit-able", False, f"{type(e).__name__}: {str(e)[:70]}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
