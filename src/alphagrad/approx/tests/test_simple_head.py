#!/usr/bin/env python3
"""Tests for the SIMPLIFIED head: 16 outputs, fixed factor/reduce/dtype rules."""
from __future__ import annotations
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from jax.flatten_util import ravel_pytree

from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures, COMPRESS_KINDS, OP_COMPRESS, OP_DIAG,
    OP_END, OP_QUANT, QUANT_DTYPES, precompute_factor_tables)
from alphagrad.approx.unified_head import (
    HEAD_WIDTH, MAX_PAIR_IDX, NUM_APPROX_OPS, block_count)
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
    ck("head is 16 wide", HEAD_WIDTH == 16, f"got {HEAD_WIDTH}")
    ck("3 ops (no NONE)", NUM_APPROX_OPS == 3)

    print("\n=== fixed rule: block count = min(n_i, n_j) ===")
    ck("square 8x8 -> 8 blocks (pure diagonal)",
       int(block_count(jnp.int32(8), jnp.int32(8))) == 8)
    ck("2x8 -> 2 blocks (each 1x4)",
       int(block_count(jnp.int32(2), jnp.int32(8))) == 2)
    ck("8x2 -> 2 blocks", int(block_count(jnp.int32(8), jnp.int32(2))) == 2)

    print("\n=== emitted MicroAction obeys the fixed rules ===")
    f = feats([8, 8, 4, 2, 16, 16, 8, 4])
    bad_fac = bad_kind = bad_dt = bad_axis = 0
    seen = set()
    for s in range(400):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(1000 + s))
        op0 = int(acts.op_type[0])
        seen.add(op0)
        if op0 == OP_DIAG:
            i, j = int(acts.i[0]), int(acts.j[0])
            ni, nj = int(f.size[i]), int(f.size[j])
            g = int(tables.gcd[ni, nj])
            fac = int(acts.factor[0])
            # must divide the gcd and not exceed min(ni,nj)
            if fac < 1 or g % fac != 0 or fac > min(ni, nj):
                bad_fac += 1
        if op0 == OP_COMPRESS:
            if int(acts.compress_kind[0]) != COMPRESS_KINDS.index("mean"):
                bad_kind += 1
            if int(acts.i[0]) != 0:
                bad_axis += 1
        if op0 == OP_QUANT:
            dt = QUANT_DTYPES[int(acts.quant_dtype[0])]
            if str(dt) != "bfloat16" and getattr(dt, "__name__", "") != "bfloat16":
                bad_dt += 1
    ck("DIAG factor divides gcd and <= min(n_i,n_j)", bad_fac == 0, f"{bad_fac} bad")
    ck("REDUCE always kind=mean", bad_kind == 0, f"{bad_kind} bad")
    ck("REDUCE always axis 0", bad_axis == 0, f"{bad_axis} bad")
    ck("QUANT always bfloat16", bad_dt == 0, f"{bad_dt} bad")
    ck("all ops exercised", seen >= {OP_DIAG, OP_COMPRESS, OP_QUANT, OP_END},
       f"seen={sorted(seen)}")

    print("\n=== square pair really gives a PURE diagonal ===")
    fsq = feats([8, 8, 8, 8, 8, 8, 8, 8])
    facs = set()
    for s in range(200):
        acts, *_ = pol.sample(ctx, fsq, tables, jrand.PRNGKey(2000 + s))
        if int(acts.op_type[0]) == OP_DIAG:
            facs.add(int(acts.factor[0]))
    ck("8x8 pairs -> factor 8 (1x1 blocks)", facs <= {8}, f"factors={sorted(facs)}")

    print("\n=== round-trip: sample -> evaluate ===")
    diffs = []
    for s in range(400):
        acts, lp, ent, arity, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(3000 + s))
        lp2, ent2, *_ = pol.evaluate(ctx, f, tables, acts)
        diffs.append(abs(float(lp) - float(lp2)))
    ck("log_prob round-trips exactly", max(diffs) < 1e-5,
       f"max|dlogp|={max(diffs):.3e} over 400")

    print("\n=== masks / gradient / jit ===")
    only_q = jnp.array([0.0, 0.0, 1.0, 0.0])
    ops = set()
    for s in range(150):
        acts, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(4000 + s),
                              op_legality_override=only_q)
        ops.add(int(acts.op_type[0]))
    ck("op_legality_override honoured", ops <= {OP_QUANT, OP_END},
       f"ops={sorted(ops)}")

    g = eqx.filter_grad(lambda p: p.sample(ctx, f, tables, jrand.PRNGKey(5))[1])(pol)
    gn = float(jnp.linalg.norm(ravel_pytree(eqx.filter(g, eqx.is_inexact_array))[0]))
    ck("gradient flows", np.isfinite(gn) and gn > 0, f"|grad|={gn:.4g}")
    try:
        fj = eqx.filter_jit(lambda p, c, k: p.sample(c, f, tables, k)[1])
        _ = fj(pol, ctx, jrand.PRNGKey(9))
        ck("jit-able", True)
    except Exception as e:
        ck("jit-able", False, f"{type(e).__name__}: {str(e)[:60]}")

    nparam = sum(x.size for x in
                 jax.tree_util.tree_leaves(eqx.filter(pol, eqx.is_inexact_array)))
    print(f"\n  head parameters: {nparam:,}")
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
