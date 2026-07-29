#!/usr/bin/env python3
"""The PPO-critical property: sample() and evaluate() must score the SAME
variable. If the round-trip log-prob differs, the importance ratio is wrong and
every update is biased -- so this is the test that decides whether the unified
head can be trained at all.
"""
from __future__ import annotations
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from alphagrad.approx.heads import (
    AxisTokenFeatures, AXIS_TAG_BITS, MAX_PRIMES, OP_COMPRESS, OP_DIAG,
    OP_END, OP_QUANT, precompute_factor_tables)
from alphagrad.approx.unified_micro import UnifiedMicroPolicy

FAIL = []


def ck(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def make_features(n_axes, sizes):
    sz = jnp.asarray(sizes, dtype=jnp.int32)
    return AxisTokenFeatures(
        size=sz,
        log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((n_axes, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n_axes,), jnp.int32),
        valid_mask=jnp.ones((n_axes,), jnp.float32),
    )


def main():
    E, S, N = 32, 16, 8
    tables = precompute_factor_tables(64)
    pol = UnifiedMicroPolicy(embd_dim=E, max_substeps=S, key=jrand.PRNGKey(0))
    feats = make_features(N, [16, 16, 8, 8, 4, 4, 2, 2])
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))

    print("=== round-trip: sample -> evaluate ===")
    diffs, ops_seen = [], set()
    for s in range(300):
        acts, lp, ent, arity, *_ = pol.sample(
            ctx, feats, tables, jrand.PRNGKey(1000 + s))
        lp2, ent2, arity2, *_ = pol.evaluate(ctx, feats, tables, acts)
        diffs.append(abs(float(lp) - float(lp2)))
        ops_seen.add(int(acts.op_type[0]))
    ck("log_prob round-trips exactly", max(diffs) < 1e-4,
       f"max|dlogp|={max(diffs):.3e} over 300 samples")
    ck("all four ops exercised", ops_seen == {OP_DIAG, OP_COMPRESS, OP_QUANT, OP_END},
       f"ops={sorted(ops_seen)}")

    print("\n=== emitted MicroAction is well-formed ===")
    bad_fac, bad_mult, n_multi = 0, 0, 0
    for s in range(300):
        acts, *_ = pol.sample(ctx, feats, tables, jrand.PRNGKey(7000 + s))
        op0 = int(acts.op_type[0])
        if op0 == OP_DIAG:
            i, j = int(acts.i[0]), int(acts.j[0])
            g = int(tables.gcd[int(feats.size[i]), int(feats.size[j])])
            f = int(acts.factor[0])
            if f < 1 or g % f != 0:
                bad_fac += 1
        if op0 == OP_COMPRESS:
            n = int(jnp.sum(acts.op_type == OP_COMPRESS))
            if n > 1:
                n_multi += 1
            # every COMPRESS sub-step must sit in a distinct axis slot
            idx = [int(x) for x, o in zip(acts.i, acts.op_type) if int(o) == OP_COMPRESS]
            if len(set(idx)) != len(idx):
                bad_mult += 1
    ck("DIAG factor always divides gcd", bad_fac == 0, f"{bad_fac}/300 bad")
    ck("COMPRESS axes distinct", bad_mult == 0, f"{bad_mult} dup-axis cases")
    ck("multi-axis COMPRESS emitted", n_multi > 0, f"{n_multi} multi-axis samples")

    print("\n=== masks respected ===")
    only_diag = jnp.array([1.0, 0.0, 0.0, 0.0])
    ops = set()
    for s in range(150):
        acts, *_ = pol.sample(ctx, feats, tables, jrand.PRNGKey(3000 + s),
                              op_legality_override=only_diag)
        ops.add(int(acts.op_type[0]))
    ck("op_legality_override honoured", ops <= {OP_DIAG, OP_END}, f"ops={sorted(ops)}")

    cv = jnp.array([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    viol = 0
    for s in range(150):
        acts, *_ = pol.sample(ctx, feats, tables, jrand.PRNGKey(4000 + s),
                              compress_valid=cv)
        for x, o in zip(acts.i, acts.op_type):
            if int(o) == OP_COMPRESS and int(x) > 1:
                viol += 1
    ck("compress_valid honoured", viol == 0, f"{viol} violations")

    print("\n=== gradient + jit ===")
    def loss(p):
        return p.sample(ctx, feats, tables, jrand.PRNGKey(5))[1]
    g = eqx.filter_grad(loss)(pol)
    from jax.flatten_util import ravel_pytree
    gn = float(jnp.linalg.norm(ravel_pytree(eqx.filter(g, eqx.is_inexact_array))[0]))
    ck("gradient flows through sample", np.isfinite(gn) and gn > 0, f"|grad|={gn:.4g}")

    def eloss(p):
        acts, *_ = p.sample(ctx, feats, tables, jrand.PRNGKey(6))
        return p.evaluate(ctx, feats, tables, acts)[0]
    ge = eqx.filter_grad(eloss)(pol)
    gen = float(jnp.linalg.norm(ravel_pytree(eqx.filter(ge, eqx.is_inexact_array))[0]))
    ck("gradient flows through evaluate", np.isfinite(gen) and gen > 0,
       f"|grad|={gen:.4g}")

    try:
        f = eqx.filter_jit(lambda p, c, k: p.sample(c, feats, tables, k)[1])
        _ = f(pol, ctx, jrand.PRNGKey(9))
        ck("jit-able", True)
    except Exception as e:
        ck("jit-able", False, f"{type(e).__name__}: {str(e)[:70]}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
