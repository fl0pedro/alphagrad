#!/usr/bin/env python3
"""Tests for the unified approx head and the set-transformer pointer."""
from __future__ import annotations
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, "/Users/assmuth/audit")
from unified_head import (  # noqa: E402
    HEAD_WIDTH, MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES, NUM_REDUCE_FNS,
    N_PRIME_GATES, OP_BLOCKDIAG, OP_NONE, OP_QUANT, OP_REDUCE, PRIMES,
    PRIME_GATES, UnifiedApproxHead)
from set_pointer import SetPointerVertexPolicy  # noqa: E402

FAIL = []


def ck(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def main():
    key = jrand.PRNGKey(0)
    E, V = 32, 13

    # ================= unified head =================
    print("=== UnifiedApproxHead ===")
    ck("layout width is 64", HEAD_WIDTH == 64, f"got {HEAD_WIDTH}")
    ck("prime gates sum to 32", N_PRIME_GATES == 32, f"got {N_PRIME_GATES}")
    ck("field widths",
       (1 + NUM_APPROX_OPS + 2 * MAX_PAIR_IDX + N_PRIME_GATES
        + NUM_REDUCE_AXES + NUM_REDUCE_FNS + 1) == HEAD_WIDTH)

    head = UnifiedApproxHead(E, key=key)
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))
    a = head(ctx, jrand.PRNGKey(2))
    ck("emits all fields", a.factor.shape == () and a.exps.shape == (7,)
       and a.axes.shape == (NUM_REDUCE_AXES,))
    ck("i,j in 1..6", 1 <= int(a.i) <= 6 and 1 <= int(a.j) <= 6,
       f"i={int(a.i)} j={int(a.j)}")
    ck("factor is a prime product",
       int(a.factor) == int(np.prod(np.array(PRIMES) ** np.asarray(a.exps))),
       f"factor={int(a.factor)} exps={list(np.asarray(a.exps))}")

    # i != j across many samples
    bad = 0
    for s in range(300):
        aa = head(ctx, jrand.PRNGKey(100 + s))
        if int(aa.i) == int(aa.j):
            bad += 1
    ck("i != j always", bad == 0, f"{bad}/300 collisions")

    # exponent ranges respect the gate caps
    mx = np.zeros(7, dtype=int)
    for s in range(400):
        aa = head(ctx, jrand.PRNGKey(5000 + s))
        mx = np.maximum(mx, np.asarray(aa.exps))
    caps = np.array([n for _, n in PRIME_GATES])
    ck("exponents within gate caps", bool((mx <= caps).all()),
       f"max={list(mx)} caps={list(caps)}")

    # op masking is honoured
    only_reduce = jnp.array([0.0, 1.0, 0.0, 0.0])
    # skip=True legitimately rewrites op to NONE, so only inspect ACTED samples
    _acts = [head(ctx, jrand.PRNGKey(900 + s), op_mask=only_reduce)
             for s in range(200)]
    ops = {int(a.op) for a in _acts if not bool(a.skip)}
    ck("op_mask honoured (acted samples)", ops <= {OP_REDUCE}, f"sampled {ops}")

    # axis mask is honoured
    am = jnp.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0])
    viol = 0
    for s in range(80):
        aa = head(ctx, jrand.PRNGKey(1500 + s), axis_mask=am)
        if bool(jnp.any(aa.axes & (am < 0.5))):
            viol += 1
    ck("axis_mask honoured", viol == 0, f"{viol} violations")

    # multiple axes can fire at once (the point of independent gates)
    multi = max(int(jnp.sum(head(ctx, jrand.PRNGKey(2000 + s)).axes))
                for s in range(200))
    ck("reduce can pick multiple axes", multi >= 2, f"max axes set={multi}")

    # force_pure_diag overrides the factor
    hp = UnifiedApproxHead(E, force_pure_diag=True, key=key)
    ap = hp(ctx, jrand.PRNGKey(7), max_factor=jnp.asarray(64, jnp.int32))
    ck("force_pure_diag takes max factor", int(ap.factor) == 64,
       f"got {int(ap.factor)}")

    # sampled factors must be CLAMPED to the largest legal one
    mf = jnp.asarray(64, jnp.int32)
    over = 0
    for s in range(200):
        aa = head(ctx, jrand.PRNGKey(6000 + s), max_factor=mf)
        if int(aa.factor) > 64:
            over += 1
    ck("factor clamped to max_factor", over == 0, f"{over}/200 exceeded")

    # gcd/forced factor and the sampled prime factor can AGREE -> both rewarded
    hp2 = UnifiedApproxHead(E, force_pure_diag=True, key=key)
    ag = [hp2(ctx, jrand.PRNGKey(7000 + s), max_factor=jnp.asarray(2, jnp.int32))
          for s in range(300)]
    n_agree = sum(1 for a in ag if bool(a.factor_agrees))
    ck("agreement flag fires when sampled == forced", n_agree > 0,
       f"{n_agree}/300 agreed on factor=2")
    ck("forced factor still applied", all(int(a.factor) == 2 for a in ag))
    # prime-gate logp must NOT be dropped by the override
    lp_forced = [float(a.log_prob) for a in ag if not bool(a.skip)
                 and int(a.op) == OP_BLOCKDIAG]
    ck("prime gates still trained under force_pure_diag",
       len(lp_forced) > 0 and all(l < -0.5 for l in lp_forced),
       f"n={len(lp_forced)} max_logp={max(lp_forced) if lp_forced else 0:.3f}")

    # skip zeroes the op, and branch masking keeps log_prob finite
    lps = [float(head(ctx, jrand.PRNGKey(3000 + s)).log_prob) for s in range(200)]
    ck("log_prob finite", all(np.isfinite(lps)), f"min={min(lps):.3f}")
    ents = [float(head(ctx, jrand.PRNGKey(3000 + s)).entropy) for s in range(200)]
    ck("entropy finite and >= 0", all(np.isfinite(e) and e >= -1e-6 for e in ents),
       f"min={min(ents):.4f} max={max(ents):.4f}")
    sk = [head(ctx, jrand.PRNGKey(4000 + s)) for s in range(200)]
    ck("skip forces op=NONE",
       all(int(x.op) == OP_NONE for x in sk if bool(x.skip)))

    # gradient flows to the head
    def loss(h):
        return h(ctx, jrand.PRNGKey(11)).log_prob
    g = eqx.filter_grad(loss)(head)
    gn = float(jnp.linalg.norm(ravel_pytree(
        eqx.filter(g, eqx.is_inexact_array))[0]))
    ck("gradient flows", np.isfinite(gn) and gn > 0, f"|grad|={gn:.4g}")

    # jit-able
    try:
        f = eqx.filter_jit(lambda h, c, k: h(c, k).factor)
        _ = f(head, ctx, jrand.PRNGKey(12))
        ck("jit-able", True)
    except Exception as e:
        ck("jit-able", False, f"{type(e).__name__}: {str(e)[:70]}")

    # ================= set pointer =================
    print("\n=== SetPointerVertexPolicy ===")
    pol = SetPointerVertexPolicy(num_vertices=V, embd_dim=E, num_heads=2,
                                 num_blocks=2, key=jrand.PRNGKey(3))
    vmem = jrand.normal(jrand.PRNGKey(4), (V + 1, E))
    vmask = jnp.ones((V + 1,))
    lg, reprs = pol.from_vertex_memory(vmem, vmask)
    ck("logits shape (V,)", lg.shape == (V,), f"{lg.shape}")
    ck("contexts shape (V,E)", reprs.shape == (V, E), f"{reprs.shape}")

    # no fixed per-vertex parameter table
    names = [n for n, _ in jax.tree_util.tree_flatten_with_path(
        eqx.filter(pol, eqx.is_inexact_array))[0]]
    ck("no vertex_embedding table",
       not any("vertex_embedding" in str(n) for n in names))

    # PERMUTATION EQUIVARIANCE: relabelling vertices permutes the logits
    perm = np.random.default_rng(0).permutation(V)
    full = np.concatenate([perm, [V]])
    lg2, _ = pol.from_vertex_memory(vmem[full], vmask[full])
    ck("permutation equivariant",
       np.allclose(np.asarray(lg)[perm], np.asarray(lg2), atol=1e-4),
       f"maxdiff={np.max(np.abs(np.asarray(lg)[perm]-np.asarray(lg2))):.2e}")

    # masked slots score -1e9 and never win
    vm2 = vmask.at[3].set(0.0)
    lg3, _ = pol.from_vertex_memory(vmem, vm2)
    ck("masked slot suppressed", float(lg3[3]) < -1e8, f"{float(lg3[3]):.3g}")

    # SIZE AGNOSTIC: the PARAMETER SET must not depend on V at all. Build the
    # same policy for a different vertex count and compare parameter shapes --
    # if any shape moves, some weight is indexed by vertex id (the exact defect
    # the old Embedding(num_vertices, E) table had).
    pol20 = SetPointerVertexPolicy(num_vertices=20, embd_dim=E, num_heads=2,
                                   num_blocks=2, key=jrand.PRNGKey(3))
    sh13 = [tuple(x.shape) for x in
            jax.tree_util.tree_leaves(eqx.filter(pol, eqx.is_inexact_array))]
    sh20 = [tuple(x.shape) for x in
            jax.tree_util.tree_leaves(eqx.filter(pol20, eqx.is_inexact_array))]
    ck("parameter shapes independent of V", sh13 == sh20,
       f"{len(sh13)} tensors, identical={sh13 == sh20}")
    lg4, _ = pol20.from_vertex_memory(
        jrand.normal(jrand.PRNGKey(9), (21, E)), jnp.ones((21,)))
    ck("runs at V=20", lg4.shape == (20,), f"{lg4.shape}")

    # raw-stream path with eqn_ids equals the pooled path
    S = 40
    enc = jrand.normal(jrand.PRNGKey(6), (S, E))
    tok = jnp.ones((S,))
    ids = jnp.asarray(np.random.default_rng(1).integers(0, V + 1, S))
    lg5, _ = pol(enc, tok, eqn_ids=ids)
    sums = jax.ops.segment_sum(enc, ids, num_segments=V + 1)
    cnts = jax.ops.segment_sum(tok, ids, num_segments=V + 1)
    pooled = sums / jnp.maximum(cnts, 1.0)[:, None]
    lg6, _ = pol.from_vertex_memory(pooled, (cnts > 0).astype(enc.dtype))
    ck("stream path == pooled path",
       np.allclose(np.asarray(lg5), np.asarray(lg6), atol=1e-5),
       f"maxdiff={np.max(np.abs(np.asarray(lg5)-np.asarray(lg6))):.2e}")

    gp = eqx.filter_grad(lambda p: jnp.sum(p.from_vertex_memory(vmem, vmask)[0]))(pol)
    gpn = float(jnp.linalg.norm(ravel_pytree(
        eqx.filter(gp, eqx.is_inexact_array))[0]))
    ck("pointer gradient flows", np.isfinite(gpn) and gpn > 0, f"|grad|={gpn:.4g}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
