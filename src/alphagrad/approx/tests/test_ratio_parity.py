#!/usr/bin/env python3
"""T1: is the PPO ratio well-behaved for the unified approximation head?

Existing coverage (test_unified_head32.py) already pins sample<->evaluate
parity under IDENTICAL weights and masks: max|dlogp| = 0.000e+00. So a plain
parity bug is ruled out numerically, not by reading. What that test does NOT
cover is the quantity PPO actually forms:

    ratio = exp(logp_new - logp_old)

with logp_new from an UPDATED policy. The unified head's log-prob is a SUM over
up to 13 independent decisions -- skip, op, i, j, NINE axis Bernoulli gates,
reduce-fn, dtype -- whereas the --no-approx-head control's is a single vertex
categorical. If each term shifts slightly under one gradient step, the log-ratio
accumulates ~13 shifts and exp() of that can be enormous. That would explain a
1.77e8 PPO loss at EPISODE 0 with healthy per-component KLs (each term moves a
little; only the SUM is large) and it is consistent with the two runs differing
by exactly one flag.

So this measures three things:
  A. parity at identical weights          -> ratio must be exactly 1
  B. log-prob magnitude and term count    -> how many terms enter the sum
  C. ratio after a small weight perturbation, unified vs vertex-only
     -> does the ratio explode, and does it explode MORE for the head?
"""
from __future__ import annotations
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np

from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
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


def perturb(pol, key, scale):
    """One SGD-sized step in a random direction — a stand-in for an update."""
    params, static = eqx.partition(pol, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jrand.split(key, len(leaves))
    new = [l + scale * jrand.normal(k, l.shape) for l, k in zip(leaves, keys)]
    return eqx.combine(jax.tree_util.tree_unflatten(treedef, new), static)


def main():
    E, S = 32, 16
    tables = precompute_factor_tables(64)
    pol = UnifiedMicroPolicy(embd_dim=E, max_substeps=S, key=jrand.PRNGKey(0))
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))
    f = feats([8, 8, 4, 3, 16, 5, 8, 4])

    print("=== A. parity at identical weights (ratio must be exactly 1) ===")
    d = []
    for s in range(300):
        acts, lp, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(s))
        lp2, *_ = pol.evaluate(ctx, f, tables, acts)
        d.append(abs(float(lp) - float(lp2)))
    ck("sample==evaluate logp", max(d) < 1e-6, f"max|dlogp|={max(d):.3e}")

    print("\n=== B. how many terms enter the log-prob sum ===")
    lps = []
    for s in range(300):
        _, lp, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(s))
        lps.append(float(lp))
    lps = np.array(lps)
    print(f"  logp: mean={lps.mean():.3f} min={lps.min():.3f} max={lps.max():.3f} "
          f"std={lps.std():.3f}")
    ck("logp magnitude is bounded", abs(lps.mean()) < 50,
       f"mean {lps.mean():.2f} (a sum over ~13 decisions)")

    print("\n=== C. ratio after a weight perturbation ===")
    print(f"  {'scale':>8} {'median_r':>12} {'p99_r':>14} {'max_r':>14} "
          f"{'max|dlogp|':>11}")
    blow = None
    for scale in (1e-4, 1e-3, 1e-2, 1e-1):
        pol2 = perturb(pol, jrand.PRNGKey(7), scale)
        rs, dl = [], []
        for s in range(300):
            acts, lp_old, *_ = pol.sample(ctx, f, tables, jrand.PRNGKey(s))
            lp_new, *_ = pol2.evaluate(ctx, f, tables, acts)
            dd = float(lp_new) - float(lp_old)
            dl.append(abs(dd))
            rs.append(float(np.exp(np.clip(dd, -700, 700))))
        rs = np.array(rs)
        print(f"  {scale:>8.0e} {np.median(rs):>12.4f} "
              f"{np.percentile(rs, 99):>14.4g} {rs.max():>14.4g} "
              f"{max(dl):>11.3f}")
        if blow is None and rs.max() > 1e3:
            blow = (scale, rs.max())
    if blow:
        ck("ratio stays below 1e3 for small steps", False,
           f"blew up at scale={blow[0]:.0e}, max ratio {blow[1]:.3g}")
    else:
        ck("ratio stays below 1e3 for small steps", True)

    print("\n=== C2. same test, VERTEX-ONLY log-prob (the healthy control) ===")
    print("  the control's ratio comes from ONE categorical, not a 13-term sum")
    for scale in (1e-3, 1e-2, 1e-1):
        k = jrand.PRNGKey(11)
        logits_old = jrand.normal(k, (13,))
        d2 = jrand.normal(jrand.PRNGKey(12), (13,)) * scale
        p_old = jax.nn.log_softmax(logits_old)
        p_new = jax.nn.log_softmax(logits_old + d2)
        r = np.exp(np.array(p_new - p_old))
        print(f"  {scale:>8.0e} {np.median(r):>12.4f} "
              f"{np.percentile(r, 99):>14.4g} {r.max():>14.4g}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
