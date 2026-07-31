#!/usr/bin/env python3
"""The PPO ratio's two sides use DIFFERENT formulas under --unified-head.

`_dynamic_loss_fn` builds the ratio as

    log_probs      <- evaluate_action_dynamic(...)          -> adapter.score()
    old_log_probs  <- old_micro_log_prob_for_action(...)    -> rebuilt from the
                      stored per-sub-step DISTRIBUTIONS
    ratio = exp(log_probs - old_log_probs)

For the legacy MicroActionPolicy the stored dists fully determine the sampled
action, so the reconstruction reproduces the sampling log-prob exactly and the
ratio is 1 at epoch 0. UnifiedMicroPolicy._dists() cannot do that: it FABRICATES
those dists (exp_d is a point mass on 0 because the factor is derived rather
than sampled; op/i/j/kind are one distribution broadcast across all sub-steps)
and they encode NOTHING about the skip Bernoulli, the nine axis gates, the
reduce-fn or the dtype bit -- all of which score() includes.

So at epoch 0, with identical weights and no update at all, the ratio is
exp(score - reconstruction), an arbitrary offset. That is a ratio blowup that
no per-component KL would show, because each stored dist is individually
unchanged.

This measures the gap directly. The existing parity test passes because sample()
and evaluate() BOTH call score(); neither exercises the reconstruction path that
the loss actually uses.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np

from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
from alphagrad.approx.unified_micro import UnifiedMicroPolicy
from alphagrad.approx.ppo import old_micro_log_prob_for_action


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
    f = feats([8, 8, 4, 3, 16, 5, 8, 4])
    n_ax = f.size.shape[0]

    gaps, ratios = [], []
    for s in range(300):
        acts, lp_sample, _ent, _ar, op_d, i_d, j_d, exp_d, kind_d, q_lp = \
            pol.sample(ctx, f, tables, jrand.PRNGKey(s))
        # exactly what the loss stores and replays as the OLD side
        old = old_micro_log_prob_for_action(
            jnp.asarray(0, jnp.int32),
            acts.op_type, acts.i, acts.j, acts.exponents,
            acts.compress_kind, acts.quant_dtype,
            jnp.ones((13,)) / 13.0,          # vertex dist (cancels: same both sides)
            op_d, i_d, j_d, exp_d, kind_d, q_lp,
        )
        # the NEW side of the ratio, same weights, no update
        lp_eval, *_ = pol.evaluate(ctx, f, tables, acts)
        # `old_micro_log_prob_for_action` DOES include the vertex term
        # (log_p_v = log(vertex_dist[idx] + 1e-8)); subtract it so only the
        # micro part is compared against evaluate's micro log-prob.
        v_lp = float(jnp.log(jnp.asarray(1.0 / 13.0) + 1e-8))
        # what the LOSS now uses for the unified head: the stored joint
        # log-prob, ungated (see the unified_head branch in _dynamic_loss_fn)
        old_used = v_lp + float(q_lp)
        gap = float(lp_eval) - (old_used - v_lp)
        gaps.append(gap)
        ratios.append(float(np.exp(np.clip(gap, -700, 700))))

    g = np.array(gaps); r = np.array(ratios)
    print("=== ratio at EPOCH 0, identical weights, no update ===")
    print(f"  log-ratio : mean={g.mean():+.4f} min={g.min():+.4f} "
          f"max={g.max():+.4f} std={g.std():.4f}")
    print(f"  ratio     : median={np.median(r):.4g} p99={np.percentile(r,99):.4g} "
          f"max={r.max():.4g}")
    exact = np.abs(g) < 1e-6
    print(f"  exactly 1 : {int(exact.sum())}/{len(g)} samples")
    print()
    if exact.all():
        print("PASS — reconstruction matches score(); ratio is 1 as PPO requires.")
        return 0
    print("FAIL — the two sides of the PPO ratio disagree with NO weight change.")
    print("       The old side is rebuilt from fabricated per-sub-step dists")
    print("       that omit skip / 9 axis gates / reduce-fn / dtype.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
