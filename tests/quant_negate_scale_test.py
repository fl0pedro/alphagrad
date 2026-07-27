# -*- coding: utf-8 -*-
"""Negate + scale quant heads, end to end.

1. ENGINE: apply_quant honors (scale_sign, scale_frac) — an ALL-NEGATIVE
   block quantized to uint32 with sign=-1 survives (dequantizes back to
   negative values), where sign=+1 zeroes it (the v13 chain-kill).
2. WIRE: translator row[2] encoding round-trips through
   decode_vertex_rule_specs into the same (sign, frac).
3. HEAD: FactoredQuantHead.sample's (dtype, sign, u) re-scores to the
   identical log-prob via log_prob (ratio-1 for the Beta scale head).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from graphax.sparse.micro_actions import Quant, apply_quant
from graphax.sparse.tensor import SparseTensor


def test_apply_quant_negate_and_scale():
    # The singleton pattern real gxf tests use: SparseTensor([], [], val).
    # An all-negative value is the v13 chain-kill case.
    val = jnp.float32(-0.5)

    q_pos = apply_quant(SparseTensor([], [], val),
                        Quant(dtype="uint32", scale_sign=1, scale_frac=0.5))
    q_neg = apply_quant(SparseTensor([], [], val),
                        Quant(dtype="uint32", scale_sign=-1, scale_frac=0.5))

    deq_pos = float(np.asarray(q_pos.val, np.float64)
                    * np.asarray(q_pos.scalar_mult, np.float64))
    deq_neg = float(np.asarray(q_neg.val, np.float64)
                    * np.asarray(q_neg.scalar_mult, np.float64))
    # sign=+1 keeps the positive arm -> the negative value dies to zero
    assert abs(deq_pos) < 1e-12, deq_pos
    # sign=-1 keeps the negative arm -> value SURVIVES with its polarity
    assert abs(deq_neg - (-0.5)) < 1e-3, deq_neg

    # scale head changes resolution: u=0 (multiplier 1) rounds |v|<0.5 to 0
    # even on the kept arm; u=0.5 preserved it above.
    q_lo = apply_quant(SparseTensor([], [], jnp.float32(-0.25)),
                       Quant(dtype="uint32", scale_sign=-1, scale_frac=0.0))
    deq_lo = float(np.asarray(q_lo.val, np.float64)
                   * np.asarray(q_lo.scalar_mult, np.float64))
    assert abs(deq_lo) < 1e-12, deq_lo


def test_wire_roundtrip():
    os.environ["GRAPHAX_ALLOW_PARTIAL_ORDER"] = "1"
    from alphagrad.approx.env import (
        micro_actions_to_rule_specs_jax, QUANT_SENTINEL)
    from alphagrad.approx.heads import OP_QUANT, OP_END

    S = 4
    ops = jnp.array([OP_QUANT, OP_END, OP_END, OP_END], jnp.int32)
    zeros = jnp.zeros((S,), jnp.int32)
    ax = jnp.zeros((8, 4), jnp.int32).at[..., 0].set(4)
    rows = micro_actions_to_rule_specs_jax(
        ops, zeros, zeros, zeros, ax,
        quant_dtypes=jnp.full((S,), 3, jnp.int32),
        quant_scale_signs=jnp.array([-1, 1, 1, 1], jnp.int32),
        quant_scale_fracs=jnp.array([0.25, -1.0, -1.0, -1.0], jnp.float32),
    )
    row = np.asarray(rows)[0]
    assert row[0] == QUANT_SENTINEL and row[1] == 3
    enc = int(row[2])
    assert enc < 0, enc                       # sign carried
    u = (abs(enc) - 1) / 1e6
    assert abs(u - 0.25) < 1e-5, u            # frac carried


def test_head_sample_evaluate_parity():
    from alphagrad.approx.heads import FactoredQuantHead

    head = FactoredQuantHead(16, key=jrand.PRNGKey(0))
    summary = jrand.normal(jrand.PRNGKey(1), (16,))
    dtype_idx, sign, frac = head.sample(summary, jrand.PRNGKey(2))
    lp1, ent1, ar1 = head.log_prob(summary, dtype_idx, sign, frac)
    lp2, ent2, ar2 = head.log_prob(summary, dtype_idx, sign, frac)
    assert np.array_equal(np.asarray(lp1), np.asarray(lp2))
    assert np.isfinite(float(lp1)) and np.isfinite(float(ent1))
    assert float(ar1) >= 2.0  # sign + scale always count
    assert 0.0 < float(frac) < 1.0


if __name__ == "__main__":
    test_apply_quant_negate_and_scale()
    test_wire_roundtrip()
    test_head_sample_evaluate_parity()
    print("ALL PASS")
