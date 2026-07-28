"""Equiprobable five-way start: {skip, diag, compress, quant, none}.

v17 began with p(END) ~= 0.76-0.97 and p(DIAG) ~= 1e-7 — the only
structure-preserving operator was effectively absent from the action space
before a single gradient step, and later underflowed to exactly 0.0 (softmax
gradients scale with p, so p ~ 1e-30 cannot recover).

The fix is an unbiased PRIOR, not a probability floor:
  * OpTypeHead weight+bias zeroed  -> masked softmax uniform over LEGAL ops
  * skip_head output bias = logit(0.2) -> p_skip = 1/5
so the five composite outcomes each start at 0.2.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alphagrad.approx.heads import NUM_OPS, OP_COMPRESS, OpTypeHead


def test_op_head_uniform_when_all_legal():
    h = OpTypeHead(32, key=jr.PRNGKey(0))
    for seed in (1, 7, 99):
        d = np.asarray(h(jr.normal(jr.PRNGKey(seed), (32,)),
                         jnp.ones((NUM_OPS,))))
        np.testing.assert_allclose(d, 1.0 / NUM_OPS, atol=1e-6)


def test_op_head_uniform_over_legal_subset():
    """Masking must renormalise, not skew: 3 legal ops -> 1/3 each."""
    h = OpTypeHead(32, key=jr.PRNGKey(0))
    mask = jnp.ones((NUM_OPS,)).at[OP_COMPRESS].set(0.0)
    d = np.asarray(h(jr.normal(jr.PRNGKey(2), (32,)), mask))
    assert d[OP_COMPRESS] < 1e-8
    live = [i for i in range(NUM_OPS) if i != OP_COMPRESS]
    np.testing.assert_allclose(d[live], 1.0 / len(live), atol=1e-6)


def test_op_head_input_independent_at_init():
    """A zero-init head must ignore its input entirely at step 0."""
    h = OpTypeHead(32, key=jr.PRNGKey(0))
    a = np.asarray(h(jnp.zeros((32,)), jnp.ones((NUM_OPS,))))
    b = np.asarray(h(jnp.full((32,), 50.0), jnp.ones((NUM_OPS,))))
    np.testing.assert_array_equal(a, b)


def test_skip_gate_starts_at_one_fifth():
    from alphagrad.approx.heads import FacePathPolicy
    fp = FacePathPolicy(embd_dim=32, num_heads=2, max_faces=4, num_slots=3,
                        key=jr.PRNGKey(3))
    p = float(jnn.sigmoid(fp.skip_head(jnp.zeros((32,)))[0]))
    assert abs(p - 0.2) < 1e-3, p


def test_five_way_composite_is_uniform():
    """skip + (keep x 4 ops) must give 0.2 to each of the five outcomes."""
    from alphagrad.approx.heads import FacePathPolicy
    fp = FacePathPolicy(embd_dim=32, num_heads=2, max_faces=4, num_slots=3,
                        key=jr.PRNGKey(5))
    p_skip = float(jnn.sigmoid(fp.skip_head(jnp.zeros((32,)))[0]))
    op = np.asarray(fp.head.op_head(jnp.zeros((32,)), jnp.ones((NUM_OPS,))))
    outcomes = np.concatenate([[p_skip], (1.0 - p_skip) * op])
    np.testing.assert_allclose(outcomes, 0.2, atol=2e-3)
    np.testing.assert_allclose(outcomes.sum(), 1.0, atol=1e-6)


def test_env_flag_restores_learned_init():
    """ALPHAGRAD_OP_HEAD_UNIFORM=0 must give back the old random init."""
    os.environ["ALPHAGRAD_OP_HEAD_UNIFORM"] = "0"
    try:
        h = OpTypeHead(32, key=jr.PRNGKey(0))
        d = np.asarray(h(jr.normal(jr.PRNGKey(1), (32,)), jnp.ones((NUM_OPS,))))
        assert not np.allclose(d, 1.0 / NUM_OPS, atol=1e-6)
    finally:
        os.environ["ALPHAGRAD_OP_HEAD_UNIFORM"] = "1"
