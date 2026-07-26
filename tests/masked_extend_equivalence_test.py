"""The masked fixed-shape extend must match the variable-length one EXACTLY.

Not "closely" — exactly. The PPO ratio-1 invariant requires the rollout encode
and the loss-time re-encode to produce identical distributions on the first
epoch; a 1e-7 drift in enc_x becomes a ratio != 1 and a spurious first-epoch
gradient. So these assert bitwise equality on the valid prefix, and that pad
steps leave the palimpsa carry untouched.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alphagrad.approx.incremental_encoder import (
    _extend_block,
    _extend_block_masked,
    init_state,
)
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent


def _agent(seed=0):
    return MicroPPOAgent(vocab_size=512, embd_dim=32, num_layers=2,
                         num_heads=2, hidden_dim=64, num_vertices=16,
                         value_dims=[32], key=jr.PRNGKey(seed),
                         max_substeps=1, policy="palimpsa")


def _carries(agent):
    return init_state(agent).carries


def test_valid_prefix_is_bitwise_identical():
    agent = _agent()
    c0 = _carries(agent)
    n, D = 5, 12
    toks = jnp.arange(1, n + 1, dtype=jnp.int32)
    pos = jnp.arange(n, dtype=jnp.int32)

    ref_carry, ref_rows = _extend_block(agent, c0, toks, pos)

    padded = jnp.pad(toks, (0, D - n))
    pos_p = jnp.pad(pos, (0, D - n))
    valid = jnp.arange(D) < n
    got_carry, got_rows = _extend_block_masked(agent, c0, padded, pos_p, valid)

    np.testing.assert_array_equal(
        np.asarray(ref_rows), np.asarray(got_rows[:n]),
        err_msg="rows drifted — ratio-1 would break")
    for (rm, ri), (gm, gi) in zip(ref_carry, got_carry):
        np.testing.assert_array_equal(np.asarray(rm), np.asarray(gm))
        np.testing.assert_array_equal(np.asarray(ri), np.asarray(gi))


def test_pad_steps_emit_zero_rows():
    agent = _agent()
    n, D = 4, 10
    toks = jnp.pad(jnp.arange(1, n + 1, dtype=jnp.int32), (0, D - n))
    pos = jnp.pad(jnp.arange(n, dtype=jnp.int32), (0, D - n))
    valid = jnp.arange(D) < n
    _, rows = _extend_block_masked(agent, _carries(agent), toks, pos, valid)
    assert np.all(np.asarray(rows[n:]) == 0.0), "pad rows must be zero"


def test_an_all_pad_call_is_a_no_op_on_the_carry():
    """A step whose delta is empty must not move the recurrence at all."""
    agent = _agent()
    c0 = _carries(agent)
    D = 8
    toks = jnp.zeros(D, jnp.int32)
    pos = jnp.zeros(D, jnp.int32)
    valid = jnp.zeros(D, bool)
    got, rows = _extend_block_masked(agent, c0, toks, pos, valid)
    for (om, oi), (gm, gi) in zip(c0, got):
        np.testing.assert_array_equal(np.asarray(om), np.asarray(gm))
        np.testing.assert_array_equal(np.asarray(oi), np.asarray(gi))
    assert np.all(np.asarray(rows) == 0.0)


def test_chained_deltas_match_one_long_encode():
    """THE property the append-only path rests on: encoding a stream in
    chunks equals encoding it in one pass."""
    agent = _agent()
    c0 = _carries(agent)
    total = 9
    toks = jnp.arange(1, total + 1, dtype=jnp.int32)
    pos = jnp.arange(total, dtype=jnp.int32)
    ref_carry, ref_rows = _extend_block(agent, c0, toks, pos)

    D = 6
    carry = c0
    chunks = [(0, 4), (4, 7), (7, 9)]
    rows_out = []
    for lo, hi in chunks:
        k = hi - lo
        t = jnp.pad(toks[lo:hi], (0, D - k))
        p = jnp.pad(pos[lo:hi], (0, D - k))
        v = jnp.arange(D) < k
        carry, r = _extend_block_masked(agent, carry, t, p, v)
        rows_out.append(r[:k])
    got_rows = jnp.concatenate(rows_out, axis=0)

    np.testing.assert_array_equal(np.asarray(ref_rows), np.asarray(got_rows))
    for (rm, ri), (gm, gi) in zip(ref_carry, carry):
        np.testing.assert_array_equal(np.asarray(rm), np.asarray(gm))
        np.testing.assert_array_equal(np.asarray(ri), np.asarray(gi))
