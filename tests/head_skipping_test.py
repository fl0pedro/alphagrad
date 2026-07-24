"""Head-skipping: a head contributes to the joint log-prob / entropy / arity
ONLY when it faced a real choice — its legality mask left >= 2 options.

A head with exactly 1 legal option is FORCED (its masked softmax is a point
mass, carrying no information and no gradient); one with 0 is empty. Both are
skipped, so a forced action is *taken* but never inflates the PPO ratio or the
entropy-normalisation denominator. This is asserted through the ``arity``
(the emitted-component count), which the gate drives directly.
"""
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.approx.heads import (
    MAX_PRIMES, NUM_QUANT_DTYPES, OP_DIAG, OP_END, OP_QUANT, MicroAction,
    MicroActionHead, precompute_factor_tables)

E, N = 8, 3


def _bits(key=jrand.PRNGKey(0)):
    head = MicroActionHead(E, key=key)
    tables = precompute_factor_tables(8)
    summary = jnp.zeros((E,))
    axis_tokens = jnp.zeros((N, E))
    axis_sizes = jnp.array([4, 4, 4], dtype=jnp.int32)
    return head, tables, summary, axis_tokens, axis_sizes


def _action(op, i=0, j=0, quant=0):
    return MicroAction(
        op_type=jnp.array(op, jnp.int32),
        i=jnp.array(i, jnp.int32), j=jnp.array(j, jnp.int32),
        exponents=jnp.zeros((MAX_PRIMES,), jnp.int32),
        factor=jnp.array(0, jnp.int32),
        compress_kind=jnp.array(0, jnp.int32),
        quant_dtype=jnp.array(quant, jnp.int32),
    )


def _arity(head, tables, s, at, sz, action, op_mask, i_diag, i_comp, j_mask, q):
    _, _, arity, *_ = head.log_prob_step(
        action, s, at, sz, op_mask, i_diag, i_comp, j_mask, q, tables)
    return float(arity)


def test_forced_end_op_contributes_nothing():
    """When only END is legal (exact / ve_only mode) the op head is forced, so
    the whole sub-step carries no decision and its arity is 0."""
    head, tables, s, at, sz = _bits()
    op_mask = jnp.array([0., 0., 0., 1.])          # only END legal
    z = jnp.zeros(N)
    a = _arity(head, tables, s, at, sz, _action(OP_END),
               op_mask, z, z, jnp.zeros((N, N)), jnp.ones(NUM_QUANT_DTYPES))
    assert a == 0.0, "a forced-END sub-step must contribute 0 to the arity"


def test_forced_quant_dtype_is_skipped_but_op_still_counts():
    """QUANT with exactly one legal dtype is forced — the dtype head is skipped
    (this is the int4-style 'only one legal quantization' case). The op head
    still counts because DIAG/QUANT/END gave it a real choice."""
    head, tables, s, at, sz = _bits()
    op_mask = jnp.array([0., 0., 1., 1.])          # QUANT + END -> op is a real choice
    z, zz = jnp.zeros(N), jnp.zeros((N, N))
    one_dtype = jnp.zeros(NUM_QUANT_DTYPES).at[0].set(1.)
    two_dtypes = jnp.zeros(NUM_QUANT_DTYPES).at[:2].set(1.)
    forced = _arity(head, tables, s, at, sz, _action(OP_QUANT, quant=0),
                    op_mask, z, z, zz, one_dtype)
    free = _arity(head, tables, s, at, sz, _action(OP_QUANT, quant=0),
                  op_mask, z, z, zz, two_dtypes)
    assert forced == 1.0, "only the op head counts when the dtype is forced"
    assert free == 2.0, "op + dtype count when >1 dtype is legal"
    assert free - forced == 1.0


def test_forced_j_partner_is_skipped():
    """A DIAG whose chosen ``i`` has exactly one legal partner forces ``j`` —
    the j head is skipped (the coupled-``i`` case, now uniform with every other
    forced head)."""
    head, tables, s, at, sz = _bits()
    op_mask = jnp.array([1., 0., 0., 1.])          # DIAG + END -> op is a real choice
    i_diag = jnp.array([1., 1., 0.])               # i has >= 2 options
    i_comp = jnp.zeros(N)
    q = jnp.ones(NUM_QUANT_DTYPES)
    j_forced = jnp.zeros((N, N)).at[0, 1].set(1.)                  # one partner
    j_free = jnp.zeros((N, N)).at[0, 1].set(1.).at[0, 2].set(1.)   # two partners
    forced = _arity(head, tables, s, at, sz, _action(OP_DIAG, i=0, j=1),
                    op_mask, i_diag, i_comp, j_forced, q)
    free = _arity(head, tables, s, at, sz, _action(OP_DIAG, i=0, j=1),
                  op_mask, i_diag, i_comp, j_free, q)
    assert free - forced == 1.0, "the j head counts only when >1 partner is legal"
