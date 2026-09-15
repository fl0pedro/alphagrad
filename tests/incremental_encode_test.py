# -*- coding: utf-8 -*-
"""Phase 3b — incremental autoregressive encode (--incremental-encode).

What ratio-1 actually rests on, tested directly:

1. CHUNKED EXTEND IS EXACT: extending the carry by a stream in two calls
   equals one call, bitwise, for the same window size. This is the identity
   the loss relies on when it re-derives a step's encoding from the stored
   pre-step carry + delta (the rollout consumed the same tokens through the
   same window-shaped program).
2. The property survives a LONGER split (the carry is the whole state; there
   is nothing recomputed from the whole stream). The relational-gate case
   this used to be -- a nonzero rel_gate driven by a causal equation-id
   histogram -- is gone: the gate and the ids were removed on 2026-09-13.
3. VERTEX-MEMORY FOLD IS ASSOCIATIVE bitwise (sums/counts, explicit ids).
4. heads_from_memory is deterministic and shape-correct, and feeding the
   same triple through sample/evaluate's `precomputed` hook yields the
   same vertex distribution.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from alphagrad.approx.ppo import (
    EncCarry,
    NUM_VALUE_HEADS,
    _build_agent,
    make_argparser,
)
from alphagrad.approx import vertex_memory as vmem


@pytest.fixture(autouse=True)
def _exact_read():
    """THIS MODULE PINS A PROPERTY OF THE EXACT OPERATOR, so it forces
    ``ALPHAGRAD_PALIMPSA_READ=exact`` even though the shipped default is now
    ``fast``.

    The claim is that a chunked extend split at an ARBITRARY token (100 of 300,
    130 of 250) is bitwise the same as one call. Under the fast read a split
    that is not on a multiple of 32 starts a fresh chunk, so the two sides read
    different chunks on purpose. That is the operator, not a defect.

    The fast read is pinned separately, on the properties it does have, in
    ``tests/fast_read_extend_parity_test.py``.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
        yield


TOTAL_V = 5
EMBD = 16


def _agent(key=None):
    args = make_argparser().parse_args([])
    args.dynamic_substeps = True
    args.vocab_size = 64
    args.embd_dim = EMBD
    args.num_heads = 2
    args.num_layers = 2
    args.hidden_dim = 32
    return _build_agent(
        args, TOTAL_V, num_factors=4, max_rules=4,
        key=key if key is not None else jrand.PRNGKey(0),
    )


def _stream(key, n, max_tokens=512):
    """Append-only-style buffer: n real tokens (ids >= 1), trailing pad 0.

    ONE buffer: the parallel equation-id buffer was removed on 2026-09-13.
    """
    k1, _k2 = jrand.split(key)
    toks = jrand.randint(k1, (n,), 1, 60)
    return jnp.zeros((max_tokens,), jnp.int32).at[:n].set(toks)


def _carry_equal(a, b):
    for name in EncCarry._fields:
        na, nb = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert np.array_equal(na, nb), f"carry field {name} differs"


def test_chunked_extend_bitwise():
    agent = _agent()
    toks = _stream(jrand.PRNGKey(1), 300)
    W = 512

    c0 = agent.carry_init()
    one, rows_one, valid_one = agent.encode_extend(c0, toks, 300, window=W)

    mid, rows_a, valid_a = agent.encode_extend(c0, toks, 100, window=W)
    assert int(mid.pos) == 100
    two, rows_b, valid_b = agent.encode_extend(mid, toks, 200, window=W)
    assert int(two.pos) == 300

    _carry_equal(one, two)
    ra = np.asarray(rows_one)[:300]
    rb = np.concatenate([np.asarray(rows_a)[:100], np.asarray(rows_b)[:200]])
    assert np.array_equal(ra, rb), "chunked rows differ from one-shot rows"
    # Rows past the valid prefix are hard zeros (they must not leak into
    # the vertex-memory fold even without the valid mask).
    assert np.all(np.asarray(rows_one)[300:] == 0.0)


def test_prefix_property_survives_an_uneven_split():
    """A DIFFERENT split of a DIFFERENT stream, for the same reason: the
    carry IS the whole state, so where the split falls cannot matter.

    This case used to force a nonzero ``rel_gate`` on every layer, so that a
    broken causal histogram could not pass as a zero-initialised no-op. Both
    the gate and the histogram were removed on 2026-09-13 together with the
    equation ids that fed them, so the extra state this was guarding no
    longer exists.
    """
    agent = _agent()
    toks = _stream(jrand.PRNGKey(2), 250)
    W = 512
    c0 = agent.carry_init()
    one, rows_one, _ = agent.encode_extend(c0, toks, 250, window=W)
    mid, rows_a, _ = agent.encode_extend(c0, toks, 130, window=W)
    two, rows_b, _ = agent.encode_extend(mid, toks, 120, window=W)
    _carry_equal(one, two)
    ra = np.asarray(rows_one)[:250]
    rb = np.concatenate([np.asarray(rows_a)[:130], np.asarray(rows_b)[:120]])
    assert np.array_equal(ra, rb)
    assert int(one.pos) == 250


def test_the_encoder_carry_is_three_leaves():
    """``cumhist`` (a (MAX_EQNS,) histogram) and ``nvalid`` are gone."""
    assert EncCarry._fields == ("M", "I", "pos")


def test_vmem_fold_associative_and_heads():
    agent = _agent()
    key = jrand.PRNGKey(3)
    rows = jrand.normal(key, (40, EMBD), jnp.float32)
    ids = jnp.concatenate([
        jnp.full((10,), 0), jnp.full((10,), 2),
        jnp.full((10,), -1), jnp.full((10,), TOTAL_V + 7),  # overflow -> clamped
    ]).astype(jnp.int32)
    valid = jnp.ones((40,), jnp.float32).at[35:].set(0.0)

    s0 = jnp.zeros((TOTAL_V + 1, EMBD), jnp.float32)
    c0 = jnp.zeros((TOTAL_V + 1,), jnp.float32)
    s1, c1 = vmem.update_ids(s0, c0, rows, ids, valid)
    # REPLAY IS BITWISE: the loss never re-folds history — it repeats the
    # SAME single fold on the stored memory. That's the ratio-1 invariant.
    s1r, c1r = vmem.update_ids(s0, c0, rows, ids, valid)
    assert np.array_equal(np.asarray(s1), np.asarray(s1r))
    assert np.array_equal(np.asarray(c1), np.asarray(c1r))
    # Chunked folding agrees up to float addition order WITHIN a segment
    # (segment_sum reduction order changes when a segment straddles the
    # chunk boundary — vertex_memory's documented caveat, irrelevant to
    # ratio-1 because production replays identical chunking).
    sa, ca = vmem.update_ids(s0, c0, rows[:17], ids[:17], valid[:17])
    sb, cb = vmem.update_ids(sa, ca, rows[17:], ids[17:], valid[17:])
    assert np.allclose(np.asarray(s1), np.asarray(sb), rtol=0, atol=1e-5)
    assert np.array_equal(np.asarray(c1), np.asarray(cb))  # counts are exact
    # invalid rows contributed nothing
    assert float(jnp.sum(c1)) == 35.0

    vl, vc, val = agent.heads_from_memory(s1, c1)
    assert vl.shape == (TOTAL_V,)
    assert vc.shape == (TOTAL_V, EMBD)
    assert val.shape == (NUM_VALUE_HEADS,)
    vl2, vc2, val2 = agent.heads_from_memory(s1, c1)
    assert np.array_equal(np.asarray(vl), np.asarray(vl2))  # deterministic


def test_precomputed_hook_consistency():
    """sample_action_dynamic and evaluate_action_dynamic given the SAME
    precomputed triple produce the SAME masked vertex distribution."""
    from alphagrad.approx.env import MAX_AXES_PER_VERTEX
    from alphagrad.approx.heads import precompute_factor_tables

    agent = _agent()
    s = jnp.abs(jrand.normal(jrand.PRNGKey(4), (TOTAL_V + 1, EMBD)))
    c = jnp.ones((TOTAL_V + 1,), jnp.float32) * 3.0
    pre = agent.heads_from_memory(s, c)

    ax_state = jnp.zeros((TOTAL_V, MAX_AXES_PER_VERTEX, 4), jnp.int32)
    ax_state = ax_state.at[..., 0].set(8)
    ax_mask = jnp.zeros((TOTAL_V, MAX_AXES_PER_VERTEX), jnp.float32).at[:, :2].set(1.0)
    ft = precompute_factor_tables(64)
    avail = jnp.ones((TOTAL_V,), jnp.float32)
    op_override = jnp.ones((4,), jnp.float32)
    toks = jnp.zeros((64,), jnp.int32)

    out = agent.sample_action_dynamic(
        toks, avail, ax_state, ax_mask, ft, op_override,
        jrand.PRNGKey(5), precomputed=pre,
    )
    vertex_idx, actions, vertex_dist = out[0], out[1], out[2]
    ev = agent.evaluate_action_dynamic(
        toks, vertex_idx, actions, avail, ax_state, ax_mask, ft,
        jrand.PRNGKey(6), op_legality_override=op_override, precomputed=pre,
    )
    ev_dist = ev[3]
    assert np.array_equal(np.asarray(vertex_dist), np.asarray(ev_dist))


if __name__ == "__main__":
    test_chunked_extend_bitwise()
    test_prefix_property_nonzero_relgate()
    test_vmem_fold_associative_and_heads()
    test_precomputed_hook_consistency()
    print("ALL PASS")
