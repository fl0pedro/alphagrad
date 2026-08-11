"""#92: base-stream tokens must NOT be credited to a vertex slot.

``eqns`` from the tokenizer are per-token equation-SEGMENT ids drawn from a
stream-global running counter (graphax.jaxpr.last_eqn_ids: "built for
relational consumers that only compare ids"). They are NOT vertex indices.
``base_tokens`` emits the whole base equation block in ONE ``_emit_eqns``
call, so every base equation token shares the counter start value, 0.

``init_carry`` used to read that as a vertex index, which credited the entire
base stream to vertex 1 (measured 298/323 rows on nn256) and left every other
vertex slot empty forever -- making vertex 1 the only distinguishable vertex
at the root of every episode, in BOTH trainers.

These tests need no environment, no measurement and no real graph: they drive
``init_carry``/``advance`` on synthetic rows, so they run in well under a
second and pin the attribution contract directly.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx import vertex_memory as _vmem
from alphagrad.approx.common import carry_stream as cs

TOTAL_V = 6
EMBD = 4
WINDOW = 12


class _StubAgent:
    """Minimal encode_extend: one unit row per token, no learned state."""

    def carry_init(self):
        return jnp.zeros((), jnp.float32)

    def encode_extend(self, carry, tokens, eqns, count, *, window, start,
                      chunk=None, budget=None):
        n = tokens.shape[0]
        rows = jnp.ones((n, EMBD), jnp.float32)
        valid = (jnp.arange(n) < count).astype(jnp.float32)
        return carry + 1.0, rows, valid, eqns


def _base_block(seg_id=0, n_eqn=9, n_hdr=3):
    """A base block shaped like the real one: header rows at -1, the rest
    sharing ONE segment id (what base_tokens actually emits)."""
    eqns = np.full((WINDOW,), -1, np.int32)
    eqns[n_hdr:n_hdr + n_eqn] = seg_id
    toks = np.ones((WINDOW,), np.int32)
    return (jnp.asarray(toks), jnp.asarray(eqns),
            jnp.asarray(n_hdr + n_eqn, jnp.int32))


def test_base_tokens_never_land_in_a_vertex_slot():
    toks, eqns, count = _base_block(seg_id=0)
    _, sums, counts = cs.init_carry(
        _StubAgent(), toks, eqns, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD)
    counts = np.asarray(counts)
    assert counts[:TOTAL_V].sum() == 0.0, (
        f"base rows leaked into vertex slots: {counts[:TOTAL_V]}")
    assert counts[TOTAL_V] == float(count), (
        "every consumed base row belongs to the global slot")
    assert float(jnp.linalg.norm(jnp.asarray(sums)[0])) == 0.0


@pytest.mark.parametrize("seg_id", [0, 1, 3, TOTAL_V - 1])
def test_attribution_is_independent_of_the_segment_counter(seg_id):
    """The counter value must not steer rows into a slot. A base block whose
    segment id happens to equal a valid vertex index is the exact case the
    old code got wrong."""
    toks, eqns, count = _base_block(seg_id=seg_id)
    _, _, counts = cs.init_carry(
        _StubAgent(), toks, eqns, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD)
    assert np.asarray(counts)[:TOTAL_V].sum() == 0.0


def test_advance_still_credits_the_owning_vertex():
    """The delta path is the one place a vertex attribution IS known, and it
    must keep working: `advance` uses `owner`, not the segment id."""
    toks, eqns, count = _base_block()
    carry, sums, counts = cs.init_carry(
        _StubAgent(), toks, eqns, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD)

    d_eqns = np.full((WINDOW,), -1, np.int32)
    d_eqns[:4] = 7                      # a segment id unrelated to the owner
    owner = 2
    _, sums2, counts2 = cs.advance(
        _StubAgent(), carry, sums, counts,
        jnp.ones((WINDOW,), jnp.int32), jnp.asarray(d_eqns),
        jnp.asarray(4, jnp.int32), owner, window=WINDOW)

    delta = np.asarray(counts2) - np.asarray(counts)
    assert delta[owner] == 4.0, f"owner slot did not receive the delta: {delta}"
    assert delta[TOTAL_V] == 0.0, "delta rows must not reach the global slot"
    assert delta.sum() == 4.0, "no rows may be double-counted"


def test_summary_is_unchanged_by_the_attribution():
    """Where rows land must not change the token mean the value heads read --
    the slot sums re-add to the same total (vertex_memory.summary)."""
    toks, eqns, count = _base_block()
    _, sums, counts = cs.init_carry(
        _StubAgent(), toks, eqns, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD)
    got = np.asarray(_vmem.summary(sums, counts))
    np.testing.assert_allclose(got, np.ones(EMBD), rtol=0, atol=1e-6)
