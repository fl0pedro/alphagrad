"""#92: base-stream tokens must NOT be credited to a vertex slot.

``eqns`` from the tokenizer were per-token equation-SEGMENT ids drawn from a
stream-global running counter (graphax.jaxpr.last_eqn_ids: "built for
relational consumers that only compare ids"). They were NOT vertex indices.
``base_tokens`` emits the whole base equation block in ONE ``_emit_eqns``
call, so every base equation token shared the counter start value, 0.

``init_carry`` used to read that as a vertex index, which credited the entire
base stream to vertex 1 (measured 298/323 rows on nn256) and left every other
vertex slot empty forever -- making vertex 1 the only distinguishable vertex
at the root of every episode, in BOTH trainers.

THE IDS THEMSELVES ARE GONE (2026-09-13): they fed only the palimpsa
relational forget gate, which went with them. So the bug cannot be
reintroduced by reading them -- there is nothing to read. What these tests
pin now is the positive rule that replaced it: attribution comes from
``base_owners`` on the base path and from ``owner`` / ``participants`` on the
delta path, and from nothing else.

These tests need no environment, no measurement and no real graph: they drive
``init_carry``/``advance`` on synthetic rows, so they run in well under a
second and pin the attribution contract directly.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest  # noqa: F401

from alphagrad.approx import vertex_memory as _vmem
from alphagrad.approx.common import carry_stream as cs

TOTAL_V = 6
EMBD = 4
WINDOW = 12


class _StubAgent:
    """Minimal encode_extend: one unit row per token, no learned state."""

    def carry_init(self):
        return jnp.zeros((), jnp.float32)

    def encode_extend(self, carry, tokens, count, *, window, start,
                      chunk=None, budget=None):
        n = tokens.shape[0]
        rows = jnp.ones((n, EMBD), jnp.float32)
        valid = (jnp.arange(n) < count).astype(jnp.float32)
        return carry + 1.0, rows, valid


def _base_block(n_eqn=9, n_hdr=3):
    """A base block shaped like the real one. THE SEGMENT IDS ARE GONE (they
    were removed on 2026-09-13); attribution has come from
    ``base_owners`` alone since bug #92, and this file is what pins that."""
    toks = np.ones((WINDOW,), np.int32)
    return jnp.asarray(toks), jnp.asarray(n_hdr + n_eqn, jnp.int32)


def test_base_tokens_never_land_in_a_vertex_slot():
    toks, count = _base_block()
    _, sums, counts = cs.init_carry(
        _StubAgent(), toks, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD, path="rollout")
    counts = np.asarray(counts)
    assert counts[:TOTAL_V].sum() == 0.0, (
        f"base rows leaked into vertex slots: {counts[:TOTAL_V]}")
    assert counts[TOTAL_V] == float(count), (
        "every consumed base row belongs to the global slot")
    assert float(jnp.linalg.norm(jnp.asarray(sums)[0])) == 0.0


def test_attribution_needs_base_owners_and_nothing_else():
    """WITHOUT ``base_owners`` every base row goes to the GLOBAL slot, which
    is the safe #92 behaviour, and there is no second signal that could steer
    it anywhere else. This case used to be parametrised over the segment id
    that could collide with a vertex index; that id no longer exists."""
    toks, count = _base_block()
    own = np.zeros((WINDOW,), np.int32)
    own[3:3 + 9] = 2                      # the OWNER says vertex 2 (1-based)
    _, _, counts = cs.init_carry(
        _StubAgent(), toks, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD,
        base_owners=jnp.asarray(own), path="rollout")
    counts = np.asarray(counts)
    assert counts[1] == 9.0, (
        f"the owner's slot (vertex 2 -> index 1) did not get its rows: "
        f"{counts}")
    assert counts[TOTAL_V] == float(count) - 9.0


def test_advance_still_credits_the_owning_vertex():
    """The delta path is the one place a vertex attribution IS known, and it
    must keep working: `advance` uses `owner`, not the segment id.

    Slot layout is ``total_v + 2`` (carry_stream.zero_memory): 0..V-1 the
    vertices, V the GLOBAL slot, V+1 the SUMMARY slot. Since ad6e2856 every
    row is credited TWICE by design -- once to the slot(s) it participates
    in, once to the SUMMARY slot (exactly once per row), because per-slot
    sums no longer re-add to the token total. So a 4-row delta moves the
    owner slot by 4 AND the summary slot by 4, never the global slot."""
    toks, count = _base_block()
    carry, sums, counts = cs.init_carry(
        _StubAgent(), toks, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD, path="rollout")

    owner = 2
    _, sums2, counts2 = cs.advance(
        _StubAgent(), carry, sums, counts,
        jnp.ones((WINDOW,), jnp.int32),
        jnp.asarray(4, jnp.int32), owner, window=WINDOW, path="rollout")

    delta = np.asarray(counts2) - np.asarray(counts)
    assert delta[owner] == 4.0, f"owner slot did not receive the delta: {delta}"
    # EVERY valid row is the owner's now. The structural half that used to go
    # to the global slot was identified by `delta_eqns < 0`, and the equation
    # ids are gone (see carry_stream.advance).
    assert delta[TOTAL_V] == 0.0, "delta rows must not reach the global slot"
    assert delta[TOTAL_V + 1] == 4.0, (
        f"the SUMMARY slot must be credited exactly once per row: {delta}")
    assert delta.sum() == 8.0, (
        f"each row is credited once to a participant slot and once to the "
        f"SUMMARY slot -- nowhere else: {delta}")


def test_summary_is_unchanged_by_the_attribution():
    """Where rows land must not change the token mean the value heads read --
    the slot sums re-add to the same total (vertex_memory.summary)."""
    toks, count = _base_block()
    _, sums, counts = cs.init_carry(
        _StubAgent(), toks, count,
        window=WINDOW, total_v=TOTAL_V, embd_dim=EMBD, path="rollout")
    got = np.asarray(_vmem.summary(sums, counts))
    np.testing.assert_allclose(got, np.ones(EMBD), rtol=0, atol=1e-6)
