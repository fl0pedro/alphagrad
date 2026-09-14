# -*- coding: utf-8 -*-
"""The rollout extend and the loss extend must read a delta the same way.

WHY THIS FILE EXISTS
--------------------
``ALPHAGRAD_PALIMPSA_READ=fast`` swaps the palimpsa read for a chunked
approximation whose chunk boundary is where it stops approximating. That makes
the CHUNK GRID part of the operator. The rollout walks a delta in blocks of
``ALPHAGRAD_EXTEND_CHUNK``; the loss walks the same delta in blocks of
``ALPHAGRAD_FOLD_CHUNK`` (and, with a budget, of
``ALPHAGRAD_LOSS_EXTEND_CHUNK``). If those two block grids cut a delta at
different places, the two sides compute genuinely different numbers and the
PPO ratio leaves 1 at epoch 0 with no other symptom -- the failure the whole
chunk-alignment rule exists to prevent.

The rule is: the fast grid is measured from token 0 OF THE DELTA and every
outer block must start on a multiple of 32. These tests drive the real
``Agent`` through both paths at several block sizes and check they land on the
same rows, the same carry and the same gradient.

The exact read is checked in the same shape, so a regression that broke only
one of the two settings cannot hide.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "256")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from alphagrad.approx.common import delta_fold as _fold
from alphagrad.approx.common.agent_factory import (
    apply_policy_arch, build_and_init_agent)
from alphagrad.transformer.fast_palimpsa_pallas import CHUNK_C

import alphagrad.approx.ppo as P

WINDOW = 4 * CHUNK_C          # 128 tokens of delta window
COUNT = 2 * CHUNK_C + 19      # a delta that ends mid-chunk, on purpose
SEED = 11
READS = ["exact", "fast"]


@pytest.fixture(scope="module")
def agent():
    """The smallest real ``Agent`` with a palimpsa backbone.

    A stub would not do here: the thing under test is which palimpsa read the
    layer stack uses, and a stub has no layer stack.
    """
    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=True,
        face_actions=False, unified_face_head=False, live_faces=False,
        max_substeps=1, axis_group_embedding=False)
    ns.embd_dim = 32
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    return build_and_init_agent(ns, 6, num_factors=4, max_rules=4, seed=SEED)


@pytest.fixture(scope="module")
def tokens():
    r = np.random.RandomState(SEED)
    return jnp.asarray(r.randint(1, 200, size=WINDOW).astype(np.int32))


def _carry(agent):
    return agent.carry_init()


def _rollout(agent, carry, toks, count, chunk):
    """What the rollout runs: ``encode_extend`` with its own block size."""
    return agent.encode_extend(carry, toks, jnp.asarray(count, jnp.int32),
                               window=WINDOW, start=0, chunk=chunk)


def _rollout_diff(agent, carry, toks, count, chunk):
    """The rollout extend in the form a DIFFERENTIATED caller gets.

    Without a budget the chunked extend is a ``lax.while_loop``, which has no
    transpose rule -- that is the whole reason ``budget`` exists (see
    ``encode_extend``'s docstring). The loss always passes one, so this is the
    form whose gradient has to match the fold's.
    """
    return agent.encode_extend(carry, toks, jnp.asarray(count, jnp.int32),
                               window=WINDOW, start=0, chunk=chunk,
                               budget=jnp.asarray(count, jnp.int32))


def _folded(agent, carry, toks, count, chunk):
    """What the loss runs: ``extend_fold`` reducing the rows as it goes."""
    init, fold = _fold.sum_reducer(agent.embd_dim)
    return _fold.extend_fold(
        agent, carry, toks, jnp.asarray(count, jnp.int32), window=WINDOW,
        chunk=chunk, init_acc=init, fold_fn=fold,
        budget=jnp.asarray(count, jnp.int32))


def _reduce(rows, valid):
    w = jnp.asarray(valid, jnp.float32)
    return jnp.sum(rows * w[:, None], axis=0), jnp.sum(w)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


# --------------------------------------------------------------------------
@pytest.mark.parametrize("read", READS)
def test_the_rollout_extend_and_the_loss_fold_agree_on_the_same_delta(
        read, agent, tokens, monkeypatch):
    """Rows and carry, both paths, at the block sizes production uses."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", read)
    c0 = _carry(agent)
    c_r, rows, valid = _rollout(agent, c0, tokens, COUNT, 2 * CHUNK_C)
    ref_sum, ref_n = _reduce(rows, valid)
    c_f, (got_sum, got_n) = _folded(agent, c0, tokens, COUNT, 2 * CHUNK_C)
    assert _rel(got_sum, ref_sum) < 1e-5
    assert float(jnp.abs(got_n - ref_n)) == 0.0
    assert _rel(c_f.M, c_r.M) < 1e-5
    assert _rel(c_f.I, c_r.I) < 1e-5
    assert int(c_f.pos) == int(c_r.pos)


@pytest.mark.parametrize("read", READS)
@pytest.mark.parametrize("chunk", [0, CHUNK_C, 2 * CHUNK_C, 4 * CHUNK_C])
def test_the_outer_block_size_does_not_change_what_a_delta_reads(
        read, chunk, agent, tokens, monkeypatch):
    """0 is the flat form (one block for the whole window); the rest are
    aligned multiples of 32. All four must agree, because the fast grid is
    measured from token 0 of the delta and every one of these starts its
    blocks on a multiple of 32."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", read)
    c0 = _carry(agent)
    c_a, rows_a, valid_a = _rollout(agent, c0, tokens, COUNT, 0)
    c_b, rows_b, _valid = _rollout(agent, c0, tokens, COUNT, chunk)
    assert _rel(rows_b, rows_a) < 1e-5
    assert _rel(c_b.M, c_a.M) < 1e-5
    assert _rel(c_b.I, c_a.I) < 1e-5


@pytest.mark.parametrize("read", READS)
def test_a_delta_split_across_two_calls_equals_one_call(
        read, agent, tokens, monkeypatch):
    """A step's delta is consumed once, but the SAME carry is then extended by
    the next step's delta. Splitting on a chunk multiple has to be free."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", read)
    c0 = _carry(agent)
    whole, rows_w, _v = _rollout(agent, c0, tokens, WINDOW, 0)
    half = 2 * CHUNK_C
    c1, rows_1, _v1 = agent.encode_extend(
        c0, tokens[:half], jnp.asarray(half, jnp.int32), window=half,
        start=0, chunk=0)
    c2, rows_2, _v2 = agent.encode_extend(
        c1, tokens[half:], jnp.asarray(WINDOW - half, jnp.int32),
        window=WINDOW - half, start=0, chunk=0)
    joined = jnp.concatenate([rows_1, rows_2], axis=0)
    assert _rel(joined, rows_w) < 1e-5
    assert _rel(c2.M, whole.M) < 1e-5
    assert _rel(c2.I, whole.I) < 1e-5


@pytest.mark.parametrize("read", READS)
def test_the_gradient_agrees_between_the_two_paths(
        read, agent, tokens, monkeypatch):
    """Values agreeing is not enough. The loss differentiates the fold, the
    rollout's numbers are what the ratio's denominator was recorded from, and
    an epoch-0 ratio of 1 needs both."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", read)
    c0 = _carry(agent)
    dyn, static = eqx.partition(agent, eqx.is_inexact_array)

    def loss_rollout(d):
        a = eqx.combine(d, static)
        _c, rows, valid = _rollout_diff(a, c0, tokens, COUNT, 2 * CHUNK_C)
        s, _n = _reduce(rows, valid)
        return jnp.sum(s ** 2)

    def loss_fold(d):
        a = eqx.combine(d, static)
        _c, (s, _n) = _folded(a, c0, tokens, COUNT, 2 * CHUNK_C)
        return jnp.sum(s ** 2)

    g_r = jax.grad(loss_rollout)(dyn)
    g_f = jax.grad(loss_fold)(dyn)
    leaves_r = [np.asarray(x) for x in jax.tree.leaves(g_r)]
    leaves_f = [np.asarray(x) for x in jax.tree.leaves(g_f)]
    assert len(leaves_r) == len(leaves_f)
    worst = max(_rel(a, b) for a, b in zip(leaves_f, leaves_r)
                if np.max(np.abs(b)) > 0)
    assert worst < 1e-4, f"gradient gap {worst:.3e}"


def test_the_flag_actually_changes_the_read(agent, tokens, monkeypatch):
    """A flag that did nothing would pass every test above. This one fails if
    `fast` silently fell back to the exact path anywhere."""
    c0 = _carry(agent)
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
    _c, rows_e, _v = _rollout(agent, c0, tokens, COUNT, 0)
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "fast")
    _c2, rows_f, _v2 = _rollout(agent, c0, tokens, COUNT, 0)
    assert not np.allclose(np.asarray(rows_e), np.asarray(rows_f),
                           rtol=1e-4, atol=1e-6)


def test_a_misaligned_outer_chunk_is_refused_under_the_fast_read(
        agent, tokens, monkeypatch):
    """Silently reading a misaligned grid is the one failure mode that would
    show up only as a drifting ratio, so it raises instead."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "fast")
    with pytest.raises(ValueError, match="not a multiple of the fast-palimpsa"):
        _rollout(agent, _carry(agent), tokens, COUNT, 24)


def test_a_misaligned_outer_chunk_is_still_allowed_under_the_exact_read(
        agent, tokens, monkeypatch):
    """The alignment rule belongs to the fast read alone; the exact read has
    no chunk grid to align, and its callers must not start failing."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
    _c, rows, _v = _rollout(agent, _carry(agent), tokens, COUNT, 24)
    assert np.all(np.isfinite(np.asarray(rows)))


def test_an_unknown_read_is_rejected_rather_than_defaulted(monkeypatch):
    from alphagrad.transformer.fast_palimpsa_pallas import palimpsa_read
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "approximate")
    with pytest.raises(ValueError, match="must be 'exact' or 'fast'"):
        palimpsa_read()


def test_the_fold_rounds_its_chunk_up_to_the_fast_grid(monkeypatch):
    """`plan_chunks` is what every folded caller sizes its side arrays from,
    so the rounding has to happen there and be visible in what it returns."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "fast")
    C, nb, padded = _fold.plan_chunks(4096, 100)
    assert C == 128 and C % CHUNK_C == 0
    assert nb * C == padded and padded >= 4096
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
    C2, _nb2, _p2 = _fold.plan_chunks(4096, 100)
    assert C2 == 100


def test_a_window_shorter_than_one_chunk_still_reads_under_the_fast_grid():
    """`plan_chunks` clamps the chunk to the window. A single block starts at
    token 0 whatever its width, so the clamp cannot misalign anything."""
    import os as _os
    prev = _os.environ.get("ALPHAGRAD_PALIMPSA_READ")
    _os.environ["ALPHAGRAD_PALIMPSA_READ"] = "fast"
    try:
        C, nb, padded = _fold.plan_chunks(11, 1024)
        assert C == 11 and nb == 1 and padded == 11
    finally:
        if prev is None:
            _os.environ.pop("ALPHAGRAD_PALIMPSA_READ", None)
        else:
            _os.environ["ALPHAGRAD_PALIMPSA_READ"] = prev


# --------------------------------------------------------------------------
# the count-proportional backward
# --------------------------------------------------------------------------
@pytest.mark.parametrize("read", READS)
def test_the_count_proportional_backward_agrees_with_the_scan_form(
        read, agent, tokens, monkeypatch):
    """``ALPHAGRAD_COUNT_VJP=1`` replaces the fold's ``scan`` + ``cond`` with a
    hand-written ``while_loop`` in both directions (``common/count_vjp.py``).

    It is ORTHOGONAL to the read: it decides HOW MANY chunks run, the read
    decides what happens inside one. Under the fast read a chunk body is a
    Pallas ``custom_vjp`` rather than a token scan, and ``count_loop`` rebuilds
    that body's VJP with ``jax.vjp`` -- which is exactly what a nested
    ``custom_vjp`` is for. This pins that the two forms still land on the same
    value and the same gradient with the fast read in force.
    """
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", read)
    c0 = _carry(agent)
    dyn, static = eqx.partition(agent, eqx.is_inexact_array)

    def loss(d):
        a = eqx.combine(d, static)
        _c, (s, _n) = _folded(a, c0, tokens, COUNT, 2 * CHUNK_C)
        return jnp.sum(s ** 2)

    monkeypatch.setenv("ALPHAGRAD_COUNT_VJP", "0")
    v_scan = float(loss(dyn))
    g_scan = jax.grad(loss)(dyn)
    monkeypatch.setenv("ALPHAGRAD_COUNT_VJP", "1")
    v_loop = float(loss(dyn))
    g_loop = jax.grad(loss)(dyn)

    assert abs(v_loop - v_scan) <= 1e-5 * max(abs(v_scan), 1.0)
    pairs = list(zip(jax.tree.leaves(g_loop), jax.tree.leaves(g_scan)))
    worst = max(_rel(a, b) for a, b in pairs if np.max(np.abs(b)) > 0)
    assert worst < 1e-4, f"count_loop gradient gap {worst:.3e}"
