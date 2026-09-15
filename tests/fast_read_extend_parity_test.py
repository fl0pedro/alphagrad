# -*- coding: utf-8 -*-
"""The rollout extend and the loss extend must read a delta the same way.

WHY THIS FILE EXISTS
--------------------
``ALPHAGRAD_PALIMPSA_READ_ROLLOUT`` and ``ALPHAGRAD_PALIMPSA_READ_LOSS`` each
pick ``exact`` or ``fast``. ``fast`` swaps the palimpsa read for a chunked
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

THE MISMATCHED PAIR. Owner ruling 2026-09-15 allows the rollout and the loss
to read with DIFFERENT operators, because the fast read costs the rollout
about 10 s per episode and saves the loss about 13.7 s. The price is that the
PPO ratio at epoch 0 is no longer 1. Every case above runs with the two reads
EQUAL and keeps its bit-identical claims. Two cases at the end run them
different: one checks that the trainer refuses without
``ALPHAGRAD_ALLOW_READ_MISMATCH=1``, and one sets the allow flag, MEASURES the
epoch-0 ratio drift and prints it instead of asserting 1.
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

ROLLOUT_ENV = "ALPHAGRAD_PALIMPSA_READ_ROLLOUT"
LOSS_ENV = "ALPHAGRAD_PALIMPSA_READ_LOSS"


def _set_read(monkeypatch, rollout, loss=None):
    """Set the PAIR. ``loss=None`` means "the same on both", which is what
    every bit-identical case in this file wants: with one operator on both
    sides the rollout and the loss compute the same function and the old
    claims stand unchanged."""
    monkeypatch.setenv(ROLLOUT_ENV, rollout)
    monkeypatch.setenv(LOSS_ENV, rollout if loss is None else loss)


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
                               window=WINDOW, start=0, chunk=chunk,
                               path="rollout")


def _rollout_diff(agent, carry, toks, count, chunk):
    """The rollout extend in the form a DIFFERENTIATED caller gets.

    Without a budget the chunked extend is a ``lax.while_loop``, which has no
    transpose rule -- that is the whole reason ``budget`` exists (see
    ``encode_extend``'s docstring). The loss always passes one, so this is the
    form whose gradient has to match the fold's.
    """
    return agent.encode_extend(carry, toks, jnp.asarray(count, jnp.int32),
                               window=WINDOW, start=0, chunk=chunk,
                               budget=jnp.asarray(count, jnp.int32),
                               path="loss")


def _folded(agent, carry, toks, count, chunk):
    """What the loss runs: ``extend_fold`` reducing the rows as it goes."""
    init, fold = _fold.sum_reducer(agent.embd_dim)
    return _fold.extend_fold(
        agent, carry, toks, jnp.asarray(count, jnp.int32), window=WINDOW,
        chunk=chunk, init_acc=init, fold_fn=fold,
        budget=jnp.asarray(count, jnp.int32), path="loss")


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
    _set_read(monkeypatch, read)
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
    _set_read(monkeypatch, read)
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
    _set_read(monkeypatch, read)
    c0 = _carry(agent)
    whole, rows_w, _v = _rollout(agent, c0, tokens, WINDOW, 0)
    half = 2 * CHUNK_C
    c1, rows_1, _v1 = agent.encode_extend(
        c0, tokens[:half], jnp.asarray(half, jnp.int32), window=half,
        start=0, chunk=0, path="rollout")
    c2, rows_2, _v2 = agent.encode_extend(
        c1, tokens[half:], jnp.asarray(WINDOW - half, jnp.int32),
        window=WINDOW - half, start=0, chunk=0, path="rollout")
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
    _set_read(monkeypatch, read)
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
    _set_read(monkeypatch, "exact")
    _c, rows_e, _v = _rollout(agent, c0, tokens, COUNT, 0)
    _set_read(monkeypatch, "fast")
    _c2, rows_f, _v2 = _rollout(agent, c0, tokens, COUNT, 0)
    assert not np.allclose(np.asarray(rows_e), np.asarray(rows_f),
                           rtol=1e-4, atol=1e-6)


def test_a_misaligned_outer_chunk_is_refused_under_the_fast_read(
        agent, tokens, monkeypatch):
    """Silently reading a misaligned grid is the one failure mode that would
    show up only as a drifting ratio, so it raises instead."""
    _set_read(monkeypatch, "fast")
    with pytest.raises(ValueError, match="not a multiple of the fast-palimpsa"):
        _rollout(agent, _carry(agent), tokens, COUNT, 24)


def test_a_misaligned_outer_chunk_is_still_allowed_under_the_exact_read(
        agent, tokens, monkeypatch):
    """The alignment rule belongs to the fast read alone; the exact read has
    no chunk grid to align, and its callers must not start failing."""
    _set_read(monkeypatch, "exact")
    _c, rows, _v = _rollout(agent, _carry(agent), tokens, COUNT, 24)
    assert np.all(np.isfinite(np.asarray(rows)))


@pytest.mark.parametrize("env", [ROLLOUT_ENV, LOSS_ENV])
def test_an_unknown_read_is_rejected_rather_than_defaulted(env, monkeypatch):
    from alphagrad.transformer.fast_palimpsa_pallas import palimpsa_read
    _set_read(monkeypatch, "fast")
    monkeypatch.setenv(env, "approximate")
    path = "rollout" if env == ROLLOUT_ENV else "loss"
    with pytest.raises(ValueError, match="must be 'exact' or 'fast'"):
        palimpsa_read(path)


def test_the_retired_single_flag_is_refused_by_name(monkeypatch):
    """`ALPHAGRAD_PALIMPSA_READ` used to select ONE read for both paths. A
    launcher that still exports it would otherwise get the default pair in
    silence, which is the one thing a read flag must never do."""
    from alphagrad.transformer.fast_palimpsa_pallas import palimpsa_read
    _set_read(monkeypatch, "fast")
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
    with pytest.raises(ValueError, match="is retired"):
        palimpsa_read("loss")


def test_a_reader_that_does_not_name_a_path_is_refused(monkeypatch):
    """The path is not optional. A reader that forgot it would read one
    operator while the other side read the other, and nothing would say so."""
    from alphagrad.transformer.fast_palimpsa_pallas import palimpsa_read
    _set_read(monkeypatch, "fast")
    with pytest.raises(ValueError, match="path must be one of"):
        palimpsa_read("both")


def test_the_fold_refuses_a_chunk_that_would_misalign_the_fast_grid(
        monkeypatch):
    """`plan_chunks` is what every folded caller sizes its side arrays from,
    so this is where a misaligned request has to be caught.

    It RAISES rather than rounding 100 up to 128. Rounding is silent, and a
    launcher that asked for 100 and got 128 has no way to find out. The exact
    read has no chunk grid, so the same request is fine there."""
    _set_read(monkeypatch, "fast")
    with pytest.raises(ValueError, match="not a multiple of the fast-palimpsa"):
        _fold.plan_chunks(4096, 100, path="loss")
    _set_read(monkeypatch, "exact")
    C2, _nb2, _p2 = _fold.plan_chunks(4096, 100, path="loss")
    assert C2 == 100


def test_an_aligned_fold_chunk_is_returned_unchanged(monkeypatch):
    _set_read(monkeypatch, "fast")
    C, nb, padded = _fold.plan_chunks(4096, 128, path="loss")
    assert C == 128 and C % CHUNK_C == 0
    assert nb == 32 and padded == 4096


def test_a_window_shorter_than_one_chunk_still_reads_under_the_fast_grid():
    """`plan_chunks` clamps the chunk to the window. A single block starts at
    token 0 whatever its width, so the clamp cannot misalign anything."""
    import os as _os
    prev = _os.environ.get(LOSS_ENV)
    _os.environ[LOSS_ENV] = "fast"
    try:
        C, nb, padded = _fold.plan_chunks(11, 1024, path="loss")
        assert C == 11 and nb == 1 and padded == 11
    finally:
        if prev is None:
            _os.environ.pop(LOSS_ENV, None)
        else:
            _os.environ[LOSS_ENV] = prev


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
    _set_read(monkeypatch, read)
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


# --------------------------------------------------------------------------
# the per-face escape hatch
# --------------------------------------------------------------------------
def test_the_read_override_actually_changes_what_the_readers_see(monkeypatch):
    """The face pipeline rests on this, so it is pinned on its own.

    It shipped once as a context manager that set a variable nothing read.
    Every caller went on using the shipped read, the face rollout and the
    face replay went on disagreeing, and the only symptom was a log-prob
    8.4e-3 out at three faces. A context manager that does nothing looks
    exactly like one that works.
    """
    from alphagrad.transformer.fast_palimpsa_pallas import (
        palimpsa_read, read_override)
    _set_read(monkeypatch, "fast")
    assert palimpsa_read("loss") == "fast"
    assert _fold._fast_read("loss") is True
    with read_override("exact"):
        assert palimpsa_read("loss") == "exact"
        assert _fold._fast_read("loss") is False
    assert palimpsa_read("loss") == "fast"


def test_the_read_override_is_restored_when_the_body_raises(monkeypatch):
    from alphagrad.transformer.fast_palimpsa_pallas import (
        palimpsa_read, read_override)
    _set_read(monkeypatch, "fast")
    with pytest.raises(RuntimeError):
        with read_override("exact"):
            raise RuntimeError("boom")
    assert palimpsa_read("loss") == "fast"


def test_the_read_override_refuses_a_mode_it_does_not_know():
    from alphagrad.transformer.fast_palimpsa_pallas import read_override
    with pytest.raises(ValueError, match="takes 'exact' or 'fast'"):
        with read_override("approximate"):
            pass


def test_the_override_reaches_the_fold_through_plan_chunks(monkeypatch):
    """`plan_chunks` refuses a misaligned chunk under the fast read. The face
    replay folds a window whose chunk it does not choose, so the override has
    to lift that refusal too, or the face path would raise instead of reading
    exact."""
    _set_read(monkeypatch, "fast")
    from alphagrad.transformer.fast_palimpsa_pallas import read_override
    with pytest.raises(ValueError, match="not a multiple of the fast-palimpsa"):
        _fold.plan_chunks(4096, 100, path="loss")
    with read_override("exact"):
        assert _fold.plan_chunks(4096, 100, path="loss")[0] == 100


# --------------------------------------------------------------------------
# THE MISMATCHED PAIR (owner ruling 2026-09-15)
# --------------------------------------------------------------------------
# Everything above runs the two paths on ONE operator, where the rollout and
# the loss compute the same function and every claim is bit-identical. These
# cases run them on DIFFERENT operators, which is what the ruling allows and
# what the guard exists for.
def test_a_mismatched_pair_is_refused_without_the_allow_flag(monkeypatch):
    """The guard the trainer runs at startup. A mismatched pair is a
    deliberate choice with a price, so it has to be asked for in writing."""
    from alphagrad.transformer.fast_palimpsa_pallas import (
        require_read_agreement)
    monkeypatch.delenv("ALPHAGRAD_ALLOW_READ_MISMATCH", raising=False)
    _set_read(monkeypatch, "exact", "fast")
    with pytest.raises(RuntimeError) as exc:
        require_read_agreement()
    msg = str(exc.value)
    # The refusal NAMES both flags and says what the mismatch does, because a
    # refusal that only says "no" sends the reader back to the source.
    assert ROLLOUT_ENV in msg and LOSS_ENV in msg
    assert "ALPHAGRAD_ALLOW_READ_MISMATCH" in msg
    assert "ratio" in msg


def test_a_matched_pair_needs_no_allow_flag(monkeypatch):
    from alphagrad.transformer.fast_palimpsa_pallas import (
        require_read_agreement)
    monkeypatch.delenv("ALPHAGRAD_ALLOW_READ_MISMATCH", raising=False)
    for mode in READS:
        _set_read(monkeypatch, mode)
        assert require_read_agreement(echo=False) == (mode, mode)


def test_the_allow_flag_lets_a_mismatched_pair_run(monkeypatch):
    from alphagrad.transformer.fast_palimpsa_pallas import (
        require_read_agreement)
    monkeypatch.setenv("ALPHAGRAD_ALLOW_READ_MISMATCH", "1")
    _set_read(monkeypatch, "exact", "fast")
    assert require_read_agreement(echo=False) == ("exact", "fast")


def test_the_two_flags_really_are_two(monkeypatch):
    """A second flag that did nothing would pass every case above. This one
    fails if one path's flag silently answered for the other."""
    from alphagrad.transformer.fast_palimpsa_pallas import palimpsa_read
    monkeypatch.setenv("ALPHAGRAD_ALLOW_READ_MISMATCH", "1")
    _set_read(monkeypatch, "exact", "fast")
    assert palimpsa_read("rollout") == "exact"
    assert palimpsa_read("loss") == "fast"
    _set_read(monkeypatch, "fast", "exact")
    assert palimpsa_read("rollout") == "fast"
    assert palimpsa_read("loss") == "exact"


def test_the_mixer_refuses_to_guess_a_path_under_a_mismatched_pair(
        monkeypatch):
    """`palimpsa_mix` is the one reader with no path argument (it lives
    inside the mixer module and `agent.encode()` reaches it from both sides).
    With the two reads equal there is one honest answer and it gives it; with
    them different there is none, so it raises rather than picking."""
    from alphagrad.transformer.fast_palimpsa_pallas import (
        current_read_path, read_path)
    monkeypatch.setenv("ALPHAGRAD_ALLOW_READ_MISMATCH", "1")
    _set_read(monkeypatch, "fast")
    assert current_read_path() in ("rollout", "loss")
    _set_read(monkeypatch, "exact", "fast")
    with pytest.raises(RuntimeError, match="outside any `read_path` block"):
        current_read_path()
    with read_path("loss"):
        assert current_read_path() == "loss"
    with read_path("rollout"):
        assert current_read_path() == "rollout"


# --------------------------------------------------------------------------
# THE DRIFT BOUND, AND ITS ARITHMETIC
# --------------------------------------------------------------------------
# The PPO ratio is exp(new_logp - old_logp). With one operator on both sides
# it is exactly 1 at epoch 0 (measured on the GPU: ratio/max_log of order
# 1e-12, which is float noise). With two operators it is not, and this is how
# far it can go.
#
#   1. THE KERNEL'S OWN ERROR. tests/fast_palimpsa_kernel_test.py pins the
#      fast read against its oracle at a RELATIVE 5e-3, forward and in every
#      one of the nine gradients. That is upstream's own tolerance for its
#      tensor-core dots, so 5e-3 is the per-chunk figure, not a guess.
#
#   2. HOW MANY CHUNKS ONE STEP IS. The fast grid is CHUNK_C = 32 tokens
#      measured from token 0 of the delta, so a step of N tokens is
#      ceil(N / 32) chunks. Each chunk hands its boundary state to the next,
#      so the state's relative error compounds ONCE PER CHUNK, not once per
#      token:
#
#          rel_state(N)  <=  (1 + 5e-3) ** ceil(N / 32)  -  1
#
#      This is the "over the token count of one step" term. At COUNT = 83
#      tokens that is 3 chunks and rel_state <= 1.508e-2.
#
#   3. FROM THE STATE TO THE LOG-PROB. The vertex logits are a linear map of
#      the pooled rows, and log-softmax is 2-Lipschitz in the sup norm of its
#      input (|d log softmax_i| <= 2 |dx|_inf). A relative error rel_state on
#      logits of magnitude |logp| therefore moves one action's log-prob by at
#      most 2 * |logp| * rel_state. |logp| is floored at 1 so the bound does
#      not collapse on a near-uniform head.
#
#          max_log_bound  =  2 * max(|logp|, 1) * rel_state(N)
#
#      At COUNT = 83 and this six-vertex head that is about 6e-2.
#
# The number MEASURED here is much smaller than the bound, and it is PRINTED
# rather than asserted equal to anything: the point of the case is to publish
# the drift, and the assertion is only a ceiling that a real regression would
# break.
KERNEL_REL = 5e-3


def _drift_bound(count, logp):
    import math
    n_chunks = -(-int(count) // CHUNK_C)
    rel_state = (1.0 + KERNEL_REL) ** n_chunks - 1.0
    return 2.0 * max(abs(float(logp)), 1.0) * rel_state, n_chunks, rel_state


def _step_logp(agent, toks, count, read, action):
    """One step's vertex log-prob, read end to end on ONE path's operator.

    This is the quantity the PPO ratio is built from: the base encode, the
    step's delta advance, the vertex head, log-softmax, one action. Running
    it twice under two operators and subtracting IS `ratio/max_log` at
    epoch 0.
    """
    import os as _os
    from alphagrad.approx.common import carry_stream as _cs
    prev = _os.environ.get(ROLLOUT_ENV), _os.environ.get(LOSS_ENV)
    _os.environ[ROLLOUT_ENV] = read
    _os.environ[LOSS_ENV] = read
    try:
        total_v = 6
        base = toks[: 2 * CHUNK_C]
        enc0, bs, bc = _cs.init_carry(
            agent, base, jnp.asarray(base.shape[0], jnp.int32),
            window=int(base.shape[0]), total_v=total_v,
            embd_dim=agent.embd_dim, path="rollout")
        vs0, vc0 = _cs.zero_memory(total_v, agent.embd_dim)
        part = jnp.zeros((total_v + 1,), jnp.float32).at[0].set(1.0)
        _c, vs, vc = _cs.advance(
            agent, enc0, vs0, vc0, toks, jnp.asarray(count, jnp.int32),
            jnp.asarray(0, jnp.int32), window=WINDOW, participants=part,
            path="rollout")
        vlog, _ctx, _val = _cs.heads(agent, vs, vc, base_mem=(bs, bc),
                                     preference=None)
        return float(jax.nn.log_softmax(vlog)[action])
    finally:
        for env, val in zip((ROLLOUT_ENV, LOSS_ENV), prev):
            if val is None:
                _os.environ.pop(env, None)
            else:
                _os.environ[env] = val


def test_a_matched_pair_leaves_the_epoch_zero_ratio_at_one(agent, tokens):
    """The control for the case below. One operator on both sides and the
    two log-probs are the SAME number, so the ratio is exactly 1."""
    a = _step_logp(agent, tokens, COUNT, "fast", 0)
    b = _step_logp(agent, tokens, COUNT, "fast", 0)
    assert a == b


@pytest.mark.parametrize("rollout,loss", [("exact", "fast"),
                                          ("fast", "exact")])
def test_a_mismatched_pair_reports_its_epoch_zero_ratio_drift(
        rollout, loss, agent, tokens, capsys):
    """MEASURE AND REPORT, do not assert 1.

    With the rollout sampling under one operator and the loss scoring under
    another the epoch-0 ratio is 1 plus the operator difference. The number is
    printed so a reader of the test output sees it; the assertion is the
    ceiling derived in the block comment above.
    """
    lp_rollout = _step_logp(agent, tokens, COUNT, rollout, 0)
    lp_loss = _step_logp(agent, tokens, COUNT, loss, 0)
    max_log = abs(lp_loss - lp_rollout)
    bound, n_chunks, rel_state = _drift_bound(COUNT, lp_rollout)
    with capsys.disabled():
        print(f"\n[ratio drift] rollout={rollout} loss={loss}  "
              f"logp_rollout={lp_rollout:.6f} logp_loss={lp_loss:.6f}  "
              f"ratio/max_log={max_log:.3e}  ratio={np.exp(max_log):.6f}  "
              f"bound={bound:.3e} "
              f"(N={COUNT} tokens = {n_chunks} chunks of {CHUNK_C}, "
              f"rel_state={rel_state:.3e}, kernel rel={KERNEL_REL:g})")
    assert max_log < bound, (
        f"epoch-0 ratio drift {max_log:.3e} exceeds the derived bound "
        f"{bound:.3e}")
