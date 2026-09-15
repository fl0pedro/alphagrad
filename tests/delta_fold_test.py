# -*- coding: utf-8 -*-
"""The chunk-fold must equal the full-width reduction, value AND gradient.

Driven by a STUB agent rather than a real policy. The fold's own risk is not
the recurrence -- that is the shipped ``encode_extend`` either way -- it is the
plumbing: chunk counts, boundaries, and the position-dependent face key. A stub
isolates exactly that and runs in milliseconds, so the boundary cases can be
swept densely instead of sampled.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.common.delta_fold import (
    extend_fold, face_key_fn, plan_chunks, segment_reducer, sum_reducer)

E = 4


class StubAgent:
    """A minimal `encode_extend`: a real carried recurrence, cheap rows.

    carry is a scalar accumulator; row t = carry_after_t * (1 + token). That
    makes rows depend on ALL previous tokens, so a fold that mishandled the
    carry across a chunk boundary could not accidentally still match.
    """

    embd_dim = E

    def encode_extend(self, carry, toks, count, *, window, start=0,
                      chunk=None, budget=None):
        del start, chunk, budget
        idx = jnp.arange(window, dtype=jnp.int32)
        valid = idx < count

        def step(c, x):
            tok, ok = x
            c2 = jnp.where(ok, c + tok.astype(jnp.float32), c)
            row = jnp.where(ok, c2 * (1.0 + jnp.arange(E, dtype=jnp.float32)),
                            jnp.zeros((E,), jnp.float32))
            return c2, row

        c_out, rows = lax_scan(step, carry, (toks[:window], valid))
        return c_out, rows, valid


def lax_scan(f, init, xs):
    from jax import lax
    return lax.scan(f, init, xs)


def _mk(window, count, seed=0):
    r = np.random.RandomState(seed)
    toks = jnp.asarray(r.randint(1, 9, size=window).astype(np.int32))
    return toks, jnp.asarray(count, jnp.int32)


def _flat_sums(agent, toks, count, window):
    """The reduction as the LIVE code does it: full-width rows, then sum.

    TWO accumulators, not four: the eqn-owned / structural split went with
    the equation ids (see ``carry_stream.advance``).
    """
    c, rows, valid = agent.encode_extend(0.0, toks, count, window=window)
    w = jnp.asarray(valid, jnp.float32)
    return c, (jnp.sum(rows * w[:, None], axis=0), jnp.sum(w))


# THE CHUNKS ARE MULTIPLES OF 32, and 48 is allowed because the window clamps
# it to a single block. Under ALPHAGRAD_PALIMPSA_READ=fast, which is the
# shipped default, a multi-block fold has to start every block on a multiple
# of 32 (see `plan_chunks`). The COUNTS still sweep the interesting
# remainders -- 0, 1, 17, 37, 63, and the full window -- which is what this
# file is actually about. `test_a_misaligned_chunk_is_refused_under_the_fast_read`
# below pins the refusal itself.
@pytest.mark.parametrize("window,count,chunk", [
    (64, 64, 32), (64, 0, 32), (64, 1, 32), (64, 63, 32), (64, 17, 32),
    (64, 64, 64), (64, 64, 128), (100, 37, 32), (100, 100, 32), (48, 48, 64),
])
def test_fold_matches_full_width_reduction(window, count, chunk):
    agent = StubAgent()
    toks, cnt = _mk(window, count)
    c_ref, ref = _flat_sums(agent, toks, cnt, window)
    init, fold = sum_reducer(E)
    c_fold, acc = extend_fold(agent, 0.0, toks, cnt, window=window,
                              chunk=chunk, init_acc=init, fold_fn=fold)
    assert float(jnp.abs(c_fold - c_ref)) < 1e-5, "carry diverged"
    for a, b in zip(acc, ref):
        assert bool(jnp.allclose(a, b, atol=1e-5)), "%s vs %s" % (a, b)


def test_gradient_matches_too():
    """Values agreeing is not enough -- the loss differentiates through this."""
    agent = StubAgent()
    window, chunk = 64, 32
    toks, cnt = _mk(window, 40, seed=3)

    def f_flat(c0):
        _, (te, _ne) = _flat_sums(agent, toks, cnt, window)
        return jnp.sum(te) + c0 * 0.0

    def f_fold(c0):
        init, fold = sum_reducer(E)
        _, (te, _ne) = extend_fold(
            agent, c0, toks, cnt, window=window, chunk=chunk,
            init_acc=init, fold_fn=fold)
        return jnp.sum(te)

    def g_flat(c0):
        c, rows, valid = agent.encode_extend(
            c0, toks, cnt, window=window)
        w = jnp.asarray(valid, jnp.float32)
        return jnp.sum(rows * w[:, None])

    def g_fold(c0):
        init, fold = sum_reducer(E)
        _, (te, _ne) = extend_fold(
            agent, c0, toks, cnt, window=window, chunk=chunk,
            init_acc=init, fold_fn=fold)
        return jnp.sum(te)

    assert bool(jnp.allclose(g_flat(0.5), g_fold(0.5), atol=1e-5))
    d_flat = jax.grad(g_flat)(0.5)
    d_fold = jax.grad(g_fold)(0.5)
    assert bool(jnp.allclose(d_flat, d_fold, atol=1e-4)), \
        "grad %s vs %s" % (d_flat, d_fold)
    assert bool(jnp.isfinite(d_fold))


def test_face_key_offset_is_applied():
    """THE regression that would be silent: dropping the chunk offset.

    Without ``off`` every chunk after the first keys to face 0, which is a
    plausible-looking wrong answer, not a crash.
    """
    ends = jnp.asarray([10, 25, 40], jnp.int32)
    key = face_key_fn(ends)
    first = key(10, 0)
    second = key(10, 10)
    third = key(10, 30)
    assert list(np.asarray(first)) == [0] * 10
    assert list(np.asarray(second)) == [1] * 10          # NOT 0
    assert list(np.asarray(third)) == [2] * 10           # NOT 0


def test_segment_fold_matches_full_width_scatter():
    agent = StubAgent()
    window, chunk, F = 64, 32, 4
    toks, cnt = _mk(window, 50, seed=5)
    ends = jnp.asarray([12, 28, 44, 64], jnp.int32)

    _c, rows, valid = agent.encode_extend(
        0.0, toks, cnt, window=window)
    w = jnp.asarray(valid, jnp.float32)
    pos = jnp.arange(window, dtype=jnp.int32)
    fid = jnp.searchsorted(ends, pos, side="right").astype(jnp.int32)
    fid_c = jnp.clip(fid, 0, F - 1)
    ref_sums = jnp.zeros((F, E)).at[fid_c].add(rows * w[:, None])
    ref_counts = jnp.zeros((F,)).at[fid_c].add(w)

    init, fold = segment_reducer(E, F, face_key_fn(ends))
    _cf, (sums, counts) = extend_fold(
        agent, 0.0, toks, cnt, window=window, chunk=chunk,
        init_acc=init, fold_fn=fold)
    assert bool(jnp.allclose(sums, ref_sums, atol=1e-5))
    assert bool(jnp.allclose(counts, ref_counts, atol=1e-5))


def test_chunk_size_does_not_change_the_answer():
    agent = StubAgent()
    window = 96
    toks, cnt = _mk(window, 70, seed=7)
    outs = []
    for chunk in (32, 64, 96, 128):
        init, fold = sum_reducer(E)
        _c, acc = extend_fold(agent, 0.0, toks, cnt, window=window,
                              chunk=chunk, init_acc=init, fold_fn=fold)
        outs.append(acc[0])
    for o in outs[1:]:
        assert bool(jnp.allclose(o, outs[0], atol=1e-5))


def test_chunk_zero_is_the_encode_extend_sentinel_not_an_error():
    """chunk=0 means "flat, no dynamic trip count" in the encode_extend API.

    advance's docstring tells reverse-differentiated callers without a budget
    to pass exactly that, so rejecting it breaks every such caller -- which is
    how it broke palimpsa_base_grad. The fold has no flat mode, so 0 falls
    back to the fold's own chunk and the answer is unchanged.
    """
    agent = StubAgent()
    toks, cnt = _mk(64, 40, seed=21)
    ref = _flat_sums(agent, toks, cnt, 64)[1]
    init, fold = sum_reducer(E)
    _c, acc = extend_fold(agent, 0.0, toks, cnt, window=64, chunk=0,
                          init_acc=init, fold_fn=fold)
    for a, b in zip(acc, ref):
        assert bool(jnp.allclose(a, b, atol=1e-5))


def test_rejects_a_negative_chunk():
    with pytest.raises(ValueError, match="chunk must be positive"):
        init, fold = sum_reducer(E)
        extend_fold(StubAgent(), 0.0, *_mk(16, 8), window=16, chunk=-4,
                    init_acc=init, fold_fn=fold)


def test_a_misaligned_chunk_is_refused_under_the_fast_read(monkeypatch):
    """The chunk sizes above are multiples of 32 because this is what happens
    otherwise, and it has to be loud.

    The fast read's own chunk grid starts at token 0 of the delta. A fold that
    began its blocks at 0, 7, 14, ... would cut the delta in places the
    rollout's own blocking never cuts it, and the two sides would then read
    genuinely different numbers. Rounding the request up would be silent, so
    `plan_chunks` raises.
    """
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "fast")
    init, fold = sum_reducer(E)
    with pytest.raises(ValueError, match="not a multiple of the fast-palimpsa"):
        extend_fold(StubAgent(), 0.0, *_mk(64, 40), window=64, chunk=7,
                    init_acc=init, fold_fn=fold)


def test_one_block_is_exempt_from_the_alignment_rule(monkeypatch):
    """A window under the chunk leaves ONE block, and one block starts at
    token 0 whatever its width. That is what makes `plan_chunks`'s clamp to
    the window safe, and a window bin of 11 is a real case."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "fast")
    C, nb, padded = plan_chunks(11, 1024)
    assert (C, nb, padded) == (11, 1, 11)


def test_a_misaligned_chunk_is_still_fine_under_the_exact_read(monkeypatch):
    """The alignment rule belongs to the fast read alone. The exact recurrence
    has no chunk grid, so no size is wrong for it."""
    monkeypatch.setenv("ALPHAGRAD_PALIMPSA_READ", "exact")
    agent = StubAgent()
    toks, cnt = _mk(64, 40, seed=31)
    ref = _flat_sums(agent, toks, cnt, 64)[1]
    init, fold = sum_reducer(E)
    _c, acc = extend_fold(agent, 0.0, toks, cnt, window=64, chunk=7,
                          init_acc=init, fold_fn=fold)
    for a, b in zip(acc, ref):
        assert bool(jnp.allclose(a, b, atol=1e-5))


@pytest.mark.parametrize("window,count,chunk,budget", [
    (128, 20, 32, 20), (128, 20, 32, 32), (128, 0, 32, 0),
    (128, 128, 32, 128), (128, 65, 32, 65), (96, 40, 96, 40),
])
def test_budget_skipping_is_exact(window, count, chunk, budget):
    """Skipping chunks past the budget must not change the answer.

    A skipped chunk is all-invalid, so `_step` freezes the carry and emits
    zero rows, and every reducer here is a `valid`-weighted sum -- so the
    skip is exact, not approximate. Without this the fold walks the WHOLE
    window regardless of the delta length, which would make a 65536 bound
    cost 64x a 1024 delta: the cost folding exists to remove.
    """
    agent = StubAgent()
    toks, cnt = _mk(window, count, seed=11)
    init, fold = sum_reducer(E)
    c_no, acc_no = extend_fold(agent, 0.0, toks, cnt, window=window,
                               chunk=chunk, init_acc=init, fold_fn=fold)
    init2, fold2 = sum_reducer(E)
    c_bd, acc_bd = extend_fold(agent, 0.0, toks, cnt, window=window,
                               chunk=chunk, init_acc=init2, fold_fn=fold2,
                               budget=budget)
    assert float(jnp.abs(c_bd - c_no)) < 1e-5, "carry diverged under budget"
    for a, b in zip(acc_bd, acc_no):
        assert bool(jnp.allclose(a, b, atol=1e-5)), "%s vs %s" % (a, b)


def test_budget_gradient_survives_the_cond():
    """cond must stay transposable -- while_loop here would kill the loss."""
    agent = StubAgent()
    toks, cnt = _mk(128, 30, seed=13)

    def g(c0):
        init, fold = sum_reducer(E)
        _, (te, _ne) = extend_fold(
            agent, c0, toks, cnt, window=128, chunk=32,
            init_acc=init, fold_fn=fold, budget=30)
        return jnp.sum(te)

    d = jax.grad(g)(0.5)
    assert bool(jnp.isfinite(d)) and float(jnp.abs(d)) > 0.0
