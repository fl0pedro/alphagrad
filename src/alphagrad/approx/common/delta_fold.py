# -*- coding: utf-8 -*-
"""Chunk-folded delta extension: never materialise ``(window, E)`` rows.

THE PROBLEM
-----------
``encode_extend`` returns one row per token of the delta WINDOW, and reverse
mode stores that ``(window, E)`` block. Peak memory is therefore set by the
window and by nothing else -- which is why the ``ALPHAGRAD_EXTEND_CHUNK``
sweep found peak FLAT at 2051 MB across every chunk size, and why the fix
attempted in 5407eec was to shrink the window instead. ``_extend_sequential``
says so in its own comment: *"The zero rows are still materialised at full
width, so every downstream reduction sees the identical array."*

Shrinking the window does not work, because no window is correct. MEASURED
(job 61350, HEAD): NN256 truncates 40 times at 4096 (up to 5756 tokens) and
TLM truncates at 16384 (19657 tokens). Every bound so far was fitted to a
distribution the next run exceeded; the tail is policy-dependent.

THE OBSERVATION THAT MAKES THIS WORK
------------------------------------
Every consumer of those rows is a REDUCTION along the window axis:

  * ``carry_stream.advance``      two weighted sums -> (E,) + 2 scalars
  * ``carry_stream.base_memory``  scatter by base_owners -> (V+2, E)
  * ``ppo.py:2143 _face_encode``  single-segment scatter_mean -> (E,)
  * ``ppo.py:2318 _face_replay``  segmented scatter_mean by face -> (F, E)

Nothing indexes an individual row. And palimpsa is a RECURRENCE, so running
tokens 0..C-1 then C..2C-1 with the carry threaded is the same computation as
one pass -- exactly what ``encode_extend`` already does across STEPS. So the
window can be walked in fixed-size chunks with the reduction folded as we go,
and the big array simply never exists.

Peak becomes O(chunk * E) and ``MAX_DELTA_TOKENS`` stops being a memory floor:
it degrades to a loop count, so an over-tall bound costs iterations on the
rare long deltas rather than memory on every step.

This module deliberately calls the EXISTING ``agent.encode_extend`` at width
``chunk`` rather than reaching into its internals. The rows it returns are
then ``(chunk, E)`` by construction, so the win needs no surgery inside
``ppo.py`` and the per-token semantics are whatever the shipped path already
does -- including the frozen-carry no-op on invalid steps.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import lax

from . import count_vjp as _cvjp


def default_chunk() -> int:
    return int(os.environ.get("ALPHAGRAD_FOLD_CHUNK", "1024"))


def _use_parallel() -> bool:
    return os.environ.get("ALPHAGRAD_FOLD_PARALLEL", "1") != "0"


def _encode_chunk(agent, enc, tk, c_cnt, C, parallel):
    """One chunk's encode. PARALLEL inside the chunk by default.

    Chunking and parallelism are ORTHOGONAL: the chunk bounds how much is
    live at once (memory), the scan inside it decides whether the tokens are
    walked serially or with log depth (speed). `_extend_parallel` is the
    associative-scan form, and reverse mode of an associative scan is itself
    an associative scan -- so a parallel forward buys a parallel backward for
    free.

    `encode_extend` picks the two apart by a global env var, which is not
    what a caller wants here, so the parallel path is called directly with
    the same `valid` mask `encode_extend` would have built. Falls back to
    `encode_extend` when the agent has no parallel path (test stubs).
    """
    if parallel:
        par = getattr(agent, "_extend_parallel", None)
        if par is not None:
            valid = jnp.arange(C, dtype=jnp.int32) < jnp.asarray(c_cnt,
                                                                 jnp.int32)
            return par(enc, tk, valid, c_cnt)
    return agent.encode_extend(enc, tk, c_cnt, window=C, start=0, chunk=0)


def plan_chunks(window, chunk=None):
    """``(C, nb, padded_len)`` for a window.

    EVERY caller with a per-token SIDE array (base_owners ids, face keys)
    must pad it to ``padded_len``, not to ``window``: the fold pads its token
    buffers up to ``nb * C`` and slices side arrays by the same offsets, so a
    window-length side array runs off the end of the last chunk. That is not
    a silent failure -- ``dynamic_slice`` raises on the shape -- but it is
    easy to hit, so the length lives here rather than in each caller.

    ``C`` is clamped to the window: a chunk larger than the whole window
    would pad more than it processes.
    """
    # chunk=0 is a SENTINEL in the encode_extend API, not an error: it means
    # "flat scan, no dynamic trip count", which a reverse-differentiated
    # caller passes when it has no budget (see advance's docstring). The fold
    # has no flat mode -- its transposability comes from the scan/cond pair,
    # not from being unchunked -- so 0 means "use the fold's own chunk", the
    # same as None. Only a NEGATIVE chunk is a caller error.
    C = int(chunk) if chunk else default_chunk()
    if C < 0:
        raise ValueError("fold chunk must be positive, got %r" % (C,))
    W = int(window)
    if W <= 0:
        return C, 0, 0
    C = min(C, W)
    nb = -(-W // C)
    return C, nb, nb * C


def extend_fold(agent, carry, tokens, count, *, window, chunk=None,
                init_acc, fold_fn, budget=None, remat=None, parallel=None,
                start=None, row=None):
    """Extend ``carry`` over ``window`` tokens, folding rows into an acc.

    ``fold_fn(acc, rows_c, valid_c, offset) -> acc`` sees one chunk at a time
    (the ``eqns_c`` argument went with the equation-id stream). ``offset`` is
    the chunk's first index IN THE WINDOW and must be
    used by any position-dependent reducer -- ``_face_replay``'s key is
    ``searchsorted(ends, position)``, and dropping the offset would silently
    misattribute every chunk after the first to the wrong face rather than
    raising.

    A chunk entirely past ``count`` still runs, and is a no-op for the same
    reason the existing skip is: ``_step`` freezes the whole carry and emits a
    zero row when a token is invalid. Its cotangent is zero too, so folding it
    changes neither value nor gradient.

    Returns ``(new_carry, acc)``. NOTE the rows are never returned -- that is
    the point; a caller who needs them wants ``encode_extend``.

    THE EPISODE STREAM (``start`` / ``row``). With neither, ``tokens`` is a
    standalone ``(window,)`` buffer and the chunks are a reshape of it --
    the historical path, untouched. With ``start``, ``tokens`` is the
    EPISODE STREAM and chunk ``j`` is ``dynamic_slice(tokens, start + j*C,
    C)``: no per-step window is materialised anywhere. ``row`` additionally
    selects the environment when the stream arrives as ``(E, L)`` (the loss
    holds every environment's row and each sample reads its own), which is
    the same slice with one more leading index.

    THE ROW MUST BE LONG ENOUGH FOR ``padded``, NOT ONLY FOR ``window``:
    ``dynamic_slice`` CLAMPS an out-of-range start rather than raising, so a
    short row returns shifted tokens in silence. ``episode_stream.
    stream_tail`` sizes it; this is the reason that function exists.
    """
    C, nb, padded = plan_chunks(window, chunk)
    W = int(window)
    if start is None and row is None:
        pad = padded - W
        if pad:
            tokens = jnp.concatenate(
                [tokens, jnp.zeros((pad,), tokens.dtype)])
        b_tok = tokens[: nb * C].reshape(nb, C)
    else:
        b_tok = None
        _base = (jnp.zeros((), jnp.int32) if start is None
                 else jnp.asarray(start, jnp.int32))
        if row is None:
            if tokens.ndim != 1:
                raise ValueError(
                    "extend_fold: start= reads a 1-D stream, got shape "
                    f"{tokens.shape}; pass row= for an (E, L) stream")

            def _chunk_tokens(off):
                return lax.dynamic_slice(tokens, (_base + off,), (C,))
        else:
            if tokens.ndim != 2:
                raise ValueError(
                    "extend_fold: row= reads an (E, L) stream, got shape "
                    f"{tokens.shape}")
            _row = jnp.asarray(row, jnp.int32)

            def _chunk_tokens(off):
                return lax.dynamic_slice(
                    tokens, (_row, _base + off), (1, C)).reshape(C)
    cnt = jnp.asarray(count, jnp.int32)

    # DYNAMIC TRIP COUNT, same reasoning as _extend_sequential's budget form.
    # Without this the fold walks every chunk of the window regardless of the
    # real delta length, so raising the bound to 65536 would cost 64x the work
    # of a 1024 delta -- exactly the cost folding exists to remove. A skipped
    # chunk is EXACT, not approximate: all its tokens are invalid, so `_step`
    # freezes the carry and emits zero rows, and a zero row contributes zero
    # to every reducer here (all of them are weighted sums with the weight
    # coming from `valid`). Its cotangent is zero for the same reason.
    #
    # `budget` must be UNBATCHED -- a batch-wide bound -- so that under the
    # loss's vmap the predicate stays scalar and vmap keeps a real `cond`
    # instead of lowering it to `select_n` over both branches, which would
    # compute the skipped chunk anyway and save nothing.
    par = _use_parallel() if parallel is None else bool(parallel)

    if budget is None:
        nb_live = nb
    else:
        nb_live = jnp.minimum(
            (jnp.maximum(jnp.asarray(budget, jnp.int32), 0) + C - 1) // C,
            nb).astype(jnp.int32)

    def _make_run(i, tk):
        off = i * C
        # Tokens remaining once this chunk starts, clipped into [0, C]. A
        # chunk beyond the delta gets 0 and contributes nothing.
        c_cnt = jnp.clip(cnt - off, 0, C)

        def _run(s):
            enc, acc = s
            # THE READ IS INSIDE THE BRANCH (review finding 8). On the
            # streaming path the chunk is a `dynamic_slice` out of the whole
            # episode stream, which is megabytes; hoisting it above the
            # `cond` made every SKIPPED chunk pay its gather anyway, in the
            # forward pass and again under remat in the backward pass. The
            # pre-batched path keeps its `tk`, which is a scan input and
            # therefore already free.
            tk_c = _chunk_tokens(off) if tk is None else tk
            enc2, rows_c, valid_c = _encode_chunk(
                agent, enc, tk_c, c_cnt, C, par)
            return (enc2, fold_fn(acc, rows_c, valid_c, off))

        return _run

    def _body(state, xs):
        if b_tok is None:
            i, tk = xs, None     # read INSIDE `_run`; see above
        else:
            i, tk = xs
        _run = _make_run(i, tk)
        if budget is None:
            return _run(state), None
        return lax.cond(i < nb_live, _run, lambda s: s, state), None

    use_remat = (os.environ.get("ALPHAGRAD_FOLD_REMAT", "1") != "0"
                 if remat is None else bool(remat))

    # COUNT-PROPORTIONAL BACKWARD (owner ruling 2026-09-14). With a budget the
    # scan below is `nb = ceil(window / chunk)` iterations long whatever the
    # real delta is, in the forward pass and again in the backward pass,
    # because reverse-mode AD cannot transpose a `while_loop`. Inside a
    # `custom_vjp` it never has to: `count_vjp.count_loop` runs the live
    # chunks with a `while_loop` in BOTH directions and hand-writes the
    # reverse sweep. See `common/count_vjp.py` for why the numbers do not
    # move and why the residual claim does not either.
    #
    # Only on the differentiated path (`budget is not None`): with no budget
    # the trip count is already the static `nb` and there is nothing to skip.
    # Only with remat: the custom_vjp ALWAYS recomputes the chunk's forward in
    # the backward, so `ALPHAGRAD_FOLD_REMAT=0`, which exists to ask for the
    # stored-residual form, keeps the scan.
    if budget is not None and use_remat and _cvjp.enabled():
        _cf0, _ci0, _cspec = _cvjp.split_inexact(carry)
        if len(_ci0) <= 1:
            def _cbody(i, st):
                cf, acc = st
                # The integer leaf of the encode carry is `pos`, and the chunk
                # body never reads it: `_encode_chunk` either calls the
                # parallel path (which takes its tokens as an argument) or
                # `encode_extend(start=0)`. So freezing it here changes
                # nothing, and its final value is the clipped addition below.
                enc = _cvjp.merge_inexact(cf, _ci0, _cspec)
                _tk = None if b_tok is None else b_tok[i]
                enc2, acc2 = _make_run(i, _tk)((enc, acc))
                _f2, _o2, _ = _cvjp.split_inexact(enc2)
                return (_f2, acc2), None

            (cf_f, acc_f), _ = _cvjp.count_loop(
                _cbody, (_cf0, init_acc), nb=nb, nb_live=nb_live)
            # `pos` advances by `clip(cnt - i*C, 0, C)` on every LIVE chunk,
            # which sums to `min(cnt, nb_live * C)` exactly -- the same
            # integer the scan/cond form arrives at, since the skipped chunks
            # freeze the carry and advance nothing.
            _adv = jnp.clip(cnt, 0, nb_live * C)
            enc_f = _cvjp.merge_inexact(
                cf_f, [x + _adv for x in _ci0], _cspec)
            return enc_f, acc_f

    body = jax.checkpoint(_body) if use_remat else _body
    _xs = (jnp.arange(nb, dtype=jnp.int32) if b_tok is None
           else (jnp.arange(nb, dtype=jnp.int32), b_tok))
    (enc_f, acc_f), _ = lax.scan(body, (carry, init_acc), _xs)
    return enc_f, acc_f


# ---------------------------------------------------------------- reducers --
# Each mirrors one existing consumer. They are written as (init, fold) pairs so
# the equivalence test can drive them against the same reduction applied to the
# full-width rows.

def sum_reducer(embd_dim):
    """``advance``'s weighted row sum and count.

    TWO accumulators, not four: the eqn-owned / structural split is gone with
    the equation ids (see ``carry_stream.advance``).
    """
    init = (jnp.zeros((embd_dim,), jnp.float32), jnp.zeros((), jnp.float32))

    def fold(acc, rows, valid, off):
        tot, n = acc
        w = jnp.asarray(valid, jnp.float32)
        return (tot + jnp.sum(rows * w[:, None], axis=0), n + jnp.sum(w))
    return init, fold


def segment_reducer(embd_dim, n_segments, key_fn):
    """Scatter-sum by a caller-supplied key.

    ``key_fn(n_rows, offset) -> (C,) int32`` receives the OFFSET so a
    position-dependent key (face id via searchsorted) stays correct across
    chunk boundaries.
    """
    init = (jnp.zeros((n_segments, embd_dim), jnp.float32),
            jnp.zeros((n_segments,), jnp.float32))

    def fold(acc, rows, valid, off):
        sums, counts = acc
        w = jnp.asarray(valid, jnp.float32)
        k = key_fn(rows.shape[0], off)
        ok = w * (k >= 0).astype(jnp.float32)
        ks = jnp.clip(k, 0, n_segments - 1)
        return (sums.at[ks].add(rows * ok[:, None]),
                counts.at[ks].add(ok))
    return init, fold


def face_key_fn(face_ends):
    """``_face_replay``'s key: the face whose prefix interval owns the token.

    THE OFFSET IS LOAD-BEARING. The live code computes
    ``searchsorted(ends, arange(rows.shape[0]))`` over the FULL window; under
    chunking the same token sits at ``off + j``, so omitting ``off`` sends
    every chunk after the first to face 0.
    """
    def key(n_rows, off):
        pos = off + jnp.arange(int(n_rows), dtype=jnp.int32)
        return jnp.searchsorted(face_ends, pos, side="right").astype(jnp.int32)
    return key
