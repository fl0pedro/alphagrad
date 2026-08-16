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


def default_chunk() -> int:
    return int(os.environ.get("ALPHAGRAD_FOLD_CHUNK", "1024"))


def extend_fold(agent, carry, tokens, eqns, count, *, window, chunk=None,
                init_acc, fold_fn, remat=None):
    """Extend ``carry`` over ``window`` tokens, folding rows into an acc.

    ``fold_fn(acc, rows_c, valid_c, eqns_c, offset) -> acc`` sees one chunk at
    a time. ``offset`` is the chunk's first index IN THE WINDOW and must be
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
    """
    C = int(chunk if chunk is not None else default_chunk())
    if C <= 0:
        raise ValueError("fold chunk must be positive, got %r" % (C,))
    W = int(window)
    nb = -(-W // C)                       # ceil
    pad = nb * C - W
    if pad:
        tokens = jnp.concatenate([tokens, jnp.zeros((pad,), tokens.dtype)])
        eqns = jnp.concatenate([eqns, jnp.full((pad,), -1, eqns.dtype)])
    b_tok = tokens[: nb * C].reshape(nb, C)
    b_eqn = eqns[: nb * C].reshape(nb, C)
    cnt = jnp.asarray(count, jnp.int32)

    def _body(state, xs):
        enc, acc = state
        i, tk, eq = xs
        off = i * C
        # Tokens remaining once this chunk starts, clipped into [0, C]. A
        # chunk beyond the delta gets 0 and contributes nothing.
        c_cnt = jnp.clip(cnt - off, 0, C)
        enc2, rows_c, valid_c, eqns_c = agent.encode_extend(
            enc, tk, eq, c_cnt, window=C, start=0, chunk=0)
        return (enc2, fold_fn(acc, rows_c, valid_c, eqns_c, off)), None

    use_remat = (os.environ.get("ALPHAGRAD_FOLD_REMAT", "1") != "0"
                 if remat is None else bool(remat))
    body = jax.checkpoint(_body) if use_remat else _body
    (enc_f, acc_f), _ = lax.scan(
        body, (carry, init_acc),
        (jnp.arange(nb, dtype=jnp.int32), b_tok, b_eqn))
    return enc_f, acc_f


# ---------------------------------------------------------------- reducers --
# Each mirrors one existing consumer. They are written as (init, fold) pairs so
# the equivalence test can drive them against the same reduction applied to the
# full-width rows.

def sum_reducer(embd_dim):
    """``advance``'s two weighted sums: eqn-owned rows and structural rows."""
    init = (jnp.zeros((embd_dim,), jnp.float32), jnp.zeros((), jnp.float32),
            jnp.zeros((embd_dim,), jnp.float32), jnp.zeros((), jnp.float32))

    def fold(acc, rows, valid, eqns, off):
        tot_e, n_e, tot_s, n_s = acc
        w = jnp.asarray(valid, jnp.float32)
        w_eqn = w * (eqns >= 0).astype(jnp.float32)
        w_str = w - w_eqn
        return (tot_e + jnp.sum(rows * w_eqn[:, None], axis=0),
                n_e + jnp.sum(w_eqn),
                tot_s + jnp.sum(rows * w_str[:, None], axis=0),
                n_s + jnp.sum(w_str))
    return init, fold


def segment_reducer(embd_dim, n_segments, key_fn):
    """Scatter-sum by a caller-supplied key.

    ``key_fn(eqns_c, offset) -> (C,) int32`` receives the OFFSET so a
    position-dependent key (face id via searchsorted) stays correct across
    chunk boundaries.
    """
    init = (jnp.zeros((n_segments, embd_dim), jnp.float32),
            jnp.zeros((n_segments,), jnp.float32))

    def fold(acc, rows, valid, eqns, off):
        sums, counts = acc
        w = jnp.asarray(valid, jnp.float32)
        k = key_fn(eqns, off)
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
    def key(_eqns, off):
        pos = off + jnp.arange(_eqns.shape[0], dtype=jnp.int32)
        return jnp.searchsorted(face_ends, pos, side="right").astype(jnp.int32)
    return key
