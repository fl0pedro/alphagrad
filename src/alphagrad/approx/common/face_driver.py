"""The `--live-faces` driver: stream construction, host callbacks, step binding.

This machinery used to live inside ``ppo.main()``'s closure, which made it
unimportable -- AlphaZero could not run the same per-face token pipeline PPO
runs without a second copy of it, and a second copy is exactly how the
face-index shift survived unnoticed for months. Everything here is lifted
verbatim; the only differences are that the closed-over values arrive as
arguments and that the prefix-cache capacity is an EXPLICIT caller argument
rather than a formula in ``num_envs`` (AZ builds its env with ``num_envs=0``,
so PPO's ``max(64, 4 * num_envs)`` would degenerate).

Three entry points, in the order a trainer uses them:

``build_live_face_stream``   once, at setup: the :class:`LiveFaceStream`.
``make_face_callbacks``      once: the ``pure_callback`` wrappers around it.
``bind_step_callbacks``      per rollout step: binds this step's prefix
                             (order / specs / step count / face history) so
                             the face loop sees the two-argument shapes the
                             agent expects.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.live_faces import LiveFaceStream

__all__ = [
    "build_live_face_stream",
    "make_face_callbacks",
    "bind_step_callbacks",
]


def _default_window():
    from alphagrad.approx.env import MAX_DELTA_TOKENS
    return int(MAX_DELTA_TOKENS)


def _dist(key, value):
    """Record one size sample into the env module's distribution sink.

    Lazy import (env imports nothing from here, but the reverse is a cycle
    at module scope) and a no-op unless ALPHAGRAD_PROFILE_DIST=1.
    """
    try:
        from alphagrad.approx.env import _dist_add
        _dist_add(key, value)
    except Exception:
        pass


def build_live_face_stream(jaxpr, argnums, consts, args, *, max_faces,
                           max_axes, vocab=None, window=None, cache):
    """The per-face token stream for one graph.

    ``cache`` is the prefix-tokenizer capacity and is REQUIRED: it must be
    sized from the caller's live working set (one prefix per env per vertex
    step for PPO, one for a single-env AZ search), and a formula in
    ``num_envs`` is wrong for a trainer that has none.
    ``ALPHAGRAD_FACE_PREFIX_CACHE`` overrides it, exactly as before.

    ``window`` defaults to the env's per-step delta cap -- a chunk is a slice
    of the step delta, so the delta cap is the one honest window: truncation
    becomes impossible whenever the delta itself fits, and the stored counts
    stay exact for the loss's cumsum boundaries.
    """
    if vocab is None:
        vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))
    if window is None:
        window = _default_window()
    return LiveFaceStream(
        jaxpr, argnums, consts, args,
        vocab=int(vocab),
        max_faces=int(max_faces),
        max_axes=int(max_axes),
        window=int(window),
        cache=int(os.environ.get("ALPHAGRAD_FACE_PREFIX_CACHE", str(cache))),
    )


def make_face_callbacks(live_faces, *, window, prof_sink=None):
    """``(chunk_cb, count_cb)`` -- the device-side face callbacks.

    ``chunk_cb(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
    face_rows, face_skips, face_hist, skip_hist)`` returns
    ``(tokens (window,), eqn_ids (window,), count)``.
    ``count_cb(order, spec_hist, step_count, vertex_idx, face_hist,
    skip_hist)`` returns the vertex's face count.

    ``prof_sink(name, seconds)`` accumulates host time (PPO passes the env
    module's shared sink); ``None`` disables the timing entirely.
    """
    W = int(window)
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _live_face_host(order, spec_hist, step_count, vertex_idx,
                        vertex_specs, face_rows, face_skips, f,
                        face_hist, skip_hist):
        # face_hist/skip_hist are the PREFIX's per-face wires -- the (N,
        # MAX_FACES, FACE_SLOTS, 3) / (N, MAX_FACES) history arrays carried by
        # the env state, aligned with `order` exactly like `spec_hist` is.
        # Without them the prefix replay rebuilt an EXACT graph while the
        # measurement built the approximated one, and the prefix cache keyed
        # two different plans to one tokenizer.
        _pt0 = _perf() if _perf is not None else None
        try:
            _order = np.asarray(order)
            if _order.ndim == 1:
                tok, ids, cnt, _nf = live_faces.chunk(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, vertex_specs,
                    face_rows, face_skips, int(np.asarray(f)),
                    face_hist, skip_hist,
                )
                _dist("face_chunk_len", cnt)
                return tok, ids, np.asarray(cnt, np.int32)
            # BATCHED (vmap_method="broadcast_all"): ONE host dispatch per
            # face substep for all envs. The sequential vmap ran E separate
            # callbacks with a device round-trip between each — the GPU
            # idled through E dispatch+sync latencies per substep (the
            # dominant share of the approx-vs-exact non-host gap). Values
            # are identical: the same per-env chunk() calls, in env order.
            B = _order.shape[0]
            toks = np.zeros((B, W), np.int32)
            idss = np.zeros((B, W), np.int32)
            cnts = np.zeros((B,), np.int32)
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi, _vs = np.asarray(vertex_idx), np.asarray(vertex_specs)
            _fr, _fs = np.asarray(face_rows), np.asarray(face_skips)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            _ff = np.asarray(f)
            for i in range(B):
                tok, ids, cnt, _nf = live_faces.chunk(
                    _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                    _vs[i], _fr[i], _fs[i], int(_ff[i]),
                    _fh[i], _kh[i],
                )
                toks[i], idss[i], cnts[i] = tok, ids, np.int32(cnt)
                _dist("face_chunk_len", cnt)
            return toks, idss, cnts
        finally:
            if _perf is not None:
                prof_sink("faces.live_chunk", _perf() - _pt0)

    def _live_face(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
                   face_rows, face_skips, face_hist, skip_hist):
        # f rides as an OPERAND: inside the while_loop it is a tracer, and
        # a partial would freeze it into the callback as a python object
        # (TracerArrayConversionError at the first body run).
        return jax.pure_callback(
            _live_face_host,
            (jax.ShapeDtypeStruct((W,), jnp.int32),
             jax.ShapeDtypeStruct((W,), jnp.int32),
             jax.ShapeDtypeStruct((), jnp.int32)),
            order, spec_hist, step_count, vertex_idx, vertex_specs,
            face_rows, face_skips, f, face_hist, skip_hist,
            vmap_method="broadcast_all",
        )

    def _live_face_count_host(order, spec_hist, step_count, vertex_idx,
                              face_hist, skip_hist):
        # TIMED, like its sibling. This callback was the only host stage in
        # the rollout with no prof sink, and it is the one that pays the
        # prefix-tokenizer miss: `n_faces` calls `_tokenizer_at`, so the
        # FIRST request at each new step rebuilds the whole prefix while the
        # chunk callbacks that follow hit the entry it just built. Untimed,
        # that cost surfaced only as a device-side gap and was read as
        # `env.step` device time for two rounds of profiling.
        _ct0 = _perf() if _perf is not None else None
        try:
            _order = np.asarray(order)
            if _order.ndim == 1:
                _nf1 = int(live_faces.n_faces(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist))
                _dist("faces_per_vertex", _nf1)
                return np.int32(_nf1)
            # batched: one dispatch per vertex step (see _live_face_host)
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi = np.asarray(vertex_idx)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            _nfs = [int(live_faces.n_faces(_order[i], _sh[i], int(_sc[i]),
                                           int(_vi[i]) + 1, _fh[i], _kh[i]))
                    for i in range(_order.shape[0])]
            for _n in _nfs:
                _dist("faces_per_vertex", _n)
            return np.asarray(_nfs, np.int32)
        finally:
            if _perf is not None:
                prof_sink("faces.live_count", _perf() - _ct0)

    def _live_face_count(order, spec_hist, step_count, vertex_idx,
                         face_hist, skip_hist):
        return jax.pure_callback(
            _live_face_count_host, jax.ShapeDtypeStruct((), jnp.int32),
            order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            vmap_method="broadcast_all")

    return _live_face, _live_face_count


def bind_step_callbacks(chunk_cb, count_cb, order, spec_hist, step_count,
                        face_hist, skip_hist):
    """Bind this step's prefix to ``(face_chunk_fn, face_count_fn)``.

    The FULL per-face history rides along: the face loop supplies the CURRENT
    vertex's in-flight decisions (``_rows``/``_skips``), while ``face_hist``/
    ``skip_hist`` are every decision already committed to the prefix. The
    prefix replay needs the latter or it rebuilds an exact graph the
    measurement never builds.
    """

    def face_chunk_fn(_f, _v, _vspecs, _rows, _skips,
                      _o=order, _s=spec_hist, _k=step_count,
                      _fh=face_hist, _kh=skip_hist):
        return chunk_cb(_f, _o, _s, _k, _v, _vspecs, _rows, _skips, _fh, _kh)

    def face_count_fn(_v, _o=order, _s=spec_hist, _k=step_count,
                      _fh=face_hist, _kh=skip_hist):
        return count_cb(_o, _s, _k, _v, _fh, _kh)

    return face_chunk_fn, face_count_fn
