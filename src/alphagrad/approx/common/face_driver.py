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
    "EdgeSlotTable",
]


class EdgeSlotTable:
    """Host-side edge-key -> emem-slot map (--face-edge-mem, dossier sec 8).

    One insertion-ordered dict PER ENV (the batched callback walks envs in
    a stable order, so the loop index IS the env identity, exactly as the
    per-env prefix tokenizers are keyed). A slot is assigned on the FIRST
    EMISSION of a res-edge key -- i.e. when the face that creates/updates
    that edge is being decided for the vertex being eliminated -- and only
    then; lookups (a later face resolving its lhs/rhs OPERAND edge) never
    assign. Past ``capacity`` the OLDEST key is evicted and its slot reused
    (`evictions` in the stats -- a nonzero steady state means K is under-
    sized, not that anything is wrong). The table resets when an env's
    ``step_count`` regresses: a new episode, new graph history, dead keys.

    Determinism: dict order is insertion order, the face loop visits faces
    in order, the batched callback visits envs in order -- so a replayed
    identical episode assigns identical slots (pinned in
    tests/edge_mem_test.py). Callback re-execution is harmless: assignment
    is idempotent per key.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._tables: dict = {}      # env index -> {edge key: slot}
        self._last_step: dict = {}   # env index -> last seen step_count
        self.stats = {"assigned": 0, "evictions": 0, "nonzero_reads": 0,
                      "resets": 0, "lookups": 0}

    def begin(self, env: int, step_count: int):
        """Per-(env, step) prologue: reset on episode restart."""
        last = self._last_step.get(env)
        if last is not None and step_count < last:
            self._tables[env] = {}
            self.stats["resets"] += 1
        self._last_step[env] = int(step_count)

    def assign(self, env: int, key) -> int:
        """Slot of ``key``, assigning (evict-oldest past capacity) if new."""
        t = self._tables.setdefault(env, {})
        s = t.get(key)
        if s is not None:
            return s
        if len(t) < self.capacity:
            s = len(t)
        else:
            oldest = next(iter(t))
            s = t.pop(oldest)
            self.stats["evictions"] += 1
        t[key] = s
        self.stats["assigned"] += 1
        return s

    def lookup(self, env: int, keys) -> int:
        """First assigned slot among ``keys`` (candidate edge ids); -1 if
        none. NEVER assigns -- an unwritten edge must read a zero row."""
        t = self._tables.get(env)
        self.stats["lookups"] += 1
        if t:
            for k in keys:
                s = t.get(k)
                if s is not None:
                    self.stats["nonzero_reads"] += 1
                    return s
        return -1

    def consume_stats(self) -> dict:
        out = dict(self.stats)
        for k in self.stats:
            self.stats[k] = 0
        return out


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


def make_face_callbacks(live_faces, *, window, prof_sink=None,
                        edge_table=None, emit_head=False):
    """``(chunk_cb, count_cb)`` -- the device-side face callbacks.

    ``chunk_cb(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
    face_rows, face_skips, face_hist, skip_hist)`` returns
    ``(tokens (window,), eqn_ids (window,), count, endpoints (2,))``.
    ``endpoints`` is the face's own ``(in_edge, out_edge)`` vertex pair,
    1-based with 0 = "no vertex" -- the face's IDENTITY, which the head
    gathers its two endpoint contexts from. It costs nothing: the face
    enumeration that produces the chunk already computed the key.
    ``count_cb(order, spec_hist, step_count, vertex_idx, face_hist,
    skip_hist)`` returns the vertex's face count.

    ``prof_sink(name, seconds)`` accumulates host time (PPO passes the env
    module's shared sink); ``None`` disables the timing entirely.

    ``edge_table`` (an :class:`EdgeSlotTable`, --face-edge-mem) appends a
    5th output ``einfo (4,) int32 = [lhs_slot, rhs_slot, res_slot, head]``:
    the emem slots of the face's two OPERAND edges (lookup only, -1 =
    never written -> zero row), the slot ASSIGNED to its res edge (first
    emission assigns; -1 for a dropped face), and the chunk's approx-echo
    prefix length (the write path's span correction). With ``None`` the
    callback shapes -- and the flag-off trace -- are exactly the v63 ones.

    ``emit_head`` (``--face-read`` != chunk-mean) appends ONE MORE int32
    scalar after everything else: the chunk's approx-echo prefix length
    ``head``, which the read-point fix masks the head's pooling at
    (``docs/FACE_READ_POINT_TRACE.md`` "Minimal change"). It rides
    separately from the edge-mem ``einfo[3]`` so the two flags compose and
    so ``--face-read chunk-mean`` keeps the historical callback arity --
    i.e. the flag-off TRACE, not merely the flag-off values, is unchanged.
    """
    W = int(window)
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _einfo_host(env_i, step_count, ekey, cvx, head, wrok):
        """One face's edge-slot wire, resolved against the host table."""
        edge_table.begin(env_i, int(step_count))
        lhs = rhs = res = -1
        i, j = int(ekey[0]), int(ekey[1])
        if i >= 0 and j >= 0:
            cands = [int(c) for c in cvx if c >= 0]
            # Lookups FIRST (they can never hit this face's own res edge --
            # operand edges are incident to the central vertex, res edges
            # bypass it -- but the order keeps that a structural fact).
            lhs = edge_table.lookup(env_i, [(i, c) for c in cands])
            rhs = edge_table.lookup(env_i, [(c, j) for c in cands])
            if int(wrok):
                res = edge_table.assign(env_i, (i, j))
        return np.asarray([lhs, rhs, res, int(head)], np.int32)

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
                if edge_table is not None:
                    _sc1 = int(np.asarray(step_count))
                    (tok, ids, cnt, _nf, ends, ekey, cvx, head,
                     wrok) = live_faces.chunk_ex(
                        order, spec_hist, _sc1,
                        int(np.asarray(vertex_idx)) + 1, vertex_specs,
                        face_rows, face_skips, int(np.asarray(f)),
                        face_hist, skip_hist,
                    )
                    _dist("face_chunk_len", cnt)
                    out1 = (tok, ids, np.asarray(cnt, np.int32),
                            np.asarray(ends, np.int32),
                            _einfo_host(0, _sc1, ekey, cvx, head, wrok))
                    if emit_head:
                        out1 = out1 + (np.asarray(head, np.int32),)
                    return out1
                tok, ids, cnt, _nf, ends, head = live_faces.chunk(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, vertex_specs,
                    face_rows, face_skips, int(np.asarray(f)),
                    face_hist, skip_hist,
                )
                _dist("face_chunk_len", cnt)
                out1 = (tok, ids, np.asarray(cnt, np.int32),
                        np.asarray(ends, np.int32))
                if emit_head:
                    out1 = out1 + (np.asarray(head, np.int32),)
                return out1
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
            ends = np.zeros((B, 2), np.int32)
            einf = -np.ones((B, 4), np.int32)
            heads = np.zeros((B,), np.int32)
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi, _vs = np.asarray(vertex_idx), np.asarray(vertex_specs)
            _fr, _fs = np.asarray(face_rows), np.asarray(face_skips)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            _ff = np.asarray(f)
            for i in range(B):
                if edge_table is not None:
                    (tok, ids, cnt, _nf, end, ekey, cvx, head,
                     wrok) = live_faces.chunk_ex(
                        _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                        _vs[i], _fr[i], _fs[i], int(_ff[i]),
                        _fh[i], _kh[i],
                    )
                    einf[i] = _einfo_host(i, int(_sc[i]), ekey, cvx,
                                          head, wrok)
                else:
                    tok, ids, cnt, _nf, end, head = live_faces.chunk(
                        _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                        _vs[i], _fr[i], _fs[i], int(_ff[i]),
                        _fh[i], _kh[i],
                    )
                toks[i], idss[i], cnts[i] = tok, ids, np.int32(cnt)
                ends[i] = end
                heads[i] = np.int32(head)
                _dist("face_chunk_len", cnt)
            outB = ((toks, idss, cnts, ends, einf) if edge_table is not None
                    else (toks, idss, cnts, ends))
            if emit_head:
                outB = outB + (heads,)
            return outB
        finally:
            if _perf is not None:
                prof_sink("faces.live_chunk", _perf() - _pt0)

    def _live_face(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
                   face_rows, face_skips, face_hist, skip_hist):
        # f rides as an OPERAND: inside the while_loop it is a tracer, and
        # a partial would freeze it into the callback as a python object
        # (TracerArrayConversionError at the first body run).
        _shapes = (jax.ShapeDtypeStruct((W,), jnp.int32),
                   jax.ShapeDtypeStruct((W,), jnp.int32),
                   jax.ShapeDtypeStruct((), jnp.int32),
                   jax.ShapeDtypeStruct((2,), jnp.int32))
        if edge_table is not None:
            _shapes = _shapes + (jax.ShapeDtypeStruct((4,), jnp.int32),)
        if emit_head:
            _shapes = _shapes + (jax.ShapeDtypeStruct((), jnp.int32),)
        return jax.pure_callback(
            _live_face_host,
            _shapes,
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


def make_face_sizes_callback(live_faces, *, max_faces, max_axes,
                             prof_sink=None):
    """``sizes_cb(order, spec_hist, step_count, vertex_idx, face_hist,
    skip_hist)`` -> ``((F, N) int32 live per-face dim sizes, (F,) float32
    per-face QUANT legality)``.

    THE SIZES HALF OF ``--per-face-masks`` ON THE ``--live-faces`` PATH.
    A1 wired the head's per-face ``AxisTokenFeatures.size`` to
    ``LiveVertexMaskOracle.face_dim_sizes``, and ``--live-faces`` runs no
    oracle (``ppo._NO_ORACLE``) -- so on the campaign path that half was
    inert and the head kept masking with the STATIC per-VERTEX axis vector.
    :meth:`LiveFaceStream.face_dim_sizes` produces the same quantity from the
    stream's own elimination, and this is its device-side wire.

    Shaped and batched exactly like ``count_cb``: ONE host dispatch per
    VERTEX step for every env (``vmap_method="broadcast_all"``), not one per
    face -- the underlying probe is memoized per (prefix, vertex), so the
    whole face loop is served by two extra eliminations.

    The arrays must ride out of the rollout and be STORED, because they enter
    the masks: the loss re-scores the stored action against them, and a
    replay that recomputes (or drops) them is not the behaviour policy's
    distribution -- the PPO ratio would leave 1 at epoch 0 with no error
    anywhere. ``ppo.sample_action_dynamic`` returns them in ``face_out`` for
    exactly that reason.
    """
    F, N = int(max_faces), int(max_axes)
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _sizes_host(order, spec_hist, step_count, vertex_idx,
                    face_hist, skip_hist):
        _t0 = _perf() if _perf is not None else None
        try:
            _order = np.asarray(order)
            if _order.ndim == 1:
                sz, qt, _n = live_faces.face_dim_sizes(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist)
                return (np.asarray(sz, np.int32)[:F, :N],
                        np.asarray(qt, np.float32)[:F])
            B = _order.shape[0]
            szs = np.zeros((B, F, N), np.int32)
            qts = np.zeros((B, F), np.float32)
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi = np.asarray(vertex_idx)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            for i in range(B):
                sz, qt, _n = live_faces.face_dim_sizes(
                    _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                    _fh[i], _kh[i])
                szs[i] = np.asarray(sz, np.int32)[:F, :N]
                qts[i] = np.asarray(qt, np.float32)[:F]
            return szs, qts
        finally:
            if _perf is not None:
                prof_sink("faces.live_sizes", _perf() - _t0)

    def _sizes_cb(order, spec_hist, step_count, vertex_idx,
                  face_hist, skip_hist):
        return jax.pure_callback(
            _sizes_host,
            (jax.ShapeDtypeStruct((F, N), jnp.int32),
             jax.ShapeDtypeStruct((F,), jnp.float32)),
            order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            vmap_method="broadcast_all")

    return _sizes_cb


def make_face_slot_legality_callback(live_faces, *, max_faces, max_axes,
                                     prof_sink=None):
    """``cb(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist)``
    -> ``(sizes (F,S,N) int32, quant (F,S) f32, pair (F,S,N,N) f32,
    comp (F,S,N) f32, n_out (F,S) int32)``.

    The ``--face-slot-frames`` sibling of :func:`make_face_sizes_callback`
    (ticket .18, D3): :meth:`LiveFaceStream.face_slot_legality` per slot
    instead of :meth:`face_dim_sizes` for the result site. Same batching,
    same memo, same binding (:func:`bind_sizes_callback`); the first two
    outputs replace ``face_sizes`` / ``face_quant`` with a slot axis, the
    next two replace the STATIC ``face_pair_valid`` / ``face_comp_valid`` of
    the live path with per-slot masks, and ``n_out`` feeds the wire encoder.
    All but ``n_out`` enter the head's masks and are stored, so the loss
    re-masks with exactly what the behaviour policy masked with.
    """
    F, N = int(max_faces), int(max_axes)
    S = 3   # lhs, rhs, new -- live_faces._SLOT_SITES
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _one(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist):
        sz, qt, pr, cp, no, _n = live_faces.face_slot_legality(
            order, spec_hist, int(np.asarray(step_count)),
            int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist)
        return (np.asarray(sz, np.int32)[:F, :S, :N],
                np.asarray(qt, np.float32)[:F, :S, :2],
                np.asarray(pr, np.float32)[:F, :S, :N, :N],
                np.asarray(cp, np.float32)[:F, :S, :N],
                np.asarray(no, np.int32)[:F, :S])

    def _host(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist):
        _t0 = _perf() if _perf is not None else None
        try:
            _order = np.asarray(order)
            if _order.ndim == 1:
                return _one(order, spec_hist, step_count, vertex_idx,
                            face_hist, skip_hist)
            B = _order.shape[0]
            outs = (np.zeros((B, F, S, N), np.int32),
                    np.zeros((B, F, S, 2), np.float32),
                    np.zeros((B, F, S, N, N), np.float32),
                    np.zeros((B, F, S, N), np.float32),
                    np.zeros((B, F, S), np.int32))
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi = np.asarray(vertex_idx)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            for i in range(B):
                got = _one(_order[i], _sh[i], _sc[i], _vi[i], _fh[i], _kh[i])
                for dst, src in zip(outs, got):
                    dst[i] = src
            return outs
        finally:
            if _perf is not None:
                prof_sink("faces.live_slot_legality", _perf() - _t0)

    def _cb(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist):
        return jax.pure_callback(
            _host,
            (jax.ShapeDtypeStruct((F, S, N), jnp.int32),
             jax.ShapeDtypeStruct((F, S, 2), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S), jnp.int32)),
            order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            vmap_method="broadcast_all")

    return _cb


def bind_sizes_callback(sizes_cb, order, spec_hist, step_count,
                        face_hist, skip_hist):
    """Bind this step's prefix to ``face_sizes_fn(vertex_idx)``.

    The sibling of :func:`bind_step_callbacks`, kept separate so the
    historical ``(chunk, count)`` pair -- and every caller that unpacks it --
    is untouched when ``--per-face-masks`` is off.
    """

    def face_sizes_fn(_v, _o=order, _s=spec_hist, _k=step_count,
                      _fh=face_hist, _kh=skip_hist):
        return sizes_cb(_o, _s, _k, _v, _fh, _kh)

    return face_sizes_fn


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
