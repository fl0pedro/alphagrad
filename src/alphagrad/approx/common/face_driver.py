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

from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES

from alphagrad.approx.common.token_vocab import (
    DELTA_TOKEN_DTYPE as _TOKEN_DTYPE,
    incr_token_vocab,
)
from alphagrad.approx.live_faces import LiveFaceStream

__all__ = [
    "build_live_face_stream",
    "make_face_callbacks",
    "fit_chunk_to_window",
    "make_face_vertex_decide_callback",
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


def _hk_kw(live_faces, env, frh, fsh, n):
    """``{"hist_key": ...}`` for this environment's prefix, or ``{}``.

    EMPTY IS ALWAYS CORRECT. The stream then builds the dense key itself and
    gets the same answer, slowly -- which is what a stand-in stream in a test,
    a stream from a module that predates the chain, and
    `ALPHAGRAD_FACE_KEY_CHAIN=0` all want. So the fast key is passed only when
    there is something to pass and somebody to pass it to, and no caller has
    to know it exists.
    """
    fn = getattr(live_faces, "hist_key", None)
    if fn is None:
        return {}
    k = fn(env, frh, fsh, n)
    return {} if k is None else {"hist_key": k}


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
    # THE one resolver (`common.token_vocab`). The face stream and the
    # observation stream MUST be tokenized at the same id space -- a chunk is
    # a slice of the same emission -- so neither of them names a default.
    vocab = incr_token_vocab(vocab)
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


def fit_chunk_to_window(tok, cnt, window):
    """One face's chunk in the WINDOW BIN's buffer, with the RAW count.

    `window` is the per-step delta window BIN and the declared width of the
    chunk callback's token output. `LiveFaceStream.window` is the stream's
    OWN buffer, which stays at the HARD CAP: the stream then never truncates
    a chunk below the cap, and its prefix tokenizer cache is shared by every
    bin instead of being rebuilt per bin. When the bin is smaller the buffer
    is cut here, and only here.

    THE COUNT THAT RIDES OUT IS NOT CUT. A chunk longer than the bin is a
    WINDOW OVERFLOW, and the rollout has to be able to SEE it: it carries
    `sum(counts) > bin` out as a device flag, and the driver discards the
    whole attempt and repeats the episode one window bin up. Cutting the
    count here instead would truncate the tokens the head reads, which
    changes the ACTION and not merely the padding, and would do it in
    silence -- the defect this replaces (`stats["truncated"]` was its only
    trace).
    """
    W = int(window)
    t = np.asarray(tok)
    c = int(cnt)
    if t.shape[0] == W:
        return t, c
    out = np.zeros((W,), _TOKEN_DTYPE)
    keep = min(c, W, int(t.shape[0]))
    if keep > 0:
        out[:keep] = t[:keep]
    return out, c


def make_face_callbacks(live_faces, *, window, prof_sink=None,
                        edge_table=None, emit_head=False):
    """``(chunk_cb, count_cb)`` -- the device-side face callbacks.

    ``chunk_cb(f, order, spec_hist, step_count, vertex_idx, vertex_specs,
    face_rows, face_skips, face_hist, skip_hist)`` returns
    ``(tokens (window,) uint8, count, endpoints (2,))`` -- the narrow token
    wire, see ``common.token_vocab``. There is no equation-id buffer any more
    (removed 2026-09-13 with the palimpsa relational forget gate).
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

    def _fit(tok, cnt):
        return fit_chunk_to_window(tok, cnt, W)

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
                # THE COMPACT HISTORY KEY, built once per environment per
                # step and handed to every cache the call touches. See
                # `LiveFaceStream.hist_key`: the dense key it replaces is
                # 6.5 megabytes of mostly padding, and it was materialised
                # and hashed about five times per environment per step.
                _hk1 = _hk_kw(
                    live_faces, 0, np.asarray(face_hist),
                    np.asarray(skip_hist), int(np.asarray(step_count)))
                if edge_table is not None:
                    _sc1 = int(np.asarray(step_count))
                    (tok, cnt, _nf, ends, ekey, cvx, head,
                     wrok) = live_faces.chunk_ex(
                        order, spec_hist, _sc1,
                        int(np.asarray(vertex_idx)) + 1, vertex_specs,
                        face_rows, face_skips, int(np.asarray(f)),
                        face_hist, skip_hist, **_hk1,
                    )
                    _dist("face_chunk_len", cnt)
                    tok, cnt = _fit(tok, cnt)
                    out1 = (tok, np.asarray(cnt, np.int32),
                            np.asarray(ends, np.int32),
                            _einfo_host(0, _sc1, ekey, cvx, head, wrok))
                    if emit_head:
                        out1 = out1 + (np.asarray(head, np.int32),)
                    return out1
                tok, cnt, _nf, ends, head = live_faces.chunk(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, vertex_specs,
                    face_rows, face_skips, int(np.asarray(f)),
                    face_hist, skip_hist, **_hk1,
                )
                _dist("face_chunk_len", cnt)
                tok, cnt = _fit(tok, cnt)
                out1 = (tok, np.asarray(cnt, np.int32),
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
            # NARROW WIRE: same dtype as `LiveFaceStream.chunk` produces and
            # as the declared callback shapes below, or `pure_callback`
            # rejects the result.
            toks = np.zeros((B, W), _TOKEN_DTYPE)
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
                _hk = _hk_kw(live_faces, i, _fh[i], _kh[i], int(_sc[i]))
                if edge_table is not None:
                    (tok, cnt, _nf, end, ekey, cvx, head,
                     wrok) = live_faces.chunk_ex(
                        _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                        _vs[i], _fr[i], _fs[i], int(_ff[i]),
                        _fh[i], _kh[i], **_hk,
                    )
                    einf[i] = _einfo_host(i, int(_sc[i]), ekey, cvx,
                                          head, wrok)
                else:
                    tok, cnt, _nf, end, head = live_faces.chunk(
                        _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                        _vs[i], _fr[i], _fs[i], int(_ff[i]),
                        _fh[i], _kh[i], **_hk,
                    )
                _tk_i, _ct_i = _fit(tok, cnt)
                toks[i], cnts[i] = _tk_i, np.int32(_ct_i)
                ends[i] = end
                heads[i] = np.int32(head)
                _dist("face_chunk_len", cnt)
            outB = ((toks, cnts, ends, einf) if edge_table is not None
                    else (toks, cnts, ends))
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
        _shapes = (jax.ShapeDtypeStruct((W,), jnp.dtype(_TOKEN_DTYPE)),
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
                _hk1 = _hk_kw(
                    live_faces, 0, np.asarray(face_hist),
                    np.asarray(skip_hist), int(np.asarray(step_count)))
                _nf1 = int(live_faces.n_faces(
                    order, spec_hist, int(np.asarray(step_count)),
                    int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist,
                    **_hk1))
                _dist("faces_per_vertex", _nf1)
                return np.int32(_nf1)
            # batched: one dispatch per vertex step (see _live_face_host)
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi = np.asarray(vertex_idx)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            _nfs = [int(live_faces.n_faces(
                        _order[i], _sh[i], int(_sc[i]), int(_vi[i]) + 1,
                        _fh[i], _kh[i],
                        **_hk_kw(live_faces, i, _fh[i], _kh[i],
                                 int(_sc[i]))))
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
                    int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist,
                    **_hk_kw(live_faces, 0, np.asarray(face_hist),
                             np.asarray(skip_hist),
                             int(np.asarray(step_count))))
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
                    _fh[i], _kh[i],
                    **_hk_kw(live_faces, i, _fh[i], _kh[i],
                             int(_sc[i])))
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

    THESE MASKS ARE STATIC, AND THAT IS STILL A KNOWN DEFECT FOR SLOT 2
    (dsnn-3qm.59 fault 2, finding 75). ``face_slot_legality`` reads every slot's
    tensor from ONE recording probe per vertex in which no decision has been
    made. Measured on TLM, minimum Markowitz, 5 seeds, through the real apply
    path: that is EXACT for ``lhs`` and ``rhs`` -- nothing at this vertex moves
    the in-edge and out-edge Jacobians, 0 rejections of 36 requests -- and WRONG
    for ``res:new``, which holds their product: 6 of 26 Diag and 4 of 40 Reduce
    rows the mask cleared are refused at apply time once the operands are armed,
    against 0 of 27 with ``new`` armed alone. On the production draw convention
    (all three slots from one head call) the same defect measures 23 of 526 rows
    over 5 seeds, and BOTH exact passes --
    :meth:`~alphagrad.approx.live_faces.LiveFaceStream.decide_faces` and
    :meth:`~alphagrad.approx.live_faces.LiveFaceStream.decide_vertex_faces` --
    measure 0 of 522 on the same walk.

    THE FIX IS :func:`make_face_vertex_decide_callback`, WHICH IS NOT WIRED
    HERE EITHER, and the reason is transport rather than design. Slot 2's mask
    does not exist until slots 0 and 1 are decided and applied, so an exact mask
    cannot be handed to the device loop UP FRONT, which is the only shape this
    callback has. Either the decision moves to the host (one call per vertex,
    rows and masks out together -- that callback) or a second host call
    refreshes slot 2 after a first device pass (two round trips; finding 75
    section 8 costed that shape at 7.06 ms/vertex by composition). The one-call
    shape is the cheaper of the two AND the exact one, and it is affordable
    because :meth:`LiveFaceStream.decide_vertex_faces` runs NO speculative
    elimination and therefore takes its draws outside any jaxpr trace -- the
    ``jax.ensure_compile_time_eval`` escape finding 75's chooser design needed is
    what cost it 8.957 ms/vertex on the head channel.

    What neither can do without ``ppo.py``: ``UnifiedPolicy._face_loop`` decides
    on device inside a ``lax.while_loop`` and DISCARDS the ``(F, S, 3)`` spec
    rows it computes. A host decision has to replace that loop's draw, or a
    device refresh has to read its rows. That is the same transport
    ``--approx-add choose`` and the learned join slots wait at, and it is
    deliberately not half-landed here.
    """
    F, N = int(max_faces), int(max_axes)
    # THE TRAINER CONSUMES EVERY SLOT THE WIDTH HAS, and nothing is narrowed.
    #
    # `face_slot_legality` returns one mask row per slot the entry builder places
    # a hook for, which is `env.wire_slots()`: three under lossy / lossless /
    # choose, four under learned1 (+ the OLD EDGE), five under learned2 (+ the
    # SUMMED EDGE). Until 2026-09-11 the policy's per-slot shapes and the rollout
    # wire stopped at the three CONTRACTION slots, so this callback narrowed to
    # them and RAISED for anything wider. Both are now `wire_slots()` wide
    # (ticket dsnn-3qm.56), so the rows go through untouched and NARROWING would
    # be the defect: a slot the width has would go unscored while the engine
    # still applies its wire row.
    #
    # The prefix assertion below is load-bearing for a different reason now: the
    # head's slot 0/1/2 are the contraction operands by position, so if a future
    # value reordered the bands those three mask rows would belong to other
    # tensors -- the mask/tensor mismatch of finding 72, from the other side.
    from alphagrad.approx.env import FACE_SLOTS as _CONTRACTION_SLOTS
    from alphagrad.approx.env import face_slot_sites as _sites
    from alphagrad.approx.env import wire_slots as _wire_slots
    # THE WIRE'S WIDTH, NOT THE CONTRACTION BAND (2026-09-11, ticket
    # dsnn-3qm.56). This used to be `S = FACE_SLOTS` plus a NotImplementedError
    # for anything wider, because the trainer wire stopped at the contraction
    # band. It no longer does: `UnifiedFacePolicy` sizes every per-slot shape
    # from `head_layout(approx_add).n_slots` and the rollout carries
    # `wire_slots()` rows, so handing the head all of the rows
    # `face_slot_legality` computed is now correct -- and narrowing would be
    # the defect (a slot the width HAS going unscored while the engine applies
    # its row).
    S = int(_wire_slots())
    _topology = _sites()
    _C = int(_CONTRACTION_SLOTS)
    if tuple(x[0] for x in _topology[:_C]) != ("lhs", "rhs", "res:new"):
        raise RuntimeError(
            f"the contraction slots are no longer the prefix of the face slot "
            f"topology ({_topology}); the head's slot 0/1/2 masks would come "
            f"from other tensors.")
    if len(_topology) != S:
        raise RuntimeError(
            f"the face slot topology has {len(_topology)} sites ({_topology}) "
            f"and env.wire_slots() says {S}. Both derive from "
            f"unified_face_head._LAYOUT_SPEC, so they cannot disagree unless "
            f"one of them stopped reading it -- and a mask row per site is the "
            f"invariant finding 72 exists for.")
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _one(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
             env=0):
        sz, qt, pr, cp, no, _n = live_faces.face_slot_legality(
            order, spec_hist, int(np.asarray(step_count)),
            int(np.asarray(vertex_idx)) + 1, face_hist, skip_hist,
            **_hk_kw(live_faces, env, np.asarray(face_hist),
                     np.asarray(skip_hist), int(np.asarray(step_count))))
        return (np.asarray(sz, np.int32)[:F, :S, :N],
                np.asarray(qt, np.float32)[:F, :S, :NUM_FACE_QUANT_DTYPES],
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
                    np.zeros((B, F, S, NUM_FACE_QUANT_DTYPES), np.float32),
                    np.zeros((B, F, S, N, N), np.float32),
                    np.zeros((B, F, S, N), np.float32),
                    np.zeros((B, F, S), np.int32))
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi = np.asarray(vertex_idx)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            for i in range(B):
                got = _one(_order[i], _sh[i], _sc[i], _vi[i], _fh[i],
                           _kh[i], env=i)
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
             jax.ShapeDtypeStruct((F, S, NUM_FACE_QUANT_DTYPES), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S), jnp.int32)),
            order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            vmap_method="broadcast_all")

    return _cb


def replay_stage1_draw(f, s, _legality, rows):
    """The ``draw`` of a STAGE-2 REFRESH: slots 0 and 1 replay the rows a
    device loop already drew (``rows`` is its ``(F, S, 3)`` carry, ``-1`` in
    column 0 for "exact"), slot 2 draws nothing. Driving
    :meth:`~alphagrad.approx.live_faces.LiveFaceStream.decide_vertex_faces`
    with this makes its returned masks the legality of every slot ON THE
    DECIDED OPERANDS, and leaves the slot-2 draw to the device (ticket .59
    fault 2). Shared by ``ppo.py`` and the policy gate so the two cannot
    drift."""
    if s >= 2:
        return None
    r = rows[int(f), int(s)]
    return None if int(r[0]) == -1 else tuple(int(x) for x in r)


def make_face_vertex_decide_callback(live_faces, *, max_faces, max_axes,
                                     draw, prof_sink=None):
    """``cb(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
    skips, draw_args...)`` -> ``(rows (F,S,3) int32, sizes (F,S,N) int32,
    quant (F,S,K) f32, pair (F,S,N,N) f32, comp (F,S,N) f32, n_out (F,S) int32,
    n_faces ())``.

    ONE HOST CALL PER VERTEX, and the rows come back WITH the masks.

    WHY THE ROWS AND THE MASKS IN ONE CALL. Slot 2's tensor IS the contraction
    of slots 0 and 1, so its mask does not exist until they are decided
    (dsnn-3qm.59 fault 2, measured: 23 of 526 rows the STATIC mask cleared are
    refused at apply time). A mask callback that runs before the draw therefore
    cannot answer for slot 2, and a second callback after the draw is a second
    host round trip per vertex -- the shape finding 75 section 8 costed at
    7.06 ms/vertex by composition. Taking the DRAW inside the one call removes
    the round trip instead of adding one, and it is affordable for exactly one
    reason: :meth:`~alphagrad.approx.live_faces.LiveFaceStream.decide_vertex_faces`
    runs NO speculative elimination, so nothing here is inside a jaxpr trace and
    the draw pays no ``jax.ensure_compile_time_eval``. That escape -- not the
    head's arithmetic -- is what cost finding 75's chooser design 8.957
    ms/vertex on the head channel against the static mask's 0.437.

    ``draw`` is a HOST callable ``draw(f, s, legality, *args) -> row | None``,
    where ``args`` are the extra (already-concrete) arrays the callback was
    handed after ``skips``. It is called once per (face, slot) the pass reaches,
    in stage order: every face's slot 0 and slot 1, then every face's slot 2.
    A slot's draw is conditioned on nothing but its OWN mask row
    (``UnifiedFaceHead.sample`` gives slot ``s`` its own key slice and its own
    logit block; 0 of 252 field cells differed between per-slot and joint draws,
    finding 75), which is what makes the stage split produce the joint draw's
    own sample and keeps PPO's stored log-prob over the variable that was acted
    on.

    THE MASKS ARE RETURNED BECAUSE PPO HAS TO STORE THEM. The loss replay
    re-reads ``face_pair_valid`` / ``face_comp_valid`` / ``face_sizes`` /
    ``face_quant`` verbatim and never recomputes them, so the arrays the
    behaviour policy masked with are the arrays ``evaluate`` must rescore with.

    NOT WIRED INTO THE ROLLOUT HERE, and the reason is transport, not design:
    ``UnifiedPolicy._face_loop`` draws on device inside a ``lax.while_loop`` and
    discards the ``(F, S, 3)`` spec rows it computes, and that loop is not this
    module's to restructure. What this callback needs from it is the loop
    REPLACED by this one call's ``rows`` -- the decision moves to the host, and
    the log-prob / entropy / arity rescore runs on device from the returned
    masks, which is the same ``evaluate`` path the loss already uses.
    """
    F, N = int(max_faces), int(max_axes)
    from alphagrad.approx.env import FACE_SLOTS as _CONTRACTION_SLOTS
    from alphagrad.approx.env import face_slot_sites as _sites
    S = int(_CONTRACTION_SLOTS)
    _topology = _sites()
    # The same prefix assertion `make_face_slot_legality_callback` carries, and
    # for the same reason: `decide_vertex_faces` answers the CONTRACTION band
    # only, so if a future value reordered the bands this would hand the head
    # masks belonging to other tensors.
    if tuple(x[0] for x in _topology[:S]) != ("lhs", "rhs", "res:new"):
        raise RuntimeError(
            f"the contraction slots are no longer the prefix of the face slot "
            f"topology ({_topology}).")
    if len(_topology) != S:
        raise NotImplementedError(
            f"decide_vertex_faces answers the {S} contraction slots; this "
            f"--approx-add gives the face {len(_topology)} ({_topology}). The "
            f"OLD EDGE (res:jr) and the SUMMED EDGE (res:jres) depend on "
            f"SIBLING faces' decisions, so they are not a function of a face's "
            f"own operands: use LiveFaceStream.decide_faces for those.")
    _perf = None
    if prof_sink is not None:
        import time as _time
        _perf = _time.perf_counter

    def _one(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
             skips, args, env=0):
        dec = live_faces.vertex_face_decisions(
            order, spec_hist, int(np.asarray(step_count)),
            int(np.asarray(vertex_idx)) + 1,
            lambda f, s, L: draw(f, s, L, *args),
            skips=np.asarray(skips),
            face_rows_hist=face_hist, face_skips_hist=skip_hist,
            **_hk_kw(live_faces, env, np.asarray(face_hist),
                     np.asarray(skip_hist), int(np.asarray(step_count))))
        return (np.asarray(dec.rows, np.int32)[:F, :S, :3],
                np.asarray(dec.sizes, np.int32)[:F, :S, :N],
                np.asarray(dec.quant, np.float32)[:F, :S, :NUM_FACE_QUANT_DTYPES],
                np.asarray(dec.pair, np.float32)[:F, :S, :N, :N],
                np.asarray(dec.comp, np.float32)[:F, :S, :N],
                np.asarray(dec.nout, np.int32)[:F, :S],
                np.asarray(dec.n_faces, np.int32))

    def _host(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
              skips, *args):
        _t0 = _perf() if _perf is not None else None
        try:
            _order = np.asarray(order)
            if _order.ndim == 1:
                return _one(order, spec_hist, step_count, vertex_idx,
                            face_hist, skip_hist, skips, args)
            B = _order.shape[0]
            outs = (np.full((B, F, S, 3), -1, np.int32),
                    np.zeros((B, F, S, N), np.int32),
                    np.zeros((B, F, S, NUM_FACE_QUANT_DTYPES), np.float32),
                    np.zeros((B, F, S, N, N), np.float32),
                    np.zeros((B, F, S, N), np.float32),
                    np.zeros((B, F, S), np.int32),
                    np.zeros((B,), np.int32))
            outs[0][..., 2] = 0
            _sh, _sc = np.asarray(spec_hist), np.asarray(step_count)
            _vi, _sk = np.asarray(vertex_idx), np.asarray(skips)
            _fh, _kh = np.asarray(face_hist), np.asarray(skip_hist)
            _a = [np.asarray(x) for x in args]
            for i in range(B):
                got = _one(_order[i], _sh[i], _sc[i], _vi[i], _fh[i], _kh[i],
                           _sk[i], tuple(x[i] for x in _a), env=i)
                for dst, src in zip(outs, got):
                    dst[i] = src
            return outs
        finally:
            if _perf is not None:
                prof_sink("faces.vertex_decide", _perf() - _t0)

    def _cb(order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            skips, *args):
        return jax.pure_callback(
            _host,
            (jax.ShapeDtypeStruct((F, S, 3), jnp.int32),
             jax.ShapeDtypeStruct((F, S, N), jnp.int32),
             jax.ShapeDtypeStruct((F, S, NUM_FACE_QUANT_DTYPES), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S, N), jnp.float32),
             jax.ShapeDtypeStruct((F, S), jnp.int32),
             jax.ShapeDtypeStruct((), jnp.int32)),
            order, spec_hist, step_count, vertex_idx, face_hist, skip_hist,
            skips, *args, vmap_method="broadcast_all")

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
