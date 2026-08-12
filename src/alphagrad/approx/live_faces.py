"""The token chunk palimpsa reads before it approximates face ``f``.

WHAT WAS WRONG. The rollout ran ONE ``encode_extend`` per env step, fed both
heads from the same ``heads_from_memory`` summary, and took a single env action
per vertex. So the approximation head decided every face of a vertex BEFORE the
env had emitted any of that vertex's contraction tokens -- it was approximating
a contraction it had never read, and every face looked identical to it. Handing
it the face's axis SIZES (``LiveVertexMaskOracle.face_features``) made the faces
distinguishable but is not the same thing as reading them.

WHAT THIS DOES. Reproduces, host-side, the pipeline

    base => palimpsa -> VE head
         => face_1,1 contraction
         => palimpsa -> approximation head
         => approximated contraction & face_1,2 contraction
         => palimpsa -> approximation head
         => ... (all faces of vertex 1)
         => palimpsa -> VE head => face_2,1 contraction => ...

``=>`` is a token handoff (new tokens, palimpsa re-encodes); ``->`` is a head
delegation. :meth:`LiveFaceStream.chunk` returns exactly one ``=>`` chunk: the
approximation face ``f-1`` produced, followed by face ``f``'s contraction. The
tokens come from graphax's own ``IncrementalPathTokenizer`` -- the same emitters
and the same vocabulary as the observation stream, not a parallel synthetic one.

WHY IT RE-ELIMINATES PER FACE. graphax exposes no resumable elimination: the
per-face hook is a callback inside ``_eliminate_vertex``, so there is no way to
stop at face ``f``, return to the caller, and continue. Face ``f``'s chunk is
therefore produced by re-running the vertex with faces ``0..f-1`` carrying their
DECIDED approximations and a recording hook on face ``f``. That is ``n_faces``
eliminations per env step rather than one.

Correctness of that replay hinges on one thing: a re-run must give the same
tokens for the faces it repeats. Variable names are drawn from a sequential
generator, and re-tracing makes NEW ``Var`` objects, so without care run ``f``
would name the same variables differently from run ``f-1`` and cross-face
coreference in the stream would be noise. :func:`_snapshot` therefore rewinds
the name generator along with ``_names``/``_fns``, which it can do because names
are drawn one per entry of those two dicts. With it rewound, run ``f`` creates
its Vars in the same order and hands them the same names as run ``f-1`` for
every face they share.

The prefix -- everything eliminated before this step -- is replayed ONCE per
distinct prefix and cached; only the current vertex is re-eliminated per face.
"""
from __future__ import annotations

import os
import warnings

import numpy as np


# Sized from the MEASURED distribution on the MNIST xent graph, not guessed:
# 192 chunks, mean 403 tokens, max 1456. At 96 the window clipped 83 of 206
# chunks -- i.e. 40% of the time the head read only the TAIL of the
# contraction it was approximating, which is the same blindness this module
# removes, just quieter. 1024 clips 8 of 192 (4%). Raise it if
# `truncated` in the health line is a large fraction of `chunks`.
FACE_TOKEN_WINDOW = int(os.environ.get("ALPHAGRAD_FACE_TOKEN_WINDOW", "1024"))


# One process-wide shout the first time the emitted faces are a PROPER
# subsequence of the enumerated ones with something still emitted -- the
# only shape of divergence that can mis-index, and the one never yet
# observed. Total drops (every face of a vertex) are routine and ride the
# `face_dropped` counter instead of shouting once per run.
_PARTIAL_DROP_WARNED = [False]


def _copy_graph(g):
    from alphagrad.approx.common.masks import _shallow_copy_graph
    return _shallow_copy_graph(g)


class _Snapshot:
    """Everything one speculative ``eliminate`` mutates, restored on exit.

    The tokenizer and its ``IncrementalJaxpr`` are extended IN PLACE (that is
    the point of the incremental builder -- the persistent trace is what makes
    a step cheap), so a speculative elimination has to be undone rather than
    run on a throwaway copy: rebuilding the builder would mean replaying the
    whole prefix.

    ``graph``/``tgraph``/``vo`` are swapped for copies so the originals are
    never touched. The append-only lists (traced equations, steps, face
    records, transform records) are truncated back. ``_namegen`` cannot be
    rewound, so it is rebuilt and fast-forwarded to the number of names drawn
    -- one per ``_names`` entry plus one per ``_fns`` entry, the only two
    consumers.
    """

    def __init__(self, tk):
        self.tk = tk
        ij = tk.ij
        self.ij = ij
        self.graph, self.tgraph, self.vo = ij.graph, ij.tgraph, ij.vo
        self.n_eqns = len(ij.trace.frame.tracing_eqns)
        self.n_steps = len(ij.steps)
        self.n_faces = (len(ij.face_sink.faces)
                        if ij.face_sink is not None else 0)
        self.n_xlog = len(ij.xlog.records)
        self.names = dict(tk._names)
        self.fns = dict(tk._fns)
        self.n_names = len(self.names) + len(self.fns)
        self.eqn_seg = tk._eqn_seg
        self.tk_steps = tk._n_steps
        self.flatten_uid = tk._flatten_uid

    def __enter__(self):
        ij = self.ij
        ij.graph = _copy_graph(self.graph)
        ij.tgraph = _copy_graph(self.tgraph)
        ij.vo = dict(self.vo) if isinstance(self.vo, dict) else self.vo
        return self

    def __exit__(self, *exc):
        from graphax.jaxpr import name_gen_python_style
        ij, tk = self.ij, self.tk
        ij.graph, ij.tgraph, ij.vo = self.graph, self.tgraph, self.vo
        del ij.trace.frame.tracing_eqns[self.n_eqns:]
        del ij.steps[self.n_steps:]
        if ij.face_sink is not None:
            del ij.face_sink.faces[self.n_faces:]
        del ij.xlog.records[self.n_xlog:]
        tk._names = self.names
        tk._fns = self.fns
        tk._eqn_seg = self.eqn_seg
        tk._n_steps = self.tk_steps
        tk._flatten_uid = self.flatten_uid
        tk._namegen = name_gen_python_style(
            tk.digit_base, tk.digit_base + tk._name_alphabet)
        for _ in range(self.n_names):
            next(tk._namegen)
        return False


# Serve a prefix miss by extending the n-1 tokenizer by one vertex instead of
# replaying the whole prefix. ALPHAGRAD_FACE_PREFIX_EXTEND=0 restores the
# O(T^2) rebuild (kept as an A/B switch, not because the rebuild is wanted).
_PREFIX_EXTEND = os.environ.get("ALPHAGRAD_FACE_PREFIX_EXTEND", "1") == "1"

class LiveFaceStream:
    """Per-face token chunks for one graph, cached across env steps."""

    def __init__(self, jaxpr, argnums, consts, args, *, vocab: int,
                 max_faces: int = 8, max_axes: int = 8,
                 window: int = FACE_TOKEN_WINDOW, cache: int = 64):
        # ``cache`` is a PREFIX-tokenizer capacity, and with the face wires in
        # the key the live working set is one prefix per ENV per vertex step
        # (all faces of one vertex share it, nothing else does). Below the env
        # count the FIFO evicts entries the very next face substep needs, so
        # callers size it from num_envs -- see ppo.py.
        self.jaxpr = jaxpr
        self.argnums = tuple(argnums)
        self.consts = list(consts)
        self.args = list(args)
        self.vocab = int(vocab)
        self.max_faces = int(max_faces)
        self.max_axes = int(max_axes)
        self.window = int(window)
        self.cache_cap = int(cache)
        self._prefix: dict = {}       # prefix key -> tokenizer at that prefix
        self._chunks: dict = {}       # full key -> result tuple
        # tok_total/tok_max/chunks size the WINDOW from the real
        # distribution: a window below the typical chunk silently keeps only
        # the tail of the contraction the head is meant to read.
        self.stats = {"prefix_miss": 0, "prefix_hit": 0, "elims": 0,
                      # `prefix_ext`: misses served by extending the n-1
                      # tokenizer by ONE vertex instead of replaying the
                      # whole prefix (see `_tokenizer_at`).
                      "prefix_ext": 0,
                      "chunk_hit": 0, "failures": 0, "truncated": 0,
                      "tok_total": 0, "tok_max": 0, "chunks": 0,
                      # The face <-> segment correspondence (see `chunk`).
                      # `face_key_seg_mismatch` / `face_dropped` are
                      # SURVIVABLE and mapped around; the other two are
                      # structural corruption and raise. They are named
                      # separately so the health line says which, instead
                      # of everything landing in `failures` next to a
                      # graph graphax simply could not trace.
                      "face_key_seg_mismatch": 0,
                      "face_dropped": 0,
                      "face_seg_not_tiled": 0,
                      "face_header_mismatch": 0}

    # -- prefix ------------------------------------------------------------
    @staticmethod
    def _hist(face_rows_hist, face_skips_hist):
        """``(rows, skips)`` as int32 arrays, or ``(None, None)``."""
        if face_rows_hist is None or face_skips_hist is None:
            return None, None
        return (np.asarray(face_rows_hist, np.int32),
                np.asarray(face_skips_hist, np.int32))

    def _tokenizer_at(self, order, specs, n, face_rows_hist=None,
                      face_skips_hist=None):
        from graphax import IncrementalPathTokenizer
        from alphagrad.approx.env import decode_vertex_rule_specs
        from alphagrad.approx.common.masks import make_live_masked_hook

        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        # THE PREFIX'S FACE WIRES BELONG IN THE KEY. Under --live-faces ppo.py
        # sets the per-vertex specs to all-exact END rows for every vertex --
        # approximation is purely per-face -- so ``specs[:n]`` is a CONSTANT
        # and a key built from (order, specs) alone degenerates to the vertex
        # ORDER. Every plan sharing an elimination order was then served one
        # tokenizer no matter what the face head had decided.
        key = (order[:n].tobytes(), specs[:n].tobytes(),
               b"" if frh is None else frh[:n].tobytes(),
               b"" if fsh is None else fsh[:n].tobytes())
        hit = self._prefix.get(key)
        if hit is not None:
            self.stats["prefix_hit"] += 1
            return hit
        self.stats["prefix_miss"] += 1

        def _apply(tk, k):
            """Replay prefix vertex ``k`` onto ``tk``.

            Lifted verbatim out of the cold loop so the extend path below
            and the cold path apply the SAME sequence of operations to the
            same tokenizer state -- that identity is what makes the extend
            observationally invisible.
            """
            v = int(order[k])
            try:
                rules = decode_vertex_rule_specs(
                    self.jaxpr, v, specs[k], is_last=False)
            except Exception:
                rules = ()
            # Hook-wrapped exactly like the measurement path: a rule that does
            # not fit one face's operand is skipped for that face instead of
            # raising, which is what keeps key enumeration alive.
            hooks = (make_live_masked_hook(tuple(rules)),) if rules else ()
            # ... AND SO DO THE PREFIX'S APPROXIMATIONS. This loop used to
            # pass no face transforms at all, so every chunk the head read was
            # computed on an EXACT prefix while the measurement
            # (env._face_transforms_for_order -> ft_by_vertex) built an
            # approximated one: the observation and the measured object
            # diverged. ``is_last=False`` is what that builder uses for every
            # vertex but the last of the ORDER, and a prefix vertex here is
            # never that one.
            ft = None
            if frh is not None:
                try:
                    _keys, ft = self._decided(
                        tk, v, frh[k], fsh[k], int(frh.shape[1]),
                        is_last=False)
                except Exception:
                    ft = None
            tk.eliminate(v, hooks, ft or None)

        # POP-EXTEND. A rollout asks for prefixes 1, 2, 3, ... in order, and
        # the key grows by one vertex each time, so EVERY step missed and
        # rebuilt the tokenizer from `base_tokens()` by replaying all n
        # eliminations. That is O(T^2) per episode: measured at 1.8 ms per
        # replayed vertex it ramped the per-decision cost from 17 ms at
        # step 0 to ~190 ms at step 94 (~8 s of the ~13 s rollout), and it
        # was invisible because it runs on the host inside the face-count
        # `pure_callback` -- device-side timers charged it to `env.step`.
        #
        # The n-1 tokenizer is already in the cache and, once step n starts,
        # nothing will ask for it again. POP it (rather than copy: a copy of
        # the whole IncrementalJaxpr per step is the cost we are removing)
        # and push it forward by the single new vertex. Popping is what keeps
        # this invisible -- no live entry is ever mutated under a reader, and
        # any consumer that really does want the n-1 prefix back simply takes
        # a cold rebuild, exactly as it would have on any other eviction.
        #
        # `order[:n-1]` / `specs[:n-1]` / the face wires below index n-1 are
        # bitwise stable across the step: `env.step` only shift-and-inserts
        # at `idx = step_count`, so rows below it never move.
        tk = None
        if _PREFIX_EXTEND and n > 0:
            pkey = (order[:n - 1].tobytes(), specs[:n - 1].tobytes(),
                    b"" if frh is None else frh[:n - 1].tobytes(),
                    b"" if fsh is None else fsh[:n - 1].tobytes())
            tk = self._prefix.pop(pkey, None)
            if tk is not None:
                try:
                    _apply(tk, n - 1)
                    self.stats["prefix_ext"] += 1
                except Exception:
                    # Half-applied: discard and take the cold path, which
                    # rebuilds from scratch and cannot see the damage.
                    tk = None
        if tk is None:
            tk = IncrementalPathTokenizer(
                self.jaxpr, self.argnums, list(self.consts), list(self.args),
                vocab_size=self.vocab)
            tk.base_tokens()
            for k in range(n):
                _apply(tk, k)
        if len(self._prefix) >= self.cache_cap:
            for dk in list(self._prefix)[: max(1, self.cache_cap // 4)]:
                self._prefix.pop(dk, None)
        self._prefix[key] = tk
        return tk

    # -- decoded per-face transforms for the DECIDED faces -----------------
    def _decided(self, tk, vertex, face_rows, face_skips, upto,
                 is_last=True):
        """``{face_key: slots|SKIP_FACE}`` for faces ``0..upto-1``.

        ``is_last`` mirrors ``env._face_dict_for_vertex``'s
        ``is_last_honored``: True for the vertex whose faces are being decided
        right now (the newest of the prefix), False when replaying an OLDER
        prefix vertex, which is exactly how ``_face_transforms_for_order``
        decodes it.
        """
        from graphax import SKIP_FACE
        from alphagrad.approx.env import (
            FACE_SLOTS, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs)
        from alphagrad.approx.common.masks import make_live_masked_hook

        keys = list(tk.ij.faces(int(vertex)))
        ft: dict = {}
        for f in range(min(upto, len(keys))):
            if int(face_skips[f]) == 1:
                ft[keys[f]] = SKIP_FACE
                continue
            slots = []
            for s in range(FACE_SLOTS):
                row = [list(int(x) for x in face_rows[f][s])] + [
                    [-1, -1, 0]] * (MAX_RULES_PER_VERTEX - 1)
                try:
                    # is_last=True: the vertex the face loop is deciding is
                    # the NEWEST of the prefix, which is exactly when
                    # decode_vertex_rule_specs admits COMPRESS. Decoding it
                    # with is_last=False drops every COMPRESS row SILENTLY,
                    # so the chunk would describe an exact contraction while
                    # the env applied a reduction.
                    rules = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row, is_last=bool(is_last))
                except Exception:
                    rules = ()
                slots.append(make_live_masked_hook(tuple(rules))
                             if rules else None)
            if any(sl is not None for sl in slots):
                ft[keys[f]] = tuple(slots)
        return keys, ft

    # -- what the elimination ACTUALLY emitted ------------------------------
    @staticmethod
    def _emitted(tk, keys, f):
        """``(emitted face keys, expected `path` header tokens of ``keys[f]``)``.

        MUST be called INSIDE the :class:`_Snapshot` block. ``__exit__``
        truncates ``face_sink.faces`` back to its pre-elimination length and
        restores ``tk._names``, so the FaceRecords that pair a segment with
        its ``(in_edge, out_edge)`` key -- and the names their header was
        emitted under -- exist only in there.

        The header is rebuilt with the tokenizer's OWN emitter, so it is the
        exact byte sequence graphax would write. ``_var_name`` is a pure
        lookup at this point (the header was just emitted for these vars),
        and even if it were not, the snapshot restores ``_names`` and rewinds
        ``_namegen`` on exit, so nothing can leak into the stream.
        """
        ij = tk.ij
        sink = getattr(ij, "face_sink", None)
        if sink is None or not ij.steps:
            return None, None
        vidx = sink.vidx
        if vidx is None:
            from graphax.core import _vidx_for
            vidx = _vidx_for(ij.jaxpr)
        recs = list(ij.step_faces(len(ij.steps) - 1))
        ekeys = [(vidx.get(fr.in_edge), vidx.get(fr.out_edge)) for fr in recs]
        hdr = None
        if 0 <= f < len(keys) and keys[f] in ekeys:
            hdr = []
            tk._emit_face_header(recs[ekeys.index(keys[f])], hdr)
            hdr = [int(t) for t in hdr]
        return ekeys, hdr

    # -- the chunk ---------------------------------------------------------
    def chunk(self, order, specs, n, vertex, vertex_specs,
              face_rows, face_skips, f,
              face_rows_hist=None, face_skips_hist=None):
        """``(tokens (W,), eqn_ids (W,), count, n_faces)``.

        ``tokens`` is the ``=>`` handoff before face ``f``'s decision: face
        ``f-1``'s approximation equations followed by face ``f``'s contraction
        (face ``0`` gets the contraction alone).

        This is the TOKEN channel only. Axis sizes and DIAG/COMPRESS legality
        still come from :class:`LiveVertexMaskOracle` -- they must, because the
        loss re-scores the stored action against the STORED masks and the two
        have to be derived from the same operand or the gcd the head used and
        the mask it was drawn under disagree.

        Everything fails soft to an empty chunk: a graph graphax cannot trace
        must not take the trainer down, and an empty chunk simply leaves the
        palimpsa carry where it was, so the head decides on the vertex context
        alone (the old behaviour) for that face only.
        """
        W = self.window
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex, f = int(n), int(vertex), int(f)
        rows = np.asarray(face_rows, np.int32)
        skips = np.asarray(face_skips, np.int32)
        vspecs = np.asarray(vertex_specs, np.int32)
        empty = (np.zeros((W,), np.int32), -np.ones((W,), np.int32),
                 np.int32(0), np.int32(0))

        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        ck = (order[:n].tobytes(), specs[:n].tobytes(), vertex,
              vspecs.tobytes(), rows[:f].tobytes(), skips[:f].tobytes(), f,
              b"" if frh is None else frh[:n].tobytes(),
              b"" if fsh is None else fsh[:n].tobytes())
        hit = self._chunks.get(ck)
        if hit is not None:
            self.stats["chunk_hit"] += 1
            return hit

        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh)
        except Exception:
            self.stats["failures"] += 1
            return empty
        try:
            keys, ft = self._decided(tk, vertex, rows, skips, f)
        except Exception:
            self.stats["failures"] += 1
            return empty
        n_faces = len(keys)
        if f >= n_faces:
            res = (empty[0], empty[1], empty[2], np.int32(n_faces))
            self._chunks[ck] = res
            return res

        # NO hook on face f. Installing a callable is not free: it puts
        # graphax's core on its approx code path, which produces a genuinely
        # different index structure (see probe_faces), so a recording hook on
        # an UNDECIDED face would make the head read a contraction the
        # measurement will not build. Faces 0..f-1 carry hooks precisely
        # because the real run will carry them too.

        # The PER-VERTEX micro action is already sampled by the time the face
        # loop runs, and it applies to every face, so the contraction the head
        # reads has to carry it -- otherwise the tokens describe a graph the
        # measurement will never build.
        from alphagrad.approx.env import decode_vertex_rule_specs
        from alphagrad.approx.common.masks import make_live_masked_hook
        try:
            vrules = decode_vertex_rule_specs(
                self.jaxpr, vertex, vspecs.tolist(), is_last=True)
        except Exception:
            vrules = ()
        vhooks = (make_live_masked_hook(tuple(vrules)),) if vrules else ()

        with _Snapshot(tk):
            try:
                self.stats["elims"] += 1
                toks = [int(t) for t in tk.eliminate(vertex, vhooks, ft)]
                ids = [int(g) for g in tk.last_eqn_ids()]
                segs = tk.last_face_segments()
                # Everything the checks below need that dies with the
                # snapshot. Pure reads -- they must not raise in here or
                # the except would file a structural defect as a soft
                # "failure" and hand back an empty chunk.
                ekeys, exp_hdr = self._emitted(tk, keys, f)
            except Exception:
                self.stats["failures"] += 1
                return empty

        # Segment index == FACE index. A skipped face is NOT absent from the
        # stream: since graphax 00a60fa it emits its `path` header with an
        # EMPTY contraction block plus a lone `approx SKIP {}`, and it gets
        # its own segment entry. Do NOT shift by the number of earlier skips
        # -- that reads segment f-k for face f, which silently hands the head
        # another face's contraction AND another face's approximation tail,
        # so it never sees its own skip decision.
        #
        # BUT `gi = f` is only sound while THE KEY LIST AND THE SEGMENT LIST
        # AGREE, and graphax does not promise that. `faces_of` -- the list the
        # head's action space and `n_faces` (the rollout while_loop's trip
        # count) are BOTH sized from -- enumerates OPTIMISTICALLY: a face whose
        # edge Jacobian forces to None (an unevaluated LazyEdge behind a
        # stop_gradient, say) is never visited and never emitted, so `keys` is
        # a SUPERSET of the emitted faces.
        #
        # MEASURED, not hypothetical. Per full elimination: Perceptron (the
        # graph tests/delta_buffer_equivalence_test.py drives) vertex 11
        # enumerates 4 keys and emits 0 (forward order), vertex 12 enumerates
        # 1 and emits 0 (reverse); the flagship nn256 drops 8 of 84 enumerated
        # faces (vertices 13 and 16, 4 keys -> 0) forward and 1 of 25 reverse.
        # So this is the steady state, not an alarm -- which is exactly why it
        # must be mapped around and COUNTED rather than raised on.
        #
        # Two different things hide behind it, and one silent `failures` bump
        # used to cover both:
        #
        #   * face f itself was not emitted. The head has already DECIDED an
        #     approximation for it, and paid log-prob for that decision, on a
        #     contraction that does not exist. There is no chunk to hand back,
        #     so the empty chunk stands -- but under its own name,
        #     `face_dropped`, instead of disappearing into the same counter as
        #     a graph graphax could not trace. A persistently non-zero
        #     `face_dropped` means the action space is wider than the stream,
        #     which only the policy side can fix.
        #
        #   * face f WAS emitted but an earlier one was not, so segment f
        #     belongs to a later-keyed face. That is a real mis-index -- the
        #     same defect class as the `skipped_before` shift 5036daf removed,
        #     from a different cause -- and it was completely silent. Map by
        #     the face's OWN key instead of by position: a lookup in a <=12
        #     entry list, and it cannot mis-index. (A SKIPPED face is a
        #     different thing and needs no compensation: it IS emitted and DOES
        #     get a segment, so its key is in `ekeys` like any other.)
        if ekeys is None or len(ekeys) != len(segs):
            self.stats["failures"] += 1
            return empty
        if ekeys != keys:
            self.stats["face_key_seg_mismatch"] += 1
            if ekeys and not _PARTIAL_DROP_WARNED[0]:
                # Partial drop: the case positional indexing gets WRONG rather
                # than merely empty. Never observed -- say so out loud once.
                _PARTIAL_DROP_WARNED[0] = True
                warnings.warn(
                    f"[alphagrad.approx.live_faces] vertex {vertex}: faces_of "
                    f"enumerated {keys} but the elimination emitted {ekeys} -- "
                    f"a PARTIAL drop. Chunks are now mapped by face key, so "
                    f"the head still reads its own face; positional indexing "
                    f"would have handed it a later face's contraction. See "
                    f"`face_key_seg_mismatch` in the face-stream health line.",
                    stacklevel=2)

        # INVARIANT: THE CHUNKS CONCATENATE TO EXACTLY THE STEP DELTA.
        # `_face_replay` scores the stored actions by pooling rows
        # [cumsum(counts)[f-1] : cumsum(counts)[f]) of ONE scan over the
        # stored emission, so its boundaries address the window the head
        # actually read only if the segments TILE that emission: first starts
        # at 0, each ends where the next begins, the last ends at the end.
        # ppo.TrainState.face_counts pins the property ("the chunks
        # concatenate to exactly the step delta") and the loss depends on it;
        # nothing checked it. (The chunks cover the tiling MINUS the last
        # face's approximation tail, which no chunk contains -- that tail is
        # the only slack, and it is exactly what `_face_replay` documents.)
        _prev = 0
        for _s, _sp, _e in segs:
            if _s != _prev or not (_s <= _sp <= _e):
                self.stats["face_seg_not_tiled"] += 1
                raise RuntimeError(
                    f"vertex {vertex}: face segments {segs} do not tile the "
                    f"{len(toks)}-token emission -- the per-face chunks no "
                    f"longer concatenate to the step delta, so the loss's "
                    f"cumsum boundaries pool the wrong rows.")
            _prev = _e
        if _prev != len(toks):
            self.stats["face_seg_not_tiled"] += 1
            raise RuntimeError(
                f"vertex {vertex}: face segments end at {_prev} but the "
                f"emission is {len(toks)} tokens -- the per-face chunks no "
                f"longer concatenate to the step delta.")

        if keys[f] not in ekeys:
            # Decided, then never contracted. Empty chunk (the head falls
            # back to the vertex context for this face alone, as it does
            # for any soft failure), but counted as what it is.
            self.stats["face_dropped"] += 1
            res = (empty[0], empty[1], empty[2], np.int32(n_faces))
            self._chunks[ck] = res
            return res
        gi = ekeys.index(keys[f])

        # INVARIANT: FACE f'S CHUNK OPENS ON FACE f'S OWN `path` HEADER.
        # The mapping is only as good as the record it was read from, so pin
        # the other end at the TOKEN level: segment gi must begin with the
        # `path <central> & <in_edge> & <out_edge>` graphax emits for the face
        # whose key is keys[f]. This is the check that fires if emission order
        # ever changes underneath the sink.
        if exp_hdr is None or toks[segs[gi][0]:segs[gi][0] + len(exp_hdr)] != exp_hdr:
            self.stats["face_header_mismatch"] += 1
            raise RuntimeError(
                f"vertex {vertex} face {f} (key {keys[f]}): segment {gi} does "
                f"not open with that face's `path` header -- got "
                f"{toks[segs[gi][0]:segs[gi][0] + 12]}, expected "
                f"{None if exp_hdr is None else exp_hdr[:12]}. The head would "
                f"be reading another face's contraction.")

        chunk: list[int] = []
        cids: list[int] = []
        if gi > 0:
            # face f-1's approximation: `approx <TYPE> <args>` + the equations
            # it produced. Empty when that face ran exact.
            _s, split, end = segs[gi - 1]
            chunk += toks[split:end]
            cids += ids[split:end]
        start, split, _e = segs[gi]
        chunk += toks[start:split]
        cids += ids[start:split]

        cnt = len(chunk)
        self.stats["tok_total"] += cnt
        self.stats["tok_max"] = max(self.stats["tok_max"], cnt)
        self.stats["chunks"] += 1
        if cnt > W:
            # Keep the TAIL: the face being decided is at the end, and it is
            # the part the decision is about. Counted, because a window that
            # silently drops the contraction would read as a healthy run.
            self.stats["truncated"] += 1
            chunk, cids, cnt = chunk[-W:], cids[-W:], W
        tok_a = np.zeros((W,), np.int32)
        ids_a = -np.ones((W,), np.int32)
        tok_a[:cnt] = np.asarray(chunk, np.int32)
        ids_a[:cnt] = np.asarray(cids, np.int32)

        res = (tok_a, ids_a, np.int32(cnt), np.int32(n_faces))
        if len(self._chunks) >= 4096:
            for dk in list(self._chunks)[:1024]:
                self._chunks.pop(dk, None)
        self._chunks[ck] = res
        return res

    def n_faces(self, order, specs, n, vertex, face_rows_hist=None,
                face_skips_hist=None):
        """Face count of ``vertex`` on the live prefix graph -- the rollout
        while_loop's trip count. Raises if it ever exceeds ``max_faces``:
        the width is the provable ancestors-x-descendants bound, so an
        excess means the bound argument is violated and a silent clamp
        would shrink the action space behind a healthy-looking run."""
        try:
            tk = self._tokenizer_at(
                np.asarray(order).reshape(-1), np.asarray(specs), int(n),
                face_rows_hist, face_skips_hist)
            k = len(list(tk.ij.faces(int(vertex))))
        except Exception:
            self.stats["failures"] += 1
            return 0
        if k > self.max_faces:
            raise RuntimeError(
                f"vertex {vertex}: {k} faces exceed the derived bound "
                f"{self.max_faces}")
        return int(k)

    def consume_stats(self) -> dict:
        out = dict(self.stats)
        for k in self.stats:
            self.stats[k] = 0
        return out
