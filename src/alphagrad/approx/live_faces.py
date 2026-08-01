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
import numpy as np


# Sized from the MEASURED distribution on the MNIST xent graph, not guessed:
# 192 chunks, mean 403 tokens, max 1456. At 96 the window clipped 83 of 206
# chunks -- i.e. 40% of the time the head read only the TAIL of the
# contraction it was approximating, which is the same blindness this module
# removes, just quieter. 1024 clips 8 of 192 (4%). Raise it if
# `truncated` in the health line is a large fraction of `chunks`.
FACE_TOKEN_WINDOW = int(os.environ.get("ALPHAGRAD_FACE_TOKEN_WINDOW", "1024"))


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


class LiveFaceStream:
    """Per-face token chunks for one graph, cached across env steps."""

    def __init__(self, jaxpr, argnums, consts, args, *, vocab: int,
                 max_faces: int = 8, max_axes: int = 8,
                 window: int = FACE_TOKEN_WINDOW, cache: int = 64):
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
                      "chunk_hit": 0, "failures": 0, "truncated": 0,
                      "tok_total": 0, "tok_max": 0, "chunks": 0}

    # -- prefix ------------------------------------------------------------
    def _tokenizer_at(self, order, specs, n):
        from graphax import IncrementalPathTokenizer
        from alphagrad.approx.env import decode_vertex_rule_specs
        from alphagrad.approx.common.masks import make_live_masked_hook

        key = (order[:n].tobytes(), specs[:n].tobytes())
        hit = self._prefix.get(key)
        if hit is not None:
            self.stats["prefix_hit"] += 1
            return hit
        self.stats["prefix_miss"] += 1
        tk = IncrementalPathTokenizer(
            self.jaxpr, self.argnums, list(self.consts), list(self.args),
            vocab_size=self.vocab)
        tk.base_tokens()
        for k in range(n):
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
            tk.eliminate(v, hooks)
        if len(self._prefix) >= self.cache_cap:
            for dk in list(self._prefix)[: max(1, self.cache_cap // 4)]:
                self._prefix.pop(dk, None)
        self._prefix[key] = tk
        return tk

    # -- decoded per-face transforms for the DECIDED faces -----------------
    def _decided(self, tk, vertex, face_rows, face_skips, upto):
        """``{face_key: slots|SKIP_FACE}`` for faces ``0..upto-1``."""
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
                        self.jaxpr, int(vertex), row, is_last=True)
                except Exception:
                    rules = ()
                slots.append(make_live_masked_hook(tuple(rules))
                             if rules else None)
            if any(sl is not None for sl in slots):
                ft[keys[f]] = tuple(slots)
        return keys, ft

    # -- the chunk ---------------------------------------------------------
    def chunk(self, order, specs, n, vertex, vertex_specs,
              face_rows, face_skips, f):
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

        ck = (order[:n].tobytes(), specs[:n].tobytes(), vertex,
              vspecs.tobytes(), rows[:f].tobytes(), skips[:f].tobytes(), f)
        hit = self._chunks.get(ck)
        if hit is not None:
            self.stats["chunk_hit"] += 1
            return hit

        try:
            tk = self._tokenizer_at(order, specs, n)
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
            except Exception:
                self.stats["failures"] += 1
                return empty

        # SKIPPED faces emit nothing, so segment index != face index. Faces
        # 0..f-1 are the only ones that can be skipped (face f's decision has
        # not been made yet), so the shift is exactly how many of them were.
        skipped_before = int(np.sum(skips[:f] == 1))
        gi = f - skipped_before
        if gi < 0 or gi >= len(segs):
            self.stats["failures"] += 1
            return empty

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

    def n_faces(self, order, specs, n, vertex):
        """Face count of ``vertex`` on the live prefix graph -- the rollout
        while_loop's trip count. Raises if it ever exceeds ``max_faces``:
        the width is the provable ancestors-x-descendants bound, so an
        excess means the bound argument is violated and a silent clamp
        would shrink the action space behind a healthy-looking run."""
        try:
            tk = self._tokenizer_at(
                np.asarray(order).reshape(-1), np.asarray(specs), int(n))
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
