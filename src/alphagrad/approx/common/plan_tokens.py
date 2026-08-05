"""Per-decision token deltas for ONE elimination plan, on the path tokenizer.

PPO's env streams observations from ``graphax.IncrementalPathTokenizer``: the
base function once (``env.base_observation()``), then the block each
``eliminate`` emits as that step's DELTA (``env._delta_observation``). AZ needs
exactly that, and additionally has to BRANCH -- a speculative expansion inside
the search must not advance the committed plan.

graphax's tokenizer is neither resumable nor cheaply clonable (the persistent
trace is the whole point of the incremental builder), so branching is done with
:class:`alphagrad.approx.live_faces._Snapshot`, which saves exactly what one
``eliminate`` mutates. ONE snapshot covers a whole speculative CHAIN: its
``__exit__`` truncates the append-only lists back to their entry length and
restores the graph objects and the name maps, so any number of eliminations
inside are undone together. Measured on nn256 (VmappedNeuralNetwork, mnist):

    depth   __enter__   eliminate   __exit__
    0        0.016 ms     0.27 ms    0.016 ms
    8        0.016 ms    17.47 ms    0.153 ms
    16       0.016 ms     5.45 ms    0.124 ms
    23       0.020 ms     6.02 ms    0.174 ms

i.e. the snapshot is 1-3% of the elimination it protects, and a chain pays it
once rather than per step.

THE PARALLEL GRAPH MODEL IS GONE. ``tk.ij`` builds its graph with the same
``_build_graph``/``_prune_graph`` and advances it with the same
``_eliminate_vertex`` a hand-rolled search model would, so
:meth:`PlanTokenizer.legal` is the authoritative legal set. Keeping a second
one is the two-MDPs hazard: the search plans on one graph and the tokens
describe another.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np

__all__ = ["PlanTokenizer"]


class PlanTokenizer:
    """A single live tokenizer plus speculative branching over it.

    ``vertex_specs`` / ``face_rows`` / ``face_skips`` are the SAME wire arrays
    the env measurement consumes, decoded through the SAME
    ``decode_vertex_rule_specs`` + ``make_live_masked_hook`` pair -- so the
    tokens describe the graph the measurement will build, not the intent.
    """

    def __init__(self, jaxpr, argnums, consts, args, *, vocab=None,
                 max_faces=None):
        from graphax import IncrementalPathTokenizer

        if vocab is None:
            vocab = int(os.environ.get("ALPHAGRAD_INCR_TOKEN_VOCAB", "512"))
        self.jaxpr = jaxpr
        self.argnums = tuple(int(a) for a in argnums)
        self.consts = list(consts)
        self.args = list(args)
        self.vocab = int(vocab)
        self.max_faces = max_faces
        self.tk = IncrementalPathTokenizer(
            jaxpr, self.argnums, list(consts), list(args), vocab_size=self.vocab)
        self._base = None
        self.stats = {"eliminations": 0, "branches": 0, "failures": 0}

    def reset(self):
        """Back to the empty prefix, for a new episode.

        A fresh ``IncrementalPathTokenizer`` rather than an unwind: episodes
        are independent and the base block is re-emitted identically (it is a
        constant of the jaxpr), so there is nothing to preserve and a stale
        name pool is one fewer thing that can silently drift.
        """
        from graphax import IncrementalPathTokenizer

        self.tk = IncrementalPathTokenizer(
            self.jaxpr, self.argnums, list(self.consts), list(self.args),
            vocab_size=self.vocab)
        toks = [int(t) for t in self.tk.base_tokens()]
        ids = [int(g) for g in self.tk.last_eqn_ids()]
        if self._base is not None:
            assert toks == self._base[0] and ids == self._base[1], (
                "base stream is not a constant of the jaxpr -- the carry "
                "bootstrap and every episode's stream would disagree")
        else:
            self._base = (toks, ids)
        return self._base

    # -- base --------------------------------------------------------------
    def base(self):
        """``(tokens, eqn_ids)`` of the base function; emitted exactly once."""
        if self._base is None:
            toks = [int(t) for t in self.tk.base_tokens()]
            ids = [int(g) for g in self.tk.last_eqn_ids()]
            guard = os.environ.get("ALPHAGRAD_VOCAB_SIZE")
            if guard is not None and self.tk.max_token_id() >= int(guard):
                raise ValueError(
                    f"incremental token ids reach {self.tk.max_token_id()} but "
                    f"the policy embedding has only {guard} rows -- raise "
                    f"--vocab-size or lower ALPHAGRAD_INCR_TOKEN_VOCAB. (JAX "
                    f"CLAMPS an out-of-range gather, silently reading the "
                    f"wrong row.)")
            self._base = (toks, ids)
        return self._base

    # -- branching ---------------------------------------------------------
    @contextlib.contextmanager
    def branch(self):
        """Everything eliminated inside is undone on exit.

        Nestable (verified: an inner chain repeated after its own unwind emits
        byte-identical tokens, and so does the outer one), but one level per
        speculative chain is enough -- ``__exit__`` truncates rather than pops.
        """
        from alphagrad.approx.live_faces import _Snapshot

        self.stats["branches"] += 1
        with _Snapshot(self.tk):
            yield self

    # -- the delta ---------------------------------------------------------
    def _hooks(self, vertex, vertex_specs, is_last):
        from alphagrad.approx.common.masks import make_live_masked_hook
        from alphagrad.approx.env import decode_vertex_rule_specs

        if vertex_specs is None:
            return ()
        rows = np.asarray(vertex_specs, np.int32).reshape(-1, 3).tolist()
        if all(int(r[0]) == -1 for r in rows):
            return ()
        try:
            rules = decode_vertex_rule_specs(
                self.jaxpr, int(vertex), rows, is_last=bool(is_last))
        except Exception:
            rules = ()
        return (make_live_masked_hook(tuple(rules)),) if rules else ()

    def face_transforms(self, vertex, face_rows, face_skips, *, is_last=True,
                        keys=None):
        """``{face_key: slots|SKIP_FACE}`` for EVERY face of ``vertex``.

        Same decoder ``live_faces.LiveFaceStream._decided`` uses, run to the
        full face count -- the measurement applies every decided face, not a
        prefix of them.

        ``keys`` OVERRIDES the live enumeration, and passing it is REQUIRED on
        a tokenizer that has been speculated on. Why:

        ``faces_of`` filters an edge out when ``_known_none_edge`` says its
        Jacobian is None, and that predicate is answerable only for a LazyEdge
        whose thunk has ALREADY RUN (an unforced one is listed optimistically).
        A speculative elimination forces LazyEdges IN PLACE, and ``_Snapshot``
        restores the two graph DICT levels but not the memo inside the shared
        LazyEdge objects -- so the enumeration SHRINKS across a branch that is
        otherwise a perfect no-op. Measured on nn256 (mnist), 8 branches of 8
        eliminations at the empty prefix, then committing vertex 27: vertices
        13, 16 and 17 went from 1/2/2 faces to 0/0/0 while the legal set was
        bit-identical. The head had already decided face 0 of vertex 13 off the
        FRESH enumeration, so a shrunk key list either drops its decision or --
        worse -- applies ``face_rows[f]`` to a different face.

        The elimination RESULT is unaffected (a known-None face is skipped
        either way); only the key list the per-face plan is indexed by is. So
        the fix is to index by the enumeration the DECIDING tokenizer used.
        """
        from alphagrad.approx.env import (
            FACE_SLOTS, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs)
        from alphagrad.approx.common.masks import make_live_masked_hook
        from graphax import SKIP_FACE

        if face_rows is None and face_skips is None:
            return None
        if keys is None:
            keys = list(self.tk.ij.faces(int(vertex)))
        else:
            keys = list(keys)
        rows = (None if face_rows is None
                else np.asarray(face_rows, np.int32))
        skips = (None if face_skips is None
                 else np.asarray(face_skips, np.int32).reshape(-1))
        ft: dict = {}
        for f in range(len(keys)):
            if skips is not None and f < skips.shape[0] and int(skips[f]) == 1:
                ft[keys[f]] = SKIP_FACE
                continue
            if rows is None or f >= rows.shape[0]:
                continue
            slots = []
            for s in range(FACE_SLOTS):
                row = [[int(x) for x in rows[f][s]]] + [
                    [-1, -1, 0]] * (MAX_RULES_PER_VERTEX - 1)
                try:
                    r = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row, is_last=bool(is_last))
                except Exception:
                    r = ()
                slots.append(make_live_masked_hook(tuple(r)) if r else None)
            if any(sl is not None for sl in slots):
                ft[keys[f]] = tuple(slots)
        return ft or None

    def eliminate(self, vertex, vertex_specs=None, face_rows=None,
                  face_skips=None, *, is_last=True, face_keys=None):
        """Advance the tokenizer by one vertex; return ``(tokens, eqn_ids)``.

        MUTATES. Wrap in :meth:`branch` for a speculative expansion, call bare
        to commit a decision. Pass ``face_keys`` from the tokenizer the per-face
        plan was DECIDED on -- see :meth:`face_transforms`.
        """
        hooks = self._hooks(vertex, vertex_specs, is_last)
        ft = self.face_transforms(vertex, face_rows, face_skips,
                                  is_last=is_last, keys=face_keys)
        self.stats["eliminations"] += 1
        toks = [int(t) for t in self.tk.eliminate(int(vertex), hooks, ft)]
        ids = [int(g) for g in self.tk.last_eqn_ids()]
        return toks, ids

    # -- the graph ---------------------------------------------------------
    def legal(self, valid_vertices):
        """The still-eliminable subset of ``valid_vertices``.

        Read off ``tk.ij.graph`` -- the tokenizer's OWN graph, advanced by the
        same ``_eliminate_vertex`` a separate search model would run. This is
        the only graph model in the search.
        """
        g = self.tk.ij.graph
        eqns = self.jaxpr.eqns
        return [int(v) for v in valid_vertices if eqns[int(v) - 1].outvars[0] in g]

    def n_faces(self, vertex):
        """Faces of ``vertex`` ON THIS TOKENIZER'S LIVE GRAPH.

        NOT authoritative after speculation -- see :meth:`face_transforms` for
        why the count can shrink across a branch. Use the deciding tokenizer's
        enumeration when the answer has to line up with a per-face plan.
        """
        k = len(list(self.tk.ij.faces(int(vertex))))
        if self.max_faces is not None and k > self.max_faces:
            raise RuntimeError(
                f"vertex {vertex}: {k} faces exceed the derived bound "
                f"{self.max_faces}")
        return int(k)

    def face_central_name(self, vertex, f):
        """The tokenized ``path`` header's central variable for face ``f``.

        Index-mapping assertion helper: face ``f``'s chunk must open with the
        header naming THIS vertex's variable, not a neighbour's.
        """
        keys = list(self.tk.ij.faces(int(vertex)))
        if f >= len(keys):
            return None
        return keys[f]

    def consume_stats(self):
        out = dict(self.stats)
        for k in self.stats:
            self.stats[k] = 0
        return out
