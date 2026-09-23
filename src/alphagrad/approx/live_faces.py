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

ONE ELIMINATION PER VERTEX (dsnn-dfw.119). The faces of one vertex do not
depend on each other (owner ruling 2026-09-23): a face's equations are a
function of its own two operand edges, its own decision and the vertex's
approx flag (``graphax.core._is_approx_cfg``). :meth:`LiveFaceStream.chunk_ex`
therefore eliminates each (prefix, vertex) ONCE with no face decided, and per
face only re-eliminates face ``f-1``, on a graph cut down to that one face, when
its decision is not exact. The chunks are then rendered with the tokenizer's own
emitter on a name state that walks the faces in the order the full run would
emit them, so the variable names are the ones the per-face re-run would draw.
:meth:`LiveFaceStream.chunk_ex_per_face` is the re-eliminating path described
next; the two return identical chunks (``tests/onelim_face_chunk_test.py``).

WHY IT RE-ELIMINATES PER FACE (``chunk_ex_per_face``). graphax exposes no
resumable elimination: the
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

from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES
# THE NARROW ID WIRE. Imported from ``common.token_vocab`` and not from
# ``env``: this module stays JAX-free at import time (the measure actors load
# it), and ``token_vocab`` is numpy + graphax only. ``env`` re-exports the
# same objects, so there is one definition, not two.
from alphagrad.approx.common.token_vocab import (
    DELTA_TOKEN_DTYPE as _TOKEN_DTYPE,
    check_delta_ids as _check_delta_ids,
    incr_token_vocab as _incr_token_vocab,
)


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


# --face-edge-mem: how many central-vertex vidx candidates ride the wire.
# A vertex owns one vidx per OUTVAR of its equation; >1 outvar is rare and
# >8 unobserved. Candidates past the width are dropped (the read then
# resolves fewer lhs/rhs slot candidates -- a zero row, never a wrong one).
EDGE_CVX_WIDTH = 8


def _ends(key, vmap):
    """A face key as the policy's own 2-vector ENDPOINT wire.

    ``graphax.core.faces_of`` keys a face by ``(vidx[in_edge],
    vidx[out_edge])`` where ``vidx`` is ``_stable_var_index``: a position
    over (constvars, invars, then every equation's outvars). That is NOT the
    trainer's vertex numbering, which is the 1-based EQUATION index -- on the
    gate's graph the two differ by the four invars, so reading a key
    straight off the wire would hand the head a different vertex's context
    (and clip anything past V onto the last row). ``vmap`` is the
    stable-index -> 1-based vertex map; anything absent from it is a jaxpr
    INPUT or const, which has no vertex and no context, and is written as 0.
    """
    i, j = key
    return np.asarray([int(vmap.get(i, 0)), int(vmap.get(j, 0))], np.int32)


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

# Rebuild the whole face-wire key chain from the arrays on every call and
# raise on any disagreement. See `LiveFaceStream.hist_key`; off by default
# because it costs exactly what the chain saves.
_FACE_KEY_VERIFY = os.environ.get("ALPHAGRAD_FACE_KEY_VERIFY", "0") == "1"

# ALPHAGRAD_FACE_KEY_CHAIN=0 restores the DENSE prefix key (the whole
# `frh[:n].tobytes()`), which is what every cache here used before the chain.
# An A/B switch, not a fallback: the two answer identically and differ only in
# what they cost.
_FACE_KEY_CHAIN = os.environ.get("ALPHAGRAD_FACE_KEY_CHAIN", "1") == "1"

# ALPHAGRAD_FACE_ONE_ELIM=0 serves every chunk from `chunk_ex_per_face`. An A/B
# switch: the two paths return identical chunks.
_ONE_ELIM = os.environ.get("ALPHAGRAD_FACE_ONE_ELIM", "1") == "1"


class _Irregular(Exception):
    pass


_MISS = object()


class _NameOverlay(dict):
    # The names drawn on top of the prefix tokenizer's own map. The emitter
    # reads its maps only through `get` and item assignment.
    __slots__ = ("base",)

    def get(self, k, d=None):
        v = dict.get(self, k, _MISS)
        if v is _MISS:
            return self.base.get(k, d)
        return v


class _NameCursor:
    # `tk._namegen` positioned at draw `pos`, over one shared list of names.
    __slots__ = ("memo", "gen", "pos")

    def __init__(self, memo, gen, pos):
        self.memo, self.gen, self.pos = memo, gen, pos

    def __iter__(self):
        return self

    def __next__(self):
        m = self.memo
        while len(m) <= self.pos:
            m.append(next(self.gen))
        nm = m[self.pos]
        self.pos += 1
        return nm


class _RenderState:
    __slots__ = ("names", "fns", "pos", "uid")

    def __init__(self, names, fns, pos, uid):
        self.names, self.fns, self.pos, self.uid = names, fns, pos, uid

    def fork(self):
        return _RenderState(_NameOverlay(self.names), _NameOverlay(self.fns),
                            self.pos, self.uid)


class _OneElim:
    # One (prefix, vertex): the faces of its single undecided elimination, the
    # faces re-eliminated alone since, and the render cursor.
    __slots__ = ("tk", "keys", "kidx", "armed0", "order", "pos", "ends",
                 "traces", "cursor")


def _frozen_eqn(tr):
    # The equation `frame.get_eqns()` would build for this entry, for the
    # emitter only. A collected weakref shifts every later index of the
    # per-face path, so that case is left to the per-face path.
    from weakref import ReferenceType
    e = tr() if isinstance(tr, ReferenceType) else tr
    if e is None:
        raise _Irregular("a traced equation of this face was collected")
    if hasattr(e, "in_tracers"):
        return SimpleNamespace(invars=[t.val for t in e.in_tracers],
                               outvars=e.outvars, primitive=e.primitive,
                               params=e.params)
    return e


def _vertex_armed(vhooks, ft) -> bool:
    # `graphax.core._eliminate_vertex`'s `_is_approx_cfg`, from the same inputs.
    from graphax import core as _gc
    return (any(isinstance(t, (_gc.Diag, _gc.Compress)) or callable(t)
                for t in vhooks)
            or bool(_gc.face_config_is_approx(ft)))


def _face_row_key(row, skiprow):
    """One vertex's face wire, as a few dozen bytes instead of 69 kilobytes.

    The wire pads with -1 and the skips with 0, so the positions of the
    entries that are NOT padding, together with their values, describe the
    row completely and injectively for a fixed shape. Measured occupancy on
    the campaign graph is 1.3 faces of 1920, so this is three orders of
    magnitude smaller than the row it stands for.
    """
    r = np.ascontiguousarray(row).reshape(-1)
    s = np.ascontiguousarray(skiprow).reshape(-1)
    ri = np.nonzero(r != -1)[0].astype(np.int32)
    si = np.nonzero(s != 0)[0].astype(np.int32)
    return (ri.tobytes(), r[ri].astype(np.int32).tobytes(),
            si.tobytes(), s[si].astype(np.int32).tobytes())


def _hist_key_parts(frh, fsh, n):
    """The DENSE fallback key for the face wires of prefix ``[0, n)``.

    What every cache in this module keyed on before the chain existed, and
    what a caller that keeps no per-environment chain still gets. Correct and
    slow: `frh[:n]` is 6.5 megabytes on the campaign graph.
    """
    return (b"" if frh is None else frh[:n].tobytes(),
            b"" if fsh is None else fsh[:n].tobytes())

# The three operand slots of one face, in the head's slot order
# (``env.FACE_SLOTS``: pre = lhs, post = rhs, new). ``face_slot_legality``
# records and reports them in this order.
_SLOT_SITES = ("lhs", "rhs", "new")

class DecidedFaces(NamedTuple):
    """What one vertex's dynamic decision pass produced.

    ``rows`` is the wire -- ``(F, S, 3)`` int32, ``-1`` where no approximation
    was chosen -- and the five mask fields are the legality each row was drawn
    under, per face and per slot, read off the live tensor AT THAT SLOT'S SITE
    after every earlier decision had landed. The masks are part of the result
    rather than a by-product because PPO's loss replay rescores from the STORED
    mask and never recomputes it, so these are the arrays that have to reach
    the trajectory for ``sample`` and ``evaluate`` to score the same variable.

    ``sizes`` / ``nout`` are the slot's own frame, which is the frame the wire
    row is written in (``bi2 = j - n_out``) -- the same contract
    ``masks.SlotLegality`` states per tensor.
    """
    rows: np.ndarray      # (F, S, 3) int32
    sizes: np.ndarray     # (F, S, N) int32
    quant: np.ndarray     # (F, S, K) float32, K = len(masks.FACE_QUANT_DTYPES)
    pair: np.ndarray      # (F, S, N, N) float32
    comp: np.ndarray      # (F, S, N) float32
    nout: np.ndarray      # (F, S) int32
    n_faces: np.int32


def _wire_slots() -> int:
    from alphagrad.approx.env import wire_slots
    return wire_slots()


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
        # THE one resolver (`common.token_vocab`): checked here, at
        # construction, so a vocabulary the tokenizer cannot fit raises from
        # the CALLER rather than from `_tokenizer_at`, whose
        # `except Exception` soft-failure path would turn it into an empty
        # chunk and a bumped `failures` counter.
        self.vocab = _incr_token_vocab(vocab)
        self.max_faces = int(max_faces)
        self.max_axes = int(max_axes)
        self.window = int(window)
        self.cache_cap = int(cache)
        self._prefix: dict = {}       # prefix key -> tokenizer at that prefix
        self._chunks: dict = {}       # full key -> result tuple
        # THE PER-ENV FACE-WIRE KEY CHAIN (see `hist_key`).
        self._histkeys: dict = {}     # env index -> (n, key tuple)
        # tok_total/tok_max/chunks size the WINDOW from the real
        # distribution: a window below the typical chunk silently keeps only
        # the tail of the contraction the head is meant to read.
        self.stats = {"prefix_miss": 0, "prefix_hit": 0, "elims": 0,
                      # `prefix_ext`: misses served by extending the n-1
                      # tokenizer by ONE vertex instead of replaying the
                      # whole prefix (see `_tokenizer_at`).
                      "prefix_ext": 0,
                      # The face-wire key chain (see `hist_key`): `ext` is a
                      # key built by adding one row to the previous step's,
                      # `cold` a key rebuilt from the whole history. A steady
                      # state of one cold per episode per env is the episode
                      # boundary; more than that means something is asking
                      # for prefixes out of order and the saving is gone.
                      "hist_key_ext": 0, "hist_key_cold": 0,
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
                      "face_header_mismatch": 0,
                      # --per-face-masks SIZES half (`face_dim_sizes`).
                      # `size_probe` counts the extra eliminations it
                      # runs (2 per (prefix, vertex), cached across the
                      # whole face loop); `size_miss` counts ENUMERATED
                      # faces the probe never visited, which come back as
                      # a zero size vector (= no approximation offered).
                      "size_probe": 0, "size_hit": 0,
                      "size_probe_fail": 0, "size_miss": 0,
                      # --face-slot-frames (`face_slot_legality`): the same
                      # four, for the per-SLOT probe.
                      "slot_probe": 0, "slot_hit": 0,
                      "slot_probe_fail": 0, "slot_miss": 0,
                      # `decide_faces` (.59 fault 2): `decide_probe` counts
                      # the ONE extra elimination per vertex, `decide_draw`
                      # the decisions taken inside it, `decide_self_skip`
                      # rows the apply path refused ON THE TENSOR IT WAS
                      # DRAWN FROM (must be 0 -- that is the whole claim),
                      # `decide_multi_site` invocations of a slot's chooser
                      # past the first (0 under every --approx-add value,
                      # non-zero only if a future value puts one slot's hook
                      # at two sites again).
                      "decide_probe": 0, "decide_draw": 0,
                      "decide_probe_fail": 0, "decide_self_skip": 0,
                      "decide_multi_site": 0,
                      # `decide_vertex_faces` (#77): the EXACT per-vertex pass,
                      # NO elimination. `vertex_probe` counts the passes,
                      # `vertex_operand_probes` the n + m forced operand edges
                      # and `vertex_contractions` the n*m structural
                      # compositions -- the three terms of the proof, so the
                      # "no speculative elimination" claim is a counter rather
                      # than a comment. `vertex_face_absent` is an enumerated
                      # face whose edge Jacobian forces to None, which graphax
                      # skips too (an all-zero mask row, i.e. nothing offered).
                      # `vertex_self_skip` must be 0 -- a row the apply hook
                      # refuses on the very tensor it was drawn from is a defect
                      # in `slot_legality`, not staleness.
                      # `vertex_key_collision` is the multi-output face-key
                      # collision (measured 0 on TLM and nn256; it RAISES);
                      # `vertex_flag_undecided` / `vertex_flag_flip` are the one
                      # case graphax's `_is_approx_cfg` is not a function of
                      # stage 1 alone (a `lossy` join armed only by slot 2).
                      "vertex_probe": 0, "vertex_probe_fail": 0,
                      "vertex_operand_probes": 0, "vertex_contractions": 0,
                      "vertex_draw": 0, "vertex_face_absent": 0,
                      "vertex_self_skip": 0, "vertex_key_collision": 0,
                      "vertex_flag_undecided": 0, "vertex_flag_flip": 0,
                      # `--approx-add choose`: the entry builder needs a
                      # per-face join BIT this pass does not hold, so the
                      # approx flag is ASSUMED True (the conservative arm).
                      "vertex_flag_assumed": 0,
                      # `chunk_ex`: `elims` counts the one full elimination
                      # per (prefix, vertex); `face_elims` the one-face
                      # eliminations of a decided face; `onelim_fallback` the
                      # chunks served by `chunk_ex_per_face` instead.
                      "face_elims": 0, "face_renders": 0,
                      "onelim_fallback": 0}
        self.one_elim = _ONE_ELIM
        self.last_onelim_error: str | None = None
        self._onelim: dict = {}       # (prefix, vertex) -> _OneElim
        self._namememo: dict = {}     # name alphabet -> (names, generator)
        # The last exception `decide_vertex_faces` swallowed, as text. See the
        # `except` there: a failure COUNT is not a diagnosis.
        self.last_vertex_error: str | None = None
        self._sizes: dict = {}        # (prefix, vertex) -> (sizes, quant, n)
        self._slots: dict = {}        # (prefix, vertex) -> per-slot legality

    # -- prefix ------------------------------------------------------------
    @staticmethod
    def _hist(face_rows_hist, face_skips_hist):
        """``(rows, skips)`` as int32 arrays, or ``(None, None)``."""
        if face_rows_hist is None or face_skips_hist is None:
            return None, None
        return (np.asarray(face_rows_hist, np.int32),
                np.asarray(face_skips_hist, np.int32))

    def hist_key(self, env: int, frh, fsh, n: int):
        """THE FACE-WIRE HISTORY OF PREFIX ``[0, n)``, AS A FEW HUNDRED BYTES.

        Every cache in this class used to key on ``frh[:n].tobytes()``. On the
        campaign graph that slice is `n x MAX_FACES x FACE_SLOTS x 3` int32 --
        6.5 megabytes at step 94 -- and measured on pgi14 it costs 1.1
        milliseconds to materialise and hash. The rollout builds that key
        about five times per environment per step (the face count, the chunks,
        the sizes, the slot legality and the vertex decision), so at sixteen
        environments it was about 93 milliseconds of the 246 milliseconds of
        host Python the rollout profile of 2026-09-15 attributes to the face
        path: roughly nine seconds per episode spent hashing padding.

        The wire pads with -1 (and the skips with 0), so ``(positions of the
        entries that are not padding, their values)`` is a COMPLETE and
        INJECTIVE description of a row for a fixed shape -- the same argument
        `env._face_wire_keys` already makes for the tokenizer's own cache. The
        measured occupancy is 1.3 faces of 1920, so the compact row key is a
        few dozen bytes and costs 3.5 microseconds.

        AND IT IS BUILT ONCE PER STEP, NOT ONCE PER CALL. The key for prefix n
        is the key for prefix n-1 with one more row on the end, so the chain
        per environment is extended by one row per step. Three structural
        facts make that sound, and the code checks the first two rather than
        assuming them:

        1. ``n`` increases by exactly one per rollout step. A call at any
           other ``n`` takes the cold path below and rebuilds the whole chain
           from the arrays in hand.
        2. A REPEAT of an episode (a bin overflow) resets ``step_count`` to 0,
           which is `n != cur + 1` and so is a cold rebuild.
        3. Rows below ``step_count`` never move: ``env.step`` shift-and-inserts
           at ``idx = step_count`` only. That is the same fact the pop-extend
           in :meth:`_tokenizer_at` already rests on, stated in its comment.

        ``ALPHAGRAD_FACE_KEY_VERIFY=1`` rebuilds the whole chain from the
        arrays on every call and raises on any disagreement. It is off by
        default because it costs exactly what the chain saves.
        """
        if frh is None or fsh is None or not _FACE_KEY_CHAIN:
            return None
        n = int(n)
        cur = self._histkeys.get(int(env))
        key = None
        if n <= 0:
            key = ()
        else:
            # THE LAST ROW IS ALWAYS RECOMPUTED, never taken on trust. It is
            # the row this step decided, so it is the one that changes when a
            # discarded attempt is repeated at the same prefix length. The
            # rows below it are taken from the chain, on fact 3 above.
            last = _face_row_key(frh[n - 1], fsh[n - 1])
            if cur is not None and len(cur[1]) == n and cur[1][n - 1] == last:
                key = cur[1]
            elif cur is not None and cur[0] == n - 1 and len(cur[1]) == n - 1:
                key = cur[1] + (last,)
                self.stats["hist_key_ext"] += 1
        if key is None:
            key = tuple(_face_row_key(frh[j], fsh[j]) for j in range(n))
            self.stats["hist_key_cold"] += 1
        if _FACE_KEY_VERIFY:
            want = tuple(_face_row_key(frh[j], fsh[j]) for j in range(n))
            if want != key:
                raise RuntimeError(
                    f"face-wire key chain desynchronised for env {env} at "
                    f"prefix {n}: the chain says {len(key)} rows and the "
                    f"history in hand says {len(want)}. The chain assumes "
                    f"step_count rises by one per step and that rows below "
                    f"it never move (live_faces.hist_key).")
        self._histkeys[int(env)] = (n, key)
        return key

    def _tokenizer_at(self, order, specs, n, face_rows_hist=None,
                      face_skips_hist=None, *, hist_key=None):
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
        #
        # `hist_key` is the COMPACT form of those wires (see `hist_key`); the
        # dense `tobytes()` slices below are the fallback for a caller that
        # does not keep a per-environment chain -- every test and probe that
        # calls this directly. The two are different key spaces, so mixing
        # them costs a cache miss and can never produce a false hit.
        _hk = (_hist_key_parts(frh, fsh, n) if hist_key is None
               else (hist_key,))
        key = (order[:n].tobytes(), specs[:n].tobytes()) + _hk
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
                rules = decode_vertex_rule_specs(self.jaxpr, v, specs[k])
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
            # diverged. The decode is position-independent now, so replaying a
            # prefix vertex here and deciding it fresh give the same rules.
            ft = None
            if frh is not None:
                try:
                    _keys, ft = self._decided(
                        tk, v, frh[k], fsh[k], int(frh.shape[1]))
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
            _phk = (_hist_key_parts(frh, fsh, n - 1) if hist_key is None
                    else (hist_key[:n - 1],))
            pkey = (order[:n - 1].tobytes(),
                    specs[:n - 1].tobytes()) + _phk
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

    def _vertex_of(self, tk):
        """``{stable var index: 1-based vertex}`` for ``tk``'s jaxpr.

        Built from the SAME ``_stable_var_index`` graphax keys faces with, so
        the two cannot drift; memoized on the jaxpr object because the prefix
        tokenizers all share one.
        """
        from graphax.core import _vidx_for
        jx = tk.ij.jaxpr
        cached = getattr(self, "_vmap_memo", None)
        if cached is not None and cached[0] is jx:
            return cached[1]
        vidx = _vidx_for(jx)
        m = {}
        for k, eqn in enumerate(jx.eqns):
            for ov in eqn.outvars:
                if ov in vidx:
                    m[vidx[ov]] = k + 1
        self._vmap_memo = (jx, m)
        return m

    def _central_vidx(self, tk, vertex):
        """``vertex``'s own stable var indices, ``(EDGE_CVX_WIDTH,)`` int32.

        The inverse read of :meth:`_vertex_of` (one vidx per outvar of the
        vertex's equation), padded with -1. These are the CENTRAL halves of
        the face's operand edge keys (--face-edge-mem): lhs edge
        ``(i, vidx(v))``, rhs edge ``(vidx(v), j)`` -- section 8's edge
        identity, in exactly the key space ``faces_of`` keys faces with.
        """
        out = -np.ones((EDGE_CVX_WIDTH,), np.int32)
        k = 0
        for vx, vv in self._vertex_of(tk).items():
            if vv == vertex and k < EDGE_CVX_WIDTH:
                out[k] = int(vx)
                k += 1
        return out

    # -- decoded per-face transforms for the DECIDED faces -----------------
    def _decided(self, tk, vertex, face_rows, face_skips, upto):
        """``{face_key: slots|SKIP_FACE}`` for faces ``0..upto-1``.

        Built by ``env._face_dict_for_vertex`` -- THE builder the measurement
        uses -- so the prefix this stream replays carries exactly the
        transforms the measured graph carries, each row decoded in its slot
        tensor's frame at apply time. This used to decode the rows in the
        vertex frame (``decode_vertex_rule_specs``); the stream's operands
        drifted from the measured ones and the decide-time masks cleared rows
        that were no-ops on the real operand (see the builder's docstring).
        """
        from types import SimpleNamespace
        from alphagrad.approx.env import _face_dict_for_vertex

        keys = list(tk.ij.faces(int(vertex)))
        # THE NARROWED WIRE MUST NOT DROP A DECISION SILENTLY. The builder
        # BREAKS out of its loop when the row runs out (`f >= face_row.shape[0]`),
        # which is right for a caller that genuinely has fewer rows and wrong
        # for `--face-wire-faces`: a vertex with more faces than the wire
        # carries would run its tail exactly while every counter reported a
        # healthy run. That is the fault class MAX_FACES exists for, so it
        # raises here instead.
        _w = int(np.asarray(face_rows).shape[0])
        if len(keys) > _w:
            raise RuntimeError(
                f"vertex {vertex} has {len(keys)} faces and the face wire "
                f"carries {_w} columns; --face-wire-faces is too narrow and "
                f"the faces past {_w} would run exact in silence.")
        ft = _face_dict_for_vertex(
            SimpleNamespace(jaxpr=self.jaxpr), tk.ij, int(vertex),
            face_rows, face_skips, keys=keys, upto=int(upto))
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
              face_rows_hist=None, face_skips_hist=None, *, hist_key=None):
        """``(tokens, count, n_faces, ends, head)``.

        :meth:`chunk_ex`'s first four plus ``head`` -- the approx-echo PREFIX
        length of this chunk, i.e. how many of its leading tokens belong to
        face ``f-1``'s APPROXIMATION rather than to face ``f``'s own
        contraction (``docs/FACE_READ_POINT_TRACE.md`` (E)(i)). It is
        returned UNCONDITIONALLY, not only under --face-edge-mem, because
        ``--face-read own-span-mean`` / ``last-row`` need it to mask the
        head's pooling to face ``f``'s own span. Cost: zero -- it is already
        computed (and already truncation-corrected) inside ``chunk_ex``.
        """
        r = self.chunk_ex(order, specs, n, vertex, vertex_specs,
                          face_rows, face_skips, f,
                          face_rows_hist, face_skips_hist, hist_key=hist_key)
        return r[:4] + (r[6],)

    def chunk_ex(self, order, specs, n, vertex, vertex_specs,
                 face_rows, face_skips, f,
                 face_rows_hist=None, face_skips_hist=None, *,
                 hist_key=None):
        """``(tokens (W,), count, n_faces, ends (2,),
        ekey (2,), cvx (EDGE_CVX_WIDTH,), head, wrok)``.

        ONE token buffer: the parallel equation-id buffer was removed on
        2026-09-13 with the ids themselves (they fed only the palimpsa
        relational forget gate, which went too).

        The last four are the --face-edge-mem wires (section 8), all cheap
        reads of state the enumeration already computed: ``ekey`` is the
        face's RAW key ``(i, j)`` in stable-var-index space -- the RES edge
        this face creates/updates; ``cvx`` the central vertex's own vidx
        candidates (so the operand edge keys are ``(i, cvx)`` / ``(cvx,
        j)``); ``head`` the length of the approx-echo PREFIX of this chunk
        (face f-1's approximation equations), which is what lets the write
        path recover face f's TRUE emission span [start_f, end_f) --
        ``last_face_segments``'s tiling -- from the stored chunk counts:
        ``start_f = cumsum(counts)[f-1] + head_f``; ``wrok`` is 1 iff this
        face was actually emitted (its res edge WILL be written this step;
        a dropped face must not claim a slot). All four are (-1/-1s/0/0) on
        every soft-failure path.

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
        if not self.one_elim:
            return self.chunk_ex_per_face(
                order, specs, n, vertex, vertex_specs, face_rows, face_skips,
                f, face_rows_hist, face_skips_hist, hist_key=hist_key)
        W = self.window
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex, f = int(n), int(vertex), int(f)
        rows = np.asarray(face_rows, np.int32)
        skips = np.asarray(face_skips, np.int32)
        vspecs = np.asarray(vertex_specs, np.int32)
        _no_edge = (-np.ones((2,), np.int32),
                    -np.ones((EDGE_CVX_WIDTH,), np.int32),
                    np.int32(0), np.int32(0))
        empty = (np.zeros((W,), _TOKEN_DTYPE),
                 np.int32(0), np.int32(0), np.zeros((2,), np.int32)
                 ) + _no_edge

        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        ck = ((order[:n].tobytes(), specs[:n].tobytes(), vertex,
               vspecs.tobytes(), rows[:f].tobytes(), skips[:f].tobytes(), f)
              + (_hist_key_parts(frh, fsh, n) if hist_key is None
                 else (hist_key,)))
        hit = self._chunks.get(ck)
        if hit is not None:
            self.stats["chunk_hit"] += 1
            return hit

        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
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
            res = (empty[0], empty[1], np.int32(n_faces),
                   empty[3]) + _no_edge
            self._chunks[ck] = res
            return res

        from alphagrad.approx.env import decode_vertex_rule_specs
        from alphagrad.approx.common.masks import make_live_masked_hook
        try:
            vrules = decode_vertex_rule_specs(
                self.jaxpr, vertex, vspecs.tolist())
        except Exception:
            vrules = ()
        vhooks = (make_live_masked_hook(tuple(vrules)),) if vrules else ()

        try:
            ekeys, tail, contr = self._one_elim_parts(
                tk, ck[:4] + ck[7:], vertex, vhooks, keys, ft, rows, skips, f)
        except Exception as exc:
            # A graph the one-face cut cannot serve exactly (a collected
            # equation, a repeated face key, a trace failure) is served by the
            # per-face path, which returns what it always returned.
            self.stats["onelim_fallback"] += 1
            self.last_onelim_error = f"{type(exc).__name__}: {exc}"
            return self.chunk_ex_per_face(
                order, specs, n, vertex, vertex_specs, face_rows, face_skips,
                f, face_rows_hist, face_skips_hist, hist_key=hist_key)

        if ekeys != keys:
            self.stats["face_key_seg_mismatch"] += 1
            if ekeys and not _PARTIAL_DROP_WARNED[0]:
                _PARTIAL_DROP_WARNED[0] = True
                warnings.warn(
                    f"[alphagrad.approx.live_faces] vertex {vertex}: faces_of "
                    f"enumerated {keys} but the elimination emitted {ekeys} -- "
                    f"a PARTIAL drop. Chunks are now mapped by face key, so "
                    f"the head still reads its own face; positional indexing "
                    f"would have handed it a later face's contraction. See "
                    f"`face_key_seg_mismatch` in the face-stream health line.",
                    stacklevel=2)

        _k2i = lambda x: np.int32(int(x) if x is not None else -1)  # noqa: E731
        _ekey = np.asarray([_k2i(keys[f][0]), _k2i(keys[f][1])], np.int32)
        _cvx = self._central_vidx(tk, vertex)
        if contr is None:
            self.stats["face_dropped"] += 1
            res = (empty[0], empty[1], np.int32(n_faces),
                   _ends(keys[f], self._vertex_of(tk)),
                   _ekey, _cvx, np.int32(0), np.int32(0))
            self._chunks[ck] = res
            return res

        chunk: list[int] = list(tail)
        head = len(chunk)
        chunk += contr

        cnt = len(chunk)
        self.stats["tok_total"] += cnt
        self.stats["tok_max"] = max(self.stats["tok_max"], cnt)
        self.stats["chunks"] += 1
        if cnt > W:
            self.stats["truncated"] += 1
            head = max(0, head - (cnt - W))
            chunk, cnt = chunk[-W:], W
        tok_a = np.zeros((W,), _TOKEN_DTYPE)
        if cnt:
            _tk64 = np.asarray(chunk, np.int64)
            _check_delta_ids(_tk64, where="LiveFaceStream.chunk_ex")
            tok_a[:cnt] = _tk64.astype(_TOKEN_DTYPE)

        res = (tok_a, np.int32(cnt), np.int32(n_faces),
               _ends(keys[f], self._vertex_of(tk)),
               _ekey, _cvx, np.int32(head), np.int32(1))
        if len(self._chunks) >= 4096:
            for dk in list(self._chunks)[:1024]:
                self._chunks.pop(dk, None)
        self._chunks[ck] = res
        return res

    # The per-face path: one full elimination of the vertex per face, with faces
    # 0..f-1 decided. `chunk_ex` returns the same chunks from one elimination.
    def chunk_ex_per_face(self, order, specs, n, vertex, vertex_specs,
                          face_rows, face_skips, f,
                          face_rows_hist=None, face_skips_hist=None, *,
                          hist_key=None):
        W = self.window
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex, f = int(n), int(vertex), int(f)
        rows = np.asarray(face_rows, np.int32)
        skips = np.asarray(face_skips, np.int32)
        vspecs = np.asarray(vertex_specs, np.int32)
        # 5th slot: the face's ENDPOINT VERTICES, as the head's identity for
        # it. `faces_of` keys are `(vidx[in_edge], vidx[out_edge])` and
        # `vidx.get` is None for a var no equation produces (a jaxpr input),
        # so the wire is 1-BASED with 0 = "no vertex" -- the device side
        # gathers a zero context for 0 rather than an arbitrary row.
        _no_edge = (-np.ones((2,), np.int32),
                    -np.ones((EDGE_CVX_WIDTH,), np.int32),
                    np.int32(0), np.int32(0))
        # NARROW WIRE: the chunk buffer carries the SAME tokens as the step
        # delta (a chunk is a slice of the same emission), so it carries the
        # same dtype -- uint8. See env.py's "THE NARROW TOKEN WIRE" note.
        # Layout: (tokens, count, n_faces, ends) + the four edge wires.
        empty = (np.zeros((W,), _TOKEN_DTYPE),
                 np.int32(0), np.int32(0), np.zeros((2,), np.int32)
                 ) + _no_edge

        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        ck = ((order[:n].tobytes(), specs[:n].tobytes(), vertex,
               vspecs.tobytes(), rows[:f].tobytes(), skips[:f].tobytes(), f)
              + (_hist_key_parts(frh, fsh, n) if hist_key is None
                 else (hist_key,)))
        hit = self._chunks.get(ck)
        if hit is not None:
            self.stats["chunk_hit"] += 1
            return hit

        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
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
            res = (empty[0], empty[1], np.int32(n_faces),
                   empty[3]) + _no_edge
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
                self.jaxpr, vertex, vspecs.tolist())
        except Exception:
            vrules = ()
        vhooks = (make_live_masked_hook(tuple(vrules)),) if vrules else ()

        with _Snapshot(tk):
            try:
                self.stats["elims"] += 1
                toks = [int(t) for t in tk.eliminate(vertex, vhooks, ft)]
                # `tk.last_eqn_ids()` is NOT read: the equation-id stream was
                # removed on 2026-09-13. graphax still produces it; alphagrad
                # no longer asks.
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

        _k2i = lambda x: np.int32(int(x) if x is not None else -1)  # noqa: E731
        _ekey = np.asarray([_k2i(keys[f][0]), _k2i(keys[f][1])], np.int32)
        _cvx = self._central_vidx(tk, vertex)
        if keys[f] not in ekeys:
            # Decided, then never contracted. Empty chunk (the head falls
            # back to the vertex context for this face alone, as it does
            # for any soft failure), but counted as what it is. The edge
            # KEY still rides (the face exists and its operand edges are
            # readable); wrok=0 -- its res edge is never emitted, so it
            # must not claim a write slot.
            self.stats["face_dropped"] += 1
            res = (empty[0], empty[1], np.int32(n_faces),
                   _ends(keys[f], self._vertex_of(tk)),
                   _ekey, _cvx, np.int32(0), np.int32(0))
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
        if gi > 0:
            # face f-1's approximation: `approx <TYPE> <args>` + the equations
            # it produced. Empty when that face ran exact.
            _s, split, end = segs[gi - 1]
            chunk += toks[split:end]
        # --face-edge-mem: the approx-echo PREFIX length of this chunk. The
        # write path subtracts it from the cumsum boundary to recover face
        # f's OWN `last_face_segments` span in the true emission.
        head = len(chunk)
        start, split, _e = segs[gi]
        chunk += toks[start:split]

        cnt = len(chunk)
        self.stats["tok_total"] += cnt
        self.stats["tok_max"] = max(self.stats["tok_max"], cnt)
        self.stats["chunks"] += 1
        if cnt > W:
            # Keep the TAIL: the face being decided is at the end, and it is
            # the part the decision is about. Counted, because a window that
            # silently drops the contraction would read as a healthy run.
            self.stats["truncated"] += 1
            head = max(0, head - (cnt - W))
            chunk, cnt = chunk[-W:], W
        tok_a = np.zeros((W,), _TOKEN_DTYPE)
        if cnt:
            _tk64 = np.asarray(chunk, np.int64)
            # SAME raise as the observation path, at the OTHER producer of
            # these tokens. A face chunk is a slice of the step emission.
            _check_delta_ids(_tk64, where="LiveFaceStream.chunk_ex")
            tok_a[:cnt] = _tk64.astype(_TOKEN_DTYPE)

        res = (tok_a, np.int32(cnt), np.int32(n_faces),
               _ends(keys[f], self._vertex_of(tk)),
               _ekey, _cvx, np.int32(head), np.int32(1))
        if len(self._chunks) >= 4096:
            for dk in list(self._chunks)[:1024]:
                self._chunks.pop(dk, None)
        self._chunks[ck] = res
        return res

    # -- one elimination per vertex (dsnn-dfw.119) --------------------------
    # The per-face path emits, for face f, the run R_f = every face of the
    # vertex with faces 0..f-1 decided, and hands out face f-1's approximation
    # tail and face f's contraction from it. A face's equations depend only on
    # its own operand edges, its own decision and the vertex's approx flag, so
    # R_f is rebuilt here face by face: an undecided face comes from the ONE
    # full elimination of the vertex (or, if a decision armed the vertex's
    # approx flag, from its own one-face elimination under that flag), a
    # decided face from its own one-face elimination. The names are drawn by
    # rendering those faces in R_f's order with the tokenizer's own emitter.

    def _one_elim_parts(self, tk, pk, vertex, vhooks, keys, ft, rows, skips,
                        f):
        """``(emitted keys, face f-1's tail, face f's contraction or None)``."""
        st = self._onelim.get(pk)
        if st is None or st.tk is not tk or st.keys != keys:
            st = self._eliminate_once(tk, vertex, vhooks, keys)
            if len(self._onelim) >= self.cache_cap:
                for dk in list(self._onelim)[: max(1, self.cache_cap // 4)]:
                    self._onelim.pop(dk, None)
            self._onelim[pk] = st
        gi = st.pos.get(keys[f])
        if gi is None:
            return st.order, None, None
        armed = _vertex_armed(vhooks, ft)
        sigs = []
        for k in st.order[:gi]:
            j = st.kidx[k]
            sigs.append((rows[j].tobytes(), int(skips[j])) if k in ft
                        else None)
        sigs = tuple(sigs)
        tail = self._advance_cursor(tk, st, vertex, vhooks, ft, armed, sigs)
        rs = st.cursor[2].fork()
        toks, split = self._render(
            tk, rs, self._face_trace(tk, st, vertex, vhooks, ft, keys[f],
                                     None, armed))
        return st.order, tail, toks[:split]

    def _eliminate_once(self, tk, vertex, vhooks, keys):
        st = _OneElim()
        st.tk = tk
        st.keys = list(keys)
        st.kidx = {k: j for j, k in enumerate(keys)}
        if len(st.kidx) != len(keys):
            raise _Irregular(f"vertex {vertex}: repeated face keys {keys}")
        frame = tk.ij.trace.frame
        if getattr(frame, "auto_dce", False):
            from weakref import ReferenceType
            if any(isinstance(r, ReferenceType) and r() is None
                   for r in frame.tracing_eqns):
                raise _Irregular("a prefix equation was collected")
        self.stats["elims"] += 1
        caps = self._elim_capture(tk, vertex, vhooks, {})
        sink = tk.ij.face_sink
        vidx = sink.vidx
        if vidx is None:
            from graphax.core import _vidx_for
            vidx = _vidx_for(tk.ij.jaxpr)
        st.order, st.pos, st.ends, st.traces = [], {}, {}, {}
        st.armed0 = _vertex_armed(vhooks, {})
        for fr, eqns in caps:
            k = (vidx.get(fr.in_edge), vidx.get(fr.out_edge))
            if k in st.pos or k not in st.kidx:
                raise _Irregular(f"vertex {vertex}: emitted face {k} is "
                                 f"repeated or not enumerated")
            if st.order and st.kidx[k] < st.kidx[st.order[-1]]:
                raise _Irregular(f"vertex {vertex}: faces emitted out of "
                                 f"enumeration order")
            st.pos[k] = len(st.order)
            st.order.append(k)
            st.ends[k] = (fr.central, fr.in_edge, fr.out_edge)
            st.traces[(k, None, st.armed0)] = (fr, eqns)
        st.cursor = None
        return st

    def _face_trace(self, tk, st, vertex, vhooks, ft, k, sig, armed):
        key = (k, sig, armed)
        t = st.traces.get(key)
        if t is None:
            self.stats["face_elims"] += 1
            caps = self._elim_capture(tk, vertex, vhooks, ft, only=st.ends[k])
            if len(caps) != 1:
                raise _Irregular(f"vertex {vertex}: face {k} alone emitted "
                                 f"{len(caps)} faces")
            t = caps[0]
            st.traces[key] = t
        return t

    def _advance_cursor(self, tk, st, vertex, vhooks, ft, armed, sigs):
        # The name state after the emitted faces `st.order[:len(sigs)]` of
        # R_f, each rendered in full, and the approximation tail of the last.
        base = (len(tk._names) + len(tk._fns), tk._flatten_uid)
        cur = st.cursor
        if (cur is None or cur[0] != armed or cur[4] != base
                or len(cur[1]) > len(sigs) or sigs[:len(cur[1])] != cur[1]):
            cur = (armed, (), _RenderState(_NameOverlay(), _NameOverlay(),
                                           base[0], base[1]), [], base)
        rs, tail = cur[2], cur[3]
        for i in range(len(cur[1]), len(sigs)):
            toks, split = self._render(
                tk, rs, self._face_trace(tk, st, vertex, vhooks, ft,
                                         st.order[i], sigs[i], armed))
            tail = toks[split:]
        st.cursor = (armed, sigs, rs, tail, base)
        return tail if sigs else []

    def _elim_capture(self, tk, vertex, vhooks, ft, only=None):
        """``[(face record, its equations)]`` of one speculative elimination.

        ``only=(central, in_edge, out_edge)`` cuts the vertex down to that one
        face first. Nothing the elimination touches survives the call.
        """
        ij = tk.ij
        sink = ij.face_sink
        if sink is None:
            raise _Irregular("the tokenizer tracks no faces")
        teq = ij.trace.frame.tracing_eqns
        g0, t0, vo0 = ij.graph, ij.tgraph, ij.vo
        n_eq, n_st = len(teq), len(ij.steps)
        n_fc, n_x = len(sink.faces), len(ij.xlog.records)
        if only is None:
            g, t = _copy_graph(g0), _copy_graph(t0)
        else:
            central, ie, oe = only
            g, t = dict(g0), dict(t0)
            for ov in ij.jaxpr.eqns[int(vertex) - 1].outvars:
                if ov is not central:
                    g.pop(ov, None)
            g[central] = {oe: g0[central][oe]}
            t[central] = {ie: t0[central][ie]}
            # The rows the elimination writes: the new edge in_edge -> out_edge
            # and the removal of the central vertex from both neighbours.
            if g0.get(ie) is not None:
                g[ie] = dict(g0[ie])
            if t0.get(oe) is not None:
                t[oe] = dict(t0[oe])
        ij.graph, ij.tgraph = g, t
        ij.vo = dict(vo0) if isinstance(vo0, dict) else vo0
        try:
            ij.eliminate(int(vertex), vhooks, ft)
            out = []
            for fr in sink.faces[n_fc:]:
                s, e = fr.start, fr.end
                if not n_eq <= s <= e <= len(teq):
                    raise _Irregular(f"face range {s}..{e} outside the step")
                apx = []
                for r in fr.approx:
                    if not s <= r.start <= r.end <= e:
                        raise _Irregular(f"approximation range {r.start}.."
                                         f"{r.end} outside face {s}..{e}")
                    apx.append(r._replace(start=r.start - s, end=r.end - s))
                out.append((fr._replace(start=0, end=e - s, approx=apx),
                            [_frozen_eqn(teq[i]) for i in range(s, e)]))
        finally:
            ij.graph, ij.tgraph, ij.vo = g0, t0, vo0
            del teq[n_eq:]
            del ij.steps[n_st:]
            del sink.faces[n_fc:]
            del ij.xlog.records[n_x:]
        return out

    def _render(self, tk, rs, trace):
        """``(tokens, split)`` of one face, emitted on name state ``rs``."""
        fr, eqns = trace
        mk = (tk.digit_base, tk._name_alphabet)
        memo = self._namememo.get(mk)
        if memo is None:
            from graphax.jaxpr import name_gen_python_style
            memo = self._namememo[mk] = (
                [], name_gen_python_style(tk.digit_base,
                                          tk.digit_base + tk._name_alphabet))
        saved = (tk._names, tk._fns, tk._namegen, tk._flatten_uid,
                 getattr(tk, "_cur_spans", None),
                 getattr(tk, "_cur_eqn_spans", None))
        rs.names.base, rs.fns.base = saved[0], saved[1]
        cur = _NameCursor(memo[0], memo[1], rs.pos)
        tk._names, tk._fns, tk._namegen, tk._flatten_uid = (
            rs.names, rs.fns, cur, rs.uid)
        tk._cur_spans = None
        tk._cur_eqn_spans = None
        out: list = []
        try:
            split = tk._emit_face(fr, eqns, out)
        finally:
            rs.pos, rs.uid = cur.pos, tk._flatten_uid
            (tk._names, tk._fns, tk._namegen, tk._flatten_uid,
             tk._cur_spans, tk._cur_eqn_spans) = saved
        self.stats["face_renders"] += 1
        return [int(x) for x in out], split

    def n_faces(self, order, specs, n, vertex, face_rows_hist=None,
                face_skips_hist=None, *, hist_key=None):
        """Face count of ``vertex`` on the live prefix graph -- the rollout
        while_loop's trip count. Raises if it ever exceeds ``max_faces``:
        the width is the provable ancestors-x-descendants bound, so an
        excess means the bound argument is violated and a silent clamp
        would shrink the action space behind a healthy-looking run."""
        try:
            tk = self._tokenizer_at(
                np.asarray(order).reshape(-1), np.asarray(specs), int(n),
                face_rows_hist, face_skips_hist, hist_key=hist_key)
            k = len(list(tk.ij.faces(int(vertex))))
        except Exception:
            self.stats["failures"] += 1
            return 0
        if k > self.max_faces:
            raise RuntimeError(
                f"vertex {vertex}: {k} faces exceed the derived bound "
                f"{self.max_faces}")
        # AND THE WIRE MUST BE WIDE ENOUGH FOR THIS VERTEX, checked HERE,
        # before the step that decides its faces has written anything. Under
        # `--face-wire-faces` the history handed to the next step carries only
        # the first columns; a vertex with more faces than that would have its
        # tail decided this step and dropped from every later prefix.
        if face_rows_hist is not None:
            _w = int(np.asarray(face_rows_hist).shape[1])
            if k > _w:
                raise RuntimeError(
                    f"vertex {vertex} has {k} faces and the live-face wire "
                    f"carries {_w} columns (--face-wire-faces). Raise it: the "
                    f"faces past {_w} would be decided this step and lost "
                    f"from every later prefix.")
        return int(k)

    # -- per-face LIVE dim sizes (--per-face-masks, SIZES half) -------------
    # THE SIZES HALF ON THE CAMPAIGN PATH. `--per-face-masks` has two halves
    # (see `common/masks.py`, "Two independent halves"): the apply-time
    # projection, which runs everywhere, and the SIZES the head reads, which
    # A1 could only source from `LiveVertexMaskOracle.face_dim_sizes` -- and
    # `--live-faces` runs no oracle (`ppo._NO_ORACLE`), so on every v57-v66
    # run and R1-R3 the head masked with the STATIC per-VERTEX axis vector.
    #
    # It does not need the oracle. This stream already re-runs the real
    # elimination of the current vertex once per face; the SAME machinery
    # (`_Snapshot` + `graphax.core._eliminate_vertex`) yields the live
    # per-face operand directly, on the prefix tokenizer that is already
    # built and cached. Cost: TWO extra eliminations per (prefix, vertex) --
    # not per face -- memoized in `self._sizes`, against the `n_faces`
    # eliminations `chunk_ex` already pays for the same vertex.
    #
    # DELIBERATELY IDENTICAL IN FORM to the oracle's
    # `face_masks_and_sizes(per_face=True)`, so the two are comparable (and
    # `tests/per_face_sizes_live_test.py` asserts they agree face for face):
    #
    #   * probed on the single contraction engine (dsnn-3qm.65);
    #   * `quant[k]` is 1 iff some dtype the 94-slot head can emit is a
    #     legal, NON-idempotent cast on the operand;
    #   * a per-vertex transform (an identity callable) is installed for the
    #     probe, exactly as `probe_faces` does, because that is what arms
    #     graphax's `_is_approx_cfg` -- the code path the real run takes.
    #
    # ONE DELIBERATE DIFFERENCE, and it is a fix. The oracle indexes its
    # faces by PROBE VISIT ORDER; the face index `f` the policy and
    # `chunk_ex` use is a position in `faces_of` -- the ENUMERATION, of which
    # the visited faces are a subsequence (a face whose edge Jacobian forces
    # to None is enumerated and never visited). Indexing sizes positionally
    # would hand face `f` a later face's dims whenever anything is dropped.
    # So the probe is keyed BY FACE KEY: the recording hook rides in
    # `face_transforms`, which graphax looks up by exactly the
    # `(vidx[in_edge], vidx[out_edge])` pair `faces_of` returns. A callable
    # there is not a `Diag`/`Compress`/`SKIP_FACE`, so it does NOT arm
    # `_is_approx_cfg` (core.py's `if face_transforms:` test) -- the arming
    # still comes from the per-vertex identity, as in the oracle.
    #
    # An enumerated-but-unvisited face gets an ALL-ZERO size row. That reads
    # back through `UnifiedFacePolicy._face_feats_1` as an all-invalid axis
    # set (`valid = sz > 0`), i.e. no legal pair and no legal compress axis,
    # so no approximation is offered on a face the elimination never
    # contracts. That is the correct answer, and it is counted (`size_miss`)
    # rather than being silently indistinguishable from a real zero.

    @staticmethod
    def _size_probe_identity(st):
        """The per-vertex transform the size probe installs.

        Identity in value, NOT in effect: ``graphax.core._eliminate_vertex``
        sets ``_is_approx_cfg`` from ``any(isinstance(t, (Diag, Compress)) or
        callable(t) for t in transforms)``, so installing a callable is what
        puts the probe on the same code path the approximated run takes.
        ``LiveVertexMaskOracle.probe_faces`` does exactly this.
        """
        return st

    def _probe_faces(self, tk, vertex, keys, slots: bool = False,
                     stat: str = "size"):
        """``{face key: live SparseTensor}`` from ONE speculative elimination.

        One probe, on the one contraction engine (dsnn-3qm.65): the second
        dispatch mode this used to intersect over is gone with the planner.

        Runs INSIDE a :class:`_Snapshot`, so the speculative elimination is
        undone in full -- graph, transpose graph, ``vo``, traced equations,
        face records, transform log, variable names and the name generator.
        Nothing it touches survives, which is what lets it run against the
        live prefix tokenizer instead of a private rebuild of it.

        ``slots=True`` (ticket .18 D3) records ONE TENSOR PER GRAPHAX SITE,
        through ``env.face_entry_from_slots`` itself -- the single place a
        face wire becomes a graphax entry -- so the sites recorded here are
        by construction the sites the measurement installs. Returns
        ``{face key: {site: live SparseTensor}}`` keyed by
        ``env.face_slot_sites()``'s names, which under ``--approx-add`` are
        ``"lhs"``, ``"rhs"`` and ``"res:new"`` -- one site per slot, all of them
        PRE-JOIN.

        Finding 72 / ticket .59 fault 1: this used to hard-code
        ``((lhs, rhs, new), (None, None, None))``, recording the fresh
        contraction and nothing else, while the measurement's entry put the
        SAME ``new`` hook on graphax's ``jr`` -- the existing OLD EDGE, a
        different tensor with its own index structure. The mask then cleared
        Diags the engine refused there (6 of 32 on TLM). Hard-coding the entry
        form in a second place is what allowed the drift, so it is not
        hard-coded in a second place any more.
        """
        from jax._src import core as _jcore
        from graphax.core import _eliminate_vertex

        seen: dict = {}

        def _mk(k, site=None):
            def _rec(st):
                # setdefault: in the rare multi-outvar case two faces can
                # collide on one key (faces_of's own note) and the FIRST is
                # the one `chunk_ex` maps `f` to.
                if site is None:
                    seen.setdefault(k, st)
                else:
                    seen.setdefault(k, {}).setdefault(site, st)
                return st
            return _rec

        if slots:
            from alphagrad.approx.env import face_entry_from_slots
            # ONE recorder per SITE: face_entry_from_slots installs a slot's
            # single hook at several sites, and `at_site` is the only way to
            # tell those invocations apart (the objects are identical).
            # with_policy=False: the slot HOOKS go in (so every tensor a slot
            # will meet is recorded, INCLUDING the old edge and the summed edge)
            # but no join POLICY, and the arm is not consulted. The probe needs
            # the tensors, which the hooks deliver; the reconciliation only
            # changes values and this elimination is undone in full by the
            # snapshot. Keeping it would make every probe pay the
            # reconciliation's arithmetic for a result nothing reads -- and
            # under `--approx-add choose` the probe does not hold the per-face
            # bit, so asking for the arm would raise.
            from alphagrad.approx.env import wire_slots
            ft = {k: face_entry_from_slots(
                      tuple(range(wire_slots())),
                      at_site=lambda site, _h, k=k: _mk(k, site),
                      with_policy=False)
                  for k in keys}
        else:
            ft = {k: (None, None, _mk(k)) for k in keys}
        with _Snapshot(tk) as snap:
            ij = snap.ij
            try:
                self.stats[f"{stat}_probe"] += 1
                with _jcore.set_current_trace(ij.trace):
                    _eliminate_vertex(
                        int(vertex), ij.jaxpr, ij.graph, ij.tgraph, ij.vo,
                        False, transforms=(self._size_probe_identity,),
                        face_transforms=ft,
                    )
            except Exception:
                # A vertex graphax cannot trace has no legal
                # approximation either: keep the faces seen so far (the
                # rest stay zero, i.e. nothing offered) and never let a
                # probe take the rollout down. Same contract as
                # `probe_faces`.
                self.stats[f"{stat}_probe_fail"] += 1
        return seen

    def face_dim_sizes(self, order, specs, n, vertex,
                       face_rows_hist=None, face_skips_hist=None, *,
                       hist_key=None):
        """``(sizes (F, N) int32, quant (F,) float32, n_faces)``.

        ``sizes[k]`` is face ``k``'s LIVE ``logical_size`` vector over
        ``out_dims ++ primal_dims`` -- ``masks.dim_logical_sizes``, i.e. THE
        NUMBERING ``Diag(i, j)``, ``diag_valid_mask``,
        ``diag_pair_factor_space`` and ``rule_is_legal`` are indexed in, which
        is the numbering the decision is actually made in at apply time.

        NOT ``val.shape``. The physical axes of the sparse tensor are a
        different (shorter, order-dependent) list -- a coupled pair stores two
        logical dims in one physical axis -- so handing those to the head
        would align with nothing it emits. ``face_features`` keeps
        ``val.shape``; this deliberately does not.

        ``quant[k]`` is 1.0 iff at least one of ``FACE_QUANT_DTYPES`` (the two
        dtypes the 94-slot head's Bernoulli can actually request) is a legal
        and non-idempotent cast on face ``k``'s operand under BOTH dispatch
        modes.

        Rows ``>= n_faces``, and any enumerated face the elimination did not
        visit, are zero -- which the head reads as "no valid axis", hence no
        legal DIAG pair and no legal COMPRESS axis on that face.

        NO VERTEX RULES ARE APPLIED. The probe carries the per-vertex identity
        only, exactly as the oracle's does: under ``--live-faces`` the
        per-vertex wire rows are always the exact END rows (approximation is
        purely per-face), so there is nothing to apply, and keeping it that
        way makes this bit-comparable with `face_masks_and_sizes`.
        """
        from alphagrad.approx.common.masks import (
            FACE_QUANT_DTYPES, dim_logical_sizes, quant_valid_mask)

        F, N = self.max_faces, self.max_axes
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex = int(n), int(vertex)
        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        # The sizes depend on the PREFIX and the vertex only -- not on the
        # in-flight per-face decisions of this vertex. Faces of one vertex
        # write disjoint (in_edge, out_edge) edges and read only edges
        # incident to the central vertex, so approximating face f-1 cannot
        # move face f's operand. That is what makes one probe serve the whole
        # face loop instead of one per face.
        ck = ((order[:n].tobytes(), specs[:n].tobytes(), vertex)
              + (_hist_key_parts(frh, fsh, n) if hist_key is None
                 else (hist_key,)))
        hit = self._sizes.get(ck)
        if hit is not None:
            self.stats["size_hit"] += 1
            return hit

        sizes = np.zeros((F, N), np.int32)
        quant = np.zeros((F,), np.float32)
        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
            keys = list(tk.ij.faces(vertex))
        except Exception:
            self.stats["failures"] += 1
            return sizes, quant, np.int32(0)

        src = self._probe_faces(tk, vertex, keys)
        # Index space = the ENUMERATION, because that is what `f` means in
        # `_face_loop` / `chunk_ex` / `n_faces`.
        n_faces = min(len(keys), F)
        for k in range(n_faces):
            kk = keys[k]
            st = src.get(kk)
            if st is None:
                self.stats["size_miss"] += 1
                continue
            try:
                sizes[k] = dim_logical_sizes(st, N)
            except Exception:
                self.stats["size_miss"] += 1
                continue
            try:
                qm = quant_valid_mask(st, FACE_QUANT_DTYPES)
            except Exception:
                qm = np.zeros((len(FACE_QUANT_DTYPES),), dtype=bool)
            quant[k] = 1.0 if bool(qm.any()) else 0.0

        res = (sizes, quant, np.int32(n_faces))
        if len(self._sizes) >= 4096:
            for dk in list(self._sizes)[:1024]:
                self._sizes.pop(dk, None)
        self._sizes[ck] = res
        return res

    # -- per-face, per-SLOT legality (--face-slot-frames, ticket .18 D3) ----
    def face_slot_legality(self, order, specs, n, vertex,
                           face_rows_hist=None, face_skips_hist=None, *,
                           hist_key=None):
        """``(sizes (F,S,N) int32, quant (F,S) f32, pair (F,S,N,N) f32,
        comp (F,S,N) f32, n_out (F,S) int32, n_faces)`` -- per face AND per
        operand slot ``S = (lhs, rhs, new)``.

        :meth:`face_dim_sizes` records the RESULT site and the policy
        broadcast that one vector to all three slots; finding 54 shows the
        slots are not alike (rhs and new have no out side under a scalar
        loss). This probe records the three slots the env's hooks are applied
        at, and each entry is ``masks.slot_legality`` on that slot's tensor:
        the head's Diag pair / Reduce axis / Quant on slot ``s`` is legal iff
        slot ``s``'s hook will apply it. ``n_out`` is what the wire encoder
        needs to write ``bi2 = j - n_out`` in the slot's own frame.

        Same memo, same dispatch-mode AND, same face-key indexing and same
        "unvisited face = all zero" contract as :meth:`face_dim_sizes`; a
        pure read of the prefix tokenizer, like it.

        AND OVER SITES, not just over dispatch modes (finding 72, .59 fault
        1). A slot's ONE hook is invoked at every site
        ``env.face_slot_sites()`` lists for it, on a DIFFERENT tensor each
        time. Under ``--approx-add`` (finding 73) every slot has exactly ONE
        site, so the AND is over one tensor; the retired ``--approx-old same``
        also landed ``new`` on the existing old edge, which is the drift this
        construction exists to survive. The first site names the slot and supplies
        ``sizes`` / ``n_out`` (the frame the wire is written in); the rest are
        handed to ``slot_legality(also=...)``, which re-decodes the same wire
        row in each one's frame. A merge-free face has no old edge, graphax
        never reaches its join position, the probe records no tensor for that
        site, and its mask is unchanged.
       
        EVERY ROW HERE IS READ BEFORE ANY DECISION IS MADE, and for the three
        DEPENDENT sites that is stale rather than wrong. Two methods ask the
        same question after the decisions land:
        :meth:`decide_faces` (one speculative elimination, every slot a graphax
        chooser -- the only one that can answer for ``res:jr`` and
        ``res:jres``), and :meth:`decide_vertex_faces` (NO elimination at all:
        ``n`` in-edge forces + ``m`` out-edge forces + ``n*m`` calls of
        ``graphax.contract_face_operands``, which answers for ``res:new`` and
        refuses the other two).

        This method stays, and stays memoised, because its answer is a pure
        function of the prefix and because the two OPERAND rows it produces are
        exact (measured: 0 rejections of 36 requests with ``lhs`` and ``rhs``
        armed). ``res:new`` / ``res:jr`` / ``res:jres`` are not: on TLM, 5
        seeds, one graph, the production draw convention, 15 of 526 rows this
        mask clears are REFUSED at apply time, against 0 of 522 for either
        exact pass (job 64928). It also costs MORE than the exact one: 7.663
        ms/vertex here against 5.492 for :meth:`decide_vertex_faces` on the
        same prefix path, because this runs a speculative elimination and that
        one runs none (the 7.663 was measured with the two dispatch-mode probes
        of the two-engine era; dsnn-3qm.65 left one).
        """
        from alphagrad.approx.common.masks import slot_legality
        from alphagrad.approx.env import face_slot_sites

        F, N = self.max_faces, self.max_axes
        # S FROM THE SITE TOPOLOGY, not from the 3-name contraction tuple. The
        # mask must have one row per slot the entry builder places a hook for,
        # or a slot's decision would be drawn under a row nobody filled in --
        # which is finding 72's fault 1 with the bands the other way round.
        # `face_slot_sites()` derives that list by calling the entry builder, so
        # this widens by itself when a value adds a slot.
        sites = face_slot_sites()
        S = len(sites)
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex = int(n), int(vertex)
        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        ck = ((order[:n].tobytes(), specs[:n].tobytes(), vertex)
              + (_hist_key_parts(frh, fsh, n) if hist_key is None
                 else (hist_key,)))
        hit = self._slots.get(ck)
        if hit is not None:
            self.stats["slot_hit"] += 1
            return hit

        sizes = np.zeros((F, S, N), np.int32)
        quant = np.zeros((F, S, NUM_FACE_QUANT_DTYPES), np.float32)
        pair = np.zeros((F, S, N, N), np.float32)
        comp = np.zeros((F, S, N), np.float32)
        nout = np.zeros((F, S), np.int32)
        empty = (sizes, quant, pair, comp, nout, np.int32(0))
        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
            keys = list(tk.ij.faces(vertex))
        except Exception:
            self.stats["failures"] += 1
            return empty

        src = self._probe_faces(tk, vertex, keys, slots=True, stat="slot")
        n_faces = min(len(keys), F)
        for k in range(n_faces):
            kk = keys[k]
            by_site = src.get(kk)
            for s, site_list in enumerate(sites):
                site = site_list[0]
                st = None if by_site is None else by_site.get(site)
                if st is None:
                    self.stats["slot_miss"] += 1
                    continue
                also = () if by_site is None else tuple(
                    by_site[x] for x in site_list[1:] if by_site.get(x)
                    is not None)
                try:
                    L = slot_legality(st, N, also=also)
                except Exception:
                    self.stats["slot_miss"] += 1
                    continue
                sizes[k, s] = L.sizes
                nout[k, s] = L.n_out
                pair[k, s] = L.pair.astype(np.float32)
                comp[k, s] = L.comp.astype(np.float32)
                quant[k, s] = L.quant.astype(np.float32)

        res = (sizes, quant, pair, comp, nout, np.int32(n_faces))
        if len(self._slots) >= 4096:
            for dk in list(self._slots)[:1024]:
                self._slots.pop(dk, None)
        self._slots[ck] = res
        return res


    # -- the DYNAMIC per-slot mask (ticket .59 fault 2) ---------------------
    #
    # WHAT WAS STALE. :meth:`face_slot_legality` records its tensors in ONE
    # speculative elimination per vertex whose slot hooks are RECORDERS -- they
    # return the operand unchanged -- so every tensor it answers about is the
    # tensor an ALL-EXACT plan would produce. The mask is therefore computed
    # against a graph in which no decision has been made, and three things move
    # the tensor between that reading and the apply (findings 72 section "Fault
    # 2", 73 section 9b, 74 section 8; all measured on TLM, minimum Markowitz,
    # 5 seeds, through the real apply path with the engine's own
    # ``skipped_<kind>`` counters):
    #
    #   1. WITHIN A FACE, across slots. ``res:new`` holds the contraction of
    #      ``lhs`` and ``rhs``. Approximate either operand and the tensor the
    #      ``new`` mask described no longer exists. ``new`` armed alone rejects
    #      0 of 27 Diag; with the operands armed, 6 of 26 Diag and 4 of 40
    #      Reduce.
    #   2. ACROSS FACES at one vertex. An earlier face's merge WRITES the edge a
    #      later face's ``res:jr`` reads, so arming the contraction slots stales
    #      learned1's mask WITHIN one elimination step: 0 of 27 alone, 3 of 594
    #      with them armed.
    #   3. THE SUMMED EDGE. ``res:jres`` IS ``new + old``, so it carries this
    #      face's own contraction approximations: 0 of 181 alone, 20 of 741 with
    #      them armed.
    #
    # All three are ORDER, not arithmetic: graphax's face loop is strictly
    # sequential (``core._eliminate_vertex``: ``lhs`` -> ``rhs`` -> contract ->
    # ``res:new`` -> ``jl`` -> ``jr`` -> ``+`` -> ``res:jres``, faces visited
    # out-edge-major / in-edge-minor, and ``graph``/``tgraph`` are written once
    # per face inside the innermost loop), so every tensor a decision needs to
    # be masked from ALREADY EXISTS by the time that decision's hook is
    # invoked -- just not before the elimination starts.
    #
    # WHAT THIS DOES. ONE speculative elimination per vertex in which every slot
    # is a graphax CHOOSER (``core._apply_face_transform``'s callable branch,
    # which exists for exactly this: "a policy cannot know their index structure
    # until this moment; masking legal actions requires seeing the tensor"). At
    # each site the chooser
    #
    #   * computes ``masks.slot_legality`` on THE TENSOR IN HAND,
    #   * calls the caller's ``draw`` with it -- outside the jaxpr trace, see
    #     ``jax.ensure_compile_time_eval`` below -- to get a wire row,
    #   * and decides that row through ``env.make_slot_frame_hook``, THE APPLY
    #     PATH'S OWN HOOK, handing the action it picks back to graphax -- which
    #     applies AND RECORDS it, so the tensor the next site sees is the tensor
    #     the measurement will produce and the block the stream carries is the
    #     block the measurement carries.
    #
    # so the mask is recomputed as decisions land without a single re-probe: the
    # elimination itself is the propagation. Nothing is copied and nothing is
    # retraced -- :class:`_Snapshot` undoes the pass in place, exactly as it
    # already does for the recording probe, and the persistent
    # ``IncrementalJaxpr`` trace is extended and truncated rather than rebuilt.
    #
    # WHY NOT RE-PROBE PER DECISION (the obvious reading of "recompute as
    # decisions land"): it is the same answer at F*S times the cost, and that
    # cost is the one ``_tokenizer_at``'s prefix cache exists to remove. See
    # finding 75 for the measurement that rejected it.
    #
    # WHY THE DECIDED ROW, NOT THE CHOSEN ACTION, IS WHAT LEAVES THIS PASS. The
    # chooser could hand graphax the micro-action directly and never mention a
    # wire row; the measured elimination would then be the only place the
    # decision exists. It has to be a ROW because the row is what the rest of
    # the system already transports: ``_decided`` replays faces ``0..f-1`` from
    # rows to build face ``f``'s token chunk (so a decision that only exists
    # after the real elimination is circular), ``plan_log`` records rows, and
    # the PPO replay rescores from the stored mask. Deciding into a row keeps
    # ONE graphax-facing apply path (``make_slot_frame_hook`` decides, graphax
    # applies) and leaves the measured elimination byte-identical to what this
    # pass traced.
    def decide_faces(self, tk, vertex, keys, draw, *, skips=None):
        """Decide every face slot of ``vertex`` IN APPLY ORDER, one elimination.

        ``draw(f, s, legality)`` is called once per (face ``f``, slot ``s``)
        that graphax actually reaches, in the order graphax reaches them, with
        ``legality`` a :class:`~alphagrad.approx.common.masks.SlotLegality`
        computed on the live tensor at that slot's own site. It returns the
        wire row ``(b0, b1, b2)`` to install there, or ``None`` for "leave this
        slot exact".

        ``draw`` IS CALLED OUTSIDE THE JAXPR TRACE. The pass runs under
        ``jcore.set_current_trace(ij.trace)``, so an unguarded ``jnp`` op inside
        a hook is traced into the persistent frame and its result is a
        ``DynamicJaxprTracer`` -- measured: ``jnp.asarray(1.) + jnp.asarray(2.)``
        inside a hook is a tracer, and the same expression inside
        ``jax.ensure_compile_time_eval()`` is the concrete ``3.0``. A policy head
        is jnp, and a tracer cannot be turned into a wire row, so the escape is
        a requirement of the design and not a convenience.

        ``skips`` -- the ``(F,)`` per-face SKIP decisions, which must already be
        made: graphax tests ``face_transforms[key] is SKIP_FACE`` BEFORE it runs
        any of the face's hooks, so a skip cannot be chosen by a chooser. This
        costs nothing in distribution terms: the skip draw reads the head's
        logit 0 and its only masks are ``face_valid`` and ``approx_ok``, neither
        of which is a per-slot tensor legality (``UnifiedFacePolicy.sample_face``
        takes ``approx_ok`` from ``allow_skip`` / ``op_legality_override``, never
        from ``pair_valid`` / ``comp_valid``), so the skip drawn before this pass
        is the skip a rescore against the FINAL masks reproduces.

        Returns :class:`DecidedFaces`: the wire rows AND the per-(face, slot)
        legality each row was drawn under. The masks are returned because they
        are what PPO has to store -- the loss replay rescores from the stored
        ``face_pair_valid`` / ``face_comp_valid`` / ``face_sizes`` /
        ``face_quant`` arrays and never recomputes them, so storing THESE keeps
        ``sample`` and ``evaluate`` scoring the same variable. That they can be
        drawn slot by slot and rescored in one call is a property of the head,
        measured rather than assumed: ``UnifiedFaceHead.sample`` splits
        ``1 + 6*n_slots (+1)`` keys and slot ``s`` reads ``keys[1+5s:1+5(s+1)]``
        plus ``dt_keys[s]`` against slot ``s``'s own logit block and slot ``s``'s
        own masks, so nothing is conditioned on another slot's draw (probe
        t75_feas, question 3: 0 of 252 field cells differed between ``S``
        sequential calls with progressively refined masks and ONE call with all
        of them).

        ``draw`` CANNOT READ A TOKEN CHUNK, and that is a hard limit of this
        shape rather than an omission. :meth:`chunk_ex` builds face ``f``'s chunk
        by calling :meth:`_tokenizer_at` and then :meth:`_decided` and running a
        whole speculative elimination of its own, inside its own
        :class:`_Snapshot` on the same tokenizer -- so calling it from inside
        this pass would truncate THIS pass's equations out from under it. A
        caller that needs the ``--live-faces`` interleave (read face ``f``'s
        contraction tokens, THEN approximate face ``f``) must therefore use this
        pass as a mask REFRESH rather than as the single decision point: draw
        slots 0 and 1 in the per-face loop that reads the chunks, call this with
        a ``draw`` that returns those rows for slots 0 and 1 and ``None`` for the
        rest -- which records their masks without deciding them -- and draw the
        dependent slots in a second pass over the same per-face loop. That is
        sound for ``res:new`` specifically, because ``res:new`` is the
        contraction of THIS face's own two operands and of nothing a sibling face
        touches. It is NOT sound for ``res:jr`` / ``res:jres``, whose tensors
        depend on sibling faces' decisions, so those two have to be decided in
        one pass -- which is what the join-slot tests do, and why the trainer
        does not offer them yet.
        """
        from jax._src import core as _jcore
        import jax as _jax
        from graphax import SKIP_FACE
        from graphax.core import _eliminate_vertex
        from alphagrad.approx.common.masks import slot_legality
        from alphagrad.approx.env import (
            face_entry_from_slots, make_slot_frame_hook, wire_slots)

        F, N = self.max_faces, self.max_axes
        S = wire_slots()
        rows = np.full((F, S, 3), -1, np.int32)
        rows[..., 2] = 0
        sizes = np.zeros((F, S, N), np.int32)
        quant = np.zeros((F, S, NUM_FACE_QUANT_DTYPES), np.float32)
        pair = np.zeros((F, S, N, N), np.float32)
        comp = np.zeros((F, S, N), np.float32)
        nout = np.zeros((F, S), np.int32)
        n_faces = min(len(keys), F)
        decided: dict = {}

        def _mk(f, s):
            # The slot hook of the LAST call, so graphax's outcome callback
            # reaches the object whose counters it is about. The hook IS a
            # chooser now, so what this returns is an action and graphax is
            # what applies and RECORDS it -- which is how a decided rule
            # reaches this pass's token stream as an ``approx`` block, exactly
            # as it reaches the measurement's.
            held: dict = {}

            def _chooser(st):
                if (f, s) in decided:
                    # A slot's hook reached a SECOND site. Under every current
                    # --approx-add value `face_slot_sites()` is one site per
                    # slot, so this cannot fire; if a future value brings the
                    # two-site form back (the shape of finding 72's fault 1),
                    # the row already drawn is what the measurement will install
                    # at both sites, so installing it here too is what keeps
                    # this pass faithful -- and the counter says it happened.
                    self.stats["decide_multi_site"] += 1
                    row = decided[(f, s)]
                    if row is None:
                        return None
                    h = make_slot_frame_hook(row)
                    held["hook"] = h
                    return h(st)
                L = slot_legality(st, N)
                sizes[f, s] = L.sizes
                nout[f, s] = L.n_out
                pair[f, s] = L.pair.astype(np.float32)
                comp[f, s] = L.comp.astype(np.float32)
                quant[f, s] = L.quant.astype(np.float32)
                with _jax.ensure_compile_time_eval():
                    w = draw(f, s, L)
                self.stats["decide_draw"] += 1
                if w is None:
                    decided[(f, s)] = None
                    return None
                row = tuple(int(x) for x in w)
                decided[(f, s)] = row
                rows[f, s] = row
                # THE APPLY PATH'S OWN HOOK, not a second copy of the decode
                # (finding 72's fault 1 was exactly a second copy of the entry
                # form). `stats` is a LOCAL dict so the pass never touches
                # `env._PER_FACE_STATS`, and a skip counted in it means the
                # apply path refused a row on the very tensor it was drawn
                # from -- i.e. the mask and the hook disagree, which is the
                # defect this pass exists to remove, so it is counted and
                # visible rather than papered over.
                seen: dict = {}
                h = make_slot_frame_hook(row, stats=seen, gated=False)
                held["hook"] = h
                out = h(st)
                if seen.get("skipped"):
                    self.stats["decide_self_skip"] += 1
                return out

            def _chosen_applied(action, applied):
                h = held.get("hook")
                if h is not None:
                    h.chosen_applied(action, applied)

            _chooser.chosen_applied = _chosen_applied
            return _chooser

        # with_policy=False, for the reason `_probe_faces` uses it and one more.
        # A join POLICY is not a slot hook: it answers to no mask and applies no
        # wire row, so it can change no decision taken here. Under every value
        # in `APPROX_ADD_CHOICES` that is provable rather than probable -- the
        # only value whose join is `lossy` has exactly THREE slots, so it has no
        # hook at `jr` or `jres` for a reconciliation to sit in front of, and
        # `learned1` / `learned2` reconcile with the UNION, which installs no
        # policy at all (env._JOIN_SEMANTICS_OF, finding 74 section 7). It is
        # also the only form that can answer under `--approx-add choose`, where
        # the arm is a per-face bit this pass does not hold.
        ft = {}
        for f in range(n_faces):
            if skips is not None and int(np.asarray(skips).reshape(-1)[f]) == 1:
                ft[keys[f]] = SKIP_FACE
                continue
            ft[keys[f]] = face_entry_from_slots(
                tuple(_mk(f, s) for s in range(S)), with_policy=False)

        with _Snapshot(tk) as snap:
            ij = snap.ij
            try:
                self.stats["decide_probe"] += 1
                with _jcore.set_current_trace(ij.trace):
                    _eliminate_vertex(
                        int(vertex), ij.jaxpr, ij.graph, ij.tgraph, ij.vo,
                        False, transforms=(self._size_probe_identity,),
                        face_transforms=ft)
            except Exception:
                # Same contract as `_probe_faces`: a vertex graphax cannot
                # trace has no legal approximation either, so keep the
                # decisions taken so far (the rest stay -1, i.e. exact) and
                # never let a probe take the rollout down.
                self.stats["decide_probe_fail"] += 1
        return DecidedFaces(rows=rows, sizes=sizes, quant=quant, pair=pair,
                            comp=comp, nout=nout, n_faces=np.int32(n_faces))

    def face_slot_decisions(self, order, specs, n, vertex, draw, *,
                            skips=None, face_rows_hist=None,
                            face_skips_hist=None, hist_key=None):
        """:meth:`decide_faces` against the PREFIX tokenizer of step ``n``.

        The tokenizer comes from :meth:`_tokenizer_at`, so the decisions are
        taken on the same graph ``face_slot_legality`` reads and at the same
        cost in prefix terms -- one cache lookup, not a replay.

        NOT MEMOISED, and it must not be: the result depends on ``draw``, i.e.
        on the policy and the key, so a ``(prefix, vertex)`` memo would serve
        one step's decisions to the next. :meth:`face_slot_legality`'s memo is
        sound only because its answer is a pure function of the prefix.
        """
        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex = int(n), int(vertex)
        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
            keys = list(tk.ij.faces(vertex))
        except Exception:
            self.stats["failures"] += 1
            F, N, S = self.max_faces, self.max_axes, _wire_slots()
            rows = np.full((F, S, 3), -1, np.int32)
            rows[..., 2] = 0
            return DecidedFaces(
                rows=rows, sizes=np.zeros((F, S, N), np.int32),
                quant=np.zeros((F, S, NUM_FACE_QUANT_DTYPES), np.float32),
                pair=np.zeros((F, S, N, N), np.float32),
                comp=np.zeros((F, S, N), np.float32),
                nout=np.zeros((F, S), np.int32), n_faces=np.int32(0))
        return self.decide_faces(tk, vertex, keys, draw, skips=skips)

    # -- the EXACT per-vertex mask, NO speculative elimination (.59, #77) ---
    #
    # THE PROOF. Let vertex ``j`` have predecessors ``P = {i1..in}`` and
    # successors ``S = {k1..km}``. Its faces are ``P x S``, and for face
    # ``(p, s)``
    #
    #     reads  = { e(p,j), e(j,s), e(p,s) }      writes = { e(p,s) }
    #
    # For ``f != g``, ``writes(f) & reads(g)`` is EMPTY:
    #
    #   * ``e(pf,sf) == e(pg,j)`` needs ``sf == j``, impossible -- ``sf`` is a
    #     successor of ``j`` and the graph is acyclic;
    #   * ``e(pf,sf) == e(j,sg)`` needs ``pf == j``, impossible likewise;
    #   * ``e(pf,sf) == e(pg,sg)`` needs ``f == g``.
    #
    # So the face -> written-edge map is a BIJECTION onto ``P x S``, and the
    # ``lhs`` / ``rhs`` reads are re-seeded per face from graphax's own
    # ``_pre_raw`` / ``_post_raw`` (``core.py``: "an ``rhs`` transform must
    # affect THIS face only, not the rest of the out_edge's fan-in"). Therefore
    #
    #     new(p,s) = contract(a_ps(e(p,j)), b_ps(e(j,s)))
    #
    # with ``a_ps`` / ``b_ps`` this face's own operand decisions and NOTHING
    # else -- no sibling face, no ordering. Every ``val.shape`` and every
    # ``SparseTensor`` metadata field the mask reads is a PURE FUNCTION of the
    # PRE-elimination graph and the decision vector.
    #
    # WHAT THAT BUYS. :meth:`decide_faces` reached the same answer by running a
    # whole speculative elimination per vertex with every slot a graphax
    # chooser; the arithmetic there is right, but it pays an elimination and --
    # because the choosers run under ``jcore.set_current_trace(ij.trace)`` -- it
    # pays ``jax.ensure_compile_time_eval`` on EVERY draw, which is where
    # finding 75 measured the head channel going 0.437 -> 8.957 ms/vertex. This
    # pass pays NEITHER:
    #
    #   * ``n`` forces of the in-edge Jacobians + ``m`` forces of the out-edge
    #     Jacobians + ``n*m`` calls of ``graphax.contract_face_operands`` --
    #     THE FUNCTION THE REAL ELIMINATION CALLS, not a second copy of the
    #     structure algebra (a second copy is exactly what produced finding
    #     72's fault 1);
    #   * every ``draw`` is taken OUTSIDE the jaxpr trace, because the trace is
    #     only needed where jax equations are actually emitted (the forces and
    #     the contractions), and this pass can leave it per face instead of
    #     running inside it from end to end.
    #
    # WHY TWO STAGES. The head draws all three slots from ONE vector, so the
    # operand decisions and the ``new`` decision are nominally simultaneous;
    # but ``new``'s tensor IS the contraction of the decided operands, so its
    # mask cannot exist until they are decided. The resolution needs no
    # elimination and no re-probe: draw slots 0 and 1 for every face against
    # masks that are pure functions of the prefix (stage 1), then compose and
    # draw slot 2 (stage 2). Finding 75's measurement that a slot's draw is
    # conditioned on nothing but its OWN mask row
    # (``UnifiedFaceHead.sample`` splits ``1 + 6*n_slots (+1)`` keys and gives
    # slot ``s`` its own slice and its own logit block; 0 of 252 field cells
    # differed) is what makes the split scoreable with no new trajectory field.
    #
    # THE STAGES ARE PER VERTEX, NOT PER FACE, and that is not a convenience:
    # graphax's ``_is_approx_cfg`` is computed ONCE for the whole elimination
    # from the WHOLE ``face_transforms`` dict, and it gates the reconciler peel
    # and the re-evaluation of ``need_contract`` inside
    # ``prepare_face_operands``. A per-face split would have to guess it.
    # Hoisting stage 2 behind all of stage 1 means the flag is known exactly
    # (:func:`graphax.face_config_is_approx` on the entry forms the measurement
    # will install) before the first ``new`` structure is composed.
    #
    # WHAT IT DOES NOT COVER, and must not pretend to: ``res:jr`` (the OLD
    # EDGE) and ``res:jres`` (the SUMMED EDGE), i.e. slots 3 and 4 under
    # ``--approx-add learned1`` / ``learned2``. Those two tensors DO depend on
    # sibling faces -- an earlier face's merge writes the edge a later face's
    # ``jr`` reads -- so the bijection above says nothing about them and this
    # pass raises rather than answering. :meth:`decide_faces` is the pass for
    # those, and it stays for exactly that reason.
    def decide_vertex_faces(self, tk, vertex, draw, *, skips=None,
                            approx_cfg=None):
        """Decide every CONTRACTION slot of ``vertex`` with NO elimination.

        ``draw(f, s, legality)`` is called once per (face ``f``, slot ``s``),
        with ``legality`` a
        :class:`~alphagrad.approx.common.masks.SlotLegality` computed on the
        tensor that slot's hook will be handed at apply time. It returns the
        wire row ``(b0, b1, b2)`` or ``None`` for "leave this slot exact".
        ``draw`` is called OUTSIDE the jaxpr trace and needs no
        ``jax.ensure_compile_time_eval``: the only work this pass does under
        the trace is forcing the operand edges and composing the contractions.

        CALL ORDER IS STAGE ORDER, NOT APPLY ORDER: every face's slot 0 and
        slot 1, then every face's slot 2. See the block comment above for why
        that is exact and why the split has to be per vertex.

        ``skips`` -- the ``(F,)`` per-face SKIP decisions, already made, for
        the same reason :meth:`decide_faces` needs them: graphax tests
        ``face_transforms[key] is SKIP_FACE`` before running any of a face's
        hooks, so a skip cannot be chosen by a mask.

        There used to be an ``approx_dispatch`` argument here for graphax's
        ``approx_active`` flag, the second staleness source this pass had to
        close: the flag selected the elemental / planner lowering inside
        ``sparse_matmul``, so the two settings contracted the SAME operands into
        tensors with different index structure and dtype (measured on TLM: 3 of
        2145 per-(face, slot) mask fields differed between the forced and the
        caller's mode, all three the QUANT row at ``res:new``). dsnn-3qm.65
        removed the second engine and the flag with it; there is one lowering
        now and nothing left to match.

        ``approx_cfg`` -- ``None`` (the default) DERIVES graphax's
        ``_is_approx_cfg`` from the entry forms the measurement will install, via
        :func:`graphax.face_config_is_approx`, which is the elimination's own
        predicate. It is the one flag that is an ARGUMENT of the contraction
        (it gates the reconciler peel and the re-evaluation of
        ``need_contract``), and the derivation is what makes this pass exact
        about `IncrementalJaxpr.eliminate`, whose ``transforms=()`` leaves the
        flag to the face dict. Override it only to answer about a DIFFERENT
        caller: ``True`` is what :meth:`decide_faces` and :meth:`_probe_faces`
        effectively run under, because they install a per-vertex CALLABLE
        transform, and comparing the two passes' answers is only meaningful with
        both flags matched.

        Returns :class:`DecidedFaces`, with ``rows`` the wire and the five mask
        fields the legality each row was drawn under -- the arrays PPO has to
        store, because its loss replay rescores from the stored mask and never
        recomputes it.

        Raises:
            NotImplementedError: if ``--approx-add`` gives the face more than
                the three contraction slots. Slots 3 / 4 sit on tensors that
                DO depend on sibling faces; answering them from a per-face
                composition would be the staleness this pass exists to remove,
                one level down.
            RuntimeError: if two of ``vertex``'s faces share a
                ``face_transforms`` key. ``faces_of``'s key omits the central
                variable, so a multi-output equation whose output variables
                share an ``(in_edge, out_edge)`` pair would have ONE row
                configure TWO faces. Measured zero on TLM and nn256 under
                minimum Markowitz (finding 77: zero eliminated equations have
                more than one output variable at all), which is why this
                raises instead of widening the key -- a silent collision is
                the one outcome that must not happen.
        """
        from graphax.core import face_specs_of
        from alphagrad.approx.env import FACE_SLOTS, wire_slots

        F, N = self.max_faces, self.max_axes
        S = wire_slots()
        if S != FACE_SLOTS:
            raise NotImplementedError(
                f"decide_vertex_faces answers the {FACE_SLOTS} CONTRACTION "
                f"slots, and --approx-add gives the face {S}. Slots "
                f"{FACE_SLOTS}.. sit on the OLD EDGE (graphax res:jr) and the "
                f"SUMMED EDGE (res:jres), whose tensors depend on SIBLING "
                f"faces' decisions -- an earlier face's merge writes the edge a "
                f"later face's jr reads -- so they are not a function of this "
                f"face's own operands and cannot be composed here. Use "
                f"decide_faces (one speculative elimination, every slot a "
                f"chooser) for those.")
        rows = np.full((F, S, 3), -1, np.int32)
        rows[..., 2] = 0
        sizes = np.zeros((F, S, N), np.int32)
        quant = np.zeros((F, S, NUM_FACE_QUANT_DTYPES), np.float32)
        pair = np.zeros((F, S, N, N), np.float32)
        comp = np.zeros((F, S, N), np.float32)
        nout = np.zeros((F, S), np.int32)

        ij = tk.ij
        specs = face_specs_of(ij.graph, ij.tgraph, int(vertex), ij.jaxpr)
        # ONE ROW, ONE FACE. `faces_of`'s key omits the central variable, so a
        # repeated key would have one wire row configure two faces and one mask
        # row describe two tensors. Counted before anything else because the
        # whole pass is indexed by `f`, which is `faces_of`'s position.
        _seen_keys: dict = {}
        for _sp in specs:
            if _sp.key in _seen_keys:
                self.stats["vertex_key_collision"] += 1
                raise RuntimeError(
                    f"vertex {int(vertex)} has two faces with the same "
                    f"face_transforms key {_sp.key}: central variables "
                    f"{_seen_keys[_sp.key]} and {_sp.central}. One wire row "
                    f"would configure both faces and one mask row would "
                    f"describe both tensors. faces_of's key omits the central "
                    f"variable; this equation has "
                    f"{len(ij.jaxpr.eqns[int(vertex) - 1].outvars)} output "
                    f"variables.")
            _seen_keys[_sp.key] = _sp.central
        n_faces = min(len(specs), F)
        specs = specs[:n_faces]

        def _skipped(f):
            return (skips is not None
                    and int(np.asarray(skips).reshape(-1)[f]) == 1)

        with _Snapshot(tk):
            self.stats["vertex_probe"] += 1
            try:
                self._decide_vertex_body(
                    ij, int(vertex), specs, draw, _skipped, N, S,
                    rows, sizes, quant, pair, comp, nout,
                    approx_cfg=approx_cfg)
            except Exception as exc:
                # Same contract as `_probe_faces` and `decide_faces`: a
                # vertex graphax cannot trace has no legal approximation
                # either, so keep the decisions taken so far (the rest stay
                # -1 = exact) and never let a probe take the rollout down.
                #
                # The LAST failure is kept, because "the pass failed on 196
                # of 475 vertices" is not a diagnosis and the counter alone
                # cannot become one. A probe or test that sees
                # `vertex_probe_fail` non-zero can print
                # `last_vertex_error` and say WHY.
                self.stats["vertex_probe_fail"] += 1
                self.last_vertex_error = (
                    f"vertex {int(vertex)}: "
                    f"{type(exc).__name__}: {exc}")
        return DecidedFaces(rows=rows, sizes=sizes, quant=quant, pair=pair,
                            comp=comp, nout=nout, n_faces=np.int32(n_faces))

    def _decide_vertex_body(self, ij, vertex, specs, draw, skipped, N,
                            S, rows, sizes, quant, pair, comp, nout,
                            approx_cfg=None):
        """The two stages. Split out so the snapshot / arming wrapper above
        stays readable and so the ``except`` there covers exactly this."""
        from jax._src import core as _jcore
        from graphax import SKIP_FACE
        from graphax.core import (_apply_face_transform, contract_face_operands,
                                  face_config_is_approx, prepare_face_operands,
                                  _force)
        from alphagrad.approx.common.masks import slot_legality
        from alphagrad.approx.env import (
            face_entry_from_slots, make_slot_frame_hook)

        # ---- the n + m OPERAND PROBES -----------------------------------
        #
        # UNDER THE TRACE, because forcing an unevaluated ``LazyEdge`` emits jax
        # equations and they have to land in the persistent frame (the snapshot
        # truncates them). ``_force`` memoises on the edge, so this forces each
        # edge exactly once no matter how many faces read it -- which is the
        # ``n + m`` of the proof rather than ``2nm``.
        #
        # The tensors are the ones graphax hands the ``lhs`` / ``rhs`` hooks
        # VERBATIM: ``_eliminate_vertex`` sets ``pre_val = _pre_raw`` and
        # re-seeds ``post_val = _post_raw`` per face before applying them, so
        # the operand a slot hook meets depends on ONE of the two neighbours
        # and on no decision at all.
        lhs_st: dict = {}
        rhs_st: dict = {}
        with _jcore.set_current_trace(ij.trace):
            for sp in specs:
                ck = id(sp.central)
                kl, kr = (ck, id(sp.in_edge)), (ck, id(sp.out_edge))
                if kl not in lhs_st:
                    lhs_st[kl] = _force(ij.tgraph[sp.central][sp.in_edge])
                if kr not in rhs_st:
                    rhs_st[kr] = _force(ij.graph[sp.central][sp.out_edge])
        self.stats["vertex_operand_probes"] += len(lhs_st) + len(rhs_st)

        def _operands(sp):
            ck = id(sp.central)
            return (lhs_st[(ck, id(sp.in_edge))],
                    rhs_st[(ck, id(sp.out_edge))])

        # ---- STAGE 1: every face's lhs and rhs --------------------------
        #
        # OUTSIDE THE TRACE. `slot_legality` reads shapes and dims, and `draw`
        # is a policy head; neither needs the jaxpr frame, and keeping them out
        # of it is what removes `jax.ensure_compile_time_eval` from the draw.
        mask1: dict = {}
        for sp in specs:
            f = sp.f
            if skipped(f):
                continue
            lhs, rhs = _operands(sp)
            if lhs is None or rhs is None:
                # graphax `continue`s a face whose edge Jacobian forces to
                # None, so no hook of it is ever invoked: the all-zero row is
                # the right answer and it is counted rather than guessed at.
                self.stats["vertex_face_absent"] += 1
                continue
            for s, st in ((0, lhs), (1, rhs)):
                L = slot_legality(st, N)
                mask1[(f, s)] = L
                sizes[f, s] = L.sizes
                nout[f, s] = L.n_out
                pair[f, s] = L.pair.astype(np.float32)
                comp[f, s] = L.comp.astype(np.float32)
                quant[f, s] = L.quant.astype(np.float32)
                w = draw(f, s, L)
                self.stats["vertex_draw"] += 1
                if w is not None:
                    rows[f, s] = tuple(int(x) for x in w)

        # ---- the APPROX FLAG, exactly as the elimination will compute it -
        #
        # `_is_approx_cfg` gates the reconciler peel and the re-evaluation of
        # `need_contract` from the PEELED operands, so a wrong flag is a wrong
        # `new` structure. It is a property of the WHOLE face_transforms dict,
        # which is why stage 2 sits behind ALL of stage 1: the entry forms the
        # measurement will install are built here from the rows just drawn and
        # handed to graphax's own predicate.
        ft_probe: dict = {}
        _entry_unknown = False
        for sp in specs:
            if skipped(sp.f):
                ft_probe[sp.key] = SKIP_FACE
                continue
            hooks = tuple(
                None if int(rows[sp.f, s, 0]) == -1
                else make_slot_frame_hook(tuple(int(x) for x in rows[sp.f, s]))
                for s in range(S))
            if all(h is None for h in hooks):
                continue  # an all-None face installs no entry (env's gate)
            try:
                ft_probe[sp.key] = face_entry_from_slots(hooks)
            except Exception:
                # `--approx-add choose` resolves the join from a PER-FACE BIT
                # that this pass does not hold, and `resolve_join_mode` rightly
                # RAISES rather than guessing (finding 74). The bit only
                # decides between `lossy` and `lossless`, and the `lossy` arm
                # installs a `FaceJoinPolicy` on an ARMED face, which ARMS
                # `_is_approx_cfg`. So the conservative answer -- the one that
                # cannot mask from a flag the elimination will not have -- is
                # True, and it is counted so a `choose` run does not look
                # exact by accident.
                _entry_unknown = True
        if approx_cfg is not None:
            approx = bool(approx_cfg)
        elif _entry_unknown:
            approx = True
            self.stats["vertex_flag_assumed"] += 1
        else:
            approx = bool(face_config_is_approx(ft_probe))
        # THE ONE CASE THE FLAG IS NOT YET DECIDED. Under a `lossy` join an
        # ARMED face carries a `FaceJoinPolicy`, which arms the flag -- and slot
        # 2 can arm a face whose operands are both exact. The flag is then a
        # function of the stage-2 draws, which is the one circularity the
        # hoisting does not break. Under every other value `face_slot_sites`
        # installs NO policy at all (env._JOIN_SEMANTICS_OF) so nothing in
        # stage 2 can move the flag; this detects the `lossy`-and-nothing-armed
        # case and says so rather than silently masking from the wrong flag.
        _flag_undecided = (approx_cfg is None and not approx
                           and not _entry_unknown
                           and self._lossy_join_active())
        if _flag_undecided:
            self.stats["vertex_flag_undecided"] += 1

        # ---- STAGE 2: every face's `new` --------------------------------
        for sp in specs:
            f = sp.f
            if skipped(f) or (f, 0) not in mask1:
                continue
            lhs, rhs = _operands(sp)
            with _jcore.set_current_trace(ij.trace):
                # THE APPLY PATH'S OWN HOOK on the operands, so the tensors fed
                # to the contraction are the tensors the measurement will feed
                # it. Not a second copy of the decode: `make_slot_frame_hook`
                # is the object `env._face_dict_for_vertex` installs.
                a, b = lhs, rhs
                for s in (0, 1):
                    if int(rows[f, s, 0]) == -1:
                        continue
                    seen: dict = {}
                    h = make_slot_frame_hook(
                        tuple(int(x) for x in rows[f, s]), stats=seen,
                        gated=False)
                    # THROUGH GRAPHAX'S OWN SLOT WRAPPER, not by calling the
                    # hook directly: `_apply_face_transform` is what the
                    # elimination wraps every slot hook in, and it owns two
                    # behaviours a direct call does not have -- a ValueError is
                    # the documented best-effort MISS (operand returned
                    # unchanged, nothing recorded) and the result is put to
                    # `_assert_sparse_tensor_consistency`. Calling the hook
                    # bare would make a miss an exception here and a skip
                    # there, which is a mask/apply divergence of exactly the
                    # kind this pass exists to remove.
                    if s == 0:
                        a = _apply_face_transform(a, h, "lhs", int(vertex),
                                                  None)
                    else:
                        b = _apply_face_transform(b, h, "rhs", int(vertex),
                                                  None)
                    if seen.get("skipped"):
                        self.stats["vertex_self_skip"] += 1
                # ``contract_face_operands(prepare_face_operands(...))`` IS
                # ``_eliminate_vertex``'s contraction -- the same two functions,
                # in the same order, with the same flags. `post` is the
                # out-edge Jacobian and `pre` the in-edge one, which is the
                # operand order the engine uses (``_post_val @ _pre_val``).
                ops = prepare_face_operands(b, a, approx=approx)
                new_st = contract_face_operands(ops, count_ops=False).val
            self.stats["vertex_contractions"] += 1
            L = slot_legality(new_st, N)
            s = 2
            sizes[f, s] = L.sizes
            nout[f, s] = L.n_out
            pair[f, s] = L.pair.astype(np.float32)
            comp[f, s] = L.comp.astype(np.float32)
            quant[f, s] = L.quant.astype(np.float32)
            w = draw(f, s, L)
            self.stats["vertex_draw"] += 1
            if w is None:
                continue
            row = tuple(int(x) for x in w)
            rows[f, s] = row
            if _flag_undecided:
                # The flag DID move: this face is armed only by slot 2, the
                # join is lossy, so the measurement will run with
                # `_is_approx_cfg` True while this mask was computed with it
                # False. Loud, because the alternative is a stale mask.
                self.stats["vertex_flag_flip"] += 1
            # THE PASS'S OWN SELF-CHECK, the same one `decide_faces` carries: a
            # skip here means `make_slot_frame_hook` refuses, ON THE VERY
            # TENSOR the row was drawn from, a row `slot_legality` cleared --
            # i.e. the mask and the hook disagree, which is a defect in
            # `slot_legality` and not staleness.
            seen = {}
            with _jcore.set_current_trace(ij.trace):
                _apply_face_transform(
                    new_st, make_slot_frame_hook(row, stats=seen, gated=False),
                    "res", int(vertex), None, log_slot="res:new")
            if seen.get("skipped"):
                self.stats["vertex_self_skip"] += 1

    @staticmethod
    def _lossy_join_active() -> bool:
        """Does the running ``--approx-add`` install a ``FaceJoinPolicy``?

        Asked of the entry builder rather than of the flag's name, so a future
        value cannot change the answer without changing this one too.
        """
        from graphax.sparse.ops.join import FaceJoinPolicy
        from alphagrad.approx.env import face_entry_from_slots, wire_slots

        def _probe(st):
            return st
        try:
            entry = face_entry_from_slots(
                tuple(_probe for _ in range(wire_slots())))
        except Exception:
            return True  # cannot rule it out -> say so
        from graphax.core import _iter_face_hooks
        return any(isinstance(h, FaceJoinPolicy)
                   for h in _iter_face_hooks(entry))

    def vertex_face_decisions(self, order, specs, n, vertex, draw, *,
                              skips=None, face_rows_hist=None,
                              face_skips_hist=None, approx_cfg=None,
                              hist_key=None):
        """:meth:`decide_vertex_faces` against the PREFIX tokenizer of step
        ``n`` -- the :meth:`face_slot_decisions` of the structural pass.

        NOT MEMOISED, for :meth:`face_slot_decisions`' reason: the result
        depends on ``draw``, i.e. on the policy and the key. A ``draw`` that
        returns ``None`` everywhere makes it a pure MASK read, and THAT answer
        is a pure function of the prefix -- which is what lets it stand in for
        :meth:`face_slot_legality`.
        """
        from alphagrad.approx.env import wire_slots

        order = np.asarray(order).reshape(-1)
        specs = np.asarray(specs)
        n, vertex = int(n), int(vertex)
        frh, fsh = self._hist(face_rows_hist, face_skips_hist)
        try:
            tk = self._tokenizer_at(order, specs, n, frh, fsh,
                                    hist_key=hist_key)
        except Exception:
            self.stats["failures"] += 1
            F, N, S = self.max_faces, self.max_axes, wire_slots()
            rows = np.full((F, S, 3), -1, np.int32)
            rows[..., 2] = 0
            return DecidedFaces(
                rows=rows, sizes=np.zeros((F, S, N), np.int32),
                quant=np.zeros((F, S, NUM_FACE_QUANT_DTYPES), np.float32),
                pair=np.zeros((F, S, N, N), np.float32),
                comp=np.zeros((F, S, N), np.float32),
                nout=np.zeros((F, S), np.int32), n_faces=np.int32(0))
        return self.decide_vertex_faces(tk, vertex, draw, skips=skips,
                                        approx_cfg=approx_cfg)

    def consume_stats(self) -> dict:
        out = dict(self.stats)
        for k in self.stats:
            self.stats[k] = 0
        return out
