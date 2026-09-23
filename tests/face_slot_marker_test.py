#!/usr/bin/env python3
"""A face slot's DECISION leaves a marker in the token stream.

Until 2026-09-13 it did not. graphax writes an ``approx`` block only from
``core._record_micro``, which runs for a literal ``Diag`` / ``Compress`` /
``Quant`` or for a CHOOSER callable that RETURNS one; a callable that returns a
TENSOR is applied and recorded by nobody. ``env.make_slot_frame_hook`` was that
third kind, so every DIAG, COMPRESS and QUANT the face head placed changed the
Jacobian and left no trace in the stream. Only SKIP -- a wire bit graphax
records itself -- was ever visible (finding 64).

The hook is a CHOOSER now: it decides and hands the action back, graphax
applies AND records it. This module pins the three things that follow.

1. An applied rule emits exactly ONE block, on the slot it was applied to. A
   rule the mask refuses, and a rule that is the identity on its operand, emit
   NOTHING -- because nothing was applied -- and are still counted skipped.
2. The decide path and the measurement builder tokenize the SAME stream for
   the same wire. That is the one-decoder principle of finding 61: a drift
   between what the head decided against and what the engine measured is the
   defect the single builder exists to remove.
3. Counting stays honest. ``applied`` is the post-apply identity test, the same
   test that decides whether a block exists, and graphax reports it back
   through the chooser's ``chosen_applied`` callback.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                               # noqa: E402
import pytest                                                    # noqa: E402

import jax                                                       # noqa: E402

VOCAB = 256          # the one resolver's answer (common.token_vocab)


# --------------------------------------------------------------------------
# the graph and the wire
# --------------------------------------------------------------------------

def _perceptron():
    """The repo's smallest multi-face target, built as the trainer builds it."""
    import jax.random as jrand
    from graphax.examples import Perceptron

    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    args = (x, y, W1, b1, W2, b2, gamma, beta)
    cj = jax.make_jaxpr(Perceptron)(*args)
    return cj.jaxpr, cj.literals, args


_ARGNUMS = (2, 3, 4, 5)


def _quant_row(dtype="bfloat16"):
    """The wire row for ``QUANT <dtype>``: the cast is legal on any slot that
    stores values, so it is the row that finds an applied rule without a
    search over geometry."""
    from graphax.sparse.micro_actions import QUANT_DTYPES
    from alphagrad.approx.env import QUANT_SENTINEL

    return (QUANT_SENTINEL, QUANT_DTYPES.index(dtype), 0)


def _exact_rows(F, S):
    return -np.ones((F, S, 3), np.int32)


def _step(jaxpr, consts, args, vertex, rows, skips):
    """``(tokenizer, tokens, [(start, split, end)])`` of ONE elimination, built
    through ``env._face_dict_for_vertex`` -- THE builder the measurement and
    ``LiveFaceStream._decided`` both use."""
    from types import SimpleNamespace

    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.env import _face_dict_for_vertex

    tk = IncrementalPathTokenizer(jaxpr, _ARGNUMS, list(consts), list(args),
                                  vocab_size=VOCAB)
    tk.base_tokens()
    keys = list(tk.ij.faces(int(vertex)))
    ft = _face_dict_for_vertex(SimpleNamespace(jaxpr=jaxpr), tk.ij,
                               int(vertex), rows, skips, keys=keys)
    toks = [int(t) for t in tk.eliminate(int(vertex), (), ft)]
    return tk, toks, tk.last_face_segments()


def _n_faces(jaxpr, consts, args, vertex):
    from graphax import IncrementalPathTokenizer

    tk = IncrementalPathTokenizer(jaxpr, _ARGNUMS, list(consts), list(args),
                                  vocab_size=VOCAB)
    tk.base_tokens()
    return len(list(tk.ij.faces(int(vertex))))


def _first_applied_site(jaxpr, consts, args, row, slots=(0, 1)):
    """``(vertex, face)`` of the first site where ``row`` in ``slots`` really
    applies, i.e. really emits a block. Skipped when the target offers none --
    a test that silently passes on a vertex where nothing applied would be
    asserting nothing. A QUANT row sits on BOTH contraction slots (owner
    ruling 2026-09-23: a face Quant is two-sided)."""
    from alphagrad.approx.env import wire_slots

    S = wire_slots()
    for v in range(1, len(jaxpr.eqns) + 1):
        try:
            nf = _n_faces(jaxpr, consts, args, v)
        except Exception:
            continue
        for f in range(nf):
            rows = _exact_rows(max(nf, 1), S)
            for s in slots:
                rows[f, s] = row
            skips = np.zeros((max(nf, 1),), np.int32)
            try:
                _tk, _toks, segs = _step(jaxpr, consts, args, v, rows, skips)
            except Exception:
                continue
            if f < len(segs) and segs[f][2] > segs[f][1]:
                return v, f, rows, skips
    return None


# --------------------------------------------------------------------------
# 1. one applied rule, one block; a refused rule, no block
# --------------------------------------------------------------------------

def test_an_applied_slot_rule_emits_exactly_one_approximation_block():
    """The decision reaches the stream: one block on the face, with the type
    and the arguments the wire asked for. A face QUANT is two-sided, so it is
    recorded on lhs and rhs and read once as the face's ``~ <dtype>``."""
    from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX

    jaxpr, consts, args = _perceptron()
    site = _first_applied_site(jaxpr, consts, args, _quant_row())
    assert site is not None, (
        "no vertex of this graph applied a QUANT on both contraction slots, "
        "so there is no applied rule to read a block off")
    vertex, f, rows, skips = site

    tk, toks, segs = _step(jaxpr, consts, args, vertex, rows, skips)
    recs = list(tk.ij.step_faces(0))[f].approx
    assert [(r.atype, r.slot) for r in recs] == [
        ("QUANT", "lhs"), ("QUANT", "rhs")], (
        f"vertex {vertex} face {f} recorded {len(recs)} approximations for "
        f"one two-sided wire row: {[(r.atype, r.slot) for r in recs]}")
    assert all(r.params == {"dtype": "bfloat16"} for r in recs)

    start, split, end = segs[f]
    assert end > split, "the applied rule left no approximation part"
    assert tk.decode(toks[split:end]).startswith(
        f"approx~{QUANT_DTYPE_INDEX['bfloat16']}^^")

    # Every OTHER face of the same vertex stays silent: the row was placed on
    # one face and a block on a second one would mean a rule leaked sideways.
    for g, (_s, sp, e) in enumerate(segs):
        if g != f:
            assert sp == e, f"face {g} emitted an approximation nobody asked for"


def test_a_row_the_decoder_drops_emits_no_block_and_counts_as_skipped():
    """A row that names no dim of this slot is a MISS: nothing is applied, so
    nothing is marked, and the miss is counted rather than forgotten."""
    from alphagrad.approx.env import wire_slots
    from alphagrad.approx.common.masks import arm_face_counts, disarm_face_counts
    from alphagrad.approx.env import _PER_FACE_STATS

    jaxpr, consts, args = _perceptron()
    vertex = next(v for v in range(1, len(jaxpr.eqns) + 1)
                  if _n_faces(jaxpr, consts, args, v) >= 1)
    nf = _n_faces(jaxpr, consts, args, vertex)
    S = wire_slots()
    rows = _exact_rows(nf, S)
    # A DIAG whose factor divides nothing on this operand: the decoder drops
    # the row, so the slot decodes to no rule at all.
    rows[0, 0] = (0, 0, 7)
    skips = np.zeros((nf,), np.int32)

    _PER_FACE_STATS.clear()
    arm_face_counts()
    try:
        tk, _toks, segs = _step(jaxpr, consts, args, vertex, rows, skips)
    finally:
        disarm_face_counts()
    stats = dict(_PER_FACE_STATS)
    _PER_FACE_STATS.clear()

    assert all(sp == e for _s, sp, e in segs), (
        f"a row the decoder dropped still emitted a block: {segs}")
    assert not any(fr.approx for fr in tk.ij.step_faces(0))
    assert stats.get("skipped", 0) >= 1, stats
    assert stats.get("skipped_diag", 0) >= 1, stats
    assert stats.get("applied", 0) == 0, stats


def test_an_identity_decision_counts_as_a_no_op_skip_and_emits_no_block():
    """Quantizing to the dtype the operand already has changes nothing. The
    engine hands the operand back unchanged, so there is no block -- and the
    counter must say ``skipped_quant_noop``, not ``applied``. The two
    statements are the SAME statement, which is why graphax reports the outcome
    back to the chooser instead of letting it guess."""
    from alphagrad.approx.env import wire_slots, _PER_FACE_STATS
    from alphagrad.approx.common.masks import arm_face_counts, disarm_face_counts

    jaxpr, consts, args = _perceptron()
    site = _first_applied_site(jaxpr, consts, args, _quant_row())
    assert site is not None
    vertex, f, _rows, _skips = site

    nf = _n_faces(jaxpr, consts, args, vertex)
    S = wire_slots()
    rows = _exact_rows(nf, S)
    rows[f, 0] = rows[f, 1] = _quant_row("float32")   # the dtype it carries
    skips = np.zeros((nf,), np.int32)

    _PER_FACE_STATS.clear()
    arm_face_counts()
    try:
        tk, _toks, segs = _step(jaxpr, consts, args, vertex, rows, skips)
    finally:
        disarm_face_counts()
    stats = dict(_PER_FACE_STATS)
    _PER_FACE_STATS.clear()

    assert all(sp == e for _s, sp, e in segs), (
        f"an identity quant emitted an approximation block: {segs}")
    assert not any(fr.approx for fr in tk.ij.step_faces(0))
    assert stats.get("applied", 0) == 0, stats
    assert stats.get("skipped_quant_noop", 0) >= 1, stats
    assert stats.get("skipped_quant", 0) >= stats.get("skipped_quant_noop", 0)


def test_a_skipped_face_still_emits_the_slotless_skip_block():
    """The SKIP channel is unchanged by the move to a chooser: it is a wire bit
    graphax records itself, and it keeps its own slot-less form."""
    from alphagrad.approx.env import wire_slots

    jaxpr, consts, args = _perceptron()
    vertex = next(v for v in range(1, len(jaxpr.eqns) + 1)
                  if _n_faces(jaxpr, consts, args, v) >= 1)
    nf = _n_faces(jaxpr, consts, args, vertex)
    rows = _exact_rows(nf, wire_slots())
    skips = np.zeros((nf,), np.int32)
    skips[0] = 1

    tk, toks, segs = _step(jaxpr, consts, args, vertex, rows, skips)
    start, split, end = segs[0]
    assert end > split
    assert tk.decode(toks[split:end]).startswith("approxSKIP")


# --------------------------------------------------------------------------
# 2. the decide path and the measurement builder agree, token for token
# --------------------------------------------------------------------------

def test_the_decide_path_and_the_measurement_builder_emit_the_same_stream():
    """ONE DECODER (finding 61). ``LiveFaceStream`` replays the decided prefix
    on its own persistent tokenizer; the measurement builds a fresh one. For
    the same wire the two must produce the same tokens -- including the
    approximation blocks, which is the part that did not exist before.
    """
    from alphagrad.approx.env import wire_slots
    from alphagrad.approx.live_faces import LiveFaceStream, _Snapshot

    jaxpr, consts, args = _perceptron()
    site = _first_applied_site(jaxpr, consts, args, _quant_row())
    assert site is not None
    vertex, f, rows, skips = site
    nf = _n_faces(jaxpr, consts, args, vertex)

    ref_tk, ref_toks, ref_segs = _step(jaxpr, consts, args, vertex, rows, skips)
    assert ref_segs[f][2] > ref_segs[f][1], "the reference emitted no block"

    V = len(jaxpr.eqns)
    lfs = LiveFaceStream(jaxpr, _ARGNUMS, consts, args, vocab=VOCAB,
                         max_faces=max(nf, 8), max_axes=8, window=8192)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, 8, 3), np.int32)

    tk = lfs._tokenizer_at(order, specs, 0, None, None)
    keys, ft = lfs._decided(tk, vertex, rows, skips, nf)
    with _Snapshot(tk):
        live_toks = [int(t) for t in tk.eliminate(int(vertex), (), ft)]
        live_segs = tk.last_face_segments()

    assert live_toks == ref_toks, (
        "the decide path and the measurement builder tokenized the same wire "
        "differently")
    assert live_segs == ref_segs


def test_the_stored_chunk_of_the_next_face_opens_on_the_decided_rules_block():
    """The stream's own carrier. ``chunk`` hands face ``f`` its predecessor's
    approximation echo, and for a RULE that echo is now non-empty -- which is
    what the encoder reads the decision off."""
    from alphagrad.approx.env import wire_slots
    from alphagrad.approx.live_faces import LiveFaceStream

    jaxpr, consts, args = _perceptron()
    row = _quant_row()
    S = wire_slots()
    V = len(jaxpr.eqns)

    chosen = None
    for v in range(1, V + 1):
        try:
            nf = _n_faces(jaxpr, consts, args, v)
        except Exception:
            continue
        if nf < 2:
            continue
        rows = _exact_rows(max(nf, 8), S)
        rows[0, 0] = rows[0, 1] = row
        skips = np.zeros((max(nf, 8),), np.int32)
        try:
            _tk, _toks, segs = _step(jaxpr, consts, args, v, rows, skips)
        except Exception:
            continue
        if segs[0][2] > segs[0][1]:
            chosen = (v, nf, rows, skips, segs)
            break
    assert chosen is not None, (
        "no vertex with two or more faces applied a QUANT on face 0")
    vertex, nf, rows, skips, ref_segs = chosen
    _ref_tk, ref_toks, _ = _step(jaxpr, consts, args, vertex, rows, skips)

    lfs = LiveFaceStream(jaxpr, _ARGNUMS, consts, args, vocab=VOCAB,
                         max_faces=max(nf, 8), max_axes=8, window=8192)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, 8, 3), np.int32)
    vspecs = -np.ones((8, 3), np.int32)
    # `chunk` returns FIVE values: the equation-id buffer was removed on
    # 2026-09-13 with the palimpsa relational forget gate it fed.
    tok, cnt, n_f, _ends, head = lfs.chunk(
        order, specs, 0, vertex, vspecs, rows, skips, 1)
    cnt, head = int(cnt), int(head)

    assert cnt > 0, f"the stream fell to its soft-failure path: {dict(lfs.stats)}"
    _s0, split0, end0 = ref_segs[0]
    assert head == end0 - split0, (
        f"face 1's chunk reports a {head}-token echo against face 0's "
        f"{end0 - split0}-token approximation block")
    assert [int(t) for t in tok[:head]] == ref_toks[split0:end0]
