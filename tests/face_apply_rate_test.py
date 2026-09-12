"""Apply rate equals request rate ON THE REAL APPLY PATH (dsnn-3qm.59, D3).

``tests/hierarchical_legality_test.py::test_tlm_sampled_action_zero_rejection``
reads the legality arrays out of ``LiveFaceStream.face_slot_legality``, hands
those SAME arrays to ``UnifiedFacePolicy.sample_face`` as the mask, and then
asserts membership in them. No hook runs and no tensor is touched, so its
``applied`` count equals its ``requested`` count by construction. That test
states deliverable 1 twice; it cannot state deliverable 3.

This test drives every sampled wire row through the path the MEASUREMENT uses
-- ``env._face_dict_for_vertex`` -> ``env.make_slot_frame_hook`` ->
``masks.make_live_masked_hook`` -- inside ``masks.arm_face_counts()``, and then
reads the engine's own counters. A rejection is ``skipped_<kind>`` and the
assertion is that it is zero.

TWO ARMS, BECAUSE TWO DEFECTS. ``_walk_one_graph`` puts the mask and the apply
on ONE tokenizer, so it states the per-slot legality claim alone. ``_walk``
keeps the production pair of graphs (ppo.py rides the stream tokenizer for the
mask and a second ``IncrementalJaxpr`` for the face keys), so it states that
claim PLUS the duplication. Mixing them could not name either: the duplicated
graph rejects Reduce rows on its own, and that would hide whether the legality
itself is sound.

AGGREGATED OVER SEEDS. A rejection appears on some seeds and not others, so a
per-seed strict xfail XPASSes on the clean ones and fails for the wrong reason.
Each claim is ONE test summing all ``N_SAMPLES`` seeds.

Targets: nn256 (VmappedNeuralNetwork on mnist) and TLM, five random samples
each, minimum Markowitz degree order.

The idempotent re-request is NOT a rejection and not a win either: the hook
applies it and nothing changes (masks.rule_is_idempotent_noop). It lands in
``applied_<kind>``, so the test reports the honest split beside the counts.

THE MASK IS DYNAMIC SINCE #75 AND EXACT PER VERTEX SINCE #77 (2026-09-11). The
four cases that used to be STRICT xfails naming dsnn-3qm.59 fault 2 -- `res:new`
with the operands armed on TLM and on nn256, `res:jr` and `res:jres` with the
contraction slots armed -- are ASSERTIONS, and they are reached by TWO different
passes on purpose:

* the two ``res:new`` claims go through ``LiveFaceStream.decide_vertex_faces``
  (#77): NO speculative elimination, ``n`` in-edge forces + ``m`` out-edge
  forces + ``n*m`` calls of ``graphax.contract_face_operands`` -- the function
  ``_eliminate_vertex`` itself calls -- with the operand rows drawn first and
  applied through the apply path's own hook. Sound because the face ->
  written-edge map of one vertex elimination is a BIJECTION, so no face's
  operands depend on another face's decision;
* the two learned-join claims go through ``LiveFaceStream.decide_faces`` (#75):
  ONE speculative elimination per vertex with every slot a graphax chooser.
  ``res:jr`` and ``res:jres`` DO depend on sibling faces -- an earlier face's
  merge writes the edge a later face's ``jr`` reads -- so the bijection says
  nothing about them and the per-vertex composition refuses to answer.

The five that remain xfailing are the DUPLICATED GRAPH, a separate open defect
neither pass can touch because the mask and the apply run on two graphs there.

ONE HEAD CALL PER (face, slot), not one per face, because slot ``s``'s mask does
not exist until slots before it have been decided and applied. That changes no
draw, and the draw is JITTED -- both pinned by tests at the bottom of this file.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from graphax import inline_call_primitives
from graphax.incremental import IncrementalJaxpr
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                   # pragma: no cover
    from jax._src.core import ClosedJaxpr

import alphagrad.approx.env as envmod
from alphagrad.approx.common import get_args, get_fn
from alphagrad.approx.common.masks import (
    arm_face_counts, disarm_face_counts, rule_is_idempotent_noop)
from alphagrad.approx.common.order import markowitz_order
from alphagrad.approx.env import FACE_SLOTS, MAX_FACES, _face_dict_for_vertex
from alphagrad.approx.heads import AXIS_TAG_BITS, AxisTokenFeatures, \
    precompute_factor_tables
from alphagrad.approx.live_faces import LiveFaceStream, _SLOT_SITES
from alphagrad.approx.unified_face_head import (
    OP_BLOCKDIAG, OP_NONE, OP_QUANT, OP_REDUCE)
from alphagrad.approx.unified_face_policy import UnifiedFacePolicy
from alphagrad.elimrl.baselines import tlm_target

N_AX = 8
N_SAMPLES = 5
KINDS = ("diag", "compress", "quant")
_OP_KIND = {OP_BLOCKDIAG: "diag", OP_REDUCE: "compress", OP_QUANT: "quant"}


def _closed(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    return cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)


# ==========================================================================
# nn256 BARELY EXERCISES THE FACE ADD, so a green nn256 run is NOT evidence
# that the ADD works. Measured, finding 73:
#
#   jobs 64653 / 64659: with the `new` slot armed alone AND with all three
#     armed, graphax's `res:jl` and `res:jr` were invoked ZERO times on every
#     seed -- NO ARMED FACE on this target is a merge face.
#   job 64803: the legality probe, which visits every face and not only the
#     armed ones, recorded a tensor at `res:jr` on 1 of 33 faces. So exactly one
#     merge face exists; it is just never the one an approximation lands on.
#
# A face merge needs the contraction to land on an edge that ALREADY exists, and
# VmappedNeuralNetwork on mnist almost never does. Every nn256 case below
# therefore tests the per-slot CONTRACTION legality only. The ADD, and the two
# learned join slots, are measured on TLM.
# ==========================================================================
@pytest.fixture(scope="module")
def nn256():
    _, args_key, _ = jrand.split(jrand.PRNGKey(7), 3)
    fn = get_fn("VmappedNeuralNetwork")
    xs = get_args("VmappedNeuralNetwork", args_key, dataset="mnist")
    cj = _closed(fn, list(xs))
    return cj.jaxpr, list(cj.literals), list(xs), tuple(range(len(xs)))


@pytest.fixture(scope="module")
def tlm():
    os.environ["ALPHAGRAD_TLM_SEQ"] = "16"
    os.environ["ALPHAGRAD_TLM_DMODEL"] = "64"
    os.environ["ALPHAGRAD_TLM_VOCAB"] = "256"
    fn, args, argnums = tlm_target(seq=16, dmodel=64, vocab=256)
    cj = _closed(fn, list(args))
    return (cj.jaxpr, list(cj.literals), list(args),
            tuple(range(len(args))))


def _valid_vertices(jaxpr, args, consts, argnums):
    _, _, _, vo = envmod._build_graph(jaxpr, args, consts, argnums)
    return tuple(i for i, eqn in enumerate(jaxpr.eqns, 1)
                 if eqn.outvars[0] not in jaxpr.outvars or i in vo)


def _policy(F):
    return (UnifiedFacePolicy(32, num_heads=2, max_faces=F,
                              key=jrand.PRNGKey(0), allow_skip=False),
            precompute_factor_tables(64))


def _features():
    """ONE AxisTokenFeatures. ``sample_face`` builds the per-slot list itself
    (``_slot_inputs`` -> ``_face_feats_1``) from ``face_sizes_f``, so handing it
    a list here is what the rank check rejects -- the per-slot sizes ride
    ``face_sizes_f`` (S, N), not the features."""
    sz = jnp.ones((N_AX,), jnp.int32) * 2
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((N_AX, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((N_AX,), jnp.int32),
        valid_mask=jnp.ones((N_AX,), jnp.float32))


def _jit_draw(pol, feats, ctx, tables):
    """ONE jitted draw, reused by every walk.

    A DYNAMIC mask needs one head call per (face, slot) instead of one per face,
    and an EAGER ``sample_face`` costs 35 ms against 0.28 ms jitted (probe
    t75_cost) -- 78 ms eager inside the decide pass's trace escape. Left eager
    this file would take hours and would mostly be measuring the harness. The
    rollout jits it too (``UnifiedPolicy._face_loop`` runs inside the jitted
    rollout step), so jitting here is the faithful choice as well as the fast
    one, and ``test_the_jitted_draw_is_the_eager_draw`` pins that it changes no
    draw.
    """
    # WARM THE HARDWARE SCAN EAGERLY FIRST. `sample_face` calls
    # `graphax.sparse.micro_actions.quant_hardware_masks()` when
    # `quant_legality_mask` is None, and that function MEMOISES a jnp array --
    # its own docstring says "Warm this once at build (eagerly, before any jit)
    # so the jnp.dot probes never run under trace." If the first caller in a
    # process is under jit, the cache holds a DynamicJaxprTracer and the next
    # EAGER caller dies with UnexpectedTracerError (measured: job 64889's
    # episode probe, leak created at micro_actions.py:299).
    from graphax.sparse.micro_actions import quant_hardware_masks
    quant_hardware_masks()

    @jax.jit
    def _d(key, f, pr, cp, sz, qt):
        sk, row, _lp, _e, _ar, _sp, _od = pol.sample_face(
            feats, tables, key, f, pr, cp, jnp.asarray(1.0),
            face_context=ctx, face_sizes_f=sz, face_quant_f=qt)
        return (sk, row["op_type"], row["i"], row["j"], row["quant_dtype"])
    return _d


def _diag_wire(i, j, n_out):
    """``(bi1, bi2)`` or None, MIRRORING ``env.micro_actions_to_rule_specs_jax``.

    The wire's bi1 is the OUT-side relative position and bi2 the PRIMAL-side
    one, so a pair the head emits as (primal, out) has to be SWAPPED, and a
    same-side pair cannot be expressed at all. ``masks.slot_legality`` fills
    ``pair[i][j]`` for both orders of a legal pair (it builds the Diag with the
    out index first either way), so an encoder that does not swap asks the
    engine for a row it cannot decode -- measured on TLM as 43 of 68 Diag
    requests "rejected" when the fault was here.
    """
    i, j, n_out = int(i), int(j), int(n_out)
    out_i, out_j = i < n_out, j < n_out
    if out_i == out_j:
        return None                      # same side: inexpressible
    rel_i = i if out_i else i - n_out
    rel_j = j if out_j else j - n_out
    return (rel_i, rel_j) if out_i else (rel_j, rel_i)


def _row_to_wire(op, i, j, axis, dtype_idx, n_out):
    """The (op, fields) the head chose -> the ``[bi1, bi2, factor]`` wire row.
    ``None`` when the head's pair has no wire form, which is NOT a rejection:
    the engine marks the row unused too (``diag_used & ~same_side``)."""
    if op == OP_BLOCKDIAG:
        bi = _diag_wire(i, j, n_out)
        return None if bi is None else (bi[0], bi[1], -1)
    if op == OP_REDUCE:
        return (int(envmod.COMPRESS_SENTINEL), int(axis), 0)
    if op == OP_QUANT:
        return (int(envmod.QUANT_SENTINEL), int(dtype_idx), 0)
    return (-1, -1, 0)



def _walk_one_graph(target, seed, slots_on=None, pass_="vertex"):
    """One sampled plan with the mask and the apply ON THE SAME GRAPH.

    ``_walk`` keeps the production pair of graphs; this owns a single
    ``IncrementalPathTokenizer``, decides every face slot on it through
    ``LiveFaceStream.decide_faces`` (whose ``_Snapshot`` undoes the speculative
    elimination) and then runs the real elimination on that same graph.

    WHY BOTH. The two arrangements fail for different reasons, and a test that
    mixed them could not name either: the duplicated graph contributes its own
    rejections (finding 71: 4 of 9 Reduce, plus a face-key-list disagreement on
    3 of 95 steps), which would mask whether the per-slot legality itself is
    sound. This arm is the legality claim; ``_walk`` is the legality claim PLUS
    the duplication.

    THE MASK IS DYNAMIC (#75, .59 fault 2). Every slot is drawn against
    ``masks.slot_legality`` read off the LIVE tensor at that slot's own graphax
    site, inside the one speculative elimination that also APPLIES each decided
    row as it is taken -- so ``res:new``'s mask describes the contraction of the
    ALREADY-APPROXIMATED operands, ``res:jr``'s the old edge as earlier faces'
    merges left it, and ``res:jres``'s the sum of this face's own approximated
    addends. It used to read all of them from one recording probe in which no
    decision had been made, which is what made the three strict xfails below
    strict xfails.

    ONE HEAD CALL PER (face, slot) rather than one per face, because a slot's
    mask does not exist until the slots before it have been decided. That
    changes NO draw: ``UnifiedFaceHead.sample`` gives slot ``s`` its own key
    slice and its own logit block and conditions it on nothing but slot ``s``'s
    own masks, so ``S`` calls with progressively filled mask rows draw exactly
    what one call with all of them draws -- pinned by
    ``test_the_per_slot_draws_equal_one_joint_draw``.

    THE MASK IS THE EXACT PER-VERTEX ONE SINCE #77. ``decide_vertex_faces``
    runs NO speculative elimination: it forces the ``n`` in-edge and ``m``
    out-edge Jacobians ONCE each, applies the drawn operand rows through the
    apply path's own hook, and composes each face's ``res:new`` structure with
    ``graphax.contract_face_operands`` -- THE FUNCTION ``_eliminate_vertex``
    ITSELF CALLS. ``pass_="decide"`` selects finding 75's chooser pass (one
    speculative elimination per vertex) instead, which is what the
    learned-join-slot arm still needs and what
    ``test_the_two_passes_draw_the_same_rows`` compares against.
    """
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.env import face_slot_sites

    jaxpr, consts, args, argnums = target
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    F = MAX_FACES

    lf = LiveFaceStream(jaxpr, argnums, consts, args,
                        vocab=512, max_faces=F, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    config = SimpleNamespace(jaxpr=jaxpr)
    pol, tables = _policy(F)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    draw_jit = _jit_draw(pol, feats, ctx, tables)
    # THE TOPOLOGY'S WIDTH, not a restated FACE_SLOTS. `face_slot_sites()`
    # reports as many slots as --approx-add has (three under the default
    # lossless, four under learned1, five under learned2), so the mask arrays
    # are sized from it and this walk is correct at every width. The prefix
    # assertion is `face_driver`'s: the contraction slots must be the first
    # three rows, or narrowing to them would hand the head other tensors' masks.
    sites = face_slot_sites()
    S_ALL = len(sites)
    assert tuple(x[0] for x in sites[:FACE_SLOTS]) == (
        "lhs", "rhs", "res:new"), sites

    envmod._PER_FACE_STATS.clear()
    requested = {k: 0 for k in KINDS}

    for n in range(len(order)):
        v = int(order[n])
        keys = list(tk.ij.faces(v))
        # The mask rows fill in AS THE DECISIONS LAND: row s is written by the
        # chooser at slot s's own site, and the head call for slot s reads this
        # array. Rows past s are still zero, which cannot reach slot s's draw.
        sizes = np.zeros((F, S_ALL, N_AX), np.int32)
        pair = np.zeros((F, S_ALL, N_AX, N_AX), np.float32)
        comp = np.zeros((F, S_ALL, N_AX), np.float32)
        quant = np.zeros((F, S_ALL, 2), np.float32)
        nout = np.zeros((F, S_ALL), np.int32)
        skips = np.zeros((F,), np.int32)

        def _draw(f, s, L, _n=n):
            sizes[f, s] = L.sizes
            nout[f, s] = L.n_out
            pair[f, s] = L.pair.astype(np.float32)
            comp[f, s] = L.comp.astype(np.float32)
            quant[f, s] = L.quant.astype(np.float32)
            if s >= FACE_SLOTS:
                # A LEARNED join slot. This walk is the CONTRACTION claim and
                # the policy has per-slot features for three slots only
                # (`_face_masks` loops `range(FACE_SLOTS)`), so the mask row is
                # recorded above -- which is what proves it CAN be -- and
                # nothing is drawn. `_walk_join_slots` is the arm that draws it.
                return None
            if slots_on is not None and _SLOT_SITES[s] not in slots_on:
                return None
            key = jrand.PRNGKey(seed * 1000003 + _n * 97 + f)
            # SLICED TO THE CONTRACTION BAND, like face_driver does: the policy
            # builds per-slot features for the three contraction slots only.
            _skip, op_t, ii, jj, dt = draw_jit(
                key, f,
                jnp.asarray(pair[f][:FACE_SLOTS]),
                jnp.asarray(comp[f][:FACE_SLOTS]),
                jnp.asarray(sizes[f][:FACE_SLOTS]),
                jnp.asarray(quant[f][:FACE_SLOTS]))
            op = int(np.asarray(op_t)[s])
            if op == OP_NONE:
                return None
            ii, jj, dt = np.asarray(ii), np.asarray(jj), np.asarray(dt)
            w = _row_to_wire(op, ii[s], jj[s], ii[s], dt[s], L.n_out)
            if w is None:
                return None
            requested[_OP_KIND[op]] += 1
            return w

        if pass_ == "vertex":
            dec = lf.decide_vertex_faces(tk, v, _draw, skips=skips)
        else:
            dec = lf.decide_faces(tk, v, keys, _draw, skips=skips)
        per_face = _face_dict_for_vertex(config, tk.ij, v, dec.rows, skips)
        arm_face_counts()
        try:
            tk.ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()

    out = dict(envmod._PER_FACE_STATS)
    out.update({f"lf_{k}": v for k, v in lf.consume_stats().items()
                if (k.startswith("decide") or k.startswith("vertex")) and v})
    envmod._PER_FACE_STATS.clear()
    return requested, out


def _totals(walk, target, label, slots_on):
    """Sum one walk over every seed. AGGREGATED ON PURPOSE: a rejection shows
    up on some seeds and not others, so a per-seed strict xfail would XPASS on
    the clean ones and fail for the wrong reason."""
    req = {k: 0 for k in KINDS}
    got: dict = {}
    for seed in range(N_SAMPLES):
        r, st = walk(target, seed, slots_on)
        for k in KINDS:
            req[k] += r[k]
        for k, val in st.items():
            got[k] = got.get(k, 0) + int(val)
    assert sum(req.values()) > 0, (
        f"{label}: the head requested nothing, so the test proved nothing")
    return req, got


def _assert_no_rejection(walk, target, label, slots_on=None):
    req, stats = _totals(walk, target, label, slots_on)
    for kind in KINDS:
        skipped = int(stats.get(f"skipped_{kind}", 0))
        assert skipped == 0, (
            f"{label}: {skipped} {kind} rows the mask cleared were REJECTED "
            f"at apply time over {N_SAMPLES} seeds "
            f"({int(stats.get(f'applied_{kind}', 0))} applied, {req[kind]} "
            f"requested). stats={stats}")
    assert int(stats.get("skipped", 0)) == 0, stats
    # THE DECIDE PASS'S OWN SELF-CHECK. `decide_faces` applies each decided row
    # through `env.make_slot_frame_hook` -- the apply path's own hook -- on the
    # tensor the row was drawn from, with a LOCAL stats dict. A skip counted
    # there means the mask cleared a row the hook refuses ON THAT VERY TENSOR,
    # i.e. a defect in `slot_legality` rather than staleness. It is a different
    # claim from the engine counters above and it is checked separately.
    assert int(stats.get("lf_decide_self_skip", 0)) == 0, (
        f"{label}: the decide pass's own apply refused "
        f"{stats['lf_decide_self_skip']} rows it had just cleared")
    assert int(stats.get("lf_decide_multi_site", 0)) == 0, stats
    # #77's pass carries the same self-check, and two more: a face-key collision
    # RAISES (so a non-zero count could only mean the raise was caught) and a
    # `lossy` join armed only by slot 2 would have been masked under the wrong
    # `_is_approx_cfg`.
    assert int(stats.get("lf_vertex_self_skip", 0)) == 0, (
        f"{label}: the vertex pass's own apply refused "
        f"{stats['lf_vertex_self_skip']} rows it had just cleared")
    assert int(stats.get("lf_vertex_key_collision", 0)) == 0, stats
    assert int(stats.get("lf_vertex_flag_flip", 0)) == 0, stats
    assert int(stats.get("lf_vertex_probe_fail", 0)) == 0, (
        f"{label}: the vertex pass failed on "
        f"{stats['lf_vertex_probe_fail']} vertices, so its mask rows are "
        f"whatever it had taken before the failure")


def _walk(target, seed, slots_on=None):
    """One sampled plan on the PRODUCTION pair of graphs: the decisions are
    taken on the stream's tokenizer and the apply runs on a separate
    ``IncrementalJaxpr`` advanced with the same wire rows.

    The mask is DYNAMIC here too (#75), so whatever this arm still rejects is
    the DUPLICATION and nothing else -- which is the point of keeping the two
    arms apart.
    """
    from alphagrad.approx.env import face_slot_sites

    jaxpr, consts, args, argnums = target
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    total_v = len(jaxpr.eqns)
    specs = np.zeros((total_v, 1, 3), np.int32)

    lf = LiveFaceStream(jaxpr, argnums, consts, args,
                        vocab=512, max_faces=MAX_FACES, max_axes=N_AX)
    ij = IncrementalJaxpr(jaxpr, argnums, list(consts), list(args),
                          track_faces=False)
    config = SimpleNamespace(jaxpr=jaxpr)
    pol, tables = _policy(MAX_FACES)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    draw_jit = _jit_draw(pol, feats, ctx, tables)
    S_ALL = len(face_slot_sites())

    rows_hist = np.full((total_v, MAX_FACES, S_ALL, 3), -1, np.int32)
    rows_hist[..., 2] = 0
    skips_hist = np.zeros((total_v, MAX_FACES), np.int32)

    envmod._PER_FACE_STATS.clear()
    requested = {k: 0 for k in KINDS}

    for n in range(len(order)):
        v = int(order[n])
        F = MAX_FACES
        sizes = np.zeros((F, S_ALL, N_AX), np.int32)
        pair = np.zeros((F, S_ALL, N_AX, N_AX), np.float32)
        comp = np.zeros((F, S_ALL, N_AX), np.float32)
        quant = np.zeros((F, S_ALL, 2), np.float32)
        nout = np.zeros((F, S_ALL), np.int32)

        def _draw(f, s, L, _n=n):
            sizes[f, s] = L.sizes
            nout[f, s] = L.n_out
            pair[f, s] = L.pair.astype(np.float32)
            comp[f, s] = L.comp.astype(np.float32)
            quant[f, s] = L.quant.astype(np.float32)
            if s >= FACE_SLOTS:
                # A LEARNED join slot. This walk is the CONTRACTION claim and
                # the policy has per-slot features for three slots only
                # (`_face_masks` loops `range(FACE_SLOTS)`), so the mask row is
                # recorded above -- which is what proves it CAN be -- and
                # nothing is drawn. `_walk_join_slots` is the arm that draws it.
                return None
            if slots_on is not None and _SLOT_SITES[s] not in slots_on:
                return None
            key = jrand.PRNGKey(seed * 1000003 + _n * 97 + f)
            _skip, op_t, ii, jj, dt = draw_jit(
                key, f,
                jnp.asarray(pair[f][:FACE_SLOTS]),
                jnp.asarray(comp[f][:FACE_SLOTS]),
                jnp.asarray(sizes[f][:FACE_SLOTS]),
                jnp.asarray(quant[f][:FACE_SLOTS]))
            op = int(np.asarray(op_t)[s])
            if op == OP_NONE:
                return None
            ii, jj, dt = np.asarray(ii), np.asarray(jj), np.asarray(dt)
            w = _row_to_wire(op, ii[s], jj[s], ii[s], dt[s], L.n_out)
            if w is None:
                return None
            requested[_OP_KIND[op]] += 1
            return w

        dec = lf.face_slot_decisions(order, specs, n, v, _draw,
                                     skips=skips_hist[n],
                                     face_rows_hist=rows_hist,
                                     face_skips_hist=skips_hist)
        rows_hist[n] = dec.rows
        per_face = _face_dict_for_vertex(config, ij, v, rows_hist[n],
                                        skips_hist[n])
        arm_face_counts()
        try:
            ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()

    out = dict(envmod._PER_FACE_STATS)
    envmod._PER_FACE_STATS.clear()
    return requested, out


# --------------------------------------------------------------------------
# ONE GRAPH: the per-slot legality claim by itself.
#
# The mask and the apply run on the SAME tokenizer, so nothing here can be
# blamed on the duplicated elimination graph. Measured on TLM, 3 seeds, the
# rejections attributed to the graphax SITE the hook ran at (job 64637, after
# the site-set fix):
#
#   slots armed    diag req/skip   reduce req/skip   quant req/skip
#   lhs                 18 / 0           31 / 0           50 / 0
#   rhs                 18 / 0           41 / 0           54 / 0
#   new                 27 / 0           39 / 0           49 / 0
#   lhs+rhs             36 / 0           72 / 0          104 / 0
#   lhs+rhs+new         60 / 6          114 / 4          152 / 0   <- fault 2
#
# THE LAST ROW IS NOW ZERO (#75): the masks are read off the live tensors in
# apply order, so `res:new`'s describes the contraction of the APPROXIMATED
# operands. Its test is an assertion below, not an xfail. The three rows above it
# were always clean and stay assertions -- they are what separated "the mask is
# wrong for its tensor" from "the mask went stale", and the answer was the second.
# --------------------------------------------------------------------------
def test_tlm_operand_slots_reject_nothing_on_one_graph(tlm):
    _assert_no_rejection(_walk_one_graph, tlm, "tlm one-graph lhs+rhs",
                         slots_on=("lhs", "rhs"))


def test_nn256_operand_slots_reject_nothing_on_one_graph(nn256):
    _assert_no_rejection(_walk_one_graph, nn256, "nn256 one-graph lhs+rhs",
                         slots_on=("lhs", "rhs"))


def test_tlm_result_slot_alone_rejects_nothing_on_one_graph(tlm):
    """FAULT 1, FIXED AT THE ROOT (#73).

    The `new` hook used to be installed at TWO graphax sites under the retired
    ``--approx-old same`` -- ``res:new`` on the fresh contraction and
    ``res:jr`` on the EXISTING OLD EDGE -- and the legality probe recorded the
    first only: ``res:jr`` refused 6 of 7 Diag invocations and 1 of 3 Reduce
    (job 64633). Finding 72 made the MASK cover both sites, which fixed the
    rejections at the cost of 32 -> 27 Diag requests. ``--approx-add`` removes
    the second site instead: the ADD is a join POLICY, so one wire row meets
    one tensor and the requests come back."""
    _assert_no_rejection(_walk_one_graph, tlm, "tlm one-graph new",
                         slots_on=("new",))


def test_nn256_result_slot_alone_rejects_nothing_on_one_graph(nn256):
    _assert_no_rejection(_walk_one_graph, nn256, "nn256 one-graph new",
                         slots_on=("new",))


def test_every_site_a_slot_hook_reaches_is_recorded_by_the_probe(tlm):
    """THE INVARIANT FAULT 1 BROKE, stated directly.

    ``face_slot_sites()`` is derived from ``face_entry_from_slots`` itself, so
    it IS the set of sites the measurement installs. The probe must record a
    tensor for each of them, or a slot's mask answers for a tensor the hook is
    not applied to. Checked under BOTH ``--approx-add`` settings.

    #73: under BOTH values every slot now has exactly ONE site, and the `new`
    slot's is ``res:new`` -- the fresh contraction, which is the tensor the
    probe records and the mask is computed on. That is fault 1 removed at the
    root rather than masked around: the retired ``--approx-old same`` listed
    ``("res:new", "res:jr")`` here, two tensors for one wire row.
    """
    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    prev = os.environ.get(envmod._APPROX_ADD_ENV)
    os.environ.pop(envmod._APPROX_OLD_ENV, None)
    try:
        # Every value, INCLUDING `choose`. Under `choose` the join arm is a
        # per-face decision, so `face_slot_sites` derives BOTH arms and requires
        # them to agree -- which is the claim being checked here: the site
        # topology is a property of where the slot HOOKS go, and the arms differ
        # only in what sits at `jr`, which is not a slot hook.
        _SITES = {
            "lossy":    (("lhs",), ("rhs",), ("res:new",)),
            "lossless": (("lhs",), ("rhs",), ("res:new",)),
            "choose":   (("lhs",), ("rhs",), ("res:new",)),
            "learned1": (("lhs",), ("rhs",), ("res:new",), ("res:jr",)),
            "learned2": (("lhs",), ("rhs",), ("res:new",), ("res:jr",),
                         ("res:jres",)),
        }
        for want in envmod.APPROX_ADD_CHOICES:
            os.environ[envmod._APPROX_ADD_ENV] = want
            sites = envmod.face_slot_sites()
            # The WIDTH is the value's since 2026-09-11: three sites under the
            # contraction-only values, four under learned1, five under learned2.
            assert sites == _SITES[want], (want, sites)
            # ONE site per slot is the invariant, not the count.
            assert all(len(x) == 1 for x in sites), (want, sites)
            flat = {x for per in sites for x in per}
            lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                                max_faces=MAX_FACES, max_axes=N_AX)
            specs = np.zeros((len(jaxpr.eqns), 1, 3), np.int32)
            tk = lf._tokenizer_at(np.asarray(order), specs, 0)
            seen: set = set()
            for v in [int(x) for x in order[:12]]:
                got = lf._probe_faces(tk, v, list(tk.ij.faces(v)), True,
                                      slots=True, stat="slot") or {}
                for by_site in got.values():
                    seen |= set(by_site)
            assert seen, f"{want}: the probe recorded no site at all"
            assert seen <= flat, (
                f"{want}: the probe records sites the entry never installs: "
                f"{sorted(seen - flat)}")
            # Every site in the topology is now unconditional (a join POLICY
            # is not a slot hook and has no mask), so the probe must record
            # exactly the topology -- never a site outside it, never fewer.
            assert {"lhs", "rhs", sites[2][0]} <= seen, (
                f"{want}: probe missed "
                f"{sorted({'lhs', 'rhs', sites[2][0]} - seen)}")
    finally:
        if prev is None:
            os.environ.pop(envmod._APPROX_ADD_ENV, None)
        else:
            os.environ[envmod._APPROX_ADD_ENV] = prev


# ==========================================================================
# THE TWO LEARNED JOIN SLOTS (#73): learned1 on the OLD EDGE (graphax
# ``res:jr``) and learned2 on the SUMMED EDGE (``res:jres``). Each is masked
# from the tensor its decision actually lands on -- the probe records one tensor
# per site and ``slot_legality`` is computed on that site's own tensor.
#
# MEASURED (job 64808, TLM, one graph, min-Markowitz, 5 seeds -- the same
# N_SAMPLES these tests use), rejection rate from the engine's own counters:
#
#   --approx-add   armed slots              requested   rejected   rate
#   learned1       3      (learned1 alone)         27          0  0.000
#   learned2       4      (learned2 alone)        181          0  0.000
#   learned1       0,1,2,3                        594          3  0.005
#   learned2       0,1,2,4                        741         20  0.027
#   learned2       0,1,2,3,4                      767         21  0.027
#
# THE VALUE IS NOT A FREE CHOICE NEXT TO THE ARM (2026-09-11): slot 3 exists
# only under `learned1` and above, slot 4 only under `learned2`, because the
# head width and the wire width are the same number. `_join_totals` derives the
# value from the arm for exactly that reason.
#
# With the contraction slots UNARMED both masks were already exactly right:
# nothing moves their tensor between the mask and the apply. With them armed both
# rejected, at rates 5x apart, and both were dsnn-3qm.59 fault 2 by two different
# routes. BOTH ARE NOW ZERO (#75) and both cases are assertions.
#
# A 3-SEED PROBE SAID learned1 WAS IMMUNE (0 of 354) AND IT WAS WRONG. The
# 5-seed test caught it. Kept because the lesson is the useful part: a rate
# this low is invisible in a short run, and the structural argument that
# produced the wrong prediction (the old edge is built by earlier eliminations,
# so this face cannot touch it) was incomplete -- it ignored sibling faces at
# the same vertex. It is also the reason the fix had to be ORDER and not a
# per-slot special case: the same pass has to serve both routes.
#
# Do not relax any of these to a skip: a mask that clears an action the engine
# then refuses is the defect deliverable 3 exists to forbid.
# ==========================================================================
def _walk_join_slots(target, seed, arm):
    """Arm the JOIN slots (and optionally the contraction slots), each drawn
    against ITS OWN DYNAMIC mask row, and return (requested, engine stats).

    Each slot is sampled with a separate head call against its own mask row.
    That is a PROBE, not the production path -- the rollout wire does not carry
    the join band yet -- but the MASK and the TENSOR are the production ones,
    which is what a rejection rate is about.

    #75: the mask row for slot ``sl`` is now read off the live tensor at slot
    ``sl``'s own graphax site INSIDE the one speculative elimination that also
    applies every earlier decision, so ``res:jr`` is masked from the old edge AS
    EARLIER FACES' MERGES LEFT IT and ``res:jres`` from the sum of this face's
    own approximated addends. Before, all five rows came from one recording
    probe in which nothing had been decided.

    ``--approx-add`` MUST ALREADY NAME A VALUE WIDE ENOUGH FOR ``arm`` (the
    caller sets it; see ``_join_totals``). Since 2026-09-11 arming slot 3 IS
    ``learned1`` and arming slot 4 IS ``learned2``: the width is not a knob a
    probe can turn independently of the configuration, and ``face_slot_sites``
    / ``wire_slots`` answer from the configuration alone.
    """
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.env import face_slot_sites

    jaxpr, consts, args, argnums = target
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    F = MAX_FACES
    sites = face_slot_sites()
    S_ALL = len(sites)

    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=F, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    config = SimpleNamespace(jaxpr=jaxpr)
    pol, tables = _policy(F)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    draw_jit = _jit_draw(pol, feats, ctx, tables)

    envmod._PER_FACE_STATS.clear()
    requested = {k: 0 for k in KINDS}
    for n in range(len(order)):
        v = int(order[n])
        keys = list(tk.ij.faces(v))
        skips = np.zeros((F,), np.int32)

        def _draw(f, sl, L, _n=n):
            if sl not in arm:
                return None
            key = jrand.fold_in(
                jrand.PRNGKey(seed * 1000003 + _n * 97 + f), sl)

            def _rep(a):
                return jnp.asarray(np.repeat(
                    np.asarray(a)[None], FACE_SLOTS, 0))

            _sk, op_t, ii, jj, dt = draw_jit(
                key, f, _rep(L.pair.astype(np.float32)),
                _rep(L.comp.astype(np.float32)), _rep(L.sizes),
                _rep(L.quant.astype(np.float32)))
            op = int(np.asarray(op_t)[0])
            if op == OP_NONE:
                return None
            ii, jj, dt = np.asarray(ii), np.asarray(jj), np.asarray(dt)
            w = _row_to_wire(op, ii[0], jj[0], ii[0], dt[0], L.n_out)
            if w is None:
                return None
            requested[_OP_KIND[op]] += 1
            return w

        dec = lf.decide_faces(tk, v, keys, _draw, skips=skips)
        per_face = _face_dict_for_vertex(config, tk.ij, v, dec.rows, skips)
        arm_face_counts()
        try:
            tk.ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()
    out = dict(envmod._PER_FACE_STATS)
    out.update({f"lf_{k}": val for k, val in lf.consume_stats().items()
                if k.startswith("decide") and val})
    envmod._PER_FACE_STATS.clear()
    return requested, out


def _join_totals(target, arm):
    """Totals over ``N_SAMPLES`` seeds, under the ``--approx-add`` value whose
    WIDTH owns the armed slots.

    Slot 3 exists only under ``learned1`` and above; slot 4 only under
    ``learned2``. So the value is derived from the arm rather than passed in:
    asking for slot 4 under ``learned1`` has to be impossible, not merely
    discouraged, and ``env.face_entry_from_slots`` raises on the width mismatch
    if it ever is.
    """
    want = "learned2" if max(arm) >= 4 else "learned1"
    prev = os.environ.get(envmod._APPROX_ADD_ENV)
    os.environ.pop(envmod._APPROX_OLD_ENV, None)
    os.environ[envmod._APPROX_ADD_ENV] = want
    try:
        assert envmod.wire_slots() == (5 if want == "learned2" else 4)
        req = {k: 0 for k in KINDS}
        skip = {k: 0 for k in KINDS}
        for seed in range(N_SAMPLES):
            r, st = _walk_join_slots(target, seed, arm)
            for k in KINDS:
                req[k] += r[k]
                skip[k] += int(st.get(f"skipped_{k}", 0))
            assert int(st.get("lf_decide_self_skip", 0)) == 0, st
            assert int(st.get("lf_decide_multi_site", 0)) == 0, st
        return req, skip
    finally:
        if prev is None:
            os.environ.pop(envmod._APPROX_ADD_ENV, None)
        else:
            os.environ[envmod._APPROX_ADD_ENV] = prev


def test_learned1_alone_rejects_nothing(tlm):
    """THE MASK IS RIGHT FOR ITS TENSOR. With the contraction slots unarmed,
    nothing upstream moves the old edge between the mask and the apply, and
    learned1 refuses nothing: 0 of 27 over 5 seeds. Run under
    ``--approx-add learned1``, the 125-logit / 4-slot width."""
    req, skip = _join_totals(tlm, (3,))
    assert sum(req.values()) > 0, "nothing was requested -- vacuous"
    assert sum(skip.values()) == 0, (req, skip)


def test_learned2_alone_rejects_nothing(tlm):
    """Same statement for the summed edge: 0 of 181 over 5 seeds. Run under
    ``--approx-add learned2``, the 156-logit / 5-slot width."""
    req, skip = _join_totals(tlm, (4,))
    assert sum(req.values()) > 0, "nothing was requested -- vacuous"
    assert sum(skip.values()) == 0, (req, skip)


# Both learned slots DO reject once the contraction slots are armed, at very
# different rates, and both are dsnn-3qm.59 fault 2 -- by two different routes.
#
#   armed        requested  rejected   rate
#   0,1,2,3            594         3  0.005
#   0,1,2,4            741        20  0.027
#   0,1,2,3,4          767        21  0.027
#
# learned2, the SUMMED edge: it IS `new + old`, so it directly carries this
# face's own contraction approximations. Plain fault 2.
#
# learned1, the OLD EDGE: I predicted this one was immune, on the argument that
# the old edge is built by EARLIER eliminations and this face's rules cannot
# touch it. A 3-seed probe agreed (0 of 354) and the 5-seed test did not (3 of
# 594). The argument was incomplete: a vertex has SEVERAL faces, and an earlier
# face's merge WRITES the edge a later face's `jr` READS, so arming the
# contraction slots changes sibling faces' old edges WITHIN one elimination step.
# That is the intra-vertex cross-face coupling finding 72 listed as fault 2's
# sibling. The rate is 5x lower than learned2's, which is consistent: it needs
# two faces at one vertex rather than one face's own product.
#
# THEY FLIPPED (#75, 2026-09-11), and these are now ASSERTIONS.
#
# Both rates go to ZERO once each slot is masked from the tensor it will actually
# be applied to. `LiveFaceStream.decide_faces` takes the whole vertex's decisions
# inside ONE speculative elimination in which every slot is a graphax chooser, so
# learned1 sees the old edge AS THE EARLIER FACE'S MERGE LEFT IT and learned2 the
# sum of this face's own approximated addends. Measured, finding 75 job 64889,
# TLM, min-Markowitz, 5 seeds, the same walk run twice in one process with only
# the mask source changed:
#
#   armed        static: requested / rejected     dynamic: requested / rejected
#   0,1,2,3        594 / 3                          595 / 0
#   0,1,2,4        741 / 20                         734 / 0
#
# THESE TWO STAY ON `decide_faces`, and that is not laziness. `res:jr` and
# `res:jres` are the two tensors #77's per-vertex composition does NOT reach: an
# earlier face's merge WRITES the edge a later face's `jr` reads, so they are
# not a function of a face's own operands. `decide_vertex_faces` raises rather
# than answering for them -- see
# `test_the_vertex_pass_refuses_the_learned_join_slots`.
#
# The request counts move because the dynamic mask is a DIFFERENT mask -- it
# clears what the live tensor allows, not what an all-exact graph allowed -- so
# a changed request count is the feature working, not a lost action.
def test_learned1_rejects_nothing_with_the_contraction_slots_armed(tlm):
    """THE OLD EDGE, intra-vertex, under ``--approx-add learned1``.

    A vertex has SEVERAL faces and an earlier face's merge WRITES the edge a
    later face's ``jr`` READS, so arming the contraction slots used to stale
    learned1's mask WITHIN one elimination step: 3 of 594 over 5 seeds, against
    0 of 27 with them unarmed. The decide pass visits the faces in graphax's own
    order and applies each decision as it is taken, so the later face's ``jr``
    mask is read off the edge the earlier merge produced.
    """
    req, skip = _join_totals(tlm, (0, 1, 2, 3))
    assert sum(req.values()) > 0, "nothing was requested -- vacuous"
    assert sum(skip.values()) == 0, (req, skip)


def test_learned2_rejects_nothing_with_the_contraction_slots_armed(tlm):
    """THE SUMMED EDGE, under ``--approx-add learned2``.

    ``res:jres`` IS ``new + old``, so it carries this face's own contraction
    approximations, and its mask used to be read before those rows were drawn:
    20 of 741 over 5 seeds, against 0 of 181 with the contraction slots
    unarmed. The decide pass reaches ``res:jres`` only after ``lhs``, ``rhs``,
    ``res:new`` and the add have all happened, which is the whole of the fix.
    """
    req, skip = _join_totals(tlm, (0, 1, 2, 4))
    assert sum(req.values()) > 0, "nothing was requested -- vacuous"
    assert sum(skip.values()) == 0, (req, skip)


# --------------------------------------------------------------------------
# THE FRESH CONTRACTION, and it is FIXED (#75).
#
# Arming lhs and rhs used to make `res:new` itself reject: 6 of 26 Diag and 4 of
# 40 Reduce on one graph, against 0 of 27 and 0 of 39 with `new` armed alone.
# `new` holds the PRODUCT of lhs and rhs, and the head drew all three slots from
# ONE distribution against ONE mask computed before any of them existed.
#
# The mask is now read at `res:new`'s own site INSIDE the elimination that has
# already applied the lhs and rhs rows, so it describes the tensor the hook will
# meet. The head is called once per (face, slot) instead of once per face, which
# changes no draw -- see `test_the_per_slot_draws_equal_one_joint_draw`.
#
# AND SINCE #77 THE MASK NEEDS NO SPECULATIVE ELIMINATION AT ALL. `_walk_one_graph`
# decides through `decide_vertex_faces`: n in-edge forces + m out-edge forces +
# n*m calls of `graphax.contract_face_operands`, the function `_eliminate_vertex`
# itself calls.
#
# MEASURED, job 64928, TLM, min-Markowitz, 5 seeds, one process, one machine, the
# SAME walk with only the mask source changed (`.../probes/t77vertexmask/t77_vertex.py`):
#   static (one recording probe per vertex)  526 requested, 15 REJECTED
#   decide (#75, one elimination, choosers)  522 requested,  0 REJECTED
#   vertex (#77, no elimination)             522 requested,  0 REJECTED
# and on nn256, 97/2 -> 96/0 -> 96/0. The #77 pass's composed `res:new` structure
# agreed with the tensor the real apply path hands the hook on 2145 of 2145
# per-(face, slot) mask fields on TLM and 495 of 495 on nn256
# (`test_the_structural_contraction_is_the_apply_paths_own`).
# --------------------------------------------------------------------------
def test_tlm_every_slot_rejects_nothing_on_one_graph(tlm):
    _assert_no_rejection(_walk_one_graph, tlm, "tlm one-graph all slots")


def test_nn256_every_slot_rejects_nothing_on_one_graph(nn256):
    """Same claim on nn256 -- BUT NOT COVERAGE OF THE ADD.

    nn256 has ZERO merge faces among armed faces (finding 73: `res:jl` and
    `res:jr` invoked 0 times on every seed, with `new` armed alone AND with all
    three armed; the legality probe, which visits every face, found exactly one
    merge face in 33 and an approximation never lands on it). So this case tests
    the WITHIN-FACE source only. Sources 2 and 3 -- the old edge and the summed
    edge -- cannot be exercised on this target at all, and a green nn256 run is
    not evidence about them.
    """
    _assert_no_rejection(_walk_one_graph, nn256, "nn256 one-graph all slots")


# --------------------------------------------------------------------------
# TWO GRAPHS: the production arrangement, and a SEPARATE open defect.
#
# ppo.py rides the stream tokenizer for the mask and a second
# ``IncrementalJaxpr`` for the face keys. Finding 71 measured that the two
# disagree about the FACE KEY LIST on 3 of 95 TLM steps (first at step 3,
# vertex 8: the stream graph reports 0 keys, the apply graph 1), so row `f` of
# the wire can describe a different face than the mask did. It also owns
# Reduce rejections the single graph does not: with only lhs+rhs armed -- where
# neither fault above can fire -- two graphs reject 3 Reduce over 5 TLM seeds
# and one graph rejects none. Measured identically on the branch tip BEFORE the
# site-set fix (job 64642 against 07fc4c4), so it is not a consequence of it.
#
# The fix is ticket .59's single-graph merge: one IncrementalJaxpr owns both
# the legality and the apply. Until then, STRICT xfail.
#
# #75 DID NOT FIX THESE AND WAS NEVER GOING TO. The decide pass runs on the
# STREAM tokenizer and the apply on the separate `IncrementalJaxpr`, so the mask
# is now right about the stream graph's tensors and the apply still happens on
# another graph's. That these five still xfail while the four one-graph cases
# flipped is the cleanest available evidence that the duplication is a SEPARATE
# defect and not a symptom of the stale mask.
# --------------------------------------------------------------------------
@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 (one graph): the legality tokenizer and "
                          "the apply IncrementalJaxpr disagree on 3 of 95 TLM "
                          "steps; 3 Reduce rows rejected over 5 seeds with only "
                          "lhs+rhs armed, 0 on one graph")
def test_tlm_operand_slots_reject_nothing_on_two_graphs(tlm):
    _assert_no_rejection(_walk, tlm, "tlm two-graph lhs+rhs",
                         slots_on=("lhs", "rhs"))


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 (one graph): same duplication defect on "
                          "nn256, 1 Reduce row over 5 seeds")
def test_nn256_operand_slots_reject_nothing_on_two_graphs(nn256):
    _assert_no_rejection(_walk, nn256, "nn256 two-graph lhs+rhs",
                         slots_on=("lhs", "rhs"))


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 (one graph): the `new` slot alone is "
                          "clean on one graph and rejects 2 Reduce rows over 5 "
                          "TLM seeds on two")
def test_tlm_result_slot_alone_rejects_nothing_on_two_graphs(tlm):
    _assert_no_rejection(_walk, tlm, "tlm two-graph new", slots_on=("new",))


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59: fault 2 and the duplicated graph "
                          "together")
def test_tlm_every_slot_rejects_nothing_on_two_graphs(tlm):
    _assert_no_rejection(_walk, tlm, "tlm two-graph all slots")


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59: fault 2 and the duplicated graph "
                          "together, nn256")
def test_nn256_every_slot_rejects_nothing_on_two_graphs(nn256):
    _assert_no_rejection(_walk, nn256, "nn256 two-graph all slots")


def test_the_counters_are_silent_outside_the_armed_scope(tlm):
    """Every hook here is built ``gated=True``: the tokenizer and the face-enum
    replay invoke the same objects on graphs nothing is measured on, so only
    ``arm_face_counts()`` may count. Without this the apply rate above would be
    a sum over three replays of one plan."""
    jaxpr, consts, args, argnums = tlm
    config = SimpleNamespace(jaxpr=jaxpr)
    ij = IncrementalJaxpr(jaxpr, argnums, list(consts), list(args),
                          track_faces=False)
    rows = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    rows[..., 2] = 0
    rows[0, 0] = (int(envmod.COMPRESS_SENTINEL), 0, 0)
    skips = np.zeros((MAX_FACES,), np.int32)
    v = int(len(jaxpr.eqns) - 1)
    envmod._PER_FACE_STATS.clear()
    per_face = _face_dict_for_vertex(config, ij, v, rows, skips)
    ij.eliminate(v, (), per_face or None)      # NOT armed
    assert envmod._PER_FACE_STATS == {}, envmod._PER_FACE_STATS


# ==========================================================================
# THE DYNAMIC MASK ITSELF (#75, dsnn-3qm.59 fault 2). Three properties, each
# stated so it can FAIL rather than merely be described.
# ==========================================================================
def test_the_per_slot_draws_equal_one_joint_draw():
    """SAMPLING SLOT BY SLOT CHANGES NO DRAW, so PPO still scores one variable.

    A dynamic mask forces one head call per (face, slot): slot ``s``'s mask does
    not exist until slots ``< s`` have been decided and applied. That is only
    safe if slot ``s``'s draw is a function of ``(key, ctx, slot s's masks)``
    alone -- otherwise the log-prob the trainer stores would be over a
    distribution the replay cannot rebuild, and the PPO ratio would not be 1 at
    epoch 0 even with the mask stored verbatim.

    ``UnifiedFaceHead.sample`` splits ``1 + 6*n_slots (+1)`` keys and gives slot
    ``s`` the slice ``keys[1+5s:1+5(s+1)]`` plus ``dt_keys[s]`` against its own
    31-logit block, so nothing is conditioned on a previously drawn slot. This
    states the CONSEQUENCE rather than the mechanism: ``S`` calls whose mask rows
    fill in one at a time must draw exactly what ONE call with every row filled
    draws, field for field, and the skip bit must agree too (the skip logit's
    only masks are ``face_valid`` and ``approx_ok``, never a slot's tensor
    legality).
    """
    pol, tables = _policy(MAX_FACES)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    rng = np.random.default_rng(0)
    S = FACE_SLOTS
    checked = 0
    for trial in range(8):
        final_pair = (rng.random((S, N_AX, N_AX)) < 0.5).astype(np.float32)
        final_comp = (rng.random((S, N_AX)) < 0.5).astype(np.float32)
        sz = np.full((S, N_AX), 4, np.int32)
        qt = np.ones((S, 2), np.float32)
        key = jrand.PRNGKey(trial)
        sk_all, row_all, lp_all, *_ = pol.sample_face(
            feats, tables, key, 0, jnp.asarray(final_pair),
            jnp.asarray(final_comp), jnp.asarray(1.0), face_context=ctx,
            face_sizes_f=jnp.asarray(sz), face_quant_f=jnp.asarray(qt))
        # the partially-filled arrays a decide pass really hands the head:
        # rows 0..s are live, rows past s are still ZERO.
        for s in range(S):
            live = (np.arange(S) <= s)
            pr = np.where(live[:, None, None], final_pair, 0.0)
            cp = np.where(live[:, None], final_comp, 0.0)
            sk, row, lp, *_ = pol.sample_face(
                feats, tables, key, 0, jnp.asarray(pr), jnp.asarray(cp),
                jnp.asarray(1.0), face_context=ctx,
                face_sizes_f=jnp.asarray(sz), face_quant_f=jnp.asarray(qt))
            np.testing.assert_array_equal(
                np.asarray(sk), np.asarray(sk_all))
            for fld in row:
                got = np.asarray(row[fld])
                want = np.asarray(row_all[fld])
                if got.ndim == 0:
                    continue
                # array_equal, not `==`: `exponents` is (S, MAX_PRIMES), so
                # `got[s]` is a VECTOR for that field and `==` is ambiguous.
                np.testing.assert_array_equal(
                    got[s], want[s],
                    err_msg=f"trial {trial} slot {s} field {fld}")
                checked += 1
    assert checked > 0


def test_the_operand_slots_were_never_stale_and_the_dependent_ones_were(tlm):
    """THE DEFECT, LOCALISED -- and the reason only some slots needed fixing.

    ``face_slot_legality`` reads every slot's tensor from one recording probe in
    which NO decision has been made. Measured (findings 72-74) that is harmless
    for ``lhs`` and ``rhs`` -- nothing at this vertex moves the in-edge and
    out-edge Jacobians, and arming them alone rejects nothing -- and wrong for
    the three dependent sites.

    So the claim here is a DIFFERENCE, not an equality: on the same vertex and
    the same graph, the dynamic pass must agree with the static probe on slots 0
    and 1 (otherwise the fix moved something it had no business moving) and must
    DISAGREE somewhere on a dependent slot (otherwise it is not dynamic at all
    and the xfails below would have flipped for no reason).
    """
    from alphagrad.approx.env import face_slot_sites

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    specs = np.zeros((len(jaxpr.eqns), 1, 3), np.int32)
    S_ALL = len(face_slot_sites())
    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=MAX_FACES, max_axes=N_AX)
    rows_hist = np.full((len(jaxpr.eqns), MAX_FACES, S_ALL, 3), -1, np.int32)
    rows_hist[..., 2] = 0
    skips_hist = np.zeros((len(jaxpr.eqns), MAX_FACES), np.int32)

    same_operand = dep_diff = dep_same = 0
    rng = np.random.default_rng(0)
    for n in range(len(order)):
        v = int(order[n])
        st_sz, st_q, st_pair, st_comp, st_no, nf = lf.face_slot_legality(
            order, specs, n, v, rows_hist, skips_hist)
        nf = int(nf)
        if nf == 0:
            continue
        dyn_pair = np.zeros_like(st_pair)
        dyn_comp = np.zeros_like(st_comp)
        seen = set()

        def _draw(f, sl, L):
            dyn_pair[f, sl] = L.pair.astype(np.float32)
            dyn_comp[f, sl] = L.comp.astype(np.float32)
            seen.add((f, sl))
            # A REAL approximation on every slot, or the dependent tensors would
            # never move and this test could not tell the two passes apart.
            ax = np.flatnonzero(L.comp)
            if ax.size == 0:
                return None
            return (int(envmod.COMPRESS_SENTINEL),
                    int(rng.choice(ax)), 0)

        dec = lf.face_slot_decisions(order, specs, n, v, _draw,
                                     skips=skips_hist[n],
                                     face_rows_hist=rows_hist,
                                     face_skips_hist=skips_hist)
        rows_hist[n] = dec.rows
        for f in range(nf):
            for sl in range(S_ALL):
                if (f, sl) not in seen:
                    continue
                eq = (np.array_equal(dyn_pair[f, sl], st_pair[f, sl])
                      and np.array_equal(dyn_comp[f, sl], st_comp[f, sl]))
                if sl < 2:
                    assert eq, (
                        f"vertex {v} face {f} slot {sl}: the dynamic pass "
                        f"disagrees with the static probe on an OPERAND slot, "
                        f"whose tensor nothing at this vertex moves")
                    same_operand += 1
                elif eq:
                    dep_same += 1
                else:
                    dep_diff += 1
    assert same_operand > 0, "no operand slot was reached -- vacuous"
    assert dep_diff > 0, (
        f"the dynamic mask never differed from the static one on a dependent "
        f"slot ({dep_same} agreed, {dep_diff} differed) -- it is not dynamic")


def test_the_decide_pass_answers_at_every_approx_add_width(tlm):
    """ONE SITE PER SLOT, AT ALL FIVE WIDTHS, and a decision at each.

    ``--approx-add`` is five head widths (94 / 95 / 125 / 156 logits, slot base
    ``1 + 31*s``) and three wire widths (3 / 4 / 5 slots). The decide pass sizes
    itself from ``env.wire_slots()`` and installs one chooser per slot through
    ``env.face_entry_from_slots``, so it must answer under every one of them --
    including ``choose``, where the join arm is a per-face bit the pass does not
    hold (it builds the entry ``with_policy=False``, the same reason
    ``face_slot_sites`` can).

    ``decide_multi_site`` must stay 0: a slot whose chooser fired twice would be
    one wire row meeting two tensors, which is finding 72's fault 1.
    """
    from graphax import IncrementalPathTokenizer

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    prev = os.environ.get(envmod._APPROX_ADD_ENV)
    os.environ.pop(envmod._APPROX_OLD_ENV, None)
    try:
        for want in envmod.APPROX_ADD_CHOICES:
            os.environ[envmod._APPROX_ADD_ENV] = want
            S = envmod.wire_slots()
            sites = envmod.face_slot_sites()
            assert len(sites) == S and all(len(x) == 1 for x in sites), sites
            lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                                max_faces=MAX_FACES, max_axes=N_AX)
            tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts),
                                          list(args), vocab_size=512)
            tk.base_tokens()
            rng = np.random.default_rng(1)
            reached = set()
            decided = 0
            for v in [int(x) for x in order[:20]]:
                keys = list(tk.ij.faces(v))

                def _draw(f, sl, L):
                    reached.add(sl)
                    ax = np.flatnonzero(L.comp)
                    if ax.size == 0:
                        return None
                    return (int(envmod.COMPRESS_SENTINEL),
                            int(rng.choice(ax)), 0)

                dec = lf.decide_faces(tk, v, keys, _draw)
                assert dec.rows.shape == (MAX_FACES, S, 3), dec.rows.shape
                # `!= -1`, NOT `>= 0`: a Reduce row's first field is
                # COMPRESS_SENTINEL = -2 and a Quant row's is -3. `-1` is the
                # only "no decision" value, which is what `decide_faces`
                # initialises the array to.
                decided += int((dec.rows[..., 0] != -1).sum())
                # `choose` decides the join PER FACE, so the wire must carry
                # the bit and `_face_dict_for_vertex` RAISES without it rather
                # than defaulting -- which is the right behaviour and the reason
                # this is passed here instead of being worked around. The decide
                # pass itself never needs the bit (it builds the entry
                # `with_policy=False`), which is what lets it answer under
                # `choose` at all.
                _join = (np.zeros((MAX_FACES,), np.int32)
                         if want == "choose" else None)
                per_face = _face_dict_for_vertex(
                    SimpleNamespace(jaxpr=jaxpr), tk.ij, v, dec.rows,
                    np.zeros((MAX_FACES,), np.int32), face_join=_join)
                tk.ij.eliminate(v, (), per_face or None)
            st = lf.consume_stats()
            assert st["decide_probe"] == 20, (want, st)
            assert st["decide_multi_site"] == 0, (want, st)
            assert st["decide_self_skip"] == 0, (want, st)
            assert st["decide_probe_fail"] == 0, (want, st)
            assert decided > 0, f"{want}: nothing was decided -- vacuous"
            assert reached >= {0, 1, 2}, (want, sorted(reached))
    finally:
        if prev is None:
            os.environ.pop(envmod._APPROX_ADD_ENV, None)
        else:
            os.environ[envmod._APPROX_ADD_ENV] = prev


def test_the_jitted_draw_is_the_eager_draw():
    """THE WALKS ABOVE JIT THE HEAD, so jit must move no sample.

    A dynamic mask costs one head call per (face, slot); eager that is 35-78 ms
    a call and this file would take hours. If jit changed a draw -- a fused
    reduction reordering a float and flipping a categorical at a tie -- the
    rejection counts would be over a different plan than findings 73/74
    measured, and the xfail flips would be comparing two different things.
    Stated over masks that are HALF ILLEGAL, because a half-masked softmax is
    where a tie is reachable at all.
    """
    pol, tables = _policy(MAX_FACES)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    dj = _jit_draw(pol, feats, ctx, tables)
    rng = np.random.default_rng(11)
    S = FACE_SLOTS
    n = 0
    for t in range(40):
        pr = (rng.random((S, N_AX, N_AX)) < 0.5).astype(np.float32)
        cp = (rng.random((S, N_AX)) < 0.5).astype(np.float32)
        sz = np.full((S, N_AX), 4, np.int32)
        qt = np.ones((S, 2), np.float32)
        k = jrand.PRNGKey(t)
        _sk, row, *_ = pol.sample_face(
            feats, tables, k, 0, jnp.asarray(pr), jnp.asarray(cp),
            jnp.asarray(1.0), face_context=ctx, face_sizes_f=jnp.asarray(sz),
            face_quant_f=jnp.asarray(qt))
        got = dj(k, 0, jnp.asarray(pr), jnp.asarray(cp), jnp.asarray(sz),
                 jnp.asarray(qt))
        want = (_sk, row["op_type"], row["i"], row["j"], row["quant_dtype"])
        for a, b in zip(want, got):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            n += 1
    assert n == 40 * 5


# ==========================================================================
# #77: THE EXACT PER-VERTEX MASK. The four tests below are the claims the
# design rests on, in the order they have to hold.
# ==========================================================================
def _real_res_tensors(lf, tk, vertex, keys, rows, skips):
    """``{(f, s): the tensor the REAL elimination hands that slot's hook}``.

    The recorders are installed THROUGH ``env.face_entry_from_slots`` -- the one
    place a face wire becomes a graphax entry -- so the sites recorded are by
    construction the sites the measurement installs; hard-coding the entry form
    in a second place is finding 72's fault 1 exactly.

    ``transforms=()``, which is what ``IncrementalJaxpr.eliminate(v, (), ft)``
    passes, so ``_is_approx_cfg`` comes from
    ``graphax.face_config_is_approx(face_transforms)`` and not from a probe's
    per-vertex callable. That matters: arming it would compare the pass against
    a THIRD code path (it gates the reconciler peel and the re-evaluation of
    ``need_contract``), and under ``--approx-add lossless`` the measurement runs
    with it OFF.
    """
    from jax._src import core as _jcore
    from graphax.core import _eliminate_vertex
    from alphagrad.approx.env import (
        face_entry_from_slots, make_slot_frame_hook, wire_slots)
    from alphagrad.approx.live_faces import _Snapshot

    S = wire_slots()
    seen: dict = {}

    def _wrap(f, s):
        row = tuple(int(x) for x in rows[f, s])
        inner = None if row[0] == -1 else make_slot_frame_hook(row)

        def _h(st):
            seen.setdefault((f, s), st)
            return st if inner is None else inner(st)
        return _h

    ft = {}
    for f in range(min(len(keys), MAX_FACES)):
        if int(np.asarray(skips).reshape(-1)[f]) == 1:
            continue
        ft[keys[f]] = face_entry_from_slots(
            tuple(_wrap(f, s) for s in range(S)), with_policy=False)
    with _Snapshot(tk) as snap:
        ij = snap.ij
        with _jcore.set_current_trace(ij.trace):
            _eliminate_vertex(int(vertex), ij.jaxpr, ij.graph, ij.tgraph,
                              ij.vo, False, transforms=(),
                              face_transforms=ft)
    return seen


@pytest.mark.parametrize("target_name", ["tlm", "nn256"])
def test_the_structural_contraction_is_the_apply_paths_own(target_name, tlm,
                                                           nn256):
    """THE CLAIM THAT MAKES #77 SOUND: the composed structure IS the real one.

    ``decide_vertex_faces`` computes ``res:new``'s mask from
    ``graphax.prepare_face_operands`` + ``graphax.contract_face_operands``
    applied to the decided operands, WITHOUT running an elimination. Those two
    functions are the two halves of ``_eliminate_vertex``'s own contraction --
    lifted out of it and called BY it -- so the mask is not a second copy of the
    structure algebra. A second copy is what produced finding 72's fault 1 (6 of
    32 Diag rows the mask cleared refused at apply time), so "not a second copy"
    has to be a measurement and not an assurance.

    ON EVERY FACE AND EVERY CONTRACTION SLOT, with the decided rows installed:
    the five ``SlotLegality`` fields the pass recorded against
    ``masks.slot_legality`` on the tensor the real elimination hands that slot's
    hook. Field by field, not just ``sizes``: ``pair`` is the ``(N, N)`` Diag
    mask and it is the field a storage difference moves (finding 72's first
    disagreeing field was ``pair``, with ``sizes`` and ``n_out`` identical).

    A face graphax never visits -- an edge Jacobian that forces to ``None``,
    which ``faces_of`` lists optimistically -- must be ABSENT ON BOTH SIDES: the
    real elimination never invokes its hook and the pass must leave an all-zero
    mask row, i.e. offer nothing. Counted separately so the test cannot pass by
    comparing nothing.
    """
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.common.masks import slot_legality

    target = {"tlm": tlm, "nn256": nn256}[target_name]
    jaxpr, consts, args, argnums = target
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    F = MAX_FACES
    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=F, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    config = SimpleNamespace(jaxpr=jaxpr)
    pol, tables = _policy(F)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)
    draw_jit = _jit_draw(pol, feats, ctx, tables)

    compared = both_absent = 0
    bad = []
    for n in range(len(order)):
        v = int(order[n])
        keys = list(tk.ij.faces(v))
        skips = np.zeros((F,), np.int32)
        sizes = np.zeros((F, FACE_SLOTS, N_AX), np.int32)
        pair = np.zeros((F, FACE_SLOTS, N_AX, N_AX), np.float32)
        comp = np.zeros((F, FACE_SLOTS, N_AX), np.float32)
        quant = np.zeros((F, FACE_SLOTS, 2), np.float32)

        def _draw(f, s, L, _n=n):
            sizes[f, s] = L.sizes
            pair[f, s] = L.pair.astype(np.float32)
            comp[f, s] = L.comp.astype(np.float32)
            quant[f, s] = L.quant.astype(np.float32)
            key = jrand.PRNGKey(_n * 97 + f)
            _sk, op_t, ii, jj, dt = draw_jit(
                key, f, jnp.asarray(pair[f]), jnp.asarray(comp[f]),
                jnp.asarray(sizes[f]), jnp.asarray(quant[f]))
            op = int(np.asarray(op_t)[s])
            if op == OP_NONE:
                return None
            ii, jj, dt = np.asarray(ii), np.asarray(jj), np.asarray(dt)
            return _row_to_wire(op, ii[s], jj[s], ii[s], dt[s], L.n_out)

        dec = lf.decide_vertex_faces(tk, v, _draw, skips=skips)
        real = _real_res_tensors(lf, tk, v, keys, dec.rows, skips)
        for f in range(min(len(keys), F)):
            for s in range(FACE_SLOTS):
                st = real.get((f, s))
                if st is None:
                    zero = not (np.asarray(dec.sizes[f, s]).any()
                                or np.asarray(dec.pair[f, s]).any()
                                or np.asarray(dec.comp[f, s]).any()
                                or np.asarray(dec.quant[f, s]).any())
                    assert zero, (
                        f"vertex {v} face {f} slot {s}: the real elimination "
                        f"never invoked this hook, but the pass offered a "
                        f"non-empty mask for it")
                    both_absent += 1
                    continue
                L = slot_legality(st, N_AX)
                got = (np.asarray(dec.sizes[f, s]), int(dec.nout[f, s]),
                       np.asarray(dec.pair[f, s]) > 0.5,
                       np.asarray(dec.comp[f, s]) > 0.5,
                       np.asarray(dec.quant[f, s]) > 0.5)
                want = (L.sizes, L.n_out, L.pair, L.comp, L.quant)
                for name, g, w in zip(
                        ("sizes", "n_out", "pair", "comp", "quant"), got, want):
                    compared += 1
                    if not np.array_equal(np.asarray(g), np.asarray(w)):
                        bad.append(
                            f"vertex {v} face {f} slot {s} field {name}: "
                            f"composed {np.asarray(g).tolist()} != real "
                            f"{np.asarray(w).tolist()} (real val.shape="
                            f"{None if st.val is None else st.val.shape}, "
                            f"dtype={st.dtype}, rows="
                            f"{[tuple(int(x) for x in dec.rows[f, q]) for q in range(FACE_SLOTS)]})")
        per_face = _face_dict_for_vertex(config, tk.ij, v, dec.rows, skips)
        tk.ij.eliminate(v, (), per_face or None)

    assert compared > 0, "nothing was compared -- vacuous"
    assert not bad, (
        f"{len(bad)} of {compared} field comparisons disagree between the "
        f"composed structure and the tensor the real apply path hands the hook "
        f"({both_absent} (face, slot) pairs absent on both sides). The "
        f"structure algebra has a second copy after all:\n  "
        + "\n  ".join(bad[:12]))


def test_faces_of_is_face_specs_of_projected(tlm):
    """ONE ENUMERATION. ``face_specs_of`` is the loop and ``faces_of`` is its
    projection onto ``key``, so a caller that needs a face's OPERANDS cannot
    drift from the key list ``_eliminate_vertex`` looks up. Checked at every
    vertex of a real order, because the graph is rewired by every elimination
    and the two could agree on the first one and not the fifty-first.

    The same walk counts the MULTI-OUTPUT FACE-KEY COLLISION ``faces_of``'s
    docstring calls rare. Measured on TLM and nn256 under minimum Markowitz
    (finding 77): ZERO eliminated equations have more than one output variable
    at all, so the collision's PRECONDITION never occurs -- which is why
    ``decide_vertex_faces`` raises on a collision rather than widening the key.
    A target that did collide would otherwise have one wire row configure two
    faces and one mask row describe two tensors, silently.
    """
    from graphax import IncrementalPathTokenizer
    from graphax.core import face_specs_of, faces_of

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    n_faces = n_multi = n_collide = 0
    for v in order:
        v = int(v)
        g, tg = tk.ij.graph, tk.ij.tgraph
        specs = face_specs_of(g, tg, v, jaxpr)
        assert [sp.key for sp in specs] == faces_of(g, tg, v, jaxpr), v
        assert [sp.f for sp in specs] == list(range(len(specs))), v
        for sp in specs:
            assert sp.central in g, (v, sp)
            assert sp.in_edge in (tg.get(sp.central) or {}), (v, sp)
            assert sp.out_edge in g[sp.central], (v, sp)
        n_faces += len(specs)
        if len({id(sp.central) for sp in specs}) > 1:
            n_multi += 1
        if len({sp.key for sp in specs}) != len(specs):
            n_collide += 1
        tk.ij.eliminate(v, (), None)
    assert n_faces > 0
    assert n_multi == 0, (
        f"{n_multi} vertices have more than one LIVE output variable on TLM; "
        f"finding 77 measured 0, so the collision precondition has appeared "
        f"and decide_vertex_faces's raise may now fire")
    assert n_collide == 0, f"{n_collide} vertices have a repeated face key"


def test_the_vertex_pass_runs_no_elimination(tlm):
    """THE "NO SPECULATIVE ELIMINATION" CLAIM, AS COUNTERS.

    The pass is supposed to cost ``n`` in-edge forces + ``m`` out-edge forces +
    ``n*m`` structural contractions per vertex and NOT one elimination, so the
    three terms of the proof are counted and the elimination counters of the two
    other passes must be untouched. ``n + m <= 2 * n*m`` with equality only at
    ``n == m == 1`` is the cheap algebraic check that the operand probes are
    DEDUPLICATED -- the whole point of re-seeding from ``_pre_raw`` /
    ``_post_raw`` being a per-neighbour read rather than a per-face one.
    """
    from graphax import IncrementalPathTokenizer

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=MAX_FACES, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    faces = 0
    for v in order:
        v = int(v)
        faces += len(list(tk.ij.faces(v)))
        lf.decide_vertex_faces(tk, v, lambda f, s, L: None,
                               skips=np.zeros((MAX_FACES,), np.int32))
        tk.ij.eliminate(v, (), None)
    st = lf.consume_stats()
    assert st["vertex_probe"] == len(order), st
    assert st["vertex_probe_fail"] == 0, st
    for dead in ("size_probe", "slot_probe", "decide_probe"):
        assert st[dead] == 0, (
            f"{dead} = {st[dead]}: the vertex pass ran a speculative "
            f"elimination, which is the thing it exists not to do. {st}")
    nm = st["vertex_contractions"]
    npm = st["vertex_operand_probes"]
    assert nm > 0 and npm > 0, st
    assert nm + st["vertex_face_absent"] == faces, (nm, st, faces)
    assert npm <= 2 * nm, (
        f"{npm} operand probes for {nm} faces: the in-edge and out-edge "
        f"Jacobians are not being forced once each. {st}")


def test_the_vertex_pass_refuses_the_learned_join_slots(tlm):
    """IT SAYS SO RATHER THAN ANSWERING. ``res:jr`` (the old edge) and
    ``res:jres`` (the summed edge) are the two tensors the #77 proof does NOT
    cover: an earlier face's merge WRITES the edge a later face's ``jr`` reads,
    so they are not a function of a face's own operands and no per-face
    composition can reach them. Measured, finding 75: 3 of 594 and 20 of 741
    rows the static mask cleared are refused there.

    ``decide_faces`` -- one speculative elimination per vertex, every slot a
    chooser -- is the pass for those, and it is kept for exactly this reason;
    the learned-join assertions above are what exercise it.
    """
    from graphax import IncrementalPathTokenizer

    jaxpr, consts, args, argnums = tlm
    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=MAX_FACES, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    prev = os.environ.get(envmod._APPROX_ADD_ENV)
    try:
        for want, n_slots in (("lossless", 3), ("learned1", 4),
                              ("learned2", 5)):
            os.environ[envmod._APPROX_ADD_ENV] = want
            assert envmod.wire_slots() == n_slots, want
            if n_slots == FACE_SLOTS:
                lf.decide_vertex_faces(tk, int(1), lambda f, s, L: None)
                continue
            with pytest.raises(NotImplementedError, match="res:jr"):
                lf.decide_vertex_faces(tk, int(1), lambda f, s, L: None)
    finally:
        if prev is None:
            os.environ.pop(envmod._APPROX_ADD_ENV, None)
        else:
            os.environ[envmod._APPROX_ADD_ENV] = prev


def test_the_vertex_pass_answers_under_approx_add_choose(tlm):
    """`choose` HAS THREE SLOTS, SO THE PASS MUST ANSWER -- and it cannot know
    the join arm.

    ``--approx-add choose`` gives the face the three CONTRACTION slots and picks
    the join semantics from a PER-FACE BIT the head draws.
    ``env.face_entry_from_slots`` rightly RAISES without that bit rather than
    defaulting to an arm (finding 74), and this pass builds an entry only to ask
    graphax for ``_is_approx_cfg``. The bit decides between ``lossy`` and
    ``lossless``, and only ``lossy`` installs a ``FaceJoinPolicy``, which is what
    arms the flag -- so the answer that cannot mask from a flag the elimination
    will not have is TRUE, taken and COUNTED (``vertex_flag_assumed``) rather
    than left to an exception that would lose the whole vertex's decisions.

    What this asserts is exactly that: it ANSWERS (no ``vertex_probe_fail``), it
    offers something, and it says out loud that it assumed.
    """
    from graphax import IncrementalPathTokenizer

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                        max_faces=MAX_FACES, max_axes=N_AX)
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=512)
    tk.base_tokens()
    prev = os.environ.get(envmod._APPROX_ADD_ENV)
    os.environ[envmod._APPROX_ADD_ENV] = "choose"
    try:
        assert envmod.wire_slots() == FACE_SLOTS
        armed = 0
        for v in list(order)[:25]:
            v = int(v)

            def _draw(f, s, L):
                ax = np.flatnonzero(L.comp)
                if ax.size == 0:
                    return None
                return (int(envmod.COMPRESS_SENTINEL), int(ax[0]), 0)

            dec = lf.decide_vertex_faces(tk, v, _draw)
            armed += int((np.asarray(dec.rows)[..., 0] != -1).sum())
            tk.ij.eliminate(v, (), None)
        st = lf.consume_stats()
    finally:
        if prev is None:
            os.environ.pop(envmod._APPROX_ADD_ENV, None)
        else:
            os.environ[envmod._APPROX_ADD_ENV] = prev
    assert st["vertex_probe_fail"] == 0, (
        f"the pass failed under --approx-add choose: {st}")
    assert armed > 0, "nothing was offered -- vacuous"
    assert st["vertex_flag_assumed"] > 0, (
        f"the pass did not have to assume the approx flag under `choose`, which "
        f"means `face_entry_from_slots` answered without the per-face join bit "
        f"-- the guard finding 74 landed is gone. {st}")


def test_the_two_passes_draw_the_same_rows(tlm):
    """#77's pass and #75's pass DECIDE THE SAME PLAN.

    Both are exact, so they must agree -- and they reach the answer by different
    routes (a composition of two graphax functions against a whole speculative
    elimination with choosers), which is what makes the agreement evidence
    rather than a tautology. The draw is made deterministic in ``(f, s)`` and
    independent of the mask so that a disagreement localises to the MASK: if the
    two passes offered different legality, the rows would still be equal, and
    the mask arrays below are what would differ.

    CALL ORDER DIFFERS ON PURPOSE. #75 draws in APPLY order (face 0's lhs, rhs,
    new, then face 1's), #77 in STAGE order (every face's lhs and rhs, then
    every face's new). The rows and masks must not depend on that, and this is
    the test that says so.

    MODE-MATCHED, because the comparison would otherwise be about the flag
    and not about the two passes. ONE of graphax's settings is an ARGUMENT of
    the contraction and ``decide_faces`` forces it: ``_is_approx_cfg`` (it
    gates the reconciler peel and the re-evaluation of ``need_contract``),
    because that pass installs a per-vertex CALLABLE transform.
    ``decide_vertex_faces`` derives it from the caller instead, which is what
    makes it exact about ``IncrementalJaxpr.eliminate``. Here it is asked for
    the forced value, so the only remaining difference is the ROUTE: a
    composition of two graphax functions against a whole speculative
    elimination with choosers. (The second flag this used to match,
    ``approx_active``, went with the second engine, dsnn-3qm.65.)
    """
    from graphax import IncrementalPathTokenizer

    jaxpr, consts, args, argnums = tlm
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    rng_rows = {}

    def _mk(seen):
        def _draw(f, s, L):
            seen.append((f, s))
            ax = np.flatnonzero(L.comp)
            if ax.size == 0:
                return None
            # deterministic in (f, s), NOT in call order
            return (int(envmod.COMPRESS_SENTINEL),
                    int(ax[(f * 3 + s) % ax.size]), 0)
        return _draw

    def _run(pass_):
        lf = LiveFaceStream(jaxpr, argnums, consts, args, vocab=512,
                            max_faces=MAX_FACES, max_axes=N_AX)
        tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts),
                                      list(args), vocab_size=512)
        tk.base_tokens()
        config = SimpleNamespace(jaxpr=jaxpr)
        out = []
        for v in order:
            v = int(v)
            keys = list(tk.ij.faces(v))
            skips = np.zeros((MAX_FACES,), np.int32)
            seen: list = []
            if pass_ == "vertex":
                dec = lf.decide_vertex_faces(tk, v, _mk(seen), skips=skips,
                                             approx_cfg=True)
            else:
                dec = lf.decide_faces(tk, v, keys, _mk(seen), skips=skips)
            out.append((dec.rows.copy(), dec.sizes.copy(), dec.nout.copy(),
                        dec.pair.copy(), dec.comp.copy(), dec.quant.copy(),
                        int(dec.n_faces), sorted(seen)))
            per_face = _face_dict_for_vertex(config, tk.ij, v, dec.rows, skips)
            tk.ij.eliminate(v, (), per_face or None)
        return out

    a, b = _run("vertex"), _run("decide")
    assert len(a) == len(b)
    names = ("rows", "sizes", "nout", "pair", "comp", "quant", "n_faces",
             "draw set")
    diffs = []
    for n, (x, y) in enumerate(zip(a, b)):
        for name, xi, yi in zip(names, x, y):
            same = (xi == yi if name in ("n_faces", "draw set")
                    else np.array_equal(xi, yi))
            if not same:
                diffs.append(f"step {n} vertex {int(order[n])} field {name}")
    assert sum(int(x[6]) for x in a) > 0, "no faces -- vacuous"
    assert not diffs, (
        f"{len(diffs)} per-vertex fields differ between the #77 structural pass "
        f"and the #75 chooser pass; both claim to be exact, so at most one of "
        f"them is:\n  " + "\n  ".join(diffs[:20]))
