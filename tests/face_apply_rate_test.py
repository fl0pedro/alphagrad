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



def _walk_one_graph(target, seed, slots_on=None):
    """One sampled plan with the mask and the apply ON THE SAME GRAPH.

    ``_walk`` keeps the production pair of graphs; this owns a single
    ``IncrementalPathTokenizer``, reads the per-slot legality off it through
    ``LiveFaceStream._probe_faces`` (whose ``_Snapshot`` undoes the speculative
    elimination) and then runs the real elimination on that same graph.

    WHY BOTH. The two arrangements fail for different reasons, and a test that
    mixed them could not name either: the duplicated graph contributes its own
    rejections (finding 71: 4 of 9 Reduce, plus a face-key-list disagreement on
    3 of 95 steps), which would mask whether the per-slot legality itself is
    sound. This arm is the legality claim; ``_walk`` is the legality claim PLUS
    the duplication.
    """
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.common.masks import slot_legality
    from alphagrad.approx.env import face_slot_sites

    jaxpr, consts, args, argnums = target
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = markowitz_order(jaxpr, argnums, consts, args, vv)
    total_v = len(jaxpr.eqns)
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
        src = lf._probe_faces(tk, v, keys, True, slots=True, stat="slot") or {}
        nf = min(len(keys), F)
        sizes = np.zeros((F, S_ALL, N_AX), np.int32)
        pair = np.zeros((F, S_ALL, N_AX, N_AX), np.float32)
        comp = np.zeros((F, S_ALL, N_AX), np.float32)
        quant = np.zeros((F, S_ALL, 2), np.float32)
        nout = np.zeros((F, S_ALL), np.int32)
        for k in range(nf):
            by = src.get(keys[k]) or {}
            for sl, site_list in enumerate(sites):
                st = by.get(site_list[0])
                if st is None:
                    continue
                also = tuple(by[x] for x in site_list[1:]
                             if by.get(x) is not None)
                L = slot_legality(st, N_AX, also=also)
                sizes[k, sl], nout[k, sl] = L.sizes, L.n_out
                pair[k, sl], comp[k, sl], quant[k, sl] = L.pair, L.comp, L.quant

        rows = np.full((F, FACE_SLOTS, 3), -1, np.int32)
        rows[..., 2] = 0
        skips = np.zeros((F,), np.int32)
        for f in range(nf):
            key = jrand.PRNGKey(seed * 1000003 + n * 97 + f)
            # SLICED TO THE CONTRACTION BAND, like face_driver does: the policy
            # builds per-slot features for the three contraction slots, and the
            # join rows above were computed to prove they CAN be (and to mask
            # learned1/learned2 once the wire carries them), not to be fed in.
            _skip, row, *_ = pol.sample_face(
                feats, tables, key, f,
                jnp.asarray(pair[f][:FACE_SLOTS]),
                jnp.asarray(comp[f][:FACE_SLOTS]), jnp.asarray(1.0),
                face_context=ctx,
                face_sizes_f=jnp.asarray(sizes[f][:FACE_SLOTS]),
                face_quant_f=jnp.asarray(quant[f][:FACE_SLOTS]))
            for sl in range(FACE_SLOTS):
                if slots_on is not None and _SLOT_SITES[sl] not in slots_on:
                    continue
                op = int(row["op_type"][sl])
                if op == OP_NONE:
                    continue
                w = _row_to_wire(op, row["i"][sl], row["j"][sl], row["i"][sl],
                                 row["quant_dtype"][sl], nout[f, sl])
                if w is None:
                    continue
                requested[_OP_KIND[op]] += 1
                rows[f, sl] = w
        per_face = _face_dict_for_vertex(config, tk.ij, v, rows, skips)
        arm_face_counts()
        try:
            tk.ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()

    out = dict(envmod._PER_FACE_STATS)
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


def _walk(target, seed, slots_on=None):
    """One sampled plan on the PRODUCTION pair of graphs: the legality comes
    from the stream's tokenizer and the apply runs on a separate
    ``IncrementalJaxpr`` advanced with the same wire rows."""
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

    rows_hist = np.full((total_v, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    rows_hist[..., 2] = 0
    skips_hist = np.zeros((total_v, MAX_FACES), np.int32)

    envmod._PER_FACE_STATS.clear()
    requested = {k: 0 for k in KINDS}
    noops = {k: 0 for k in KINDS}

    for n in range(len(order)):
        v = int(order[n])
        sizes, quant, pair, comp, nout, nf = lf.face_slot_legality(
            order, specs, n, v, rows_hist, skips_hist)
        nf = int(nf)
        rows = rows_hist[n]
        skips = skips_hist[n]
        for f in range(nf):
            key = jrand.PRNGKey(seed * 1000003 + n * 97 + f)
            _skip, row, _lp, _ent, _ar, _, _ = pol.sample_face(
                feats, tables, key, f,
                jnp.asarray(pair[f]), jnp.asarray(comp[f]),
                jnp.asarray(1.0), face_context=ctx,
                face_sizes_f=jnp.asarray(sizes[f]),
                face_quant_f=jnp.asarray(quant[f]))
            for s in range(FACE_SLOTS):
                if slots_on is not None and _SLOT_SITES[s] not in slots_on:
                    continue
                op = int(row["op_type"][s])
                if op == OP_NONE:
                    continue
                w = _row_to_wire(op, row["i"][s], row["j"][s], row["i"][s],
                                 row["quant_dtype"][s], nout[f, s])
                if w is None:
                    # No wire form. The engine drops it too, so it is not a
                    # request the apply path ever sees.
                    continue
                requested[_OP_KIND[op]] += 1
                rows[f, s] = w
        per_face = _face_dict_for_vertex(config, ij, v, rows, skips)
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
# With the contraction slots UNARMED both masks are exactly right: nothing moves
# their tensor between the mask and the apply. With them armed both reject, at
# rates 5x apart, and both are dsnn-3qm.59 fault 2 by two different routes --
# see the xfail reasons below.
#
# A 3-SEED PROBE SAID learned1 WAS IMMUNE (0 of 354) AND IT WAS WRONG. The
# 5-seed test caught it. Recorded because the lesson is the useful part: a rate
# this low is invisible in a short run, and the structural argument that
# produced the wrong prediction (the old edge is built by earlier eliminations,
# so this face cannot touch it) was incomplete -- it ignored sibling faces at
# the same vertex.
#
# Do not relax either xfail to a skip: a mask that clears an action the engine
# then refuses is the defect deliverable 3 exists to forbid.
# ==========================================================================
def _walk_join_slots(target, seed, arm):
    """Arm the JOIN slots (and optionally the contraction slots), each drawn
    against ITS OWN mask row, and return (requested, engine stats).

    Each slot is sampled with a separate head call against its own mask row.
    That is a PROBE, not the production path -- the rollout wire does not carry
    the join band yet -- but the MASK and the TENSOR are the production ones,
    which is what a rejection rate is about.

    ``--approx-add`` MUST ALREADY NAME A VALUE WIDE ENOUGH FOR ``arm`` (the
    caller sets it; see ``_join_totals``). Since 2026-09-11 arming slot 3 IS
    ``learned1`` and arming slot 4 IS ``learned2``: the width is not a knob a
    probe can turn independently of the configuration, and ``face_slot_sites``
    / ``wire_slots`` answer from the configuration alone.
    """
    from graphax import IncrementalPathTokenizer
    from alphagrad.approx.common.masks import slot_legality
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

    envmod._PER_FACE_STATS.clear()
    requested = {k: 0 for k in KINDS}
    for n in range(len(order)):
        v = int(order[n])
        keys = list(tk.ij.faces(v))
        src = lf._probe_faces(tk, v, keys, True, slots=True, stat="slot") or {}
        nf = min(len(keys), F)
        sizes = np.zeros((F, S_ALL, N_AX), np.int32)
        pair = np.zeros((F, S_ALL, N_AX, N_AX), np.float32)
        comp = np.zeros((F, S_ALL, N_AX), np.float32)
        quant = np.zeros((F, S_ALL, 2), np.float32)
        nout = np.zeros((F, S_ALL), np.int32)
        for k in range(nf):
            by = src.get(keys[k]) or {}
            for sl, site_list in enumerate(sites):
                st = by.get(site_list[0])
                if st is None:
                    continue
                L = slot_legality(st, N_AX)
                sizes[k, sl], nout[k, sl] = L.sizes, L.n_out
                pair[k, sl], comp[k, sl], quant[k, sl] = (
                    L.pair, L.comp, L.quant)

        rows = np.full((F, S_ALL, 3), -1, np.int32)
        rows[..., 2] = 0
        skips = np.zeros((F,), np.int32)
        for f in range(nf):
            key = jrand.PRNGKey(seed * 1000003 + n * 97 + f)
            for sl in arm:
                def _rep(a):
                    return jnp.asarray(a[f][sl][None].repeat(FACE_SLOTS, 0))
                _sk, row, *_ = pol.sample_face(
                    feats, tables, jrand.fold_in(key, sl), f,
                    _rep(pair), _rep(comp), jnp.asarray(1.0),
                    face_context=ctx, face_sizes_f=_rep(sizes),
                    face_quant_f=_rep(quant))
                op = int(row["op_type"][0])
                if op == OP_NONE:
                    continue
                w = _row_to_wire(op, row["i"][0], row["j"][0],
                                 row["i"][0], row["quant_dtype"][0],
                                 nout[f, sl])
                if w is None:
                    continue
                requested[_OP_KIND[op]] += 1
                rows[f, sl] = w
        per_face = _face_dict_for_vertex(config, tk.ij, v, rows, skips)
        arm_face_counts()
        try:
            tk.ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()
    out = dict(envmod._PER_FACE_STATS)
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
# STRICT xfails, so they flip the moment fault 2 is fixed.
@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 fault 2, intra-vertex, under "
                          "--approx-add learned1 (125 logits, 4 slots): an "
                          "earlier face's merge writes the edge a later face's "
                          "jr reads, so arming the contraction slots stales "
                          "learned1's mask within one elimination step. "
                          "Measured 3 of 594 on TLM over 5 seeds (1 Diag, 2 "
                          "Reduce); 0 of 27 with the contraction slots "
                          "unarmed.")
def test_learned1_rejects_nothing_with_the_contraction_slots_armed(tlm):
    req, skip = _join_totals(tlm, (0, 1, 2, 3))
    assert sum(skip.values()) == 0, (req, skip)


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 fault 2, one level further along, under "
                          "--approx-add learned2 (156 logits, 5 slots): the "
                          "SUMMED edge is new + old, so it carries this face's "
                          "own contraction approximations, and learned2's mask "
                          "is read from it BEFORE those rows are drawn. "
                          "Measured 20 of 741 on TLM over 5 seeds (9 Diag, 11 "
                          "Reduce); 0 of 181 with the contraction slots "
                          "unarmed.")
def test_learned2_rejects_nothing_with_the_contraction_slots_armed(tlm):
    req, skip = _join_totals(tlm, (0, 1, 2, 4))
    assert sum(skip.values()) == 0, (req, skip)


# --------------------------------------------------------------------------
# FAULT 2, STILL OPEN. Arming lhs and rhs makes `res:new` itself reject: 6 of
# 26 Diag and 4 of 40 Reduce on one graph (job 64637), against 0 of 27 and 0
# of 39 with `new` armed alone. `new` holds the PRODUCT of lhs and rhs, so
# approximating an operand changes the tensor `new`'s mask was read from -- and
# the head draws all three slots from ONE distribution, so no mask computed
# before the draw can know the operands' choices. NOT fixable at the mask as
# structured: it needs the slots decoded in APPLY order (re-probe `new` once
# lhs/rhs are sampled) or the `new` slot turned into a graphax CHOOSER
# (core.py:1328 -- the hook is handed the live operand and returns the action
# it picked).
#
# STRICT xfail, so it flips to a failure the moment fault 2 is fixed. Do not
# relax it to a skip: a mask that clears an action the engine then refuses is
# the defect dsnn-3qm.59 deliverable 3 exists to forbid.
# --------------------------------------------------------------------------
@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 fault 2: arming lhs/rhs moves the "
                          "contraction result, so the `new` mask read off the "
                          "un-approximated product clears rows the engine "
                          "refuses (6 of 26 Diag, 4 of 40 Reduce on TLM)")
def test_tlm_every_slot_rejects_nothing_on_one_graph(tlm):
    _assert_no_rejection(_walk_one_graph, tlm, "tlm one-graph all slots")


@pytest.mark.xfail(strict=True,
                   reason="dsnn-3qm.59 fault 2: same operand-coupling defect "
                          "on nn256")
def test_nn256_every_slot_rejects_nothing_on_one_graph(nn256):
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
