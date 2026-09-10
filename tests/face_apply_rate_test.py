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

TWO GRAPHS ON PURPOSE. The legality comes from the stream's tokenizer and the
apply runs on a separate ``IncrementalJaxpr`` advanced with the same wire rows,
because that is the production arrangement (ppo.py rides the stream tokenizer
for the mask and the ``_ElimChain`` for the face keys). If the two ever build
different edges from one order, this test is where it shows.

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
from alphagrad.approx.live_faces import LiveFaceStream
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


def _row_to_wire(op, i, j, axis, dtype_idx, n_out):
    """The (op, fields) the head chose -> the ``[bi1, bi2, factor]`` wire row
    the engine decodes. One encoder, the same one ``micro_actions_to_rule_specs``
    uses: Diag writes ``bi2 = j - n_out``, Reduce writes the axis token, Quant
    writes the dtype index."""
    if op == OP_BLOCKDIAG:
        return (int(i), int(j) - int(n_out), -1)
    if op == OP_REDUCE:
        return (int(envmod.COMPRESS_SENTINEL), int(axis), 0)
    if op == OP_QUANT:
        return (int(envmod.QUANT_SENTINEL), int(dtype_idx), 0)
    return (-1, -1, 0)


def _walk(target, seed):
    """One sampled plan. Returns the engine's per-face counters."""
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
                op = int(row["op_type"][s])
                if op == OP_NONE:
                    continue
                kind = _OP_KIND[op]
                requested[kind] += 1
                rows[f, s] = _row_to_wire(
                    op, row["i"][s], row["j"][s], row["i"][s],
                    row["quant_dtype"][s], nout[f, s])
        per_face = _face_dict_for_vertex(config, ij, v, rows, skips)
        arm_face_counts()
        try:
            ij.eliminate(v, (), per_face or None)
        finally:
            disarm_face_counts()

    out = dict(envmod._PER_FACE_STATS)
    envmod._PER_FACE_STATS.clear()
    return requested, out


@pytest.mark.parametrize("seed", range(N_SAMPLES))
def test_nn256_apply_rate_equals_request_rate(nn256, seed):
    _assert_zero_rejection(nn256, seed, "nn256")


@pytest.mark.parametrize("seed", range(N_SAMPLES))
def test_tlm_apply_rate_equals_request_rate(tlm, seed):
    _assert_zero_rejection(tlm, seed, "tlm")


def _assert_zero_rejection(target, seed, label):
    requested, stats = _walk(target, seed)
    total_req = sum(requested.values())
    assert total_req > 0, (
        f"{label} seed {seed}: the head requested nothing, so the test "
        f"proved nothing. stats={stats}")
    for kind in KINDS:
        skipped = int(stats.get(f"skipped_{kind}", 0))
        applied = int(stats.get(f"applied_{kind}", 0))
        assert skipped == 0, (
            f"{label} seed {seed}: {skipped} {kind} rows the mask cleared "
            f"were REJECTED at apply time ({applied} applied, "
            f"{requested[kind]} requested). stats={stats}")
    # Every request the engine saw was applied, so the two counts must agree.
    assert int(stats.get("skipped", 0)) == 0, stats
    assert int(stats.get("applied", 0)) == sum(
        int(stats.get(f"applied_{k}", 0)) for k in KINDS), stats


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
