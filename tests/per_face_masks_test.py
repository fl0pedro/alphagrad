"""``--per-face-masks``: per-FACE legality for EVERY approximation.

WHAT IS UNDER TEST. Approximation legality is decided at three layers that
disagree about granularity -- nominal/per-vertex (``decode_vertex_rule_specs``),
the oracle's face probe (``face_masks``), and the live operand at apply time
(``rule_is_legal``) -- and only the last one decides. These tests pin the two
halves of expressing the first two in the last one's terms:

  * the SIZES the head reads (``face_dim_sizes``), which make its hardcoded
    ``factor = gcd(N_i, N_j)`` legal BY CONSTRUCTION rather than by luck;
  * the apply-time PROJECTION, generalised from DIAG to COMPRESS and QUANT.

and the two properties whose violation is silent: flag-off bit-identity, and
sampling == replay (the PPO ratio at epoch 0).

The fixture is the same split-gcd graph ``diag_per_face_test`` uses, because it
is the situation a per-VERTEX quantity provably cannot serve: the two faces of
one vertex have gcds 4 and 3 and therefore NO shared legal factor > 1.
"""
import contextlib
import math
import os

os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from graphax import inline_call_primitives
from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.common import masks as M
from alphagrad.approx.common.masks import (
    FACE_QUANT_DTYPES, LiveVertexMaskOracle, compress_valid_mask,
    diag_pair_factor_space, dim_logical_sizes, face_legal_actions,
    legal_quant_actions, make_live_masked_hook, masked_micro_chooser,
    project_compress_to_face, project_quant_to_face, project_rule_to_face,
    quant_is_noop, quant_valid_mask, rule_is_idempotent_noop, rule_is_legal,
    set_diag_per_face, set_per_face_masks)

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8


# --------------------------------------------------------------------------
#   e = A @ x        A: (12, 6)   ->  e: (12,)
#   return B @ e, C @ e           B: (8, 12), C: (9, 12)
#
# gcd(8, 12) = 4 and gcd(9, 12) = 3: the two faces of ``e`` share no legal
# factor > 1, so ONE per-vertex factor cannot serve both.
# --------------------------------------------------------------------------
_A = jnp.asarray(np.linspace(0.1, 0.9, 12 * 6, dtype=np.float32).reshape(12, 6))
_B = jnp.asarray(np.linspace(0.2, 0.8, 8 * 12, dtype=np.float32).reshape(8, 12))
_C = jnp.asarray(np.linspace(0.3, 0.7, 9 * 12, dtype=np.float32).reshape(9, 12))
_X = jnp.asarray(np.linspace(0.1, 0.9, 6, dtype=np.float32))


def _split_gcd(x):
    e = _A @ x
    return _B @ e, _C @ e


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_MLP_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)


def _oracle_for(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    o = LiveVertexMaskOracle(closed.jaxpr, list(closed.literals), list(xs),
                             tuple(range(len(xs))), max_axes=N_AX)
    return closed, o


def _all_faces(fn, xs):
    """``(oracle, [(vertex, face_index, live SparseTensor), ...])``."""
    closed, o = _oracle_for(fn, xs)
    out = []
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        try:
            faces = o.probe_faces(v, approx=True)
        except Exception:                                  # pragma: no cover
            faces = []
        for k, st in enumerate(faces):
            out.append((v, k, st))
        o.advance(v, rules=())
    return o, out


@contextlib.contextmanager
def _in_trace(o):
    from jax._src import core as _jcore
    with _jcore.set_current_trace(o._incrs[True].trace):
        yield


@contextlib.contextmanager
def _flag(on, **kw):
    try:
        set_per_face_masks(bool(on), **kw)
        yield
    finally:
        set_per_face_masks(False)


# ==========================================================================
# 1. FLAG-OFF BIT-IDENTITY
# ==========================================================================
def test_default_is_off():
    assert not M.per_face_masks_enabled()


def test_face_masks_is_unchanged_by_the_new_combined_method():
    """``face_masks`` now delegates to ``face_masks_and_sizes(per_face=False)``.
    The delegation must be byte-for-byte, and the flag-off path must produce
    NO sizes and NO quant bits (they are not stored, so they must not exist)."""
    closed, o = _oracle_for(_split_gcd, [_X])
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        fp, fc, n = o.face_masks(v, 8)
        fp2, fc2, sz2, q2, n2 = o.face_masks_and_sizes(v, 8, per_face=False)
        assert n == n2
        assert np.array_equal(fp, fp2) and np.array_equal(fc, fc2)
        assert not sz2.any() and not q2.any()


def test_flag_off_hook_is_bit_identical_for_every_kind():
    """With the flag off an illegal COMPRESS / QUANT is still DROPPED, with the
    historical counters and nothing else -- no projection, no noop split."""
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    seen = 0
    for _v, _k, st in faces:
        ndim = int(getattr(getattr(st, "val", None), "ndim", 0))
        rules = (Compress((ndim + 3,), "mean"), Quant(dtype="float32"))
        s: dict = {}
        with _in_trace(_o):
            out = make_live_masked_hook(rules, max_dims=N_AX, max_axes=N_AX,
                                        stats=s)(st)
        assert "repaired_compress" not in s
        assert "skipped_compress_noop" not in s
        assert "skipped_quant_noop" not in s
        assert s.get("skipped_raised", 0) == 0
        assert out is not None
        seen += 1
    assert seen, "fixture produced no faces"


def test_masked_micro_chooser_default_still_excludes_quant():
    """The pre-existing wrapper's action list must not change shape: its one
    live user (tests/face_slot_chooser_test) enumerates DIAG + COMPRESS."""
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        got = []
        masked_micro_chooser(lambda _st, acts: got.extend(acts))(st)
        assert not any(isinstance(a, Quant) for a in got)
        withq = face_legal_actions(st, max_dims=N_AX, max_axes=N_AX,
                                   quant_dtypes=FACE_QUANT_DTYPES)
        assert len(withq) >= len(got)
        return
    pytest.skip("fixture produced no faces")


# ==========================================================================
# 2. THE FACES HAVE DIFFERENT LEGAL SETS -- and per-vertex admits an illegal
#    rule where per-face does not.
# ==========================================================================
def test_per_vertex_factor_is_illegal_on_a_face_that_per_face_serves():
    """THE CORE CLAIM.

    On the split-gcd vertex the head's ``factor = gcd(N_i, N_j)`` taken over
    the vertex's NOMINAL sizes is a single number, and the two faces admit
    disjoint factor sets (divisors of 4 vs divisors of 3, both > 1). So at
    least one face must reject the per-vertex factor -- while the SAME pair
    with that face's OWN sizes is legal on it.
    """
    closed, o = _oracle_for(_split_gcd, [_X])
    found = False
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        fp, fc, sizes, quant, n = o.face_masks_and_sizes(v, 8, per_face=True)
        if n < 2:
            o.advance(v, rules=())
            continue
        faces = o.probe_faces(v, approx=True)
        # Every pair the PER-FACE mask admits must be legal on its own face at
        # the factor the head derives from that face's own sizes.
        for k in range(min(n, len(faces))):
            st = faces[k]
            for i in range(N_AX):
                for j in range(N_AX):
                    if not fp[k, i, j]:
                        continue
                    g = math.gcd(int(sizes[k, i]), int(sizes[k, j]))
                    assert g > 1
                    assert rule_is_legal(st, Diag(i, j, g), max_dims=N_AX), (
                        f"v{v} face{k} pair({i},{j}) factor {g} rejected by "
                        "the live operand -- the mask and the head disagree")
                    # ... and the OTHER face's factor is a different number,
                    # which is what a per-vertex factor cannot express.
                    for k2 in range(min(n, len(faces))):
                        if k2 == k:
                            continue
                        g2 = math.gcd(int(sizes[k2, i]), int(sizes[k2, j]))
                        if g2 > 1 and g2 != g:
                            found = True
        o.advance(v, rules=())
    assert found, (
        "fixture did not produce two faces with different legal factor sets")


def test_face_dim_sizes_are_the_diag_numbering_not_val_shape():
    """``face_dim_sizes`` must be indexed the way ``Diag(i, j)`` is.

    ``face_features`` keeps ``val.shape`` -- the PHYSICAL axes -- which is a
    different list, so feeding it to the head would align with nothing. This
    pins that the two are genuinely different and that the new one matches
    ``dim_logical_sizes`` of the very tensor the mask read.
    """
    differed = 0
    checked = 0
    for fn, xs in ((_split_gcd, [_X]), (_mlp, list(_MLP_ARGS))):
        closed, o = _oracle_for(fn, xs)
        for v in range(1, len(closed.jaxpr.eqns) + 1):
            faces = o.probe_faces(v, approx=True)
            sizes, n = o.face_dim_sizes(v, 8)
            vshapes, _nv = o.face_features(v, 8)
            for k in range(min(n, len(faces))):
                # THE INVARIANT: what the head is handed IS what the mask read.
                assert np.array_equal(sizes[k],
                                      dim_logical_sizes(faces[k], N_AX))
                checked += 1
                if not np.array_equal(sizes[k], vshapes[k]):
                    differed += 1
            o.advance(v, rules=())
    assert checked, "no faces probed"
    assert differed, (
        "logical dim sizes never differed from val.shape on EITHER fixture -- "
        "the distinction this method exists for was not exercised")


# ==========================================================================
# 3. QUANT: the operator that had no per-face mask at all
# ==========================================================================
def test_quant_to_the_operands_own_dtype_is_a_noop():
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        val = getattr(st, "val", None)
        if val is None:
            assert quant_is_noop(st, "float32")
            continue
        cur = str(val.dtype)
        assert quant_is_noop(st, cur)
        assert not quant_valid_mask(st, (cur,))[0]
        assert rule_is_idempotent_noop(st, Quant(dtype=cur))
        # ... and it is still LEGAL, which is exactly why it used to be
        # counted as applied.
        assert rule_is_legal(st, Quant(dtype=cur))


def test_quant_noop_is_skipped_and_labelled_when_the_flag_is_on():
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    checked = 0
    for _v, _k, st in faces:
        val = getattr(st, "val", None)
        if val is None:
            continue
        cur = str(val.dtype)
        s_off: dict = {}
        with _in_trace(_o):
            make_live_masked_hook((Quant(dtype=cur),), max_dims=N_AX,
                                  max_axes=N_AX, stats=s_off)(st)
        assert s_off.get("applied_quant") == 1, "flag off: counted as applied"
        s_on: dict = {}
        with _flag(True), _in_trace(_o):
            out = make_live_masked_hook((Quant(dtype=cur),), max_dims=N_AX,
                                        max_axes=N_AX, stats=s_on)(st)
        assert out is st, "an idempotent QUANT must leave the operand alone"
        assert s_on.get("applied_quant") is None
        assert s_on.get("skipped_quant") == 1
        assert s_on.get("skipped_quant_noop") == 1
        assert s_on.get("skipped_raised", 0) == 0
        checked += 1
    assert checked, "fixture produced no materialised-val face"


def test_a_real_cast_is_still_applied_under_the_flag():
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        val = getattr(st, "val", None)
        if val is None or str(val.dtype) == "bfloat16":
            continue
        s: dict = {}
        with _flag(True), _in_trace(_o):
            out = make_live_masked_hook((Quant(dtype="bfloat16"),),
                                        max_dims=N_AX, max_axes=N_AX,
                                        stats=s)(st)
        assert s.get("applied_quant") == 1
        assert out is not st
        assert legal_quant_actions(st, ("bfloat16",))
        return
    pytest.skip("fixture produced no quantisable face")


def test_quant_projection_never_invents_a_different_dtype():
    """A DIAG snapped to a legal factor is the same approximation in the
    operand's coordinates; a QUANT snapped to a different dtype would be a
    DIFFERENT approximation. The projection must refuse."""
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        val = getattr(st, "val", None)
        if val is None:
            continue
        cur = str(val.dtype)
        assert project_quant_to_face(st, Quant(dtype=cur)) is None
        alt = project_rule_to_face(st, Quant(dtype=cur), max_dims=N_AX,
                                   max_axes=N_AX)
        assert alt is None
        return
    pytest.skip("fixture produced no materialised-val face")


# ==========================================================================
# 4. COMPRESS: projected, not dropped silently
# ==========================================================================
def test_out_of_range_compress_axis_snaps_to_a_legal_one():
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    checked = 0
    for _v, _k, st in faces:
        mask = compress_valid_mask(st, N_AX)
        if not mask.any():
            continue
        bad = int(np.max(np.nonzero(mask)[0])) + 2      # past the live ndim
        if bad >= N_AX:
            continue
        req = Compress((bad,), "mean")
        assert not rule_is_legal(st, req, max_dims=N_AX, max_axes=N_AX)

        s_off: dict = {}
        with _in_trace(_o):
            off = make_live_masked_hook((req,), max_dims=N_AX, max_axes=N_AX,
                                        stats=s_off)(st)
        assert off is st and s_off.get("skipped_compress") == 1
        assert "repaired_compress" not in s_off

        s_on: dict = {}
        with _flag(True), _in_trace(_o):
            on = make_live_masked_hook((req,), max_dims=N_AX, max_axes=N_AX,
                                       stats=s_on)(st)
        alt = project_compress_to_face(st, req, max_axes=N_AX)
        assert alt is not None and rule_is_legal(st, alt, max_dims=N_AX,
                                                 max_axes=N_AX)
        assert s_on.get("applied_compress") == 1
        assert s_on.get("repaired_compress") == 1
        assert s_on.get("skipped_raised", 0) == 0
        assert on is not st
        checked += 1
    assert checked, "no repairable COMPRESS request found in the fixture"


def test_compress_repair_can_be_turned_off():
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        mask = compress_valid_mask(st, N_AX)
        if not mask.any():
            continue
        bad = int(np.max(np.nonzero(mask)[0])) + 2
        if bad >= N_AX:
            continue
        assert project_compress_to_face(st, Compress((bad,), "mean"),
                                        max_axes=N_AX,
                                        repair_axis=False) is None
        return
    pytest.skip("no out-of-range axis available")


def test_compress_slot_mask_is_the_bound_apply_compress_enforces():
    """THE BUG THE PROJECTION FOUND.

    ``Compress.axes`` are CANONICAL SLOTS, and ``apply_compress`` raises
    "entry {a} out of range" against ``len(canonical_axis_order(st))``.
    ``compress_valid_mask`` bounds by ``val.ndim`` instead. As long as the
    policy only ever asked for axes the vertex-level screen had cleared, the
    difference was invisible; the moment the per-face projection started
    snapping axes to the top of the ``val.ndim`` range it produced this
    module's first non-zero ``skipped_raised`` (3 on this fixture).
    """
    from graphax.sparse.micro_actions import canonical_axis_order
    from alphagrad.approx.common.masks import compress_slot_mask

    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    for _v, _k, st in faces:
        if getattr(st, "val", None) is None:
            continue
        n_slots = len(canonical_axis_order(st))
        m = compress_slot_mask(st, N_AX)
        for a in range(N_AX):
            if m[a]:
                assert a < n_slots, (
                    f"slot {a} admitted but apply_compress only accepts "
                    f"< {n_slots}")


def test_projected_compress_never_raises():
    """Every repair must be applicable. This is the property that keeps
    ``skipped_raised`` at 0 -- it is asserted directly, not inferred."""
    from alphagrad.approx.common.masks import compress_slot_mask

    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    tried = 0
    for _v, _k, st in faces:
        for bad in range(N_AX):
            if compress_slot_mask(st, N_AX)[bad]:
                continue
            alt = project_compress_to_face(st, Compress((bad,), "mean"),
                                           max_axes=N_AX, repair_axis=True)
            if alt is None:
                continue
            with _in_trace(_o):
                from graphax.sparse.micro_actions import apply_compress
                out = apply_compress(st, alt)      # must not raise
            assert out is not None
            tried += 1
    assert tried, "no COMPRESS repair was exercised"


def test_compress_on_a_structure_only_edge_is_a_noop_not_a_failure():
    """``val is None`` -- graphax accepts the Compress and does nothing."""
    class _Fake:
        val = None
    assert rule_is_idempotent_noop(_Fake(), Compress((0,), "mean"))
    assert project_compress_to_face(_Fake(), Compress((0,), "mean")) is None


# ==========================================================================
# 5. NOTHING EVER RAISES  (skipped_raised stays 0)
# ==========================================================================
@pytest.mark.parametrize("on", [False, True])
def test_skipped_raised_stays_zero_under_an_absurd_rule_set(on):
    _o, faces = _all_faces(_mlp, list(_MLP_ARGS))
    rules = (Diag(0, 1, 7919), Compress((7,), "mean"), Quant(dtype="float32"),
             Diag(3, 5, 6), Compress((0,), "max"))
    total: dict = {}
    for _v, _k, st in faces:
        with _flag(on), _in_trace(_o):
            make_live_masked_hook(rules, max_dims=N_AX, max_axes=N_AX,
                                  stats=total)(st)
    assert total.get("skipped_raised", 0) == 0
    assert total, "the hook was never consulted"


# ==========================================================================
# 6. SAMPLING == REPLAY (the PPO ratio at epoch 0)
# ==========================================================================
def test_sample_equals_evaluate_with_per_face_sizes_and_quant():
    """The loss-side mirror must carry ``face_sizes`` AND ``face_quant``.

    Both enter the MASKS, so dropping either scores a different distribution
    and the ratio silently leaves 1 -- with no error anywhere.
    """
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

    E, F = 32, 6
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F,
                            key=jrand.PRNGKey(0))
    n = 6
    sz = jnp.asarray([8, 8, 4, 16, 6, 4], jnp.int32)
    feats = AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))
    fpv = jnp.ones((F, n, n), jnp.float32)
    fcv = jnp.ones((F, n), jnp.float32)
    fval = jnp.concatenate(
        [jnp.ones((4,)), jnp.zeros((F - 4,))]).astype(jnp.float32)
    # Deliberately DIFFERENT per face, and one face with QUANT masked off.
    fsz = jnp.asarray(np.array([[4, 12, 6, 3, 0, 0],
                                [9, 12, 3, 0, 0, 0],
                                [8, 8, 8, 8, 2, 2],
                                [6, 4, 2, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]], np.int32))
    fq = jnp.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], jnp.float32)

    out = pol.sample(None, feats, tables, jrand.PRNGKey(3), fpv, fcv, fval,
                     face_sizes=fsz, face_quant=fq)
    fa, logp, ent, arity = out[0], out[1], out[2], out[3]
    logp2, ent2, arity2 = pol.evaluate(
        None, feats, tables, fa, fpv, fcv, fval,
        face_sizes=fsz, face_quant=fq)[:3]
    assert float(jnp.abs(logp - logp2)) < 1e-5, (
        f"ratio != 1 at epoch 0: {float(logp)} vs {float(logp2)}")
    assert float(jnp.abs(ent - ent2)) < 1e-5
    assert float(arity) == float(arity2)

    # The mirror is LOAD-BEARING: replaying WITHOUT the per-face arrays must
    # score a different number, or these tests would prove nothing.
    logp_blind = pol.evaluate(None, feats, tables, fa, fpv, fcv, fval)[0]
    assert float(jnp.abs(logp - logp_blind)) > 1e-6, (
        "dropping face_sizes/face_quant changed nothing -- the plumbing is "
        "not actually reaching the masks")


def test_flag_off_policy_path_is_unchanged():
    """``face_sizes=None`` must reproduce the pre-flag policy exactly."""
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

    E, F, n = 32, 5, 6
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F, key=jrand.PRNGKey(0))
    sz = jnp.asarray([8, 8, 4, 16, 6, 4], jnp.int32)
    feats = AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))
    fpv = jnp.ones((F, n, n), jnp.float32)
    fcv = jnp.ones((F, n), jnp.float32)
    fval = jnp.ones((F,), jnp.float32)
    a = pol.sample(None, feats, tables, jrand.PRNGKey(5), fpv, fcv, fval)
    b = pol.sample(None, feats, tables, jrand.PRNGKey(5), fpv, fcv, fval,
                   face_sizes=None, face_quant=None)
    assert float(a[1]) == float(b[1])
    assert bool(jnp.all(a[0].op_type == b[0].op_type))
    assert bool(jnp.all(a[0].factor == b[0].factor))
