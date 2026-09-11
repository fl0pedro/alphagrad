#!/usr/bin/env python3
"""THE FOUR USES OF THE FACE ACTION RECORD MUST AGREE, and this is the proof.

One face decision is consumed in four places:

1. ``face_action.FaceAction`` -- the fields ``UnifiedFacePolicy.sample`` emits;
2. ``ppo.Trajectory`` -- what the rollout STORES;
3. ``ppo.TrainBatch`` -> ``Agent._face_replay`` -> ``evaluate_face`` -- what PPO
   RE-SCORES;
4. ``Agent.to_env_action_dynamic`` -- where the fields become the env's
   ``(F, S, 3)`` wire rows plus the per-face channels.

Use 3 is the dangerous one. A field that ``sample`` draws and ``evaluate`` does
not read (or the reverse) raises NOTHING: the PPO ratio simply stops being 1 at
epoch 0, and every number the run reports stays plausible.

HOW THIS MODULE CANNOT BE SATISFIED BY EDITING A LIST
-----------------------------------------------------
Nothing here enumerates the fields. Every expectation is computed from
``face_action.FACE_ACTION_FIELDS`` -- the declaration -- at each
``--approx-add`` width:

* use 1 is checked by asking the POLICY for a record and comparing it field by
  field against ``face_action.shapes(mode, F)``;
* uses 2 and 3 are checked STRUCTURALLY: the carriers hold the record as ONE
  leaf, so there is no list of names to go stale, and ``check_carrier`` refuses
  a carrier that grew the flat ``face_*`` leaves back;
* use 3's reader is checked with an ACCESS RECORDER: ``evaluate_face`` is run
  against a proxy that logs every field it touches, and the set it touched must
  equal ``face_action.scored_names(mode)``. A declared scored field nobody reads
  therefore FAILS, and so does a derived field somebody reads;
* use 4 is checked with the same recorder on ``to_env_action_dynamic``, against
  ``translator_names()`` plus the per-face channels;
* and :func:`test_a_NEW_field_in_the_declaration_breaks_every_use` injects a
  field into the declaration and requires the checks above to FAIL. That is the
  actual deliverable: the test proves the mechanism, not a snapshot of today's
  ten fields.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from alphagrad.approx import face_action as REC
from alphagrad.approx.face_action import FaceAction
from alphagrad.approx.env import (
    AXIS_FEATURE_DIM, MAX_AXES_PER_VERTEX, StepAction, wire_slots,
)
from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures, MicroAction, OP_END,
    precompute_factor_tables,
)
from alphagrad.approx.unified_face_head import FACE_SLOTS, head_layout
from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

MODES = list(REC.ALL_MODES)
E, F = 32, 4
N_AX = 6


# --------------------------------------------------------------- fixtures
def _feats(sizes=(8, 8, 4, 16, 6, 4)):
    sz = jnp.asarray(sizes, jnp.int32)
    n = len(sizes)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _policy(mode, seed=0):
    return UnifiedFacePolicy(E, num_heads=2, max_faces=F,
                             key=jrand.PRNGKey(seed), approx_add=mode)


def _env(mode):
    """``(policy, features, tables, pair, comp, valid)`` at one width."""
    pol = _policy(mode)
    f = _feats()
    n = f.size.shape[0]
    return (pol, f, precompute_factor_tables(64),
            jnp.ones((F, n, n), jnp.float32), jnp.ones((F, n), jnp.float32),
            jnp.concatenate([jnp.ones((2,)), jnp.zeros((F - 2,))]
                            ).astype(jnp.float32))


def _one_face_record(pol, f, tables, pair, comp, valid, key=11):
    """``sample_face`` for face 0, lifted to a ONE-FACE action record.

    Built from the row's OWN keys, so a field the wire row gained arrives here
    without this helper being edited -- which is the property under test.
    """
    sk, row, *_ = pol.sample_face(f, tables, jrand.PRNGKey(key), 0,
                                  pair[0], comp[0], valid[0])
    return FaceAction(skip=jnp.asarray(sk)[None],
                      **{k: jnp.asarray(v)[None] for k, v in row.items()})


class _Recorder:
    """A record proxy that logs which fields a consumer reads.

    This is what makes use 3 and use 4 checkable WITHOUT restating the field
    list: the expected access set is computed from the declaration.
    """

    def __init__(self, fa):
        object.__setattr__(self, "_fa", fa)
        object.__setattr__(self, "seen", set())

    def __getattr__(self, name):
        if name.startswith("_") or name == "seen":
            raise AttributeError(name)
        object.__getattribute__(self, "seen").add(name)
        return getattr(object.__getattribute__(self, "_fa"), name)


# ============================================================= the record
def test_the_record_IS_the_declaration():
    """``FaceAction`` is generated from ``FACE_ACTION_FIELDS``, in order."""
    assert FaceAction._fields == REC.names(), (
        FaceAction._fields, REC.names())
    # Width-restricted fields default to None -- "this decision has no field",
    # never a value the head did not draw.
    opt = [f.name for f in REC.FACE_ACTION_FIELDS if f.modes is not None]
    assert FaceAction._field_defaults == {k: None for k in opt}


@pytest.mark.parametrize("mode", MODES)
def test_the_slot_axis_is_the_HEADS_width_not_FACE_SLOTS(mode, monkeypatch):
    """One number: the declaration, the head and the wire all read one table."""
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    assert REC.n_slots(mode) == head_layout(mode).n_slots
    assert REC.n_slots(mode) == wire_slots()
    assert REC.n_slots(mode) >= FACE_SLOTS
    # FACE_SLOTS keeps meaning the CONTRACTION band, so `range(FACE_SLOTS)`
    # loops stay right at every width. It is NEVER the shape.
    assert FACE_SLOTS == 3
    for f in REC.fields(mode):
        want = ((F, REC.n_slots(mode)) if f.per_slot else (F,)) + f.tail
        assert f.shape(mode, F) == want, (mode, f.name)


# ================================================================== use 1
@pytest.mark.parametrize("mode", MODES)
def test_use1_sample_emits_EXACTLY_the_declared_record(mode):
    pol, f, tables, pair, comp, valid = _env(mode)
    fa, *_ = pol.sample(None, f, tables, jrand.PRNGKey(3), pair, comp, valid)
    # The validator is itself derived; this line is the whole check, and it
    # raises naming the field and the width.
    REC.check(fa, mode, F, where=f"sample at --approx-add {mode}")
    for fld in REC.FACE_ACTION_FIELDS:
        got = getattr(fa, fld.name)
        if fld.present(mode):
            assert got is not None, (mode, fld.name)
            assert tuple(got.shape) == fld.shape(mode, F)
            assert np.dtype(got.dtype) == fld.np_dtype
        else:
            assert got is None, (mode, fld.name)


# ============================================================ uses 2 and 3
def test_uses2and3_carry_the_record_WHOLE():
    """The carriers hold ONE leaf, so there is no list of names to go stale.

    Imported lazily: ppo is 13k lines and the declaration must not need it.
    """
    from alphagrad.approx.ppo import Trajectory, TrainBatch
    REC.check_carrier(Trajectory)
    REC.check_carrier(TrainBatch)
    for cls in (Trajectory, TrainBatch):
        for fld in REC.FACE_ACTION_FIELDS:
            assert fld.carrier not in cls._fields, (cls.__name__, fld.carrier)
        assert "face_action" in cls._fields


@pytest.mark.parametrize("mode", MODES)
def test_use3_evaluate_reads_every_SCORED_field_and_no_DERIVED_one(mode):
    """THE test this module exists for.

    ``evaluate_face`` must read back exactly the fields the head DREW. A
    declared scored field it does not read is a variable ``sample`` drew and
    the loss ignores; a derived field it DOES read is a variable the loss
    scores that nothing decided. Both move the PPO ratio off 1 silently.
    """
    pol, f, tables, pair, comp, valid = _env(mode)
    fa = _one_face_record(pol, f, tables, pair, comp, valid)
    rec = _Recorder(fa)
    pol.evaluate_face(f, tables, rec, 0, pair[0], comp[0], valid[0])
    assert rec.seen == set(REC.scored_names(mode)), (
        mode, sorted(rec.seen), sorted(REC.scored_names(mode)))
    assert not (rec.seen & set(REC.derived_names(mode)))


@pytest.mark.parametrize("mode", MODES)
def test_use3_a_dropped_field_RAISES_instead_of_being_defaulted(mode):
    """Every scored field, deleted one at a time, must raise.

    The old failure was the opposite: a dropped field became a default, the
    replay scored it, and nothing complained.
    """
    pol, f, tables, pair, comp, valid = _env(mode)
    fa = _one_face_record(pol, f, tables, pair, comp, valid)
    for name in REC.scored_names(mode):
        broken = fa._replace(**{name: None})
        with pytest.raises((ValueError, TypeError, AttributeError)):
            REC.check(broken, mode, 1, where="dropped " + name)
        # And the reader itself must not substitute a value for the per-FACE
        # decisions it cannot re-derive.
        if not REC.field(name).per_slot and name != "skip":
            with pytest.raises(ValueError):
                pol.evaluate_face(f, tables, broken, 0,
                                  pair[0], comp[0], valid[0])


# ================================================================== use 4
@pytest.mark.parametrize("mode", MODES)
def test_use4_the_wire_takes_EXACTLY_the_declared_translator_fields(mode,
                                                                   monkeypatch):
    """``to_env_action_dynamic`` reads the translator fields plus the per-face
    channels, and nothing else.

    It uses no ``self``, so it is called unbound -- this test costs no Agent.
    """
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    from alphagrad.approx.ppo import Agent
    pol, f, tables, pair, comp, valid = _env(mode)
    fa = _one_face_record(pol, f, tables, pair, comp, valid)
    rec = _Recorder(fa)
    zero = jnp.zeros((1,), jnp.int32)
    act = MicroAction(
        op_type=jnp.full((1,), OP_END, jnp.int32), i=zero, j=zero,
        exponents=jnp.zeros((1, 9), jnp.int32), factor=zero,
        compress_kind=zero, quant_dtype=zero,
        quant_scale_sign=jnp.ones((1,), jnp.int32),
        quant_scale_frac=jnp.zeros((1,), jnp.float32))
    ax = jnp.zeros((2, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), jnp.int32)
    sa = Agent.to_env_action_dynamic(None, 0, act, ax, face_action=rec)
    want = set(REC.translator_names()) | set(REC.per_face_names(mode))
    assert rec.seen == want, (mode, sorted(rec.seen), sorted(want))
    assert isinstance(sa, StepAction)
    assert tuple(sa.face_rows.shape) == (1, REC.n_slots(mode), 3)
    # The per-face channels exist iff the width declares them.
    assert (sa.face_join is None) == ("join" not in REC.names(mode))


@pytest.mark.parametrize("mode", MODES)
def test_use4_a_MIS_SLOTTED_row_array_raises(mode, monkeypatch):
    """A wire row array at the wrong width must RAISE, not be sliced or padded.

    A narrow array would drop a slot the head drew and scored; a wide one
    would apply a row the head has no logits for.
    """
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    from alphagrad.approx.env import wire_slots_of_rows
    S = REC.n_slots(mode)
    assert wire_slots_of_rows(np.full((F, S, 3), -1, np.int32)) == S
    for bad in (S - 1, S + 1):
        if bad < 1:
            continue
        with pytest.raises(ValueError):
            wire_slots_of_rows(np.full((F, bad, 3), -1, np.int32))
    # The record's own validator says the same thing about the record.
    with pytest.raises(ValueError):
        REC.check(REC.zeros(mode, F)._replace(
            op_type=jnp.zeros((F, S + 1), jnp.int32)), mode, F)


# =================================================== the codec is a PAIR
@pytest.mark.parametrize("mode", MODES)
def test_the_codec_round_trips_every_field_including_the_DERIVED_ones(mode):
    """``FaceFields -> _wire_row -> _fields_of -> _wire_row`` is the identity.

    The derived fields (``factor`` / ``exponents``, the quant-scale constants)
    are NOT read back by the scorer, so the only thing that can keep them
    honest is that the replay RE-DERIVES them bit for bit. That is what this
    asserts, per field, from the declaration.
    """
    pol, f, tables, pair, comp, valid = _env(mode)
    for seed in range(8):
        sk, row, *_ = pol.sample_face(f, tables, jrand.PRNGKey(500 + seed), 0,
                                      pair[0], comp[0], valid[0])
        fa = FaceAction(skip=jnp.asarray(sk)[None],
                        **{k: jnp.asarray(v)[None] for k, v in row.items()})
        back = pol._fields_of(fa, 0)
        row2 = pol._wire_row(back, pol._face_feats_1(f, None), tables)
        assert set(row2) == set(row)
        for k in row:
            assert np.array_equal(np.asarray(row[k]), np.asarray(row2[k])), (
                mode, seed, k)


# ====================================================== THE META-PROPERTY
def test_a_NEW_field_in_the_declaration_breaks_every_use(monkeypatch):
    """Declare a field nobody implements; every use must FAIL.

    This is the deliverable. It is why the tests above cannot be satisfied by
    editing a list: the list is the declaration, and adding to it here makes
    the same checks fail without any of them being touched.
    """
    extra = REC.FaceField("ghost", "int32", per_slot=True, fill=0,
                          role=REC.SCORED, translator=False,
                          doc="declared, implemented nowhere")
    monkeypatch.setattr(REC, "FACE_ACTION_FIELDS",
                        REC.FACE_ACTION_FIELDS + (extra,))
    monkeypatch.setitem(REC._BY_NAME, "ghost", extra)

    mode = "lossless"
    # use 1: the policy's record now lacks a declared field.
    pol, f, tables, pair, comp, valid = _env(mode)
    fa, *_ = pol.sample(None, f, tables, jrand.PRNGKey(3), pair, comp, valid)
    with pytest.raises(ValueError, match="ghost"):
        REC.check(fa, mode, F)
    # the wire-row key set no longer matches the declaration.
    with pytest.raises(ValueError, match="ghost"):
        REC.check_wire_row_keys(
            {k: None for k in ("op_type", "i", "j", "exponents", "factor",
                               "compress_kind", "quant_dtype",
                               "quant_scale_sign", "quant_scale_frac")},
            mode)
    # use 3: evaluate reads the scored fields, and `ghost` is now one of them.
    one = FaceAction(skip=fa.skip[:1],
                     **{k: getattr(fa, k)[:1] for k in REC.per_slot_names(mode)
                        if getattr(fa, k) is not None})
    rec = _Recorder(one)
    pol.evaluate_face(f, tables, rec, 0, pair[0], comp[0], valid[0])
    assert "ghost" not in rec.seen
    assert rec.seen != set(REC.scored_names(mode)), (
        "the use-3 agreement test would have PASSED with a declared field "
        "nobody reads -- the recorder is not doing its job")


def test_the_declaration_is_importable_without_ppo():
    """The declaration every use must agree with cannot need the 13k-line
    trainer to be importable, or the agreement would be unprovable from the
    env side.
    """
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; import alphagrad.approx.face_action as F; "
         "assert F.names(); "
         "print('ppo' in str(sorted(k for k in sys.modules "
         "if 'alphagrad' in k)))"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "False", r.stdout
