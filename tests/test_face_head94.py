#!/usr/bin/env python3
"""Pin the face head's LAYOUT PER WIDTH and its scoring.

The filename records the DEFAULT width: 94 logits, which is what
``--approx-add lossy`` / ``lossless`` build. Since the 2026-09-11 ruling
(ticket dsnn-3qm.56) the width is a function of ``--approx-add``, so every
layout claim here is parametrised over all five values rather than restated
for one:

    --approx-add   width            contents
    lossy          W*3 + 1          skip + the three contraction slots
    lossless       W*3 + 1          skip + the three contraction slots
    choose         W*3 + 2          + ONE Bernoulli, lossy vs lossless per face
    learned1       W*4 + 1          + slot 3, the OLD EDGE's approximation
    learned2       W*5 + 1          + slot 4, the ADD OUTPUT's approximation

with ``W = SLOT_WIDTH = 30 + len(masks.FACE_QUANT_DTYPES)``: 31 (94 logits)
while the dtype field was a Bernoulli over {float32, bfloat16}, 34 (103) with
the four-float set. The filename keeps the historical 94.

The properties that matter are the ones whose violation is SILENT:

  A. the slot blocks TILE from 1 upwards with no gap or overlap, at every
     width, so slot ``s`` is at ``1 + W*s`` whatever the value is. An
     off-by-one here reads another field's logit and nothing ever raises.
  A2. A FIELD A WIDTH DOES NOT CONTAIN CANNOT BE INDEXED AT ALL. This replaced
     the older "a gated-off field contributes exactly zero" test: at a narrower
     width there is no logit to perturb, so the claim is that the head REFUSES
     the out-of-range index -- an ``IndexError`` from ``slot_base`` /
     ``choose_index``, never a wrap-around into the neighbouring field.
  B. sample() == score(): the log-prob AND the entropy returned at sample time
     are what evaluate would recompute, EXACTLY 0 apart, at every width. This
     is the property whose absence WAS the T1 ratio bug, and it is what holds
     the PPO ratio at 1 on epoch 0.
  C. one axis per COMPRESS. The whole point of the softmax: no unrolling, no
     truncation rule, no zero-axes fallback.
  D. branch masking: changing a logit the chosen op does NOT consume must not
     move the log-prob at all.
  E. gates: a padding face and a skipped face contribute exactly zero.
  F. masks are respected -- an illegal op/axis is never drawn, and a mask whose
     row count disagrees with the width RAISES rather than being padded or
     sliced.
  G. gradients are finite wherever the forward is.

This module is a pytest module, not a script: every claim above used to live
inside a ``main()`` that pytest collected ZERO tests from, so none of them were
enforced by the suite.
"""
from __future__ import annotations
import equinox as eqx
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np
import pytest

from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES
from alphagrad.approx.unified_face_head import (
    CONTRACTION_LAYOUT, UnifiedFaceHead, FaceFields, SLOT_WIDTH, FACE_SLOTS,
    NUM_APPROX_OPS, MAX_PAIR_IDX, NUM_REDUCE_AXES, NUM_REDUCE_FNS,
    head_layout, slot_base, S_OP, S_I, S_J, S_AXIS, S_RFN, S_DTYPE,
    OP_BLOCKDIAG, OP_REDUCE, OP_QUANT, OP_NONE, O_SKIP, O_SLOT0,
    JOIN_LOSSY, JOIN_LOSSLESS, _bern_logp_ent,
)

#: THE OWNER'S ARITHMETIC, restated as a literal so a change to the layout
#: table has to change this number too. `W*N + 1` for the learned values --
#: NOT `+2`: they have no choose bit.
_W = SLOT_WIDTH
WIDTHS = {"lossy": 3 * _W + 1, "lossless": 3 * _W + 1, "choose": 3 * _W + 2,
          "learned1": 4 * _W + 1, "learned2": 5 * _W + 1}
MODES = list(WIDTHS)
E = 32


def _head(mode, seed=0):
    return UnifiedFaceHead(E, key=jrand.PRNGKey(seed), approx_add=mode)


def _ctx(seed=1):
    return jrand.normal(jrand.PRNGKey(seed), (E,))


def _masks(n, all_legal=True):
    return (jnp.ones((n, NUM_APPROX_OPS), jnp.float32),
            jnp.ones((n, MAX_PAIR_IDX), jnp.float32),
            jnp.ones((n, MAX_PAIR_IDX), jnp.float32),
            jnp.ones((n, NUM_REDUCE_AXES), jnp.float32))


def _fields(n, op_val, *, skip=0, join=None):
    return FaceFields(
        skip=jnp.array(skip, jnp.int32),
        op=jnp.full((n,), op_val, jnp.int32),
        i=jnp.zeros((n,), jnp.int32), j=jnp.ones((n,), jnp.int32),
        axis=jnp.zeros((n,), jnp.int32),
        reduce_fn=jnp.zeros((n,), jnp.int32),
        dtype_idx=jnp.zeros((n,), jnp.int32),
        join=join)


# ===================================================================== A
@pytest.mark.parametrize("mode", MODES)
def test_width_is_the_owners_arithmetic(mode):
    lay = head_layout(mode)
    assert lay.width == WIDTHS[mode], (mode, lay.width)
    assert lay.width == O_SLOT0 + SLOT_WIDTH * lay.n_slots + int(lay.has_choose)
    assert SLOT_WIDTH == S_DTYPE + NUM_FACE_QUANT_DTYPES
    assert SLOT_WIDTH == 34, "the four-float set: 30 + 4"
    assert O_SKIP == 0 and O_SLOT0 == 1
    # The learned values are W*N + 1, NOT +2: no choose bit.
    if mode in ("learned1", "learned2"):
        assert not lay.has_choose
        assert lay.width == SLOT_WIDTH * lay.n_slots + 1


@pytest.mark.parametrize("mode", MODES)
def test_slot_bases_are_one_multiply_at_every_width(mode):
    lay = head_layout(mode)
    bases = [lay.slot_base(s) for s in range(lay.n_slots)]
    assert bases == [O_SLOT0 + SLOT_WIDTH * s for s in range(lay.n_slots)]
    # The three CONTRACTION bases are the same number at every width -- that
    # width-independence is what the 2026-09-11 layout buys, and it is why a
    # bare `slot_base(s)` over range(FACE_SLOTS) is right under every value.
    assert bases[:FACE_SLOTS] == [1, 1 + _W, 1 + 2 * _W]
    assert bases[:FACE_SLOTS] == [slot_base(s) for s in range(FACE_SLOTS)]


@pytest.mark.parametrize("mode", MODES)
def test_the_logits_tile_exactly_once(mode):
    lay = head_layout(mode)
    covered = np.zeros(lay.width, int)
    covered[O_SKIP] += 1
    if lay.has_choose:
        covered[lay.choose_index] += 1
    for s in range(lay.n_slots):
        b = lay.slot_base(s)
        for lo, hi in ((S_OP, S_I), (S_I, S_J), (S_J, S_AXIS),
                       (S_AXIS, S_RFN), (S_RFN, S_DTYPE),
                       (S_DTYPE, SLOT_WIDTH)):
            covered[b + lo:b + hi] += 1
    assert (covered == 1).all(), (mode, covered.min(), covered.max())
    assert ((S_I - S_OP) + (S_J - S_I) + (S_AXIS - S_J) + (S_RFN - S_AXIS)
            + (S_DTYPE - S_RFN) + 1 == SLOT_WIDTH)


@pytest.mark.parametrize("mode", MODES)
def test_proj_emits_exactly_the_layout_width(mode):
    assert _head(mode).logits(_ctx()).shape == (WIDTHS[mode],)


def test_the_collision_at_94_is_the_design_not_an_accident():
    """``choose``'s bit and ``learned1``'s slot 4 BOTH sit at index 94.

    That is the consequence of the slot blocks tiling from 1: the bit goes
    immediately after the last block. The two never coexist -- ``choose`` has
    three slots and no fourth block, the learned values have no bit -- so one
    index can carry both meanings without ambiguity, and ``slot_base`` needs no
    branch. The earlier layout reserved 94 for the bit at EVERY value and
    started the join slots at 95, which is the branch that is gone.
    """
    assert head_layout("choose").choose_index == 94
    assert head_layout("learned1").slot_base(FACE_SLOTS) == 94
    assert head_layout("learned2").slot_base(FACE_SLOTS) == 94
    assert not head_layout("learned1").has_choose
    assert head_layout("choose").n_slots == FACE_SLOTS


# ==================================================================== A2
@pytest.mark.parametrize("mode", MODES)
def test_a_slot_the_width_does_not_have_cannot_be_indexed(mode):
    """THE REPLACEMENT FOR THE GATED-LOGIT PERTURBATION TEST.

    The old test perturbed a gated-off logit and required the score not to
    move. At a narrower width there is no logit to perturb, so the claim is
    stronger and different: the index is REFUSED. ``IndexError``, not a
    wrap-around into the next field and not a clamp onto the last one.
    """
    lay = head_layout(mode)
    for s in (lay.n_slots, lay.n_slots + 1, -1):
        with pytest.raises(IndexError):
            lay.slot_base(s)
    # and the bare module function answers for the contraction band only
    with pytest.raises(IndexError):
        slot_base(FACE_SLOTS)
    assert CONTRACTION_LAYOUT.n_slots == FACE_SLOTS


@pytest.mark.parametrize("mode", MODES)
def test_the_choose_bit_exists_only_under_choose(mode):
    lay = head_layout(mode)
    if mode == "choose":
        assert lay.choose_index == 94
        return
    with pytest.raises(IndexError):
        lay.choose_index


@pytest.mark.parametrize("mode", MODES)
def test_a_join_bit_at_a_width_without_one_raises(mode):
    """Either mismatch raises, in both directions.

    A bit under a layout that has none cannot be scored -- there is no logit --
    and a MISSING bit under ``choose`` must not become ``JOIN_LOSSY``: the plan
    would be measured under a join the policy did not pick while the stored
    log-prob scored the one it did.
    """
    lay = head_layout(mode)
    head = _head(mode)
    z = head.logits(_ctx())
    om, im, jm, am = _masks(lay.n_slots)
    kw = dict(op_mask=om, i_mask=im, j_mask=jm, axis_mask=am)
    if lay.has_choose:
        with pytest.raises(ValueError):
            head.score(z, _fields(lay.n_slots, OP_BLOCKDIAG), **kw)
        head.score(z, _fields(lay.n_slots, OP_BLOCKDIAG,
                              join=jnp.array(JOIN_LOSSLESS, jnp.int32)), **kw)
    else:
        with pytest.raises(ValueError):
            head.score(z, _fields(lay.n_slots, OP_BLOCKDIAG,
                                  join=jnp.array(JOIN_LOSSY, jnp.int32)), **kw)
        head.score(z, _fields(lay.n_slots, OP_BLOCKDIAG), **kw)


def test_unknown_approx_add_has_no_width():
    with pytest.raises(ValueError):
        head_layout("same")
    with pytest.raises(ValueError):
        UnifiedFaceHead(E, key=jrand.PRNGKey(0), approx_add="exact")


# ===================================================================== B
def _parity_head(mode, n_draws):
    """sample -> score INSIDE the head: the logits and the FaceFields."""
    lay = head_layout(mode)
    head = _head(mode)
    ctx = _ctx()
    om, im, jm, am = _masks(lay.n_slots)
    dlp, dent = [], []
    for s in range(n_draws):
        z, f, lp, ent, ar = head.sample(
            ctx, jrand.PRNGKey(7000 + s), op_mask=om, i_mask=im, j_mask=jm,
            axis_mask=am)
        lp2, e2, a2 = head.score(z, f, op_mask=om, i_mask=im, j_mask=jm,
                                 axis_mask=am)
        dlp.append(abs(float(lp) - float(lp2)))
        dent.append(abs(float(ent) - float(e2)))
        assert float(ar) == float(a2)
        # The `choose` bit is part of the decision being scored, so a width
        # that has one must actually be carrying it here.
        assert (f.join is None) != lay.has_choose
    return dlp, dent


def _parity_record(mode, n_draws):
    """sample -> THE ACTION RECORD -> evaluate, through the POLICY.

    THE EXTENSION (2026-09-11, ticket dsnn-3qm.56). The head agreeing with
    itself is NOT the property PPO needs: `sample` emits an action RECORD that
    the rollout stores and the loss re-reads, and a field the record drops
    leaves the head-level parity at exactly 0 while the ratio walks off 1. The
    `choose` bit was exactly that field -- the head drew it and the wire could
    not carry it -- so the parity claim is stated at BOTH levels now, with the
    same numbers and the same name.
    """
    from alphagrad.approx.face_action import FaceAction
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

    sizes = (8, 8, 4, 16, 6, 4)
    sz = jnp.asarray(sizes, jnp.int32)
    feats = AxisTokenFeatures(
        size=sz, log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((len(sizes), AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((len(sizes),), jnp.int32),
        valid_mask=jnp.ones((len(sizes),), jnp.float32))
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=2,
                            key=jrand.PRNGKey(0), approx_add=mode)
    n = len(sizes)
    pv = jnp.ones((n, n), jnp.float32)
    cv = jnp.ones((n,), jnp.float32)
    dlp, dent = [], []
    for s in range(n_draws):
        sk, row, lp, ent, ar, _sp, _od = pol.sample_face(
            feats, tables, jrand.PRNGKey(7000 + s), 0, pv, cv,
            jnp.asarray(1.0))
        fa = FaceAction(skip=jnp.asarray(sk)[None],
                        **{k: jnp.asarray(v)[None] for k, v in row.items()})
        lp2, e2, a2, _sp2, _od2 = pol.evaluate_face(
            feats, tables, fa, 0, pv, cv, jnp.asarray(1.0))
        dlp.append(abs(float(lp) - float(lp2)))
        dent.append(abs(float(ent) - float(e2)))
        assert float(ar) == float(a2)
    return dlp, dent


@pytest.mark.parametrize("level", ["head", "record"])
@pytest.mark.parametrize("mode", MODES)
def test_sample_score_parity_is_exactly_zero(mode, level):
    """log-prob AND entropy, EXACTLY 0 apart, at every width AND at both levels.

    The failure mode is silent: a field drawn in sample() and forced in
    score() (or the reverse) just moves the PPO ratio off 1.

    ``level="head"``   sample -> score on the logits and the FaceFields.
    ``level="record"`` sample_face -> the ACTION RECORD -> evaluate_face, i.e.
                       including the round trip through the wire fields the
                       rollout stores. This is the level the PPO ratio actually
                       lives at, and the one the `choose` bit used to fail.
    """
    dlp, dent = (_parity_head if level == "head" else _parity_record)(mode, 120)
    assert max(dlp) == 0.0, (mode, level, max(dlp))
    assert max(dent) == 0.0, (mode, level, max(dent))


@pytest.mark.parametrize("mode", MODES)
def test_sample_draws_a_join_bit_iff_the_width_has_one(mode):
    lay = head_layout(mode)
    head = _head(mode)
    om, im, jm, am = _masks(lay.n_slots)
    seen = set()
    for s in range(60):
        _, f, *_ = head.sample(_ctx(), jrand.PRNGKey(400 + s), op_mask=om,
                               i_mask=im, j_mask=jm, axis_mask=am)
        if not lay.has_choose:
            assert f.join is None
        else:
            seen.add(int(f.join))
    if lay.has_choose:
        assert seen <= {JOIN_LOSSY, JOIN_LOSSLESS} and seen


def test_the_key_budget_moves_no_existing_draw():
    """``split(key, n)[i]`` does not depend on ``n``.

    That is the whole reason the key budget can be width-dependent
    (``1 + 6*n_slots`` plus one under ``choose``) without moving the skip's or
    any contraction slot's draw. Stated on jax itself, since the head's own
    parameters differ across widths and so cannot be compared directly.
    """
    k = jrand.PRNGKey(1234)
    a = jrand.split(k, 19)
    for n in (20, 25, 31, 32):
        b = jrand.split(k, n)
        assert np.array_equal(np.asarray(a), np.asarray(b[:19])), n


# ===================================================================== C
@pytest.mark.parametrize("mode", MODES)
def test_one_axis_per_compress(mode):
    lay = head_layout(mode)
    head = _head(mode)
    om, im, jm, am = _masks(lay.n_slots)
    n_rd = 0
    for s in range(150):
        _, f, *_ = head.sample(_ctx(), jrand.PRNGKey(1000 + s), op_mask=om,
                               i_mask=im, j_mask=jm, axis_mask=am)
        assert f.axis.shape == (lay.n_slots,)
        assert f.axis.dtype == jnp.int32
        n_rd += int(np.sum(np.asarray(f.op) == OP_REDUCE))
    assert n_rd > 0


# ===================================================================== D
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("op_val,unused_lo", [
    (OP_BLOCKDIAG, S_AXIS),      # DIAG ignores the reduce axis
    (OP_REDUCE, S_I),            # COMPRESS ignores i
    (OP_QUANT, S_J),             # QUANT ignores j
    (OP_NONE, S_I),              # END ignores everything
])
def test_branch_masking_on_every_slot(mode, op_val, unused_lo):
    """Only the fields the chosen op consumes carry log-prob.

    Checked on EVERY slot the width has, not just slot 0: a learned join slot
    is the same 31-wide block and must mask the same way.
    """
    lay = head_layout(mode)
    head = _head(mode)
    z0 = head.logits(_ctx())
    om, im, jm, am = _masks(lay.n_slots)
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)
    f = _fields(lay.n_slots, op_val, join=join)
    kw = dict(op_mask=om, i_mask=im, j_mask=jm, axis_mask=am)
    lp_a, *_ = head.score(z0, f, **kw)
    for s in range(lay.n_slots):
        z1 = z0.at[lay.slot_base(s) + unused_lo].add(5.0)
        lp_b, *_ = head.score(z1, f, **kw)
        assert abs(float(lp_a) - float(lp_b)) < 1e-6, (mode, s)


@pytest.mark.parametrize("mode", MODES)
def test_every_slot_the_width_has_is_LIVE(mode):
    """The converse of branch masking, and the reason the old gates are gone.

    A slot the width contains must MOVE the score -- otherwise a learned slot
    could be stuck off while the engine still applied its wire row. So the
    width is the gate, and the gate is on.
    """
    lay = head_layout(mode)
    head = _head(mode)
    z0 = head.logits(_ctx())
    om, im, jm, am = _masks(lay.n_slots)
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)
    f = _fields(lay.n_slots, OP_BLOCKDIAG, join=join)
    kw = dict(op_mask=om, i_mask=im, j_mask=jm, axis_mask=am)
    lp_a, e_a, ar_a = head.score(z0, f, **kw)
    for s in range(lay.n_slots):
        z1 = z0.at[lay.slot_base(s) + S_OP + OP_BLOCKDIAG].add(3.0)
        lp_b, e_b, _ = head.score(z1, f, **kw)
        assert float(lp_a) != float(lp_b), (mode, s)
    assert float(ar_a) == 1.0 + lay.n_slots      # the face + every live slot
    if lay.has_choose:
        z1 = z0.at[lay.choose_index].add(17.0)
        lp_b, e_b, _ = head.score(z1, f, **kw)
        assert float(lp_a) != float(lp_b)


# ===================================================================== E
@pytest.mark.parametrize("mode", MODES)
def test_padding_and_skipped_faces_contribute_zero(mode):
    lay = head_layout(mode)
    head = _head(mode)
    ctx = _ctx()
    om, im, jm, am = _masks(lay.n_slots)
    _, f, lp_pad, e_pad, a_pad = head.sample(
        ctx, jrand.PRNGKey(7), op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
        face_valid=False)
    assert float(lp_pad) == 0.0 and float(e_pad) == 0.0
    assert float(a_pad) == 0.0

    z0 = head.logits(ctx)
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)
    f_skip = _fields(lay.n_slots, OP_BLOCKDIAG, skip=1, join=join)
    lp_s, _, ar_s = head.score(z0, f_skip, op_mask=om, i_mask=im, j_mask=jm,
                               axis_mask=am)
    lp_only_skip, _ = _bern_logp_ent(z0[O_SKIP], jnp.array(True))
    if lay.has_choose:
        # the bit is a property of the FACE, not of a slot, so a skipped face
        # still scores it -- and nothing else.
        lp_cho, _ = _bern_logp_ent(z0[lay.choose_index], jnp.array(False))
        lp_only_skip = lp_only_skip + lp_cho
    assert abs(float(lp_s) - float(lp_only_skip)) < 1e-6, mode
    assert float(ar_s) == 1.0


# ===================================================================== F
@pytest.mark.parametrize("mode", MODES)
def test_masks_are_respected(mode):
    lay = head_layout(mode)
    head = _head(mode)
    om = jnp.zeros((lay.n_slots, NUM_APPROX_OPS),
                   jnp.float32).at[:, OP_QUANT].set(1.0)
    am = jnp.zeros((lay.n_slots, NUM_REDUCE_AXES), jnp.float32).at[:, 3].set(1.0)
    im = jnp.ones((lay.n_slots, MAX_PAIR_IDX), jnp.float32)
    bad_op = bad_ax = 0
    for s in range(120):
        _, f, *_ = head.sample(_ctx(), jrand.PRNGKey(2000 + s), op_mask=om,
                               i_mask=im, j_mask=im, axis_mask=am)
        bad_op += int(np.sum(np.asarray(f.op) != OP_QUANT))
        bad_ax += int(np.sum(np.asarray(f.axis) != 3))
    assert bad_op == 0 and bad_ax == 0


@pytest.mark.parametrize("mode", MODES)
def test_a_mask_row_count_that_disagrees_with_the_width_raises(mode):
    """NEITHER padded NOR sliced.

    A padded row would clear actions on a tensor no mask was computed from
    (finding 72's defect); a short one would leave a slot the width HAS
    unscored while the engine still applies its row. Both are silent, so both
    raise -- and the message names the unfinished trainer wire, which is the
    real reason a 3-row mask reaches a 4-slot head.
    """
    lay = head_layout(mode)
    head = _head(mode)
    z0 = head.logits(_ctx())
    om, im, jm, am = _masks(lay.n_slots)
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)
    f = _fields(lay.n_slots, OP_BLOCKDIAG, join=join)
    for n in (1, 2, 3, 4, 5, 6):
        if n == lay.n_slots:
            continue
        bad = jnp.ones((n, NUM_APPROX_OPS), jnp.float32)
        with pytest.raises(ValueError):
            head.score(z0, f, op_mask=bad, i_mask=im, j_mask=jm, axis_mask=am)
    # mixed widths, even when one of them is right
    other = 1 if lay.n_slots != 1 else 2
    with pytest.raises(ValueError):
        head.score(z0, f, op_mask=om,
                   i_mask=jnp.ones((other, MAX_PAIR_IDX), jnp.float32),
                   j_mask=jm, axis_mask=am)


# ===================================================================== G
@pytest.mark.parametrize("mode", MODES)
def test_no_nan_under_a_fully_dead_mask(mode):
    lay = head_layout(mode)
    head = _head(mode)
    om = jnp.zeros((lay.n_slots, NUM_APPROX_OPS), jnp.float32)
    _, _, lp3, e3, _ = head.sample(
        _ctx(), jrand.PRNGKey(3), op_mask=om,
        i_mask=jnp.ones((lay.n_slots, MAX_PAIR_IDX), jnp.float32),
        j_mask=jnp.ones((lay.n_slots, MAX_PAIR_IDX), jnp.float32),
        axis_mask=jnp.ones((lay.n_slots, NUM_REDUCE_AXES), jnp.float32))
    assert np.isfinite(float(lp3)) and np.isfinite(float(e3))


def _grad_masks(n, op_live, i_live, ax_live):
    def row(w, live):
        v = np.zeros((w,), np.float32)
        for k in live:
            v[k] = 1.0
        return jnp.asarray(np.tile(v, (n, 1)))
    return (row(NUM_APPROX_OPS, op_live), row(MAX_PAIR_IDX, i_live),
            row(MAX_PAIR_IDX, i_live), row(NUM_REDUCE_AXES, ax_live))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tag", ["all live", "one op live", "one axis live",
                                "nothing live"])
@pytest.mark.parametrize("fv", [True, False])
@pytest.mark.parametrize("skip", [0, 1])
def test_gradients_are_finite_where_the_forward_is(mode, tag, fv, skip):
    """Non-finite GRADIENT under a finite forward -- the silent failure."""
    lay = head_layout(mode)
    n = lay.n_slots
    head = UnifiedFaceHead(64, key=jrand.PRNGKey(11), approx_add=mode)
    ctx = jrand.normal(jrand.PRNGKey(12), (64,))
    live = {
        "all live": (range(NUM_APPROX_OPS), range(MAX_PAIR_IDX),
                     range(NUM_REDUCE_AXES)),
        "one op live": ([0], range(MAX_PAIR_IDX), range(NUM_REDUCE_AXES)),
        "one axis live": (range(NUM_APPROX_OPS), [0], [0]),
        "nothing live": ([], [], []),
    }[tag]
    om, im, jm, am = _grad_masks(n, *live)
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)

    def loss(h):
        z = h.logits(ctx)
        fields = FaceFields(
            skip=jnp.array(skip),
            op=jnp.arange(n) % NUM_APPROX_OPS,
            i=jnp.zeros((n,), jnp.int32), j=jnp.ones((n,), jnp.int32),
            axis=jnp.zeros((n,), jnp.int32),
            reduce_fn=jnp.zeros((n,), jnp.int32),
            dtype_idx=jnp.zeros((n,), jnp.int32), join=join)
        lp, ent, _ar = h.score(
            z, fields, op_mask=om, i_mask=im, j_mask=jm, axis_mask=am,
            pair_ok=None, face_valid=jnp.array(fv),
            approx_ok=jnp.array(True))
        # The loss differentiates BOTH -- the trainer's ppo term rides on the
        # log-prob and its entropy bonus on the entropy, and only one of the
        # two hit the trap.
        return jnp.nan_to_num(lp, neginf=0.0) + ent

    g = eqx.filter_grad(loss)(head)
    bad = sum(int(np.sum(~np.isfinite(np.asarray(v))))
              for v in jax.tree_util.tree_leaves(g) if eqx.is_array(v))
    assert bad == 0, (mode, tag, fv, skip, bad)


@pytest.mark.parametrize("mode", MODES)
def test_gradient_finite_with_a_masked_logit_far_above_the_live_max(mode):
    """The exp-overflow case.

    It is NOT enough to make every logit large: the shift is by the max over
    LIVE entries, so a uniform bias moves zmax with it and nothing overflows.
    The trap needs ONE MASKED index far above the live ones, which is what a
    legality mask routinely produces once the head has learned to want an
    action the oracle forbids.
    """
    lay = head_layout(mode)
    n = lay.n_slots
    head = UnifiedFaceHead(64, key=jrand.PRNGKey(11), approx_add=mode)
    ctx = jrand.normal(jrand.PRNGKey(12), (64,))
    om, im, jm, am = _grad_masks(n, [1], [0, 1], [0])   # op 0 is MASKED OUT
    _b = np.asarray(head.proj.layers[-1].bias).copy()
    for _s in range(n):
        _b[lay.slot_base(_s) + S_OP + 0] = 400.0        # the masked one
    big = eqx.tree_at(lambda h: h.proj.layers[-1].bias, head, jnp.asarray(_b))
    join = (jnp.array(JOIN_LOSSY, jnp.int32) if lay.has_choose else None)

    def loss_big(h):
        z = h.logits(ctx)
        fields = FaceFields(
            skip=jnp.array(0),
            op=jnp.ones((n,), jnp.int32),                # the LIVE op
            i=jnp.zeros((n,), jnp.int32), j=jnp.ones((n,), jnp.int32),
            axis=jnp.zeros((n,), jnp.int32),
            reduce_fn=jnp.zeros((n,), jnp.int32),
            dtype_idx=jnp.zeros((n,), jnp.int32), join=join)
        lp, ent, _ = h.score(z, fields, op_mask=om, i_mask=im, j_mask=jm,
                             axis_mask=am, pair_ok=None,
                             face_valid=jnp.array(True),
                             approx_ok=jnp.array(True))
        return jnp.nan_to_num(lp, neginf=0.0) + ent

    g = eqx.filter_grad(loss_big)(big)
    bad = sum(int(np.sum(~np.isfinite(np.asarray(v))))
              for v in jax.tree_util.tree_leaves(g) if eqx.is_array(v))
    assert bad == 0, (mode, bad)
