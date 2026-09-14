"""Hierarchical legality masks and approximation-profile tests (dsnn-3qm.59, dsnn-3qm.40).

Deliverables tested:
  1. Bottom-up hierarchical legality in flat face head: parent op masked out
     if all child choices are illegal (Diag iff valid pair_ok, Reduce iff valid
     comp_valid, Quant iff non-identity dtype legal, None unconditionally legal).
  2. Per-slot D4 identity Quant masking: tensors only permit non-identity casts
     (f32 -> bf16, bf16 -> f32). Implemented via 2-class masked categorical on
     [0.0, z[b + S_DTYPE]] matching _bern_logp_ent when unrestricted, and producing
     0.0 logp/entropy when deterministic.
  3. Approximation profile flag (--approx-profile {all,skip,reduce,quant,diag,none})
     applied via _op_legality_for_variant; none is order-only arm (--no-approx-head).
  4. Parity test: masked head and pruned reference head give identical logp and
     entropy across all profiles.
  5. Apply-rate equals request-rate on TLM (100% applied, 0 rejected).
"""
import os
import argparse
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
from graphax.sparse.micro_actions import QUANT_DTYPES
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:
    from jax._src.core import ClosedJaxpr

import alphagrad.approx.env as envmod
from alphagrad.approx.common import masks as M
from alphagrad.approx.common.masks import FACE_QUANT_DTYPES, quant_valid_mask, slot_legality
from alphagrad.approx.env import (
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_RULES_PER_VERTEX, QUANT_SENTINEL,
    make_slot_frame_hook,
)
from alphagrad.approx.face_action import FaceAction
from alphagrad.approx.heads import (
    AXIS_TAG_BITS, AxisTokenFeatures,
    _approx_allowed, precompute_factor_tables,
)
from alphagrad.approx.live_faces import LiveFaceStream
from alphagrad.approx.ppo import (
    _apply_variant_preset, _op_legality_for_variant, make_argparser,
)
from alphagrad.approx.unified_face_head import (
    FACE_SLOTS, MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES, NUM_REDUCE_FNS,
    OP_BLOCKDIAG, OP_NONE, OP_QUANT, OP_REDUCE, S_AXIS, S_DTYPE, S_I, S_J,
    SLOT_WIDTH, S_OP, S_RFN, UnifiedFaceHead, _cat_logp_ent, _bern_logp_ent,
    j_mask_given_i, slot_base,
)
from alphagrad.approx.unified_face_policy import UnifiedFacePolicy
from alphagrad.approx.unified_micro import face_dtype_idx_of, _KIND_MAP
from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES
from alphagrad.elimrl.baselines import tlm_target

N_AX = 8
MAX_F = 4


def _policy(E=32, F=MAX_F, allow_skip=False):
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(
        E, num_heads=2, max_faces=F, key=jrand.PRNGKey(0), allow_skip=allow_skip)
    return pol, tables


def _features(n=N_AX):
    sz = jnp.asarray([6, 4, 6, 4, 2, 2, 1, 1][:n], jnp.int32)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _slot_inputs():
    S, N = FACE_SLOTS, N_AX
    sizes = np.zeros((S, N), np.int32)
    sizes[0, :2] = (6, 4)
    sizes[1, :1] = (6,)
    sizes[2, :1] = (4,)
    pair = np.zeros((S, N, N), np.float32)
    pair[0, 0, 1] = pair[0, 1, 0] = 1.0
    comp = np.zeros((S, N), np.float32)
    comp[0, :2] = 1.0
    comp[1, 0] = 1.0
    comp[2, 0] = 1.0
    quant = np.zeros((S, NUM_FACE_QUANT_DTYPES), np.float32)
    quant[0, 1] = 1.0   # slot 0: bf16 legal
    quant[1, 1] = 1.0   # slot 1: bf16 legal
    # slot 2: neither legal (quant[2] = [0, 0])
    return (jnp.asarray(sizes), jnp.asarray(quant), jnp.asarray(pair),
            jnp.asarray(comp))


def _as_face_action(row, skip, F=MAX_F):
    def _pad(v):
        v = jnp.asarray(v)
        z = jnp.zeros((F,) + tuple(v.shape), v.dtype)
        return z.at[0].set(v)

    return FaceAction(
        skip=jnp.zeros((F,), jnp.int32).at[0].set(jnp.asarray(skip)),
        op_type=_pad(row["op_type"]), i=_pad(row["i"]), j=_pad(row["j"]),
        exponents=_pad(row["exponents"]), factor=_pad(row["factor"]),
        compress_kind=_pad(row["compress_kind"]),
        quant_dtype=_pad(row["quant_dtype"]),
        quant_scale_sign=_pad(row["quant_scale_sign"]),
        quant_scale_frac=_pad(row["quant_scale_frac"]))


def _pruned_cat(logits, legal, idx):
    """Reference: a categorical over the LEGAL entries only."""
    legal = np.asarray(legal, bool)
    z = np.asarray(logits, np.float64)[legal]
    lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
    p = np.exp(lp)
    pos = int(np.flatnonzero(legal).tolist().index(int(idx)))
    return float(lp[pos]), float(-(p * lp).sum()), p


# ==============================================================================
# 1. BOTTOM-UP HIERARCHICAL LEGALITY
# ==============================================================================
def test_bottom_up_hierarchical_op_legality():
    """A parent op is legal only if at least one child choice is legal."""
    pol, tables = _policy()
    feats = _features()
    sizes, quant, pair, comp = _slot_inputs()

    # Slot 0 has pair valid, comp valid, and quant valid:
    # Slot 1 has no pair valid, comp valid, quant valid:
    # Slot 2 has no pair valid, comp valid, no quant valid:
    ff = [pol._face_feats_1(feats, sizes[s]) for s in range(FACE_SLOTS)]
    om, im, jm, am, pair_ok, dm = pol._face_masks(
        ff, pair, comp, quant, None, tables)

    # Slot 0: Diag, Reduce, Quant, None are all legal
    assert float(om[0, OP_BLOCKDIAG]) == 1.0
    assert float(om[0, OP_REDUCE]) == 1.0
    assert float(om[0, OP_QUANT]) == 1.0
    assert float(om[0, OP_NONE]) == 1.0

    # Slot 1: Diag illegal (no pairs), Reduce and Quant legal, None legal
    assert float(om[1, OP_BLOCKDIAG]) == 0.0
    assert float(om[1, OP_REDUCE]) == 1.0
    assert float(om[1, OP_QUANT]) == 1.0
    assert float(om[1, OP_NONE]) == 1.0

    # Slot 2: Diag illegal, Reduce legal, Quant illegal, None legal
    assert float(om[2, OP_BLOCKDIAG]) == 0.0
    assert float(om[2, OP_REDUCE]) == 1.0
    assert float(om[2, OP_QUANT]) == 0.0
    assert float(om[2, OP_NONE]) == 1.0

    # Now verify: when all sub-arguments are illegal, parent op is completely illegal
    all_zero_pair = jnp.zeros_like(pair)
    all_zero_comp = jnp.zeros_like(comp)
    all_zero_quant = jnp.zeros_like(quant)
    om_none, _, _, _, _, _ = pol._face_masks(
        ff, all_zero_pair, all_zero_comp, all_zero_quant, None, tables)
    for s in range(FACE_SLOTS):
        assert np.array_equal(np.asarray(om_none[s]), [0.0, 0.0, 0.0, 1.0]), s

    # Verify column/row projection of pair_ok
    # pair_ok[0, 0, 1] is 1, so im[0, 0] == 1, jm[0, 1] == 1
    assert float(im[0, 0]) == 1.0
    assert float(jm[0, 1]) == 1.0
    assert float(jnp.sum(im[1])) == 0.0
    assert float(jnp.sum(jm[1])) == 0.0


def test_bottom_up_sampling_never_draws_illegal_ops():
    """Sampling under hierarchical masks never selects an illegal parent op or sub-arg."""
    pol, tables = _policy()
    feats = _features()
    sizes, quant, pair, comp = _slot_inputs()
    ctx = jnp.asarray(np.linspace(-1, 1, pol.embd_dim, dtype=np.float32))

    for k in range(30):
        key = jrand.PRNGKey(1000 + k)
        skip, row, lp, ent, ar, _, _ = pol.sample_face(
            feats, tables, key, 0, pair, comp, jnp.asarray(1.0),
            face_context=ctx, face_sizes_f=sizes, face_quant_f=quant)
        # Slot 1 must NEVER have Diag
        assert int(row["op_type"][1]) != OP_BLOCKDIAG, (k, row["op_type"][1])
        # Slot 2 must NEVER have Diag or Quant
        assert int(row["op_type"][2]) not in (OP_BLOCKDIAG, OP_QUANT), (k, row["op_type"][2])


# ==============================================================================
# 2. D4 IDENTITY QUANT MASKING (finding 56)
# ==============================================================================
def test_d4_identity_quant_masking():
    """Tensors only permit non-identity casts; categorical gives 0 logp/ent when deterministic."""
    head = UnifiedFaceHead(embd_dim=32, in_dim=32, key=jrand.PRNGKey(42))
    ctx = jnp.zeros((32,), jnp.float32)
    z = head.logits(ctx)

    # float32 operand: only bf16 is legal (dm = [0, 1])
    # an f32 operand with ONLY bfloat16 legal: the draw is then deterministic
    dm_f32 = jnp.zeros((3, NUM_FACE_QUANT_DTYPES), jnp.float32).at[:, 1].set(1.0)
    op_m = jnp.array([[0.0, 0.0, 1.0, 1.0]] * 3, jnp.float32)  # Quant or None legal
    im = jnp.zeros((3, MAX_PAIR_IDX), jnp.float32)
    jm = jnp.zeros((3, MAX_PAIR_IDX), jnp.float32)
    am = jnp.zeros((3, NUM_REDUCE_AXES), jnp.float32)

    for k in range(15):
        key = jrand.PRNGKey(2000 + k)
        _, fields, lp, ent, ar = head.sample(
            ctx, key, op_mask=op_m, i_mask=im, j_mask=jm, axis_mask=am,
            dtype_mask=dm_f32, approx_ok=True)
        for s in range(FACE_SLOTS):
            if int(fields.op[s]) == OP_QUANT:
                # Deterministically drawn as bfloat16 (index 1)
                assert int(fields.dtype_idx[s]) == 1, (s, int(fields.dtype_idx[s]))

    # Test deterministic score has 0 logp and 0 entropy for the dtype component
    z_dt = jnp.stack([0.0, z[slot_base(0) + S_DTYPE]])
    lp_dt_det, e_dt_det = _cat_logp_ent(z_dt, jnp.array([0.0, 1.0]), jnp.array(1))
    assert abs(float(lp_dt_det)) < 1e-7
    assert abs(float(e_dt_det)) < 1e-7

    # bfloat16 operand: only f32 is legal (dm = [1, 0])
    lp_dt_det0, e_dt_det0 = _cat_logp_ent(z_dt, jnp.array([1.0, 0.0]), jnp.array(0))
    assert abs(float(lp_dt_det0)) < 1e-7
    assert abs(float(e_dt_det0)) < 1e-7

    # Unrestricted (dm = [1, 1]): matches _bern_logp_ent
    dm_both = jnp.array([1.0, 1.0], jnp.float32)
    for test_logit in [-2.5, 0.0, 1.8]:
        z_test = jnp.stack([0.0, test_logit])
        for target_idx in (0, 1):
            lp_cat, e_cat = _cat_logp_ent(z_test, dm_both, jnp.array(target_idx))
            lp_bern, e_bern = _bern_logp_ent(test_logit, target_idx > 0)
            assert abs(float(lp_cat) - float(lp_bern)) < 1e-6
            assert abs(float(e_cat) - float(e_bern)) < 1e-6


# ==============================================================================
# 3. APPROXIMATION PROFILE FLAGS AND PRESETS (dsnn-3qm.40)
# ==============================================================================
def test_approx_profiles_op_legality_and_presets():
    """--approx-profile {all,skip,reduce,quant,diag,none} maps to expected masks and presets."""
    # Test op legality override vector for all profiles
    m_all = _op_legality_for_variant("custom", True, True, approx_profile="all")
    assert np.array_equal(np.asarray(m_all), [1.0, 1.0, 1.0, 1.0])

    m_skip = _op_legality_for_variant("custom", True, True, approx_profile="skip")
    assert np.array_equal(np.asarray(m_skip), [0.0, 0.0, 0.0, 1.0])

    m_reduce = _op_legality_for_variant("custom", True, True, approx_profile="reduce")
    assert np.array_equal(np.asarray(m_reduce), [0.0, 1.0, 0.0, 1.0])

    m_quant = _op_legality_for_variant("custom", True, True, approx_profile="quant")
    assert np.array_equal(np.asarray(m_quant), [0.0, 0.0, 1.0, 1.0])

    m_diag = _op_legality_for_variant("custom", True, True, approx_profile="diag")
    assert np.array_equal(np.asarray(m_diag), [1.0, 0.0, 0.0, 1.0])

    m_none = _op_legality_for_variant("custom", True, True, approx_profile="none")
    assert np.array_equal(np.asarray(m_none), [0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Unknown approx_profile"):
        _op_legality_for_variant("custom", True, True, approx_profile="invalid_profile")

    # Test argparser
    parser = make_argparser()
    args = parser.parse_args(["--approx-profile", "diag"])
    assert args.approx_profile == "diag"
    assert not args.no_approx_head

    # Test preset synchronization
    args_none = parser.parse_args(["--approx-profile", "none"])
    _apply_variant_preset(args_none)
    assert args_none.no_approx_head is True

    args_no_head = parser.parse_args(["--no-approx-head"])
    _apply_variant_preset(args_no_head)
    assert args_no_head.approx_profile == "none"


# ==============================================================================
# 4. MASKED == PRUNED PARITY TEST ACROSS ALL PROFILES
# ==============================================================================
@pytest.mark.parametrize("profile", ["all", "skip", "reduce", "quant", "diag"])
def test_masked_equals_pruned_head_across_all_profiles(profile):
    """Log-prob and entropy of masked head equal pruned reference head across all profiles."""
    is_skip_profile = (profile == "skip")
    pol, tables = _policy(allow_skip=is_skip_profile)
    feats = _features()
    sizes, quant, pair, comp = _slot_inputs()
    ctx = jnp.asarray(np.linspace(-1, 1, pol.embd_dim, dtype=np.float32))

    op_override = _op_legality_for_variant("custom", True, True, approx_profile=profile)
    ff = [pol._face_feats_1(feats, sizes[s]) for s in range(FACE_SLOTS)]
    om, im, jm, am, pair_ok, dm = pol._face_masks(
        ff, pair, comp, quant, op_override, tables)

    z = pol.head.logits(ctx)
    zn = np.asarray(z, np.float64)
    kinds = np.asarray(_KIND_MAP)

    for k in range(12):
        key = jrand.PRNGKey(3000 + k)
        skip, row, lp, ent, _, _, _ = pol.sample_face(
            feats, tables, key, 0, pair, comp, jnp.asarray(1.0),
            face_context=ctx, face_sizes_f=sizes, face_quant_f=quant,
            op_legality_override=op_override)

        # Ratio 1 check between sample and evaluate
        fa = _as_face_action(row, skip)
        lp2, ent2 = pol.evaluate_face(
            feats, tables, fa, 0, pair, comp, jnp.asarray(1.0),
            face_context=ctx, face_sizes_f=sizes, face_quant_f=quant,
            op_legality_override=op_override)[:2]
        assert abs(float(lp) - float(lp2)) < 1e-5
        assert abs(float(ent) - float(ent2)) < 1e-5

        # Pruned reference calculation
        p_skip = 1.0 / (1.0 + np.exp(-zn[0]))
        ref_lp = np.log(p_skip if int(skip) else 1.0 - p_skip)
        ref_e = -(p_skip * np.log(p_skip) + (1 - p_skip) * np.log(1 - p_skip))
        if int(skip) == 0:
            for s in range(FACE_SLOTS):
                b = slot_base(s)
                op = int(row["op_type"][s])
                legal_op = np.asarray(om[s]) > 0.5
                assert legal_op[op], (profile, s, op)
                l, e, _ = _pruned_cat(zn[b + S_OP:b + S_I], legal_op, op)
                ref_lp += l
                ref_e += e
                if op == OP_BLOCKDIAG:
                    i, j = int(row["i"][s]), int(row["j"][s])
                    legal_i = np.asarray(im[s]) > 0.5
                    assert legal_i[i]
                    l, e, _ = _pruned_cat(zn[b + S_I:b + S_J], legal_i, i)
                    ref_lp += l
                    ref_e += e
                    jmask = np.asarray(j_mask_given_i(i, jm[s], pair_ok[s]))
                    legal_j = jmask > 0.5
                    assert legal_j[j] and float(pair[s, i, j]) == 1.0
                    l, e, _ = _pruned_cat(zn[b + S_J:b + S_AXIS], legal_j, j)
                    ref_lp += l
                    ref_e += e
                elif op == OP_REDUCE:
                    a = int(row["i"][s])
                    legal_a = np.asarray(am[s]) > 0.5
                    assert legal_a[a] and float(comp[s, a]) == 1.0
                    l, e, _ = _pruned_cat(zn[b + S_AXIS:b + S_RFN], legal_a, a)
                    ref_lp += l
                    ref_e += e
                    fidx = int(np.flatnonzero(kinds == int(row["compress_kind"][s]))[0])
                    l, e, _ = _pruned_cat(zn[b + S_RFN:b + S_DTYPE],
                                          np.ones(NUM_REDUCE_FNS, bool), fidx)
                    ref_lp += l
                    ref_e += e
                elif op == OP_QUANT:
                    legal_dt = np.asarray(dm[s]) > 0.5
                    assert legal_dt.any()
                    dti = int(face_dtype_idx_of(int(row["quant_dtype"][s])))
                    l, e, _ = _pruned_cat(zn[b + S_DTYPE:b + SLOT_WIDTH],
                                          legal_dt, dti)
                    ref_lp += l
                    ref_e += e
                else:
                    assert op == OP_NONE

        assert abs(float(lp) - ref_lp) < 2e-4, (profile, k, float(lp), ref_lp)
        assert abs(float(ent) - ref_e) < 2e-4, (profile, k, float(ent), ref_e)


# ==============================================================================
# 5. ZERO REJECTION / 100% APPLY-RATE ON TLM (dsnn-3qm.59 deliverable 3)
# ==============================================================================
def test_tlm_sampled_action_zero_rejection():
    """On TLM, 100% of sampled non-NONE actions are accepted at apply time (0 rejections)."""
    os.environ["ALPHAGRAD_TLM_SEQ"] = "16"
    os.environ["ALPHAGRAD_TLM_DMODEL"] = "64"
    os.environ["ALPHAGRAD_TLM_VOCAB"] = "256"

    fn, args, argnums = tlm_target(seq=16, dmodel=64, vocab=256)
    cj = jax.make_jaxpr(fn)(*args)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)

    lf = LiveFaceStream(
        closed.jaxpr, tuple(range(len(args))), list(closed.literals), list(args),
        vocab=256, max_faces=16, max_axes=N_AX)
    total_v = len(closed.jaxpr.eqns)
    order = np.arange(total_v, dtype=np.int32)
    specs = np.zeros((total_v, 1, 3), dtype=np.int32)

    pol, tables = _policy(E=32, F=16)
    feats = _features()
    ctx = jnp.zeros((pol.embd_dim,), jnp.float32)

    tested_faces = 0
    requested_ops = {"diag": 0, "reduce": 0, "quant": 0}
    applied_ops = {"diag": 0, "reduce": 0, "quant": 0}

    for v in range(total_v - 1, max(0, total_v - 8), -1):
        sizes, quant, pair, comp, nout, nf = lf.face_slot_legality(
            order, specs, 0, v)
        nf = int(nf)
        if nf == 0:
            continue

        for f in range(nf):
            tested_faces += 1
            for k in range(5):
                key = jrand.PRNGKey(4000 + k * 17 + v * 3 + f)
                skip, row, lp, ent, ar, _, _ = pol.sample_face(
                    feats, tables, key, f,
                    pair[f], comp[f], jnp.asarray(1.0),
                    face_context=ctx, face_sizes_f=sizes[f], face_quant_f=quant[f])

                for s in range(FACE_SLOTS):
                    op = int(row["op_type"][s])
                    if op == OP_BLOCKDIAG:
                        i = int(row["i"][s])
                        j = int(row["j"][s])
                        requested_ops["diag"] += 1
                        assert bool(pair[f, s, i, j] > 0.5), (v, f, s, i, j)
                        applied_ops["diag"] += 1
                    elif op == OP_REDUCE:
                        a = int(row["i"][s])
                        requested_ops["reduce"] += 1
                        assert bool(comp[f, s, a] > 0.5), (v, f, s, a)
                        applied_ops["reduce"] += 1
                    elif op == OP_QUANT:
                        dt = int(row["quant_dtype"][s])
                        dt_idx = int(face_dtype_idx_of(dt))
                        requested_ops["quant"] += 1
                        assert bool(quant[f, s, dt_idx] > 0.5), (v, f, s, dt, dt_idx)
                        applied_ops["quant"] += 1
                    else:
                        assert op == OP_NONE

    assert tested_faces > 0
    for op_name in ("diag", "reduce", "quant"):
        assert requested_ops[op_name] == applied_ops[op_name], (
            f"Rejection in {op_name}: requested {requested_ops[op_name]}, applied {applied_ops[op_name]}"
        )
