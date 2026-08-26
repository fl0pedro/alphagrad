"""sec 12.7 credit-layer fix: --lag-causal-mask + --adv-winsorize.

Pins the two mechanisms that break the v62/v63 anti-none runaway
(docs/QUALITY_COLLAPSE_INVESTIGATION.md sec 12):

1. Causal quality mask m(e,t): the quality-channel advantage weight is
   lambda*m(e,t), m=1 iff step t of env e took ANY causal face action
   (skip counts; any slot op != OP_NONE on a valid face counts; padding
   faces never count). All-none plans carry ZERO quality advantage on
   every step; a plan with approximations at steps {3,7} carries the
   quality term at exactly those steps.
2. Winsorize: per-channel normalized advantages clip at [-Z, +Z] BEFORE
   preference weighting; Z=0/off is bit-identical.
3. Both flags off => the scalarization is bit-identical to the pre-fix
   formula sum(norm_adv_components * preference).
4. Rollout/replay equality: the mask is a pure function of the stored
   batch fields (face_valid/face_skip/face_op_type), so any slicing or
   shuffling of the batch commutes with computing the mask -- the same
   guarantee the loss's replay path relies on.

sec 12.9 (--lag-raw-viol-adv, H-ZNEUT):

5. NEUTRALIZATION IMMUNITY (THE test): the raw-scale quality advantage is
   bitwise invariant to any perturbation of the PopArt (mu, sigma) for
   fixed raw advantages, while the z path it replaces scales by 1/sigma
   -- the v64b fade (sigma_q widening as violations grew) cannot touch it.
6. Flag off / other channels: the override touches ONLY the quality slot.
7. Composition with --lag-causal-mask: masked steps carry ZERO quality
   term, unmasked steps carry exactly lambda*A_raw.
8. Bounded: |A_raw| <= _RAW_VIOL_ADV_CLIP (= 2.0) for any critic error.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from alphagrad.approx.ppo import (                              # noqa: E402
    HEAD_NAMES,
    NUM_VALUE_HEADS,
    _RAW_VIOL_ADV_CLIP,
    _causal_quality_mask,
    _raw_viol_override,
    _winsorize_adv,
)
from alphagrad.approx.unified_face_head import OP_NONE          # noqa: E402

QHEAD = HEAD_NAMES.index("quality")
E, T, F, S = 3, 10, 5, 2


def _mk_faces(seed=0, valid_p=0.6):
    """All-none baseline face fields: every op OP_NONE, no skips, a random
    validity pattern with at least one valid face per step."""
    rng = np.random.default_rng(seed)
    fv = (rng.random((E, T, F)) < valid_p).astype(np.float32)
    fv[..., 0] = 1.0                       # >=1 valid face per step
    fsk = np.zeros((E, T, F), dtype=np.int32)
    fop = np.full((E, T, F, S), OP_NONE, dtype=np.int32)
    return fv, fsk, fop


def _scalarize(comps, pref, mask=None, winsorize=0.0):
    """Mirror of ppo.py's post-fix scalarization (winsorize BEFORE
    preference weighting; mask multiplies the quality slot only)."""
    comps = jnp.asarray(comps)
    pref = jnp.asarray(pref)
    if winsorize > 0.0:
        comps, _ = _winsorize_adv(comps, winsorize)
    if mask is not None:
        pref = pref.at[..., QHEAD].multiply(jnp.asarray(mask))
    return jnp.sum(comps * pref, axis=-1)


# ---------------------------------------------------------------------------
# mask correctness
# ---------------------------------------------------------------------------

def test_all_none_plan_zero_quality_everywhere():
    fv, fsk, fop = _mk_faces()
    m = np.asarray(_causal_quality_mask(fv, fsk, fop))
    assert m.shape == (E, T)
    np.testing.assert_array_equal(m, 0.0)
    # and through the scalarization: quality contributes 0 to EVERY step.
    rng = np.random.default_rng(1)
    comps = rng.normal(size=(E, T, NUM_VALUE_HEADS)).astype(np.float32)
    lam = 10.0
    pref = np.ones((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    pref[..., QHEAD] = lam
    got = np.asarray(_scalarize(comps, pref, mask=m))
    want = np.asarray(_scalarize(
        comps[..., [j for j in range(NUM_VALUE_HEADS) if j != QHEAD]],
        pref[..., [j for j in range(NUM_VALUE_HEADS) if j != QHEAD]]))
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_approx_steps_carry_quality_term_only_there():
    fv, fsk, fop = _mk_faces()
    # env 0: approximations at steps {3, 7} on a valid face.
    fop[0, 3, 0, 1] = (OP_NONE + 1) % 4    # some non-NONE op
    fop[0, 7, 0, 0] = (OP_NONE + 2) % 4
    m = np.asarray(_causal_quality_mask(fv, fsk, fop))
    want = np.zeros((E, T), dtype=np.float32)
    want[0, 3] = 1.0
    want[0, 7] = 1.0
    np.testing.assert_array_equal(m, want)
    # scalarization: quality term present at exactly those steps.
    comps = np.zeros((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    comps[..., QHEAD] = -5.6               # the v63 destroyed-plan z
    pref = np.ones((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    pref[..., QHEAD] = 10.0
    got = np.asarray(_scalarize(comps, pref, mask=m))
    assert got[0, 3] == -56.0 and got[0, 7] == -56.0
    others = np.delete(got.reshape(-1), [0 * T + 3, 0 * T + 7])
    np.testing.assert_array_equal(others, 0.0)


def test_skip_counts_as_causal():
    fv, fsk, fop = _mk_faces()
    fsk[1, 4, 0] = 1                       # skip on a valid face, ops all NONE
    m = np.asarray(_causal_quality_mask(fv, fsk, fop))
    assert m[1, 4] == 1.0
    assert m.sum() == 1.0


def test_invalid_faces_never_causal():
    fv, fsk, fop = _mk_faces()
    fv[2, 5, 3] = 0.0                      # padding face...
    fsk[2, 5, 3] = 1                       # ...skipped
    fop[2, 5, 3, :] = (OP_NONE + 1) % 4    # ...and op'd
    m = np.asarray(_causal_quality_mask(fv, fsk, fop))
    np.testing.assert_array_equal(m, 0.0)


# ---------------------------------------------------------------------------
# winsorize
# ---------------------------------------------------------------------------

def test_winsorize_clips_at_z_and_reports_clip_frac():
    rng = np.random.default_rng(2)
    comps = (rng.normal(size=(E, T, NUM_VALUE_HEADS)) * 4.0).astype(
        np.float32)
    comps[0, 0, QHEAD] = -5.6              # the sec-12 destroyed-plan z
    z = 3.0
    clipped, frac = _winsorize_adv(jnp.asarray(comps), z)
    clipped, frac = np.asarray(clipped), np.asarray(frac)
    assert clipped.min() >= -z and clipped.max() <= z
    assert clipped[0, 0, QHEAD] == -z
    inside = np.abs(comps) <= z
    np.testing.assert_array_equal(clipped[inside], comps[inside])
    np.testing.assert_allclose(
        frac, np.mean(np.abs(comps) > z, axis=(0, 1)), rtol=1e-6)
    assert frac.shape == (NUM_VALUE_HEADS,)
    assert frac[QHEAD] > 0.0


def test_winsorize_off_is_identity():
    # Z=0 disables the clip in ppo.py (guarded by _wz > 0.0); the helper
    # itself must also be exact identity for any Z that bounds the data.
    rng = np.random.default_rng(3)
    comps = rng.normal(size=(E, T, NUM_VALUE_HEADS)).astype(np.float32)
    clipped, frac = _winsorize_adv(jnp.asarray(comps), 1e9)
    np.testing.assert_array_equal(np.asarray(clipped), comps)
    np.testing.assert_array_equal(np.asarray(frac), 0.0)


# ---------------------------------------------------------------------------
# both flags off == pre-fix formula, bitwise
# ---------------------------------------------------------------------------

def test_flags_off_bit_identical_to_prefix_scalarization():
    rng = np.random.default_rng(4)
    comps = (rng.normal(size=(E, T, NUM_VALUE_HEADS)) * 7.0).astype(
        np.float32)
    pref = rng.uniform(0.5, 12.0, (E, T, NUM_VALUE_HEADS)).astype(np.float32)
    old = np.asarray(jnp.sum(jnp.asarray(comps) * jnp.asarray(pref), axis=-1))
    new = np.asarray(_scalarize(comps, pref, mask=None, winsorize=0.0))
    np.testing.assert_array_equal(old, new)   # BITWISE


# ---------------------------------------------------------------------------
# rollout-collected vs replay-recomputed mask equality
# ---------------------------------------------------------------------------

def test_mask_commutes_with_batch_slicing():
    """The loss's replay path consumes shuffled/sliced minibatches of the
    SAME stored face fields the rollout collected. The mask is a pure
    per-(env,step) function of those fields, so mask(slice) == slice(mask)
    for any index selection -- rollout and replay can never disagree."""
    fv, fsk, fop = _mk_faces(seed=5)
    rng = np.random.default_rng(6)
    fsk[rng.random((E, T, F)) < 0.15] = 1
    fop[rng.random((E, T, F, S)) < 0.2] = (OP_NONE + 1) % 4
    full = np.asarray(_causal_quality_mask(fv, fsk, fop))
    # flatten (E,T) -> N and take a shuffled minibatch, as the update does.
    N = E * T
    perm = rng.permutation(N)
    mb = perm[: N // 2]
    fv_f = fv.reshape(N, F)[mb]
    fsk_f = fsk.reshape(N, F)[mb]
    fop_f = fop.reshape(N, F, S)[mb]
    replay = np.asarray(_causal_quality_mask(fv_f, fsk_f, fop_f))
    np.testing.assert_array_equal(replay, full.reshape(N)[mb])


# ---------------------------------------------------------------------------
# sec 12.9: --lag-raw-viol-adv
# ---------------------------------------------------------------------------

def _mk_raw_advantages(seed=7, scale=0.6):
    """Raw-scale GAE advantages: cost channels at symlog scale, quality
    channel in the bounded -violation range (tau 0.75 -> [-1.25, ~1])."""
    rng = np.random.default_rng(seed)
    adv = (rng.normal(size=(E, T, NUM_VALUE_HEADS)) * 8.0).astype(np.float32)
    adv[..., QHEAD] = rng.uniform(-1.25, 1.0, (E, T)).astype(np.float32) \
        * scale
    return adv


def test_raw_adv_invariant_to_popart_stats():
    """THE test: for fixed RAW advantages the raw-path quality term is
    BITWISE invariant to any (mu, sigma) perturbation, while the z path it
    replaces scales by 1/sigma.

    NOTE the framing changed with sec 12.9. This was written as H-ZNEUT
    immunity ("sigma_q widens as violations grow, so the price fades");
    H-ZNEUT is REFUTED -- v64b's sigma_q SHRANK 0.1446 -> 0.1265 and the
    per-unit price ROSE. The INVARIANCE property tested here is unaffected
    and is exactly what makes the flag the sec-12.10 arm-D control: the
    quality term stops depending on the critic-driven scale in EITHER
    direction."""
    adv = _mk_raw_advantages()
    lam = 10.0
    pref = np.ones((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    pref[..., QHEAD] = lam
    outs, raws, zs = [], [], []
    # a 10x sigma_q perturbation in the direction H-ZNEUT predicted. The
    # measured v64b move was the OTHER way and smaller (0.1446 -> 0.1265);
    # 10x is used here because the test asserts INVARIANCE, so the widest
    # perturbation is the strongest statement. mu never enters the
    # advantage at all (GAE takes differences); sigma is the only scale.
    for sigma_q in (0.128, 1.28):
        sigma = np.ones((NUM_VALUE_HEADS,), dtype=np.float32)
        sigma[QHEAD] = sigma_q
        comps = jnp.asarray(adv) / jnp.asarray(sigma)     # PopArt step
        comps_ov, raw_q = _raw_viol_override(comps, jnp.asarray(adv), QHEAD)
        outs.append(np.asarray(jnp.sum(comps_ov * pref, axis=-1)))
        raws.append(np.asarray(raw_q))
        zs.append(np.asarray(comps[..., QHEAD]))
    np.testing.assert_array_equal(raws[0], raws[1])       # BITWISE immune
    np.testing.assert_array_equal(outs[0], outs[1])       # through scalarize
    np.testing.assert_array_equal(
        raws[0], np.clip(adv[..., QHEAD], -_RAW_VIOL_ADV_CLIP,
                         _RAW_VIOL_ADV_CLIP))
    # control: WITHOUT the override the z path scales by exactly 1/sigma --
    # the scale dependence the flag removes (and the reason the flag is a
    # clean PopArt-layer control regardless of which way sigma moved).
    np.testing.assert_allclose(zs[0], zs[1] * 10.0, rtol=1e-5)


def test_raw_adv_touches_only_quality_slot():
    adv = _mk_raw_advantages(seed=8)
    rng = np.random.default_rng(9)
    comps = jnp.asarray(
        (rng.normal(size=(E, T, NUM_VALUE_HEADS)) * 2.0).astype(np.float32))
    comps_ov, _ = _raw_viol_override(comps, jnp.asarray(adv), QHEAD)
    other = [j for j in range(NUM_VALUE_HEADS) if j != QHEAD]
    np.testing.assert_array_equal(
        np.asarray(comps_ov)[..., other], np.asarray(comps)[..., other])
    # flag off == the ppo path never calls the override (Python-level if);
    # the untouched components ARE the pre-12.9 scalarization input.


def test_raw_adv_composes_with_causal_mask():
    """Masked steps carry ZERO quality term; unmasked steps carry exactly
    lambda*A_raw (the mask multiplies the PREFERENCE quality slot, the
    override rewrites the COMPONENT quality slot -- independent axes)."""
    adv = _mk_raw_advantages(seed=10)
    lam = 10.0
    pref = np.ones((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    pref[..., QHEAD] = lam
    rng = np.random.default_rng(11)
    m = (rng.random((E, T)) < 0.4).astype(np.float32)
    comps = jnp.asarray(adv) / 0.3                        # some sigma
    comps_ov, raw_q = _raw_viol_override(comps, jnp.asarray(adv), QHEAD)
    got = np.asarray(_scalarize(comps_ov, pref, mask=m))
    cost_only = np.asarray(_scalarize(
        np.asarray(comps_ov)[..., [j for j in range(NUM_VALUE_HEADS)
                                   if j != QHEAD]],
        pref[..., [j for j in range(NUM_VALUE_HEADS) if j != QHEAD]]))
    np.testing.assert_allclose(
        got - cost_only, lam * np.asarray(raw_q) * m, rtol=1e-5, atol=1e-6)
    # masked steps: quality term exactly zero.
    np.testing.assert_array_equal((got - cost_only)[m == 0.0], 0.0)


def test_raw_adv_bounded_for_any_critic_error():
    """|A_raw| <= 2.0 even for a transiently insane critic (warm start);
    the raw path is NOT winsorized, the clip is its only guard."""
    adv = np.zeros((E, T, NUM_VALUE_HEADS), dtype=np.float32)
    adv[..., QHEAD] = np.linspace(-1e6, 1e6, E * T).reshape(E, T)
    comps = jnp.zeros((E, T, NUM_VALUE_HEADS), jnp.float32)
    _, raw_q = _raw_viol_override(comps, jnp.asarray(adv), QHEAD)
    raw_q = np.asarray(raw_q)
    assert np.abs(raw_q).max() <= _RAW_VIOL_ADV_CLIP
    assert raw_q.min() == -_RAW_VIOL_ADV_CLIP
    assert raw_q.max() == _RAW_VIOL_ADV_CLIP
    # in-range values pass through untouched (identity inside the bound).
    adv2 = _mk_raw_advantages(seed=12)
    _, raw_q2 = _raw_viol_override(
        jnp.zeros_like(jnp.asarray(adv2)), jnp.asarray(adv2), QHEAD)
    inside = np.abs(adv2[..., QHEAD]) <= _RAW_VIOL_ADV_CLIP
    np.testing.assert_array_equal(
        np.asarray(raw_q2)[inside], adv2[..., QHEAD][inside])
