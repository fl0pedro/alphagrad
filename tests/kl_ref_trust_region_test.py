"""A5: --kl-ref-weight / --kl-ref-target / --target-kl. Pins the trust region.

TWO OPPOSITE FAILURES ARE ON RECORD, and this guard is aimed at exactly one
of them:

1. DIFFUSION (v62/v63). An unbounded entropy BONUS (0.05, then 0.005) pushed
   the face head toward uniform over 94 actions x ~118 faces until every plan
   was destroyed; the entropy FLOOR hinge was worse, firing continuously from
   ep0. A KL penalty against a FROZEN reference is the bounded version of the
   same intent: it costs nothing while the policy stays near identity and
   grows without a floor's always-on gradient.

2. PARKING (R2). Every entropy term 0 and FACE_NONE_BIAS=6: the policy sat at
   identity for all 250 episodes with quality spread 3e-06 and latency spread
   1.7us over 16 plans -- ZERO advantage contrast, nothing to learn from.

**THE TRUST REGION DOES NOT FIX (2) AND CANNOT.** It penalises MOVEMENT away
from identity, which is the direction R2 already refused to go. R2's problem
was that all 16 plans were the same plan, not that the policy moved too much;
a KL penalty makes that MORE stable, not less. The contrast knob is
ALPHAGRAD_FACE_NONE_BIAS (agent_factory.py:87-112) and it needs no code.
Sweep the two together: KL = stability, bias = contrast.

ALPHAGRAD_FACE_NONE_BIAS SWEEP SPEC (no code; env var read at
common/agent_factory.py:87-112 and printed at init):

The identity init adds +B to every slot's OP_NONE logit and -B to SKIP. The
factory prints ``P(approx/face) ~ 3*e^-B``. That label is off by one factor:
the 3 is NUM_APPROX_OPS-1 = the three non-NONE ops (DIAG/COMPRESS/QUANT), so
``3*e^-B`` is the rate PER SLOT, and a face carries FACE_SLOTS = 3 slots. The
expected approximations per PLAN over the TLM's ~118 live faces is therefore
``118 * 3 * 3 * e^-B``:

    B    3*e^-B (per slot)   E[approx/plan]   verdict
    6      0.00744               2.63         in band, at the FLOOR  <- R2 ran here
    5      0.02021               7.15         in band, CENTRE        <- RECOMMENDED
    4      0.05495              19.45         just above the band
    3      0.14936              52.9          far above
    2      0.40601             143.7          destruction regime (v62/v63)

The ``x3x3`` reading is not a guess: it reproduces the two independently
MEASURED anchors on record (~2.6 approx/plan at B=6, ~19 at B=4) to two
significant figures, which the mislabelled per-face reading (0.88 and 6.5)
does not. Every measured win in this campaign has lived in the 1-15
approx/plan band, so the sweep is {6 (the parked control), 5 (recommended),
4 (upper bracket)} and B<=3 is not worth an arm.

What is pinned here:
  * OFF BY DEFAULT, and off is a static Python gate (bit-identity).
  * The penalty is exactly 0 when the policy equals the reference, positive
    otherwise, and monotone in how far the PARAMETERS were perturbed.
  * The adaptive coefficient rises above target, falls below, is a no-op at
    target, and never leaves its clip bounds.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402

from alphagrad.approx.heads import (                            # noqa: E402
    AXIS_TAG_BITS,
    AxisTokenFeatures,
    FacePathPolicy,
    precompute_factor_tables,
)
from alphagrad.approx.ppo import (                              # noqa: E402
    _kl_ref_dual_update,
    _kl_ref_estimate,
    kl_ref_enabled,
)

_N = 4
_F = 4
_S = 3
_E = 32


# --------------------------------------------------------------------- gating
def test_off_by_default():
    """Neither flag set -> OFF. This is what makes a default run bit-identical
    to the one before the feature existed: every added term is behind this."""
    assert kl_ref_enabled(SimpleNamespace()) is False
    assert kl_ref_enabled(
        SimpleNamespace(kl_ref_weight=0.0, kl_ref_target=0.0)) is False
    assert kl_ref_enabled(
        SimpleNamespace(kl_ref_weight=None, kl_ref_target=None)) is False


def test_either_flag_arms_it():
    assert kl_ref_enabled(SimpleNamespace(kl_ref_weight=0.01)) is True
    # the ADAPTIVE arm alone must arm it too -- otherwise --kl-ref-target
    # with the default weight 0 would silently do nothing.
    assert kl_ref_enabled(SimpleNamespace(kl_ref_target=0.01)) is True


# ------------------------------------------------------- the estimator itself
def test_zero_when_identical():
    lp = jnp.asarray([-1.0, -2.5, -0.3, -7.0])
    assert float(_kl_ref_estimate(lp, lp)) == 0.0


def test_non_negative_and_grows_with_the_log_ratio():
    """k3 = (r-1) - log r is non-negative PER SAMPLE (not just in mean) and
    convex with its minimum at r=1, so it grows in BOTH directions."""
    ref = jnp.zeros((64,))
    prev = 0.0
    for d in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0):
        for sign in (+1.0, -1.0):
            k = float(_kl_ref_estimate(ref + sign * d, ref))
            assert k >= 0.0
        k = float(_kl_ref_estimate(ref + d, ref))
        if d > 0.0:
            assert k > prev, f"k3 not increasing at delta={d}"
        prev = k


# ------------------------------------------------- perturbed REAL parameters
def _feats():
    tb = jnp.zeros((_N, AXIS_TAG_BITS)).at[:, 0].set(1.0)
    return AxisTokenFeatures(
        size=jnp.full((_N,), 4, dtype=jnp.int32),
        log_size=jnp.log(jnp.ones(_N) * 4),
        tag_bits=tb,
        group_id=jnp.zeros(_N, dtype=jnp.int32),
        valid_mask=jnp.ones(_N),
    )


def _masks(n_valid=_F):
    pair = jnp.zeros((_F, _N, _N)).at[:, 0, 1].set(1.0).at[:, 1, 0].set(1.0)
    comp = jnp.zeros((_F, _N)).at[:, 0].set(1.0)
    valid = jnp.array([1.0] * n_valid + [0.0] * (_F - n_valid))
    return pair, comp, valid


def _perturbed(pol, direction, eps):
    """The reference policy displaced along ONE FIXED random direction in
    parameter space, scaled by eps. A fixed direction (rather than fresh
    noise per eps) is what makes the KL a smooth function of eps: the
    log-prob shift is ~eps*g, so k3 ~ eps^2 * var(g) / 2 -- strictly
    increasing, which is the property the trust region relies on."""
    upd = jax.tree_util.tree_map(lambda d: eps * d, direction)
    return eqx.apply_updates(pol, upd)


_N_ACTIONS = 8


def _setup():
    """Reference policy, ONE fixed displacement direction, and a fixed set of
    actions drawn from the reference. Built once: every eps below re-scores
    the SAME actions, which is what makes the comparison a function of the
    parameter displacement alone."""
    ref = FacePathPolicy(_E, 4, max_faces=_F, num_slots=_S,
                         key=jrand.PRNGKey(0))
    params = eqx.filter(ref, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    dkeys = jrand.split(jrand.PRNGKey(1234), max(len(leaves), 1))
    direction = jax.tree_util.tree_unflatten(
        treedef,
        [jrand.normal(k, x.shape, x.dtype) for k, x in zip(dkeys, leaves)],
    )
    tables = precompute_factor_tables(8)
    pair, comp, valid = _masks()
    ctx = jnp.ones((_E,)) * 0.1
    feats = _feats()
    acts = [ref.sample(ctx, feats, tables, jrand.PRNGKey(s),
                       pair, comp, valid)[0]
            for s in range(_N_ACTIONS)]
    return ref, direction, (ctx, feats, tables, pair, comp, valid), acts


def _logps(pol, env, acts):
    ctx, feats, tables, pair, comp, valid = env
    return jnp.stack([
        pol.evaluate(ctx, feats, tables, fa, pair, comp, valid)[0]
        for fa in acts])


def test_zero_at_the_reference_and_growing_away_from_it():
    """eps=0 IS the reference: the penalty must be exactly 0, or the trust
    region taxes a policy that has not moved (which is how the entropy FLOOR
    hinge went wrong -- it fired continuously from ep0). Then it must GROW
    as the parameters are displaced, or it is not a trust region at all."""
    ref, direction, env, acts = _setup()
    lp_ref = _logps(ref, env, acts)

    k0 = float(_kl_ref_estimate(_logps(_perturbed(ref, direction, 0.0),
                                       env, acts), lp_ref))
    assert k0 == 0.0, f"non-zero KL at the reference itself: {k0}"

    ks = [float(_kl_ref_estimate(
              _logps(_perturbed(ref, direction, e), env, acts), lp_ref))
          for e in (0.02, 0.05, 0.1, 0.2)]
    assert all(k > 0.0 for k in ks), ks
    assert ks == sorted(ks), f"KL not monotone in the perturbation: {ks}"


# ------------------------------------------------------------ adaptive dual
_LO, _HI = 1e-3, 1e3


def _step(coef, kl, target=0.01, eta=1.0):
    return _kl_ref_dual_update(coef, kl, target, eta, _LO, _HI)


def test_dual_rises_above_target_and_falls_below():
    c = 1.0
    assert _step(c, 0.10) > c          # KL 10x target -> price up
    assert _step(c, 0.001) < c         # KL 0.1x target -> price down
    assert _step(c, 0.01) == c         # exactly on target -> no move


def test_dual_is_a_no_op_without_a_target():
    """--kl-ref-weight alone is the FIXED-price arm; the dual must not move
    it, whatever the KL does."""
    for kl in (0.0, 1.0, 1e6):
        assert _kl_ref_dual_update(0.7, kl, 0.0, 1.0, _LO, _HI) == 0.7


def test_dual_move_is_bounded_per_step():
    """+-eta*20%% per episode by construction (the relative error is clipped
    to +-0.2), whatever the KL is -- one pathological episode cannot blow the
    price up."""
    for kl in (0.0, 1e-9, 1.0, 1e9):
        c = _step(1.0, kl)
        assert 0.8 - 1e-9 <= c <= 1.2 + 1e-9, (kl, c)


def test_dual_stays_inside_its_clip_bounds():
    c = 1.0
    for _ in range(500):               # KL pinned far above target
        c = _step(c, 1e6)
    assert c == _HI
    for _ in range(2000):              # and far below
        c = _step(c, 0.0)
    assert c == _LO


def test_dual_converges_toward_the_target_on_a_toy_plant():
    """Closed loop on a monotone plant kl(coef) = 1/coef: the controller must
    settle near the target rather than oscillate away from it."""
    target = 0.01
    c = 1.0
    for _ in range(400):
        c = _step(c, 1.0 / c, target=target)
    assert abs((1.0 / c) - target) / target < 0.05, c
