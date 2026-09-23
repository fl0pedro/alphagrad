# -*- coding: utf-8 -*-
"""THE ABSORBER (ticket dsnn-dfw): --face-none-bias > 0 stops the face head
from deciding ANYTHING, from episode 0.

WHAT THE CLUSTER MEASURED, and what this file reproduces on CPU.

Three smoke runs of arm C on TransformerLM, same stack (alphagrad ab568f4 /
3ac95da / 8fe379d, graphax 3df48f2), same node, and -- proven by the argv
recorded in wandb-metadata.json -- exactly ONE flag apart:

| job   | --face-none-bias | entropy/approx_head ep0 | approx_prob at episode 0                          |
|-------|------------------|-------------------------|---------------------------------------------------|
| 66136 | 0                | 0.7732                  | skip .5038 none .2371 diag .0311 comp .0859 q .1421 |
| 66201 | 2                | 4.37e-05                | skip 0 none 0.9999999 diag 0 comp 0 quant 0         |
| 66167 | 4                | 2.91e-05                | skip 0 none 1 diag 0 comp 0 quant 0                 |

Episode 0 is the FIRST rollout: no PPO update has happened, so nothing about
the loss can explain it. ``approx_prob/*`` is the realized per-(valid face,
slot) decision frequency and the five classes partition it, so the bias-0 row
is the head behaving exactly as ``apply_face_none_bias`` documents
(``p_skip = sigmoid(-0) = 1/2``) and the bias-2 / bias-4 rows are a head that
emits NONE with probability 1 and never skips.

THE SKIP BERNOULLI IS WHAT MAKES THIS A DEFECT AND NOT A TUNING RESULT. No
legality mask touches it -- ``UnifiedFaceHead.sample`` draws it from one logit
and gates it only by ``face_valid * approx_ok`` -- so at bias 4 the face head's
entropy cannot be below ``H(sigmoid(-4)) = 0.088`` nats however the op masks
fall. It is measured at 2.9e-05, three thousand times smaller, i.e. the head is
SATURATED at init rather than tilted by B.

``tests/face_none_bias_flag_test.py`` already pins the analytic init
(``p_skip = sigmoid(-B)``, ``p_none = e^B/(e^B+3)``) and is green -- on a ZERO
context, with ``--face-logit-clamp 0`` and ``--init-scheme campaign``. The
campaign arm runs with a REAL face latent, ``--face-logit-clamp 15`` and
``--init-scheme classic``, and none of those three is covered anywhere. This
file adds exactly that cross-product, so the run's configuration is asserted
rather than assumed.

RUN IT ON pgi15-cpu1 (the suite is CPU-only):
    JAX_PLATFORMS=cpu pytest tests/face_none_bias_campaign_config_test.py -q
A FAILURE HERE IS THE DEFECT, and the failing parameter names the knob that
carries it. If every case PASSES, the head is innocent and the decisions are
dropped downstream of the draw -- the next instrument is then the per-face
census (``ALPHAGRAD_FACE_DEBUG=1``, commit cefbabf8), whose
``approx_ok_zero`` / ``legal_*`` / ``drawn_*`` counters separate "masked
illegal" from "drawn and discarded".
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from init_scheme_test import _agent, _ns                         # noqa: E402

from alphagrad.approx import unified_face_head as _ufh           # noqa: E402
from alphagrad.approx.unified_face_head import (                 # noqa: E402
    FACE_SLOTS, NUM_APPROX_OPS, OP_NONE, O_QUANT, O_SKIP, S_OP, set_logit_clamp,
    slot_base)

#: The campaign arm's own values (fq_smoke_C_tlm.sbatch, job 66201).
CAMPAIGN_CLAMP = 15.0
CAMPAIGN_FACE_SCALE = 0.1
CAMPAIGN_SCHEME = "classic"

#: The biases the owner is choosing between, and 0 as the control.
BIASES = (0.0, 1.0, 2.0, 4.0)


@pytest.fixture
def clamp():
    """Set and RESTORE the module-global face-logit clamp.

    ``LOGIT_CLAMP`` is process-global state that ``ppo.main`` writes once from
    ``--face-logit-clamp``; a test that left it set would change every later
    test's head in the same process.
    """
    before = float(_ufh.LOGIT_CLAMP[0])

    def _set(c):
        set_logit_clamp(float(c))

    yield _set
    set_logit_clamp(before)


def _campaign_agent(bias, scheme=CAMPAIGN_SCHEME, seed=11):
    return _agent(_ns(face_none_bias=float(bias),
                      scale_face_head=CAMPAIGN_FACE_SCALE,
                      init_scheme=scheme,
                      face_logit_clamp=CAMPAIGN_CLAMP), seed=seed)


def _ctx(agent, key=None):
    """The head's input: zeros (the pinned analytic case) or a real latent.

    The width is the head's own ``in_dim``; a real palimpsa readout is
    standardised, so unit-normal draws are the honest stand-in.
    """
    n = agent.face_path_policy.head.proj.layers[0].weight.shape[-1]
    if key is None:
        return jnp.zeros((n,), jnp.float32)
    return jrand.normal(key, (n,), jnp.float32)


def _skip_and_none(agent, ctx):
    """``(p_skip, p_none per contraction slot)`` straight off the head."""
    z = agent.face_path_policy.head.logits(ctx)
    p_skip = float(jax.nn.sigmoid(z[O_SKIP]))
    p_none = []
    for s in range(FACE_SLOTS):
        b = slot_base(s) + S_OP
        p = jax.nn.softmax(z[b:b + NUM_APPROX_OPS])
        p_none.append(float(p[OP_NONE]))
    return p_skip, p_none


# ------------------------------------------------- 1. the skip Bernoulli

@pytest.mark.parametrize("bias", BIASES)
@pytest.mark.parametrize("scheme", ["campaign", "classic"])
def test_skip_probability_is_sigmoid_minus_bias_under_the_campaign_clamp(
        bias, scheme, clamp):
    """``p_skip = sigmoid(-B)`` must survive the clamp and both schemes.

    On a zero context every Linear bias is 0 after the init, so the head's
    logits ARE its output bias and the clamp is ``15*tanh(+-B/15)``, i.e.
    near-identity for the biases in use. A run at bias 4 whose faces never
    skip is only possible if this number is wrong.
    """
    clamp(CAMPAIGN_CLAMP)
    agent = _campaign_agent(bias, scheme=scheme)
    p_skip, _ = _skip_and_none(agent, _ctx(agent))
    want = float(jax.nn.sigmoid(
        CAMPAIGN_CLAMP * np.tanh(-bias / CAMPAIGN_CLAMP)))
    assert p_skip == pytest.approx(want, rel=1e-4), (scheme, bias, p_skip)
    # And the number the cluster contradicts: at bias 4 one face in 55 skips.
    if bias == 4.0:
        assert p_skip > 0.015, p_skip


@pytest.mark.parametrize("bias", BIASES)
def test_none_probability_is_the_softmax_tilt_under_the_campaign_clamp(
        bias, clamp):
    clamp(CAMPAIGN_CLAMP)
    agent = _campaign_agent(bias)
    _, p_none = _skip_and_none(agent, _ctx(agent))
    b = CAMPAIGN_CLAMP * np.tanh(bias / CAMPAIGN_CLAMP)
    want = float(np.exp(b) / (np.exp(b) + 2.0))
    for s, p in enumerate(p_none):
        assert p == pytest.approx(want, rel=1e-4), (bias, s, p)
        # Never a point mass: the head must keep mass on the three classes.
        assert p < 0.999, (bias, s, p)


# --------------------------------------- 2. the same, on a REAL face latent

@pytest.mark.parametrize("bias", [0.0, 2.0, 4.0])
def test_the_bias_shifts_the_logits_by_exactly_b_on_a_real_latent(
        bias, clamp):
    """The bias is ADDITIVE, so on ONE context the only difference between
    the bias-0 head and the bias-B head is +B on the three OP_NONE logits,
    -B on the skip logit and -B on the quant bit. Asserted on a real
    (non-zero) latent, with the
    clamp OFF so the comparison is on the raw head.

    This is the assertion the run falsifies at the level of behaviour: at
    bias 2 the measured skip rate falls from 0.504 to 0.000, which no
    additive -2 can do.
    """
    clamp(0.0)
    a0 = _campaign_agent(0.0)
    ab = _campaign_agent(bias)
    ctx = _ctx(a0, jrand.PRNGKey(3))
    z0 = np.asarray(a0.face_path_policy.head.logits(ctx))
    zb = np.asarray(ab.face_path_policy.head.logits(ctx))
    want = np.zeros_like(z0)
    want[O_SKIP] = -bias
    want[O_QUANT] = -bias
    for s in range(FACE_SLOTS):
        want[slot_base(s) + S_OP + OP_NONE] = bias
    np.testing.assert_allclose(zb - z0, want, atol=1e-4)


# --------------------------------------------- 3. the draw, not the logits

@pytest.mark.parametrize("bias", [0.0, 2.0, 4.0])
def test_the_drawn_skip_rate_matches_the_bias(bias, clamp):
    """2000 draws through the policy's OWN sampler, every op legal.

    The rollout draws one face per key; this is that draw, with the campaign
    clamp on. The tolerance is three standard errors of a Bernoulli at
    ``sigmoid(-B)`` plus a floor, so a correct head cannot fail it and the
    measured ZERO cannot pass it.
    """
    clamp(CAMPAIGN_CLAMP)
    agent = _campaign_agent(bias)
    pol = agent.face_path_policy
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
    tables = precompute_factor_tables(64)
    _NA = 4
    feats = AxisTokenFeatures(
        size=jnp.full((_NA,), 4, jnp.int32),
        log_size=jnp.log(jnp.full((_NA,), 4.0)),
        tag_bits=jnp.zeros((_NA, AXIS_TAG_BITS)).at[:, 0].set(1.0),
        group_id=jnp.zeros((_NA,), jnp.int32),
        valid_mask=jnp.ones((_NA,), jnp.float32),
    )
    n = 2000
    keys = jrand.split(jrand.PRNGKey(0), n)
    pair = (jnp.ones((_NA, _NA), jnp.float32)
            - jnp.eye(_NA, dtype=jnp.float32))
    comp = jnp.ones((_NA,), jnp.float32)

    def _one(k):
        sk, _row, _lp, _e, _ar, _sp, _od = pol.sample_face(
            feats, tables, k, 0, pair, comp, jnp.float32(1.0))
        return sk

    drawn = np.asarray(jax.vmap(_one)(keys)).reshape(-1)
    rate = float((drawn > 0).mean())
    want = float(jax.nn.sigmoid(
        CAMPAIGN_CLAMP * np.tanh(-bias / CAMPAIGN_CLAMP)))
    tol = 3.0 * float(np.sqrt(max(want * (1 - want), 1e-6) / n)) + 0.01
    assert abs(rate - want) < tol, (bias, rate, want)
