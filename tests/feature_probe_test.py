"""THE PROBE MUST NOT TRAIN THE THING IT MEASURES. This pins that.

The whole value of an online probe is that it is a passive reader: palimpsa,
the vertex policy and the face head are trained by PPO's reward ALONE, and the
probe decodes whatever that produces. If any probe gradient leaks into the
encoder, the probe stops measuring the representation and starts creating it --
and the resulting R2 would be an artefact of the probe's own supervision, which
is exactly the confound that separates this from the offline campaign (where
the whole agent WAS trained end-to-end on the probe loss).

"The probe weight is small" is not the same guarantee and is not accepted here.
The test differentiates the probe loss with respect to the AGENT and asserts
the cotangent is exactly zero on every leaf -- an algebraic property of
stop_gradient, so it holds at any weight, any learning rate, any number of
steps.

The companion assertion matters just as much: the probe's OWN parameters must
receive a nonzero gradient. Two zeros would satisfy "no leak" trivially while
the probe silently learned nothing -- the same failure mode
palimpsa_base_grad_test exists to catch on the encoder side.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "512")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common import feature_probe as FP  # noqa: E402

EMBD = 32
NAXES = 8
NFACE = 6
NV = 5


@pytest.fixture(scope="module")
def probes():
    # LEAN = exactly the live head's input (the face latent alone).
    return FP.FeatureProbes(
        embd_dim=EMBD, max_axes=NAXES, n_vertex_out=3, width=32,
        key=jax.random.PRNGKey(0), arm=FP.FaceProbeArm.LEAN)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    return dict(
        ctx_i=jnp.asarray(rng.normal(size=(NFACE, EMBD)).astype(np.float32)),
        ctx_j=jnp.asarray(rng.normal(size=(NFACE, EMBD)).astype(np.float32)),
        lat=jnp.asarray(rng.normal(size=(NFACE, EMBD)).astype(np.float32)),
        tgt=jnp.asarray(rng.normal(size=(NFACE, FP.NFT)).astype(np.float32)),
        valid=jnp.asarray(np.array([1, 1, 1, 1, 0, 0], np.float32)),
    )


def _face_loss(probes, d):
    pred = jax.vmap(probes.face_predict)(d["lat"])
    return FP.masked_mse(pred, d["tgt"], d["valid"]).sum()


def test_probe_loss_has_no_gradient_to_its_inputs(probes, data):
    """The representation is a CONSTANT to the probe loss."""
    def loss_wrt_inputs(lat):
        return _face_loss(probes, dict(data, lat=lat))

    g = (jax.grad(loss_wrt_inputs)(data["lat"]),)
    for name, gi in zip(("face_latent",), g):
        arr = np.asarray(gi)
        assert np.all(arr == 0.0), (
            f"probe gradient LEAKED into {name}: max|g|={np.abs(arr).max()}. "
            "stop_gradient is missing or was applied after the concat.")


def test_probe_parameters_do_get_a_gradient(probes, data):
    """...and the probe itself must actually learn, or the above is vacuous."""
    g = eqx.filter_grad(lambda p: _face_loss(p, data))(probes)
    leaves = [x for x in jax.tree_util.tree_leaves(g)
              if eqx.is_inexact_array(x)]
    assert leaves, "no differentiable probe parameters at all"
    tot = sum(float(jnp.sum(jnp.abs(x))) for x in leaves)
    assert tot > 0.0, "the probe's own gradient is identically zero"


def test_masked_mse_ignores_padded_rows(data):
    """Padding must not enter the denominator.

    Averaging over padded slots is how approx_prob came to report 99.2% when
    it was reporting the padding ratio; the same arithmetic here would make a
    probe look better the more padding it was handed.
    """
    pred = jnp.zeros((NFACE, FP.NFT), jnp.float32)
    tgt = jnp.zeros((NFACE, FP.NFT), jnp.float32)
    # Put a huge error ONLY in an invalid row.
    tgt = tgt.at[5].set(1e6)
    out = FP.masked_mse(pred, tgt, data["valid"])
    assert float(jnp.max(jnp.abs(out))) == 0.0, (
        "an invalid row contributed to the loss")


def test_masked_mse_is_zero_not_nan_when_nothing_is_valid():
    pred = jnp.ones((NFACE, FP.NFT), jnp.float32)
    tgt = jnp.zeros((NFACE, FP.NFT), jnp.float32)
    out = FP.masked_mse(pred, tgt, jnp.zeros((NFACE,), jnp.float32))
    assert np.all(np.isfinite(np.asarray(out))), "empty mask produced NaN"


def test_within_step_r2_is_one_for_a_perfect_decode():
    rng = np.random.default_rng(3)
    n, S, C = 40, 4, 2
    step = jnp.asarray(rng.integers(0, S, (n,)).astype(np.int32))
    tgt = jnp.asarray(rng.normal(size=(n, C)).astype(np.float32))
    valid = jnp.ones((n,), jnp.float32)
    r2 = FP.within_step_r2(tgt, tgt, valid, step, S)
    assert np.allclose(np.asarray(r2), 1.0, atol=1e-4), np.asarray(r2)


def test_within_step_r2_is_zero_for_the_per_step_mean():
    """Predicting only each step's mean must score 0, not something flattering.

    This is the entire reason the offline campaign used the WITHIN-step
    statistic: the per-step mean is recoverable from the step index alone, so
    a probe that learns only that has learned nothing about the face.
    """
    rng = np.random.default_rng(4)
    n, S = 200, 5
    step = np.asarray(rng.integers(0, S, (n,)))
    base = rng.normal(size=(S,)) * 10.0
    tgt = (base[step] + rng.normal(size=(n,))).astype(np.float32)[:, None]
    pred = base[step].astype(np.float32)[:, None]
    r2 = FP.within_step_r2(
        jnp.asarray(pred), jnp.asarray(tgt), jnp.ones((n,), jnp.float32),
        jnp.asarray(step.astype(np.int32)), S)
    assert abs(float(r2[0])) < 0.05, (
        f"per-step-mean predictor scored R2={float(r2[0]):.3f}, expected ~0")


def test_degenerate_target_scores_zero_not_one():
    """A constant column has no variance to explain; it must not read 1.0."""
    n, S = 30, 3
    step = jnp.asarray(np.arange(n) % S, dtype=jnp.int32)
    const = jnp.ones((n, 1), jnp.float32) * 7.0
    r2 = FP.within_step_r2(const, const, jnp.ones((n,), jnp.float32), step, S)
    assert float(r2[0]) == 0.0, "a degenerate target reported a perfect decode"


def test_reference_arms_refuse_to_run_without_their_extra_inputs():
    """A reference arm must FAIL LOUDLY rather than silently score the lean
    input under a different name -- that would corrupt the paired rows."""
    ep = FP.FeatureProbes(
        embd_dim=EMBD, max_axes=NAXES, n_vertex_out=3, width=16,
        key=jax.random.PRNGKey(2), arm=FP.FaceProbeArm.ENDPOINTS)
    with pytest.raises(ValueError, match="ENDPOINTS"):
        ep.face_predict(jnp.zeros(EMBD))
    ex = FP.FeatureProbes(
        embd_dim=EMBD, max_axes=NAXES, n_vertex_out=3, width=16,
        key=jax.random.PRNGKey(3), arm=FP.FaceProbeArm.EXTENTS)
    with pytest.raises(ValueError, match="EXTENTS"):
        ex.face_predict(jnp.zeros(EMBD))


def test_lean_arm_input_width_matches_the_live_head():
    """The whole point: the probe sees what UnifiedFaceHead sees.
    UnifiedFacePolicy builds UnifiedFaceHead(embd_dim, in_dim=embd_dim), and
    _repr returns the face latent unchanged -- so LEAN must be E, not 3E."""
    assert FP.face_input_dim(FP.FaceProbeArm.LEAN, EMBD, NAXES) == EMBD
    assert FP.face_input_dim(FP.FaceProbeArm.ENDPOINTS, EMBD, NAXES) == 3 * EMBD
    assert FP.face_input_dim(FP.FaceProbeArm.EXTENTS, EMBD, NAXES) == EMBD + NAXES


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown face probe arm"):
        FP.FeatureProbes(embd_dim=EMBD, max_axes=NAXES, n_vertex_out=3,
                         width=16, key=jax.random.PRNGKey(4), arm="nope")


def test_steps_to_threshold():
    assert FP.steps_to_threshold([0.0, 0.2, 0.55, 0.8], 0.5) == 2
    assert FP.steps_to_threshold([0.0, 0.1], 0.5) == -1


def test_bars_are_labelled_as_a_reference_not_a_lean_target():
    """The bars were measured on the ENDPOINTS arm plus extents. The lean arm
    removes ctx_i/ctx_j on top of that, so BARS is a reference line."""
    for name in FP.FACE_TARGETS:
        assert name in FP.BARS
        assert name in FP.OFFLINE_ENDPOINTS_NULL


def test_names_and_arity_agree():
    assert len(FP.FACE_NAMES) == FP.NFT
