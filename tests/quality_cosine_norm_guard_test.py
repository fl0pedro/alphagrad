"""The quality cosine is guarded against a ZERO norm, not against a SMALL one.

Defect found 2026-09-16 on the recurrent SHD family and fixed in alphagrad
`727d9070`.  `env._quality_metrics` divided by
``max(||exact||, sqrt(1e-7)) * max(||approx||, sqrt(1e-7))``, which is a FLOOR
at a gradient norm of 3.16e-4.  Below that norm the cosine of a plan that
reproduced the reference EXACTLY reads ``||g||^2 / 1e-7`` instead of 1.0, and
it reads it silently.  Measured on the two-copy window arm (job 66082): two
bit-identical Jacobians, maximum difference exactly 0.0, scored 0.3247,
because the sampled step's gradient norm was 2.19e-4.

These three tests are the ones `727d9070` shipped.  They live in a module of
their own here, because the module they shipped in (`tests/temporal_rule_test.py`)
belongs to the SNN line and is not on this branch, and the thesis matrix takes
that one commit and nothing else from it.

WHY THE THESIS MATRIX CARES.  Every arm trains on `--quality-metric
grad_cosine`, and the quality channel is the constraint of the three
Lagrangian arms.  A floor in that channel prices the gradient's SIZE instead
of the plan whenever the probe batch is small in norm.  The smoke therefore
reports the minimum gradient norm each target's quality channel saw, so the
distance to the old floor is on record.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-4, 1e-6])
def test_two_identical_gradients_score_one_at_every_scale(scale):
    import alphagrad.approx.env as envmod
    g = tuple(jnp.asarray(np.random.RandomState(0).randn(*s), jnp.float32)
              * scale for s in ((8, 5), (5, 5), (3, 5)))
    cos, rel = envmod._quality_metrics(g, g)
    assert abs(float(cos) - 1.0) < 1e-5, (scale, float(cos))
    assert float(rel) < 1e-5


def test_a_zero_reference_still_scores_zero_and_is_dropped():
    """The cosine is UNDEFINED against a zero reference, and the channel drops
    such a batch rather than counting it. The guard against zero stays; only
    the guard against SMALL is gone."""
    import alphagrad.approx.env as envmod
    z = tuple(jnp.zeros(s, jnp.float32) for s in ((8, 5), (5, 5)))
    a = tuple(jnp.ones(s, jnp.float32) for s in ((8, 5), (5, 5)))
    cos, rel = envmod._quality_metrics(z, a)
    assert float(cos) == 0.0
    assert float(rel) == 1.0


def test_a_small_gradient_scores_the_angle_and_not_its_size():
    """Two gradients at a fixed angle score the same cosine whatever their
    length. That is what a cosine means, and what the floor broke."""
    import alphagrad.approx.env as envmod
    rs = np.random.RandomState(1)
    e0 = tuple(jnp.asarray(rs.randn(*s), jnp.float32) for s in ((8, 5), (5, 5)))
    a0 = tuple(x + 0.25 * jnp.asarray(rs.randn(*x.shape), jnp.float32)
               for x in e0)
    ref = float(envmod._quality_metrics(e0, a0)[0])
    for scale in (1e-3, 1e-5, 1e-7):
        e = tuple(x * scale for x in e0)
        a = tuple(x * scale for x in a0)
        got = float(envmod._quality_metrics(e, a)[0])
        assert abs(got - ref) < 1e-4, (scale, got, ref)
