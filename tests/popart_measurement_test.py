"""C1 + C2: PopArt normalization and the measurement protocol.

PopArt replaces the per-batch advantage z-score, which has a collapse
ratchet: once a batch is uniformly degenerate a channel's std -> 0 and that
channel's (opposing) advantage VANISHES. A debiased EMA carried across
episodes keeps a stable per-channel scale instead.

Two bugs these pin, both found while wiring it:
  * winsorizing against the cold (0, 1) init crushed a 1e7-scale channel
    to +/-5 and debiasing could never recover the scale;
  * returning the DEBIASED value and feeding it back as the raw EMA
    accumulator double-counts (mu ran 1e7 -> 5e8 in two updates).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import EnvConfig
from alphagrad.approx.ppo import (
    NUM_VALUE_HEADS,
    _popart_derive,
    _popart_update,
    make_argparser,
)

SIGMA_MIN, SIGMA_MAX, BETA, WINSOR_K = 0.1, 1e12, 1e-2, 5.0


def _mixed_scale_returns(seed=0):
    """(E, T, K) targets with the real cross-channel gap: a ~1e7 cost channel
    and a ~1 quality channel."""
    r = np.random.default_rng(seed).normal(size=(8, 6, NUM_VALUE_HEADS))
    r = r.astype(np.float32)
    r[..., 0] = r[..., 0] * 1e6 + 1e7      # flops-like
    r[..., 2] = r[..., 2] * 0.05 + 0.9     # cosine-like
    return jnp.asarray(r)


def _zeros():
    z = jnp.zeros((NUM_VALUE_HEADS,), dtype=jnp.float32)
    return z, z, z


def test_cold_state_is_the_identity_transform():
    m1, m2, w = _zeros()
    mu, sigma = _popart_derive(m1, m2, w, SIGMA_MIN, SIGMA_MAX)
    assert np.allclose(np.asarray(mu), 0.0)
    assert np.allclose(np.asarray(sigma), 1.0)


def test_first_update_adopts_the_batch_scale():
    """Debiasing must make update #1 land on the batch stats — not crawl away
    from the (0,1) init, and not be clipped to it by the winsorizer."""
    m1, m2, w = _zeros()
    R = _mixed_scale_returns()
    m1, m2, w = _popart_update(m1, m2, w, R, BETA, SIGMA_MIN, SIGMA_MAX, WINSOR_K)
    mu, _ = _popart_derive(m1, m2, w, SIGMA_MIN, SIGMA_MAX)
    assert float(mu[0]) == pytest.approx(float(jnp.mean(R[..., 0])), rel=0.2)
    assert float(mu[2]) == pytest.approx(float(jnp.mean(R[..., 2])), rel=0.2)


def test_repeated_updates_stay_stable():
    """The accumulator must not re-inflate when fed back (the 1e7 -> 5e8 bug)."""
    m1, m2, w = _zeros()
    R = _mixed_scale_returns()
    for _ in range(8):
        m1, m2, w = _popart_update(m1, m2, w, R, BETA, SIGMA_MIN, SIGMA_MAX, WINSOR_K)
    mu, sigma = _popart_derive(m1, m2, w, SIGMA_MIN, SIGMA_MAX)
    assert 5e6 < float(mu[0]) < 2e7, f"cost-channel mu drifted: {float(mu[0])}"
    assert 0.5 < float(mu[2]) < 1.5, f"quality-channel mu drifted: {float(mu[2])}"
    assert float(sigma[0]) > 0.0 and float(sigma[2]) > 0.0


def test_normalization_puts_both_scales_on_one_footing():
    """The whole point: a 1e7 channel and a [0,1] channel both become O(1)."""
    m1, m2, w = _zeros()
    R = _mixed_scale_returns()
    for _ in range(4):
        m1, m2, w = _popart_update(m1, m2, w, R, BETA, SIGMA_MIN, SIGMA_MAX, WINSOR_K)
    mu, sigma = _popart_derive(m1, m2, w, SIGMA_MIN, SIGMA_MAX)
    z = (R - mu) / sigma
    for k in range(NUM_VALUE_HEADS):
        assert float(jnp.mean(jnp.abs(z[..., k]))) < 5.0


def test_sigma_floor_prevents_advantage_blowup_on_a_uniform_channel():
    """A channel that goes uniform (the collapse case) must not get an
    unbounded 1/sigma amplification."""
    m1, m2, w = _zeros()
    R = np.zeros((4, 4, NUM_VALUE_HEADS), dtype=np.float32)
    R[..., 2] = 0.5                      # perfectly uniform channel
    Rj = jnp.asarray(R)
    for _ in range(5):
        m1, m2, w = _popart_update(m1, m2, w, Rj, BETA, SIGMA_MIN, SIGMA_MAX, WINSOR_K)
    _, sigma = _popart_derive(m1, m2, w, SIGMA_MIN, SIGMA_MAX)
    assert float(sigma[2]) >= SIGMA_MIN


# --------------------------------------------------------------------------- #
# C2 — measurement protocol
# --------------------------------------------------------------------------- #

def test_measurement_flags_are_wired_not_swallowed():
    """`from_jaxpr(**_compat)` used to silently eat these, so passing them
    looked like it worked and did nothing."""
    a = make_argparser().parse_args(["--example", "X"])
    assert (a.num_data_points, a.reps_per_point) == (5, 4)   # spec 5 x 4 = 20
    cfg = EnvConfig(jaxpr=None, argnums=(), has_aux=False, sparse=False,
                    cmp_type="flops", mem_type="peak_memory")
    assert (cfg.num_data_points, cfg.reps_per_point) == (5, 4)


def test_advantage_norm_defaults_to_popart():
    a = make_argparser().parse_args(["--example", "X"])
    assert a.advantage_norm == "popart"
