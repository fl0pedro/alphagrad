"""PopArtStats.seed unit test (B1).

Pins the three properties the warm start depends on:
  1. a channel WITH spread is seeded exactly (mu/sigma == the batch's), and
     the accumulators are consistent with the debiased-EMA reads;
  2. a CONSTANT channel is left COLD (w == 0, mu/sigma at the (0,1) init) so
     the debiased EMA still adopts its FIRST real batch exactly;
  3. seeding does not advance ``n_updates`` (consumers gate on real updates).
"""
import numpy as np
import pytest

from alphagrad.approx.common.popart import PopArtStats

K = 3


def _stats(**kw):
    return PopArtStats(K, beta=1e-2, sigma_min=0.1, sigma_max=1e12, **kw)


def test_seed_adopts_batch_stats_on_channels_with_spread():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=[1e6, -3.0, 0.0], scale=[1e5, 2.0, 1.0], size=(64, K))
    s = _stats()
    s.seed(x)
    assert np.allclose(s.mu, x.mean(axis=0), rtol=1e-5)
    assert np.allclose(s.sigma, x.std(axis=0), rtol=1e-5)
    assert np.allclose(s._w, 1.0)
    assert s.n_updates == 0
    # accumulators consistent with the debiased reads
    assert np.allclose(s._mu_acc / s._w, x.mean(axis=0))


def test_seed_leaves_constant_channel_cold():
    x = np.stack([
        np.linspace(1.0, 2.0, 16),      # spread
        np.full(16, 7.0),               # CONSTANT -> must stay cold
        np.linspace(-5.0, 5.0, 16),     # spread
    ], axis=1)
    s = _stats()
    s.seed(x)
    assert s._w[0] == 1.0 and s._w[2] == 1.0
    assert s._w[1] == 0.0
    # (0, 1) init preserved on the cold channel: normalisation is a no-op
    assert s.mu[1] == 0.0 and s.sigma[1] == 1.0
    assert s._mu_acc[1] == 0.0 and s._nu_acc[1] == 0.0


def test_cold_channel_adopts_first_real_batch_exactly():
    x = np.stack([np.linspace(1.0, 2.0, 16), np.full(16, 7.0),
                  np.linspace(-5.0, 5.0, 16)], axis=1)
    s = _stats()
    s.seed(x)
    y = np.stack([np.linspace(0.0, 1.0, 32), np.linspace(100.0, 200.0, 32),
                  np.linspace(0.0, 1.0, 32)], axis=1)
    s.update(y)
    # channel 1 was cold => its debiased mean is the new batch's mean exactly
    assert s.mu[1] == pytest.approx(y[:, 1].mean(), rel=1e-6)
    # a WARM channel must NOT jump to the new batch (it crawls at beta)
    assert abs(float(s.mu[0]) - y[:, 0].mean()) > 1e-3


def test_seed_is_a_noop_on_the_scalar_w_semantics():
    """Without seed(), the per-channel _w must behave exactly like the old
    scalar: identical across channels and equal to 1-(1-beta)**n."""
    s = _stats()
    x = np.arange(3 * K, dtype=np.float64).reshape(3, K)
    for _ in range(5):
        s.update(x + np.random.default_rng(1).normal(size=(3, K)))
    assert np.allclose(s._w, s._w[0])
    assert float(s._w[0]) == pytest.approx(1.0 - (1.0 - 1e-2) ** 5)


def test_seed_rejects_bad_input():
    s = _stats()
    with pytest.raises(ValueError):
        s.seed(np.zeros((4, K + 1)))
    with pytest.raises(ValueError):
        s.seed(np.full((4, K), np.nan))
    with pytest.raises(ValueError):
        s.seed(np.zeros((0, K)))


if __name__ == "__main__":
    test_seed_adopts_batch_stats_on_channels_with_spread()
    test_seed_leaves_constant_channel_cold()
    test_cold_channel_adopts_first_real_batch_exactly()
    test_seed_is_a_noop_on_the_scalar_w_semantics()
    test_seed_rejects_bad_input()
    print("ALL PASS")
