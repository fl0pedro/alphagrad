from __future__ import annotations

import ast
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import alphagrad.approx.ppo as ppo

E, T, K = 16, 80, 3
BETA, SIGMA_MIN, WINSOR = 1e-2, 0.1, 5.0


def _reward():
    # Environment 0 is excluded: its terminal row is the hard sentinel.
    r = np.zeros((E, T, ppo.NUM_REWARDS), np.float32)
    r[0, -1, list(ppo.COMPUTE_REWARD_INDICES)] = ppo.SENTINEL_COST
    return jnp.asarray(r)


def _returns(seed):
    g = np.random.default_rng(seed).normal(size=(E, T, K)).astype(np.float32)
    g[0] = ppo.SENTINEL_COST
    return jnp.asarray(g)


@pytest.mark.parametrize("warm", [False, True])
def test_an_excluded_environment_leaves_sigma_at_the_live_only_value(warm):
    _, live = ppo.refused_env_mask(_reward())
    assert float(live[0].max()) == 0.0 and float(live[1:].min()) == 1.0
    state = tuple(jnp.zeros((K,), jnp.float32) for _ in range(3))
    if warm:
        state = ppo._popart_update(*state, _returns(1)[1:], BETA, SIGMA_MIN,
                                   1e12, WINSOR)
    ret = _returns(0)
    got = ppo._popart_update(*state, ret, BETA, SIGMA_MIN, 1e12, WINSOR,
                             live=live)
    want = ppo._popart_update(*state, ret[1:], BETA, SIGMA_MIN, 1e12, WINSOR)
    mu, sigma = ppo._popart_derive(*got, SIGMA_MIN, 1e12)
    mu_live, sigma_live = ppo._popart_derive(*want, SIGMA_MIN, 1e12)
    np.testing.assert_allclose(np.asarray(sigma), np.asarray(sigma_live),
                               rtol=1e-5)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(mu_live),
                               rtol=1e-5, atol=1e-6)


def test_the_cold_seed_reads_only_the_live_environments():
    _, live = ppo.refused_env_mask(_reward())
    head = np.zeros((E, T, K), np.float32)
    head[:, -1] = np.random.default_rng(2).normal(size=(E, K))
    head[0, -1] = ppo.SENTINEL_COST
    done = np.zeros((E, T), np.float32)
    done[:, -1] = 1.0
    ret = ppo._popart_seed_returns(jnp.asarray(head), jnp.asarray(done),
                                   jnp.ones((E, T), jnp.float32))
    mu, sigma, n = ppo._popart_seed_stats(ret, live, SIGMA_MIN)
    mu_live, sigma_live, n_live = ppo._popart_seed_stats(
        ret[1:], jnp.ones((E - 1, T), jnp.float32), SIGMA_MIN)
    assert float(n) == float(n_live) == (E - 1) * T
    np.testing.assert_allclose(np.asarray(sigma), np.asarray(sigma_live),
                               rtol=1e-5)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(mu_live),
                               rtol=1e-5, atol=1e-6)


def test_a_batch_with_no_live_environment_moves_nothing():
    state = (jnp.full((K,), 0.5, jnp.float32), jnp.full((K,), 2.0, jnp.float32),
             jnp.full((K,), 0.3, jnp.float32))
    got = ppo._popart_update(*state, _returns(0), BETA, SIGMA_MIN, 1e12,
                             WINSOR, live=jnp.zeros((E, T), jnp.float32))
    for new, old in zip(got, state):
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def test_the_episode_update_gives_popart_only_the_live_environments():
    tree = ast.parse(inspect.getsource(ppo))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_episode_update")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    updates = [c for c in calls if c.func.id == "_popart_update"]
    assert updates
    for c in updates:
        assert "live" in {k.arg for k in c.keywords}, ast.unparse(c)
    assert any(c.func.id == "_popart_seed_stats" for c in calls)
