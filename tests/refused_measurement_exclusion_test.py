"""A REFUSED terminal measurement trains nothing (AGENTS.md: missing data).

The defect this pins, from smoke job 66201 (``[health ep0] ppo=1.227e+09``):

``_is_degen`` in ``train_episode`` is a per-(env, step) test on the RAW
reward, and under ``--terminal-rewards-only`` only the LAST step of an
environment carries a reward at all. GAE has already run when that test is
taken, and its backward recursion

    advantage_t = delta_t + discount * gae_lambda * advantage_{t+1}

carries the terminal row's -1e10 into every earlier step of the SAME
environment -- undiminished at the campaign's ``--discount 1.0 --gae-lambda
1.0``. Masking only the sentinel step left T-1 contaminated transitions per
refused environment in the actor gradient, in the critic's target and in the
logged ``ppo`` statistic. Under ``--cost-form paired-log`` the latency and
peak-memory channels are exempt from symlog, so those rows arrive at the loss
at the full -1e10 scale, which is where a 1.2e9 loss comes from.

What the tests below pin:

1. ``sentinel_step_mask`` still finds exactly the sentinel row.
2. The GAE really does carry it backwards -- the mechanism, measured, so a
   later refactor cannot quietly re-open the hole.
3. ``refused_env_mask`` excludes EVERY step of the refused environment.
4. The actor and the critic gradients of a two-environment episode with one
   refused environment are IDENTICAL to the one-environment episode without
   it, and so is the logged loss statistic.
5. ``_live_mean`` is bitwise ``jnp.mean`` when nothing is refused.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.common.gae import get_advantages          # noqa: E402
from alphagrad.approx.ppo import (                              # noqa: E402
    _live_mean,
    _per_channel_value_loss,
    refused_env_mask,
    sentinel_step_mask,
)

NUM_REWARDS = int(ppo.NUM_REWARDS)
SENTINEL_COST = float(ppo.SENTINEL_COST)
COMPUTE = tuple(int(i) for i in ppo.COMPUTE_REWARD_INDICES)

E = 2          # two environments, the second one refused
T = 4          # four steps, so three of them are NON-terminal
K = 4          # value heads, as HEAD_REWARD_INDICES has


def _reward_tensor(refused_env=1):
    """(E, T, NUM_REWARDS) raw rewards: terminal-only, one refused row."""
    r = np.zeros((E, T, NUM_REWARDS), dtype=np.float32)
    # A measured terminal for every environment: negated costs, bounded
    # quality slots. Nothing here is near the sentinel.
    for e in range(E):
        r[e, T - 1, :] = -1.0 * (e + 2)
        r[e, T - 1, int(ppo.REWARD_INDEX["latency_ns"])] = -(3.0e5 + e)
        r[e, T - 1, int(ppo.REWARD_INDEX["peak_memory"])] = -(7.0e7 + e)
        r[e, T - 1, int(ppo.REWARD_INDEX["cosine_sim"])] = 0.5
    if refused_env is not None:
        # THE REFUSAL: the exact sentinel vector the env returns.
        r[refused_env, T - 1, :] = 0.0
        for c in COMPUTE:
            r[refused_env, T - 1, c] = SENTINEL_COST
        r[refused_env, T - 1, int(ppo.REWARD_INDEX["frob_residual"])] = -1.0
    return jnp.asarray(r)


def _gae_inputs(reward):
    """Head rewards, dones, discounts and values for the real GAE.

    ``--cost-form paired-log`` exempts latency and peak_memory from symlog,
    so the head rewards carry the sentinel RAW. That is the campaign's own
    configuration and the one the 1.2e9 loss came from.
    """
    idx = jnp.asarray(
        [int(ppo.REWARD_INDEX["latency_ns"]),
         int(ppo.REWARD_INDEX["peak_memory"]),
         int(ppo.REWARD_INDEX["cosine_sim"]),
         int(ppo.REWARD_INDEX["frob_residual"])], dtype=jnp.int32)
    head_rewards = reward[..., idx]                              # (E,T,K)
    done = jnp.zeros((E, T), jnp.float32).at[:, T - 1].set(1.0)
    discount = jnp.ones((E, T), jnp.float32)                     # --discount 1.0
    rng = np.random.default_rng(7)
    value = jnp.asarray(rng.normal(0.0, 0.3, (E, T, K)).astype(np.float32))
    next_value = jnp.asarray(
        rng.normal(0.0, 0.3, (E, T, K)).astype(np.float32))
    return head_rewards, done, discount, value, next_value


def test_sentinel_step_mask_finds_exactly_the_refused_terminal():
    m = np.asarray(sentinel_step_mask(_reward_tensor(refused_env=1)))
    assert m.shape == (E, T)
    want = np.zeros((E, T), dtype=bool)
    want[1, T - 1] = True
    assert np.array_equal(m, want)


def test_a_merely_expensive_plan_is_not_a_sentinel():
    """The ep-39 cliff: channel 0 is -muls_adds_fmas, ~1e12 on nn256."""
    r = np.zeros((1, 1, NUM_REWARDS), dtype=np.float32)
    r[0, 0, 0] = -4.0e12       # more negative than the sentinel, one channel
    r[0, 0, 3] = -2.0e12
    assert not bool(np.asarray(sentinel_step_mask(jnp.asarray(r)))[0, 0])


def test_the_gae_carries_the_sentinel_into_every_earlier_step():
    """THE MECHANISM, measured. Not a regression guard on our own fix: this
    is the reason a per-step mask was never enough."""
    reward = _reward_tensor(refused_env=1)
    hr, done, disc, v, nv = _gae_inputs(reward)
    _, _, advantages = get_advantages(hr, done, v, nv, disc, 1.0)
    adv = np.asarray(advantages)
    # Every step of the refused environment, terminal and non-terminal, is
    # at the sentinel scale.
    assert np.abs(adv[1, :, 0]).min() > 1e9
    # The measured environment is nowhere near it.
    assert np.abs(adv[0, :, 0]).max() < 1e7


def test_refused_env_mask_excludes_every_step_of_that_environment():
    env_refused, live = refused_env_mask(_reward_tensor(refused_env=1))
    assert np.array_equal(np.asarray(env_refused), np.array([False, True]))
    assert np.array_equal(
        np.asarray(live), np.array([[1.0] * T, [0.0] * T], dtype=np.float32))


def test_nothing_is_excluded_when_nothing_is_refused():
    env_refused, live = refused_env_mask(_reward_tensor(refused_env=None))
    assert not bool(np.asarray(env_refused).any())
    assert np.array_equal(np.asarray(live), np.ones((E, T), np.float32))


def test_the_masked_advantage_of_a_refused_environment_is_exactly_zero():
    reward = _reward_tensor(refused_env=1)
    hr, done, disc, v, nv = _gae_inputs(reward)
    _, estim_returns, advantages = get_advantages(hr, done, v, nv, disc, 1.0)
    _, live = refused_env_mask(reward)
    advantages = advantages * live[..., None]
    # (b) THE VALUE TARGET: the critic's own prediction, so its residual on
    # those rows is 0 -- partial-episode bootstrapping, the treatment
    # `_truncated_reward` documents.
    estim_returns = jnp.where(
        live[..., None] > 0.5, estim_returns, ppo._value_decode(v))
    assert np.array_equal(
        np.asarray(advantages)[1], np.zeros((T, K), np.float32))
    assert np.allclose(
        np.asarray(estim_returns)[1], np.asarray(ppo._value_decode(v))[1],
        rtol=0, atol=0)


# ---------------------------------------------------------------------------
# THE GRADIENT PROOF.
#
# `_dynamic_loss_fn` lives inside `main()`'s closure and cannot be imported,
# so the three terms of `total_loss` that consume a measurement are rebuilt
# here in EXACTLY the form the loss uses them -- the clipped surrogate over
# `norm_adv`, `_per_channel_value_loss` over `estim_returns`, and the entropy
# bonus -- each through the same `_live_mean` the loss now takes. What is
# proved is the property the fix is about: the two-environment batch with one
# refused environment produces the SAME loss and the SAME gradient as the
# one-environment batch that never had it.
# ---------------------------------------------------------------------------

def _flat_batch(reward):
    """The flattened per-sample batch the loss sees, with the trainer's own
    masking already applied (advantage 0, neutral value target)."""
    hr, done, disc, v, nv = _gae_inputs(reward)
    _, estim_returns, advantages = get_advantages(hr, done, v, nv, disc, 1.0)
    _, live = refused_env_mask(reward)
    advantages = advantages * live[..., None]
    estim_returns = jnp.where(
        live[..., None] > 0.5, estim_returns, ppo._value_decode(v))
    # --advantage-norm none: norm_adv_components IS advantages (SHARED_CLI).
    pref = jnp.asarray(
        np.full((K,), 0.25, dtype=np.float32))
    norm_adv = jnp.sum(advantages * pref, axis=-1)               # (E,T)
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(0.0, 0.5, (E, T)).astype(np.float32))
    e_in = jnp.asarray(rng.normal(1.0, 0.2, (E, T)).astype(np.float32))
    v_in = jnp.asarray(rng.normal(0.0, 0.4, (E, T, K)).astype(np.float32))
    return {
        "x": x.reshape(-1),
        "e_in": e_in.reshape(-1),
        "v_in": v_in.reshape(-1, K),
        "norm_adv": norm_adv.reshape(-1),
        "estim_returns": estim_returns.reshape(-1, K),
        "live": live.reshape(-1),
    }


def _keep_env(batch, e):
    """The same batch with ONLY environment `e`'s samples, all live."""
    sl = slice(e * T, (e + 1) * T)
    out = {k: v[sl] for k, v in batch.items()}
    out["live"] = jnp.ones((T,), jnp.float32)
    return out


def _surrogate(theta, b):
    """ppo_loss + value_weight * value_loss - entropy_weight * entropy, each
    averaged the way `_dynamic_loss_fn` now averages it."""
    ratio = jnp.exp(theta[0] * b["x"])
    obj = jnp.minimum(
        ratio * b["norm_adv"],
        jnp.clip(ratio, 0.8, 1.2) * b["norm_adv"])
    ppo_loss = _live_mean(-obj, b["live"])
    values = theta[1] * b["v_in"]
    value_loss, _ = _per_channel_value_loss(
        values, b["estim_returns"], b["live"])
    entropy = _live_mean(theta[0] * b["e_in"], b["live"])
    return ppo_loss + 0.5 * value_loss - 0.05 * entropy


def test_actor_and_critic_gradients_ignore_the_refused_environment():
    both = _flat_batch(_reward_tensor(refused_env=1))
    only_measured = _keep_env(both, 0)
    theta = jnp.asarray([0.3, 0.7], jnp.float32)
    g_both = np.asarray(jax.grad(_surrogate)(theta, both))
    g_one = np.asarray(jax.grad(_surrogate)(theta, only_measured))
    assert np.all(np.isfinite(g_both))
    assert np.allclose(g_both, g_one, rtol=0.0, atol=0.0), (g_both, g_one)


def test_the_logged_loss_statistic_ignores_the_refused_environment():
    both = _flat_batch(_reward_tensor(refused_env=1))
    only_measured = _keep_env(both, 0)
    theta = jnp.asarray([0.3, 0.7], jnp.float32)
    l_both = float(_surrogate(theta, both))
    l_one = float(_surrogate(theta, only_measured))
    assert l_both == l_one
    # And it is a sane number, not the 1.2e9 of job 66201.
    assert abs(l_both) < 1e3


def test_without_the_environment_mask_the_statistic_explodes():
    """The counterfactual: the per-STEP mask, which is what job 66201 ran."""
    reward = _reward_tensor(refused_env=1)
    hr, done, disc, v, nv = _gae_inputs(reward)
    _, _, advantages = get_advantages(hr, done, v, nv, disc, 1.0)
    step_live = (~sentinel_step_mask(reward)).astype(jnp.float32)
    advantages = advantages * step_live[..., None]
    pref = jnp.asarray(np.full((K,), 0.25, dtype=np.float32))
    norm_adv = jnp.sum(advantages * pref, axis=-1).reshape(-1)
    assert float(jnp.mean(-norm_adv)) > 1e8


# ---------------------------------------------------------------------------
# `_live_mean` itself.
# ---------------------------------------------------------------------------

def test_live_mean_is_bitwise_the_plain_mean_when_everything_is_live():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(0.0, 1.0, (32,)).astype(np.float32))
    w = jnp.ones((32,), jnp.float32)
    assert float(_live_mean(x, w)) == float(jnp.mean(x))
    y = jnp.asarray(rng.normal(0.0, 1.0, (32, 5)).astype(np.float32))
    a = np.asarray(_live_mean(y, w, axis=0))
    b = np.asarray(jnp.mean(y, axis=0))
    assert np.array_equal(a, b)


def test_live_mean_equals_the_mean_over_the_kept_rows():
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(0.0, 1.0, (10,)).astype(np.float32))
    w = jnp.asarray(np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 1], np.float32))
    kept = jnp.asarray(np.asarray(x)[np.asarray(w) > 0.5])
    assert float(_live_mean(x, w)) == float(jnp.mean(kept))


def test_live_mean_of_an_all_refused_batch_is_zero_not_nan():
    x = jnp.asarray(np.array([1.0, 2.0, 3.0], np.float32))
    w = jnp.zeros((3,), jnp.float32)
    out = float(_live_mean(x, w))
    assert out == 0.0
    assert np.isfinite(out)
