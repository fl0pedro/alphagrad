"""GAE-λ and reward-normalisation helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp

from alphagrad.utils import symexp, symlog


def reward_normalization_fn(reward):
    """Symlog-based reward normalisation; matches the existing PPO/GDPO behaviour."""
    return symlog(reward)


def inverse_reward_normalization_fn(reward):
    return symexp(reward)


def get_num_clipping_triggers(ratio, eps):
    """Approximate count of how many ratios fell into the PPO clip region."""
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    """GAE-λ over a single env trajectory; vmapped over the batch dimension."""

    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj

        mask = 1.0 - done
        episodic_return = reward + discount * episodic_return * mask

        value_raw = inverse_reward_normalization_fn(value)
        next_value_raw = inverse_reward_normalization_fn(next_value)

        delta = reward + next_value_raw * discount * mask - value_raw
        advantage = delta + discount * gae_lambda * lastgaelam * mask

        estim_return = advantage + value_raw
        return (episodic_return, advantage), (episodic_return, estim_return, advantage)

    inputs = (rewards, dones, values, next_values, discounts)
    rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
    init_val = jnp.zeros_like(rewards[0])
    _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
    return jax.tree.map(lambda x: x[::-1], output)
