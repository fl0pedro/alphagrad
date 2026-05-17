"""GAE-λ, reward-normalisation, and advantage-aggregation helpers.

This module hosts three closely related concerns:

1. **Scalar reward normalisation** (`reward_normalization_fn` / its
   inverse). Historically applied unconditionally inside GAE via
   ``symlog`` to squash multi-decade reward magnitudes; the GDPO
   pipeline (see `gdpo_normalise_advantages` below) makes this
   compression redundant, so callers can opt out via the factory.
2. **Generalized advantage estimation** (`get_advantages`). Element-wise
   over any trailing axis, so single-channel (legacy) and per-channel
   (GDPO) callers share the same scan kernel.
3. **GDPO-style advantage aggregation** (`gdpo_normalise_advantages`).
   Per-channel z-score over a minibatch, priority-weighted sum, then
   batch-norm of the summed advantage. Matches the construction from
   NVIDIA's GDPO paper (arXiv:2601.05242).
"""

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


def _identity(x):
    return x


def make_get_advantages(use_symlog: bool):
    """Build a jitted ``get_advantages`` variant.

    With ``use_symlog=True`` the legacy behaviour is preserved: GAE
    treats stored ``value`` / ``next_value`` as symlog-encoded and runs
    them through ``symexp`` before the delta computation. With
    ``use_symlog=False`` the value targets are raw — appropriate when
    the caller normalises advantages via :func:`gdpo_normalise_advantages`
    (which subsumes the magnitude compression symlog used to provide).
    """
    inv = inverse_reward_normalization_fn if use_symlog else _identity

    @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
    def _get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
        def loop_fn(carry, traj):
            episodic_return, lastgaelam = carry
            reward, done, value, next_value, discount = traj

            mask = 1.0 - done
            episodic_return = reward + discount * episodic_return * mask

            value_raw = inv(value)
            next_value_raw = inv(next_value)

            delta = reward + next_value_raw * discount * mask - value_raw
            advantage = delta + discount * gae_lambda * lastgaelam * mask

            estim_return = advantage + value_raw
            return (episodic_return, advantage), (episodic_return, estim_return, advantage)

        inputs = (rewards, dones, values, next_values, discounts)
        rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
        init_val = jnp.zeros_like(rewards[0])
        _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
        return jax.tree.map(lambda x: x[::-1], output)

    return jax.jit(_get_advantages)


# Legacy entry point — preserves call sites that imported `get_advantages`
# directly. Equivalent to ``make_get_advantages(use_symlog=True)``.
get_advantages = make_get_advantages(True)


def get_num_clipping_triggers(ratio, eps):
    """Approximate count of how many ratios fell into the PPO clip region."""
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@jax.jit
def gdpo_normalise_advantages(
    advantages,        # (B, K) per-minibatch per-channel
    channel_mask,      # (K,) 1 where reward_weight != 0
    sparse_mask,       # (K,) 1 where channel is sparse-terminal (cos, frob)
    priority_weights,  # (K,) user --lambda-* values (now pure priorities)
):
    """GDPO advantage aggregation in three stages:

    1. **Per-channel z-score** over the minibatch (axis 0). Channels
       with ``channel_mask == 0`` get neutralised (μ=0, σ=1) so their
       contribution to the summed advantage is exactly zero regardless
       of input.
    2. **Priority-weighted sum** across channels — the ``--lambda-*``
       knobs now express pure priority because the per-channel
       z-score already removed magnitude.
    3. **Batch-wise normalisation** of the summed advantage. The
       GDPO paper's Appendix B shows omitting this step occasionally
       leads to convergence failures as the channel count grows.

    Sparse-terminal channels (cos / frob) are mostly zero at
    intermediate timesteps; their per-batch mean would be near zero and
    z-scoring would inflate noise. The override divides them by σ
    without recentering, so the terminal-step signal still surfaces but
    intermediate zeros stay zero.

    Returns shape ``(B,)`` — the scalar advantage the PPO surrogate
    consumes.
    """
    eps = 1e-6

    # Per-channel statistics. Inactive channels (mask 0) get μ=0/σ=1
    # so the (a − μ)/σ step is a no-op — they emerge as their raw
    # values, then get zeroed by the `* channel_mask` in the sum below.
    raw_mu = jnp.mean(advantages, axis=0)
    raw_var = jnp.var(advantages, axis=0)
    mu = jnp.where(channel_mask > 0.5, raw_mu, 0.0)
    sigma = jnp.where(channel_mask > 0.5, jnp.sqrt(raw_var), 1.0)
    sigma = sigma + eps

    z_dense = (advantages - mu[None, :]) / sigma[None, :]
    # Sparse-terminal channels: rescale without recentre. Most entries
    # are zero so recentring would shift them to a (large) negative
    # constant; the terminal step would then look only marginally
    # different from the rest.
    z_sparse = advantages / sigma[None, :]
    z = jnp.where(sparse_mask[None, :] > 0.5, z_sparse, z_dense)

    # Priority-weighted sum. ``channel_mask`` zeros out inactive heads
    # for safety even if ``priority_weights[k] != 0`` somehow.
    weights = priority_weights * channel_mask  # (K,)
    a_sum = jnp.sum(z * weights[None, :], axis=-1)  # (B,)

    # Batch-wise final normalisation. ``+ eps`` floor handles the
    # degenerate case of a constant minibatch (all rollouts produced
    # identical advantages on every channel).
    a_hat = (a_sum - jnp.mean(a_sum)) / (jnp.std(a_sum) + eps)
    return a_hat
