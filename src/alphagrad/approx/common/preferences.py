"""Per-episode preference-vector sampling.

Used by the multi-reward trainers (alpha0 / mu0 / gdpo / gfn) when
``--preference-conditioned`` is on: each env in the rollout is assigned a
preference vector ``w ∈ Δ^{NUM_REWARDS-1}`` (a point on the probability
simplex), drawn fresh per episode. The agent's network is conditioned on
``w`` so a single trained model covers the entire reward simplex —
querying it with different ``w`` at inference time recovers the
corresponding Pareto-frontier point.

The default sampling is uniform on the simplex (Dirichlet(1) — equivalent
to taking exponentials of the unit hypercube and normalising). Lower
``alpha`` (< 1) concentrates samples at the corners of the simplex
(emphasising one reward dimension at a time); higher ``alpha`` (> 1)
concentrates near the centre (balanced preferences).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand


def sample_preferences(
    key,
    num_rewards: int,
    num_envs: int,
    dirichlet_alpha: float = 1.0,
    active_mask: "jax.Array | None" = None,
) -> jax.Array:
    """Draw ``num_envs`` preference vectors from ``Dirichlet(alpha · 1)``.

    Returns shape ``(num_envs, num_rewards)`` — each row sums to 1.

    ``active_mask`` (optional, shape ``(num_rewards,)``): when given, the
    Dirichlet is sampled only over the active dimensions and the inactive
    components are forced to zero. Useful when only a subset of the reward
    vector contributes to training (e.g. ``--rewards cmp acc`` selects 2
    of the 8 components — the simplex of interest is 2-D, not 8-D).
    """
    if active_mask is None:
        return jrand.dirichlet(
            key,
            jnp.full((num_rewards,), float(dirichlet_alpha)),
            shape=(num_envs,),
        )

    # Use Gamma → normalise so we can zero out inactive components cleanly.
    alpha = jnp.full((num_rewards,), float(dirichlet_alpha))
    alpha = alpha * active_mask  # zero alpha → degenerate gamma at 0
    # Replace zero alpha with a small positive value to keep the gamma
    # sampler well-defined; we mask the result anyway.
    safe_alpha = jnp.where(alpha > 0.0, alpha, 1.0)
    raw = jrand.gamma(key, safe_alpha, shape=(num_envs, num_rewards))
    raw = raw * active_mask[None, :]                       # zero-out inactive
    norm = raw / jnp.maximum(jnp.sum(raw, axis=-1, keepdims=True), 1e-8)
    return norm
