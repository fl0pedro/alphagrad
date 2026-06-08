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


# ---------------------------------------------------------------------------
# Low-discrepancy preference sampling (RQ9 / Pitch B)
#
# Dirichlet has O(N^{-1/2}) discrepancy on the simplex; the Kronecker /
# R_d quasirandom sequence has O(log N / N). For preference-conditioned
# RL, that means far more uniform coverage of the front in the same
# number of samples — directly addresses Pitch B's "front coverage"
# motivation (``docs/experiments/pareto_front_tchebycheff.md``).
# ---------------------------------------------------------------------------


def _plastic_constant(num_dims: int) -> float:
    """Unique positive real root of ``x^(d+1) = x + 1``.

    For d=1 this is the golden ratio φ; for d=2 the plastic constant
    ψ ≈ 1.32472; for higher d, the "generalised golden ratios" from
    Roberts 2018 ("The Unreasonable Effectiveness of Quasirandom
    Sequences"). Solved by Newton's method — a few iterations from
    x=1.5 converges to machine precision.
    """
    x = 1.5
    for _ in range(64):
        # f(x) = x^(d+1) - x - 1; f'(x) = (d+1) x^d - 1
        x_pow = x ** num_dims
        f = x * x_pow - x - 1.0
        df = (num_dims + 1) * x_pow - 1.0
        x_new = x - f / df
        if abs(x_new - x) < 1e-15:
            return float(x_new)
        x = x_new
    return float(x)


def kronecker_preferences(
    num_rewards: int,
    num_envs: int,
    *,
    offset: int = 0,
    active_mask: "jax.Array | None" = None,
) -> jax.Array:
    """Draw ``num_envs`` preference vectors from the R_d low-discrepancy
    sequence on the simplex.

    Args:
        num_rewards: K — dimensionality of the preference vector.
        num_envs: B — number of preference vectors to return.
        offset: global step counter (e.g. ``episode * num_envs``) so
            successive calls cover the simplex deterministically rather
            than re-drawing the first B points. Pass the running
            ``episode * num_envs`` to walk the sequence.
        active_mask: optional ``(num_rewards,)`` mask zeroing inactive
            components; remaining are renormalised to a simplex on the
            active subset (same semantics as Dirichlet's active_mask).

    Returns ``(num_envs, num_rewards)`` float32; each row sums to 1.

    The construction (Roberts 2018):
        phi_K        = unique positive real root of x^(K+1) = x + 1
        g_k          = 1 / phi_K^(k+1)        for k ∈ [1, K]
        raw[n, k]    = (offset + n) * g_k mod 1
        prefs[n, :]  = raw[n, :] / sum(raw[n, :])

    For K=1 the construction degenerates (preferences are trivially
    1.0 in 1-D); for K=2 ``phi_K`` is the golden ratio and you get the
    classical golden-ratio Kronecker sequence on the line; for K>2
    you get the R_d generalisation.
    """
    K = int(num_rewards)
    B = int(num_envs)
    assert K >= 1, f"num_rewards must be >= 1, got {K}"
    if K == 1:
        out = jnp.ones((B, 1), dtype=jnp.float32)
        if active_mask is not None:
            out = out * active_mask[None, :]
        return out

    n = jnp.arange(offset, offset + B, dtype=jnp.float32)
    if K == 2:
        # The K=2 simplex is 1-D ({(w, 1-w) : w ∈ [0, 1]}). Sample w
        # from the canonical golden-ratio Kronecker sequence on [0, 1]
        # — uniform-discrepancy on [0, 1] directly translates to
        # uniform coverage of the 1-D simplex (no renormalisation
        # distortion). For K>2 we need the R_d generalisation below.
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        w0 = jnp.mod(n * phi, 1.0)
        prefs = jnp.stack([w0, 1.0 - w0], axis=-1)
        if active_mask is not None:
            prefs = prefs * active_mask[None, :]
            row_sum = jnp.sum(prefs, axis=-1, keepdims=True)
            prefs = jnp.where(row_sum > 1e-8, prefs / row_sum, prefs)
        return prefs

    # K >= 3: R_d sequence. Stride g_k = 1 / phi_K^(k+1) where phi_K is
    # the unique positive root of x^(K+1) = x + 1 (Roberts 2018).
    phi = _plastic_constant(K)
    g = jnp.asarray(
        [1.0 / (phi ** (k + 1)) for k in range(K)],
        dtype=jnp.float32,
    )  # (K,)
    raw = jnp.mod(n[:, None] * g[None, :], 1.0)  # (B, K)
    if active_mask is not None:
        raw = raw * active_mask[None, :]
    # Renormalise to the simplex. A row of all-zeros (possible with a
    # narrow active_mask) falls back to uniform-on-active.
    row_sum = jnp.sum(raw, axis=-1, keepdims=True)
    safe = jnp.where(row_sum > 1e-8, raw / row_sum, 1.0 / jnp.maximum(
        jnp.sum(active_mask) if active_mask is not None else float(K), 1.0,
    ))
    return safe
