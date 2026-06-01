"""Multi-objective scalarization functions (RQ9 / Pitch B scaffold).

Currently only the linear-sum scalarization is wired into trainers
(via ``--lambda-cmp/mem/frob`` × per-channel reward). Augmented
Tchebycheff is here as a self-contained function so RQ9 can flip
``--scalarization tchebycheff`` and the trainer-side hookup is one
``apply_scalarization`` call.

Design + reachability argument: see
``docs/experiments/pareto_front_tchebycheff.md``.
"""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp


ScalarizationKind = Literal["linear", "tchebycheff"]


def apply_scalarization(
    per_channel_rewards: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    kind: ScalarizationKind = "linear",
    ideal_point: jnp.ndarray | None = None,
    rho: float = 0.05,
) -> jnp.ndarray:
    """Reduce a per-channel reward to a scalar via ``kind``.

    Args:
        per_channel_rewards: ``(..., K)`` per-channel reward magnitudes.
            Convention: higher = better (cost channels should already be
            negated, fidelity channels are positive).
        weights: ``(..., K)`` simplex weights (``sum_k w_k = 1``,
            ``w_k >= 0``). Broadcast-compatible with
            ``per_channel_rewards``.
        kind: ``"linear"`` (default) is ``sum_k w_k * r_k`` — the
            current behaviour. ``"tchebycheff"`` is the augmented
            Tchebycheff scalarization
            ``-(max_k w_k * |r_k - z_k^*|) - rho * sum_k |r_k - z_k^*|``
            with ``z_k^*`` the (positive-orientation) ideal point per
            channel.
        ideal_point: ``(K,)`` per-channel ideal (best-ever) values; only
            used by ``tchebycheff``. When None, falls back to the
            per-batch max along the leading axis — but the caller is
            expected to maintain a stable EMA estimate of this and pass
            it in for reachability theorems to apply (see Miettinen 1999
            §3.4 — Tchebycheff reachability needs a stable ideal point).
        rho: augmentation weight on the ``sum_k |.|`` term. Default 0.05
            following Steuer 1986; small values rule out weakly Pareto-
            optimal points without distorting the front shape.

    Returns:
        ``(...,)`` scalar reward (same leading shape as inputs minus
        the last axis).
    """
    if kind == "linear":
        return jnp.sum(per_channel_rewards * weights, axis=-1)
    if kind == "tchebycheff":
        if ideal_point is None:
            ideal_point = jnp.max(per_channel_rewards, axis=tuple(
                range(per_channel_rewards.ndim - 1),
            ))
        gap = jnp.abs(per_channel_rewards - ideal_point)
        # Tchebycheff term: maximize the smallest weighted distance to
        # the ideal point. We negate so that "higher reward = better"
        # matches the linear case's convention.
        tcheb = -jnp.max(weights * gap, axis=-1)
        # Augmentation term: rule out weakly Pareto-optimal points.
        aug = -float(rho) * jnp.sum(gap, axis=-1)
        return tcheb + aug
    raise ValueError(f"unknown scalarization kind={kind!r}")


def update_ideal_point_ema(
    z_star: jnp.ndarray,
    fresh_rewards: jnp.ndarray,
    *,
    beta: float = 0.99,
) -> jnp.ndarray:
    """EMA-update the ideal point ``z_star`` from a batch of fresh
    per-channel rewards.

    ``fresh_rewards`` shape ``(B, K)``. The ideal point per channel is
    the *running best* (max) value seen so far — but the strict
    running-max would lock in early outliers. An EMA over the per-batch
    max provides a slowly-tracking ideal that still satisfies the
    Tchebycheff reachability theorem in the limit (the EMA converges
    to a stable point under repeated sampling).

    Returns ``(K,)`` updated ideal.
    """
    batch_max = jnp.max(fresh_rewards, axis=0)
    return (1.0 - beta) * z_star + beta * batch_max
