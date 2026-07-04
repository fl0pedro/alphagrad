"""PopArt value-target normalisation.

van Hasselt et al. 2016, "Learning values across many orders of
magnitude" (arXiv:1602.07714); multi-channel exactly as in IMPALA's
multi-task PopArt (Hessel et al. 2018). Two pieces:

* :class:`PopArtStats` — per-channel running (mu, sigma) over the VALUE
  TARGETS (the per-channel GAE returns), maintained as a debiased EMA at
  a slow, quasi-static rate (beta ~ 1e-2 per update). sigma is FLOORED
  (``sigma = max(sigma, sigma_min)``) so the normaliser can NEVER
  amplify noise: a homogeneous batch (returns identical everywhere)
  drives sigma to the floor, not to 0 — unlike a per-rollout z-score,
  which divides by the shrinking batch std and blows small differences
  up to +-1.
* :func:`popart_rescale_mlp_head` — the "Art" (Adaptively Rescaling
  Targets, preserving outputs): when the stats step mu->mu',
  sigma->sigma', the FINAL linear layer of the value head is rewritten

      W <- W * (sigma / sigma'),   b <- (sigma * b + mu - mu') / sigma'

  so the DE-normalised predictions ``sigma' * v_hat' + mu'`` are
  numerically unchanged — the critic never sees a target-scale jump as
  a gradient.

The consumer (ppo_ray_worker) keeps the critic learning NORMALISED
values ``v_hat``; raw values (GAE bootstrap) are recovered affinely via
``v = sigma * v_hat + mu``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np


class PopArtStats:
    """Per-channel debiased EMA of target mean / std.

    ``update`` consumes a batch of raw per-channel value targets and
    advances the stats by ONE EMA step (``beta`` is per-update,
    independent of batch size — quasi-static by design). Debiasing (the
    ``_w`` accumulator, the Adam pattern; equivalently van Hasselt's
    stream update with step size beta_t = beta / w_t) makes the FIRST
    update adopt the batch stats exactly instead of crawling away from
    the arbitrary (0, 1) init over ~1/beta updates.
    """

    def __init__(
        self,
        num_channels: int,
        beta: float = 1e-2,
        sigma_min=0.1,
        sigma_max: float = 1e6,
    ) -> None:
        self.num_channels = int(num_channels)
        self.beta = float(beta)
        # Fix 4: sigma_min may be a scalar OR a per-channel array. A
        # near-homogeneous channel (bkstep/cosine at convergence) drives
        # var->0 and clamps to the floor; too small a floor (0.1) lets the
        # A_k/sigma_k advantage over-amplify when the batch goes uniform.
        # Per-channel floors let bkstep(9)/cosine(6) sit higher (~0.2)
        # without over-flattening the wide-dynamic-range cost channels.
        _sm = np.asarray(sigma_min, dtype=np.float64)
        if _sm.ndim == 0:
            _sm = np.full(self.num_channels, float(_sm), dtype=np.float64)
        if _sm.shape != (self.num_channels,):
            raise ValueError(
                f"sigma_min must be scalar or ({self.num_channels},), got "
                f"{_sm.shape}",
            )
        self.sigma_min = _sm
        self.sigma_max = float(sigma_max)
        # Debiased EMA accumulators (float64 — the raw cost channels span
        # ~9 decades and the second moment squares that).
        self._mu_acc = np.zeros(self.num_channels, dtype=np.float64)
        self._nu_acc = np.zeros(self.num_channels, dtype=np.float64)
        self._w = 0.0
        self.n_updates = 0
        # Public stats. Init (0, 1): normalisation is an exact no-op
        # until the first update.
        self.mu = np.zeros(self.num_channels, dtype=np.float32)
        self.sigma = np.ones(self.num_channels, dtype=np.float32)

    def update(self, targets: np.ndarray):
        """One EMA step from ``targets`` of shape (M, K).

        Returns ``(old_mu, old_sigma, new_mu, new_sigma)`` — feed these
        straight into :func:`popart_rescale_mlp_head`.
        """
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != self.num_channels:
            raise ValueError(
                f"targets must be (M, {self.num_channels}), got "
                f"{targets.shape}",
            )
        if not np.isfinite(targets).all():
            raise ValueError("PopArtStats.update: non-finite targets")
        old_mu = self.mu.copy()
        old_sigma = self.sigma.copy()
        b = self.beta
        self._mu_acc = (1.0 - b) * self._mu_acc + b * targets.mean(axis=0)
        self._nu_acc = (1.0 - b) * self._nu_acc + b * (targets ** 2).mean(axis=0)
        self._w = (1.0 - b) * self._w + b
        mu = self._mu_acc / self._w
        var = np.maximum(self._nu_acc / self._w - mu ** 2, 0.0)
        sigma = np.clip(np.sqrt(var), self.sigma_min, self.sigma_max)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        self.n_updates += 1
        return old_mu, old_sigma, self.mu.copy(), self.sigma.copy()


def _final_linear_index(seq_layers) -> int:
    """Index of the LAST eqx.nn.Linear in an eqx.nn.Sequential's layers
    (alphagrad.transformer.MLP ends [..., Linear, Lambda(final_act)])."""
    idx = -1
    for i, lyr in enumerate(seq_layers):
        if isinstance(lyr, eqx.nn.Linear):
            idx = i
    if idx < 0:
        raise ValueError("value head has no eqx.nn.Linear layer to rescale")
    return idx


def popart_rescale_mlp_head(mlp, old_mu, old_sigma, new_mu, new_sigma):
    """Output-preserving final-layer rescale of an alphagrad MLP head.

    ``W <- W * (sigma/sigma')`` (per output row) and
    ``b <- (sigma * b + mu - mu') / sigma'`` so that
    ``sigma' * head'(x) + mu' == sigma * head(x) + mu`` for every x, to
    float precision. Returns the rescaled MLP (functional, eqx.tree_at).
    """
    seq = mlp.layers.layers  # MLP.layers is an eqx.nn.Sequential
    li = _final_linear_index(seq)
    lin = seq[li]
    dt = lin.weight.dtype
    old_mu = jnp.asarray(old_mu, dtype=dt)
    old_sigma = jnp.asarray(old_sigma, dtype=dt)
    new_mu = jnp.asarray(new_mu, dtype=dt)
    new_sigma = jnp.asarray(new_sigma, dtype=dt)
    ratio = old_sigma / new_sigma                          # (K,)
    new_w = lin.weight * ratio[:, None]                    # rows = channels
    new_b = (old_sigma * lin.bias + old_mu - new_mu) / new_sigma
    return eqx.tree_at(
        lambda m: (m.layers.layers[li].weight, m.layers.layers[li].bias),
        mlp,
        (new_w, new_b),
    )
