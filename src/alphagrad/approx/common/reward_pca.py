"""Online PCA-2 compressor for the 6 cost-reward channels.

Implements Pitch A's Stage 1 (per-channel z-score against EMA mean/var),
Stage 2 (EMA correlation PCA → top-2 eigenvectors → whitened latents),
and Stage 3 (slow refit + sign-pin + Procrustes alignment).

See ``docs/experiments/reward_pipeline_pca.md`` for the design and the
RQ8 success criteria.

This module is JAX-free at the state-update boundary — the EMA stats
are numpy arrays kept in plain Python state so they can be pickled into
checkpoints and resumed on restart. The projection ``P`` is consumed
JAX-side via ``project_jax`` which converts at call time.

Lifecycle:

    state = PCA2State.create(num_cost_channels=6, refit_every=100)
    for episode in episodes:
        # Update EMA stats from this episode's per-step cost vectors
        # (shape (T*N_envs, 6) of negative-cost values).
        state = state.observe_episode(cost_batch)
        # Periodic refit (also pins signs + Procrustes-aligns):
        if state.should_refit():
            state = state.refit()
        # Project per-step costs into the 2-latent space for the trainer.
        latents = state.project(cost_batch)  # → (T*N_envs, 2)

The refit is idempotent and cheap (6×6 eigendecomp) — running it every
~100 episodes adds negligible overhead while keeping the projection
responsive to slow drift in the cost distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


# Default EMA decay for mean/var/correlation stats. The cost vector
# updates ~192× per episode (12 vertices × 16 envs), so over 100
# episodes that's 19200 samples — beta=0.99 gives an effective window
# of ~100 samples, beta=0.999 gives ~1000. We want the projection to
# respond to slow distribution drift but not chase noise, so 0.99 is
# the conservative default.
_DEFAULT_BETA = 0.99


@dataclass
class PCA2State:
    """All state for the online PCA-2 compressor.

    Designed to round-trip through pickle (no JAX arrays) so a long
    training run can resume from a checkpoint without re-warming the
    EMA stats.
    """

    num_channels: int
    # Per-channel EMA estimates (shape (C,)):
    mu: np.ndarray
    var: np.ndarray
    # EMA correlation matrix on z-scored channels (shape (C, C)).
    # Updated alongside (mu, var) so the projection refit sees the
    # current distribution.
    corr: np.ndarray
    # Active projection matrix (shape (C, 2)). None until the first
    # refit; before that, ``project`` falls back to z-score-only
    # (returning the first 2 channels) to keep the trainer running.
    projection: Optional[np.ndarray]
    # Reference loadings used for sign-pinning across refits (shape (C, 2)).
    # First refit sets this; subsequent refits use it to flip eigenvector
    # signs.
    reference: Optional[np.ndarray]
    # Eigenvalues of the top-2 PCs at the most recent refit (shape (2,)).
    # Logged to wandb as the "variance explained" diagnostic.
    eigenvalues: Optional[np.ndarray]
    # Episode-counter bookkeeping.
    episodes_observed: int
    refit_every: int
    warmup_episodes: int
    last_refit_at: int
    # EMA decay used for both (mu, var) and corr.
    beta: float

    @classmethod
    def create(
        cls,
        *,
        num_cost_channels: int = 6,
        refit_every: int = 100,
        warmup_episodes: int = 50,
        beta: float = _DEFAULT_BETA,
    ) -> "PCA2State":
        return cls(
            num_channels=int(num_cost_channels),
            mu=np.zeros(num_cost_channels, dtype=np.float64),
            var=np.ones(num_cost_channels, dtype=np.float64),
            corr=np.eye(num_cost_channels, dtype=np.float64),
            projection=None,
            reference=None,
            eigenvalues=None,
            episodes_observed=0,
            refit_every=int(refit_every),
            warmup_episodes=int(warmup_episodes),
            last_refit_at=-1,
            beta=float(beta),
        )

    # ------------------------------------------------------------------
    # EMA updates
    # ------------------------------------------------------------------
    def observe_batch(self, cost_batch: np.ndarray) -> "PCA2State":
        """EMA-update mean / var / corr from a batch of cost vectors.

        ``cost_batch`` shape ``(B, C)`` — typically ``B = T * N_envs``
        per rollout, where T is the rollout length and N_envs is the
        env count. The C cost channels should be the *raw* (negative)
        cost values; this method handles z-scoring internally.

        Returns a new state (not in-place — matches the rest of the
        common/ package's immutable-state convention).
        """
        cost = np.asarray(cost_batch, dtype=np.float64)
        if cost.ndim == 1:
            cost = cost[None, :]
        assert cost.shape[1] == self.num_channels, (
            f"expected {self.num_channels} channels, got {cost.shape[1]}"
        )
        # The EMA absorbs the batch by averaging it first — this is the
        # standard "batched EMA" reduction. Equivalent to running the
        # single-sample EMA recurrence over the batch sequentially when
        # beta is small; cheaper and order-independent.
        batch_mean = cost.mean(axis=0)
        batch_centered = cost - batch_mean
        batch_var = (batch_centered ** 2).mean(axis=0)
        mu = (1.0 - self.beta) * self.mu + self.beta * batch_mean
        # Use the OLD mu when computing var so var is on the same
        # reference as the EMA mean (avoids transient drift on
        # batches where mean shifts a lot).
        var = (1.0 - self.beta) * self.var + self.beta * (
            batch_var + (batch_mean - self.mu) ** 2
        )

        # Z-score the batch against the freshly-updated mu/var and
        # accumulate the correlation matrix.
        std = np.sqrt(np.maximum(var, 1e-12))
        z = (cost - mu) / std  # (B, C)
        batch_corr = (z.T @ z) / max(z.shape[0], 1)
        corr = (1.0 - self.beta) * self.corr + self.beta * batch_corr
        return replace(self, mu=mu, var=var, corr=corr)

    def observe_episode(self, cost_batch: np.ndarray) -> "PCA2State":
        """Same as ``observe_batch`` but also increments the episode
        counter — call this once per rollout, not per micro-step."""
        new_state = self.observe_batch(cost_batch)
        return replace(new_state, episodes_observed=new_state.episodes_observed + 1)

    # ------------------------------------------------------------------
    # Refit
    # ------------------------------------------------------------------
    def should_refit(self) -> bool:
        """True when the warmup window has passed AND the configured
        refit interval has elapsed since the last refit (or no refit
        has run yet)."""
        if self.episodes_observed < self.warmup_episodes:
            return False
        if self.projection is None:
            return True  # first refit after warmup
        return (self.episodes_observed - self.last_refit_at) >= self.refit_every

    def refit(self) -> "PCA2State":
        """Re-eigendecompose the EMA correlation and update the
        projection. Pins sign + Procrustes-aligns to the previous
        projection so the latent reward stays continuous across refits.

        Returns a new state with ``projection``, ``eigenvalues``, and
        ``last_refit_at`` updated. ``reference`` is set on the first
        refit only.
        """
        # Symmetrise (numerical drift can break the eigendecomp).
        corr_sym = 0.5 * (self.corr + self.corr.T)
        eigvals, eigvecs = np.linalg.eigh(corr_sym)
        # ``eigh`` returns ascending eigenvalues; take the top 2.
        idx = np.argsort(eigvals)[::-1][:2]
        top_vals = eigvals[idx]
        top_vecs = eigvecs[:, idx]  # (C, 2)

        # Sign-pin: flip each eigenvector so its dot-product with the
        # reference loading is non-negative. First refit's eigenvectors
        # ARE the reference.
        if self.reference is None:
            reference = top_vecs.copy()
        else:
            reference = self.reference
            for k in range(2):
                if np.dot(top_vecs[:, k], reference[:, k]) < 0:
                    top_vecs[:, k] = -top_vecs[:, k]

        # Procrustes-align against the previous projection (if any) so
        # axis swaps don't introduce reward discontinuities. Solves the
        # orthogonal Procrustes problem
        # ``min_R ||top_vecs @ R - prev_proj||_F`` over orthonormal R.
        if self.projection is not None:
            M = top_vecs.T @ self.projection      # (2, 2)
            U, _S, Vt = np.linalg.svd(M)
            R = U @ Vt                            # (2, 2) orthonormal
            top_vecs = top_vecs @ R
            # Re-apply sign-pin after rotation in case Procrustes
            # introduced a flip relative to the reference.
            for k in range(2):
                if np.dot(top_vecs[:, k], reference[:, k]) < 0:
                    top_vecs[:, k] = -top_vecs[:, k]

        # Whiten by sqrt(eigenvalue) so both latents are unit-variance.
        whiten = np.diag(1.0 / np.sqrt(np.maximum(top_vals, 1e-12)))
        projection = top_vecs @ whiten  # (C, 2)

        return replace(
            self,
            projection=projection,
            reference=reference,
            eigenvalues=top_vals,
            last_refit_at=self.episodes_observed,
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(self, cost_batch: np.ndarray) -> np.ndarray:
        """Map raw cost vectors → 2-latent space.

        Z-scores against the current EMA stats, then applies the
        projection matrix. Returns shape ``(B, 2)``.

        Before the first refit, falls back to passing through the first
        two z-scored channels unchanged so the trainer can warm up
        without crashing. This is a benign fallback — the latents are
        unit-variance, just not yet decorrelated.
        """
        cost = np.asarray(cost_batch, dtype=np.float64)
        if cost.ndim == 1:
            cost = cost[None, :]
        std = np.sqrt(np.maximum(self.var, 1e-12))
        z = (cost - self.mu) / std  # (B, C)
        if self.projection is None:
            # Fallback: first two z-scored channels. Cheap, valid;
            # the trainer doesn't need to handle a None projection.
            return z[:, :2]
        return z @ self.projection

    def variance_explained(self) -> float:
        """Fraction of total correlation variance captured by the top
        2 PCs at the most recent refit. Logged to wandb as the
        ``reward/pca2/variance_explained`` diagnostic — should be ≥ 0.99
        if the 6 cost channels are as collinear as we expect."""
        if self.eigenvalues is None:
            return 0.0
        # Total variance of correlation matrix == trace == num_channels
        # (since z-scored channels are unit-variance).
        total = float(self.num_channels)
        return float(np.sum(self.eigenvalues) / max(total, 1e-12))

    def loadings(self) -> Optional[np.ndarray]:
        """Return the (C, 2) projection matrix, or None pre-first-refit.

        Logged to wandb per refit so the eigenvector grouping
        (arithmetic axis vs memory-traffic axis) is auditable.
        """
        return self.projection
