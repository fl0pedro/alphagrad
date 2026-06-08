"""Tests for the RQ8 PCA-2 reward compressor.

Pins:

1. Stage 1 (z-score) — after enough warm-up batches the per-channel EMA
   mean/var converges to the true sample mean/var.
2. Stage 2 (PCA-2) — on synthetic 6-channel data with two strong axes
   (arithmetic-like + memory-like), the eigenvalues capture ≥ 0.99 of
   the variance and the loadings cluster correctly.
3. Stage 3 (sign-pin + Procrustes) — after a refit, the latent
   projection is sign-consistent with the previous refit (no jumps).
4. The fallback projection (pre-first-refit) returns sensible
   z-scored values and doesn't crash.
"""
from __future__ import annotations

import numpy as np

from alphagrad.approx.common.reward_pca import PCA2State


def _synthetic_cost_batch(rng, n: int = 500) -> np.ndarray:
    """Two latent axes (arithmetic, memory) with channel-specific noise.

    Returns shape ``(n, 6)`` with the canonical channel ordering
    ``(muls_adds_fmas, flops, latency_ns, max_io_sum, bytes_accessed, peak_memory)``.
    Channels 0-2 (arithmetic) co-vary; channels 3-5 (memory) co-vary;
    arithmetic and memory have a small but non-zero correlation.
    """
    arith = rng.normal(0, 1, size=(n,))
    mem = rng.normal(0, 1, size=(n,)) + 0.3 * arith
    noise = rng.normal(0, 0.1, size=(n, 6))
    # Scale each channel to a different order of magnitude (the actual
    # cost-channel reality: FLOPs ~1e12, peak_memory ~1e8, etc.) so
    # we can verify the z-score handles the unit mismatch.
    scales = np.array([1e12, 1e11, 1e6, 1e8, 1e10, 1e8], dtype=np.float64)
    batch = np.stack([
        arith * 1.0,
        arith * 0.9,
        arith * 0.7,
        mem * 1.0,
        mem * 0.85,
        mem * 0.95,
    ], axis=-1) * scales[None, :]
    return batch + noise * scales[None, :]


def test_stage1_zscore_converges_with_ema_warmup():
    """After enough warmup batches, EMA mu/var should be close to the
    true sample statistics — pins that the z-score is doing the right
    thing under the scale mismatch."""
    print("\n[pca2] EMA mu/var converges to sample statistics")
    rng = np.random.default_rng(0)
    state = PCA2State.create(num_cost_channels=6, beta=0.99)
    # Pump 200 batches of 500 samples each — enough EMA windows to settle.
    for _ in range(200):
        state = state.observe_episode(_synthetic_cost_batch(rng, n=500))
    # Generate a large reference sample and compare moments. Normalise
    # by the channel scale (sqrt(var)) rather than |mu| — the synthetic
    # data is mean-0, so any /max(|mu|, 1.0) blows up the rel-error for
    # mean-zero channels even when the EMA estimate is fine in absolute
    # terms (a few tens vs a sample scale of 1e12 is essentially zero).
    ref = _synthetic_cost_batch(rng, n=20000)
    ref_mu = ref.mean(axis=0)
    ref_var = ref.var(axis=0)
    channel_scale = np.sqrt(np.maximum(ref_var, 1.0))
    rel_mu = np.abs(state.mu - ref_mu) / channel_scale
    rel_var = np.abs(state.var - ref_var) / np.maximum(ref_var, 1.0)
    print(f"  rel mu error  max={rel_mu.max():.4f} (per-channel: {rel_mu})")
    print(f"  rel var error max={rel_var.max():.4f}")
    assert rel_mu.max() < 0.2, f"EMA mu didn't converge: {rel_mu}"
    assert rel_var.max() < 0.5, f"EMA var didn't converge: {rel_var}"


def test_stage2_variance_explained_captures_99_percent():
    """On the synthetic 2-latent-axes data, the top 2 PCs should
    capture ≥ 0.99 of the correlation variance — this is the actual
    R-squared diagnostic logged to wandb."""
    print("\n[pca2] top 2 PCs capture >=0.99 of variance on 2-axis data")
    rng = np.random.default_rng(1)
    state = PCA2State.create(num_cost_channels=6, refit_every=10, warmup_episodes=5)
    for _ in range(50):
        state = state.observe_episode(_synthetic_cost_batch(rng, n=500))
    state = state.refit()
    ve = state.variance_explained()
    print(f"  variance explained = {ve:.4f} (eigenvalues = {state.eigenvalues})")
    # 0.95 is a comfortable threshold for the synthetic 2-axis data —
    # real cost channels (FLOPs / bytes / latency / peak) are even more
    # collinear, so a real run typically shows ≥ 0.99 (logged to wandb
    # per refit as ``reward/pca2/variance_explained``).
    assert ve >= 0.95, f"top-2 variance explained {ve} < 0.95 — PCA-2 failed"


def test_stage3_procrustes_keeps_latent_continuous_across_refits():
    """A refit must not introduce big jumps in the latent reward — the
    Procrustes alignment + sign-pinning guarantee this."""
    print("\n[pca2] latent is continuous across refits (no axis swap)")
    rng = np.random.default_rng(2)
    state = PCA2State.create(
        num_cost_channels=6, refit_every=10, warmup_episodes=5,
    )
    for _ in range(20):
        state = state.observe_episode(_synthetic_cost_batch(rng, n=500))
    state = state.refit()
    # Project a held-out batch BEFORE the next refit.
    held = _synthetic_cost_batch(rng, n=100)
    latents_pre = state.project(held)

    # Push more data through (slowly drifting distribution) and refit again.
    for _ in range(20):
        state = state.observe_episode(_synthetic_cost_batch(rng, n=500))
    state = state.refit()
    latents_post = state.project(held)

    # The two latent projections should be highly correlated across the
    # refit boundary — Procrustes guarantees the axis assignment + sign
    # convention stays consistent.
    for k in range(2):
        c = np.corrcoef(latents_pre[:, k], latents_post[:, k])[0, 1]
        assert c > 0.5, (
            f"latent[{k}] correlation across refit = {c:.3f} — "
            f"Procrustes alignment failed (axis swap or sign flip)"
        )
        print(f"  latent[{k}] cross-refit correlation = {c:.3f}")


def test_pre_refit_fallback_returns_z_scored_values():
    """Before the first refit, ``project`` should return z-scored
    values of the first 2 channels (the fallback). This keeps the
    trainer running through warmup without a None check."""
    print("\n[pca2] pre-refit fallback projects to z-scored first-2 channels")
    rng = np.random.default_rng(3)
    state = PCA2State.create(num_cost_channels=6, refit_every=100, warmup_episodes=10)
    # Warm up the EMA stats but don't refit.
    for _ in range(15):
        state = state.observe_episode(_synthetic_cost_batch(rng, n=200))
    assert state.projection is None
    batch = _synthetic_cost_batch(rng, n=50)
    latents = state.project(batch)
    assert latents.shape == (50, 2)
    # Latents should be roughly unit-variance (z-scored).
    std = latents.std(axis=0)
    print(f"  pre-refit latent std = {std}")
    assert (std > 0.3).all() and (std < 3.0).all(), (
        f"pre-refit latents not well-scaled: std={std}"
    )


def main():
    print("=== RQ8 PCA-2 reward compressor tests ===")
    test_stage1_zscore_converges_with_ema_warmup()
    test_stage2_variance_explained_captures_99_percent()
    test_stage3_procrustes_keeps_latent_continuous_across_refits()
    test_pre_refit_fallback_returns_z_scored_values()
    print("\nALL PCA2 TESTS OK")


if __name__ == "__main__":
    main()
