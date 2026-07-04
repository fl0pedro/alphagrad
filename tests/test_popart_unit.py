"""PopArt unit tests (plain python, srun-runnable — no GPU needed).

1. OUTPUT PRESERVATION (the Art): after every stats update + final-layer
   rescale, the DE-normalised predictions are unchanged to float32
   precision — checked over 60 repeated updates with target scales
   sweeping 1e-2 .. 1e4 per channel.
2. SIGMA FLOOR: constant (homogeneous) targets drive sigma to exactly
   sigma_min — the normaliser can never amplify noise.
3. GRADIENTS: value-loss gradients through the (heavily) rescaled head
   stay finite.

Run: cd ~/dsnn && uv run --no-sync python alphagrad/tests/test_popart_unit.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx

from alphagrad.transformer import MLP
from alphagrad.approx.common.popart import PopArtStats, popart_rescale_mlp_head

K = 10
EMBD = 32
mlp = MLP(EMBD, K, (64, 64), key=jrand.PRNGKey(0))
x = jrand.normal(jrand.PRNGKey(1), (128, EMBD))
stats = PopArtStats(K, beta=1e-2, sigma_min=0.1)
rng = np.random.default_rng(0)


def denorm_pred(m, mu, sigma):
    v_hat = np.asarray(jax.vmap(m)(x))            # (B, K) normalised
    return sigma[None, :] * v_hat + mu[None, :]   # raw


# ---- 1) output preservation under repeated stats updates -------------
# Preservation is EXACT in exact arithmetic; in float32 the residual is
# rounding at the channel scale (~eps32 * sigma). The honest metric is
# therefore |after - before| / sigma' (== the error of the NORMALISED
# prediction v_hat', which is what the critic actually trains on).
max_abs, max_scaled = 0.0, 0.0
for step in range(60):
    scale = 10.0 ** rng.uniform(-2, 4, size=K)
    targets = rng.normal(loc=scale, scale=scale, size=(256, K))
    before = denorm_pred(mlp, stats.mu, stats.sigma)
    o_mu, o_sig, n_mu, n_sig = stats.update(targets)
    mlp = popart_rescale_mlp_head(mlp, o_mu, o_sig, n_mu, n_sig)
    after = denorm_pred(mlp, stats.mu, stats.sigma)
    aerr = float(np.max(np.abs(after - before)))
    serr = float(np.max(np.abs(after - before) / n_sig[None, :]))
    max_abs, max_scaled = max(max_abs, aerr), max(max_scaled, serr)
print(f"[1] output preservation over 60 updates (target scales 1e-2..1e4): "
      f"max_abs_err={max_abs:.3e} "
      f"max_err/sigma={max_scaled:.3e} (float32 eps ~1.2e-7)")
assert max_scaled < 1e-5, f"preservation broken: err/sigma {max_scaled}"

# ---- 2) sigma floor ---------------------------------------------------
s2 = PopArtStats(K, beta=1e-2, sigma_min=0.1)
for _ in range(10):
    s2.update(np.full((64, K), 3.14))
print(f"[2] sigma floor: constant targets -> sigma={s2.sigma[0]} "
      f"(min={s2.sigma_min}), mu[0]={s2.mu[0]:.4f}")
assert np.all(s2.sigma == np.float32(0.1)), s2.sigma
assert np.allclose(s2.mu, 3.14, atol=1e-5)

# ---- 3) finite gradients through the rescaled head --------------------
tgt = jnp.asarray((targets[:128] - stats.mu) / stats.sigma, dtype=jnp.float32)


def loss(m):
    v_hat = jax.vmap(m)(x)
    return jnp.mean((v_hat - tgt) ** 2)


g = eqx.filter_grad(loss)(mlp)
leaves = jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array))
n_bad = sum(int((~jnp.isfinite(l)).sum()) for l in leaves)
gmax = max(float(jnp.abs(l).max()) for l in leaves)
print(f"[3] gradients: {len(leaves)} leaves, non-finite={n_bad}, "
      f"absmax={gmax:.3e}")
assert n_bad == 0

print("ALL POPART UNIT TESTS PASSED")
