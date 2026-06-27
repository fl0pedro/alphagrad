"""Offline validation of the B_kstep-light proxy (_bkstep_score).

Run on a CPU compute node (srun cpu1). Checks:
  1. finite values
  2. DISCRIMINATIVE:
       - faithful approx (== exact grad) -> drop_a ≈ drop_e -> score ≈ 1
       - garbage approx (zero grad)      -> drop_a ≈ 0      -> score ≈ 0
       - anti grad (ascends loss)        -> drop_a < 0       -> score = 0 (clipped)
       - half-magnitude grad             -> partial descent  -> 0 < score < 1
"""
import os
import jax
import jax.numpy as jnp
import jax.random as jrand

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from alphagrad.approx.common.examples import _neural_network, scalar_loss_fn

from alphagrad.approx import env as ENV

# Build the env's small MLP proxy model in grad mode: scalar MSE loss.
loss = scalar_loss_fn(_neural_network)
argnums = (2, 3, 4, 5)  # W1, b1, W2, b2

key = jrand.PRNGKey(0)
H = 16
N = 8   # batch
Dx = 12
Dy = 4
k = jrand.split(key, 6)
x = jrand.normal(k[0], (N, Dx))
y = jrand.normal(k[1], (N, Dy)) * 0.1
W1 = jrand.normal(k[2], (H, Dx)) * 0.3
b1 = jnp.zeros(H)
W2 = jrand.normal(k[3], (Dy, H)) * 0.3
b2 = jnp.zeros(Dy)
args0 = [x, y, W1, b1, W2, b2]

# EXACT compiled fn: (value, grads) via jax.value_and_grad.
exact_fn = jax.jit(jax.value_and_grad(loss, argnums=argnums))

# Faithful approx == exact.
faithful_fn = exact_fn

# Garbage approx: returns true loss value but ZERO gradients (no descent).
def _garbage(*a):
    v, g = jax.value_and_grad(loss, argnums=argnums)(*a)
    g0 = tuple(jnp.zeros_like(gi) for gi in g)
    return v, g0
garbage_fn = jax.jit(_garbage)

# Anti approx: gradient pointing the WRONG way (ascends loss).
def _anti(*a):
    v, g = jax.value_and_grad(loss, argnums=argnums)(*a)
    return v, tuple(-2.0 * gi for gi in g)
anti_fn = jax.jit(_anti)

# Half approx: half-magnitude grad (still descends, but less than exact).
def _half(*a):
    v, g = jax.value_and_grad(loss, argnums=argnums)(*a)
    return v, tuple(0.5 * gi for gi in g)
half_fn = jax.jit(_half)

seeds = [args0]  # single-seed is enough to validate discriminativeness
# add a 2nd seed (S=2) with different data
k2 = jrand.split(k[4], 2)
x2 = jrand.normal(k2[0], (N, Dx))
y2 = jrand.normal(k2[1], (N, Dy)) * 0.1
seeds.append([x2, y2, W1, b1, W2, b2])

def run(name, approx_fn):
    s = ENV._bkstep_score(approx_fn, exact_fn, seeds, argnums, K=3, lr=0.1)
    print(f"  {name:10s} score={s:.4f}  finite={s==s}")
    return s

print("[validate] B_kstep-light proxy")
sf = run("faithful", faithful_fn)
sg = run("garbage", garbage_fn)
sa = run("anti", anti_fn)
sh = run("half", half_fn)

ok = True
if not (0.95 <= sf <= 1.0):
    print(f"FAIL: faithful score {sf} not ~1"); ok = False
if not (sg <= 0.05):
    print(f"FAIL: garbage score {sg} not ~0"); ok = False
if not (sa <= 0.05):
    print(f"FAIL: anti score {sa} not ~0"); ok = False
if not (0.05 < sh < 0.95):
    print(f"FAIL: half score {sh} not strictly between"); ok = False

print("VALIDATION", "PASS" if ok else "FAIL")
