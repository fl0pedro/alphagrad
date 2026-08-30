"""CPU smoke: the Pallas kernel under interpret=True must match palimpsa_ref,
forward AND gradient. This is what licenses demoting palimpsa_ref to a test
oracle -- if these disagree, the single-recurrence change is not safe."""
import os
import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.transformer.palimpsa_pallas import (
    palimpsa, palimpsa_ref, palimpsa_attention, _resolve_backend)

print("resolved:", _resolve_backend())

B, T, H, DK, DV = 2, 24, 2, 16, 16
k0 = jrand.PRNGKey(0)
ks = jrand.split(k0, 6)
q = jrand.normal(ks[0], (B, T, H, DK), jnp.float32)
k = jrand.normal(ks[1], (B, T, H, DK), jnp.float32)
v = jrand.normal(ks[2], (B, T, H, DV), jnp.float32)
b = jrand.normal(ks[3], (B, T, H, DV), jnp.float32) ** 2 + 0.1
gt = jax.nn.sigmoid(jrand.normal(ks[4], (B, T, H), jnp.float32))
g = jnp.abs(jrand.normal(ks[5], (H,), jnp.float32)) + 0.1
Ip = jnp.abs(jrand.normal(ks[5], (H,), jnp.float32)) + 0.5

o_ref = palimpsa_ref(q, k, v, b, gt, g, Ip, None)
o_int = palimpsa_attention(q, k, v, b, gt, g, Ip, None, 16)
d = float(jnp.max(jnp.abs(o_ref - o_int)))
r = float(jnp.max(jnp.abs(o_ref - o_int)) / (jnp.max(jnp.abs(o_ref)) + 1e-12))
print("FWD  max_abs=%.3e  max_rel=%.3e" % (d, r))


def loss_ref(q_):
    return jnp.sum(palimpsa_ref(q_, k, v, b, gt, g, Ip, None) ** 2)


def loss_int(q_):
    return jnp.sum(palimpsa_attention(q_, k, v, b, gt, g, Ip, None, 16) ** 2)


gr = jax.grad(loss_ref)(q)
gi = jax.grad(loss_int)(q)
gd = float(jnp.max(jnp.abs(gr - gi)))
grel = float(jnp.max(jnp.abs(gr - gi)) / (jnp.max(jnp.abs(gr)) + 1e-12))
print("GRAD max_abs=%.3e  max_rel=%.3e" % (gd, grel))
print("dispatcher path == interpret kernel:",
      bool(jnp.allclose(palimpsa(q, k, v, b, gt, g, Ip), o_int)))
print("VERDICT:", "PASS" if (r < 1e-4 and grel < 1e-4) else "FAIL")
