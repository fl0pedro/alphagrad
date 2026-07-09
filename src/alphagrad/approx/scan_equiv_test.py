# -*- coding: utf-8 -*-
"""Verify the SCAN-based incremental_encoder.extend matches full re-encode
AND the append/branch API works (same acceptance as incremental_encoder_test)."""
import os
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("ALPHAGRAD_NN_HIDDEN", "256")
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")

from alphagrad.approx.common.examples import get_fn, get_args, infer_argnums, scalar_loss_fn
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent
from alphagrad.approx import incremental_encoder as ie

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
k = jax.random.PRNGKey(0); ak, _ = jax.random.split(k)
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
jaxpr = jax.make_jaxpr(LOSS)(*xs).jaxpr
NV = len(jaxpr.eqns)
agent = MicroPPOAgent(vocab_size=512, embd_dim=128, num_layers=4, num_heads=4,
                      hidden_dim=256, num_vertices=NV, value_dims=(128,128),
                      key=jax.random.PRNGKey(0), max_substeps=16, policy="palimpsa")

def full_encode(toks):
    ex, _ = agent.encode_tokens(jnp.asarray(toks, jnp.int32), key=jax.random.PRNGKey(0), eqn_ids=None)
    return ex

rng = np.random.default_rng(12345)
gmax = 0.0
# 1) full-sequence extend via scan
for i in range(20):
    S = int(rng.integers(5, 31))
    toks = rng.integers(1, NV+1, size=S).astype(np.int32)
    st = ie.init_state(agent)
    ie.extend(agent, st, [int(t) for t in toks])
    inc = ie.enc_x(st)
    full = full_encode(toks)
    d = float(jnp.max(jnp.abs(inc - full)))
    gmax = max(gmax, d)
    print(f"[seq {i:2d}] len={S:2d} max_abs_diff={d:.3e}", flush=True)
print(f"[ASSERT] full-extend max_abs_diff={gmax:.6e} -> {'PASS' if gmax<1e-4 else 'FAIL'}", flush=True)

# 2) prefix-once + branch (mirrors loop's copy()+delta pattern)
prefix = rng.integers(1, NV+1, size=12).astype(np.int32)
base = ie.init_state(agent)
ie.extend(agent, base, [int(t) for t in prefix])
bmax = 0.0
for j in range(3):
    nxt = int(rng.integers(1, NV+1))
    st = base.copy()
    ie.extend(agent, st, [nxt])
    inc_last = ie.enc_x(st)[-1]
    full = full_encode(np.concatenate([prefix, [nxt]]).astype(np.int32))
    d = float(jnp.max(jnp.abs(inc_last - full[-1])))
    bmax = max(bmax, d)
    print(f"[branch {j}] next={nxt:3d} max_abs_diff={d:.3e}", flush=True)
print(f"[ASSERT] branch max_abs_diff={bmax:.6e} -> {'PASS' if bmax<1e-4 else 'FAIL'}", flush=True)
print(f"[OVERALL] {'PASS' if (gmax<1e-4 and bmax<1e-4) else 'FAIL'}", flush=True)
