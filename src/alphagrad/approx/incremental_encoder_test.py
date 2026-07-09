# -*- coding: utf-8 -*-
"""
INCREMENTAL (single-token, O(1)-in-seq-length) causal palimpsa encoder.

Proves that processing tokens ONE AT A TIME while carrying per-layer palimpsa
(M, I) state reproduces the FULL re-encode (agent.encode_tokens) bit-for-bit
(within 1e-4). eqn_ids=None throughout (relational gate OFF => exactly causal).

Run on a free Blackwell GPU:
    cd ~/dsnn && export PATH=$HOME/.local/bin:$PATH \
      && uv run --no-sync python \
         src/alphagrad/approx/incremental_encoder_test.py
"""
import os
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("ALPHAGRAD_NN_HIDDEN", "256")

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn

# Blackwell GPUs default to TF32/bf16 tensor-core accumulation for float32
# matmuls. That tiling differs between a BATCHED GEMM (the full re-encode's
# `jax.vmap(proj)(S,E)`) and an UNBATCHED matvec (the incremental per-token
# `proj(E,)`), producing a ~6e-4 numeric gap that is PURE GEMM-tiling noise,
# NOT an algorithmic difference (verified: identical to <2e-15 under float64).
# Forcing 'highest' precision makes both paths use true-float32 accumulation
# so the comparison isolates the recurrence algorithm itself.
jax.config.update("jax_default_matmul_precision", "highest")

from alphagrad.approx.common.examples import get_fn, get_args, infer_argnums, scalar_loss_fn
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent

# ---------------------------------------------------------------------------
# 1. Build the env jaxpr (like gumbel_planner.py) and the MicroPPOAgent.
# ---------------------------------------------------------------------------
LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")

k = jax.random.PRNGKey(0)
ak, _ = jax.random.split(k)
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
closed = jax.make_jaxpr(LOSS)(*xs)
jaxpr = closed.jaxpr
NUM_VERTICES = len(jaxpr.eqns)
print(f"[setup] num_vertices (len jaxpr.eqns) = {NUM_VERTICES}", flush=True)
print(f"[setup] device = {jax.devices()[0]}", flush=True)

agent = MicroPPOAgent(
    vocab_size=512,
    embd_dim=128,
    num_layers=4,
    num_heads=4,
    hidden_dim=256,
    num_vertices=NUM_VERTICES,
    value_dims=(128, 128),
    key=jax.random.PRNGKey(0),
    max_substeps=16,
    policy="palimpsa",
)

ENC = agent.encoder
NUM_LAYERS = ENC.num_layers
SCALE = None  # palimpsa uses head_dim**-0.5


# ---------------------------------------------------------------------------
# 2. Incremental per-token palimpsa recurrence + full per-token block.
# ---------------------------------------------------------------------------
def _palimpsa_step(mixer, y_t, carry):
    """One-token palimpsa recurrence, MASK ALL-ONES (fully-valid prefix).

    Reproduces palimpsa_ref exactly for a single token given carry (M, I):
        decay_t = exp(-gt_t * g)
        M_t = v_t (x) k_t   + decay_t*M_{t-1}
        I_t = b_t (x) k_t^2 + (1-decay_t)*Ip + decay_t*I_{t-1}
        mu_t = M_t / I_t
        out_t = einsum('hdn,hn->hd', mu_t, q_t*scale)
    y_t: (E,) already attn-normed token. carry (M,I): each (H, DV, DK).
    Returns (attn_out (E,), new_carry).
    """
    H = mixer.num_heads
    d = mixer.head_dim
    scale = d ** -0.5

    q = mixer.query_proj(y_t).reshape(H, d)
    kk = mixer.key_proj(y_t).reshape(H, d)
    v = mixer.value_proj(y_t).reshape(H, d)
    b = jnn.softplus(mixer.bias_proj(y_t)).reshape(H, d)
    gt = jnn.softplus(mixer.gate_proj(y_t))          # (H,)
    g = jnn.softplus(mixer.g_raw)                     # (H,)
    Ip = jnn.softplus(mixer.Ip_raw)                   # (H,)

    M_prev, I_prev = carry                            # (H, DV, DK) each
    decay = jnp.exp(-gt[:, None, None] * g[:, None, None])   # (H,1,1)
    outer_M = v[:, :, None] * kk[:, None, :]          # (H, DV, DK)
    outer_I = b[:, :, None] * (kk[:, None, :] ** 2)
    M = outer_M + decay * M_prev
    I = outer_I + (1.0 - decay) * Ip[:, None, None] + decay * I_prev
    mu = M / I
    out_t = jnp.einsum('hdn,hn->hd', mu, q * scale)   # (H, d)
    out = out_t.reshape(H * d)
    attn_out = mixer.output_proj(out)                 # (E,)
    return attn_out, (M, I)


def _layer_step(layer, x_t, carry):
    """One EncoderLayer applied to one token x_t (E,) with palimpsa carry.

    y = attn(attn_norm(x)); x = x + y; y = mlp(mlp_norm(x)); x = x + y.
    attn_norm / mlp_norm are per-token LayerNorm; mlp is per-token SwiGLU.
    Only the palimpsa attn mixes across tokens (via carry).
    """
    mixer = layer.attn_layer
    y = layer.attn_norm(x_t)
    attn_out, new_carry = _palimpsa_step(mixer, y, carry)
    x_t = x_t + attn_out
    y = layer.mlp_norm(x_t)
    y = layer.mlp(y)
    x_t = x_t + y
    return x_t, new_carry


def _init_carry(mixer):
    """M0 = 0, I0 = broadcast(Ip)  (matches palimpsa_ref)."""
    H = mixer.num_heads
    d = mixer.head_dim
    Ip = jnn.softplus(mixer.Ip_raw)                   # (H,)
    M0 = jnp.zeros((H, d, d), jnp.float32)
    I0 = jnp.broadcast_to(Ip[:, None, None], (H, d, d)).astype(jnp.float32)
    return (M0, I0)


def _embed_token(agent, tok, pos):
    """Embedding + additive positional encoding for a single token at pos."""
    x = agent.embedding(jnp.asarray(tok, dtype=jnp.int32))   # (E,)
    x = x + agent.pos_enc.pe[pos, :]
    return x


def incremental_encode(agent, token_ids):
    """Process tokens ONE AT A TIME carrying per-layer palimpsa (M,I) state.

    Returns (S, embd) == final_norm(encoder(embed+pe)) computed causally.
    eqn_ids=None throughout (relational gate OFF).
    Fully-valid prefix assumed (no pad/zeros) => mask all-ones.
    """
    enc = agent.encoder
    layers = enc.layers
    # per-layer palimpsa carry
    carries = [_init_carry(layer.attn_layer) for layer in layers]

    outs = []
    for pos, tok in enumerate(token_ids):
        x_t = _embed_token(agent, tok, pos)           # (E,)
        for li, layer in enumerate(layers):
            x_t, carries[li] = _layer_step(layer, x_t, carries[li])
        enc_t = agent.final_norm(x_t)                 # (E,)
        outs.append(enc_t)
    return jnp.stack(outs, axis=0)                     # (S, E)


# jit the single-layer step and embed for speed / precision consistency.
_layer_step_j = jax.jit(_layer_step)
_embed_j = jax.jit(_embed_token, static_argnums=())
_finalnorm_j = jax.jit(agent.final_norm)


def incremental_encode_fast(agent, token_ids):
    layers = agent.encoder.layers
    carries = [_init_carry(layer.attn_layer) for layer in layers]
    outs = []
    for pos, tok in enumerate(token_ids):
        x_t = agent.embedding(jnp.asarray(tok, dtype=jnp.int32)) + agent.pos_enc.pe[pos, :]
        for li, layer in enumerate(layers):
            x_t, carries[li] = _layer_step_j(layer, x_t, carries[li])
        outs.append(agent.final_norm(x_t))
    return jnp.stack(outs, axis=0)


# ---------------------------------------------------------------------------
# 3. ACCEPTANCE TEST: 20 random sequences, compare vs full re-encode.
# ---------------------------------------------------------------------------
def full_encode(agent, tokens):
    enc_x, _ = agent.encode_tokens(
        jnp.asarray(tokens, dtype=jnp.int32),
        key=jax.random.PRNGKey(0),
        eqn_ids=None,
    )
    return enc_x


def main():
    print(f"[precision] jax_default_matmul_precision = "
          f"{jax.config.jax_default_matmul_precision}", flush=True)
    rng = np.random.default_rng(12345)
    global_max = 0.0
    per_seq = []
    for i in range(20):
        S = int(rng.integers(5, 31))
        toks = rng.integers(1, NUM_VERTICES + 1, size=S).astype(np.int32)
        inc = incremental_encode_fast(agent, list(toks))
        full = full_encode(agent, toks)
        diff = float(jnp.max(jnp.abs(inc - full)))
        per_seq.append((S, diff))
        global_max = max(global_max, diff)
        print(f"[seq {i:2d}] len={S:2d}  max_abs_diff={diff:.3e}", flush=True)

    print(f"\n[ASSERT] max_abs_diff = {global_max:.6e}", flush=True)
    ok = global_max < 1e-4
    print(f"[ASSERT] {'PASS' if ok else 'FAIL'} (threshold 1e-4)", flush=True)

    # -----------------------------------------------------------------------
    # 4. BRANCHING: encode a prefix, extend by 3 different next tokens from
    #    the SAME carried state; verify each matches full re-encode of
    #    prefix+[token].
    # -----------------------------------------------------------------------
    print("\n[branch] branching test from shared carried state", flush=True)
    prefix = rng.integers(1, NUM_VERTICES + 1, size=12).astype(np.int32)

    # carry the state through the prefix
    layers = agent.encoder.layers
    carries = [_init_carry(layer.attn_layer) for layer in layers]
    for pos, tok in enumerate(prefix):
        x_t = agent.embedding(jnp.asarray(tok, dtype=jnp.int32)) + agent.pos_enc.pe[pos, :]
        for li, layer in enumerate(layers):
            x_t, carries[li] = _layer_step_j(layer, x_t, carries[li])
    prefix_carries = [(M, I) for (M, I) in carries]  # snapshot

    branch_max = 0.0
    for j in range(3):
        nxt = int(rng.integers(1, NUM_VERTICES + 1))
        pos = len(prefix)
        # branch from the shared snapshot
        bc = [(M, I) for (M, I) in prefix_carries]
        x_t = agent.embedding(jnp.asarray(nxt, dtype=jnp.int32)) + agent.pos_enc.pe[pos, :]
        for li, layer in enumerate(layers):
            x_t, bc[li] = _layer_step_j(layer, x_t, bc[li])
        inc_last = agent.final_norm(x_t)             # (E,) for the new token

        full_seq = np.concatenate([prefix, [nxt]]).astype(np.int32)
        full = full_encode(agent, full_seq)
        full_last = full[-1]
        d = float(jnp.max(jnp.abs(inc_last - full_last)))
        branch_max = max(branch_max, d)
        print(f"[branch {j}] next_tok={nxt:3d}  last_token max_abs_diff={d:.3e}", flush=True)

    print(f"\n[BRANCH] max_abs_diff = {branch_max:.6e}", flush=True)
    print(f"[BRANCH] {'PASS' if branch_max < 1e-4 else 'FAIL'} (threshold 1e-4)", flush=True)


if __name__ == "__main__":
    main()
