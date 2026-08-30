"""decode_check: prove the harness's ONE concatenated encode == the live
sequential carry_stream.init_carry/advance path (bitwise up to float assoc).
"""
import os, sys
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
import numpy as np, jax, jax.numpy as jnp, jax.random as jrand, equinox as eqx
from types import SimpleNamespace
from alphagrad.approx.ppo import _build_agent
from alphagrad.approx.common import carry_stream as cs
from alphagrad.approx import vertex_memory as vmem

d = np.load(sys.argv[1], allow_pickle=True)
TOK, OWN, EQN, NTOK, PREF = d["tok"], d["own"], d["eqn"], d["ntok"], d["pref"]
VFEAT = d["vfeat"]; NV = VFEAT.shape[0]
E = 32
args = SimpleNamespace(vocab_size=512, embd_dim=E, op_embd_dim=8, num_layers=3,
                       num_heads=2, hidden_dim=64, value_dims="64,32",
                       set_pointer=True, set_pointer_blocks=2,
                       dynamic_substeps=False, no_approx_head=True,
                       live_faces=False, face_actions=False,
                       unified_head=False, unified_face_head=False,
                       max_substeps=1)
agent = _build_agent(args, NV, 1, 1, jrand.PRNGKey(0))

i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(NTOK[i]); pref = PREF[i]
tok = jnp.asarray(TOK[i, :n]); own = jnp.asarray(OWN[i, :n]); eqn = jnp.asarray(EQN[i, :n])

# --- A: the harness path (ONE encode over the concatenated stream) ---
os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "1"
c0 = agent.carry_init()
_, rows, valid, _ = agent.encode_extend(c0, tok, eqn, n, window=n, start=0, chunk=0)
ar = jnp.arange(n)


def fold(w):
    ids = jnp.where(own < 0, NV, jnp.minimum(own, NV - 1)).astype(jnp.int32)
    s = jax.ops.segment_sum(rows * w[:, None], ids, num_segments=NV + 1)
    c = jax.ops.segment_sum(w, ids, num_segments=NV + 1)
    return s, c


# --- B: the live path (init_carry over the base, then advance per delta) ---
# reconstruct the delta boundaries from the owner stream: a delta is a maximal
# run with a single owner slot, and the base is everything before the first
# delta.  Use the recorded prefix lengths for the query checkpoints instead.
# Sequential = repeated encode_extend on the SAME buffer with carry.pos.
os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "0"
cut = [0] + [int(p) for p in pref] + [n]
cut = sorted(set(cut))
carry = agent.carry_init()
S = jnp.zeros((NV + 1, E), jnp.float32); C = jnp.zeros((NV + 1,), jnp.float32)
seqS = {}
for a, b in zip(cut[:-1], cut[1:]):
    w = b - a
    carry, r, v, _ = agent.encode_extend(carry, tok, eqn, w, window=w, start=a,
                                         chunk=0)
    ids = jnp.where(own[a:b] < 0, -1, own[a:b])
    S, C = vmem.update_ids(S, C, r, ids, v)
    seqS[b] = (S, C)

print("checkpoint |  max|dSums|   max|dCounts|  relerr(ctx)")
for qi, p in enumerate(pref):
    p = int(p)
    if p not in seqS:
        continue
    sA, cA = fold((ar < p).astype(jnp.float32))
    sB, cB = seqS[p]
    _, ctxA, _ = agent.heads_from_memory(sA, cA, vertex_features=jnp.asarray(VFEAT))
    _, ctxB, _ = agent.heads_from_memory(sB, cB, vertex_features=jnp.asarray(VFEAT))
    rel = float(jnp.max(jnp.abs(ctxA - ctxB)) / (jnp.max(jnp.abs(ctxB)) + 1e-9))
    print(f"  t={qi} p={p:6d} | {float(jnp.max(jnp.abs(sA-sB))):.3e}  "
          f"{float(jnp.max(jnp.abs(cA-cB))):.3e}   {rel:.3e}")
