"""decode2_face_check: the decode_check proof, for the FACE input path.

Proves that the harness's ONE concatenated encode, pooled over a face's own
[start, split) chunk, equals what the LIVE path builds -- carry_stream
init_carry over the base then one encode_extend per elimination delta, with
the face chunk pooled inside that delta's rows (ppo._face_encode / _face_pool).

Also proves the per-step vertex memory the harness rebuilds by segment-summing
the deltas equals the sequential vmem.update_ids fold.
"""
import os, sys
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
import numpy as np, jax, jax.numpy as jnp, jax.random as jrand
from types import SimpleNamespace
from alphagrad.approx.ppo import _build_agent
from alphagrad.approx import vertex_memory as vmem

d = np.load(sys.argv[1], allow_pickle=True)
TOK, OWN, EQN, DID, NTOK = d["tok"], d["own"], d["eqn"], d["did"], d["ntok"]
VFEAT = d["vfeat"]; NV = VFEAT.shape[0]; E = 32
i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
args = SimpleNamespace(vocab_size=512, embd_dim=E, op_embd_dim=8, num_layers=3,
                       num_heads=2, hidden_dim=64, value_dims="64,32",
                       set_pointer=True, set_pointer_blocks=2,
                       dynamic_substeps=False, no_approx_head=True,
                       live_faces=False, face_actions=False,
                       unified_head=False, unified_face_head=False,
                       max_substeps=1)
agent = _build_agent(args, NV, 1, 1, jrand.PRNGKey(0))

n = int(NTOK[i])
tok = jnp.asarray(TOK[i, :n]); eqn = jnp.asarray(EQN[i, :n])
own = np.asarray(OWN[i, :n]); did = np.asarray(DID[i, :n])

# --- A: ONE encode over the concatenated stream --------------------------
os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "1"
c0 = agent.carry_init()
_, rowsA, validA, _ = agent.encode_extend(c0, tok, eqn, n, window=n, start=0,
                                          chunk=0)

# --- B: base, then ONE encode_extend per elimination delta ---------------
os.environ["ALPHAGRAD_CHUNKED_EXTEND"] = "0"
cuts = [0]
for t in range(int(did.max()) + 1):
    w = np.nonzero(did == t)[0]
    if len(w):
        cuts.append(int(w[-1]) + 1)
cuts = sorted(set(cuts + [n]))
carry = agent.carry_init()
rowsB = np.zeros((n, E), np.float32)
S = jnp.zeros((NV + 1, E), jnp.float32); C = jnp.zeros((NV + 1,), jnp.float32)
seq_mem = {}
for a, b in zip(cuts[:-1], cuts[1:]):
    w = b - a
    carry, r, v, _ = agent.encode_extend(carry, tok, eqn, w, window=w, start=a,
                                         chunk=0)
    rowsB[a:b] = np.asarray(r[:w])
    seq_mem[a] = (S, C)                      # memory BEFORE this block
    ids = jnp.where(jnp.asarray(own[a:b]) < 0, -1, jnp.asarray(own[a:b]))
    S, C = vmem.update_ids(S, C, r[:w], ids, v[:w])
seq_mem[n] = (S, C)

rowsA_np = np.asarray(rowsA[:n])
den = np.max(np.abs(rowsB)) + 1e-9
print(f"rows      max|dA-B| {np.max(np.abs(rowsA_np - rowsB)):.3e}  "
      f"rel {np.max(np.abs(rowsA_np - rowsB))/den:.3e}")

# --- face chunk pooling ---------------------------------------------------
f_traj = d["f_traj"]; sel = np.nonzero(f_traj == i)[0]
worst = 0.0
for kk in sel[:200]:
    s_, p_ = int(d["f_start"][kk]), int(d["f_split"][kk])
    if p_ <= s_ or p_ > n:
        continue
    a = rowsA_np[s_:p_].mean(0); b = rowsB[s_:p_].mean(0)
    worst = max(worst, float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9)))
print(f"face chunk means: {len(sel)} faces, worst relerr {worst:.3e}")

# --- per-step vertex memory: harness segment-sum vs sequential fold -------
oid = np.where(own < 0, NV, np.minimum(own, NV - 1))
NSTEP = int(did.max()) + 1
base = did < 0
bs = np.zeros((NV + 1, E), np.float32); bc = np.zeros(NV + 1, np.float32)
np.add.at(bs, oid[base], rowsA_np[base]); np.add.at(bc, oid[base], 1.0)
Sh, Ch = bs.copy(), bc.copy()
worstS = worstC = 0.0
for t in range(NSTEP):
    a0 = int(np.nonzero(did == t)[0][0]) if (did == t).any() else None
    if a0 is not None and a0 in seq_mem:
        sB, cB = seq_mem[a0]
        worstS = max(worstS, float(np.max(np.abs(Sh - np.asarray(sB)))
                                   / (np.max(np.abs(np.asarray(sB))) + 1e-9)))
        worstC = max(worstC, float(np.max(np.abs(Ch - np.asarray(cB)))))
    m = did == t
    if m.any():
        np.add.at(Sh, oid[m], rowsA_np[m]); np.add.at(Ch, oid[m], 1.0)
print(f"per-step vmem: worst rel|dS| {worstS:.3e}  max|dC| {worstC:.3e}")
