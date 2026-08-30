"""decode3_check: prove the harness's ONE concatenated encode reproduces the
LIVE sequential path of the 2026-08-14 architecture.

Nothing here is assumed.  Every input path the two trainers use is compared,
element for element, against the code the rollout actually runs:

  rows      one encode over the concatenated stream   vs  carry_stream
            .init_carry over the base + one encode_extend per elimination
            delta (the loop .advance runs).
  identity  decode3_arch.identity over the full stream  vs  carry_stream
            .base_identity_stream over the base alone, both through
            Agent.identity_pool.
  vmem      decode3_arch.mem_at(t)  vs  the sequential (V+2) memory built by
            carry_stream.advance's participation crediting with the STORED
            (V+1) mask.
  contexts  decode3_arch.heads  vs  Agent.heads_from_memory.
  face      the [start, split) slice mean  vs  the live SIDE CARRY:
            one encode per face chunk branching off the step's carry, pooled
            by Agent._face_pool.  These are NOT identical by construction (the
            side carry never reads the step header), so this number is what
            decides whether the cheap pooling is a faithful stand-in.

Every window is PADDED to one static width per loop, so the whole check is
three compiles, not one per delta length.

Usage: decode3_check.py DS.npz [TRAJ]
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp, jax.random as jrand

import decode3_arch as ARCH
from alphagrad.approx.common import carry_stream as CS

d = np.load(sys.argv[1], allow_pickle=True)
i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
E = 32
TOK, OWN, EQN, DID, NTOK = d["tok"], d["own"], d["eqn"], d["did"], d["ntok"]
PART = d["part"]
NV = d["vfeat"].shape[0]
QSTEPS = [int(x) for x in d["qsteps"]]
NSTEP = PART.shape[1]
print(f"traj {i}  NV={NV} NSTEP={NSTEP} ntok={int(NTOK[i])} "
      f"arm={'B' if int(d['shapes']) else 'A'}", flush=True)

agent = ARCH.build(E, 3, 2, 2, NV, jrand.PRNGKey(0))

n = int(NTOK[i])
tok_np = np.asarray(TOK[i, :n]); eqn_np = np.asarray(EQN[i, :n])
tok = jnp.asarray(tok_np); eqn = jnp.asarray(eqn_np)
own = np.asarray(OWN[i, :n]); did = np.asarray(DID[i, :n]).astype(np.int32)
part = jnp.asarray(PART[i].astype(np.float32))

# ---------------- A: harness ----------------------------------------------
rowsA, wA = ARCH.encode(agent, tok, eqn, n, n)
identA = ARCH.identity(agent, rowsA, wA, jnp.asarray(own), jnp.asarray(did), NV)
tab = ARCH.memory_tables(rowsA, wA, jnp.asarray(own), eqn, jnp.asarray(did),
                         part, NV, NSTEP)
rowsA_np = np.asarray(rowsA)

# ---------------- B: live sequential --------------------------------------
nb = int(np.nonzero(did >= 0)[0][0])
base_owner = np.where(own[:nb] >= 0, own[:nb] + 1, 0).astype(np.int32)

carry, S, C = CS.init_carry(agent, tok[:nb], eqn[:nb], nb, window=nb,
                            total_v=NV, embd_dim=E, base_owners=base_owner)
identB = CS.base_identity_stream(agent, tok[:nb], eqn[:nb], nb, window=nb,
                                 total_v=NV, base_owners=base_owner)
identB_pool = agent.identity_pool(identB[0], identB[1], identB[2], NV + 2)
print(f"identity  max|dA-B| "
      f"{float(jnp.max(jnp.abs(identA - identB_pool))):.3e}  "
      f"scale {float(jnp.max(jnp.abs(identB_pool))):.3e}", flush=True)

# per-step delta slices, PADDED to one static window
spans = []
for t in range(NSTEP):
    ws = np.nonzero(did == t)[0]
    spans.append((int(ws[0]), int(ws[-1]) + 1) if len(ws) else (nb, nb))
WD = max(1, max(b - a for a, b in spans))
TD = np.zeros((NSTEP, WD), np.int32)
ED = np.full((NSTEP, WD), -1, np.int32)
CD = np.zeros(NSTEP, np.int32)
for t, (a, b) in enumerate(spans):
    TD[t, :b - a] = tok_np[a:b]; ED[t, :b - a] = eqn_np[a:b]; CD[t] = b - a
TD = jnp.asarray(TD); ED = jnp.asarray(ED)
print(f"delta window {WD} (one compile)", flush=True)

worst_S = worst_C = worst_R = worst_ctx = 0.0
mem_seq = {}
step_carries = {}
for t in range(NSTEP):
    mem_seq[t] = (S, C)
    step_carries[t] = carry
    cnt = int(CD[t])
    if cnt == 0:
        continue
    carry, rows_t, valid_t, eqn_t = agent.encode_extend(
        carry, TD[t], ED[t], cnt, window=WD, start=0, chunk=0)
    a, b = spans[t]
    r = np.asarray(rows_t[:cnt])
    worst_R = max(worst_R, float(np.max(np.abs(rowsA_np[a:b] - r))
                                 / (np.max(np.abs(r)) + 1e-9)))
    # carry_stream.advance's memory half, on the rows just produced: the
    # comparison isolates the CREDITING from the encoder.
    w_ = valid_t.astype(jnp.float32)
    w_eqn = w_ * (eqn_t >= 0).astype(jnp.float32)
    w_str = w_ - w_eqn
    p = part[t]
    S = S.at[:NV + 1].add(p[:, None] * jnp.sum(rows_t * w_eqn[:, None], 0))
    C = C.at[:NV + 1].add(p * jnp.sum(w_eqn))
    S = S.at[NV].add(jnp.sum(rows_t * w_str[:, None], 0))
    C = C.at[NV].add(jnp.sum(w_str))
    S = S.at[NV + 1].add(jnp.sum(rows_t * w_[:, None], 0))
    C = C.at[NV + 1].add(jnp.sum(w_))
mem_seq[NSTEP] = (S, C)

for t in list(QSTEPS) + [NSTEP]:
    if t not in mem_seq:
        continue
    Sh, Ch = ARCH.mem_at(tab, jnp.asarray(t), NV, E, NSTEP)
    Sb, Cb = mem_seq[t]
    worst_S = max(worst_S, float(jnp.max(jnp.abs(Sh - Sb))
                                 / (jnp.max(jnp.abs(Sb)) + 1e-9)))
    worst_C = max(worst_C, float(jnp.max(jnp.abs(Ch - Cb))))
    ctxh, _ = ARCH.heads(agent, Sh, Ch, identA)
    _l, ctxb, _v = agent.heads_from_memory(Sb, Cb, identity_stream=identB)
    worst_ctx = max(worst_ctx, float(jnp.max(jnp.abs(ctxh - ctxb))
                                     / (jnp.max(jnp.abs(ctxb)) + 1e-9)))

print(f"rows      worst rel {worst_R:.3e}   (one encode vs per-delta extend)",
      flush=True)
print(f"vmem      worst rel|dS| {worst_S:.3e}  max|dC| {worst_C:.3e}",
      flush=True)
print(f"contexts  worst rel {worst_ctx:.3e}   "
      f"(harness heads vs Agent.heads_from_memory)", flush=True)

# ---------------- face chunk: slice mean vs live SIDE CARRY ---------------
f_traj = d["f_traj"]; sel = np.nonzero(f_traj == i)[0]
f_q, f_s, f_p = d["f_q"], d["f_start"], d["f_split"]
STEPS_CHK = sorted(set(int(f_q[k]) for k in sel))[:12]
items = [(int(f_q[k]), int(f_s[k]), int(f_p[k])) for k in sel
         if int(f_q[k]) in STEPS_CHK and int(f_p[k]) > int(f_s[k])
         and int(f_p[k]) <= n]
WF = max(1, max(p - s for _q, s, p in items)) if items else 1
CAP = int(WF)
TF = np.zeros((len(items), WF), np.int32)
EF = np.full((len(items), WF), -1, np.int32)
CF = np.zeros(len(items), np.int32)
for k, (_q, s_, p_) in enumerate(items):
    TF[k, :p_ - s_] = tok_np[s_:p_]; EF[k, :p_ - s_] = eqn_np[s_:p_]
    CF[k] = p_ - s_
TF = jnp.asarray(TF); EF = jnp.asarray(EF)
print(f"face window {WF} over {len(items)} faces (one compile)", flush=True)

worst_f = 0.0
cur_t = None
side = None
for k, (q_, s_, p_) in enumerate(items):
    if q_ != cur_t:
        cur_t, side = q_, step_carries[q_]
    side, rw, vv, _e = agent.encode_extend(
        side, TF[k], EF[k], int(CF[k]), window=WF, start=0, chunk=0)
    live = agent._face_pool(rw, vv, jnp.asarray(int(CF[k])))
    har = ARCH.face_latent(rowsA, wA, jnp.asarray(s_), jnp.asarray(p_), CAP)
    worst_f = max(worst_f, float(jnp.max(jnp.abs(har - live))
                                 / (float(jnp.max(jnp.abs(live))) + 1e-9)))
print(f"face pool worst rel {worst_f:.3e}   "
      f"(harness slice mean vs live side carry)", flush=True)

# ---------------- participation sanity ------------------------------------
fs_i, fs_j, fv = d["f_si"], d["f_sj"], d["f_v"]
bad = 0
chk = 0
for t in range(NSTEP):
    ks = [k for k in sel if int(f_q[k]) == t]
    if not ks:
        continue
    want = {int(fv[ks[0]])}
    for k in ks:
        for s_ in (int(fs_i[k]), int(fs_j[k])):
            if s_ >= 0:
                want.add(s_)
    got = set(np.nonzero(np.asarray(PART[i, t]))[0].tolist())
    chk += 1
    if not want <= got:
        bad += 1
print(f"participation: stored mask covers {{v}} u face endpoints at "
      f"{chk - bad}/{chk} steps -> {'OK' if bad == 0 else 'MISMATCH'}",
      flush=True)
