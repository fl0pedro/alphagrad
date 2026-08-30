"""decode6_dscheck: is a STORED dataset still what the CURRENT tokenizer emits?

decode3_dsA.npz / decode3_dsB.npz were built 2026-08-14 19:51/19:54, i.e.
BEFORE aa0774c (token budgets, env.py+ppo.py) and 09e7a3c (examples.py
TransformerLM3).  Neither commit is supposed to touch what
``decode3_data.py`` emits -- it imports ``graphax.IncrementalPathTokenizer``
and ``alphagrad.elimrl.*`` directly, not ``alphagrad.approx.env`` -- but
"supposed to" is not a check.  The generator is deterministic (trajectory i
uses seed 1000+i, mode alternates rev/rand), so rebuilding the FIRST 8
trajectories with today's code must reproduce the stored rows EXACTLY.

Usage: decode6_dscheck.py STORED.npz TINY.npz
Exit 0 only if every compared array is bit-identical.
"""
import sys
import numpy as np

S = np.load(sys.argv[1], allow_pickle=True)
T = np.load(sys.argv[2], allow_pickle=True)
n = int(T["tok"].shape[0])
print(f"stored {sys.argv[1]}  tok{S['tok'].shape}  faces {len(S['f_traj'])}")
print(f"tiny   {sys.argv[2]}  tok{T['tok'].shape}  faces {len(T['f_traj'])}")
print(f"comparing the first {n} trajectories")
print(f"shapes flag stored={int(S['shapes'])} tiny={int(T['shapes'])}")

bad = []
if int(S["shapes"]) != int(T["shapes"]):
    bad.append("shapes flag differs")

# ---- per-trajectory token stream, byte for byte over the REAL length
for i in range(n):
    ns, nt = int(S["ntok"][i]), int(T["ntok"][i])
    if ns != nt:
        bad.append(f"traj {i}: ntok {ns} != {nt}")
        continue
    for k in ("tok", "own", "eqn", "did"):
        a = S[k][i, :ns]
        b = T[k][i, :nt]
        if not np.array_equal(a, b):
            d = int((a != b).sum())
            bad.append(f"traj {i}: {k} differs in {d}/{ns} positions "
                       f"(first at {int(np.nonzero(a != b)[0][0])})")

# ---- face records belonging to those trajectories
ms = S["f_traj"] < n
mt = T["f_traj"] < n
if int(ms.sum()) != int(mt.sum()):
    bad.append(f"face count {int(ms.sum())} != {int(mt.sum())}")
else:
    for k in ("f_traj", "f_q", "f_start", "f_split", "f_si", "f_sj", "f_v",
              "f_tgt", "f_ext"):
        a, b = S[k][ms], T[k][mt]
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} != {b.shape}")
        elif not np.array_equal(a, b):
            bad.append(f"{k}: {int((a != b).sum())} differing entries")

# ---- the static per-vertex side and the participation mask
for k in ("vfeat", "part", "mode"):
    a, b = S[k][:n], T[k][:n]
    if a.shape != b.shape:
        bad.append(f"{k}: shape {a.shape} != {b.shape}")
    elif not np.array_equal(a, b):
        bad.append(f"{k}: {int((a != b).sum())} differing entries")

# ---- budget report: what the LIVE pipeline would need for this stream
d = np.diff(np.sort(np.unique(S["did"][S["did"] >= 0])))
print("\nSTREAM BUDGET (stored, all trajectories)")
print(f"  stream len  mean {S['ntok'].mean():.0f}  max {int(S['ntok'].max())}")
per = []
for i in range(S["tok"].shape[0]):
    dd = S["did"][i, :int(S["ntok"][i])]
    dd = dd[dd >= 0]
    if len(dd):
        per.append(np.bincount(dd.astype(np.int64)).max())
if per:
    print(f"  max single delta {int(max(per))}  "
          f"(ALPHAGRAD_MAX_DELTA_TOKENS default is 32768 since aa0774c)")
    over = int(sum(1 for p in per if p > 32768))
    print(f"  trajectories whose worst delta EXCEEDS 32768: {over}/{len(per)}")
print(f"  max token id {int(S['tok'].max())}  (embedding vocab 512)")

if bad:
    print("\nDATASET IS STALE / NOT REPRODUCIBLE:")
    for b in bad[:40]:
        print("  ", b)
    raise SystemExit(1)
print("\nDSCHECK PASS -- stored rows are bit-identical to a fresh build "
      "with today's code")
