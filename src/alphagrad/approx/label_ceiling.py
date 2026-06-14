"""Label ceiling for the quality-signal experiment: how high CAN any signal's
Spearman vs the 20-seed accuracy go, given seed noise? Uses the stored 20
per-seed accuracies per rule (no retraining) via split-half reliability +
Spearman-Brown; ceiling = sqrt(reliability_20seed).

  uv run label_ceiling.py
"""
import os, glob, json
import numpy as np

rng = np.random.default_rng(0)


def load():
    out = {"cmorl": [], "mogfn": []}
    for s in ("cmorl", "mogfn"):
        for f in glob.glob(os.path.expanduser(f"~/dsnn/train_exp/{s}_shard*.json")):
            try: d = json.load(open(f))
            except Exception: continue
            for r in d.get("results", []):
                a = r.get("accs")
                if a and len(a) >= 10:
                    out[s].append(np.array(a, float))
    return out


def rank(x): return np.argsort(np.argsort(np.asarray(x, float))).astype(float)
def pear(a, b): a = a - a.mean(); b = b - b.mean(); return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def spear(x, y): return pear(rank(x), rank(y))


def split_half(accs, n_splits=200):
    """Mean Spearman between two random disjoint halves of the seeds, across rules."""
    A = np.stack(accs)  # (R, S)
    R, S = A.shape
    h = S // 2
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(S)
        a = A[:, perm[:h]].mean(1); b = A[:, perm[h:2 * h]].mean(1)
        rs.append(spear(a, b))
    return float(np.mean(rs)), R, S


for s, accs in load().items():
    if not accs:
        print(f"{s}: no data"); continue
    r_hh, R, S = split_half(accs)
    # Spearman-Brown: reliability of the FULL S-seed mean from the (S/2)-seed half.
    r_full = 2 * r_hh / (1 + r_hh)
    ceiling = np.sqrt(max(r_full, 0))
    print(f"{s}: rules={R} seeds={S} | split-half(S/2 vs S/2) rho={r_hh:.3f} "
          f"| {S}-seed reliability={r_full:.3f} | CEILING rho<= {ceiling:.3f}")

# both combined
allaccs = sum(load().values(), [])
r_hh, R, S = split_half(allaccs)
r_full = 2 * r_hh / (1 + r_hh)
print(f"both: rules={R} seeds={S} | split-half rho={r_hh:.3f} | reliability={r_full:.3f} | CEILING rho<= {np.sqrt(max(r_full,0)):.3f}")
print("\n(compare: aligned_cos achieved Spearman ~0.79 both / 0.69 cmorl / 0.82 mogfn)")
