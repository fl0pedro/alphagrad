"""Detect whether the running sweep's cosine is contaminated by the mid-run
_align_jac deploy: compare cosine in early vs late rows of the sweep jsonl. If late
rows have systematically higher cosine, passes measured after the fix use aligned
cosine while earlier passes don't.
"""
import json
import numpy as np

f = "/Users/assmuth/dsnn/search_space_grad/results_seed12345.jsonl"
lines = open(f).readlines()
N = len(lines)
early, late = [], []
for k, line in enumerate(lines):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("invalid"):
        continue
    rv = d.get("reward_vec")
    if not rv or rv[6] != rv[6]:
        continue
    if k < 150000:
        early.append(rv[6])
    elif k > N - 150000:
        late.append(rv[6])
e, l = np.array(early), np.array(late)
print(f"total lines={N}")
print(f"EARLY (first 150k lines): n={len(e)} mean_cos={e.mean():.3f} frac>0.9={(e>0.9).mean():.2f} frac<0.05={(e<0.05).mean():.2f}")
print(f"LATE  (last 150k lines):  n={len(l)} mean_cos={l.mean():.3f} frac>0.9={(l>0.9).mean():.2f} frac<0.05={(l<0.05).mean():.2f}")
