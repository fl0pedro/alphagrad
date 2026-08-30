#!/usr/bin/env python3
"""Leaf-by-leaf bit-identity between two ALPHAGRAD_EQ_DUMP prefixes."""
from __future__ import annotations
import glob, os, pickle, sys
import numpy as np

def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from leaves(obj[k], f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj

def main(a, b):
    fa = sorted(glob.glob(f"{a}.ep*.pkl"))
    fb = sorted(glob.glob(f"{b}.ep*.pkl"))
    if not fa or not fb:
        print(f"NO DUMPS: base={len(fa)} post={len(fb)}"); return 2
    if len(fa) != len(fb):
        print(f"EPISODE COUNT DIFFERS: {len(fa)} vs {len(fb)}"); return 2
    total = same = diff = 0
    for pa, pb in zip(fa, fb):
        A = pickle.load(open(pa, "rb")); B = pickle.load(open(pb, "rb"))
        la = dict(leaves(A)); lb = dict(leaves(B))
        if la.keys() != lb.keys():
            only = set(la) ^ set(lb)
            print(f"{os.path.basename(pa)}: KEY SET DIFFERS ({len(only)}): "
                  f"{sorted(only)[:6]}")
            return 2
        for k in la:
            total += 1
            x, y = np.asarray(la[k]), np.asarray(lb[k])
            if x.shape == y.shape and np.array_equal(x, y, equal_nan=True):
                same += 1
            else:
                diff += 1
                if diff <= 8:
                    print(f"  DIFF {k}: shapes {x.shape} vs {y.shape}")
        print(f"{os.path.basename(pa)}: {len(la)} leaves compared")
    print(f"\nTOTAL {total} leaves: {same} bit-identical, {diff} different")
    print("EQ_DUMP RESULT: " + ("PASS (bit-identical)" if diff == 0 else "FAIL"))
    return 0 if diff == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
