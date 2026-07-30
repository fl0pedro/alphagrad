#!/usr/bin/env python3
"""Fix GA/SA IndexError: elimination can PRUNE vertices, so random orders may be
SHORTER than NV — swap-mutation indices must use len(order), not NV."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
n = 0
old_ga = '                if _orng.random() < 0.3:\n                    a, b = int(_orng.integers(NV)), int(_orng.integers(NV)); co[a], co[b] = co[b], co[a]'
new_ga = '                if _orng.random() < 0.3 and len(co) >= 2:\n                    _L = len(co)\n                    a, b = int(_orng.integers(_L)), int(_orng.integers(_L)); co[a], co[b] = co[b], co[a]'
if old_ga in s:
    s = s.replace(old_ga, new_ga, 1); n += 1
old_sa = '                o2 = list(cur[0]); a, b = int(_orng.integers(NV)), int(_orng.integers(NV))\n                o2[a], o2[b] = o2[b], o2[a]; nxt = (o2, _rand_approx(_orng, NV))'
new_sa = '                o2 = list(cur[0]); _L = max(len(o2), 2)\n                a, b = int(_orng.integers(_L)), int(_orng.integers(_L))\n                o2[a], o2[b] = o2[b], o2[a]; nxt = (o2, _rand_approx(_orng, NV))'
if old_sa in s:
    s = s.replace(old_sa, new_sa, 1); n += 1
p.write_text(s)
print(f"GA/SA length fix: {n} sites patched" if n else "NO anchors matched (check manually)")
