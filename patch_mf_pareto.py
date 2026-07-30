#!/usr/bin/env python3
"""Persist EVERY measured (raw, order, approx) in the model-free optimizer arms
so each run's Pareto front over {lat, xla_peak, cos} is computable offline."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "_mf_all" in s:
    print("ALREADY PATCHED"); sys.exit(0)

o1 = '    _mf_bufY = []\n'
n1 = '    _mf_bufY = []\n    _mf_all = []          # every measured (n, raw, order, approx) -> offline Pareto front\n'
assert o1 in s, "bufY anchor"; s = s.replace(o1, n1, 1)

o2 = ("            _n[0] += 1  # only VALID measurements count toward the budget\n"
      "            _mf_bufY.append(r); out.append((cand, r))")
n2 = ("            _n[0] += 1  # only VALID measurements count toward the budget\n"
      "            _mf_bufY.append(r); out.append((cand, r))\n"
      "            _mf_all.append({\"n\": _n[0], \"raw\": r.tolist(),\n"
      "                            \"order\": list(map(int, cand[0])),\n"
      "                            \"approx\": ([list(m) if m else None for m in cand[1]]\n"
      "                                       if cand[1] is not None else None)})")
assert o2 in s, "measure anchor"; s = s.replace(o2, n2, 1)

o3 = ('    _json_mf.dump({"optimizer": _opt, "approx": _APPROX, "seed": int(A.seed),\n'
      '                   "budget": int(_budget), "best": _mf_best, "history": _mf_hist},')
n3 = ('    _json_mf.dump({"optimizer": _opt, "approx": _APPROX, "seed": int(A.seed),\n'
      '                   "budget": int(_budget), "best": _mf_best, "history": _mf_hist,\n'
      '                   "all_measurements": _mf_all},')
assert o3 in s, "dump anchor"; s = s.replace(o3, n3, 1)
p.write_text(s)
print("optimizer arms now persist all measurements")
