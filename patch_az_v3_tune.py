#!/usr/bin/env python3
"""Tune the v3 optimizer branch: sparse micro-actions (most vertices none),
valid-only budget counting (redraws free), DIAG i!=j, tighter default axis range."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "_MICRO_P" in s:
    print("ALREADY TUNED"); sys.exit(0)

old_rm = (
    '    def _rand_micro(rng):\n'
    '        op = _OPS[int(rng.integers(len(_OPS)))]\n'
    '        if op == "none":\n'
    '            return None'
)
new_rm = (
    '    _MICRO_P = float(_os_mf.environ.get("ALPHAGRAD_MF_MICRO_P", "0.25"))\n'
    '    _OPS_NN = [o for o in _OPS if o != "none"] or ["quant"]\n'
    '    def _rand_micro(rng):\n'
    '        if (not _APPROX) or rng.random() >= _MICRO_P:\n'
    '            return None\n'
    '        op = _OPS_NN[int(rng.integers(len(_OPS_NN)))]\n'
    '        if op == "none":\n'
    '            return None'
)
assert old_rm in s, "anchor rand_micro"; s = s.replace(old_rm, new_rm, 1)

old_d = (
    '        if op == "diag":\n'
    '            return ("d", int(rng.integers(_MAXAX)), int(rng.integers(_MAXAX)),\n'
    '                    _FACS[int(rng.integers(len(_FACS)))])'
)
new_d = (
    '        if op == "diag":\n'
    '            i = int(rng.integers(_MAXAX)); j = int(rng.integers(_MAXAX))\n'
    '            if j == i: j = (i + 1) % max(_MAXAX, 2)\n'
    '            return ("d", i, j, _FACS[int(rng.integers(len(_FACS)))])'
)
assert old_d in s, "anchor diag"; s = s.replace(old_d, new_d, 1)

old_meas = (
    '            r = _measure1(cand); tries = 0\n'
    '            while r is None and _APPROX and cand[1] is not None and tries < _REDRAW:\n'
    '                cand = (cand[0], _rand_approx(_orng, len(cand[0]))); r = _measure1(cand); tries += 1\n'
    '                _n[0] += 1  # a redraw is a fresh measurement\n'
    '            _n[0] += 1\n'
    '            if r is None:\n'
    '                out.append((cand, None)); continue\n'
    '            _mf_bufY.append(r); out.append((cand, r))'
)
new_meas = (
    '            r = _measure1(cand); tries = 0\n'
    '            while r is None and _APPROX and cand[1] is not None and tries < _REDRAW:\n'
    '                cand = (cand[0], _rand_approx(_orng, len(cand[0]))); r = _measure1(cand); tries += 1\n'
    '            if r is None:\n'
    '                out.append((cand, None)); continue\n'
    '            _n[0] += 1  # only VALID measurements count toward the budget\n'
    '            _mf_bufY.append(r); out.append((cand, r))'
)
assert old_meas in s, "anchor measure"; s = s.replace(old_meas, new_meas, 1)

s = s.replace('_MAXAX = int(_os_mf.environ.get("ALPHAGRAD_MF_MAX_AX", "3"))',
              '_MAXAX = int(_os_mf.environ.get("ALPHAGRAD_MF_MAX_AX", "2"))', 1)
p.write_text(s)
print("v3 tuned: sparse micro (MICRO_P=0.25), i!=j, valid-only budget, MAX_AX=2")
