#!/usr/bin/env python3
"""Route the model-free optimizers' _measure1 through the subprocess measure
server when ALPHAGRAD_MEASURE_SERVER=1 (CUDA-poison isolation for the random/
SA/GA arms — a poisoned kernel kills the child, the parent redraws)."""
import pathlib, sys
p = pathlib.Path("src/alphagrad/approx/autoscheduler_loop.py")
s = p.read_text()
if "_MS_CLIENT" in s:
    print("ALREADY PATCHED"); sys.exit(0)
anchor = "    def _measure1(cand):"
new = ('    _MS_ON = _os_mf.environ.get("ALPHAGRAD_MEASURE_SERVER", "0") == "1"\n'
       '    _MS_CLIENT = [None]\n'
       '    def _measure1(cand):\n'
       '        if _MS_ON:\n'
       '            if _MS_CLIENT[0] is None:\n'
       '                from alphagrad.approx.measure_client import MeasureClient\n'
       '                _MS_CLIENT[0] = MeasureClient()\n'
       '            return _MS_CLIENT[0].measure_seq(_mf_seq(cand))\n'
       '        return _measure1_inproc(cand)\n'
       '    def _measure1_inproc(cand):')
assert anchor in s, "measure1 anchor"
s = s.replace(anchor, new, 1)
p.write_text(s)
print("optimizer _measure1 routed through measure server (flag-gated)")
