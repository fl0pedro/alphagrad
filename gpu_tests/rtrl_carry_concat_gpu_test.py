# The GPU check of dsnn-dfw.304 on mem_objective_test's RSNN_SHD rtrl program: the exact plan joins the two carry
# stacks and is not bigger for it. From graphax d5114d3 XLA merges the W-stack and V-stack dots of a hidden state into
# one dot on a concatenation f32[512,105984]. Under the measure compiler options that cost 1.43 times core-v2's temp
# and 1.14 times its time (jobs 68449 to 68455). Without them (dsnn-dfw.287) the joined program holds less temp than
# core-v2's (job 68488), and on the thesis target the exact plan runs at 0.995 of core-v2, so the owner closed dfw.304
# without a carry-layout change (2026-09-27). The check pins "joined, not bigger": it reports the join and refuses
# only a temp above core-v2's. It compiles the program the measurement itself lowers (the capture of
# tools/b11_gpu_ab.py, which job 68488 measured); its own lowering of measured_program held 411,036,104 B on core-v2
# and 444,371,912 B on the head in that job, so it is not what the plans pay. XLA:CPU never merges the stacks, so the
# check runs on the GPU. It is outside the CPU suite's testpaths. An sbatch job on a GPU node runs it:
#   JAX_PLATFORMS=cuda python -m pytest -q -s gpu_tests/rtrl_carry_concat_gpu_test.py
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")

_spec = importlib.util.spec_from_file_location(
    "b11_gpu_ab", Path(__file__).resolve().parents[1] / "tools" / "b11_gpu_ab.py")
AB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AB)
# The measurement's environment for the memobj_rtrl target of the A/B, set before alphagrad reads it.
AB._apply_env(AB.target_setup("memobj_rtrl")[0], [])

import jax  # noqa: E402

if jax.default_backend() != "gpu":
    raise RuntimeError(f"this check runs on a GPU, the backend is {jax.default_backend()}")

# Job 68488 (pgi15-gpu19, RTX PRO 6000 Blackwell Max-Q, no measure compiler options), the exact plan: temp on core-v2
# 432d0ce with the stacks apart, and on the batch-11 head 8dd9423 with them joined. The argument bytes name the
# program the two were measured on.
_CORE_V2_TEMP = 443_425_736
_HEAD_8DD9423_TEMP = 426_862_792
_ARG_BYTES = 716_408
# The widths of the two stacks side by side: 89600 + 16384 as a matrix, 700 + 128 in the stacked layout.
_JOINED = re.compile(r"\[[0-9,]*\b(105984|828)\b[0-9,]*\]")


def test_on_rsnn_shd_rtrl_the_exact_plan_joins_the_carry_stacks_and_is_not_bigger_on_the_gpu():
    lm, env, ev = AB._build("memobj_rtrl", {})
    spy = AB.capture(lm, env, ev, AB._order(lm, env, "reverse"), [], compile_it=True)
    ma = spy.compiled.memory_analysis()
    temp = int(ma.temp_size_in_bytes)
    joined = sorted({m.group(0) for m in _JOINED.finditer(spy.compiled.as_text())})
    print(f"[rtrl-carry-concat-gpu] {jax.devices()[0].device_kind}: exact plan temp {temp} B "
          f"(core-v2 {_CORE_V2_TEMP} B, head 8dd9423 {_HEAD_8DD9423_TEMP} B), "
          f"shapes with a joined width: {joined}")
    assert int(ma.argument_size_in_bytes) == _ARG_BYTES, "not the program job 68488 measured"
    assert temp <= _CORE_V2_TEMP, temp
