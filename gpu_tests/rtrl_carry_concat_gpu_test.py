# The GPU check of dsnn-dfw.304: on RSNN_SHD rtrl the exact plan's program keeps the two carry stacks apart.
# From graphax d5114d3 XLA merged the W-stack and V-stack dots of a hidden state into one dot on a 217 MB
# concatenation f32[512,105984]: the exact plan had 644 MB of temp against 450 MB on core-v2 and ran 1.14 times
# as long (jobs 68449 to 68455, RTX PRO 6000 Blackwell Max-Q). XLA:CPU never merges them, so the check runs
# on the GPU. It is outside the CPU suite's testpaths. An sbatch job on a GPU node runs it:
#   JAX_PLATFORMS=cuda python -m pytest -q -s gpu_tests/rtrl_carry_concat_gpu_test.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import jax  # noqa: E402

import alphagrad.approx.env as envmod  # noqa: E402
import mem_objective_test as M  # noqa: E402
from alphagrad.approx.common.rsnn_shd import measure_args  # noqa: E402

if jax.default_backend() != "gpu":
    raise RuntimeError(f"this check runs on a GPU, the backend is {jax.default_backend()}")

_paired_log_with_the_plan_log = M._paired_log_with_the_plan_log
# Job 68449: the exact plan's temp on core-v2 432d0ce, 450,033,096 B; from d5114d3 on, 643,732,936 B.
_CORE_V2_TEMP = 450_033_096
# The widths of the two stacks side by side: 89600 + 16384 as a matrix, 700 + 128 in the stacked layout.
_JOINED = re.compile(r"\[[0-9,]*\b(105984|828)\b[0-9,]*\]")


def test_on_rsnn_shd_rtrl_the_exact_plan_keeps_the_carry_stacks_apart_on_the_gpu():
    _lm, env, _ev = M._rtrl_env()
    cfg = env.config
    order = sorted(int(v) for v in env.valid_vertices)[::-1]
    fn = envmod.measured_program(cfg, order, env.consts, transforms=[], face_transforms=None, sparse=True)
    args = jax.device_put(tuple(measure_args(cfg, env.args)), jax.devices()[0])
    ex = envmod._compile_measure(jax.jit(fn, keep_unused=True).lower(*args))
    hlo = ex.as_text()
    temp = int(ex.memory_analysis().temp_size_in_bytes)
    joined = sorted({m.group(0) for m in _JOINED.finditer(hlo)})
    print(f"[rtrl-carry-concat-gpu] {jax.devices()[0].device_kind}: exact plan temp {temp} B "
          f"(core-v2 {_CORE_V2_TEMP} B), shapes with a joined width: {joined}")
    assert not joined, joined
    assert temp < 1.05 * _CORE_V2_TEMP, temp
