# The Diag temp check of tests/mem_objective_test.py on the GPU (owner ruling 2026-09-26, Q16 a: checks of approximated
# plans run on the GPU). It is outside the CPU suite's testpaths. An sbatch job on a GPU node runs it:
#   JAX_PLATFORMS=cuda python -m pytest -q -s gpu_tests/mem_objective_gpu_test.py
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import jax  # noqa: E402

import mem_objective_test as M  # noqa: E402

if jax.default_backend() != "gpu":
    raise RuntimeError(f"this check runs on a GPU, the backend is {jax.default_backend()}")

_paired_log_with_the_plan_log = M._paired_log_with_the_plan_log


def test_on_rsnn_shd_a_diag_on_the_carried_face_cuts_the_temp_term_on_the_gpu():
    lm, env, eval_samples = M._rtrl_env()
    order = sorted(int(v) for v in env.valid_vertices)[::-1]
    exact = M._measure_rsnn(lm, env, eval_samples, order, [], "exact")
    diag = M._measure_rsnn(lm, env, eval_samples, order,
                           M._diag_on_the_carried_face(lm, env, order), "diag")
    assert exact["ref_temp_bytes"] == diag["ref_temp_bytes"]
    assert diag["mem_ratios"]["out"] == exact["mem_ratios"]["out"] == 1.0
    assert diag["mem_ratios"]["args"] == exact["mem_ratios"]["args"] == 1.0
    # Job 68449 on the GPU: 450,033,096 B of temp exact (core-v2) against 3,570,888 B Diag at every graphax commit.
    assert diag["mem_ratios"]["temp"] < exact["mem_ratios"]["temp"] / 20.0
    print(f"[mem-objective-gpu] RSNN_SHD rtrl temp: exact {exact['mem_temp_bytes']:.0f} B, "
          f"Diag {diag['mem_temp_bytes']:.0f} B, reference {exact['ref_temp_bytes']:.0f} B")
