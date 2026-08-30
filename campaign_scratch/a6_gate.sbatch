#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_gate_%j.log
cd /Users/assmuth/dsnn/alphagrad
export SMOKE_OUT=/Users/assmuth/dsnn/alphagrad/a6_out/gate
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a6
echo "########## tools/smoke.sh (patched ppo.py) ##########"
tools/smoke.sh
echo "SMOKE_GATE_RC=$?"
echo "########## ratio-1 unit tests ##########"
export JAX_PLATFORMS=cpu
uv run --no-sync python -m pytest -q \
  tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant \
  tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes \
  tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks \
  tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay \
  tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical \
  2>&1 | tail -25
echo "PYTEST_RC=$?"
echo "===== A6 GATE DONE ====="
