#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 1:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_out/batch_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a6
export ALPHAGRAD_SKIP_COUNT_OPS=1
echo "===== A: the five ratio gates in ONE pytest process ====="
uv run --no-sync python -m pytest \
  tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant \
  tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes \
  tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks \
  tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical \
  tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay \
  -q --tb=long 2>&1 | tail -100
echo "A rc=${PIPESTATUS[0]}"
echo "===== B: forced pre-import of env, then endpoint_read alone ====="
uv run --no-sync python -c "
import alphagrad.approx.env as E
print(\"pre-import MAX_DELTA_TOKENS =\", E.MAX_DELTA_TOKENS)
import pytest, sys
sys.exit(pytest.main([\"tests/endpoint_read_test.py\",\"-q\",\"--tb=line\"]))
" 2>&1 | tail -40
echo "B rc=${PIPESTATUS[0]}"
echo "===== DONE ====="
