#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_out/verify_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a6
export ALPHAGRAD_SKIP_COUNT_OPS=1
echo "##### 1. standalone endpoint_read (must be 7 passed)"
uv run --no-sync python -m pytest tests/endpoint_read_test.py -q 2>&1 | tail -6
echo "1 rc=${PIPESTATUS[0]}"
echo "##### 2. standalone edge_mem + face_bucket (the other two patched files)"
uv run --no-sync python -m pytest tests/edge_mem_test.py -q 2>&1 | tail -5
echo "edge_mem rc=${PIPESTATUS[0]}"
uv run --no-sync python -m pytest tests/face_bucket_test.py -q 2>&1 | tail -5
echo "face_bucket rc=${PIPESTATUS[0]}"
echo "##### 3. THE OLD FAILING INVOCATION: five gates in ONE process"
uv run --no-sync python -m pytest \
  tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant \
  tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes \
  tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks \
  tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical \
  tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay \
  -q -rs --tb=short 2>&1 | tail -18
echo "3 rc=${PIPESTATUS[0]}"
echo "##### 4. tools/ratio_gates.sh (the CI-ish form)"
PY="uv run --no-sync python" bash tools/ratio_gates.sh 2>&1 | tail -60
echo "4 rc=${PIPESTATUS[0]}"
echo "##### DONE"
