#!/bin/bash
#SBATCH -J perf2r3
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/perf2_r3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/perf2_out
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64

echo "############ TESTS (corrected) ############"
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q \
  tests/face_live_state_test.py tests/face_prefix_cache_test.py \
  tests/pool_face_passthrough_test.py tests/per_face_apply_test.py \
  tests/face_actions_env_test.py tests/live_face_prefix_test.py \
  tests/stream_prefix_property_test.py tests/incremental_encode_test.py \
  > $R/tests3.log 2>&1
echo "pytest rc=$?"; tail -25 $R/tests3.log
echo "############ GATE ############"
uv run --no-sync python tests/policy_regression_gate.py > $R/gate3.log 2>&1
echo "gate rc=$?"; grep -iE "PASS|FAIL|mismatch" $R/gate3.log | head -20
echo "############ PERF2 ROUND 3 DONE ############"
