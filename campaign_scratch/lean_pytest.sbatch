#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_pytest_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

TAG=${LEAN_TAG:-run}
# ONE PROCESS PER FILE: the whole-suite single process SIGKILLs, pre-existing.
FILES="
tests/carry_stream_base_attribution_test.py
tests/vertex_memory_test.py
tests/incremental_encode_test.py
tests/face_bucket_test.py
tests/delta_buffer_equivalence_test.py
tests/test_sampled_az_gradflow.py
tests/policy_regression_gate_test.py
tests/face_path_policy_test.py
tests/test_face_head94.py
tests/test_face_policy.py
tests/append_only_tokens_test.py
tests/masked_extend_equivalence_test.py
tests/stream_prefix_property_test.py
tests/face_live_state_test.py
tests/live_vertex_mask_test.py
tests/head_skipping_test.py
tests/alignment_test.py
tests/palimpsa_base_grad_test.py
"
echo "===== PYTEST BASELINE ($TAG) ====="
for f in $FILES; do
  if [ ! -f "$f" ]; then echo "SKIP(missing) $f"; continue; fi
  out=$(JAX_PLATFORMS=cpu timeout 900 uv run --no-sync python -m pytest -q "$f" 2>&1)
  rc=$?
  line=$(echo "$out" | tail -3 | tr '\n' ' ')
  echo "[$TAG] rc=$rc  $f  :: $line"
  echo "$out" > "lean_pytest_${TAG}_$(basename $f .py).txt"
done
echo "===== PYTEST $TAG DONE ====="
