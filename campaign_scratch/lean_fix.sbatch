#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_fix_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

echo "===== AZ gradflow with the fixture repaired ====="
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q tests/test_sampled_az_gradflow.py 2>&1 | tail -8
echo
echo "===== gate still bit-identical after the test edit (no src change, but prove it) ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py 2>&1 | grep -E "^\[gate\]"
echo "===== LEAN FIX DONE ====="
