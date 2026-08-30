#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_before_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

echo "===== XLA_FLAGS validation ====="
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo
echo "===== BEFORE: policy regression gate (must PASS on the unpatched build) ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py
echo "gate rc=$?"

echo
echo "===== BEFORE: gradient-to-palimpsa probe ====="
JAX_PLATFORMS=cpu uv run --no-sync python lean_probe.py
echo "probe rc=$?"

echo
echo "===== BEFORE: DAG-agnosticism audit ====="
JAX_PLATFORMS=cpu uv run --no-sync python lean_audit.py
echo "audit rc=$?"

echo "===== LEAN BEFORE DONE ====="
