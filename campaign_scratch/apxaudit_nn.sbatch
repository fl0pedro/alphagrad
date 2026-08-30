#!/bin/bash
#SBATCH --job-name=apxNN
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/apxaudit_out/nn_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_QUALITY_METRIC=cosine
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_apxaudit
mkdir -p "$JAX_COMPILATION_CACHE_DIR" /Users/assmuth/dsnn/alphagrad/apxaudit_out
cd ~/dsnn/alphagrad

echo "=== node=$(hostname) $(date) ==="
echo "### PHASE 1: correctness on CPU (NeuralNetwork) ###"
JAX_PLATFORMS=cpu uv run --no-sync python apxaudit_matrix2.py \
    --target NeuralNetwork --phase correctness --npos 4 \
    --out apxaudit_out/nn_correct.json
echo "### rc=$? ###"

echo "### PHASE 2: cost on GPU (NeuralNetwork) ###"
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true" \
uv run --no-sync python apxaudit_matrix2.py \
    --target NeuralNetwork --phase cost --npos 4 --pairs 3 \
    --out apxaudit_out/nn_cost.json
echo "### rc=$? ###"
echo "=== DONE $(date) ==="
