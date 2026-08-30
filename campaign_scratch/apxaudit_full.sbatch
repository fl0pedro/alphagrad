#!/bin/bash
#SBATCH --job-name=apxFULL
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/apxaudit_out/full_%j.out
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
GPUFLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"

echo "=== node=$(hostname) $(date) ==="

echo "### 1. NN correctness (CPU, 6 SPREAD positions) ###"
JAX_PLATFORMS=cpu uv run --no-sync python apxaudit_matrix2.py \
    --target NeuralNetwork --phase correctness --npos 6 \
    --out apxaudit_out/nn_correct2.json
echo "### rc=$? ###"

echo "### 2. NN cost (GPU, paired x5) ###"
XLA_FLAGS="$GPUFLAGS" uv run --no-sync python apxaudit_matrix2.py \
    --target NeuralNetwork --phase cost --npos 6 --pairs 5 \
    --out apxaudit_out/nn_cost2.json
echo "### rc=$? ###"

echo "### 3. TransformerLM correctness (CPU, 4 SPREAD positions) ###"
JAX_PLATFORMS=cpu uv run --no-sync python apxaudit_matrix2.py \
    --target TransformerLM --phase correctness --npos 4 \
    --out apxaudit_out/tlm_correct.json
echo "### rc=$? ###"

echo "### 4. TransformerLM cost (GPU, paired x3) ###"
XLA_FLAGS="$GPUFLAGS" uv run --no-sync python apxaudit_matrix2.py \
    --target TransformerLM --phase cost --npos 3 --pairs 3 \
    --out apxaudit_out/tlm_cost.json
echo "### rc=$? ###"
echo "=== DONE $(date) ==="
