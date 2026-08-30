#!/bin/bash
#SBATCH --job-name=apxdiag
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/apxaudit_out/diag_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_QUALITY_METRIC=cosine
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_apxaudit
mkdir -p "$JAX_COMPILATION_CACHE_DIR" /Users/assmuth/dsnn/alphagrad/apxaudit_out
cd ~/dsnn/alphagrad
echo "=== node=$(hostname) $(date) ==="
uv run --no-sync python apxaudit_diag.py \


echo "=== RC=$? $(date) ==="
