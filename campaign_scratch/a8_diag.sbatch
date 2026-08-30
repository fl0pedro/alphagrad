#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_diag_%j.log
export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu ALPHAGRAD_SKIP_COUNT_OPS=1 GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
cd /Users/assmuth/dsnn/alphagrad
uv run --no-sync python -m pytest tests/live_vertex_mask_test.py -q -p no:cacheprovider 2>&1 | tail -60
