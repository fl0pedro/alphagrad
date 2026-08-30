#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_out/order_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a6
export ALPHAGRAD_SKIP_COUNT_OPS=1
echo "===== PAIRED (edge_mem first) ====="
uv run --no-sync python -m pytest tests/edge_mem_test.py tests/endpoint_read_test.py -q --tb=long 2>&1 | tail -120
echo "rc=${PIPESTATUS[0]}"
echo "===== DONE ====="
