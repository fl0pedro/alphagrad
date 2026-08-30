#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 0:40:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a3_test_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a3
export ALPHAGRAD_SKIP_COUNT_OPS=1
echo "===== A3 unit test ====="
uv run --no-sync python -m pytest tests/walk_heldout_test.py -q 2>&1 | tail -40
echo "a3 rc=${PIPESTATUS[0]}"
echo "===== A4 regression (grad coverage / walk argnums) ====="
uv run --no-sync python -m pytest tests/grad_coverage_test.py -q 2>&1 | tail -20
echo "===== DONE ====="
