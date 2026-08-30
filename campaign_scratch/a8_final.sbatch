#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_final_%j.log
export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu ALPHAGRAD_SKIP_COUNT_OPS=1 ALPHAGRAD_SKIP_COST_ANALYSIS=1 GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
cd /Users/assmuth/dsnn/alphagrad
echo "########## whole-suite, patched, after the vacuity-bound fix"
timeout 7200 uv run --no-sync python -m pytest tests -q -p no:cacheprovider \
  --ignore=tests/transforms/symmetry_test.py 2>&1 | grep -E "^FAILED|^ERROR|passed|failed" | tail -20
echo "########## A8 FINAL DONE"
