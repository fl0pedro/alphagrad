#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_tests_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
echo "########## test_jacobian_equals_grad"
uv run --no-sync python -m pytest src/alphagrad/approx/tests/test_jacobian_equals_grad.py -q -s -p no:cacheprovider 2>&1 | tail -80
echo "########## test_seed_vertices_dropped"
uv run --no-sync python -m pytest src/alphagrad/approx/tests/test_seed_vertices_dropped.py -q -p no:cacheprovider 2>&1 | tail -30
echo "########## approx/tests (whole dir)"
uv run --no-sync python -m pytest src/alphagrad/approx/tests -q -p no:cacheprovider 2>&1 | tail -40
echo "########## A8 TESTS DONE"
