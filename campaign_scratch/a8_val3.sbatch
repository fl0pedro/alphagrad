#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_val3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
echo "########## TLM3 alone"
uv run --no-sync python -m pytest "src/alphagrad/approx/tests/test_jacobian_equals_grad.py::test_jacve_equals_jax_grad[TransformerLM3]" -q -s -p no:cacheprovider 2>&1 | tail -12
echo "########## repo tests/ (skip the duplicate-basename collector)"
timeout 10800 uv run --no-sync python -m pytest tests -q -p no:cacheprovider \
  --ignore=tests/transforms/symmetry_test.py 2>&1 | tail -45
echo "########## A8 VAL3 DONE"
