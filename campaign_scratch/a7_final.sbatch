#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 2:30:00
#SBATCH -J a7final
#SBATCH -o a7_final.out
export PATH=$HOME/.local/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jc_a7
cd $HOME/dsnn/alphagrad
echo "########## BEFORE/AFTER (final)"
JAX_PLATFORMS=cpu GRAPHAX_ALLOW_PARTIAL_ORDER=1 ALPHAGRAD_SKIP_COUNT_OPS=1 \
  uv run --no-sync python a7_beforeafter.py
echo "########## JACOBIAN==GRAD + A4"
JAX_PLATFORMS=cpu GRAPHAX_ALLOW_PARTIAL_ORDER=1 ALPHAGRAD_SKIP_COUNT_OPS=1 \
  uv run --no-sync python -m pytest \
  src/alphagrad/approx/tests/test_jacobian_equals_grad.py \
  src/alphagrad/approx/tests/test_seed_vertices_dropped.py \
  -q --tb=short -p no:cacheprovider -s
echo "########## SMOKE (re-run on the final tree)"
SMOKE_OUT=$HOME/dsnn/alphagrad/smoke_a7 SMOKE_TIMEOUT=3000 tools/smoke.sh Helmholtz NeuralNetwork
echo "smoke rc=$?"
