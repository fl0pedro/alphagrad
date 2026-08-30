#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a5_tests2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a5_out
mkdir -p $R
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a5
for T in static_objective_test gae_value_encoding_test; do
  echo "----- $T -----"
  timeout 1800 uv run --no-sync python -m pytest -q tests/$T.py > $R/test_$T.log 2>&1
  echo "$T rc=$?"
  tail -4 $R/test_$T.log
done
echo "===== A5 TESTS2 DONE ====="
