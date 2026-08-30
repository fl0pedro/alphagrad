#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a2_tests_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a2_out
mkdir -p $R
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a2
FAIL=0
for T in fidelity_channel_test test_reward_scaling grad_coverage_test credit_fix_test lagrangian_reward_test static_objective_test gae_value_encoding_test; do
  echo "----- $T -----"
  timeout 2400 uv run --no-sync python -m pytest -q tests/$T.py > $R/test_$T.log 2>&1
  RC=$?
  echo "$T rc=$RC"
  [ $RC -ne 0 ] && FAIL=1
  tail -6 $R/test_$T.log
done
echo "A2 TESTS FAIL=$FAIL"
echo "===== A2 TESTS DONE ====="
