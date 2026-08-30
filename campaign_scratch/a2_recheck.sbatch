#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a2_recheck_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a2
echo "HEAD=$(git rev-parse --short HEAD)  dirty(src,tests)=$(git status --porcelain -- src tests | wc -l)"
FAIL=0
for T in fidelity_channel_test grad_coverage_test test_reward_scaling; do
  timeout 1200 uv run --no-sync python -m pytest -q tests/$T.py 2>&1 | tail -2
  RC=${PIPESTATUS[0]}
  echo "$T rc=$RC"
  [ $RC -ne 0 ] && FAIL=1
done
echo "A2 RECHECK FAIL=$FAIL"
echo "===== A2 RECHECK DONE ====="
