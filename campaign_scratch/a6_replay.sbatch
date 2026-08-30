#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -J a6_replay
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_replay_%j.log
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
unset ALPHAGRAD_QUALITY_METRIC
unset ALPHAGRAD_PLAN_LOG
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_APPEND_ONLY_JAXPR=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
# THE APPARATUS MUST MATCH THE RECORDING. The recorded rows are the
# frozen-gradient guards SENTINEL; replaying with the guard OFF measures the
# plan normally and the comparison is meaningless.
export ALPHAGRAD_REJECT_FROZEN_GRADS=1
export ALPHAGRAD_GRAD_COVERAGE=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a6
echo "########## structure + REPLAY of the recorded losers ##########"
uv run --no-sync python a6_live_check.py \
  --log /Users/assmuth/dsnn/alphagrad/a6_out/ray_plan_log.jsonl \
  --example NeuralNetwork --seed 42 --replay 4 2>&1 \
  | grep -v nanobind | grep -v "leaked "
echo "CHECK_RC=${PIPESTATUS[0]}"
echo "########## A6 REPLAY DONE ##########"
