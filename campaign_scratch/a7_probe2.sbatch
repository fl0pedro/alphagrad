#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 5:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_probe2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7
O=/Users/assmuth/dsnn/alphagrad/a7_out
mkdir -p $O
SMALL="--budgets 1,5,all --single-skip 0:0 --single-skip 1:0 --single-skip 2:0 \
--walk-steps 5 --num-data-points 1 --reps-per-point 1 --latency-inner-reps 5 \
--latency-warmup 1 --num-eval-samples 2"
echo "##### A/B: Helmholtz with the CHANNEL OFF (is the TypeError mine?)"
ALPHAGRAD_SPARSITY=0 timeout 900 uv run --no-sync python tools/sparsity_probe.py \
  --example Helmholtz --dataset none $SMALL --out $O/rows_Helmholtz_off.csv 2>&1 | grep -E "MEASUREMENT FAILED|a7\]|SUMMARY|n_rows" | head -25
echo "##### A/B: Helmholtz with the CHANNEL OFF and COVERAGE OFF"
ALPHAGRAD_SPARSITY=0 ALPHAGRAD_GRAD_COVERAGE=0 timeout 900 uv run --no-sync python tools/sparsity_probe.py \
  --example Helmholtz --dataset none $SMALL --out $O/rows_Helmholtz_nocov.csv 2>&1 | grep -E "MEASUREMENT FAILED|a7\]|SUMMARY|n_rows" | head -25
for EX in NeuralNetwork VmappedNeuralNetwork; do
  echo "##### $EX"
  timeout 3000 uv run --no-sync python tools/sparsity_probe.py \
    --example $EX --dataset none $SMALL --out $O/rows_$EX.csv 2>&1 | tail -45
done
