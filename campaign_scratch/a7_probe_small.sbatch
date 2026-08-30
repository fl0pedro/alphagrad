#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_probe_small_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7
mkdir -p /Users/assmuth/dsnn/alphagrad/a7_out
for EX in Helmholtz NeuralNetwork; do
  echo "##### $EX"
  timeout 3000 uv run --no-sync python tools/sparsity_probe.py \
    --example $EX --dataset none --budgets 1,5,all \
    --single-skip 0:0 --single-skip 1:0 --single-skip 2:0 \
    --walk-steps 5 --num-data-points 1 --reps-per-point 1 \
    --latency-inner-reps 1 --latency-warmup 1 --num-eval-samples 2 \
    --out /Users/assmuth/dsnn/alphagrad/a7_out/rows_$EX.csv 2>&1 | tail -70
  echo "rc=$?"
done
