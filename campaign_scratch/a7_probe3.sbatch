#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=120G
#SBATCH -c 32
#SBATCH -t 6:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_probe3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7tlm
O=/Users/assmuth/dsnn/alphagrad/a7_out
S="--budgets 1,5,all --single-skip 0:0 --single-skip 1:0 --single-skip 2:0 \
--num-data-points 1 --reps-per-point 1 --latency-inner-reps 5 \
--latency-warmup 1 --num-eval-samples 2 --repeat-identity 3"
for EX in NeuralNetwork VmappedNeuralNetwork; do
  echo "##### $EX (coverage MEASURED, guard off)"
  timeout 3000 uv run --no-sync python tools/sparsity_probe.py \
    --example $EX --dataset none --walk-steps 5 $S \
    --out $O/v3_$EX.csv 2>&1 | tail -35
done
echo "##### TransformerLM (coverage MEASURED -- the exact reference is free then)"
ALPHAGRAD_GRAD_COVERAGE=1 timeout 12000 uv run --no-sync python tools/sparsity_probe.py \
  --example TransformerLM --dataset wikitext2 --quality-metric none $S \
  --out $O/v3_TLM_cov.csv 2>&1 | tail -35
echo "##### TransformerLM (coverage OFF -- measures the FALLBACK price)"
ALPHAGRAD_GRAD_COVERAGE=0 timeout 12000 uv run --no-sync python tools/sparsity_probe.py \
  --example TransformerLM --dataset wikitext2 --quality-metric none $S \
  --out $O/v3_TLM_nocov.csv 2>&1 | tail -35
