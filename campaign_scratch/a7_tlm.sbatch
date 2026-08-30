#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=120G
#SBATCH -c 32
#SBATCH -t 8:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_tlm_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7tlm
O=/Users/assmuth/dsnn/alphagrad/a7_out
mkdir -p $O
# COVERAGE OFF on TLM: its exact reference is the 9.70 s / 4.24 GB execution,
# and the sparsity denominator needs only the TRACE. This is also the
# configuration that MEASURES the fallback price -- with no free reference,
# every plan pays an abstract walk, and `fallback_traces` counts them.
export ALPHAGRAD_GRAD_COVERAGE=0
export ALPHAGRAD_REJECT_FROZEN_GRADS=0
COMMON="--budgets 1,5,all --single-skip 0:0 --single-skip 1:0 \
--quality-metric none --num-data-points 1 --reps-per-point 1 \
--latency-inner-reps 5 --latency-warmup 1 --num-eval-samples 1 \
--repeat-identity 2"
echo "########## TransformerLM small (1 layer) -- feasibility first"
timeout 7000 uv run --no-sync python tools/sparsity_probe.py \
  --example TransformerLM --dataset none --num-layers 1 --hidden-dim 128 \
  --embd-dim 64 --num-heads 2 --vocab-size 512 $COMMON \
  --out $O/rows_TLM_small.csv 2>&1 | tail -40
echo "rc=$?"
echo "########## TransformerLM flagship (3 layers, hidden 256, wikitext2)"
timeout 14000 uv run --no-sync python tools/sparsity_probe.py \
  --example TransformerLM --dataset wikitext2 --num-layers 3 --hidden-dim 256 \
  --embd-dim 128 --num-heads 4 --vocab-size 512 $COMMON \
  --out $O/rows_TLM.csv 2>&1 | tail -40
echo "rc=$?"
