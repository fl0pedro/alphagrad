#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_nan_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a6_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a6
BASE="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --grad-window 0 --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"
for EX in Helmholtz NeuralNetwork; do
  echo "########## $EX  (ALPHAGRAD_HEALTH_EPISODES=99) ##########"
  ALPHAGRAD_HEALTH_EPISODES=99 timeout 3300 uv run --no-sync \
    python src/alphagrad/approx/ppo.py $BASE --example $EX --name a6_$EX \
    > $R/a6_$EX.log 2>&1
  echo "$EX rc=$?"
  echo "--- popart-init line ---"; grep -n "popart-init" $R/a6_$EX.log | head
  echo "--- ALL health lines ---"; grep -n "\[health" $R/a6_$EX.log | head -40
done
echo "===== A6 DONE ====="
