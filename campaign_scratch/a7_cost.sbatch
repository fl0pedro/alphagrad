#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_cost_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a7_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
unset ALPHAGRAD_SPARSITY ALPHAGRAD_SPARSITY_WEIGHT
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_HEALTH_EPISODES=99
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7cost
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--grad-window 0 --example NeuralNetwork --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--advantage-norm popart --episodes 4 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"
echo "########## ARM A: channel OFF (the reference wall)"
/usr/bin/time -f "ARM_A_WALL %e s" uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --name a7_cost_off > $R/cost_off.log 2>&1
tail -3 $R/cost_off.log
grep -E "sec/ep" $R/cost_off.log | tail -4
echo "########## ARM B: --sparsity-log (channel LOGGED, guard ON by default)"
/usr/bin/time -f "ARM_B_WALL %e s" uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --sparsity-log --name a7_cost_on > $R/cost_on.log 2>&1
grep -E "sec/ep|\[sparsity ep|\[cfg\] sparsity" $R/cost_on.log | tail -12
echo "########## ARM C: --sparsity-weight 1 (TRAINED; guard on)"
uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --sparsity-weight 1 --name a7_cost_train > $R/cost_train.log 2>&1
grep -E "sec/ep|\[sparsity ep|\[cfg\] sparsity|heads=" $R/cost_train.log | tail -12
echo "########## ARM D: --sparsity-weight 1 --no-reject-frozen-grads (MUST REFUSE)"
uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --sparsity-weight 1 --no-reject-frozen-grads --name a7_cost_bad > $R/cost_bad.log 2>&1
echo "rc=$? (non-zero expected)"
grep -E "ValueError|reject-frozen-grads" $R/cost_bad.log | tail -5
