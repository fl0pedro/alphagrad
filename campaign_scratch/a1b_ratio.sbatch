#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_ratio_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a1b_out; mkdir -p $R
export JAX_PLATFORMS=cpu
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp --lambda-cmp 1 --advantage-norm popart --num-envs 2 \
--minibatches 1 --latency-inner-reps 5 --vocab-size 512 \
--no-reject-frozen-grads --wandb disabled"

for ARM in on off; do
  EXTRA=""; [ $ARM = on ] && EXTRA="--per-face-masks"
  echo "===== RATIO ARM=$ARM ====="
  ALPHAGRAD_HEALTH_EPISODES=1 timeout 1800 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example NeuralNetwork --dataset none $COMMON $EXTRA \
    --episodes 2 --name a1b_r_$ARM > $R/ratio_$ARM.log 2>&1
  echo "rc=$?"
  grep -E "per-face-masks" $R/ratio_$ARM.log | head -2
  grep -oE "\[health ep[0-9]+\] ppo=[^ ]+ value=[^ ]+ ent=[^ ]+ ratio/max_log=[^ ]+ kl/approx=[^ ]+" $R/ratio_$ARM.log | head -4
  grep -E "Traceback|Error:" $R/ratio_$ARM.log | head -5
done
echo "===== RATIO DONE ====="
