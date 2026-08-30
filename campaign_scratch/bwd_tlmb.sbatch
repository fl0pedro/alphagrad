#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu17
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/bwd_tlmb_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/bwd_out_b
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256          # NEVER 0 (flat 29x trap)
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_HEALTH_EPISODES=40
export JAX_COMPILATION_CACHE_DIR=$S/.jc_bwdb
# DELTA_OVERFLOW deliberately LEFT AT THE DEFAULT (raise): a 4096 window that
# is too small must be loud, not clipped away.

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

NN="--example VmappedNeuralNetwork --dataset mnist --hidden-dim 256"
TLM="--example TransformerLM --dataset wikitext2"

echo "########## TLM arm B ONLY: FIX 1 without remat (separates the two fixes) ##########"
run () {   # $1 tag  $2 remat  $3 window  $4 K  $5 episodes  $6.. target args
  local TAG=$1 RM=$2 W=$3 K=$4 EP=$5; shift 5
  : > $R/gpumem_$TAG.txt
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0
      sleep 2
    done ) > $R/gpumem_$TAG.txt 2>/dev/null &
  local MON=$!
  local T0=$(date +%s)
  ALPHAGRAD_CARRY_HEADS_REMAT=$RM ALPHAGRAD_MAX_DELTA_TOKENS=$W \
  ALPHAGRAD_NN_HIDDEN=256 timeout 5400 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    "$@" $COMMON --episodes $EP --grad-window $K --name bwd-$TAG \
    > $R/$TAG.log 2>&1
  local RC=$?
  local T1=$(date +%s)
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  local PEAK=$(sort -n $R/gpumem_$TAG.txt | tail -1)
  local SECS=$(grep -oE "sec/ep=[0-9.]+" $R/$TAG.log | tail -3 | tr '\n' ' ')
  echo "RESULT $TAG remat=$RM window=$W K=$K rc=$RC wall=$((T1-T0))s peakMiB=$PEAK sec/ep(last3)= $SECS"
  grep -iE "RESOURCE_EXHAUSTED|out of memory|Killed|token DELTA truncated" $R/$TAG.log | head -3
  if [ $RC -ne 0 ]; then echo "  --- tail:"; tail -20 $R/$TAG.log; fi
}
for K in 1 4 16; do
  run tlmB_k$K 0 4096 $K 6 $TLM
done
echo "===== BWD TLMB DONE ====="
