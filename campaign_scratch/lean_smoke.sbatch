#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH -c 32
#SBATCH -t 8:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/lean_out
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256          # NEVER 0 (flat 29x trap)
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_lean

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

# NOTE: the background subshell MUST have its stdout redirected away, or
# `PID=$(poll ...)` never returns -- command substitution waits for every
# writer of the pipe to close it, and a `while true` background job never
# does. That cost one 26-minute hung job.
poll () {  # $1 = outfile; sets POLL_PID
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$1"
      sleep 5
    done ) >/dev/null 2>&1 &
  POLL_PID=$!
}

echo "##### S1: Helmholtz 10 episodes -- does the LEAN model TRAIN? #####"
poll $R/gpumem_helm.txt; PID=$POLL_PID
ALPHAGRAD_HEALTH_EPISODES=10 timeout 3600 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --episodes 10 --name lean_h \
  > $R/smoke_helm.log 2>&1
RC1=$?
echo "rc=$RC1"; kill $PID 2>/dev/null
grep -E "base token stream|policy params|Traceback|Error" $R/smoke_helm.log | grep -v live-faces | head -10
grep -E "health ep" $R/smoke_helm.log | grep -v live-faces | head -12
echo "peak MiB: $(sort -n $R/gpumem_helm.txt | tail -1)"
if [ $RC1 -ne 0 ]; then
  echo "S1 FAILED -- aborting before burning the node on the K sweep"
  tail -60 $R/smoke_helm.log
  exit 1
fi

echo
echo "##### S2: --grad-window K SWEEP on TransformerLM (V=95), 6 episodes each"
echo "##### The DEFAULT IS NOT CHANGED. This is the measurement the owner"
echo "##### asked for so THEY can choose K."
for K in 1 2 4 8 16; do
  echo "----- K=$K  $(date +%H:%M:%S)"
  : > $R/gpumem_k$K.txt
  poll $R/gpumem_k$K.txt; PID=$POLL_PID
  ALPHAGRAD_HEALTH_EPISODES=6 timeout 5400 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example TransformerLM --dataset wikitext2 $COMMON \
    --episodes 6 --grad-window $K --name lean_k$K \
    > $R/k$K.log 2>&1
  rc=$?
  kill $PID 2>/dev/null
  echo "  rc=$rc  peakMiB=$(sort -n $R/gpumem_k$K.txt | tail -1)"
  grep -E "base token stream" $R/k$K.log | head -1
  grep -E "health ep" $R/k$K.log | grep -v live-faces | tail -3
  grep -iE "RESOURCE_EXHAUSTED|out of memory|OOM|Killed" $R/k$K.log | head -3
  if [ $rc -ne 0 ]; then
    echo "  K=$K FAILED -- this is the affordability ceiling"
    tail -25 $R/k$K.log
    break
  fi
done

echo
echo "##### SUMMARY: K vs cost #####"
for K in 1 2 4 8 16; do
  [ -f $R/k$K.log ] || continue
  SEC=$(grep -oE "sec/ep=[0-9.]+" $R/k$K.log | tail -3 | tr '\n' ' ')
  echo "K=$K  peakMiB=$(sort -n $R/gpumem_k$K.txt | tail -1)  sec/ep(last3)= $SEC"
done
echo "===== LEAN SMOKE DONE ====="
