#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode4_ab_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode4_out

run_ab () {   # gpu, budget, extend_chunk, tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; sleep 2; done ) > $R/gpumem_$4.txt &
  MON=$!
  T0=$(date +%s)
  ALPHAGRAD_MAX_DELTA_TOKENS=$2 ALPHAGRAD_EXTEND_CHUNK=$3 \
  ALPHAGRAD_DELTA_OVERFLOW=clip ALPHAGRAD_POLICY=palimpsa \
  ALPHAGRAD_INCREMENTAL_TOKENS=1 ALPHAGRAD_NN_HIDDEN=256 \
  ALPHAGRAD_SKIP_COST_ANALYSIS=1 ALPHAGRAD_MAX_EQNS=512 \
  ALPHAGRAD_EXTEND_UNROLL=32 ALPHAGRAD_PROFILE=1 \
  GRAPHAX_ALLOW_PARTIAL_ORDER=1 \
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d4ab$3 CUDA_VISIBLE_DEVICES=$1 timeout 3000 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example VmappedNeuralNetwork --dataset mnist --hidden-dim 256 \
    --vocab-size 512 --unified-head --unified-face-head \
    --variant full --face-actions --incremental-encode --measure-grad \
    --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
    --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
    --advantage-norm popart --num-envs 4 --minibatches 4 \
    --episodes 2 --wandb disabled --name d4-$4 > $R/ab_$4.log 2>&1
  RC=$?
  T1=$(date +%s)
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$4.txt | tail -1)
  echo "  AB $4: budget=$2 extend_chunk=$3 rc=$RC wall=$((T1-T0))s gpu_peak_MiB=$PEAK"
  tail -3 $R/ab_$4.log
  grep -E "prof/(encode|rollout|update)|truncat|WARNING: ALPHAGRAD_EXTEND_CHUNK" $R/ab_$4.log | tail -6
}
( run_ab 0 1024  0   b1024_c0   ; run_ab 0 32768 0   b32768_c0 ) &
( run_ab 1 1024  256 b1024_c256 ; run_ab 1 32768 256 b32768_c256 ) &
wait
echo "===== AB DONE ====="
