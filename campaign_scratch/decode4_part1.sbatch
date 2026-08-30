#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode4_part1_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode4_out
mkdir -p $R

echo "===== XLA_FLAGS validation ====="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo "===== (a)(b) constants ====="
JAX_PLATFORMS=cpu uv run --no-sync python -c "
from alphagrad.approx.env import LEGACY_STREAM_TOKENS, MAX_DELTA_TOKENS, MAX_BASE_TOKENS
print('LEGACY_STREAM_TOKENS', LEGACY_STREAM_TOKENS)
print('MAX_DELTA_TOKENS    ', MAX_DELTA_TOKENS)
print('MAX_BASE_TOKENS     ', MAX_BASE_TOKENS)
" || exit 1

echo "===== (a) ALPHAGRAD_MAX_TOKENS must be a HARD ERROR ====="
ALPHAGRAD_MAX_TOKENS=8192 JAX_PLATFORMS=cpu uv run --no-sync python -c "import alphagrad.approx.env" 2>&1 | tail -4
echo "  (expect a RuntimeError above)"

echo "===== (c) delta overflow is LOUD ====="
JAX_PLATFORMS=cpu uv run --no-sync python -c "
from alphagrad.approx import env
try:
    env._record_delta_truncation(env.MAX_DELTA_TOKENS + 1)
    print('FAIL: no raise')
except ValueError as e:
    print('RAISE OK:', str(e)[:120])
"
ALPHAGRAD_DELTA_OVERFLOW=clip JAX_PLATFORMS=cpu uv run --no-sync python -c "
from alphagrad.approx import env
env._record_delta_truncation(env.MAX_DELTA_TOKENS + 7)
print('clip mode returned (stderr line above is the loud part)')
"

echo "===== touched tests ====="
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -x -q \
  tests/per_face_apply_test.py 2>&1 | tail -25

echo "===== (d) encode_extend prefix-proportionality (GPU) ====="
JAX_COMPILATION_CACHE_DIR=$S/.jc_d4 CUDA_VISIBLE_DEVICES=0 \
  uv run --no-sync python decode4_extend_bench.py $R/extend_bench.json 2>&1 | tee $R/extend_bench.log

echo "===== (d) end-to-end nn256 rollout, 1024 vs 32768 ====="
run_ab () {   # budget, extend_chunk, tag
  export ALPHAGRAD_MAX_DELTA_TOKENS=$1
  export ALPHAGRAD_EXTEND_CHUNK=$2
  export ALPHAGRAD_DELTA_OVERFLOW=clip
  export ALPHAGRAD_POLICY=palimpsa
  export ALPHAGRAD_INCREMENTAL_TOKENS=1
  export ALPHAGRAD_NN_HIDDEN=256
  export ALPHAGRAD_SKIP_COST_ANALYSIS=1
  export ALPHAGRAD_MAX_EQNS=512
  export ALPHAGRAD_EXTEND_UNROLL=32
  export ALPHAGRAD_PROFILE=1
  export GRAPHAX_ALLOW_PARTIAL_ORDER=1
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1; sleep 2; done ) > $R/gpumem_$3.txt &
  MON=$!
  T0=$(date +%s)
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d4ab CUDA_VISIBLE_DEVICES=1 timeout 2400 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example VmappedNeuralNetwork --dataset mnist --hidden-dim 256 \
    --variant full --face-actions --incremental-encode --measure-grad \
    --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
    --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
    --advantage-norm popart --num-envs 4 --minibatches 4 \
    --episodes 2 --wandb disabled --name d4-$3 > $R/ab_$3.log 2>&1
  RC=$?
  T1=$(date +%s)
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$3.txt | tail -1)
  echo "  AB $3: budget=$1 extend_chunk=$2 rc=$RC wall=$((T1-T0))s gpu_peak_MiB=$PEAK"
  grep -E "prof/(encode|rollout|update)|episode .* wall|truncat" $R/ab_$3.log | tail -8
}
run_ab 1024  0   b1024_c0
run_ab 32768 0   b32768_c0
run_ab 1024  256 b1024_c256
run_ab 32768 256 b32768_c256
echo "===== PART 1 DONE ====="
