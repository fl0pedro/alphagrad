#!/bin/bash
#SBATCH -J tlm3gaz
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/tlm3_gaz_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/tlm3_out
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
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64

echo "chunk = $ALPHAGRAD_EXTEND_CHUNK (MUST NOT be 0)"

# az_gumbel's CLI differs from ppo's: --wandb is store_true and the name flag
# is --wandb-name. Passing `--wandb disabled --name X` is an argparse ERROR.
run_gaz () {  # $1 = task, $2 = gpu, $3 = tag, $4 = measurements
  T0=$(date +%s)
  JAX_COMPILATION_CACHE_DIR=$S/.jc_tlm3gaz CUDA_VISIBLE_DEVICES=$2 timeout 7200 \
    stdbuf -oL -eL uv run --no-sync python -m alphagrad.approx.az_gumbel \
    --task $1 --dataset wikitext2 --seed 7 \
    --total-measurements $4 --popart-init-episodes 0 \
    --latency-inner-reps 50 --out $R/gaz_$3.json 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/gazb_$3.log
  T1=$(date +%s)
  echo "  GAZ[$3] task=$1 measurements=$4 total_wall=$((T1-T0))s"
  grep -iE "ALPHAGRAD_EXTEND_CHUNK" $R/gazb_$3.log | head -2
  echo "  --- per-episode / per-measurement lines ---"
  grep -oE "^[0-9]+ .*(ep=[0-9]+|episode [0-9]+|meas[a-z]* [0-9]+).*" $R/gazb_$3.log | tail -12
  echo "  --- tail ---"
  tail -20 $R/gazb_$3.log
}

echo "##### GAZ (az_gumbel) affordability #####"
echo "===== 2-block reference ====="
run_gaz TransformerLM  0 t2 6
echo "===== 3-block target ====="
run_gaz TransformerLM3 1 t3 6
echo "##### GAZ DONE #####"
