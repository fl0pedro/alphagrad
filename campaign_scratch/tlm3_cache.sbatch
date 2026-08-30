#!/bin/bash
#SBATCH -J tlm3cache
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/tlm3_cache_%j.log
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
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

# PAIRED, back-to-back on the SAME GPU in the SAME job: GPU state drifts
# ~18-20% between jobs, so an unpaired A/B is not evidence (feedback #paired).
run () {  # $1 = cache flag, $2 = tag
  export ALPHAGRAD_FACE_ENUM_CACHE=$1
  T0=$(date +%s)
  ALPHAGRAD_HEALTH_EPISODES=9 CUDA_VISIBLE_DEVICES=0 timeout 10800 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example TransformerLM3 --dataset wikitext2 $COMMON \
    --episodes 9 --name cache$2 > $R/cache_$2.log 2>&1
  RC=$?
  T1=$(date +%s)
  echo "  CACHE[$2] ALPHAGRAD_FACE_ENUM_CACHE=$1 rc=$RC total_wall=$((T1-T0))s"
  grep -E "health ep" $R/cache_$2.log | grep -v live-faces | head -10
  grep -oE "\[prof ep=[ 0-9]+\] host_total=[ 0-9.]+s  cb.face_enum=[0-9.]+s" $R/cache_$2.log | head -10
  grep -oE "face_enum\(ext/cold\)=[0-9]+/[0-9]+" $R/cache_$2.log | head -10
}

echo "##### TLM3 face-enum cache A/B (paired, same GPU, same job) #####"
echo "===== OFF (shipped default) ====="
run 0 off
echo "===== ON  (ALPHAGRAD_FACE_ENUM_CACHE=1) ====="
run 1 on
echo "##### CACHE A/B DONE #####"
