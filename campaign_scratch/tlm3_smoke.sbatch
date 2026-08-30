#!/bin/bash
#SBATCH -J tlm3smk
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/tlm3_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/tlm3_out
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

# ---- the 29x trap: EXTEND_CHUNK MUST BE NON-ZERO -------------------------
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
# --------------------------------------------------------------------------
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_PROFILE=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_TLM_DMODEL=128
export ALPHAGRAD_TLM_VOCAB=2048

echo "===== XLA_FLAGS validation ====="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo "===== chunk sanity: ALPHAGRAD_EXTEND_CHUNK=$ALPHAGRAD_EXTEND_CHUNK (MUST NOT be 0) ====="
[ "$ALPHAGRAD_EXTEND_CHUNK" = "0" ] && { echo "FATAL: chunk=0 is the 29x trap"; exit 1; }

# ts: prefix each line with epoch seconds so per-episode wall is measurable
run_ppo () {   # $1 = example, $2 = gpu id, $3 = tag, $4 = episodes
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $2; sleep 2; done ) > $R/gpumem_$3.txt &
  MON=$!
  T0=$(date +%s)
  JAX_COMPILATION_CACHE_DIR=$S/.jc_tlm3 CUDA_VISIBLE_DEVICES=$2 timeout 12000 \
    stdbuf -oL -eL uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example $1 --dataset wikitext2 \
    --variant full --face-actions --unified-face-head --live-faces \
    --dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
    --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
    --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
    --advantage-norm popart --num-envs 4 --minibatches 4 \
    --latency-inner-reps 50 --vocab-size 512 \
    --episodes $4 --wandb disabled --name tlm3-$3 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/ppo_$3.log
  RC=$?
  T1=$(date +%s)
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$3.txt | tail -1)
  echo "  PPO[$3] example=$1 episodes=$4 rc=$RC total_wall=$((T1-T0))s gpu_peak_MiB=$PEAK"
  echo "  --- chunk warning check (must be EMPTY) ---"
  grep -i "ALPHAGRAD_EXTEND_CHUNK" $R/ppo_$3.log | head -3
  echo "  --- per-episode markers (epoch prefix = wall) ---"
  grep -E "\[prof ep=|\[mem ep=" $R/ppo_$3.log | head -20
  echo "  --- tail ---"
  tail -25 $R/ppo_$3.log
}

echo "===== A) reference: TransformerLM (2 blocks, 96 eqns / 95 elim) ====="
run_ppo TransformerLM 0 tlm2 3

echo "===== B) target: TransformerLM3 (3 blocks, 136 eqns / 135 elim) ====="
run_ppo TransformerLM3 1 tlm3 3

run_gaz () {   # $1 = task, $2 = gpu id, $3 = tag, $4 = measurements
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $2; sleep 2; done ) > $R/gpumem_$3.txt &
  MON=$!
  T0=$(date +%s)
  JAX_COMPILATION_CACHE_DIR=$S/.jc_tlm3gaz CUDA_VISIBLE_DEVICES=$2 timeout 7200 \
    stdbuf -oL -eL uv run --no-sync python -m alphagrad.approx.az_gumbel \
    --task $1 --dataset wikitext2 --seed 7 \
    --total-measurements $4 --popart-init-episodes 0 \
    --latency-inner-reps 50 --wandb disabled --name tlm3-$3 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/gaz_$3.log
  RC=$?
  T1=$(date +%s)
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$3.txt | tail -1)
  echo "  GAZ[$3] task=$1 measurements=$4 rc=$RC total_wall=$((T1-T0))s gpu_peak_MiB=$PEAK"
  grep -i "ALPHAGRAD_EXTEND_CHUNK" $R/gaz_$3.log | head -3
  tail -30 $R/gaz_$3.log
}

echo "===== C) GAZ arm: az_gumbel on TransformerLM3 ====="
run_gaz TransformerLM3 0 gaz3 3

echo "===== SMOKE DONE ====="
