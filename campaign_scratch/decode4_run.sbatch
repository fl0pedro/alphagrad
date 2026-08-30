#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -c 128
#SBATCH -t 10:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode4_run_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
# Blackwell: the Triton GEMM tiling pass asserts (dst_remaining_size == 1,
# rc=134) on three of these programs. Flag names validated below, not guessed.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode4_out
mkdir -p $R
DA=$S/decode3_dsA.npz

echo "===== XLA_FLAGS validation (an unknown flag aborts every arm at import) ====="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo "===== TRACE + PARAMETER-SHAPE AUDIT, all four pools ====="
for P in mean last sumtok attn; do
  echo "########## pool $P"
  JAX_PLATFORMS=cpu uv run --no-sync python decode4_face_train.py --data $DA \
    --out /dev/null --pool $P --trace-only 2>&1 | tail -45 || exit 1
done

echo "===== FOUR-POOL A/B, 5 seeds ====="
runp () {  # gpu, pool, seed
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d4_$1 CUDA_VISIBLE_DEVICES=$1 \
    uv run --no-sync python decode4_face_train.py --data $DA \
    --out $R/face_${2}_s${3}.json --pool $2 --seed $3 --steps 3000 \
    --batch 4 --chunk-cap 4096 --lr 1e-3 > $R/face_${2}_s${3}.log 2>&1
  rc=$?
  echo "  face_${2}_s${3} rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "  face_${2}_s${3} retrying at batch 3 (different fusion shapes)"
    JAX_COMPILATION_CACHE_DIR=$S/.jc_d4r_$1 CUDA_VISIBLE_DEVICES=$1 \
      uv run --no-sync python decode4_face_train.py --data $DA \
      --out $R/face_${2}_s${3}.json --pool $2 --seed $3 --steps 3000 \
      --batch 3 --chunk-cap 4096 --lr 1e-3 > $R/face_${2}_s${3}.log 2>&1
    echo "  face_${2}_s${3} (batch 3) rc=$?"
  fi
}

# One serial queue per GPU: 5 seeds x 4 pools over 8 GPUs.
( runp 0 mean   0 ; runp 0 mean   3 ) &
( runp 1 mean   1 ; runp 1 mean   4 ) &
( runp 2 mean   2 ; runp 2 attn   3 ) &
( runp 3 attn   0 ; runp 3 attn   4 ) &
( runp 4 attn   1 ; runp 4 last   3 ) &
( runp 5 attn   2 ; runp 5 last   4 ) &
( runp 6 last   0 ; runp 6 last   1 ; runp 6 last   2 ) &
( runp 7 sumtok 0 ; runp 7 sumtok 1 ; runp 7 sumtok 2 ) &
wait
( runp 0 sumtok 3 ) &
( runp 1 sumtok 4 ) &

echo "===== (d) end-to-end nn256 rollout, 1024 vs 32768 ====="
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
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d4ab CUDA_VISIBLE_DEVICES=$1 timeout 3000 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example VmappedNeuralNetwork --dataset mnist --hidden-dim 256 \
    --vocab-size 512 \
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
  grep -E "prof/(encode|rollout|update)|truncat|WARNING: ALPHAGRAD_EXTEND_CHUNK" $R/ab_$4.log | tail -6
}
( run_ab 2 1024  0   b1024_c0   ; run_ab 2 32768 0   b32768_c0 ) &
( run_ab 3 1024  256 b1024_c256 ; run_ab 3 32768 256 b32768_c256 ) &
wait

echo "===== SUMMARY ====="
uv run --no-sync python decode4_summary.py $R
echo "===== DECODE4 DONE ====="
