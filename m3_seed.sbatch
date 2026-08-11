#!/bin/bash
#SBATCH -J m3_pomo
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 24:00:00
set -x
SEED=${SEED:?set SEED}
OUT=$HOME/dsnn/elimrl_m3_s${SEED}
mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jax_compile_cache
export PYTHONPATH="$HOME/dsnn/graphax/src:$HOME/dsnn/alphagrad/src"
export ALPHAGRAD_SKIP_COUNT_OPS=1
nvidia-smi -L
cd /Users/assmuth/dsnn/alphagrad
uv run --no-sync python -m alphagrad.elimrl.pomo_train \
    --out-dir "$OUT" \
    --seed "$SEED" --target tlm --updates 500 --n-traj 8 \
    --max-unique 4000 --max-hours 23.0 \
    --baselines --random-baselines 8 \
    --greedy-every 10 --grad-chunk 0 --lr 1e-3 \
    --reps 10 --inner 5 --timeout 400
