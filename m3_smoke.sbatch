#!/bin/bash
#SBATCH -J m3_smoke
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o /Users/assmuth/dsnn/elimrl_m3_smoke/smoke.%j.out
#SBATCH -e /Users/assmuth/dsnn/elimrl_m3_smoke/smoke.%j.err

set -x
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
export JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jax_compile_cache
export PYTHONPATH="$HOME/dsnn/graphax/src:$HOME/dsnn/alphagrad/src"
export ALPHAGRAD_SKIP_COUNT_OPS=1
nvidia-smi
cd /Users/assmuth/dsnn/alphagrad
uv run --no-sync python -m alphagrad.elimrl.pomo_train \
    --out-dir /Users/assmuth/dsnn/elimrl_m3_smoke \
    --seed 0 --target tlm --updates 20 --n-traj 8 \
    --max-unique 4000 --max-hours 3.5 \
    --baselines --random-baselines 32 \
    --greedy-every 5 --grad-chunk 2 \
    --reps 10 --inner 5 --timeout 1500
