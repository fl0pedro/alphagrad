#!/bin/bash
#SBATCH -J abprobe
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu17
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/ab_probe_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export GRAPHAX_PLANNER_EXACT=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_TLM_DMODEL=128
export ALPHAGRAD_TLM_VOCAB=256
unset JAX_COMPILATION_CACHE_DIR
echo "===== NODE ====="; hostname
env | grep -E '^(GRAPHAX_|ALPHAGRAD_|XLA_)' | sort
timeout 4800 stdbuf -oL -eL uv run --no-sync python ab_pomo_probe.py 2>&1
echo "rc=$?"
echo "===== PROBE DONE ====="
