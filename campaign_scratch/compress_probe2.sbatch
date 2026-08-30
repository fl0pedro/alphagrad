#!/bin/bash
#SBATCH --job-name=cmprB
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/compress_out/probeB_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
cd ~/dsnn/alphagrad
echo "=== node=$(hostname) $(date) ==="
uv run --no-sync python compress_probe2.py NeuralNetwork TransformerLM
echo "=== RC=$? $(date) ==="
