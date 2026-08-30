#!/bin/bash
#SBATCH --job-name=cmprA
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/compress_out/probeA_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
cd ~/dsnn/alphagrad
echo "=== node=$(hostname) $(date) ==="
uv run --no-sync python compress_probe.py NeuralNetwork Perceptron
echo "=== RC=$? $(date) ==="
