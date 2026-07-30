#!/bin/bash
#SBATCH --job-name=ctlppo
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/ctlppo_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export ALPHAGRAD_MEASURE_VIA_AOJ=0
export PYTHONUNBUFFERED=1
cd ~/dsnn/alphagrad
echo "=== node=$(hostname) $(date) AOJ=$ALPHAGRAD_MEASURE_VIA_AOJ ==="
uv run --no-sync python -m alphagrad.approx.run ppo \
  --episodes 3 --num-envs 1 --wandb disabled --seed 0
echo "=== RC=$? $(date) ==="
