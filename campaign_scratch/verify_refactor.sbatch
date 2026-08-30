#!/bin/bash
#SBATCH -J verify
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/verify_refactor_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
echo "=== import check ==="
JAX_PLATFORMS=cpu uv run --no-sync python -c "import alphagrad.approx.ppo, alphagrad.approx.az_gumbel, alphagrad.approx.unified_face_policy; print(\"imports OK\")"
echo "=== policy regression gate ==="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py
echo "=== gate exit: $? ==="
