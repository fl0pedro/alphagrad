#!/bin/bash
#SBATCH -J tlm3cnt
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 00:50:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/tlm3_count_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad

echo "=== import check ==="
JAX_PLATFORMS=cpu uv run --no-sync python -c "import alphagrad.approx.ppo, alphagrad.approx.az_gumbel, alphagrad.approx.unified_face_policy; print('imports OK')"
echo "=== import exit: $? ==="

echo "=== TLM2 vs TLM3 equation / eliminable-vertex counts ==="
JAX_PLATFORMS=cpu uv run --no-sync python tlm3_count.py
echo "=== count exit: $? ==="

echo "=== policy regression gate (must stay bit-identical) ==="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py
echo "=== gate exit: $? ==="
