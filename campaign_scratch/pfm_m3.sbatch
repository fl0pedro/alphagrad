#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=160G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/pfm_m3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
unset ALPHAGRAD_MAX_TOKENS
echo "===== FINAL MEASUREMENT ====="
timeout 12000 uv run --no-sync python pfm_measure.py NeuralNetwork TransformerLM3 > pfm_m3.txt 2>&1
echo "rc=$?"; cat pfm_m3.txt
echo "===== M3 DONE ====="
