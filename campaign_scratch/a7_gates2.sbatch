#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 5:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_gates2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7g
echo "########## tools/ratio_gates.sh"
PY="uv run --no-sync python" tools/ratio_gates.sh
echo "gates rc=$?"
