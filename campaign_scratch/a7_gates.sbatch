#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 5:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_gates_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7g
echo "########## tools/smoke.sh (flag OFF: A7 is inert)"
SMOKE_OUT=/Users/assmuth/dsnn/alphagrad/a7_out/smoke_off tools/smoke.sh Helmholtz NeuralNetwork
echo "smoke_off rc=$?"
echo "########## tools/smoke.sh with --sparsity-log (channel LOGGED)"
export ALPHAGRAD_SMOKE_EXTRA="--sparsity-log"
SMOKE_OUT=/Users/assmuth/dsnn/alphagrad/a7_out/smoke_on tools/smoke.sh NeuralNetwork
echo "smoke_on rc=$?"
unset ALPHAGRAD_SMOKE_EXTRA
echo "########## tools/ratio_gates.sh"
PY="uv run --no-sync python" tools/ratio_gates.sh
echo "gates rc=$?"
