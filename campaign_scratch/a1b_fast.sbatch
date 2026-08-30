#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 0:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_fast_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b
export ALPHAGRAD_SKIP_COUNT_OPS=1
echo "########## PAIRED --live-faces arms"
uv run --no-sync python pfm_measure.py --live NeuralNetwork TransformerLM3 2>&1 | tail -45
echo "===== PFM2 DONE ====="
echo "===== SUITE edge_mem ====="
uv run --no-sync python -m pytest tests/edge_mem_test.py -q 2>&1 | tail -12
echo "edge_mem rc=${PIPESTATUS[0]}"
echo "===== FAST DONE ====="
