#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_pfm_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b
export ALPHAGRAD_SKIP_COUNT_OPS=1

echo "########## A1b: applied/requested, ORACLE regime AND --live-faces regime"
uv run --no-sync python pfm_measure.py --both NeuralNetwork 2>&1 | tail -60
echo
uv run --no-sync python pfm_measure.py --both TransformerLM3 2>&1 | tail -60
echo "===== A1B PFM DONE ====="
