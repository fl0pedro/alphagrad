#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_run2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b
export ALPHAGRAD_SKIP_COUNT_OPS=1

echo "===== A1b unit test (fixed) ====="
uv run --no-sync python -m pytest tests/per_face_sizes_live_test.py -q 2>&1 | tail -25
echo "unit rc=${PIPESTATUS[0]}"
for T in endpoint_read edge_mem; do
  echo "===== SUITE $T ====="
  uv run --no-sync python -m pytest tests/${T}_test.py -q 2>&1 | tail -12
  echo "$T rc=${PIPESTATUS[0]}"
done
echo "########## PAIRED --live-faces arms"
uv run --no-sync python pfm_measure.py --live NeuralNetwork 2>&1 | tail -30
echo
uv run --no-sync python pfm_measure.py --live TransformerLM3 2>&1 | tail -30
echo "===== A1B RUN2 DONE ====="
