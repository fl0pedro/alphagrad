#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_tests_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b
export ALPHAGRAD_SKIP_COUNT_OPS=1

echo "===== A1b new: per_face_sizes_live ====="
uv run --no-sync python -m pytest tests/per_face_sizes_live_test.py -q 2>&1 | tail -60
echo "rc=${PIPESTATUS[0]}"

for T in per_face_masks diag_per_face face_bucket face_slot_chooser \
         face_mask_oracle face_live_state per_face_apply endpoint_read edge_mem; do
  echo "===== SUITE $T ====="
  uv run --no-sync python -m pytest tests/${T}_test.py -q 2>&1 | tail -14
  echo "$T rc=${PIPESTATUS[0]}"
done
echo "===== DONE ====="
