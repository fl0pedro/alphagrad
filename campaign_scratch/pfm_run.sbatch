#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/pfm_run_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
unset ALPHAGRAD_MAX_TOKENS
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_pfm
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "===== A. MEASUREMENT (applied/requested before vs after) ====="
timeout 3600 uv run --no-sync python pfm_measure.py NeuralNetwork 2>&1 | tail -40

echo
echo "===== B. TESTS ====="
FILES="
tests/per_face_masks_test.py
tests/diag_per_face_test.py
tests/face_bucket_test.py
tests/face_slot_chooser_test.py
tests/face_mask_oracle_test.py
tests/face_live_state_test.py
tests/per_face_apply_test.py
tests/var_probe_test.py
tests/endpoint_read_test.py
tests/edge_mem_test.py
"
for f in $FILES; do
  if [ ! -f "$f" ]; then echo "SKIP(missing) $f"; continue; fi
  out=$(timeout 1800 uv run --no-sync python -m pytest -q "$f" 2>&1)
  rc=$?
  echo "[pfm] rc=$rc  $f  :: $(echo "$out" | tail -3 | tr "\n" " ")"
  echo "$out" > "pfm_pytest_$(basename $f .py).txt"
done
echo "===== DONE ====="
