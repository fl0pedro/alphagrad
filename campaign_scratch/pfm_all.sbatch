#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/pfm_all_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
unset ALPHAGRAD_MAX_TOKENS
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_pfm
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
which uv; uv --version
echo "===== A. MEASUREMENT v2 (all three slots) ====="
timeout 5400 uv run --no-sync python pfm_measure.py NeuralNetwork > pfm_meas_out.txt 2>&1
echo "meas rc=$?"
cat pfm_meas_out.txt
echo
echo "===== B. TESTS ====="
for f in tests/per_face_masks_test.py tests/diag_per_face_test.py tests/face_bucket_test.py tests/face_slot_chooser_test.py tests/face_mask_oracle_test.py tests/face_live_state_test.py tests/per_face_apply_test.py tests/var_probe_test.py tests/endpoint_read_test.py tests/edge_mem_test.py ; do
  if [ ! -f "$f" ]; then echo "SKIP(missing) $f"; continue; fi
  b=$(basename $f .py)
  timeout 1800 uv run --no-sync python -m pytest -q "$f" > "pfm_pytest_$b.txt" 2>&1
  rc=$?
  echo "[pfm] rc=$rc  $f  :: $(tail -3 pfm_pytest_$b.txt | tr "\n" " ")"
done
echo "===== DONE ====="
