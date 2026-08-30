#!/bin/bash
#SBATCH -J cmprV4
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify4_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
R=/Users/assmuth/dsnn/alphagrad/compress_out
H=/tmp/cmpr_head_tree
echo '===== 0. PRE-EXISTING CHECK: clean HEAD snapshot, HEAD code + HEAD tests ====='
rm -rf $H && mkdir -p $H && git archive HEAD | tar -x -C $H
cd $H
PYTHONPATH=$H/src uv run --no-sync --project /Users/assmuth/dsnn/alphagrad python -m pytest -q \
  tests/delta_obs_emission_test.py tests/live_face_prefix_test.py tests/face_prefix_cache_test.py \
  2>&1 | grep -v nanobind | grep -v leaked | tail -20
cd /Users/assmuth/dsnn/alphagrad
echo '===== 1. face_prefix_cache with the missing _FACE_LIVE_STATE shimmed in ====='
cat > /tmp/cmpr_shim.py <<'PYEOF'
import sys
import alphagrad.approx.env as E
if not hasattr(E, "_FACE_LIVE_STATE"):
    E._FACE_LIVE_STATE = False   # agent-1 in-flight symbol, not yet in env.py
import pytest
sys.exit(pytest.main(["-q", "tests/face_prefix_cache_test.py"]))
PYEOF
uv run --no-sync python /tmp/cmpr_shim.py 2>&1 | grep -v nanobind | grep -v leaked | tail -25
echo '===== 2. gate trace dump ====='
uv run --no-sync python tests/policy_regression_gate.py --dump $R/gate_live.json 2>&1 | grep -v nanobind | grep -v leaked | tail -12
echo "=== DONE $(date) ==="
