#!/bin/bash
#SBATCH -J cmprV5
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify5_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
R=/Users/assmuth/dsnn/alphagrad/compress_out
H=/tmp/cmpr_head_tree
echo '===== 1. equivalence: HEAD src vs patched src ====='
cp compress_equiv.py $H/ 2>/dev/null
PYTHONPATH=$H/src uv run --no-sync python $H/compress_equiv.py $R/equiv_head.json 2>&1 | tail -2
uv run --no-sync python compress_equiv.py $R/equiv_new.json 2>&1 | tail -2
python3 - <<'PYEOF'
import json
a=json.load(open('/Users/assmuth/dsnn/alphagrad/compress_out/equiv_head.json'))
b=json.load(open('/Users/assmuth/dsnn/alphagrad/compress_out/equiv_new.json'))
for kind in a:
    diffs=[i for i,(x,y) in enumerate(zip(a[kind],b[kind])) if x!=y]
    print(f'{kind:9s}: {len(a[kind])} prefix lengths, differing at {diffs}')
PYEOF
echo '===== 2. RE-RECORD the golden (deliberate semantic change) ====='
cp tests/golden/policy_gate_golden.json $R/policy_gate_golden.PRE_COMPRESS_FIX.json
uv run --no-sync python tests/policy_regression_gate.py --record 2>&1 | grep -v nanobind | grep -v leaked | tail -6
echo '===== 3. GATE (must PASS against the new golden) ====='
uv run --no-sync python tests/policy_regression_gate.py 2>&1 | grep -v nanobind | grep -v leaked | tail -8
echo "gate rc=${PIPESTATUS[0]}"
echo '===== 4. full targeted suite, final state (with shim) ====='
cat > /tmp/cmpr_shim2.py <<'PYEOF'
import sys
import alphagrad.approx.env as E
if not hasattr(E, "_FACE_LIVE_STATE"):
    E._FACE_LIVE_STATE = False
import pytest
sys.exit(pytest.main(["-q",
  "tests/stream_prefix_property_test.py", "tests/rule_replay_test.py",
  "tests/face_prefix_cache_test.py", "tests/delta_obs_emission_test.py",
  "tests/live_face_prefix_test.py", "tests/az_plan_tokens_test.py",
  "tests/append_only_tokens_test.py", "tests/incremental_encode_test.py",
  "tests/per_face_apply_test.py", "tests/face_actions_env_test.py",
  "tests/masked_extend_equivalence_test.py", "tests/face_mask_oracle_test.py",
  "tests/policy_regression_gate_test.py"]))
PYEOF
uv run --no-sync python /tmp/cmpr_shim2.py 2>&1 | grep -v nanobind | grep -v leaked | tail -20
echo "=== DONE $(date) ==="
