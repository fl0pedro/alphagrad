#!/bin/bash
#SBATCH -J cmprV
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
unset ALPHAGRAD_TOKENS_MID_COMPRESS
echo "=== node=$(hostname) $(date) ==="
echo '===== 1. imports ====='
uv run --no-sync python -c "import alphagrad.approx.env, alphagrad.approx.ppo, alphagrad.approx.live_faces, alphagrad.approx.common.plan_tokens; print('imports OK')"
echo "rc=$?"
echo '===== 2. deleted-knob hard error ====='
ALPHAGRAD_TOKENS_MID_COMPRESS=0 uv run --no-sync python -c "import alphagrad.approx.env" 2>&1 | tail -3
echo '===== 3. targeted tests ====='
uv run --no-sync python -m pytest -x -q \
  tests/stream_prefix_property_test.py \
  tests/rule_replay_test.py \
  tests/face_prefix_cache_test.py \
  tests/delta_obs_emission_test.py \
  tests/live_face_prefix_test.py \
  tests/az_plan_tokens_test.py \
  tests/append_only_tokens_test.py \
  tests/incremental_encode_test.py \
  tests/per_face_apply_test.py \
  tests/face_actions_env_test.py 2>&1 | tail -40
echo "pytest rc=$?"
echo '===== 4. POLICY REGRESSION GATE ====='
uv run --no-sync python tests/policy_regression_gate.py 2>&1 | tail -40
echo "gate rc=$?"
echo "=== DONE $(date) ==="
