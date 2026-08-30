#!/bin/bash
#SBATCH -J cmprV7
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 00:45:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify7_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
echo '===== face_prefix_cache, NO shim ====='
uv run --no-sync python -m pytest -q tests/face_prefix_cache_test.py 2>&1 | grep -v nanobind | grep -v leaked | tail -6
echo '===== full targeted suite, NO shim (gate test run separately) ====='
uv run --no-sync python -m pytest -q \
  tests/stream_prefix_property_test.py tests/rule_replay_test.py \
  tests/face_prefix_cache_test.py tests/delta_obs_emission_test.py \
  tests/live_face_prefix_test.py tests/az_plan_tokens_test.py \
  tests/append_only_tokens_test.py tests/incremental_encode_test.py \
  tests/per_face_apply_test.py tests/face_actions_env_test.py \
  tests/masked_extend_equivalence_test.py tests/face_mask_oracle_test.py \
  tests/micro_action_mask_test.py tests/test_live_faces.py \
  2>&1 | grep -v nanobind | grep -v leaked | tail -12
echo '===== GATE ====='
uv run --no-sync python tests/policy_regression_gate.py 2>&1 | grep -v nanobind | grep -v leaked | grep -E 'PASS|FAIL'
echo '===== imports of every touched module ====='
uv run --no-sync python -c "import alphagrad.approx.env, alphagrad.approx.ppo, alphagrad.approx.az_gumbel, alphagrad.approx.live_faces, alphagrad.approx.common.plan_tokens, alphagrad.approx.unified_face_policy; print('IMPORTS OK')"
echo "=== DONE $(date) ==="
