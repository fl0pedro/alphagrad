#!/bin/bash
#SBATCH -J cmprV2
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
R=/Users/assmuth/dsnn/alphagrad/compress_out
echo "=== node=$(hostname) $(date) ==="
echo '===== 1. targeted tests (NO -x) ====='
uv run --no-sync python -m pytest -q \
  tests/stream_prefix_property_test.py \
  tests/rule_replay_test.py \
  tests/face_prefix_cache_test.py \
  tests/delta_obs_emission_test.py \
  tests/live_face_prefix_test.py \
  tests/az_plan_tokens_test.py \
  tests/append_only_tokens_test.py \
  tests/incremental_encode_test.py \
  tests/per_face_apply_test.py \
  tests/face_actions_env_test.py \
  tests/masked_extend_equivalence_test.py \
  tests/face_mask_oracle_test.py 2>&1 | tail -30
echo '===== 2. gate trace dump (live, current code) ====='
uv run --no-sync python tests/policy_regression_gate.py --dump $R/gate_live.json 2>&1 | tail -5
echo "=== DONE $(date) ==="
