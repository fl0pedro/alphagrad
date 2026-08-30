#!/bin/bash
#SBATCH -J cmprV3
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
echo '===== A: face_prefix_cache (is it _FACE_LIVE_STATE?) ====='
uv run --no-sync python -m pytest -q tests/face_prefix_cache_test.py::test_face_enum_cache_equals_cold_and_engages 2>&1 | grep -v nanobind | grep -v 'leaked' | tail -25
echo '===== A2: SAME TEST AT HEAD (stash-free: git show into a temp file) ====='
git show HEAD:tests/face_prefix_cache_test.py > /tmp/fpc_head_test.py
uv run --no-sync python -m pytest -q /tmp/fpc_head_test.py::test_face_enum_cache_equals_cold_and_engages 2>&1 | grep -v nanobind | grep -v leaked | tail -20
echo '===== B: delta_obs diag=True ====='
uv run --no-sync python -m pytest -q 'tests/delta_obs_emission_test.py::test_base_plus_deltas_reconstructs_the_stream[True]' 2>&1 | grep -v nanobind | grep -v leaked | tail -35
echo '===== C: live_face_prefix ====='
uv run --no-sync python -m pytest -q tests/live_face_prefix_test.py::test_prefix_replay_matches_the_measurements_graph 2>&1 | grep -v nanobind | grep -v leaked | tail -35
echo "=== DONE $(date) ==="
