#!/bin/bash
#SBATCH -J cmprV6
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 00:45:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/verify6_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export PYTHONUNBUFFERED=1
H=/tmp/cmpr_head_tree
echo '===== A: gate test ALONE (patched tree) ====='
uv run --no-sync python -m pytest -q tests/policy_regression_gate_test.py 2>&1 | grep -v nanobind | grep -v leaked | tail -6
echo '===== B: per_face_apply ALONE (patched tree) ====='
uv run --no-sync python -m pytest -q tests/per_face_apply_test.py 2>&1 | grep -v nanobind | grep -v leaked | tail -6
echo '===== C: the PAIR, patched tree (pollution check) ====='
uv run --no-sync python -m pytest -q tests/per_face_apply_test.py tests/policy_regression_gate_test.py 2>&1 | grep -v nanobind | grep -v leaked | tail -25
echo '===== D: the SAME PAIR at HEAD (pre-existing?) ====='
cd $H
PYTHONPATH=$H/src uv run --no-sync --project /Users/assmuth/dsnn/alphagrad python -m pytest -q tests/per_face_apply_test.py tests/policy_regression_gate_test.py 2>&1 | grep -v nanobind | grep -v leaked | tail -20
echo "=== DONE $(date) ==="
