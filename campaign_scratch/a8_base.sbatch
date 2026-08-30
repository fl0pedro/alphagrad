#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_base_%j.log
export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
B=/Users/assmuth/dsnn/.a8_base
rm -rf $B && mkdir -p $B
cd /Users/assmuth/dsnn/alphagrad
git archive 4744979 | tar x -C $B
echo "########## BASELINE 4744979 whole-suite"
cd $B
PYTHONPATH=$B/src timeout 10800 uv run --no-sync python -m pytest tests -q -p no:cacheprovider \
  --ignore=tests/transforms/symmetry_test.py 2>&1 | tail -25
echo "########## PATCHED: each of the 9 in its OWN process"
cd /Users/assmuth/dsnn/alphagrad
for g in \
  "tests/carry_stream_base_attribution_test.py::test_advance_still_credits_the_owning_vertex" \
  "tests/delta_obs_emission_test.py::test_base_plus_deltas_reconstructs_the_stream" \
  "tests/live_vertex_mask_test.py::test_mask_is_not_vacuous" \
  "tests/per_face_apply_test.py::test_delta_budget_is_the_only_bound_and_both_widths_are_settable" \
  "tests/policy_regression_gate_test.py::test_policy_path_is_bit_identical_to_golden" \
  "tests/test_all_cost_channels.py" \
  "tests/test_sparse_terminal.py::test_sparse_terminal_indices_match_canonical_channels" ; do
  r=$(uv run --no-sync python -m pytest "$g" -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|error|no tests ran" | tail -1)
  echo "PATCHED  $g  ->  $r"
done
echo "########## BASELINE: the same 9 in their OWN process"
cd $B
for g in \
  "tests/carry_stream_base_attribution_test.py::test_advance_still_credits_the_owning_vertex" \
  "tests/delta_obs_emission_test.py::test_base_plus_deltas_reconstructs_the_stream" \
  "tests/live_vertex_mask_test.py::test_mask_is_not_vacuous" \
  "tests/per_face_apply_test.py::test_delta_budget_is_the_only_bound_and_both_widths_are_settable" \
  "tests/policy_regression_gate_test.py::test_policy_path_is_bit_identical_to_golden" \
  "tests/test_all_cost_channels.py" \
  "tests/test_sparse_terminal.py::test_sparse_terminal_indices_match_canonical_channels" ; do
  r=$(PYTHONPATH=$B/src uv run --no-sync python -m pytest "$g" -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|error|no tests ran" | tail -1)
  echo "BASELINE $g  ->  $r"
done
echo "########## A8 BASE DONE"
