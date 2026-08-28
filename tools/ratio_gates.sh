#!/usr/bin/env bash
# The ratio-pinning gates: sampling log-prob == replay log-prob, i.e. the
# PPO importance ratio is exactly 1 at epoch 0. If one of these is red, the
# gradients of any run using that feature were taken against a wrong ratio.
#
# EACH GATE RUNS IN ITS OWN PYTHON PROCESS, ON PURPOSE.
# alphagrad.approx.env freezes MAX_DELTA_TOKENS / MAX_FACES at its FIRST
# import, so a module that asks for a small test scale only gets it when it
# imports alphagrad first. Running these five in ONE pytest process is what
# produced
#     ValueError: Incompatible shapes for broadcasting:
#                 shapes=[(32768,), (128,), ()]
# in tests/endpoint_read_test.py -- scaffolding, not the mirror. See
# tests/_scale_guard.py.
#
# A SKIP IS A FAILURE HERE. tests/_scale_guard.py skips exactly when the
# scale request did not take, and a gate that did not run pins nothing.
#
# WHAT THIS FILE DOES NOT COVER, and where that lives.
# These gates pin what the numbers MEAN; not one of them starts a measure
# pool, and tools/smoke.sh's canonical config has no --ray-measure in it at
# all. So a --ray-measure run that measures NOTHING passes every gate in this
# file (87cdc49: the pool forwarded an `episode` kwarg the actor wrapper did
# not accept, every pooled dispatch died with TypeError, the pool sentinelled
# the row and killed the actor, and every terminal reward was the degenerate
# sentinel -- while the run exited 0 and printed finite health rows). That is
# tools/pool_liveness_gate.sh's job; it is a SIBLING rather than a member of
# GATES below because it costs ~6 minutes and a real Ray run rather than a
# pytest process, and it runs in the SAME W0-A pre-flight (see
# tools/gen_fq_launchers.py, arm w0_cpu_gates, GATE 3/3).
#
# Usage:  tools/ratio_gates.sh          (uses $PY, default: python)
#         PY="uv run --no-sync python" tools/ratio_gates.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export GRAPHAX_ALLOW_PARTIAL_ORDER="${GRAPHAX_ALLOW_PARTIAL_ORDER:-1}"
export ALPHAGRAD_SKIP_COUNT_OPS="${ALPHAGRAD_SKIP_COUNT_OPS:-1}"
PY=${PY:-python}

GATES=(
  "tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay"
  "tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant"
  "tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes"
  "tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks"
  "tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical"
  "tests/policy_regression_gate_test.py"
)

rc=0
for g in "${GATES[@]}"; do
  out=$(${PY} -m pytest "$g" -q --tb=short -p no:cacheprovider 2>&1); grc=$?
  last=$(printf "%s\n" "$out" | grep -E "passed|failed|error|no tests ran" | tail -1)
  if [ ${grc} -ne 0 ]; then
    echo "FAIL  ${g}"
    echo "      ${last}"
    printf "%s\n" "$out" | tail -40 | sed "s/^/      /"
    rc=1
  elif printf "%s\n" "$out" | grep -qE "[0-9]+ skipped"; then
    echo "FAIL  ${g}   SKIPPED -- a gate that did not run pins nothing"
    printf "%s\n" "$out" | grep -iE "skip" | head -6 | sed "s/^/      /"
    rc=1
  elif ! printf "%s\n" "$out" | grep -qE "[0-9]+ passed"; then
    echo "FAIL  ${g}   no test ran"
    echo "      ${last}"
    rc=1
  else
    echo "ok    ${g}"
    echo "      ${last}"
  fi
done

if [ ${rc} -eq 0 ]; then
  echo "ALL RATIO GATES PASS (ratio == 1 at epoch 0 is pinned)"
else
  echo "RATIO GATES RED -- do not trust gradients from the affected feature"
fi
exit ${rc}
