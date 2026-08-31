#!/usr/bin/env bash
# The whole suite, ONE PYTHON PROCESS PER TEST MODULE.
#
# WHY THIS EXISTS
# ---------------
# A single shared pytest process cannot give this suite a trustworthy answer.
# 195 module-scope `os.environ` writes live across the two test roots -- almost
# all `setdefault` at import, because most ALPHAGRAD_* knobs must be set BEFORE
# alphagrad.approx.env is imported to have any effect. In one process each of
# those writes leaks into every module imported after it, so a module's result
# depends on who was collected first.
#
# Measured, 2026-08-31: src/alphagrad/approx/tests/test_env_callback.py gave two
# DISJOINT failure sets alone vs in the suite. Bisecting the 21 variables that
# differed found ALPHAGRAD_INCREMENTAL_TOKENS=1 sufficient on its own. See
# .scratch/trustworthy-approx-search/findings/47-import-order-contamination.md
#
# WHY NOT A CONFTEST FIXTURE
# --------------------------
# Tried and reverted (574ed634). Restoring os.environ per test cannot work:
# env.py FREEZES MAX_DELTA_TOKENS / MAX_FACES / MAX_TOKENS into module constants
# at its first import, and no fixture reaches a constant that is already bound.
# Worse, when module B does setdefault(K, x) that module A already won, B does
# not own K, so restoring "B's own vars" DELETES K -- leaving the env var absent
# while the constant derived from it is still set. That third state is coherent
# for nobody: it took the suite from 8 failures to 50 failures + 7 errors.
#
# One process per module is the only correct isolation. tools/ratio_gates.sh
# already does this for six hand-picked gates and says so in its own header;
# this is the same idea applied to everything.
#
# Usage:
#   tools/suite_per_module.sh                 # both roots, 8-way parallel
#   JOBS=1 tools/suite_per_module.sh          # serial
#   tools/suite_per_module.sh tests/foo_test.py tests/bar_test.py
#
# Exit status is the number of modules that failed, capped at 250.
set -uo pipefail
cd "$(dirname "$0")/.."

# PRISTINE environment on purpose: each module must DECLARE the knobs it needs.
# Do not add ALPHAGRAD_*/GRAPHAX_* defaults here -- that would re-create the
# borrowing this script exists to eliminate. JAX_PLATFORMS and the compile cache
# are harness settings, not behaviour knobs.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache_suite}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

PY=${PY:-uv run --no-sync python}
JOBS=${JOBS:-8}
OUT=${OUT:-$(mktemp -d)}
mkdir -p "$OUT"   # mktemp -d creates it; a caller-supplied OUT does not

if [ "$#" -gt 0 ]; then
  MODULES=("$@")
else
  # Mirror [tool.pytest.ini_options] testpaths + python_files exactly.
  mapfile -t MODULES < <(
    find tests src/alphagrad/approx/tests \
         \( -name 'test_*.py' -o -name '*_test.py' \) -type f | sort
  )
fi

echo "== ${#MODULES[@]} modules, one process each, JOBS=$JOBS =="
echo "== per-module logs in $OUT =="

run_one() {
  local m="$1" out="$2"
  local log="$out/$(echo "$m" | tr '/' '_').log"
  # shellcheck disable=SC2086
  $PY -m pytest "$m" -q --tb=short -p no:cacheprovider > "$log" 2>&1
  local rc=$?
  local last
  # Anchor on pytest's own summary line ("1 failed, 1 passed, ... in 42.03s"),
  # NOT a bare word match: the nanobind leak report this venv prints at exit
  # contains " - ... skipped remainder", which swallowed a real summary and
  # undercounted the failing-test total.
  last=$(grep -E "^([0-9]+ (passed|failed|skipped|xfailed|xpassed|error|warning)|no tests ran)" "$log" | tail -1)
  printf "%s\t%d\t%s\n" "$m" "$rc" "${last:-<no summary>}"
}
export -f run_one
export PY

printf '%s\n' "${MODULES[@]}" \
  | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$OUT" \
  > "$OUT/results.tsv"

# pytest exit codes: 0 = all passed, 1 = tests failed, 5 = NO TESTS COLLECTED.
# 5 is not a failure here -- it is what a module-level quarantine skip and a
# script-style module with no test functions both produce, and this suite has
# 62 of those. Counting them red hides the modules that are actually broken.
echo
echo "-- modules with FAILING tests --"
sort "$OUT/results.tsv" | while IFS=$'\t' read -r m rc last; do
  case "$rc" in 0|5) ;; *) printf "  %-58s %s\n" "$m" "$last" ;; esac
done

nfail=$(awk -F'\t' '$2 != 0 && $2 != 5' "$OUT/results.tsv" | wc -l | tr -d ' ')
npass=$(awk -F'\t' '$2 == 0' "$OUT/results.tsv" | wc -l | tr -d ' ')
nempty=$(awk -F'\t' '$2 == 5' "$OUT/results.tsv" | wc -l | tr -d ' ')
ntests=$(awk -F'\t' '$2 != 0 && $2 != 5 {match($3, /[0-9]+ failed/); if (RSTART) print substr($3, RSTART, RLENGTH)}' \
         "$OUT/results.tsv" | awk '{s += $1} END {print s + 0}')
echo
echo "== $npass modules clean | $nfail modules failing ($ntests tests) | $nempty ran no tests =="
echo "== logs: $OUT =="
[ "$nfail" -gt 250 ] && nfail=250
exit "$nfail"
