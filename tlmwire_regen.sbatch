#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH -J node-pgi15-cpu1
#SBATCH --dependency=singleton
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -o /Scratch/assmuth/mrg/tlmwire/tlmwire_regen_%j.log

set -uo pipefail
S=/Scratch/assmuth/mrg/tlmwire
PY=/Scratch/assmuth/t57/stack/venv/bin/python
OUT=/Scratch/assmuth/mrg/runs/thesis_launchers
W=/Scratch/assmuth/mrg/tlmwire/regen_$SLURM_JOB_ID
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH=$S/src
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
mkdir -p $W
cd $S
echo "ag=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git status --porcelain

echo "=== 1. the launcher tests ==="
$PY -m pytest -q -p no:cacheprovider tests/gen_fq_launchers_thesis_test.py
echo "LAUNCHER_TESTS rc=$?"

echo "=== 2. --check against $OUT, BEFORE the write ==="
$PY tools/gen_fq_launchers.py --check --out $OUT > $W/precheck.log 2>&1
echo "PRECHECK rc=$?"
echo "ok=$(grep -c '^ok  ' $W/precheck.log)"
echo "drift=$(grep -c '^DRIFT' $W/precheck.log)"
echo "missing=$(grep -c '^MISSING' $W/precheck.log)"
echo "-- drift, added lines, unique --"
grep -E '^\+' $W/precheck.log | grep -v '^+++' | sed 's/^\+//' | sort -u | head -20
echo "-- drift, removed lines, unique --"
grep -E '^-' $W/precheck.log | grep -v '^---' | sed 's/^-//' | sort -u | head -20

echo "=== 3. regenerate into $OUT ==="
$PY tools/gen_fq_launchers.py --out $OUT > $W/gen.log 2>&1
echo "GEN rc=$?"
echo "wrote=$(grep -c '^wrote ' $W/gen.log)"

echo "=== 4. --check against $OUT, AFTER the write ==="
$PY tools/gen_fq_launchers.py --check --out $OUT > $W/postcheck.log 2>&1
echo "POSTCHECK rc=$?"
echo "ok=$(grep -c '^ok  ' $W/postcheck.log)"
echo "drift=$(grep -c '^DRIFT' $W/postcheck.log)"
echo "missing=$(grep -c '^MISSING' $W/postcheck.log)"

echo "=== 5. the footprint ==="
echo "with face-wire-faces 128 (tlm)     : $(grep -l -- '--face-wire-faces 128' $OUT/fq_*.sbatch | wc -l)"
echo "with face-wire-faces 64  (others)  : $(grep -l -- '--face-wire-faces 64' $OUT/fq_*.sbatch | wc -l)"
echo "with cache-clear export            : $(grep -l -- 'export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=100' $OUT/fq_*.sbatch | wc -l)"
echo "with retry-on-oom export           : $(grep -l -- 'export ALPHAGRAD_RECYCLE_RETRY_ON_OOM=1' $OUT/fq_*.sbatch | wc -l)"
echo "with proactive-recycle export      : $(grep -l -- 'export ALPHAGRAD_PROACTIVE_RECYCLE_EVERY=100' $OUT/fq_*.sbatch | wc -l)"
echo "with --target-kl                   : $(grep -l -- '--target-kl 0.1' $OUT/fq_*.sbatch | wc -l)"
echo "with --dual-clip                   : $(grep -l -- '--dual-clip 3.0' $OUT/fq_*.sbatch | wc -l)"
echo "total                              : $(ls -1 $OUT/fq_*.sbatch | wc -l)"

echo "=== 6. the one named row, dsnn-dfw report needs it ==="
grep -n -- '--face-wire-faces\|ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY\|ALPHAGRAD_RECYCLE_RETRY_ON_OOM\|ALPHAGRAD_PROACTIVE_RECYCLE_EVERY' \
    $OUT/fq_condC_tlm_s250197.sbatch
echo "===== DONE ====="
