#!/bin/bash
#SBATCH --job-name=t28b-gx
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/suites
#SBATCH -o /Scratch/assmuth/t57/runs/suites/%x-%j.out
# graphax's own pytest suite on the lane-B clone under /Scratch, from the stack
# venv.  Copy of probes/gx_suite_scratch.sbatch with the node and the job name
# changed (ticket dsnn-3qm.67).
set -uo pipefail
GX=${GX:-/Scratch/assmuth/t57/agents/t28b-graphax}
STACK=/Scratch/assmuth/t57/stack
LOCAL=/tmp/gxsuite_$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL/home" "$LOCAL/jaxcache" "$LOCAL/tmp"
export HOME=$LOCAL/home TMPDIR=$LOCAL/tmp
grep -qx "home = $STACK/uv-python/bin" "$STACK/venv/pyvenv.cfg" || sed -i "s|^home = .*|home = $STACK/uv-python/bin|" "$STACK/venv/pyvenv.cfg"
export PY=$STACK/venv/bin/python
export PATH=$STACK/venv/bin:$STACK/uv-python/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$GX/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$LOCAL/jaxcache
cd "$GX"
echo "node=$(hostname -s) gx=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
$PY -c "import graphax; print(graphax.__file__)"
$PY -m pytest tests -q -p no:cacheprovider ${PYTEST_ARGS:-} 2>&1 | tail -60
echo "pytest exit=${PIPESTATUS[0]}"
