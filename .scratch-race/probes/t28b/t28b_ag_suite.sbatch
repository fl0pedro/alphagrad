#!/bin/bash
#SBATCH --job-name=t28b-cpu
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/suites
#SBATCH -o /Scratch/assmuth/t57/runs/suites/%x-%j.out
# alphagrad per-module suite on the integration code with the lane-B graphax,
# from the /Scratch stack (pgi15-cpu2 has no home).  Copy of
# probes/ag_suite_scratch.sbatch (ticket dsnn-3qm.67).
set -uo pipefail
CLONE=${CLONE:-t28b-alphagrad}
STACK=/Scratch/assmuth/t57/stack
LOCAL=/tmp/suite_$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL/home" "$LOCAL/jaxcache" "$LOCAL/tmp"
cp -a /Scratch/assmuth/t57/home/. "$LOCAL/home/"
export HOME=$LOCAL/home TMPDIR=$LOCAL/tmp
grep -qx "home = $STACK/uv-python/bin" "$STACK/venv/pyvenv.cfg" || sed -i "s|^home = .*|home = $STACK/uv-python/bin|" "$STACK/venv/pyvenv.cfg"
export PY=$STACK/venv/bin/python
export PATH=$STACK/venv/bin:$STACK/uv-python/bin:/usr/local/bin:/usr/bin:/bin
GX=${GX:-/Scratch/assmuth/t57/agents/t28b-graphax}
export PYTHONPATH=/Scratch/assmuth/t57/agents/$CLONE/src:$GX/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$LOCAL/jaxcache
export WANDB_MODE=disabled
cd /Scratch/assmuth/t57/agents/$CLONE
echo "node=$(hostname -s) sha=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) gx=$(git -C $GX rev-parse --short HEAD) gxbranch=$(git -C $GX rev-parse --abbrev-ref HEAD)"
$PY -c "import alphagrad, graphax; print(alphagrad.__file__); print(graphax.__file__)"
JOBS=8 OUT=/Scratch/assmuth/t57/runs/suites/logs-$SLURM_JOB_ID bash tools/suite_per_module.sh ${MODULES:-}
echo "suite exit=$?"
