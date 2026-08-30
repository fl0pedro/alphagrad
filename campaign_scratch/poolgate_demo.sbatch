#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=96G
#SBATCH -c 24
#SBATCH -t 2:00:00
#SBATCH -J poolgate
#SBATCH -o /Users/assmuth/dsnn/alphagrad/poolgate_demo_%j.log
# ONE allocation, three phases, sequential. Nothing else runs on this node.
#   A. gate on the PRE-FIX tree (87cdc49 reverted in a SCRATCH COPY) -> must be RED
#   B. gate at HEAD                                                  -> must be GREEN
# The repo's own src/ is NEVER modified: the pre-fix tree is a copy under
# /Users/assmuth/dsnn/poolgate_scratch and is reached with PYTHONPATH.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
REPO=/Users/assmuth/dsnn/alphagrad
SC=/Users/assmuth/dsnn/poolgate_scratch
VENVPY=/Users/assmuth/dsnn/.venv/bin/python
cd $REPO

echo "########## host=$(hostname) job=$SLURM_JOB_ID"
echo "########## HEAD=$(git rev-parse --short HEAD)"

echo
echo "############################################################"
echo "########## PHASE A: PRE-FIX TREE (87cdc49 reverted) #########"
echo "############################################################"
rm -rf $SC
mkdir -p $SC
cp -r src tools $SC/
git show 87cdc49 -- src/alphagrad/approx/cpu_approx_actors.py | patch -R -p1 -d $SC
echo "PATCH_RC=$?"
echo "--- the ONLY difference between the scratch tree and HEAD: ---"
diff -u src/alphagrad/approx/cpu_approx_actors.py \
        $SC/src/alphagrad/approx/cpu_approx_actors.py
echo "--- (diff rc=$? ; 1 == they differ, as intended) ---"
echo "--- repo working tree src/ is UNTOUCHED: ---"
git status --short src/

cd $SC
T0=$(date +%s)
PYTHONPATH=$SC/src PY=$VENVPY POOL_GATE_OUT=$SC/out \
  RAY_TMPDIR=/tmp/ray_poolgate_A_$SLURM_JOB_ID \
  bash $SC/tools/pool_liveness_gate.sh
RED_RC=$?
T1=$(date +%s)
echo "PHASE_A_GATE_RC=$RED_RC   wall=$((T1-T0))s   (MUST be non-zero)"
echo "--- proof the PATCHED source is what the actors ran (TypeError text): ---"
grep -c "unexpected keyword argument 'episode'" $SC/out/pool_liveness_run.log
grep -m3 "unexpected keyword argument 'episode'" $SC/out/pool_liveness_run.log
echo "--- plan log size: ---"
wc -l -c $SC/out/pool_liveness_plan_log.jsonl 2>&1

echo
echo "############################################################"
echo "########## PHASE B: HEAD                          ##########"
echo "############################################################"
cd $REPO
T2=$(date +%s)
PY="uv run --no-sync python" POOL_GATE_OUT=$REPO/pool_gate_out \
  RAY_TMPDIR=/tmp/ray_poolgate_B_$SLURM_JOB_ID \
  bash tools/pool_liveness_gate.sh
GREEN_RC=$?
T3=$(date +%s)
echo "PHASE_B_GATE_RC=$GREEN_RC   wall=$((T3-T2))s   (MUST be 0)"

echo
echo "############################################################"
echo "RED_ON_PREFIX_RC=$RED_RC   GREEN_ON_HEAD_RC=$GREEN_RC"
echo "PHASE_A_WALL=$((T1-T0))s   PHASE_B_WALL=$((T3-T2))s"
echo "--- repo working tree src/ after everything: ---"
git -C $REPO status --short src/
echo "########## POOLGATE DEMO DONE ##########"
