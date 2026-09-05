#!/bin/bash
# t28a (race lane A, ticket dsnn-3qm.66): the landing test on GPU.
# Same probe as t65 (finding 61); the "planner" arm is the CANDIDATE (the
# dot_general emission + plain dense x dense), the "tiled" arm is the
# INCUMBENT, untouched in this tree.  Runs from the /Scratch stack venv of
# finding 57 with THIS LANE's graphax and alphagrad clones on PYTHONPATH.
#
#   sbatch --export=ALL,EXAMPLE=TransformerLM,DATASET=wikitext2 t65_gpu.sbatch
#   sbatch --export=ALL,EXAMPLE=NeuralNetwork,DATASET=mnist     t65_gpu.sbatch
#SBATCH -J t28a-gpu
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu15
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/t28a
#SBATCH -o /Scratch/assmuth/t57/runs/t28a/%x-%j.out
set -uo pipefail
EXAMPLE=${EXAMPLE:-TransformerLM}
DATASET=${DATASET:-wikitext2}
T57=/Scratch/assmuth/t57
LOCAL=/tmp/t28a_$SLURM_JOB_ID
RUN=$T57/runs/t28a/${EXAMPLE}-gpu-$SLURM_JOB_ID
AG=$T57/agents/t28a-alphagrad
GX=$T57/agents/t28a-graphax
trap 'echo "[t28a] cleanup rm -rf $LOCAL"; rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL" "$RUN"
echo "[t28a] job=$SLURM_JOB_ID host=$(hostname -s) example=$EXAMPLE dataset=$DATASET $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/[t28a] gpu: /'

STACK=$T57/stack
export T57_LOCAL=$LOCAL WANDB_DIR=$RUN WANDB_MODE=disabled
source "$T57/probes/t57_env.sh" "$STACK" "$T57/home" || exit $?
# THIS LANE's trees, not the stack's.
export PYTHONPATH=$GX/src:$AG/src
source "$T57/probes/t28a/t65_shared_env.sh"
export ALPHAGRAD_MEASURE_ACTOR=1
echo "[t28a] ag=$(git -C $AG rev-parse --short HEAD) gx=$(git -C $GX rev-parse --short HEAD)"

cd "$AG"
"$PY" -c "import alphagrad, graphax; print('[t28a]', alphagrad.__file__); print('[t28a]', graphax.__file__)"
t0=$SECONDS
timeout 6600 "$PY" "$T57/probes/t28a/t65_engine_probe.py" "$RUN" "$EXAMPLE" "$DATASET" > "$RUN/probe.log" 2>&1
RC=$?
echo "[t28a] probe rc=$RC in $((SECONDS-t0))s  log=$RUN/probe.log"
grep -a '"phase": "floor"\|"phase": "summary"\|"phase": "oracle_jax_grad"\|"error"' "$RUN/probe.log" | cut -c1-600
grep -a "^\[t65\] =====\|^\[t65\] drift\|^\[t65\] order\|^\[t65\] reverse\|^\[t65\] markowitz" "$RUN/probe.log"
[ $RC -ne 0 ] && tail -60 "$RUN/probe.log"
echo "[t28a] done $(date -u +%FT%TZ)"
exit $RC
