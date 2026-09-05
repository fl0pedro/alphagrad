#!/bin/bash
# t28b: the landing test of the .28 race, lane B (ticket dsnn-3qm.67), GPU.
# Runs from the /Scratch stack of finding 57 (no home on the GPU nodes).
#
#   sbatch --export=ALL,EXAMPLE=TransformerLM,DATASET=wikitext2 t65_gpu.sbatch
#   sbatch --export=ALL,EXAMPLE=NeuralNetwork,DATASET=mnist     t65_gpu.sbatch
#SBATCH -J t28b-gpu
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/t28b
#SBATCH -o /Scratch/assmuth/t57/runs/t28b/%x-%j.out
set -uo pipefail
EXAMPLE=${EXAMPLE:-TransformerLM}
DATASET=${DATASET:-wikitext2}
T57=/Scratch/assmuth/t57
LOCAL=/tmp/t28b_$SLURM_JOB_ID
RUN=$T57/runs/t28b/${EXAMPLE}-gpu-$SLURM_JOB_ID
trap 'echo "[cleanup] rm -rf $LOCAL"; rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL" "$RUN"
echo "[t65] job=$SLURM_JOB_ID host=$(hostname -s) example=$EXAMPLE dataset=$DATASET $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/[t65] gpu: /'

STACK=$T57/stack
export T57_LOCAL=$LOCAL WANDB_DIR=$RUN WANDB_MODE=disabled
source "$T57/probes/t57_env.sh" "$STACK" "$T57/home" || exit $?
source "$T57/probes/t28b/t65_shared_env.sh"
# lane B code: the /Scratch clones of wip/t28b-20260905, not the stack trees.
export PYTHONPATH=$T57/agents/t28b-graphax/src:$T57/agents/t28b-alphagrad/src
export ALPHAGRAD_MEASURE_ACTOR=1
echo "[t28b] ag=$(git -C $T57/agents/t28b-alphagrad rev-parse --short HEAD) gx=$(git -C $T57/agents/t28b-graphax rev-parse --short HEAD)"
"$PY" -c "import alphagrad, graphax; print(alphagrad.__file__); print(graphax.__file__)"

cd "$T57/agents/t28b-alphagrad"
t0=$SECONDS
timeout 6600 "$PY" "$T57/probes/t28b/t28b_engine_probe.py" "$RUN" "$EXAMPLE" "$DATASET" > "$RUN/probe.log" 2>&1
RC=$?
echo "[t65] probe rc=$RC in $((SECONDS-t0))s  log=$RUN/probe.log"
grep -a '"phase": "floor"\|"phase": "summary"\|"phase": "oracle_jax_grad"\|"error"' "$RUN/probe.log" | cut -c1-600
grep -a "^\[t65\] =====\|^\[t65\] drift\|^\[t65\] order\|^\[t65\] reverse\|^\[t65\] markowitz" "$RUN/probe.log"
[ $RC -ne 0 ] && tail -60 "$RUN/probe.log"
echo "[t65] done $(date -u +%FT%TZ)"
exit $RC
