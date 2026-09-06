#!/bin/bash
# t284: the three emissions of a single implicit sparse axis, GPU, one job.
# Runs from the /Scratch stack of finding 57 (no home on the GPU nodes) and
# from the lane F clones of wip/t284-20260906.
#
#   sbatch t284_gpu.sbatch
#SBATCH -J t284-gpu
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu15
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/t284
#SBATCH -o /Scratch/assmuth/t57/runs/t284/%x-%j.out
set -uo pipefail
T57=/Scratch/assmuth/t57
LOCAL=/tmp/t284_$SLURM_JOB_ID
RUN=$T57/runs/t284/gpu-$SLURM_JOB_ID
trap 'echo "[cleanup] rm -rf $LOCAL"; rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL" "$RUN"
echo "[t284] job=$SLURM_JOB_ID host=$(hostname -s) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/[t284] gpu: /'

STACK=$T57/stack
export T57_LOCAL=$LOCAL WANDB_DIR=$RUN WANDB_MODE=disabled
source "$T57/probes/t57_env.sh" "$STACK" "$T57/home" || exit $?
source "$T57/probes/t284/t65_shared_env.sh"
export PYTHONPATH=$T57/agents/t284-graphax/src:$T57/agents/t284-alphagrad/src
export ALPHAGRAD_MEASURE_ACTOR=1
echo "[t284] ag=$(git -C $T57/agents/t284-alphagrad rev-parse --short HEAD) gx=$(git -C $T57/agents/t284-graphax rev-parse --short HEAD)"
"$PY" -c "import alphagrad, graphax; print(alphagrad.__file__); print(graphax.__file__)"

cd "$T57/agents/t284-alphagrad"
P=$T57/probes/t284

# --- deliverable (a): the synthetic three-emission sweep -------------------
echo "[t284] ===== sweep (GPU) ====="
t0=$SECONDS
timeout 3000 "$PY" "$P/t284_three_emissions_sweep.py" "$RUN/sweep" gpu > "$RUN/sweep.log" 2>&1
echo "[t284] sweep rc=$? in $((SECONDS-t0))s"
grep -a "^\[t284\] M=" "$RUN/sweep.log" | tail -90
tail -5 "$RUN/sweep.log"

# --- deliverable (b): the three emissions in the real engine ---------------
echo "[t284] ===== engine probe, TransformerLM (campaign shape) ====="
t0=$SECONDS
timeout 7200 "$PY" "$P/t284_emission_probe.py" "$RUN/tlm" TransformerLM wikitext2 \
  > "$RUN/tlm.log" 2>&1
RC1=$?
echo "[t284] tlm rc=$RC1 in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/tlm.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] reverse\|^\[t284\] markowitz\|^\[t284\] order" "$RUN/tlm.log"
[ $RC1 -ne 0 ] && tail -40 "$RUN/tlm.log"

echo "[t284] ===== engine probe, NeuralNetwork (mnist) ====="
t0=$SECONDS
T284_ARMS=bcast_in_dot,kept_on_storing_side,mul_then_reduce,legacy \
  timeout 3600 "$PY" "$P/t284_emission_probe.py" "$RUN/nn" NeuralNetwork mnist \
  > "$RUN/nn.log" 2>&1
RC2=$?
echo "[t284] nn rc=$RC2 in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/nn.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] reverse\|^\[t284\] markowitz\|^\[t284\] order" "$RUN/nn.log"
[ $RC2 -ne 0 ] && tail -40 "$RUN/nn.log"

echo "[t284] done $(date -u +%FT%TZ) RUN=$RUN"
exit 0
