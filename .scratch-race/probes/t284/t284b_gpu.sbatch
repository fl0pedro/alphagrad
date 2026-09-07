#!/bin/bash
# t284 round 2, GPU: attribute the 10 percent (owner D12).
#SBATCH -J t284-gpu2
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu15
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -D /Scratch/assmuth/t57/runs/t284
#SBATCH -o /Scratch/assmuth/t57/runs/t284/%x-%j.out
set -uo pipefail
T57=/Scratch/assmuth/t57
LOCAL=/tmp/t284b_$SLURM_JOB_ID
RUN=$T57/runs/t284/gpu2-$SLURM_JOB_ID
trap 'echo "[t284] cleanup"; rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL" "$RUN"
echo "[t284] job=$SLURM_JOB_ID host=$(hostname -s) $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/[t284] gpu: /'
STACK=$T57/stack
export T57_LOCAL=$LOCAL WANDB_DIR=$RUN WANDB_MODE=disabled
source "$T57/probes/t57_env.sh" "$STACK" "$T57/home" || exit $?
source "$T57/probes/t284/t65_shared_env.sh"
unset ALPHAGRAD_FORCE_REV_ORDER
export PYTHONPATH=$T57/agents/t284-graphax/src:$T57/agents/t284-alphagrad/src
export ALPHAGRAD_MEASURE_ACTOR=1
echo "[t284] ag=$(git -C $T57/agents/t284-alphagrad rev-parse --short HEAD) gx=$(git -C $T57/agents/t284-graphax rev-parse --short HEAD)"
cd "$T57/agents/t284-alphagrad"
P=$T57/probes/t284

echo "[t284] ===== copy attribution (GPU) ====="
t0=$SECONDS
timeout 1800 "$PY" "$P/t284_copy_attribution.py" "$RUN/copy" gpu > "$RUN/copy.log" 2>&1
echo "[t284] copy rc=$? in $((SECONDS-t0))s"
grep -a "^\[t284k\]" "$RUN/copy.log"

echo "[t284] ===== engine probe, TLM reverse only, exact + Quant ====="
t0=$SECONDS
T284_ORDERS=reverse T284_PLANS=exact,quant_all \
  timeout 3600 "$PY" "$P/t284_emission_probe.py" "$RUN/tlm_rev" TransformerLM wikitext2 \
  > "$RUN/tlm_rev.log" 2>&1
RC=$?
echo "[t284] tlm_rev rc=$RC in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/tlm_rev.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] reverse" "$RUN/tlm_rev.log"
[ $RC -ne 0 ] && tail -30 "$RUN/tlm_rev.log"
echo "[t284] done $(date -u +%FT%TZ) RUN=$RUN"
exit 0
