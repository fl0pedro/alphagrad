#!/bin/bash
# t284 round 2, CPU: the growing-broadcast census by case, and the CPU side of
# the copy attribution.
#SBATCH -J t284-cpu2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
RUN=$HOME/agents/t284/cpu2-$SLURM_JOB_ID
mkdir -p "$RUN"
echo "[t284] job=$SLURM_JOB_ID host=$(hostname -s) $(date -u +%FT%TZ)"
export PYTHONPATH=$HOME/agents/t284-graphax/src:$HOME/agents/t284-alphagrad/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_$(hostname -s)
export WANDB_MODE=disabled
source "$HOME/agents/t284/t65_shared_env.sh"
unset ALPHAGRAD_FORCE_REV_ORDER
PY=$HOME/dsnn/.venv/bin/python
P=$HOME/agents/t284
echo "[t284] ag=$(git -C $HOME/agents/t284-alphagrad rev-parse --short HEAD) gx=$(git -C $HOME/agents/t284-graphax rev-parse --short HEAD)"
cd "$HOME/agents/t284-alphagrad"

echo "[t284] ===== growing-broadcast census by case ====="
t0=$SECONDS
timeout 3600 "$PY" "$P/t284_broadcast_census.py" "$RUN/census" > "$RUN/census.log" 2>&1
echo "[t284] census rc=$? in $((SECONDS-t0))s"
grep -a "^\[t284c\]" "$RUN/census.log"

echo "[t284] ===== copy attribution (CPU) ====="
t0=$SECONDS
timeout 1800 "$PY" "$P/t284_copy_attribution.py" "$RUN/copy" cpu > "$RUN/copy.log" 2>&1
echo "[t284] copy rc=$? in $((SECONDS-t0))s"
grep -a "^\[t284k\]" "$RUN/copy.log"
echo "[t284] done $(date -u +%FT%TZ) RUN=$RUN"
exit 0
