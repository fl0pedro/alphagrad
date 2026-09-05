#!/bin/bash
# t28b: the landing test of the .28 race, lane B (ticket dsnn-3qm.67), CPU.
# pgi15-cpu2 has no home directory (finding 57), so everything comes from the
# /Scratch stack, the way probes/ag_suite_scratch.sbatch does it.
#
#   sbatch --export=ALL,EXAMPLE=TransformerLM,DATASET=wikitext2 t28b_cpu.sbatch
#   sbatch --export=ALL,EXAMPLE=NeuralNetwork,DATASET=mnist     t28b_cpu.sbatch
#SBATCH -J t28b-cpu
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -D /Scratch/assmuth/t57/runs/t28b
#SBATCH -o /Scratch/assmuth/t57/runs/t28b/%x-%j.out
set -uo pipefail
EXAMPLE=${EXAMPLE:-TransformerLM}
DATASET=${DATASET:-wikitext2}
T57=/Scratch/assmuth/t57
STACK=$T57/stack
LOCAL=/tmp/t28b_$SLURM_JOB_ID
RUN=$T57/runs/t28b/${EXAMPLE}-cpu-$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT TERM INT
mkdir -p "$LOCAL/home" "$LOCAL/jaxcache" "$LOCAL/tmp" "$RUN"
# the wikitext cache and the wandb settings of the stack home (finding 57)
cp -a "$T57/home/." "$LOCAL/home/"
export HOME=$LOCAL/home TMPDIR=$LOCAL/tmp
echo "[t28b] job=$SLURM_JOB_ID host=$(hostname -s) example=$EXAMPLE dataset=$DATASET $(date -u +%FT%TZ)"
grep -qx "home = $STACK/uv-python/bin" "$STACK/venv/pyvenv.cfg" || sed -i "s|^home = .*|home = $STACK/uv-python/bin|" "$STACK/venv/pyvenv.cfg"
export PY=$STACK/venv/bin/python
export PATH=$STACK/venv/bin:$STACK/uv-python/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$T57/agents/t28b-graphax/src:$T57/agents/t28b-alphagrad/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$LOCAL/jaxcache
export WANDB_MODE=disabled WANDB_DIR=$RUN
source "$T57/probes/t28b/t65_shared_env.sh"
export ALPHAGRAD_MEASURE_ACTOR=0
echo "[t28b] ag=$(git -C $T57/agents/t28b-alphagrad rev-parse --short HEAD) gx=$(git -C $T57/agents/t28b-graphax rev-parse --short HEAD)"
"$PY" -c "import alphagrad, graphax; print(alphagrad.__file__); print(graphax.__file__)"

cd "$T57/agents/t28b-alphagrad"
t0=$SECONDS
timeout 14000 "$PY" "$T57/probes/t28b/t28b_engine_probe.py" "$RUN" "$EXAMPLE" "$DATASET" > "$RUN/probe.log" 2>&1
RC=$?
echo "[t28b] probe rc=$RC in $((SECONDS-t0))s  log=$RUN/probe.log"
grep -a '"phase": "floor"\|"phase": "summary"\|"phase": "oracle_jax_grad"\|"error"' "$RUN/probe.log" | cut -c1-600
grep -a "^\[t65\] =====\|^\[t65\] drift\|^\[t65\] order\|^\[t65\] reverse\|^\[t65\] markowitz" "$RUN/probe.log"
[ $RC -ne 0 ] && tail -60 "$RUN/probe.log"
echo "[t28b] done $(date -u +%FT%TZ)"
exit $RC
