#!/bin/bash
# t28a (race lane A, ticket dsnn-3qm.66): the landing test on CPU.
# pgi15-cpu1 is the one node with the home directory, so this runs from the
# lane's home clones ~/agents/t28a-{alphagrad,graphax} and the live venv.
#
#   sbatch -p pgi15-cpu -w pgi15-cpu1 --export=ALL,EXAMPLE=TransformerLM,DATASET=wikitext2 t65_cpu.sbatch
#   sbatch -p pgi15-cpu -w pgi15-cpu1 --export=ALL,EXAMPLE=NeuralNetwork,DATASET=mnist     t65_cpu.sbatch
#SBATCH -J t28a-cpu
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
EXAMPLE=${EXAMPLE:-TransformerLM}
DATASET=${DATASET:-wikitext2}
AG=$HOME/agents/t28a-alphagrad
GX=$HOME/agents/t28a-graphax
RUN=$HOME/agents/t28a/${EXAMPLE}-cpu-$SLURM_JOB_ID
mkdir -p "$RUN"
echo "[t28a] job=$SLURM_JOB_ID host=$(hostname -s) example=$EXAMPLE dataset=$DATASET $(date -u +%FT%TZ)"
export PYTHONPATH=$GX/src:$AG/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_$(hostname -s)
export WANDB_MODE=disabled
source "$HOME/agents/t28a/t65_shared_env.sh"
export ALPHAGRAD_MEASURE_ACTOR=0
PY=$HOME/dsnn/.venv/bin/python
echo "[t28a] ag=$(git -C $AG rev-parse --short HEAD) gx=$(git -C $GX rev-parse --short HEAD)"
$PY -c "import alphagrad, graphax; print('[t28a]', alphagrad.__file__); print('[t28a]', graphax.__file__)"

cd "$AG"
t0=$SECONDS
timeout 14000 "$PY" "$HOME/agents/t28a/t65_engine_probe.py" "$RUN" "$EXAMPLE" "$DATASET" > "$RUN/probe.log" 2>&1
RC=$?
echo "[t28a] probe rc=$RC in $((SECONDS-t0))s  log=$RUN/probe.log"
grep -a '"phase": "floor"\|"phase": "summary"\|"phase": "oracle_jax_grad"\|"error"' "$RUN/probe.log" | cut -c1-600
grep -a "^\[t65\] =====\|^\[t65\] drift\|^\[t65\] order\|^\[t65\] reverse\|^\[t65\] markowitz" "$RUN/probe.log"
[ $RC -ne 0 ] && tail -60 "$RUN/probe.log"
echo "[t28a] done $(date -u +%FT%TZ)"
exit $RC
