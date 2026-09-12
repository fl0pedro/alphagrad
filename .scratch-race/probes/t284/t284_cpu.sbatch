#!/bin/bash
# t284: the three emissions of a single implicit sparse axis, CPU, one job.
# pgi15-cpu1 is the only node with the home directory mounted (finding 57).
#
#   sbatch -p pgi15-cpu -w pgi15-cpu1 t284_cpu.sbatch
#SBATCH -J t284-cpu
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 05:00:00
#SBATCH -o %x-%j.out
set -uo pipefail
RUN=$HOME/agents/t284/cpu-$SLURM_JOB_ID
mkdir -p "$RUN"
echo "[t284] job=$SLURM_JOB_ID host=$(hostname -s) $(date -u +%FT%TZ)"
export PYTHONPATH=$HOME/agents/t284-graphax/src:$HOME/agents/t284-alphagrad/src
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_$(hostname -s)
export WANDB_MODE=disabled
source "$HOME/agents/t284/t65_shared_env.sh"
# dsnn-3qm.64 deleted this env var; masks.py raises when it is set.
unset ALPHAGRAD_FORCE_REV_ORDER
export ALPHAGRAD_MEASURE_ACTOR=0
PY=$HOME/dsnn/.venv/bin/python
P=$HOME/agents/t284
echo "[t284] ag=$(git -C $HOME/agents/t284-alphagrad rev-parse --short HEAD) gx=$(git -C $HOME/agents/t284-graphax rev-parse --short HEAD)"
$PY -c "import alphagrad, graphax; print('[t284]', alphagrad.__file__, graphax.__file__)"

cd "$HOME/agents/t284-alphagrad"

# --- deliverable (a): the synthetic three-emission sweep -------------------
if [ "${T284_SKIP_SWEEP:-0}" = "0" ]; then
echo "[t284] ===== sweep (CPU) ====="
t0=$SECONDS
timeout 4000 "$PY" "$P/t284_three_emissions_sweep.py" "$RUN/sweep" cpu > "$RUN/sweep.log" 2>&1
echo "[t284] sweep rc=$? in $((SECONDS-t0))s"
grep -a "^\[t284\] M=" "$RUN/sweep.log" | tail -90
tail -5 "$RUN/sweep.log"
else
echo "[t284] sweep skipped (T284_SKIP_SWEEP=1)"
fi

# --- deliverable (b): the three emissions in the real engine ---------------
# TLM reverse takes the campaign protocol; TLM Markowitz is 100x slower per
# execution (finding 61 verdict 7), so it takes the shrunk protocol and its
# ratios read to two digits.
echo "[t284] ===== engine probe, TransformerLM, reverse (campaign protocol) ====="
t0=$SECONDS
T284_ORDERS=reverse timeout 9000 "$PY" "$P/t284_emission_probe.py" \
  "$RUN/tlm_rev" TransformerLM wikitext2 > "$RUN/tlm_rev.log" 2>&1
RC1=$?
echo "[t284] tlm_rev rc=$RC1 in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/tlm_rev.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] reverse" "$RUN/tlm_rev.log"
[ $RC1 -ne 0 ] && tail -40 "$RUN/tlm_rev.log"

echo "[t284] ===== engine probe, TransformerLM, markowitz (shrunk protocol) ====="
t0=$SECONDS
T284_ORDERS=markowitz T284_INNER=5 T284_POINTS=1 T284_REPS=2 T284_ROUNDS=2 \
  timeout 9000 "$PY" "$P/t284_emission_probe.py" \
  "$RUN/tlm_mk" TransformerLM wikitext2 > "$RUN/tlm_mk.log" 2>&1
RC2=$?
echo "[t284] tlm_mk rc=$RC2 in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/tlm_mk.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] markowitz" "$RUN/tlm_mk.log"
[ $RC2 -ne 0 ] && tail -40 "$RUN/tlm_mk.log"

echo "[t284] ===== engine probe, NeuralNetwork (mnist) ====="
t0=$SECONDS
T284_ARMS=bcast_in_dot,kept_on_storing_side,mul_then_reduce,legacy \
  timeout 3600 "$PY" "$P/t284_emission_probe.py" "$RUN/nn" NeuralNetwork mnist \
  > "$RUN/nn.log" 2>&1
RC3=$?
echo "[t284] nn rc=$RC3 in $((SECONDS-t0))s"
grep -a '"phase": "floor"\|"phase": "summary"\|"error"' "$RUN/nn.log" | cut -c1-900
grep -a "^\[t284\] =====\|^\[t284\] drift\|^\[t284\]   \|^\[t284\] reverse\|^\[t284\] markowitz" "$RUN/nn.log"
[ $RC3 -ne 0 ] && tail -40 "$RUN/nn.log"

echo "[t284] done $(date -u +%FT%TZ) RUN=$RUN"
exit 0
