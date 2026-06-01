#!/bin/bash
#SBATCH --job-name=full_approx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Master research run — sequentially executes RQ1 → RQ5 and the
# downstream-MNIST evaluation for every recorded best sequence.
# Allocated 14 days; expected wall time ~7–10 days at 500 episodes
# per RL run + 10k MNIST steps per downstream eval.
#
# **Resumable**: each phase writes a marker file in $STATE_DIR/done_<phase>
# when it completes. Restarting the script (or re-sbatch'ing it after a
# timeout / preempt) picks up at the first unfinished phase. Set
# RESET_STATE=1 to clear all markers and start over.
#
# **Per-phase overrides**: each phase is a single ``bash <script>`` call;
# adjust env vars like EPISODES, NUM_ENVS, SEEDS by editing the
# individual scripts. Defaults give the matrix described in
# ``docs/experiments/research_plan.md``.
#
# **Total runs** (per individual script's header):
#   RQ1: 27 PPO runs (ve_only × 3 examples × 3 cmp-types × 3 seeds)
#   RQ2: 18 PPO runs (6 family/argument-policy × 3 seeds)
#   RQ3: 36 PPO runs (3 families × 4 max-rules × 3 seeds)
#   RQ4: 18 PPO runs (3 mixings × 2 curricula × 3 seeds)
#   RQ5: 15 PPO runs (5 anti-degeneracy configs × 3 seeds)
#   = 114 PPO training runs at 500 episodes each
#
# **Downstream evaluation** at the end: every recorded best sequence
# from every RQ phase is piped through downstream_train.py to measure
# real MNIST test accuracy. Baselines (jax_grad, graphax_rev) included.

set -uo pipefail  # NOT -e: per-phase failures must not abort the whole batch
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

STATE_DIR=~/dsnn/logs_full_research
MASTER_LOG="$STATE_DIR/master.log"
mkdir -p "$STATE_DIR" slurm

# Allow a clean restart via RESET_STATE=1.
if [[ "${RESET_STATE:-0}" == "1" ]]; then
    echo "[$(date +%F-%H:%M:%S)] RESET_STATE=1 — clearing phase markers."
    rm -f "$STATE_DIR"/done_*
fi

# Log helper. Both stdout (visible in slurm output) and master.log
# (persistent record across slurm restarts).
log() {
    echo "[$(date +%F-%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

phase_done() {
    [[ -f "$STATE_DIR/done_$1" ]]
}

mark_done() {
    date +%F-%H:%M:%S > "$STATE_DIR/done_$1"
}

# Wrapper: run a phase script, capture its exit code, log it. Only
# mark the phase done on rc=0 so a failed phase (e.g. bad CLI flag)
# is automatically retried on the next sbatch resubmit. Per-variant
# logs live inside each script's own log_dir (logs_rq1_ve_only/,
# logs_rq2_per_family/, etc.). Subsequent phases still start even if
# the prior one failed — a broken RQ2 shouldn't block RQ3.
run_phase() {
    local name=$1
    local cmd=$2
    if phase_done "$name"; then
        log "phase=$name: already marked done, skipping"
        return 0
    fi
    log "phase=$name: starting"
    local start_ts=$(date +%s)
    bash -c "$cmd" >> "$MASTER_LOG" 2>&1
    local rc=$?
    local end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    if [[ $rc -eq 0 ]]; then
        log "phase=$name: ok (rc=0, elapsed=${elapsed}s)"
        mark_done "$name"
    else
        log "phase=$name: FAILED (rc=$rc, elapsed=${elapsed}s) — not marking done so a resubmit retries"
    fi
}

log "================================================================="
log "Full research run starting on $(hostname)"
log "STATE_DIR=$STATE_DIR  MASTER_LOG=$MASTER_LOG"
log "Resumable: $(ls "$STATE_DIR"/done_* 2>/dev/null | wc -l) phases already complete"
log "================================================================="

# ---------------------------------------------------------------------
# Phase 1: RQ1 — foundation sanity check (27 runs)
# ---------------------------------------------------------------------
run_phase rq1 'bash alphagrad/run_rq1_ve_only.sh'

# Aggregate RQ1 best_sequences against the graphax_rev baseline.
run_phase rq1_analyse \
    'uv run --no-sync alphagrad/src/alphagrad/approx/analyse_rq1.py \
        --wandb-dir ~/dsnn/wandb \
        --log-dir ~/dsnn/logs_rq1_ve_only \
        --output-csv ~/dsnn/out/rq1_summary.csv'

# ---------------------------------------------------------------------
# Phase 2: RQ2 — per-family sweep (18 runs)
# ---------------------------------------------------------------------
run_phase rq2 'bash alphagrad/run_rq2_per_family.sh'

# ---------------------------------------------------------------------
# Phase 3: RQ3 — dynamic per-vertex stack (36 runs)
# Uses placeholder FAMILY_WINNERS at the top of run_rq3 — review RQ2
# downstream results and edit before this phase if you want
# winner-driven selection. Default is to sweep all 3 families.
# ---------------------------------------------------------------------
run_phase rq3 'bash alphagrad/run_rq3_dynamic_stack.sh'

# ---------------------------------------------------------------------
# Phase 4: RQ4 — mixed families × curriculum (18 runs)
# Same caveat: SINGLE_WINNER / RQ3_BEST_MAX_RULES are placeholders.
# ---------------------------------------------------------------------
run_phase rq4 'bash alphagrad/run_rq4_mixed_families.sh'

# ---------------------------------------------------------------------
# Phase 5: RQ5 — anti-degeneracy mechanism sweep (15 runs)
# ---------------------------------------------------------------------
run_phase rq5 'bash alphagrad/run_rq5_anti_degeneracy.sh'

# ---------------------------------------------------------------------
# Phase 6: downstream MNIST evaluation for every recorded best sequence
# across all five RQ phases. Each variant runs downstream_train.py for
# $TRAIN_STEPS MNIST steps (default 10000) and writes a CSV the
# plot_downstream.py renderer consumes.
# ---------------------------------------------------------------------
run_phase downstream_eval \
    'bash alphagrad/run_downstream_eval_all.sh'

# ---------------------------------------------------------------------
# Phase 7: aggregate plots — produces loss / acc / wall / peak_mem /
# cossim PNGs + summary.csv keyed by gradient source.
# ---------------------------------------------------------------------
run_phase downstream_plots \
    'uv run --no-sync alphagrad/src/alphagrad/approx/plot_downstream.py \
        --csv-glob "$HOME/dsnn/out/downstream_full/*_seed*.csv" \
        --output-dir "$HOME/dsnn/out/downstream_full/plots"'

# ---------------------------------------------------------------------
# Phase 8: static-metric comparison table — cheap one-shot table over
# every recorded best sequence vs jax / graphax baselines.
# ---------------------------------------------------------------------
run_phase compare_baselines \
    'WANDB_ROOT="$HOME/dsnn/wandb";
     mapfile -t ALL_RUNS < <(
         for d in "$WANDB_ROOT"/run-*; do
             meta="$d/files/wandb-metadata.json"
             [[ -f "$meta" ]] && grep -qE "\"rq[1-5]_" "$meta" && echo "$d"
         done
     );
     if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
         echo "[compare] no rq[1-5]_* wandb runs found — running baseline-only table";
         uv run --no-sync alphagrad/src/alphagrad/approx/compare_baselines.py \
            --example VmappedNeuralNetwork \
            --dataset mnist \
            --wandb-runs \
            --channels best_overall \
            --measure-latency \
            --wandb offline \
            --wandb-project dsnn-compare-baselines \
            --name compare_baselines_only_$(date +%Y%m%d) \
            --output-json "$HOME/dsnn/out/compare_full.json" \
            --output-csv "$HOME/dsnn/out/compare_full.csv";
     else
         uv run --no-sync alphagrad/src/alphagrad/approx/compare_baselines.py \
            --example VmappedNeuralNetwork \
            --dataset mnist \
            --wandb-runs "${ALL_RUNS[@]}" \
            --measure-latency \
            --wandb offline \
            --wandb-project dsnn-compare-baselines \
            --name compare_full_$(date +%Y%m%d) \
            --output-json "$HOME/dsnn/out/compare_full.json" \
            --output-csv "$HOME/dsnn/out/compare_full.csv";
     fi'

log "================================================================="
log "Full research run finished. State markers:"
ls -la "$STATE_DIR"/done_* 2>/dev/null | tee -a "$MASTER_LOG"
log "Outputs:"
log "  ~/dsnn/out/rq1_summary.csv"
log "  ~/dsnn/out/downstream_full/*.csv"
log "  ~/dsnn/out/downstream_full/plots/*.png"
log "  ~/dsnn/out/compare_full.{json,csv}"
log "================================================================="
