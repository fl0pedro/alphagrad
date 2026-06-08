#!/bin/bash
#SBATCH --job-name=full_research
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --nodelist=pgi15-gpu15,pgi15-gpu16,pgi15-gpu17,pgi15-gpu18

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
#
# **CRITICAL**: every phase below trains / evaluates against
# ``VmappedNeuralNetwork + mnist`` exclusively. The recorded best
# sequences are defined over that jaxpr's vertex ordering; replaying
# against any other (example, dataset) would silently mismatch vertex
# indices. The downstream-train and compare-baselines helpers actively
# reject other values; the RL sbatches pass --example VmappedNeuralNetwork
# --dataset mnist as part of their COMMON_ARGS. Do not change.
#
#   RQ1:  9 PPO runs (ve_only × VmappedNeuralNetwork × 3 cmp-types × 3 seeds)
#   RQ2: 18 PPO runs (6 family/argument-policy × 3 seeds)
#   RQ3: 36 PPO runs (3 families × 4 max-rules × 3 seeds)
#   RQ4:  9 PPO runs (3 mixings × 3 seeds; curriculum axis dropped → RQ6)
#   RQ5: 15 PPO runs (5 anti-degeneracy configs × 3 seeds)
#   RQ6:  6 PPO runs (2 curricula × 3 seeds — granular additive vs none)
#   RQ7: 12 PPO runs (4 Dirichlet-α configs × 3 seeds)
#   = 105 PPO training runs at 500 episodes each (all on VmappedNeuralNetwork+mnist)
#
# **Downstream evaluation** at the end: every recorded best sequence
# from every RQ phase is piped through downstream_train.py to measure
# real MNIST test accuracy. Baselines (jax_grad, graphax_rev) included.

set -uo pipefail  # NOT -e: per-phase failures must not abort the whole batch
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

# Cross-job JAX-compile cache reuse — share the on-disk XLA artefact
# cache across resubmissions of this script on the same node. Without
# this, each new SLURM_JOB_ID gets a fresh /tmp/dsnn-jax-cache-<id>/
# and we re-pay the cold-compile cost on every restart. With it set,
# compiles land in /tmp/dsnn-jax-cache-shared-<hostname>/ and are
# reused across jobs. See common/cache.py for the path resolution.
# (Trade-off: cache stays warm across job sequences; correctness still
# guarded by JAX/XLA's internal HLO-hash keying, so different code
# revisions don't collide.)
export DSNN_JAX_CACHE_REUSE="${DSNN_JAX_CACHE_REUSE:-1}"

# wandb destination — the user's runs land in the public cloud entity
# ``dll-streetview`` (https://wandb.ai/dll-streetview). Defaults below
# can be overridden by setting these env vars before sbatch'ing.
#
# Why the explicit base URL: ``~/.config/wandb/settings`` on pgi15 was
# found containing the literal placeholder ``base_url = redacted``
# which fails wandb's pydantic URL validator and breaks ``wandb sync``
# (see master.log on 2026-05-22). Setting WANDB_BASE_URL via env var
# bypasses the broken file.
#
# To override for a single run:
#   sbatch -W WANDB_ENTITY=other_team WANDB_PROJECT=foo \
#       alphagrad/run_full_research.sh
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
# Per-project: each helper reads its own dedicated env var name so
# inheriting from a generic WANDB_PROJECT doesn't cross-contaminate.
export WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-dsnn-vertex}"          # ppo_ray / mu0_ray / gfn_ray training
export WANDB_PROJECT_DOWN="${WANDB_PROJECT_DOWN:-dsnn-downstream}"  # downstream_train.py per-step harness
export WANDB_PROJECT_CMP="${WANDB_PROJECT_CMP:-dsnn-compare-baselines}"  # compare_baselines.py static table
export WANDB_PROJECT_SEQ="${WANDB_PROJECT_SEQ:-rq-mnist-seq}"       # per-RQ nn_mnist.py eval (consumed by run_rq_eval_seqs.sh)
export WANDB_MODE="${WANDB_MODE:-online}"

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

# Upload every ``offline-run-*`` dir under ~/dsnn/wandb to the wandb
# cloud. All RL phases run with ``--wandb offline`` (so a missing
# login can't kill them); this step makes them visible on the
# dashboard. Run after every phase that produces new offline runs.
# Failures are logged but never block the master sbatch — a wandb-side
# outage shouldn't stop training.
WANDB_SYNC_LOG="$STATE_DIR/wandb_sync.log"
sync_wandb() {
    local phase="${1:-?}"
    local wandb_root="$HOME/dsnn/wandb"
    local n_offline
    n_offline=$(find "$wandb_root" -maxdepth 1 -name 'offline-run-*' -type d 2>/dev/null | wc -l)
    if [[ "$n_offline" -eq 0 ]]; then
        log "wandb sync (after $phase): no offline runs to sync"
        return 0
    fi

    # Pre-check: wandb settings file with literal ``redacted`` string
    # crashes ``wandb sync`` deep inside pydantic URL validation. Detect
    # and skip-with-actionable-message instead of failing per-call.
    local settings_file="$HOME/.config/wandb/settings"
    if [[ -f "$settings_file" ]] && grep -qE '^base_url\s*=\s*redacted\s*$' "$settings_file" \
       && [[ -z "${WANDB_BASE_URL:-}" ]]; then
        log "wandb sync (after $phase): SKIPPED — ~/.config/wandb/settings has 'base_url = redacted'"
        log "    Fix once, then re-run this sbatch:"
        log "      sed -i 's|^base_url.*|base_url = https://wandb.fz-juelich.de|' $settings_file"
        log "    (or pass WANDB_BASE_URL=<url> when sbatching the master script)"
        log "    The $n_offline offline run(s) under $wandb_root are safe on disk and can be"
        log "    synced manually later with ``wandb sync $wandb_root/offline-run-*``"
        return 0
    fi

    log "wandb sync (after $phase): syncing $n_offline offline run(s)"
    # ``wandb sync`` is idempotent — once a run has uploaded successfully
    # it's marked synced and re-running is a fast no-op. Run with a
    # generous timeout so a slow network doesn't hang the master sbatch.
    if timeout 600 uv run --no-sync wandb sync "$wandb_root"/offline-run-* \
            >> "$WANDB_SYNC_LOG" 2>&1; then
        log "wandb sync (after $phase): ok"
    else
        local rc=$?
        log "wandb sync (after $phase): rc=$rc (see $WANDB_SYNC_LOG) — continuing"
    fi
}

# Submit one of the per-RQ nn-mnist-seq eval scripts as its own slurm
# job, FIRE-AND-FORGET so the master keeps running the next RQ in
# parallel. To enforce the user's "at most 2 slurm jobs concurrent"
# rule, this function FIRST waits for the previous eval job (if any)
# to leave the slurm queue, THEN submits the new one.
#
# Net effect:
#   * While eval N is running, master is also running RQ_{N+1}
#     training inside its own job → 2 jobs total ✓
#   * When master next reaches submit_eval_phase, it pauses ONLY if
#     eval N is somehow still running (unlikely — RQ training is
#     usually slower) before submitting eval N+1.
#   * The eval's final exit code is checked async via a small
#     polling helper at end-of-script (so we still mark_done /
#     report failure cleanly).
#
# Each eval is a separate slurm job (1 GPU, 4 CPU, 8h walltime —
# matches nn-mnist-seq.bash's resource profile).
PREV_EVAL_JOBID=""
PREV_EVAL_PHASE=""

_wait_for_prev_eval() {
    [[ -z "$PREV_EVAL_JOBID" ]] && return 0
    # squeue lists active jobs; once the job has exited the queue it
    # disappears. Poll every 30s — eval should be much faster than RQ
    # training in steady state so this normally returns immediately.
    local poll=0
    while squeue -j "$PREV_EVAL_JOBID" -h -o "%T" 2>/dev/null | grep -qE 'RUNNING|PENDING|REQUEUED|SUSPENDED|CONFIGURING'; do
        if (( poll == 0 )); then
            log "waiting for prev eval job $PREV_EVAL_JOBID ($PREV_EVAL_PHASE) to exit queue (2-job cap)"
        fi
        sleep 30
        poll=$((poll + 1))
    done
    # Grace period — slurmctld can lag in flushing the just-finished
    # job's exit code to the sacct DB. Without this, sacct returns
    # empty and we'd mark the phase as failed even on a clean rc=0.
    sleep 5
    # Retry sacct up to 3 times with a short backoff in case the
    # db-flush is slow.
    local rc=""
    for retry in 1 2 3; do
        rc=$(sacct -j "$PREV_EVAL_JOBID" --noheader --parsable2 -o ExitCode 2>/dev/null \
                | head -1 | cut -d: -f1)
        if [[ -n "$rc" ]]; then
            break
        fi
        sleep 5
    done
    rc="${rc:-?}"
    if [[ "$rc" == "0" ]]; then
        log "prev eval $PREV_EVAL_PHASE (job $PREV_EVAL_JOBID) ok"
        mark_done "$PREV_EVAL_PHASE"
    elif [[ "$rc" == "?" ]]; then
        log "prev eval $PREV_EVAL_PHASE (job $PREV_EVAL_JOBID) — sacct didn't return an exit code; not marking done (resubmit will retry)"
    else
        log "prev eval $PREV_EVAL_PHASE (job $PREV_EVAL_JOBID) FAILED (rc=$rc) — not marking done"
    fi
    PREV_EVAL_JOBID=""
    PREV_EVAL_PHASE=""
}

submit_eval_phase() {
    local rq=$1
    local phase_name="rq${rq}_seq_eval"
    if phase_done "$phase_name"; then
        log "phase=$phase_name: already marked done, skipping"
        return 0
    fi
    # Drain the previous eval before submitting a new one so we never
    # exceed 2 concurrent jobs (master + 1 eval).
    _wait_for_prev_eval

    log "phase=$phase_name: submitting nested sbatch (fire-and-forget; master continues)"
    local jobid
    jobid=$(sbatch --parsable "alphagrad/run_rq_eval_seqs.sh" "$rq" 2>>"$MASTER_LOG")
    local rc=$?
    if [[ $rc -ne 0 || -z "$jobid" ]]; then
        log "phase=$phase_name: sbatch submission FAILED (rc=$rc)"
        return $rc
    fi
    log "phase=$phase_name: submitted as job $jobid; master proceeding to next RQ"
    PREV_EVAL_JOBID="$jobid"
    PREV_EVAL_PHASE="$phase_name"
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
# Phase 0: reference baselines (graphax fwd/rev) on the CPU partition.
# Fired FIRST, before any RL training, so the reference numbers exist
# in wandb while the RL search is still running. Eval job runs on
# pgi15-cpu and doesn't compete with the RL job for GPUs.
# ---------------------------------------------------------------------
submit_eval_phase ref

# ---------------------------------------------------------------------
# Phase 1: RQ1 — foundation sanity check (9 runs: VmappedNN × 3 cmp × 3 seed)
# ---------------------------------------------------------------------
run_phase rq1 'bash alphagrad/run_rq1_ve_only.sh'
sync_wandb rq1
submit_eval_phase 1

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
sync_wandb rq2
submit_eval_phase 2

# ---------------------------------------------------------------------
# Phase 3: RQ3 — dynamic per-vertex stack (36 runs)
# Uses placeholder FAMILY_WINNERS at the top of run_rq3 — review RQ2
# downstream results and edit before this phase if you want
# winner-driven selection. Default is to sweep all 3 families.
# ---------------------------------------------------------------------
run_phase rq3 'bash alphagrad/run_rq3_dynamic_stack.sh'
sync_wandb rq3
submit_eval_phase 3

# ---------------------------------------------------------------------
# Phase 4: RQ4 — mixed families (9 runs; curriculum axis dropped → RQ6)
# Same caveat: SINGLE_WINNER / RQ3_BEST_MAX_RULES are placeholders.
# ---------------------------------------------------------------------
run_phase rq4 'bash alphagrad/run_rq4_mixed_families.sh'
sync_wandb rq4
submit_eval_phase 4

# ---------------------------------------------------------------------
# Phase 5: RQ5 — anti-degeneracy mechanism sweep (15 runs)
# ---------------------------------------------------------------------
run_phase rq5 'bash alphagrad/run_rq5_anti_degeneracy.sh'
sync_wandb rq5
submit_eval_phase 5

# ---------------------------------------------------------------------
# Phase 6: RQ6 — granular additive curriculum (6 runs).
# Edit RQ3_BEST_MAX_RULES + RQ5_BEST_DELTA at the top of the sbatch
# based on RQ3 / RQ5 outcomes before this phase fires.
# ---------------------------------------------------------------------
run_phase rq6 'bash alphagrad/run_rq6_granular_curriculum.sh'
sync_wandb rq6
submit_eval_phase 6

# ---------------------------------------------------------------------
# Phase 7: RQ7 — Dirichlet-preference-conditioned reward (12 runs).
# Edit WINNING_VARIANT / WINNING_CURRICULUM / RQ3_BEST_MAX_RULES /
# RQ5_BEST_DELTA at the top of the sbatch based on RQ4/5/6 outcomes
# before this phase fires.
# ---------------------------------------------------------------------
run_phase rq7 'bash alphagrad/run_rq7_dirichlet_preference.sh'
sync_wandb rq7
submit_eval_phase 7

# ---------------------------------------------------------------------
# Phase 8: RQ8 — Pitch A reward pipeline (PCA-2 + per-component GDPO +
# RCPO-style fidelity constraints). 3 arms × 3 seeds = 9 runs.
# Edit RQ4_WINNING_VARIANT / RQ3_BEST_MAX_RULES / RQ5_BEST_DELTA at
# the top of the sbatch based on the prior phases' outcomes. RQ9
# (Pareto front, Pitch B) is intentionally NOT in the master flow —
# run on demand once RQ8 has settled.
# ---------------------------------------------------------------------
run_phase rq8 'bash alphagrad/run_rq8_reward_pipeline.sh'
sync_wandb rq8
submit_eval_phase 8

# ---------------------------------------------------------------------
# Phase 8: static-metric comparison table — cheap one-shot table over
# every recorded best sequence vs jax / graphax baselines.
# The per-RQ ``rq{N}_seq_eval`` phases above already train each
# sequence on MNIST via ``nn_mnist.py`` and stream the results to
# wandb (project rq-mnist-seq, entity dll-streetview). The old
# ``downstream_eval`` + ``downstream_plots`` (downstream_train.py-
# based) phases are NOT in the master flow anymore — superseded by
# the per-RQ nn-mnist-seq evaluation. The standalone
# ``run_downstream_eval_all.sh`` + ``plot_downstream.py`` scripts are
# kept around for manual one-off runs but skipped here.
# ---------------------------------------------------------------------
run_phase compare_baselines \
    'WANDB_ROOT="$HOME/dsnn/wandb";
     mapfile -t ALL_RUNS < <(
         for d in "$WANDB_ROOT"/run-* "$WANDB_ROOT"/offline-run-*; do
             [[ -d "$d" ]] || continue
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
            --wandb "$WANDB_MODE" \
            --wandb-project "$WANDB_PROJECT_CMP" \
            --wandb-entity "$WANDB_ENTITY" \
            --name compare_baselines_only_$(date +%Y%m%d) \
            --output-json "$HOME/dsnn/out/compare_full.json" \
            --output-csv "$HOME/dsnn/out/compare_full.csv";
     else
         uv run --no-sync alphagrad/src/alphagrad/approx/compare_baselines.py \
            --example VmappedNeuralNetwork \
            --dataset mnist \
            --wandb-runs "${ALL_RUNS[@]}" \
            --measure-latency \
            --wandb "$WANDB_MODE" \
            --wandb-project "$WANDB_PROJECT_CMP" \
            --wandb-entity "$WANDB_ENTITY" \
            --name compare_full_$(date +%Y%m%d) \
            --output-json "$HOME/dsnn/out/compare_full.json" \
            --output-csv "$HOME/dsnn/out/compare_full.csv";
     fi'
sync_wandb compare_baselines

# Drain any still-running final eval (e.g. RQ7's) before exit so the
# master sbatch doesn't terminate while a child eval is still pending —
# otherwise the eval's marker file would not be written even when it
# eventually succeeds.
_wait_for_prev_eval

log "================================================================="
log "Full research run finished. State markers:"
ls -la "$STATE_DIR"/done_* 2>/dev/null | tee -a "$MASTER_LOG"
log "Outputs:"
log "  ~/dsnn/out/rq1_summary.csv"
log "  ~/dsnn/out/downstream_full/*.csv"
log "  ~/dsnn/out/downstream_full/plots/*.png"
log "  ~/dsnn/out/compare_full.{json,csv}"
log "================================================================="
