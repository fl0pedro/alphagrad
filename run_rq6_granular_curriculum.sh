#!/bin/bash
#SBATCH --job-name=rq6_granular_curr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ6: does an "additive" curriculum (one new action-space dimension
# per stage) outperform NO curriculum at all?
#
# Two arms (the 7-stage curriculum was dropped — its design has the
# large simple→difficult action-space jump that the research
# background says PPO's trust region can't absorb; running it would
# only confirm a known a-priori weakness):
#
#   none      — no curriculum, --variant full from step 0
#   granular  — additive schedule that expands the action space by
#               exactly ONE preset per stage. Smaller per-stage policy
#               perturbation; PPO's clip-trust region absorbs it
#               gracefully.
#
# 2 configs × 3 seeds = 6 runs. Built on RQ4's best mixing (default
# `full`) and RQ5's best δ (default 0.05 placeholder — applies only
# if anti-degeneracy is re-enabled per RQ5 outcome).
#
# See docs/experiments/curriculum_granularity.md for the research
# background (Bengio 2009; Soviany 2022; Florensa 2018).

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq6_granular_curr
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"
RQ5_BEST_DELTA="${RQ5_BEST_DELTA:-0.05}"

# Granular curriculum: 9 stages, each adding exactly ONE new preset
# from VARIANT_PRESETS (variants.py). Episode counts: 1/9 of EPISODES
# per stage (with leftover absorbed by the final 'full' stage).
N_STAGES=9
PER_STAGE=$(( EPISODES / N_STAGES ))
FULL_STAGE=$(( EPISODES - PER_STAGE * (N_STAGES - 1) ))
GRANULAR_SPEC="ve_only:${PER_STAGE},diag_gcd:${PER_STAGE},compress_scalar:${PER_STAGE},quant_smallest_float:${PER_STAGE},all_simple:${PER_STAGE},diag_factor:${PER_STAGE},compress:${PER_STAGE},quantize:${PER_STAGE},full:${FULL_STAGE}"

declare -a CONFIGS=(
    "none::full:"
    "granular::full:${GRANULAR_SPEC}"
)
SEEDS=(42 250197 1337)

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    # CPU pool 1:1 with envs (see run_rq1_ve_only.sh comment).
    --num-cpu-workers "$NUM_ENVS"
    --measure-latency
    # peak_memory / max_io_sum as rollout-wide max (see RQ1 sbatch).
    --running-max-channels peak_memory,max_io_sum
    # 2-phase cost schedule: graphax-only until ep 200, then full.
    --cost-pipeline-schedule cheap_first
    --phase-cutover-ep 200
    --max-rules "$RQ3_BEST_MAX_RULES"
    --dynamic-substeps
    --wandb "${WANDB_MODE:-online}"
    --wandb-project "${WANDB_PROJECT_RL:-dsnn-vertex}"
    --wandb-entity "${WANDB_ENTITY:-dll-streetview}"
)

run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    # Each PPO variant uses ALL 4 GPUs via Ray sharding. Variants run
    # SEQUENTIALLY in this sbatch so each gets the full GPU allocation.
    echo "[${tag}] all 4 GPUs (Ray-sharded) -> $logfile"
    if uv run --no-sync "$PPO" --name "rq6_${tag}" \
            "${COMMON_ARGS[@]}" "$@" \
            > "$logfile" 2>&1; then
        echo "[${tag}] OK"
        return 0
    else
        local rc=$?
        echo "[${tag}] FAILED (rc=${rc}) — continuing to next variant"
        return $rc
    fi
}


FAIL=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r cfg_tag _ variant curr_spec <<< "$cfg"
    for sd in "${SEEDS[@]}"; do
        tag="${cfg_tag}_seed${sd}"
        extra_args=(--variant "$variant" --seed "$sd")
        if [[ -n "$curr_spec" ]]; then
            extra_args+=(--curriculum "$curr_spec")
        fi
        run_variant "$tag" "${extra_args[@]}" || FAIL=$((FAIL + 1))
    done
done
echo "All 6 RQ6 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
