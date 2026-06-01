#!/bin/bash
#SBATCH --job-name=rq7_dirichlet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ7: does a preference-conditioned policy (Dirichlet-sampled reward
# weighting at training time) produce a Pareto frontier of (test_acc,
# FLOPs) trade-offs that dominates the single-preference baselines
# from RQ1..RQ6?
#
# Four configs × 3 seeds = 12 PPO runs at 500 episodes:
#   baseline         : fixed lambdas (no preference-conditioning, RQ4 winner)
#   pref_corner      : Dirichlet α=0.2 (corner-biased, Yang et al. 2019 default)
#   pref_uniform     : Dirichlet α=1.0 (uniform over preferences)
#   pref_center      : Dirichlet α=5.0 (centered, balanced trade-offs)
#
# Inference-time evaluation needs a separate step (see
# docs/experiments/dirichlet_preference_reward.md §"Inference-time
# evaluation"): query each trained policy with 16 grid preferences
# and downstream-train each rolled-out sequence to get the Pareto
# frontier. That part is stubbed pending an
# inference_pareto.py helper.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq7_dirichlet
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

WINNING_VARIANT="${WINNING_VARIANT:-full}"
WINNING_CURRICULUM="${WINNING_CURRICULUM:-}"   # empty = no curriculum, override after RQ6
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"
RQ5_BEST_DELTA="${RQ5_BEST_DELTA:-0.05}"

# (tag, --preference-conditioned (bool), α). Baseline uses no flag.
declare -a CONFIGS=(
    "baseline:0:0.0"
    "pref_corner:1:0.2"
    "pref_uniform:1:1.0"
    "pref_center:1:5.0"
)
SEEDS=(42 250197 1337)

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
    --anti-degeneracy delta_ceiling
    --anti-degeneracy-delta "$RQ5_BEST_DELTA"
    --cosine-lower-bound 0.0
    --cosine-upper-bound 1.0
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
    --variant "$WINNING_VARIANT"
    --wandb "${WANDB_MODE:-online}"
    --wandb-project "${WANDB_PROJECT_RL:-dsnn-vertex}"
    --wandb-entity "${WANDB_ENTITY:-dll-streetview}"
)
if [[ -n "$WINNING_CURRICULUM" ]]; then
    COMMON_ARGS+=(--curriculum "$WINNING_CURRICULUM")
fi

run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    # Each PPO variant uses ALL 4 GPUs via Ray sharding. Variants run
    # SEQUENTIALLY in this sbatch so each gets the full GPU allocation.
    echo "[${tag}] all 4 GPUs (Ray-sharded) -> $logfile"
    if uv run --no-sync "$PPO" --name "rq7_${tag}" \
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
    IFS=':' read -r cfg_tag pref_on alpha <<< "$cfg"
    for sd in "${SEEDS[@]}"; do
        tag="${cfg_tag}_seed${sd}"
        extra_args=(--seed "$sd")
        if [[ "$pref_on" == "1" ]]; then
            extra_args+=(
                --preference-conditioned
                --preference-dirichlet-alpha "$alpha"
                --dirichlet-mix-ratio 0.5
            )
        fi
        run_variant "$tag" "${extra_args[@]}" || FAIL=$((FAIL + 1))
    done
done
echo "All 12 RQ7 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
