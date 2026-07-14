#!/bin/bash
#SBATCH --job-name=rq8_reward_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ8 (Pitch A): does refactoring the reward pipeline — PCA-2 of cost
# channels + per-component GDPO + RCPO-style fidelity constraints +
# telescoping max-over-vertices for peak memory — produce more stable
# training AND find lower-FLOPs points than the legacy weighted-sum
# formulation?
#
# 4 arms × 3 seeds = 12 PPO runs at 500 episodes each, all built on
# the RQ4/RQ5 winning configuration (full variant + delta_ceiling
# anti-degeneracy). The legacy arm matches RQ4 exactly so we can read
# the marginal contribution of each pipeline stage.
#
# See docs/experiments/reward_pipeline_pca.md for the full design + the
# per-arm success criteria. Reward pipeline modules:
# common/reward_pca.py, common/telescoping.py, common/scalarization.py.
#
# **Skip GFN.** PPO + MuZero only per the user's scope. The arm tags
# include "ppo_" so the eval pipeline picks them up alongside the
# existing RQ1–RQ7 runs.

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export DSNN_JAX_CACHE_REUSE="${DSNN_JAX_CACHE_REUSE:-1}"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_rq8_reward_pipeline
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-dsnn-vertex}"

# Winners from RQ4/RQ5 — edit if those phases revise the defaults.
RQ4_WINNING_VARIANT="${RQ4_WINNING_VARIANT:-full}"
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"
RQ5_BEST_DELTA="${RQ5_BEST_DELTA:-0.05}"

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    --num-cpu-workers "$NUM_ENVS"
    --measure-latency
    # peak_memory / max_io_sum as rollout-wide max — applied to every
    # RQ8 arm so the cost-channel semantics match the rest of the
    # research plan (user 2026-05-23: "add it to all the ppo runs that
    # reference it").
    --running-max-channels peak_memory,max_io_sum
    # 2-phase cost schedule: graphax-only until ep 200, then full.
    # NB: PCA-2 fits only the channels actually populated, so phase 1
    # PCA refits only see muls_adds_fmas+max_io_sum non-zero; the rest
    # of the cost vector is zero. Refit gets re-triggered after the
    # cutover episode automatically.
    --cost-pipeline-schedule cheap_first
    --phase-cutover-ep 200
    --max-rules "$RQ3_BEST_MAX_RULES"
    --dynamic-substeps
    --variant "$RQ4_WINNING_VARIANT"
    --wandb "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT_RL"
    --wandb-entity "$WANDB_ENTITY"
)

# (tag, --reward-pipeline, --reward-as-constraints)
# 3 arms × 3 seeds = 9 PPO runs. With telescoping promoted to a global
# default the prior 4-arm design's pca2_rcpo_max arm is identical to
# pca2_rcpo, so it's been removed.
declare -a ARMS=(
    "baseline:legacy:"
    "pca2:pca2:"
    "pca2_rcpo:pca2:cosine_sim,frob_residual"
)
SEEDS=(42 250197 1337)


run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    # All 4 GPUs Ray-sharded per variant; variants sequentialise.
    echo "[${tag}] all 4 GPUs (Ray-sharded) -> $logfile"
    if uv run --no-sync "$PPO" --name "rq8_${tag}" \
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
for arm in "${ARMS[@]}"; do
    IFS=':' read -r arm_tag pipeline as_constraints <<< "$arm"
    for sd in "${SEEDS[@]}"; do
        tag="${arm_tag}_seed${sd}"
        extra_args=(
            --reward-pipeline "$pipeline"
            --seed "$sd"
        )
        if [[ -n "$as_constraints" ]]; then
            extra_args+=()
        fi
        run_variant "$tag" "${extra_args[@]}" || FAIL=$((FAIL + 1))
    done
done

echo "All 9 RQ8 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
