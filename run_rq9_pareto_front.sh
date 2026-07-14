#!/bin/bash
#SBATCH --job-name=rq9_pareto_front
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ9 (Pitch B): does preference-conditioning + augmented Tchebycheff
# scalarization + low-discrepancy preference sampling produce a
# Pareto-front hypervolume that strictly dominates RQ7's
# linear-scalarisation + Dirichlet baseline?
#
# ** USER FLAGGED THIS AS LOWEST PRIORITY ** — this sbatch is a
# scaffold; the trainer-side hookup for `--scalarization tchebycheff`
# and `--preference-sampler kronecker` (and the PPO-Ray
# preference-into-policy gap noted in
# docs/experiments/pareto_front_tchebycheff.md) needs to land before
# this can actually run. Submitting it today would silently fall back
# to the linear+Dirichlet baseline (defaults preserve current
# behaviour). NOT wired into the master sbatch.
#
# Built on top of the RQ8 winner so the within-objective reward is
# sane (Pitch A and Pitch B compose — Pitch A is the within-objective
# fix; Pitch B is the front-coverage layer on top).
#
# 4 arms × 3 seeds = 12 PPO runs at 500 episodes each. Inference-time
# Pareto-frontier evaluation queries each trained policy with 16 grid
# preferences and runs downstream-MNIST per rolled-out sequence — that
# helper lives at alphagrad/src/alphagrad/approx/inference_pareto.py
# (also pending; shared with RQ7).
#
# **Skip GFN.** PPO + MuZero only.

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export DSNN_JAX_CACHE_REUSE="${DSNN_JAX_CACHE_REUSE:-1}"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq9_pareto_front
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-dsnn-vertex}"

RQ4_WINNING_VARIANT="${RQ4_WINNING_VARIANT:-full}"
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"
RQ5_BEST_DELTA="${RQ5_BEST_DELTA:-0.05}"
RQ8_WINNING_PIPELINE="${RQ8_WINNING_PIPELINE:-pca2_rcpo}"   # placeholder

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
    --max-rules "$RQ3_BEST_MAX_RULES"
    --dynamic-substeps
    --variant "$RQ4_WINNING_VARIANT"
    --preference-conditioned
    --preference-dirichlet-alpha 1.0
    --dirichlet-mix-ratio 0.5
    --reward-pipeline "${RQ8_WINNING_PIPELINE/pca2_rcpo/pca2}"
    --wandb "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT_RL"
    --wandb-entity "$WANDB_ENTITY"
)

# (tag, --preference-sampler, --scalarization)
# Baseline arm == RQ7 winner under linear+Dirichlet sampling.
declare -a ARMS=(
    "baseline_pref:dirichlet:linear"
    "tcheb_only:dirichlet:tchebycheff"
    "kronecker_only:kronecker:linear"
    "full:kronecker:tchebycheff"
)
SEEDS=(42 250197 1337)


run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    echo "[${tag}] all 4 GPUs -> $logfile"
    if uv run --no-sync "$PPO" --name "rq9_${tag}" \
            "${COMMON_ARGS[@]}" "$@" \
            > "$logfile" 2>&1; then
        echo "[${tag}] OK"
        return 0
    else
        local rc=$?
        echo "[${tag}] FAILED (rc=${rc}) — continuing"
        return $rc
    fi
}


FAIL=0
for arm in "${ARMS[@]}"; do
    IFS=':' read -r arm_tag sampler scalar <<< "$arm"
    for sd in "${SEEDS[@]}"; do
        tag="${arm_tag}_seed${sd}"
        run_variant "$tag" \
            --preference-sampler "$sampler" \
            --scalarization "$scalar" \
            --tchebycheff-rho 0.05 \
            --seed "$sd" || FAIL=$((FAIL + 1))
    done
done

echo "All 12 RQ9 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
echo ""
echo "NEXT (separate helper, not in this script):"
echo "  uv run --no-sync alphagrad/src/alphagrad/approx/inference_pareto.py \\"
echo "    --runs-glob '\$HOME/dsnn/wandb/run-*-*' --filter rq9_ \\"
echo "    --grid-preferences 16 --downstream-steps 10000"
echo "(That helper is pending; see docs/experiments/pareto_front_tchebycheff.md.)"
exit "$FAIL"
