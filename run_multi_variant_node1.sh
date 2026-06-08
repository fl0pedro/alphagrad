#!/bin/bash
#SBATCH --job-name=mv_node1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=pgi15-gpu15
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Multi-variant RQ — node 1 of 3 (pgi15-gpu15, 64 CPU, 770 GB, 4× Blackwell).
# Phases (sequential): diag_gcd → compress → diag_compress.
# Each phase: 5 seeds × 500 episodes, all 4 GPUs Ray-sharded per run.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_multi_variant_node1
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-48}"
MINIBATCHES="${MINIBATCHES:-2}"
PHASES=(diag_gcd compress diag_compress)
SEEDS=(42 250197 1337 7 100)

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-dsnn-variants}"

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem
    --cmp-type latency
    --mem-type peak_memory
    --advantage-norm scalar
    --ppo-epochs 4
    --anti-degeneracy none
    --cosine-lower-bound 0.0
    --cosine-upper-bound 1.0
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    --num-cpu-workers "$NUM_ENVS"
    --minibatches "$MINIBATCHES"
    --measure-latency
    --running-max-channels peak_memory,max_io_sum
    --calibrate-steps 0
    # Disable CPU-actor callback timeouts: the 5×4 measurement loop
    # on a cold (order, specs, shape) can exceed the default 1800s
    # initial timeout, causing the actor pool to recycle in a loop and
    # never complete a step. 0 = wait until done.
    --cpu-callback-timeout 0
    --cpu-callback-initial-timeout 0
    --wandb "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT_RL"
    --wandb-entity "$WANDB_ENTITY"
)

run_one() {
    local phase=$1; local seed=$2
    local tag="${phase}_seed${seed}"
    local logfile="$LOG_DIR/${tag}.out"
    echo "[node1] [$tag] START $(date +%FT%T%z)"
    SECONDS=0
    if uv run --no-sync "$PPO" \
            --name "mv_node1_${tag}" \
            --variant "$phase" \
            --seed "$seed" \
            "${COMMON_ARGS[@]}" 2>&1 \
            | awk -W interactive '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
            > "$logfile"; then
        echo "[node1] [$tag] OK ($SECONDS s)"
    else
        local rc=$?
        echo "[node1] [$tag] FAILED rc=$rc after $SECONDS s — continuing"
    fi
}

echo "================================================================"
echo "MULTI-VARIANT NODE 1: $(hostname)"
echo "Phases: ${PHASES[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "Start:  $(date)"
echo "================================================================"

for phase in "${PHASES[@]}"; do
    echo
    echo ">>> PHASE: $phase ($(date)) <<<"
    for seed in "${SEEDS[@]}"; do
        run_one "$phase" "$seed"
    done
done

echo
echo "================================================================"
echo "NODE 1 DONE: $(date)"
echo "Logs in $LOG_DIR/"
echo "================================================================"
