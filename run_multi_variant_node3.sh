#!/bin/bash
#SBATCH --job-name=mv_node3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=pgi15-gpu18
#SBATCH --time=20-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Multi-variant RQ — node 3 of 3 (pgi15-gpu18, 64 CPU, 770 GB, 4× Blackwell).
# Phases (sequential): compress_scalar → quantize → compress_quant → full.
# Note: 4 phases (one extra for the all-three "full" variant).

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_multi_variant_node3
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-48}"
MINIBATCHES="${MINIBATCHES:-2}"
PHASES=(compress_scalar quantize compress_quant full)
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
    echo "[node3] [$tag] START $(date +%FT%T%z)"
    SECONDS=0
    if uv run --no-sync "$PPO" \
            --name "mv_node3_${tag}" \
            --variant "$phase" \
            --seed "$seed" \
            "${COMMON_ARGS[@]}" 2>&1 \
            | awk -W interactive '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
            > "$logfile"; then
        echo "[node3] [$tag] OK ($SECONDS s)"
    else
        local rc=$?
        echo "[node3] [$tag] FAILED rc=$rc after $SECONDS s — continuing"
    fi
}

echo "================================================================"
echo "MULTI-VARIANT NODE 3: $(hostname)"
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
echo "NODE 3 DONE: $(date)"
echo "Logs in $LOG_DIR/"
echo "================================================================"
