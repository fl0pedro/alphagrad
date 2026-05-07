#!/bin/bash
#SBATCH --job-name=approx_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

cd ~/dsnn

alias uv="$HOME/.local/bin/uv" # redundant w/ PATH
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_smoke_mnist
mkdir -p "$LOG_DIR" slurm

EPISODES=3000
NUM_ENVS=16

COMMON=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --num-eval-samples 10
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    --rewards cmp mem acc
    --top-n 20
    --wandb offline
    --seed 250197
)

declare -a PIDS=()

run_variant() {
    local tag=$1
    local gpu=$2
    shift 2
    local logfile="$LOG_DIR/${tag}.out"
    echo "[${tag}] CUDA_VISIBLE_DEVICES=$gpu -> $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run "$PPO" --name "smoke_mnist_${tag}" "${COMMON[@]}" "$@" \
        > "$logfile" 2>&1 &
    PIDS+=("$!:${tag}")
}

run_variant "mlp_single"  0  --no-ptr --not-autoreg
run_variant "ptr_single"  1  --not-autoreg
run_variant "ptr_pairs"   2  --max-rules 4 --factors=-1
run_variant "ptr_pairs_F" 3  --max-rules 4 --factors=1,2,4,7,8,14,16,28,32,49,56,64,98,112,196,392,784

echo "Launched ${#PIDS[@]} variants. Waiting for completion."

FAIL=0
for entry in "${PIDS[@]}"; do
    pid=${entry%%:*}
    tag=${entry#*:}
    if wait "$pid"; then
        echo "[${tag}] OK (pid=${pid})"
    else
        rc=$?
        echo "[${tag}] FAILED (pid=${pid}, rc=${rc})"
        FAIL=1
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "One or more variants failed. See per-variant logs in ${LOG_DIR}."
    exit 1
fi

echo "All four variants finished cleanly. Logs in ${LOG_DIR}."
