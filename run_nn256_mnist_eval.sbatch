#!/bin/bash
#SBATCH --job-name=eval_nn256_mnist
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

cd ~/dsnn

export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

LOG_DIR=~/dsnn/logs_eval_mnist
mkdir -p "$LOG_DIR" slurm

EPISODES=1000

run_variant() {
    local name=$1
    local gpu=$2
    shift 2
    local logfile="$LOG_DIR/${name}.out"
    echo "[${name}] CUDA_VISIBLE_DEVICES=$gpu -> $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync alphagrad/src/alphagrad/approx/ppo.py \
            --example VmappedNeuralNetwork \
            --dataset mnist \
            --hidden-dim 256 \
            --episodes "$EPISODES" \
            --name "$name" \
            "$@" > "$logfile" 2>&1 &
}

# GPU 0: EXACT run (NO micro actions, purely VE)
run_variant "exact_mnist_256" 0 --exact

# GPU 1: APPROX run (Full action space)
run_variant "approx_mnist_256" 1 --variant full

echo "Launched Exact and Approx jobs. Waiting for completion."
wait

echo "Jobs completed cleanly."
