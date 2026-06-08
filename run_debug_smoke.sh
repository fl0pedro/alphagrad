#!/bin/bash
#SBATCH --job-name=debug_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=pgi15-gpu15
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Single-variant debug smoke. Carries [DBG-rrt] / [DBG-rvm] prints from
# ppo_ray_worker.py so we can pinpoint where the hang is.
# Minimal-config: RQ1-shape (16 envs, calibration off, no new env-loop
# flags). If the rollout DOES progress we see per-t and per-r [DBG] lines;
# whichever one stops appearing is the hang site.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1
export RAY_DEDUP_LOGS=0

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_debug_smoke
mkdir -p "$LOG_DIR" slurm

V=${V:-diag_gcd}
SEED=42
EPISODES=2

echo "==== DEBUG SMOKE: variant=$V eps=$EPISODES seed=$SEED ===="
echo "==== Start: $(date) ===="

uv run --no-sync "$PPO" \
    --name "debug_smoke_${V}" \
    --variant "$V" \
    --seed "$SEED" \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --rewards cmp \
    --cmp-type latency \
    --advantage-norm scalar \
    --ppo-epochs 4 \
    --anti-degeneracy none \
    --cosine-lower-bound 0.0 \
    --cosine-upper-bound 1.0 \
    --episodes "$EPISODES" \
    --num-envs 16 \
    --num-cpu-workers 16 \
    --minibatches 4 \
    --num-data-points 2 \
    --reps-per-point 1 \
    --measure-latency \
    --running-max-channels peak_memory,max_io_sum \
    --calibrate-steps 0 \
    --wandb offline 2>&1 \
    | awk -W interactive '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
    > "$LOG_DIR/${V}.out"

echo "==== DONE: $(date) ===="
