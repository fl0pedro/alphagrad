#!/bin/bash
#SBATCH --job-name=prof16
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Realistic 16-env profiling: full per-episode phase breakdown
# ([DBG-prof] line) + env terminal phases ([DBG-env]). 4 episodes of
# diag_gcd so we see cold (ep0) vs warm (ep1-3). With the affinity fix.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG=~/dsnn/logs_profile/profcpu2.out
mkdir -p ~/dsnn/logs_profile slurm

echo "==== PROFILE-CPU2-384core: 16 env / 16 worker, 4 eps diag_gcd ===="
echo "==== Start: $(date) ===="

uv run --no-sync "$PPO" \
    --name "profile16" \
    --variant diag_gcd \
    --seed 42 \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --rewards cmp mem \
    --cmp-type latency \
    --mem-type peak_memory \
    --advantage-norm scalar \
    --ppo-epochs 4 \
    --episodes 4 \
    --num-envs 16 \
    --num-cpu-workers 16 \
    --minibatches 16 \
    --num-data-points 5 \
    --reps-per-point 4 \
    --percentile-keep 0.60 \
    --slow-exec-cutoff-seconds 8.0 \
    --measure-queue \
    --flop-gate-threshold 2e11 \
    --reserved-driver-cores 0 \
    --measure-latency \
    --running-max-channels peak_memory,max_io_sum \
    --calibrate-steps 0 \
    --cpu-callback-timeout 0 \
    --cpu-callback-initial-timeout 0 \
    --wandb offline 2>&1 \
    | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
    > "$LOG"

echo "==== DONE: $(date) ===="
echo "=== PER-EP PHASE BREAKDOWN ==="
grep "DBG-prof" "$LOG"
echo "=== terminal env phases (sample) ==="
grep -E "exec_loop|quality_split" "$LOG" | tail -8
