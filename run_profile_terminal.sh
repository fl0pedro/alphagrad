#!/bin/bash
#SBATCH --job-name=prof_term
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=pgi15-gpu18
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Profiling harness: ONE env, ONE CPU worker — removes the 16-actor
# contention so the per-phase [DBG-env] timings are clean and honest
# (paired with the block_until_ready fix that attributes exact-jacrev
# cost to the quality phase instead of leaking it into the next approx
# measurement). 3 episodes of diag_gcd so we see ep0 (cold) vs ep1/ep2
# (warm) terminal-step breakdown without contention.
#
# Reads: per-phase env timing (approx_compile / exact_compile /
# exec_loop / quality_split), the slow-order cutoff, and per-ep wall.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG=~/dsnn/logs_profile/diag_gcd.out
mkdir -p ~/dsnn/logs_profile slurm

echo "==== PROFILE terminal step: 1 env, 1 worker, no contention ===="
echo "==== Start: $(date) ===="

uv run --no-sync "$PPO" \
    --name "profile_terminal" \
    --variant diag_gcd \
    --seed 42 \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --rewards cmp mem \
    --cmp-type latency \
    --mem-type peak_memory \
    --advantage-norm scalar \
    --ppo-epochs 4 \
    --anti-degeneracy none \
    --cosine-lower-bound 0.0 \
    --cosine-upper-bound 1.0 \
    --episodes 3 \
    --num-envs 1 \
    --num-cpu-workers 1 \
    --minibatches 1 \
    --num-data-points 5 \
    --reps-per-point 4 \
    --percentile-keep 0.60 \
    --slow-exec-cutoff-seconds 8.0 \
    --measure-latency \
    --running-max-channels peak_memory,max_io_sum \
    --calibrate-steps 0 \
    --cpu-callback-timeout 0 \
    --cpu-callback-initial-timeout 0 \
    --wandb offline 2>&1 \
    | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
    > "$LOG"

echo "==== DONE: $(date) ===="
echo "=== env phase timings ==="
grep "DBG-env" "$LOG" | tail -30
echo "=== per-ep ==="
tr "\r" "\n" < "$LOG" | grep -oE "[0-9]/3 ..[0-9:]+.[0-9:]+, [0-9.]+s/it." | tail -3
