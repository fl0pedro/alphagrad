#!/bin/bash
#SBATCH --job-name=smoke_variants
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=pgi15-gpu18
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Larger smoke: 2 episodes per variant × seed=42 × all 10 variants.
# Validates: every variant boots + completes 2 eps, and reports ep2
# (post-warmup) wall-clock. Carries the jacrev-exact fix + the
# terminal-rewards-only CPU-worker fix + quality-decoupled-from-reps.
# ALPHAGRAD_DBG_TIMING shows the per-phase env breakdown so we can
# confirm the quality phase is now cheap.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_smoke_variants
mkdir -p "$LOG_DIR" slurm

VARIANTS=(
    diag_gcd compress diag_compress
    diag_factor quant_smallest_float diag_quant
    compress_scalar quantize compress_quant
    full
)
SEED=42
EPISODES=2

# Multi-channel reward: latency + peak_memory + frob (lambda-frob=1.0
# default). Light measurement for the smoke (2 data points × 2 reps);
# the full run scales these up.
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
    --num-envs 16
    --num-cpu-workers 16
    --minibatches 4
    --num-data-points 5
    --reps-per-point 4
    --percentile-keep 0.60
    --measure-latency
    --running-max-channels peak_memory,max_io_sum
    --calibrate-steps 0
    --cpu-callback-timeout 0
    --cpu-callback-initial-timeout 0
    --wandb offline
)

run_variant() {
    local v=$1
    local logfile="$LOG_DIR/${v}.out"
    echo
    echo "==========================================================="
    echo "[$v] START $(date +%FT%T%z)"
    echo "==========================================================="
    SECONDS=0
    if uv run --no-sync "$PPO" \
            --name "smoke_${v}" \
            --variant "$v" \
            --seed "$SEED" \
            "${COMMON_ARGS[@]}" 2>&1 \
            | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }' \
            > "$logfile"; then
        echo "[$v] OK ($SECONDS s wall-clock for $EPISODES eps)"
    else
        echo "[$v] FAILED (rc=$? after $SECONDS s) — continuing"
    fi
    echo "[$v] best + env-timing:"
    tr "\r" "\n" < "$logfile" | grep -E "ppo_ray best:" | tail -1 || true
    grep "DBG-env" "$logfile" | tail -3 || true
}

echo "================================================================"
echo "LARGER SMOKE: ${#VARIANTS[@]} variants × $EPISODES eps × seed=$SEED"
echo "Fixes: jacrev-exact + terminal-rewards-only CPU + quality/reps decouple"
echo "Node: $(hostname); $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x4"
echo "Start: $(date)"
echo "================================================================"

for v in "${VARIANTS[@]}"; do
    run_variant "$v"
done

echo
echo "================================================================"
echo "SMOKE DONE: $(date)"
echo "Per-variant logs: $LOG_DIR/"
echo "================================================================"
