#!/bin/bash
# Search-space sampler for the "full" variant, on cpu2 (no GPU needed —
# CPU-latency measurement). Generates N reproducible random full-variant
# configs, then measures all cost channels + quality over a FIXED 8
# MNIST points x 10 reps each, RAW (every measurement stored). Gates OFF
# (no flop-gate, no slow-exec trim). Reproducible & resumable: bump
# --num-samples to extend; already-measured samples are skipped.
#
#   sbatch run_sampler_cpu2.sh                 # 10k samples, seed 12345 (Jacobian)
#   NUM_SAMPLES=100000 sbatch run_sampler_cpu2.sh   # extends to 100k
#   # GRAD-MODE rerun (value_and_grad + deterministic xla_peak_memory),
#   # SAME configs (same seed) into a SEPARATE dir so the old Jacobian data is
#   # preserved:
#   MEASURE_GRAD=1 LATENCY_TIMER=rm OUT_DIR=~/dsnn/search_space_full_grad \
#       sbatch run_sampler_cpu2.sh

#SBATCH --job-name=ssfull
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15-cpu
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --nodes=1
#SBATCH --exclusive

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
# Disable Ray's `uv run` hook (else it uploads ~/dsnn as working_dir, >512MB).
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

mkdir -p slurm
SAMPLER=alphagrad/src/alphagrad/approx/sampler_full.py

NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEED="${SEED:-12345}"
# 384 logical cores on cpu2; 48 actors -> 8 logical (4 physical) cores each.
# 48 (not 64) leaves RAM headroom: 64 long-lived workers + the cost_analysis
# C++ leak filled the 755GB node (~700GB) and Ray's memory-monitor began
# killing workers. With 48 workers + aggressive RECYCLE_EVERY the resident set
# stays well bounded.
NUM_WORKERS="${NUM_WORKERS:-48}"
OUT_DIR="${OUT_DIR:-$HOME/dsnn/search_space_full}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"   # network hidden width (sampler-only override)
# Grad-mode measurement (value_and_grad of the scalar loss) + deterministic
# xla_peak_memory channel — matches the MORL trainers. MEASURE_GRAD=0 keeps the
# legacy Jacobian measurement. Use a SEPARATE OUT_DIR for grad runs so the two
# paradigms' jsonl files don't mix.
MEASURE_GRAD="${MEASURE_GRAD:-0}"
LAT_TIMER="${LATENCY_TIMER:-perf_counter}"
GRAD_FLAGS=""; [ "$MEASURE_GRAD" = "1" ] && GRAD_FLAGS="--measure-grad"

echo "==== sampler start $(date): N=$NUM_SAMPLES seed=$SEED workers=$NUM_WORKERS hidden=$HIDDEN_DIM grad=$MEASURE_GRAD timer=$LAT_TIMER out=$OUT_DIR ===="
uv run --no-sync "$SAMPLER" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --out-dir "$OUT_DIR" \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --num-data-points 8 \
    --num-passes 10 \
    --num-eval-samples 8 \
    --max-exec-seconds 0 \
    --hidden-dim "$HIDDEN_DIM" \
    --latency-timer "$LAT_TIMER" $GRAD_FLAGS
echo "==== sampler done $(date) ===="
