#!/bin/bash
#SBATCH --job-name=compare_baselines
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# One-shot static-metric comparison: jax.jacfwd/rev, graphax fwd/rev,
# and every recorded best-sequence channel from the 3 latest wandb runs
# (PPO j2yl2wn7, MuZero 7lt7oiwm, GFN k8qzjo23).
#
# Output: a Markdown-style table on stdout + a JSON dump that the
# downstream-plotter can consume. No Ray cluster — single python
# invocation on one GPU is sufficient.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

ENTRY=alphagrad/src/alphagrad/approx/compare_baselines.py
OUT_DIR=~/dsnn/out/compare
mkdir -p "$OUT_DIR" slurm

uv run --no-sync "$ENTRY" \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --measure-latency \
    --output-json "$OUT_DIR/table.json" \
    --output-csv "$OUT_DIR/table.csv" \
    2>&1 | tee "$OUT_DIR/run.out"

echo
echo "[compare] outputs in $OUT_DIR"
