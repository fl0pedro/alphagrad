#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Two-phase dispatch: calibration (no lambdas) -> compute lambdas from
# observed reward distribution -> full training with computed lambdas.
# Edit experiments and the lambda formula in alphagrad/dispatch_nns.py.

cd ~/dsnn
alias uv="~/.local/bin/uv"

uv run alphagrad/dispatch_nns.py \
    --calibration-episodes 30 \
    --full-episodes 500 \
    --top-n 20 \
    --example VmappedNeuralNetwork \
    --exec-on-gpu \
    --log-dir ~/dsnn/logs_dispatch
