#!/bin/bash
#SBATCH --job-name=nn_mnist
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# VmappedNeuralNetwork PPO sweep on MNIST (784-dim input, 10-class output).
#
# Three agent variants run in parallel (one GPU each):
#   ptr_autoreg : default (pointer vertex + autoregressive RuleDecoder)
#   ptr_single  : --not-autoreg
#   mlp_single  : --no-ptr --not-autoreg
#
# Phase 1 (per-reward, isolated): 30 episodes with --rewards {cmp, mem, acc}
# in turn for every variant. 3 waves × 3 variants = 9 calibration runs total,
# 3 in parallel per wave.
# Phase 2: 500 episodes per variant with the lambdas computed from phase 1.
#
# MNIST graph is heavier than the synthetic generator (larger jaxpr, real data
# in the env eval callback). Budget ~10h calibration + ~50h full ≈ 2.5d on the
# heaviest variant. 4d gives buffer for compile slowness on the first iter.

set -euo pipefail

cd ~/dsnn

# `alias uv=...` is shell-only; use the absolute path so subprocess.Popen finds it too.
export PATH="$HOME/.local/bin:$PATH"

uv run alphagrad/dispatch_nns.py \
    --calibration-mode per-reward \
    --calibration-episodes 30 \
    --full-episodes 500 \
    --rewards cmp mem acc \
    --top-n 20 \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --num-eval-samples 10 \
    --name-prefix mnist \
    --log-dir ~/dsnn/logs_dispatch/mnist
