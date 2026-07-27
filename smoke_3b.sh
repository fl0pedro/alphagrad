#!/bin/bash
# Phase 3b end-to-end smoke: 2 episodes on CPU, full incremental stack
# (append-only tokens + palimpsa carry + vertex memory + face actions).
cd "$HOME/dsnn/alphagrad" || exit 1
export PYTHONPATH="$HOME/dsnn/gxf/src:$HOME/dsnn/alphagrad/src"
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_MAX_TOKENS=8192
export ALPHAGRAD_MAX_DELTA_TOKENS=1024
export ALPHAGRAD_NN_HIDDEN=16
exec uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example NeuralNetwork --episodes 2 \
  --incremental-encode --face-actions \
  --wandb disabled --name 3b-smoke
