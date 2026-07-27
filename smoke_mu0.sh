#!/bin/bash
# Gumbel-MuZero revival smoke: 2 episodes on CPU against the CURRENT env API.
# Latent-space planning (learned dynamics/reward inside the tree); real env
# measurements only as training targets. CPU also sidesteps the Blackwell
# XLA-autotuner SIGSEGV; the GPU launcher adds --xla_gpu_autotune_level=0.
cd "$HOME/dsnn/alphagrad" || exit 1
export PYTHONPATH="$HOME/dsnn/gxf/src:$HOME/dsnn/alphagrad/src"
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_NN_HIDDEN=16
exec uv run --no-sync python src/alphagrad/approx/mu0.py \
  --example NeuralNetwork --episodes 2 \
  --num-envs 8 --num-simulations 8 --minibatches 4 \
  --wandb disabled --name mu0-smoke
