#!/bin/bash
# Degenerate-plan forensics: the EXACT v13 target/config on CPU, 2 episodes,
# with the instrumented guard naming each sentinel's cause.
cd "$HOME/dsnn/alphagrad" || exit 1
export PYTHONPATH="$HOME/dsnn/gxf/src:$HOME/dsnn/alphagrad/src"
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_MAX_DELTA_TOKENS=8192
export ALPHAGRAD_NN_HIDDEN=256
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_DEBUG_DEGEN=1
exec uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example VmappedNeuralNetwork --dataset mnist --hidden-dim 256 \
  --variant full --face-actions --incremental-encode \
  --reward-mode mult --cmp-type latency --mem-type peak_memory \
  --episodes 2 --wandb disabled --name degen-forensics
