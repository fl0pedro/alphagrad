#!/bin/bash
# --measure-grad smoke: verify the GRADIENT pipeline (spec-correct target)
# produces non-zero approx/exact norms and sane frob, before spending GPU time.
# The historical failure mode was ||approx|| == 0 from a graph mismatch.
cd "$HOME/dsnn/alphagrad" || exit 1
export PYTHONPATH="$HOME/dsnn/gxf/src:$HOME/dsnn/alphagrad/src"
export JAX_PLATFORMS=cpu
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_MAX_DELTA_TOKENS=8192
export ALPHAGRAD_NN_HIDDEN=256
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_PROFILE=1
export ALPHAGRAD_DEBUG_DEGEN=1
exec uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example VmappedNeuralNetwork --dataset mnist --hidden-dim 256 \
  --variant full --face-actions --incremental-encode \
  --measure-grad \
  --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
  --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
  --advantage-norm popart --num-envs 4 --minibatches 4 \
  --episodes 2 --wandb disabled --name grad-smoke
