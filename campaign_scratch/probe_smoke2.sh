#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_probe
cd /Users/assmuth/dsnn/alphagrad
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_FEATURE_PROBE=1
export ALPHAGRAD_FEATURE_PROBE_FACES=16
export ALPHAGRAD_FEATURE_PROBE_WIDTH=64
export ALPHAGRAD_FEATURE_PROBE_LR=2e-2
COMMON="--variant full --face-actions --unified-face-head --live-faces --dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad --cmp-type latency --mem-type peak_memory --terminal-rewards-only --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 --advantage-norm popart --num-envs 8 --minibatches 2 --latency-inner-reps 5 --vocab-size 512 --wandb disabled"
timeout 3000 stdbuf -oL -eL uv run --no-sync python -u src/alphagrad/approx/ppo.py --example Helmholtz --dataset none $COMMON --episodes 6 --ppo-epochs 4 --name probe_smoke2 2>&1
echo "rc=$?"
