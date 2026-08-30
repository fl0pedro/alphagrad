#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 7:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a3_probe_w20_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a3
export ALPHAGRAD_SKIP_COUNT_OPS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
unset ALPHAGRAD_MAX_TOKENS
export A3_EPISODES=10
export A3_ARMS=base,held
export A3_PLANS=identity,24:0,prim:dot_general
export A3_WALK_STEPS=20
export A3_OUT=/Users/assmuth/dsnn/alphagrad/a3_probe_rows_w20.csv
uv run --no-sync python a3_probe.py
echo "a3_probe rc=$?"
