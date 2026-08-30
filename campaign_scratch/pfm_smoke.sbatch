#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/pfm_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/pfm_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_pfm
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 2 --minibatches 2 \
--vocab-size 512 --wandb disabled --episodes 1"
for ARM in off on; do
  EXTRA=""
  [ "$ARM" = "on" ] && EXTRA="--per-face-masks"
  echo "##### ARM=$ARM #####"
  timeout 3000 uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example Helmholtz --dataset none $COMMON $EXTRA --name pfm_$ARM \
    > $R/smoke_$ARM.log 2>&1
  echo "rc=$?"
  grep -nE "Traceback|Error|error" $R/smoke_$ARM.log | head -8
  grep -oE "ratio[^ ,]*[=:] *[0-9.eE+-]+" $R/smoke_$ARM.log | head -8
  grep -oE "(per_face|approx_applied|approx_skipped)/[a-z_]+[=:] *[0-9.eE+-]+" $R/smoke_$ARM.log | head -20
  tail -25 $R/smoke_$ARM.log
done
echo "===== SMOKE DONE ====="
