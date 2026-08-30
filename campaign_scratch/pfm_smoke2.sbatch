#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/pfm_smoke2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/pfm_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_pfm
# NO --live-faces: this is the path where the oracle RUNS, i.e. the only one
# where the per-face SIZES half of --per-face-masks is active.
COMMON="--variant full --face-actions --unified-face-head \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 2 --minibatches 2 \
--vocab-size 512 --wandb disabled --episodes 1"
for ARM in off on; do
  EXTRA=""
  [ "$ARM" = "on" ] && EXTRA="--per-face-masks"
  echo "##### SIZES ARM=$ARM #####"
  timeout 3000 uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example Helmholtz --dataset none $COMMON $EXTRA --name pfms_$ARM \
    > $R/sizes_$ARM.log 2>&1
  echo "rc=$?"
  grep -aoE "per-face-masks[^\"]{0,120}" $R/sizes_$ARM.log | head -2
  grep -naE "Traceback|TypeError|ValueError" $R/sizes_$ARM.log | head -5
  grep -aoE "ratio[^ ,]*=[0-9.eE+-]+" $R/sizes_$ARM.log | head -6
done
echo "===== SMOKE2 DONE ====="
