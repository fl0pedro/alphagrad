#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a2_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a2_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a2
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --grad-window 0 --example Helmholtz --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"

echo "### A: channel ON, guard ON (the shared-exact-reference path)"
timeout 3300 uv run --no-sync python src/alphagrad/approx/ppo.py $COMMON \
  --fidelity-weight 1.0 --cos-log-every 1 --name a2_smoke_on \
  > $R/smoke_on.log 2>&1
echo "A rc=$?"
grep -E "^\[cfg\] fidelity|^\[fidelity ep|^\[grad-cov ep|heads=" $R/smoke_on.log | head -20
echo "--- non-finite / traceback check ---"
grep -cE "Traceback|nan|NaN" $R/smoke_on.log

echo "### B: channel ON, guard OFF (the channel pays for its own reference)"
timeout 3300 uv run --no-sync python src/alphagrad/approx/ppo.py $COMMON \
  --fidelity-weight 1.0 --no-reject-frozen-grads --name a2_smoke_noguard \
  > $R/smoke_noguard.log 2>&1
echo "B rc=$?"
grep -E "^\[cfg\] fidelity|^\[fidelity ep|^\[grad-cov ep" $R/smoke_noguard.log | head -20

echo "### C: baseline, channel OFF, guard ON (the amortisation denominator)"
timeout 3300 uv run --no-sync python src/alphagrad/approx/ppo.py $COMMON \
  --name a2_smoke_off > $R/smoke_off.log 2>&1
echo "C rc=$?"
grep -E "^\[cfg\] fidelity|^\[fidelity ep|^\[grad-cov ep" $R/smoke_off.log | head -20

echo "--- per-episode wall, all three arms ---"
for L in on noguard off; do
  printf "%-8s " $L
  grep -oE "sec/ep=[0-9.na/]+" $R/smoke_$L.log | tr '\n' ' '
  echo
done
echo "===== A2 SMOKE DONE ====="
