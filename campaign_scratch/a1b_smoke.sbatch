#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a1b_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a1b_out
mkdir -p $R

export JAX_PLATFORMS=cpu
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_a1b

echo "===== unit tests (fixed) ====="
uv run --no-sync python -m pytest tests/per_face_sizes_live_test.py -q 2>&1 | tail -25
echo "unit rc=${PIPESTATUS[0]}"

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 2 --minibatches 2 \
--latency-inner-reps 5 --vocab-size 512 --wandb disabled"

echo "===== SMOKE A: --live-faces WITH --per-face-masks (sizes half ON) ====="
ALPHAGRAD_HEALTH_EPISODES=2 timeout 3000 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --per-face-masks \
  --episodes 2 --name a1b_on > $R/smoke_on.log 2>&1
echo "rc=$?"
grep -E "per-face-masks|faces.live_sizes|Traceback|Error:" $R/smoke_on.log | head -20
grep -iE "ratio|approx_kl|kl=" $R/smoke_on.log | head -12
grep -E "health ep|approx_applied|approx_skipped" $R/smoke_on.log | head -12
tail -25 $R/smoke_on.log

echo
echo "===== SMOKE B: --live-faces WITHOUT the flag (control) ====="
ALPHAGRAD_HEALTH_EPISODES=2 timeout 3000 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON \
  --episodes 2 --name a1b_off > $R/smoke_off.log 2>&1
echo "rc=$?"
grep -E "per-face-masks|Traceback|Error:" $R/smoke_off.log | head -20
grep -iE "ratio|approx_kl|kl=" $R/smoke_off.log | head -12
tail -15 $R/smoke_off.log
echo "===== A1B SMOKE DONE ====="
