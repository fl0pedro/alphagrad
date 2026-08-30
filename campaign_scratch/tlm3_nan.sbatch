#!/bin/bash
#SBATCH -J tlm3nan
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/tlm3_nan_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/tlm3_out
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256          # NEVER 0 (flat 29x trap)
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

echo "##### D1: Helmholtz (tiniest target) 4 episodes -- does the NaN reproduce off TLM? #####"
ALPHAGRAD_HEALTH_EPISODES=4 CUDA_VISIBLE_DEVICES=0 timeout 1800 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --episodes 4 --name d1 \
  > $R/nan_d1.log 2>&1
echo "rc=$?"; grep -E "health ep|popart-init|Traceback|Error" $R/nan_d1.log | grep -v live-faces | head -20

echo "##### D2: Helmholtz --entropy-weight 0 -- is the entropy term the poison? #####"
ALPHAGRAD_HEALTH_EPISODES=4 CUDA_VISIBLE_DEVICES=0 timeout 1800 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --episodes 4 \
  --entropy-weight 0 --name d2 > $R/nan_d2.log 2>&1
echo "rc=$?"; grep -E "health ep" $R/nan_d2.log | grep -v live-faces | head -10

echo "##### D3: Helmholtz, popart OFF (--advantage-norm none) -- is PopArt the poison? #####"
ALPHAGRAD_HEALTH_EPISODES=4 CUDA_VISIBLE_DEVICES=0 timeout 1800 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none \
  --variant full --face-actions --unified-face-head --live-faces \
  --dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
  --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
  --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
  --advantage-norm none --num-envs 4 --minibatches 4 \
  --latency-inner-reps 50 --vocab-size 512 --wandb disabled \
  --episodes 4 --name d3 > $R/nan_d3.log 2>&1
echo "rc=$?"; grep -E "health ep" $R/nan_d3.log | grep -v live-faces | head -10

echo "##### D4: PRE-REFACTOR ca43369 (alphagrad only; graphax unchanged) Helmholtz #####"
PRE=$S/tlm3_prerefactor
rm -rf $PRE && mkdir -p $PRE
git archive ca43369 | tar -x -C $PRE
echo "  pre-refactor tree at $PRE (git archive, NO .git mutation, NO stash)"
ALPHAGRAD_HEALTH_EPISODES=4 PYTHONPATH=$PRE/src CUDA_VISIBLE_DEVICES=1 timeout 1800 \
  uv run --no-sync python $PRE/src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --episodes 4 --name d4 \
  > $R/nan_d4.log 2>&1
echo "rc=$?"; grep -E "health ep|Traceback|Error|unrecognized" $R/nan_d4.log | grep -v live-faces | head -20

echo "##### D5: JAX_DEBUG_NANS on Helmholtz -- locate the FIRST non-finite op #####"
ALPHAGRAD_HEALTH_EPISODES=2 JAX_DEBUG_NANS=1 CUDA_VISIBLE_DEVICES=0 timeout 2400 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example Helmholtz --dataset none $COMMON --episodes 2 --name d5 \
  > $R/nan_d5.log 2>&1
echo "rc=$?"; grep -E "health ep|FloatingPointError|invalid value|Traceback" $R/nan_d5.log | head -5
echo "  --- last 60 lines of the debug-nans traceback ---"
tail -60 $R/nan_d5.log

echo "##### D6: TransformerLM 12 episodes -- does the NaN CLEAR after warm-up? #####"
ALPHAGRAD_HEALTH_EPISODES=12 CUDA_VISIBLE_DEVICES=1 timeout 5400 \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example TransformerLM --dataset wikitext2 $COMMON --episodes 12 --name d6 \
  > $R/nan_d6.log 2>&1
echo "rc=$?"; grep -E "health ep" $R/nan_d6.log | grep -v live-faces | head -14

echo "##### NAN DIAGNOSTIC DONE #####"
