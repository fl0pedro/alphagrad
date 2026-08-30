#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/bwd_ab_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/bwd_out
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256          # NEVER 0 (flat 29x trap)
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_HEALTH_EPISODES=40
export JAX_COMPILATION_CACHE_DIR=$S/.jc_bwd
# DELTA_OVERFLOW deliberately LEFT AT THE DEFAULT (raise): a 4096 window that
# is too small must be loud, not clipped away.

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

NN="--example VmappedNeuralNetwork --dataset mnist --hidden-dim 256"
TLM="--example TransformerLM --dataset wikitext2"

echo "########## PHASE 0: correctness gates (CPU) ##########"
echo "--- numerics proof: remat on vs off, forward AND gradient bit-identical"
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q \
  tests/carry_heads_remat_equiv_test.py 2>&1 | tail -12
RC_EQ=${PIPESTATUS[0]}
echo "remat-equivalence rc=$RC_EQ"

echo "--- gradient still reaches palimpsa's base encode"
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q \
  tests/palimpsa_base_grad_test.py 2>&1 | tail -8
RC_BG=${PIPESTATUS[0]}
echo "base-grad rc=$RC_BG"

echo "--- policy regression gate: MUST be bit-identical, NO re-golden"
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py \
  > $R/gate.log 2>&1
RC_GATE=$?
grep -E "^\[gate\]|MISMATCH|Traceback" $R/gate.log | head -30
echo "gate rc=$RC_GATE"

if [ $RC_EQ -ne 0 ] || [ $RC_BG -ne 0 ] || [ $RC_GATE -ne 0 ]; then
  echo "!!!!! PHASE 0 FAILED -- NOT running the A/B. Behaviour changed."
  tail -40 $R/gate.log
  exit 1
fi

# ---------------------------------------------------------------------------
run () {   # $1 tag  $2 remat(0/1)  $3 window  $4 K  $5 episodes  $6.. target args
  local TAG=$1 RM=$2 W=$3 K=$4 EP=$5; shift 5
  : > $R/gpumem_$TAG.txt
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0
      sleep 2
    done ) > $R/gpumem_$TAG.txt 2>/dev/null &
  local MON=$!
  local T0=$(date +%s)
  ALPHAGRAD_CARRY_HEADS_REMAT=$RM ALPHAGRAD_MAX_DELTA_TOKENS=$W \
  ALPHAGRAD_NN_HIDDEN=256 timeout 5400 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    "$@" $COMMON --episodes $EP --grad-window $K --name bwd-$TAG \
    > $R/$TAG.log 2>&1
  local RC=$?
  local T1=$(date +%s)
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  local PEAK=$(sort -n $R/gpumem_$TAG.txt | tail -1)
  local SECS=$(grep -oE "sec/ep=[0-9.]+" $R/$TAG.log | tail -3 | tr '\n' ' ')
  echo "RESULT $TAG remat=$RM window=$W K=$K rc=$RC wall=$((T1-T0))s peakMiB=$PEAK sec/ep(last3)= $SECS"
  grep -iE "RESOURCE_EXHAUSTED|out of memory|Killed|token DELTA truncated" \
    $R/$TAG.log | head -3
  if [ $RC -ne 0 ]; then echo "  --- tail:"; tail -20 $R/$TAG.log; fi
}

echo
echo "########## PHASE 1: NN256 A/B  (3 arms x K in {1,4,16}) ##########"
echo "## arm A = BEFORE (no remat, window 32768)"
echo "## arm B = FIX 1 only (no remat, window 4096)"
echo "## arm C = AFTER   (remat + window 4096)"
for K in 1 4 16; do
  run nnA_k$K 0 32768 $K 6 $NN
  run nnB_k$K 0 4096  $K 6 $NN
  run nnC_k$K 1 4096  $K 6 $NN
done

echo
echo "########## PHASE 2: palimpsa kernel chunk_size ##########"
echo "--- (a) is the Pallas kernel entered at all on the production path?"
ALPHAGRAD_NN_HIDDEN=256 ALPHAGRAD_MAX_DELTA_TOKENS=4096 timeout 2400 \
  uv run --no-sync python bwd_kernel_probe.py \
  $NN $COMMON --episodes 1 --grad-window 1 --name bwd-probe \
  > $R/kernel_probe.log 2>&1
echo "probe rc=$?"
grep -E "kernel-probe" $R/kernel_probe.log

echo "--- (b) standalone kernel sweep at the encoder's real shapes (H=4 d=32)"
BWD_TS=4096,32768 BWD_CS=8,16,32,64,128 timeout 3600 \
  uv run --no-sync python bwd_kernel_chunksweep.py > $R/chunksweep.log 2>&1
echo "chunksweep rc=$?"
grep -E "chunksweep" $R/chunksweep.log

echo
echo "########## PHASE 3: TransformerLM confirmation ##########"
for K in 1 4 16; do
  run tlmA_k$K 0 32768 $K 6 $TLM
  run tlmC_k$K 1 4096  $K 6 $TLM
done

echo
echo "########## SUMMARY ##########"
grep -h "^RESULT " $R/../bwd_ab_${SLURM_JOB_ID}.log 2>/dev/null | sort
echo "===== BWD AB DONE ====="
