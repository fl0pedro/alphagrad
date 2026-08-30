#!/bin/bash
#SBATCH -J abprof
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/ab_prof_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/ab_prof_out
mkdir -p $R

# =====================================================================
# MATCHED PPO-vs-POMO PROFILING.  Every lowering/measurement-relevant
# variable below is exported ONCE and inherited by BOTH arms (POMO's
# measure worker is a subprocess spawned with env=dict(os.environ)+overrides,
# so it inherits all of this too).
# =====================================================================
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export GRAPHAX_PLANNER_EXACT=1          # <-- THE historical confound. Same on both.
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_TLM_DMODEL=128
export ALPHAGRAD_TLM_VOCAB=256
export ALPHAGRAD_EXTEND_CHUNK=256       # NEVER 0 (flat 29x trap)
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_PROFILE=1
export ALPHAGRAD_HEALTH_EPISODES=64
unset ALPHAGRAD_MAX_TOKENS
unset JAX_COMPILATION_CACHE_DIR

echo "===== NODE ====="; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===== PINNED ENV (identical on BOTH arms) ====="
env | grep -E '^(GRAPHAX_|ALPHAGRAD_|XLA_|JAX_)' | sort
echo "===== git ====="; git rev-parse --short HEAD; git -C /Users/assmuth/dsnn/graphax rev-parse --short HEAD

echo "===== XLA_FLAGS validation ====="
uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum(), jax.default_backend())" || { echo "XLA_FLAGS REJECTED"; exit 1; }
[ "$ALPHAGRAD_EXTEND_CHUNK" = "0" ] && { echo "FATAL: chunk=0"; exit 1; }

echo "===== PREFLIGHT: graph size at the PINNED target config ====="
uv run --no-sync python - <<'PY' 2>&1 | tail -6
import jax
from alphagrad.approx.common.examples import (
    get_fn, get_args, infer_argnums, scalar_loss_fn)
fn = scalar_loss_fn(get_fn("TransformerLM"))
argnums = infer_argnums("TransformerLM")
args = get_args("TransformerLM", jax.random.PRNGKey(0), dataset="wikitext2")
jx = jax.make_jaxpr(fn)(*args)
print("PREFLIGHT TransformerLM seq64/dm128/vocab256:",
      "eqns=", len(jx.eqns), "invars=", len(jx.jaxpr.invars),
      "outvars=", len(jx.jaxpr.outvars), "argnums=", argnums)
PY

gpumon_start () {  # $1 = tag
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 2; done ) > $R/gpumem_$1.txt &
  echo $!
}

# ---------------------------------------------------------------------
# MATCHED MEASUREMENT BUDGET
#   PPO : --num-data-points 1 --reps-per-point 10 --latency-inner-reps 5
#   POMO: --reps 10 --inner 5
#   => 50 timed executions of the plan per measured plan on BOTH arms.
# ---------------------------------------------------------------------
PPO_COMMON="--example TransformerLM --dataset wikitext2 \
--variant full --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --cmp-type latency --mem-type peak_memory \
--terminal-rewards-only --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--lambda-frob 1 --advantage-norm popart --num-envs 4 --minibatches 4 \
--num-data-points 1 --reps-per-point 10 --latency-inner-reps 5 \
--vocab-size 512 --wandb disabled"

run_ppo () {   # $1 = tag  $2 = episodes  $3.. = extra flags
  TAG=$1; shift; EPS=$1; shift
  MON=$(gpumon_start $TAG)
  T0=$(date +%s)
  timeout 10800 stdbuf -oL -eL uv run --no-sync python src/alphagrad/approx/ppo.py \
    $PPO_COMMON --episodes $EPS --name ab-$TAG "$@" 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/ppo_$TAG.log
  RC=$?
  T1=$(date +%s)
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$TAG.txt | tail -1)
  echo "### PPO[$TAG] eps=$EPS rc=$RC wall=$((T1-T0))s gpu_peak_MiB=$PEAK extra='$@'"
  grep -E "sec/ep=" $R/ppo_$TAG.log | sed 's/.*\(\[health ep[0-9]*\]\)/\1/' | head -40
  echo "  --- prof lines ---"
  grep -oE "\[prof ep=[ 0-9]*\] host_total=.*" $R/ppo_$TAG.log | head -40
  echo "  --- tail ---"; tail -12 $R/ppo_$TAG.log
}

run_pomo () {  # $1 = tag  $2 = updates  $3 = compile-cache dir ("" = none)
  TAG=$1; UPD=$2; CC=$3
  MON=$(gpumon_start $TAG)
  T0=$(date +%s)
  if [ -n "$CC" ]; then rm -rf $CC; mkdir -p $CC; export JAX_COMPILATION_CACHE_DIR=$CC;
  else unset JAX_COMPILATION_CACHE_DIR; fi
  rm -rf $R/$TAG; mkdir -p $R/$TAG
  timeout 7200 stdbuf -oL -eL uv run --no-sync python -m alphagrad.elimrl.pomo_train \
    --out-dir $R/$TAG --seed 0 --target tlm \
    --seq 64 --dmodel 128 --vocab 256 \
    --updates $UPD --n-traj 8 --max-unique 100000 --max-hours 1.5 \
    --greedy-every 0 --check-every 0 --reps 10 --inner 5 \
    --require-gpu --worker-platforms cuda 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/pomo_$TAG.log
  RC=$?
  T1=$(date +%s)
  unset JAX_COMPILATION_CACHE_DIR
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$TAG.txt | tail -1)
  echo "### POMO[$TAG] updates=$UPD rc=$RC wall=$((T1-T0))s gpu_peak_MiB=$PEAK cache='$CC'"
  grep -E "^\[m3\]|\[u0" $R/pomo_$TAG.log | sed 's/^[0-9]* //' | head -40
  echo "  --- tail ---"; tail -12 $R/pomo_$TAG.log
}

echo "#################### ARM A: PPO, FACE ACTIONS ON ####################"
run_ppo faceon 12 --face-actions --unified-face-head --live-faces

echo "#################### ARM B: PPO, ORDER-ONLY (--no-approx-head) ####################"
run_ppo orderonly 12 --no-approx-head

echo "#################### ARM C: POMO, no persistent compile cache ####################"
run_pomo nocache 14 ""

echo "#################### ARM D: POMO, WITH persistent compile cache ####################"
run_pomo withcache 14 /Users/assmuth/dsnn/.jc_abprof_pomo

echo "#################### ARM E: PPO faces-ON at the DEFAULT measure budget (5x4x50) ####################"
PPO_COMMON="--example TransformerLM --dataset wikitext2 \
--variant full --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --cmp-type latency --mem-type peak_memory \
--terminal-rewards-only --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--lambda-frob 1 --advantage-norm popart --num-envs 4 --minibatches 4 \
--num-data-points 5 --reps-per-point 4 --latency-inner-reps 50 \
--vocab-size 512 --wandb disabled"
run_ppo faceon_bigbudget 6 --face-actions --unified-face-head --live-faces

echo "#################### AB PROFILING DONE ####################"
