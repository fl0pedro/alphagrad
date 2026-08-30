#!/bin/bash
#SBATCH -J abpomo
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu17
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/ab_pomo_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/ab_prof_out
mkdir -p $R

# IDENTICAL pinned environment to ab_prof.sbatch (the PPO arms).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export GRAPHAX_PLANNER_EXACT=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_TLM_DMODEL=128
export ALPHAGRAD_TLM_VOCAB=256
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_PROFILE=1
unset ALPHAGRAD_MAX_TOKENS
unset JAX_COMPILATION_CACHE_DIR

echo "===== NODE ====="; hostname
echo "===== PINNED ENV ====="; env | grep -E '^(GRAPHAX_|ALPHAGRAD_|XLA_|JAX_)' | sort

run_pomo () {  # $1 = tag  $2 = updates  $3 = compile-cache dir ("" = none)
  TAG=$1; UPD=$2; CC=$3
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 2; done ) > $R/gpumem_$TAG.txt &
  MON=$!
  T0=$(date +%s)
  if [ -n "$CC" ]; then rm -rf $CC; mkdir -p $CC; export JAX_COMPILATION_CACHE_DIR=$CC;
  else unset JAX_COMPILATION_CACHE_DIR; fi
  rm -rf $R/$TAG; mkdir -p $R/$TAG
  timeout 4200 stdbuf -oL -eL uv run --no-sync python -m alphagrad.elimrl.pomo_train \
    --out-dir $R/$TAG --seed 0 --target tlm \
    --seq 64 --dmodel 128 --vocab 256 \
    --updates $UPD --n-traj 8 --max-unique 100000 --max-hours 1.0 \
    --greedy-every 0 --check-every 0 --reps 10 --inner 5 \
    --require-gpu --worker-platforms cuda 2>&1 \
    | awk '{ printf "%d %s\n", systime(), $0; fflush() }' > $R/pomo_$TAG.log
  RC=$?
  T1=$(date +%s)
  unset JAX_COMPILATION_CACHE_DIR
  kill $MON 2>/dev/null
  PEAK=$(sort -n $R/gpumem_$TAG.txt | tail -1)
  echo "### POMO[$TAG] updates=$UPD rc=$RC wall=$((T1-T0))s gpu_peak_MiB=$PEAK cache='$CC'"
  grep -E '^\[m3\]|^\[u0' $R/pomo_$TAG.log | sed 's/^[0-9]* //' | head -30
  echo "  --- tail ---"; tail -10 $R/pomo_$TAG.log
}

echo "#################### ARM C: POMO, no persistent compile cache ####################"
run_pomo nocache17 14 ""

echo "#################### ARM D: POMO, WITH persistent compile cache ####################"
run_pomo withcache17 14 /Users/assmuth/dsnn/.jc_abprof_pomo

echo "#################### AB POMO DONE ####################"
