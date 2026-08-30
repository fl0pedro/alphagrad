#!/bin/bash
#SBATCH -J perf2r1
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=220G
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/perf2_r1_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/perf2_out
mkdir -p $R

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/alphagrad/.jaxcache_perf2
mkdir -p $JAX_COMPILATION_CACHE_DIR
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64

echo "############ (3) CHUNK SWEEP ############" 
uv run --no-sync python perf2_chunk_sweep.py $R/chunk_sweep.json > $R/chunk_sweep.log 2>&1
echo "chunk_sweep rc=$?"
tail -70 $R/chunk_sweep.log

echo "############ (1) FACE-ENUM MICROBENCH ############"
for CFG in "TransformerLM 4 0 0.0 tlm2_e4" "TransformerLM3 4 0 0.0 tlm3_e4" "TransformerLM3 1 0 0.0 tlm3_e1" "TransformerLM3 4 0 0.30 tlm3_e4_c30"; do
  set -- $CFG
  echo "--- faceenum $5 (target=$1 envs=$2 T=$3 cfrac=$4)"
  JAX_PLATFORMS=cpu uv run --no-sync python perf2_faceenum_bench.py \
      $1 $2 $3 $4 $R/faceenum_$5.json > $R/faceenum_$5.log 2>&1
  echo "  rc=$?"
  grep -E "^\[perf2\]" $R/faceenum_$5.log | tail -60
done

echo "############ (2) PROFILED PPO -- PER-PHASE ATTRIBUTION ############"
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
--cmp-type latency --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --num-envs 4 --minibatches 4 \
--latency-inner-reps 50 --vocab-size 512 --wandb disabled"

prof () {   # $1 example  $2 cacheflag  $3 tag
  export ALPHAGRAD_FACE_ENUM_CACHE=$2
  export ALPHAGRAD_PROFILE=1
  export ALPHAGRAD_PROFILE_POLICY=1
  export ALPHAGRAD_PROFILE_DIST=1
  T0=$(date +%s)
  ALPHAGRAD_HEALTH_EPISODES=3 CUDA_VISIBLE_DEVICES=0 timeout 7200 \
    uv run --no-sync python src/alphagrad/approx/ppo.py \
    --example $1 --dataset wikitext2 $COMMON \
    --episodes 3 --name perf2$3 > $R/prof_$3.log 2>&1
  RC=$?; T1=$(date +%s)
  echo "=== PROF[$3] example=$1 cache=$2 rc=$RC wall=$((T1-T0))s"
  grep -E "face width|health ep" $R/prof_$3.log | grep -v live-faces
  grep -oE "\[prof ep=[^\n]*" $R/prof_$3.log
  grep -E "\[dist|delta_len|\[pp " $R/prof_$3.log | head -30
}

prof TransformerLM  0 tlm2_off
prof TransformerLM  1 tlm2_on
prof TransformerLM3 0 tlm3_off
prof TransformerLM3 1 tlm3_on

echo "############ GATE ############"
unset ALPHAGRAD_PROFILE ALPHAGRAD_PROFILE_POLICY ALPHAGRAD_PROFILE_DIST
unset ALPHAGRAD_FACE_ENUM_CACHE
uv run --no-sync python tests/policy_regression_gate.py > $R/gate.log 2>&1
echo "gate rc=$?"; tail -20 $R/gate.log
echo "############ PERF2 ROUND 1 DONE ############"
