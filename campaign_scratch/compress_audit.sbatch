#!/bin/bash
#SBATCH -J cmprC
#SBATCH -p pgi15
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/compress_out/auditC_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/compress_out
mkdir -p $R
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export ALPHAGRAD_FACE_ENUM_CACHE=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export ALPHAGRAD_TLM_SEQ=64
export ALPHAGRAD_TLM_DMODEL=128
export ALPHAGRAD_TLM_VOCAB=2048
export ALPHAGRAD_COMPRESS_AUDIT=1
export ALPHAGRAD_COMPRESS_AUDIT_DIR=$R/audit_$SLURM_JOB_ID
echo "=== node=$(hostname) $(date) ==="
JAX_COMPILATION_CACHE_DIR=$S/.jc_tlm3 timeout 3000 stdbuf -oL -eL \
  uv run --no-sync python src/alphagrad/approx/ppo.py \
  --example TransformerLM --dataset wikitext2 \
  --variant full --face-actions --unified-face-head --live-faces \
  --dynamic-substeps --max-substeps 1 --incremental-encode --measure-grad \
  --cmp-type latency --mem-type peak_memory --terminal-rewards-only \
  --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
  --advantage-norm popart --num-envs 4 --minibatches 4 \
  --latency-inner-reps 50 --vocab-size 512 \
  --episodes 4 --wandb disabled --name cmpr-audit > $R/audit_ppo_$SLURM_JOB_ID.log 2>&1
echo "=== RC=$? $(date) ==="
echo '--- audit json ---'
cat $ALPHAGRAD_COMPRESS_AUDIT_DIR/*.json 2>/dev/null
echo
echo '--- audit stderr lines ---'
grep -h 'compress-audit' $R/audit_ppo_$SLURM_JOB_ID.log | tail -20
