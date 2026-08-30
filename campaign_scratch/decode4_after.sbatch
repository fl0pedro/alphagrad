#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:4
#SBATCH --mem=600G
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode4_after_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode4_out
DA=$S/decode3_dsA.npz
DT=$S/decode4_transfer.npz

echo "===== DAG-AGNOSTICISM: two graph sizes, same parameter shapes? ====="
JAX_PLATFORMS=cpu uv run --no-sync python decode4_audit.py 2>&1 | tail -20

echo "===== SUMMARY (four-pool A/B) ====="
uv run --no-sync python decode4_summary.py $R

echo "===== STRETCH: transfer to a SECOND graph (smaller TLM) ====="
if [ ! -f $DT ]; then
  DECODE4_TLM=16,64,512 JAX_PLATFORMS=cpu uv run --no-sync python \
    decode4_data.py $DT 64 > $R/build_transfer.log 2>&1 || echo "BUILD FAILED"
fi
tail -14 $R/build_transfer.log
for P in mean last sumtok attn; do
  SRC=$R/face_${P}_s0.json
  if [ ! -f $SRC.eqx ]; then echo "  $P: no saved params, skipped"; continue; fi
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d4t CUDA_VISIBLE_DEVICES=0 \
    uv run --no-sync python decode4_face_train.py --data $DT \
    --out $R/transfer_$P.json --pool $P --eval-only \
    --load-params $SRC.eqx --norm $SRC.norm.npz --batch 2 \
    > $R/transfer_$P.log 2>&1
  echo "  transfer $P rc=$?"
  grep -E "loaded params|within|TRANSFER|Error|shape" $R/transfer_$P.log | tail -20
done
echo "===== AFTER DONE ====="
