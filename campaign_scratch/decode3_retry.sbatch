#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:5
#SBATCH --mem=800G
#SBATCH -c 64
#SBATCH -t 08:00:00
#SBATCH -d singleton
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode3_retry_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
# The three arm-A FACE runs died with
#   F triton_tiling_propagation.cc:547] Check failed: dst_remaining_size == 1
# an XLA assert inside the Triton GEMM tiling pass -- a COMPILE crash, not a
# result.  Flag names verified against xla_cuda_plugin.so strings, not guessed.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"

echo "== XLA_FLAGS validation (an unknown flag aborts every arm at import) =="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "import jax; import jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED - aborting"; exit 1; }

S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode3_out
DA=$S/decode3_dsA.npz

runf () {  # gpu, arm
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d3r_$1 CUDA_VISIBLE_DEVICES=$1 \
    uv run --no-sync python decode3_face_train.py --data $DA \
    --out $R/face_A_$2.json --arm $2 --steps 3000 --batch 4 \
    --chunk-cap 4096 --lr 1e-3 > $R/face_A_$2.log 2>&1
  rc=$?
  echo "  face_A_$2 rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "  face_A_$2 retrying at batch 3 (different fusion shapes)"
    JAX_COMPILATION_CACHE_DIR=$S/.jc_d3r2_$1 CUDA_VISIBLE_DEVICES=$1 \
      uv run --no-sync python decode3_face_train.py --data $DA \
      --out $R/face_A_$2.json --arm $2 --steps 3000 --batch 3 \
      --chunk-cap 4096 --lr 1e-3 > $R/face_A_$2.log 2>&1
    echo "  face_A_$2 (batch 3) rc=$?"
  fi
}

runf 0 new        &
runf 1 new_sizes  &
runf 2 old_today  &
( JAX_COMPILATION_CACHE_DIR=$S/.jc_d3r_3 CUDA_VISIBLE_DEVICES=3 \
  uv run --no-sync python decode3_vertex_train.py --data $DA \
  --out $R/vert_A_new_slots.json --arm new_slots --steps 3000 --batch 8 \
  --lr 1e-3 > $R/vert_A_new_slots.log 2>&1; echo "  vert_A_new_slots rc=$?" ) &
wait
echo "########## SUMMARY ##########"
uv run --no-sync python decode3_summary.py $R
echo "########## RETRY DONE ##########"
