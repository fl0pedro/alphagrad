#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -c 64
#SBATCH -t 01:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode3_smoke2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
S=/Users/assmuth/dsnn
J=$S/.jc_d3s
mkdir -p $J
D=$S/decode3_tiny.npz

echo "########## BUILD TINY (short horizon) ##########"
JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $D 6 0,2,4,6 || exit 1

for ARM in new new_sizes sizes_only old_today; do
  echo "########## FACE $ARM ##########"
  JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=0 uv run --no-sync python decode3_face_train.py \
    --data $D --out /tmp/d3_f_$ARM.json --arm $ARM --steps 2 --batch 2 --n-test 2 --chunk-cap 4096 \
    2>&1 | grep -vE "^W[0-9]|split_k" | tail -25 || exit 1
done
for ARM in new new_slots anchor no_ident no_dyn; do
  echo "########## VERTEX $ARM ##########"
  JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=1 uv run --no-sync python decode3_vertex_train.py \
    --data $D --out /tmp/d3_v_$ARM.json --arm $ARM --steps 2 --batch 2 --n-test 2 \
    2>&1 | grep -vE "^W[0-9]|split_k" | tail -25 || exit 1
done
echo "########## SMOKE ALL DONE ##########"
