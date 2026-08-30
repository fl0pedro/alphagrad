#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu20
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH -c 64
#SBATCH -t 10:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode2_wave1_%j.log
cd ~/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false

D=/Users/assmuth/dsnn/decode2_ds_192.npz
R=/Users/assmuth/dsnn/alphagrad/decode2_out
mkdir -p $R
FSTEPS=3000
VSTEPS=3000

runf () {  # gpu arm
  JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache_f$2 CUDA_VISIBLE_DEVICES=$1 uv run --no-sync python decode2_face_train.py \
    --data $D --out $R/face_$2.json --arm $2 --steps $FSTEPS --batch 4 \
    --chunk-cap 2048 --lr 1e-3 --eval-every 100 > $R/face_$2.log 2>&1
}
runv () {  # gpu variant
  JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache_v$2 CUDA_VISIBLE_DEVICES=$1 uv run --no-sync python decode2_vertex_train.py \
    --data $D --out $R/vert_$2.json --variant $2 --steps $VSTEPS --batch 8 \
    --lr 1e-3 --eval-every 100 > $R/vert_$2.log 2>&1
}

runf 0 today &
runf 1 today_bnd &
runf 2 endpoints &
runf 3 endpoints_sizes &
( runf 4 sizes_only ; runv 4 mix_nobias ) &
runv 5 fold &
runv 6 split_part &
runv 7 mix_adj &
wait
echo ALL DONE
