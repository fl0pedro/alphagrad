#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu20
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH -c 64
#SBATCH -t 10:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode2_wave2_%j.log
cd ~/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=/Users/assmuth/dsnn/decode2_ds_192.npz
R=/Users/assmuth/dsnn/alphagrad/decode2_out
mkdir -p $R
runf () {
  JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache_f$2$4 CUDA_VISIBLE_DEVICES=$1 \
  uv run --no-sync python decode2_face_train.py \
    --data $D --out $R/face_$2$4.json --arm $2 --steps $3 --batch 4 \
    --chunk-cap 2048 --lr 1e-3 --eval-every 100 --seed ${5:-0} > $R/face_$2$4.log 2>&1
}
runv () {
  JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache_v$2$4 CUDA_VISIBLE_DEVICES=$1 \
  uv run --no-sync python decode2_vertex_train.py \
    --data $D --out $R/vert_$2$4.json --variant $2 --steps $3 --batch 8 \
    --lr 1e-3 --eval-every 100 --seed ${5:-0} > $R/vert_$2$4.log 2>&1
}
runv 0 boundary 3000 &
runv 1 split_auth 3000 &
runf 2 today_sizes 3000 &
( runv 3 fold 3000 _s1 1 ; runv 3 split_part 3000 _s1 1 ) &
( runv 4 mix_adj 3000 _s1 1 ; runv 4 mix_nobias 3000 _s1 1 ) &
runf 5 today 3000 _s1 1 &
runf 6 endpoints_sizes 3000 _s1 1 &
runf 7 endpoints 3000 _s1 1 &
wait
echo ALL DONE
