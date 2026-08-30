#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu20
#SBATCH --gres=gpu:6
#SBATCH --mem=700G
#SBATCH -c 48
#SBATCH -t 03:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode2_wave3_%j.log
cd ~/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
D=/Users/assmuth/dsnn/decode2_ds_192.npz
R=/Users/assmuth/dsnn/alphagrad/decode2_out_fine
mkdir -p $R
# FINE early-phase resolution: eval every 10 steps for the first 400.
runv () {
  JAX_COMPILATION_CACHE_DIR=/Users/assmuth/.jaxcache_v$2 CUDA_VISIBLE_DEVICES=$1 \
  uv run --no-sync python decode2_vertex_train.py \
    --data $D --out $R/vert_$2$3.json --variant $2 --steps 400 --batch 8 \
    --lr 1e-3 --eval-every 10 --seed ${4:-0} > $R/vert_$2$3.log 2>&1
}
runv 0 fold &
runv 1 split_part &
runv 2 split_auth &
runv 3 boundary &
runv 4 mix_nobias &
runv 5 mix_adj &
wait
echo ALL DONE
