#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=600G
#SBATCH -c 64
#SBATCH -t 01:30:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode3_smoke_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
S=/Users/assmuth/dsnn
J=/Users/assmuth/.jaxcache_d3
mkdir -p $J

echo "########## BUILD SMOKE ARM A ##########"
JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $S/decode3_smokeA.npz 6 || exit 1
echo "########## BUILD SMOKE ARM B ##########"
JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $S/decode3_smokeB.npz 6 --shapes || exit 1

echo "########## CHECK ARM A ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=0 uv run --no-sync python decode3_check.py $S/decode3_smokeA.npz 0 || exit 1
echo "########## CHECK ARM B ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=0 uv run --no-sync python decode3_check.py $S/decode3_smokeB.npz 0 || exit 1

echo "########## TRAIN SMOKE: face new ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=0 uv run --no-sync python decode3_face_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_f.json --arm new --steps 3 --batch 2 --n-test 2 --chunk-cap 4096 || exit 1
echo "########## TRAIN SMOKE: face new_sizes ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=1 uv run --no-sync python decode3_face_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_fs.json --arm new_sizes --steps 3 --batch 2 --n-test 2 --chunk-cap 4096 || exit 1
echo "########## TRAIN SMOKE: face old_today ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=1 uv run --no-sync python decode3_face_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_fo.json --arm old_today --steps 3 --batch 2 --n-test 2 --chunk-cap 4096 || exit 1
echo "########## TRAIN SMOKE: vertex new ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=2 uv run --no-sync python decode3_vertex_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_v.json --arm new --steps 3 --batch 2 --n-test 2 || exit 1
echo "########## TRAIN SMOKE: vertex anchor ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=2 uv run --no-sync python decode3_vertex_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_va.json --arm anchor --steps 3 --batch 2 --n-test 2 || exit 1
echo "########## TRAIN SMOKE: vertex no_dyn ##########"
JAX_COMPILATION_CACHE_DIR=$J CUDA_VISIBLE_DEVICES=2 uv run --no-sync python decode3_vertex_train.py \
  --data $S/decode3_smokeA.npz --out /tmp/d3_vn.json --arm no_dyn --steps 3 --batch 2 --n-test 2 || exit 1
echo "########## SMOKE ALL DONE ##########"
