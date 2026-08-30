#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -c 128
#SBATCH -t 12:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode3_run_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1

S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode3_out
mkdir -p $R
DA=$S/decode3_dsA.npz
DB=$S/decode3_dsB.npz
NT=192
FSTEPS=3000
VSTEPS=3000

if [ ! -f $DA ]; then
  echo "########## BUILD ARM A ##########"
  JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $DA $NT > $R/buildA.log 2>&1 || exit 1
fi
tail -25 $R/buildA.log
if [ ! -f $DB ]; then
  echo "########## BUILD ARM B ##########"
  JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $DB $NT --shapes > $R/buildB.log 2>&1 || exit 1
fi
tail -25 $R/buildB.log

echo "########## EQUIVALENCE CHECK ##########"
JAX_COMPILATION_CACHE_DIR=$S/.jc_d3 CUDA_VISIBLE_DEVICES=0 uv run --no-sync \
  python decode3_check.py $DA 0 2>&1 | tee $R/checkA.log
JAX_COMPILATION_CACHE_DIR=$S/.jc_d3 CUDA_VISIBLE_DEVICES=0 uv run --no-sync \
  python decode3_check.py $DB 1 2>&1 | tee $R/checkB.log

runf () {  # gpu, dataset-tag, arm
  local D=$DA; [ "$2" = "B" ] && D=$DB
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d3_$1 CUDA_VISIBLE_DEVICES=$1 \
    uv run --no-sync python decode3_face_train.py --data $D \
    --out $R/face_$2_$3.json --arm $3 --steps $FSTEPS --batch 4 \
    --chunk-cap 4096 --lr 1e-3 > $R/face_$2_$3.log 2>&1
  echo "  done face_$2_$3 rc=$?"
}
runv () {  # gpu, dataset-tag, arm
  local D=$DA; [ "$2" = "B" ] && D=$DB
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d3_$1 CUDA_VISIBLE_DEVICES=$1 \
    uv run --no-sync python decode3_vertex_train.py --data $D \
    --out $R/vert_$2_$3.json --arm $3 --steps $VSTEPS --batch 8 \
    --lr 1e-3 > $R/vert_$2_$3.log 2>&1
  echo "  done vert_$2_$3 rc=$?"
}

echo "########## TRAINING ##########"
runf 0 A new           &
runf 1 A new_sizes     &
runf 2 B new           &
runf 3 B new_sizes     &
( runv 4 A new       ; runf 4 A sizes_only  ) &
( runv 5 A anchor    ; runf 5 A old_today   ) &
( runv 6 B new       ; runv 6 A no_dyn      ) &
( runv 7 A no_ident  ; runf 7 B sizes_only  ) &
wait
echo "########## SUMMARY ##########"
uv run --no-sync python decode3_summary.py $R
echo "########## ALL DONE ##########"
