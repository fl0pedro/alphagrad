#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -c 128
#SBATCH -t 20:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode6_run_%j.log
# decode6: the SETTLED decode4/decode5 comparison with ONE variable changed --
# per-equation SHAPES in the tokenization.  decode4_run.sbatch:20 set
# DA=$S/decode3_dsA.npz and every arm used $DA, so the shapes dataset was never
# put in front of the pools.  Both datasets were verified bit-identical to a
# fresh build with today's code by decode6_pre (job 61255).
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256          # never 0: measured 29x FLAT penalty
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode6_out
R5=/Users/assmuth/dsnn/alphagrad/decode5_out   # stage-B dsA reference arms
mkdir -p $R
DA=$S/decode3_dsA.npz
DB=$S/decode3_dsB.npz
STEPS=${STEPS:-3000}

run () {  # gpu, ds, pool, mp, extents, seed
  local G=$1 DS=$2 P=$3 M=$4 X=$5 SD=$6
  local D=$DA; [ "$DS" = "B" ] && D=$DB
  local N=d6_${DS}_${P}_mp${M}_ex${X}_s${SD}
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d6_$G CUDA_VISIBLE_DEVICES=$G \
    uv run --no-sync python decode5_face_train.py --data $D \
    --out $R/$N.json --pool $P --mp $M --extents $X --seed $SD \
    --steps $STEPS --batch 4 --chunk-cap 4096 --lr 1e-3 \
    > $R/$N.log 2>&1
  local rc=$?
  echo "  $N rc=$rc  $(date +%H:%M:%S)"
  if [ $rc -ne 0 ]; then
    echo "  $N retrying at batch 2 (smaller fusion shapes)"
    JAX_COMPILATION_CACHE_DIR=$S/.jc_d6r_$G CUDA_VISIBLE_DEVICES=$G \
      uv run --no-sync python decode5_face_train.py --data $D \
      --out $R/$N.json --pool $P --mp $M --extents $X --seed $SD \
      --steps $STEPS --batch 2 --chunk-cap 4096 --lr 1e-3 \
      > $R/$N.log 2>&1
    echo "  $N (batch 2) rc=$?  $(date +%H:%M:%S)"
  fi
}

# One arm per GPU.  Lane 7 carries the two dsA replication arms because dsA
# streams are 2.8x shorter, so two of them fit in one dsB slot.
#
# SEED IS THE OUTER LOOP: every arm gets seed s before any arm gets s+1, so a
# job cut short still yields a PAIRED comparison (an unpaired comparison has
# produced a fake 17.5% "win" in this repo before).
lane () {  # gpu, seed, "ds:pool:mp:ex ..."
  local G=$1 SD=$2; shift 2
  for A_ in $@; do
    IFS=: read -r DS P M X <<< "$A_"
    run $G $DS $P $M $X $SD
  done
}

for SD in 0 1 2; do
  echo "########## SEED ROUND $SD  $(date +%H:%M:%S)"
  lane 0 $SD "B:mean:0:0"                 &   # THE POINT: shapes, no extents
  lane 1 $SD "B:mean:0:1"                 &   # shapes + extents
  lane 2 $SD "B:mean:1:0"                 &   # THE POINT at the best arch (MP)
  lane 3 $SD "B:mean:1:1"                 &   # ceiling
  lane 4 $SD "B:attn:0:0"                 &   # four-pool sweep, on shapes
  lane 5 $SD "B:sumtok:0:0"               &
  lane 6 $SD "B:last:0:0"                 &
  lane 7 $SD "A:mean:0:0 A:mean:0:1"      &   # no-shapes control + the BAR
  wait
  echo "########## SEED ROUND $SD COMPLETE  $(date +%H:%M:%S) -- summary 0..$SD"
  uv run --no-sync python decode6_summary.py $R $R5 2>&1 \
    | tee $R/summary_thru_s${SD}.txt
done

echo
echo "===== FINAL SUMMARY (decode6 arms only) ====="
uv run --no-sync python decode6_summary.py $R
echo
echo "===== FINAL SUMMARY (+ stage-B dsA reference arms, 5 seeds) ====="
uv run --no-sync python decode6_summary.py $R /Users/assmuth/dsnn/alphagrad/decode5_out
echo "===== DECODE6 DONE ====="
