#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu17
#SBATCH --gres=gpu:4
#SBATCH --mem=700G
#SBATCH -c 64
#SBATCH -t 16:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode5_run_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
# Blackwell: the Triton GEMM tiling pass asserts on some of these programs.
# Flag names validated below, not guessed -- an unknown XLA flag is fatal at
# import and would kill every arm.
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode5_out
mkdir -p $R
DA=$S/decode3_dsA.npz

echo "===== XLA_FLAGS validation ====="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo
echo "===== PARAMETER-SHAPE AUDIT + ALT-SHAPE RE-AUDIT, all 8 arms ====="
echo "  (the alt-shape re-audit rebuilds each arm at V=137 / NSTEP=53 and"
echo "   compares leaf shapes elementwise -- strictly stronger than checking"
echo "   that decode4's coincidence flags vanish on another graph)"
for P in attn mean; do for M in 0 1; do for X in 0 1; do
  ( JAX_PLATFORMS=cpu uv run --no-sync python decode5_face_train.py --data $DA \
      --out /dev/null --pool $P --mp $M --extents $X --audit \
      > $R/audit_${P}_mp${M}_ex${X}.txt 2>&1 ; echo "audit $P/$M/$X rc=$?" ) &
done; done; done
# 3*embd_dim becomes 99 at --embd-dim 33: if the (head_width, 96) flag follows
# to (head_width, 99) it was the concat width, never V.
( JAX_PLATFORMS=cpu uv run --no-sync python decode5_face_train.py --data $DA \
    --out /dev/null --pool attn --mp 1 --extents 0 --embd-dim 33 --audit \
    > $R/audit_embd33.txt 2>&1 ; echo "audit embd33 rc=$?" ) &
wait
for f in $R/audit_*.txt; do
  echo "########## $f"
  grep -E "^ARM |OP_TYPE_VOCAB|NOT DAG-AGNOSTIC|DAG-AGNOSTIC:|ALT-SHAPE|LEAF |ALL [0-9]+ LEAF|total trainable|\*\*\*" $f
done

echo
echo "===== TRACE (shape/grad sanity, CPU, all 8 arms) ====="
for P in attn mean; do for M in 0 1; do for X in 0 1; do
  ( JAX_PLATFORMS=cpu uv run --no-sync python decode5_face_train.py --data $DA \
      --out /dev/null --pool $P --mp $M --extents $X --trace-only \
      > $R/trace_${P}_mp${M}_ex${X}.txt 2>&1 ; echo "trace $P/$M/$X rc=$?" ) &
done; done; done
wait
grep -h "TRACE OK" $R/trace_*.txt
NOK=$(grep -l "TRACE OK" $R/trace_*.txt | wc -l)
echo "traces OK: $NOK / 8"
if [ "$NOK" -ne 8 ]; then
  echo "TRACE FAILED -- aborting before burning GPU time"
  for f in $R/trace_*.txt; do echo "##### $f"; tail -25 $f; done
  exit 1
fi

echo
echo "===== 2x2x2 FACTORIAL, 5 seeds, one arm per GPU ====="
run () {  # gpu, pool, mp, extents, seed
  local G=$1 P=$2 M=$3 X=$4 SD=$5
  local N=f5_${P}_mp${M}_ex${X}_s${SD}
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d5_$G CUDA_VISIBLE_DEVICES=$G \
    uv run --no-sync python decode5_face_train.py --data $DA \
    --out $R/$N.json --pool $P --mp $M --extents $X --seed $SD \
    --steps 3000 --batch 4 --chunk-cap 4096 --lr 1e-3 \
    > $R/$N.log 2>&1
  local rc=$?
  echo "  $N rc=$rc  $(date +%H:%M:%S)"
  if [ $rc -ne 0 ]; then
    echo "  $N retrying at batch 3 (different fusion shapes)"
    JAX_COMPILATION_CACHE_DIR=$S/.jc_d5r_$G CUDA_VISIBLE_DEVICES=$G \
      uv run --no-sync python decode5_face_train.py --data $DA \
      --out $R/$N.json --pool $P --mp $M --extents $X --seed $SD \
      --steps 3000 --batch 3 --chunk-cap 4096 --lr 1e-3 \
      > $R/$N.log 2>&1
    echo "  $N (batch 3) rc=$?"
  fi
}

# This node has FOUR GPUs (only gpu19 has eight, and it is busy).  Two arms
# per GPU lane, ONE process at a time -- no co-tenancy, so no OOM roulette and
# no contention skew between arms.  Each lane pairs one MP arm with one
# non-MP arm so the lanes finish together.
#
# SEED IS THE OUTER LOOP: every arm gets seed s before any arm gets seed s+1.
# If the job has to be cut short, all eight arms have the SAME seeds and the
# comparison stays paired (an unpaired comparison has produced a fake 17.5%
# "win" in this repo before).
LANE0="attn:0:0 attn:1:1"
LANE1="attn:0:1 attn:1:0"
LANE2="mean:0:0 mean:1:1"
LANE3="mean:0:1 mean:1:0"

lane () {   # gpu, seed, "p:m:x p:m:x"
  local G=$1 SD=$2; shift 2
  for A_ in $@; do
    IFS=: read -r P M X <<< "$A_"
    run $G $P $M $X $SD
  done
}

for SD in 0 1 2 3 4; do
  echo "########## SEED ROUND $SD  $(date +%H:%M:%S)"
  lane 0 $SD $LANE0 &
  lane 1 $SD $LANE1 &
  lane 2 $SD $LANE2 &
  lane 3 $SD $LANE3 &
  wait
  echo "########## SEED ROUND $SD COMPLETE -- running summary over seeds 0..$SD"
  uv run --no-sync python decode5_summary.py $R 2>&1 | tee $R/summary_thru_s${SD}.txt
done

echo
echo "===== FINAL SUMMARY ====="
uv run --no-sync python decode5_summary.py $R
echo "===== DECODE5 DONE ====="
