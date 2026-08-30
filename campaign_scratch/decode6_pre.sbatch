#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:8
#SBATCH --mem=1200G
#SBATCH -c 128
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/decode6_pre_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256          # never 0: measured 29x FLAT penalty
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS
S=/Users/assmuth/dsnn
R=/Users/assmuth/dsnn/alphagrad/decode6_out
mkdir -p $R
DA=$S/decode3_dsA.npz     # arm A: production tokenizer, NO per-eqn shapes
DB=$S/decode3_dsB.npz     # arm B: ShapeTokenizer, per-eqn OUTPUT SHAPES stated

echo "===== XLA_FLAGS validation (an unknown flag is fatal at import) ====="
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -c \
  "import jax, jax.numpy as jnp; print('XLA_FLAGS OK', jnp.ones(4).sum())" \
  || { echo "XLA_FLAGS REJECTED"; exit 1; }

echo
echo "===== (1) DATASET FRESHNESS: rebuild the first 8 trajectories with"
echo "          TODAY'S code and compare bit-for-bit against the stored npz ====="
( JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $S/decode6_tinyA.npz 8 \
    > $R/tinyA.log 2>&1; echo "buildA rc=$?" ) &
( JAX_PLATFORMS=cpu uv run --no-sync python decode3_data.py $S/decode6_tinyB.npz 8 --shapes \
    > $R/tinyB.log 2>&1; echo "buildB rc=$?" ) &
wait
tail -3 $R/tinyA.log; tail -3 $R/tinyB.log
uv run --no-sync python decode6_dscheck.py $DA $S/decode6_tinyA.npz 2>&1 | tee $R/dscheckA.txt
RA=${PIPESTATUS[0]}
uv run --no-sync python decode6_dscheck.py $DB $S/decode6_tinyB.npz 2>&1 | tee $R/dscheckB.txt
RB=${PIPESTATUS[0]}
echo "dscheck A rc=$RA  B rc=$RB"
if [ "$RA" -ne 0 ] || [ "$RB" -ne 0 ]; then
  echo "STALE DATASET -- decode6 must rebuild before trusting any arm"; exit 1
fi

echo
echo "===== (2) TRACE all nine arms on BOTH datasets (CPU, shapes/grad sanity) ====="
tr () {  # tag, data, pool, mp, ext
  JAX_PLATFORMS=cpu uv run --no-sync python decode5_face_train.py --data $2 \
    --out /dev/null --pool $3 --mp $4 --extents $5 --trace-only \
    > $R/trace_$1.txt 2>&1
  echo "trace $1 rc=$?"
}
tr B_mean_mp0_ex0 $DB mean 0 0 &
tr B_mean_mp0_ex1 $DB mean 0 1 &
tr B_mean_mp1_ex0 $DB mean 1 0 &
tr B_mean_mp1_ex1 $DB mean 1 1 &
tr B_attn_mp0_ex0 $DB attn 0 0 &
tr B_sumtok_mp0_ex0 $DB sumtok 0 0 &
tr B_last_mp0_ex0 $DB last 0 0 &
tr A_mean_mp0_ex0 $DA mean 0 0 &
tr A_mean_mp0_ex1 $DA mean 0 1 &
wait
grep -h "TRACE OK" $R/trace_*.txt
NOK=$(grep -l "TRACE OK" $R/trace_*.txt | wc -l)
echo "traces OK: $NOK / 9"
if [ "$NOK" -ne 9 ]; then
  echo "TRACE FAILED -- aborting before burning GPU time"
  for f in $R/trace_*.txt; do echo "##### $f"; tail -25 $f; done
  exit 1
fi
echo
echo "----- chunk-cap truncation, as reported by the loader (must be 0.0%) -----"
grep -h "^chunk len\|^data N\|^faces/traj\|^buckets\|^DEGENERATE" $R/trace_B_mean_mp0_ex0.txt $R/trace_A_mean_mp0_ex0.txt $R/trace_B_sumtok_mp0_ex0.txt

echo
echo "===== (3) COST + PEAK-MEMORY SMOKE on the SHAPES dataset ====="
echo "  dsB streams are 2.8x dsA (mean 105594 vs 43678, max 339603 vs 122910),"
echo "  so batch 4 at the largest bucket is the OOM risk and the cost driver."
( nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 \
    > $R/nvsmi.csv 2>&1 ) &
NVPID=$!
smoke () {  # gpu, tag, data, pool, mp, ext, batch
  JAX_COMPILATION_CACHE_DIR=$S/.jc_d6_$1 CUDA_VISIBLE_DEVICES=$1 \
    uv run --no-sync python decode5_face_train.py --data $3 \
    --out $R/smoke_$2.json --pool $4 --mp $5 --extents $6 --seed 0 \
    --steps 30 --batch $7 --chunk-cap 4096 --lr 1e-3 \
    > $R/smoke_$2.log 2>&1
  echo "smoke $2 (batch $7) rc=$?  $(grep -c 'eval @' $R/smoke_$2.log) evals"
}
smoke 0 B_mp1_b4 $DB mean 1 0 4 &
smoke 1 B_mp0_b4 $DB mean 0 0 4 &
smoke 2 B_mp0_b2 $DB mean 0 0 2 &
smoke 3 A_mp0_b4 $DA mean 0 0 4 &
wait
kill $NVPID 2>/dev/null
for t in B_mp1_b4 B_mp0_b4 B_mp0_b2 A_mp0_b4; do
  echo "##### $t"
  grep -E "^data N|^chunk len|^buckets|eval @ 1 |eval @ 30 |Error|error|RESOURCE|OOM" $R/smoke_$t.log | head -12
  tail -3 $R/smoke_$t.log
done
echo
echo "----- peak GPU memory seen during the smoke, per GPU (MiB) -----"
uv run --no-sync python - <<'PY'
import collections
mx = collections.defaultdict(int)
for line in open("/Users/assmuth/dsnn/alphagrad/decode6_out/nvsmi.csv"):
    p = line.split(",")
    if len(p) != 2:
        continue
    try:
        mx[int(p[0])] = max(mx[int(p[0])], int(p[1].strip().split()[0]))
    except ValueError:
        pass
for g in sorted(mx):
    print(f"  gpu{g}  peak {mx[g]} MiB")
PY
echo "===== DECODE6 PREFLIGHT DONE ====="
