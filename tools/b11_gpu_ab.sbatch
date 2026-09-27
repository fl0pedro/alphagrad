#!/bin/bash
# Merge batch 11 on the GPU, in one job: the paired A/B of three graphax trees on RSNN_SHD rtrl and bptt
# (mem_objective_test's program and the thesis rows), NN256 and TLM, then the GPU tests of dsnn-dfw.304 and .305.
#   core       core-v2 432d0ce, the base
#   head       agent/rsnnfix 8dd9423: the interior pad in the join, the barrier at every narrow private sum
#   nobarrier  fee0f72: the same head without the barrier at narrow private sums
# The measurement compiles with the compiler options of this alphagrad tree (dsnn-dfw.287 decides them).
# Submit from an alphagrad clone at the commit that carries this file, with the graphax commits on GX_REPO:
#   sbatch tools/b11_gpu_ab.sbatch
# Another node: sbatch -w pgi15-gpuNN -J node-pgi15-gpuNN tools/b11_gpu_ab.sbatch (4 Blackwell GPUs, not gpu17).
# Variables (sbatch --export=ALL,NAME=value): PY GX_REPO GX_CORE GX_HEAD GX_NOBARRIER TARGETS OUT.
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu19
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH -t 06:00:00
#SBATCH -J node-pgi15-gpu19
#SBATCH --dependency=singleton
#SBATCH -o b11_gpu_ab_%j.out
set -uo pipefail

PY=${PY:-/Scratch/assmuth/t57/stack/venv/bin/python}
GX_REPO=${GX_REPO:-/Scratch/assmuth/mrg/relay/graphax.git}
GX_CORE=${GX_CORE:-432d0ce4d706a6fa270106cba2b41d665bc0ebc4}
GX_HEAD=${GX_HEAD:-8dd94239c19896d84590b11c7d23bd4ed7113fb6}
GX_NOBARRIER=${GX_NOBARRIER:-fee0f7222ecda7ab0ac255087f222ea249f0a8b8}
TARGETS=${TARGETS:-memobj_rtrl memobj_bptt nn256 tlm rsnn_rtrl rsnn_bptt}
AG=${SLURM_SUBMIT_DIR:-$PWD}
OUT=${OUT:-/Scratch/assmuth/mrg/b11-gpu-ab/${SLURM_JOB_ID:-local}}
mkdir -p "$OUT/logs" "$OUT/gx"
cd "$AG"

# ---------------------- MEASURE TOOLCHAIN (as the thesis rows) ---------------------
FQ_CUDA_WANT=12.9
FQ_CUDA_BIN=""
for d in /usr/local/cuda-*/bin; do
  [ -x "$d/ptxas" ] && [ -x "$d/nvlink" ] || continue
  pv=$("$d/ptxas" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
  nv=$("$d/nvlink" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
  if [ "$pv" = "$FQ_CUDA_WANT" ] && [ "$nv" = "$FQ_CUDA_WANT" ]; then FQ_CUDA_BIN=$d; fi
done
if [ -n "$FQ_CUDA_BIN" ]; then
  export PATH="$FQ_CUDA_BIN:$PATH"
fi
FQ_PTXAS_VER=$(ptxas --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
FQ_NVLINK_VER=$(nvlink --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
echo "[cfg] measure toolchain on $(hostname -s): PATH<-${FQ_CUDA_BIN:-<none>} ptxas ${FQ_PTXAS_VER:-none} nvlink ${FQ_NVLINK_VER:-none}"
if [ "$FQ_PTXAS_VER" != "$FQ_CUDA_WANT" ] || [ "$FQ_NVLINK_VER" != "$FQ_CUDA_WANT" ]; then
  echo "ABORT(72): measure toolchain on $(hostname -s) is ptxas ${FQ_PTXAS_VER:-none} / nvlink ${FQ_NVLINK_VER:-none}, want $FQ_CUDA_WANT"
  exit 72
fi

# ---------------------- TREES ---------------------
if [ -n "$(git -C "$AG" status --short --untracked-files=no)" ]; then
  echo "ABORT(73): the alphagrad clone $AG has changes; this job runs on committed code"
  git -C "$AG" status --short --untracked-files=no
  exit 73
fi
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-local} GPUS=${CUDA_VISIBLE_DEVICES:-unset} OUT=$OUT"
echo "alphagrad $(git -C "$AG" rev-parse HEAD) ($AG)"
declare -A GX=([core]=$GX_CORE [head]=$GX_HEAD [nobarrier]=$GX_NOBARRIER)
for name in core head nobarrier; do
  sha=${GX[$name]}
  if ! git -C "$GX_REPO" cat-file -e "$sha^{commit}"; then
    echo "ABORT(74): graphax $name commit $sha is not in $GX_REPO"
    exit 74
  fi
  full=$(git -C "$GX_REPO" rev-parse "$sha^{commit}")
  mkdir -p "$OUT/gx/$name"
  git -C "$GX_REPO" archive --format=tar "$full" src | tar -x -C "$OUT/gx/$name"
  echo "graphax $name = $full $(git -C "$GX_REPO" log -1 --format=%s "$full" | cut -c1-100)"
done
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#GPUS[@]}" -lt 4 ]; then
  echo "ABORT(75): this job wants 4 GPUs and has ${CUDA_VISIBLE_DEVICES:-none}"
  exit 75
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
T0=$SECONDS

# ---------------------- 1. PLANS (CPU, under the head) ---------------------
for t in $TARGETS; do
  JAX_PLATFORMS=cpu PYTHONPATH="$OUT/gx/head/src:$AG/src" \
    "$PY" tools/b11_gpu_ab.py plans --target "$t" --out "$OUT" > "$OUT/logs/plans_$t.log" 2>&1 &
done
wait
for t in $TARGETS; do
  echo "--- plans $t"; grep -E "^\[plans\]|Error" "$OUT/logs/plans_$t.log" | tail -12
done
echo "[time] plans done at $((SECONDS - T0)) s"

# ---------------------- 2. COMPILE (one GPU per tree) ---------------------
compile_variant() {  # name gpu
  local v=$1 gpu=$2
  for t in $TARGETS; do
    CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/$v/src:$AG/src" \
      "$PY" tools/b11_gpu_ab.py compile --target "$t" --variant "$v" --out "$OUT" \
      > "$OUT/logs/compile_${v}_$t.log" 2>&1
  done
}
compile_variant core "${GPUS[0]}" &
compile_variant head "${GPUS[1]}" &
compile_variant nobarrier "${GPUS[2]}" &
wait
grep -h "^\[compile\]" "$OUT"/logs/compile_*.log | "$PY" -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line[len("[compile] "):])
    print("%-9s %-12s %-18s ok=%s temp=%10.2f MB compile=%s s barriers=%s concat=%s MB joined=%s %s" % (
        r["variant"], r["target"], r["pid"], r["ok"], r.get("temp", 0) / 2 ** 20, r.get("compile_s"),
        r.get("barriers"), r.get("largest_concat_mb"), r.get("carry_stacks_joined"), r.get("err", "")[:160]))'
echo "[time] compile done at $((SECONDS - T0)) s"

# ---------------------- 3. TIME (one process, paired, base = core-v2) ---------------------
CUDA_VISIBLE_DEVICES=${GPUS[0]} JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/head/src:$AG/src" \
  "$PY" tools/b11_gpu_ab.py time --out "$OUT" --variants core,head,nobarrier \
  --targets "$(echo $TARGETS | tr ' ' ,)" > "$OUT/logs/time.log" 2>&1
echo "time exit $?"
echo "[time] timing done at $((SECONDS - T0)) s"

# ---------------------- 4. GPU TESTS (one GPU each) ---------------------
gpu_test() {  # label gx-tree gpu test-file
  CUDA_VISIBLE_DEVICES=$3 JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/$2/src:$AG/src" \
    "$PY" -m pytest -q -s -p no:cacheprovider "gpu_tests/$4" > "$OUT/logs/test_$1.log" 2>&1
  echo "$1 exit $?" > "$OUT/logs/test_$1.rc"
}
gpu_test head_narrow head "${GPUS[0]}" narrow_private_sum_gpu_test.py &
gpu_test nobarrier_narrow nobarrier "${GPUS[1]}" narrow_private_sum_gpu_test.py &
gpu_test head_concat head "${GPUS[2]}" rtrl_carry_concat_gpu_test.py &
gpu_test core_concat core "${GPUS[3]}" rtrl_carry_concat_gpu_test.py &
wait

echo
echo "=================== SUMMARY (job ${SLURM_JOB_ID:-local}, $((SECONDS - T0)) s) ==================="
echo "base = graphax core ${GX[core]}; head = ${GX[head]}; nobarrier = ${GX[nobarrier]}"
cat "$OUT/summary.txt"
echo
echo "GPU tests (expected: head_narrow 0, nobarrier_narrow 1, head_concat 1 while dsnn-dfw.304 is open, core_concat 0):"
cat "$OUT"/logs/test_*.rc
for f in "$OUT"/logs/test_*.log; do
  echo "--- $(basename "$f")"; grep -E "^\[|passed|failed|Error" "$f" | tail -6
done
echo "rows: $OUT/rows.jsonl; optimized HLO: $OUT/hlo/; logs: $OUT/logs/"
