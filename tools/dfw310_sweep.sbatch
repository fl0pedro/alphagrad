#!/bin/bash
# dsnn-dfw.310 on the GPU, in one job: which graphax commit of merge batch 11 slows RSNN_SHD's rtrl Diag container
# and its bptt exact plan. Job 68488 timed the head 8dd9423 against core-v2 432d0ce at 1.08 on the thesis target's
# rtrl diag_container, 1.04 on its bptt exact_rev and 1.15 on mem_objective_test's rtrl diag_container, with and
# without the narrow-sum barrier. This job times those plans, and mem_objective_test's rtrl exact plan (1.04 there),
# under every graphax src tree from core-v2 to the head, paired in ONE process against core-v2 (the time step of
# tools/b11_gpu_ab.py). A commit whose src tree an earlier commit already had is the same program: it is compiled
# once, under the earlier commit's name. Then the GPU test of dsnn-dfw.304 under the head; core-v2's side of it is
# its compile row here, since the test's bound is core-v2's own temp.
# Submit from an alphagrad clone at the commit that carries this file, with the graphax commits on GX_REPO:
#   sbatch tools/dfw310_sweep.sbatch
# Another node: sbatch -w pgi15-gpuNN -J node-pgi15-gpuNN tools/dfw310_sweep.sbatch (4 Blackwell GPUs, not gpu17).
# Variables (sbatch --export=ALL,NAME=value): PY GX_REPO GX_BASE GX_HEAD OUT.
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu16
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH -t 04:00:00
#SBATCH -J node-pgi15-gpu16
#SBATCH --dependency=singleton
#SBATCH -o dfw310_sweep_%j.out
set -uo pipefail

PY=${PY:-/Scratch/assmuth/t57/stack/venv/bin/python}
GX_REPO=${GX_REPO:-/Scratch/assmuth/mrg/relay/graphax.git}
GX_BASE=${GX_BASE:-432d0ce4d706a6fa270106cba2b41d665bc0ebc4}
GX_HEAD=${GX_HEAD:-8dd94239c19896d84590b11c7d23bd4ed7113fb6}
AG=${SLURM_SUBMIT_DIR:-$PWD}
OUT=${OUT:-/Scratch/assmuth/mrg/dfw310-sweep/${SLURM_JOB_ID:-local}}
# The plans of job 68488 this job times, by target.
TARGETS="rsnn_rtrl rsnn_bptt memobj_rtrl"
declare -A ONLY=([rsnn_rtrl]="diag_container" [rsnn_bptt]="exact_rev" [memobj_rtrl]="diag_container exact_rev")
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
for c in "$GX_BASE" "$GX_HEAD"; do
  if ! git -C "$GX_REPO" cat-file -e "$c^{commit}"; then
    echo "ABORT(74): graphax commit $c is not in $GX_REPO"
    exit 74
  fi
done
if ! git -C "$GX_REPO" merge-base --is-ancestor "$GX_BASE" "$GX_HEAD"; then
  echo "ABORT(74): graphax $GX_BASE is not an ancestor of $GX_HEAD"
  exit 74
fi
# The variants in commit order: core-v2 first (the base of every ratio), then each new src tree up to the head.
VARIANTS=()
declare -A SEEN=()
for c in "$GX_BASE" $(git -C "$GX_REPO" rev-list --reverse "$GX_BASE..$GX_HEAD"); do
  full=$(git -C "$GX_REPO" rev-parse "$c^{commit}")
  tree=$(git -C "$GX_REPO" rev-parse "$full:src")
  subject=$(git -C "$GX_REPO" log -1 --format=%s "$full" | cut -c1-100)
  if [ -n "${SEEN[$tree]:-}" ]; then
    echo "graphax ${full:0:7} = ${SEEN[$tree]} (the same src tree) $subject"
    continue
  fi
  SEEN[$tree]=${full:0:7}
  VARIANTS+=("${full:0:7}")
  mkdir -p "$OUT/gx/${full:0:7}"
  git -C "$GX_REPO" archive --format=tar "$full" src | tar -x -C "$OUT/gx/${full:0:7}"
  echo "graphax ${full:0:7} $full $subject"
done
BASE=${VARIANTS[0]}
TIP=${SEEN[$(git -C "$GX_REPO" rev-parse "$GX_HEAD:src")]}
echo "variants ${#VARIANTS[@]}: ${VARIANTS[*]} (base $BASE, head $TIP)"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#GPUS[@]}" -lt 4 ]; then
  echo "ABORT(75): this job wants 4 GPUs and has ${CUDA_VISIBLE_DEVICES:-none}"
  exit 75
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
T0=$SECONDS

# ---------------------- 1. PLANS (CPU, under the head) ---------------------
for t in $TARGETS; do
  only=()
  for p in ${ONLY[$t]}; do only+=(--only "$p"); done
  JAX_PLATFORMS=cpu PYTHONPATH="$OUT/gx/$TIP/src:$AG/src" \
    "$PY" tools/b11_gpu_ab.py plans --target "$t" --out "$OUT" "${only[@]}" > "$OUT/logs/plans_$t.log" 2>&1 &
done
wait
for t in $TARGETS; do
  echo "--- plans $t"; grep -E "^\[plans\]|Error" "$OUT/logs/plans_$t.log" | tail -12
done
echo "[time] plans done at $((SECONDS - T0)) s"

# ---------------------- 2. COMPILE (four GPUs, the variants dealt round-robin) ---------------------
compile_variants() {  # gpu, then the variants it compiles one after another
  local gpu=$1 v t
  shift
  for v in "$@"; do
    for t in $TARGETS; do
      CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/$v/src:$AG/src" \
        "$PY" tools/b11_gpu_ab.py compile --target "$t" --variant "$v" --out "$OUT" \
        > "$OUT/logs/compile_${v}_$t.log" 2>&1
    done
  done
}
for w in 0 1 2 3; do
  mine=()
  for i in "${!VARIANTS[@]}"; do
    if [ $((i % 4)) = "$w" ]; then mine+=("${VARIANTS[$i]}"); fi
  done
  compile_variants "${GPUS[$w]}" "${mine[@]}" &
done
wait
for v in "${VARIANTS[@]}"; do grep -h "^\[compile\]" "$OUT"/logs/compile_"$v"_*.log; done | "$PY" -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line[len("[compile] "):])
    print("%-9s %-12s %-18s ok=%s temp=%10.2f MB compile=%s s barriers=%s concat=%s MB joined=%s %s" % (
        r["variant"], r["target"], r["pid"], r["ok"], r.get("temp", 0) / 2 ** 20, r.get("compile_s"),
        r.get("barriers"), r.get("largest_concat_mb"), r.get("carry_stacks_joined"), r.get("err", "")[:160]))'
echo "[time] compile done at $((SECONDS - T0)) s"

# ---------------------- 3. TIME (one process, paired, base = core-v2) ---------------------
CUDA_VISIBLE_DEVICES=${GPUS[0]} JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/$TIP/src:$AG/src" \
  "$PY" tools/b11_gpu_ab.py time --out "$OUT" --variants "$(IFS=,; echo "${VARIANTS[*]}")" \
  --targets "$(echo $TARGETS | tr ' ' ,)" > "$OUT/logs/time.log" 2>&1
echo "time exit $?"
echo "[time] timing done at $((SECONDS - T0)) s"

# ---------------------- 4. GPU TEST of dsnn-dfw.304 (under the head) ---------------------
CUDA_VISIBLE_DEVICES=${GPUS[0]} JAX_PLATFORMS=cuda PYTHONPATH="$OUT/gx/$TIP/src:$AG/src" \
  "$PY" -m pytest -q -s -p no:cacheprovider gpu_tests/rtrl_carry_concat_gpu_test.py > "$OUT/logs/test_head_concat.log" 2>&1
echo "head_concat exit $?" > "$OUT/logs/test_head_concat.rc"

echo
echo "=================== SUMMARY (job ${SLURM_JOB_ID:-local}, $((SECONDS - T0)) s) ==================="
echo "base = graphax core-v2 $BASE; variants in commit order: ${VARIANTS[*]}"
cat "$OUT/summary.txt"
echo
echo "GPU test of dsnn-dfw.304 under the head (expected 0: joined or not, temp not above core-v2's):"
cat "$OUT/logs/test_head_concat.rc"
grep -E "^\[|passed|failed|Error" "$OUT/logs/test_head_concat.log" | tail -6
echo "rows: $OUT/rows.jsonl; optimized HLO: $OUT/hlo/; logs: $OUT/logs/"
