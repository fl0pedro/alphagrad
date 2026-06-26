#!/bin/bash
# Phase 0 dispersion gate: re-measure a stratified config sample with single-core
# latency (each shard pins its own core; OMP=MKL=1) + aligned cosine + xla_peak.
# Output: ~/dsnn/train_exp/dispersion/d_<k>.jsonl
set -u
cd ~/dsnn/alphagrad
NSH="${NSHARDS:-48}"
NSAMPLE="${NSAMPLE:-400}"
NDATA="${NDATA:-16}"
OUT=~/dsnn/train_exp/dispersion
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
UV="$HOME/.local/bin/uv"
echo "[dispersion] $NSH shards, nsample=$NSAMPLE, ndata=$NDATA $(date)"
pids=()
for k in $(seq 0 $((NSH-1))); do
    "$UV" run --no-sync python src/alphagrad/approx/measure_dispersion.py \
        --shard "$k" --nshards "$NSH" --nsample "$NSAMPLE" --ndata "$NDATA" \
        --out "$OUT/d_${k}.jsonl" >"$OUT/d_${k}.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"
echo "[dispersion] measurement done $(date); rows: $(cat $OUT/d_*.jsonl 2>/dev/null | wc -l)"
echo "[dispersion] affinity sample:"; grep -h "pinned=" "$OUT"/d_0.log "$OUT"/d_1.log 2>/dev/null
"$UV" run --no-sync python src/alphagrad/approx/analyze_cosine_dispersion.py --dir "$OUT"
echo "[dispersion] ALL DONE $(date)"
