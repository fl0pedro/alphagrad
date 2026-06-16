#!/bin/bash
# Re-measure cosine_sim for all dumped MORL candidates, sharded across cpu1 cores.
# Runs cmorl then mogfn (bounds peak RAM). Output: ~/dsnn/train_exp/remeasure/<src>_<k>.jsonl
set -u
cd ~/dsnn/alphagrad
NSH="${NSHARDS:-48}"
OUT=~/dsnn/train_exp/remeasure
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
UV="$HOME/.local/bin/uv"
for SRC in cmorl mogfn; do
    echo "[remeasure] $SRC : $NSH shards $(date)"
    pids=()
    for k in $(seq 0 $((NSH-1))); do
        "$UV" run --no-sync python src/alphagrad/approx/remeasure_cossim.py \
            --src "$SRC" --shard "$k" --nshards "$NSH" --out "$OUT/${SRC}_${k}.jsonl" \
            >"$OUT/${SRC}_${k}.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    echo "[remeasure] $SRC done $(date); rows: $(cat $OUT/${SRC}_*.jsonl | wc -l)"
done
echo "[remeasure] ALL DONE $(date)"
