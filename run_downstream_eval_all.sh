#!/bin/bash
# Downstream MNIST evaluation for every recorded best sequence across
# all rq[1-5]_* wandb runs + the two reference baselines (jax_grad and
# graphax_rev). Designed to be invoked by ``run_full_research.sh``
# Phase 6, but can also be run standalone after individual RQ phases.
#
# For each wandb run dir with ``rq[1-5]_`` in its name args, this
# replays ``best_overall`` through downstream_train.py. To also replay
# per-channel bests (best_per_channel/flops, …), set EVAL_PER_CHANNEL=1.
#
# Outputs land in $OUT_DIR (default ~/dsnn/out/downstream_full/) as
# ``<run_tag>__<channel>_seed${SEED}.csv``. plot_downstream.py picks
# them up via ``--csv-glob``.

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

ENTRY=alphagrad/src/alphagrad/approx/downstream_train.py
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SEED="${SEED:-250197}"
OUT_DIR="${OUT_DIR:-$HOME/dsnn/out/downstream_full}"
EVAL_PER_CHANNEL="${EVAL_PER_CHANNEL:-0}"
WANDB_DIR="${WANDB_DIR:-$HOME/dsnn/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"            # online | offline | disabled
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-downstream}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
mkdir -p "$OUT_DIR" slurm

# Discover every wandb run whose --name was "rq[1-5]_*". Includes
# both online (``run-<id>-<name>``) and offline
# (``offline-run-<ts>-<id>``, no name in dir) layouts.
mapfile -t RQ_RUNS < <(
    for d in "$WANDB_DIR"/run-* "$WANDB_DIR"/offline-run-*; do
        [[ -d "$d" ]] || continue
        meta="$d/files/wandb-metadata.json"
        if [[ -f "$meta" ]] && grep -qE '"rq[1-5]_' "$meta"; then
            echo "$d"
        fi
    done
)
echo "[downstream-all] discovered ${#RQ_RUNS[@]} RQ wandb runs."

# Build the (label, --gradient-source) list.
declare -a VARIANTS=(
    "jax_grad:jax_grad"
    "graphax_rev:graphax_rev"
)
for run_dir in "${RQ_RUNS[@]}"; do
    tag=$(python3 -c "
import json, sys
m = json.load(open('$run_dir/files/wandb-metadata.json'))
argv = m.get('args', [])
for i, a in enumerate(argv):
    if a == '--name' and i + 1 < len(argv):
        print(argv[i + 1]); break
")
    [[ -z "$tag" ]] && continue
    # Replay best_overall.
    VARIANTS+=("${tag}__best_overall:wandb:${run_dir}:best_overall")
    if [[ "$EVAL_PER_CHANNEL" == "1" ]]; then
        # Per-channel bests — five channels per run, but most converge to
        # the same sequence; the overall is usually enough for analysis.
        for ch in flops peak_memory cosine_sim frob_residual; do
            VARIANTS+=("${tag}__bpc_${ch}:wandb:${run_dir}:best_per_channel/${ch}")
        done
    fi
done

echo "[downstream-all] launching ${#VARIANTS[@]} variants (TRAIN_STEPS=$TRAIN_STEPS)"

declare -a PIDS=()
run_variant() {
    local label=$1
    local gpu=$2
    local source=$3
    local logfile="$OUT_DIR/${label}.out"
    local csv="$OUT_DIR/${label}_seed${SEED}.csv"
    # Skip if already done (resumable mid-phase).
    if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -gt $((TRAIN_STEPS / 2)) ]]; then
        echo "[${label}] skipping (CSV exists with >$((TRAIN_STEPS / 2)) rows)"
        return 0
    fi
    echo "[${label}] gpu=$gpu  source=${source}"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync "$ENTRY" \
            --gradient-source "$source" \
            --train-steps "$TRAIN_STEPS" \
            --eval-every "$EVAL_EVERY" \
            --seed "$SEED" \
            --output-csv "$csv" \
            --wandb "$WANDB_MODE" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-entity "$WANDB_ENTITY" \
            --name "downstream_${label}" \
            > "$logfile" 2>&1 &
    PIDS+=("$!:${label}")
}

drain_batch() {
    local fail=0
    for entry in "${PIDS[@]}"; do
        local pid=${entry%%:*}
        local tag=${entry#*:}
        if wait "$pid"; then
            echo "[${tag}] OK"
        else
            local rc=$?
            echo "[${tag}] FAILED (rc=${rc}) — continuing"
            fail=$((fail + 1))
        fi
    done
    PIDS=()
    [[ $fail -gt 0 ]] && echo "[downstream-all] batch had $fail failure(s)"
}

gpu=0
batch=0
for entry in "${VARIANTS[@]}"; do
    label="${entry%%:*}"
    source="${entry#*:}"
    run_variant "$label" "$gpu" "$source"
    gpu=$(( (gpu + 1) % 4 ))
    batch=$((batch + 1))
    if (( batch >= 4 )); then
        drain_batch
        batch=0
    fi
done
if (( ${#PIDS[@]} > 0 )); then
    drain_batch
fi

echo "[downstream-all] done. CSVs in ${OUT_DIR}."
echo "Plot via:"
echo "  uv run alphagrad/plots/plot_downstream.py \\"
echo "    --csv-glob '${OUT_DIR}/*_seed*.csv' --output-dir ${OUT_DIR}/plots"
