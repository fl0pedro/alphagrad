#!/bin/bash
#SBATCH --job-name=rq2_eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# After run_rq2_per_family.sh finishes, this script pipes every learned
# best sequence through downstream_train.py to measure REAL downstream
# MNIST training accuracy — the load-bearing measurement per the
# research plan, not the recorded cossim.
#
# Variants:
#   jax_grad      — exact reference (dashed line in plots)
#   graphax_rev   — graphax cross-country reverse mode (no approx)
#   rq2/<family_tag>/best_overall  — the agent's overall best
#   rq2/<family_tag>/best_per_channel/<channel>  — per-channel bests
#
# Each runs for $TRAIN_STEPS (default 10000) and writes a CSV that the
# downstream-plotter consumes.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

ENTRY=alphagrad/src/alphagrad/approx/downstream_train.py
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SEED="${SEED:-250197}"
OUT_DIR=~/dsnn/out/rq2_downstream/T${TRAIN_STEPS}
mkdir -p "$OUT_DIR" slurm

# Find every wandb run dir whose run name starts with rq2_.
WANDB_DIR=~/dsnn/wandb
mapfile -t RQ2_RUNS < <(
    for d in "$WANDB_DIR"/run-*; do
        meta="$d/files/wandb-metadata.json"
        if [[ -f "$meta" ]] && grep -q '"rq2_' "$meta"; then
            echo "$d"
        fi
    done
)

echo "Found ${#RQ2_RUNS[@]} RQ2 wandb runs."

declare -a VARIANTS=(
    "jax_grad:jax_grad"
    "graphax_rev:graphax_rev"
)
for run_dir in "${RQ2_RUNS[@]}"; do
    # Pull the rq2_<tag> name from wandb-metadata.json's args.
    tag=$(python3 -c "
import json
m = json.load(open('$run_dir/files/wandb-metadata.json'))
argv = m.get('args', [])
for i, a in enumerate(argv):
    if a == '--name' and i + 1 < len(argv):
        print(argv[i + 1]); break
")
    VARIANTS+=("${tag}__best_overall:wandb:${run_dir}:best_overall")
done

declare -a PIDS=()

run_variant() {
    local label=$1
    local gpu=$2
    local source=$3
    local logfile="$OUT_DIR/${label}.out"
    local csv="$OUT_DIR/${label}_seed${SEED}.csv"
    echo "[${label}] CUDA_VISIBLE_DEVICES=$gpu  source=${source}"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync "$ENTRY" \
            --gradient-source "$source" \
            --train-steps "$TRAIN_STEPS" \
            --eval-every "$EVAL_EVERY" \
            --seed "$SEED" \
            --output-csv "$csv" \
            > "$logfile" 2>&1 &
    PIDS+=("$!:${label}")
}

drain_batch() {
    local FAIL=0
    for entry in "${PIDS[@]}"; do
        local pid=${entry%%:*}
        local tag=${entry#*:}
        if wait "$pid"; then
            echo "[${tag}] OK"
        else
            local rc=$?
            echo "[${tag}] FAILED (rc=${rc})"
            FAIL=1
        fi
    done
    PIDS=()
    if [[ $FAIL -ne 0 ]]; then
        echo "Some downstream evals failed; continuing anyway."
    fi
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

echo "All downstream evals done. Plot with:"
echo "  uv run alphagrad/src/alphagrad/approx/plot_downstream.py \\"
echo "    --csv-glob '${OUT_DIR}/*_seed*.csv' --output-dir ${OUT_DIR}/plots"
