#!/bin/bash
#SBATCH --job-name=downstream_train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Trains VmappedNeuralNetwork on MNIST for `TRAIN_STEPS` steps using each
# gradient source in `VARIANTS` (one per GPU, in parallel). The output is
# a per-source CSV under `OUT_DIR` that `plot_downstream.py` then renders
# into learning-curve comparison plots.
#
# Per the research plan (Infra 3): the load-bearing measurement is
# "does this approximated gradient actually train MNIST?" — not the
# cossim against exact AD. The CSV columns track train loss, test acc,
# wall time, and cossim_vs_exact per step so the plotter can show
# all four side-by-side.
#
# Variants:
#   jax_grad     — exact ``jax.grad`` reference (dashed line in plots).
#   graphax_rev  — graphax cross-country, reverse mode (no approx).
#   wandb:...    — replay learned best sequences from the 3 latest
#                  runs (PPO j2yl2wn7, MuZero 7lt7oiwm, GFN k8qzjo23).
#
# Adjust TRAIN_STEPS for 1k / 10k / 100k budgets per the plan.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

TRAIN_STEPS="${TRAIN_STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
LR="${LR:-1e-3}"
SEED="${SEED:-250197}"
ENTRY=alphagrad/src/alphagrad/approx/downstream_train.py
OUT_DIR=~/dsnn/out/downstream/T${TRAIN_STEPS}_s${SEED}
mkdir -p "$OUT_DIR" slurm

# (variant_label, --gradient-source value). The label is the file
# stem; downstream-plotter groups runs by it.
declare -a VARIANTS=(
    "jax_grad:jax_grad"
    "graphax_rev:graphax_rev"
    "ppo_overall:wandb:/Users/assmuth/dsnn/wandb/run-20260517_073107-j2yl2wn7:best_overall"
    "mu0_overall:wandb:/Users/assmuth/dsnn/wandb/run-20260517_072719-7lt7oiwm:best_overall"
)

declare -a PIDS=()

run_variant() {
    local tag=$1
    local gpu=$2
    local source=$3
    local logfile="$OUT_DIR/${tag}.out"
    local csv="$OUT_DIR/${tag}_seed${SEED}.csv"
    echo "[${tag}] CUDA_VISIBLE_DEVICES=$gpu  source=${source}  -> ${logfile}"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync "$ENTRY" \
            --gradient-source "$source" \
            --train-steps "$TRAIN_STEPS" \
            --eval-every "$EVAL_EVERY" \
            --lr "$LR" \
            --seed "$SEED" \
            --output-csv "$csv" \
            > "$logfile" 2>&1 &
    PIDS+=("$!:${tag}")
}

gpu=0
for entry in "${VARIANTS[@]}"; do
    tag="${entry%%:*}"
    source="${entry#*:}"
    run_variant "$tag" "$gpu" "$source"
    gpu=$((gpu + 1))
done

echo "Launched ${#PIDS[@]} variants. Waiting for completion."

FAIL=0
for entry in "${PIDS[@]}"; do
    pid=${entry%%:*}
    tag=${entry#*:}
    if wait "$pid"; then
        echo "[${tag}] OK (pid=${pid})"
    else
        rc=$?
        echo "[${tag}] FAILED (pid=${pid}, rc=${rc})"
        FAIL=1
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "One or more variants failed. See per-variant logs in ${OUT_DIR}."
    exit 1
fi

echo "All variants finished cleanly. CSVs in ${OUT_DIR}."
echo "Plot with:"
echo "  uv run alphagrad/src/alphagrad/approx/plot_downstream.py --csv-glob '${OUT_DIR}/*_seed*.csv' --output-dir ${OUT_DIR}/plots"
