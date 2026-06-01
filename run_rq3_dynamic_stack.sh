#!/bin/bash
#SBATCH --job-name=rq3_dynamic_stack
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ3 (research plan E3): does the per-vertex op-count matter, and
# should it be dynamic? For each family's RQ2-winner variant, sweep
# --max-rules ∈ {1, 2, 3, 16}. The 16-case is "dynamic stack" — the
# agent can stack up to MAX_RULES_PER_VERTEX ops per vertex via the
# OP_END truncation (see docs/experiments/per_vertex_stack_semantics.md).
#
#   3 families × 4 max_rules × 3 seeds = 36 runs
#
# Uses the LOCAL ppo.py trainer (not the Ray one) because only ppo.py
# accepts --max-rules > 1 with the MicroActionHead scan.
#
# Set FAMILY_WINNERS before running; edit per the actual RQ2 outcome.
# Defaults below use the SIMPLE variants per family as placeholder.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq3_dynamic_stack
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

# (family_tag, --variant). Override after RQ2 finishes — pick whichever
# (strict | dynamic) won on the downstream-MNIST plot.
declare -a FAMILY_WINNERS=(
    "diag:diag_factor"          # placeholder; replace after RQ2
    "compress:compress_scalar"  # placeholder
    "quant:quantize"            # placeholder
)
MAX_RULES_VALUES=(1 2 3 16)
SEEDS=(42 250197 1337)

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
    # No anti-degeneracy — see RQ2 sbatch for rationale.
    --anti-degeneracy none
    --cosine-lower-bound 0.0
    --cosine-upper-bound 1.0
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    # CPU pool 1:1 with envs (see run_rq1_ve_only.sh comment).
    --num-cpu-workers "$NUM_ENVS"
    --measure-latency
    # peak_memory / max_io_sum as rollout-wide max (see RQ1 sbatch).
    --running-max-channels peak_memory,max_io_sum
    # 2-phase cost schedule: graphax-only until ep 200, then full.
    --cost-pipeline-schedule cheap_first
    --phase-cutover-ep 200
    --dynamic-substeps
    --wandb "${WANDB_MODE:-online}"
    --wandb-project "${WANDB_PROJECT_RL:-dsnn-vertex}"
    --wandb-entity "${WANDB_ENTITY:-dll-streetview}"
)

run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    # Each PPO variant uses ALL 4 GPUs via Ray sharding. Variants run
    # SEQUENTIALLY in this sbatch so each gets the full GPU allocation.
    echo "[${tag}] all 4 GPUs (Ray-sharded) -> $logfile"
    if uv run --no-sync "$PPO" --name "rq3_${tag}" \
            "${COMMON_ARGS[@]}" "$@" \
            > "$logfile" 2>&1; then
        echo "[${tag}] OK"
        return 0
    else
        local rc=$?
        echo "[${tag}] FAILED (rc=${rc}) — continuing to next variant"
        return $rc
    fi
}


FAIL=0
for entry in "${FAMILY_WINNERS[@]}"; do
    family_tag="${entry%%:*}"
    variant="${entry#*:}"
    for mr in "${MAX_RULES_VALUES[@]}"; do
        for sd in "${SEEDS[@]}"; do
            tag="${family_tag}_mr${mr}_seed${sd}"
            run_variant "$tag" \
                --variant "$variant" \
                --max-rules "$mr" \
                --seed "$sd" || FAIL=$((FAIL + 1))
        done
    done
done

echo "All 36 RQ3 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
