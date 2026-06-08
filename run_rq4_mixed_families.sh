#!/bin/bash
#SBATCH --job-name=rq4_mixed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ4 (research plan E4): do mixed-family sequences push the
# downstream frontier further than single-family ones?
#
#   mixing ∈ {ve_only (control), single (RQ2 winner), full (3-family)}
#   3 mixings × 3 seeds = 9 runs
#
# **Curriculum axis dropped** — it moved to RQ6 (granular additive
# curriculum). RQ4 keeps the mixing axis only so the comparison
# between single-family and mixed-family is uncontaminated by
# curriculum scheduling effects.
#
# RQ4's "single" arm is whichever (family + arg-policy) won RQ2 on
# downstream MNIST accuracy. EDIT SINGLE_WINNER / RQ3_BEST_MAX_RULES
# below before launching.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq4_mixed
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

# Edit after RQ2 + RQ3 results are reviewed.
SINGLE_WINNER="${SINGLE_WINNER:-diag_factor}"     # placeholder
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"     # placeholder

declare -a MIXINGS=(
    "ve_only:ve_only"
    "single:${SINGLE_WINNER}"
    "full:full"
)
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
    --max-rules "$RQ3_BEST_MAX_RULES"
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
    if uv run --no-sync "$PPO" --name "rq4_${tag}" \
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
for mix_entry in "${MIXINGS[@]}"; do
    mix_tag="${mix_entry%%:*}"
    variant="${mix_entry#*:}"
    for sd in "${SEEDS[@]}"; do
        tag="${mix_tag}_seed${sd}"
        run_variant "$tag" --variant "$variant" --seed "$sd" || FAIL=$((FAIL + 1))
    done
done
echo "All 9 RQ4 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
