#!/bin/bash
#SBATCH --job-name=rq5_anti_degen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ5 (research plan E5): what's the right anti-degeneracy mechanism to
# stop the policy collapsing to cossim=1.0? Sweep:
#
#   anti-degeneracy ∈ {none, delta_ceiling (δ ∈ {0.01, 0.05, 0.1}), corridor}
#   = 5 configs × 3 seeds = 15 runs
#
# Run on the RQ4-winning (mixing × curriculum) combination. Defaults
# below use `full + curriculum` as a placeholder; replace SETUP after
# RQ4's downstream-eval is reviewed.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
LOG_DIR=~/dsnn/logs_rq5_anti_degen
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"
WINNING_VARIANT="${WINNING_VARIANT:-full}"
WINNING_CURRICULUM="${WINNING_CURRICULUM:-full}"
RQ3_BEST_MAX_RULES="${RQ3_BEST_MAX_RULES:-2}"

# (cfg_tag, --anti-degeneracy, --anti-degeneracy-delta).
# corridor / none ignore the delta value.
declare -a CONFIGS=(
    "none:none:0.01"
    "delta01:delta_ceiling:0.01"
    "delta05:delta_ceiling:0.05"
    "delta10:delta_ceiling:0.10"
    "corridor:corridor:0.01"
)
SEEDS=(42 250197 1337)

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
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
    --variant "$WINNING_VARIANT"
    --curriculum "$WINNING_CURRICULUM"
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
    if uv run --no-sync "$PPO" --name "rq5_${tag}" \
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
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r tag mode delta <<< "$cfg"
    for sd in "${SEEDS[@]}"; do
        run_variant "${tag}_seed${sd}" \
            --seed "$sd" || FAIL=$((FAIL + 1))
    done
done
echo "All 15 RQ5 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
