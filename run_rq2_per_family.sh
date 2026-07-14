#!/bin/bash
#SBATCH --job-name=rq2_per_family
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ2 (research plan E2): does each approximation family on its own
# admit a useful learning signal? Single-example (VmappedNeuralNetwork)
# sweep over the 6 family/argument-policy pairs × 3 seeds = 18 runs.
#
#   family         | strict variant       | dynamic variant
#   -------------- | -------------------- | ---------------
#   diag           | diag_gcd             | diag_factor
#   compress       | compress_scalar      | compress
#   quant          | quant_smallest_float | quantize
#
# Anti-degeneracy is on (corridor [0.8, 0.9]) so the policy can't
# collapse to cossim=1.0 — the failure mode of the j2yl2wn7 baseline.
# Reward is the 3-channel (cmp, mem, acc) so the policy weighs cost vs
# quality; lambdas use the dispatch-time calibration where available
# (here we just set them to 1.0 — Infra 7 recommends calibrating but
# RQ2 is intentionally a baseline pass).
#
# Each run writes a best_sequences.json that the post-processor pipes
# through downstream_train.py to measure REAL downstream MNIST
# accuracy with the recorded gradient.

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_rq2_per_family
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

# (family_tag, --variant arg) — order matters for downstream analysis.
declare -a VARIANTS=(
    "diag_strict:diag_gcd"
    "diag_dynamic:diag_factor"
    "compress_strict:compress_scalar"
    "compress_dynamic:compress"
    "quant_strict:quant_smallest_float"
    "quant_dynamic:quantize"
)
SEEDS=(42 250197 1337)

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem acc
    --lambda-cmp 1.0
    --lambda-mem 1.0
    --lambda-frob 1.0
    # No anti-degeneracy: cossim is completely unconstrained. RQ5
    # specifically studies whether a constraint is needed and at what
    # threshold; until then we want to SEE the unconstrained dynamics
    # (incl. potential cossim=1 collapse) cleanly, without confound.
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    # CPU pool 1:1 with envs (see run_rq1_ve_only.sh comment).
    --num-cpu-workers "$NUM_ENVS"
    --measure-latency
    # peak_memory / max_io_sum as rollout-wide max (see RQ1 sbatch).
    --running-max-channels peak_memory,max_io_sum
    # 2-phase cost schedule: phase 1 = graphax symbolic only (cheap,
    # ~10x faster), phase 2 = full path. Cutover at episode 200.
    --cost-pipeline-schedule cheap_first
    --phase-cutover-ep 200
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
    if uv run --no-sync "$PPO" --name "rq2_${tag}" \
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
for entry in "${VARIANTS[@]}"; do
    family_tag="${entry%%:*}"
    variant="${entry#*:}"
    for sd in "${SEEDS[@]}"; do
        tag="${family_tag}_seed${sd}"
        run_variant "$tag" --variant "$variant" --seed "$sd" || FAIL=$((FAIL + 1))
    done
done

echo "All 18 RQ2 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
