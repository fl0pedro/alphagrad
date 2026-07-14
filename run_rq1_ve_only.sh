#!/bin/bash
#SBATCH --job-name=rq1_ve_only
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# RQ1 (research plan E1): foundation sanity check. With approximation
# ops disabled (`--variant ve_only`), confirm the current PPO agent
# recovers Jamie Lohoff's vertex-elimination result on
# **VmappedNeuralNetwork + MNIST** (the target the rest of the
# research plan is built on), and that the learned order is
# consistent across the three cmp-type proxies
# (graphax → muls_adds_fmas, flops, latency_ns).
#
#   1 example × 3 cmp-types × 3 seeds = 9 runs
#   500 episodes each, 4-way GPU parallel = ~3 batches
#
# CRITICAL: this sbatch is locked to ``--example VmappedNeuralNetwork
# --dataset mnist`` to match the rest of the research plan (RQ2..RQ5
# and downstream-MNIST eval all train against the same model+data).
# Do not introduce other examples here — the analyse_rq1 baseline
# computation also assumes a single (example, dataset).

set -euo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_rq1_ve_only
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"

# Locked: only the model+data the entire research plan targets.
EXAMPLE="VmappedNeuralNetwork"
DATASET="mnist"
CMP_TYPES=(graphax flops latency)
SEEDS=(42 250197 1337)

# Run only the cost channel (cmp). Disable the corridor / anti-degeneracy
# wholesale — RQ1 has no approximations, so the cossim channel is
# constant 1.0 (or undefined) and no Lagrangian pressure is wanted.
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"
WANDB_PROJECT_RL="${WANDB_PROJECT_RL:-dsnn-vertex}"

COMMON_ARGS=(
    --example "$EXAMPLE"
    --dataset "$DATASET"
    --variant ve_only
    --rewards cmp
    --advantage-norm scalar
    --ppo-epochs 4
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    # CPU pool must be 1:1 with env count, otherwise the rollout
    # batch picks N=$NUM_ENVS actors via popleft() and surplus slots
    # get immediate sentinel rewards (zeroed cost channels). Seen
    # in run 46020: pool=4 vs envs=16 → 144/192 sentinel rate.
    --num-cpu-workers "$NUM_ENVS"
    # Force the wall-clock latency reading so latency_ns channel
    # populates every step alongside the other 5 cost channels.
    # Without this flag, env._callback hardcodes latency_ns=0 and
    # downstream comparison loses that axis.
    --measure-latency
    # peak_memory / max_io_sum are max-over-vertices quantities; the
    # default sum-along-rollout buffer over-states the true cost.
    # Telescoping increments make cumsum==rollout-wide max (GAE-additive).
    --running-max-channels peak_memory,max_io_sum
    --wandb "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT_RL"
    --wandb-entity "$WANDB_ENTITY"
)


run_variant() {
    local tag=$1
    shift 1
    local logfile="$LOG_DIR/${tag}.out"
    # Each PPO variant uses ALL 4 GPUs via Ray sharding — single
    # ppo_ray.py invocation, Ray internally distributes the actor
    # pool across the GPUs (see --actor-num-gpus). Variants run
    # SEQUENTIALLY in this sbatch so each gets the full GPU
    # allocation.
    echo "[${tag}] all 4 GPUs (Ray-sharded) -> $logfile"
    if uv run --no-sync "$PPO" --name "rq1_${tag}" \
            "${COMMON_ARGS[@]}" "$@" \
            > "$logfile" 2>&1; then
        echo "[${tag}] OK"
    else
        local rc=$?
        echo "[${tag}] FAILED (rc=${rc}) — continuing to next variant"
        # Non-fatal: continue to the next variant rather than
        # aborting the entire phase. The master sbatch will only
        # mark_done if ALL variants returned 0.
    fi
}

FAIL=0
for ct in "${CMP_TYPES[@]}"; do
    for sd in "${SEEDS[@]}"; do
        tag="${EXAMPLE}_${ct}_seed${sd}"
        if ! run_variant "$tag" --cmp-type "$ct" --seed "$sd"; then
            FAIL=1
        fi
    done
done

echo "All 9 RQ1 variants finished (fail count: $FAIL). Logs in ${LOG_DIR}."
exit "$FAIL"
echo "Run analyser:"
echo "  uv run alphagrad/src/alphagrad/approx/analyse_rq1.py --log-dir ${LOG_DIR}"
