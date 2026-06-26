#!/bin/bash
#SBATCH --job-name=mu0_mnist
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Hierarchical-MCTS MuZero sweep on VmappedNeuralNetwork / MNIST.
#
# Four variants run in parallel (one GPU each) — each combines the new
# hierarchical-MCTS action layout with one of the diversity / replay
# mechanisms. mu0 runs MCTS entirely in latent space (dynamics network
# replaces env.step inside the search) so larger ``num-simulations`` is
# cheap relative to alpha0:
#
#   mu0_baseline   : --max-rules 1 --factors=-1
#                    Single-rule, no factor choice. DECISION_DEPTH=3.
#
#   mu0_factors    : --max-rules 1 --factors=-1,1,2,4
#                    Single-rule, discrete factor choice.
#
#   mu0_replay     : --max-rules 2 --factors=-1,1,2,4
#                    --replay-buffer-size 1024 --replay-warmup 5
#                    --replay-priority-alpha 0.5
#                    Self-play archive with prioritised replay (priority
#                    = shifted positive episode return). Tests whether
#                    re-sampling high-return trajectories accelerates
#                    convergence.
#
#   mu0_pref       : --max-rules 2 --factors=-1,1,2,4
#                    --preference-conditioned --preference-dirichlet-alpha 1.0
#                    Single conditional model across the reward simplex.
#                    Per-env Dirichlet preference is added to the leading
#                    latent via ``MuZeroAgent.pref_proj``.
#
# Visit-count distillation (per-depth CE) is on by default in the new
# mu0 — every variant gets it. The reward head is trained against
# ``[0, 0, ..., env_reward]`` per real step (zero at intermediate
# decision steps, env reward at commit).
#
# Wall-time budget ~3d per variant.

set -euo pipefail

cd ~/dsnn

export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

MU0=alphagrad/src/alphagrad/approx/mu0.py
LOG_DIR=~/dsnn/logs_mu0_mnist
mkdir -p "$LOG_DIR" slurm

EPISODES=300
NUM_ENVS=16
NUM_SIMULATIONS=50

COMMON=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --num-eval-samples 10
    --episodes "$EPISODES"
    --num-envs "$NUM_ENVS"
    --num-simulations "$NUM_SIMULATIONS"
    --unroll-steps 2
    --rewards cmp mem acc
    --temperature 1.0
    --temperature-final 0.1
    --temperature-schedule cosine
    --wandb offline
    --seed 250197
)

declare -a PIDS=()

run_variant() {
    local tag=$1
    local gpu=$2
    shift 2
    local logfile="$LOG_DIR/${tag}.out"
    echo "[${tag}] CUDA_VISIBLE_DEVICES=$gpu -> $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync "$MU0" --name "mu0_mnist_${tag}" "${COMMON[@]}" "$@" \
        > "$logfile" 2>&1 &
    PIDS+=("$!:${tag}")
}

run_variant "baseline" 0 \
    --max-rules 1 --factors=-1
run_variant "factors"  1 \
    --max-rules 1 --factors=-1,1,2,4
run_variant "replay"   2 \
    --max-rules 2 --factors=-1,1,2,4 \
    --replay-buffer-size 1024 --replay-warmup 5 \
    --replay-priority-alpha 0.5
run_variant "pref"     3 \
    --max-rules 2 --factors=-1,1,2,4 \
    --preference-conditioned --preference-dirichlet-alpha 1.0

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
    echo "One or more variants failed. See per-variant logs in ${LOG_DIR}."
    exit 1
fi

echo "All four variants finished cleanly. Logs in ${LOG_DIR}."
