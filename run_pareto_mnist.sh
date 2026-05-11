#!/bin/bash
#SBATCH --job-name=pareto_mnist
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Preference-conditioned Pareto-frontier sweep on VmappedNeuralNetwork / MNIST.
#
# One trainer per GPU — all four are conditioned on a per-env Dirichlet
# preference w ∈ Δ^{NUM_REWARDS-1}, so a single trained network covers
# the entire reward simplex. After training, the same net can be queried
# with any w to recover the corresponding Pareto-optimal policy.
#
# This is the headline "simplex coverage" experiment — comparing how
# the four trainer paradigms handle the multi-reward conditioning:
#
#   ppo_pref       : PPO + GAE + autoreg policy + preference projection
#                    The straight-shot baseline. Stage F preference path
#                    is the original PPO implementation.
#
#   alpha0_pref    : Hierarchical MCTS over (vertex, pair, factor) with
#                    visit-count distillation at every depth.
#                    --max-rules 2 --factors=-1,1,2,4
#
#   mu0_pref       : Hierarchical MCTS in latent space (no env.step
#                    inside the search). pref_proj injected into the
#                    leading latent so the dynamics carries it through.
#                    --max-rules 2 --factors=-1,1,2,4
#
#   gfn_pref       : Trajectory-Balance with β annealing + replay.
#                    GFlowNet's natural strength is mode coverage; with
#                    preference conditioning it should sample-from the
#                    full Pareto surface rather than concentrate on one
#                    corner. --beta-schedule linear --replay-buffer-size 1024
#
# All four use --preference-conditioned --preference-dirichlet-alpha 1.0
# (uniform on simplex). Switch alpha to 0.3 to favour corner-only
# preferences (extremal Pareto points), or 3.0 for centroid-like ones.

set -euo pipefail

cd ~/dsnn

export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

PPO=alphagrad/src/alphagrad/approx/ppo.py
ALPHA0=alphagrad/src/alphagrad/approx/alpha0.py
MU0=alphagrad/src/alphagrad/approx/mu0.py
GFN=alphagrad/src/alphagrad/approx/gfn.py

LOG_DIR=~/dsnn/logs_pareto_mnist
mkdir -p "$LOG_DIR" slurm

EPISODES=400
NUM_ENVS=16

COMMON_ENV=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --num-eval-samples 10
    --num-envs "$NUM_ENVS"
    --rewards cmp mem acc
    --wandb offline
    --seed 250197
)

PREF=(
    --preference-conditioned
    --preference-dirichlet-alpha 1.0
)

declare -a PIDS=()

run_variant() {
    local tag=$1
    local gpu=$2
    local entry=$3
    shift 3
    local logfile="$LOG_DIR/${tag}.out"
    echo "[${tag}] CUDA_VISIBLE_DEVICES=$gpu -> $logfile"
    CUDA_VISIBLE_DEVICES="$gpu" \
        uv run --no-sync "$entry" --name "pareto_mnist_${tag}" \
        "${COMMON_ENV[@]}" "${PREF[@]}" "$@" \
        > "$logfile" 2>&1 &
    PIDS+=("$!:${tag}")
}

# PPO: --max-rules controls the rule decoder; autoreg + factors are
# default-on. Long horizon since PPO needs more episodes than the search-
# based methods to find good policies.
run_variant "ppo"    0 "$PPO" \
    --episodes 800 \
    --max-rules 2 \
    --factors=-1,1,2,4 \
    --top-n 20

# AlphaZero: hierarchical MCTS with cache-encoding to keep encoder cost
# in check across the deeper tree.
run_variant "alpha0" 1 "$ALPHA0" \
    --episodes "$EPISODES" \
    --num-simulations 50 \
    --max-rules 2 \
    --factors=-1,1,2,4 \
    --cache-encoding \
    --temperature-init 1.0 --temperature-final 0.1 \
    --temperature-schedule cosine

# MuZero: same hierarchical action space; learned dynamics in latent
# space so num-simulations can be a touch higher.
run_variant "mu0"    2 "$MU0" \
    --episodes "$EPISODES" \
    --num-simulations 50 \
    --unroll-steps 2 \
    --max-rules 2 \
    --factors=-1,1,2,4 \
    --temperature 1.0 --temperature-final 0.1 \
    --temperature-schedule cosine

# GFlowNet: TB with β warm-up (broad exploration → mode-focused) and a
# replay buffer to retain high-reward trajectories across the simplex.
run_variant "gfn"    3 "$GFN" \
    --episodes "$EPISODES" \
    --max-rules 2 \
    --factors=-1,1,2,4 \
    --beta 0.1 --beta-final 1.0 --beta-schedule linear \
    --replay-buffer-size 1024 --replay-warmup 5 \
    --replay-priority-alpha 0.5 \
    --top-n 20

echo "Launched ${#PIDS[@]} trainers. Waiting for completion."

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
    echo "One or more trainers failed. See per-variant logs in ${LOG_DIR}."
    exit 1
fi

echo "All four trainers finished cleanly. Logs in ${LOG_DIR}."
