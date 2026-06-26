#!/bin/bash
#SBATCH --job-name=alpha0_mnist
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
##SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Hierarchical-MCTS AlphaZero sweep on VmappedNeuralNetwork / MNIST.
#
# Four variants run in parallel (one GPU each) — each exercises a
# different point on the (action-space-richness × diversity-mechanism)
# axis. All four use the new PPO-Agent backbone (encoder + pointer
# vertex + AutoregRulePolicy) with hierarchical MCTS:
#
#   alpha0_baseline : --max-rules 1 --factors=-1
#                     Single-rule, factor pinned at -1. DECISION_DEPTH=3.
#                     The minimal "vertex / pair / factor" hierarchy.
#
#   alpha0_factors  : --max-rules 1 --factors=-1,1,2,4
#                     Single-rule with discrete factor choice.
#                     DECISION_DEPTH=3 (factor head now non-degenerate).
#
#   alpha0_multi    : --max-rules 2 --factors=-1,1,2,4
#                     Two-rule autoreg with factor choice. DECISION_DEPTH=5.
#                     Tests whether richer per-vertex sparsity patterns
#                     find better elimination orders.
#
#   alpha0_pref     : --max-rules 2 --factors=-1,1,2,4
#                     --preference-conditioned --preference-dirichlet-alpha 1.0
#                     Same action space as alpha0_multi but trains one
#                     conditional policy across the whole reward simplex.
#
# Visit-count distillation (per-depth CE against MCTS visits) is on by
# default in the new alpha0 — every variant gets it. Encoder cache is
# also on so the rollout's MCTS expansions reuse the initial-jaxpr
# encoding via residual_state.
#
# MNIST graph is heavy + the env eval-callback runs the compiled approx
# fn each step. Hierarchical MCTS adds DECISION_DEPTH-fold expansions
# vs. flat alpha0; budget ~3-4d per variant to be safe.

set -euo pipefail

cd ~/dsnn

# `alias uv=...` is shell-only; export PATH so subprocess.Popen can find it too.
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

ALPHA0=alphagrad/src/alphagrad/approx/alpha0.py
LOG_DIR=~/dsnn/logs_alpha0_mnist
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
    --rewards cmp mem acc
    --cache-encoding
    --temperature-init 1.0
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
        uv run --no-sync "$ALPHA0" --name "alpha0_mnist_${tag}" "${COMMON[@]}" "$@" \
        > "$logfile" 2>&1 &
    PIDS+=("$!:${tag}")
}

run_variant "baseline" 0 \
    --max-rules 1 --factors=-1
run_variant "factors"  1 \
    --max-rules 1 --factors=-1,1,2,4
run_variant "multi"    2 \
    --max-rules 2 --factors=-1,1,2,4
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
