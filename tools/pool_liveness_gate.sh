#!/usr/bin/env bash
# THE POOLED-MEASUREMENT LIVENESS GATE.  Sibling of tools/ratio_gates.sh, run
# from the SAME wave-0 pre-flight (gen_fq_launchers.py's w0_cpu_gates arm).
#
# WHAT IT PINS: that a --ray-measure run MEASURES SOMETHING.
#
# 87cdc49 fixed a fifteen-line regression (introduced by 4c4d872) in which
# CpuApproxPool.evaluate/evaluate_batch forwarded an `episode` kwarg that
# CpuApproximationActor.evaluate did not accept.  Every pooled dispatch died
# with `TypeError: got an unexpected keyword argument 'episode'`, the pool
# sentinelled the row and killed the actor, and under --ray-measure NOTHING
# WAS MEASURED: every terminal reward of every env of every episode was the
# degenerate sentinel (-1e10 on all six cost channels, grad_coverage -1,
# quality 0).
#
# IT WAS INVISIBLE TO EVERY EXISTING GATE.  ratio_gates.sh pins the PPO
# importance ratio; smoke.sh pins finite health rows; neither starts a pool,
# and the smoke config has no --ray-measure at all.  The run exited 0, printed
# health rows and stepped PPO.  The only trace was [SENTINEL] lines that
# nothing reads.  EVERY wave-1..4 arm of docs/EXPERIMENT_PLAN.md runs
# `--ray-measure 3`, so this was ~177 node-hours of pure sentinel that would
# still have looked like a training run.
#
# TWO PARTS, and BOTH ALWAYS RUN -- there is no fast-fail:
#   1. CONTRACT  a static ast read of the pool -> actor -> server call chain.
#                ~10 ms, no ray, no jax.  Names the offending kwarg.
#   2. LIVE      a short real --ray-measure run on NeuralNetwork, then
#                tools/pool_liveness_check.py verdict: >0 terminal plans
#                measured INSIDE a measure actor, ZERO [SENTINEL] lines, and
#                no terminal reward degenerate on all six cost channels.
# Part 1 is a PROXY and part 2 is the EVIDENCE; skipping the evidence because
# the proxy is red is the same mistake this gate exists to correct, and a
# bypass flag on a gate is a bypass flag on the campaign.  In the green case
# both run anyway, so this costs nothing that matters.
#
# A SKIP IS A FAILURE (116c540).  "The gate could not run" exits 2 -- still
# non-zero -- but is reported as HARNESS MISCONFIGURED rather than as
# MEASUREMENT DEAD, because those two call for opposite responses.  The most
# common instance is the one that has already been hit once: --ray-measure
# raises ValueError without ALPHAGRAD_BATCHED_CALLBACK=1.  This script sets it.
#
# Run it on a compute node, never the head node:
#   srun -p pgi15-cpu -w pgi15-cpu1 --mem=96G -c 24 -t 0:30:00 \
#        tools/pool_liveness_gate.sh
#
# Usage:  tools/pool_liveness_gate.sh
#   PY=...                 python launcher     (default: uv run --no-sync python)
#   POOL_GATE_OUT=DIR      artefact directory  (default: ./pool_gate_out)
#   POOL_GATE_TIMEOUT=SEC  wall cap on the run (default: 1500)
#   POOL_GATE_ACTORS=N     measure actors      (default: 2)
#   POOL_GATE_ENVS=N       envs                (default: 4)
#   POOL_GATE_EPISODES=N   episodes            (default: 1)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
ROOT=$PWD
OUT=${POOL_GATE_OUT:-$ROOT/pool_gate_out}
mkdir -p "$OUT" || exit 2
PY=${PY:-uv run --no-sync python}
ACTORS=${POOL_GATE_ACTORS:-2}
ENVS=${POOL_GATE_ENVS:-4}
EPISODES=${POOL_GATE_EPISODES:-1}
GATE_TIMEOUT=${POOL_GATE_TIMEOUT:-1500}

RUN_LOG=$OUT/pool_liveness_run.log
PLAN_LOG=$OUT/pool_liveness_plan_log.jsonl
rm -f "$RUN_LOG" "$PLAN_LOG"

echo "=== POOL LIVENESS 1/2: static wire contract (pool -> actor -> server) ==="
${PY} tools/pool_liveness_check.py contract --repo "$ROOT"
CRC=$?
echo "contract rc=$CRC"

echo
echo "=== POOL LIVENESS 2/2: a real --ray-measure run (${ACTORS} actors, ${ENVS} envs, ${EPISODES} ep) ==="

export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu
# THE TRAP THE HARNESS SETS FOR ITSELF: without this, ppo.py raises
#   ValueError: --ray-measure needs ALPHAGRAD_BATCHED_CALLBACK=1
# and the gate would report a dead measurement path when the truth is that it
# never got to test one.  Set here, and re-detected BY NAME in the checker so
# a future caller that unsets it is told "misconfigured", not "dead".
export ALPHAGRAD_BATCHED_CALLBACK=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_MAX_EQNS=512
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/dsnn/.jax_compile_cache}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_poolgate_${SLURM_JOB_ID:-$$}}
mkdir -p "$RAY_TMPDIR"
# ppo.configure_plan_log EXPORTS these; a stale value in the environment would
# fight the --plan-log flag this gate depends on.
unset ALPHAGRAD_PLAN_LOG ALPHAGRAD_MAX_TOKENS ALPHAGRAD_QUALITY_METRIC

# WHY --no-reject-frozen-grads, ON A GATE, DELIBERATELY.  The frozen-gradient
# guard is DEFAULT ON and it is right to be on in production -- but it RETURNS
# EARLY with `_SENTINEL_BAD_REWARD` before the cost channels are measured.  At
# episode 0 with an untrained face policy on a 25-vertex graph, the sampled
# SKIPs freeze every trainable leaf, so every terminal plan is guard-rejected
# and every recorded reward is degenerate FOR A POLICY REASON.  Measured: at
# HEAD with the guard on, 16/16 plans came back `sentinelled` with all six cost
# channels at -1e10 -- indistinguishable, on the reward vector alone, from the
# dead-pool failure this gate exists to detect.  The guard is therefore turned
# OFF here so that the verdict depends on the TRANSPORT and not on what a
# random policy happened to sample.  A guard-sentinelled plan still proves the
# callback ran in the actor (it built the exact reference and the coverage
# census there), but it proves nothing about whether a COST was measured, and
# "a cost was measured" is the whole claim.
T0=$(date +%s)
timeout "$GATE_TIMEOUT" ${PY} src/alphagrad/approx/ppo.py \
    --variant full --face-actions --unified-face-head --live-faces \
    --set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
    --grad-window 0 --dataset none \
    --cmp-type flops --mem-type peak_memory --terminal-rewards-only \
    --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
    --advantage-norm none --episodes "$EPISODES" --seed 42 \
    --num-envs "$ENVS" --minibatches 1 --no-reject-frozen-grads \
    --vocab-size 512 --wandb disabled --example NeuralNetwork \
    --ray-measure "$ACTORS" --ray-measure-timeout 300 \
    --plan-log "$PLAN_LOG" --name pool_liveness_gate \
    > "$RUN_LOG" 2>&1
RRC=$?
T1=$(date +%s)
echo "run rc=$RRC  wall=$((T1 - T0))s  log=$RUN_LOG"
if [ $RRC -eq 124 ]; then
    echo "  (the run hit the ${GATE_TIMEOUT}s cap -- raise POOL_GATE_TIMEOUT)"
fi

echo
${PY} tools/pool_liveness_check.py verdict \
    --log "$RUN_LOG" --plan-log "$PLAN_LOG" --rc "$RRC"
VRC=$?
echo "verdict rc=$VRC"
if [ $VRC -ne 0 ]; then
    echo
    echo "----- last 30 lines of $RUN_LOG -----"
    tail -30 "$RUN_LOG"
    echo "-------------------------------------"
fi

# ---- combine ---------------------------------------------------------------
# 1 (MEASUREMENT DEAD) dominates 2 (COULD NOT RUN): if either half proved the
# path dead, that is the headline.  Both are non-zero; a SKIP is a FAILURE.
echo
echo "=== POOL LIVENESS SUMMARY: contract=$CRC live=$VRC ==="
if [ $CRC -eq 1 ] || [ $VRC -eq 1 ]; then
    echo "POOLED-MEASUREMENT LIVENESS GATE: RED -- the --ray-measure path is"
    echo "DEAD or its wire contract is broken. A run in this state still exits"
    echo "0, still prints health rows and MEASURES NOTHING: every terminal"
    echo "reward is the degenerate sentinel. DO NOT LAUNCH ANY WAVE."
    exit 1
fi
if [ $CRC -ne 0 ] || [ $VRC -ne 0 ]; then
    echo "POOLED-MEASUREMENT LIVENESS GATE: RED (COULD NOT RUN) -- the harness"
    echo "is misconfigured, so nothing about the measurement path was tested."
    echo "A gate that did not run pins nothing; treat this as a failure."
    exit 2
fi
echo "POOLED-MEASUREMENT LIVENESS GATE: GREEN (contract + live)"
exit 0
