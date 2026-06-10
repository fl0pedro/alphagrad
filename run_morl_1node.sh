#!/bin/bash
# Self-contained single-node MORL run: policy on this node's 4 GPUs, ALL
# measurement on this node's own CPUs (local Ray cluster; measurement actors
# pinned to CPU via JAX_PLATFORMS=cpu). No heterogeneous job, no dedicated CPU
# node — each method gets its own 4-GPU node so they never share CPUs.
#
# Usage (pick the node at submit time):
#   sbatch --nodelist=pgi15-gpu10 run_morl_1node.sh cmorl
#   sbatch --nodelist=pgi15-gpu11 run_morl_1node.sh mogfn
# Override: EPISODES=300 SEED=7 sbatch --nodelist=pgi15-gpu12 run_morl_1node.sh cmorl
#
#SBATCH --job-name=morl1n
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

METHOD="${1:-cmorl}"
EPISODES="${EPISODES:-500}"
# 4 envs is the proven default: the full micro-action agent CUDA-OOMs XLA
# autotuning at 16 envs on the 4×4090 nodes (every production launch overrides
# to 4 anyway — make the default match reality).
NUM_ENVS="${NUM_ENVS:-4}"
SEED="${SEED:-42}"
WANDB="${WANDB:-offline}"
BETA="${BETA:-16}"
OBJ="${OBJECTIVES:-latency_ns,peak_memory,frob_residual}"
# Latency-measurement noise control (2026-06-10 overhaul). Each reading is a
# perf_counter inner-loop of INNER_REPS executions (amortizes dispatch noise;
# replaces the buggy ResourceMonitor wall-timer), latency aggregated by a
# symmetric winsorized mean (LAT_WINSOR) — empirically the most reproducible
# estimator. reps_per_point cut to 2 since each reading is internally denoised.
INNER_REPS="${LATENCY_INNER_REPS:-8}"
LAT_WARMUP="${LATENCY_WARMUP:-3}"
LAT_WINSOR="${LATENCY_WINSOR:-0.2}"
REPS_PER_POINT="${REPS_PER_POINT:-2}"
NUM_DATA_POINTS="${NUM_DATA_POINTS:-5}"
# Single-core-per-actor measurement: each measurement actor pins to exactly 1
# core (cleanest per-reading latency CV, ~25-30× tighter — Agent C). Single-
# threaded exec is slower, so we run MANY actors (≈ #cores) and fan the per-
# (env,point) measurement tasks out across them (--measure-queue) to recover
# throughput via cross-actor parallelism. CPU_CORES_PER_ACTOR=0 reverts to the
# legacy multi-core slice. NUM_CPU_WORKERS defaults high to fill the cores.
CPA="${CPU_CORES_PER_ACTOR:-1}"
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-56}"   # ~one actor per core (64-core node, 8 reserved)
# Off-policy C-MORL: replay buffer of per-env trajectories + V-trace value
# targets / clipped-IS (genuine off-policy actor-critic, parity with MOGFN).
# 0 reverts to on-policy PPO on the fresh rollout.
REPLAY_SIZE="${REPLAY_BUFFER_SIZE:-2048}"
REPLAY_SAMPLE="${REPLAY_SAMPLE_TRAJS:-0}"   # 0 -> num_envs trajectories/update
LOG_DIR=~/dsnn/logs_morl_1node
mkdir -p "$LOG_DIR" slurm
HOST="$(hostname)"
echo "[morl1n] method=$METHOD host=$HOST gpus=4 cpus=$(nproc) $(date)"

# Latency measured on THIS node's CPUs: NUM_DATA_POINTS×REPS_PER_POINT (5×2)
# perf_counter inner-loop readings, aggregated by the LAT_WINSOR winsorized
# mean. (--percentile-keep only governs peak_memory once winsor is active.)
# Local Ray (no --ray-address): the driver starts a cluster on this node; the
# GPU trainer actor is num_cpus=0/num_gpus=4 and the num_cpus=1 measurement
# actors (JAX_PLATFORMS=cpu) run on the node's remaining cores.

if [ "$METHOD" = "cmorl" ]; then
    uv run --no-sync alphagrad/src/alphagrad/approx/cmorl_ray.py \
        --name cmorl_1n_${HOST}_s${SEED} --seed "$SEED" \
        --example VmappedNeuralNetwork --dataset mnist \
        --objectives "$OBJ" \
        --num-prefs 6 --extension-steps 20 --extension-episodes 10 \
        --extension-beta 0.9 --ipo-t 1.0 \
        --episodes "$EPISODES" --num-envs "$NUM_ENVS" \
        --num-cpu-workers "$NUM_CPU_WORKERS" --actor-num-gpus 4 \
        --cpu-cores-per-actor "$CPA" --measure-queue \
        --replay-buffer-size "$REPLAY_SIZE" --replay-sample-trajs "$REPLAY_SAMPLE" \
        --minibatches 4 --ppo-epochs 4 \
        --cmp-type latency --mem-type peak_memory --advantage-norm scalar \
        --num-data-points "$NUM_DATA_POINTS" --reps-per-point "$REPS_PER_POINT" --percentile-keep 0.60 \
        --latency-inner-reps "$INNER_REPS" --latency-warmup "$LAT_WARMUP" --latency-winsor "$LAT_WINSOR" \
        --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 --measure-latency \
        --dynamic-substeps --max-substeps 16 --variant full \
        --calibrate-steps 0 --wandb "$WANDB" \
        2>&1 | tee "$LOG_DIR/cmorl_${HOST}_s${SEED}.out"
elif [ "$METHOD" = "mogfn" ]; then
    uv run --no-sync alphagrad/src/alphagrad/approx/gfn_ray.py \
        --name mogfn_1n_${HOST}_s${SEED} --seed "$SEED" \
        --example VmappedNeuralNetwork --dataset mnist \
        --mogfn-pc --preference-channels "$OBJ" \
        --reward-normalization zscore --scalarization ws --beta "$BETA" \
        --episodes "$EPISODES" --num-envs "$NUM_ENVS" \
        --num-cpu-workers "$NUM_CPU_WORKERS" --spmd-gpus 4 \
        --cpu-cores-per-actor "$CPA" \
        --cmp-type latency --measure-latency \
        --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 \
        --num-data-points "$NUM_DATA_POINTS" --reps-per-point "$REPS_PER_POINT" --percentile-keep 0.60 \
        --latency-inner-reps "$INNER_REPS" --latency-warmup "$LAT_WARMUP" --latency-winsor "$LAT_WINSOR" \
        --dynamic-substeps --max-substeps 16 --variant full --variant-sweep full \
        --calibrate-steps 0 --wandb "$WANDB" \
        2>&1 | tee "$LOG_DIR/mogfn_${HOST}_s${SEED}.out"
else
    echo "unknown method '$METHOD' (use cmorl|mogfn)"; exit 1
fi
# Capture the training pipeline's exit status. ``pipefail`` makes ``$?`` reflect
# a non-zero ``uv run`` even though it's piped into ``tee``. We must propagate it
# as the script's exit code — otherwise the trailing echo (exit 0) masks a crash
# and SLURM reports COMPLETED for a run that actually died (e.g. a CUDA OOM /
# compile failure mid-training).
rc=$?

echo "[morl1n] $METHOD done (rc=$rc) $(date)"
exit "$rc"
