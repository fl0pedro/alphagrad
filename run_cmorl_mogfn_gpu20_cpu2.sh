#!/bin/bash
# Two trainers sharing one heterogeneous Ray cluster, DYNAMIC SUBSTEPS ON:
#   het-group 0 = pgi15-gpu20 (8 GPUs) -> Run A: C-MORL (4 GPU) + Run B: MOGFN (4 GPU)
#   het-group 1 = pgi15-cpu2  (384 CPU) -> ALL measurement actors
#
# Both GPU trainer actors are scheduled with num_cpus=0 (ppo_ray + the gfn_ray
# patch), and the gpu20 Ray head is started with --num-cpus=0, so every
# num_cpus=1 measurement actor is forced onto cpu2. Both runs claim disjoint
# cpu2 core slices via --cpu-cores-shared (the cluster-wide core allocator).
#
# Dynamic substeps: --dynamic-substeps + --variant full (full DIAG/COMPRESS/
# QUANT micro-action space). MOGFN's agent is autoregressive over micro-actions
# by construction; --variant full unlocks the same op set.
#
# Objectives: latency_ns, peak_memory, frob_residual (peak_memory=0 == "<~50MB,
# too small to measure" — kept as useful signal). Latency = env default 5×4 @
# 60th-slowest, measured on cpu2.
#
# Submit:  sbatch run_cmorl_mogfn_gpu20_cpu2.sh
# Override: EPISODES=300 SEED=7 sbatch run_cmorl_mogfn_gpu20_cpu2.sh
#
#SBATCH --job-name=cmorlmogfn
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15
#SBATCH --nodelist=pgi15-gpu20
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH hetjob
#SBATCH --partition=pgi15-cpu
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --cpus-per-task=256

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

CMORL=alphagrad/src/alphagrad/approx/cmorl_ray.py
MOGFN=alphagrad/src/alphagrad/approx/gfn_ray.py
LOG_DIR=~/dsnn/logs_cmorl_mogfn
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-16}"   # per run; 2 runs => 32 actors on cpu2
GPUS_PER_RUN="${GPUS_PER_RUN:-4}"
OBJECTIVES="${OBJECTIVES:-latency_ns,peak_memory,frob_residual}"
SEED="${SEED:-42}"
WANDB="${WANDB:-offline}"
BETA="${BETA:-16}"                          # MOGFN reward exponent (paper: 16-96)

RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-0}"   # GPU actors are num_cpus=0
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-$((2 * NUM_CPU_WORKERS + 16))}"
RAY_PORT=6379

GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "GPU node: $GPU_NODE | CPU node: $CPU_NODE | head ip: $HEAD_IP"

# Ray head on gpu20: 8 GPUs, ZERO logical CPUs (forces measurement to cpu2).
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT --num-gpus=8 --num-cpus=$RAY_CPUS_GPUNODE --block" &
sleep 25

# Ray worker on cpu2: the measurement pool.
srun --het-group=1 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        export JAX_PLATFORMS=cpu && \
        uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
        --num-cpus=$RAY_CPUS_CPUNODE --block" &
sleep 20

echo "==== Ray cluster up. Launching C-MORL + MOGFN (4 GPUs each). $(date) ===="

# ---- Run A: C-MORL with dynamic substeps ----
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && \
        uv run --no-sync $CMORL --name cmorl_g20_dyn_s${SEED} --seed $SEED \
            --example VmappedNeuralNetwork --dataset mnist \
            --objectives $OBJECTIVES \
            --num-prefs 6 --extension-steps 20 --extension-episodes 10 \
            --extension-beta 0.9 --ipo-t 1.0 \
            --episodes $EPISODES --num-envs $NUM_ENVS \
            --num-cpu-workers $NUM_CPU_WORKERS --actor-num-gpus $GPUS_PER_RUN \
            --minibatches 4 --ppo-epochs 4 \
            --cmp-type latency --mem-type peak_memory --advantage-norm scalar \
            --num-data-points 5 --reps-per-point 4 --percentile-keep 0.60 \
            --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 --measure-latency \
            --dynamic-substeps --variant full \
            --cpu-cores-shared --calibrate-steps 0 \
            --ray-address $HEAD_IP:$RAY_PORT --wandb $WANDB" \
    > "$LOG_DIR/cmorl_s${SEED}.out" 2>&1 &
PID_A=$!
echo "[C-MORL] launched (4 GPUs) -> $LOG_DIR/cmorl_s${SEED}.out"
sleep 8

# ---- Run B: MOGFN (gfn_ray --mogfn-pc) with dynamic substeps ----
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && \
        uv run --no-sync $MOGFN --name mogfn_g20_dyn_s${SEED} --seed $SEED \
            --example VmappedNeuralNetwork --dataset mnist \
            --mogfn-pc --preference-channels $OBJECTIVES \
            --reward-normalization zscore --scalarization ws --beta $BETA \
            --episodes $EPISODES --num-envs $NUM_ENVS \
            --num-cpu-workers $NUM_CPU_WORKERS --spmd-gpus $GPUS_PER_RUN \
            --cmp-type latency --measure-latency \
            --dynamic-substeps --variant full --variant-sweep full \
            --cpu-cores-shared --calibrate-steps 0 \
            --ray-address $HEAD_IP:$RAY_PORT --wandb $WANDB" \
    > "$LOG_DIR/mogfn_s${SEED}.out" 2>&1 &
PID_B=$!
echo "[MOGFN] launched (4 GPUs) -> $LOG_DIR/mogfn_s${SEED}.out"

echo "==== both runs launched; waiting. $(date) ===="
wait "$PID_A"; echo "C-MORL exited ($?)"
wait "$PID_B"; echo "MOGFN exited ($?)"

echo "==== all runs done: $(date) ===="
srun --het-group=0 -N1 -n1 bash -lc "uv run --no-sync ray stop" || true
