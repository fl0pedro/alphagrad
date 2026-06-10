#!/bin/bash
# C-MORL (cmorl_ray.py) on a heterogeneous 2-node Ray cluster:
#   het-group 0 = a GPU node  -> the policy/trainer actor (num_gpus=N, num_cpus=0)
#   het-group 1 = cpu2        -> the measurement actors (num_cpus=1, JAX_PLATFORMS=cpu)
#
# The model trains on the GPU node; ALL latency/cost measurement runs on the
# single CPU node. This is enforced by:
#   * the GPU trainer actor reserving num_cpus=0 (set in cmorl_ray.py), so it
#     cannot consume the CPU node's slots, and
#   * a deliberately small Ray logical-CPU count on the GPU node
#     (RAY_CPUS_GPUNODE) so the num_cpus=1 measurement actors cannot fit there
#     and spill entirely onto the CPU node. (No --spread-cpu-actors: we want
#     them ON ONE CPU NODE, not split.)
#
# Latency protocol (C-MORL default objectives latency_ns,peak_memory,frob_residual):
#   VmappedNeuralNetwork (hidden via env), MNIST, 5 data points x 4 reps,
#   reduced at the 60th-slowest percentile (--num-data-points 5 --reps-per-point 4
#   --percentile-keep 0.60 --measure-latency).
#
# Submit:
#   sbatch run_cmorl_2node.sh
#   EPISODES=2000 EXTENSION_STEPS=40 sbatch run_cmorl_2node.sh
#
#SBATCH --job-name=cmorl2n
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15-single-gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH hetjob
#SBATCH --partition=pgi15-cpu
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-cpu2
#SBATCH --cpus-per-task=384

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

CMORL=alphagrad/src/alphagrad/approx/cmorl_ray.py
LOG_DIR=~/dsnn/logs_cmorl_2node
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-16}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
OBJECTIVES="${OBJECTIVES:-latency_ns,peak_memory,frob_residual}"
NUM_PREFS="${NUM_PREFS:-8}"
EXTENSION_STEPS="${EXTENSION_STEPS:-30}"
EXTENSION_EPISODES="${EXTENSION_EPISODES:-10}"
WANDB="${WANDB:-offline}"
SEED="${SEED:-42}"

# Keep the GPU node's Ray logical-CPU count tiny so the num_cpus=1
# measurement actors cannot be packed there and spill onto cpu2. The GPU
# trainer actor uses num_cpus=0 so it still fits. cpu2 gets a generous
# logical share (still below its 384 physical cores to avoid the
# prestart fork-storm).
RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-6}"
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-$((NUM_CPU_WORKERS + 8))}"
RAY_PORT=6379

GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "GPU node: $GPU_NODE | CPU node: $CPU_NODE | head ip: $HEAD_IP"

# Ray head on the GPU node (declares the GPUs; tiny logical-CPU count).
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT --num-gpus=$ACTOR_GPUS --num-cpus=$RAY_CPUS_GPUNODE \
        --block" &
sleep 25

# Ray worker on cpu2 (the measurement node).
srun --het-group=1 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        export JAX_PLATFORMS=cpu && \
        uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
        --num-cpus=$RAY_CPUS_CPUNODE --block" &
sleep 20

echo "==== Ray cluster up. Launching C-MORL. $(date) ===="

# Driver runs on the GPU node (het-group 0) and joins the cluster via
# --ray-address. The GPU trainer actor lands on the GPU node; the CPU
# measurement actors land on cpu2 (see RAY_CPUS_GPUNODE rationale above).
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && \
        uv run --no-sync $CMORL \
            --name cmorl2n_s${SEED} \
            --example VmappedNeuralNetwork --dataset mnist \
            --objectives $OBJECTIVES \
            --num-prefs $NUM_PREFS \
            --extension-steps $EXTENSION_STEPS \
            --extension-episodes $EXTENSION_EPISODES \
            --extension-beta 0.9 --ipo-t 1.0 \
            --episodes $EPISODES --num-envs $NUM_ENVS \
            --num-cpu-workers $NUM_CPU_WORKERS --actor-num-gpus $ACTOR_GPUS \
            --minibatches 4 --ppo-epochs 4 \
            --cmp-type latency --mem-type peak_memory \
            --advantage-norm scalar \
            --num-data-points 5 --reps-per-point 4 --percentile-keep 0.60 \
            --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 \
            --measure-latency \
            --calibrate-steps 0 \
            --ray-address $HEAD_IP:$RAY_PORT \
            --seed $SEED --wandb $WANDB" \
    2>&1 | tee "$LOG_DIR/cmorl2n_s${SEED}.out"
EXIT_CODE=${PIPESTATUS[0]}

echo "==== C-MORL done (exit=$EXIT_CODE): $(date) ===="
srun --het-group=0 -N1 -n1 bash -lc "uv run --no-sync ray stop" || true
exit "$EXIT_CODE"
