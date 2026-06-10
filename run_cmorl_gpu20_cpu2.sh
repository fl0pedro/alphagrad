#!/bin/bash
# Two C-MORL runs sharing one heterogeneous Ray cluster:
#   het-group 0 = pgi15-gpu20 (8 GPUs)  -> two policy/trainer actors, 4 GPUs each
#   het-group 1 = pgi15-cpu2  (384 CPU) -> ALL measurement actors
#
# The gpu20 Ray head is started with --num-cpus=0, so the num_cpus=1
# measurement actors physically cannot be placed there and ALL latency/cost
# measurement lands on cpu2. The GPU trainer actors use num_cpus=0 / num_gpus=4
# so the 8 GPUs split into two 4-GPU runs. Both runs share cpu2's measurement
# pool via the per-node core allocator (--cpu-cores-shared).
#
# Submit:  sbatch run_cmorl_gpu20_cpu2.sh
# Override: EPISODES=300 EXTENSION_STEPS=10 sbatch run_cmorl_gpu20_cpu2.sh
#
#SBATCH --job-name=cmorl2x4
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
#SBATCH --cpus-per-task=384

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

CMORL=alphagrad/src/alphagrad/approx/cmorl_ray.py
LOG_DIR=~/dsnn/logs_cmorl_gpu20_cpu2
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-16}"     # per run; 2 runs => 32 actors on cpu2
ACTOR_GPUS="${ACTOR_GPUS:-4}"
OBJECTIVES="${OBJECTIVES:-latency_ns,peak_memory,frob_residual}"
NUM_PREFS="${NUM_PREFS:-6}"
EXTENSION_STEPS="${EXTENSION_STEPS:-20}"
EXTENSION_EPISODES="${EXTENSION_EPISODES:-10}"
WANDB="${WANDB:-offline}"
SEEDS=(${SEEDS:-42 43})                       # two parallel runs (replicas by seed)

# gpu20 contributes GPUs only (0 logical CPUs) so every num_cpus=1 measurement
# actor is forced onto cpu2. cpu2 gets a logical-CPU count above the total
# measurement-actor count (2*NUM_CPU_WORKERS) but well below 384 physical
# (avoids Ray's prestart fork-storm; real cores are claimed via affinity).
RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-0}"
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-$((2 * NUM_CPU_WORKERS + 16))}"
RAY_PORT=6379

GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "GPU node: $GPU_NODE | CPU node: $CPU_NODE | head ip: $HEAD_IP"

# Ray head on gpu20: all 8 GPUs, ZERO logical CPUs (forces measurement to cpu2).
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

echo "==== Ray cluster up. Launching ${#SEEDS[@]} C-MORL runs (4 GPUs each). $(date) ===="

COMMON_ARGS=(
    --example VmappedNeuralNetwork --dataset mnist
    --objectives "$OBJECTIVES"
    --num-prefs "$NUM_PREFS"
    --extension-steps "$EXTENSION_STEPS" --extension-episodes "$EXTENSION_EPISODES"
    --extension-beta 0.9 --ipo-t 1.0
    --episodes "$EPISODES" --num-envs "$NUM_ENVS"
    --num-cpu-workers "$NUM_CPU_WORKERS" --actor-num-gpus "$ACTOR_GPUS"
    --minibatches 4 --ppo-epochs 4
    --cmp-type latency --mem-type peak_memory --advantage-norm scalar
    --num-data-points 5 --reps-per-point 4 --percentile-keep 0.60
    --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 --measure-latency
    --num-eval-samples 5 --calibrate-steps 0
    --cpu-cores-shared
    --ray-address "$HEAD_IP:$RAY_PORT"
    --wandb "$WANDB"
)

pids=()
for sd in "${SEEDS[@]}"; do
    srun --het-group=0 -N1 -n1 --overlap \
        bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
            export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && \
            uv run --no-sync $CMORL --name cmorl_g20_s${sd} --seed $sd ${COMMON_ARGS[*]}" \
        > "$LOG_DIR/cmorl_s${sd}.out" 2>&1 &
    pids+=($!)
    echo "[seed $sd] launched (4 GPUs) -> $LOG_DIR/cmorl_s${sd}.out"
    sleep 8
done

echo "==== both runs launched; waiting. $(date) ===="
for p in "${pids[@]}"; do wait "$p"; done

echo "==== all runs done: $(date) ===="
srun --het-group=0 -N1 -n1 bash -lc "uv run --no-sync ray stop" || true
