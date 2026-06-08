#!/bin/bash
# Two-node pooled run: 8-GPU Blackwell node + cpu1 (384c, no GPU), joined
# into ONE Ray cluster. 4 variants run in PARALLEL, each on 2 GPUs
# (minibatches=4 fits 2 Blackwell GPUs), all 4 sharing the combined
# 128+384 = 512-core measurement pool (disjoint slices via the per-node
# core allocator, --cpu-cores-shared). Goal: see how 4 variants run
# together when CPUs and GPUs are pooled across both nodes.
#
# Heterogeneous SLURM job:
#   het-group 0 = the 8-GPU node (partition pgi15, 8 GPU, 128 CPU)
#   het-group 1 = cpu1          (partition pgi15-cpu, 384 CPU)
#
# Submit with the GPU node chosen at submit time (whichever 8-GPU node is
# idle), e.g.:
#   GPUNODE=pgi15-gpu19 sbatch run_2node_4parallel.sh

#SBATCH --job-name=mv2n4p
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15-h100
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=256
#SBATCH hetjob
#SBATCH --partition=pgi15-cpu
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-cpu1
#SBATCH --cpus-per-task=384

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_2node_4parallel
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
# 4 variants to run in parallel this batch (first 4 of the 8 difficult set).
VARIANTS=(diag_factor compress quantize full)
SEEDS=(42 42 42 42)

# Ray logical-CPU caps (prestart-worker control — physical cores are
# claimed via affinity). These ALSO force the CPU-actor split across
# both nodes: Ray won't place more num_cpus=1 actors on a node than its
# logical count, so with 4×16=64 measurement actors + 4 PPOActors we cap
# each node below 64 to FORCE spill onto both (SPREAD alone is a soft
# hint Ray ignored — it packed all 64 onto cpu1). cpu1 (384 phys cores)
# gets the larger logical share than gpu14 (256 phys). Kept well below
# physical counts to avoid the prestart fork-storm.
RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-36}"
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-44}"
RAY_PORT=6379

# --- resolve node hostnames per het-group ---
GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "GPU node: $GPU_NODE | CPU node: $CPU_NODE | head ip: $HEAD_IP"

# --- start Ray head on the GPU node ---
# IMPORTANT: --overlap on every step, and NO per-step --gpus. A step that
# requests --gpus=8 holds all GPUs exclusively and blocks every later
# step ("Requested nodes are busy"). Instead, declare the 8 GPUs to Ray
# (--num-gpus=8) and let RAY assign 2 to each variant's PPOActor.
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT --num-gpus=8 --num-cpus=$RAY_CPUS_GPUNODE \
        --block" &
sleep 25

# --- start Ray worker on cpu1 (joins the head) ---
srun --het-group=1 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
        --num-cpus=$RAY_CPUS_CPUNODE --block" &
sleep 20

echo "==== Ray cluster up. Launching ${#VARIANTS[@]} parallel variants. $(date) ===="

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --rewards cmp mem
    --cmp-type latency
    --mem-type peak_memory
    --advantage-norm scalar
    --ppo-epochs 4
    --anti-degeneracy none
    --cosine-lower-bound 0.0
    --cosine-upper-bound 1.0
    --episodes "$EPISODES"
    --num-envs 16
    --num-cpu-workers 16
    --minibatches 4
    --num-data-points 5
    --reps-per-point 4
    --percentile-keep 0.60
    # FULL FIDELITY: no flop-gate, no slow-exec-cutoff — every order is
    # really executed and timed, full 5×4 samples. True CPU-latency
    # signal (no FLOP surrogates, no sample capping). Slower, but the
    # pooled 640 cores + 4-parallel keep it tractable.
    --slow-exec-cutoff-seconds 0
    --flop-gate-threshold 0
    --measure-latency
    --running-max-channels peak_memory,max_io_sum
    --calibrate-steps 0
    --cpu-callback-timeout 0
    --cpu-callback-initial-timeout 0
    --reserved-driver-cores 0
    --actor-num-gpus 2
    --cpu-cores-shared
    --spread-cpu-actors
    --ray-address "$HEAD_IP:$RAY_PORT"
    --wandb offline
)

# Launch the 4 variants in parallel on the head node (het-group 0). Each
# joins the shared cluster, grabs 2 GPUs, and draws CPU actors from the
# pooled 512-core measurement pool (disjoint slices via the allocator).
pids=()
for i in "${!VARIANTS[@]}"; do
    v="${VARIANTS[$i]}"; sd="${SEEDS[$i]}"
    srun --het-group=0 -N1 -n1 --overlap \
        bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
            export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ALPHAGRAD_DBG_TIMING=1 && \
            uv run --no-sync $PPO --name mv2n4p_${v}_s${sd} \
            --variant $v --seed $sd ${COMMON_ARGS[*]}" \
        > "$LOG_DIR/${v}_s${sd}.out" 2>&1 &
    pids+=($!)
    echo "[$v seed$sd] launched (2 GPUs) -> $LOG_DIR/${v}_s${sd}.out"
    sleep 5
done

echo "==== all ${#VARIANTS[@]} variants running. waiting. $(date) ===="
for p in "${pids[@]}"; do wait "$p"; done

echo "==== all variants done: $(date) ===="
srun --het-group=0 -N1 -n1 bash -lc "uv run --no-sync ray stop" || true
