#!/bin/bash
# PRODUCTION MATRIX: all 8 variants, run 5 times, ALWAYS as two sets of 4
# in parallel (variant [0-3, 4-7] * 5). Settled by the 4-vs-8 experiment:
# 8-at-once on one CPU node saturates it (ep1 never finished after 99min,
# exec_loops 89-482s at 3 cores/actor) — 2 sets of 4 is ~4x throughput.
#
# Structure: bring up ONE 2-node Ray cluster (8-GPU node + cpu1), then run
# 10 waves SEQUENTIALLY through it (cluster + compile cache + core
# allocator persist across waves):
#
#   step 0 (seed 42)     : waveA[0-3]  then  waveB[4-7]
#   step 1 (seed 250197) : waveA[0-3]  then  waveB[4-7]
#   ...
#   step 4 (seed 100)    : waveA[0-3]  then  waveB[4-7]
#
# Each wave = 4 variants in parallel, 2 GPUs each (all 8 GPUs used), all 4
# drawing disjoint core slices from cpu1's 384-core measurement pool
# (16 actors/variant x 4 = 64 actors x 6 cores = 384, exact fill -> the
# allocator self-resets each wave by wrapping). Full fidelity: no flop
# gate, no slow-exec cutoff, 5x4 samples, P60.
#
# het-group 0 = 8-GPU node (pgi15-h100, default gpu14); het-group 1 = cpu1.
# Submit (optionally override the GPU node if gpu14 is busy):
#   sbatch run_matrix_2node.sh
#   GPUNODE=pgi15-gpu19 sbatch run_matrix_2node.sh

#SBATCH --job-name=mvstatic
#SBATCH --time=12-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15-h100
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=256
#SBATCH hetjob
#SBATCH --partition=pgi15-cpu
#SBATCH --nodelist=pgi15-cpu1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=384

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_matrix_static
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"

# The 8 variants, in two sets of 4. Set A = [0-3], Set B = [4-7].
SET_A=(diag_gcd diag_factor compress quantize)
SET_B=(diag_compress diag_quant compress_quant full)
# 5 steps -> 5 distinct seeds. Each step runs set A then set B at that seed.
SEEDS=(42 250197 1337 7 100)

# cpu1 is the SOLE measurement node (consistent cores = clean latency
# reward, no node-mix muddiness). The GPU node is declared with ZERO Ray
# CPUs: the 4 PPOActors are num_cpus=0 (GPU-bound, they fit on a 0-CPU
# node via their GPU reservation), so the num_cpus=1 measurement actors
# physically cannot land on the GPU node and ALL go to cpu1. cpu1's cap
# (80) covers 64 measurement actors + the 0.1-CPU compile-cache
# coordinator, while staying well below 384 phys cores (no fork-storm).
RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-0}"
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-80}"
RAY_PORT=6379

# --- resolve node hostnames per het-group ---
GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "GPU node: $GPU_NODE | CPU node: $CPU_NODE | head ip: $HEAD_IP"

# --- start Ray head on the GPU node (8 GPUs declared to Ray) ---
srun --het-group=0 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT --num-gpus=8 --num-cpus=$RAY_CPUS_GPUNODE \
        --block" &
sleep 25

# --- start Ray worker on cpu1 (joins the head; the measurement pool) ---
srun --het-group=1 -N1 -n1 --overlap \
    bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
        uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
        --num-cpus=$RAY_CPUS_CPUNODE --block" &
sleep 20

echo "==== Ray cluster up. Matrix: ${#SEEDS[@]} steps x 2 sets of 4. $(date) ===="

COMMON_ARGS=(
    --example VmappedNeuralNetwork
    --dataset mnist
    --dynamic-substeps
    --rewards cmp mem
    --lambda-cmp 0.0717
    --lambda-mem 0.0535
    --lambda-frob 1.443
    --cmp-type latency
    --mem-type peak_memory
    --advantage-norm scalar
    --ppo-epochs 4
    --episodes "$EPISODES"
    --num-envs 16
    --num-cpu-workers 16
    --minibatches 4
    --num-data-points 5
    --reps-per-point 4
    --percentile-keep 0.60
    # FULL FIDELITY: no flop-gate, no slow-exec-cutoff. Every order really
    # executed and timed, full 5x4 samples, true single-thread CPU latency.
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
    --ray-address "$HEAD_IP:$RAY_PORT"
    --wandb online
)

# run_wave <step-tag> <variant...>: launch the given variants (one seed,
# 2 GPUs each) in parallel, wait for all to finish.
run_wave() {
    local tag="$1"; shift
    local sd="$1"; shift
    local set_name="$1"; shift
    local variants=("$@")
    local wpids=()
    echo "---- wave ${tag} set${set_name} seed${sd}: ${variants[*]} | $(date) ----"
    for v in "${variants[@]}"; do
        srun --het-group=0 -N1 -n1 --overlap \
            bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
                export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ALPHAGRAD_DBG_TIMING=1 && \
                uv run --no-sync $PPO --name mvstatic_${v}_s${sd} \
                --variant $v --seed $sd ${COMMON_ARGS[*]}" \
            > "$LOG_DIR/${v}_s${sd}.out" 2>&1 &
        wpids+=($!)
        echo "  [$v seed$sd] launched (2 GPUs) -> $LOG_DIR/${v}_s${sd}.out"
        sleep 5
    done
    for p in "${wpids[@]}"; do wait "$p"; done
    echo "---- wave ${tag} set${set_name} seed${sd} DONE | $(date) ----"
}

# 5 steps; each step = set A (4-parallel) then set B (4-parallel).
for i in "${!SEEDS[@]}"; do
    sd="${SEEDS[$i]}"
    run_wave "$i" "$sd" A "${SET_A[@]}"
    run_wave "$i" "$sd" B "${SET_B[@]}"
done

echo "==== full 8x5 matrix done: $(date) ===="
srun --het-group=0 -N1 -n1 bash -lc "uv run --no-sync ray stop" || true
