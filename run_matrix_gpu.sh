#!/bin/bash
# GPU + constrained matrix: 8 variants, each on 1 GPU of an 8-GPU node, with
# GPU Jacobian measurement (--exec-on-gpu) and a Lagrangian CEILING constraint
# cosine_sim <= COSINE_MAX (default 0.9) that FORCES the policy off the exact
# solution into the approximate regime (unconstrained it just picks exact).
# Each variant: trainer (0.5 GPU) + measurement actor (0.5 GPU) co-located on
# one GPU via Ray fractional allocation -> 8 variants = 8 GPUs, all parallel.
# Single node (no cpu1 het-job: measurement is on the GPUs now).
#
#   sbatch run_matrix_gpu.sh                 # cosine<=0.9, 5 seeds
#   COSINE_MAX=0.7 sbatch run_matrix_gpu.sh  # sweep the ceiling

#SBATCH --job-name=mvgpu
#SBATCH --time=12-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15-h100
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=256

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1
export ALPHAGRAD_DBG_TIMING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
LOG_DIR=~/dsnn/logs_matrix_gpu
mkdir -p "$LOG_DIR" slurm

EPISODES="${EPISODES:-500}"
COSINE_MAX="${COSINE_MAX:-0.9}"   # cosine_sim <= COSINE_MAX (accuracy CEILING)
VARIANTS=(diag_gcd diag_factor compress quantize diag_compress diag_quant compress_quant full)
SEEDS=(42 250197 1337 7 100)

RAY_PORT=6379
HEAD_IP=$(hostname -i | awk '{print $1}')
echo "==== matrix-gpu start $(date): node=$(hostname) head=$HEAD_IP cosine_max=$COSINE_MAX ===="

# Ray head on this 8-GPU node. PREALLOCATE=false so the 2 fractional actors
# per GPU don't each grab all HBM.
uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=8 --num-cpus=64 --block &
sleep 25
echo "==== ray up. $(date) ===="

COMMON_ARGS=(
    --example VmappedNeuralNetwork --dataset mnist
    --rewards cmp mem --cmp-type latency --mem-type peak_memory
    --advantage-norm scalar --ppo-epochs 4 --anti-degeneracy none
    # minibatches=32 (not 4): the policy's attention is f32[heads*mb, 4096,
    # 4096]; a big minibatch (minibatches=4 -> mb~283) wants ~76GB and OOMs a
    # shared H100. 32 -> mb~35 -> ~9GB, fits alongside the measurement actor.
    --episodes "$EPISODES" --num-envs 16 --num-cpu-workers 1 --minibatches 32
    --num-data-points 5 --reps-per-point 4 --percentile-keep 0.60
    --slow-exec-cutoff-seconds 0 --flop-gate-threshold 0 --measure-latency
    # GPU measurement: trainer 0.5 GPU + measurement actor 0.5 GPU = 1 GPU/variant.
    --exec-on-gpu --actor-num-gpus 0.5 --cpu-actor-num-gpus 0.5
    # CEILING constraint that forces approximation (dual-ascent enforces it).
    --lagrangian-constraint "cosine_sim<=${COSINE_MAX}"
    --lagrangian-lr 1e-2 --lagrangian-warmup-eps 20
    --running-max-channels peak_memory,max_io_sum
    --calibrate-steps 0 --cpu-callback-timeout 0 --cpu-callback-initial-timeout 0
    --ray-address "$HEAD_IP:$RAY_PORT" --wandb offline
)

run_seed() {
    local sd="$1"; local pids=()
    echo "---- seed $sd: 8 variants (1 GPU each) | $(date) ----"
    for v in "${VARIANTS[@]}"; do
        uv run --no-sync "$PPO" --name "mvgpu_${v}_s${sd}_c${COSINE_MAX}" \
            --variant "$v" --seed "$sd" "${COMMON_ARGS[@]}" \
            > "$LOG_DIR/${v}_s${sd}.out" 2>&1 &
        pids+=($!)
        echo "  [$v s$sd] launched (1 GPU)"
        sleep 5
    done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "---- seed $sd DONE | $(date) ----"
}

for sd in "${SEEDS[@]}"; do run_seed "$sd"; done
echo "==== matrix-gpu done $(date) ===="
uv run --no-sync ray stop || true
