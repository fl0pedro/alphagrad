#!/bin/bash
# Search-space sampler with GPU Jacobian measurement (--exec-on-gpu), on a
# 4-GPU node. 4 workers, 1 GPU each. Same reproducible/resumable/shuffled
# design as the CPU sampler; only the exec device changes (so latency now
# reflects GPU). Node/GPU tracked per worker via the actor ready() log.
#
#   sbatch run_sampler_gpu.sh

#SBATCH --job-name=ssgpu
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15
#SBATCH --nodelist=pgi15-gpu15
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
# GPU: let JAX see CUDA (override the sampler's default cpu setdefault).
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0   # else Ray uploads ~/dsnn as working_dir

mkdir -p slurm
SAMPLER=alphagrad/src/alphagrad/approx/sampler_full.py

NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SEED="${SEED:-12345}"
NUM_WORKERS="${NUM_WORKERS:-4}"          # 4 GPUs on the node, 1 per worker
OUT_DIR="${OUT_DIR:-$HOME/dsnn/search_space_full_gpu}"

echo "==== sampler-GPU start $(date): node=$(hostname) N=$NUM_SAMPLES seed=$SEED workers=$NUM_WORKERS ===="
uv run --no-sync "$SAMPLER" \
    --num-samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    --out-dir "$OUT_DIR" \
    --example VmappedNeuralNetwork \
    --dataset mnist \
    --num-data-points 8 \
    --num-passes 10 \
    --num-eval-samples 8 \
    --max-exec-seconds 0 \
    --exec-on-gpu \
    --actor-num-gpus 1
echo "==== sampler-GPU done $(date) ===="
