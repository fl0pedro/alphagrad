#!/bin/bash
#SBATCH --job-name=aojsub1
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/Users/assmuth/dsnn/alphagrad/aojsub1_%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONUNBUFFERED=1
# THE point of this run: measured executable built by the IncrementalJaxpr.
# Measured executable via jacve: graphax's AOJ value+Jacobian is still in dev,
# and the two are equivalent for a given elimination+approximation sequence.
# The AOJ stays the intended source for the policy's incremental state.
export ALPHAGRAD_MEASURE_VIA_AOJ=0
export ALPHAGRAD_POLICY=palimpsa
# Per-face approximation through jacve callables (see approx/README.md).
export ALPHAGRAD_PER_FACE_MVP=1
# The env measures orders that have not eliminated everything yet (at reset the
# order is empty). graphax's strict full-order guard rejects that; we are not
# consuming intermediate Jacobians, so accept the partial order.
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
# Let the policy see (and learn to drop/approximate) the full graph.
export GRAPHAX_PRUNE=0
cd ~/dsnn/alphagrad
echo "=== node=$(hostname) $(date) ==="
echo "=== AOJ=$ALPHAGRAD_MEASURE_VIA_AOJ graphax=$(git -C ../graphax log --oneline -1) ==="
echo "=== alphagrad=$(git log --oneline -1) ==="
uv run --no-sync python src/alphagrad/approx/ppo_ray.py \
  --name nn256_aoj_sub1_s7 --variant full --seed 7 \
  --example VmappedNeuralNetwork --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type xla_peak_memory \
  --exec-on-gpu --measure-latency --latency-inner-reps 50 \
  --entropy-coef 0.05 --entropy-coef-final 0.05 \
  --dynamic-substeps --max-substeps 1 \
  --actor-num-gpus 1 --cpu-actor-num-gpus 1 --num-cpu-workers 3 \
  --cpu-cores-per-actor 8 --cpu-cores-shared \
  --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
  --cpu-worker-recycle-every 6 \
  --advantage-norm scalar --ppo-epochs 4 --lr 1e-4 \
  --lr-decay-min-mult 1.0 \
  --episodes 3400 --num-envs 4 --minibatches 4 \
  --num-data-points 5 --reps-per-point 2 \
  --checkpoint-path campaign_aoj_sub1/nn256/ckpt --checkpoint-every 100 \
  --best-sequences-json campaign_aoj_sub1/nn256/best.json --best-sequences-every 1 \
  --calibrate-steps 0 \
  --wandb online --wandb-project dsnn-aoj-perpath
echo "=== RC=$? $(date) ==="
