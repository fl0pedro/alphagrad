#!/bin/bash
# Train the VmappedNN on MNIST with every Pareto-front rule (approx gradient),
# 20 seeds each + reverse-mode baseline. Runs on CPU with SINGLE-THREADED kernels
# (matching how the front latencies were measured) — many processes in parallel,
# each handling a shard of the rules. Run each model on its OWN node's CPUs:
#   sbatch --nodelist=pgi15-gpu16 run_train_experiment.sh cmorl
#   sbatch --nodelist=pgi15-gpu17 run_train_experiment.sh mogfn
#   SMOKE=1 NPROC=1 NSHARDS=56 sbatch --nodelist=pgi15-gpu16 run_train_experiment.sh cmorl
#SBATCH --job-name=trainexp
#SBATCH --time=12:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
# CPU, single-threaded kernels — same condition the front latencies were measured
# under (clean single-core signal; compile still uses all cores).
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NPROC_OMP_NUM_THREADS=1

SOURCE="${1:?usage: run_train_experiment.sh cmorl|mogfn}"
FRONT="${FRONT:-$HOME/dsnn/morl_fronts/${SOURCE}_front.json}"
OUT="${OUT:-$HOME/dsnn/train_exp}"
NPROC="${NPROC:-24}"          # parallel worker processes (each single-threaded)
NS="${NSHARDS:-$NPROC}"
SMOKE="${SMOKE:-0}"
EXTRA=""; [ "$SMOKE" = "1" ] && EXTRA="--smoke"
mkdir -p "$OUT" slurm
S=alphagrad/src/alphagrad/approx/train_rules_experiment.py

echo "[trainexp] $SOURCE front=$FRONT nproc=$NPROC nshards=$NS smoke=$SMOKE CPU single-thread $(date)"
for k in $(seq 0 $((NPROC-1))); do
  uv run --no-sync "$S" --source "$SOURCE" \
    --front "$FRONT" --shard "$k" --nshards "$NS" \
    --out "$OUT/${SOURCE}_shard${k}.json" $EXTRA \
    > "$OUT/${SOURCE}_shard${k}.log" 2>&1 &
done
wait
echo "[trainexp] $SOURCE done $(date)"
