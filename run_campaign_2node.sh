#!/bin/bash
# Quality-signals CVaR-gate full campaign on a heterogeneous 2-node Ray cluster.
#   het-group 0 = pgi15-h100 (gpu14, 4xH100)  -> the 4 GPU trainer actors
#   het-group 1 = pgi15-cpu1  (384 cores)      -> ALL latency/memory measurement
#
# Every run trains on a GPU; ALL measurement runs single-core on the dedicated
# CPU node (clean latency + deterministic XLA memory). Enforced by:
#   * GPU trainer actor reserves num_cpus=0 (PPOActor, --actor-num-gpus 1), and
#   * the GPU node's Ray logical-CPU count is tiny (RAY_CPUS_GPUNODE), so the
#     num_cpus=1 measure actors cannot fit there and spill entirely onto cpu1.
#   * --cpu-cores-shared: the 4 concurrent runs claim DISJOINT 1-core slices on
#     cpu1 from a per-node allocator (a cluster named actor) -> no contention,
#     clean per-reading latency. 4 x NUM_CPU_WORKERS must be <= cpu1 cores (384).
#
# Staggered waves of 4 (rotation [0123]->[1230]->[2301]->[3012], x2 = 8 waves):
#   models 0=VmappedNeuralNetwork 1=VmappedConvNet 2=VmappedMoE 3=VmappedViT.
#   Every concurrent group-of-4 is fully diverse; each model runs 8 times.
#   Ray assigns each wave's 4 trainers a distinct GPU; a wave barrier frees the
#   GPUs (and dead actors' core slices) before the next wave.
#
# Signal stack: off-policy V-trace (--replay-buffer-size, reuses the expensive
#   CPU measurements across updates), single-core measure (--cpu-cores-per-actor
#   1, threads off), deterministic XLA peak-memory reward (--mem-type
#   xla_peak_memory; RM peak still measured at idx 5), aligned cosine quality,
#   perf_counter winsorized latency, --quant-once, --measure-grad, full variant.
#
# Smoke vs full via env overrides, e.g. a 1-wave 2-episode probe:
#   WAVES=1 EPISODES=2 MAX_WALL=0 NUM_CPU_WORKERS=60 sbatch run_campaign_2node.sh
#
#SBATCH --job-name=qsig_camp2n
#SBATCH --time=16:00:00
#SBATCH --output=/Users/assmuth/dsnn/campaign/logs/camp2n_%j.out
#SBATCH --partition=pgi15-h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH hetjob
#SBATCH --partition=pgi15-cpu
#SBATCH --nodelist=pgi15-cpu1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=384

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
CAMP=~/dsnn/campaign
mkdir -p "$CAMP/logs"

# ---- tunables (env-overridable for smoke) --------------------------------
WAVES="${WAVES:-8}"                       # 8 = full staggered campaign
EPISODES="${EPISODES:-500}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-qsig-cvar-seed}"
MAX_WALL="${MAX_WALL:-5400}"              # per-run wall cap (s); 0 = run to EPISODES
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-24}"  # per run; 4*this <= 384
REPLAY_CAP="${REPLAY_CAP:-128}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_CPUS_GPUNODE="${RAY_CPUS_GPUNODE:-0}"          # tiny -> measure spills to cpu1
RAY_CPUS_CPUNODE="${RAY_CPUS_CPUNODE:-$((4 * NUM_CPU_WORKERS + 16))}"

MODELS=(VmappedNeuralNetwork VmappedConvNet VmappedMoE VmappedViT)
# --seed-vertices ~doubles the graph (NN 15->36 eqns etc.), enlarging the policy's
# relational attention; envs=4 + higher minibatches keeps jit_update_step in 80GB.
ENVS=(4 4 4 4)
MBS=(8 8 8 16)

# Raise fd / process limits: the measure pool puts 4*NUM_CPU_WORKERS actors on
# cpu1; each holds plasma-store fds. The default soft limit (1024) makes the
# raylet crash with "Failed to receive the fd" (EMFILE). Lift to the hard max.
ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'

GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "########## CAMPAIGN2N $(date) | GPU=$GPU_NODE CPU=$CPU_NODE head=$HEAD_IP waves=$WAVES eps=$EPISODES wall=$MAX_WALL workers=$NUM_CPU_WORKERS ##########"

# Ray head on the GPU node (declares 4 GPUs; tiny logical-CPU count).
srun --het-group=0 -N1 -n1 --overlap bash -lc \
  "$ULIM cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
   uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
   --port=$RAY_PORT --num-gpus=4 --num-cpus=$RAY_CPUS_GPUNODE --block" &
sleep 25

# Ray worker on cpu1 (the measurement node; JAX pinned to CPU).
srun --het-group=1 -N1 -n1 --overlap bash -lc \
  "$ULIM cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && export JAX_PLATFORMS=cpu && \
   uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
   --num-cpus=$RAY_CPUS_CPUNODE --block" &
sleep 20
echo "==== Ray cluster up $(date) ===="

# The wave orchestrator runs on the GPU node and joins the cluster as the driver
# for each of the 4 concurrent runs (each is its own ppo_ray process; Ray hands
# each PPOActor a distinct GPU, measure actors land on cpu1 via the allocator).
run_wave() {
  local WAVE=$1
  local pids=()
  for SLOT in 0 1 2 3; do
    local MIDX=$(( (WAVE + SLOT) % 4 ))
    local M=${MODELS[$MIDX]} NE=${ENVS[$MIDX]} MB=${MBS[$MIDX]}
    local SEED=$(( 1000 + MIDX*100 + WAVE ))
    local OUT=$CAMP/${M}/run${WAVE}
    local LOG=$CAMP/logs/${M}_w${WAVE}.log
    if [ -f "$OUT/.done" ]; then echo "[w$WAVE s$SLOT] SKIP $M (done)"; continue; fi
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "[w$WAVE s$SLOT] START $M envs=$NE mb=$MB seed=$SEED"
    srun --het-group=0 -N1 -n1 --overlap bash -lc \
      "$ULIM cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH \
         XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 && \
       uv run --no-sync $PPO --name camp_${M}_w${WAVE} --variant full --seed $SEED \
         --example $M --dataset mnist \
         --rewards cmp mem acc --cmp-type latency --mem-type xla_peak_memory \
         --measure-grad --measure-latency --quant-once --seed-vertices \
         --latency-timer perf_counter --latency-inner-reps 5 --latency-warmup 2 --latency-winsor 0.2 \
         --cpu-cores-per-actor 1 --cpu-cores-shared --num-cpu-workers $NUM_CPU_WORKERS \
         --actor-num-gpus 1 --ray-address $HEAD_IP:$RAY_PORT \
         --replay-buffer-size $REPLAY_CAP --vtrace-rho-bar 1.0 --vtrace-c-bar 1.0 \
         --advantage-norm scalar --ppo-epochs 4 --anti-degeneracy none \
         --cosine-lower-bound 0.0 --cosine-upper-bound 1.0 \
         --episodes $EPISODES --num-envs $NE --minibatches $MB \
         --max-wall-seconds $MAX_WALL \
         --num-data-points 5 --reps-per-point 2 \
         --best-sequences-json $OUT/best.json --best-sequences-every 5 \
         --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > $LOG 2>&1
       rc=\$?; echo \"EXIT=\$rc model=$M wave=$WAVE seed=$SEED\" >> $LOG
       if [ \$rc -eq 0 ] && [ -s $OUT/best.json ]; then touch $OUT/.done; fi
       exit \$rc" &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  echo "===== WAVE $WAVE COMPLETE ($fail failed) :: $(date) ====="
}

for ((W=0; W<WAVES; W++)); do
  W0=$(( W % 4 )); W1=$(( (W+1) % 4 )); W2=$(( (W+2) % 4 )); W3=$(( (W+3) % 4 ))
  echo "===== WAVE $W :: ${MODELS[$W0]} ${MODELS[$W1]} ${MODELS[$W2]} ${MODELS[$W3]} :: $(date) ====="
  run_wave "$W"
done

echo "########## CAMPAIGN2N DONE $(date) ##########"
for M in "${MODELS[@]}"; do for ((w=0; w<WAVES; w++)); do
  f=$CAMP/${M}/run${w}/best.json
  [ -s "$f" ] && echo "OK  $M run$w" || echo "MISS $M run$w"
done; done

srun --het-group=0 -N1 -n1 --overlap bash -lc "cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && uv run --no-sync ray stop" 2>/dev/null || true
