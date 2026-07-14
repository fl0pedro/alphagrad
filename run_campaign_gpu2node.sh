#!/bin/bash
# Quality-signals seed-vertex campaign on TWO GPU nodes (no CPU measure node).
#   het0 = a 4-GPU blackwell node, het1 = an 8-GPU blackwell node => 12 GPUs.
#   Per run: 2 GPUs for the (SPMD data-parallel) PPO trainer + 1 GPU for the
#   grad/measurement model => 3 GPUs/run, 4 concurrent (=12 GPUs). Ray schedules
#   the 4 PPOActors (num_gpus=2) + 4 measure actors (num_gpus=1) across both nodes.
#
#   ALL measurement runs ON GPU (--exec-on-gpu): peak_memory = ResourceMonitor
#   high-water mark (the WATERLINE, exact on GPU), latency + cosine on GPU too.
#   Pure on-policy PPO (no replay/V-trace). Keep --seed-vertices (tangent+adjoint
#   seeds as eliminable vertices, eval at t=0). Per-MODEL reward lambdas.
#   No-barrier 4-slot queue (a fast run frees its 3 GPUs for the next immediately).
#
#SBATCH --job-name=qsig_gpu2n
#SBATCH --time=4-00:00:00
#SBATCH --output=/Users/assmuth/dsnn/campaign/logs/gpu2n_%j.out
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --cpus-per-task=32
#SBATCH hetjob
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:8
#SBATCH --cpus-per-task=32

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
CAMP=~/dsnn/campaign_g2
mkdir -p "$CAMP/logs"

WAVES="${WAVES:-8}"
EPISODES="${EPISODES:-500}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-qsig-gpu2node}"
MAXJOBS="${MAXJOBS:-4}"                 # 4 concurrent runs x 3 GPUs = 12
PPO_GPUS="${PPO_GPUS:-2}"               # GPUs for the SPMD PPO trainer per run
GRAD_GPUS="${GRAD_GPUS:-1}"             # GPUs for the grad/measure model per run
RAY_PORT="${RAY_PORT:-6379}"
RAY_CPUS_PER_NODE="${RAY_CPUS_PER_NODE:-24}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"         # cosine already [0,1]
LAMBDA_FROB="${LAMBDA_FROB:-0.0}"

MODELS=(VmappedNeuralNetwork VmappedConvNet VmappedMoE VmappedViT)
ENVS=(4 4 4 4)                          # multiple of PPO_GPUS=2 for SPMD shard
MBS=(8 8 8 16)
# Per-model lambdas = 1/(that model's measured |symlog| scale); GPU waterline
# peak-mem scale ~ matches latency (re-tune after first runs if needed).
LAMBDA_CMP_ARR=(0.060 0.046 0.050 0.046)
LAMBDA_MEM_ARR=(0.057 0.045 0.049 0.047)

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
NODE1=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 -N1 -n1 hostname -i | awk '{print $1}')
echo "########## GPU2NODE $(date) | head=$NODE0($HEAD_IP, 4gpu) worker=$NODE1(8gpu) waves=$WAVES eps=$EPISODES ##########"

# Ray head on the 4-GPU node (batch script runs here); declares its 4 GPUs.
( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=4 --num-cpus=$RAY_CPUS_PER_NODE --block ) &
disown
sleep 25
# Ray worker on the 8-GPU node; declares its 8 GPUs.
srun --het-group=1 -N1 -n1 bash -lc \
  "$ULIM unset CUDA_VISIBLE_DEVICES; cd ~/dsnn && export PATH=\$HOME/.local/bin:\$PATH && \
   uv run --no-sync ray start --address=$HEAD_IP:$RAY_PORT \
   --num-cpus=$RAY_CPUS_PER_NODE --block" &
disown
sleep 20
echo "==== Ray cluster up (12 GPUs across 2 nodes) $(date) ===="

launch_one() {
    local WAVE=$1 SLOT=$2
    local MIDX=$(( (WAVE + SLOT) % 4 ))
    local M=${MODELS[$MIDX]} NE=${ENVS[$MIDX]} MB=${MBS[$MIDX]}
    local LCMP=${LAMBDA_CMP_ARR[$MIDX]} LMEM=${LAMBDA_MEM_ARR[$MIDX]}
    local SEED=$(( 1000 + MIDX*100 + WAVE ))
    local OUT=$CAMP/${M}/run${WAVE}
    local LOG=$CAMP/logs/${M}_w${WAVE}.log
    if [ -f "$OUT/.done" ]; then echo "[w$WAVE s$SLOT] SKIP $M (done)"; return 0; fi
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "[w$WAVE s$SLOT] START $M envs=$NE mb=$MB seed=$SEED $(date +%H:%M:%S)"
    # Driver is a plain process on the head node; JAX_PLATFORMS=cpu so the 4
    # concurrent drivers don't grab GPU memory (the PPOActor + measure actor,
    # Ray-scheduled, do all GPU work). PPOActor: PPO_GPUS (SPMD). Measure actor:
    # --exec-on-gpu + GRAD_GPUS GPU (peak_memory waterline, latency, cosine on GPU).
    JAX_PLATFORMS=cpu uv run --no-sync $PPO --name camp_${M}_w${WAVE} --variant full --seed $SEED \
      --example $M --dataset mnist \
      --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
      --lambda-cmp $LCMP --lambda-mem $LMEM --lambda-acc $LAMBDA_ACC --lambda-frob $LAMBDA_FROB \
      --measure-grad --measure-latency --quant-once --seed-vertices \
      --exec-on-gpu --use-placement-group --actor-num-gpus $PPO_GPUS --cpu-actor-num-gpus $GRAD_GPUS --num-cpu-workers 1 \
      --latency-timer perf_counter --latency-inner-reps 5 --latency-warmup 2 --latency-winsor 0.2 \
      --ray-address $HEAD_IP:$RAY_PORT \
      --advantage-norm scalar --ppo-epochs 4 \
      --episodes $EPISODES --num-envs $NE --minibatches $MB \
      --num-data-points 5 --reps-per-point 2 \
      --best-sequences-json $OUT/best.json --best-sequences-every 5 \
      --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
    local RC=$?
    echo "EXIT=$RC model=$M wave=$WAVE seed=$SEED" >> "$LOG"
    if [ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ]; then touch "$OUT/.done"; fi
    echo "[w$WAVE s$SLOT] END   $M (rc=$RC) $(date +%H:%M:%S)"
}

echo "===== QUEUE START: $WAVES waves x4 = $((WAVES*4)) runs, $MAXJOBS concurrent :: $(date) ====="
for ((W=0; W<WAVES; W++)); do
  for SLOT in 0 1 2 3; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    launch_one "$W" "$SLOT" &
  done
done
wait

echo "########## GPU2NODE DONE $(date) ##########"
for M in "${MODELS[@]}"; do for ((w=0; w<WAVES; w++)); do
  f=$CAMP/${M}/run${w}/best.json
  [ -s "$f" ] && echo "OK  $M run$w" || echo "MISS $M run$w"
done; done
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
