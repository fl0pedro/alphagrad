#!/bin/bash
#SBATCH --job-name=nn256_ablA_curric
#SBATCH --time=7-00:00:00
#SBATCH --output=/Users/assmuth/dsnn/nn256_ablA_curric_%j.out
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu16
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --cpus-per-task=128

# ============================================================================
# GRAD-MODE POSTER CHILD, PopArt-OFF variant (bridge-cse, USER-DIRECTED)
# Reference working grad-mode run = wandb nskom9jd (full_nn256_v2_popart_s7_ss16)
#   MODIFICATIONS vs nskom9jd:
#     * GRAD MODE (--measure-grad): latency, peak_memory AND flops measured on
#       the GRADIENT executable (graphax value_and_grad). NOT Jacobian mode.
#     * PopArt ON (always): per-channel EMA value-target normalisation.
#     * bkstep OFF (ALPHAGRAD_BKSTEP=0): no closed-loop trainability probe.
#     * QUALITY = cosine_sim ON THE GRAD (grad-vs-exact-grad cosine, weight 1.0).
#       In grad mode env compares out_approx[1] vs out_exact[1] (the grads).
#       Because cosine_sim carries a nonzero reward weight, _quality_is_rewarded
#       -> True -> the exact grad is COMPUTED (skip gate b5c9d0b does NOT fire),
#       so cosine is sensible (NOT Jacobian-mode 0.0).
#     * 4 REWARD CHANNELS via the per-channel selector (ALPHAGRAD_REWARD_CHANNELS,
#       commit 5172fcb): peak_memory, flops, latency_ns, cosine_sim.
#     * MANUAL LAMBDAS replicate PopArt's per-channel 1/sigma scaling (PopArt OFF):
#       reward = sum_k w_k * raw_channel_k (costs stored NEGATED so lower=better).
#       w_k = 0.06 / sigma_k for the cost channels, cosine_sim:1.0.
#       NN sigmas (nskom9jd FINAL PopArt sigmas): sigma_latency_ns=815346.5625,
#       sigma_peak_memory=57755560. sigma_flops CALIBRATED here (see WEIGHTS).
#     * 500 episodes, flat LR 1e-4 (--lr-decay-min-mult 1.0), seed 7.
#     * best sequences logged EVERY step (--best-sequences-every 1).
#
# H100 EVEN GPU+CPU DISTRIBUTION (8x H100-80GB, 256 cores):
#   trainer 1 GPU (--actor-num-gpus 1) + 7 GPU measure workers
#   (--num-cpu-workers 7 --cpu-actor-num-gpus 1 --exec-on-gpu) => 8 GPUs, one
#   per worker. Cores split evenly: 256/8 = 32 (--cpu-cores-per-actor 32
#   --cpu-cores-shared). Ray head --num-cpus=128 (NOT 256, raylet SIGABRTs) +
#   RAY_memory_monitor_refresh_ms=0 (6f50736: mem-monitor aborts on 1.5TB node).
# ============================================================================

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
# 6f50736 H100-safe Ray head sizing.
export RAY_memory_monitor_refresh_ms=0
export ALPHAGRAD_SKIP_COST_ANALYSIS=0   # need flops (idx1) measured on the grad exec
export DSNN_JAX_CACHE_REUSE=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export ALPHAGRAD_MAX_MEASURE_TOKENS=0
export ALPHAGRAD_MAX_MEASURE_MEM_GIB=${ALPHAGRAD_MAX_MEASURE_MEM_GIB:-60}
export ALPHAGRAD_MEASURE_MEM_SAFETY=${ALPHAGRAD_MEASURE_MEM_SAFETY:-1.3}
export ALPHAGRAD_MEASURE_MEM_FLOOR_GIB=${ALPHAGRAD_MEASURE_MEM_FLOOR_GIB:-1.0}
export ALPHAGRAD_SENTINEL_K=0.0
export ALPHAGRAD_PREVALIDATE_MEASURE=1
export ALPHAGRAD_MEASURE_INPROC_LRU=${ALPHAGRAD_MEASURE_INPROC_LRU:-0}
export ALPHAGRAD_MEASURE_DELETE_BUFFERS=${ALPHAGRAD_MEASURE_DELETE_BUFFERS:-1}
export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=${ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY:-64}
# >>> COST-HEAD AUXILIARY TASK (bridge-cse) <<<
# cost_head predicts the terminal MEASURED 4-tuple (symlog); aux Huber loss
# into the shared encoder (weight 0.3). Held CONSTANT across both ablation
# arms so the comparison stays clean.
export ALPHAGRAD_COST_HEAD_AUX=${ALPHAGRAD_COST_HEAD_AUX:-1}
export ALPHAGRAD_COST_HEAD_AUX_WEIGHT=${ALPHAGRAD_COST_HEAD_AUX_WEIGHT:-0.3}
export ALPHAGRAD_RECYCLE_RETRY_ON_OOM=${ALPHAGRAD_RECYCLE_RETRY_ON_OOM:-1}
export ALPHAGRAD_PROACTIVE_RECYCLE_EVERY=${ALPHAGRAD_PROACTIVE_RECYCLE_EVERY:-0}
# promotion-safe QUANT dtypes (drop the TypePromotionError dtypes).
export ALPHAGRAD_QUANT_ALLOWED="int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"
# Absolute device peak (real GPU high-water, not cached-exec entry-delta).
export ALPHAGRAD_PEAK_MEMORY_ABSOLUTE=1

# >>> SCALE-UP <<<
export ALPHAGRAD_NN_HIDDEN=256

# >>> bkstep OFF (USER-DIRECTED) <<<
export ALPHAGRAD_BKSTEP=0

# >>> ADDITIVE reward, RAW cost (no symlog): manual lambdas ARE 1/sigma <<<
# With PopArt OFF the per-channel weights must fold in 1/sigma themselves, so
# the cost channels must enter the reward RAW (no symlog compression). Keep the
# additive-symlog pass OFF and the cost cap OFF so w_k * raw_channel_k is exactly
# the PopArt-equivalent (raw / sigma) * w_outer.
export ALPHAGRAD_REWARD_MODE=additive
export ALPHAGRAD_ADDITIVE_SYMLOG_COST=0
export ALPHAGRAD_COST_SYMLOG_CAP=0
# Raw full-range cossim into the reward, no guide cap.
export ALPHAGRAD_COSSIM_GUIDE_CAP=${ALPHAGRAD_COSSIM_GUIDE_CAP:-0}
export ALPHAGRAD_RAW_COSSIM_THREAD=1

# >>> POLICY V2 (same backbone as nskom9jd) <<<
export ALPHAGRAD_POLICY=palimpsa

# >>> 4 REWARD CHANNELS via the per-channel selector (commit 5172fcb) <<<
# name:weight pairs -> build_reward_weights writes w[REWARD_INDEX[name]]=weight
# directly (takes PRECEDENCE over --rewards cmp/mem/acc). Costs are stored
# NEGATED in the env reward vector, so a POSITIVE weight = "reward low cost".
# WEIGHTS (0.06 / sigma for cost channels; cosine_sim:1.0):
#   latency_ns : 0.06 / 815346.5625 = 7.36e-8   (nskom9jd PopArt sigma)
#   peak_memory: 0.06 / 57755560    = 1.04e-9    (nskom9jd PopArt sigma)
#   flops      : 0.06 / SIGMA_FLOPS  (CALIBRATED, injected below)
#   cosine_sim : 1.0
FLOPS_WEIGHT="${FLOPS_WEIGHT:-1.167e-9}"   # 0.06 / calibrated sigma_flops
export ALPHAGRAD_REWARD_CHANNELS="xla_peak_memory:0.06,latency_ns:0.06,flops:0.06,cosine_sim:1.0"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
EPISODES="${EPISODES:-3400}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_gradmode_nn256_cse}"
RAY_PORT="${RAY_PORT:-6385}"
SEED="${SEED:-7}"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-4}"
MAX_SUBSTEPS="${MAX_SUBSTEPS:-16}"
ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
ENTROPY_COEF_FINAL="${ENTROPY_COEF_FINAL:-0.05}"
LR="${LR:-1e-4}"

# EVEN H100 layout: 1 trainer GPU + 7 measure GPUs; 32 cores each.
NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-7}"
CPU_CORES_PER_ACTOR="${CPU_CORES_PER_ACTOR:-32}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"   # 6f50736: 128 NOT 256 (else raylet SIGABRT)

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
NAME="${NAME:-gradmode_nn256_cse_s${SEED}_ss${MAX_SUBSTEPS}}"

echo "########## GRADMODE_NN256_CSE $(date) | head=$NODE0 eps=$EPISODES seed=$SEED lr=$LR channels=$ALPHAGRAD_REWARD_CHANNELS ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=$RAY_NUM_GPUS --num-cpus=$RAY_NUM_CPUS --block ) &
disown
sleep 25
echo "==== Ray up ($RAY_NUM_GPUS GPU / $RAY_NUM_CPUS CPU on $NODE0) $(date) ===="

OUT=$CAMP/nn256
LOG=$CAMP/logs/nn256.log
rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"

uv run --no-sync $PPO --name $NAME --variant ${VARIANT:-full} --seed $SEED \
  --example VmappedNeuralNetwork --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type xla_peak_memory \
  --measure-grad --exec-on-gpu --measure-latency --latency-inner-reps 50 \
  --entropy-coef $ENTROPY_COEF --entropy-coef-final $ENTROPY_COEF_FINAL \
  --dynamic-substeps --max-substeps ${MAX_SUBSTEPS} \
  --actor-num-gpus 1 --cpu-actor-num-gpus 1 --num-cpu-workers $NUM_CPU_WORKERS \
  --cpu-cores-per-actor $CPU_CORES_PER_ACTOR --cpu-cores-shared \
  --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
  --cpu-worker-recycle-every ${CPU_WORKER_RECYCLE_EVERY:-6} \
  --ray-address $HEAD_IP:$RAY_PORT \
  --advantage-norm scalar --ppo-epochs 4 \
  --lr $LR --lr-decay-min-mult 1.0 \
  --episodes $EPISODES --num-envs $NUM_ENVS --minibatches $MBS \
  --num-data-points 5 --reps-per-point 2 \
  --checkpoint-path $OUT/ckpt --checkpoint-every 100 \
  --best-sequences-json $OUT/best.json --best-sequences-every 1 \
  --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC seed=$SEED" >> "$LOG"
[ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
echo "########## GRADMODE_NN256_CSE DONE (rc=$RC) $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
