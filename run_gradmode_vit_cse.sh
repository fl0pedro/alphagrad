#!/bin/bash
#SBATCH --job-name=gradmode_vit_cse
#SBATCH --time=24:00:00
#SBATCH --output=/Users/assmuth/dsnn/gradmode_vit_cse_%j.out
#SBATCH --partition=pgi15-h100
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=256

# ============================================================================
# GRAD-MODE POSTER CHILD (ViT), PopArt-OFF variant (bridge-cse, USER-DIRECTED)
# Reference working grad-mode run = wandb nskom9jd (full_nn256_v2_popart_s7_ss16)
#   Same MODIFICATIONS as the NN launcher (grad-mode, PopArt OFF, bkstep OFF,
#   cosine_sim-on-grad quality weight 1.0, 4 channels via ALPHAGRAD_REWARD_CHANNELS,
#   manual 1/sigma lambdas, 500 eps, flat LR 1e-4, seed 7, best-seq-every 1),
#   with ViT-specific changes:
#     * --example VmappedViT, ALPHAGRAD_VIT_POW2=1 (shape-storm guard),
#       ALPHAGRAD_PREVALIDATE_MEASURE=1.
#     * ViT sigmas are MUCH larger than the NN sigmas, so ALL of ViT's sigmas
#       (latency_ns, peak_memory, flops) are CALIBRATED here (see WEIGHTS).
#     * --latency-inner-reps TUNED DOWN (ViT's grad exec is ms-scale, so far
#       fewer reps still give a stable reading — see the calibration CV sweep).
# H100 EVEN GPU+CPU layout identical to the NN launcher (1 trainer + 7 measure
# GPUs, 32 cores each, Ray head 8 GPU / 128 CPU, mem-monitor OFF).
# ============================================================================

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_memory_monitor_refresh_ms=0
export ALPHAGRAD_SKIP_COST_ANALYSIS=0
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
export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=${ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY:-0}
export ALPHAGRAD_RECYCLE_RETRY_ON_OOM=${ALPHAGRAD_RECYCLE_RETRY_ON_OOM:-1}
export ALPHAGRAD_PROACTIVE_RECYCLE_EVERY=${ALPHAGRAD_PROACTIVE_RECYCLE_EVERY:-0}
export ALPHAGRAD_QUANT_ALLOWED="int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"
export ALPHAGRAD_PEAK_MEMORY_ABSOLUTE=1

# >>> ViT measurability stack <<<
export ALPHAGRAD_VIT_POW2=1

# >>> bkstep OFF <<<
export ALPHAGRAD_BKSTEP=0

# >>> ADDITIVE reward, RAW cost (manual lambdas ARE 1/sigma) <<<
export ALPHAGRAD_REWARD_MODE=additive
export ALPHAGRAD_ADDITIVE_SYMLOG_COST=0
export ALPHAGRAD_COST_SYMLOG_CAP=0
export ALPHAGRAD_COSSIM_GUIDE_CAP=${ALPHAGRAD_COSSIM_GUIDE_CAP:-0}
export ALPHAGRAD_RAW_COSSIM_THREAD=1

export ALPHAGRAD_POLICY=palimpsa

# >>> 4 REWARD CHANNELS (per-channel selector, commit 5172fcb) <<<
# ViT weights = 0.06 / sigma_vit_channel (CALIBRATED), cosine_sim:1.0.
LATENCY_WEIGHT="${LATENCY_WEIGHT:-9.386e-8}"
PEAK_WEIGHT="${PEAK_WEIGHT:-1.312e-11}"
FLOPS_WEIGHT="${FLOPS_WEIGHT:-1.109e-11}"
export ALPHAGRAD_REWARD_CHANNELS="peak_memory:0.06,latency_ns:0.06,flops:0.06,cosine_sim:1.0"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py
EPISODES="${EPISODES:-500}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_gradmode_vit_cse}"
RAY_PORT="${RAY_PORT:-6386}"
SEED="${SEED:-7}"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-8}"
MAX_SUBSTEPS="${MAX_SUBSTEPS:-16}"
ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
ENTROPY_COEF_FINAL="${ENTROPY_COEF_FINAL:-0.05}"
LR="${LR:-1e-4}"
# ViT grad exec is ms-scale -> far fewer inner reps still stable (see CV sweep).
LATENCY_INNER_REPS="${LATENCY_INNER_REPS:-20}"

NUM_CPU_WORKERS="${NUM_CPU_WORKERS:-7}"
CPU_CORES_PER_ACTOR="${CPU_CORES_PER_ACTOR:-32}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-8}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-128}"

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
NAME="gradmode_vit_cse_s${SEED}_ss${MAX_SUBSTEPS}"

echo "########## GRADMODE_VIT_CSE $(date) | head=$NODE0 eps=$EPISODES seed=$SEED lr=$LR inner_reps=$LATENCY_INNER_REPS channels=$ALPHAGRAD_REWARD_CHANNELS ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=$RAY_NUM_GPUS --num-cpus=$RAY_NUM_CPUS --block ) &
disown
sleep 25
echo "==== Ray up ($RAY_NUM_GPUS GPU / $RAY_NUM_CPUS CPU on $NODE0) $(date) ===="

OUT=$CAMP/vit
LOG=$CAMP/logs/vit.log
rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"

uv run --no-sync $PPO --name $NAME --variant ${VARIANT:-full} --seed $SEED \
  --example ${EXAMPLE:-VmappedViT} --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
  --measure-grad --exec-on-gpu --measure-latency --latency-inner-reps $LATENCY_INNER_REPS \
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
  --best-sequences-json $OUT/best.json --best-sequences-every 1 \
  --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC seed=$SEED" >> "$LOG"
[ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
echo "########## GRADMODE_VIT_CSE DONE (rc=$RC) $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
