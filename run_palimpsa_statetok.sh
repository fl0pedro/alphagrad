#!/bin/bash
#SBATCH --job-name=jac_palimpsa_statetok
#SBATCH --time=2-00:00:00
#SBATCH --output=/Users/assmuth/dsnn/jac_palimpsa_statetok%j.out
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu16
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --cpus-per-task=64

# ViT-Palimpsa rerun from palimpsapprox-statetok: same recipe as
# run_palimpsa.sh (job 50646, legacy tokenizer) EXCEPT
#   * GRAPHAX_STATE_TOKENS=1  -> graphax's gated append-only state-tokenizer
#     (order + per-vertex DIAG/COMPRESS/QUANT micro-actions), no per-step
#     Jacobian re-trace; ~10x shorter token stream on ViT.
#   * NUM_ENVS=4, --cpu-actor-num-gpus 1 --num-cpu-workers 3 (1 measure
#     actor/GPU), prevalidate on, ALPHAGRAD_POLICY=palimpsa.
#   * runs on a FREE 4-GPU node (gpu16) != gpu15 != gpu20 so 50646 (the
#     untouched legacy baseline) keeps running for A/B. GPU budget: PPO=1
#     GPU + 3 measure-actor GPUs = 4 (a single-GPU node deadlocks: PPO
#     grabs the only GPU and the measure actors stay pending forever).
# The policy retrains FRESH on the new (shorter) state representation.

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
# Compile/cache fixes inherited from palimpsapprox.
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export DSNN_JAX_CACHE_REUSE=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export ALPHAGRAD_MAX_MEASURE_TOKENS=0
export ALPHAGRAD_MAX_MEASURE_MEM_GIB=0
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_PREVALIDATE_MEASURE=1
# >>> THE STATE-TOKENIZER SWITCH <<<
export GRAPHAX_STATE_TOKENS=1

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py

EPISODES="${EPISODES:-1000}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_jac_statetok}"
PPO_GPUS="${PPO_GPUS:-1}"
RAY_PORT="${RAY_PORT:-6380}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
LAMBDA_FROB="${LAMBDA_FROB:-0.0}"

MODELS_STR="${MODELS_STR:-VmappedViT}"
read -r -a MODELS <<< "$MODELS_STR"
NUM_ENVS="${NUM_ENVS:-4}"                 # OOM at 16 (MORL note); 4 is safe
MBS="${MBS:-32}"

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
echo "########## JAC_STATETOK $(date) | head=$NODE0 eps=$EPISODES models=${MODELS[*]} STATE_TOKENS=1 ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=4 --num-cpus=64 --block ) &
disown
sleep 25
echo "==== Ray up (4 GPU $NODE0) $(date) ===="

launch_one() {
    local SLOT=$1
    local M=${MODELS[$SLOT]}
    local SEED=$(( 2000 + SLOT ))
    local OUT=$CAMP/${M}/slot${SLOT}
    local LOG=$CAMP/logs/${M}_slot${SLOT}.log
    if [ -f "$OUT/.done" ]; then echo "[s$SLOT] SKIP $M (done)"; return 0; fi
    rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"
    case "$M" in
      VmappedNeuralNetwork) local LCMP=0.060 LMEM=0.057;;
      VmappedConvNet)       local LCMP=0.046 LMEM=0.045;;
      VmappedViT)           local LCMP=0.050 LMEM=0.050;;
      *)                    local LCMP=0.050 LMEM=0.050;;
    esac
    echo "[s$SLOT] START $M envs=$NUM_ENVS seed=$SEED $(date +%H:%M:%S)"
    uv run --no-sync $PPO --name jacstatetok_${M}_s${SLOT} --variant full --seed $SEED \
      --example $M --dataset mnist \
      --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
      --lambda-cmp $LCMP --lambda-mem $LMEM --lambda-acc $LAMBDA_ACC --lambda-frob $LAMBDA_FROB \
      --measure-latency --dynamic-substeps --max-substeps 16 \
      --actor-num-gpus $PPO_GPUS --cpu-actor-num-gpus 1 --exec-on-gpu --num-cpu-workers 3 --cpu-cores-per-actor 8 --cpu-cores-shared \
      --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
      --cpu-worker-recycle-every 0 \
      --latency-timer perf_counter --latency-inner-reps 5 --latency-warmup 2 --latency-winsor 0.2 \
      --ray-address $HEAD_IP:$RAY_PORT \
      --advantage-norm scalar --ppo-epochs 4 --anti-degeneracy none \
      --cosine-lower-bound 0.0 --cosine-upper-bound 1.0 \
      --episodes $EPISODES --num-envs $NUM_ENVS --minibatches $MBS \
      --num-data-points 5 --reps-per-point 2 \
      --best-sequences-json $OUT/best.json --best-sequences-every 5 \
      --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
    local RC=$?
    echo "EXIT=$RC model=$M slot=$SLOT seed=$SEED" >> "$LOG"
    [ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
    echo "[s$SLOT] END   $M (rc=$RC) $(date +%H:%M:%S)"
}

for ((S=0; S<${#MODELS[@]}; S++)); do launch_one "$S"; done

echo "########## JAC_STATETOK DONE $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
