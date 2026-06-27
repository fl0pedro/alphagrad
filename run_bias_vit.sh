#!/bin/bash
#SBATCH --job-name=jacbiasvit
#SBATCH --time=04:00:00
#SBATCH --output=/Users/assmuth/dsnn/jacbiasvit_%j.out
#SBATCH --partition=pgi15-h100
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64

# Estimator-bias quality-proxy bake-off run (proxy-bias worktree).
#   * Replaces single-point Jacobian cosine with the BIAS half of the
#     gradient-error bias/variance decomposition:
#       bias = ||mean_s g_approx - mean_s g_exact|| / ||mean_s g_exact||
#       quality = exp(-bias)  (written into the cosine_sim channel)
#     Predicts converged-to-wrong-optimum (systematic offset) that a
#     per-sample cosine is blind to.
#   * Requires the cross-sample MEAN of the raw per-sample gradient pytrees,
#     which is formable ONLY on the single-call all-points path
#     (env._callback point_idx=-1). We DO NOT pass --measure-queue, so the
#     terminal step uses _fan_out_tokenize -> evaluate(point_idx=-1) and all
#     per-sample pytrees land together in out_approxs/out_exacts.
#   * Uses MY worktree source via PYTHONPATH + an absolute PPO path so the
#     patched env.py is the one that loads (banner verifies the path).
#   * Node = gpu9 (4x RTX 4090), != gpu15/16/17 (don't touch other runs).

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export DSNN_JAX_CACHE_REUSE=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export ALPHAGRAD_MAX_MEASURE_TOKENS=0
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_PREVALIDATE_MEASURE=1
export GRAPHAX_STATE_TOKENS=1
# ViT grads are larger; on the 80GB H100 the 8G live-free floor is over-
# conservative. Lower the floor and raise the static cap (coordinator guidance).
export ALPHAGRAD_MEASURE_MEM_FLOOR_GIB="${ALPHAGRAD_MEASURE_MEM_FLOOR_GIB:-4}"
export ALPHAGRAD_MAX_MEASURE_MEM_GIB="${ALPHAGRAD_MAX_MEASURE_MEM_GIB:-60}"

# >>> BIAS PROXY + WORKTREE SOURCE <<<
WT=~/dsnn/wt-bias
export PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}"
export ALPHAGRAD_QUALITY_PROXY=bias       # the proxy under test
export ALPHAGRAD_REWARD_MODE=mult         # fidelity-gated cheapness gates on it
export ALPHAGRAD_LOCAL_TOKENIZE=1         # non-terminal tokens computed locally
export ALPHAGRAD_DEBUG_QUALITY=1          # log bias vs per-sample cosine

PPO="$WT/src/alphagrad/approx/ppo_ray.py"

EPISODES="${EPISODES:-1000}"
WANDB="${WANDB:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_bias_vit}"
PPO_GPUS="${PPO_GPUS:-1}"
RAY_PORT="${RAY_PORT:-6385}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
LAMBDA_FROB="${LAMBDA_FROB:-0.0}"
NDP="${NDP:-8}"                            # >= 8 for a trustworthy cross-sample mean
MAXWALL="${MAXWALL:-0}"                     # >0 = stop after this many wall seconds

MODELS_STR="${MODELS_STR:-VmappedViT}"
read -r -a MODELS <<< "$MODELS_STR"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-32}"

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
echo "########## JACBIAS $(date) | head=$NODE0 eps=$EPISODES models=${MODELS[*]} NDP=$NDP ##########"
echo "==== PYTHONPATH=$PYTHONPATH ===="
echo "==== PPO=$PPO ===="
# Banner: verify the patched env.py is the one that loads.
uv run --no-sync python -c "import alphagrad.approx.env as e; print('==== env.py import path:', e.__file__); print('==== has _estimator_bias:', hasattr(e, '_estimator_bias'))"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=2 --num-cpus=32 --block ) &
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
    uv run --no-sync $PPO --name jacbias_${M}_s${SLOT} --variant full --seed $SEED \
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
      --max-wall-seconds $MAXWALL \
      --num-data-points $NDP --reps-per-point 2 \
      --best-sequences-json $OUT/best.json --best-sequences-every 5 \
      --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
    local RC=$?
    echo "EXIT=$RC model=$M slot=$SLOT seed=$SEED" >> "$LOG"
    [ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
    echo "[s$SLOT] END   $M (rc=$RC) $(date +%H:%M:%S)"
}

for ((S=0; S<${#MODELS[@]}; S++)); do launch_one "$S"; done

echo "########## JACBIAS DONE $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
