#!/bin/bash
#SBATCH --job-name=jac_bkstep
#SBATCH --time=04:00:00
#SBATCH --output=/Users/assmuth/dsnn/jac_bkstep_%j.out
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu18
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --cpus-per-task=64

# B_kstep-light QUALITY proxy bake-off run.
#   * Uses the proxy-bkstep WORKTREE src (~/dsnn/wt-bkstep/src) prepended on
#     PYTHONPATH so this run imports the patched env.py (NOT ~/dsnn/alphagrad).
#   * ALPHAGRAD_QUALITY_PROXY=bkstep  -> terminal cosine_sim channel carries the
#     K-step SGD true-loss-drop ratio in [0,1] instead of single-point Jacobian
#     cosine; the multiplicative cosine-gate reward then gates on B_kstep.
#   * ALPHAGRAD_REWARD_MODE=mult      -> fidelity-gated cheapness (anti-hacking).
#   * --measure-grad                  -> compiled fns return (scalar_loss, grads)
#     so B_kstep can SGD-step + read the loss; required by the proxy.
#   * VmappedNeuralNetwork on MNIST   -> the env's 128-hidden MLP proxy model
#     (fast; B_kstep is meaningful for a trainable MLP).
#   * Node pgi15-gpu18 (idle Blackwell, != gpu15/16/17; never touch bi 50918).

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
# >>> WORKTREE IMPORT <<< prepend the proxy-bkstep src so env.py is the patched one.
export PYTHONPATH="$HOME/dsnn/wt-bkstep/src:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export DSNN_JAX_CACHE_REUSE=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export ALPHAGRAD_MAX_MEASURE_TOKENS=0
export ALPHAGRAD_MAX_MEASURE_MEM_GIB=0
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_PREVALIDATE_MEASURE=1
export GRAPHAX_STATE_TOKENS=1
# >>> B_kstep proxy switches <<<
export ALPHAGRAD_QUALITY_PROXY=bkstep
export ALPHAGRAD_REWARD_MODE=mult
export ALPHAGRAD_LOCAL_TOKENIZE=1
export ALPHAGRAD_BKSTEP_K="${ALPHAGRAD_BKSTEP_K:-3}"
export ALPHAGRAD_BKSTEP_S="${ALPHAGRAD_BKSTEP_S:-2}"
export ALPHAGRAD_BKSTEP_LR="${ALPHAGRAD_BKSTEP_LR:-0.1}"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py

EPISODES="${EPISODES:-1000}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_bkstep}"
PPO_GPUS="${PPO_GPUS:-1}"
RAY_PORT="${RAY_PORT:-6381}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
LAMBDA_FROB="${LAMBDA_FROB:-0.0}"

MODELS_STR="${MODELS_STR:-VmappedNeuralNetwork}"
read -r -a MODELS <<< "$MODELS_STR"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-32}"
MAX_SUBSTEPS="${MAX_SUBSTEPS:-16}"

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
echo "########## JAC_BKSTEP $(date) | head=$NODE0 eps=$EPISODES models=${MODELS[*]} QUALITY_PROXY=bkstep K=$ALPHAGRAD_BKSTEP_K S=$ALPHAGRAD_BKSTEP_S ##########"
echo "PYTHONPATH=$PYTHONPATH"
uv run --no-sync python -c "import alphagrad.approx.env as e; print('[banner] env.py import path =', e.__file__)"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=4 --num-cpus=64 --block ) &
disown
sleep 25
echo "==== Ray up (4 GPU $NODE0) $(date) ===="

launch_one() {
    local SLOT=$1
    local M=${MODELS[$SLOT]}
    local SEED=$(( 3000 + SLOT ))
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
    uv run --no-sync $PPO --name jacbkstep_${M}_s${SLOT} --variant full --seed $SEED \
      --example $M --dataset mnist \
      --measure-grad \
      --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
      --lambda-cmp $LCMP --lambda-mem $LMEM --lambda-acc $LAMBDA_ACC --lambda-frob $LAMBDA_FROB \
      --measure-latency --dynamic-substeps --max-substeps $MAX_SUBSTEPS \
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

echo "########## JAC_BKSTEP DONE $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
