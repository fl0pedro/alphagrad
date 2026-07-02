#!/bin/bash
#SBATCH --job-name=full_nn256_grad
#SBATCH --time=24:00:00
#SBATCH --output=/Users/assmuth/dsnn/full_nn256_grad_%j.out
#SBATCH --partition=pgi15
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu15
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:4
#SBATCH --cpus-per-task=64

# ============================================================================
# FULL-variant (DIAG+QUANT+COMPRESS) PPO search, SCALED UP:
#   * VmappedNeuralNetwork, hidden=256 (ALPHAGRAD_NN_HIDDEN=256), MNIST
#   * variant=full (COMPRESS kept in)
#   * --measure-grad (approx loss-gradient, commit 489c8d6) executed ON GPU
#   * REWARD CHANNELS = {cmp: LATENCY, mem: PEAK_MEMORY, acc: B_kstep trainability
#     accuracy}. The acc channel is B_kstep (ALPHAGRAD_ACC_PROXY=bkstep): run K
#     Adam steps of REAL MNIST training with the rule's OWN approx gradient, then
#     read MNIST test accuracy in [0,1] — replaces cosine as the quality signal.
#   * ADDITIVE bkstep-dominant reward: lambda_acc=1.0 (bkstep-acc in [0,1]),
#     lambda_cmp=lambda_mem=0.005, symlog cost.
#   * flops (idx1) + xla_peak_memory (idx8) ALSO measured+logged (NOT rewarded):
#     ALPHAGRAD_SKIP_COST_ANALYSIS=0 so the XLA cost analysis runs.
#   * 1000 episodes, WANDB online
#
# LAYOUT (ViT-proven multi-GPU, no single-GPU deadlock):
#   PPO trainer on 1 GPU (--actor-num-gpus 1) + 3 GPU measure-actors
#   (--cpu-actor-num-gpus 1 --exec-on-gpu --num-cpu-workers 3) => 4 GPUs.
#   BOTH the PPO model AND the grad measurements run on GPU.
# ============================================================================

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DISABLE_IMPORT_WARNING=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=0  # RUN XLA cost analysis -> flops (idx1) + xla_peak_memory (idx8) computed+logged (analytic, not rewarded)
export DSNN_JAX_CACHE_REUSE=1
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export ALPHAGRAD_MAX_MEASURE_TOKENS=0
# Fix 2(b)(ii): give the mem-gate real headroom so a too-big COMPRESS is CLEANLY
# GATED (bounded sentinel) instead of racing to a device OOM. Lower the static
# cap 72->60G AND restore a safety multiplier 1.0->1.3 on the per-order estimate:
# the gate now skips at est*1.3 > 60G, leaving ~36G of the 96GB Blackwell as
# margin for the (unpredicted) autotuner/densify workspace the estimate can't
# see. Genuine COMPRESS up to ~46G real still runs; only the OOM-prone ones skip.
export ALPHAGRAD_MAX_MEASURE_MEM_GIB=60  # was 72 — 36G headroom absorbs COMPRESS densify workspace under-count
export ALPHAGRAD_MEASURE_MEM_SAFETY=1.3  # was 1.0 — margin so OOMs are rare (gate fires before exec, not after)
export ALPHAGRAD_PREVALIDATE_MEASURE=1

# >>> Fix 2(a): QUANT dtype restriction (drop the TypePromotionError dtypes) <<<
# graphax's mixed-precision shim (dtype_compute._NARROW_PROMOTION_REP) covers
# scaled-mul but NOT every grad-measure op path, so the sub-byte ints
# (int2/int4/uint2/uint4), float4, the exotic float8 variants (*fnuz / e3m4 /
# e8m0fnu / e4m3b11fnuz), and complex still raise TypePromotionError mid-measure
# -> sentinel. The plain-width ints (int32/64, uint*) and float32/64 are no-op
# "quantizations" (not narrower than the f32 grad). Restrict the QUANT head to
# the standard, promotion-safe NARROW dtypes: int8/int16, the two standard
# float8s, bf16, f16. Masked in sample AND log_prob (heads.py::_quant_dtype_mask)
# so PPO ratios stay consistent. Keeps QUANT in the action space, just bounded.
export ALPHAGRAD_QUANT_ALLOWED="int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"

# >>> SCALE-UP + REWARD KNOBS <<<
export ALPHAGRAD_NN_HIDDEN=256
export ALPHAGRAD_REWARD_MODE=additive
export ALPHAGRAD_ADDITIVE_SYMLOG_COST=1
# Bidirectional Palimpsa (gated linear-attention, O(seq)) policy backbone —
# efficient over the now-unclipped ~8.2k-token grad graph (MAX_TOKENS=16384).
export ALPHAGRAD_POLICY=palimpsa_bi

# >>> B_kstep TRAINABILITY as the acc reward channel <<<
# ALPHAGRAD_ACC_PROXY=bkstep routes the --rewards acc weight to the B_kstep
# channel (idx 9) instead of cosine_sim; ALPHAGRAD_BKSTEP=1 turns on the probe
# inside env._callback (terminal + --measure-grad). K=40 Adam steps x 2 seeds is
# the cheap-but-predictive sweet spot (memory: ~40 is the plateau; 2 seeds keeps
# per-episode cost bounded — ~80 tiny approx-grad calls per terminal step).
export ALPHAGRAD_ACC_PROXY=bkstep
export ALPHAGRAD_BKSTEP=1
export ALPHAGRAD_BKSTEP_K="${ALPHAGRAD_BKSTEP_K:-40}"
export ALPHAGRAD_BKSTEP_SEEDS="${ALPHAGRAD_BKSTEP_SEEDS:-2}"

# >>> CAPPED-COSSIM GUIDE (anti flat-zero-basin) <<<
# The untrained policy sits at cos<=0 (all-COMPRESS -> degenerate Jacobian ->
# cos clamped 0), where B_kstep(fracred) has NO gradient (measured: fracred~0
# for cos<=0, lifting to ~0.30 by cos~0.1). ALPHAGRAD_COSSIM_GUIDE_CAP=C makes
# env._callback emit min(cossim, C) on the cosine_sim channel (min, NOT
# clip-at-0 -> monotonic climb from NEGATIVE cos up to C). --lambda-cossim-guide
# weights that channel: below the edge it dominates the (now symlog'd) cost so
# the policy climbs; above C it is constant -> B_kstep resolves the tradeoff.
# C=0.1 = the measured trainability edge (fracred liftoff ~cos 0.03->0.1, prior
# sweep 0.10-0.15). lambda_guide=1.5 -> below-edge swing ~1.5*0.4=0.6 >> cost
# ~0.12; above-edge cap 1.5*0.1=0.15 < lambda_acc*bkstep(~0.5-0.66).
export ALPHAGRAD_COSSIM_GUIDE_CAP="${ALPHAGRAD_COSSIM_GUIDE_CAP:-0.1}"
LAMBDA_COSSIM_GUIDE="${LAMBDA_COSSIM_GUIDE:-1.5}"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py

EPISODES="${EPISODES:-1000}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_full_nn256_grad}"
RAY_PORT="${RAY_PORT:-6381}"
SEED="${SEED:-2000}"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-32}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
LAMBDA_CMP="${LAMBDA_CMP:-0.001}"
LAMBDA_MEM="${LAMBDA_MEM:-0.001}"
# Fix 3: entropy-coef anneal range. Start high (0.05) so the 6-head action
# space explores early, decay linearly to a small floor (0.002) by the end of
# the run so the policy COMMITS to the best rule (entropy was pinned ~1.99 =
# never converging because the coef was frozen). Wired into the per-episode
# anneal in ppo_ray_worker.run_rollout_and_train.
ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
ENTROPY_COEF_FINAL="${ENTROPY_COEF_FINAL:-0.002}"

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
echo "########## FULL_NN256_GRAD $(date) | head=$NODE0 eps=$EPISODES NN_HIDDEN=256 variant=full measure-grad ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=4 --num-cpus=64 --block ) &
disown
sleep 25
echo "==== Ray up (4 GPU $NODE0) $(date) ===="

OUT=$CAMP/nn256
LOG=$CAMP/logs/nn256.log
rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"

uv run --no-sync $PPO --name full_nn256_grad_s${SEED} --variant full --seed $SEED \
  --example VmappedNeuralNetwork --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
  --measure-grad --exec-on-gpu --measure-latency --latency-inner-reps 50 \
  --lambda-cmp $LAMBDA_CMP --lambda-mem $LAMBDA_MEM --lambda-acc $LAMBDA_ACC --lambda-frob 0.0 \
  --lambda-cossim-guide $LAMBDA_COSSIM_GUIDE \
  --entropy-coef $ENTROPY_COEF --entropy-coef-final $ENTROPY_COEF_FINAL \
  --dynamic-substeps --max-substeps 16 \
  --actor-num-gpus 1 --cpu-actor-num-gpus 1 --num-cpu-workers 3 \
  --cpu-cores-per-actor 8 --cpu-cores-shared \
  --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
  --cpu-worker-recycle-every 0 \
  --ray-address $HEAD_IP:$RAY_PORT \
  --advantage-norm scalar --ppo-epochs 4 --anti-degeneracy none \
  --cosine-lower-bound 0.0 --cosine-upper-bound 1.0 \
  --episodes $EPISODES --num-envs $NUM_ENVS --minibatches $MBS \
  --num-data-points 5 --reps-per-point 2 \
  --best-sequences-json $OUT/best.json --best-sequences-every 5 \
  --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC seed=$SEED" >> "$LOG"
[ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
echo "########## FULL_NN256_GRAD DONE (rc=$RC) $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
