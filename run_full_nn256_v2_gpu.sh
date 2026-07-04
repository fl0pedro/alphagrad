#!/bin/bash
#SBATCH --job-name=full_nn256_v2
#SBATCH --time=24:00:00
#SBATCH --output=/Users/assmuth/dsnn/full_nn256_v2_%j.out
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
# ESTIMATE-BASED mem-gate (was fixed 8G floor). The old default floor skipped
# a ~0.32G diag measure whenever live free < 8G even with tens of GiB truly
# free -> spurious sentinels/failed_transitions under moderate node load
# (51557 ep235/236 + ep362->363 spike). Now the real protection is
# est*safety > headroom*free (COMPRESS/ViT densify still gated); the floor is
# just a small absolute minimum so we never measure into a near-empty device.
export ALPHAGRAD_MEASURE_MEM_FLOOR_GIB=1.0  # was 8.0 (fixed) — now a min, not the gate
export ALPHAGRAD_SENTINEL_K=0.0  # mu neutral sentinel (no -2sigma penalty; unknown != bad)
export ALPHAGRAD_PREVALIDATE_MEASURE=1
# MEASURE-GPU LEAK BOUND. Each measure = a DISTINCT per-(order,micro-action)
# jax.jit(jacve()).compile() executable on the measure GPU (exec_on_gpu). Orders
# vary per-step -> unbounded distinct executables; under PREALLOCATE=false PJRT
# holds each loaded executable + buffers, so the measure GPU fills over thousands
# of measures and even a KiB alloc OOMs (RESOURCE_EXHAUSTED at non-terminal steps,
# count CLIMBING across episodes — masked before by the old 8G mem-floor). The
# CpuApproximationServer calls jax.clear_caches()+gc every N evaluate calls (and
# always on an OOM sentinel) so XLA releases those executables; the on-disk
# compile cache survives -> recurring configs reload cheap. Keeps measure-GPU
# memory FLAT instead of monotonically growing.
export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=${ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY:-64}

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
# >>> COST SENSITIVITY: lambda INSIDE the symlog (symlog(lambda_inner*raw)) <<<
# The cost term was lambda_outer*symlog(raw): huge raw costs (latency ~1.13e5 ns,
# peak_memory ~1.30e8 B from ep-20 best.json) saturate symlog's log regime, so
# a 2x-cheaper order barely moved the reward (~few-e-4) — cost was rank-blind.
# Fix: per-channel lambda_inner ~ 1/typical_raw_cost puts a typical cost in the
# LINEAR regime (|lambda_inner*cost|~1) so order-of-magnitude differences are
# PRESERVED (sensitive), outliers still log-bounded. Then a SMALL outer weight
# (lambda_cmp/lambda_mem=0.06 below) keeps the cost a minor nudge (~0.08 typical
# combined) vs bkstep (~0.5). w_outer*symlog(lambda_inner*raw).
export ALPHAGRAD_INNER_LAMBDA_LATENCY_NS="${ALPHAGRAD_INNER_LAMBDA_LATENCY_NS:-9e-6}"   # 1/1.13e5
export ALPHAGRAD_INNER_LAMBDA_PEAK_MEMORY="${ALPHAGRAD_INNER_LAMBDA_PEAK_MEMORY:-7.7e-9}" # 1/1.30e8
# >>> V2 ANTI-HACK (pf7ityh6 post-mortem) <<<
# 1) COST_SYMLOG_CAP: clip symlog(lambda_inner*raw) to +/-1.25. The untrained
#    micro policy starts at ~20x-typical cost (latency symlog ~3.1), which gave
#    the cost term ~0.23 of scalar leverage (designed ~0.09) — the policy hacked
#    cost (COMPRESS/QUANT spam) while bkstep decayed 0.22->0.16. With the cap,
#    beyond ~2.4x typical the cost term SATURATES: no reward for making the
#    graph cheaper by making it worse; max combined cost term = 0.06*1.25*2 =
#    0.15 << quality (~0.8) at a good operating point.
export ALPHAGRAD_COST_SYMLOG_CAP="${ALPHAGRAD_COST_SYMLOG_CAP:-1.25}"
# 2) FAILED_ADV_STAMP: the -2.0 failed penalty is z-scored; an ALL-fail rollout
#    (constant -2) normalises to advantage ~0 => the basin is ABSORBING (pf7ityh6
#    pinned at mean_return -2/-16 for 80+ eps). Post-z-score overwrite of failed
#    rows' advantage with -1.0 keeps a repulsive gradient in all-fail batches.
export ALPHAGRAD_FAILED_ADV_STAMP="${ALPHAGRAD_FAILED_ADV_STAMP:-1.0}"
# Unidirectional Palimpsa (gated linear-attention, O(seq)) policy backbone —
# efficient over the now-unclipped ~8.2k-token grad graph (MAX_TOKENS=16384).
# V2: the uni mixer now carries the zero-init eqn_ids relational DAG-degree
# forget-gate modulation (same construction as palimpsa_bi).
export ALPHAGRAD_POLICY=palimpsa

# >>> POLICY V2 (this launcher's whole point) <<<
# ALPHAGRAD_RAY_MICROPOLICY=1 swaps SimplePPOAgent for MicroPPOAgent on the
# Ray PPO path: PointerVertexPolicy (per-vertex queries cross-attending the
# per-token encoder embeddings) + heads.MicroActionPolicy (autoregressive
# typed sub-episodes, REAL --max-substeps with sticky END + hard cap, gated
# joint log-prob/entropy) + a separate attention pool for the 10-channel
# value head. Reward stack / stabilisation fixes are untouched.
export ALPHAGRAD_RAY_MICROPOLICY="${ALPHAGRAD_RAY_MICROPOLICY:-1}"

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
CAMP="${CAMP:-campaign_full_nn256_v2}"
RAY_PORT="${RAY_PORT:-6384}"
SEED="${SEED:-2000}"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-32}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
# w_outer for the cost channels. lambda is now INSIDE the symlog (per-channel
# ALPHAGRAD_INNER_LAMBDA_* above); this OUTER weight just scales the whole
# symlog'd term. w_outer=0.06 -> typical combined cost term ~0.08 (symlog(~1)~
# 0.69 per channel), a minor nudge vs bkstep(~0.5); 2x-cheaper vs typical moves
# reward by ~0.035 (was ~0.0014 with the old lambda*symlog(raw) at 0.001).
LAMBDA_CMP="${LAMBDA_CMP:-0.06}"   # w_outer (latency); lambda_inner is 1/typical_raw
LAMBDA_MEM="${LAMBDA_MEM:-0.06}"   # w_outer (peak_memory)
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
# PopArt value normalisation (--value-norm popart): flag-gated, default
# baseline = byte-identical legacy path (symlog value loss + rollout
# advantage z-score). Set VALUE_NORM=popart (or ALPHAGRAD_POPART=1).
VALUE_NORM="${VALUE_NORM:-baseline}"
MAX_SUBSTEPS="${MAX_SUBSTEPS:-16}"
# ep49-collapse Fix 1 (job 51516): scale-only advantage bound (no mean-subtract)
# + hard clip backstop on the PopArt scalar advantage path — caps the std-13
# blowup directly. Env-gated / revertible (ADV_CLIP=0 disables).
export ALPHAGRAD_ADV_STD_FLOOR="${ALPHAGRAD_ADV_STD_FLOOR:-0.5}"
export ALPHAGRAD_ADV_CLIP="${ALPHAGRAD_ADV_CLIP:-8.0}"
# Fix 4: per-channel PopArt sigma floor. Global floor 0.1 over-amplified the
# near-homogeneous bkstep(9)+cosine(6) channels (var->0 -> A_k/sigma_k inflated).
# Floor those quality channels higher (0.2); cost channels keep base 0.1.
export ALPHAGRAD_POPART_SIGMA_MIN="${ALPHAGRAD_POPART_SIGMA_MIN:-0.1}"
export ALPHAGRAD_POPART_SIGMA_MIN_QUALITY="${ALPHAGRAD_POPART_SIGMA_MIN_QUALITY:-0.2}"
# Fix 2: PPO KL early-stop (reject a catastrophic ep49-type update). 0 disables.
export ALPHAGRAD_PPO_TARGET_KL="${ALPHAGRAD_PPO_TARGET_KL:-0.15}"
NAME="full_nn256_v2${VARIANT:+_$VARIANT}$([ "$VALUE_NORM" = popart ] && echo _popart)_s${SEED}_ss${MAX_SUBSTEPS}"

echo "########## FULL_NN256_V2 $(date) | head=$NODE0 eps=$EPISODES NN_HIDDEN=256 variant=${VARIANT:-full} value_norm=$VALUE_NORM measure-grad ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=4 --num-cpus=64 --block ) &
disown
sleep 25
echo "==== Ray up (4 GPU $NODE0) $(date) ===="

OUT=$CAMP/nn256
LOG=$CAMP/logs/nn256.log
rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"

uv run --no-sync $PPO --name $NAME --variant ${VARIANT:-full} --seed $SEED \
  --example VmappedNeuralNetwork --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
  --measure-grad $([ "${EXEC_ON_GPU:-1}" = 1 ] && echo --exec-on-gpu) --measure-latency --latency-inner-reps 50 \
  --lambda-cmp $LAMBDA_CMP --lambda-mem $LAMBDA_MEM --lambda-acc $LAMBDA_ACC --lambda-frob 0.0 \
  --lambda-cossim-guide $LAMBDA_COSSIM_GUIDE \
  --entropy-coef $ENTROPY_COEF --entropy-coef-final $ENTROPY_COEF_FINAL \
  --dynamic-substeps --max-substeps ${MAX_SUBSTEPS} \
  --actor-num-gpus 1 --cpu-actor-num-gpus 1 --num-cpu-workers 3 \
  --cpu-cores-per-actor 8 --cpu-cores-shared \
  --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
  --cpu-worker-recycle-every ${CPU_WORKER_RECYCLE_EVERY:-20} \
  --ray-address $HEAD_IP:$RAY_PORT \
  --advantage-norm scalar --value-norm $VALUE_NORM --ppo-epochs 4 --anti-degeneracy none \
  --cosine-lower-bound 0.0 --cosine-upper-bound 1.0 \
  --episodes $EPISODES --num-envs $NUM_ENVS --minibatches $MBS \
  --num-data-points 5 --reps-per-point 2 \
  --best-sequences-json $OUT/best.json --best-sequences-every 5 \
  --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC seed=$SEED" >> "$LOG"
[ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
echo "########## FULL_NN256_V2 DONE (rc=$RC) $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
