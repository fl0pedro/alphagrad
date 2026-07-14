#!/bin/bash
#SBATCH --job-name=vit_h100
#SBATCH --time=24:00:00
#SBATCH --output=/Users/assmuth/dsnn/vit_h100_%j.out
#SBATCH --partition=pgi15-h100
#SBATCH --nodes=1
#SBATCH --nodelist=pgi15-gpu14
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --cpus-per-task=128

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
# >>> C2 "winner" DISPROVEN AT SCALE (long-validation job 51621/51622/51623, 2026-07) <<<
# The claimed "flat ~8.8 GiB, 0 OOM" for LRU=128+delbuf was a SHORT-PROBE
# ARTIFACT. Long standalone-harness validation (CpuApproximationServer.evaluate
# over the live diag_gcd/256-NN grad measure, hundreds of measures) proved:
#   * LRU 16 == 32 == 128 -> IDENTICAL 34345 MiB monotonic climb (no plateau):
#     the OrderedDict eviction .delete() does NOT reclaim device memory.
#   * jax.clear_caches() at every=16 AND every=1 -> SAME climb + recompile tax:
#     does NOT reclaim either. The retention is XLA/PJRT-INTERNAL, below any
#     Python/JAX API. => NO in-process method bounds the leak.
# On the live run this fills 66-73 GiB (the ~71 GiB PJRT cap) WITHIN episode 1
# (seed7 51613 reached call=7543 still in ep1 -> 6237 RESOURCE_EXHAUSTED). Only
# PROCESS TEARDOWN (recycle) frees the memory, but per-EPISODE recycle cannot
# bound a single episode's thousands of measures. MEMORY CONFIG ALONE CANNOT
# reach sentinels->0 at scale -> measurement-surrogate heads are required (see
# thesis/ scope note). Until then: LRU OFF (it only wastes RAM), tight process
# recycle = least-bad partial bound (frees cross-episode; NOT within-episode).
export ALPHAGRAD_MEASURE_INPROC_LRU=${ALPHAGRAD_MEASURE_INPROC_LRU:-0}
export ALPHAGRAD_MEASURE_DELETE_BUFFERS=${ALPHAGRAD_MEASURE_DELETE_BUFFERS:-1}
export ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=${ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY:-0}
# >>> PRIMARY LEAK FIX: recycle+retry-on-OOM (revertible) <<<
# Memory config PROVEN unable to bound the per-measure XLA compile leak; only
# PROCESS TEARDOWN frees the XLA-internal executable retention. When a measure
# OOMs, CpuApproxPool RECYCLES that measure actor (kill+respawn -> fresh
# process -> memory freed) and RETRIES the measure ONCE on the fresh actor, so
# the run keeps progressing with REAL measurements instead of hanging /
# sentinel-storming at the ~71 GiB PJRT cap. Exactly one recycle+retry per
# OOMd measure; a retry that also OOMs keeps the (neutral) sentinel. Set to 0
# to revert to the old bounded-sentinel-on-OOM behaviour.
export ALPHAGRAD_RECYCLE_RETRY_ON_OOM=${ALPHAGRAD_RECYCLE_RETRY_ON_OOM:-1}
# Optional PROACTIVE recycle (A/B alternative): recycle a measure actor after
# it has served N measures, BEFORE it reaches the OOM point. 0 = off
# (reactive-on-OOM only, the shipped default).
export ALPHAGRAD_PROACTIVE_RECYCLE_EVERY=${ALPHAGRAD_PROACTIVE_RECYCLE_EVERY:-0}

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
# >>> RAW COST INTO POPART (bridge-cse, USER-DIRECTED) <<<
# PopArt now does ALL the magnitude normalisation. The cost channels
# (latency_ns, peak_memory) enter the reward as RAW (negated) values — NO
# symlog, NO INNER_LAMBDA rescale, NO symlog cap — so PopArt's per-channel
# (G_k - mu_k)/sigma_k normalises them exactly like the quality channels.
#   * ALPHAGRAD_ADDITIVE_SYMLOG_COST=0 disables the whole
#     symlog(lambda_inner*raw)+cap pass in ppo_ray_worker (the pass is gated
#     behind this flag, and lambda_inner is ONLY applied inside that pass, so
#     turning it off ALSO stops lambda_inner from being applied).
#   * ALPHAGRAD_COST_SYMLOG_CAP=0 removes the +/-1.25 latency/peak_mem cap.
# CAVEAT: raw cost -> PopArt sigma is OUTLIER-SENSITIVE (a single 100x-latency
# outlier inflates the per-channel sigma EMA and can transiently shrink the
# cost advantage). Mitigated by the SLOW PopArt beta (quasi-static EMA, so one
# outlier barely moves sigma) and the SMALL outer cost weights
# (lambda_cmp/lambda_mem=0.06) that keep the cost term a minor nudge regardless
# of scale. If sigma proves too jumpy, options are: raise POPART_SIGMA_MIN for
# the cost channels, or a soft (percentile) cost clip — NOT the symlog cap.
export ALPHAGRAD_ADDITIVE_SYMLOG_COST=0   # was 1 — cost now RAW into PopArt (no symlog / no lambda_inner)
# INNER_LAMBDA_* kept exported for documentation, but INERT while symlog is
# off (only applied inside the symlog pass). Harmless if left set.
export ALPHAGRAD_INNER_LAMBDA_LATENCY_NS="${ALPHAGRAD_INNER_LAMBDA_LATENCY_NS:-9e-6}"   # INERT (symlog off)
export ALPHAGRAD_INNER_LAMBDA_PEAK_MEMORY="${ALPHAGRAD_INNER_LAMBDA_PEAK_MEMORY:-7.7e-9}" # INERT (symlog off)
export ALPHAGRAD_COST_SYMLOG_CAP=0   # was 1.25 — RETIRED (no cost cap; PopArt handles scale)
# >>> PURE-POPART ADVANTAGE (bridge-cse, USER-DIRECTED) <<<
# The advantage is EXACTLY the per-channel PopArt-normalised residual
# collapsed by the weights, sum_k (A_k/sigma_k)*w_k — NO extra rollout z-score,
# NO ADV_STD_FLOOR, NO ADV_CLIP, NO failed-row stamp (double normalisation).
# KL early-stop (PPO_TARGET_KL below) is the blow-up guard instead of the clip.
export ALPHAGRAD_POPART_PURE_ADV="${ALPHAGRAD_POPART_PURE_ADV:-1}"
# RETIRED exports (bridge-cse): ALPHAGRAD_FAILED_ADV_STAMP, ALPHAGRAD_ADV_STD_FLOOR,
# ALPHAGRAD_ADV_CLIP, ALPHAGRAD_FAILED_PENALTY — all inert under neutral-mu +
# pure-PopArt advantage (a failed row is already A==0). No longer exported.
# Unidirectional Palimpsa (gated linear-attention, O(seq)) policy backbone —
# efficient over the now-unclipped ~8.2k-token grad graph (MAX_TOKENS=16384).
# V2: the uni mixer now carries the zero-init eqn_ids relational DAG-degree
# forget-gate modulation (same construction as palimpsa_bi).
export ALPHAGRAD_POLICY=palimpsa

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

# >>> ViT MEASURABILITY FIX STACK (bridge-cse) <<<
# (1) POW2 ViT internal dims: pad the token sequence 17->32 so every internal
#     weight tensor is a power of two — kills the non-pow2 ``slice
#     limit_indices`` graphax shape-storm (166 dim-17 -> 2 in the jaxpr). MNIST
#     input(784)/output(10) untouched. Revert with ALPHAGRAD_VIT_POW2=0.
export ALPHAGRAD_VIT_POW2="${ALPHAGRAD_VIT_POW2:-1}"
# (2) peak_memory reward reads the ABSOLUTE device high-water mark (baseline+peak)
#     rather than the near-zero entry-delta of a cached executable, so the
#     channel carries a real non-zero GPU peak. Revert with =0.
export ALPHAGRAD_PEAK_MEMORY_ABSOLUTE="${ALPHAGRAD_PEAK_MEMORY_ABSOLUTE:-1}"
# (3) B_kstep probe is now dynamic over the 18-arg ViT (no env flag needed).

# >>> UN-SCALED COSSIM (user request, bridge-cse) <<<
# The cosine_sim channel now enters the reward in its REGULAR RAW RANGE,
# UNSCALARIZED except for PopArt's per-channel normalisation:
#   (a) WEIGHT = 1.0  (LAMBDA_COSSIM_GUIDE 1.5 -> 1.0): the ×1.5 boost is gone,
#       so cosine_sim carries the same natural weight as the bkstep quality
#       channel instead of an artificial 1.5× guide weight.
#   (b) GUIDE CAP DISABLED (ALPHAGRAD_COSSIM_GUIDE_CAP 0.1 -> 0): the old
#       min(cossim, 0.1) clamp is removed, so the FULL cosine range reaches the
#       reward/GAE buffer. ALPHAGRAD_COSSIM_GUIDE_CAP=0 (any <=0) means "no cap"
#       (a strictly-positive value would re-install a cap). PopArt then
#       normalises the raw cosine_sim channel like every other channel.
# Net: cosine_sim reward = raw cosine value (full range), weight 1.0, PopArt-
# normalised. (bkstep is already raw + weight 1.0 (lambda_acc) with no cap —
# unchanged.)
export ALPHAGRAD_COSSIM_GUIDE_CAP="${ALPHAGRAD_COSSIM_GUIDE_CAP:-0}"
LAMBDA_COSSIM_GUIDE="${LAMBDA_COSSIM_GUIDE:-1.0}"

PPO=alphagrad/src/alphagrad/approx/ppo_ray.py

EPISODES="${EPISODES:-1000}"
WANDB="${WANDB:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-dsnn-jac-gpu}"
CAMP="${CAMP:-campaign_vit_v2}"
RAY_PORT="${RAY_PORT:-6384}"
SEED="${SEED:-2000}"
NUM_ENVS="${NUM_ENVS:-4}"
MBS="${MBS:-8}"
LAMBDA_ACC="${LAMBDA_ACC:-1.0}"
# w_outer for the cost channels. lambda is now INSIDE the symlog (per-channel
# ALPHAGRAD_INNER_LAMBDA_* above); this OUTER weight just scales the whole
# symlog'd term. w_outer=0.06 -> typical combined cost term ~0.08 (symlog(~1)~
# 0.69 per channel), a minor nudge vs bkstep(~0.5); 2x-cheaper vs typical moves
# reward by ~0.035 (was ~0.0014 with the old lambda*symlog(raw) at 0.001).
LAMBDA_CMP="${LAMBDA_CMP:-0.06}"   # w_outer (latency); lambda_inner is 1/typical_raw
LAMBDA_MEM="${LAMBDA_MEM:-0.06}"   # w_outer (peak_memory)
# ENTROPY-COEF ANNEAL DISABLED (user request, bridge-cse). The linear decay
# init->final (commit 2701cce, wired in ppo_ray_worker.run_rollout_and_train)
# is turned OFF by setting FINAL == INIT: the per-episode anneal formula
# ``init + (final-init)*progress`` is then CONSTANT for all episodes. Entropy
# coefficient is held FIXED at 0.05 (no decay). To re-enable annealing, set
# ENTROPY_COEF_FINAL to a value below ENTROPY_COEF.
ENTROPY_COEF="${ENTROPY_COEF:-0.05}"
ENTROPY_COEF_FINAL="${ENTROPY_COEF_FINAL:-0.05}"  # == ENTROPY_COEF -> constant (anneal disabled)

ULIM='ulimit -n $(ulimit -Hn) 2>/dev/null || ulimit -n 262144 2>/dev/null; ulimit -u $(ulimit -Hu) 2>/dev/null || true;'
eval "$ULIM"

NODE0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 1p)
HEAD_IP=$(srun --nodes=1 --nodelist=$NODE0 hostname -i | awk '{print $1}')
# PopArt value normalisation is ALWAYS ON (per-channel EMA normalisation
# of the value targets). Knobs: ALPHAGRAD_POPART_BETA,
# ALPHAGRAD_POPART_SIGMA_MIN.
MAX_SUBSTEPS="${MAX_SUBSTEPS:-16}"
# RETIRED (bridge-cse): the scale-only advantage bound + hard clip
# (ALPHAGRAD_ADV_STD_FLOOR / ALPHAGRAD_ADV_CLIP) were the double-normalisation
# on top of PopArt. Dropped — the advantage is now pure PopArt (see
# ALPHAGRAD_POPART_PURE_ADV above); KL early-stop is the blow-up guard.
# Fix 4: per-channel PopArt sigma floor. Global floor 0.1 over-amplified the
# near-homogeneous bkstep(9)+cosine(6) channels (var->0 -> A_k/sigma_k inflated).
# Floor those quality channels higher (0.2); cost channels keep base 0.1.
export ALPHAGRAD_POPART_SIGMA_MIN="${ALPHAGRAD_POPART_SIGMA_MIN:-0.1}"
export ALPHAGRAD_POPART_SIGMA_MIN_QUALITY="${ALPHAGRAD_POPART_SIGMA_MIN_QUALITY:-0.2}"
# Fix 2: PPO KL early-stop (reject a catastrophic ep49-type update). 0 disables.
export ALPHAGRAD_PPO_TARGET_KL="${ALPHAGRAD_PPO_TARGET_KL:-0.15}"
# PER-COMPONENT KL (bridge-cse): the early-stop checks the MEAN-per-active-
# component KL (joint KL / n_active_components), so target_kl=0.15 is a proper
# single-action PPO threshold instead of a joint-sum one (which tripped after
# ~1 minibatch). Both joint + per-component KL are logged. 0 = legacy joint.
export ALPHAGRAD_KL_PER_COMPONENT="${ALPHAGRAD_KL_PER_COMPONENT:-1}"
# ROBUST PopArt sigma (bridge-cse): winsorize value targets to
# median +/- k*(1.4826*MAD) per channel before the EMA so a raw-cost outlier
# can't spike sigma. sigma_max raised to admit the raw cost scale (~1e9).
export ALPHAGRAD_POPART_ROBUST_STD="${ALPHAGRAD_POPART_ROBUST_STD:-1}"
export ALPHAGRAD_POPART_WINSOR_K="${ALPHAGRAD_POPART_WINSOR_K:-5.0}"
export ALPHAGRAD_POPART_SIGMA_MAX="${ALPHAGRAD_POPART_SIGMA_MAX:-1e12}"
NAME="vit_v2${VARIANT:+_$VARIANT}_s${SEED}_ss${MAX_SUBSTEPS}"

echo "########## FULL_NN256_V2 $(date) | head=$NODE0 eps=$EPISODES NN_HIDDEN=256 variant=${VARIANT:-full} measure-grad ##########"

( cd ~/dsnn && uv run --no-sync ray start --head --node-ip-address=$HEAD_IP \
    --port=$RAY_PORT --num-gpus=${NUM_GPUS:-4} --num-cpus=${RAY_NUM_CPUS:-64} --block ) &
disown
sleep 25
echo "==== Ray up (4 GPU $NODE0) $(date) ===="

OUT=$CAMP/vit
LOG=$CAMP/logs/vit.log
rm -rf "$OUT"; mkdir -p "$OUT" "$CAMP/logs"

uv run --no-sync $PPO --name $NAME --variant ${VARIANT:-full} --seed $SEED \
  --example ${EXAMPLE:-VmappedViT} --dataset mnist \
  --rewards cmp mem acc --cmp-type latency --mem-type peak_memory \
  --measure-grad $([ "${EXEC_ON_GPU:-1}" = 1 ] && echo --exec-on-gpu) --measure-latency --latency-inner-reps 50 \
  --lambda-cmp $LAMBDA_CMP --lambda-mem $LAMBDA_MEM --lambda-acc $LAMBDA_ACC --lambda-frob 0.0 \
  --lambda-cossim-guide $LAMBDA_COSSIM_GUIDE \
  --entropy-coef $ENTROPY_COEF --entropy-coef-final $ENTROPY_COEF_FINAL \
  --dynamic-substeps --max-substeps ${MAX_SUBSTEPS} \
  --actor-num-gpus 1 --cpu-actor-num-gpus 1 --num-cpu-workers ${NUM_CPU_WORKERS:-3} \
  --cpu-cores-per-actor 8 --cpu-cores-shared \
  --cpu-callback-timeout 1800 --cpu-callback-initial-timeout 1800 \
  --cpu-worker-recycle-every ${CPU_WORKER_RECYCLE_EVERY:-6} \
  --ray-address $HEAD_IP:$RAY_PORT \
  --advantage-norm scalar --ppo-epochs 4 \
  --episodes $EPISODES --num-envs $NUM_ENVS --minibatches $MBS \
  --num-data-points 5 --reps-per-point 2 \
  --best-sequences-json $OUT/best.json --best-sequences-every 5 \
  --calibrate-steps 0 --wandb $WANDB --wandb-project $WANDB_PROJECT > "$LOG" 2>&1
RC=$?
echo "EXIT=$RC seed=$SEED" >> "$LOG"
[ "$RC" -eq 0 ] && [ -s "$OUT/best.json" ] && touch "$OUT/.done"
echo "########## FULL_NN256_V2 DONE (rc=$RC) $(date) ##########"
( cd ~/dsnn && uv run --no-sync ray stop ) 2>/dev/null || true
