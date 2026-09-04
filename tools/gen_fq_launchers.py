#!/usr/bin/env python3
"""THE ONE SKELETON every fq_*.sbatch in this campaign is generated from.

Launchers drift.  The v57..v66 series drifted into "the v66 arms differ from
v65 in FIVE simultaneous ways, not one" (docs/run_analysis/params_matrix.md
C7) and fq_v58_tlm_env16.sbatch was overwritten in place while its job was
pending, so job 61494's command line is unrecoverable (C8).  Both failures are
edit-a-file-by-hand failures.  Here the SHARED stack exists exactly once, each
arm is a dict of DIFFERENCES from it, and regenerating is the only supported
way to change a launcher.

    python3 tools/gen_fq_launchers.py --out ~/dsnn            # write
    python3 tools/gen_fq_launchers.py --out ~/dsnn --check    # diff only

Every emitted file is `bash -n` checked before it is written (a previous bulk
edit silently uncommented ~40 launchers; syntax is verified, never eyeballed).

The plan these launchers execute, with the registered predictions and the
decision table, is docs/EXPERIMENT_PLAN.md.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# THE SETTLED REWARD CONFIGURATION (owner decision 2026-08-28, not re-litigated
# here).  Three TRAINED channels: real measured latency, real measured peak
# memory, and the gradient cosine at init with K=1 (949f1af: 0.874 Pearson /
# 0.805 Spearman / 0.003 s per plan -- the most predictive AND the cheapest
# variant measured).  Everything else is LOGGED.
#
# ALPHAGRAD_QUALITY_METRIC=auto still resolves to loss_drop (env.py:3516), so
# every arm names grad_cosine EXPLICITLY.  A launcher that forgets is a
# launcher that silently ran the 200x-dearer channel.
# ---------------------------------------------------------------------------

REPO = "/Users/assmuth/dsnn/alphagrad"
HOME_DSNN = "/Users/assmuth/dsnn"

# ---------------------------------------------------------------------------
# THE LANDSCAPE MEASUREMENT ARMS import from the LIVE trees, not from the
# .ag_pin_landscape / .gx_pin_landscape snapshot directories the hand-written
# fq_face_attrib.sbatch used.  Three measured reasons, 2026-08-30:
#
#  1. THE PIN CANNOT DO grad_cosine.  The settled quality channel is
#     grad_cosine, but the pinned alphagrad (c2b8104, 2026-08-27) predates it:
#     its env.quality_metric() has no grad_cosine branch at all and its
#     landscape_map.py offers choices=[loss_drop, cosine, none].  A
#     `--quality-metric grad_cosine` phase against the pin argparse-errors,
#     and if it did not it would raise in quality_metric().
#  2. THE PIN DIRECTORY IS MUTABLE and has already been overwritten in place:
#     .ag_pin_landscape's landscape_map.py was replaced on 2026-08-27
#     03:00:32, four minutes after the PA-PH job that used it finished.  A
#     directory name is not a provenance statement; a git sha is.
#  3. THE INSTRUMENT IS NOW COMMITTED.  landscape_map.py's face-inventory and
#     singleton-sweep work lived uncommitted for two days -- which is why the
#     launcher had to point at a snapshot at all.  It is committed now, so the
#     repo path IS the reproducible instrument and the hack is unnecessary.
#
# The PA-PH rows under run_analysis/landscape stay reproducible from their own
# recipe -- alphagrad c2b8104 for the library, landscape_map.py from 907c231
# (sha256 cb7c7667bcbd588b, the blob those rows are stamped with), graphax
# 4ea0bf8 -- which is recorded in UNBIASED_PARETO_AND_MEASUREMENT.md.  They are
# loss_drop rows and are NOT comparable with anything this arm produces.
# ---------------------------------------------------------------------------
LANDSCAPE_TOOL = f"{REPO}/src/alphagrad/approx/tools/landscape_map.py"

# Flags fq_face_attrib passes to landscape_map.py.  Grepped against THE TOOL
# THAT IS ACTUALLY INVOKED, not against ppo.py: this launcher lived outside
# git for two days passing --face-inventory / --singleton-skip-sweep /
# --sweep-stride, which at the time NO COMMITTED landscape_map.py defined.
# That is the 43-line-stale-launcher failure mode, and this list is the guard.
LANDSCAPE_FLAGS = [
    "--example", "--dataset", "--hidden-dim", "--vocab-size", "--num-layers",
    "--seed", "--exec-on-gpu", "--cmp-type", "--mem-type",
    "--num-data-points", "--reps-per-point", "--quality-metric",
    "--walk-steps", "--out-dir", "--latency-inner-reps", "--reps",
    "--warmup-trials", "--ladder", "--no-all-rung", "--ops", "--no-skip-plan",
    "--archive", "--archive-target", "--archive-tol", "--archive-all-points",
    "--archive-max-measure", "--face-inventory", "--inventory-only",
    "--singleton-skip-sweep", "--sweep-stride", "--noise-floor-reps",
    "--report-only", "--tag", "--config-note", "--max-seconds",
]
# wandb, as THREE constants rather than one opaque string, so the
# pre-flight below and the emitted command line are provably the same
# entity/project.  A wave arm that logs to the wrong entity is invisible
# on the team dashboard and is indistinguishable, from the owner's side,
# from "wandb is not syncing at all".
WANDB_MODE = "online"
WANDB_ENTITY = "dll-streetview"
WANDB_PROJECT = "dsnn-vertex"
WANDB = (f"--wandb {WANDB_MODE} --wandb-entity {WANDB_ENTITY}"
         f" --wandb-project {WANDB_PROJECT}")

# Flags whose ABSENCE from ppo.py must abort the job before any setup noise.
# The R1-R4 battery shipped with this guard and it caught three missing flags.
REQUIRED_FLAGS = [
    "--quality-metric",
    "--symlog-channels",
    "--init-scheme",
    "--face-read",
    "--discount",
    "--gae-lambda",
    "--walk-rotate",
    "--sparsity-log",
    "--cos-log-every",
    "--pareto-dump-every",
    "--latency-inner-reps",
    "--kl-ref-weight",
    "--exact",
    "--var-probe",
    "--face-edge-mem",
    "--per-face-masks",
    "--plan-log",
]

class _Delete:
    """Sentinel: an arm sets a key to _DELETE to REMOVE it, in `cli` or `env`.

    Deleting an inherited env var is not cosmetic.  SHARED_ENV is the
    TRAINING stack; a measurement arm that silently inherits
    ALPHAGRAD_QUALITY_GATE_MIN=0.05 has its cost channels FLOORED at the
    exact-reverse reference for every plan scoring below 0.05 quality --
    which for a SKIP sweep is precisely the plans being measured.
    """


_DELETE = _Delete()


# ---------------------------------------------------------------------------
# SHARED ENV.  Byte-identical across every training arm -- a controlled
# variable, not a per-run parameter.  Arms override by key in `env`.
# ---------------------------------------------------------------------------

SHARED_ENV = [
    ("RAY_TMPDIR", "/tmp/ray_$SLURM_JOB_ID"),
    ("PATH", '"$HOME/.local/bin:$PATH"'),
    ("PYTHONPATH", '"$HOME/dsnn/graphax/src:$HOME/dsnn/alphagrad/src"'),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
    ("XLA_FLAGS",
     '"--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"'),
    # --- graphax: PLANNER path for exact lowering + L5 demand-emit (v42-proven)
    ("GRAPHAX_PLANNER_EXACT", "1"),
    ("GRAPHAX_DEMAND_EMIT", "1"),
    # PULLUP, not pulldown.  Every TLM arm before R1 ran PULLDOWN=1, so
    # bf16-native compute was the thing paying for the approximation; at
    # pullup the cost axis is the approximation ITSELF.
    ("GRAPHAX_QUANT_PULLDOWN", "0"),
    # --- TLM target shape (v42 baseline).  The target's size comes from THESE,
    #     not from --hidden-dim/--vocab-size/--num-layers, which size the POLICY.
    ("ALPHAGRAD_TLM_SEQ", "32"),
    ("ALPHAGRAD_TLM_DMODEL", "128"),
    ("ALPHAGRAD_TLM_VOCAB", "1024"),
    # --- face width: probed derived_max_faces=2538 for TLM under grad.
    #     MUST be an env var: cpu_approx_worker value-binds MAX_FACES at import
    #     and the trainer only derives it after ray.init.
    ("ALPHAGRAD_MAX_FACES", "2538"),
    ("ALPHAGRAD_MAX_DELTA_TOKENS", "32768"),
    ("ALPHAGRAD_MAX_EQNS", "512"),
    # G4 2026-08-28: graphax's working tree accepts the res-slot two-op form
    # (tests/misc/test_face_two_op_form.py, committed e5fd46c) and =1 hooks
    # BOTH addends.  Under =1 on a graphax that does NOT accept it, every plan
    # putting a rule in the res/new slot dies in _trace_truncate SILENTLY --
    # UNBIASED_PARETO_AND_MEASUREMENT.md sec 7(c).  The pre-flight VERIFIES it.
    ("ALPHAGRAD_NEW_SLOT_JOIN", "1"),
    ("ALPHAGRAD_POLICY", "palimpsa"),
    # Vertex choice pinned to reverse elimination via legality, so ONLY the
    # approximations are learned.  Read at IMPORT time in
    # common/masks.py:148 -- it must be exported before python starts.
    ("ALPHAGRAD_FORCE_REV_ORDER", "1"),
    # The additive quality gate.  Plans below qmin pay the exact-rev reference
    # cost, so the SKIP cliff earns nothing.
    ("ALPHAGRAD_QUALITY_GATE_MIN", "0.05"),
    # K=1 is both the most predictive and the cheapest (949f1af); K>1 is
    # strictly worse.  Named here so no arm inherits a stale export.
    ("ALPHAGRAD_GRAD_COSINE_K", "1"),
    ("ALPHAGRAD_ACTOR_PROF_EVERY", "20"),
    ("ALPHAGRAD_INCREMENTAL_TOKENS", "1"),
    ("ALPHAGRAD_DEBUG_APPROX_PROB", "1"),
    ("ALPHAGRAD_CLEAR_JIT_CACHES_EVERY", "0"),
    ("ALPHAGRAD_DEBUG_MEM", "1"),
    ("ALPHAGRAD_SKIP_COST_ANALYSIS", "1"),
    ("ALPHAGRAD_DEBUG_MEASURE", "1"),
    ("ALPHAGRAD_DEBUG_DEGEN", "1"),
    ("ALPHAGRAD_PROFILE", "1"),
    ("ALPHAGRAD_EXTEND_CHUNK", "128"),
    ("ALPHAGRAD_EXTEND_UNROLL", "32"),
    ("ALPHAGRAD_MULS_SENTINEL_CAP", "5e12"),
    # Project memory: ALWAYS skip the count pass (77% of host time) and use the
    # spec-native direct measurement.
    ("ALPHAGRAD_SKIP_COUNT_OPS", "1"),
    ("ALPHAGRAD_DIRECT_MEASURE", "1"),
    # --- hostperf stack
    ("ALPHAGRAD_FACE_ENUM_CACHE", "1"),
    ("ALPHAGRAD_UNIFIED_FACE_ENUM", "1"),
    ("ALPHAGRAD_BATCHED_CALLBACK", "1"),
    ("JAX_COMPILATION_CACHE_DIR", "$HOME/dsnn/.jax_compile_cache"),
    ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2"),
]

# ---------------------------------------------------------------------------
# SHARED CLI.  Ordered list of (group-comment, [tokens]).  Arms override by
# flag name in `cli` (None deletes the flag).
# ---------------------------------------------------------------------------

SHARED_CLI = [
    ("--example", "TransformerLM"),
    ("--dataset", "wikitext2"),
    ("--exec-on-gpu", None),          # None value == bare store_true flag
    ("--seed", "250197"),
    # THE MEASUREMENT PROTOCOL.  50 inner reps is the standing project rule;
    # at 5 the whole scale moves (identity reads 169038 ns vs 140090, and the
    # archived winner's ratio moves 0.5280 -> 0.6012 on the rep count alone).
    ("--measure-latency", None),
    ("--latency-inner-reps", "50"),
    ("--num-data-points", "5"),
    ("--reps-per-point", "4"),
    ("--incremental-encode", None),
    # REAL measured latency and REAL measured peak memory. Non-negotiable.
    ("--cmp-type", "latency"),
    ("--mem-type", "peak_memory"),
    ("--terminal-rewards-only", None),
    ("--rewards", "cmp mem acc"),
    ("--reward-mode", "additive"),
    # symlog on the COST channels only, never on quality.  Slots 6/8/9/10 are
    # structurally symlog-exempt anyway (NO_SYMLOG_REWARD_INDICES).
    ("--symlog-channels", "cost"),
    ("--lambda-cmp", "1"),
    ("--lambda-mem", "1"),
    ("--lambda-acc", "170"),
    ("--advantage-norm", "none"),
    # THE CREDIT HORIZON.  ppo.py defaults are 0.99/0.95; at 95 eliminations
    # (gamma*lambda)^95 = 0.0079, i.e. the terminal reward reaches the first
    # decision at 0.8 percent strength.  R3 restored the defaults and drifted
    # to destruction at ep131 while R2 at 1.0/1.0 held.  MANDATORY, and named
    # explicitly because no campaign run v57-v66 ever set them.
    ("--discount", "1.0"),
    ("--gae-lambda", "1.0"),
    # ("--reject-frozen-grads", the gradient-coverage guard, sat here from
    # wave 1 until 2026-09-03; removed by owner ruling 2026-09-03, ticket dsnn-3qm.15.)
    # THE TRAINED QUALITY CHANNEL.  auto still means loss_drop; name it.
    ("--quality-metric", "grad_cosine"),
    # Sampling variance in the quality signal is WANTED.  --walk-rotate is
    # named for the loss-drop walk but env._walk_seed is SHARED, so it rotates
    # the grad-cosine probe batch too: without it grad_cosine scores every
    # plan of every episode on one frozen batch.
    ("--walk-rotate", None),
    ("--init-scheme", "classic"),
    ("--face-read", "last-row"),
    ("--num-envs", "16"),
    ("--minibatches", "4"),
    ("--grad-window", "0"),
    # One PPO epoch makes the importance ratio identically 1 for the whole
    # update (R2's choice).
    ("--ppo-epochs", "1"),
    ("--entropy-weight", "0"),
    ("--face-entropy-weight", "0"),
    ("--face-entropy-floor", "0"),
    ("--face-logit-clamp", "0"),
    ("--set-pointer", None),
    ("--face-actions", None),
    ("--unified-face-head", None),
    ("--live-faces", None),
    ("--hidden-dim", "256"),
    ("--vocab-size", "512"),
    ("--num-layers", "3"),
    ("--ray-measure", "3"),
    ("--ray-measure-timeout", "600"),
    # LOGGED, NOT TRAINED: sparsity (weight 0) and the legacy Jacobian cosine
    # (subsampled).  Clipped relative Frobenius rides slot 8 automatically
    # because grad_cosine materialises the exact reference it needs.
    ("--sparsity-log", None),
    ("--cos-log-every", "20"),
    ("--pareto-dump-every", "10"),
    # A6.  --pareto-dump-every persists the FRONT; this persists EVERYTHING,
    # one append-only JSONL per run holding every terminal plan with its
    # replayable wire, all 11 reward slots and the per-kind
    # requested/applied/idempotent counts.
    # X3 is an analysis OF THE LOSERS and they were previously discarded at
    # the end of every episode.  It is pure logging -- no reward, no action,
    # no device work, no extra exact reference -- so it rides every arm.
    # "auto" = <wandb-run-dir>/plan_log_<name>.jsonl.
    ("--plan-log", "auto"),
    ("--episodes", "250"),
]

PREAMBLE = r"""export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
cd {repo}
"""

# ---------------------------------------------------------------------------
# THE ARMS.
# ---------------------------------------------------------------------------

ARMS: list[dict] = []


def arm(**kw):
    ARMS.append(kw)


# ===========================  WAVE 0  =======================================

arm(
    name="w0_cpu_gates",
    job="w0-cpu-gates",
    kind="cpu",
    node="pgi15-cpu1",
    time="2:00:00",
    purpose="""WAVE 0 / CPU GATES -- the cheap pre-flight for the whole campaign.

Runs, in order: the ratio gates (sampling log-prob == replay log-prob, i.e.
the PPO importance ratio is exactly 1 at epoch 0 -- a red gate here means the
gradients of every arm using that feature were taken against a wrong ratio; a
past instance of this bug ran the ratio to 2.3e23 invisibly under a
batch-averaged KL), then the standard smoke on NeuralNetwork, then the
POOLED-MEASUREMENT LIVENESS GATE.

Helmholtz is DELIBERATELY NOT SMOKED: landscape_map/_callback on Helmholtz
carries a pre-existing TypeError from the scalar-loss retarget (744fc3d),
unrelated to anything this campaign changes.

GATE 3 IS THE NEWEST AND IT PINS WHAT THE OTHER TWO CANNOT: that a
--ray-measure run MEASURES SOMETHING.  Neither of the first two starts a
measure pool -- the smoke's canonical config has no --ray-measure at all --
and 87cdc49 is the proof that this matters: 4c4d872 made
CpuApproxPool.evaluate forward an `episode` kwarg that
CpuApproximationActor.evaluate did not accept, every pooled dispatch died with
TypeError, the pool sentinelled the row and killed the actor, and under
--ray-measure NOTHING WAS MEASURED for 19 hours -- while the run still exited
0, still printed health rows and still stepped PPO.  The only trace was
[SENTINEL] lines no gate reads.  EVERY wave-1..4 arm here runs --ray-measure
3, so that is 177 node-hours of pure sentinel one launch away.

COST: ~2 CPU-h + ~7 min for the liveness gate.  Blackwell node-hours: ZERO.""",
    prediction="""ratio_gates 6/6 ok with NO skips (a skip is a failure here:
a gate that did not run pins nothing).  smoke NeuralNetwork rc=0 with every
post-warm-up [health ep..] row finite.  pool_liveness_gate.sh GREEN: contract
ok, 0 [SENTINEL] lines, every terminal plan measured inside a measure actor
and carrying a real cost vector.  Since 949f1af the first two passed; the
liveness gate is measured GREEN at 2ddbeef and measured RED with 87cdc49
reverted, so it is demonstrated to fail on the bug it targets.""",
    falsifier="Any gate red, or any gate SKIPPED, blocks every later wave.",
    body=r"""
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export JAX_COMPILATION_CACHE_DIR=$HOME/dsnn/.jax_compile_cache

echo "=== GATE 1/3: tools/ratio_gates.sh ==="
PY="uv run --no-sync python" tools/ratio_gates.sh
RG=$?
echo "ratio_gates rc=$RG"

echo "=== GATE 2/3: tools/smoke.sh NeuralNetwork ==="
SMOKE_OUT=$HOME/dsnn/run_analysis/w0_smoke uv run --no-sync tools/smoke.sh NeuralNetwork
SM=$?
echo "smoke rc=$SM"

# GATE 3/3.  Does --ray-measure measure anything?  rc 1 = MEASUREMENT DEAD,
# rc 2 = the gate could not run (harness misconfigured).  BOTH are failures --
# a gate that did not run pins nothing (116c540).
echo "=== GATE 3/3: tools/pool_liveness_gate.sh ==="
POOL_GATE_OUT=$HOME/dsnn/run_analysis/w0_pool_liveness \
RAY_TMPDIR=/tmp/ray_poolgate_$SLURM_JOB_ID \
PY="uv run --no-sync python" tools/pool_liveness_gate.sh
PL=$?
echo "pool_liveness rc=$PL"

if [ $RG -ne 0 ] || [ $SM -ne 0 ] || [ $PL -ne 0 ]; then
  echo "W0 CPU GATES RED (ratio_gates=$RG smoke=$SM pool_liveness=$PL)" \
       "-- do not launch wave 1"
  exit 1
fi
echo "W0 CPU GATES GREEN"
""",
)

arm(
    name="w0_probe",
    job="w0-probe",
    kind="probe",
    node="pgi15-gpu15",
    time="12:00:00",
    gpus=4,
    purpose="""WAVE 0 / SINGLE-NODE PROBE -- three verifications that would each
otherwise consume a training node, run sequentially on one.

  P1  RE-DERIVE THE CANDIDATE LIST AND THE BEST-FACE CLAIM.  The singleton
      SKIP sweep over every live face of the fixed-reverse TLM elimination,
      paired against its own exact reference.  This is X1's and X2's
      prerequisite.  The campaign's headline "k21/f1 is the best free single
      skip nobody found" (ratio 0.5325, quality 0.9257) came from a sweep
      under a DIFFERENT configuration, and A3 found that K21/F1 IS NOT A LIVE
      FACE IN THIS GRAPH -- it substituted k14/f0 (vertex 81).  NOTHING THAT
      NAMES k21/f1 MAY BE WRITTEN UP UNTIL THIS PHASE REPLACES IT.

  P2  DOES DIAG EVER APPLY ON A LIVE RUN?  DIAG applied 0 of 103 requested
      rules at every budget under both pulldown polarities: "every 'diag'
      result in the campaign is an identity plan wearing a diag label".  A1
      (690971a) could NOT reproduce the 0/103 -- but its harness draws
      uniformly over legal ops rather than from a collapsed trained policy, so
      the two numbers are not the same quantity and neither settles the other.
      This phase measures applied/requested per kind, on the production
      configuration, under (a) neither flag, (b) --per-face-masks, (c)
      --per-face-masks --diag-per-face, so the MINIMAL flag that moves DIAG
      off 0/103 is identified.  The owner keeps DIAG in the action space but
      its FACTOR IS NOT AN ACTION, so (c) is measured for attribution only and
      is NOT a candidate for any later wave.

  P3  RE-BASELINE AFTER 744fc3d.  The scalar-loss retarget landed after every
      quality number the campaign quotes.  LIF_SNN_SHD and ADALIF_SNN_SEQ
      moved (58->56 and 37->35 equations, two dead vertices dropped) and the
      seven analytic AD benchmarks went back to their full multi-output
      Jacobian target (Simple 7->4 eqns, Helmholtz 8->6, RoeFlux_1d 104->101,
      BlackScholes 416->414).  Any comparison against a pre-744fc3d number is
      invalid until this runs.  Note also that LIF_SNN / ADALIF_SNN had never
      run under --measure-grad at all (TypeError on their 7-leaf tuple), which
      744fc3d fixed.

COST: 1 node, <= 12 h.  4 GPUs are REQUESTED but ONE visible device is used,
because peak_memory is a DEVICE-WIDE counter and a co-resident process
inflates it -- CV was observed going 0.0000% -> 49.7% under a noisy
neighbour.  Holding the node is what makes the memory column mean anything.""",
    prediction="""REGISTERED BEFORE THE RUN.
  P1: the 118-live-face inventory reproduces; ~31 faces at quality > 0.9 and
      ratio < 0.99, of which ~6 are gradient-destroyers near 0.72, ~5 are
      genuine at 0.76-0.79 and ~26 are mild at 0.90-0.99; k21/f1 DOES NOT
      APPEAR among the live faces.
  P2: DIAG stays at 0/103 with neither flag and moves off it under
      --per-face-masks ALONE, i.e. the DIAG factor does not need to become an
      action for DIAG to apply.
  P3: the two SNN families and the seven analytic benchmarks move; every other
      family is equation-, vertex-, face- and value-identical.""",
    falsifier="""If P1 finds no face at quality > 0.9 whose ratio beats the
1.0007 +/- 0.0008 drift floor, the entire singleton-skip premise is dead and
X1/X2 are CANCELLED, not reinterpreted.
If P2 shows DIAG still 0/103 under BOTH flags, DIAG is structurally inert on
this target; it stays in the action space as a DOCUMENTED NO-OP (owner's
instruction) and no later wave may credit it with anything.""",
    body=r"""
# --- P1/P2/P3 all run as measure actors on ONE visible device.
export ALPHAGRAD_MEASURE_ACTOR=1
export ALPHAGRAD_MEASURE_WARMUP=1
OUT=$HOME/dsnn/run_analysis/w0
mkdir -p $OUT

echo "===================== P1: singleton skip sweep ====================="
# NOTE, AND DO NOT "FIX" IT: landscape_map's --quality-metric choices are
# loss_drop/cosine/none -- the tool cannot NAME grad_cosine.  On
# TransformerLM (a scalar-loss target) "cosine" is the DEPRECATED ALIAS that
# resolves to grad_cosine (env.py:3489), which is exactly the settled channel.
# It warns once and loudly.  Passing loss_drop here would measure a different
# channel from every training arm.
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
  src/alphagrad/approx/tools/landscape_map.py \
  --example TransformerLM --dataset wikitext2 \
  --hidden-dim 256 --vocab-size 512 --num-layers 3 --seed 250197 \
  --exec-on-gpu --cmp-type latency --mem-type peak_memory \
  --num-data-points 5 --reps-per-point 4 --latency-inner-reps 50 \
  --latency-warmup 2 --warmup-trials 2 \
  --quality-metric cosine \
  --face-inventory --singleton-skip-sweep --sweep-stride 1 \
  --reps 3 --noise-floor-reps 10 --noise-floor-plan identity \
  --out-dir $OUT --tag p1_singleton --max-seconds 18000
echo "P1 exited with $?"

echo "===================== P2: DIAG applied/requested ===================="
# Three modes, same ladder, same reps.  What is read is applied/requested per
# KIND, with idempotent no-ops removed from the denominator (~83% of DIAG's
# "failures" are correct no-ops, so the raw ratio understates it).
for MODE in none pfm pfm_diag; do
  echo "--- P2 mode=$MODE ---"
  case $MODE in
    none)     export ALPHAGRAD_PER_FACE_MASKS=0 ALPHAGRAD_DIAG_PER_FACE=0 ;;
    pfm)      export ALPHAGRAD_PER_FACE_MASKS=1 ALPHAGRAD_DIAG_PER_FACE=0 ;;
    pfm_diag) export ALPHAGRAD_PER_FACE_MASKS=1 ALPHAGRAD_DIAG_PER_FACE=1 ;;
  esac
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
    src/alphagrad/approx/tools/landscape_map.py \
    --example TransformerLM --dataset wikitext2 \
    --hidden-dim 256 --vocab-size 512 --num-layers 3 --seed 250197 \
    --exec-on-gpu --cmp-type latency --mem-type peak_memory \
    --num-data-points 5 --reps-per-point 4 --latency-inner-reps 50 \
    --quality-metric cosine \
    --ladder 1,5,15,50 --ops quant,diag,compress --reps 1 \
    --config-note "diagprobe:$MODE" \
    --out-dir $OUT --tag p2_diag_$MODE --max-seconds 3600
  echo "P2 mode=$MODE exited with $?"
done
unset ALPHAGRAD_PER_FACE_MASKS ALPHAGRAD_DIAG_PER_FACE

echo "===================== P3: post-744fc3d re-baseline =================="
for EX in LIF_SNN ADALIF_SNN ADALIF_SNN_SEQ LIF_SNN_SHD \
          Simple Lighthouse RobotArm_6DOF RoeFlux_1d BlackScholes_Jacobian; do
  echo "--- P3 example=$EX ---"
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
    src/alphagrad/approx/tools/landscape_map.py \
    --example $EX --dataset none --seed 250197 \
    --exec-on-gpu --cmp-type latency --mem-type peak_memory \
    --num-data-points 5 --reps-per-point 4 --latency-inner-reps 50 \
    --ladder 1,5 --ops quant,diag,compress --reps 3 \
    --noise-floor-reps 5 --noise-floor-plan identity \
    --out-dir $OUT --tag p3_rebase_$EX --max-seconds 1800
  echo "P3 $EX exited with $?"
done
echo "W0 PROBE COMPLETE"
""",
)


# The w0_x2_screen arm (the coverage-constrained frontier, run through
# tools/coverage_beam.py) was removed with the gradient-coverage guard on
# 2026-09-03 (owner ruling 2026-09-03, ticket dsnn-3qm.15).

# ===========================  WAVE 1  =======================================
#
# CONTRAST x PRICE.  The R battery bracketed the space and left exactly one
# hole:
#     R1  random init (B=0), no gate      -> destroyed absorber, no contrast
#     R2  identity init (B=6), gamma=1    -> identity FIXED POINT, no contrast
#     R3  identity init,       gamma<1    -> drifts to destruction (ep131)
# R2 held (none 0.998, quality median 0.8853) but its quality spread was
# 3e-06 and its latency spread 1.7 us across 16 plans: the advantage was about
# zero and there was nothing to learn from.  The campaign's own registered next
# step is "the R2 configuration plus a BOUNDED exploration pressure, made safe
# by the gradient-coverage guard so that exploring toward skips cannot be
# rewarded for freezing gradients".
#
# TWO knobs supply that, and both are in the backlog:
#
#  * ALPHAGRAD_FACE_NONE_BIAS -- ppo.py:4727 calls it "the contrast knob" in so
#    many words.  A5's CORRECTED arithmetic (the factory's printed
#    P(approx/face) ~ 3*e^-B is a PER-SLOT rate and each face carries 3 slots;
#    the true expectation on TLM is 118*3*3*e^-B, so the old reading was wrong
#    by 3x) puts B=6 -- where R2 ran -- at 2.6 approximations per plan, the
#    FLOOR of the useful band.  B=5 is 7.2, the centre.  B=4 is 19.5.
#    Every measured win in the whole campaign lives at 1-15 approximations per
#    plan: v57 ep41 101us/q0.885 at ONE approximation, v57 ep117 96us/q0.882 at
#    14, and the paired one-face SKIP cluster at ratio 0.578 / quality 0.9258.
#    Both learners then walked away from that region toward 155-2900
#    approximations per batch, where BOTH axes are worse.
#
#  * lambda_acc -- WITHOUT PopArt's per-channel sigma division (these arms all
#    run --advantage-norm none) the realized pull is lambda*sigma_q/sigma_cost
#    with the ep0 census sigma_q = 0.15852 raw, sigma_lat = 2.119 symlog,
#    sigma_mem = 3.158.  So lambda=10 buys 0.75:1 and lambda=16 buys 1.20:1 --
#    NOT 10:1 and 16:1.  Matching the pricing v64b actually realized under
#    PopArt needs lambda ~ 134 (latency) / 199 (memory).  R1-R3 ran
#    lambda_acc=16, i.e. quality UNDER-PRICED about 10x; that was safe only
#    because they had no contrast to spend.  Adding exploration at lambda=16 is
#    adding it with the brakes off.  The abort rule that registered
#    lambda in {130, 170, 210} was never executed.
# ---------------------------------------------------------------------------

W1_HEAD = """WAVE 1 -- CONTRAST x PRICE.  The settled reward configuration:
three TRAINED channels (real measured latency, real measured peak memory, and
the gradient cosine at init with K=1), terminal rewards only, gamma = GAE
lambda = 1, classic init, symlog on the cost channels only, and sparsity /
the legacy Jacobian cosine / the clipped relative Frobenius LOGGED but never
trained.  (Wave 1 ran with the gradient-coverage guard armed; the guard was
removed 2026-09-03 by owner ruling, ticket dsnn-3qm.15, and a regenerated
launcher no longer passes it.)"""

for _n, _bias, _lam, _node in [
    ("w1a_bias6_lam170", 6, 170, "pgi15-gpu15"),
    ("w1b_bias5_lam170", 5, 170, "pgi15-gpu16"),
    ("w1c_bias4_lam170", 4, 170, "pgi15-gpu17"),
    ("w1d_bias5_lam16", 5, 16, "pgi15-gpu18"),
]:
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="12:00:00",
        gpus=4,
        env={"ALPHAGRAD_FACE_NONE_BIAS": str(_bias)},
        cli={"--lambda-acc": str(_lam), "--name": _n.replace("_", "-")},
        purpose=W1_HEAD + f"""

THIS ARM: NONE-bias B={_bias} (about {{6:2.6, 5:7.2, 4:19.5}}[{_bias}]
approximations per plan), lambda_acc={_lam}.

COMPARISON PAIRS.  Each is a ONE-KNOB difference; do not read any other pair.
  w1a vs w1b vs w1c   the NONE-bias sweep at a fixed, correctly-priced lambda.
  w1b vs w1d          the price of quality at a fixed bias B=5.
w1a is additionally the bridge to R2: it is R2's configuration with exactly
two deliberate changes, the quality channel (loss_drop -> grad_cosine) and the
price (lambda 16 -> 170).  It is NOT a one-knob comparison against R2 and must
not be reported as one.

READ THESE PANELS, IN THIS ORDER:
  1. quality spread across the 16 envs per episode.  R2's was 3e-06.  Anything
     below 1e-3 means there is still nothing to learn from and every other
     panel is uninterpretable.
  2. grad_cov/rejected_this_ep against plan/n_live.  A run where they stay
     EQUAL is not learning, it is being refused -- env.py's own v16
     soft-sentinel note warns that degeneracy becomes the safe haven once the
     value baseline rises.
  3. entropy/approx_head DIVIDED BY faces/mean_valid.  The raw entropy metric
     averages a structural zero for every step whose vertex has no live face,
     so a fall that TRACKS a fall in faces/mean_valid is a graph-DESTRUCTION
     signal, not an entropy collapse (v64b's entropy fell 357x while
     mean_valid fell 91x; the quotient fell 4x).
  4. approx_prob/none, and the per-plan latency ratios against the
     1.0007 +/- 0.0008 drift floor.""",
        prediction=f"""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * B=6 (w1a) reproduces R2: the policy parks at identity, quality spread
    stays below 1e-3, approx_prob/none stays above 0.99.  A repriced lambda
    does not create contrast by itself.
  * B=5 (w1b) produces measurable contrast -- quality spread above 1e-3 AND at
    least one plan per episode outside the drift floor by ep50 -- and
    grad_cov/rejected_this_ep is NON-ZERO, i.e. the guard is actually catching
    the frozen-gradient plans exploration walks into.  This is the arm
    predicted to find the 1-15 approximation band where every measured win
    lives.
  * B=4 (w1c) over-approximates: more rules, no more speed.  This is exactly
    the effect X3 exists to explain -- over 64 archived points
    pearson(applied rules, latency ratio) = -0.12 (no relationship) while
    pearson(applied rules, quality) = -0.44, and 18 plans with >=20 faces
    average ratio 0.607 at quality 0.662 against 12 one-face plans at 0.598
    and 0.915.
  * lambda=16 (w1d) under-prices quality about 10x against w1b and drifts
    further toward destruction, with the COVERAGE GUARD rather than the reward
    doing the stopping.""",
        falsifier=f"""If NONE of w1a/w1b/w1c reaches a quality spread above
1e-3 by ep50, then contrast is NOT bias-limited, the NONE-bias hypothesis is
DEAD, and wave 3 is cancelled -- the campaign proceeds on the elimination
order axis (wave 2) alone.  Say that; do not re-run the sweep wider.
If w1d is indistinguishable from w1b on every panel, lambda is not a live knob
at this scale and the {{130, 170, 210}} relaunch is CANCELLED as answered
rather than completed.""",
    )

# ===========================  WAVE 2  =======================================

for _n, _rev, _extra_cli, _node, _what in [
    ("w2a_order_pinned", "1", {}, "pgi15-gpu15",
     "PINNED reverse -- the IN-BATTERY control.  It is re-run rather than "
     "compared against wave 1 because GPU state on this cluster moves 18-20% "
     "over a session and per-actor offsets are systematic at ~1.3%"),
    ("w2b_order_free", "0", {}, "pgi15-gpu16",
     "UNPINNED order + approximations -- both levers"),
    ("w2c_order_free_exact", "0", {"--exact": None}, "pgi15-gpu17",
     "UNPINNED order, EXACT only -- isolates the order axis from the "
     "approximation axis entirely.  THE DECISIVE ARM OF THIS WAVE"),
    ("w2d_order_free_edgemem", "0", {"--face-edge-mem": None}, "pgi15-gpu18",
     "UNPINNED order + the edge-keyed face memory, which is predicted INERT "
     "under rev-pinning and only becomes meaningful here"),
]:
    _cli = {"--name": _n.replace("_", "-"), "--episodes": "250"}
    _cli.update(_extra_cli)
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="24:00:00",
        gpus=4,
        env={"ALPHAGRAD_FORCE_REV_ORDER": _rev,
             "ALPHAGRAD_FACE_NONE_BIAS":
                 '${W1_BIAS:?export W1_BIAS to wave 1 winning NONE-bias}'},
        cli=_cli,
        depends="wave 1 (inherits its winning NONE-bias via $W1_BIAS)",
        purpose=f"""WAVE 2 -- THE ELIMINATION ORDER.  The axis with real range
(45-80x, against at most ~2x on the approximation axis) that has NEVER been
scheduled, because it was starved by the very credit horizon R2/R3 proved
causal.  Under ALPHAGRAD_FORCE_REV_ORDER=1 exactly one vertex is legal at
every step, so the pointer head has taken ZERO GRADIENT across the entire
v57-v66 campaign and R1-R3 (ve entropy -0.0e+00, max|dH/dlogits| = 0.0e+00,
every episode logging ve_head=0 macro_vertex=0).  Lifting the pin is the first
time that head is trained at all.

IT IS ALSO THE ONLY PLACE THE MEMORY CHANNEL IS ALIVE.  Under a pinned order
the memory ratio is 1.0000 +/- 0.0000 for every archived winner and every rung
of the hand-specified ladder; the ONLY plan that moves it is skip@all, at
0.0406 with quality 0.0000 -- i.e. destruction.  Across ORDERS the same
channel spans ~80x.  The owner's requirement that real measured peak memory be
a trained reward component is therefore satisfiable HERE AND NOWHERE ELSE on
this target.  Wave 1 trains it as a channel with no signal; wave 2 is where it
earns its slot.

THIS ARM: {_what}.

ALPHAGRAD_FORCE_REV_ORDER is read at IMPORT time (common/masks.py:148), so it
is exported in the env block and cannot be moved onto the CLI.

WHAT LIFTING THE PIN COSTS.  Face count 115 (rev) -> ~313 (random), ~2.7x the
face work; distinct edge keys 101 -> 206-286; the face stratum flips from
1 both-primitive / 114 one-intermediate / 0 both-intermediate to 164/311/463,
so both-intermediate becomes the MAJORITY -- precisely the stratum where the
chunk-mean read measures worst and where the edge-keyed memory was built to
help.  Random orders measure ~500x above rev.  ALPHAGRAD_MAX_FACES=2538 stays
a valid bound (derived_max_faces is order-independent).  And each distinct
order pays its OWN exact compile for the coverage guard (_EXACT_LEAF_NORMS is
keyed on the order): 8.9% of the first period under an order-searching policy
against 0.71% at steady state under a pinned one.  Hence 24 h, not 12.""",
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
w2c (exact, free order) is the decisive arm: if the pointer head can find
orders beating reverse at all, it shows up there with NO approximation
confound and with the memory channel actually moving.
  * w2c finds at least one plan whose PAIRED latency ratio against exact
    reverse is below 0.95, and moves peak memory by more than 2x.
  * w2b, which has both levers, does NO BETTER than the better of its two
    halves -- order and approximation do not compose freely, because the
    approximation wins are attached to specific faces of the reverse
    elimination and those faces do not survive reordering.
  * w2a reproduces its wave-1 counterpart within the drift floor.  If it does
    not, the wave-1 reading was drift and must be re-stated.""",
        falsifier="""If w2c never beats exact reverse outside the
1.0007 +/- 0.0008 drift floor over 250 episodes, then on TLM reverse is
optimal-or-unbeatable-by-this-policy, the 45-80x range is a property of BAD
orders rather than of reachable good ones, and THE ORDER AXIS IS CLOSED.  Say
so; do not reinterpret it and do not re-run it wider.
If w2d is indistinguishable from w2b, the edge-keyed memory is inert even off
the pin -- which is the only condition under which it was ever predicted to
help -- and the flag should be retired rather than carried further.""",
        owner_ruling=(
            "CLEAN_DESIGN_AUDIT.md row g2 marks --face-edge-mem 'would "
            "CONFLICT' on the same grounds as row g1's --face-endpoint-read: "
            "it widens the face head's input with memory rows, which the "
            "owner's design forbids. It has been inert so far, so the "
            "conflict has never bitten; off the rev pin it stops being "
            "inert. THIS ARM NEEDS AN OWNER RULING BEFORE LAUNCH. If the "
            "ruling is NO, run a second seed of w2b on this node instead -- "
            "the order axis is the one place a variance reading is worth a "
            "whole node."
            if _n.endswith("edgemem") else None),
    )

# ===========================  WAVE 3  =======================================
#
# CONDITIONAL on wave 1 finding contrast.  If w1a/w1b/w1c all park at identity
# this wave is CANCELLED, not rescoped.
# ---------------------------------------------------------------------------

for _n, _lam, _kl, _node, _what in [
    ("w3a_lam_star", "${W1_LAM:?export W1_LAM to wave 1 winning lambda-acc}",
     "0", "pgi15-gpu15", "the wave-1 winner, re-run as the IN-BATTERY control"),
    ("w3b_lam130", "130", "0", "pgi15-gpu16", "lambda_acc = 130"),
    ("w3c_lam400", "400", "0", "pgi15-gpu17", "lambda_acc = 400"),
    ("w3d_kl_ref", "${W1_LAM:?export W1_LAM to wave 1 winning lambda-acc}",
     "0.1", "pgi15-gpu18",
     "the wave-1 winner PLUS a KL trust region to the frozen identity-init "
     "policy"),
]:
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="12:00:00",
        gpus=4,
        env={"ALPHAGRAD_FACE_NONE_BIAS":
             '${W1_BIAS:?export W1_BIAS to wave 1 winning NONE-bias}'},
        cli={"--lambda-acc": _lam, "--kl-ref-weight": _kl,
             "--name": _n.replace("_", "-")},
        depends="wave 1 finding contrast (otherwise CANCELLED)",
        purpose=f"""WAVE 3 -- PRICE AND STABILITY.  Two registered items that
were both recorded and neither executed.

THE PRICE.  The abort rule attached to the v66 static battery said: if
violations climb and mean raw quality drops below 0.75 within ~20 episodes,
kill and relaunch at lambda in {{130, 170, 210}}.  It never fired and the arms
were read instead.  Separately, the design note calls the lambda sweep "the
single most consequential number in the design" and shows that a sweep of
{{1, 4, 16}} lands ENTIRELY inside the already-failed band -- lambda=1 buys
0.075:1, lambda=4 buys 0.30:1, lambda=16 buys 1.20:1 which is EXACTLY v66b,
which ended at quality +0.233 with 155 ops/plan and latency 12% WORSE than
exact.  The informative sweep is {{16, 130, 400}}.  Wave 1 already ran 16 and
170, so this wave completes it with 130 and 400 around the wave-1 winner.

THE STABILITY.  A5's KL-to-reference (4421056) penalises divergence from the
FROZEN identity-init policy on the face head plus its live-face context; the
vertex term cancels exactly and is structurally zero under the rev pin.  Its
own docstring is explicit about what it does and does not do: it bounds DRIFT,
it CANNOT create CONTRAST -- "the contrast knob is ALPHAGRAD_FACE_NONE_BIAS.
Sweep them together."  This wave is the second half of that sweep, run only
after wave 1 has established that there IS drift worth bounding.

THIS ARM: {_what}.

NOTE the interaction the design record flags: do NOT run a low lambda without
the quality gate.  Removing ALPHAGRAD_QUALITY_GATE_MIN uncaps the full SKIP
prize -- symlog(157/70) ~ +0.81 for one action against lambda_acc * 0.885 --
so SKIP wins outright at lambda=1, is marginal at 4 and is priced out at 16.
The gate stays at 0.05 in every arm here.""",
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * lambda=130 (w3b) and the wave-1 winner behave alike: both are inside the
    band that matches the pricing PopArt realized, so the outcome is flat in
    lambda across 130-210 and the exact value does not matter.
  * lambda=400 (w3c) over-prices quality: the policy holds at identity like
    w1a did, because any approximation is priced out.  This is the arm that
    establishes the UPPER edge of the usable band, which no run has ever
    located.
  * the KL arm (w3d) has the same endpoint as its control but a monotonically
    smoother path -- lower episode-to-episode variance in approx_prob/none,
    no absorber -- and does NOT improve the best plan found, because a trust
    region cannot create contrast it is not given.""",
        falsifier="""If w3b, w3c and w3a are indistinguishable, quality price
is NOT a live knob on this target across a 3x range and the lambda question is
CLOSED -- retire it from the backlog rather than sweeping wider.
If w3d finds a better plan than w3a, the KL claim in A5's own docstring is
wrong and the trust region is doing more than bounding drift; that would need
explaining before it is used.""",
    )

# ===========================  WAVE 4  =======================================

for _n, _read, _node, _what in [
    ("w4a_read_lastrow", "last-row", "pgi15-gpu15",
     "the INCUMBENT: what R1-R3 and every wave-1/2/3 arm ran"),
    ("w4b_read_chunkmean", "chunk-mean", "pgi15-gpu16",
     "the shipped DEFAULT, and the read the information-loss dossier indicts"),
    ("w4c_read_ownspan", "own-span-mean", "pgi15-gpu17",
     "the documented MINIMAL FIX: pool face f's OWN span only"),
    ("w4d_read_ownspan_seed2", "own-span-mean", "pgi15-gpu18",
     "w4c at a second seed -- the read-point effect sizes the dossier predicts "
     "are small enough that one seed cannot carry them"),
]:
    _cli = {"--face-read": _read, "--name": _n.replace("_", "-"),
            "--var-probe": None, "--var-probe-lr": "1e-3",
            "--var-probe-steps": "16"}
    if _n.endswith("seed2"):
        _cli["--seed"] = "970520"
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="12:00:00",
        gpus=4,
        env={"ALPHAGRAD_FACE_NONE_BIAS":
             '${W1_BIAS:?export W1_BIAS to wave 1 winning NONE-bias}'},
        cli=_cli,
        depends="waves 1-3 (inherits the winning bias and lambda)",
        purpose=f"""WAVE 4 -- THE FACE READ POINT.  --face-read is the CHEAPEST
UNRUN EXPERIMENT IN THE WHOLE BACKLOG: no extra encode, no extra scan, no new
parameters, no width change -- the pooled rows are already materialised and
only the mask moves.  All three modes are already pinned rollout == replay by
tests/face_read_point_test.py, so the PPO ratio is safe in every arm.

WHAT IT FIXES.  Face f's chunk is [approx-echo(f-1) || header+contraction(f)],
so chunk-mean produces the head's 94 logits from a token-count-weighted blend
of the PREVIOUS face's approximation with THIS face's contraction, and there is
no type channel to separate them -- the head cannot tell how much of its
pooled input came from the previous face.  own-span-mean masks the pooling at
the already-computed echo prefix so the head reads face f's OWN span;
last-row reads the recurrence's state after the whole chunk instead of a mean
over it.  The read point is already causally correct; only the readout is
wrong.

THIS ARM: {_what}.

NOT IN THIS WAVE, ON PURPOSE: --face-endpoint-read.  CLEAN_DESIGN_AUDIT.md row
g1 marks it CONFLICTS -- it makes the head input
[chunk_mean || vmem_slot_i || vmem_slot_j], routing the VERTEX memory into the
FACE head, "precisely what the owner forbids".  Omitting it costs no code.  It
needs an owner ruling, not a node.

All four arms carry --var-probe at --var-probe-steps 16, because the read-point
claim is a REPRESENTATION claim and probe/face/*/ndim_acc and size_r2 are how
it is adjudicated.  The full per-episode oracle replay was 210 s of a 246 s
host episode before the sampled-step flag existed; 16 steps is the affordable
form.""",
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
Offline, with a frozen random-init palimpsa and a ridge decoder, chunk-mean
reads 0.82/0.86/0.89 ndim and 0.45/0.48/0.55 size-R2 on lhs/rhs/res.  ONLINE
the same read rose to lhs ndim 0.64 by ~ep17 and then DECAYED to the 0.47
majority baseline by ~ep60 and stayed there, while the vertex probe held 1.00
throughout.  Prediction: own-span-mean beats chunk-mean on
probe/face/*/ndim_acc AT EP150 by at least 0.05, last-row lands between them,
and it is the DECAY that separates them -- the peak is not the metric.""",
        falsifier="""If all three reads decay to the majority baseline by ep60,
the read POINT is not the binding constraint: the collapse is upstream in the
shared palimpsa rows, and the next lever is row-collapse telemetry rather than
a fourth read variant.  WIDTH is already excluded as the fix -- lhs size-R2 is
pinned at ~0 at every width from E=32 to E=512, slope +0.021 per doubling, and
reaching 0.22 extrapolates to E ~ 1e5 -- so a negative here closes the cheap
options and leaves only the learned-query decoder at E=256, which is a build,
not a launch.""",
    )


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------

# ===========================  MEASUREMENT ARMS  =============================
# These read a landscape rather than train a policy: they invoke
# landscape_map.py directly, carry no wandb config, and their required-flag
# guard points at THE TOOL rather than at ppo.py.

_FACE_ATTRIB_BODY = r"""
# ---- PROVENANCE, RESOLVED AND RECORDED, NOT ASSUMED ----------------------
# The hand-written predecessor pinned PYTHONPATH at snapshot directories and
# verified the imports landed there.  This arm imports the live trees, so the
# equivalent guard is: say exactly which files were imported and exactly which
# commits they are, and refuse if a stale editable install is shadowing them.
uv run --no-sync python - <<'PYEOF'
import sys
import graphax, alphagrad
gx, ag = graphax.__file__, alphagrad.__file__
print("graphax.__file__  =", gx)
print("alphagrad.__file__=", ag)
ok = gx.startswith("@HOME_DSNN@/graphax/") and ag.startswith("@REPO@/")
if not ok:
    print("ABORT: imports did NOT resolve to the live working trees --")
    print("       an editable-install finder is beating PYTHONPATH.")
    sys.exit(70)
print("IMPORTS VERIFIED: both resolve to the live working trees.")
PYEOF
if [ $? -ne 0 ]; then
  echo "ABORT(70): import resolution failed -- refusing to produce numbers"
  echo "           whose provenance we cannot state. Nothing measured."
  exit 70
fi

# A DIRTY LIBRARY IS A PROVENANCE HOLE.  Not fatal (graphax is routinely
# mid-edit here) but it must be visible in the log beside the numbers, and
# the diffstat is recorded so the run can be reconstructed.
for R in @REPO@ @HOME_DSNN@/graphax; do
  D=$(git -C $R status --porcelain | wc -l)
  echo "TREE $R HEAD=$(git -C $R rev-parse --short HEAD) dirty=$D"
  if [ "$D" != "0" ]; then
    echo "  WARNING: $R has $D uncommitted change(s); these rows are NOT"
    echo "           reproducible from a sha alone.  Diffstat:"
    git -C $R diff --stat | sed 's/^/           /'
  fi
done

TOOL=@TOOL@
echo "TOOL $TOOL sha256=$(sha256sum $TOOL | cut -c1-16)"
COMMITTED=$(git -C @REPO@ rev-parse HEAD:src/alphagrad/approx/tools/landscape_map.py 2>/dev/null || echo none)
ONDISK=$(git -C @REPO@ hash-object "$TOOL" 2>/dev/null || echo none)
if [ "$COMMITTED" = "$ONDISK" ]; then
  echo "TOOL PROVENANCE: instrument IS the committed HEAD blob $COMMITTED"
else
  echo "TOOL PROVENANCE: WARNING -- instrument is NOT the committed HEAD blob"
  echo "                 on-disk=$ONDISK committed=$COMMITTED"
fi
AG_SHA=$(git -C @REPO@ rev-parse --short HEAD)
GX_SHA=$(git -C @HOME_DSNN@/graphax rev-parse --short HEAD)
NOTE="ag=$AG_SHA gx=$GX_SHA live tool=$(sha256sum $TOOL | cut -c1-8)"

# A NEW OUTPUT DIRECTORY, DELIBERATELY.  run_analysis/landscape holds the
# 2026-08-27 loss_drop rows measured on the pinned stack; these are
# grad_cosine rows on the live stack and the two must not share a --report-only
# glob.  (landscape_map keys its combined report on the quality metric as
# well, so pooling is prevented twice.)
OUT=@HOME_DSNN@/run_analysis/landscape_gradcos
mkdir -p $OUT
W=@REPO@/wandb
ARCH="--archive v57=$W/run-20260817_113827-it05ku34/files/pareto_front.json \
 --archive v60=$W/run-20260817_181647-ygm8n2jy/files/pareto_front.json \
 --archive v63=$W/run-20260822_165942-as9s5yrl/files/pareto_front.json \
 --archive v64b=$W/run-20260825_132646-38oyqf4g/files/pareto_front.json \
 --archive v65=$W/run-20260826_121153-8sht6x1m/files/pareto_front.json \
 --archive v66a=$W/run-20260826_121153-318ktrgq/files/pareto_front.json \
 --archive v66b=$W/run-20260826_121154-0olsxsjl/files/pareto_front.json \
 --archive v66c=$W/run-20260826_121154-s1537jdd/files/pareto_front.json"

COMMON="--example TransformerLM --dataset wikitext2 \
 --hidden-dim 256 --vocab-size 512 --num-layers 3 --seed 250197 \
 --exec-on-gpu \
 --cmp-type latency --mem-type peak_memory \
 --num-data-points 5 --reps-per-point 4 \
 --quality-metric grad_cosine --walk-steps 200 \
 --out-dir $OUT"

run_on () {   # $1 = gpu index, $2 = label, rest = args
  local g="$1"; local lbl="$2"; shift 2
  echo "=========================================================="
  echo "PHASE $lbl  gpu=$g  start $(date +%H:%M:%S)  PULLDOWN=$GRAPHAX_QUANT_PULLDOWN"
  echo "=========================================================="
  CUDA_VISIBLE_DEVICES=$g uv run --no-sync python "$TOOL" "$@"
  echo "PHASE $lbl exited rc=$? at $(date +%H:%M:%S)"
}

# ---- QA: name the face every archived plan skips -------------------------
# --reps 0 --warmup-trials 0: this phase MEASURES NOTHING.  It builds every
# archived plan, enumerates the live-face inventory on the exact prefix, and
# writes plans_*.json with each wire resolved to (step, vertex, face key,
# primitive, operand shapes/dtypes).  Costs ~2 minutes.
run_on 0 "QA-face-attribution" $COMMON $ARCH \
  --latency-inner-reps 50 --reps 0 --warmup-trials 0 \
  --ladder "" --no-all-rung --ops "" --no-skip-plan \
  --archive-all-points --archive-max-measure 40 \
  --face-inventory --noise-floor-reps 0 \
  --tag qa_attrib --config-note "$NOTE phase=QA" --max-seconds 900

# ---- QB: SINGLETON SWEEP -- every live face, skipped alone ---------------
# Each plan skips exactly ONE face, so every row IS a minimal plan and the
# best row IS the true optimum of the fixed-rev SKIP space.  Paired against
# its own exact reference, warm (2 discarded rounds), n=3.
run_on 0 "QB-singleton-sweep" $COMMON \
  --latency-inner-reps 50 --reps 3 --warmup-trials 2 \
  --ladder "" --no-all-rung --ops "" --no-skip-plan \
  --singleton-skip-sweep --sweep-stride 1 \
  --face-inventory --noise-floor-reps 0 \
  --tag qb_sweep --config-note "$NOTE phase=QB" --max-seconds 5400

# ---- QC: combined report -------------------------------------------------
run_on 0 "QC-report" $COMMON --report-only --tag COMBINED

echo "=========================================================="
echo "FACE ATTRIBUTION DONE $(date). alphagrad $AG_SHA graphax $GX_SHA"
ls -la $OUT
echo "=========================================================="
""".replace("@TOOL@", LANDSCAPE_TOOL).replace("@REPO@", REPO) \
   .replace("@HOME_DSNN@", HOME_DSNN)


arm(
    name="face_attrib",
    job="face-attribution",
    kind="tool",
    node="pgi15-gpu18",
    time="2:30:00",
    gpus=4,
    needs_tool=LANDSCAPE_TOOL,
    required_flags=LANDSCAPE_FLAGS,
    required_flags_file=LANDSCAPE_TOOL,
    env={
        # The archived winners were produced under PULLDOWN=1, so the archived
        # plans are rebuilt under pulldown.  Comparing them under pullup would
        # be comparing two different compute stacks and calling the difference
        # a result.
        "GRAPHAX_QUANT_PULLDOWN": "1",
        "ALPHAGRAD_MAX_FACES": "2538",
        "ALPHAGRAD_MAX_DELTA_TOKENS": "32768",
        "ALPHAGRAD_MEASURE_WARMUP": "1",
        "GRAPHAX_PLANNER_EXACT": "1",
        "GRAPHAX_DEMAND_EMIT": "1",
        # K=1 is the settled grad-cosine variant (949f1af): most predictive
        # AND cheapest.  Relevant here because this arm's quality channel IS
        # grad_cosine.
        "ALPHAGRAD_GRAD_COSINE_K": "1",
        # ALPHAGRAD_NEW_SLOT_JOIN is left at the shared default (1).  The
        # hand-written predecessor forced 0 because the PINNED graphax 4ea0bf8
        # rejects the res-slot two-op form; the live graphax accepts it
        # (e5fd46c) and the two-op pre-flight above VERIFIES that before any
        # phase runs.  It is inert for QB in any case -- a SKIP-only plan
        # writes no rule into any slot.
        #
        # ---- DROPPED FROM THE TRAINING STACK -----------------------------
        # SHARED_ENV describes ppo.py.  Two of these do not merely add noise
        # to a measurement arm, they CHANGE THE NUMBER, and the hand-written
        # launcher this arm replaces set neither:
        #
        # QUALITY_GATE_MIN: defaults to 0 = gate OFF.  At 0.05 the additive
        #   quality gate FLOORS latency_ns and peak_memory at the exact-rev
        #   reference for any plan scoring below 0.05.  A singleton SKIP
        #   sweep exists to price exactly those plans, so inheriting this
        #   would silently replace the measurement with the reference cost.
        # BATCHED_CALLBACK: defaults to 0.  At 1 env._callback takes the
        #   batched host path -- a different measurement path from the one
        #   every archived row was measured on.
        #
        # The rest are trainer-only and have no meaning here: there is no
        # policy, no actor pool and no episode loop in landscape_map.
        "ALPHAGRAD_QUALITY_GATE_MIN": _DELETE,
        "ALPHAGRAD_BATCHED_CALLBACK": _DELETE,
        "ALPHAGRAD_FORCE_REV_ORDER": _DELETE,
        "ALPHAGRAD_POLICY": _DELETE,
        "ALPHAGRAD_ACTOR_PROF_EVERY": _DELETE,
        "ALPHAGRAD_PROFILE": _DELETE,
        "ALPHAGRAD_DEBUG_APPROX_PROB": _DELETE,
        "ALPHAGRAD_DEBUG_DEGEN": _DELETE,
        "ALPHAGRAD_DEBUG_MEM": _DELETE,
        "ALPHAGRAD_DEBUG_MEASURE": _DELETE,
    },
    purpose="""FACE ATTRIBUTION / SINGLETON SKIP SWEEP -- which face is the win?

The archived winners at ratio ~0.52-0.58 carry ONE face wire and ZERO applied
diag/compress/quant rules -- the win is a SKIP.  So WHICH face is skipped
decides everything, and the search space is worth mapping face by face.

  QA  FACE ATTRIBUTION -- name the skipped face of every archived plan
  QB  SINGLETON SWEEP  -- skip each live face ALONE, paired, n=3
  QC  combined report

WHY ALL FOUR GPUs FOR A ONE-GPU JOB.  The peak_memory channel is a
DEVICE-WIDE counter (peak_bytes_in_use delta); a co-resident process on the
same GPU inflates it, and CV was measured going 0.0000% -> 49.7% under a
noisy neighbour.  Holding the node is what makes the memory column mean
anything.  Only CUDA_VISIBLE_DEVICES=0 is ever used.

ADOPTED INTO THE GENERATOR 2026-08-30 (ticket 22).  It ran for two days as a
hand-edited file outside git passing --face-inventory, --singleton-skip-sweep
and --sweep-stride, none of which any COMMITTED landscape_map.py defined -- it
worked only because it invoked a snapshot copy.  The required-flag guard above
now greps THE TOOL IT ACTUALLY INVOKES for every flag it passes, so that
divergence aborts in seconds instead of hours in.

TWO DELIBERATE CHANGES from the hand-written version, both forced and both
documented at the constants above and in `env`: the quality channel is
grad_cosine rather than loss_drop (owner decision), which the 2026-08-27 pin
cannot express at all, so the arm imports the live trees; and the output goes
to run_analysis/landscape_gradcos so grad_cosine rows never share a report
glob with the archived loss_drop ones.""",
    prediction="""QA resolves every archived winner's single wire to a named
(step, vertex, graphax face key, primitive, operand shapes) and costs ~2 min
with --reps 0.  QB measures ~115-118 singleton plans; the best singleton
reproduces the archived winners' ~0.53 LATENCY RATIO, confirming the win is
ONE face.  The QUALITY column will NOT match the archived runs and is not
expected to: it is a different quantity.  Note that the live stack enumerates
117 live faces where the pinned stack enumerated 118, so k/f indices are NOT
transferable between the two -- faces must be matched by (vertex, primitive,
key), not by index.""",
    falsifier="""Import resolution failing (exit 70), any flag missing from the
tool (exit 64), or QB's best singleton latency ratio not reproducing the
archived winner's ratio within the paired noise floor -- the last would mean
the one-face attribution is wrong.""",
    body=_FACE_ATTRIB_BODY,
)


def _wrap_comment(text: str, prefix: str = "# ") -> str:
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("#")
            continue
        out.append(prefix + para.rstrip())
    return "\n".join(out)


def _merge_cli(overrides: dict) -> list[tuple[str, str | None]]:
    """SHARED_CLI with per-arm overrides applied, order preserved."""
    merged: list[tuple[str, str | None]] = []
    seen = set()
    for flag, val in SHARED_CLI:
        if flag in overrides:
            new = overrides[flag]
            seen.add(flag)
            if new is _DELETE:
                continue
            merged.append((flag, new))
        else:
            merged.append((flag, val))
    for flag, val in overrides.items():
        if flag in seen or val is _DELETE:
            continue
        merged.append((flag, val))
    return merged


def render(a: dict) -> str:
    kind = a["kind"]
    gpus = a.get("gpus", 0)
    L = ["#!/bin/bash"]
    L.append("#SBATCH -p " + ("pgi15-cpu" if kind == "cpu" else "pgi15"))
    L.append(f"#SBATCH -w {a['node']}")
    if gpus:
        L.append(f"#SBATCH --gres=gpu:{gpus}")
        L.append("#SBATCH -c 64")
        L.append("#SBATCH --mem=400G")
    else:
        L.append("#SBATCH -c 8")
        L.append("#SBATCH --mem=64G")
    L.append(f"#SBATCH -t {a['time']}")
    L.append(f"#SBATCH -J {a['job']}")
    L.append(f"#SBATCH -o {HOME_DSNN}/{a['name']}_%j.log")
    L.append("#")
    L.append("# " + "=" * 72)
    L.append(_wrap_comment(a["purpose"]))
    L.append("#")
    L.append("# REGISTERED PREDICTION (recorded BEFORE the run; never edited after):")
    L.append(_wrap_comment(a["prediction"], "#   "))
    L.append("#")
    L.append("# FALSIFICATION CRITERION:")
    L.append(_wrap_comment(a["falsifier"], "#   "))
    if a.get("depends"):
        L.append("#")
        L.append("# DEPENDS ON: " + a["depends"])
    if a.get("owner_ruling"):
        L.append("#")
        L.append("# *** OWNER RULING REQUIRED BEFORE LAUNCH ***")
        L.append(_wrap_comment(a["owner_ruling"], "#   "))
    L.append("# " + "=" * 72)
    L.append("#")
    L.append("# GENERATED BY tools/gen_fq_launchers.py -- DO NOT EDIT IN PLACE.")
    L.append("# fq_v58_tlm_env16.sbatch was edited while its job was pending and")
    L.append("# job 61494's command line is unrecoverable.  Edit the generator.")
    L.append("")
    L.append("set -uo pipefail")
    L.append("")

    if kind == "cpu" and not a.get("needs_tool"):
        L.append(PREAMBLE.format(repo=REPO).rstrip())
        L.append('export PATH="$HOME/.local/bin:$PATH"')
        L.append('export PYTHONPATH="$HOME/dsnn/graphax/src:$HOME/dsnn/alphagrad/src"')
        L.append("export PYTHONDONTWRITEBYTECODE=1")
        L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
        L.append('echo "ag=$(git -C ~/dsnn/alphagrad rev-parse --short HEAD)'
                 ' gx=$(git -C ~/dsnn/graphax rev-parse --short HEAD)"')
        L.append(a["body"])
        return "\n".join(L) + "\n"

    # --- shared env block
    L.append(PREAMBLE.format(repo=REPO).rstrip())
    over = a.get("env", {})
    for k, v in SHARED_ENV:
        if k == "RAY_TMPDIR":
            continue
        nv = over.get(k, v)
        if nv is _DELETE:
            continue
        L.append(f"export {k}={nv}")
    for k, v in over.items():
        if k not in {kk for kk, _ in SHARED_ENV} and v is not _DELETE:
            L.append(f"export {k}={v}")
    L.append("")

    # --- pre-flight
    L.append("# ---------------------- PRE-FLIGHT ----------------------------")
    _has_wandb = not (kind == "probe" or a.get("needs_tool"))
    L.append("# %s layers, each failing LOUDLY with its own exit code before"
             % ("FOUR" if _has_wandb else "THREE"))
    L.append("# any setup noise reaches the log.  The R1-R4 battery shipped the")
    L.append("# first layer and it caught three missing flags.")
    _flagsrc = a.get("required_flags_file", "src/alphagrad/approx/ppo.py")
    _flags = a.get("required_flags", REQUIRED_FLAGS)
    L.append(f"#   64 = a flag this launcher needs is not defined in {_flagsrc}")
    L.append("#   65 = argparse rejected the assembled command line")
    L.append("#   66 = a tool this launcher invokes does not exist")
    L.append("#   70 = graphax cannot lower what ALPHAGRAD_NEW_SLOT_JOIN asks for")
    if _has_wandb:
        L.append("#   71 = --wandb online, but this node cannot reach or"
                 " authenticate to wandb")
    # Layer 0 BEFORE layer 1: if the file the flags are grepped from does not
    # exist, every flag reads as "missing" and the abort names the wrong
    # problem.  Existence first, then contents.
    if a.get("needs_tool"):
        t = a["needs_tool"]
        L.append(f'if [ ! -f "{t}" ]; then')
        L.append(f'  echo "ABORT(66): {t} does not exist -- this arm is'
                 ' BLOCKED ON A BUILD."')
        L.append("  exit 66")
        L.append("fi")
        L.append("")
    L.append("MISSING=\"\"")
    L.append("Q='\"'")
    L.append(f"FLAGSRC={_flagsrc}")
    L.append("for F in " + " ".join(_flags) + "; do")
    L.append('  grep -qF -- "$Q$F$Q" "$FLAGSRC"'
             ' || MISSING="$MISSING $F"')
    L.append("done")
    L.append('if [ -n "$MISSING" ]; then')
    L.append('  echo "ABORT(64): $FLAGSRC does not define:$MISSING"')
    L.append("  exit 64")
    L.append("fi")
    L.append("")
    if kind != "cpu":
        L.append("# ALPHAGRAD_NEW_SLOT_JOIN=1 emits the res-slot two-op face form.")
        L.append("# On a graphax that rejects it, EVERY plan putting a rule in the")
        L.append("# res/new slot dies in _trace_truncate SILENTLY -- no counter, no")
        L.append("# log line.  That went unnoticed for a whole campaign.  VERIFY.")
        L.append('if [ "${ALPHAGRAD_NEW_SLOT_JOIN:-1}" = "1" ]'
                 ' && [ "${FQ_SKIP_TWOOP:-0}" != "1" ]; then')
        L.append("  JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q -x \\")
        L.append("    $HOME/dsnn/graphax/tests/misc/test_face_two_op_form.py \\")
        L.append("    -p no:cacheprovider >/tmp/twoop_$SLURM_JOB_ID.log 2>&1 || {")
        L.append('    echo "ABORT(70): graphax rejects the res-slot two-op form,"')
        L.append('    echo "           but ALPHAGRAD_NEW_SLOT_JOIN=1 emits it."')
        L.append("    tail -20 /tmp/twoop_$SLURM_JOB_ID.log")
        L.append("    exit 70")
        L.append("  }")
        L.append('  echo "[preflight] graphax accepts the two-op face form"')
        L.append("fi")
        L.append("")

    if kind == "probe" or a.get("needs_tool"):
        L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
        L.append('echo "ag=$(git -C ~/dsnn/alphagrad rev-parse --short HEAD)'
                 ' gx=$(git -C ~/dsnn/graphax rev-parse --short HEAD)"')
        if kind != "cpu":
            L.append("nvidia-smi --query-gpu=index,name,memory.total"
                     " --format=csv,noheader")
        L.append(a["body"])
        return "\n".join(L) + "\n"

    # --- Layer 3: wandb.  Every arm past this point carries `--wandb online`
    #     (the WANDB constant), and an online run whose backend is unreachable
    #     -- expired credentials, a firewalled node, a typo'd entity -- trains
    #     for hours and lands NOWHERE the owner can see.  That is the "is the
    #     wandb syncing?  I don't see it on the dashboard" failure, and like
    #     the two-op form above it is SILENT.  Prove the authenticated
    #     round-trip HERE, from THIS node, against the SAME entity the command
    #     line will use, and print the dashboard URL into the slurm log.
    L.append("# Layer 3: wandb credentials + a real authenticated round-trip")
    L.append("# from THIS node to THIS entity, before hours are spent.  ~2 s.")
    L.append('if [ "${FQ_SKIP_WANDB_CHECK:-0}" != "1" ]; then')
    L.append("  JAX_PLATFORMS=cpu uv run --no-sync python - "
             "<<'FQ_WANDB_EOF' || {")
    L.append("import sys, wandb")
    L.append(f"ENT, PROJ = {WANDB_ENTITY!r}, {WANDB_PROJECT!r}")
    L.append("api = wandb.Api(timeout=30)")
    L.append("teams = list(api.viewer.teams)")
    L.append("if ENT not in teams:")
    L.append("    sys.exit('wandb entity %r not available to these"
             " credentials (viewer teams: %r)' % (ENT, teams))")
    L.append("print('[preflight] wandb reachable and authenticated:"
             " entity %s, project %s' % (ENT, PROJ))")
    L.append("FQ_WANDB_EOF")
    L.append('  echo "ABORT(71): --wandb online but this node cannot reach'
             ' or authenticate to wandb."')
    L.append('  echo "          A run started here would train blind.'
             '  Set FQ_SKIP_WANDB_CHECK=1"')
    L.append('  echo "          to override, or fix credentials with'
             ' wandb login."')
    L.append("  exit 71")
    L.append("}")
    L.append(f'  echo "[preflight] dashboard:'
             f' https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}"')
    L.append("fi")
    L.append("")

    # --- the command line, ONCE, as an array: the dry-parse and the real run
    #     cannot disagree because they are the same tokens.
    L.append("ARGS=(")
    for flag, val in _merge_cli(a.get("cli", {})):
        L.append(f"  {flag}" + (f" {val}" if val is not None else ""))
    L.append(f"  {WANDB}")
    L.append(")")
    L.append("")
    L.append("# Layer 2: run the EXACT token list through ppo.py's own argparse.")
    L.append("# This catches a bad CHOICE value (a --face-read typo, an")
    L.append("# --advantage-norm value valid on the ray surface but not this one)")
    L.append("# that a name-only grep cannot see.  JAX_PLATFORMS=cpu so it does")
    L.append("# not touch a GPU.")
    L.append("JAX_PLATFORMS=cpu uv run --no-sync python -c \\")
    L.append("\"import sys; from alphagrad.approx.ppo import make_argparser;\\")
    L.append(" make_argparser().parse_args(sys.argv[1:]);\\")
    L.append(" print('[preflight] argparse accepted the command line')\" \\")
    L.append('  "${ARGS[@]}" || { echo "ABORT(65): argparse rejected it"; exit 65; }')
    L.append("")
    L.append("# FQ_PREFLIGHT_ONLY=1 stops here.  This is how a launcher is")
    L.append("# validated without consuming a node -- every guard above has run")
    L.append("# and nothing has been submitted.")
    L.append('if [ "${FQ_PREFLIGHT_ONLY:-0}" = "1" ]; then')
    L.append('  echo "[preflight] PREFLIGHT ONLY -- all guards passed, not launching"')
    L.append("  exit 0")
    L.append("fi")
    L.append("")
    L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
    L.append('echo "ag=$(git -C ~/dsnn/alphagrad rev-parse --short HEAD)'
             ' gx=$(git -C ~/dsnn/graphax rev-parse --short HEAD)"')
    L.append("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    L.append("")
    L.append("CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python \\")
    L.append('  src/alphagrad/approx/ppo.py "${ARGS[@]}"')
    L.append('echo "TRAINER exited with $?"')
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HOME_DSNN)
    ap.add_argument("--check", action="store_true",
                    help="render, syntax-check and DIFF against what is on "
                         "disk; write nothing; exit non-zero on any drift")
    ns = ap.parse_args()

    rc = 0
    drifted = []
    for a in ARMS:
        text = render(a)
        path = os.path.join(ns.out, f"fq_{a['name']}.sbatch")
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch",
                                         delete=False) as fh:
            fh.write(text)
            tmp = fh.name
        chk = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        if chk.returncode != 0:
            print(f"SYNTAX ERROR in {path}:\n{chk.stderr}", file=sys.stderr)
            os.unlink(tmp)
            rc = 1
            continue
        if ns.check:
            # A --check that only ran `bash -n` reported "ok" for a launcher
            # whose on-disk copy had drifted arbitrarily far from the
            # generator -- it proved the FILE WAS SHELL, not that it was THIS
            # file.  Diff, and make drift a non-zero exit.
            old = None
            if os.path.exists(path):
                with open(path) as fh:
                    old = fh.read()
            if old is None:
                print(f"MISSING       {path} (would be created)")
                drifted.append(path)
                rc = 1
            elif old != text:
                print(f"DRIFT         {path}")
                sys.stdout.writelines(difflib.unified_diff(
                    old.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    fromfile=f"{path} (on disk)",
                    tofile=f"{path} (generated)"))
                drifted.append(path)
                rc = 1
            else:
                print(f"ok            {path}")
            os.unlink(tmp)
            continue
        shutil.move(tmp, path)
        os.chmod(path, 0o644)
        print(f"wrote {path}")
    if ns.check and drifted:
        print(f"\n{len(drifted)} launcher(s) differ from the generator:",
              file=sys.stderr)
        for d in drifted:
            print(f"  {d}", file=sys.stderr)
        print("Regenerate (drop --check) rather than editing them in place.",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
