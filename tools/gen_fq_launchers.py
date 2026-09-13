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
    python3 tools/gen_fq_launchers.py --dry-run --out /Scratch/.../t43_launchers \
                                      --against ~/dsnn        # write OUTSIDE the
                                      # tree, diff against the tree, touch nothing

Every emitted file is `bash -n` checked before it is written (a previous bulk
edit silently uncommented ~40 launchers; syntax is verified, never eyeballed).

The plan these launchers execute, with the registered predictions and the
decision table, is docs/EXPERIMENT_PLAN.md.  The wave 0-4 arms are the 2026-08
running comparison; THE CAMPAIGN section (tickets .50-.54, one declarative
row per arm, `campaign_arm`) is the phase 1-5 plan of finding 53, regenerated
2026-09-13 (ticket .43) under the owner's rulings of that day: the static
Markowitz order (--fixed-order markowitz, .64), ONE face-ADD value
(--approx-add lossless; the all-rev pair of .56 is dropped), the four-dtype
Quant head whose width is DERIVED (never a literal), args only (the env vars a
campaign launcher exports are the TLM target shape and the Ray/measure
plumbing; a knob that still has no flag is exported under a TODO header, never
silently), one 8-GPU Blackwell job per node on pgi15-gpu19/20, --ray-measure 1.
The gate telemetry of .45 has no switch: ppo.py emits gate/*, paired/* and
measure/* every episode; the launcher supplies G1's input (--gate-winners-table).
"""

from __future__ import annotations

import argparse
import difflib
import os
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
# ALPHAGRAD_QUALITY_METRIC=auto resolves to grad_cosine since 2026-09-02
# (ticket dsnn-3qm.39; it was loss_drop before), and every arm still names
# grad_cosine EXPLICITLY: the pin is redundant now and harmless, and it keeps
# the launcher's channel independent of the env default.
# ---------------------------------------------------------------------------

REPO = "/Users/assmuth/dsnn/alphagrad"
HOME_DSNN = "/Users/assmuth/dsnn"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")

# ---------------------------------------------------------------------------
# THE FACE ADD (ticket .56, finding 73; owner ruling 2026-09-13): ONE value,
# lossless, on every training arm.  The sum's support is the UNION of the two
# addends' supports and no non-zero is dropped.  The `lossy` / `lossless`
# paired pair that ticket .56 once asked for on the all-rev arm is DROPPED
# (the learned join values -- choose, learned1, learned2 -- are benched and
# the pair's reading depended on them).  `campaign_arm` REFUSES any other
# value; there is no `arm_per_approx_add` any more.
# ---------------------------------------------------------------------------
APPROX_ADD = "lossless"


def _face_head_geometry(mode: str):
    """(width, quant dtypes) of the face head under ``--approx-add mode``.

    DERIVED from ``alphagrad.approx.unified_face_head.head_layout`` and
    ``common.masks.FACE_QUANT_DTYPES``, never typed: the head gained nine
    logits (three slots x three extra dtype logits: a four-way softmax
    replaced the one-logit Bernoulli) when the Quant dtype set grew from
    {f32, bf16} to the four floats (f32, bf16, f8e5m2, f8e4m3fn), and a
    launcher header that still said the old number
    would have described a head that no longer exists.  RAISES when the
    library cannot be imported -- falling back to a literal is exactly the
    stale-launcher failure this function exists to prevent.
    """
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    try:
        from alphagrad.approx.unified_face_head import head_layout
        from alphagrad.approx.common.masks import FACE_QUANT_DTYPES
    except Exception as exc:  # noqa: BLE001 -- re-raised with the remedy
        raise ImportError(
            f"gen_fq_launchers derives the face-head width from "
            f"{_SRC}/alphagrad/approx/unified_face_head.py and cannot import "
            f"it ({exc!r}).  Run the generator inside the project venv "
            f"(uv run --no-sync python tools/gen_fq_launchers.py ...); it "
            f"does not fall back to a typed width.") from exc
    return int(head_layout(mode).width), tuple(FACE_QUANT_DTYPES)


FACE_HEAD_WIDTH, FACE_QUANT_DTYPES = _face_head_geometry(APPROX_ADD)

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

# GATE G1 (ticket .45) reads the sweep winners from a table that depends on
# the arm's ELIMINATION ORDER: a plan applied at vertex v on the Markowitz
# order is a different opportunity from the same vertex on the reverse order,
# and the two sweeps found different winners (4029 rows vs 2969).  An arm
# pointed at the other order's table reports a recovery fraction that means
# nothing, so the value is resolved per arm in `_merge_cli` from the arm's own
# --fixed-order.  The tables are the SWEEP64 ones (owner ruling 2026-09-13);
# CAMPAIGN_GATE_WINNERS_TABLES below is the map, and the old
# ~/dsnn/run_analysis/sweep41/winners.csv path is gone -- nothing ever wrote
# it, so every launcher carrying it logged gate/g1/present = 0 forever.
class _ByOrder:
    """Sentinel: resolve this flag's value from the arm's --fixed-order."""

    def __repr__(self) -> str:        # so a stray copy is visible in a diff
        return "<resolved from --fixed-order>"


GATE_WINNERS_TABLE = _ByOrder()

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
    "--approx-add",
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
    "--measure-toolchain-gate",
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
    "--face-none-bias",
    "--exact",
    "--var-probe",
    "--face-edge-mem",
    "--per-face-masks",
    "--plan-log",
    "--approx-add",
    # Phase-0d flags the campaign arms (tickets .50-.54) pass: the profile
    # (.40), the cost form and the quality floor (.9), the memory channel
    # (.49), the init knobs (.44), the decode frames (.18, .20), the reward
    # form and normalisation (.12, .53) and the gate telemetry inputs (.45,
    # defined in common/gate_telemetry.py, hence the second file in
    # REQUIRED_FLAGS_FILES).
    "--approx-profile",
    "--cost-form",
    "--quality-floor",
    "--mem-channel",
    "--scale-face-head",
    "--face-logit-clamp",
    "--reduce-axis-space",
    "--rewards",
    "--lambda-acc",
    "--reward-mode",
    "--advantage-norm",
    "--no-symlog",
    "--preference-conditioned",
    "--gate-winners-table",
    "--gate-offline-contrast",
    # Ticket .43 (2026-09-13): the order (.64), the Ray measurement fan-out
    # the campaign runs under, and the face-path preconditions ppo.py now
    # REQUIRES rather than defaults (--face-actions --unified-face-head
    # --live-faces --dynamic-substeps; --per-face-masks is always on with
    # --face-actions and still accepted).
    "--fixed-order",
    "--ray-measure",
    "--ray-measure-timeout",
    "--terminal-rewards-only",
    "--face-actions",
    "--unified-face-head",
    "--live-faces",
    "--dynamic-substeps",
]

# The files the pre-flight greps REQUIRED_FLAGS in.  ppo.py defines every
# trainer flag but the two gate inputs, which gate_telemetry.add_gate_args
# adds to the same parser (ticket .45).
REQUIRED_FLAGS_FILES = [
    "src/alphagrad/approx/ppo.py",
    "src/alphagrad/approx/common/gate_telemetry.py",
]

# Env vars that BECAME FLAGS (owner ruling 2026-09-03: args only, no
# fallback period).  No training arm may export one; the campaign test pins
# it against every rendered launcher.  The vars a launcher still exports are
# the measurement environment (ALPHAGRAD_SKIP_COUNT_OPS, the TLM shape,
# JAX_COMPILATION_CACHE_DIR) and the import-time settings that have no flag
# yet (ALPHAGRAD_FORCE_REV_ORDER, ALPHAGRAD_MAX_FACES, ALPHAGRAD_POLICY).
PROMOTED_ENV_VARS = [
    "ALPHAGRAD_FACE_NONE_BIAS",        # --face-none-bias (.44)
    "ALPHAGRAD_NEW_SLOT_JOIN",         # --approx-add (.56)
    "ALPHAGRAD_QUALITY_GATE_MIN",      # deleted; --quality-floor (.9)
    "ALPHAGRAD_QUALITY_METRIC",        # --quality-metric (.39)
    "ALPHAGRAD_APPROX_OLD",            # RETIRED (.56/73): env.approx_add raises
    "ALPHAGRAD_APPROX_ADD",            # --approx-add (.56, finding 73)
    "ALPHAGRAD_MEM_CHANNEL",           # --mem-channel (.49)
    "ALPHAGRAD_MEM_PARITY",            # deleted (.49): parity always recorded
    "ALPHAGRAD_COST_FORM",             # --cost-form (.9)
    "ALPHAGRAD_REDUCE_AXIS_SPACE",     # --reduce-axis-space (.20)
    "ALPHAGRAD_MEASURE_TOOLCHAIN_GATE",  # --measure-toolchain-gate (.21)
    "ALPHAGRAD_PLAN_LOG",              # --plan-log
    "ALPHAGRAD_REWARD_MODE",           # --reward-mode
    "ALPHAGRAD_COS_LOG_EVERY",         # --cos-log-every
    "ALPHAGRAD_SPARSITY",              # --sparsity-log
    "ALPHAGRAD_WALK_ROTATE",           # --walk-rotate
    "ALPHAGRAD_PER_FACE_MASKS",        # --per-face-masks
    "ALPHAGRAD_DIAG_PER_FACE",         # --diag-per-face
    # Dropped 2026-09-01 (4be3ff7d): graphax no longer reads it.  The
    # on-disk wave launchers that still export it predate that commit.
    "GRAPHAX_ALLOW_PARTIAL_ORDER",
    "ALPHAGRAD_FORCE_REV_ORDER",       # --fixed-order (.64); a set var raises
    "GRAPHAX_PLANNER_EXACT",           # removed with the second engine (.65)
]

class _Delete:
    """Sentinel: an arm sets a key to _DELETE to REMOVE it, in `cli` or `env`.

    Deleting an inherited env var is not cosmetic.  SHARED_ENV is the
    TRAINING stack; a measurement arm that silently inherited
    ALPHAGRAD_QUALITY_GATE_MIN=0.05 (deleted 2026-09-04, ticket .9) had its
    cost channels FLOORED at the exact-reverse reference for every plan
    scoring below 0.05 quality -- which for a SKIP sweep is precisely the
    plans being measured.
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
    # The face-ADD configuration (formerly ALPHAGRAD_NEW_SLOT_JOIN, then
    # ALPHAGRAD_APPROX_OLD) is an ARGUMENT now: --approx-add in SHARED_CLI
    # (ticket .56, finding 73).
    ("ALPHAGRAD_POLICY", "palimpsa"),
    # The elimination order is an ARGUMENT now: --fixed-order on every train
    # arm (ticket .64; ALPHAGRAD_FORCE_REV_ORDER fails loudly when set).
    # The additive quality gate (ALPHAGRAD_QUALITY_GATE_MIN=0.05: plans below
    # qmin paid the exact-rev reference cost, so the SKIP cliff earned nothing)
    # was DELETED on 2026-09-04 (ticket dsnn-3qm.9). Its replacement is the
    # --quality-floor argument (a hinge on the quality channel, not a clamp on
    # the cost channels); ticket .43 puts it into the campaign CLI.
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
    # PER NODE, never shared (owner ruling on ticket .21; finding 03 sec 5a):
    # an entry written on a healthy node is reused verbatim on a node whose
    # link toolchain is broken and the fault never fires.  That is why the
    # wave-1 contamination was 51% / 97% rather than 100%, and why the
    # contaminated plan set cannot be recovered from the node name.  Expanded
    # by the shell ON THE NODE at job start.
    ("JAX_COMPILATION_CACHE_DIR", "$HOME/.jaxcache_$(hostname -s)"),
    ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "2"),
]

# ---------------------------------------------------------------------------
# THE MEASURE TOOLCHAIN (finding 03, ticket dsnn-3qm.21).  The measure compile
# options split the module (xla_gpu_enable_llvm_module_compilation_parallelism)
# and splitting means LINKING: XLA shells out to the first `nvlink` it finds.
# Wave 1 found it under /usr/local/cuda -- an admin-flipped SYMLINK that on
# pgi15-gpu16/17 pointed at cuda-12.8 while the venv's ptxas (12.9.86) emits
# 12.9 cubins.  nvlink refused every one ("newer than toolkit (129 vs 128)"),
# _compile_measure silently took the degraded-fusion executable, and 51% of
# w1b and 97% of w1c were contaminated (docs/WAVE1_RESULTS.md).
#
# The fix the owner approved (option 4a): put the MATCHED toolkit first on
# PATH, in a form that names no node and never trusts the symlink, and prove
# the version before python starts.  Not xla_gpu_cuda_data_dir: it silences
# the error when pointed at ANY directory, an empty one included (measured).
# CUDA_WANT is the release of the ptxas the venv ships (nvidia_cuda_nvcc_cu12
# 12.9.86); nvlink has to be the same release, so the job proves BOTH.
#
# A CPU job (JAX_PLATFORMS=cpu) never links, so it runs the same block for the
# record but does not abort when the node has no toolkit.
# ---------------------------------------------------------------------------

CUDA_WANT = "12.9"

TOOLCHAIN_BLOCK = r"""# ---------------------- MEASURE TOOLCHAIN ---------------------
# Finding 03: XLA links the measure executables with the first nvlink on
# PATH; a 12.8 nvlink refuses 12.9 ptxas cubins and every measurement
# silently degrades.  Put the matched @WANT@ toolkit first, from whatever
# /usr/local/cuda-* this node has (never the /usr/local/cuda symlink), and
# prove the version.  72 = no matched @WANT@ toolkit on this node.
FQ_CUDA_WANT=@WANT@
FQ_CUDA_BIN=""
for d in /usr/local/cuda-*/bin; do
  [ -x "$d/ptxas" ] && [ -x "$d/nvlink" ] || continue
  pv=$("$d/ptxas" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
  nv=$("$d/nvlink" --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
  if [ "$pv" = "$FQ_CUDA_WANT" ] && [ "$nv" = "$FQ_CUDA_WANT" ]; then FQ_CUDA_BIN=$d; fi
done
if [ -n "$FQ_CUDA_BIN" ]; then
  export PATH="$FQ_CUDA_BIN:$PATH"
fi
FQ_PTXAS_VER=$(ptxas --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
FQ_NVLINK_VER=$(nvlink --version 2>&1 | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | tail -1)
echo "[cfg] measure toolchain on $(hostname -s): PATH<-${FQ_CUDA_BIN:-<none>}" \
     "ptxas ${FQ_PTXAS_VER:-none} ($(command -v ptxas || echo not-on-PATH))" \
     "nvlink ${FQ_NVLINK_VER:-none} ($(command -v nvlink || echo not-on-PATH))" \
     "/usr/local/cuda -> $(readlink -f /usr/local/cuda 2>/dev/null || echo none)"
if [ "$FQ_PTXAS_VER" != "$FQ_CUDA_WANT" ] || [ "$FQ_NVLINK_VER" != "$FQ_CUDA_WANT" ]; then
@ON_FAULT@
fi
"""

_TOOLCHAIN_ABORT = (
    '  echo "ABORT(72): measure toolchain on $(hostname -s) is'
    ' ptxas ${FQ_PTXAS_VER:-none} / nvlink ${FQ_NVLINK_VER:-none},'
    ' want $FQ_CUDA_WANT -- every measurement here would be a degraded'
    ' fallback (finding 03)"\n'
    '  exit 72')
_TOOLCHAIN_CPU_NOTE = (
    '  echo "[cfg] CPU job (JAX_PLATFORMS=cpu): XLA never links, no CUDA'
    ' toolkit required on this node"')


def _toolchain_block(kind: str) -> str:
    on_fault = _TOOLCHAIN_CPU_NOTE if kind == "cpu" else _TOOLCHAIN_ABORT
    return (TOOLCHAIN_BLOCK.replace("@WANT@", CUDA_WANT)
            .replace("@ON_FAULT@", on_fault).rstrip())

# ---------------------------------------------------------------------------
# SHARED CLI.  Ordered list of (flag, value).  Arms override by flag name in
# `cli` (None = a bare store_true flag; _DELETE removes the flag).
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
    # THE CREDIT HORIZON.  ppo.py defaulted to 0.99/0.95 until 2026-09-01;
    # at 95 eliminations (gamma*lambda)^94 = 0.0031, i.e. the terminal reward
    # reaches the first decision at 0.31 percent strength (finding 55).
    # R3 restored the old defaults on purpose and drifted
    # to destruction at ep131 while R2 at 1.0/1.0 held.  MANDATORY, and named
    # explicitly because no campaign run v57-v66 ever set them.
    ("--discount", "1.0"),
    ("--gae-lambda", "1.0"),
    # ("--reject-frozen-grads", the gradient-coverage guard, sat here from
    # wave 1 until 2026-09-03; removed by owner ruling 2026-09-03, ticket dsnn-3qm.15.)
    # THE TRAINED QUALITY CHANNEL.  auto means grad_cosine since 2026-09-02;
    # the explicit pin is redundant and kept on purpose.
    ("--quality-metric", "grad_cosine"),
    # THE FACE ADD (ticket .56, finding 73; owner ruling 2026-09-13): ONE
    # value, APPROX_ADD = lossless, on every training arm -- the sum's support
    # is the UNION of the two addends' supports and no non-zero is dropped
    # (the pre-flight below VERIFIES graphax accepts the two-op form).  Named
    # so no arm inherits an unstated default; `campaign_arm` refuses any
    # other value.
    ("--approx-add", APPROX_ADD),
    # THE FIXED ORDER (ticket .64): the static minimum Markowitz degree order
    # of the exact graph, one table (common/order.py) shared with the sweep.
    # The ppo.py default since 2026-09-05; named so every fixed-order arm
    # carries it on its own command line.  The order arms override it with
    # `free`; `reverse` is the control of .60 only.
    ("--fixed-order", "markowitz"),
    # THE MEASURE TOOLCHAIN GATE (finding 03).  abort is the default; named
    # so no arm can inherit a stale warn.  A SKIP IS A FAILURE.
    ("--measure-toolchain-gate", "abort"),
    # THE COST FORM (ticket .9): every terminal measurement also measures
    # rev-exact in the same actor, back to back, and slots 2 and 5 carry
    # -(log cost(candidate) - log cost(rev-exact)).  The ppo.py default
    # since 2026-09-05; named so the launcher's channel does not depend on
    # the env default.  The quality floor (--quality-floor tau, the hinge
    # that replaced ALPHAGRAD_QUALITY_GATE_MIN) is OFF here: raw quality is
    # the P0 form every phase-1 and phase-2 arm runs; the P1 and L arms of
    # phase 3 set it in `cli`.
    ("--cost-form", "paired-log"),
    # THE MEMORY CHANNEL (ticket .49): slot 5 holds the XLA static temp
    # bytes of the plan's own executable, the runtime watermark is logged
    # beside it (measure/mem_parity/*).  --mem-type peak_memory above still
    # selects WHICH slot --lambda-mem weights.
    ("--mem-channel", "temp"),
    # THE REDUCE AXIS SPACE (ticket .20): Reduce axes are physical val axes.
    # The ppo.py default; named so no arm inherits the pre-ticket read.
    # Ticket .18's sibling --face-slot-frames is GONE: each face slot always
    # decodes in its own tensor's frame, so there is nothing to name.
    ("--reduce-axis-space", "physical"),
    # GATE G1 (ticket .45): the sweep winners of .41 by (vertex, primitive,
    # kind).  ppo.py accepts an absent file (gate/g1/present = 0, one
    # [gate] line), so this names WHERE .41 must write and the pre-flight
    # below says whether the file is there yet.
    ("--gate-winners-table", GATE_WINNERS_TABLE),
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
    # The face head's output scale at init (CLEAN_DESIGN_AUDIT code-change
    # #1, ticket .44).  0 = off = the wave-1 head; the campaign arms below
    # set 0.1 (finding 51 D.1).  Named so the value is never inherited.
    ("--scale-face-head", "0"),
    ("--set-pointer", None),
    ("--face-actions", None),
    # ALWAYS ON with --face-actions (ppo.py only checks its precondition);
    # still accepted, and named so the launcher states the masking it runs
    # under rather than inheriting it.
    ("--per-face-masks", None),
    ("--unified-face-head", None),
    ("--live-faces", None),
    # The heads.py MicroActionPolicy (the ppo.py default; ppo.py REQUIRES it
    # beside --face-actions --unified-face-head --live-faces when the KL
    # trust region is on).  Named so the requirement is on the command line.
    ("--dynamic-substeps", None),
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


# (The `lossy` / `lossless` paired pair of ticket .56 -- `arm_per_approx_add`,
# one arm per value on the all-rev arm -- was removed 2026-09-13: the owner
# benched the learned join values and ruled ONE value, APPROX_ADD, everywhere.
# `campaign_arm` raises on any other value.)


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
export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_$(hostname -s)

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
#  * --face-none-bias (the env var ALPHAGRAD_FACE_NONE_BIAS until ticket
#    dsnn-3qm.44; args only now) -- ppo.py's --kl-ref-weight help calls it
#    "the contrast knob" in so many words.  A5's CORRECTED arithmetic (the factory's printed
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
        cli={"--face-none-bias": str(_bias), "--lambda-acc": str(_lam),
             "--name": _n.replace("_", "-")},
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

# Waves 2-4 inherit wave 1's winning NONE-bias through the shell variable
# W1_BIAS, passed as the --face-none-bias VALUE (it was the env var
# ALPHAGRAD_FACE_NONE_BIAS before dsnn-3qm.44; ppo.py now refuses that var).
W1_BIAS = '${W1_BIAS:?export W1_BIAS to wave 1 winning NONE-bias}'

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
    _cli = {"--name": _n.replace("_", "-"), "--episodes": "250",
            "--face-none-bias": W1_BIAS}
    _cli.update(_extra_cli)
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="24:00:00",
        gpus=4,
        env={},
        cli=dict(_cli, **{"--fixed-order": "reverse" if _rev == "1" else "free"}),
        depends="wave 1 (inherits its winning NONE-bias via $W1_BIAS)",
        purpose=f"""WAVE 2 -- THE ELIMINATION ORDER.  The axis with real range
(45-80x, against at most ~2x on the approximation axis) that has NEVER been
scheduled, because it was starved by the very credit horizon R2/R3 proved
causal.  Under --fixed-order reverse exactly one vertex is legal at
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

The order is the --fixed-order argument (ticket .64; previously an
import-time environment variable before 2026-09-05).

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
        cli={"--face-none-bias": W1_BIAS, "--lambda-acc": _lam,
             "--kl-ref-weight": _kl, "--name": _n.replace("_", "-")},
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
it CANNOT create CONTRAST -- "the contrast knob is --face-none-bias.  Sweep
them together."  This wave is the second half of that sweep, run only
after wave 1 has established that there IS drift worth bounding.

THIS ARM: {_what}.

NOTE the interaction the design record flags: do NOT run a low lambda without
the quality gate.  Removing ALPHAGRAD_QUALITY_GATE_MIN uncaps the full SKIP
prize -- symlog(157/70) ~ +0.81 for one action against lambda_acc * 0.885 --
so SKIP wins outright at lambda=1, is marginal at 4 and is priced out at 16.
(Historical: wave 3 was designed with the gate at 0.05 in every arm.  The
gate was DELETED 2026-09-04 by owner ruling, ticket dsnn-3qm.9; a regenerated
launcher carries --cost-form paired-log and no gate, and the low-lambda
warning above is answered by the phase-1 campaign arms' lambda_q, not here.)""",
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
            "--var-probe-steps": "16", "--face-none-bias": W1_BIAS}
    if _n.endswith("seed2"):
        _cli["--seed"] = "970520"
    arm(
        name=_n,
        job=_n.replace("_", "-"),
        kind="train",
        node=_node,
        time="12:00:00",
        gpus=4,
        cli=_cli,
        depends="waves 1-3 (inherits the winning bias and lambda)",
        purpose=f"""WAVE 4 -- THE FACE READ POINT.  --face-read is the CHEAPEST
UNRUN EXPERIMENT IN THE WHOLE BACKLOG: no extra encode, no extra scan, no new
parameters, no width change -- the pooled rows are already materialised and
only the mask moves.  All three modes are already pinned rollout == replay by
tests/face_read_point_test.py, so the PPO ratio is safe in every arm.

WHAT IT FIXES.  Face f's chunk is [approx-echo(f-1) || header+contraction(f)],
so chunk-mean produces the head's {FACE_HEAD_WIDTH} logits from a token-count-weighted blend
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


# ===========================  THE CAMPAIGN (tickets .50-.54)  ===============
#
# ONE ROW PER ARM (`campaign_arm(...)` calls below); the owner edits a row
# or one of the MVP constants and regenerates.  Every campaign arm: the TLM
# target, latency + static temp memory + grad-cosine (--mem-channel temp,
# --quality-metric grad_cosine, --cost-form paired-log), --discount 1.0
# --gae-lambda 1.0 --terminal-rewards-only --reward-mode additive
# --advantage-norm none (SHARED_CLI; the P4 and L rows override ONE of them
# by design), the static Markowitz order (--fixed-order markowitz, .64;
# `free` on the two order arms), --approx-add lossless (the ONE value),
# --face-none-bias 4, NO quality gate, ONE seed, 250 episodes, --plan-log
# auto, --ray-measure 1, the .45 gate telemetry (ppo.py emits it every
# episode, no switch; the launcher passes --gate-winners-table), one 8-GPU
# Blackwell job per node on pgi15-gpu19/20.  Phase order (finding 53 Q27;
# ticket .55: the Reduce, Quant and Diag arms waited for .16-.20, which have
# landed): SKIP-only -> Reduce -> Quant -> Diag -> order-only -> free; then
# phase 2 (channels), 3 (P0 -> P1 -> L), 4 (PopArt), 5 (five seeds).
#
# PHASE-1 TABLE (ticket .50, owner ruling 2026-09-13): SKIP-only,
# Reduce-only, Quant-only, Diag-only, order-only (the `none` profile of .40,
# the one spelling of --no-approx-head), free (order + all classes).  The
# all-rev x2 arms (the lossy / lossless pair of .56) are DROPPED: they
# depended on the learned join values, which are benched.  Diag-only is
# emitted as a live arm, not held: under the Markowitz order Diag is legal on
# rhs 112/131, new 113/131 and old 14/14 TLM sites (finding 59, ticket .64),
# which is the condition ticket .50 attached to it; ticket .25 (what Diag
# MEANS on a scalar loss) is still open and is named in the arm's DEPENDS.
#
# NAMES encode the phase, the profile, the order (free = pin lifted;
# markowitz is the default and unnamed), the channel set when not all three,
# and the price: lq<lambda_q> for a fixed lambda, pref for preference
# conditioning (P0), pref_tau<tau> for the floored P1, dual_tau<tau> for the
# Lagrangian L.  p1b_reduce_lq5 is phase 1, arm b, Reduce only, Markowitz
# order, lambda_q = 5.  (The face ADD is no longer in the name: there is one
# value.)
#
# INIT MVP (ticket .36 / finding 51 D.1; ALL TUNABLE, the owner's ruling):
# --face-none-bias 4, --scale-face-head 0.1, --face-logit-clamp 15,
# lambda_q 5 (the window is 5-6; Q30: B = 6 straddles tau = 0.9).
# ---------------------------------------------------------------------------

FACE_NONE_BIAS_MVP = "4"
SCALE_FACE_HEAD_MVP = "0.1"
FACE_LOGIT_CLAMP_MVP = "15"

# ---------------------------------------------------------------------------
# THE APPROVED REWARD (owner ruling 2026-09-13, from finding 63's measured
# absorber).  NOT the finding-51 window: that window (3.95-6.8) was measured
# on the REVERSE order with the WATERMARK memory channel, where the
# skip-everything absorber gains almost nothing.  On the Markowitz order with
# the static temp channel the absorber gains 9.8 nats, and finding 63 measures
# the contrast
#
#     contrast = R(best feasible) - max(R(absorber), R(baseline))
#
# at lambda_q = 8 -> -0.7, 12 -> +1.9, 16 -> +1.9, 32 -> +1.9 (P1 hinge,
# tau = 0.90, both channels floored at the rev-exact cost).  lambda_q = 16 is
# 1.33x the 12 the absorber needs.  EVERY phase-1 arm carries it, and the
# hinge with it: the `lq5` arms the generator emitted this morning passed
# --lambda-acc 5 and NO quality floor, which finding 63 prices at
# contrast -20.6 -- the absorber wins outright.
# ---------------------------------------------------------------------------
LAMBDA_Q_MVP = "16"
QUALITY_FLOOR_TAU = "0.90"
# Dual ascent (arm L only): lam <- clip(lam + eta*(violation - target), min, max).
DUAL_ETA = "2.0"
DUAL_LAMBDA_MIN = "12"
DUAL_LAMBDA_MAX = "32"
CAMPAIGN_SEED = "250197"

# ---------------------------------------------------------------------------
# GATE G6 (ticket .45): the pre-run offline contrast of ticket .42, by the
# order the arm runs on.  Finding 63, at the approved lambda_q = 16 / tau =
# 0.90 / P1 hinge with both channels floored at the rev-exact cost:
#
#   Markowitz  +1.9 nats, ~19 % of the absorber's 9.8-nat scale  -> 0.19
#   reverse     0.0 nats: there is nothing to gain on that order -> 0.00
#
# THE FLOOR THIS ASSUMES IS NOT IN THE TRAINER YET.  env._MEM_LOG_FLOOR_BYTES
# is still one byte, and --paired-cost-floor {byte,reference} is not in
# make_argparser as of alphagrad 0863bd4.  Under the one-byte floor the same
# configuration prices at contrast -13.4 on Markowitz.  The launchers carry
# the APPROVED number and the pre-flight below says so, so a run that lands
# before the floor change is not silently judged against a number its own
# reward cannot reach.
# ---------------------------------------------------------------------------
GATE_OFFLINE_CONTRAST = {"markowitz": "0.19", "free": "0.19", "reverse": "0.00"}

# The profiles of ticket .40 (ppo.py --approx-profile choices) and the orders
# of ticket .64 (common/order.py FIXED_ORDER_CHOICES).  Typed here because
# importing ppo.py costs a jax session; the campaign test cross-checks both
# against make_argparser so a drift is caught, not trusted.
PROFILES = ("all", "skip", "reduce", "quant", "diag", "none")
FIXED_ORDERS = ("markowitz", "reverse", "free")

# ---------------------------------------------------------------------------
# THE CAMPAIGN HARDWARE (owner ruling 2026-09-13).  One sbatch per node, all
# eight Blackwell GPUs of the node, on pgi15-gpu19 / pgi15-gpu20.  The trainer
# takes device 0 (--gpus 0, the ppo.py default) and the --ray-measure actor
# device 1 (ppo.py pins idx + 1 through RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_
# DEVICES); holding the node is what makes the timed executions mean anything
# (co-residency measured CV 0.0000% -> 49.7%).
# ---------------------------------------------------------------------------
CAMPAIGN_NODES = ("pgi15-gpu19", "pgi15-gpu20")
CAMPAIGN_GPUS = 8
CAMPAIGN_GRES = ("gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition"
                 f":{CAMPAIGN_GPUS}")
CAMPAIGN_CPUS = 128
CAMPAIGN_MEM = "800G"          # the nodes have 1.5 TB; 100 G per GPU
CAMPAIGN_RAY_MEASURE = "1"
CAMPAIGN_RAY_MEASURE_TIMEOUT = "600"

# ---------------------------------------------------------------------------
# THE CAMPAIGN STACK (finding 57).  The pgi15 GPU nodes mount NO home
# directory (/Users/assmuth is ENOENT on gpu15..20 and cpu2); /Scratch is the
# one writable filesystem every node and the head share.  So a campaign
# launcher does not `cd ~/dsnn/alphagrad` and does not `uv run`: it runs the
# relocated venv of finding 57 against alphagrad/graphax worktrees staged
# under CAMPAIGN_STACK, with a node-local $HOME that receives the two wandb
# credential files.  The pre-flight refuses (exit 66) when any of these is
# missing.  The owner stages the stack once per campaign commit:
#   git -C ~/dsnn/alphagrad worktree add --detach CAMPAIGN_STACK/alphagrad <sha>
#   git -C ~/dsnn/graphax   worktree add --detach CAMPAIGN_STACK/graphax   <sha>
# ---------------------------------------------------------------------------
CAMPAIGN_ROOT = "/Scratch/assmuth/campaign"
CAMPAIGN_STACK = f"{CAMPAIGN_ROOT}/stack"
CAMPAIGN_RUNS = f"{CAMPAIGN_ROOT}/runs"
CAMPAIGN_PY = "/Scratch/assmuth/t57/stack/venv/bin/python"
CAMPAIGN_WANDB_HOME = "/Scratch/assmuth/t57/home"     # .netrc + .config/wandb
CAMPAIGN_CACHE = "/Scratch/assmuth/mrg/cache"          # dsnn_wikitext, dsnn_mnist
# GATE G1 (ticket .45) on a node without a home: the sweep winners are read
# from /Scratch.  THE TABLE IS THE SWEEP64 ONE, BY ORDER (owner ruling
# 2026-09-13): 4029 rows at q >= 0.80 on Markowitz, 2969 on reverse.  The
# staged copy at {CAMPAIGN_ROOT}/sweep41/winners.csv is byte-for-byte the
# sweep64 MARKOWITZ table under a name that says sweep41 -- a reverse-order
# arm pointed at it would be scored against winners from the other order and
# report a recovery that means nothing.  One table per order, named here.
CAMPAIGN_GATE_WINNERS_TABLES = {
    "markowitz": "/Scratch/assmuth/sweep64/runs/markowitz/winners.csv",
    "reverse": "/Scratch/assmuth/sweep64/runs/reverse/winners.csv",
    # The free-order arms start from the Markowitz pin and the reward signal
    # lives on that order (finding 63); vertex ids are jaxpr equation indices
    # and do not depend on the elimination order, so the Markowitz winner set
    # is the meaningful one to recover.
    "free": "/Scratch/assmuth/sweep64/runs/markowitz/winners.csv",
}
CAMPAIGN_GATE_WINNERS_TABLE = CAMPAIGN_GATE_WINNERS_TABLES["markowitz"]

# ---------------------------------------------------------------------------
# THE CAMPAIGN ENVIRONMENT (owner ruling 2026-09-13: args only).  A campaign
# launcher exports EXACTLY three kinds of variable, and the campaign test
# refuses any export outside them:
#
#   CAMPAIGN_ENV      the TLM target shape (the target's size comes from these,
#                     not from --hidden-dim/--vocab-size/--num-layers, which
#                     size the POLICY) and the three measurement-plumbing
#                     variables the smoke runs under.  No XLA_*, no JAX_*.
#   NO_FLAG_ENV       knobs that CHANGE THE RUN and still have no flag in
#                     ppo.py.  Exported because the run is wrong or dead
#                     without them, and named in a TODO block in the launcher
#                     header so the departure from "args only" is never
#                     silent.  Each entry carries its evidence.
#   STACK_ENV         where the code, the data and the credentials are
#                     (finding 57).  Plumbing, not knobs.
# ---------------------------------------------------------------------------
CAMPAIGN_ENV = [
    ("ALPHAGRAD_TLM_SEQ", "32"),
    ("ALPHAGRAD_TLM_DMODEL", "128"),
    ("ALPHAGRAD_TLM_VOCAB", "1024"),
    # --ray-measure requires it (ppo.py: "Requires ALPHAGRAD_BATCHED_CALLBACK=1").
    ("ALPHAGRAD_BATCHED_CALLBACK", "1"),
    ("RAY_TMPDIR", "/tmp/ray_$SLURM_JOB_ID"),
    # Ray must leave the actor's CUDA_VISIBLE_DEVICES pin alone (ppo.py
    # exports it into the actor's runtime_env too; the driver copy is what
    # reaches Ray's worker startup).
    ("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1"),
]

# (var, value, evidence) -- see NO_FLAG_ENV above.  Promote to a flag and
# DELETE the row; the campaign test pins that every row is exported AND
# named in the header's TODO block.
NO_FLAG_ENV = [
    ("ALPHAGRAD_POLICY", "palimpsa",
     "the encoder backbone.  ppo.py make_argparser has no --policy flag; "
     "ppo.py reads os.environ ALPHAGRAD_POLICY with default 'transformer' "
     "and RAISES under --incremental-encode unless it is 'palimpsa' (the "
     "delta observation needs the causal carry).  README 'there is no CLI "
     "flag'."),
    ("ALPHAGRAD_SKIP_COST_ANALYSIS", "1",
     "skip Compiled.cost_analysis() per measurement: it leaks ~3.5 GB per "
     "episode of C++ HloCostAnalysis state (env.py), which over 250 "
     "episodes kills the job; the flops / bytes_accessed channels it feeds "
     "are LOGGED, never trained (--rewards cmp mem acc).  No flag exists."),
]

STACK_ENV_NAMES = ("HOME", "PYTHONPATH", "DSNN_WIKITEXT_DIR", "DSNN_MNIST_DIR",
                   "PATH")   # PATH: the measure toolchain block (finding 03)

#: Every `export NAME=` a campaign launcher may contain.  The test derives the
#: rendered set and asserts equality with this one.
CAMPAIGN_ENV_ALLOWED = frozenset(
    [k for k, _ in CAMPAIGN_ENV] + [k for k, _, _ in NO_FLAG_ENV]
    + list(STACK_ENV_NAMES))

# THE WINNERS.  None = not decided: the arm is emitted with a shell
# placeholder the owner exports at submit time (the W1_BIAS pattern) and
# "winner" in its name.  Set the constant and regenerate once the deciding
# phase has been read; the name then carries the real value.
P1_WINNER_PROFILE: str | None = None     # phase 1 decides; phases 2-5 use it
P2_WINNER_CHANNELS: str = "cmp mem acc"  # phase 2 decides; phases 3-5 use it
P3_WINNER_FORM: str = "fixed"            # phase 3 decides: fixed | P0 | P1 | L
FIVE_SEEDS = (CAMPAIGN_SEED, "970520", "31415", "27182", "16180")

_CHANNEL_TOKEN = {"cmp mem acc": "latmemq", "cmp acc": "latq", "mem acc": "memq"}
_FORMS = ("fixed", "P0", "P1", "L")
_P1_PLACEHOLDER = "${P1_PROFILE:?export P1_PROFILE to the phase-1 winning profile}"

CAMPAIGN_HEAD = f"""THE CAMPAIGN (tickets .50-.54) under the APPROVED REWARD
(owner ruling 2026-09-13, priced in finding 63): three trained channels --
paired log-difference latency, paired log-difference static temp memory (both
against rev-exact measured in the same actor, back to back; --cost-form
paired-log, --mem-channel temp) and the P1 HINGE on grad-cosine
(--quality-floor {QUALITY_FLOOR_TAU}, so slot 6 carries -max(0, tau - q)) --
weighted --lambda-cmp 1 --lambda-mem 1 --lambda-acc {LAMBDA_Q_MVP}.  Terminal
rewards only, gamma = GAE lambda = 1, additive, raw advantages, classic init
with the MVP face-head init (--face-none-bias {FACE_NONE_BIAS_MVP},
--scale-face-head {SCALE_FACE_HEAD_MVP}, --face-logit-clamp
{FACE_LOGIT_CLAMP_MVP}), the static Markowitz order (--fixed-order markowitz,
ticket .64) unless the row lifts it, the face ADD --approx-add {APPROX_ADD}
(the one value), one seed, 250 episodes.

WHY lambda_q = {LAMBDA_Q_MVP} AND NOT 5.  Finding 63 measured the
skip-everything absorber on THIS order and THIS channel: it gains 9.8 nats,
and the contrast R(best feasible) - max(R(absorber), R(baseline)) is -20.6 at
lambda_q = 8 without a floor, -0.7 at 8 with one, +1.9 from 12 up.  The
finding-51 window (3.95-6.8) was measured on the REVERSE order with the
WATERMARK channel, where the absorber gains nothing; it does not transfer.
{LAMBDA_Q_MVP} is 1.33x the 12 the Markowitz absorber needs.

THE COST FLOOR THIS ASSUMES IS NOT IN THE TRAINER YET.  Finding 63's numbers
floor BOTH cost channels at the paired rev-exact cost before the log;
env._MEM_LOG_FLOOR_BYTES is still one byte and there is no
--paired-cost-floor flag in make_argparser.  Under the one-byte floor the
same configuration prices at contrast -13.4 on Markowitz -- the absorber
wins.  Do not read a phase-1 result as a reward-design result until that
lands (ticket .9).

THE FACE HEAD is one flat MLP of {FACE_HEAD_WIDTH} logits (derived from
alphagrad.approx.unified_face_head.head_layout({APPROX_ADD!r}).width at
generation time, never typed): three {(FACE_HEAD_WIDTH - 1) // 3}-wide slot
blocks, each with a four-way Quant dtype softmax over
{", ".join(FACE_QUANT_DTYPES)} (the operand's own dtype is masked, ticket
.40 D4); no flag selects the dtype set.

MEASUREMENT: --ray-measure {CAMPAIGN_RAY_MEASURE} actor on its own GPU
(timeout {CAMPAIGN_RAY_MEASURE_TIMEOUT} s), ALPHAGRAD_BATCHED_CALLBACK=1,
all {CAMPAIGN_GPUS} GPUs of the node held by this job.

THE GATE G1-G6 TELEMETRY of ticket .45 (paired/*, gate/g1..g6/*, measure/*)
has NO switch: ppo.py's host_log computes it from the drained plan records
every episode (alphagrad.approx.common.gate_telemetry.episode_fields; field
table docs/GATE_TELEMETRY.md).  The launcher supplies the TWO inputs the
trainer cannot measure for itself: --gate-winners-table (G1, the sweep64
table for THIS arm's order) and --gate-offline-contrast (G6, finding 63's
pre-run number for this reward on this order).  measure/drain/* audits that
the plans the measure actors counted are the plans these fields were
computed from (ticket .7): measure/drain/ok = 0 means a counter is being
read in a process that does not own it and every panel understates."""

CAMPAIGN_P1_PREDICTION = """REGISTERED BEFORE THE RUN (finding 51 D.1, ticket .50);
NEVER EDITED AFTERWARDS.  Episode-0 survivors (q > 0 plans) >= 75 percent of
plans; a plan with paired latency ratio <= 0.6 at q >= 0.9 by ep15; that plan
HELD (present in >= 20 percent of plans) at ep100 and at ep250.  Fail =
identity drift, which confirms the objective, not the init, as the blocker."""


class CampaignRowError(ValueError):
    """A campaign row asks for something the owner's rulings forbid or the
    trainer does not have.  Raised, never rendered: a launcher that quietly
    drops or substitutes a flag is the failure this generator exists to end."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CampaignRowError(msg)


def campaign_arm(*, phase: int, tag: str, profile: str, node: str, what: str,
                 prediction: str, falsifier: str,
                 order: str = "markowitz", approx_add: str = APPROX_ADD,
                 rewards: str = "cmp mem acc", form: str = "fixed",
                 lambda_q: str = LAMBDA_Q_MVP, advantage_norm: str = "none",
                 seed: str = CAMPAIGN_SEED, time: str | None = None,
                 held: str | None = None, depends: str | None = None) -> dict:
    """One row of the campaign table -> one `arm(...)`.  Returns the arm.

    Raises :class:`CampaignRowError` on a row outside the rulings: an order
    not in FIXED_ORDERS, a profile not in PROFILES (or WINNER), a face ADD
    other than APPROX_ADD, a node off the campaign nodes, an unknown reward
    form / channel set / advantage normalisation.
    """
    _require(order in FIXED_ORDERS,
             f"order {order!r} is not one of {FIXED_ORDERS} (ticket .64)")
    _require(approx_add == APPROX_ADD,
             f"--approx-add {approx_add!r} is not allowed: owner ruling "
             f"2026-09-13 runs every arm on {APPROX_ADD!r} only (the learned "
             f"join values are benched, the lossy/lossless pair is dropped)")
    _require(profile == "WINNER" or profile in PROFILES,
             f"profile {profile!r} is not one of {PROFILES} or WINNER (.40)")
    _require(form in _FORMS, f"form {form!r} is not one of {_FORMS}")
    _require(rewards in _CHANNEL_TOKEN,
             f"rewards {rewards!r} is not one of {tuple(_CHANNEL_TOKEN)}")
    _require(advantage_norm in ("none", "popart"),
             f"advantage_norm {advantage_norm!r} is not none|popart")
    _require(node in CAMPAIGN_NODES,
             f"node {node!r} is not one of {CAMPAIGN_NODES} (one 8-GPU "
             f"Blackwell job per node, owner ruling 2026-09-13)")
    _require(bool(what and prediction and falsifier),
             "every campaign row carries what, prediction and falsifier")
    if profile == "WINNER":
        prof_tok = P1_WINNER_PROFILE or "winner"
        prof_val = P1_WINNER_PROFILE or _P1_PLACEHOLDER
    else:
        prof_tok = prof_val = profile
    tau_tok = QUALITY_FLOOR_TAU.replace(".", "").rstrip("0") or "0"
    # THE NAME STATES THE REWARD IT RUNS.  `fixed` is no longer "raw quality
    # at lambda_q": it is the approved P1 HINGE at tau, so it says so.  An arm
    # named lq5 that passes --lambda-acc 16, or named lq16 with no floor,
    # is the drift this token exists to prevent.
    lam_tok = {"fixed": f"hinge_tau{tau_tok}_lq{lambda_q}",
               "P0": f"pref_lq{lambda_q}",
               "P1": f"pref_tau{tau_tok}_lq{lambda_q}",
               "L": f"dual_tau{tau_tok}_eta{DUAL_ETA.replace('.', '')}"}[form]
    name = f"p{phase}{tag}_{prof_tok}"
    if order != "markowitz":
        name += f"_{order}"
    if rewards != "cmp mem acc":
        name += f"_{_CHANNEL_TOKEN[rewards]}"
    name += f"_{lam_tok}"
    if advantage_norm != "none":
        name += f"_{advantage_norm}"
    if seed != CAMPAIGN_SEED:
        name += f"_s{seed}"
    job = name.replace("_", "-")
    cli: dict = {
        "--name": job,
        "--seed": seed,
        "--approx-profile": prof_val,
        # The fixed order (ticket .64): static minimum Markowitz degree for
        # every fixed-order arm, reverse only as the control (.60), free
        # for the order arms.  One table, common/order.py.
        "--fixed-order": order,
        "--approx-add": approx_add,
        "--face-none-bias": FACE_NONE_BIAS_MVP,
        "--scale-face-head": SCALE_FACE_HEAD_MVP,
        "--face-logit-clamp": FACE_LOGIT_CLAMP_MVP,
        "--rewards": rewards,
        # THE APPROVED CHANNEL WEIGHTS (owner ruling 2026-09-13, finding 63).
        "--lambda-cmp": "1",
        "--lambda-mem": "1",
        "--lambda-acc": lambda_q,
        # THE P1 HINGE, ON EVERY ARM.  Reward slot 6 carries
        # -max(0, tau - q) instead of raw q (ticket .9).  This is the form
        # finding 63 priced: without it the skip-everything absorber outscores
        # every honest plan by 20+ nats and the arm learns to allocate
        # nothing.  Phase 3's P0 arm is the ONE arm that lifts it, and it
        # lifts it on purpose, as its question.
        "--quality-floor": QUALITY_FLOOR_TAU,
        # THE MEASUREMENT (owner ruling 2026-09-13): one Ray measure actor
        # on its own GPU, the node held by this job.
        "--ray-measure": CAMPAIGN_RAY_MEASURE,
        "--ray-measure-timeout": CAMPAIGN_RAY_MEASURE_TIMEOUT,
        # THE GATE .45 INPUTS, the two the trainer cannot measure for itself.
        # G1's winners table is inherited from SHARED_CLI and resolved from
        # this arm's --fixed-order by `_merge_cli` (ONE mechanism, so a wave
        # arm cannot get a different rule from a campaign arm); it lives on
        # /Scratch because the GPU nodes mount no home (finding 57).
        # G6's pre-run number (finding 63), so the run carries the contrast
        # it is judged against instead of reading NaN.
        "--gate-offline-contrast": GATE_OFFLINE_CONTRAST[order],
    }
    if form in ("P0", "P1", "L"):
        cli["--preference-conditioned"] = None
    if form == "P0":
        # THE ONE ARM WITHOUT THE FLOOR: P0 is raw quality by definition
        # (finding 63's P0 form), and phase 3 exists to compare it against
        # the hinge.  Deleted rather than overwritten so the launcher does
        # not carry a flag whose value contradicts its name.
        cli["--quality-floor"] = _DELETE
    if form == "L":
        # Lambda by dual ascent; --lambda-acc is ignored in this mode and
        # --quality-floor sets --lag-tau.  eta 2.0 with lambda in [12, 32]
        # (owner ruling 2026-09-13): one episode of full violation
        # (q = 0 at tau = 0.90) raises lambda by 1.8, and the floor of 12 is
        # the smallest weight that puts the Markowitz absorber below the
        # baseline (finding 63).
        cli["--reward-mode"] = "lagrangian"
        cli["--lag-eta"] = DUAL_ETA
        cli["--lag-min"] = DUAL_LAMBDA_MIN
        cli["--lag-max"] = DUAL_LAMBDA_MAX
        cli["--lag-init"] = lambda_q
    if advantage_norm == "popart":
        # The recorded trap (ticket .53): --no-symlog must be set with
        # PopArt, and the three symlog sites must agree.  --symlog-channels
        # none IS --no-symlog; both are passed and ppo.py checks they agree.
        cli["--advantage-norm"] = "popart"
        cli["--no-symlog"] = None
        cli["--symlog-channels"] = "none"
    # No per-arm env: the campaign environment is CAMPAIGN_ENV + NO_FLAG_ENV
    # + the stack plumbing, rendered by `render` for runtime "scratch", and
    # nothing else (owner ruling 2026-09-13: args only).
    env: dict = {}
    a = dict(
        name=name, job=job, kind="train", runtime="scratch", node=node,
        time=time or ("24:00:00" if order == "free" else "12:00:00"),
        gpus=CAMPAIGN_GPUS, env=env, cli=cli, phase=phase,
        purpose=CAMPAIGN_HEAD + f"\n\nPHASE {phase}, ARM {name}: {what}",
        prediction=prediction, falsifier=falsifier,
    )
    if held:
        a["held"] = held
    if depends:
        a["depends"] = depends
    arm(**a)
    return a


def campaign_arms() -> list[dict]:
    return [a for a in ARMS if a.get("phase")]


# ---------------------------  PHASE 1 (ticket .50)  --------------------------
# Class ablation, one seed each, on the static Markowitz order (ticket .64),
# so ONLY the approximations are learned -- except the two order arms, which
# lift the pin.  THE TABLE IS THIS TUPLE: (tag, profile, order, what,
# prediction, falsifier, depends); `campaign_arm` turns each row into an arm,
# and the nodes alternate gpu19 / gpu20 by row.  The campaign test pins the
# rendered arm list against this tuple's tags and profiles.
_P1_FALSIFIER = """If the arm drifts to identity (approx_prob/none > 0.99 and no
plan outside the drift floor by ep100) or collapses to q = 0 for the majority
of plans, the CLASS is not where the objective's contrast lives; say so, do
not retune the init on this arm."""

_P1_FIDELITY_DEP = ("ticket .55: the fidelity fixes .17-.20 (landed on the "
                    "integration branch)")

PHASE1_TABLE = (
    ("a", "skip", "markowitz",
     """SKIP-ONLY (--approx-profile skip).  The first arm of the campaign
and the class the archived wins belong to: every archived winner at ratio
0.52-0.58 carries ONE face wire and zero applied rules (the win IS a skip).
Reduce, Quant and Diag are masked; the face head chooses skip / none.""",
     CAMPAIGN_P1_PREDICTION + """
  * this arm finds the one-face SKIP band (1-15 skips per plan) and holds a
    plan at paired latency ratio <= 0.6 with q >= 0.9; the static temp ratio
    stays at 1.0 within the drift floor (a skip does not move XLA temp on
    TLM; finding 05, re-read under the Markowitz order by .63).""",
     _P1_FALSIFIER,
     None),
    ("b", "reduce", "markowitz",
     """REDUCE-ONLY (--approx-profile reduce).  The class where the memory
saving is (finding 05: 110/114 applied, 0.79x XLA temp on TLM).  Reduce axes
are physical val axes decoded per slot (--reduce-axis-space physical;
tickets .18, .20).""",
     CAMPAIGN_P1_PREDICTION + """
  * this is the arm that moves the TEMP channel: a plan at
    paired/temp_ratio_best <= 0.8 with q >= 0.9 by ep50, and the latency
    ratio inside 1.0 +/- the drift floor (Reduce saves bytes, not time).""",
     _P1_FALSIFIER + """
If paired/temp_ratio_best never leaves the drift floor over 250 episodes,
the temp channel has no reachable contrast under the Markowitz pin and phase
2's memory+q arm is CANCELLED as answered.""",
     _P1_FIDELITY_DEP),
    ("c", "quant", "markowitz",
     f"""QUANT-ONLY (--approx-profile quant).  Four target dtypes
({", ".join(FACE_QUANT_DTYPES)}); the operand's own dtype is illegal
(ticket .40 D4), so every Quant action changes the stored dtype.  Pullup:
the approximation itself pays, not bf16-native compute.""",
     CAMPAIGN_P1_PREDICTION + """
  * NO q = 0 plan in this arm (quant@all reads q ~ 0.93, finding 51 D.2);
    the best plan is a mild latency win (ratio ~ 0.95) at q > 0.9, and the
    temp ratio moves below 1 (a narrower dtype halves or quarters the
    stored bytes it touches).""",
     _P1_FALSIFIER + """
If gate/g4/q_zero_frac > 0.1 in this arm, a Quant rule destroys the
gradient on some face: that is a fidelity defect (ticket .16 class), not a
class result, and the arm is stopped and the plan log handed to .16.""",
     _P1_FIDELITY_DEP),
    ("d", "diag", "markowitz",
     """DIAG-ONLY (--approx-profile diag).  Under the static Markowitz order
Diag has an out-primal pair on rhs 112/131, new 113/131 and old 14/14 TLM
sites (finding 59) -- the condition ticket .50 attached to this arm, which
the reverse order never met (findings 52, 54: lhs only, 23 free sites).
Read for whether a Diag on a STORED slot moves temp at a quality price.""",
     CAMPAIGN_P1_PREDICTION + """
  * Diag on stored slots moves paired/temp_ratio_* below 1 on some plans
    (a block-diagonal stores fewer cells) at a quality price that keeps the
    best plan under q = 0.9; gate/g1/recovery_diag is defined (not NaN)
    because the Markowitz sweep of .63 has Diag winners to recover.""",
     _P1_FALSIFIER + """
If no Diag action changes temp or quality outside the drift floor on any
stored slot, Diag is inert on TLM under this order too and .25 is answered
by measurement: keep it in the action space as a documented no-op.""",
     "ticket .25 (what Diag MEANS on a scalar loss: still open; this arm "
     "measures it under the Markowitz order) and .55"),
    ("e", "none", "free",
     """ORDER-ONLY (--approx-profile none; the pointer head live,
--fixed-order free).  --approx-profile none removes the
approximation heads (the one spelling of --no-approx-head): the pure
elimination-order control.  The axis with real range: across orders the
temp channel spans ~80x while under a pin every archived winner reads
1.0000 (wave 2 text).  Each distinct order pays its own rev-exact pairing,
hence 24 h.""",
     """REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * at least one plan whose paired latency ratio against rev-exact is below
    0.95 and whose temp ratio is below 0.5 by ep100 -- or, if no order beats
    rev-exact on TLM, every plan sits at or above 1.0 on both channels and
    the policy converges to one order (gate/g5/n_rev_exact rises toward 16
    per episode).""",
     """If no plan beats rev-exact outside the drift floor over 250
episodes, reverse is optimal-or-unbeatable-by-this-policy on TLM and the
order axis is CLOSED for this campaign; the free arm p1f is then read as
the class arms plus noise, not as a two-lever result.""",
     None),
    ("f", "all", "free",
     """FREE (order + all classes; --approx-profile all, the pin
lifted).  Both levers.  Read ONLY against the single-class arms (same
order pin, one class each) and p1e (same order freedom, no classes); every
other pair is a two-knob difference.""",
     """REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * p1f does NO BETTER than the better of the best class arm and p1e on
    either paired channel: the approximation wins are attached to faces of
    one elimination order and do not survive reordering, and reordering
    pays a face count ~2.7x higher (115 -> ~313 faces) for the same rules.""",
     """If p1f beats BOTH halves outside the drift floor on the same
plans, order and approximation compose and the phase-2 channel arms run on
the free profile rather than the pinned one; say so before phase 2 starts.""",
     "p1a-p1d (the class arms) and p1e (order-only): the halves it is read "
     "against"),
)

for _i, (_tag, _prof, _order, _what, _pred, _fals, _dep) in enumerate(PHASE1_TABLE):
    campaign_arm(
        phase=1, tag=_tag, profile=_prof, order=_order,
        node=CAMPAIGN_NODES[_i % len(CAMPAIGN_NODES)],
        what=_what, prediction=_pred, falsifier=_fals, depends=_dep,
    )

# ---------------------------  PHASE 2 (ticket .51)  --------------------------
# Channel arms on the phase-1 winner: each cost channel alone with quality,
# before the three-channel preference conditioning is asked to amortize them.
for _tag, _rewards, _node, _what in (
    ("a", "cmp acc", CAMPAIGN_NODES[0],
     "LATENCY + QUALITY only (--rewards cmp acc): the memory head is off."),
    ("b", "mem acc", CAMPAIGN_NODES[1],
     "MEMORY + QUALITY only (--rewards mem acc): the latency head is off."),
):
    campaign_arm(
        phase=2, tag=_tag, profile="WINNER", rewards=_rewards, node=_node,
        depends="phase 1 (P1_WINNER_PROFILE, or export P1_PROFILE at submit)",
        what=_what + """  Purpose: show that this channel alone produces a
front that differs from rev-exact by more than the drift floor.""",
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * the trained channel's paired/*_ratio_best leaves the drift floor by
    ep50 and the untrained channel's does not move in the same direction
    (it is not being paid for); gate/g4/q_zero_frac stays below 0.1.""",
        falsifier="""If the trained channel's best paired ratio never leaves the
drift floor over 250 episodes, that channel has no contrast at this lambda
on the winning profile and phase 3 runs on the other channel alone.""",
    )

# ---------------------------  PHASE 3 (ticket .52)  --------------------------
# The reward-form ladder P0 -> P1 -> L on the winning channel set (.12
# decides after these run).
for _tag, _form, _node, _what in (
    ("a", "P0", CAMPAIGN_NODES[0],
     "P0: --preference-conditioned with RAW quality (3-D front; the Dirichlet "
     "preference over the three heads drives the advantage weighting)."),
    ("b", "P1", CAMPAIGN_NODES[1],
     f"P1: --preference-conditioned --quality-floor {QUALITY_FLOOR_TAU} (2-D "
     "front; slot 6 is the hinge -max(0, tau - q), so the third weight "
     "prices violations only)."),
    ("c", "L", CAMPAIGN_NODES[0],
     f"L: --reward-mode lagrangian --preference-conditioned --quality-floor "
     f"{QUALITY_FLOOR_TAU}: the Dirichlet runs over (latency, memory) only "
     "and lambda, the dual variable ascended once per episode on the mean "
     "violation, IS the quality weight."),
):
    campaign_arm(
        phase=3, tag=_tag, profile="WINNER", rewards=P2_WINNER_CHANNELS,
        form=_form, node=_node,
        depends="phases 1-2 (P1_WINNER_PROFILE, P2_WINNER_CHANNELS)",
        what=_what,
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * P0: gate/g5/spread_lat and spread_temp exceed the drift floor (the
    corners of the simplex reach different plans); the quality corner parks
    at rev-exact.
  * P1: gate/g4/q_ge_tau_frac rises toward 1 and the cost corners move
    further than P0's at q >= tau (the floor frees the price of quality
    above tau).
  * L: lambda settles (lag/lambda stops moving by ep100) with the mean
    violation near --lag-target, and the front at q >= tau matches P1's
    within the drift floor -- dual ascent finds the price P1 fixes by
    hand.""",
        falsifier="""If P1's feasible fraction does not rise above P0's, the
floor is not doing work at tau = 0.9 and .12 rules P0. If L's lambda pins at
--lag-max with the violation still above target, the constraint is
unsatisfiable at this init and L is out.""",
    )

# ---------------------------  PHASE 4 (ticket .53)  --------------------------
# PopArt re-test on the phase-3 winner.  Off by default, re-tested LAST.
campaign_arm(
    phase=4, tag="a", profile="WINNER", rewards=P2_WINNER_CHANNELS,
    form=P3_WINNER_FORM, advantage_norm="popart", node=CAMPAIGN_NODES[0],
    depends="phase 3 (P3_WINNER_FORM); its raw-advantage twin is the phase-3 winner itself",
    what="""POPART RE-TEST (--advantage-norm popart --no-symlog
--symlog-channels none) on the phase-3 winner.  The recorded trap: --no-symlog
must be set with PopArt and the three symlog sites must agree (memory
project_symlog_vs_popart); ppo.py refuses --no-symlog against --symlog-channels
cost, so both are set to none here.""",
    prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS (finding 51
C.5, QCI sec 14.2 and 14.7: the normalised-advantage runs were the collapse
group and raw advantages the group that held).
  * PopArt does NOT keep the held plan: it drifts to identity or collapses to
    q = 0 while the raw-advantage twin holds.  PopArt stays off.""",
    falsifier="""If the PopArt run keeps the winner's held plan (same ep100 /
ep250 criterion as phase 1) while its raw-advantage twin does too, PopArt is
harmless here and may be re-enabled for phase 5; if it holds and the twin
does not, the finding-51 reading is wrong and must be re-stated.""",
)

# ---------------------------  PHASE 5 (ticket .54)  --------------------------
# Five seeds on the winner, reported as a distribution of paired ratios.
for _i, _seed in enumerate(FIVE_SEEDS):
    campaign_arm(
        phase=5, tag="abcde"[_i], profile="WINNER", rewards=P2_WINNER_CHANNELS,
        form=P3_WINNER_FORM, seed=_seed, node=CAMPAIGN_NODES[_i % len(CAMPAIGN_NODES)],
        depends="phases 1-4 (the winner constants); one node per seed, one ppo job per node",
        what=f"""FIVE SEEDS, seed {_seed} ({_i + 1} of 5).  The distribution
claim: no result of this campaign is reported as a distribution before all
five have run (AGENTS.md).  Paired ratios against rev-exact, never raw
numbers.""",
        prediction="""REGISTERED BEFORE THE RUN; NEVER EDITED AFTERWARDS.
  * the held plan of the winner reproduces in >= 4 of 5 seeds (a plan at
    paired latency ratio <= 0.6 with q >= 0.9 held at ep250); the
    across-seed spread of paired/lat_ratio_best is larger than the drift
    floor (seeds matter) and smaller than the win (the win survives).""",
        falsifier="""If fewer than 3 of 5 seeds hold the plan, the phase-3
winner was one seed's luck and the campaign reports NO distribution claim;
the front handed to the benchmark harness (.14) is then the per-seed table,
not a summary.""",
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
 --quality-metric grad_cosine --approx-add @APPROX_ADD@ --walk-steps 200 \
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
   .replace("@HOME_DSNN@", HOME_DSNN).replace("@APPROX_ADD@", APPROX_ADD)


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
        # K=1 is the settled grad-cosine variant (949f1af): most predictive
        # AND cheapest.  Relevant here because this arm's quality channel IS
        # grad_cosine.
        "ALPHAGRAD_GRAD_COSINE_K": "1",
        # --approx-add is APPROX_ADD (lossless), the one value every arm
        # runs.  The hand-written predecessor forced the old edge EXACT
        # because the PINNED graphax 4ea0bf8 rejects the res-slot two-op
        # form; the live graphax accepts it (e5fd46c) and the two-op
        # pre-flight above VERIFIES that before any phase runs.  It is
        # inert for QB in any case -- a SKIP-only plan writes no rule into
        # any slot, so there is no approximated contraction for the ADD to
        # reconcile against.
        #
        # ---- DROPPED FROM THE TRAINING STACK -----------------------------
        # SHARED_ENV describes ppo.py.  One of these does not merely add noise
        # to a measurement arm, it CHANGES THE NUMBER, and the hand-written
        # launcher this arm replaces did not set it:
        #
        # (QUALITY_GATE_MIN used to be the second: at 0.05 the additive
        #   quality gate FLOORED latency_ns and peak_memory at the exact-rev
        #   reference for any plan scoring below 0.05, and a singleton SKIP
        #   sweep exists to price exactly those plans.  The variable and the
        #   gate were deleted on 2026-09-04, ticket dsnn-3qm.9.)
        # BATCHED_CALLBACK: defaults to 0.  At 1 env._callback takes the
        #   batched host path -- a different measurement path from the one
        #   every archived row was measured on.
        #
        # The rest are trainer-only and have no meaning here: there is no
        # policy, no actor pool and no episode loop in landscape_map.
        "ALPHAGRAD_BATCHED_CALLBACK": _DELETE,
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
    # G1's table follows the arm's own order (see GATE_WINNERS_TABLE).  Done
    # after the merge so the order an arm OVERRIDES is the one that decides.
    order = dict(merged).get("--fixed-order")
    if any(isinstance(v, _ByOrder) for _, v in merged):
        try:
            table = CAMPAIGN_GATE_WINNERS_TABLES[order]
        except KeyError:
            raise ValueError(
                f"--fixed-order {order!r} has no gate G1 winners table. "
                f"Known: {sorted(CAMPAIGN_GATE_WINNERS_TABLES)}. An arm whose "
                f"order has no sweep must not silently inherit another "
                f"order's winners -- G1 would report a recovery against "
                f"opportunities this run never had.") from None
        merged = [(f, table if isinstance(v, _ByOrder) else v)
                  for f, v in merged]
    return merged


def cli_tokens(a: dict) -> list[str]:
    """The ppo.py command line of a training arm as argv tokens.

    The same (flag, value) list the ARGS array is rendered from, so a test
    can run it through ``make_argparser`` exactly as the launcher's Layer-2
    pre-flight does.  Values that are shell placeholders (``${W1_BIAS:?..}``)
    come back verbatim; the caller substitutes.
    """
    toks: list[str] = []
    for flag, val in _merge_cli(a.get("cli", {})):
        toks.append(flag)
        if val is None:
            continue
        val = str(val)
        if val.startswith("${") and val.endswith("}"):
            # one shell word once expanded (the :? message is never a value)
            toks.append(val)
        else:
            toks.extend(val.split())   # "--rewards cmp mem acc" is 3 words
    toks.extend(WANDB.split())
    return toks


def is_scratch(a: dict) -> bool:
    """True for an arm that runs on the /Scratch stack of finding 57 (every
    campaign arm); False for the home-directory runtime of the wave arms."""
    return a.get("runtime", "home") == "scratch"


def _python(a: dict) -> str:
    """The interpreter invocation of this arm's runtime."""
    return "$PY" if is_scratch(a) else "uv run --no-sync python"


def _repo_paths(a: dict) -> tuple[str, str]:
    """(alphagrad checkout, graphax checkout) the launcher runs against."""
    if is_scratch(a):
        return f"{CAMPAIGN_STACK}/alphagrad", f"{CAMPAIGN_STACK}/graphax"
    return "~/dsnn/alphagrad", "~/dsnn/graphax"


def _scratch_stack_block() -> list[str]:
    """The environment of a campaign arm: the stack, the plumbing, the TLM
    shape, the measurement vars, the no-flag knobs.  Nothing else."""
    L = [
        "# ---------------------- THE STACK (finding 57) ------------------------",
        "# The pgi15 GPU nodes mount NO home directory; /Scratch is the one",
        "# filesystem every node and the head share.  66 = the stack, the venv,",
        "# the data cache or the wandb credentials are not staged on it.",
        f"PY={CAMPAIGN_PY}",
        f'for P in "$PY" {CAMPAIGN_STACK}/alphagrad/src/alphagrad/approx/ppo.py \\',
        f"         {CAMPAIGN_STACK}/graphax/src/graphax {CAMPAIGN_WANDB_HOME}/.netrc \\",
        f"         {CAMPAIGN_CACHE}/dsnn_wikitext; do",
        '  [ -e "$P" ] || { echo "ABORT(66): $P does not exist -- stage the'
        ' campaign stack (finding 57; CAMPAIGN_STACK in'
        ' tools/gen_fq_launchers.py) before submitting"; exit 66; }',
        "done",
        "# A node-local HOME with the two wandb credential files copied in",
        "# (finding 57 sec 3): wandb online needs .netrc, nothing else lives here.",
        "export HOME=/tmp/fq_home_$SLURM_JOB_ID",
        'mkdir -p "$HOME"',
        f'cp -r {CAMPAIGN_WANDB_HOME}/. "$HOME/"',
        'chmod 600 "$HOME/.netrc"',
        f"export PYTHONPATH={CAMPAIGN_STACK}/graphax/src:{CAMPAIGN_STACK}/alphagrad/src",
        f"export DSNN_WIKITEXT_DIR={CAMPAIGN_CACHE}/dsnn_wikitext",
        f"export DSNN_MNIST_DIR={CAMPAIGN_CACHE}/dsnn_mnist",
        f"cd {CAMPAIGN_STACK}/alphagrad",
        "",
        "# ---------------------- THE ENVIRONMENT (args only) -------------------",
        "# Owner ruling 2026-09-13: every knob is an ARGUMENT.  Exported here:",
        "# the TLM TARGET shape (the target's size comes from these, not from",
        "# --hidden-dim/--vocab-size/--num-layers, which size the POLICY) and",
        "# the Ray / measurement plumbing.  No XLA_*, no JAX_*; the campaign",
        "# test refuses any export outside CAMPAIGN_ENV_ALLOWED.",
    ]
    for k, v in CAMPAIGN_ENV:
        L.append(f"export {k}={v}")
    L.append("")
    L.append("# TODO (ticket .43): knobs that still have NO FLAG in ppo.py -- named")
    L.append("# in the header's TODO block with their evidence; promote and delete.")
    for k, v, _why in NO_FLAG_ENV:
        L.append(f"export {k}={v}")
    return L


def render(a: dict) -> str:
    kind = a["kind"]
    gpus = a.get("gpus", 0)
    scratch = is_scratch(a)
    py = _python(a)
    ag_repo, gx_repo = _repo_paths(a)
    L = ["#!/bin/bash"]
    L.append("#SBATCH -p " + ("pgi15-cpu" if kind == "cpu" else "pgi15"))
    L.append(f"#SBATCH -w {a['node']}")
    if scratch:
        # THE CAMPAIGN HARDWARE: the whole Blackwell node, by its gres name.
        L.append(f"#SBATCH --gres={CAMPAIGN_GRES}")
        L.append(f"#SBATCH -c {CAMPAIGN_CPUS}")
        L.append(f"#SBATCH --mem={CAMPAIGN_MEM}")
    elif gpus:
        L.append(f"#SBATCH --gres=gpu:{gpus}")
        L.append("#SBATCH -c 64")
        L.append("#SBATCH --mem=400G")
    else:
        L.append("#SBATCH -c 8")
        L.append("#SBATCH --mem=64G")
    L.append(f"#SBATCH -t {a['time']}")
    L.append(f"#SBATCH -J {a['job']}")
    if scratch:
        # -D and -o on /Scratch: a launcher whose -o names the missing home
        # fails at launch with ExitCode 0:53 (finding 57).  CAMPAIGN_RUNS
        # must exist before sbatch (slurm opens the log first): the owner
        # creates it once when staging the stack.
        L.append(f"#SBATCH -D {CAMPAIGN_STACK}/alphagrad")
        L.append(f"#SBATCH -o {CAMPAIGN_RUNS}/{a['name']}_%j.log")
    else:
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
    if a.get("held"):
        L.append("#")
        L.append("# *** HELD -- NOT TO BE SUBMITTED UNTIL THE OWNER RELEASES IT ***")
        L.append(_wrap_comment(a["held"], "#   "))
    if scratch and NO_FLAG_ENV:
        # The departure from "args only", stated where the owner reads.
        L.append("#")
        L.append("# *** TODO (ticket .43): ENV VARS WITHOUT A FLAG -- exported, never silent ***")
        L.append("#   Owner ruling 2026-09-13: args only.  These knobs have no flag in")
        L.append("#   ppo.py make_argparser yet and the run is wrong or dead without them,")
        L.append("#   so the launcher exports them HERE and says so.  Promote each to a")
        L.append("#   flag (the .44 pattern) and delete its NO_FLAG_ENV row.")
        for k, v, why in NO_FLAG_ENV:
            L.append(f"#   {k}={v}")
            L.append(_wrap_comment(why, "#       "))
    L.append("# " + "=" * 72)
    L.append("#")
    L.append("# GENERATED BY tools/gen_fq_launchers.py -- DO NOT EDIT IN PLACE.")
    L.append("# fq_v58_tlm_env16.sbatch was edited while its job was pending and")
    L.append("# job 61494's command line is unrecoverable.  Edit the generator.")
    L.append("")
    L.append("set -uo pipefail")
    L.append("")
    if a.get("held"):
        # A held arm that is submitted by mistake must refuse, not run: a
        # launch guard, not a training knob.  The owner releases it by
        # removing `held` from the row and regenerating; FQ_RELEASE_HELD=1
        # is the one-off override for a run the owner ordered by hand.
        L.append("# HELD (see the header).  73 = submitted while still held.")
        L.append('if [ "${FQ_RELEASE_HELD:-0}" != "1" ]; then')
        L.append(f'  echo "ABORT(73): {a["name"]} is HELD --'
                 ' remove held= from its row in tools/gen_fq_launchers.py,'
                 ' regenerate, then submit"')
        L.append("  exit 73")
        L.append("fi")
        L.append("")

    if kind == "cpu" and not a.get("needs_tool"):
        L.append(PREAMBLE.format(repo=REPO).rstrip())
        L.append('export PATH="$HOME/.local/bin:$PATH"')
        L.append('export PYTHONPATH="$HOME/dsnn/graphax/src:$HOME/dsnn/alphagrad/src"')
        L.append("export PYTHONDONTWRITEBYTECODE=1")
        L.append("")
        L.append(_toolchain_block(kind))
        L.append("")
        L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
        L.append('echo "ag=$(git -C ~/dsnn/alphagrad rev-parse --short HEAD)'
                 ' gx=$(git -C ~/dsnn/graphax rev-parse --short HEAD)"')
        L.append(a["body"])
        return "\n".join(L) + "\n"

    # --- the environment
    if scratch:
        if a.get("env"):
            raise CampaignRowError(
                f"{a['name']}: a campaign arm carries no per-arm env "
                f"(got {sorted(a['env'])}); every knob is an argument")
        L.extend(_scratch_stack_block())
    else:
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
    L.append(_toolchain_block(kind))
    L.append("")

    # --- pre-flight
    L.append("# ---------------------- PRE-FLIGHT ----------------------------")
    _has_wandb = not (kind == "probe" or a.get("needs_tool"))
    L.append("# %s layers, each failing LOUDLY with its own exit code before"
             % ("FOUR" if _has_wandb else "THREE"))
    L.append("# any setup noise reaches the log.  The R1-R4 battery shipped the")
    L.append("# first layer and it caught three missing flags.")
    _flagsrc = a.get("required_flags_file", " ".join(REQUIRED_FLAGS_FILES))
    _flags = a.get("required_flags", REQUIRED_FLAGS)
    L.append(f"#   64 = a flag this launcher needs is not defined in {_flagsrc}")
    L.append("#   65 = argparse rejected the assembled command line")
    L.append("#   66 = a tool this launcher invokes does not exist")
    L.append("#   70 = graphax cannot lower the two-op face form --approx-add asks for")
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
    L.append(f'FLAGSRC="{_flagsrc}"')
    L.append("for F in " + " ".join(_flags) + "; do")
    # FLAGSRC unquoted on purpose: it may name several files (ppo.py and
    # gate_telemetry.py), and grep -q over them answers "defined anywhere".
    L.append('  grep -qF -- "$Q$F$Q" $FLAGSRC'
             ' || MISSING="$MISSING $F"')
    L.append("done")
    L.append('if [ -n "$MISSING" ]; then')
    L.append('  echo "ABORT(64): $FLAGSRC does not define:$MISSING"')
    L.append("  exit 64")
    L.append("fi")
    L.append("")
    _approx_add = dict(_merge_cli(a.get("cli", {}))).get("--approx-add", APPROX_ADD)
    if kind != "cpu":
        _twoop = (f"{gx_repo}/tests/misc/test_face_two_op_form.py" if scratch
                  else "$HOME/dsnn/graphax/tests/misc/test_face_two_op_form.py")
        L.append(f"# --approx-add {_approx_add} emits the res-slot two-op face form.")
        L.append("# On a graphax that rejects it, EVERY plan putting a rule in the")
        L.append("# res/new slot dies in _trace_truncate SILENTLY -- no counter, no")
        L.append("# log line.  That went unnoticed for a whole campaign.  VERIFY.")
        L.append('if [ "${FQ_SKIP_TWOOP:-0}" != "1" ]; then')
        L.append(f"  JAX_PLATFORMS=cpu {py} -m pytest -q -x \\")
        L.append(f"    {_twoop} \\")
        L.append("    -p no:cacheprovider >/tmp/twoop_$SLURM_JOB_ID.log 2>&1 || {")
        L.append('    echo "ABORT(70): graphax rejects the res-slot two-op form,"')
        L.append('    echo "           but --approx-add emits it."')
        L.append("    tail -20 /tmp/twoop_$SLURM_JOB_ID.log")
        L.append("    exit 70")
        L.append("  }")
        L.append('  echo "[preflight] graphax accepts the two-op face form"')
        L.append("fi")
        L.append("")

    if kind == "probe" or a.get("needs_tool"):
        L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
        L.append(f'echo "ag=$(git -C {ag_repo} rev-parse --short HEAD)'
                 f' gx=$(git -C {gx_repo} rev-parse --short HEAD)"')
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
    L.append(f"  JAX_PLATFORMS=cpu {py} - "
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
    # Gate G1's input (ticket .45).  Absent is ACCEPTED by ppo.py (the sweep
    # .41 has not run: gate/g1/present = 0), so this is a statement, not a
    # gate; it puts the fact next to the numbers in the slurm log.
    _gw = dict(_merge_cli(a.get("cli", {}))).get("--gate-winners-table")
    if _gw:
        L.append(f'if [ -f "{_gw}" ]; then')
        L.append(f'  echo "[preflight] gate G1 winners table present: {_gw}"')
        L.append("else")
        L.append(f'  echo "[preflight] gate G1 winners table ABSENT ({_gw}):'
                 ' gate/g1/present will read 0 and gate/g1/recovery* will be'
                 ' meaningless for this whole run"')
        L.append("fi")
        L.append("")
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
    L.append(f"JAX_PLATFORMS=cpu {py} -c \\")
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
    L.append(f'echo "ag=$(git -C {ag_repo} rev-parse --short HEAD)'
             f' gx=$(git -C {gx_repo} rev-parse --short HEAD)"')
    L.append("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    L.append("")
    if scratch:
        # All GPUs of the node are visible: the trainer takes device 0
        # (--gpus 0, the ppo.py default) and the --ray-measure actor device 1
        # (ppo.py pins idx + 1 under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES).
        L.append(f"{py} \\")
    else:
        L.append(f"CUDA_VISIBLE_DEVICES=0,1,2,3 {py} \\")
    L.append('  src/alphagrad/approx/ppo.py "${ARGS[@]}"')
    L.append('echo "TRAINER exited with $?"')
    return "\n".join(L) + "\n"


def _bash_n(text: str) -> str | None:
    """``bash -n`` the text; return stderr on a syntax error, else None."""
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        chk = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
        return None if chk.returncode == 0 else chk.stderr
    finally:
        os.unlink(tmp)


def _diff(old: str | None, text: str, path: str) -> tuple[str, str]:
    """('MISSING' | 'DRIFT' | 'ok', unified diff text)."""
    if old is None:
        return "MISSING", ""
    if old == text:
        return "ok", ""
    return "DRIFT", "".join(difflib.unified_diff(
        old.splitlines(keepends=True), text.splitlines(keepends=True),
        fromfile=f"{path} (on disk)", tofile=f"{path} (generated)"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=HOME_DSNN,
                    help="where the launchers are written (default: the tree)")
    ap.add_argument("--check", action="store_true",
                    help="render, syntax-check and DIFF against what is on "
                         "disk under --out; write nothing; exit non-zero on "
                         "any drift")
    ap.add_argument("--dry-run", action="store_true",
                    help="write every launcher to --out, which must lie OUTSIDE "
                         "--against, and diff each against its copy in "
                         "--against.  The tree is never touched.  Also writes "
                         "<out>/DRIFT.diff (every diff) and <out>/SUMMARY.txt.")
    ap.add_argument("--against", default=HOME_DSNN,
                    help="the tree a --dry-run diffs against (default: the tree)")
    ns = ap.parse_args(argv)

    if ns.check and ns.dry_run:
        raise SystemExit("--check and --dry-run are exclusive: --check writes "
                         "nothing, --dry-run writes outside the tree")
    out = os.path.realpath(os.path.expanduser(ns.out))
    against = os.path.realpath(os.path.expanduser(ns.against))
    if ns.dry_run:
        if out == against or out.startswith(against + os.sep):
            raise SystemExit(
                f"--dry-run refuses to write into the tree it diffs against "
                f"({ns.against}); pass an --out OUTSIDE it")
        os.makedirs(out, exist_ok=True)

    rc = 0
    drifted: list[str] = []
    summary: list[str] = []
    diffs: list[str] = []
    for a in ARMS:
        text = render(a)
        fname = f"fq_{a['name']}.sbatch"
        path = os.path.join(out, fname)
        err = _bash_n(text)
        if err is not None:
            print(f"SYNTAX ERROR in {path}:\n{err}", file=sys.stderr)
            rc = 1
            continue
        if ns.check or ns.dry_run:
            ref = path if ns.check else os.path.join(against, fname)
            old = None
            if os.path.exists(ref):
                with open(ref) as fh:
                    old = fh.read()
            status, d = _diff(old, text, ref)
            summary.append(f"{status:<8} {ref}")
            if status == "MISSING":
                print(f"MISSING       {ref} (would be created)")
                drifted.append(ref)
            elif status == "DRIFT":
                print(f"DRIFT         {ref}")
                drifted.append(ref)
                diffs.append(d)
                if ns.check:
                    sys.stdout.write(d)
            else:
                print(f"ok            {ref}")
            if ns.check:
                rc = rc or (1 if status != "ok" else 0)
                continue
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o644)
        print(f"wrote {path}")
    if ns.dry_run:
        with open(os.path.join(out, "DRIFT.diff"), "w") as fh:
            fh.writelines(diffs)
        n_miss = sum(1 for s in summary if s.startswith("MISSING"))
        n_drift = sum(1 for s in summary if s.startswith("DRIFT"))
        n_ok = sum(1 for s in summary if s.startswith("ok"))
        head = (f"gen_fq_launchers --dry-run: {len(ARMS)} launchers rendered to "
                f"{out}; against {against}: {n_ok} ok, {n_drift} DRIFT, "
                f"{n_miss} MISSING; {sum(d.count(chr(10)) for d in diffs)} "
                f"diff lines in DRIFT.diff")
        with open(os.path.join(out, "SUMMARY.txt"), "w") as fh:
            fh.write(head + "\n" + "\n".join(summary) + "\n")
        print(head)
    elif ns.check and drifted:
        print(f"\n{len(drifted)} launcher(s) differ from the generator:",
              file=sys.stderr)
        for d in drifted:
            print(f"  {d}", file=sys.stderr)
        print("Regenerate (drop --check) rather than editing them in place.",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
