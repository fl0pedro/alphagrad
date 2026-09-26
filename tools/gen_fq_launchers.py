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

# ---------------------------------------------------------------------------
# THE CAMPAIGN STACK (finding 57).  The pgi15 GPU nodes mount NO home
# directory (/Users/assmuth is ENOENT on gpu15..20 and cpu2); /Scratch is the
# one writable filesystem every node and the head share.  So no launcher
# generated here -- campaign arm or wave arm alike -- `cd`s to ~/dsnn/alphagrad
# or runs `uv run`: every one of them runs the relocated venv of finding 57
# against alphagrad/graphax worktrees staged under CAMPAIGN_STACK, with a
# node-local $HOME that receives the two wandb credential files.  The
# pre-flight refuses (exit 66) when any of these is missing.  Moved above
# REPO/HOME_DSNN (owner ruling 2026-09-14, ticket dsnn-3qm.45.wave): the wave
# 0-4 arms and fq_face_attrib used to `cd ~/dsnn/alphagrad` too, back when
# that checkout was current; it is now 281 commits stale AND the home export
# it lives on is read-only, so every arm moved onto this stack instead.  The
# owner stages the stack once per campaign commit:
#   git -C ~/dsnn/alphagrad worktree add --detach CAMPAIGN_STACK/alphagrad <sha>
#   git -C ~/dsnn/graphax   worktree add --detach CAMPAIGN_STACK/graphax   <sha>
# ---------------------------------------------------------------------------
CAMPAIGN_ROOT = "/Scratch/assmuth/campaign"
CAMPAIGN_STACK = f"{CAMPAIGN_ROOT}/stack"
CAMPAIGN_RUNS = f"{CAMPAIGN_ROOT}/runs"
CAMPAIGN_PY = "/Scratch/assmuth/t57/stack/venv/bin/python"
CAMPAIGN_WANDB_HOME = "/Scratch/assmuth/t57/home"     # .netrc + .config/wandb
CAMPAIGN_CACHE = "/Scratch/assmuth/mrg/cache"          # dsnn_wikitext, dsnn_mnist, dsnn_shd

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
LANDSCAPE_TOOL = f"{CAMPAIGN_STACK}/alphagrad/src/alphagrad/approx/tools/landscape_map.py"

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
    # THE MEASUREMENT PIPELINE (owner ruling 2026-09-14).  Named so an arm
    # cannot inherit the trainer's synchronous default in silence.
    "--measure-pipeline",
    # DATA PARALLELISM OVER ENVIRONMENTS (owner ruling 2026-09-15).  Named so
    # an arm states how many GPUs roll the episode out instead of inheriting a
    # default, in either direction.
    "--rollout-shards",
    # WHERE THE PER-STEP TOKENIZATION RUNS (owner ruling 2026-09-15).  Named
    # for the same reason: it decides whether the measure actors are free
    # during a rollout, and therefore whether the pipeline overlaps a
    # measurement with the NEXT rollout or only with the previous update.
    "--tokenize-where",
    # HOW MANY FACE COLUMNS THE HOST CALLBACKS CARRY.  Named because a tree
    # without the flag would ship the whole 1920-column prefix history on
    # every callback -- 59.4 GB device to host per episode, measured -- while
    # the preflight said yes.
    "--face-wire-faces",
    "--cost-form",
    "--quality-floor",
    # The cost floor (.9, b2c89170): every training arm passes it explicitly,
    # and a tree without the flag fails at argparse AFTER the preflight said
    # yes (audit of 2026-09-14 on a 61-commit-old stack). Checked here so the
    # preflight names the missing flag instead.
    "--paired-cost-floor",
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
    # The per-plan time budget (owner ruling 2026-09-14). Every training arm
    # passes both explicitly, so a tree without them would fail at argparse
    # after the preflight said yes -- the same failure --paired-cost-floor
    # was added here for.
    "--measure-budget-secs",
    "--measure-window-secs",
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
    ("PYTHONPATH", f'"{CAMPAIGN_STACK}/graphax/src:{CAMPAIGN_STACK}/alphagrad/src"'),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    # NO XLA FLAGS, and no preallocation switch (owner ruling 2026-09-12 on
    # ticket .43): the arms measure with temp memory, so a preallocation
    # fraction buys nothing, and a Triton / autotune override changes the
    # very cost the arm measures.
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
    # 256, not the unset 0: canary job 65443 (fq_p1a on gpu19, stack at
    # 12ed4936) printed ppo.py's own warning ("ALPHAGRAD_EXTEND_CHUNK is 0,
    # so every encode_extend scans all 32768 window steps ... 256 measured
    # well") and an episode took 275 s at 0.  ppo.py has no --extend-chunk
    # flag (2026-09-14 audit).  32, not 256, since the rollout profile of
    # 2026-09-15 (scratchpad profile-rollout.md, gpu15, fast read): the
    # per-step cost outside the env callback was 363 ms at 256 and 226 ms
    # at 32, 14.5 s per episode, no code change; 32 is the fast read's
    # chunk, and the fold refuses a chunk that is not a multiple of it.
    ("ALPHAGRAD_EXTEND_CHUNK", "32"),
    ("ALPHAGRAD_EXTEND_UNROLL", "32"),
    ("ALPHAGRAD_MULS_SENTINEL_CAP", "5e12"),
    # Project memory: ALWAYS skip the count pass (77% of host time) and use the
    # spec-native direct measurement.  Canary job 65443 is why this one may
    # never be dropped: with only ALPHAGRAD_SKIP_COST_ANALYSIS=1 set (this
    # var absent, as it is on every campaign arm today) the symbolic count
    # pass ran, and env.py's ALPHAGRAD_MULS_SENTINEL_CAP refusal
    # ("muls-cap") rejected every one of the 48 plans two episodes produced
    # -- the arm measured nothing (2026-09-14 audit).
    ("ALPHAGRAD_SKIP_COUNT_OPS", "1"),
    ("ALPHAGRAD_DIRECT_MEASURE", "1"),
    # --- hostperf stack
    ("ALPHAGRAD_FACE_ENUM_CACHE", "1"),
    ("ALPHAGRAD_UNIFIED_FACE_ENUM", "1"),
    ("ALPHAGRAD_BATCHED_CALLBACK", "1"),
]

# ---------------------------------------------------------------------------
# PER-NODE PERSISTENT JAX COMPILATION CACHE (owner ruling 2026-09-14, small
# fixes #3).  /Scratch, never $HOME or /tmp: the pgi15 GPU nodes mount no
# home directory (finding 57) and a /tmp cache dies with the job, so neither
# warms across submissions.  Keyed by hostname, never shared across nodes
# (same reasoning as the old ticket .21 PER NODE rule below, and the owner's
# own: the nodes differ in GPUs and CPUs, so an executable compiled for one
# node's ISA/driver must never be handed to another) -- an entry written on
# a healthy node reused verbatim on a node whose link toolchain is broken is
# exactly how the wave-1 contamination was 51%/97% rather than 100%.
# Expanded by the shell ON THE NODE at job start; created before python
# starts so a process that never calls common.cache.setup_jax_compile_cache
# (which would otherwise os.makedirs it) does not hit ENOENT on first write.
#
# The other two exports make the cache take effect for a program of ANY
# compile time / artifact size, not just the trainer's multi-second compiles
# (installed JAX is 0.10.2.dev0+selfbuilt at /Scratch/assmuth/t57/stack/venv;
# jax/_src/config.py: jax_persistent_cache_min_compile_time_secs defaults to
# 1.0 second, jax_persistent_cache_min_entry_size_bytes defaults to 0 already
# but is named here so it is never silently overridden -- both honoured by
# this build). No XLA_FLAGS.
#
# JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="" is the fix for a race the owner's
# ruling did not anticipate but a live canary did: job 65500 on gpu19 (7
# measure actors sharing this node's cache dir) died in one actor's evaluate
# with "NOT_FOUND: .../xla_gpu_per_fusion_autotune_cache_dir/tmp/
# tmp_per_fusion_cache__..._textproto".  This JAX build's
# jax_persistent_cache_enable_xla_caches defaults to
# 'xla_gpu_per_fusion_autotune_cache_dir' (jax/_src/config.py ~1427), which
# jax/_src/compiler.py's get_compile_options (~262-283) turns on whenever the
# persistent cache is enabled: it derives an autotune-cache subdirectory
# under JAX_COMPILATION_CACHE_DIR and arms it UPDATE-mode for
# `distributed.global_state.process_id == 0`, READ-mode otherwise. Every
# measure actor here is an independent JAX runtime, not a participant in one
# jax.distributed cluster, so EVERY actor defaults to process_id 0 and all N
# of them write-mode the same per-fusion cache files -- the multi-writer
# race behind the NOT_FOUND. Disabling it (empty string; the substring check
# in get_compile_options treats "" as neither "all" nor containing either
# cache name) turns off both optional XLA-side caches and leaves only the
# base executable cache this ticket asks for -- a Python-level LRUCache
# (jax/_src/lru_cache.py) that this ticket's directory change already makes
# per-node and persistent, and which this JAX build only filelock-guards
# when eviction is on (jax_compilation_cache_max_size != -1, not set here);
# unlike the autotune cache it was neither asked for nor observed to race.
JAX_CACHE_DIR_EXPR = "/Scratch/assmuth/jaxcache/$(hostname -s)"

JAX_CACHE_ENV = [
    ("JAX_COMPILATION_CACHE_DIR", JAX_CACHE_DIR_EXPR),
    ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0"),
    ("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0"),
    ("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", ""),
]


def _jax_cache_lines() -> list[str]:
    """``mkdir -p`` the per-node cache dir, then export the four JAX_*
    names in JAX_CACHE_ENV.  Every arm that runs python calls this --
    directly (the campaign and wave/cpu/tool render branches) or via the
    ``@JAX_CACHE_BLOCK@`` placeholder (the one arm with a literal body)."""
    L = [f"mkdir -p {JAX_CACHE_DIR_EXPR}"]
    for k, v in JAX_CACHE_ENV:
        L.append(f"export {k}={v}")
    return L


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


def _toolchain_block(kind: str, node: str) -> str:
    """The block wants @WANT@ replaced with THIS NODE's matched release, not
    the venv's own 12.9 always: NODE_CUDA_WANT (defined beside NODE_CUDA_BIN,
    below) names the nodes whose matched pair is a different release, and the
    default is CUDA_WANT for every node it does not name."""
    on_fault = _TOOLCHAIN_CPU_NOTE if kind == "cpu" else _TOOLCHAIN_ABORT
    want = NODE_CUDA_WANT.get(node, CUDA_WANT)
    return (TOOLCHAIN_BLOCK.replace("@WANT@", want)
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
    # THE PAIRED REV-EXACT REFERENCE'S OWN BUDGET (owner ruling 2026-09-14).
    # Named explicitly on every training arm, not left to the CLI default,
    # for the same reason --num-data-points is: a launcher must say what it
    # measured.  The two halves of the pair are 150x apart in cost on this
    # order -- one candidate execution is 18.2 ms and one reference execution
    # is 0.121 ms -- so the SHARED 5 x 4 budget gave the candidate 18.3 s of
    # integration and the reference 0.12 s.  Measured over 128 repeats of one
    # plan (job 65468): candidate CV 0.56 percent, reference CV 4.53 percent,
    # paired log ratio sd 0.0446 nats, essentially all of it the reference.
    # 5 x 32 costs the reference about 1 s per plan, five percent of the
    # plan's measurement time, and halves the paired ratio's noise.  The
    # INNER reps stay shared at 50 above (owner ruling).
    ("--ref-num-data-points", "5"),
    ("--ref-reps-per-point", "32"),
    # THE PER-PLAN TIME BUDGET (owner ruling 2026-09-14).  Named explicitly
    # on every training arm, for the same reason every other measurement flag
    # is: a launcher must say what it measured.  --num-data-points and
    # --reps-per-point above are now the CAP on the candidate's timed
    # windows; how many of them a plan earns comes from one warm-up
    # execution, so that a plan costs about ONE SECOND of executions whatever
    # it costs per execution.  Before the ruling a plan ran a fixed
    # 5 x 4 x 50 = 1005 executions, which on the transformer arm is 18.3 s
    # per plan and 293 s of an episode's 405 s, to sample a reading whose
    # coefficient of variation is 0.56 percent.  A SLOW PLAN NOW GETS FEWER
    # WINDOWS, which the owner accepts: slow runs do not matter, they are too
    # large anyway.  The 50 ms window is what keeps a window off the dispatch
    # floor (5 executions per window read 20.7 percent high against 50; 20
    # already read within 3 percent of 50), and --latency-inner-reps 50 above
    # is its ceiling.
    ("--measure-budget-secs", "1.0"),
    ("--measure-window-secs", "0.05"),
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
    # STE: every thesis row overrides this with THESIS_MEM_CHANNEL watermark
    # (owner 2026-09-19); temp stays the campaign's channel.
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
    # ENVIRONMENTS PER SHARD (owner ruling 2026-09-15).  --num-envs is what
    # ONE GPU rolls out; --rollout-shards says how many GPUs do.
    ("--num-envs", "16"),
    # DATA PARALLELISM OVER ENVIRONMENTS: one rollout shard per GPU the job
    # holds, so an episode holds CAMPAIGN_GPUS * 16 environments and the PPO
    # update runs on all of them concatenated.  See CAMPAIGN_ROLLOUT_SHARDS.
    # ONE, for now.  The literal is checked against CAMPAIGN_ROLLOUT_SHARDS
    # where that is defined, which is further down this file than SHARED_CLI is
    # built, so the two cannot drift apart.
    ("--rollout-shards", "1"),
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
    # THE POLICY EMBEDDING's row count. It must be at least the TOKENIZER's id
    # space, which is 256 (owner's decision, 2026-09-13 -- see
    # alphagrad.approx.common.token_vocab). It used to say 512, which was the
    # tokenizer's old id space; a generated arm that still says 512 would
    # allocate twice the embedding rows the tokenizer can ever address.
    ("--vocab-size", "256"),
    ("--num-layers", "3"),
    ("--ray-measure", "3"),
    ("--ray-measure-timeout", "600"),
    # PIPELINE THE MEASUREMENT (owner ruling 2026-09-14).  The terminal step
    # submits its plans and returns; the driver dispatches the PREVIOUS
    # episode's PPO update and waits for these rewards while it runs.  See
    # CAMPAIGN_MEASURE_PIPELINE for what the one-episode lag costs.
    ("--measure-pipeline", "1"),
    # THE PER-STEP TOKENIZATION STAYS IN THE TRAINER (ruling 2026-09-15).  It
    # measures nothing, so it needs no measure actor; routing it to the pool
    # cost a Ray round trip per step AND held the actors, which is what made
    # the measurement of e overlap only the update of e-1.  With the actors
    # free for the whole of the next rollout, e's measurement now overlaps
    # the ROLLOUT of e+1.  See CAMPAIGN_TOKENIZE_WHERE.
    ("--tokenize-where", "local"),
    # THE FACE WIRE CARRIES THE COLUMNS THAT ARE USED (ruling 2026-09-15).
    # The four per-step face callbacks and the env step callback each take
    # the whole elimination-prefix face history as an operand, sized by the
    # provable bound MAX_FACES = 1920 on this graph.  Measured under an XLA
    # trace on pgi15-gpu17, one episode at 16 environments: 59.4 GB copied
    # device to host, the GPU idle for three quarters of the rollout behind
    # it, against a MEASURED occupancy of a median of one face per vertex and
    # a maximum of thirteen.  At 64 columns the same episode copies 2.4 GB
    # and its traced span falls from 48.0 s to 14.7 s.  NOT a lowered bound:
    # the state keeps every column and a vertex with more faces than this
    # stops the run by name.  See CAMPAIGN_FACE_WIRE_FACES.
    ("--face-wire-faces", "64"),
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
@JAX_CACHE_BLOCK@

echo "=== GATE 1/3: tools/ratio_gates.sh ==="
PY="$PY" tools/ratio_gates.sh
RG=$?
echo "ratio_gates rc=$RG"

echo "=== GATE 2/3: tools/smoke.sh NeuralNetwork ==="
SMOKE_OUT=@CAMPAIGN_ROOT@/run_analysis/w0_smoke tools/smoke.sh NeuralNetwork
SM=$?
echo "smoke rc=$SM"

# GATE 3/3.  Does --ray-measure measure anything?  rc 1 = MEASUREMENT DEAD,
# rc 2 = the gate could not run (harness misconfigured).  BOTH are failures --
# a gate that did not run pins nothing (116c540).
echo "=== GATE 3/3: tools/pool_liveness_gate.sh ==="
POOL_GATE_OUT=@CAMPAIGN_ROOT@/run_analysis/w0_pool_liveness \
RAY_TMPDIR=/tmp/ray_poolgate_$SLURM_JOB_ID \
PY="$PY" tools/pool_liveness_gate.sh
PL=$?
echo "pool_liveness rc=$PL"

if [ $RG -ne 0 ] || [ $SM -ne 0 ] || [ $PL -ne 0 ]; then
  echo "W0 CPU GATES RED (ratio_gates=$RG smoke=$SM pool_liveness=$PL)" \
       "-- do not launch wave 1"
  exit 1
fi
echo "W0 CPU GATES GREEN"
""".replace("@CAMPAIGN_ROOT@", CAMPAIGN_ROOT)
      .replace("@JAX_CACHE_BLOCK@", "\n".join(_jax_cache_lines())),
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
OUT=@CAMPAIGN_ROOT@/run_analysis/w0
mkdir -p $OUT

echo "===================== P1: singleton skip sweep ====================="
# NOTE, AND DO NOT "FIX" IT: landscape_map's --quality-metric choices are
# loss_drop/cosine/none -- the tool cannot NAME grad_cosine.  On
# TransformerLM (a scalar-loss target) "cosine" is the DEPRECATED ALIAS that
# resolves to grad_cosine (env.py:3489), which is exactly the settled channel.
# It warns once and loudly.  Passing loss_drop here would measure a different
# channel from every training arm.
CUDA_VISIBLE_DEVICES=0 $PY \
  src/alphagrad/approx/tools/landscape_map.py \
  --example TransformerLM --dataset wikitext2 \
  --hidden-dim 256 --vocab-size 256 --num-layers 3 --seed 250197 \
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
  CUDA_VISIBLE_DEVICES=0 $PY \
    src/alphagrad/approx/tools/landscape_map.py \
    --example TransformerLM --dataset wikitext2 \
    --hidden-dim 256 --vocab-size 256 --num-layers 3 --seed 250197 \
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
  # THE TARGET'S GRADIENT WINDOW, named because it used to be a default.
  # LIF_SNN_SHD read ALPHAGRAD_SNN_TRUNC and, unset, unrolled the whole T=100
  # sequence; the flag that replaced it (--target-grad-window) defaults to ONE
  # step, so the window this arm re-baselined is written out.  The flag RAISES
  # on a target with no time steps, which is why the other eight do not get it.
  WIN=""
  [ "$EX" = "LIF_SNN_SHD" ] && WIN="--target-grad-window 100"
  CUDA_VISIBLE_DEVICES=0 $PY \
    src/alphagrad/approx/tools/landscape_map.py \
    --example $EX --dataset none --seed 250197 $WIN \
    --exec-on-gpu --cmp-type latency --mem-type peak_memory \
    --num-data-points 5 --reps-per-point 4 --latency-inner-reps 50 \
    --ladder 1,5 --ops quant,diag,compress --reps 3 \
    --noise-floor-reps 5 --noise-floor-plan identity \
    --out-dir $OUT --tag p3_rebase_$EX --max-seconds 1800
  echo "P3 $EX exited with $?"
done
echo "W0 PROBE COMPLETE"
""".replace("@CAMPAIGN_ROOT@", CAMPAIGN_ROOT),
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
    ("w1c_bias4_lam170", 4, 170, "pgi15-gpu18"),
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
    ("w2c_order_free_exact", "0", {"--exact": None}, "pgi15-gpu18",
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
    ("w3c_lam400", "400", "0", "pgi15-gpu18", "lambda_acc = 400"),
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
    ("w4c_read_ownspan", "own-span-mean", "pgi15-gpu18",
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
# STE: every thesis row raises this cap to THESIS_DUAL_LAMBDA_MAX 64 (owner
# 2026-09-19); 32 stays the campaign's cap.
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
# THE FLOOR THIS ASSUMES IS IN THE TRAINER since alphagrad b2c89170:
# --paired-cost-floor {reference,byte} defaults to `reference`, which floors
# both channels at the paired reference's own cost.  Every campaign arm
# passes it explicitly, so the launcher records the reward it ran under
# instead of inheriting a default that may move.  Under the old one-byte
# floor (--paired-cost-floor byte) the same configuration prices at contrast
# -13.4 on Markowitz and the absorber wins.
# ---------------------------------------------------------------------------
GATE_OFFLINE_CONTRAST = {"markowitz": "0.19", "free": "0.19", "reverse": "0.00"}

#: The paired-cost floor every campaign arm runs under (ppo.py
#: --paired-cost-floor).  `reference` is what finding 63 priced and what the
#: owner approved; `byte` is the pre-2026-09-13 floor.
#: STE: every thesis row runs under THESIS_PAIRED_COST_FLOOR byte instead
#: (owner 2026-09-19), because the reference floor made the cost reward
#: min(0, -log ratio) and no plan could earn a gain below parity.
PAIRED_COST_FLOOR = "reference"

# The profiles of ticket .40 (ppo.py --approx-profile choices) and the orders
# of ticket .64 (common/order.py FIXED_ORDER_CHOICES).  Typed here because
# importing ppo.py costs a jax session; the campaign test cross-checks both
# against make_argparser so a drift is caught, not trusted.
PROFILES = ("all", "skip", "reduce", "quant", "diag", "none")
FIXED_ORDERS = ("markowitz", "reverse", "free")

# ---------------------------------------------------------------------------
# THE CAMPAIGN HARDWARE (owner ruling 2026-09-13).  One sbatch per node, all
# eight Blackwell GPUs of the node.  pgi15-gpu19 belongs to another group
# (dsnn-dfw.69): pgi15-gpu20 is the only 8-GPU node we may use, so every
# campaign row now runs there.  The trainer takes device 0 (--gpus 0, the
# ppo.py default) and the --ray-measure actor device 1 (ppo.py pins idx + 1
# through RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES); holding the node is
# what makes the timed executions mean anything (co-residency measured CV
# 0.0000% -> 49.7%).
# ---------------------------------------------------------------------------
CAMPAIGN_NODES = ("pgi15-gpu20",)
CAMPAIGN_GPUS = 8
CAMPAIGN_GRES = ("gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition"
                 f":{CAMPAIGN_GPUS}")
CAMPAIGN_CPUS = 128
CAMPAIGN_MEM = "800G"          # the nodes have 1.5 TB; 100 G per GPU
# ONE PPO GPU, EVERY OTHER GPU A MEASURE ACTOR (owner ruling 2026-09-14,
# replacing the one-actor ruling of 2026-09-13): the trainer keeps GPU 0 and
# the terminal plans of an episode spread over the remaining GPUs of the
# node. Canary job 65468 measured 16 plans per episode on ONE actor while six
# Blackwells sat idle, at about 300 s per episode. The paired protocol runs
# candidate and reference back to back inside one actor, so several actors
# do not break the pairing.
CAMPAIGN_RAY_MEASURE = str(CAMPAIGN_GPUS - 1)
CAMPAIGN_RAY_MEASURE_TIMEOUT = "600"

# THE PIPELINE (owner ruling 2026-09-14).  The terminal rewards of episode e
# are read by one thing, e's PPO update, so the rollout submits the plans and
# the driver waits for them with the PREVIOUS episode's update already running
# on the trainer's GPU.  The cost is one update of policy lag: e's trajectory
# was drawn under the policy that had absorbed e-2 while its update starts
# from the policy that has absorbed e-1, so PPO's ratio is exact but no longer
# identically 1 at epoch 0 (the arms run --ppo-epochs 1).
CAMPAIGN_MEASURE_PIPELINE = "1"

# DATA PARALLELISM OVER ENVIRONMENTS (owner ruling 2026-09-15, "use the idle
# GPUs for the rollout").  --num-envs is PER SHARD, and --rollout-shards says
# how many devices roll an episode out.  The PPO update runs on the shards
# concatenated, which is the same program a single device would run for that
# many environments.
#
# ONE, FOR NOW (owner ruling 2026-09-15, the follow-up).  Sharding was measured
# at 1, 4 and 8 shards on pgi15-gpu19 and it did not pay: the per-step host
# callbacks are Python, one thread per shard is the most concurrency a single
# process can have, and the host cost per environment does not fall.  It is
# armed here at 1 until the host path is cheap enough that the device work it
# hides is worth having.
CAMPAIGN_ROLLOUT_SHARDS = "1"
assert dict(SHARED_CLI)["--rollout-shards"] == CAMPAIGN_ROLLOUT_SHARDS, (
    "SHARED_CLI's --rollout-shards literal and CAMPAIGN_ROLLOUT_SHARDS must "
    "agree; SHARED_CLI is built before this line runs, so this is what keeps "
    "them equal.")

# WHERE THE PER-STEP TOKENIZATION RUNS (owner ruling 2026-09-15).  A
# non-terminal callback row measures nothing under terminal rewards: it
# tokenizes the prefix, decides face legality and returns the delta
# observation.  It was riding the measure actors, which cost a Ray round trip
# on every step and, worse, kept the actors busy -- so the pipelined terminal
# measurement could only be hidden behind the previous UPDATE.  Kept in the
# trainer process the actors are idle for the whole of the next rollout, and
# the measurement of episode e is hidden behind the ROLLOUT of e+1.
CAMPAIGN_TOKENIZE_WHERE = "local"

# HOW MANY FACE COLUMNS THE HOST CALLBACKS CARRY (owner ruling 2026-09-15).
# 64 against a measured maximum of 13 faces on any vertex of this graph
# (profile-rollout.md section 4: n=1520 vertices, median 1, p95 3, p99 11,
# max 13, occupancy 0.069 percent of the 1920 cap), so just under five times
# the largest thing ever seen.  The elimination order is FIXED on these arms,
# so the face count of each vertex is a property of the order rather than of
# the policy.  If a vertex ever exceeds this the run stops and the message
# names the flag; it does not truncate.
CAMPAIGN_FACE_WIRE_FACES = "64"

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
    ("ALPHAGRAD_PROFILE", "1",
     "print the per-phase wall (rollout, measurement, loss) once per "
     "episode from env.py's shared profiling sink (ppo.py ~7261).  Owner "
     "ruling 2026-09-14: canary 65468 ran about 300 s per episode and "
     "nothing in its log attributes the time, because the arms did not set "
     "this.  Printing only; no flag exists."),
    ("ALPHAGRAD_SKIP_COUNT_OPS", "1",
     "skip the symbolic muls-count pass.  Canary job 65443 "
     "(fq_p1a_skip_hinge_tau09_lq16 on gpu19, stack at 12ed4936) wrote 48 "
     "plan records over two episodes and every one was refused, reason "
     "muls-cap: with only ALPHAGRAD_SKIP_COST_ANALYSIS=1 set, env.py's "
     "count pass runs and its ALPHAGRAD_MULS_SENTINEL_CAP (5e13) refuses "
     "the exact Markowitz plan on TransformerLM outright, so the arm can "
     "never earn a reward.  The validated smoke "
     "(/Scratch/assmuth/mrg/runs/smoke_merged.sbatch) sets this and drains "
     "clean; the reward channels these arms train (cmp mem acc) do not "
     "need the count pass.  env.py has no --skip-count-ops flag."),
    ("ALPHAGRAD_EXTEND_CHUNK", "32",
     "chunk the encode_extend scan instead of walking the whole window; "
     "32 since the rollout profile of 2026-09-15 measured 14.5 s per "
     "episode less than 256 at the fast read (the kernel's own chunk).  "
     "The same canary job (65443) printed ppo.py's own warning "
     "('ALPHAGRAD_EXTEND_CHUNK is 0, so every encode_extend scans all "
     "32768 window steps ... 256 measured well') and an episode took "
     "275 s with it unset; the rendered fq_p1a launcher exports neither "
     "this nor ALPHAGRAD_SKIP_COUNT_OPS today.  ppo.py has no "
     "--extend-chunk flag."),
]

STACK_ENV_NAMES = ("HOME", "PYTHONPATH", "DSNN_WIKITEXT_DIR", "DSNN_MNIST_DIR",
                   "DSNN_SHD_DIR",  # a row thesis_arm emits (dsnn-dfw.264)
                   "PATH")   # PATH: the measure toolchain block (finding 03)

#: Every `export NAME=` a campaign launcher may contain.  The test derives the
#: rendered set and asserts equality with this one.
CAMPAIGN_ENV_ALLOWED = frozenset(
    [k for k, _ in CAMPAIGN_ENV] + [k for k, _, _ in NO_FLAG_ENV]
    + list(STACK_ENV_NAMES)
    # The per-node persistent JAX compile cache (owner ruling 2026-09-14,
    # small fixes #3) is the one JAX_*/XLA_* exception the campaign test
    # allows -- see JAX_CACHE_ENV.
    + [k for k, _ in JAX_CACHE_ENV])

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

THE COST FLOOR: --paired-cost-floor {PAIRED_COST_FLOOR} (alphagrad
b2c89170, ticket .9).  Finding 63's numbers floor BOTH cost channels at the
paired rev-exact cost before the log, and this arm passes that explicitly.
Under the old one-byte floor (--paired-cost-floor byte) the same
configuration prices at contrast -13.4 on Markowitz and the absorber wins.

THE FACE HEAD is one flat MLP of {FACE_HEAD_WIDTH} logits (derived from
alphagrad.approx.unified_face_head.head_layout({APPROX_ADD!r}).width at
generation time, never typed): three {(FACE_HEAD_WIDTH - 1) // 3}-wide slot
blocks, each with a four-way Quant dtype softmax over
{", ".join(FACE_QUANT_DTYPES)} (the operand's own dtype is masked, ticket
.40 D4); no flag selects the dtype set.

MEASUREMENT: --ray-measure {CAMPAIGN_RAY_MEASURE} actors, one per GPU the
trainer does not hold (timeout {CAMPAIGN_RAY_MEASURE_TIMEOUT} s),
ALPHAGRAD_BATCHED_CALLBACK=1, all {CAMPAIGN_GPUS} GPUs of the node held by
this job. ALPHAGRAD_PROFILE=1 prints the per-phase wall every episode.

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
        # THE COST FLOOR (ticket .9, alphagrad b2c89170).  Passed even though
        # it is the trainer default: an arm's launcher must record the reward
        # it ran under, and this one moved on 2026-09-13.
        "--paired-cost-floor": PAIRED_COST_FLOOR,
        # THE MEASUREMENT (owner ruling 2026-09-13): one Ray measure actor
        # on its own GPU, the node held by this job.
        "--ray-measure": CAMPAIGN_RAY_MEASURE,
        "--ray-measure-timeout": CAMPAIGN_RAY_MEASURE_TIMEOUT,
        "--measure-pipeline": CAMPAIGN_MEASURE_PIPELINE,
        # HOW MANY DEVICES ROLL AN EPISODE OUT (owner ruling 2026-09-15).
        "--rollout-shards": CAMPAIGN_ROLLOUT_SHARDS,
        "--tokenize-where": CAMPAIGN_TOKENIZE_WHERE,
        "--face-wire-faces": CAMPAIGN_FACE_WIRE_FACES,
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
    ("b", "mem acc", CAMPAIGN_NODES[0],
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
    ("b", "P1", CAMPAIGN_NODES[0],
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


# ===========================  THE THESIS MATRIX  ============================
# Epic `dsnn-dfw`, ticket `dsnn-dfw.4`.  The owner's rulings of 2026-09-15 and
# 2026-09-16 (thesis-plan-2026-09-15.md, HANDOFF-2026-09-16.md and the epic's
# comments).  THIS SECTION DOES NOT REPLACE THE PHASE 1-5 CAMPAIGN ABOVE: the
# campaign answered "which class, which channel, which reward form" on ONE
# seed and the static Markowitz order; the thesis matrix is the DATA
# COLLECTION that follows from its answers, on the FREE order, five seeds and
# a thousand episodes.
#
# THE MATRIX, verbatim from the owner:
#
#   arm        init bias   reward form                PopArt   conditioned
#   A          0           fixed additive, NO floor   off      no
#   B          4           fixed additive, NO floor   off      no
#   C          0           Lagrangian dual, tau 0.90  off      no
#   C_popart   0           Lagrangian dual, tau 0.90  ON       no
#   condC      0           Lagrangian dual, tau 0.90  off      YES
#
#   target         example          dataset     target-shape env
#   nn256          NeuralNetwork    mnist       ALPHAGRAD_NN_HIDDEN=256
#   tlm            TransformerLM    wikitext2   the ALPHAGRAD_TLM_* triple
#   rsnn_tbptt     RSNN_SHD         shd         none (the target is fixed)
#   rsnn_bptt      RSNN_SHD         shd         none
#   rsnn_rtrl      RSNN_SHD         shd         none
#   rsnn_window2   RSNN_SHD         shd         none
#
#   seeds      250197 250198 250199 250200 250201
#   run name   <arm>_<target>_s<seed>, e.g. C_popart_tlm_s250197 and
#              C_rsnn_bptt_s250199
#
# 2026-09-25 (owner rulings, grill rounds 1-3): condC left the matrix, and the
# defense arms A_popart, A_popart_lq4 and A_popart_lq64 joined it on NN256.
# The rsnn_tbptt and rsnn_window2 rows left it too (dsnn-dfw.232).
#
# THE RECURRENT TARGET IS FOUR TARGETS, NOT ONE.  --example RSNN_SHD
# --dataset shd is shared by all four and --temporal-rule is the one flag
# that separates them, so the rule is part of the TARGET KEY rather than a
# fourth matrix coordinate.  That is not cosmetic.  Every helper below takes
# exactly one (arm, target, seed) triple and `thesis_run_name` spells such a
# triple `<arm>_<target>_s<seed>`; with the key `rsnn_bptt` that spelling IS
# the `<arm>_rsnn_<rule>_s<seed>` the owner asked for, and there is no second
# naming rule to keep in step with the first.  The recurrent block is
# therefore 2 rules x 4 arms x 5 seeds = 40 rows (THESIS_MATRIX_RULES), and
# the whole thesis section is 40 + 40 rows, the 9 defense rows and the two
# smoke runs.
#
# --example STAYS RSNN_SHD ON ALL FOUR, window2 INCLUDED.  window2 builds a
# different graph -- two step copies joined by the temporal edge -- but
# common/rsnn_shd.target_example is the one place that says so, and it reads
# the RULE off the command line ("window2 on the ONE-STEP target is legal and
# is how a run asks for the window arm").  A launcher that named
# RSNN_SHD_W2 itself would be a second place saying the same thing.
#
# EVERY RECURRENT ROW IS HELD: the owner released the NN256/TLM first block
# only and no RSNN_SHD run.  See _THESIS_SNN_HELD.
#
# WHY A AND B CARRY NO QUALITY FLOOR (owner ruling 2026-09-16, night).  A and
# B are the CONTROLS that make the Lagrangian arm readable: the fixed additive
# form at lambda_q = 16 with raw quality is what finding 63 prices at contrast
# -20.6, so A is PREDICTED to collapse to q = 0 and B, which starts at the
# identity, is PREDICTED to stay there.  The prediction is the point.  They
# are not tuned to succeed and the floor is not added to rescue them.
#
# WHY condC IS A COMPOSITION AND NOT A FORM.  `campaign_arm` ties
# --preference-conditioned to the reward form (its P0/P1/L rows are all
# conditioned), which is right for the campaign's phase-3 ladder and wrong
# here: the owner's C is the Lagrangian dual WITHOUT conditioning and condC is
# the same dual WITH it.  ppo.py has composed the two since 2026-09-04 (it
# prints "[cfg] lagrangian + preference-conditioned: Dirichlet over (latency,
# memory) only; the quality preference IS lambda") but NO run has ever used
# the composition, which is why the smoke included one condC run.  Since
# 2026-09-25 only the frozen order-only preference rows carry condC
# (FROZEN_ARM_SPEC).
# ---------------------------------------------------------------------------

#: `thesis_arm(arm=...)` takes the ARM NAME in a parameter called `arm`, which
#: shadows the module-level `arm(...)` registrar inside that function.  The
#: alias is how the row still reaches ARMS.
arm_ = arm

THESIS_SEEDS = ("250197", "250198", "250199", "250200", "250201")
THESIS_EPISODES = "1000"
THESIS_CHECKPOINT_EVERY = "50"
THESIS_PARETO_DUMP_EVERY = "10"
THESIS_PLAN_LOG = "auto"
#: Spatial order FREE in every thesis arm (owner: "order is free in every arm,
#: spatial and temporal").  The campaign's Markowitz pin is a campaign answer.
THESIS_ORDER = "free"
#: Every approximation class is available; the thesis does not ablate classes.
THESIS_PROFILE = "all"
THESIS_LAMBDA_Q = LAMBDA_Q_MVP          # 16
THESIS_TAU = QUALITY_FLOOR_TAU          # 0.90
THESIS_TIME = "24:00:00"

# ---------------------------------------------------------------------------
# THE FOUR BLOCK SETTINGS (owner rulings 2026-09-19, epic dsnn-dfw).  They are
# named HERE, in the thesis section, and not on the campaign constants above:
# the campaign is a finished running comparison and finding 63 priced it under
# the campaign's own values.  Every thesis row -- the A/B/C arms on NN256, TLM
# and the SNN, the order-only NN256 and SNN rows, the preference-conditioned
# rows -- reads these four, because every one of them is built by `thesis_cli`.
# ---------------------------------------------------------------------------
#: THE PAIRED-COST FLOOR (a).  byte, not the campaign's `reference`: the
#: reference floor made the cost reward min(0, -log ratio), so no plan could
#: earn a gain below parity.  The Lagrangian handles the skip-all exploit.
THESIS_PAIRED_COST_FLOOR = "byte"
#: THE LAGRANGIAN CAP (b).  64, not the campaign's 32: the skip-all plan on
#: TLM earns about 17 nats of memory gain plus its latency gain, and the
#: penalty is lambda times a violation of about 0.9, so 32 barely wins and 64
#: has room.  --lag-init (THESIS_LAMBDA_Q, 16) and --lag-min (DUAL_LAMBDA_MIN,
#: 12) do not move; the code does not tie them to the cap.
THESIS_DUAL_LAMBDA_MAX = "64"
#: THE MEMORY CHANNEL (c).  watermark, not the campaign's static temp bytes:
#: the runtime water level is the real quantity and it carries the allocator
#: step.  Since 2026-09-25 slot 5 is logged and not trained on a row that
#: carries THESIS_MEM_OBJECTIVE_WEIGHT below.
THESIS_MEM_CHANNEL = "watermark"
#: THE TRAINED MEMORY CHANNEL (dsnn-mep; owner ruling 2026-09-25, round 1 Q4;
#: dsnn-xvi): reward slot 11, the three static memory_analysis() log ratios
#: against rev-exact, at weight 1.  "mem" leaves --rewards, so the value head
#: of slot 5 gets weight 0.
THESIS_MEM_OBJECTIVE_WEIGHT = "1"
#: THE MEASURE PATH (dsnn-dfw.169; owner ruling 2026-09-25, round 1 Q2):
#: env.py reads both; ppo.py has no flag for them.
THESIS_MEASURE_PATH_ENV = {"ALPHAGRAD_DIRECT_MEASURE": "1",
                           "ALPHAGRAD_UNIFIED_FACE_ENUM": "1"}
#: NO DISK CACHE (dsnn-dfw.247, dsnn-dfw.230): without this, ppo.py and
#: cpu_approx_worker.py write /tmp/dsnn-jax-cache-<job>-<host>.
THESIS_NO_DISK_CACHE_ENV = {"ALPHAGRAD_DISABLE_JIT_DISK_CACHE": "1"}
#: AUTO-STOP IS OFF ON A FINAL ROW (d).  A FINAL row is one `thesis_arm`
#: emits: the A/B/C arms of the matrix and the recurrent block.  A TUNING row
#: (orderonly_nn256_*, orderonly_rsnn_*) calls `thesis_cli` itself with
#: auto_stop=True and keeps it.  The smoke rows pass auto_stop=False already.
THESIS_FINAL_AUTO_STOP = False

#: THE FACE-ENTROPY FLOOR (dsnn-dfw.78, SEC-12 finding 2026-08-25). ppo.py's
#: own --face-entropy-floor help records that 0.3 is an always-on igniter at
#: an identity-like init (H is 0.03-0.06 there) and that v64+ uses 0.05.
#: MATRIX ROWS ONLY, unlike (a)-(d) above: the order-only tuning rows and the
#: smoke keep 0.3 (`thesis_cli`'s own default) because they sit outside the
#: thesis matrix.
THESIS_FACE_ENTROPY_FLOOR = "0.05"

#: THE FACE-ENTROPY WEIGHT ON THE RECURRENT AND TLM TARGETS (owner ruling
#: 2026-09-21, dsnn-dfw.84 and dsnn-dfw.78).  dsnn-dfw.78 fixed the floor
#: above to 0.05; dsnn-dfw.84 found the floor was not even the active term on
#: the recurrent target -- the shared 0.05 entropy BONUS alone still drifted
#: the trained policy's none-to-quantize mix with no latency reason (PopArt
#: row zvmfnj2g, approx_prob/none 0.95 -> 0.78 -> 0.65 -> 0.57 at ep
#: 50/150/290/410; symlog row 0.79 -> 0.60).  The trainer docstring
#: (_split_entropy_bonus, ppo.py) records the same mechanism for v62 job
#: 61844 and prescribes the floor PLUS a near-zero bonus, 0.005, not the
#: campaign's 0.05.  RECURRENT (THESIS_RSNN_TARGETS) AND TLM: 0.005.  NN256
#: keeps 0.05, unmeasured by dsnn-dfw.84's finding.
THESIS_FACE_ENTROPY_WEIGHT_LOW = "0.005"
THESIS_FACE_ENTROPY_WEIGHT_NN256 = "0.05"

#: THE ACTOR UPDATE BUDGET (owner ruling 2026-09-20, epic dsnn-dfw, sweep
#: rounds 2-3).  At today's SHARED_CLI defaults (--ppo-epochs 1,
#: --minibatches 4) the Lagrangian constraint is not met; --ppo-epochs 2
#: --minibatches 8 holds it (feasible 0.90-0.96 on six seeds).  MATRIX ROWS
#: ONLY, the same split as THESIS_FACE_ENTROPY_FLOOR above: the order-only
#: tuning rows and the smoke keep today's budget (`thesis_cli`'s own
#: defaults, --ppo-epochs 1 --minibatches 4) because they sit outside the
#: thesis matrix.  The sweep sections (sweepl, sweepl2, sweepl3) set their
#: own budget explicitly per row, frozen so this change cannot silently
#: move their meaning; they are unaffected.
THESIS_PPO_EPOCHS = "2"
THESIS_MINIBATCHES = "8"

#: DUAL-CLIP PPO (owner ruling 2026-09-21, dsnn-dfw.95).  The face head's PPO
#: importance ratio reached 2e4 on the recurrent target because its log-prob
#: is a SUM over 126-195 live face slots whose logits move together, and
#: PPO's one-sided clip bounds the ratio only for A > 0.  For A < 0 the
#: surrogate is r * A with r unbounded, so one violating plan pushed the
#: OP_NONE logit down with weight 2e4 while every feasible plan pushed it up
#: with weight at most 1.2 -- the approximation runaway.  Ye et al. 2020 cap
#: the objective from below at c * A on that branch.  c = 3 is the owner's
#: value.  EVERY ROW `thesis_arm` EMITS reads this: the A/B/C/C_popart/condC
#: matrix on all targets, and the smoke, which has to start the same command
#: line the matrix runs.  The order-only tuning rows call `thesis_cli`
#: directly and the three Lagrangian sweep rounds pass `dual_clip=None`, so
#: both keep the flag OFF: they are FROZEN running comparisons and a
#: launcher that moved under them would invalidate what they measured.  Off
#: is ppo.py's own default and is bit-identical to the loss before the flag
#: existed.
THESIS_DUAL_CLIP = "3.0"

#: TARGET-KL TRUST REGION BOUND (dsnn-dfw.98, owner overnight ruling
#: 2026-09-22).  The face head PPO update on the recurrent target left
#: the trust region at episode 188 of job 67408 (condC bptt s250199):
#: kl/approx went 0.01-0.03 for 187 episodes, then 1.09, then 7.46, then
#: 8.45, each update larger than the last until the run collapsed.
#: --target-kl T zeroes the policy gradient of any minibatch whose KL to
#: the rollout policy exceeds T; value and entropy keep training.  Normal
#: updates sit at 0.01-0.03, so T=0.1 does not touch a healthy update and
#: stops a runaway at its first minibatch past target.  EVERY ROW
#: `thesis_arm` EMITS reads this, exactly as THESIS_DUAL_CLIP above: the
#: A/B/C/C_popart/condC matrix on all targets, the pair launchers and the
#: smoke.  The three Lagrangian sweep rounds pass `target_kl=None` and
#: keep the flag OFF, for the same reason dual-clip is off there: they
#: are FROZEN running comparisons and a launcher that moved under them
#: would invalidate what they measured.  Off is ppo.py's own default.
THESIS_TARGET_KL = "0.1"

#: THE MEASURE ACTORS' EXECUTABLE RETENTION BOUND (dsnn-dfw.99).  Under
#: the free order almost every terminal plan is a NEW PROGRAM (3566
#: distinct plan hashes in 3646 terminal records, job 67410), so every
#: measurement leaves an XLA executable on the measure GPU and nothing
#: drops it: the seven actors of that row climbed to 72.4 GiB of a 96 GB
#: card and refused 5-6 plans of 16 per episode from episode 120.
#: 100 measurements between clears, against the ~450 the pool took to
#: fill.  The actor's clear
#: (cpu_approx_worker._maybe_clear_compile_caches) drops JAX's
#: in-process jit and compilation caches, which is what releases those
#: executables; the ON-DISK compile cache survives, so a plan met again
#: reloads instead of recompiling from HLO, and the paired reference is
#: a _LOCAL_CACHE hit that no clear drops.  EVERY ROW `thesis_arm`
#: EMITS reads this, exactly as THESIS_DUAL_CLIP and THESIS_TARGET_KL
#: above: the A/B/C/C_popart/condC matrix on all targets and the smoke.
#: The order-only tuning rows call `thesis_cli` directly and the three
#: Lagrangian sweep rounds pass `cache_clear_every=None`, so both keep
#: the export OFF: they are FROZEN running comparisons and a launcher
#: that moved under them would invalidate what they measured.  Unset is
#: the actor's own default, which is 0 and means never.
THESIS_MEASURE_CACHE_CLEAR_EVERY = "100"

#: dsnn-dfw.99's export name, in ONE place: the arm below renders it and
#: THESIS_TARGET_ENV_ALLOWED admits it.
MEASURE_CACHE_CLEAR_EVERY_VAR = "ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY"

#: THE ACTOR PROCESS RECYCLE (dsnn-dfw.99, follow-up ruling 2026-09-22,
#: 08:58 cluster time).  The in-process clear above does not return the
#: executables' device memory to the BFC pool: the resumed cadence test
#: (job 67437, ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY=100) still hit
#: RESOURCE_EXHAUSTED in two actors at calls 123/124 and 185/186, and
#: episode 261 refused 5 of 16.  Only recycling the ACTOR PROCESS frees the
#: pool.  EVERY ROW `thesis_arm` EMITS reads this, on the SAME
#: footprint as THESIS_MEASURE_CACHE_CLEAR_EVERY above: the
#: A/B/C/C_popart/condC matrix on all targets and the smoke.  The
#: order-only tuning rows call `thesis_cli` directly and the three
#: Lagrangian sweep rounds pass `proactive_recycle_every=None`, so both
#: keep the export OFF for the same reason: they are FROZEN running
#: comparisons.  Unset is cpu_approx_pool.py's own default (no proactive
#: recycle).  The pool recycles an actor whose measure OOMs on every row;
#: the retry of the OOM'd plan on the fresh actor is gone (dsnn-dfw.120,
#: owner ruling 2026-09-23: 40 of 47 retries failed again).
THESIS_PROACTIVE_RECYCLE_EVERY = "100"

#: dsnn-dfw.99's further export name, in ONE place: the arm below renders
#: it and THESIS_TARGET_ENV_ALLOWED admits it.
PROACTIVE_RECYCLE_EVERY_VAR = "ALPHAGRAD_PROACTIVE_RECYCLE_EVERY"

#: THE MEASURE DEADLINE OF A THESIS ROW (owner rulings 2026-09-23, 2026-09-24
#: Q48: one deadline of 300 s, no cold budget).  Frozen rounds keep
#: CAMPAIGN_RAY_MEASURE_TIMEOUT.
THESIS_RAY_MEASURE_TIMEOUT = "300"
#: THE CORES OF ONE TIMING ACTOR ON A THESIS ROW, on every node class (owner
#: ruling 2026-09-23).  Frozen rounds keep FROZEN_CORE_BUDGET's per_actor.
THESIS_CORES_PER_ACTOR = "8"

# ---------------------------------------------------------------------------
# THE HARDWARE.  Five Blackwell nodes we may use (dsnn-dfw.69, owner ruling
# 2026-09-20).  pgi15-gpu17 has no matched CUDA 12.9 ptxas or nvlink (job
# 66740 aborted 72) and is never a node source in this table, the default
# assignment every thesis row below uses.  pgi15-gpu19 is ours again: it was
# held for another group on 2026-09-16 and released on 2026-09-20.
# pgi15-gpu19 and pgi15-gpu20 carry eight GPUs, 128 CPUs and 1.5T; the other
# three carry four, 64 and 770G.  The measurement fan-out follows the ROW'S
# PROFILE, not the node (see THESIS_UNIFORM_GPUS): one GPU for the trainer
# and every other GPU of the profile a measure actor.
#
# THESIS_NODES is an alias of THESIS_NODES_ALL: every node this table names
# is a node we may use, so nothing is held back from it.
# ---------------------------------------------------------------------------
THESIS_NODES_ALL = ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18",
                    "pgi15-gpu19", "pgi15-gpu20")
THESIS_NODES = THESIS_NODES_ALL
THESIS_NODE_GPUS = {"pgi15-gpu15": 4, "pgi15-gpu16": 4, "pgi15-gpu17": 4,
                    "pgi15-gpu18": 4, "pgi15-gpu19": 8, "pgi15-gpu20": 8}
#: --ray-measure by node size: every GPU the trainer does not hold.
THESIS_RAY_MEASURE = {4: "3", 8: "7"}


def thesis_device_cli(gpus: int, half: int = 0) -> dict:
    # The trainer takes the first GPU of its half, and actor k the k-th after it (dsnn-dfw.245).
    trainer = int(half) * int(gpus)
    return {"--gpus": str(trainer),
            "--measure-gpus": ",".join(
                str(trainer + 1 + k)
                for k in range(int(THESIS_RAY_MEASURE[gpus])))}


#: THE UNIFORM ROW PROFILE, BY TARGET (owner ruling 2026-09-20).  A target
#: named here renders the SAME hardware profile on every Blackwell node: the
#: gres count, -c, --mem, --ray-measure and the core budget come from this
#: number, not from the node's GPU count.  NN256 is here because its five
#: seeds are one distribution: a seed that lands on an 8-GPU node used to
#: render 8 GPUs, 128 CPUs and --ray-measure 7 while its four siblings ran 4,
#: 64 and 3, and latency measured under two fan-outs is not comparable.  A
#: target NOT named here (TLM, the recurrent rules) still takes the node.
THESIS_UNIFORM_GPUS = {"nn256": 4}


def thesis_row_gpus(target: str, node: str) -> int:
    """THE ROW'S GPU PROFILE: uniform by target on Blackwell, else the node.

    The non-Blackwell tuning nodes (pgi15-gpu13, -gpu14, -gpu8) keep the
    node's own count: they carry the order-only tuning round, which is read
    within one node and never pooled with the Blackwell rows.
    """
    if target in THESIS_UNIFORM_GPUS and node in THESIS_NODE_GPUS:
        return THESIS_UNIFORM_GPUS[target]
    return node_gpu_count(node)
#: -c and --mem by node size.  The 8-GPU values are the campaign's.  -c is
#: every CPU of the node (sinfo 2026-09-25: 64 on pgi15-gpu15 to 18, 128 on
#: pgi15-gpu19 and 20).  BLACKWELL_MEM stays what the campaign and the frozen
#: rounds ask for; a row `thesis_arm` emits asks for THESIS_ROW_MEM.
BLACKWELL_CPUS = {4: 64, 8: CAMPAIGN_CPUS}
BLACKWELL_MEM = {4: "400G", 8: CAMPAIGN_MEM}
#: THE ROW'S MEMORY AND THE ORACLE'S HOST BUDGET, by GPU profile (owner rulings
#: 2026-09-25, dsnn-dfw.208; report dsnn-dfw.237).  sinfo RealMemory is
#: 770000 MB on a 4-GPU node and 1540000 MB on an 8-GPU node (752 and 1504
#: GiB): 740G and 1480G leave 12 and 24 GiB to the OS.  The peak RSS of a
#: check is 1.03 to 1.34 times the size the budget bars, so an admitted check
#: holds at most 1.34 x 300 = 402 GiB of 740 (338 GiB stay) and 1.34 x 600 =
#: 804 GiB of 1480 (676 GiB stay) for the trainer and the timing actors, whose
#: own host memory is not measured yet.
THESIS_ROW_MEM = {4: "740G", 8: "1480G"}
THESIS_GRAD_ORACLE_HOST_BUDGET_GB = {4: "300", 8: "600"}
#: --grad-oracle-cores 0: every core the trainer and the timing actors leave
#: (owner ruling 2026-09-25, round 1 Q11).  ppo.py computes that number at
#: run time; THESIS_CORE_BUDGET's oracle entry is the same number.
THESIS_GRAD_ORACLE_CORES = "0"
#: THE ROW'S CORE BUDGET, by GPU profile, over every CPU the row asks for.
#: Owner ruling Q3, 2026-09-18 (the measurement is dsnn-dfw.30 and the probe
#: dsnn-dfw.40): the trainer had no slice of its own and its host work ran on
#: the same cores as the timing actors, which is what made the candidate
#: latency bimodal.  2026-09-23: 8 cores per timing actor.  2026-09-25: the
#: oracle takes every core the two leave.
THESIS_CORE_BUDGET_CPUS = BLACKWELL_CPUS


def _thesis_core_budget(gpus: int) -> dict:
    trainer, per_actor = 8, int(THESIS_CORES_PER_ACTOR)
    return {"trainer": trainer, "per_actor": per_actor,
            "oracle": (THESIS_CORE_BUDGET_CPUS[gpus] - trainer
                       - int(THESIS_RAY_MEASURE[gpus]) * per_actor)}


THESIS_CORE_BUDGET = {g: _thesis_core_budget(g) for g in (8, 4)}
#: THE CORE BUDGET A FROZEN ROUND RAN UNDER (owner ruling Q3, 2026-09-18: a
#: timing actor needs 2 logical CPUs, the timing is one second on the GPU).
#: The order-only rounds and the three sweep rounds render these two flags.
FROZEN_CORE_BUDGET = {"trainer": 8, "per_actor": 2}


def thesis_core_layout(gpus: int):
    """The disjoint per-node layout of ``THESIS_CORE_BUDGET[gpus]``."""
    from alphagrad.approx.common.core_budget import node_core_layout
    if gpus not in THESIS_CORE_BUDGET:
        raise ValueError(
            f"no core budget for a {gpus}-GPU node; the budget covers "
            f"{sorted(THESIS_CORE_BUDGET)} GPUs")
    b = THESIS_CORE_BUDGET[gpus]
    return node_core_layout(
        THESIS_CORE_BUDGET_CPUS[gpus], int(THESIS_RAY_MEASURE[gpus]),
        trainer_cores=b["trainer"], cores_per_actor=b["per_actor"],
        oracle_cores=b["oracle"])


# The budget must fit and be disjoint on every node type, checked at import so
# a launcher can never be generated from a budget that oversubscribes a node.
for _budget_gpus in sorted(THESIS_CORE_BUDGET):
    if thesis_core_layout(_budget_gpus).spare:
        raise ValueError(
            f"the {_budget_gpus}-GPU core budget leaves cores spare; the "
            f"oracle takes every core the trainer and the timing actors leave")
#: The node the smoke runs on (owner: "SMOKE on gpu16").
THESIS_SMOKE_NODE = "pgi15-gpu16"


def blackwell_gres(gpus: int) -> str:
    """The Blackwell gres string for a node of ``gpus`` GPUs."""
    if gpus not in THESIS_RAY_MEASURE:
        raise CampaignRowError(
            f"{gpus} is not a Blackwell node size on this cluster; the nodes "
            f"carry {sorted(THESIS_RAY_MEASURE)} GPUs "
            f"(sinfo, 2026-09-16).")
    return ("gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition"
            f":{gpus}")


assert blackwell_gres(CAMPAIGN_GPUS) == CAMPAIGN_GRES, (
    "the campaign's 8-GPU gres string and the derived one must be the same "
    "string; the campaign arms render from CAMPAIGN_GRES and the thesis arms "
    "from blackwell_gres, and a drift would give two arms different hardware "
    "under one name")

# ---------------------------------------------------------------------------
# NODES THAT ARE NOT BLACKWELL (ticket dsnn-dfw.29, owner 2026-09-18: "tuning
# on any node whose toolchain gate passes, the baseline on Blackwell").  The
# thesis matrix and the campaign run on Blackwell alone and every table above
# is keyed on the GPU COUNT, which is enough while one GPU model is in play.
# A tuning round that spreads over three GPU models needs the node itself as
# the key, so the four tables below hold ONLY the non-Blackwell nodes and
# every helper falls back to the Blackwell expression it replaced.  No arm
# that existed before this section moves by one byte.
#
# `sinfo -N -o "%n %c %m %G %P"`, 2026-09-18.  Memory is the SLURM limit, not
# the hardware: a --mem above it is never scheduled at all.  pgi15-gpu8
# joined 2026-09-19 (ticket dsnn-dfw.45, owner ruling): 64 CPUs, 370000 MB
# (sinfo), the same 92 percent headroom as gpu13's 700G of 750000.
# ---------------------------------------------------------------------------
NODE_GRES_TYPE = {
    "pgi15-gpu8": "nvidia_rtx_6000_ada_generation",
    "pgi15-gpu13": "nvidia_geforce_rtx_4090",
    "pgi15-gpu14": "nvidia_h100_80gb_hbm3",
}
NODE_GPUS = {"pgi15-gpu8": 4, "pgi15-gpu13": 4, "pgi15-gpu14": 8}
NODE_CPUS = {"pgi15-gpu8": 64, "pgi15-gpu13": 64, "pgi15-gpu14": 128}
NODE_MEM = {"pgi15-gpu8": "340G", "pgi15-gpu13": "700G", "pgi15-gpu14": "1000G"}
#: pgi15-gpu14 is the only node of the `pgi15-h100` partition; every other
#: node here is in `pgi15`.
NODE_PARTITION = {"pgi15-gpu14": "pgi15-h100"}
#: A node whose matched CUDA pair is OUTSIDE /usr/local, which is all the
#: measure-toolchain block searches.  The arm puts it on PATH and the block
#: still proves the two versions, so a wrong path aborts 72 (finding 03).
#: pgi15-gpu8's matched pair IS inside /usr/local (the block finds it on its
#: own), but the row still names it here, explicitly, since it is the one
#: fact job 66542 measured (owner ruling 2026-09-19).
NODE_CUDA_BIN = {
    "pgi15-gpu14": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/12.9/bin",
    "pgi15-gpu8": "/usr/local/cuda-12/bin",
}
#: THE WANTED RELEASE, BY NODE (ticket dsnn-dfw.45).  The venv's own ptxas is
#: 12.9.86 (CUDA_WANT); pgi15-gpu8's matched pair is 12.8 (job 66542: "toolchain
#: gate OK on pgi15-gpu8: ptxas 12.8.93 ... nvlink 12.8.93").  A node not named
#: here wants CUDA_WANT, unchanged.
NODE_CUDA_WANT = {
    "pgi15-gpu8": "12.8",
}


def node_gpu_count(node: str) -> int:
    if node in THESIS_NODE_GPUS:
        return THESIS_NODE_GPUS[node]
    if node in NODE_GPUS:
        return NODE_GPUS[node]
    raise CampaignRowError(
        f"node {node!r} has no GPU count on this cluster; known nodes are "
        f"{sorted(set(THESIS_NODE_GPUS) | set(NODE_GPUS))}")


def node_gres(node: str, gpus: int) -> str:
    if node in NODE_GRES_TYPE:
        return f"gpu:{NODE_GRES_TYPE[node]}:{gpus}"
    return blackwell_gres(gpus)


def node_cpus(node: str, gpus: int) -> int:
    return NODE_CPUS[node] if node in NODE_CPUS else BLACKWELL_CPUS[gpus]


def node_mem(node: str, gpus: int) -> str:
    return NODE_MEM[node] if node in NODE_MEM else BLACKWELL_MEM[gpus]


def node_partition(node: str) -> str:
    if node in NODE_PARTITION:
        return NODE_PARTITION[node]
    return "pgi15-cpu" if node == "pgi15-cpu1" else "pgi15"


def thesis_job_name(node: str) -> str:
    """THE CROSS-AGENT PER-NODE SINGLETON NAME (ticket dsnn-dfw.65).

    Every thesis job carries `--dependency=singleton` and a job NAME that is
    the node it is pinned to, so Slurm runs exactly one of our jobs on that
    node at a time and starts the next one the moment the node frees up.
    This is the scheduling form of the project rule that two jobs of one user
    on one pgi15 node kill each other: `/etc/slurm/epilog_reset_node.sh` kills
    every process of the user when ANY of that user's jobs on the node ends
    (memory note `pgi15-epilog-kills-sibling-jobs`; seven of eight sweep64
    shards died that way on 2026-09-13).  A singleton queue cannot produce
    that state, and it needs no babysitting.

    THE NAME IS `node-<node>`, NOT `thesis-<node>` (owner 2026-09-19, ticket
    dsnn-dfw.65).  Singleton serializes only jobs that SHARE a name, and
    several agents submit to these nodes: a name only the matrix uses
    serializes the matrix against itself and nothing else, so a matrix row
    and an order-only row could hold one node together and the epilog would
    kill both.  `node-<node>` is the one name every job of ours carries, so
    `thesis_job_name` and `orderonly_job_name` now return the same string.
    """
    if node not in THESIS_NODE_GPUS and node not in NODE_GPUS:
        raise CampaignRowError(
            f"node {node!r} is not a GPU node this generator knows "
            f"({sorted(set(THESIS_NODE_GPUS) | set(NODE_GPUS))})")
    return f"node-{node}"


# ---------------------------------------------------------------------------
# THE TARGETS.  The example and the dataset are ARGUMENTS; the target's SHAPE
# is an environment variable on the NeuralNetwork side, because
# common/examples.py reads ALPHAGRAD_NN_HIDDEN at IMPORT time (module-scope
# `_EQ_NN_HIDDEN`), exactly as it reads the ALPHAGRAD_TLM_* triple.  There is
# no --nn-hidden flag; this is the same "no flag exists" departure the
# NO_FLAG_ENV block declares, and the thesis test pins that the only per-arm
# export a thesis launcher carries is this one.
#
# ALPHAGRAD_MAX_FACES is NOT exported by either target.  It is an explicit
# experiment override (env.py: `configure_max_faces` returns early when it is
# set); the trainer derives the provable bound per graph and prints it
# ("face width: derived bound N").  The campaign arms already run without it.
# ---------------------------------------------------------------------------
#: THE RECURRENT TARGET's four temporal rules, which were the four SNN arms of
#: the matrix (owner ruling 2026-09-16).  The rule says HOW the state carried
#: between time steps enters the gradient:
#:
#:   tbptt     no temporal edge.  The truncated baseline, and ppo.py's default.
#:   bptt      the one-step body plus the given FUTURE adjoint over the suffix.
#:   rtrl      the one-step body plus the given PAST Jacobian over the prefix.
#:   window2   TWO step copies joined by the temporal edge and NO given edge.
#:
#: This generator imports nothing from alphagrad -- it runs on the login host,
#: where the stack is not importable -- so the tuple is typed here and the
#: thesis test pins it against common/rsnn_shd.TEMPORAL_RULES, which is the
#: tuple ppo.py builds `--temporal-rule`'s choices from.
THESIS_TEMPORAL_RULES = ("tbptt", "bptt", "rtrl", "window2")
#: Owner, 2026-09-26: tbptt and window2 are deprecated. No row may run them.
THESIS_MATRIX_RULES = ("bptt", "rtrl")
THESIS_RSNN_EXAMPLE = "RSNN_SHD"
#: The batched one-step body, rsnn_shd.RSNN_VMAP_TARGET; window2 has none.
THESIS_RSNN_VMAP_EXAMPLE = "VmappedRSNN_SHD"
THESIS_RSNN_DATASET = "shd"
#: The target key of a temporal rule.  `rsnn_bptt`, so `thesis_run_name`
#: spells the run `<arm>_rsnn_bptt_s<seed>`.
THESIS_RSNN_TARGETS = tuple(f"rsnn_{r}" for r in THESIS_TEMPORAL_RULES)
THESIS_TARGETS = ("nn256", "tlm") + THESIS_RSNN_TARGETS
#: The batch of a Vmapped target; common/datasets.py reads it at import time.
NN_BATCH_VAR = "ALPHAGRAD_NN_BATCH"
THESIS_TARGET_CLI = {
    "nn256": {"--example": "VmappedNeuralNetwork",
              "--dataset": "mnist"},
    "tlm": {"--example": "VmappedTransformerLM",
            "--dataset": "wikitext2"},
    # THE RECURRENT TARGET.  --example and --dataset are the same in all four
    # rows; --temporal-rule is the only difference between them, and it is
    # the only flag the recurrent rows carry that the NN256 and TLM rows do
    # not.  Everything else comes from `thesis_cli`, unchanged.
    # dsnn-dfw.191: a batched rule names THESIS_RSNN_VMAP_EXAMPLE instead.
    "rsnn_tbptt": {"--example": THESIS_RSNN_VMAP_EXAMPLE,
                   "--dataset": THESIS_RSNN_DATASET,
                   "--temporal-rule": "tbptt"},
    "rsnn_bptt": {"--example": THESIS_RSNN_VMAP_EXAMPLE,
                  "--dataset": THESIS_RSNN_DATASET,
                  "--temporal-rule": "bptt"},
    "rsnn_rtrl": {"--example": THESIS_RSNN_VMAP_EXAMPLE,
                  "--dataset": THESIS_RSNN_DATASET,
                  "--temporal-rule": "rtrl"},
    "rsnn_window2": {"--example": THESIS_RSNN_EXAMPLE,
                     "--dataset": THESIS_RSNN_DATASET,
                     "--temporal-rule": "window2"},
}
THESIS_TARGET_ENV = {
    # Decision D1 (dsnn-dfw.152): 43 percent of the card under Markowitz.
    "nn256": {"ALPHAGRAD_NN_HIDDEN": "256", NN_BATCH_VAR: "4096"},
    # the ALPHAGRAD_TLM_* triple is in CAMPAIGN_ENV
    # Owner ruling Q21 (2026-09-25): 62 percent of the card under Markowitz.
    "tlm": {NN_BATCH_VAR: "64"},
    # The recurrent target's shape is NOT an environment variable: the hidden
    # width, the time constants and the init scale are module constants of
    # common/rsnn_shd.py (RSNN_HIDDEN = 128, the seven measured constants and
    # WEIGHT_SCALE), chosen by the learning gate, and nothing reads an env
    # var for them.  So these rows export nothing of their own and
    # THESIS_TARGET_ENV_ALLOWED below does not grow.
    # dsnn-dfw.191: a batched rule exports its batch.  One entry per rule, the
    # value on its own line, so that each rule's batch is its own commit.
    "rsnn_tbptt":
        {NN_BATCH_VAR: "256"},  # owner ruling Q21, dsnn-dfw.193
    "rsnn_bptt":
        {NN_BATCH_VAR: "256"},  # owner ruling Q21, dsnn-dfw.193
    "rsnn_rtrl":
        {NN_BATCH_VAR: "64"},  # owner ruling Q22 a, dsnn-dfw.193
    "rsnn_window2":
        {},
}
#: The ONLY per-arm exports a thesis launcher may carry.  `render` refuses
#: any other key, exactly as it refuses every per-arm export on a campaign arm.
THESIS_TARGET_ENV_ALLOWED = frozenset(
    k for env in THESIS_TARGET_ENV.values() for k in env
) | {MEASURE_CACHE_CLEAR_EVERY_VAR, PROACTIVE_RECYCLE_EVERY_VAR, NN_BATCH_VAR
     } | set(THESIS_MEASURE_PATH_ENV) | set(THESIS_NO_DISK_CACHE_ENV)
#: Every `export NAME=` a THESIS launcher may contain: the campaign's allowed
#: set plus the target-shape variables above.
THESIS_ENV_ALLOWED = frozenset(CAMPAIGN_ENV_ALLOWED) | THESIS_TARGET_ENV_ALLOWED


#: THE ORACLE'S BATCH BY TARGET (report dsnn-dfw.237).  NN256 checks its whole
#: batch: 8 to 70 GiB RSS at B=4096 over 4 random orders.  One TLM recording
#: asks 4.4 to 101.5 GiB of XLA temp per order, so the batch of 64 fits no
#: node and TLM checks its first 2: 10.7 to 138.3 GiB over 16 orders.  A
#: full-rollout recurrent target checks its whole batch whatever the flag says.
THESIS_GRAD_ORACLE_BATCH = {"nn256": "0", "tlm": "2",
                            **{t: "0" for t in THESIS_RSNN_TARGETS}}


def thesis_target_form(target: str, batched: bool) -> tuple[dict, dict]:
    cli = dict(THESIS_TARGET_CLI[target])
    env = dict(THESIS_TARGET_ENV[target])
    if not batched:
        # A frozen round keeps the unbatched target it ran on (dsnn-dfw.191).
        cli["--example"] = cli["--example"].removeprefix("Vmapped")
        env.pop(NN_BATCH_VAR, None)
    return cli, env


def thesis_temporal_rule(target: str) -> str | None:
    """The temporal rule of a thesis target, or None when it has no time.

    The rule lives in the target's own `cli` dict, so this reads it back
    rather than parsing the key: one place defines it.
    """
    if target not in THESIS_TARGET_CLI:
        raise CampaignRowError(
            f"target {target!r} is not one of {THESIS_TARGETS}")
    return THESIS_TARGET_CLI[target].get("--temporal-rule")


#: THE TLM FACE-WIRE BUDGET (dsnn-dfw.104, owner ruling 2026-09-22).
#: Vertex 40 of a TLM plan carried 72 live faces at episode 55, past
#: CAMPAIGN_FACE_WIRE_FACES's 64, and the trainer raised by design (the
#: budget is a hard stop, not a truncation).  Every arm on the TLM target
#: -- the matrix, the pair launchers and the smoke, i.e. every row
#: `thesis_arm` emits (`matrix_row=True`) -- renders this instead of the
#: campaign constant.  The order-only TLM final row is also target "tlm"
#: and also goes through `thesis_cli`, but it is a DIFFERENT round with its
#: own record (mirroring `rung1_row`), so it keeps 64.  Every other thesis
#: target (nn256
#: and the four recurrent rules) keeps CAMPAIGN_FACE_WIRE_FACES regardless
#: of `matrix_row`.
THESIS_TLM_FACE_WIRE_FACES = "128"


# ---------------------------------------------------------------------------
# THE ARMS.  Each row is the DIFFERENCE from the shared thesis configuration:
# the face-head init bias, the reward form, the advantage normalisation and
# whether the preference conditioning is composed on top.
# ---------------------------------------------------------------------------
#: condC left the matrix (owner rulings 2026-09-25, round 1 Q1, round 2 Q1).
THESIS_ARMS = ("A", "B", "C", "C_popart")
#: THE DEFENSE ARMS (dsnn-dfw.231; owner ruling 2026-09-25, round 3 Q1 b):
#: arm A with C_popart's PopArt form at lambda_q 16, 4 and 64, so that A, C,
#: C_popart and these span {fixed, L} x {symlog, PopArt}.  NN256 only, three
#: seeds per point, the full experiments only.
THESIS_DEFENSE_ARMS = ("A_popart", "A_popart_lq4", "A_popart_lq64")
THESIS_DEFENSE_TARGET = "nn256"
THESIS_DEFENSE_SEEDS = THESIS_SEEDS[:3]
THESIS_ARM_SPEC = {
    # arm: (face_none_bias, form, advantage_norm, conditioned, lambda_q)
    "A": ("0", "fixed", "none", False, THESIS_LAMBDA_Q),
    "B": ("4", "fixed", "none", False, THESIS_LAMBDA_Q),
    "C": ("2", "L", "none", False, THESIS_LAMBDA_Q),
    "C_popart": ("2", "L", "popart", False, THESIS_LAMBDA_Q),
    "A_popart": ("0", "fixed", "popart", False, THESIS_LAMBDA_Q),
    "A_popart_lq4": ("0", "fixed", "popart", False, "4"),
    "A_popart_lq64": ("0", "fixed", "popart", False, "64"),
}
#: The arm the frozen order-only preference rows ran (ORDERONLY_PREF_ARM).
#: `thesis_cli` reads it; `thesis_arm` refuses it.
FROZEN_ARM_SPEC = {"condC": ("2", "L", "none", True, THESIS_LAMBDA_Q)}

# ---------------------------------------------------------------------------
# RUNG 1 OF THE LADDER: THE NORMALIZED FACE-HEAD INIT (owner ruling
# 2026-09-20).  Arms C and C_popart on NN256 stop setting the none-bias
# directly and state the PLAN they want instead: `a` requested approximations
# and `kappa` requested skips per plan at init.  ppo.py derives
# B = ln(F*S*k/a - k) and Bs = ln(F/kappa - 1) from F, the live-face count of
# the REVERSE-MODE REFERENCE ORDER -- the same walk the paired cost measures
# against -- and prints every input as a [face-init] block.
#
# WHY THESE ROWS AND NO OTHERS.  dsnn-dfw.74: every C and C_popart seed on
# NN256 started at quality ~0 with ten of eleven faces approximated, so the
# constraint was violated everywhere from episode 0, the multiplier saturated
# and the policy collapsed.  --face-none-bias 2 is what put the initial plan
# there.  Asking for THREE approximations per plan instead of a bias number
# is the change under test; every other arm and every other target keeps its
# recorded bias, or the comparison is not about this.
# ---------------------------------------------------------------------------
RUNG1_ARMS = ("C", "C_popart")
RUNG1_TARGET = "nn256"
RUNG1_APPROX_PER_PLAN = "3"
RUNG1_SKIPS_PER_PLAN = "0.3"

#: RUNG 1 REACHES THE RECURRENT TARGET (owner ruling 2026-09-21, agent
#: rsnnpace's init probe).  --face-none-bias 2 asks the recurrent target for
#: 36-56 approximations per plan (F = 42 bptt / 65 rtrl), pins the quality
#: median at exactly 0 and holds the PopArt quality head frozen from episode
#: 0; the SAME plan as NN256's rung 1 -- 3 approximations, 0.3 skips -- gives
#: q median 0.80 (bptt) / 0.46 (rtrl) and a feasible plan by episode 0.
RUNG1_RSNN_ARMS = ("C", "C_popart")

#: RUNG 1 ON THE RECURRENT TARGET DROPS TO a=1 (owner ruling 2026-09-21,
#: dsnn-dfw.84 and dsnn-dfw.78).  a=3 above (RUNG1_APPROX_PER_PLAN, NN256's
#: own number) gave the recurrent target a feasible plan at episode 0, but
#: by episode 60 the trained policy still drifted its none-to-quantize mix
#: under the entropy bonus with no latency reason (dsnn-dfw.84: PopArt row
#: zvmfnj2g, approx_prob/none 0.95 -> 0.78 -> 0.65 -> 0.57 at ep
#: 50/150/290/410).  THESIS_FACE_ENTROPY_WEIGHT_LOW above is the other half
#: of the fix; this constant is the first half, on the recurrent target
#: ONLY -- NN256 keeps RUNG1_APPROX_PER_PLAN (3), unmeasured by dsnn-dfw.84's
#: finding.
#: kappa is unchanged: RUNG1_SKIPS_PER_PLAN (0.3) stays for both families.
RUNG1_RSNN_APPROX_PER_PLAN = "1"


def rung1_row(arm: str, target: str, matrix_row: bool = True) -> bool:
    """Is this (arm, target) a rung-1 row, i.e. normalized-init instead of
    a raw --face-none-bias?

    ONLY A MATRIX COORDINATE CAN BE ONE.  The order-only tuning rounds
    (dsnn-dfw.29 and .45) are also arm C on NN256 and they also go through
    `thesis_cli`, but they run --approx-profile none and are a DIFFERENT
    round with its own record; moving their init would change a launcher
    under a comparison that is not this one.  `matrix_row` is False for
    them, and False is the default so a new caller has to ask.

    THE RECURRENT TARGET IS ON THE SAME RUNG (owner ruling 2026-09-21): its
    four rules all take the same plan, RUNG1_APPROX_PER_PLAN /
    RUNG1_SKIPS_PER_PLAN, for RUNG1_RSNN_ARMS.
    """
    return (matrix_row
            and ((target == RUNG1_TARGET and arm in RUNG1_ARMS)
                 or (target in THESIS_RSNN_TARGETS and arm in RUNG1_RSNN_ARMS)))


THESIS_FLAGS_FILES = REQUIRED_FLAGS_FILES + [
    # --checkpoint-every and --resume live in common/checkpoint.py and
    # --auto-stop in common/auto_stop.py; both install their arguments on
    # ppo.py's own parser (`_ckpt.add_checkpoint_args`, `_auto.
    # add_auto_stop_args`), so the pre-flight's grep must read them too or
    # every thesis launcher would abort 64 naming a flag that is defined.
    "src/alphagrad/approx/common/checkpoint.py",
    "src/alphagrad/approx/common/auto_stop.py",
]
THESIS_REQUIRED_FLAGS = REQUIRED_FLAGS + [
    "--example", "--dataset", "--episodes", "--seed", "--name",
    "--checkpoint-every", "--resume", "--auto-stop",
    "--lag-eta", "--lag-init", "--lag-min", "--lag-max",
    "--grad-oracle-cadence",
    # The recurrent target's one extra flag.  It is defined in ppo.py, so the
    # layer-1 grep passes on every row; it is listed for the same reason
    # every other flag here is, so that a tree without it aborts 64 naming
    # the flag rather than 65 with an argparse dump.  The grep is a NAME
    # check and cannot see the flag's CHOICES, so a tree whose
    # common/rsnn_shd.TEMPORAL_RULES is short of a rule this matrix names
    # fails at layer 2 instead, by value, which is the right message.
    "--temporal-rule",
]
#: Rung 1's two flags, added to the list a RUNG-1 ROW greps for and to no
#: other row's.  A tree without them would otherwise reach argparse and
#: abort 65 with a dump; named here that row aborts 64 saying which flag the
#: tree is short of.  Per row rather than in the shared list because the
#: shared list is rendered into every thesis launcher, and a running
#: comparison's launcher may not change for a guard its own row does not
#: need.
RUNG1_REQUIRED_FLAGS = ["--face-init-approx-per-plan",
                        "--face-init-skips-per-plan"]

#: dsnn-dfw.95's flag, added to the list a row that PASSES it greps for and
#: to no other row's, exactly as RUNG1_REQUIRED_FLAGS above.  `thesis_arm`
#: adds it, and a row that renders `dual_clip=None` drops it again, so the
#: order-only rows and the three sweep rounds keep the line they already
#: have.  A running comparison's launcher may not change for a guard its own
#: row does not need.
DUAL_CLIP_REQUIRED_FLAGS = ["--dual-clip"]

#: dsnn-dfw.98's flag, added to the list a row that PASSES it greps for
#: and to no other row's, exactly as DUAL_CLIP_REQUIRED_FLAGS above.
TARGET_KL_REQUIRED_FLAGS = ["--target-kl"]

#: dsnn-mep's flag and the three oracle flags (dsnn-dfw.208, report
#: dsnn-dfw.237), each added to the list of a row that passes it and to no
#: other row's, exactly as DUAL_CLIP_REQUIRED_FLAGS above.
MEM_OBJECTIVE_REQUIRED_FLAGS = ["--mem-objective-weight"]
GRAD_ORACLE_REQUIRED_FLAGS = ["--grad-oracle-cores", "--grad-oracle-batch",
                              "--grad-oracle-host-budget-gb"]
#: dsnn-dfw.245's two flags, on the list of a row that names its devices.
MEASURE_GPUS_REQUIRED_FLAGS = ["--gpus", "--measure-gpus"]

_THESIS_SHARED_WATERMARK = f"""WHAT IS SHARED BY EVERY ARM.  Three trained channels -- paired log-difference
latency, paired log-difference runtime watermark memory (both against
rev-exact measured back to back in the same actor; --cost-form paired-log,
--mem-channel {THESIS_MEM_CHANNEL}) and grad-cosine quality -- weighted
--lambda-cmp 1 --lambda-mem 1."""

_THESIS_SHARED_MEM_OBJECTIVE = f"""WHAT IS SHARED BY EVERY ARM.  Three trained channels -- paired log-difference
latency (against rev-exact measured back to back in the same actor;
--cost-form paired-log), the static memory objective (reward slot 11: minus
the log ratio of the total static bytes, temp + args + out, against the same
rev-exact's total, the three ratios logged only (owner ruling 2026-09-26 Q1 a,
dsnn-dfw.285),
--mem-objective-weight {THESIS_MEM_OBJECTIVE_WEIGHT}) and grad-cosine quality
-- with --rewards cmp acc and --lambda-cmp 1.  The runtime watermark stays in
slot 5 (--mem-channel {THESIS_MEM_CHANNEL}), logged and not trained (dsnn-mep,
owner ruling 2026-09-25)."""


def _thesis_head(shared: str) -> str:
    return f"""THE THESIS MATRIX (epic dsnn-dfw, ticket dsnn-dfw.4) under
the owner's rulings of 2026-09-15 and 2026-09-16.  Data collection, not a
comparison of reward designs: the campaign's phases 1-5 decided the class set,
the channels and the reward form, and these runs collect the fronts the thesis
reports.

{shared}
Terminal rewards only, gamma = GAE lambda = 1, classic init with the MVP face
head (--scale-face-head {SCALE_FACE_HEAD_MVP}, --face-logit-clamp
{FACE_LOGIT_CLAMP_MVP}), the face ADD --approx-add {APPROX_ADD}, the paired
cost floor --paired-cost-floor {THESIS_PAIRED_COST_FLOOR} (owner 2026-09-19:
the reference floor made the cost reward min(0, -log ratio), so no plan could
earn a gain below parity), and THE SPATIAL ORDER FREE
(--fixed-order {THESIS_ORDER}): the policy chooses the elimination order as
well as the approximations.

WHAT EACH RUN DOES.  --episodes {THESIS_EPISODES} WITHOUT --auto-stop (owner
2026-09-19: a final row runs its full thousand episodes; the tuning rows keep
the early stop), --checkpoint-every
{THESIS_CHECKPOINT_EVERY} so a killed run can be continued
exactly, --pareto-dump-every {THESIS_PARETO_DUMP_EVERY} and --plan-log
{THESIS_PLAN_LOG} so the front over exploration and every terminal plan are on
disk while the run is alive.

SCHEDULING.  One sbatch job per run, pinned to one Blackwell node, with
`--dependency=singleton` on a job name that IS the node.  Slurm then runs one
of our jobs per node at a time and starts the next as soon as the node frees
up.  The measurement fan-out follows the node: --ray-measure
{THESIS_RAY_MEASURE[8]} on the 8-GPU nodes and {THESIS_RAY_MEASURE[4]} on the
4-GPU nodes, one actor per GPU the trainer does not hold.

DATA.  /Scratch is NOT persistent.  The nightly copy job
(tools/thesis_nightly_copy.sh, run by a cron entry on the login host at 02:23)
mirrors the run directory -- which is where the plan log, the front dumps, the
checkpoints and auto_stop.json all land, by common/checkpoint.run_directory --
to the home export's thesis-runs directory whenever that export accepts
writes, and marks a copy complete only after a checksum list verifies."""


#: The header of a frozen round, as it ran.
THESIS_HEAD = _thesis_head(_THESIS_SHARED_WATERMARK)
#: The header of a row that trains the static memory objective.
THESIS_MATRIX_HEAD = _thesis_head(_THESIS_SHARED_MEM_OBJECTIVE)

_THESIS_ARM_WHAT = {
    "A": """CONTROL A: the fixed additive form at lambda_q """ + THESIS_LAMBDA_Q
         + """ with RAW quality
and NO floor, face-head init bias 0.  This is the reward finding 63 prices at
contrast -20.6 on this channel set: the skip-everything absorber outscores
every honest plan, so the arm is expected to allocate nothing.""",
    "B": """CONTROL B: arm A with the face-head init bias at 4, which starts
the face head AT THE IDENTITY (no approximation anywhere).  A and B together
separate "the objective has no contrast" from "the initialisation cannot
leave the identity".""",
    "C": """THE ARM: the Lagrangian dual.  Quality is a CONSTRAINT at tau """
         + THESIS_TAU + """ rather
than a weighted channel; lambda is ascended once per episode on the measured
mean violation, eta """ + DUAL_ETA + """, clipped to [""" + DUAL_LAMBDA_MIN
         + ", " + THESIS_DUAL_LAMBDA_MAX + """], started at """
         + THESIS_LAMBDA_Q + """.
THE CAP IS """ + THESIS_DUAL_LAMBDA_MAX + """ AND NOT 32 (owner 2026-09-19),
so the multiplier can dominate the skip-all plan: that plan takes about 17
nats of memory gain plus its latency gain, against a penalty of lambda times
a violation of about 0.9.""",
    "C_popart": """ARM C with PopArt: the same dual with per-channel
debiased-EMA normalisation of the value targets and sigma-scaled advantages.
--no-symlog AND --symlog-channels none ride with it (ppo.py checks the two
sites agree): symlog and PopArt address the same dynamic range and stacking
them shrinks the memory channel about 14x instead of normalising it.""",
    **{a: """DEFENSE ARM """ + a + """ (dsnn-dfw.231, owner ruling 2026-09-25,
round 3 Q1 b): arm A -- the fixed additive form with RAW quality, NO floor,
face-head init bias 0 -- at lambda_q """ + THESIS_ARM_SPEC[a][4] + """, with
the PopArt form arm C_popart carries: --advantage-norm popart, --no-symlog,
--symlog-channels none.  A, C, C_popart and the three A_popart arms span
{fixed, L} x {symlog, PopArt}.  NN256 only, three seeds, the full experiments
only: not in block 1 and not in the pilot.""" for a in THESIS_DEFENSE_ARMS},
}

_THESIS_ARM_PREDICTION = {
    "A": """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER (thesis plan
2026-09-15): A COLLAPSES TO q = 0 -- the median terminal grad-cosine falls
below 0.05 and stays there, and the front holds no point at q >= 0.9.""",
    "B": """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER: B STAYS AT q = 1 --
the terminal plans stay at or next to the identity (approx_prob/none above
0.99) for the whole run.""",
    "C": """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER: C is the arm that
produces a FRONT -- at least one terminal plan with paired latency ratio
<= 0.6 at quality >= tau by episode 250, and that plan still present in at
least 20 percent of terminal plans 250 episodes later.""",
    "C_popart": """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER: the same
front as C, reached no later, with a visibly smaller spread of the scalarized
advantage across channels.""",
    **{a: """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER (owner ruling
2026-09-25, round 3 Q1 b): PopArt alone does not save the fixed form.  At
lambda_q """ + THESIS_ARM_SPEC[a][4] + """ the front holds no terminal plan
with paired latency ratio <= 0.6 at quality >= 0.9: the arm collapses to
q = 0 as A does, or stays at the identity as B does.  PopArt rescales each
head, so only the multiplier of the L form moves the trade-off when the floor
breaks.""" for a in THESIS_DEFENSE_ARMS},
}

_THESIS_DEFENSE_HELD = """The defense arms run in the full experiments only
(owner ruling 2026-09-25, round 3 Q1 b): not in block 1 and not in the pilot.
They are GENERATED so that the 2x2 {fixed, L} x {symlog, PopArt} is
reviewable, and HELD so that a stray `sbatch fq_*.sbatch` cannot start one.
Remove `held=` from the row in tools/gen_fq_launchers.py and regenerate when
the owner releases them."""

_THESIS_FALSIFIER = """If the arm neither collapses nor produces a front but
drifts (approx_prob/none > 0.99 with no plan outside the drift floor by
episode 250), the result is reported as drift, on this reward, at this init.
The arm is NOT retuned mid-matrix and no seed is dropped: the five seeds of an
arm are reported together or not at all."""

_THESIS_HELD = """The owner authorised the FIRST BLOCK only: C and C_popart on
both targets at all five seeds, then A and B at seed """ + THESIS_SEEDS[0] + """
only (condC left the matrix on 2026-09-25).  The remaining A and B
seeds are GENERATED so that the matrix is complete and reviewable, and they
are HELD so that a stray `sbatch fq_*.sbatch` cannot start one.  Remove
`held=` from the row in tools/gen_fq_launchers.py and regenerate when the
owner releases them."""


def thesis_run_name(arm: str, target: str, seed: str) -> str:
    """`<arm>_<target>_s<seed>` (owner ruling 2026-09-16)."""
    if arm not in THESIS_ARM_SPEC:
        raise CampaignRowError(
            f"arm {arm!r} is not one of {tuple(THESIS_ARM_SPEC)}")
    if target not in THESIS_TARGET_CLI:
        raise CampaignRowError(
            f"target {target!r} is not one of {THESIS_TARGETS}")
    if seed not in THESIS_SEEDS:
        raise CampaignRowError(f"seed {seed!r} is not one of {THESIS_SEEDS}")
    return f"{arm}_{target}_s{seed}"


def thesis_cli(*, arm: str, target: str, seed: str, node: str, name: str,
               episodes: str, checkpoint_every: str,
               auto_stop: bool, grad_oracle_cadence: str = "50",
               matrix_row: bool = False,
               batched: bool = False,
               face_entropy_floor: str = "0.3",
               face_entropy_weight: str | None = None,
               ppo_epochs: str = "1", minibatches: str = "4",
               dual_clip: str | None = None,
               target_kl: str | None = None,
               ray_measure_timeout: str = CAMPAIGN_RAY_MEASURE_TIMEOUT,
               cores_per_actor: str | None = None,
               mem_objective_weight: str | None = None,
               whole_node: bool = False,
               measure_gpus: bool = False) -> dict:
    """The `cli` override dict of one thesis run.

    Everything the owner fixed is HERE, once, so the block and the smoke
    cannot disagree about anything except the three arguments the smoke
    changes on purpose (episodes, checkpoint interval, auto-stop).
    """
    spec = THESIS_ARM_SPEC.get(arm) or FROZEN_ARM_SPEC.get(arm)
    if spec is None:
        raise CampaignRowError(
            f"arm {arm!r} is not one of {tuple(THESIS_ARM_SPEC)} or the "
            f"frozen {tuple(FROZEN_ARM_SPEC)}")
    bias, form, advantage_norm, conditioned, lambda_q = spec
    if face_entropy_weight is None:
        # dsnn-dfw.84 and dsnn-dfw.78 (owner ruling 2026-09-21): the
        # recurrent target and TLM render the near-zero bonus; NN256 is
        # unmeasured by dsnn-dfw.84's finding and keeps the campaign's 0.05.
        # A caller that names its own value (the smoke rows do, to stay
        # outside this finding) is left alone.
        face_entropy_weight = (THESIS_FACE_ENTROPY_WEIGHT_NN256
                               if target == "nn256"
                               else THESIS_FACE_ENTROPY_WEIGHT_LOW)
    if rung1_row(arm, target, matrix_row):
        # The recurrent target takes a=1 (dsnn-dfw.84 and dsnn-dfw.78, owner
        # ruling 2026-09-21); NN256 keeps a=3.  kappa is the same number for
        # both (RUNG1_SKIPS_PER_PLAN).
        approx_per_plan = (RUNG1_RSNN_APPROX_PER_PLAN if target in
                           THESIS_RSNN_TARGETS else RUNG1_APPROX_PER_PLAN)
        face_init_cli = {"--face-init-approx-per-plan": approx_per_plan,
                          "--face-init-skips-per-plan": RUNG1_SKIPS_PER_PLAN}
    else:
        face_init_cli = {"--face-none-bias": bias}
    gpus = thesis_row_gpus(target, node)
    cli: dict = {
        "--name": name,
        "--seed": seed,
        # --- the target
        **thesis_target_form(target, batched)[0],
        # --- the search space
        "--approx-profile": THESIS_PROFILE,
        "--fixed-order": THESIS_ORDER,
        "--approx-add": APPROX_ADD,
        # --- the face head at init.  A rung-1 row states
        #     the PLAN it wants and lets ppo.py derive B and Bs from the
        #     reference order's face count; every other row keeps its
        #     recorded bias.  The two ways are mutually exclusive: ppo.py
        #     refuses both at once.
        **face_init_cli,
        "--scale-face-head": SCALE_FACE_HEAD_MVP,
        "--face-logit-clamp": FACE_LOGIT_CLAMP_MVP,
        "--face-entropy-weight": face_entropy_weight,
        "--face-entropy-floor": face_entropy_floor,
        "--face-entropy-floor-weight": "10.0",
        # --- the reward.  dsnn-mep: a row with the memory objective trains
        #     slot 11 and logs slot 5; a frozen round keeps "mem".
        "--rewards": ("cmp mem acc" if mem_objective_weight is None
                      else "cmp acc"),
        "--lambda-cmp": "1",
        "--lambda-mem": "1",
        "--lambda-acc": lambda_q,
        **({} if mem_objective_weight is None
           else {"--mem-objective-weight": mem_objective_weight}),
        "--paired-cost-floor": THESIS_PAIRED_COST_FLOOR,
        "--mem-channel": THESIS_MEM_CHANNEL,
        "--advantage-norm": advantage_norm,
        # --- the measurement, sized by the node
        "--ray-measure": THESIS_RAY_MEASURE[gpus],
        "--ray-measure-timeout": ray_measure_timeout,
        **(thesis_device_cli(gpus) if measure_gpus else {}),
        # --- the node's core budget, disjoint by construction
        "--reserved-driver-cores": str(
            FROZEN_CORE_BUDGET["trainer"] if cores_per_actor is None
            else THESIS_CORE_BUDGET[gpus]["trainer"]),
        "--cpu-cores-per-actor": (
            str(FROZEN_CORE_BUDGET["per_actor"])
            if cores_per_actor is None else cores_per_actor),
        # --- the gradient oracle of a row that holds the whole node
        #     (dsnn-dfw.208, report dsnn-dfw.237)
        **({} if not whole_node else {
            "--grad-oracle-cores": THESIS_GRAD_ORACLE_CORES,
            "--grad-oracle-batch": THESIS_GRAD_ORACLE_BATCH[target],
            "--grad-oracle-host-budget-gb":
                THESIS_GRAD_ORACLE_HOST_BUDGET_GB[gpus]}),
        "--measure-pipeline": CAMPAIGN_MEASURE_PIPELINE,
        "--rollout-shards": CAMPAIGN_ROLLOUT_SHARDS,
        # --- the actor's update budget (owner ruling 2026-09-20).  A matrix
        # row (through `thesis_arm`) gets THESIS_PPO_EPOCHS/THESIS_MINIBATCHES;
        # a direct `thesis_cli` caller (order-only) keeps this function's own
        # defaults, today's SHARED_CLI values, unless it says otherwise.
        "--ppo-epochs": ppo_epochs,
        "--minibatches": minibatches,
        "--tokenize-where": CAMPAIGN_TOKENIZE_WHERE,
        # dsnn-dfw.104: TLM matrix coordinates (the matrix, the pair
        # launchers and the smoke -- everything `thesis_arm` emits) render
        # the raised budget.  ONLY A MATRIX COORDINATE CAN, mirroring
        # `rung1_row`: the
        # order-only TLM final row is also target "tlm" and also goes
        # through `thesis_cli`, but it is a DIFFERENT round with its own
        # record, so `matrix_row` is False for it and it keeps 64.  Every
        # other thesis target keeps the campaign constant regardless.
        "--face-wire-faces": (THESIS_TLM_FACE_WIRE_FACES
                              if target == "tlm" and matrix_row
                              else CAMPAIGN_FACE_WIRE_FACES),
        # --- the gate inputs (G1's table is resolved from --fixed-order)
        "--gate-offline-contrast": GATE_OFFLINE_CONTRAST[THESIS_ORDER],
        # --- the run
        "--episodes": episodes,
        "--checkpoint-every": checkpoint_every,
        "--grad-oracle-cadence": grad_oracle_cadence,
        "--pareto-dump-every": THESIS_PARETO_DUMP_EVERY,
        "--plan-log": THESIS_PLAN_LOG,
    }
    if dual_clip is not None:
        # dsnn-dfw.95.  A row `thesis_arm` emits renders THESIS_DUAL_CLIP;
        # a direct `thesis_cli` caller leaves the flag off, which is ppo.py's
        # own default.
        cli["--dual-clip"] = dual_clip
    if target_kl is not None:
        # dsnn-dfw.98.  A row `thesis_arm` emits renders THESIS_TARGET_KL;
        # a direct `thesis_cli` caller leaves the flag off, which is
        # ppo.py's own default.
        cli["--target-kl"] = target_kl
    if auto_stop:
        cli["--auto-stop"] = None
    if form == "L":
        # THE LAGRANGIAN DUAL.  --quality-floor IS --lag-tau in this mode
        # (ppo.py: "under lagrangian it IS the constraint threshold"), and
        # --lambda-acc is ignored: the quality slot's weight is lambda.
        cli["--reward-mode"] = "lagrangian"
        cli["--quality-floor"] = THESIS_TAU
        cli["--lag-eta"] = DUAL_ETA
        cli["--lag-min"] = DUAL_LAMBDA_MIN
        cli["--lag-max"] = THESIS_DUAL_LAMBDA_MAX
        cli["--lag-init"] = lambda_q
    else:
        # ARMS A AND B: the fixed additive form with RAW quality and NO
        # floor (owner ruling 2026-09-16, night).  --quality-floor is not in
        # SHARED_CLI, so "no floor" is the absence of the flag, and the
        # thesis test asserts the absence rather than trusting it.
        cli["--reward-mode"] = "additive"
    if conditioned:
        cli["--preference-conditioned"] = None
    if advantage_norm == "popart":
        # The recorded trap (ticket .53): --no-symlog must be set with PopArt
        # and the three symlog sites must agree.  --symlog-channels none IS
        # --no-symlog; both are passed and ppo.py checks they agree.
        cli["--no-symlog"] = None
        cli["--symlog-channels"] = "none"
    if cli["--example"].startswith("Vmapped"):
        # dsnn-qaht: every G1 winners table was recorded on the unbatched
        # TransformerLM graph, so its vertex ids do not name a batched graph.
        cli["--gate-winners-table"] = _DELETE
    return cli


def thesis_arm(*, arm: str, target: str, seed: str, node: str,
               name: str | None = None, episodes: str | None = None,
               checkpoint_every: str = THESIS_CHECKPOINT_EVERY,
               auto_stop: bool = THESIS_FINAL_AUTO_STOP,
               what: str | None = None,
               prediction: str | None = None, held: str | None = None,
               time: str = THESIS_TIME, extra_cli: dict | None = None,
               grad_oracle_cadence: str = "50",
               face_entropy_floor: str = THESIS_FACE_ENTROPY_FLOOR,
               face_entropy_weight: str | None = None,
               ppo_epochs: str = THESIS_PPO_EPOCHS,
               minibatches: str = THESIS_MINIBATCHES,
               dual_clip: str | None = THESIS_DUAL_CLIP,
               target_kl: str | None = THESIS_TARGET_KL,
               cache_clear_every: str | None =
               THESIS_MEASURE_CACHE_CLEAR_EVERY,
               proactive_recycle_every: str | None =
               THESIS_PROACTIVE_RECYCLE_EVERY,
               ray_measure_timeout: str = THESIS_RAY_MEASURE_TIMEOUT,
               cores_per_actor: str | None = THESIS_CORES_PER_ACTOR,
               jax_cache: bool = False,
               shd_dir: bool = True,
               measure_path: bool = True,
               mem_objective_weight: str | None = THESIS_MEM_OBJECTIVE_WEIGHT,
               whole_node: bool = True,
               batched: bool = True,
               measure_gpus: bool = True) -> dict:
    """One thesis run -> one `arm(...)`.  Returns the arm."""
    _require(node in THESIS_NODES,
             f"node {node!r} is not one of the permitted thesis nodes "
             f"{THESIS_NODES} (dsnn-dfw.69: pgi15-gpu17 has no matched CUDA "
             f"12.9 ptxas and is never a node source; pgi15-gpu19 was "
             f"released back to us on 2026-09-20).")
    name = name or thesis_run_name(arm, target, seed)
    _require(arm in THESIS_ARM_SPEC,
             f"arm {arm!r} is not one of {tuple(THESIS_ARM_SPEC)}")
    _require(target in THESIS_TARGET_CLI,
             f"target {target!r} is not one of {THESIS_TARGETS}")
    _require(thesis_temporal_rule(target) in (None,) + THESIS_MATRIX_RULES,
             f"target {target!r} runs a deprecated temporal rule (owner, "
             f"2026-09-26, dsnn-dfw.232: tbptt and window2 are deprecated). "
             f"The recurrent rows run {THESIS_MATRIX_RULES}.")
    _require(arm not in THESIS_DEFENSE_ARMS or target == THESIS_DEFENSE_TARGET,
             f"arm {arm!r} is a defense arm and runs on "
             f"{THESIS_DEFENSE_TARGET} only (dsnn-dfw.231), not on {target!r}")
    # The seed is checked here as well as in `thesis_run_name`, because a row
    # that passes its own `name` (the smoke rows do) never reaches that
    # helper, and a run on an unruled seed is not part of the matrix.
    _require(seed in THESIS_SEEDS,
             f"seed {seed!r} is not one of {THESIS_SEEDS}")
    episodes = THESIS_EPISODES if episodes is None else episodes
    cli = thesis_cli(arm=arm, target=target, seed=seed, node=node, name=name,
                     episodes=episodes, checkpoint_every=checkpoint_every,
                     auto_stop=auto_stop,
                     grad_oracle_cadence=grad_oracle_cadence,
                     matrix_row=True, batched=batched,
                     face_entropy_floor=face_entropy_floor,
                     face_entropy_weight=face_entropy_weight,
                     ppo_epochs=ppo_epochs, minibatches=minibatches,
                     dual_clip=dual_clip,
                     target_kl=target_kl,
                     ray_measure_timeout=ray_measure_timeout,
                     cores_per_actor=cores_per_actor,
                     mem_objective_weight=mem_objective_weight,
                     whole_node=whole_node,
                     measure_gpus=measure_gpus)
    if extra_cli:
        cli.update(extra_cli)
    gpus = thesis_row_gpus(target, node)
    if whole_node:
        b = THESIS_CORE_BUDGET[gpus]
        _require(cli["--reserved-driver-cores"] == str(b["trainer"])
                 and cli["--cpu-cores-per-actor"] == str(b["per_actor"]),
                 f"{name}: --reserved-driver-cores "
                 f"{cli['--reserved-driver-cores']} and --cpu-cores-per-actor "
                 f"{cli['--cpu-cores-per-actor']} are not the {gpus}-GPU core "
                 f"budget {b}, so the oracle's spare cores at run time are not "
                 f"the budget's")
    a = dict(
        name=name, job=thesis_job_name(node), kind="train", runtime="scratch",
        node=node, time=time, gpus=gpus, singleton=True, thesis=True,
        thesis_arm=arm, thesis_target=target, thesis_seed=seed,
        # None on NN256 and TLM, the rule name on a recurrent row.  It is the
        # coordinate a reader wants and it is DERIVED from the target, never
        # passed in, so the two cannot disagree.
        thesis_rule=thesis_temporal_rule(target),
        # dsnn-dfw.230 (owner ruling 2026-09-25, round 1 Q19 b): no JAX
        # compile cache under any order.  A frozen round passes True and
        # keeps the cache it ran with.
        jax_cache=jax_cache,
        # dsnn-dfw.264: DSNN_SHD_DIR beside the other two data caches.  A
        # frozen round passes False and keeps its environment.
        shd_dir=shd_dir,
        # dsnn-dfw.99: the retention bound and the process recycle ride
        # with the target shape, and a row that renders `None` for either of
        # the two drops that one export again.  dsnn-dfw.169: the measure
        # path rides the same way.
        env=dict(thesis_target_form(target, batched)[1],
                 **({MEASURE_CACHE_CLEAR_EVERY_VAR: cache_clear_every}
                    if cache_clear_every is not None else {}),
                 **({PROACTIVE_RECYCLE_EVERY_VAR: proactive_recycle_every}
                    if proactive_recycle_every is not None else {}),
                 **(THESIS_MEASURE_PATH_ENV if measure_path else {}),
                 **({} if jax_cache else THESIS_NO_DISK_CACHE_ENV)),
        required_flags=(THESIS_REQUIRED_FLAGS
                        + (DUAL_CLIP_REQUIRED_FLAGS
                           if dual_clip is not None else [])
                        + (TARGET_KL_REQUIRED_FLAGS
                           if target_kl is not None else [])
                        + (MEM_OBJECTIVE_REQUIRED_FLAGS
                           if mem_objective_weight is not None else [])
                        + (GRAD_ORACLE_REQUIRED_FLAGS if whole_node else [])
                        + (MEASURE_GPUS_REQUIRED_FLAGS
                           if measure_gpus else [])
                        + (RUNG1_REQUIRED_FLAGS
                           if rung1_row(arm, target) else [])),
        required_flags_file=" ".join(THESIS_FLAGS_FILES),
        cli=cli,
        purpose=(THESIS_HEAD if mem_objective_weight is None
                 else THESIS_MATRIX_HEAD)
                + f"\n\nARM {arm} ON {target.upper()}, SEED "
                  f"{seed}: " + (what or _THESIS_ARM_WHAT[arm]),
        prediction=prediction or _THESIS_ARM_PREDICTION[arm],
        falsifier=_THESIS_FALSIFIER,
    )
    if whole_node:
        # One job of ours runs per node, so the row asks for the node's
        # memory; THESIS_ROW_MEM carries the arithmetic.
        a["mem"] = THESIS_ROW_MEM[gpus]
    if held:
        a["held"] = held
    arm_(**a)
    return a


def thesis_submission_order() -> list[tuple[str, str, str]]:
    """(arm, target, seed) in THE OWNER'S PRIORITY ORDER (2026-09-16 and
    2026-09-25: condC left the matrix, the defense arms joined it).

    1. C and C_popart, both targets, five seeds.
    2. A and B at seed 250197 only.
    3. the remaining A and B seeds -- generated, HELD, not submitted.
    4. the defense arms on NN256 at three seeds -- generated, HELD.

    The node of a run is assigned round-robin over THESIS_NODES IN THIS
    ORDER, so the first block spreads over every released node instead of
    queueing behind one of them.
    """
    order: list[tuple[str, str, str]] = []
    for a in ("C", "C_popart"):
        for t in ("tlm", "nn256"):
            for s in THESIS_SEEDS:
                order.append((a, t, s))
    for a in ("A", "B"):
        for t in ("tlm", "nn256"):
            order.append((a, t, THESIS_SEEDS[0]))
    for a in ("A", "B"):
        for t in ("tlm", "nn256"):
            for s in THESIS_SEEDS[1:]:
                order.append((a, t, s))
    for a in THESIS_DEFENSE_ARMS:
        for s in THESIS_DEFENSE_SEEDS:
            order.append((a, THESIS_DEFENSE_TARGET, s))
    return order


#: How many entries of `thesis_submission_order` the owner authorised to
#: start after the smoke: C and C_popart (20), A and B at one seed (4).  The
#: rest are held.
THESIS_BLOCK1 = 20 + 4


def thesis_arms() -> list[dict]:
    return [a for a in ARMS if a.get("thesis")]


def thesis_pair_arms() -> list[dict]:
    """The paired launchers: two half rows of one arm inside one sbatch.

    A pair is not a matrix coordinate -- its two halves are, and they are
    still rows of `thesis_core_arms` or `thesis_defense_arms` -- so it is
    excluded from every count of the matrix and listed here instead.
    """
    return [a for a in ARMS if a.get("paired") and not a.get("sweepl")
            and not a.get("sweepl2") and not a.get("sweepl3")]


def thesis_block1_arms() -> list[dict]:
    # The order-only tuning rows (tickets dsnn-dfw.29 and dsnn-dfw.45) are
    # thesis arms but not matrix coordinates: block 1 is the 24 runs of the
    # matrix the owner authorised (2026-09-16, without condC since
    # 2026-09-25) and nothing else.
    return [a for a in thesis_arms()
            if not a.get("held") and not a.get("smoke") and not a.get("paired")
            and not a.get("orderonly") and not a.get("orderonly_rsnn")
            and not a.get("sweepl") and not a.get("sweepl2")
            and not a.get("sweepl3")
            and not a.get("orderonly_final")
            and not a.get("orderonly_tlm_final")]


def thesis_snn_arms() -> list[dict]:
    """The 40 rows of the recurrent target, in generation order.

    The order-only recurrent rows (ticket dsnn-dfw.45) carry a
    `thesis_rule` too -- that is what a reader wants from them -- but they
    are a tuning round, not a matrix coordinate, so they are excluded here
    exactly as the NN256 order-only rows are excluded from
    `thesis_core_arms` below.
    """
    return [a for a in thesis_arms()
            if a.get("thesis_rule") and not a.get("orderonly_rsnn")
            and not a.get("paired")]


def thesis_core_arms() -> list[dict]:
    """The 40 NN256/TLM rows: the matrix without the recurrent target, the
    defense arms, the smoke, the order-only tuning rows and the order-only
    baseline."""
    return [a for a in thesis_arms()
            if not a.get("smoke") and not a.get("thesis_rule")
            and not a.get("orderonly") and not a.get("orderonly_final")
            and not a.get("orderonly_tlm_final") and not a.get("paired")
            and not a.get("sweepl") and not a.get("sweepl2") and not a.get("sweepl3")
            and a.get("thesis_arm") not in THESIS_DEFENSE_ARMS]


def thesis_defense_arms() -> list[dict]:
    """The 9 defense rows (dsnn-dfw.231): three arms, NN256, three seeds."""
    return [a for a in thesis_arms()
            if a.get("thesis_arm") in THESIS_DEFENSE_ARMS
            and not a.get("paired")]


# Target-pinned nodes (owner ruling 2026-09-17: "max 4* 2 tlm 2 nn256").
# pgi15-gpu19 and pgi15-gpu17 are gone (dsnn-dfw.69: another group's node,
# and no matched CUDA 12.9 ptxas): 2 nodes for TLM (pgi15-gpu20 [8 GPUs, 7
# Ray actors], pgi15-gpu16 [4 GPUs, 3 Ray actors]), 2 nodes for NN256
# (pgi15-gpu18 [4 GPUs, 3 Ray actors], pgi15-gpu15 [4 GPUs, 3 Ray actors]).
# Activated via THESIS_TARGET_NODES=1.
THESIS_TARGET_ARM_NODES = {
    ("tlm", "C"): "pgi15-gpu20",
    ("tlm", "C_popart"): "pgi15-gpu16",
    ("tlm", "A"): "pgi15-gpu16",
    ("tlm", "B"): "pgi15-gpu20",
    ("nn256", "C"): "pgi15-gpu18",
    ("nn256", "C_popart"): "pgi15-gpu15",
    ("nn256", "A"): "pgi15-gpu15",
    ("nn256", "B"): "pgi15-gpu18",
}
THESIS_TARGET_NODES = {
    "tlm": "pgi15-gpu20",
    "nn256": "pgi15-gpu18",
}
_USE_TARGET_NODES = os.environ.get("THESIS_TARGET_NODES", "0") == "1"

# ---------------------------------------------------------------------------
# THE SEVEN SLOTS (owner ruling 2026-09-20).  An NN256 row occupies four
# Blackwell GPUs wherever it runs, so a 4-GPU node holds ONE row and an
# 8-GPU node holds TWO.  The slot ring is therefore three whole nodes plus
# two halves on each of the two 8-GPU nodes: seven places, not five.
#
# A HALF IS NOT A JOB.  `/etc/slurm/epilog_reset_node.sh` kills every process
# of this user on a node the moment ANY job of theirs on it ends, so two jobs
# of ours on one node kill each other (memory note
# pgi15-epilog-kills-sibling-jobs).  Two halves of one node are therefore
# rendered into ONE sbatch by `thesis_pair_arm`, and the two rows that make
# it up keep their own record and their own wandb run while their launcher
# files become stubs that refuse (exit 74) and name the pair.
# ---------------------------------------------------------------------------
def thesis_slots() -> tuple[tuple[str, int | None], ...]:
    """(node, half) for every place an NN256 row can run, in node order.

    `half` is None on a node the row fills whole, else the index of the
    half (0 = the low GPUs and the low cores, 1 = the high ones).
    """
    per = THESIS_UNIFORM_GPUS["nn256"]
    slots: list[tuple[str, int | None]] = []
    for node in THESIS_NODES:
        gpus = THESIS_NODE_GPUS[node]
        if gpus % per:
            raise CampaignRowError(
                f"node {node!r} carries {gpus} GPUs, which is not a whole "
                f"number of {per}-GPU halves; the slot ring cannot split it")
        if gpus == per:
            slots.append((node, None))
        else:
            slots.extend((node, h) for h in range(gpus // per))
    return tuple(slots)


THESIS_SLOTS = thesis_slots()
#: The letter a half is called by in a paired launcher, by half index.
_PAIR_TAGS = ("A", "B")


def thesis_pair_name(arm: str, target: str, seeds: tuple[str, str]) -> str:
    """`<arm>_<target>_s<seedA>_s<seedB>_pair`, the paired launcher's file."""
    return f"{arm}_{target}_s{seeds[0]}_s{seeds[1]}_pair"


def thesis_pair_arm(rows: list[dict]) -> dict:
    """Two half rows of ONE node and ONE arm -> one paired `arm(...)`.

    The halves keep their own `--name` (their wandb run is the matrix's own
    coordinate), their own GPUs, their own cores, their own RAY_TMPDIR and
    their own log file.  The job exits with the WORSE of the two trainer
    codes, so a half that crashes cannot be hidden by a sibling that did not.
    """
    _require(len(rows) == 2, f"a pair is two rows, not {len(rows)}")
    a, b = sorted(rows, key=lambda r: r["half"])
    node = a["node"]
    arm_name, target = a["thesis_arm"], a["thesis_target"]
    _require(b["node"] == node and b["thesis_arm"] == arm_name
             and b["thesis_target"] == target,
             "a pair is two seeds of ONE arm on ONE node")
    _require(a["thesis_seed"] != b["thesis_seed"],
             "a pair is two DIFFERENT seeds")
    _require(bool(a.get("held")) == bool(b.get("held")),
             f"{a['name']} and {b['name']} disagree about being held; a "
             f"paired job releases or holds both halves at once")
    gpus = THESIS_NODE_GPUS[node]
    per = THESIS_UNIFORM_GPUS[target]
    cpus = node_cpus(node, gpus)
    per_cpus = cpus // (gpus // per)
    halves = []
    for r in (a, b):
        h = r["half"]
        _require("--measure-gpus" in r["cli"],
                 f"{r['name']} names no measure GPUs, so its half cannot name "
                 f"its own (dsnn-dfw.245)")
        halves.append(dict(
            name=r["name"], seed=r["thesis_seed"], half=h,
            devices=",".join(str(h * per + d) for d in range(per)),
            cores=f"{h * per_cpus}-{(h + 1) * per_cpus - 1}",
            cli=dict(r["cli"], **thesis_device_cli(per, h)),
        ))
    name = thesis_pair_name(arm_name, target,
                            (a["thesis_seed"], b["thesis_seed"]))
    for r in (a, b):
        r["paired_into"] = name
    p = dict(
        name=name, job=thesis_job_name(node), kind="train", runtime="scratch",
        node=node, time=a["time"], gpus=gpus, singleton=True, thesis=True,
        paired=True, halves=halves,
        jax_cache=a["jax_cache"], shd_dir=a["shd_dir"],
        thesis_arm=arm_name, thesis_target=target,
        thesis_rule=thesis_temporal_rule(target),
        env=dict(a["env"]),
        required_flags=a["required_flags"],
        required_flags_file=a["required_flags_file"],
        # The preflight reads ONE command line (the two differ in --seed,
        # --name, --gpus and --measure-gpus alone); the ARGS arrays are
        # rendered from `halves`.
        cli=a["cli"],
        purpose=a["purpose"] + f"\n\nPAIRED ON {node}: seeds "
                f"{a['thesis_seed']} and {b['thesis_seed']} of arm "
                f"{arm_name} run CONCURRENTLY inside this one job, each on "
                f"{per} of the node's {gpus} GPUs and {per_cpus} of its "
                f"{cpus} cores.  One job, because the node epilog kills a "
                f"sibling job of this user on the same node.",
        prediction=a["prediction"], falsifier=a["falsifier"],
    )
    if a.get("mem"):
        # The two halves hold the whole node, so the pair asks for its memory.
        p["mem"] = THESIS_ROW_MEM[gpus]
    if a.get("held"):
        p["held"] = a["held"]
    arm_(**p)
    return p


# --- the 49 runs of the matrix and the defense arms ------------------------
_HALVES: dict[tuple[str, str], list[dict]] = {}
_SLOT = 0
_row: dict = {}
for _i, (_arm, _target, _seed) in enumerate(thesis_submission_order()):
    _half = None
    if _USE_TARGET_NODES:
        _node = THESIS_TARGET_ARM_NODES.get(
            (_target, _arm), THESIS_TARGET_NODES.get(_target, "pgi15-gpu16"))
    elif _target in THESIS_UNIFORM_GPUS:
        _node, _half = THESIS_SLOTS[_SLOT % len(THESIS_SLOTS)]
        _SLOT += 1
    else:
        _node = THESIS_NODES[_i % len(THESIS_NODES)]
    thesis_arm(
        arm=_arm, target=_target, seed=_seed,
        node=_node,
        held=(None if _i < THESIS_BLOCK1
              else _THESIS_DEFENSE_HELD if _arm in THESIS_DEFENSE_ARMS
              else _THESIS_HELD),
    )
    if _half is not None:
        # ARMS[-1], not the dict `thesis_arm` returns: `arm(**kw)` re-packs
        # its keywords, so the returned dict is a COPY and a `paired_into`
        # written on it would never reach the renderer.
        _row = ARMS[-1]
        _row["half"] = _half
        _HALVES.setdefault((_arm, _node), []).append(_row)
# A slot that no sibling joined stays a whole single-node job of four GPUs on
# an 8-GPU node: legal, one job on the node, the same profile as every other
# NN256 row.  Only a node that carries TWO rows of one arm becomes a pair.
# The pairs are on with the pair fix (dsnn-dfw.245): each half names its own GPUs. THESIS_PAIRS=0 turns them off.
THESIS_PAIRS = os.environ.get("THESIS_PAIRS", "1") == "1"
_rows: list[dict] = []
for _rows in _HALVES.values():
    if THESIS_PAIRS and len(_rows) == 2:
        thesis_pair_arm(_rows)
del _i, _arm, _target, _seed, _node, _half, _row, _rows, _HALVES, _SLOT


# ---------------------------------------------------------------------------
# THE RECURRENT BLOCK (owner ruling 2026-09-16).  The same four arms and the
# same five seeds as the matrix above, on --example RSNN_SHD --dataset shd,
# crossed with the two temporal rules of THESIS_MATRIX_RULES: 2 x 4 x 5 = 40
# rows (100 before condC left the matrix on 2026-09-25, 80 before the tbptt
# rows and 60 before the window2 rows left it, dsnn-dfw.232).
#
# NOTHING ELSE MOVES.  These rows go through the same `thesis_cli`, so the
# reward form per arm, the free spatial order, the thousand episodes, the
# auto-stop, the checkpoint and dump intervals, the plan log, the measurement
# protocol, the gate inputs, the per-node singleton and the --ray-measure
# rule (7 on the 8-GPU nodes, 3 on the 4-GPU nodes) are the ones the NN256
# and TLM rows carry, flag for flag.  The ONLY flag a recurrent row adds is
# --temporal-rule, and it adds no environment variable at all.
#
# THE ORDER of generation is rule, then arm, then seed.  It is a GENERATION
# order and not a submission order: `thesis_submission_order` is the owner's
# priority list and it holds the NN256 and TLM rows only, because no
# recurrent row is released to be submitted.
#
# THE NODE, per rule (owner ruling 2026-09-21, agent rsnnpace's pace
# diagnosis).  rtrl is the one rule whose measurement pipeline stalls once
# the plan carries real approximations: 3 measure actors cannot absorb its
# paired reference (a 188 MB carried Jacobian), so every rtrl row -- every
# arm -- renders on an 8-GPU node with --ray-measure 7, round-robin BY SEED
# over the two released 8-GPU nodes (gpu19 first) so a seed always lands on
# the same node and no rtrl row is paired with another on the same node
# (`thesis_pair_arm` is never called here).  bptt keeps the 4-GPU profile,
# round-robin over the three released 4-GPU nodes: 20 rows over 3 nodes is
# 7, 7 and 6.
# ---------------------------------------------------------------------------
#: The two 8-GPU nodes, in round-robin order (gpu19 first, owner ruling).
THESIS_RSNN_RTRL_NODES = ("pgi15-gpu19", "pgi15-gpu20")
#: The three 4-GPU nodes every other recurrent rule renders on.
THESIS_RSNN_OTHER_NODES = ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18")


def thesis_snn_node(target: str, seed: str, i: int) -> str:
    """The node of one recurrent row.

    rtrl: round-robin BY SEED over the two 8-GPU nodes, so every arm of one
    seed lands on the same node and a seed's node never depends on
    generation order.  Every other rule: round-robin over the three 4-GPU
    nodes by `i`, the caller's own counter over non-rtrl rows.
    """
    if thesis_temporal_rule(target) == "rtrl":
        return THESIS_RSNN_RTRL_NODES[
            THESIS_SEEDS.index(seed) % len(THESIS_RSNN_RTRL_NODES)]
    return THESIS_RSNN_OTHER_NODES[i % len(THESIS_RSNN_OTHER_NODES)]

_THESIS_RSNN_WHAT = {
    "bptt": """TEMPORAL RULE bptt: THE FUTURE FEEDS IN.  An edge from the next
state to the loss carries the adjoint dL(>t)/ds_t from a detached backward
pass over the suffix, so the gradient is exactly the contribution step t
makes to full backpropagation through time.""",
    "rtrl": """TEMPORAL RULE rtrl: THE PAST FEEDS IN.  An edge from the
weights to the carried state carries the influence matrix ds(t-1)/dW from a
detached pass over the prefix, so eliminating that vertex is one real-time
recurrent-learning step and the gradient is exactly dL_t/dW through the whole
prefix.""",
}

_THESIS_SNN_HELD = """The recurrent block is GENERATED so that the matrix is
complete and reviewable, and every one of its 40 rows is HELD: the owner
authorised the NN256/TLM first block only and released no RSNN_SHD run.  A
held launcher that is submitted by mistake aborts 73 before it starts
anything.  Remove `held=` from the recurrent loop in
tools/gen_fq_launchers.py and regenerate when the owner releases them."""


def thesis_snn_order() -> list[tuple[str, str, str]]:
    """(arm, target, seed) for the 40 recurrent rows, in GENERATION order.

    Rule, then arm, then seed.  Not a submission order: every row is held.
    """
    order: list[tuple[str, str, str]] = []
    for rule in THESIS_MATRIX_RULES:
        for a in THESIS_ARMS:
            for s in THESIS_SEEDS:
                order.append((a, f"rsnn_{rule}", s))
    return order


# --- the 40 runs of the recurrent block -------------------------------------
_rsnn_other_i = 0
for _i, (_arm, _target, _seed) in enumerate(thesis_snn_order()):
    if thesis_temporal_rule(_target) == "rtrl":
        _node = thesis_snn_node(_target, _seed, _i)
    else:
        _node = thesis_snn_node(_target, _seed, _rsnn_other_i)
        _rsnn_other_i += 1
    thesis_arm(
        arm=_arm, target=_target, seed=_seed,
        node=_node,
        what=_THESIS_ARM_WHAT[_arm] + "\n\n"
             + _THESIS_RSNN_WHAT[thesis_temporal_rule(_target)],
        held=_THESIS_SNN_HELD,
    )
del _i, _arm, _target, _seed, _node, _rsnn_other_i


# ---------------------------------------------------------------------------
# THE SMOKE (owner ruling 2026-09-16).  Two short runs on gpu16 that prove
# the block can start, BEFORE the block is queued:
#
#   1. arm C on TLM, 20 episodes, --checkpoint-every 10, --auto-stop OFF.
#      Auto-stop is off because its first check point is after 250 episodes;
#      on a 20-episode run it would only print that no check point is
#      reachable, and the smoke is about the checkpoint, not about it.
#   2. the same command line plus --resume <the episode-10 checkpoint>.  The
#      resume refuses any command line that differs from the checkpoint's in
#      anything but --episodes and --resume, so this arm MUST be byte-equal
#      to arm 1 apart from that one flag -- which is why both are generated
#      from one `thesis_cli` call instead of being written twice.  The
#      checkpoint path is a shell placeholder the submitter exports.
#
# The third run, condC on NN256, left with condC (owner rulings 2026-09-25).
# Both carry the per-node singleton name of gpu16, so they queue behind each
# other and behind the block instead of sharing the node with it (the epilog
# kills siblings).
# ---------------------------------------------------------------------------
THESIS_SMOKE_EPISODES = "20"
THESIS_SMOKE_CHECKPOINT_EVERY = "10"
_THESIS_RESUME_PLACEHOLDER = (
    "${THESIS_RESUME:?export THESIS_RESUME to the episode-10 checkpoint "
    "directory of the smoke run, e.g. .../ppo_ckpt_ep000000010}")

_SMOKE_WHAT = """THE SMOKE, not a result.  It proves that the thesis command
line starts on this stack, that a checkpoint is written, and that a resume
continues the SAME run (the plan log appends and the front dumps keep
coming).  No claim is made about learning in 20 episodes."""

_SMOKE_PREDICTION = """REGISTERED BEFORE THE RUN: the run reaches episode 20,
writes ppo_ckpt_ep000000010 and ppo_ckpt_ep000000020, appends to
plan_log_<name>.jsonl every episode and dumps a front every 10 episodes; the
resumed leg starts at episode 10, does not re-run episodes 0-9, and appends to
the same plan log and front series."""

_SMOKE_FALSIFIER = """If the resume raises, or the plan log or the front dumps
restart instead of continuing, the block is NOT submitted and the failure is
reported as it stands."""

thesis_arm(
    arm="C", target="tlm", seed=THESIS_SEEDS[0], node=THESIS_SMOKE_NODE,
    name="smoke_C_tlm", episodes=THESIS_SMOKE_EPISODES,
    # dsnn-dfw.78 and owner ruling 2026-09-20: smoke is outside the thesis
    # matrix, on the entropy floor and the update budget.  dsnn-dfw.84
    # (2026-09-21) is a TLM/recurrent matrix finding too; smoke keeps its
    # own recorded 0.05 rather than picking up the near-zero bonus.
    face_entropy_floor="0.3", face_entropy_weight="0.05",
    ppo_epochs="1", minibatches="4",
    checkpoint_every=THESIS_SMOKE_CHECKPOINT_EVERY, auto_stop=False,
    grad_oracle_cadence="10",
    time="04:00:00", what=_SMOKE_WHAT, prediction=_SMOKE_PREDICTION,
)
ARMS[-1]["smoke"] = True
ARMS[-1]["falsifier"] = _SMOKE_FALSIFIER

thesis_arm(
    arm="C", target="tlm", seed=THESIS_SEEDS[0], node=THESIS_SMOKE_NODE,
    name="smoke_C_tlm", episodes=THESIS_SMOKE_EPISODES,
    # dsnn-dfw.78 and owner ruling 2026-09-20: smoke is outside the thesis
    # matrix, on the entropy floor and the update budget.  dsnn-dfw.84
    # (2026-09-21) is a TLM/recurrent matrix finding too; smoke keeps its
    # own recorded 0.05 rather than picking up the near-zero bonus.
    face_entropy_floor="0.3", face_entropy_weight="0.05",
    ppo_epochs="1", minibatches="4",
    checkpoint_every=THESIS_SMOKE_CHECKPOINT_EVERY, auto_stop=False,
    grad_oracle_cadence="10",
    time="04:00:00", what=_SMOKE_WHAT, prediction=_SMOKE_PREDICTION,
    extra_cli={"--resume": _THESIS_RESUME_PLACEHOLDER},
)
ARMS[-1]["smoke"] = True
ARMS[-1]["falsifier"] = _SMOKE_FALSIFIER
# The FILE name differs (a launcher per file) while --name does NOT: a resume
# whose --name differed from the checkpoint's is refused, and that refusal
# already cost one agent a suite run (agent-ckpt-report section 9).
ARMS[-1]["name"] = "smoke_C_tlm_resume"


def thesis_smoke_arms() -> list[dict]:
    return [a for a in ARMS if a.get("smoke")]


# ================  LAGRANGIAN DUAL SWEEP, ROUND 1  ================
# Owner ruling 2026-09-20 (epic dsnn-dfw, same day as .78's floor fix).  A
# short one-factor-at-a-time probe of the Lagrangian dual's four knobs
# around arm C's own rung-1 center on NN256, before any of them is retuned
# for the matrix.  100 episodes, no --auto-stop.  NOT a matrix coordinate:
# `thesis_core_arms` and `thesis_block1_arms` exclude every row this section
# marks `sweepl=True`, the same way they exclude the order-only rounds.
#
# GRID ARITHMETIC (report this, do not silently fix it): one factor at a
# time around a shared center over the value sets below gives 2 + 1 + 2 + 2
# = 7 non-center configurations plus the center itself, 8 distinct
# configurations, not 9 -- lag-max names only ONE non-center value (256), so
# it contributes 1, not 2.  8 configurations x 3 seeds = 24 rows, not 27.
# The 24 rows below are exactly the 8 configurations the stated value sets
# admit; no config was invented to reach 9.
SWEEPL_TARGET = "nn256"
SWEEPL_ARM = "C"
SWEEPL_SEEDS = THESIS_SEEDS[:3]
SWEEPL_EPISODES = "100"
#: THE CENTER, read off arm C's own constants so the sweep and the matrix
#: cannot silently disagree about what "unperturbed" means.
SWEEPL_CENTER: dict[str, str] = {
    "--lag-eta": DUAL_ETA,
    "--lag-max": THESIS_DUAL_LAMBDA_MAX,
    "--lag-init": THESIS_LAMBDA_Q,
    "--quality-floor": THESIS_TAU,
}
#: ROUND 1 RAN AT TODAY'S UPDATE BUDGET (--ppo-epochs 1, --minibatches 4),
#: before the owner's 2026-09-20 ruling raised the matrix's own budget to
#: THESIS_PPO_EPOCHS/THESIS_MINIBATCHES.  Named here and merged into every
#: row's overrides so the round's recorded meaning is frozen and does not
#: drift when `thesis_arm`'s own default budget changes.
SWEEPL_TODAY_BUDGET: dict[str, str] = {
    "--ppo-epochs": "1",
    "--minibatches": "4",
}
#: One factor at a time.  Each tuple includes the center value so a reader
#: sees the whole probed range; `sweepl_configs` skips the center value when
#: it walks a factor, else the center would render twice.
SWEEPL_GRID: dict[str, tuple[str, ...]] = {
    "--lag-eta": ("0.5", "2.0", "8.0"),
    "--lag-max": ("64", "256"),
    "--lag-init": ("8", "16", "40"),
    "--quality-floor": ("0.85", "0.90", "0.95"),
}
SWEEPL_TAG = {
    "--lag-eta": "eta", "--lag-max": "lagmax",
    "--lag-init": "laginit", "--quality-floor": "qfloor",
}

_SWEEPL_WHAT = """A ONE-FACTOR-AT-A-TIME PROBE of the Lagrangian dual around
arm C's own rung-1 center on NN256 (--lag-eta 2.0, --lag-max 64, --lag-init
16, --quality-floor 0.90), 100 episodes, no --auto-stop: short enough to read
before any of the four is retuned for the matrix.

READ-OUT: the fraction of plans above the quality floor at episodes 80-100,
and the multiplier's distance from its cap (lambda / lag-max) at the same
window.  The paired latency ratio of the feasible plans among those (the
ones that cleared the quality floor) is the tie-breaker between
configurations that tie on the first number."""

_SWEEPL_LAGMIN_NOTE = """

LAG-INIT 8 IS BELOW THE MATRIX'S --lag-min 12: this row sets --lag-min 8 too,
so the initial multiplier is not clamped above its own starting point before
the first update (dsnn-dfw epic, 2026-09-20 sweep ruling)."""


def sweepl_configs() -> list[tuple[str, dict[str, str]]]:
    """(tag, cli overrides) for the 8 distinct configurations: the center,
    then one perturbation per grid value that is not already the center."""
    configs: list[tuple[str, dict[str, str]]] = [
        ("center", dict(SWEEPL_CENTER))]
    for flag, values in SWEEPL_GRID.items():
        for v in values:
            if v == SWEEPL_CENTER[flag]:
                continue
            cli = dict(SWEEPL_CENTER)
            cli[flag] = v
            if flag == "--lag-init" and v == "8":
                cli["--lag-min"] = "8"
            configs.append((f"{SWEEPL_TAG[flag]}_{v}", cli))
    return configs


def sweepl_row_name(tag: str, seed: str) -> str:
    return (f"sweepL_nn256_center_s{seed}" if tag == "center"
            else f"sweepL_nn256_{tag}_s{seed}")


def sweepl_pair_name(tag: str, seeds: tuple[str, str]) -> str:
    return f"sweepL_nn256_{tag}_s{seeds[0]}_s{seeds[1]}_pair"


def sweepl_pair_arm(rows: list[dict]) -> dict:
    """Two half rows of ONE sweep configuration, one 8-GPU node -> one paired
    `arm(...)`.  A copy of `thesis_pair_arm`'s body under a sweep-shaped
    name: that function names a pair by arm+target alone, which two sweep
    configurations of arm C on NN256 would collide on.  Carries `paired=True`
    (so the generic per-row NN256 checks skip it, exactly like a matrix pair)
    and `sweepl=True` (so `thesis_pair_arms` -- matrix pairs only -- excludes
    it); half rows point back at it through `sweepl_paired_into`, never the
    matrix's own `paired_into`, so the two pairing records cannot cross.
    """
    _require(len(rows) == 2, f"a sweep pair is two rows, not {len(rows)}")
    a, b = sorted(rows, key=lambda r: r["half"])
    node = a["node"]
    tag = a["sweepl_tag"]
    _require(b["node"] == node and b["sweepl_tag"] == tag,
             "a sweep pair is two seeds of ONE configuration on ONE node")
    _require(a["thesis_seed"] != b["thesis_seed"],
             "a sweep pair is two DIFFERENT seeds")
    gpus = THESIS_NODE_GPUS[node]
    per = THESIS_UNIFORM_GPUS[SWEEPL_TARGET]
    cpus = node_cpus(node, gpus)
    per_cpus = cpus // (gpus // per)
    halves = []
    for r in (a, b):
        h = r["half"]
        halves.append(dict(
            name=r["name"], seed=r["thesis_seed"], half=h,
            devices=",".join(str(h * per + d) for d in range(per)),
            cores=f"{h * per_cpus}-{(h + 1) * per_cpus - 1}",
            cli=r["cli"],
        ))
    name = sweepl_pair_name(tag, (a["thesis_seed"], b["thesis_seed"]))
    for r in (a, b):
        r["sweepl_paired_into"] = name
    p = dict(
        name=name, job=thesis_job_name(node), kind="train", runtime="scratch",
        node=node, time=a["time"], gpus=gpus, singleton=True, thesis=True,
        sweepl=True, paired=True, halves=halves,
        thesis_arm=SWEEPL_ARM, thesis_target=SWEEPL_TARGET,
        env=dict(a["env"]),
        required_flags=a["required_flags"],
        required_flags_file=a["required_flags_file"],
        cli=a["cli"],
        purpose=a["purpose"] + f"\n\nPAIRED ON {node}: seeds "
                f"{a['thesis_seed']} and {b['thesis_seed']} of sweep "
                f"configuration {tag} run CONCURRENTLY, each on {per} of "
                f"the node's {gpus} GPUs and {per_cpus} of its {cpus} cores.",
        prediction=a["prediction"], falsifier=a["falsifier"],
    )
    arm_(**p)
    return p


def sweepl_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl")]


def sweepl_single_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl") and not a.get("paired")]


_SWEEPL_HALVES: dict[tuple[str, str], list[dict]] = {}
_sweepl_slot = 0
for _sweepl_tag, _sweepl_overrides in sweepl_configs():
    for _sweepl_seed in SWEEPL_SEEDS:
        _sweepl_node, _sweepl_half = THESIS_SLOTS[
            _sweepl_slot % len(THESIS_SLOTS)]
        _sweepl_slot += 1
        _sweepl_what = _SWEEPL_WHAT + (
            _SWEEPL_LAGMIN_NOTE if "--lag-min" in _sweepl_overrides else "")
        thesis_arm(
            arm=SWEEPL_ARM, target=SWEEPL_TARGET, seed=_sweepl_seed,
            node=_sweepl_node,
            name=sweepl_row_name(_sweepl_tag, _sweepl_seed),
            episodes=SWEEPL_EPISODES, what=_sweepl_what,
            extra_cli=dict(_sweepl_overrides, **SWEEPL_TODAY_BUDGET),
            # dsnn-dfw.95: this round is FROZEN (see the budget note above),
            # so it keeps --dual-clip off and its launchers do not move.
            dual_clip=None,
            # dsnn-dfw.98: same reason, keeps --target-kl off too.
            target_kl=None,
            # dsnn-dfw.99: same reason, keeps the retention bound off too.
            cache_clear_every=None,
            # dsnn-dfw.99: same reason, keeps the process recycle off too.
            proactive_recycle_every=None,
            # Owner ruling 2026-09-23 moves the matrix rows; this round is
            # frozen and keeps its timeout, cores and compile cache.
            ray_measure_timeout=CAMPAIGN_RAY_MEASURE_TIMEOUT,
            cores_per_actor=None,
            jax_cache=True,
            # dsnn-dfw.264 moves the matrix rows; this round keeps its env.
            shd_dir=False,
            # The rulings of 2026-09-25 (dsnn-dfw.169, dsnn-mep, dsnn-dfw.208)
            # move the matrix rows; this round keeps its measure path, its
            # memory channel, its oracle and its memory request.
            measure_path=False,
            mem_objective_weight=None,
            whole_node=False,
            # dsnn-dfw.245 moves the matrix rows; this round keeps its devices.
            measure_gpus=False,
            # dsnn-dfw.191: same reason, keeps the unbatched target too.
            batched=False,
        )
        ARMS[-1]["sweepl"] = True
        ARMS[-1]["sweepl_tag"] = _sweepl_tag
        if _sweepl_half is not None:
            _sweepl_row = ARMS[-1]
            _sweepl_row["half"] = _sweepl_half
            _SWEEPL_HALVES.setdefault(
                (_sweepl_tag, _sweepl_node), []).append(_sweepl_row)
_sweepl_row: dict = {}
_sweepl_rows: list[dict] = []
for _sweepl_rows in _SWEEPL_HALVES.values():
    if len(_sweepl_rows) == 2:
        sweepl_pair_arm(_sweepl_rows)
del (_sweepl_tag, _sweepl_overrides, _sweepl_seed, _sweepl_node,
     _sweepl_half, _sweepl_what, _sweepl_row, _sweepl_rows, _sweepl_slot,
     _SWEEPL_HALVES)


# ================  LAGRANGIAN DUAL SWEEP, ROUND 2  ================
# Owner ruling 2026-09-20, after round 1's read-out (no configuration left
# the center seeds' own band within 100 episodes; a multiplier of 94 moves
# the feasible fraction no more than 33 does).  Round 2 asks whether the
# actor responds to the dual given more time and a bigger update, not
# whether the dual's constants matter: two finalists, the rung-1 CENTER and
# CAP256 (--lag-max 256, else center), each at today's PPO update budget
# and a larger one, 300 episodes, no --auto-stop.  NOT a matrix coordinate:
# `thesis_core_arms`, `thesis_block1_arms` and `thesis_pair_arms` exclude
# every row this section marks `sweepl2=True`, the same way they exclude
# round 1's `sweepl=True` rows.
SWEEPL2_TARGET = "nn256"
SWEEPL2_ARM = "C"
SWEEPL2_SEEDS = THESIS_SEEDS[:3]
SWEEPL2_EPISODES = "100"
#: dual tag -> the cli overrides that differ from arm C's own rung-1
#: defaults.  CENTER overrides nothing: it IS arm C's own dual constants
#: (--lag-eta 2.0, --lag-max 64, --lag-init 16, --quality-floor 0.90).
SWEEPL2_DUAL: dict[str, dict[str, str]] = {
    "center": {},
    "cap256": {"--lag-max": "256"},
}
#: budget tag -> the cli overrides that differ from today's update budget
#: (--ppo-epochs 1, --minibatches 4, the SHARED_CLI defaults).  B1 states
#: that budget EXPLICITLY (owner ruling 2026-09-20 later raised the matrix's
#: own default to THESIS_PPO_EPOCHS/THESIS_MINIBATCHES; this round is frozen
#: for comparison, not left to inherit whatever the ambient default is).
SWEEPL2_BUDGET: dict[str, dict[str, str]] = {
    "b1": {"--ppo-epochs": "1", "--minibatches": "4"},
    "b4": {"--ppo-epochs": "2", "--minibatches": "8"},
}

_SWEEPL2_WHAT = """SWEEP ROUND 2 (owner ruling 2026-09-20, after round 1): the
two finalists -- the rung-1 CENTER (--lag-eta 2.0, --lag-max 64, --lag-init
16, --quality-floor 0.90) and CAP256 (--lag-max 256, else center) -- each at
today's PPO update budget (--ppo-epochs 1, --minibatches 4) and a larger one
(--ppo-epochs 2, --minibatches 8), 300 episodes, no --auto-stop, arm C on
NN256 at the normalized init, entropy floor 0.05.

PURPOSE: does the actor respond to the dual within 300 episodes, and is the
update budget the bottleneck rather than the dual's constants.

READ-OUT: the feasible fraction at q >= 0.9 over episodes 80-99 from the
plan log, the mean q over the same window, the multiplier's trajectory
against its cap, the latency reward of the feasible plans, and how many
feasible plans run faster than the reference."""


def sweepl2_configs() -> list[tuple[str, dict[str, str]]]:
    """(tag, cli overrides) for the 4 distinct configurations: dual x
    budget, tagged `<dual>_<budget>`."""
    configs: list[tuple[str, dict[str, str]]] = []
    for dual, dual_cli in SWEEPL2_DUAL.items():
        for budget, budget_cli in SWEEPL2_BUDGET.items():
            cli = dict(dual_cli)
            cli.update(budget_cli)
            configs.append((f"{dual}_{budget}", cli))
    return configs


def sweepl2_row_name(tag: str, seed: str) -> str:
    return f"sweepL2_nn256_{tag}_s{seed}"


def sweepl2_pair_name(tag: str, seeds: tuple[str, str]) -> str:
    return f"sweepL2_nn256_{tag}_s{seeds[0]}_s{seeds[1]}_pair"


def sweepl2_pair_arm(rows: list[dict]) -> dict:
    """Two half rows of ONE sweep-round-2 configuration, one 8-GPU node ->
    one paired `arm(...)`.  A copy of `sweepl_pair_arm`'s body under round
    2's own name: half rows point back at it through `sweepl2_paired_into`,
    never round 1's `sweepl_paired_into` or the matrix's `paired_into`, so
    the three pairing records cannot cross.
    """
    _require(len(rows) == 2, f"a sweep pair is two rows, not {len(rows)}")
    a, b = sorted(rows, key=lambda r: r["half"])
    node = a["node"]
    tag = a["sweepl2_tag"]
    _require(b["node"] == node and b["sweepl2_tag"] == tag,
             "a sweep pair is two seeds of ONE configuration on ONE node")
    _require(a["thesis_seed"] != b["thesis_seed"],
             "a sweep pair is two DIFFERENT seeds")
    gpus = THESIS_NODE_GPUS[node]
    per = THESIS_UNIFORM_GPUS[SWEEPL2_TARGET]
    cpus = node_cpus(node, gpus)
    per_cpus = cpus // (gpus // per)
    halves = []
    for r in (a, b):
        h = r["half"]
        halves.append(dict(
            name=r["name"], seed=r["thesis_seed"], half=h,
            devices=",".join(str(h * per + d) for d in range(per)),
            cores=f"{h * per_cpus}-{(h + 1) * per_cpus - 1}",
            cli=r["cli"],
        ))
    name = sweepl2_pair_name(tag, (a["thesis_seed"], b["thesis_seed"]))
    for r in (a, b):
        r["sweepl2_paired_into"] = name
    p = dict(
        name=name, job=thesis_job_name(node), kind="train", runtime="scratch",
        node=node, time=a["time"], gpus=gpus, singleton=True, thesis=True,
        sweepl2=True, paired=True, halves=halves,
        thesis_arm=SWEEPL2_ARM, thesis_target=SWEEPL2_TARGET,
        env=dict(a["env"]),
        required_flags=a["required_flags"],
        required_flags_file=a["required_flags_file"],
        cli=a["cli"],
        purpose=a["purpose"] + f"\n\nPAIRED ON {node}: seeds "
                f"{a['thesis_seed']} and {b['thesis_seed']} of sweep round "
                f"2 configuration {tag} run CONCURRENTLY, each on {per} of "
                f"the node's {gpus} GPUs and {per_cpus} of its {cpus} "
                "cores.",
        prediction=a["prediction"], falsifier=a["falsifier"],
    )
    arm_(**p)
    return p


def sweepl2_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl2")]


def sweepl2_single_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl2") and not a.get("paired")]


_SWEEPL2_HALVES: dict[tuple[str, str], list[dict]] = {}
_sweepl2_slot = 0
for _sweepl2_tag, _sweepl2_overrides in sweepl2_configs():
    for _sweepl2_seed in SWEEPL2_SEEDS:
        _sweepl2_node, _sweepl2_half = THESIS_SLOTS[
            _sweepl2_slot % len(THESIS_SLOTS)]
        _sweepl2_slot += 1
        thesis_arm(
            arm=SWEEPL2_ARM, target=SWEEPL2_TARGET, seed=_sweepl2_seed,
            node=_sweepl2_node,
            name=sweepl2_row_name(_sweepl2_tag, _sweepl2_seed),
            episodes=SWEEPL2_EPISODES, what=_SWEEPL2_WHAT,
            extra_cli=dict(_sweepl2_overrides),
            # dsnn-dfw.95: this round is FROZEN (see the budget note above),
            # so it keeps --dual-clip off and its launchers do not move.
            dual_clip=None,
            # dsnn-dfw.98: same reason, keeps --target-kl off too.
            target_kl=None,
            # dsnn-dfw.99: same reason, keeps the retention bound off too.
            cache_clear_every=None,
            # dsnn-dfw.99: same reason, keeps the process recycle off too.
            proactive_recycle_every=None,
            # Owner ruling 2026-09-23 moves the matrix rows; this round is
            # frozen and keeps its timeout, cores and compile cache.
            ray_measure_timeout=CAMPAIGN_RAY_MEASURE_TIMEOUT,
            cores_per_actor=None,
            jax_cache=True,
            # dsnn-dfw.264 moves the matrix rows; this round keeps its env.
            shd_dir=False,
            # The rulings of 2026-09-25 (dsnn-dfw.169, dsnn-mep, dsnn-dfw.208)
            # move the matrix rows; this round keeps its measure path, its
            # memory channel, its oracle and its memory request.
            measure_path=False,
            mem_objective_weight=None,
            whole_node=False,
            # dsnn-dfw.245 moves the matrix rows; this round keeps its devices.
            measure_gpus=False,
            # dsnn-dfw.191: same reason, keeps the unbatched target too.
            batched=False,
        )
        ARMS[-1]["sweepl2"] = True
        ARMS[-1]["sweepl2_tag"] = _sweepl2_tag
        if _sweepl2_half is not None:
            _sweepl2_row = ARMS[-1]
            _sweepl2_row["half"] = _sweepl2_half
            _SWEEPL2_HALVES.setdefault(
                (_sweepl2_tag, _sweepl2_node), []).append(_sweepl2_row)
_sweepl2_row: dict = {}
_sweepl2_rows: list[dict] = []
for _sweepl2_rows in _SWEEPL2_HALVES.values():
    if len(_sweepl2_rows) == 2:
        sweepl2_pair_arm(_sweepl2_rows)
del (_sweepl2_tag, _sweepl2_overrides, _sweepl2_seed, _sweepl2_node,
     _sweepl2_half, _sweepl2_row, _sweepl2_rows, _sweepl2_slot,
     _SWEEPL2_HALVES)


# ================  ACTOR UPDATE-BUDGET LADDER, ROUND 3  ================
# Owner ruling 2026-09-20.  Round 2 asked whether the actor responds to the
# dual at all; this round holds the dual at its own CENTER constants
# (--lag-eta 2.0, --lag-max 64, --lag-init 16, --quality-floor 0.90) fixed
# and instead climbs the PPO update budget itself: four rungs, three seeds,
# arm C on NN256 at the normalized init, entropy floor 0.05, 100 episodes,
# no --auto-stop.  NOT a matrix coordinate: `thesis_core_arms`,
# `thesis_block1_arms` and `thesis_pair_arms` exclude every row this
# section marks `sweepl3=True`, the same way they exclude round 1's
# `sweepl=True` and round 2's `sweepl2=True` rows.
SWEEPL3_TARGET = "nn256"
SWEEPL3_ARM = "C"
SWEEPL3_SEEDS = THESIS_SEEDS[:3]
SWEEPL3_EPISODES = "100"
#: budget tag -> the cli overrides that differ from today's update budget
#: (--ppo-epochs 1, --minibatches 4, the SHARED_CLI defaults).  The dual
#: itself is never overridden here: every row carries arm C's own CENTER
#: constants (--lag-eta 2.0, --lag-max 64, --lag-init 16,
#: --quality-floor 0.90) untouched.
SWEEPL3_BUDGET: dict[str, dict[str, str]] = {
    "b4": {"--ppo-epochs": "2", "--minibatches": "8"},
    "b16": {"--ppo-epochs": "2", "--minibatches": "32"},
    "b32": {"--ppo-epochs": "4", "--minibatches": "32"},
    "b64": {"--ppo-epochs": "4", "--minibatches": "64"},
}

_SWEEPL3_WHAT = """SWEEP ROUND 3 (owner ruling 2026-09-20): the actor's PPO
update budget ladder -- b4 (--ppo-epochs 2, --minibatches 8), b16
(--ppo-epochs 2, --minibatches 32), b32 (--ppo-epochs 4, --minibatches 32)
and b64 (--ppo-epochs 4, --minibatches 64) -- at the rung-1 CENTER dual
(--lag-eta 2.0, --lag-max 64, --lag-init 16, --quality-floor 0.90) held
fixed, arm C on NN256 at the normalized init, entropy floor 0.05, 100
episodes, no --auto-stop.

PURPOSE: how far the actor's update budget can go before PPO destabilizes.

READ-OUT: the feasible fraction at q >= 0.9 over episodes 80-99 from the
plan log; kl/approx and ratio/max_log from the health line as the
destabilization guard; the plan profile (requested rules per plan) and the
latency reward of feasible plans."""


def sweepl3_configs() -> list[tuple[str, dict[str, str]]]:
    """(tag, cli overrides) for the 4 budget rungs."""
    return [(tag, dict(cli)) for tag, cli in SWEEPL3_BUDGET.items()]


def sweepl3_row_name(tag: str, seed: str) -> str:
    return f"sweepL3_nn256_{tag}_s{seed}"


def sweepl3_pair_name(tag: str, seeds: tuple[str, str]) -> str:
    return f"sweepL3_nn256_{tag}_s{seeds[0]}_s{seeds[1]}_pair"


def sweepl3_pair_arm(rows: list[dict]) -> dict:
    """Two half rows of ONE sweep-round-3 budget, one 8-GPU node -> one
    paired `arm(...)`.  A copy of `sweepl2_pair_arm`'s body under round 3's
    own name: half rows point back at it through `sweepl3_paired_into`,
    never round 1's or round 2's pairing records, so the three pairing
    records cannot cross.
    """
    _require(len(rows) == 2, f"a sweep pair is two rows, not {len(rows)}")
    a, b = sorted(rows, key=lambda r: r["half"])
    node = a["node"]
    tag = a["sweepl3_tag"]
    _require(b["node"] == node and b["sweepl3_tag"] == tag,
             "a sweep pair is two seeds of ONE configuration on ONE node")
    _require(a["thesis_seed"] != b["thesis_seed"],
             "a sweep pair is two DIFFERENT seeds")
    gpus = THESIS_NODE_GPUS[node]
    per = THESIS_UNIFORM_GPUS[SWEEPL3_TARGET]
    cpus = node_cpus(node, gpus)
    per_cpus = cpus // (gpus // per)
    halves = []
    for r in (a, b):
        h = r["half"]
        halves.append(dict(
            name=r["name"], seed=r["thesis_seed"], half=h,
            devices=",".join(str(h * per + d) for d in range(per)),
            cores=f"{h * per_cpus}-{(h + 1) * per_cpus - 1}",
            cli=r["cli"],
        ))
    name = sweepl3_pair_name(tag, (a["thesis_seed"], b["thesis_seed"]))
    for r in (a, b):
        r["sweepl3_paired_into"] = name
    p = dict(
        name=name, job=thesis_job_name(node), kind="train", runtime="scratch",
        node=node, time=a["time"], gpus=gpus, singleton=True, thesis=True,
        sweepl3=True, paired=True, halves=halves,
        thesis_arm=SWEEPL3_ARM, thesis_target=SWEEPL3_TARGET,
        env=dict(a["env"]),
        required_flags=a["required_flags"],
        required_flags_file=a["required_flags_file"],
        cli=a["cli"],
        purpose=a["purpose"] + f"\n\nPAIRED ON {node}: seeds "
                f"{a['thesis_seed']} and {b['thesis_seed']} of sweep round "
                f"3 budget {tag} run CONCURRENTLY, each on {per} of "
                f"the node's {gpus} GPUs and {per_cpus} of its {cpus} "
                "cores.",
        prediction=a["prediction"], falsifier=a["falsifier"],
    )
    arm_(**p)
    return p


def sweepl3_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl3")]


def sweepl3_single_arms() -> list[dict]:
    return [a for a in ARMS if a.get("sweepl3") and not a.get("paired")]


_SWEEPL3_HALVES: dict[tuple[str, str], list[dict]] = {}
_sweepl3_slot = 0
for _sweepl3_tag, _sweepl3_overrides in sweepl3_configs():
    for _sweepl3_seed in SWEEPL3_SEEDS:
        _sweepl3_node, _sweepl3_half = THESIS_SLOTS[
            _sweepl3_slot % len(THESIS_SLOTS)]
        _sweepl3_slot += 1
        thesis_arm(
            arm=SWEEPL3_ARM, target=SWEEPL3_TARGET, seed=_sweepl3_seed,
            node=_sweepl3_node,
            name=sweepl3_row_name(_sweepl3_tag, _sweepl3_seed),
            episodes=SWEEPL3_EPISODES, what=_SWEEPL3_WHAT,
            extra_cli=dict(_sweepl3_overrides),
            # dsnn-dfw.95: this round is FROZEN (see the budget note above),
            # so it keeps --dual-clip off and its launchers do not move.
            dual_clip=None,
            # dsnn-dfw.98: same reason, keeps --target-kl off too.
            target_kl=None,
            # dsnn-dfw.99: same reason, keeps the retention bound off too.
            cache_clear_every=None,
            # dsnn-dfw.99: same reason, keeps the process recycle off too.
            proactive_recycle_every=None,
            # Owner ruling 2026-09-23 moves the matrix rows; this round is
            # frozen and keeps its timeout, cores and compile cache.
            ray_measure_timeout=CAMPAIGN_RAY_MEASURE_TIMEOUT,
            cores_per_actor=None,
            jax_cache=True,
            # dsnn-dfw.264 moves the matrix rows; this round keeps its env.
            shd_dir=False,
            # The rulings of 2026-09-25 (dsnn-dfw.169, dsnn-mep, dsnn-dfw.208)
            # move the matrix rows; this round keeps its measure path, its
            # memory channel, its oracle and its memory request.
            measure_path=False,
            mem_objective_weight=None,
            whole_node=False,
            # dsnn-dfw.245 moves the matrix rows; this round keeps its devices.
            measure_gpus=False,
            # dsnn-dfw.191: same reason, keeps the unbatched target too.
            batched=False,
        )
        ARMS[-1]["sweepl3"] = True
        ARMS[-1]["sweepl3_tag"] = _sweepl3_tag
        if _sweepl3_half is not None:
            _sweepl3_row = ARMS[-1]
            _sweepl3_row["half"] = _sweepl3_half
            _SWEEPL3_HALVES.setdefault(
                (_sweepl3_tag, _sweepl3_node), []).append(_sweepl3_row)
_sweepl3_row: dict = {}
_sweepl3_rows: list[dict] = []
for _sweepl3_rows in _SWEEPL3_HALVES.values():
    if len(_sweepl3_rows) == 2:
        sweepl3_pair_arm(_sweepl3_rows)
del (_sweepl3_tag, _sweepl3_overrides, _sweepl3_seed, _sweepl3_node,
     _sweepl3_half, _sweepl3_row, _sweepl3_rows, _sweepl3_slot,
     _SWEEPL3_HALVES)

# ================  ORDER-ONLY SCALARIZATION TUNING, ROUND 1  ================
# Ticket `dsnn-dfw.29`, owner rulings 2026-09-18.  ROUND 1 ONLY: the weights.
# Round 2 (the PPO knobs at the winning weight) needs the owner's word after
# round 1 reports and is NOT generated here.
#
# THE ARM is the epic's REQUIRED order-only arm on NN256: --approx-profile
# none with --fixed-order free, so the policy chooses the ELIMINATION ORDER
# and nothing else.  With no approximation applied the grad cosine is 1 on
# every plan, the Lagrangian constraint at tau 0.90 is never violated and
# lambda decays to its floor, so the C form reduces to a pure SCALARIZATION
#
#     lambda_cmp * paired-log latency + lambda_mem * paired-log watermark memory
#
# and round 1 sweeps its two weights.  That is the MORL-to-SORL step: five
# fixed weights, three seeds each, against one preference-conditioned run
# that amortises the whole front.
#
# THE WEIGHTS sum to 2 on every row, so the five rows differ in the DIRECTION
# of the scalarization and not in its scale -- a run at (1, 1) and a run at
# (2, 2) would be the same objective at twice the advantage.
#
# WHY THE SEED IS THE NODE (AGENTS.md: "latency and memory numbers are not
# comparable across GPU models; compare only within one node and one job").
# Three GPU models carry this round, so the node is assigned BY SEED: all six
# configurations of one seed run on one node, one after another through the
# per-node singleton.  The weight comparison -- the thing the round decides --
# is therefore WITHIN a node and within a GPU model in every seed column, and
# the seed spread carries the cross-model variation instead of hiding it
# inside a weight.  The reward channels are paired log ratios against a
# rev-exact reference measured back to back in the same actor, which is what
# makes the columns comparable at all; the ratios are still read per column
# first and pooled only after.
# ---------------------------------------------------------------------------
ORDERONLY_TARGET = "nn256"
#: --approx-profile none IS the arm: no approximation class is available.
ORDERONLY_PROFILE = "none"
#: The C form, unchanged from the matrix (THESIS_ARM_SPEC): the Lagrangian
#: dual at tau 0.90.  Inert here because quality is constant 1, and kept so
#: this round and the matrix's C rows differ in the swept flags alone.
ORDERONLY_ARM = "C"
ORDERONLY_PREF_ARM = "condC"
#: Three seeds (owner: 3 seeds for the tuning, 5 for the baseline that
#: follows it).  The first three of the matrix's five, never a new one.
ORDERONLY_SEEDS = THESIS_SEEDS[:3]
#: (--lambda-cmp, --lambda-mem), the five pairs of the ruling, in order.
ORDERONLY_WEIGHTS = (("2", "0"), ("1.5", "0.5"), ("1", "1"),
                     ("0.5", "1.5"), ("0", "2"))
#: ONE NODE PER SEED, in seed order.  A node may carry more than one seed; a
#: seed may never be split, because the weight comparison inside a seed is the
#: thing this round decides and it must not straddle two GPU models.
#:
#: Every node here was cleared by running the measure toolchain gate ON the
#: node, and the whole command line under it for one episode, before it was
#: listed: gpu13 finds the matched pair at /usr/local/cuda-12, gpu14 needs
#: NODE_CUDA_BIN (jobs 66341 and 66342, both "toolchain gate OK" and TRAINER
#: exited 0).  OUT: pgi15-gpu9, -gpu11 and -gpu12 still carry no matched CUDA
#: pair (finding 03).  pgi15-gpu8 is cleared as of 2026-09-19 (job 66542;
#: see NODE_CUDA_WANT) and carries the dsnn-dfw.45 recurrent-target rows.
#:
#: pgi15-gpu16 held seed 250197 until 2026-09-18 19:10, when another group
#: took gpu15, gpu16, gpu17 and gpu18 on a three-day reservation.  No
#: Blackwell node is reachable, so that seed moved to gpu14, which then
#: carries two whole seeds.  The BASELINE that follows this round still goes
#: on Blackwell (owner ruling); this is the tuning round.
ORDERONLY_NODES = ("pgi15-gpu14", "pgi15-gpu13", "pgi15-gpu14")
#: --lambda-cmp and --lambda-mem are the swept flags, so the pre-flight greps
#: for them by name rather than trusting that ppo.py still defines them.
ORDERONLY_REQUIRED_FLAGS = THESIS_REQUIRED_FLAGS + ["--lambda-cmp",
                                                    "--lambda-mem"]

_ORDERONLY_HEAD = f"""ORDER-ONLY SCALARIZATION TUNING, ROUND 1 (ticket
dsnn-dfw.29) under the owner's rulings of 2026-09-18.  The epic's REQUIRED
order-only arm on NN256: --approx-profile {ORDERONLY_PROFILE} with
--fixed-order {THESIS_ORDER}, so the policy chooses THE ELIMINATION ORDER and
nothing else.  No approximation is applied, the grad cosine is 1 on every
plan and the C form's constraint at tau {THESIS_TAU} is never violated, so the
objective is the pure scalarization

    --lambda-cmp * paired-log latency + --lambda-mem * paired-log watermark memory

whose two weights this round sweeps.  Everything else is the matrix row it
comes from: --episodes {THESIS_EPISODES} with --auto-stop, --checkpoint-every
{THESIS_CHECKPOINT_EVERY}, --pareto-dump-every {THESIS_PARETO_DUMP_EVERY},
--plan-log {THESIS_PLAN_LOG}, no XLA environment, wandb online to the project
the matrix already writes to.

THE NODE IS THE SEED.  Three GPU models carry this round and latency and
memory are not comparable across models, so all six configurations of one
seed run on ONE node through the per-node singleton queue.  The weight
comparison is within a node; the seed spread carries the model variation.

ROUND 2 (the PPO knobs at the winning weight) is NOT in this file.  It needs
the owner's word after this round reports."""

_ORDERONLY_PREDICTION = """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER
(ticket dsnn-dfw.29): the five weights trace a front -- (2, 0) reaches the
lowest paired latency ratio of the five and (0, 2) the lowest paired memory
ratio, with the three mixed weights between them and no weight dominating
another on both channels.  The preference-conditioned run's front spans at
least the latency range the five fixed weights span between them."""

_ORDERONLY_FALSIFIER = """If every weight lands on the same terminal plan --
the same paired ratios within the drift floor at all five -- the scalarization
weight is NOT what decides this arm and round 2 is pointless at any weight;
the result is reported as that, and no weight is declared the winner.  If the
preference-conditioned run spans less than the fixed weights do, the
conditioning is reported as not amortising this front."""


def orderonly_run_name(lam_cmp: str, lam_mem: str, seed: str) -> str:
    """`orderonly_nn256_l<X>m<Y>_s<seed>` (owner ruling 2026-09-18)."""
    if (lam_cmp, lam_mem) not in ORDERONLY_WEIGHTS:
        raise CampaignRowError(
            f"({lam_cmp!r}, {lam_mem!r}) is not one of the five ruled weight "
            f"pairs {ORDERONLY_WEIGHTS}")
    if seed not in ORDERONLY_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_SEEDS}")
    return f"orderonly_nn256_l{lam_cmp}m{lam_mem}_s{seed}"


def orderonly_pref_run_name(seed: str) -> str:
    """`orderonly_nn256_pref_s<seed>` (owner ruling 2026-09-18)."""
    if seed not in ORDERONLY_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_SEEDS}")
    return f"orderonly_nn256_pref_s{seed}"


def orderonly_job_name(node: str) -> str:
    """THE CROSS-AGENT SINGLETON NAME (orchestrator ruling 2026-09-18).

    Slurm's `--dependency=singleton` serializes jobs of one user that share a
    NAME.  Three agents now submit to pgi15-gpu14, so a name that only one of
    them uses serializes nothing: two jobs of this user would land on the node
    together and the node epilog would kill both (memory note
    pgi15-epilog-kills-sibling-jobs).  `node-<node>` is the name EVERY agent
    uses, so one job of ours runs on a node at a time whoever submitted it.
    STE: `thesis_job_name` returns this same string since ticket dsnn-dfw.65;
    the matrix's old `thesis-<node>` served the matrix against itself alone.
    """
    if node not in THESIS_NODE_GPUS and node not in NODE_GPUS:
        raise CampaignRowError(
            f"node {node!r} is not a GPU node this generator knows "
            f"({sorted(set(THESIS_NODE_GPUS) | set(NODE_GPUS))})")
    return f"node-{node}"


def orderonly_node(seed: str) -> str:
    """THE NODE OF A SEED.  One node per seed, so every weight of one seed is
    measured on one GPU model and the weight comparison never straddles two."""
    if seed not in ORDERONLY_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_SEEDS}")
    return ORDERONLY_NODES[ORDERONLY_SEEDS.index(seed)]


def orderonly_arm(*, seed: str, lam_cmp: str | None = None,
                  lam_mem: str | None = None, pref: bool = False) -> dict:
    """One round-1 run -> one `arm(...)`.  Returns the arm.

    `pref=True` is the preference-conditioned row: the Dirichlet preference
    over (latency, memory) replaces the fixed pair, so it carries the matrix's
    own --lambda-cmp 1 --lambda-mem 1 and sweeps nothing.
    """
    node = orderonly_node(seed)
    if pref:
        _require(lam_cmp is None and lam_mem is None,
                 "the preference-conditioned row sweeps no weight: the "
                 "Dirichlet preference over (latency, memory) IS the weight, "
                 "and a fixed pair beside it would say two different things")
        name = orderonly_pref_run_name(seed)
        arm_name = ORDERONLY_PREF_ARM
    else:
        _require(lam_cmp is not None and lam_mem is not None,
                 "a fixed-weight row needs both --lambda-cmp and --lambda-mem")
        name = orderonly_run_name(lam_cmp, lam_mem, seed)
        arm_name = ORDERONLY_ARM
    cli = thesis_cli(arm=arm_name, target=ORDERONLY_TARGET, seed=seed,
                     node=node, name=name, episodes=THESIS_EPISODES,
                     checkpoint_every=THESIS_CHECKPOINT_EVERY,
                     auto_stop=True)
    cli["--approx-profile"] = ORDERONLY_PROFILE
    if not pref:
        cli["--lambda-cmp"] = lam_cmp
        cli["--lambda-mem"] = lam_mem
    gpus = thesis_row_gpus(ORDERONLY_TARGET, node)
    what = (f"THE PREFERENCE-CONDITIONED ROW at seed {seed}: one run over a "
            f"Dirichlet preference on (latency, memory), against the five "
            f"fixed weights of the same seed on the same node."
            if pref else
            f"WEIGHT (--lambda-cmp {lam_cmp}, --lambda-mem {lam_mem}) at seed "
            f"{seed}, one of the five ruled pairs.")
    a = dict(
        name=name, job=orderonly_job_name(node), kind="train",
        runtime="scratch",
        node=node, time=THESIS_TIME, gpus=gpus, singleton=True, thesis=True,
        orderonly=True, thesis_arm=arm_name, thesis_target=ORDERONLY_TARGET,
        thesis_seed=seed, orderonly_weights=(None if pref
                                             else (lam_cmp, lam_mem)),
        env=thesis_target_form(ORDERONLY_TARGET, batched=False)[1],
        required_flags=ORDERONLY_REQUIRED_FLAGS,
        required_flags_file=" ".join(THESIS_FLAGS_FILES),
        cli=cli,
        purpose=_ORDERONLY_HEAD + "\n\n" + what,
        prediction=_ORDERONLY_PREDICTION,
        falsifier=_ORDERONLY_FALSIFIER,
    )
    if node in NODE_CUDA_BIN:
        a["cuda_bin"] = NODE_CUDA_BIN[node]
    arm_(**a)
    return a


def orderonly_submission_order() -> list[tuple[str, str | None, str | None]]:
    """(seed, lambda_cmp, lambda_mem) in submission order; the pair is None
    on the preference-conditioned row.

    Seed-major, so the six runs of one node queue together behind that node's
    singleton and the three nodes fill at once.  The weights run in the ruled
    order inside a seed and the conditioned row runs last of its node, after
    the five it is compared against.
    """
    order: list[tuple[str, str | None, str | None]] = []
    for s in ORDERONLY_SEEDS:
        for lc, lm in ORDERONLY_WEIGHTS:
            order.append((s, lc, lm))
        order.append((s, None, None))
    return order


#: The 18 runs of round 1: five weights x three seeds, plus one
#: preference-conditioned run per seed.
ORDERONLY_RUNS = len(ORDERONLY_SEEDS) * (len(ORDERONLY_WEIGHTS) + 1)

for _seed, _lc, _lm in orderonly_submission_order():
    orderonly_arm(seed=_seed, lam_cmp=_lc, lam_mem=_lm, pref=_lc is None)
del _seed, _lc, _lm


def orderonly_arms() -> list[dict]:
    return [a for a in ARMS if a.get("orderonly")]


# ---------------------------------------------------------------------------
# THE 5-SEED ORDER-ONLY BASELINE ON NN256 (owner ruling 2026-09-19, the epic's
# block plan).  THE FINAL row of the order-only arm; the round above is its
# tuning.  The same arm -- --approx-profile none with --fixed-order free, the
# C form, the weight pair (2, 0) -- at FIVE seeds instead of three, on the
# Blackwell nodes, and WITHOUT --auto-stop, because a final row runs its full
# thousand episodes (THESIS_FINAL_AUTO_STOP).
#
# THE WEIGHT IS (2, 0), THE LATENCY-ONLY SCALARIZATION.  Round 1 has not
# reported, and the baseline the epic asks for is the latency arm: (2, 0)
# weights the paired-log latency alone, which is the quantity this arm exists
# to move.  When round 1 names a different pair the owner moves
# ORDERONLY_FINAL_WEIGHTS and regenerates.
#
# THE NODE IS THE SEED, round-robin over the five cleared Blackwell nodes in
# seed order.  A seed's numbers may never straddle two GPU models (AGENTS.md).
# pgi15-gpu17 stays out (dsnn-dfw.69: job 66740 aborted 72 -- /usr/local/
# cuda-12.9 on gpu17 has no bin/ptxas or nvlink); pgi15-gpu19 came back to us
# on 2026-09-20, so five seeds now sit on five nodes and no node repeats.
# EVERY seed renders the SAME profile -- 4 GPUs, 64 CPUs, 400G and
# --ray-measure 3 -- on all five nodes (THESIS_UNIFORM_GPUS), so the seed
# spread carries the node variation and nothing else; the 8-GPU nodes hold
# the row on four of their GPUs.
# ---------------------------------------------------------------------------
ORDERONLY_FINAL_SEEDS = THESIS_SEEDS
#: (--lambda-cmp, --lambda-mem) of the baseline: one of the five ruled pairs.
ORDERONLY_FINAL_WEIGHTS = ("2", "0")
ORDERONLY_FINAL_NODES = THESIS_NODES_ALL

_ORDERONLY_FINAL_HEAD = f"""THE 5-SEED ORDER-ONLY BASELINE ON NN256 (epic
dsnn-dfw, owner ruling 2026-09-19).  A FINAL row, not a tuning row: the
order-only arm of ticket dsnn-dfw.29 -- --approx-profile {ORDERONLY_PROFILE}
with --fixed-order {THESIS_ORDER}, the C form, so the policy chooses THE
ELIMINATION ORDER and nothing else -- at the weight pair
(--lambda-cmp {ORDERONLY_FINAL_WEIGHTS[0]},
--lambda-mem {ORDERONLY_FINAL_WEIGHTS[1]}), the latency-only scalarization,
across all five matrix seeds.

--episodes {THESIS_EPISODES} WITHOUT --auto-stop, --checkpoint-every
{THESIS_CHECKPOINT_EVERY}, --pareto-dump-every {THESIS_PARETO_DUMP_EVERY},
--plan-log {THESIS_PLAN_LOG}, the four block settings of 2026-09-19
(--paired-cost-floor {THESIS_PAIRED_COST_FLOOR}, --mem-channel
{THESIS_MEM_CHANNEL}, --lag-max {THESIS_DUAL_LAMBDA_MAX}), wandb online.

ONE SEED PER BLACKWELL NODE, in seed order, with the cross-agent per-node
singleton name: latency and memory are not comparable across GPU models, so
a seed is measured on one node and the seed spread carries the model
variation.  The three-seed round on the mixed nodes is the TUNING this row is
the baseline for; it is not this row, and the two are not pooled."""

_ORDERONLY_FINAL_PREDICTION = """REGISTERED BEFORE THE RUN, NEVER EDITED
AFTER: the five seeds agree -- the median terminal paired latency ratio of
the five is below 1.0 and the seed-to-seed spread of that median is smaller
than the gap between the (2, 0) and (0, 2) weights of the tuning round.  The
memory channel, weighted 0 here, is reported and not optimised."""

_ORDERONLY_FINAL_FALSIFIER = """If the five seeds disagree -- any two seeds'
terminal latency ratios not overlapping within their own per-seed spread --
the order-only arm is reported as seed-dependent at this weight, and no
single baseline number is quoted for it."""


def orderonly_final_run_name(seed: str) -> str:
    """`orderonly_nn256_final_l<X>m<Y>_s<seed>` (owner ruling 2026-09-19)."""
    if seed not in ORDERONLY_FINAL_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_FINAL_SEEDS}")
    lam_cmp, lam_mem = ORDERONLY_FINAL_WEIGHTS
    return f"orderonly_nn256_final_l{lam_cmp}m{lam_mem}_s{seed}"


def orderonly_final_node(seed: str) -> str:
    """THE NODE OF A SEED: round-robin over the five cleared Blackwell
    nodes, in seed order (dsnn-dfw.69: pgi15-gpu17 excluded)."""
    if seed not in ORDERONLY_FINAL_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_FINAL_SEEDS}")
    i = ORDERONLY_FINAL_SEEDS.index(seed)
    return ORDERONLY_FINAL_NODES[i % len(ORDERONLY_FINAL_NODES)]


def orderonly_final_arm(*, seed: str) -> dict:
    """One 5-seed order-only baseline row -> one `arm(...)`.  Returns it."""
    node = orderonly_final_node(seed)
    _require(node in THESIS_NODES_ALL,
             f"node {node!r} is not one of the cleared Blackwell nodes "
             f"{THESIS_NODES_ALL}; the baseline is a FINAL row and final rows "
             f"run on Blackwell only (AGENTS.md)")
    lam_cmp, lam_mem = ORDERONLY_FINAL_WEIGHTS
    name = orderonly_final_run_name(seed)
    cli = thesis_cli(arm=ORDERONLY_ARM, target=ORDERONLY_TARGET, seed=seed,
                     node=node, name=name, episodes=THESIS_EPISODES,
                     checkpoint_every=THESIS_CHECKPOINT_EVERY,
                     auto_stop=THESIS_FINAL_AUTO_STOP)
    cli["--approx-profile"] = ORDERONLY_PROFILE
    cli["--lambda-cmp"] = lam_cmp
    cli["--lambda-mem"] = lam_mem
    gpus = thesis_row_gpus(ORDERONLY_TARGET, node)
    a = dict(
        name=name, job=orderonly_job_name(node), kind="train",
        runtime="scratch",
        node=node, time=THESIS_TIME, gpus=gpus, singleton=True, thesis=True,
        orderonly_final=True, thesis_arm=ORDERONLY_ARM,
        thesis_target=ORDERONLY_TARGET, thesis_seed=seed,
        orderonly_weights=(lam_cmp, lam_mem),
        env=thesis_target_form(ORDERONLY_TARGET, batched=False)[1],
        required_flags=ORDERONLY_REQUIRED_FLAGS,
        required_flags_file=" ".join(THESIS_FLAGS_FILES),
        cli=cli,
        purpose=_ORDERONLY_FINAL_HEAD + f"\n\nSEED {seed} ON {node}.",
        prediction=_ORDERONLY_FINAL_PREDICTION,
        falsifier=_ORDERONLY_FINAL_FALSIFIER,
    )
    if node in NODE_CUDA_BIN:
        a["cuda_bin"] = NODE_CUDA_BIN[node]
    arm_(**a)
    return a


#: The five runs of the baseline: one weight pair, five seeds, five nodes.
ORDERONLY_FINAL_RUNS = len(ORDERONLY_FINAL_SEEDS)

for _seed in ORDERONLY_FINAL_SEEDS:
    orderonly_final_arm(seed=_seed)
del _seed


def orderonly_final_arms() -> list[dict]:
    return [a for a in ARMS if a.get("orderonly_final")]


# ---------------------------------------------------------------------------
# ONE ORDER-ONLY FINAL ROW ON TLM (owner: "run a TLM run for 1k episodes",
# 2026-09-19 evening).  The order-only arm's shape -- --approx-profile
# {ORDERONLY_PROFILE} with --fixed-order {THESIS_ORDER}, the C form -- on
# --example TransformerLM --dataset wikitext2 (the fq_C_tlm row's target
# flags; the ALPHAGRAD_TLM_* triple is SHARED_ENV, not per-target, so it
# exports here exactly as it does on every scratch-kind row), at
# --lambda-cmp 1 --lambda-mem 1: memory is a live objective on TLM (the
# orchestrator's assumption; the owner may move the weight later), unlike
# the NN256 baseline's latency-only (2, 0).
#
# A FINAL ROW, like the NN256 baseline above: THESIS_FINAL_AUTO_STOP (no
# --auto-stop, the full thousand episodes), the four block settings
# (--paired-cost-floor {THESIS_PAIRED_COST_FLOOR}, --mem-channel
# {THESIS_MEM_CHANNEL}, --lag-max {THESIS_DUAL_LAMBDA_MAX}), wandb online
# (WANDB_MODE, unchanged for a thesis-kind row), on pgi15-gpu15 -- a
# Blackwell node, final rows run on Blackwell only (AGENTS.md).
#
# ONE ROW, ONE NAME, no seed sweep and no node table: the owner named the
# exact row (orderonly_tlm_final_l1m1_s250197), not a family.
# ---------------------------------------------------------------------------
ORDERONLY_TLM_FINAL_NAME = "orderonly_tlm_final_l1m1_s250197"
ORDERONLY_TLM_FINAL_SEED = "250197"
ORDERONLY_TLM_FINAL_WEIGHTS = ("1", "1")
ORDERONLY_TLM_FINAL_NODE = "pgi15-gpu15"

_ORDERONLY_TLM_FINAL_HEAD = f"""ONE ORDER-ONLY FINAL ROW ON TLM (epic
dsnn-dfw, owner: "run a TLM run for 1k episodes", 2026-09-19 evening).  The
order-only arm -- --approx-profile {ORDERONLY_PROFILE} with --fixed-order
{THESIS_ORDER}, the C form, so the policy chooses THE ELIMINATION ORDER and
nothing else -- on --example TransformerLM --dataset wikitext2, at the
weight pair (--lambda-cmp {ORDERONLY_TLM_FINAL_WEIGHTS[0]}, --lambda-mem
{ORDERONLY_TLM_FINAL_WEIGHTS[1]}): memory is a live objective on this
target, unlike the NN256 baseline's latency-only (2, 0) -- the orchestrator's
assumption; the owner may move the weight later.

--episodes {THESIS_EPISODES} WITHOUT --auto-stop (a final row runs the full
thousand), --checkpoint-every {THESIS_CHECKPOINT_EVERY}, --pareto-dump-every
{THESIS_PARETO_DUMP_EVERY}, --plan-log {THESIS_PLAN_LOG}, the four block
settings of 2026-09-19 (--paired-cost-floor {THESIS_PAIRED_COST_FLOOR},
--mem-channel {THESIS_MEM_CHANNEL}, --lag-max {THESIS_DUAL_LAMBDA_MAX}),
wandb online, on pgi15-gpu15 (Blackwell, a final node)."""

_ORDERONLY_TLM_FINAL_PREDICTION = """REGISTERED BEFORE THE RUN, NEVER EDITED
AFTER: at roughly 64 s/episode this row completes about 1000 episodes in 18
hours; the terminal plan's paired latency and memory ratios both improve
over the C arm's own matrix row (arm C on tlm, --approx-profile all), since
order alone with no approximation should never cost more than order-plus-
approximation search does at convergence."""

_ORDERONLY_TLM_FINAL_FALSIFIER = """If the run does not reach 1000 episodes
in the allotted time, or the terminal plan's ratios are not better than the
matrix C-arm row's, that is reported as it stands; no number here is
adjusted after the fact."""


def orderonly_tlm_final_arm() -> dict:
    """The one TLM order-only final row -> one `arm(...)`.  Returns it."""
    node = ORDERONLY_TLM_FINAL_NODE
    _require(node in THESIS_NODES_ALL,
             f"node {node!r} is not one of the cleared Blackwell nodes "
             f"{THESIS_NODES_ALL}; this is a FINAL row and final rows run "
             f"on Blackwell only (AGENTS.md)")
    lam_cmp, lam_mem = ORDERONLY_TLM_FINAL_WEIGHTS
    name = ORDERONLY_TLM_FINAL_NAME
    # dsnn-dfw.84 (2026-09-21) is a thesis-matrix finding; this row is
    # order-only, not a matrix coordinate, so it keeps its own recorded 0.05
    # rather than picking up the near-zero bonus.
    cli = thesis_cli(arm=ORDERONLY_ARM, target="tlm",
                     seed=ORDERONLY_TLM_FINAL_SEED, node=node, name=name,
                     episodes=THESIS_EPISODES,
                     checkpoint_every=THESIS_CHECKPOINT_EVERY,
                     auto_stop=THESIS_FINAL_AUTO_STOP,
                     face_entropy_weight="0.05")
    cli["--approx-profile"] = ORDERONLY_PROFILE
    cli["--lambda-cmp"] = lam_cmp
    cli["--lambda-mem"] = lam_mem
    gpus = thesis_row_gpus("tlm", node)
    a = dict(
        name=name, job=orderonly_job_name(node), kind="train",
        runtime="scratch",
        node=node, time=THESIS_TIME, gpus=gpus, singleton=True, thesis=True,
        orderonly_tlm_final=True, thesis_arm=ORDERONLY_ARM,
        thesis_target="tlm", thesis_seed=ORDERONLY_TLM_FINAL_SEED,
        orderonly_weights=(lam_cmp, lam_mem),
        env=thesis_target_form("tlm", batched=False)[1],
        required_flags=ORDERONLY_REQUIRED_FLAGS,
        required_flags_file=" ".join(THESIS_FLAGS_FILES),
        cli=cli,
        purpose=_ORDERONLY_TLM_FINAL_HEAD,
        prediction=_ORDERONLY_TLM_FINAL_PREDICTION,
        falsifier=_ORDERONLY_TLM_FINAL_FALSIFIER,
    )
    if node in NODE_CUDA_BIN:
        a["cuda_bin"] = NODE_CUDA_BIN[node]
    arm_(**a)
    return a


orderonly_tlm_final_arm()


def orderonly_tlm_final_arms() -> list[dict]:
    return [a for a in ARMS if a.get("orderonly_tlm_final")]


# ---------------------------------------------------------------------------
# ORDER-ONLY SCALARIZATION TUNING ON THE RECURRENT TARGET (ticket dsnn-dfw.45,
# owner rulings 2026-09-19).  The section above, on rsnn_bptt and rsnn_rtrl
# instead of nn256: the same --approx-profile none, the same --fixed-order
# free, the same C form, the same five weight pairs, the same three seeds and
# the same one preference-conditioned row per seed.  The recurrent target
# adds --temporal-rule (bptt or rtrl); nothing else about the arm moves.
#
# THE NODE IS THE RULE, not the seed: latency and memory here compare a
# scalarization weight WITHIN one rule, so all three seeds of one rule share
# one GPU model and the rule itself is the thing that must not straddle two.
# bptt runs on pgi15-gpu14 (H100, the same node round 1 cleared).  rtrl runs
# on pgi15-gpu8 (RTX 6000 Ada), cleared 2026-09-19: a matched 12.8 pair at
# /usr/local/cuda-12, proven by job 66542 (NODE_CUDA_WANT, NODE_CUDA_BIN).
#
# The three rsnn_tbptt reference rows left with the rule (dsnn-dfw.232).
# ---------------------------------------------------------------------------
ORDERONLY_RSNN_RULES = ("bptt", "rtrl")
#: ONE NODE PER RULE (not per seed: the round above puts the seed on one
#: node because it spans three GPU models per seed; here every seed of one
#: rule already shares a node, so the rule is the axis that must not split).
ORDERONLY_RSNN_NODES = {"bptt": "pgi15-gpu14", "rtrl": "pgi15-gpu8"}

_ORDERONLY_RSNN_HEAD = f"""ORDER-ONLY SCALARIZATION TUNING ON THE RECURRENT
TARGET (ticket dsnn-dfw.45) under the owner's rulings of 2026-09-19.  The
same order-only arm as the NN256 round (dsnn-dfw.29): --approx-profile
{ORDERONLY_PROFILE} with --fixed-order {THESIS_ORDER}, the C form, on
RSNN_SHD instead of NN256, with --temporal-rule naming the recurrent rule
this row measures.  No approximation is applied, the grad cosine is 1 on
every plan, and the objective is the pure scalarization

    --lambda-cmp * paired-log latency + --lambda-mem * paired-log watermark memory

--episodes {THESIS_EPISODES} with --auto-stop, --checkpoint-every
{THESIS_CHECKPOINT_EVERY}, --pareto-dump-every {THESIS_PARETO_DUMP_EVERY},
--plan-log {THESIS_PLAN_LOG}, no XLA environment.

THE NODE IS THE RULE.  bptt and rtrl each carry one GPU model for all three
seeds and all five weights of that rule, so the weight comparison is within
a node and the rule comparison (dsnn-dfw.45's actual question, pace and
front shape under bptt versus rtrl) is the one that crosses two GPU models --
which the ticket accepts, because the two rules were never going to share a
node (finding 03 leaves no cleared Blackwell node for a third GPU model)."""

_ORDERONLY_RSNN_PREDICTION = """REGISTERED BEFORE THE RUN, NEVER EDITED AFTER
(ticket dsnn-dfw.45): as in the NN256 round, the five weights trace a front on
each rule -- (2, 0) lowest paired latency ratio, (0, 2) lowest paired memory
ratio, the three mixed weights between them -- and the preference-conditioned
run's front spans at least the latency range the five fixed weights span.
rtrl's carried Jacobian is a matrix over the whole prefix and bptt's carried
adjoint is a vector over the suffix, so rtrl is predicted the slower pace and
the larger carried-state memory of the two."""

_ORDERONLY_RSNN_FALSIFIER = """If every weight lands on the same terminal plan
within a rule, the scalarization weight is not what decides that rule and the
result is reported as that, with no weight declared the winner.  If rtrl is
not slower or does not carry more memory than bptt, the carried-Jacobian
prediction above is wrong and is reported as such, not quietly dropped."""

_ORDERONLY_RSNN_WHAT = {
    "bptt": """TEMPORAL RULE bptt: THE FUTURE FEEDS IN, exactly as in the
thesis matrix's recurrent block -- an edge from the next state to the loss
carries the adjoint dL(>t)/ds_t from a detached backward pass over the
suffix.""",
    "rtrl": """TEMPORAL RULE rtrl: THE PAST FEEDS IN -- an edge from the
weights to the carried state carries the influence matrix ds(t-1)/dW from a
detached pass over the prefix, so eliminating that vertex is one real-time
recurrent-learning step.""",
}


def orderonly_rsnn_run_name(rule: str, lam_cmp: str, lam_mem: str,
                            seed: str) -> str:
    """`orderonly_rsnn_<rule>_l<X>m<Y>_s<seed>` (owner ruling 2026-09-19)."""
    if rule not in ORDERONLY_RSNN_RULES:
        raise CampaignRowError(
            f"rule {rule!r} is not one of the order-only recurrent rules "
            f"{ORDERONLY_RSNN_RULES}")
    if (lam_cmp, lam_mem) not in ORDERONLY_WEIGHTS:
        raise CampaignRowError(
            f"({lam_cmp!r}, {lam_mem!r}) is not one of the five ruled weight "
            f"pairs {ORDERONLY_WEIGHTS}")
    if seed not in ORDERONLY_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_SEEDS}")
    return f"orderonly_rsnn_{rule}_l{lam_cmp}m{lam_mem}_s{seed}"


def orderonly_rsnn_pref_run_name(rule: str, seed: str) -> str:
    """`orderonly_rsnn_<rule>_pref_s<seed>` (owner ruling 2026-09-19)."""
    if rule not in ORDERONLY_RSNN_RULES:
        raise CampaignRowError(
            f"rule {rule!r} is not one of the order-only recurrent rules "
            f"{ORDERONLY_RSNN_RULES}")
    if seed not in ORDERONLY_SEEDS:
        raise CampaignRowError(
            f"seed {seed!r} is not one of {ORDERONLY_SEEDS}")
    return f"orderonly_rsnn_{rule}_pref_s{seed}"


def orderonly_rsnn_node(rule: str) -> str:
    """THE NODE OF A RULE.  Every weight and every seed of one rule runs on
    one GPU model, so the rule comparison alone crosses two models."""
    if rule not in ORDERONLY_RSNN_RULES:
        raise CampaignRowError(
            f"rule {rule!r} is not one of the order-only recurrent rules "
            f"{ORDERONLY_RSNN_RULES}")
    return ORDERONLY_RSNN_NODES[rule]


def orderonly_rsnn_arm(*, rule: str, seed: str, lam_cmp: str | None = None,
                       lam_mem: str | None = None, pref: bool = False) -> dict:
    """One dsnn-dfw.45 order-only row on the recurrent target -> one
    `arm(...)`.  Returns the arm.  Mirrors `orderonly_arm` exactly, on
    `rsnn_<rule>` instead of `nn256` and the rule's own node."""
    node = orderonly_rsnn_node(rule)
    target = f"rsnn_{rule}"
    if pref:
        _require(lam_cmp is None and lam_mem is None,
                 "the preference-conditioned row sweeps no weight: the "
                 "Dirichlet preference over (latency, memory) IS the weight, "
                 "and a fixed pair beside it would say two different things")
        name = orderonly_rsnn_pref_run_name(rule, seed)
        arm_name = ORDERONLY_PREF_ARM
    else:
        _require(lam_cmp is not None and lam_mem is not None,
                 "a fixed-weight row needs both --lambda-cmp and --lambda-mem")
        name = orderonly_rsnn_run_name(rule, lam_cmp, lam_mem, seed)
        arm_name = ORDERONLY_ARM
    # dsnn-dfw.84 (2026-09-21) is a thesis-matrix finding; this row is
    # order-only, not a matrix coordinate, so it keeps its own recorded 0.05
    # rather than picking up the near-zero bonus.
    cli = thesis_cli(arm=arm_name, target=target, seed=seed, node=node,
                     name=name, episodes=THESIS_EPISODES,
                     checkpoint_every=THESIS_CHECKPOINT_EVERY,
                     auto_stop=True, face_entropy_weight="0.05")
    cli["--approx-profile"] = ORDERONLY_PROFILE
    if not pref:
        cli["--lambda-cmp"] = lam_cmp
        cli["--lambda-mem"] = lam_mem
    gpus = thesis_row_gpus(target, node)
    what = (f"THE PREFERENCE-CONDITIONED ROW at seed {seed}, rule {rule}: one "
            f"run over a Dirichlet preference on (latency, memory), against "
            f"the five fixed weights of the same rule and seed."
            if pref else
            f"WEIGHT (--lambda-cmp {lam_cmp}, --lambda-mem {lam_mem}) at seed "
            f"{seed}, rule {rule}, one of the five ruled pairs.")
    a = dict(
        name=name, job=orderonly_job_name(node), kind="train",
        runtime="scratch",
        node=node, time=THESIS_TIME, gpus=gpus, singleton=True, thesis=True,
        orderonly_rsnn=True, thesis_arm=arm_name, thesis_target=target,
        thesis_rule=thesis_temporal_rule(target),
        thesis_seed=seed, orderonly_weights=(None if pref
                                             else (lam_cmp, lam_mem)),
        env=thesis_target_form(target, batched=False)[1],
        required_flags=ORDERONLY_REQUIRED_FLAGS,
        required_flags_file=" ".join(THESIS_FLAGS_FILES),
        cli=cli,
        purpose=_ORDERONLY_RSNN_HEAD + "\n\n" + _ORDERONLY_RSNN_WHAT[rule]
                + "\n\n" + what,
        prediction=_ORDERONLY_RSNN_PREDICTION,
        falsifier=_ORDERONLY_RSNN_FALSIFIER,
    )
    if node in NODE_CUDA_BIN:
        a["cuda_bin"] = NODE_CUDA_BIN[node]
    arm_(**a)
    return a


def orderonly_rsnn_submission_order(
        rule: str) -> list[tuple[str, str | None, str | None]]:
    """(seed, lambda_cmp, lambda_mem) for one rule, in submission order; the
    pair is None on the preference-conditioned row.  Seed-major, exactly as
    `orderonly_submission_order` above."""
    order: list[tuple[str, str | None, str | None]] = []
    for s in ORDERONLY_SEEDS:
        for lc, lm in ORDERONLY_WEIGHTS:
            order.append((s, lc, lm))
        order.append((s, None, None))
    return order


#: The 18 runs of one rule: five weights x three seeds, plus one
#: preference-conditioned run per seed.  36 total over the two rules.
ORDERONLY_RSNN_RUNS_PER_RULE = len(ORDERONLY_SEEDS) * (len(ORDERONLY_WEIGHTS)
                                                       + 1)

for _rule in ORDERONLY_RSNN_RULES:
    for _seed, _lc, _lm in orderonly_rsnn_submission_order(_rule):
        orderonly_rsnn_arm(rule=_rule, seed=_seed, lam_cmp=_lc, lam_mem=_lm,
                          pref=_lc is None)
del _rule, _seed, _lc, _lm


def orderonly_rsnn_arms() -> list[dict]:
    return [a for a in ARMS if a.get("orderonly_rsnn")]


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
$PY - <<'PYEOF'
import sys
import graphax, alphagrad
gx, ag = graphax.__file__, alphagrad.__file__
print("graphax.__file__  =", gx)
print("alphagrad.__file__=", ag)
ok = gx.startswith("@GX_REPO@/") and ag.startswith("@AG_REPO@/")
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
for R in @AG_REPO@ @GX_REPO@; do
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
COMMITTED=$(git -C @AG_REPO@ rev-parse HEAD:src/alphagrad/approx/tools/landscape_map.py 2>/dev/null || echo none)
ONDISK=$(git -C @AG_REPO@ hash-object "$TOOL" 2>/dev/null || echo none)
if [ "$COMMITTED" = "$ONDISK" ]; then
  echo "TOOL PROVENANCE: instrument IS the committed HEAD blob $COMMITTED"
else
  echo "TOOL PROVENANCE: WARNING -- instrument is NOT the committed HEAD blob"
  echo "                 on-disk=$ONDISK committed=$COMMITTED"
fi
AG_SHA=$(git -C @AG_REPO@ rev-parse --short HEAD)
GX_SHA=$(git -C @GX_REPO@ rev-parse --short HEAD)
NOTE="ag=$AG_SHA gx=$GX_SHA live tool=$(sha256sum $TOOL | cut -c1-8)"

# A NEW OUTPUT DIRECTORY, DELIBERATELY.  run_analysis/landscape holds the
# 2026-08-27 loss_drop rows measured on the pinned stack; these are
# grad_cosine rows on the live stack and the two must not share a --report-only
# glob.  (landscape_map keys its combined report on the quality metric as
# well, so pooling is prevented twice.)
OUT=@OUT_ROOT@/run_analysis/landscape_gradcos
mkdir -p $OUT
W=@AG_REPO@/wandb
ARCH="--archive v57=$W/run-20260817_113827-it05ku34/files/pareto_front.json \
 --archive v60=$W/run-20260817_181647-ygm8n2jy/files/pareto_front.json \
 --archive v63=$W/run-20260822_165942-as9s5yrl/files/pareto_front.json \
 --archive v64b=$W/run-20260825_132646-38oyqf4g/files/pareto_front.json \
 --archive v65=$W/run-20260826_121153-8sht6x1m/files/pareto_front.json \
 --archive v66a=$W/run-20260826_121153-318ktrgq/files/pareto_front.json \
 --archive v66b=$W/run-20260826_121154-0olsxsjl/files/pareto_front.json \
 --archive v66c=$W/run-20260826_121154-s1537jdd/files/pareto_front.json"

COMMON="--example TransformerLM --dataset wikitext2 \
 --hidden-dim 256 --vocab-size 256 --num-layers 3 --seed 250197 \
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
  CUDA_VISIBLE_DEVICES=$g $PY "$TOOL" "$@"
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
""".replace("@TOOL@", LANDSCAPE_TOOL).replace("@AG_REPO@", f"{CAMPAIGN_STACK}/alphagrad") \
   .replace("@GX_REPO@", f"{CAMPAIGN_STACK}/graphax").replace("@OUT_ROOT@", CAMPAIGN_ROOT) \
   .replace("@APPROX_ADD@", APPROX_ADD)


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
    """The interpreter invocation of this arm's runtime.  Every arm -- wave,
    cpu, tool or campaign -- runs the campaign stack's venv now (owner ruling
    2026-09-14): $HOME/dsnn is 281 commits stale and the export it lives on
    is read-only.  `$PY` is bound by `_stack_exists_check` below."""
    return "$PY"


def _repo_paths(a: dict) -> tuple[str, str]:
    """(alphagrad checkout, graphax checkout) the launcher runs against.

    Every arm runs the same staged worktrees under CAMPAIGN_STACK (owner
    ruling 2026-09-14); there is no longer a second, ~/dsnn-rooted tree."""
    return f"{CAMPAIGN_STACK}/alphagrad", f"{CAMPAIGN_STACK}/graphax"


def _stack_exists_check() -> list[str]:
    """ABORT(66) if the campaign stack (finding 57) is not staged, then bind
    a node-local $HOME carrying the wandb credentials.  Every arm now runs
    from CAMPAIGN_STACK (owner ruling 2026-09-14: the wave 0-4 arms and
    fq_face_attrib used to `cd ~/dsnn/alphagrad`, but that checkout is 281
    commits stale and the home export it lives on is read-only), so this is
    shared by the campaign arms and the wave/cpu/tool arms alike."""
    return [
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
    ]


def _scratch_stack_block(target_env: dict | None = None,
                         jax_cache: bool = True,
                         shd_dir: bool = False) -> list[str]:
    """The environment of a campaign arm: the stack, the plumbing, the TLM
    shape, the measurement vars, the no-flag knobs.  Nothing else.

    ``target_env`` is the thesis matrix's one addition: the target-shape
    variable a NeuralNetwork arm needs at import time (THESIS_TARGET_ENV).
    Empty for every campaign arm, so their rendering does not move.

    Since dsnn-dfw.99 it also carries the measure actors' retention
    bound, which is rendered in its own block above the target shape.
    """
    target_env = dict(target_env or {})
    L = _stack_exists_check() + [
        f"export PYTHONPATH={CAMPAIGN_STACK}/graphax/src:{CAMPAIGN_STACK}/alphagrad/src",
        f"export DSNN_WIKITEXT_DIR={CAMPAIGN_CACHE}/dsnn_wikitext",
        f"export DSNN_MNIST_DIR={CAMPAIGN_CACHE}/dsnn_mnist",
    ] + ([f"export DSNN_SHD_DIR={CAMPAIGN_CACHE}/dsnn_shd"] if shd_dir else []) + [
        f"cd {CAMPAIGN_STACK}/alphagrad",
        "",
        "# ---------------------- THE ENVIRONMENT (args only) -------------------",
        "# Owner ruling 2026-09-13: every knob is an ARGUMENT.  Exported here:",
        "# the TLM TARGET shape (the target's size comes from these, not from",
        "# --hidden-dim/--vocab-size/--num-layers, which size the POLICY) and",
        "# the Ray / measurement plumbing.  No XLA_*; the campaign test refuses",
        "# any export outside CAMPAIGN_ENV_ALLOWED, JAX_* included -- the four",
        "# names below (owner ruling 2026-09-14, small fixes #3) are the one",
        "# allowed exception, the per-node persistent JAX compile cache.",
    ]
    for k, v in CAMPAIGN_ENV:
        L.append(f"export {k}={v}")
    L.append("")
    L.append("# TODO (ticket .43): knobs that still have NO FLAG in ppo.py -- named")
    L.append("# in the header's TODO block with their evidence; promote and delete.")
    for k, v, _why in NO_FLAG_ENV:
        L.append(f"export {k}={v}")
    _clear_every = target_env.pop(MEASURE_CACHE_CLEAR_EVERY_VAR, None)
    if _clear_every is not None:
        L.append("")
        L.append("# THE MEASURE ACTORS' EXECUTABLE RETENTION BOUND (thesis")
        L.append("# matrix, dsnn-dfw.99).  Under the free order nearly every plan")
        L.append("# is a new program, so the actor drops its in-process JAX caches")
        L.append("# every N measurements and the executables go back to the device.")
        L.append("# The on-disk compile cache survives the clear.  ppo.py has no")
        L.append("# flag for it.")
        L.append(f"export {MEASURE_CACHE_CLEAR_EVERY_VAR}={_clear_every}")
    _proactive_recycle = target_env.pop(PROACTIVE_RECYCLE_EVERY_VAR, None)
    if _proactive_recycle is not None:
        L.append("")
        L.append("# THE MEASURE ACTORS' PROCESS RECYCLE (thesis matrix,")
        L.append("# dsnn-dfw.99 follow-up).  The retention bound above clears the")
        L.append("# in-process JAX caches but does not return the executables'")
        L.append("# device memory to the pool; only recycling the actor PROCESS")
        L.append("# does.  ppo.py has no flag for it.")
        L.append(f"export {PROACTIVE_RECYCLE_EVERY_VAR}={_proactive_recycle}")
    _measure_path = {k: target_env.pop(k) for k in THESIS_MEASURE_PATH_ENV
                     if k in target_env}
    if _measure_path:
        L.append("")
        L.append("# THE MEASURE PATH (thesis matrix, dsnn-dfw.169): the direct")
        L.append("# measurement and the unified face enumeration.  env.py reads")
        L.append("# both, and ppo.py has no flag for them.")
        for k in sorted(_measure_path):
            L.append(f"export {k}={_measure_path[k]}")
    _no_disk_cache = {k: target_env.pop(k) for k in THESIS_NO_DISK_CACHE_ENV
                      if k in target_env}
    _batch = target_env.pop(NN_BATCH_VAR, None)
    if target_env:
        L.append("")
        L.append("# THE TARGET SHAPE (thesis matrix, ticket dsnn-dfw.4).  The")
        L.append("# NeuralNetwork target's hidden width is read at IMPORT time by")
        L.append("# common/examples.py (module-scope _EQ_NN_HIDDEN) exactly as the")
        L.append("# ALPHAGRAD_TLM_* triple above is, and ppo.py has no flag for it.")
        for k in sorted(target_env):
            L.append(f"export {k}={target_env[k]}")
    if _batch is not None:
        L.append("")
        L.append("# THE BATCH (dsnn-4ay).  common/datasets.py reads it at")
        L.append("# IMPORT time, and ppo.py has no flag for it.")
        L.append(f"export {NN_BATCH_VAR}={_batch}")
    L.append("")
    if jax_cache:
        L.extend(_jax_cache_lines())
    else:
        L.append("# NO JAX COMPILE CACHE, under any order (owner rulings")
        L.append("# 2026-09-23 and 2026-09-25, dsnn-dfw.230): writing it takes time")
        L.append("# and most plans are never compiled again.")
        L.append("# ppo.py and cpu_approx_worker.py write one to /tmp unless this")
        L.append("# is 1 (dsnn-dfw.247).")
        for k in sorted(_no_disk_cache):
            L.append(f"export {k}={_no_disk_cache[k]}")
    return L


def render(a: dict) -> str:
    kind = a["kind"]
    gpus = a.get("gpus", 0)
    scratch = is_scratch(a)
    py = _python(a)
    ag_repo, gx_repo = _repo_paths(a)
    L = ["#!/bin/bash"]
    # The partition is keyed on the NODE, not the kind: pgi15-cpu1 is a
    # member only of the pgi15-cpu partition, never of pgi15 (a head node is
    # in neither).  pgi15-cpu2 sits in both; it takes the ordinary GPU-node
    # partition, pgi15, like every other non-head node here.
    L.append("#SBATCH -p " + node_partition(a["node"]))
    L.append(f"#SBATCH -w {a['node']}")
    if scratch:
        # THE CAMPAIGN HARDWARE: the whole Blackwell node, by its gres name.
        # Sized by the arm's own GPU count, because the thesis matrix runs on
        # BOTH Blackwell sizes: gpu19 and gpu20 carry eight GPUs and 128 CPUs,
        # gpu15-gpu18 carry four and 64 (sinfo, 2026-09-16).  A campaign arm
        # is 8 GPUs and renders the identical three lines it always did.
        L.append(f"#SBATCH --gres={node_gres(a['node'], gpus)}")
        L.append(f"#SBATCH -c {node_cpus(a['node'], gpus)}")
        L.append(f"#SBATCH --mem={a.get('mem') or node_mem(a['node'], gpus)}")
    elif gpus:
        L.append(f"#SBATCH --gres=gpu:{gpus}")
        L.append("#SBATCH -c 64")
        L.append("#SBATCH --mem=400G")
    else:
        L.append("#SBATCH -c 8")
        L.append("#SBATCH --mem=64G")
    L.append(f"#SBATCH -t {a['time']}")
    L.append(f"#SBATCH -J {a['job']}")
    if a.get("singleton"):
        # THE PER-NODE SINGLETON QUEUE (owner ruling 2026-09-16, the thesis
        # matrix).  `-J` is the NODE, not the run, and `--dependency=singleton`
        # holds a job until every other job of this user with the same name
        # has finished.  Slurm then runs one of our jobs per node at a time
        # and starts the next as soon as that node frees up.  Two jobs of one
        # user on one pgi15 node kill each other through the node epilog
        # (memory note pgi15-epilog-kills-sibling-jobs), so this is a
        # correctness guard, not a convenience.  The RUN's own name is
        # --name / the -o log file, not -J.
        L.append("#SBATCH --dependency=singleton")
    # -D and -o on /Scratch, for every arm (owner ruling 2026-09-14): a
    # launcher whose -o names the missing/stale home fails at launch with
    # ExitCode 0:53 (finding 57).  CAMPAIGN_RUNS must exist before sbatch
    # (slurm opens the log first): the owner creates it once when staging
    # the stack.
    L.append(f"#SBATCH -D {CAMPAIGN_STACK}/alphagrad")
    L.append(f"#SBATCH -o {CAMPAIGN_RUNS}/{a['name']}_%j.log")
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
    if a.get("paired_into"):
        L.append("#")
        L.append("# *** ONE HALF OF A PAIRED JOB -- SUBMIT"
                 f" fq_{a['paired_into']}.sbatch ***")
        L.append(_wrap_comment(
            f"This row shares its node with the other seed of its arm and "
            f"the two run CONCURRENTLY inside one sbatch, because the node "
            f"epilog kills a sibling job of this user on the same node. The "
            f"file below is the readable record of the row and the command "
            f"line the pair runs; submitting it alone aborts 74.", "#   "))
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

    if a.get("paired_into"):
        # THIS ROW IS HALF OF A PAIRED JOB (owner ruling 2026-09-20).  The
        # whole launcher is still generated, unchanged, because it is the
        # readable record of the row AND the file the paired launcher's
        # command line is diffed against.  It may not be SUBMITTED: its run
        # happens inside the pair, and two jobs of ours on one node kill
        # each other through the node epilog.  74 = submitted on its own.
        L.append("# HALF OF A PAIRED JOB (see the header).  74 = submitted"
                 " on its own.")
        L.append('if [ "${FQ_RELEASE_HALF:-0}" != "1" ]; then')
        L.append(f'  echo "ABORT(74): {a["name"]} runs as one half of'
                 f' fq_{a["paired_into"]}.sbatch -- submit THAT launcher,'
                 ' not this one"')
        L.append("  exit 74")
        L.append("fi")
        L.append("")

    if kind == "cpu" and not a.get("needs_tool"):
        L.append("export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID")
        L.extend(_stack_exists_check())
        L.append(f"export PYTHONPATH={CAMPAIGN_STACK}/graphax/src:{CAMPAIGN_STACK}/alphagrad/src")
        L.append("export PYTHONDONTWRITEBYTECODE=1")
        L.append(f"cd {CAMPAIGN_STACK}/alphagrad")
        L.append("")
        L.append(_toolchain_block(kind, a["node"]))
        L.append("")
        L.append('echo "HOST=$(hostname) JOB=$SLURM_JOB_ID"')
        L.append(f'echo "ag=$(git -C {ag_repo} rev-parse --short HEAD)'
                 f' gx=$(git -C {gx_repo} rev-parse --short HEAD)"')
        L.append(a["body"])
        return "\n".join(L) + "\n"

    # --- the environment
    if scratch:
        allowed_env = THESIS_TARGET_ENV_ALLOWED if a.get("thesis") else set()
        bad = sorted(set(a.get("env") or {}) - set(allowed_env))
        if bad:
            raise CampaignRowError(
                f"{a['name']}: a campaign arm carries no per-arm env "
                f"(got {bad}); every knob is an argument. A THESIS arm may "
                f"carry only {sorted(THESIS_TARGET_ENV_ALLOWED)}, the target "
                f"shape that common/examples.py reads at import time and for "
                f"which ppo.py has no flag.")
        _jc = bool(a.get("jax_cache", True))
        if bool(set(THESIS_NO_DISK_CACHE_ENV) & set(a.get("env") or {})) == _jc:
            raise CampaignRowError(
                f"{a['name']}: jax_cache={_jc} and "
                f"{sorted(THESIS_NO_DISK_CACHE_ENV)} "
                f"{'set' if _jc else 'unset'}; a row without the JAX compile "
                f"cache sets it, a row with the cache does not (dsnn-dfw.247)")
        L.extend(_scratch_stack_block(
            a.get("env") or {}, jax_cache=a.get("jax_cache", True),
            shd_dir=bool(a.get("shd_dir", False))))
    else:
        # A wave/cpu/tool arm (owner ruling 2026-09-14): the same stack, the
        # same node-local $HOME for wandb, and the same ABORT(66) check as a
        # campaign arm, but its OWN per-arm environment -- not args-only.
        L.append("export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID")
        L.extend(_stack_exists_check())
        L.append(f"cd {CAMPAIGN_STACK}/alphagrad")
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
        L.extend(_jax_cache_lines())
    if a.get("cuda_bin"):
        # This node's matched toolkit is outside /usr/local, which is all the
        # block below searches.  Put it on PATH here, through the environment
        # and nothing else: the block still PROVES that ptxas and nvlink both
        # read CUDA_WANT, so a wrong directory aborts 72 instead of degrading
        # every measurement of the run (finding 03).
        L.append("")
        L.append(f'export PATH="{a["cuda_bin"]}:$PATH"')
    L.append("")
    L.append(_toolchain_block(kind, a["node"]))
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
        _twoop = f"{gx_repo}/tests/misc/test_face_two_op_form.py"
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
    halves = a.get("halves") or []
    tags = [_PAIR_TAGS[h["half"]] for h in halves]
    arrays = [f"ARGS_{t}" for t in tags] if halves else ["ARGS"]
    for arr, src in zip(arrays, [h["cli"] for h in halves] or [a.get("cli", {})]):
        L.append(f"{arr}=(")
        for flag, val in _merge_cli(src):
            L.append(f"  {flag}" + (f" {val}" if val is not None else ""))
        L.append(f"  {WANDB}")
        L.append(")")
        L.append("")
    L.append("# Layer 2: run the EXACT token list through ppo.py's own argparse.")
    L.append("# This catches a bad CHOICE value (a --face-read typo, an")
    L.append("# --advantage-norm value valid on the ray surface but not this one)")
    L.append("# that a name-only grep cannot see.  JAX_PLATFORMS=cpu so it does")
    L.append("# not touch a GPU.")
    for arr in arrays:
        L.append(f"JAX_PLATFORMS=cpu {py} -c \\")
        L.append("\"import sys; from alphagrad.approx.ppo import make_argparser;\\")
        L.append(" make_argparser().parse_args(sys.argv[1:]);\\")
        L.append(" print('[preflight] argparse accepted the command line')\" \\")
        L.append(f'  "${{{arr}[@]}}" ||'
                 ' { echo "ABORT(65): argparse rejected it"; exit 65; }')
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
    if halves:
        # TWO TRAINERS, ONE JOB (owner ruling 2026-09-20).  The node epilog
        # kills a sibling job of this user on the same node, so the second
        # seed cannot be a second sbatch.  Each half gets its own GPUs, its
        # own cores, its own Ray temp directory, its own log file and its own
        # wandb run (its --name is the matrix coordinate, unchanged), and the
        # job exits with the WORSE of the two trainer codes.
        for tag, h in zip(tags, halves):
            L.append(f"# --- half {tag}: seed {h['seed']},"
                     f" GPUs {h['devices']}, cores {h['cores']}")
            L.append("(")
            L.append(f"  export CUDA_VISIBLE_DEVICES={h['devices']}")
            L.append("  export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID}"
                     f"_s{h['seed']}")
            L.append(f'  echo "[half {tag}] seed {h["seed"]}'
                     f' CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'
                     f' cores {h["cores"]} RAY_TMPDIR=$RAY_TMPDIR"')
            # WHAT THIS HALF ACTUALLY SEES.  `nvidia-smi` stood here and was
            # evidence of nothing: it talks to the driver and enumerates
            # every GPU of the node whatever CUDA_VISIBLE_DEVICES says, so
            # both halves printed the same eight lines and a half wired to
            # the wrong devices would have looked healthy.  JAX reads the
            # mask, so jax.devices() IS the half's device list.  On a node
            # with no GPU it returns the CPU device and the line still
            # prints, which is what makes this safe in a CPU test.
            _dev_dump = (
                f"import jax; print('[half {tag}] jax devices: ' + "
                "', '.join(str(d.id) + ':' + d.platform + ':' + "
                "d.device_kind for d in jax.devices()))")
            L.append(f'  {py} -c "{_dev_dump}"')
            L.append(f"  taskset -c {h['cores']} {py} \\")
            L.append(f'    src/alphagrad/approx/ppo.py "${{ARGS_{tag}[@]}}"')
            L.append(f") > {CAMPAIGN_RUNS}/{h['name']}"
                     f"_${{SLURM_JOB_ID}}_s{h['seed']}.log 2>&1 &")
            L.append(f"PID_{tag}=$!")
            L.append("")
        for tag, h in zip(tags, halves):
            L.append(f'wait "$PID_{tag}"')
            L.append(f"STATUS_{tag}=$?")
            L.append(f'echo "TRAINER half {tag} (seed {h["seed"]},'
                     f' {h["name"]}) exited with $STATUS_{tag}"')
        L.append(f"TRAINER_STATUS=$STATUS_{tags[0]}")
        for tag in tags[1:]:
            L.append(f'if [ "$STATUS_{tag}" -gt "$TRAINER_STATUS" ]; then')
            L.append(f"  TRAINER_STATUS=$STATUS_{tag}")
            L.append("fi")
        L.append('echo "TRAINER exited with $TRAINER_STATUS"')
        L.append('exit "$TRAINER_STATUS"')
        return "\n".join(L) + "\n"
    if scratch:
        # All GPUs of the node are visible: the trainer takes device 0
        # (--gpus 0, the ppo.py default) and the --ray-measure actor device 1
        # (ppo.py pins idx + 1 under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES).
        # A matrix row names both: --gpus 0 and --measure-gpus 1..n (dsnn-dfw.245).
        L.append(f"{py} \\")
    else:
        L.append(f"CUDA_VISIBLE_DEVICES=0,1,2,3 {py} \\")
    L.append('  src/alphagrad/approx/ppo.py "${ARGS[@]}"')
    L.append("TRAINER_STATUS=$?")
    L.append('echo "TRAINER exited with $TRAINER_STATUS"')
    L.append('exit "$TRAINER_STATUS"')
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
