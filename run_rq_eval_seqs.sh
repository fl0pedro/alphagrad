#!/bin/bash
#SBATCH --job-name=rq_seq_eval
#SBATCH --nodes=1
#SBATCH --partition=pgi15-cpu
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

# Per-RQ best-sequence MNIST evaluation. SUBMITTED AS ITS OWN SLURM
# JOB by the master sbatch (run_full_research.sh), which awaits the
# previous eval job before submitting the next so at most 2 slurm
# jobs exist concurrently: the master + the in-flight eval.
#
# For one RQ number (passed as positional arg or RQ env var), this script:
#
#   1. Globs every wandb offline run whose ``--name`` starts with
#      ``rq{N}_`` (RQ-tagged runs only).
#   2. Stages each one's ``best_sequences.json`` into
#      ``~/dsnn/results/rq{N}_seq_eval/sequences/<run_tag>.json``.
#   3. Writes a ``meta.json`` carrying provenance per tag
#      (variant, curriculum, source wandb run id) so the labels on the
#      wandb-mnist-seq plot are useful.
#   4. Invokes ``alphagrad/playground/nn_mnist.py`` directly (same
#      harness as ``slurm/nn-mnist-seq.bash`` but inline — we're
#      already inside the master sbatch's SLURM job, no need to nest).
#
# Output: ``~/dsnn/results/rq{N}_seq_eval/nn_results_vmapped_group0.csv``
# plus a per-sequence wandb run under project ``rq-mnist-seq`` (entity
# ``dll-streetview``) so all RQs feed the same dashboard project for
# direct comparison.
#
# Usage:
#   bash alphagrad/run_rq_eval_seqs.sh <RQ_NUMBER>
#   RQ=2 bash alphagrad/run_rq_eval_seqs.sh
#
# Env overrides:
#   MAX_STEPS              (2000)
#   TIME_LIMIT_S           (10)
#   NUM_RUNS               (10)
#   ANALYSIS_FREQUENCY     (20)
#   BATCH_SIZE             (16)
#   PPO_FACTOR_TABLE       (-1,2,3,4)
#   WANDB                  (online)
#   WANDB_PROJECT          (rq-mnist-seq)
#   WANDB_ENTITY           (dll-streetview)

set -uo pipefail
cd ~/dsnn
export PATH="$HOME/.local/bin:$PATH"

# VmappedNeuralNetwork on MNIST is small enough that CPU evaluation
# (graphax JAX compilation + a few hundred train steps) is fast and
# leaves the GPUs free for the RL search running concurrently in the
# master sbatch. Force JAX to CPU even though no GPUs are visible.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

RQ="${1:-${RQ:-}}"
if [[ -z "$RQ" ]]; then
    echo "usage: $0 <RQ_NUMBER|ref>  (or set RQ env var)"
    exit 2
fi

# Special "ref" mode: don't discover any RL runs — just train the
# built-in graphax fwd/rev baselines (nn_mnist.py EXTRA_CONFIGS) so
# the reference numbers exist before any RL phase finishes.
REF_MODE=0
if [[ "$RQ" == "ref" ]]; then
    REF_MODE=1
fi

# Eval knobs — defaults match nn-mnist-seq.bash so apples-to-apples
# comparison is possible.
MAX_STEPS="${MAX_STEPS:-2000}"
TIME_LIMIT_S="${TIME_LIMIT_S:-10}"
NUM_RUNS="${NUM_RUNS:-10}"
ANALYSIS_FREQUENCY="${ANALYSIS_FREQUENCY:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PPO_FACTOR_TABLE="${PPO_FACTOR_TABLE:--1,2,3,4}"

WANDB="${WANDB:-online}"
# Use a dedicated env var name (WANDB_PROJECT_SEQ) instead of the
# generic WANDB_PROJECT because the master sbatch exports the latter
# = dsnn-downstream for the downstream_train.py helper. Inheriting
# that here would land the eval in the wrong project.
WANDB_PROJECT="${WANDB_PROJECT_SEQ:-rq-mnist-seq}"
WANDB_ENTITY="${WANDB_ENTITY:-dll-streetview}"

WANDB_DIR="${WANDB_DIR:-$HOME/dsnn/wandb}"
EVAL_DIR="$HOME/dsnn/results/rq${RQ}_seq_eval"
SEQ_DIR="$EVAL_DIR/sequences"
mkdir -p "$SEQ_DIR" "$EVAL_DIR/.data/mnist"

# Reuse the MNIST .gz files the env trainer downloaded for the RL
# runs — saves a duplicate download. `--update=none` is the portable
# replacement for the deprecated `cp -n`; works on coreutils 9+.
if [[ -d "$HOME/.cache/dsnn_mnist" ]]; then
    for f in "$HOME/.cache/dsnn_mnist"/*.gz; do
        if [[ -f "$f" && ! -f "$EVAL_DIR/.data/mnist/$(basename "$f")" ]]; then
            cp "$f" "$EVAL_DIR/.data/mnist/"
        fi
    done
fi

# nn_mnist.py needs pandas — nn-mnist-seq.bash installs it per-job
# but the master sbatch reuses the project venv so we install it
# here (idempotent — uv pip install is a no-op if already present).
if ! uv run --no-sync python -c "import pandas" 2>/dev/null; then
    echo "  installing pandas (one-time, idempotent)"
    uv pip install pandas grain 2>&1 | tail -3
fi

echo "=== RQ${RQ} sequence evaluation ==="
echo "  EVAL_DIR=$EVAL_DIR"
echo "  WANDB=$WANDB project=$WANDB_PROJECT entity=$WANDB_ENTITY"
echo "  REF_MODE=$REF_MODE"

# Step 1: discover RQ-tagged wandb runs (both online and offline layouts).
# Skipped in ref mode — nn_mnist.py's EXTRA_CONFIGS provides gx_jacve_fwd
# and gx_jacve_rev as built-in baselines that run with no staged JSON.
declare -a RQ_RUNS=()
if (( REF_MODE == 0 )); then
    for d in "$WANDB_DIR"/run-* "$WANDB_DIR"/offline-run-*; do
        [[ -d "$d" ]] || continue
        meta="$d/files/wandb-metadata.json"
        [[ -f "$meta" ]] || continue
        if grep -qE "\"rq${RQ}_" "$meta"; then
            RQ_RUNS+=("$d")
        fi
    done
    echo "  discovered ${#RQ_RUNS[@]} rq${RQ}_* wandb runs"

    if [[ ${#RQ_RUNS[@]} -eq 0 ]]; then
        echo "  no rq${RQ}_* runs found under $WANDB_DIR — nothing to evaluate"
        exit 0
    fi
else
    echo "  ref mode: skipping discovery; nn_mnist.py will run gx_jacve_{fwd,rev} baselines only"
fi

# Step 2 + 3: stage best_sequences.json files + build meta.json.
# Skipped in ref mode (no RL runs to harvest).
if (( REF_MODE == 1 )); then
    echo "{}" > "$SEQ_DIR/meta.json"
else
echo "{" > "$SEQ_DIR/meta.json"
first=1
for run_dir in "${RQ_RUNS[@]}"; do
    bsj="$run_dir/files/best_sequences.json"
    cfg="$run_dir/files/wandb-metadata.json"
    if [[ ! -f "$bsj" ]]; then
        echo "  [skip] no best_sequences.json under $(basename "$run_dir")"
        continue
    fi

    # Read --name from wandb-metadata args. Stem becomes the JSON
    # filename + the "algo" field in the per-sequence record.
    tag=$(python3 -c "
import json, sys
m = json.load(open('$cfg'))
argv = m.get('args', [])
for i, a in enumerate(argv):
    if a == '--name' and i + 1 < len(argv):
        print(argv[i + 1]); break
")
    [[ -z "$tag" ]] && tag="$(basename "$run_dir")"

    # Sanitise filename — wandb run names can have '/'; keep just the
    # rq{N}_<rest> portion.
    safe_tag=$(echo "$tag" | tr '/' '_' | tr -cd '[:alnum:]_-')
    dest="$SEQ_DIR/${safe_tag}.json"
    cp "$bsj" "$dest"

    # Provenance for the meta.json. Extract variant, curriculum, the
    # short wandb run id (from the dir basename like
    # ``offline-run-20260522_193422-jp5wpbua`` → ``jp5wpbua``).
    run_id=$(basename "$run_dir" | sed -E 's/.*-([a-z0-9]+)$/\1/')
    variant=$(python3 -c "
import json
m = json.load(open('$cfg'))
argv = m.get('args', [])
for i, a in enumerate(argv):
    if a == '--variant' and i + 1 < len(argv):
        print(argv[i + 1]); break
")
    curriculum=$(python3 -c "
import json
m = json.load(open('$cfg'))
argv = m.get('args', [])
for i, a in enumerate(argv):
    if a == '--curriculum' and i + 1 < len(argv):
        print(argv[i + 1]); break
")

    if (( first )); then
        first=0
    else
        echo "," >> "$SEQ_DIR/meta.json"
    fi
    printf '  "%s": {"variant": "%s", "curriculum": "%s", "wandb_run_id": "%s"}' \
        "$safe_tag" "${variant:-}" "${curriculum:-}" "$run_id" \
        >> "$SEQ_DIR/meta.json"
done
echo "" >> "$SEQ_DIR/meta.json"
echo "}" >> "$SEQ_DIR/meta.json"
fi  # end of !REF_MODE block

n_staged=$(ls "$SEQ_DIR"/*.json 2>/dev/null | grep -v -E '/(meta|sequences_dedup)\.json$' | wc -l)
echo "  staged $n_staged sequence files under $SEQ_DIR"
if (( REF_MODE == 0 )) && [[ "$n_staged" -eq 0 ]]; then
    echo "  no sequence files staged — exiting"
    exit 0
fi

# Step 4: invoke nn_mnist.py in parallel groups on CPU.
# nn_mnist.py's ``--group N --groups M`` machinery splits the
# sequence list into M strides — group 0 handles indices (0, M, 2M, ...),
# group 1 handles (1, M+1, ...), etc. — so the work is shared.
# Each group emits its own CSV ``nn_results_vmapped_group{N}.csv``
# and per-sequence wandb runs.
#
# pgi15-cpu has 128 cores allocated here, so we let JAX/XLA use a
# fraction per group (16 threads/group × 8 groups = 128) by capping
# XLA's intra-op thread count.
cd "$EVAL_DIR"

WANDB_FLAGS=(
    --wandb "$WANDB"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name-prefix "rq${RQ}"
)
if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_FLAGS+=(--wandb-entity "$WANDB_ENTITY")
fi

NUM_GROUPS="${NUM_GROUPS:-8}"
THREADS_PER_GROUP="${THREADS_PER_GROUP:-16}"
echo "  invoking nn_mnist.py × $NUM_GROUPS parallel groups (CPU, $THREADS_PER_GROUP threads each):"
echo "    MAX_STEPS=$MAX_STEPS  NUM_RUNS=$NUM_RUNS  BATCH_SIZE=$BATCH_SIZE"
echo

declare -a EVAL_PIDS=()
declare -a EVAL_TAGS=()
for ((g=0; g<NUM_GROUPS; g++)); do
    logfile="$EVAL_DIR/nn-mnist-seq_rq${RQ}_group${g}.log"
    echo "    group $g log $logfile"
    (
        # XLA / OpenMP / MKL / OpenBLAS thread caps — keep one group
        # from eating all 128 cores and starving the others.
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1" \
        OMP_NUM_THREADS="$THREADS_PER_GROUP" \
        MKL_NUM_THREADS="$THREADS_PER_GROUP" \
        OPENBLAS_NUM_THREADS="$THREADS_PER_GROUP" \
            uv run --no-sync python -u ~/dsnn/alphagrad/playground/nn_mnist.py \
            --group "$g" \
            --groups "$NUM_GROUPS" \
            --sequences-dir "$SEQ_DIR" \
            --max-steps "$MAX_STEPS" \
            --time-limit-s "$TIME_LIMIT_S" \
            --num-runs "$NUM_RUNS" \
            --analysis-frequency "$ANALYSIS_FREQUENCY" \
            --batch-size "$BATCH_SIZE" \
            --ppo-factor-table="$PPO_FACTOR_TABLE" \
            --mnist-dir .data/mnist \
            "${WANDB_FLAGS[@]}"
    ) > "$logfile" 2>&1 &
    EVAL_PIDS+=("$!")
    EVAL_TAGS+=("group${g}")
done

# Wait for all groups; track exit codes individually.
FAILED=0
for i in "${!EVAL_PIDS[@]}"; do
    pid="${EVAL_PIDS[$i]}"
    tag="${EVAL_TAGS[$i]}"
    if wait "$pid"; then
        echo "  [$tag] OK"
    else
        rc=$?
        echo "  [$tag] FAILED (rc=$rc)"
        FAILED=$((FAILED + 1))
    fi
done
EXIT_CODE=$FAILED
echo
echo "=== RQ${RQ} eval done (rc=$EXIT_CODE) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "  CSV: $EVAL_DIR/nn_results_vmapped_group0.csv"
echo "  wandb: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}?query=rq${RQ}_"
exit "$EXIT_CODE"
