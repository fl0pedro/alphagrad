#!/bin/bash -l
# THE STANDARD SMOKE GATE.
#
# Runs the canonical 2-episode CPU config and FAILS (non-zero exit) when a
# post-warm-up health metric is non-finite or missing. Before this existed the
# smoke was an ad-hoc command line whose only gate was "rc=0", i.e. it proved
# the pipeline RUNS -- not that the loss is finite or the PPO ratio is sane.
# See docs/VALIDATION_GATES.md for what this does and does NOT prove.
#
#   usage: tools/smoke.sh [example ...]        (default: Helmholtz NeuralNetwork)
#
# Run it on a compute node, never the head node:
#   srun -p pgi15-cpu -w pgi15-cpu1 --mem=64G -c 8 -t 2:00:00 tools/smoke.sh
#
# WHY THE `[health warmup]` ROWS ARE NOT CHECKED: --popart-init-episodes runs
# random-plan rollouts with NO gradient step, so every loss-derived metric is
# undefined there and prints `n/a`. Those rows are skipped BY LABEL, so a real
# NaN can no longer hide among expected ones.
set -uo pipefail

EXAMPLES=("$@")
if [ ${#EXAMPLES[@]} -eq 0 ]; then
    EXAMPLES=(Helmholtz NeuralNetwork)
fi

cd "$(dirname "$0")/.." || exit 2
ROOT=$PWD
OUT=${SMOKE_OUT:-$ROOT/smoke_out}
mkdir -p "$OUT"

export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$ROOT/../.jc_smoke}
# Every training episode must print its health row, or a 2-episode smoke can
# silently check nothing.
export ALPHAGRAD_HEALTH_EPISODES=${ALPHAGRAD_HEALTH_EPISODES:-99}

# The canonical config. Keep in sync with docs/VALIDATION_GATES.md.
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --grad-window 0 --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"

FAIL=0
for EX in "${EXAMPLES[@]}"; do
    LOG=$OUT/smoke_$EX.log
    echo "########## SMOKE $EX ##########"
    timeout "${SMOKE_TIMEOUT:-3300}" uv run --no-sync \
        python src/alphagrad/approx/ppo.py $COMMON \
        --example "$EX" --name "smoke_$EX" > "$LOG" 2>&1
    RC=$?
    echo "$EX rc=$RC   log=$LOG"
    if [ $RC -ne 0 ]; then
        echo "SMOKE FAIL [$EX]: process exited $RC"
        tail -25 "$LOG"
        FAIL=1
        continue
    fi
    grep -a "\[health " "$LOG" || true
    # The gate. Only `[health ep<N>]` rows -- the post-warm-up ones -- carry
    # defined loss metrics; `[health warmup]` rows are skipped by label.
    awk -v ex="$EX" '
        {
            idx = index($0, "[health ep")
            if (idx == 0) next
            line = substr($0, idx)
            if (line !~ /^\[health ep[0-9]+\]/) next
            if (line !~ /ppo=/) next
            n++
            split(line, F, /[ \t]+/)
            for (i = 1; i <= 12; i++) {
                if (F[i] == "") continue
                split(F[i], kv, "=")
                k = kv[1]; v = kv[2]
                if (k != "ppo" && k != "value" && k != "ent" && k != "ratio/max_log") continue
                lv = tolower(v)
                if (lv ~ /nan/ || lv ~ /inf/) {
                    printf "SMOKE FAIL [%s]: %s is non-finite (%s) on health row %s\n", ex, k, v, F[2]
                    bad++
                } else if (lv == "n/a") {
                    printf "SMOKE FAIL [%s]: %s is MISSING (n/a) on health row %s -- the canonical config always defines it\n", ex, k, F[2]
                    bad++
                }
            }
        }
        END {
            if (n == 0) {
                printf "SMOKE FAIL [%s]: no post-warm-up [health ep..] row was printed -- nothing was validated\n", ex
                exit 1
            }
            printf "[%s] %d post-warm-up health row(s) checked\n", ex, n
            exit (bad > 0)
        }' "$LOG" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo "===== SMOKE FAILED ====="
    exit 1
fi
echo "===== SMOKE PASSED ====="
