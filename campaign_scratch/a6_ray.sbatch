#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=96G
#SBATCH -c 24
#SBATCH -t 3:00:00
#SBATCH -J a6_ray
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_ray_%j.log
# A6 under --ray-measure. EVERY wave-1..4 arm runs --ray-measure 3, so the
# measurement callback -- and therefore the plan-log recorder -- runs in the
# ACTOR processes and the trainer's own record list is EMPTY for every pooled
# terminal. This is the production path: without
# measure_pool.merge_pool_plan_records the log would silently contain nothing
# at all. Pinned here rather than assumed.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a6_out
mkdir -p $R
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
export ALPHAGRAD_BATCHED_CALLBACK=1
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
unset ALPHAGRAD_QUALITY_METRIC
unset ALPHAGRAD_PLAN_LOG
export ALPHAGRAD_EXTEND_CHUNK=256
export ALPHAGRAD_EXTEND_UNROLL=32
export ALPHAGRAD_DELTA_OVERFLOW=clip
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_UNIFIED_FACE_ENUM=1
export ALPHAGRAD_APPEND_ONLY_JAXPR=1
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export ALPHAGRAD_MAX_EQNS=512
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a6

LOG=$R/ray_plan_log.jsonl
rm -f $LOG
echo "########## --ray-measure 2 + --plan-log ##########"
uv run --no-sync python src/alphagrad/approx/ppo.py \
  --variant full --face-actions --unified-face-head --live-faces \
  --set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
  --grad-window 0 --dataset none \
  --cmp-type flops --mem-type peak_memory --terminal-rewards-only \
  --rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
  --advantage-norm popart --episodes 3 --seed 42 --num-envs 4 --minibatches 1 \
  --vocab-size 512 --wandb disabled --example NeuralNetwork \
  --latency-inner-reps 50 --ray-measure 2 --ray-measure-timeout 600 \
  --plan-log $LOG --name a6_ray > $R/ray.log 2>&1
echo "RAY_RC=$?"
grep -E "\[plan-log\]|\[cfg\] plan log|plan_log/" $R/ray.log | head -20
echo "########## records ##########"
wc -l -c $LOG 2>&1
uv run --no-sync python - <<'PY' 2>&1 | grep -v nanobind | grep -v "leaked "
import collections, json
from alphagrad.approx.common import plan_log as plog
recs = plog.read_records("/Users/assmuth/dsnn/alphagrad/a6_out/ray_plan_log.jsonl")
print(f"records={len(recs)}")
if not recs:
    raise SystemExit("FAIL: pooled run wrote NO plan records")
by_actor = collections.Counter(r.get("actor") for r in recs)
by_pid = collections.Counter(r.get("pid") for r in recs)
by_ep = collections.Counter(r["episode"] for r in recs)
print(f"by_actor={dict(by_actor)}")
print(f"by_pid={dict(by_pid)}")
print(f"by_episode={dict(sorted(by_ep.items()))}")
n_from_actor = sum(1 for r in recs if r.get("actor") is not None)
print(f"records measured IN A MEASURE ACTOR: {n_from_actor}/{len(recs)}")
r = recs[0]
print("sample:", json.dumps({k: v for k, v in r.items()
                             if k not in ("faces", "rules", "coverage")})[:600])
print("coverage:", json.dumps({k: v for k, v in r["coverage"].items()
                               if not isinstance(v, list)}))
assert n_from_actor > 0, "FAIL: no record came from a measure actor"
print("POOL DRAIN: ok")
PY
echo "POOL_CHECK_RC=$?"
echo "########## structure + REPLAY of the recorded losers ##########"
uv run --no-sync python a6_live_check.py --log $LOG --example NeuralNetwork \
  --seed 42 --replay 4 2>&1 | grep -v nanobind | grep -v "leaked "
echo "CHECK_RC=${PIPESTATUS[0]}"
echo "########## A6 RAY DONE ##########"
