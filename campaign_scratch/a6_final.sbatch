#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=96G
#SBATCH -c 24
#SBATCH -t 6:00:00
#SBATCH -J a6_final
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_final_%j.log
# A6 FINAL GATE PASS, after the `_reward_slots` refactor (record the reward
# slots host-side so a recorded plan costs no device->host transfer).
# Everything env.py touches is re-gated:
#   1. tests/plan_log_test.py
#   2. tools/ratio_gates.sh          (6/6, a SKIP is a FAILURE)
#   3. tools/smoke.sh NeuralNetwork  (NOT Helmholtz -- 744fc3d TypeError)
#   4. --ray-measure 2 + --plan-log: THE PRODUCTION PATH (every wave arm runs
#      --ray-measure 3, so the recorder lives in the ACTOR processes)
#   5. ALPHAGRAD_EQ_DUMP flag-off bit identity against HEAD's blobs
# ONE JOB: three separate jobs of mine on this node were SIGKILLed the instant
# a fourth was allocated on it (62715/62719/62720/62721).
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a6_out
mkdir -p $R
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID
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

echo "########## 1. A6 unit tests ##########"
uv run --no-sync python -m pytest tests/plan_log_test.py -q --tb=short \
  -p no:cacheprovider 2>&1 | grep -v nanobind | grep -v "leaked "
echo "A6_TESTS_RC=${PIPESTATUS[0]}"

echo "########## 2. ratio gates ##########"
PY="uv run --no-sync python" tools/ratio_gates.sh
echo "RATIO_GATES_RC=$?"

echo "########## 3. smoke NeuralNetwork ##########"
SMOKE_OUT=$R/smoke2 tools/smoke.sh NeuralNetwork 2>&1 | grep -avE "it/s\]|s/it\]"
echo "SMOKE_RC=${PIPESTATUS[0]}"

echo "########## 4. --ray-measure 2 + --plan-log (production path) ##########"
LOG=$R/ray_plan_log.jsonl
rm -f $LOG
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
grep -aE "\[plan-log\]|\[cfg\] plan log" $R/ray.log | head -10
wc -l -c $LOG 2>&1
uv run --no-sync python - <<'PY' 2>&1 | grep -v nanobind | grep -v "leaked "
import collections, json
from alphagrad.approx.common import plan_log as plog
P = "/Users/assmuth/dsnn/alphagrad/a6_out/ray_plan_log.jsonl"
recs = plog.read_records(P)
print(f"records={len(recs)}")
if not recs:
    print("FAIL: the pooled run wrote NO plan records"); raise SystemExit(1)
print("by_actor=", dict(collections.Counter(r.get("actor") for r in recs)))
print("by_pid=", dict(collections.Counter(r.get("pid") for r in recs)))
print("by_episode=", dict(sorted(collections.Counter(
    r["episode"] for r in recs).items())))
n_actor = sum(1 for r in recs if r.get("actor") is not None)
print(f"records measured IN A MEASURE ACTOR: {n_actor}/{len(recs)}")
r = recs[0]
print("sample:", json.dumps({k: v for k, v in r.items()
                             if k not in ("faces", "rules", "coverage")})[:700])
print("POOL DRAIN:", "ok" if n_actor else "FAIL (no record came from an actor)")
raise SystemExit(0 if n_actor else 1)
PY
echo "POOL_CHECK_RC=${PIPESTATUS[0]}"

echo "########## 5. structure + REPLAY of the recorded losers ##########"
uv run --no-sync python a6_live_check.py --log $LOG --example NeuralNetwork \
  --seed 42 --replay 4 2>&1 | grep -v nanobind | grep -v "leaked "
echo "CHECK_RC=${PIPESTATUS[0]}"

echo "########## 6. EQ_DUMP flag-off bit identity ##########"
B=/Users/assmuth/dsnn/.a6_base_src
rm -rf $B; mkdir -p $B
cp -r src/. $B/
for f in approx/env.py approx/ppo.py approx/common/measure_pool.py \
         approx/cpu_approx_actors.py; do
  git show 62e65de:src/alphagrad/$f > $B/alphagrad/$f || exit 2
done
rm -f $B/alphagrad/approx/common/plan_log.py
find $B -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
EQC="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--grad-window 0 --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled --example NeuralNetwork \
--latency-inner-reps 50"
rm -f $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl
echo "### arm A = HEAD blobs for the 4 touched files (+ no plan_log.py)"
PYTHONPATH=$B ALPHAGRAD_EQ_DUMP=$R/eq_base uv run --no-sync \
  python $B/alphagrad/approx/ppo.py $EQC --name a6_eqbase > $R/eq_base.log 2>&1
echo "EQ_BASE_RC=$?"
echo "### arm B = working tree, --plan-log ABSENT"
ALPHAGRAD_EQ_DUMP=$R/eq_off uv run --no-sync \
  python src/alphagrad/approx/ppo.py $EQC --name a6_eqoff > $R/eq_off.log 2>&1
echo "EQ_OFF_RC=$?"
echo -n "plan-log lines in arm B's log (must be 0): "
grep -ac "plan-log" $R/eq_off.log
ls -la $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl 2>&1 | head

uv run --no-sync python - <<'PY' 2>&1 | grep -v nanobind | grep -v "leaked "
import pickle, glob, numpy as np
R = "/Users/assmuth/dsnn/alphagrad/a6_out"
eps = sorted(int(f.split(".ep")[1].split(".")[0])
             for f in glob.glob(R + "/eq_base.ep*.pkl"))
if not eps:
    print("EQ: no dumps produced"); raise SystemExit(1)
bad = total = 0
for ep in eps:
    a = pickle.load(open(f"{R}/eq_base.ep{ep}.pkl", "rb"))
    b = pickle.load(open(f"{R}/eq_off.ep{ep}.pkl", "rb"))
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        xa, xb = a[k], b[k]
        if isinstance(xa, list):
            if len(xa) != len(xb):
                print(f"ep{ep} {k}: LEAF COUNT {len(xa)} vs {len(xb)}")
                bad += 1
                continue
            for i, (u, v) in enumerate(zip(xa, xb)):
                u, v = np.asarray(u), np.asarray(v)
                try:
                    same = np.array_equal(u, v, equal_nan=True)
                except TypeError:
                    same = np.array_equal(u, v)
                if not same:
                    print(f"ep{ep} {k}[{i}] shape={u.shape} DIFFERS")
                    bad += 1
        else:
            u, v = np.asarray(xa), np.asarray(xb)
            try:
                same = np.array_equal(u, v, equal_nan=True)
            except TypeError:
                same = np.array_equal(u, v)
            if not same:
                print(f"ep{ep} {k}: DIFFERS {xa} vs {xb}")
                bad += 1
    n = sum(len(a[k]) if isinstance(a[k], list) else 1 for k in a)
    total += n
    print(f"ep{ep}: {n} leaves compared")
print(f"EQ RESULT: {total - bad}/{total} leaves bit-identical",
      "-> BIT-IDENTICAL" if bad == 0 else f"-> {bad} DIFFERING LEAVES")
PY
echo "########## A6 FINAL DONE ##########"
