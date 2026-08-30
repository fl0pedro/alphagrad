#!/bin/bash -l
#SBATCH -p pgi15-cpu

#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a6_eq_%j.log
# A6 FLAG-OFF INERTNESS. The plan log must not move a DEFAULT run.
#   arm A = HEAD's blobs for the four files A6 touches (plus no plan_log.py)
#   arm B = the working tree, --plan-log ABSENT
# Same protocol as 02e2679 / 949f1af. NeuralNetwork, NOT Helmholtz: Helmholtz
# has a pre-existing scalar-loss-retarget TypeError unrelated to this work.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a6_out
mkdir -p $R
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

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--grad-window 0 --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled --example NeuralNetwork \
--latency-inner-reps 50"

# ---- arm A: working tree with the four touched files reverted to HEAD -----
B=/Users/assmuth/dsnn/.a6_base_src
rm -rf $B; mkdir -p $B
cp -r src/. $B/
for f in approx/env.py approx/ppo.py approx/common/measure_pool.py \
         approx/cpu_approx_actors.py; do
  git show HEAD:src/alphagrad/$f > $B/alphagrad/$f || exit 2
done
rm -f $B/alphagrad/approx/common/plan_log.py
find $B -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
echo "### arm A = HEAD blobs for the 4 touched files (+ no plan_log.py)"
rm -f $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl

PYTHONPATH=$B ALPHAGRAD_EQ_DUMP=$R/eq_base timeout 3300 uv run --no-sync \
  python $B/alphagrad/approx/ppo.py $COMMON --name a6_eqbase \
  > $R/eq_base.log 2>&1
echo "arm A rc=$?"

echo "### arm B = working tree, --plan-log ABSENT"
ALPHAGRAD_EQ_DUMP=$R/eq_off timeout 3300 uv run --no-sync \
  python src/alphagrad/approx/ppo.py $COMMON --name a6_eqoff \
  > $R/eq_off.log 2>&1
echo "arm B rc=$?"

grep -c "plan-log" $R/eq_off.log || true
ls -la $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl 2>&1 | head

uv run --no-sync python - <<'PY'
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
echo "===== A6 EQ DONE ====="
