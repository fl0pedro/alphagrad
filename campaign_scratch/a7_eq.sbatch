#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_eq_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a7_out
mkdir -p $R
export JAX_PLATFORMS=cpu
unset ALPHAGRAD_MAX_TOKENS
unset ALPHAGRAD_SPARSITY
unset ALPHAGRAD_SPARSITY_WEIGHT
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
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7eq
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--grad-window 0 --example Helmholtz --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"
B=/Users/assmuth/dsnn/.a7_base_src
rm -f $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl
echo "### base (pre-A7 snapshot, HEAD src) with EQ dump"
PYTHONPATH=$B ALPHAGRAD_EQ_DUMP=$R/eq_base timeout 5000 uv run --no-sync \
  python $B/alphagrad/approx/ppo.py $COMMON --name a7_eqbase > $R/eq_base.log 2>&1
echo "base rc=$?"
echo "### off (patched, --sparsity-weight and --sparsity-log absent)"
ALPHAGRAD_EQ_DUMP=$R/eq_off timeout 5000 uv run --no-sync \
  python src/alphagrad/approx/ppo.py $COMMON --name a7_eqoff > $R/eq_off.log 2>&1
echo "off rc=$?"
ls -la $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl 2>&1 | head
uv run --no-sync python - <<"PY"
import pickle, glob, numpy as np
R = "/Users/assmuth/dsnn/alphagrad/a7_out"
eps = sorted(int(f.split(".ep")[1].split(".")[0])
             for f in glob.glob(R + "/eq_base.ep*.pkl"))
if not eps:
    print("EQ: no dumps produced"); raise SystemExit(1)
bad = 0; total = 0
def same(u, v):
    u, v = np.asarray(u), np.asarray(v)
    if u.shape != v.shape: return False
    try: return np.array_equal(u, v, equal_nan=True)
    except TypeError: return np.array_equal(u, v)
for ep in eps:
    a = pickle.load(open(f"{R}/eq_base.ep{ep}.pkl","rb"))
    b = pickle.load(open(f"{R}/eq_off.ep{ep}.pkl","rb"))
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        xa, xb = a[k], b[k]
        if k == "rewards":
            ra, rb = np.asarray(xa), np.asarray(xb)
            print(f"ep{ep} rewards base{ra.shape} patched{rb.shape}")
            n0 = ra.shape[-1]
            assert rb.shape[-1] == n0 + 2, (ra.shape, rb.shape)
            total += 3
            if not same(ra, rb[..., :n0]):
                print(f"ep{ep} rewards[:, :{n0}] DIFFER"); bad += 1
            sent = np.all(rb[..., :6] <= -1e10*0.99, axis=-1)
            res, spars = rb[..., n0], rb[..., n0+1]
            if not np.all(res == 0.0):
                print(f"ep{ep} RESERVED slot 9 non-zero: {res}"); bad += 1
            if not np.all(spars[~sent] == 0.0):
                print(f"ep{ep} slot 10 NON-ZERO ON A REAL PLAN: {spars[~sent]}"); bad += 1
            if not np.all(spars[sent] == rb[..., 7][sent]):
                print(f"ep{ep} slot 10 does not mirror slot 7 on sentinels: "
                      f"{spars[sent]} vs {rb[...,7][sent]}"); bad += 1
            print(f"ep{ep} slot9={res} slot10 real={spars[~sent]} sent={spars[sent]}")
            continue
        if isinstance(xa, list):
            if len(xa) != len(xb): print(f"ep{ep} {k} len differs"); bad += 1; continue
            for i,(u,v) in enumerate(zip(xa,xb)):
                total += 1
                if not same(u,v): print(f"ep{ep} {k}[{i}] DIFFERS"); bad += 1
        elif isinstance(xa, dict):
            for kk in xa:
                total += 1
                if not same(xa[kk], xb.get(kk)): print(f"ep{ep} {k}.{kk} DIFFERS"); bad += 1
        else:
            total += 1
            if not same(xa, xb): print(f"ep{ep} {k} DIFFERS"); bad += 1
print(f"EQ RESULT: {total-bad}/{total} leaves identical, {bad} differ")
raise SystemExit(1 if bad else 0)
PY
echo "eqcmp rc=$?"
