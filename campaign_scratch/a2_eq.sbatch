#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu2
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a2_eq_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a2_out
mkdir -p $R
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
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a2
COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --grad-window 0 --example Helmholtz --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"
B=/Users/assmuth/dsnn/.a2_base_src
rm -f $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl
echo "### base (pre-A2 snapshot) with EQ dump"
PYTHONPATH=$B ALPHAGRAD_EQ_DUMP=$R/eq_base timeout 3300 uv run --no-sync \
  python $B/alphagrad/approx/ppo.py $COMMON --name a2_eqbase > $R/eq_base.log 2>&1
echo "base rc=$?"
echo "### off (patched, --fidelity-weight absent, --cos-log-every absent)"
ALPHAGRAD_EQ_DUMP=$R/eq_off timeout 3300 uv run --no-sync \
  python src/alphagrad/approx/ppo.py $COMMON --name a2_eqoff > $R/eq_off.log 2>&1
echo "off rc=$?"
ls -la $R/eq_base.ep*.pkl $R/eq_off.ep*.pkl 2>&1 | head
uv run --no-sync python - <<'PY'
import pickle, glob, numpy as np
R = "/Users/assmuth/dsnn/alphagrad/a2_out"
eps = sorted(int(f.split(".ep")[1].split(".")[0])
             for f in glob.glob(R + "/eq_base.ep*.pkl"))
if not eps:
    print("EQ: no dumps produced"); raise SystemExit(1)
bad = 0; total = 0
def same(u, v):
    u, v = np.asarray(u), np.asarray(v)
    if u.shape != v.shape:
        return False
    try:
        return np.array_equal(u, v, equal_nan=True)
    except TypeError:
        return np.array_equal(u, v)
for ep in eps:
    a = pickle.load(open(f"{R}/eq_base.ep{ep}.pkl", "rb"))
    b = pickle.load(open(f"{R}/eq_off.ep{ep}.pkl", "rb"))
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        xa, xb = a[k], b[k]
        if k == "rewards":
            # THE ONE EXPECTED DIFFERENCE. A2 APPENDED reward slot 8
            # (`fidelity`), so the vector is (E, 9) against the base's (E, 8).
            # The claim is therefore precise: the 8 pre-existing columns are
            # BIT-IDENTICAL and the new one is identically 0.0 with the flag
            # off. Anything else is a regression.
            ra, rb = np.asarray(xa), np.asarray(xb)
            print(f"ep{ep} rewards base{ra.shape} patched{rb.shape}")
            n0 = ra.shape[-1]
            assert rb.shape[-1] == n0 + 1, (ra.shape, rb.shape)
            total += 2
            if not same(ra, rb[..., :n0]):
                print(f"ep{ep} rewards[:, :{n0}] DIFFER"); bad += 1
            # WHAT THE NEW COLUMN IS ALLOWED TO BE with the flag off.
            # `total_rewards_full` is the SUM over the episode, and slot 8 is
            # 0.0 on every real plan (not measured) but -1.0 -- its clip floor,
            # like slot 7's -- on every step the env SENTINELLED. So the claim
            # is not "identically zero"; it is "zero except where slot 7 is
            # also at its floor, and by the same amount". A sentinelled row is
            # recognisable exactly as ppo's `_is_degen` recognises it: the six
            # cost channels at SENTINEL_COST.
            sent = np.all(rb[..., :6] <= -1e10 * 0.99, axis=-1)
            new_col = rb[..., n0]
            if not np.all(new_col[~sent] == 0.0):
                print(f"ep{ep} slot 8 NON-ZERO ON A REAL PLAN: "
                      f"{new_col[~sent]}"); bad += 1
            if not np.all(new_col[sent] == rb[..., 7][sent]):
                print(f"ep{ep} slot 8 does not mirror slot 7 on the "
                      f"sentinelled rows: {new_col[sent]} vs "
                      f"{rb[..., 7][sent]}"); bad += 1
            print(f"ep{ep} slot8: real rows={new_col[~sent]} "
                  f"sentinel rows={new_col[sent]} (slot7 there="
                  f"{rb[..., 7][sent]})")
            continue
        if isinstance(xa, list):
            if len(xa) != len(xb):
                print(f"ep{ep} {k}: LEAF COUNT {len(xa)} vs {len(xb)}")
                bad += 1; continue
            for i, (u, v) in enumerate(zip(xa, xb)):
                total += 1
                if not same(u, v):
                    print(f"ep{ep} {k}[{i}] shape={np.asarray(u).shape} DIFFERS")
                    bad += 1
        else:
            total += 1
            if not same(xa, xb):
                print(f"ep{ep} {k}: DIFFERS {xa} vs {xb}"); bad += 1
    print(f"ep{ep}: cumulative {total} leaves compared")
print("EQ RESULT:", "BIT-IDENTICAL" if bad == 0 else f"{bad} DIFFERING LEAVES")
PY
echo "===== A2 EQ DONE ====="
