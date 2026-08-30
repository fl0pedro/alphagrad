#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a5_validate_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
R=/Users/assmuth/dsnn/alphagrad/a5_out
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
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a5

echo "########## TESTS (serialized) ##########"
for T in kl_ref_trust_region_test credit_fix_test lagrangian_reward_test \
         static_objective_test gae_value_encoding_test; do
  echo "----- $T -----"
  timeout 2400 uv run --no-sync python -m pytest -q tests/$T.py \
    > $R/test_$T.log 2>&1
  echo "$T rc=$?"
  tail -4 $R/test_$T.log
done

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--measure-grad --grad-window 0 --example Helmholtz --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"

echo "########## ARM base (PRE-PATCH source snapshot) ##########"
B=/Users/assmuth/dsnn/.a5_base_src
PYTHONPATH=$B uv run --no-sync python -c \
  "import alphagrad.approx.ppo as m; print('BASE MODULE:', m.__file__)" \
  2>&1 | tail -2
PYTHONPATH=$B timeout 3300 uv run --no-sync python $B/alphagrad/approx/ppo.py \
  $COMMON --name a5_base > $R/smoke_base.log 2>&1
echo "base rc=$?"

echo "########## ARM off (patched, flags absent) ##########"
timeout 3300 uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --name a5_off > $R/smoke_off.log 2>&1
echo "off rc=$?"

echo "########## ARM on (patched, trust region + target-kl) ##########"
timeout 3300 uv run --no-sync python src/alphagrad/approx/ppo.py \
  $COMMON --kl-ref-weight 0.05 --kl-ref-target 0.002 --kl-ref-eta 1.0 \
  --target-kl 0.05 --name a5_on > $R/smoke_on.log 2>&1
echo "on rc=$?"

echo "########## BIT-IDENTITY: base vs off ##########"
for A in base off on; do
  grep -aoE "(ppo|value|total|ent)[a-z_]*[=:] *-?[0-9.]+(e[-+][0-9]+)?" \
    $R/smoke_$A.log | tail -60 > $R/nums_$A.txt
  grep -aoE "loss[^ ,]*=[-0-9.eE+]+|ent:[-0-9.eE+]+|kl[^ ,]*=[-0-9.eE+]+" \
    $R/smoke_$A.log >> $R/nums_$A.txt
  echo "$A numeric tokens: $(wc -l < $R/nums_$A.txt)"
done
if diff -q $R/nums_base.txt $R/nums_off.txt > /dev/null; then
  echo "BIT-IDENTITY base==off: PASS"
else
  echo "BIT-IDENTITY base==off: DIFF"
  diff $R/nums_base.txt $R/nums_off.txt | head -30
fi

echo "########## new keys / finiteness (arm on) ##########"
grep -a "\[kl-ref\]" $R/smoke_on.log | head -10
grep -a "\[target-kl\]" $R/smoke_on.log | head -3
grep -aE "nan|inf" $R/smoke_on.log | head -5
for A in base off on; do
  echo "--- tail $A ---"
  grep -naE "Traceback|Error:|error:" $R/smoke_$A.log | head -5
  tail -12 $R/smoke_$A.log
done
echo "===== A5 VALIDATE DONE ====="
