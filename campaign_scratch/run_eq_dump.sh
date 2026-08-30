#!/bin/bash
# Flag-off inertness: the gradient-cosine swap must not move a DEFAULT run.
# Arm A = HEAD's env.py/ppo.py (their sparsity work, without my patch).
# Arm B = the working tree (their sparsity work + my patch).
# Both run the canonical smoke config with ALPHAGRAD_QUALITY_METRIC unset, so
# `auto` resolves to loss_drop and my cosine changes must be inert.
# NeuralNetwork, NOT Helmholtz: landscape_map/_callback on Helmholtz has a
# pre-existing TypeError (ndim=1) unrelated to this work.
set -uo pipefail
cd ~/dsnn/alphagrad
export PATH="$HOME/.local/bin:$PATH"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax_gates"
unset ALPHAGRAD_QUALITY_METRIC
export ALPHAGRAD_POLICY=palimpsa
export ALPHAGRAD_INCREMENTAL_TOKENS=1
export ALPHAGRAD_APPEND_ONLY_JAXPR=1

COMMON="--variant full --face-actions --unified-face-head --live-faces \
--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode \
--grad-window 0 --dataset none \
--cmp-type flops --mem-type peak_memory --terminal-rewards-only \
--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 \
--advantage-norm popart --episodes 2 --seed 42 --num-envs 2 --minibatches 1 \
--vocab-size 512 --wandb disabled"

R=$HOME/dsnn/qb_eq; BASE=/tmp/qb_eqbase
rm -rf "$R" "$BASE"; mkdir -p "$R" "$BASE"
cp -r src "$BASE/"
git show HEAD:src/alphagrad/approx/env.py > "$BASE/src/alphagrad/approx/env.py"
git show HEAD:src/alphagrad/approx/ppo.py > "$BASE/src/alphagrad/approx/ppo.py"
echo "[eq] arm A = HEAD blobs, arm B = working tree"

ALPHAGRAD_EQ_DUMP="$R/eq_base" PYTHONPATH="$BASE/src" \
  uv run --no-sync python "$BASE/src/alphagrad/approx/ppo.py" $COMMON \
  --example NeuralNetwork --name eq_base > "$R/base.log" 2>&1
echo "[eq] arm A rc=$?"

ALPHAGRAD_EQ_DUMP="$R/eq_post" PYTHONPATH="$PWD/src" \
  uv run --no-sync python src/alphagrad/approx/ppo.py $COMMON \
  --example NeuralNetwork --name eq_post > "$R/post.log" 2>&1
echo "[eq] arm B rc=$?"

ls -la "$R"/*.pkl 2>/dev/null | head
uv run --no-sync python compare_eq.py "$R/eq_base" "$R/eq_post"
echo "EQ_RC=$?"
