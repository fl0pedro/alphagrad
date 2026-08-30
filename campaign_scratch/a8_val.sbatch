#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_val_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
echo "########## jacobian===grad TABLE"
uv run --no-sync python -m pytest src/alphagrad/approx/tests/test_jacobian_equals_grad.py -q -s -p no:cacheprovider 2>&1 | grep -E "^\[|passed|failed|error"
echo "########## approx/tests (minus the 4 known-stale collectors)"
uv run --no-sync python -m pytest src/alphagrad/approx/tests -q -p no:cacheprovider \
  --ignore=src/alphagrad/approx/tests/test_agent.py \
  --ignore=src/alphagrad/approx/tests/test_loss.py \
  --ignore=src/alphagrad/approx/tests/test_no_approx_head.py \
  --ignore=src/alphagrad/approx/tests/test_simple_head.py 2>&1 | tail -25
echo "########## repo tests/"
uv run --no-sync python -m pytest tests -q -p no:cacheprovider -x --timeout=1200 2>&1 | tail -40
echo "########## ratio_gates.sh"
PY="uv run --no-sync python" bash tools/ratio_gates.sh 2>&1 | tail -25
echo "ratio_gates rc=$?"
echo "########## smoke.sh"
timeout 5400 bash tools/smoke.sh 2>&1 | tail -40
echo "smoke rc=$?"
echo "########## A8 VAL DONE"
