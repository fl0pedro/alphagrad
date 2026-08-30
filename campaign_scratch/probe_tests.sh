#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_probe
cd /Users/assmuth/dsnn/alphagrad
echo '=== feature_probe_test ==='
uv run --no-sync python -u -m pytest tests/feature_probe_test.py -q 2>&1 | tail -20
echo '=== policy_regression_gate ==='
uv run --no-sync python -u tests/policy_regression_gate.py 2>&1 | tail -40
