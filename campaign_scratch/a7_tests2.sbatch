#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a7_tests2_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a7
for t in tests/sparsity_channel_test.py tests/fidelity_channel_test.py tests/grad_coverage_test.py tests/test_reward_scaling.py tests/test_sparse_terminal.py tests/symlog_channels_test.py tests/degen_detection_test.py tests/static_objective_test.py tests/lagrangian_reward_test.py tests/quality_metrics_test.py tests/resource_truncation_test.py; do
  echo "### $t"
  uv run --no-sync python -m pytest $t -q --tb=short -p no:cacheprovider 2>&1 | tail -25
done
echo "### test_all_cost_channels (RED BEFORE A7 -- capturing the reason)"
uv run --no-sync python -m pytest tests/test_all_cost_channels.py -q --tb=short -p no:cacheprovider 2>&1 | tail -45
