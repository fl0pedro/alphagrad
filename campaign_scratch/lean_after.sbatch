#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_after_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

echo "===== AFTER: import smoke ====="
JAX_PLATFORMS=cpu uv run --no-sync python -c "
import alphagrad.approx.ppo as P
import alphagrad.approx.az_gumbel  # noqa  -- import only, no run
" 2>&1 | tail -5
echo "ppo import rc=$?"
JAX_PLATFORMS=cpu uv run --no-sync python -c "import alphagrad.approx.ppo as P; print('ppo import OK')"

echo
echo "===== AFTER: gradient-to-palimpsa probe ====="
JAX_PLATFORMS=cpu uv run --no-sync python lean_probe.py
echo "probe rc=$?"

echo
echo "===== AFTER: the gradient PROOF as a test ====="
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q tests/palimpsa_base_grad_test.py
echo "grad test rc=$?"

echo
echo "===== AFTER: DAG-agnosticism audit ====="
JAX_PLATFORMS=cpu uv run --no-sync python lean_audit.py
echo "audit rc=$?"

echo
echo "===== AFTER: the gate against the OLD golden (MUST FAIL -- semantics changed) ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py 2>&1 | tail -30
echo "old-golden gate rc=$? (a FAIL here is the expected, honest outcome)"

echo
echo "===== AFTER: RE-DERIVE the golden ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py --record 2>&1 | tail -20
echo "record rc=$?"

echo
echo "===== AFTER: the gate against the RE-DERIVED golden (must PASS) ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py 2>&1 | tail -10
echo "new-golden gate rc=$?"

echo "===== LEAN AFTER DONE ====="
