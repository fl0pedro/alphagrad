#!/bin/bash -l
#SBATCH -p pgi15
#SBATCH -w pgi15-gpu18
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH -c 16
#SBATCH -t 2:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/lean_gate_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_EXTEND_CHUNK=256
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cublas_fallback=true"
unset ALPHAGRAD_MAX_TOKENS

echo "===== the gradient PROOF as a test (re-run; the scatter-purity check was fixed) ====="
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q tests/palimpsa_base_grad_test.py 2>&1 | tail -6
echo "grad test rc=$?"

echo
echo "===== gate against the OLD golden -- MUST FAIL, the semantics changed ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py 2>&1 | grep -E "^\[gate\]|MISMATCH|^  " | head -40
echo

echo "===== RE-DERIVE the golden ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py --record 2>&1 | grep -E "^\[gate\]|recorded|Traceback|Error" | head -20
echo "record rc=$?"

echo
echo "===== gate against the RE-DERIVED golden -- must PASS ====="
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py 2>&1 | grep -E "^\[gate\]|MISMATCH" | head -20
echo "new-golden gate rc=$?"

echo
echo "===== BEFORE-tree parameter/leaf audit (git archive HEAD; NO stash, NO .git mutation) ====="
PRE=/Users/assmuth/dsnn/lean_pretree
rm -rf $PRE && mkdir -p $PRE
git archive HEAD | tar -x -C $PRE
cp lean_audit.py $PRE/lean_audit_pre.py
( cd $PRE && PYTHONPATH=$PRE/src JAX_PLATFORMS=cpu uv run --no-sync --project /Users/assmuth/dsnn/alphagrad python lean_audit_pre.py 2>&1 | grep -E "^\[audit\]|\*\*\*" )
echo "pre-tree audit rc=$?"

echo "===== LEAN GATE DONE ====="
