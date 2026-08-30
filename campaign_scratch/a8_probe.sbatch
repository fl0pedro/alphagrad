#!/bin/bash -l
#SBATCH -p pgi15-cpu
#SBATCH -w pgi15-cpu1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -o /Users/assmuth/dsnn/alphagrad/a8_probe3_%j.log
export PATH=$HOME/.local/bin:$PATH
cd /Users/assmuth/dsnn/alphagrad
export JAX_PLATFORMS=cpu
export ALPHAGRAD_SKIP_COUNT_OPS=1
export ALPHAGRAD_SKIP_COST_ANALYSIS=1
export GRAPHAX_ALLOW_PARTIAL_ORDER=1
export JAX_COMPILATION_CACHE_DIR=/Users/assmuth/dsnn/.jc_a8
TAG=${A8_TAG:-after2}
echo "##### MODE=graph TAG=$TAG"
timeout 4800 uv run --no-sync python a8_probe_registry.py graph 2>&1 | tail -80
echo "##### MODE=grad TAG=$TAG"
timeout 4800 uv run --no-sync python a8_probe_registry.py grad 2>&1 | tail -80
echo "##### A8 PROBE DONE"
echo "##### FLAG CHECK"
uv run --no-sync python - <<PY
import os
os.environ.setdefault("JAX_PLATFORMS","cpu")
from types import SimpleNamespace as NS
from alphagrad.approx.common.examples import (warn_measure_grad_deprecated,
    grad_target_setup, get_fn, get_args, has_scalar_loss)
import jax
print("warned:", warn_measure_grad_deprecated(NS(measure_grad=True)))
print("warned again:", warn_measure_grad_deprecated(NS(measure_grad=True)))
print("has_scalar_loss NN/Helmholtz:", has_scalar_loss("NeuralNetwork"), has_scalar_loss("Helmholtz"))
xs = get_args("NeuralNetwork", jax.random.PRNGKey(0))
for f in (False, True):
    t, x2, an = grad_target_setup(NS(measure_grad=f, seed_vertices=False), get_fn("NeuralNetwork"), xs, "NeuralNetwork")
    print("flag", f, "jaxpr eq:", len(jax.make_jaxpr(t)(*x2).jaxpr.eqns), "out", [a.shape for a in jax.make_jaxpr(t)(*x2).out_avals])
t, x2, an = grad_target_setup(NS(measure_grad=False, seed_vertices=True), get_fn("NeuralNetwork"), xs, "NeuralNetwork")
print("seed-vertices without measure-grad OK, eqns:", len(jax.make_jaxpr(t)(*x2).jaxpr.eqns))
PY
echo "##### ARGPARSE CHECK"
uv run --no-sync python src/alphagrad/approx/tools/landscape_map.py --no-measure-grad 2>&1 | tail -3
echo "##### A8 PROBE2 DONE"
