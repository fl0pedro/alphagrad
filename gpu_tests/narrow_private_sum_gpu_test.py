# The barrier at a narrow private sum makes the sum read the stored bf16 edge (graphax 01d5393, dsnn-dfw.273):
# the values on the GPU, on an edge of 65536 elements. On XLA:CPU the rounding follows the size of the array,
# with or without the barrier (jaxlib 0.10.2: rounded up to 2048 elements, unrounded from 43690 on), so the
# check runs where the plans run. An sbatch job on a GPU node runs it:
#   JAX_PLATFORMS=cuda python -m pytest -q -s gpu_tests/narrow_private_sum_gpu_test.py
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from graphax.sparse.ops.matmul import _emit_einsum  # noqa: E402

if jax.default_backend() != "gpu":
    raise RuntimeError(f"this check runs on a GPU, the backend is {jax.default_backend()}")


def _sum_of_quantized(p):
    # The Quant cast of an f32 edge, then a contraction whose batch label only it carries.
    a = p.astype(jnp.bfloat16)
    w = jnp.ones((p.shape[1],), jnp.bfloat16)
    return _emit_einsum(a, [0, 1], w, [1], [1]).materialize()


def test_the_private_sum_of_a_large_narrow_edge_is_of_the_rounded_values_on_the_gpu():
    # 1 + 3 * 2^-9 rounds to 1 + 2^-7 in bf16. Against -1 the sum of the rounded
    # values is 2^-7, and of the unrounded ones 3 * 2^-9; both are exact in bf16.
    n = 65536 // 2
    p = jnp.asarray([[1 + 3 * 2.0 ** -9] * n, [-1.0] * n], jnp.float32)
    got = jax.jit(_sum_of_quantized)(p)
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    assert bool(jnp.all(got == 2.0 ** -7)), sorted({float(x) for x in got})
    print(f"[narrow-private-sum-gpu] {jax.devices()[0].device_kind}: the sum of {2 * n} bf16 elements "
          f"is {float(got[0])} (rounded 2^-7 = {2.0 ** -7}, unrounded 3 * 2^-9 = {3 * 2.0 ** -9})")
