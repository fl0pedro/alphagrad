#!/usr/bin/env python3
"""Ground-truth probe: what does each target family RETURN, what does
scalar_loss_fn reduce it to, and does jacve(scalar) == jax.grad(scalar)?"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "512")
os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "8")
os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "8")
os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "16")

import jax
import jax.numpy as jnp
import numpy as np
from graphax import jacve

from alphagrad.approx.common.examples import (
    get_fn, get_args, infer_argnums, scalar_loss_fn,
)

EXAMPLES = [
    ("NeuralNetwork", None),
    ("VmappedNeuralNetwork", None),
    ("Encoder", None),
    ("EncoderDecoder", None),
    ("LIF_SNN", None),
    ("ADALIF_SNN_SEQ", None),
    ("LIF_SNN_SHD", None),
    ("Helmholtz", None),
]


def describe(name, dataset):
    print("=" * 70)
    print(f"### {name}  dataset={dataset}")
    try:
        fn = get_fn(name)
        xs = get_args(name, jax.random.PRNGKey(0), dataset=dataset)
        argnums = infer_argnums(name)
        out = fn(*xs)
        leaves = jax.tree_util.tree_leaves(out)
        print(f"  raw out type={type(out).__name__} n_leaves={len(leaves)} "
              f"shapes={[tuple(np.shape(l)) for l in leaves]}")
        print(f"  argnums={argnums}")
    except Exception as e:
        print(f"  SETUP FAILED {type(e).__name__}: {str(e)[:200]}")
        return
    try:
        loss = scalar_loss_fn(fn)
        v = loss(*xs)
        print(f"  scalar_loss_fn -> shape={np.shape(v)} value={float(v):.8g}")
    except Exception as e:
        print(f"  scalar_loss_fn FAILED {type(e).__name__}: {str(e)[:300]}")
        return
    # reference reductions
    try:
        first = leaves[0]
        print(f"  ref sum(first leaf)      = {float(jnp.sum(first)):.8g}")
        print(f"  ref mean(first leaf)     = {float(jnp.mean(first)):.8g}")
        if np.ndim(first) >= 2:
            print(f"  ref mean(sum(-1)) first  = "
                  f"{float(jnp.mean(jnp.sum(first, axis=-1))):.8g}")
    except Exception as e:
        print(f"  ref FAILED {e}")
    # jacve vs jax.grad on the scalar loss
    try:
        g_ref = jax.jit(jax.grad(loss, argnums=argnums))(*xs)
        j = jax.jit(jacve(loss, order="rev", argnums=argnums))(*xs)
        jl = jax.tree_util.tree_leaves(j)
        gl = jax.tree_util.tree_leaves(g_ref)
        print(f"  jacve leaf shapes = {[tuple(x.shape) for x in jl]}")
        print(f"  grad  leaf shapes = {[tuple(x.shape) for x in gl]}")
        worst = 0.0
        for a, b in zip(jl, gl):
            a2 = jnp.reshape(a, b.shape)
            d = float(jnp.max(jnp.abs(a2 - b)))
            s = float(jnp.max(jnp.abs(b))) or 1.0
            worst = max(worst, d / s)
        print(f"  JACVE vs JAX.GRAD  max rel err = {worst:.3e}")
    except Exception as e:
        print(f"  jacve/grad FAILED {type(e).__name__}: {str(e)[:300]}")


for n, d in EXAMPLES:
    describe(n, d)

# TransformerLM needs wikitext2; try it, tolerate failure.
describe("TransformerLM", "wikitext2")
