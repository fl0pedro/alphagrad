#!/usr/bin/env python3
"""EXPLICIT BEFORE/AFTER for the unconditional scalar loss.

BEFORE = the pre-patch behaviour, reconstructed inline:
  * without --measure-grad the traced target was the RAW example;
  * scalar_loss_fn's reduction was inferred from ndim.
AFTER = what the repo does now.

Prints, per target: traced eqn count, valid-vertex count, output aval
(Jacobian rows vs scalar), and the loss value. Also re-checks A4's claim that
seed_loss_fn and scalar_loss_fn give the SAME vertex count on TransformerLM.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "4096")

from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common.examples import (
    get_args, get_fn, grad_target_setup, infer_argnums, scalar_loss_fn,
    seed_loss_fn,
)


def old_scalar_loss_fn(fn):
    """The pre-patch reduction: inferred from ndim."""
    def _loss(*a):
        out = fn(*a)
        if jnp.ndim(out) >= 2:
            return jnp.mean(jnp.sum(out, axis=-1))
        return jnp.mean(out)
    return _loss


def describe(tag, fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    n_eqns = len(cj.jaxpr.eqns)
    outs = [tuple(a.shape) for a in cj.out_avals]
    try:
        val = fn(*xs)
        leaves = jax.tree_util.tree_leaves(val)
        v = f"{float(leaves[0]):.8g}" if np.ndim(leaves[0]) == 0 else "(non-scalar)"
    except Exception as e:
        v = f"ERR {type(e).__name__}"
    print(f"    {tag:<34} eqns={n_eqns:<5} out_avals={outs}  value={v}")
    return n_eqns


def run(example, dataset=None):
    print("=" * 78)
    print(f"### {example}  dataset={dataset}")
    base = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0), dataset=dataset)
    argnums = infer_argnums(example)
    print(f"  argnums={argnums}")

    print("  BEFORE (pre-patch):")
    #   flag OFF -> raw example (a JACOBIAN target)
    n_off_before = describe("no --measure-grad: RAW example", base, xs)
    #   flag ON  -> old ndim-inferred scalar loss
    n_on_before = describe("--measure-grad: old scalar_loss",
                           old_scalar_loss_fn(base), xs)

    print("  AFTER (this patch):")
    t_off, xs_off, _ = grad_target_setup(
        NS(measure_grad=False, seed_vertices=False), base, xs, example)
    n_off_after = describe("no --measure-grad: scalar loss", t_off, xs_off)
    t_on, xs_on, _ = grad_target_setup(
        NS(measure_grad=True, seed_vertices=False), base, xs, example)
    n_on_after = describe("--measure-grad: scalar loss", t_on, xs_on)

    print(f"  DELTA eqns  flag-off: {n_off_before} -> {n_off_after} "
          f"({n_off_after - n_off_before:+d})   "
          f"flag-on: {n_on_before} -> {n_on_after} "
          f"({n_on_after - n_on_before:+d})")

    # A4 re-check: seeded vs scalar vertex count.
    ts, xss, ans = grad_target_setup(
        NS(measure_grad=True, seed_vertices=True), base, xs, example)
    n_seed = len(jax.make_jaxpr(ts)(*xss).jaxpr.eqns)
    print(f"  A4 re-check: scalar_loss eqns={n_on_after}  "
          f"seed_loss eqns={n_seed}  "
          f"({'SAME' if n_seed == n_on_after else 'DIFFERENT'})")

    # And the values they differ by: seed_loss drops the 1/N and perturbs.
    try:
        v_scalar = float(t_on(*xs_on))
        v_seed = float(ts(*xss))
        print(f"  A4 re-check: scalar_loss value={v_scalar:.8g}  "
              f"seed_loss value={v_seed:.8g}  ratio={v_seed / v_scalar:.6g}")
    except Exception as e:
        print(f"  A4 value compare failed: {type(e).__name__}: {e}")


run("NeuralNetwork")
run("VmappedNeuralNetwork")
run("TransformerLM", "wikitext2")
# The already-0-d targets: the patch must leave their graphs untouched.
run("LIF_SNN_SHD")
run("ADALIF_SNN_SEQ")
run("Helmholtz")
