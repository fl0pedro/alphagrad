#!/usr/bin/env python3
"""Do the INTERMEDIATE Jacobians stay sparse under seed-vertex grad targets?

The seeded target (`seed_loss_fn` + graphax.seed_vertices) keeps the action
space on the Jacobian graph but makes the measured object the gradient. The
risk is that an arbitrary elimination order defeats that: if the seed is
applied late, the intermediates are full Jacobians and we are back to paying
Jacobian cost. This samples RANDOM orders x RANDOM approximations and reports
how big the intermediates actually get.

`max_io_sum` (graphax's `aux["mem"]`) is exactly the metric: the sum over
Jacobian accumulations of max(in_size, out_size, edge_out_size) * itemsize —
i.e. how large the intermediate edges become. Comparing the seeded graph to
the plain Jacobian graph on the SAME kind of random plans tells us whether
sparsity survives.

Also checks CORRECTNESS: the seeded elimination must reproduce jax.grad.
"""
from __future__ import annotations

import random

import jax
import jax.numpy as jnp
import numpy as np

N_PLANS = 6
SEED = 0


def _rand_specs(rng, n_vertices, max_rules, p_approx=0.5):
    """Random approximation rows in the env's wire format."""
    from alphagrad.approx.env import COMPRESS_SENTINEL, QUANT_SENTINEL
    specs = []
    for _ in range(n_vertices):
        rows = []
        for _ in range(max_rules):
            r = rng.random()
            if r > p_approx:
                rows.append([-1, -1, 0])
            elif r < p_approx / 3:
                rows.append([0, 0, rng.choice([2, 4])])          # DIAG
            elif r < 2 * p_approx / 3:
                rows.append([COMPRESS_SENTINEL, 0, 0])           # COMPRESS
            else:
                rows.append([QUANT_SENTINEL, rng.randrange(0, 6), 0])
        specs.append(rows)
    return specs


def run(example="VmappedNeuralNetwork", seeded=True):
    from alphagrad.approx.common import (
        get_args, get_fn, grad_target_setup, infer_argnums)
    from graphax.core import vertex_elimination_jaxpr

    class _A:
        measure_grad = seeded
        seed_vertices = seeded

    base_fn = get_fn(example)
    xs0 = get_args(example, jax.random.PRNGKey(0), dataset="mnist")
    fn, xs, argnums = grad_target_setup(_A(), base_fn, xs0, example)
    cj = jax.make_jaxpr(fn)(*xs)
    n_eqns = len(cj.jaxpr.eqns)

    label = "SEEDED (grad, seed vertices)" if seeded else "PLAIN (Jacobian)"
    print(f"\n=== {label} ===")
    print(f"  vertices={n_eqns}  argnums={argnums}  n_args={len(xs)}")

    # Correctness: the un-approximated elimination must equal jax.grad.
    if seeded:
        try:
            from graphax.core import jacve
            order = list(range(1, n_eqns + 1))
            g_ve = jax.jit(jacve(fn, order, argnums=argnums))(*xs)
            leaves = jax.tree.leaves(g_ve)
            nrm = float(sum(jnp.sum(jnp.abs(l)) for l in leaves))
            print(f"  ||exact grad via jacve||_1 = {nrm:.6e}"
                  f"   {'OK' if nrm > 0 else '<-- ZERO! graph mismatch'}")
        except Exception as exc:
            print(f"  jacve check FAILED: {type(exc).__name__}: {str(exc)[:160]}")

    from alphagrad.approx.common.masks import make_live_masked_hook
    from graphax.sparse.micro_actions import Compress, Diag, Quant, QUANT_DTYPES

    def _rand_rules(rng):
        """A random approximation for one vertex, wrapped in the live-masked
        hook so illegal rules are skipped per-face instead of raising —
        exactly how the env applies policy actions."""
        r = rng.random()
        if r < 0.25:
            return ()
        if r < 0.50:
            return (Diag(i=0, j=1, factor=rng.choice([2, 4])),)
        if r < 0.75:
            return (Compress(axes=(0,), kind="mean"),)
        return (Quant(dtype=QUANT_DTYPES[rng.randrange(0, 6)]),)

    rng = random.Random(SEED)
    rows = []
    for k in range(N_PLANS):
        order = list(range(1, n_eqns + 1))
        rng.shuffle(order)
        # RANDOM APPROXIMATIONS on top of the random order (user request):
        # every vertex gets an independently sampled rule set.
        transforms = []
        n_appr = 0
        for v in order:
            rules = _rand_rules(rng)
            if rules:
                n_appr += 1
                transforms.append((int(v), (make_live_masked_hook(rules),)))
        try:
            _, aux = vertex_elimination_jaxpr(
                cj.jaxpr, order, cj.literals, *xs,
                argnums=argnums, count_ops=True, sparse_representation=True,
                transforms=transforms)
            ops = float(aux["adds"] + aux["muls"] + aux["fmas"])
            mem = float(aux["mem"])
            rows.append((ops, mem))
            print(f"  plan {k:2d}: ops={ops:.3e}  max_io={mem:.3e}"
                  f"  approx_vertices={n_appr}/{len(order)}")
        except Exception as exc:
            print(f"  plan {k:2d}: FAILED {type(exc).__name__}: {str(exc)[:110]}")
    if rows:
        o = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows])
        print(f"  --> ops  median={np.median(o):.3e}  max={o.max():.3e}")
        print(f"  --> max_io median={np.median(m):.3e}  max={m.max():.3e}"
              f"   (this is intermediate-Jacobian size: LOW == stayed sparse)")
    return rows


def main():
    plain = run(seeded=False)
    seeded = run(seeded=True)
    if plain and seeded:
        mp = np.median([r[1] for r in plain])
        ms = np.median([r[1] for r in seeded])
        print("\n" + "=" * 74)
        print(f"median max_io  plain(Jacobian)={mp:.3e}   seeded(grad)={ms:.3e}")
        print(f"ratio seeded/plain = {ms/max(mp,1):.3f}"
              f"   (<1 => the seed keeps intermediates smaller)")
        print("=" * 74)


if __name__ == "__main__":
    main()
