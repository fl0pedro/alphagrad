#!/usr/bin/env python3
"""Why does the seeded target have 36 vertices and not 13+2?

Dumps the eqn list of the plain vs seeded jaxpr so the extra vertices can be
attributed to specific primitives. The seeds are SUPPOSED to be two eliminable
nodes (one tangent injection, one adjoint contraction); anything beyond that is
scaffolding that `seed_loss_fn` builds out of ordinary JAX primitives.
"""
from __future__ import annotations

import collections

import jax


def dump(example="VmappedNeuralNetwork", seeded=False):
    from alphagrad.approx.common import get_args, get_fn, grad_target_setup

    class _A:
        measure_grad = seeded
        seed_vertices = seeded

    base_fn = get_fn(example)
    xs0 = get_args(example, jax.random.PRNGKey(0), dataset="mnist")
    fn, xs, argnums = grad_target_setup(_A(), base_fn, xs0, example)
    cj = jax.make_jaxpr(fn)(*xs)
    eqns = cj.jaxpr.eqns

    label = "SEEDED" if seeded else "PLAIN"
    print(f"\n=== {label}: {len(eqns)} eqns, argnums={argnums} ===")
    for i, e in enumerate(eqns, start=1):
        invars = " ".join(
            f"{getattr(v, 'aval', v)}" for v in e.invars)[:70]
        out = getattr(e.outvars[0], "aval", "?")
        print(f"  v{i:<3} {str(e.primitive.name):<22} -> {out}   [{invars}]")
    counts = collections.Counter(e.primitive.name for e in eqns)
    print(f"  primitive histogram: {dict(counts.most_common())}")
    return [e.primitive.name for e in eqns]


def main():
    plain = dump(seeded=False)
    seeded = dump(seeded=True)
    cp, cs = collections.Counter(plain), collections.Counter(seeded)
    print("\n=== DELTA (seeded - plain) ===")
    for name in sorted(set(cp) | set(cs)):
        d = cs.get(name, 0) - cp.get(name, 0)
        if d:
            print(f"  {name:<24} {d:+d}")
    print(f"  TOTAL {len(seeded) - len(plain):+d}  "
          f"({len(plain)} -> {len(seeded)})")
    print("\nExpectation was +2 (one tangent seed vertex, one adjoint "
          "contraction). Anything above that is scaffolding emitted as "
          "ordinary primitives.")


if __name__ == "__main__":
    main()
