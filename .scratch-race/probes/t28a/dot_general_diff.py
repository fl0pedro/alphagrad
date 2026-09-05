"""Differential check of the planner's EMISSION (ticket dsnn-3qm.66, lane A).

Same plan, same labels, same output layout; only the emitted primitive
changes. Compares, per random structured operand pair:

    GRAPHAX_PLANNER_DOT_GENERAL=0   jnp.einsum with integer labels (the old one)
    GRAPHAX_PLANNER_DOT_GENERAL=1   lax.dot_general with dimension numbers

against each other and against the dense oracle. Runs the same families the
lattice property suite draws, plus the demand-emit variant and an all-bf16
(Quant) variant.

Quick script: CPU, float64 for the oracle families, under a minute.

    JAX_PLATFORMS=cpu python dot_general_diff.py [n_cases]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.environ["GX_TESTS"])
from lattice_property_test import _mk, _dense_mat            # noqa: E402

from graphax.sparse.elemental.dispatch import set_approx_active  # noqa: E402
from graphax.sparse.ops.matmul import matmul                     # noqa: E402
import graphax.sparse.lower.matmul as LOW                        # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 20260802


def draw(i):
    rng = np.random.default_rng(SEED + i)
    Lc = int(rng.choice([4, 6, 12]))
    Lo = int(rng.choice([3, 4, 8]))
    Lp = int(rng.choice([2, 5, 6]))

    def specs(free_l, con_l, lhs_side):
        style = str(rng.choice(["dense", "implicit", "pair"]))
        if style == "pair":
            divs = [m for m in (1, 2, 3, 4, 6, 12)
                    if free_l % m == 0 and con_l % m == 0]
            meta = int(rng.choice(divs))
            if lhs_side:
                return [("pair", free_l, con_l, meta)], []
            return [("pair", con_l, free_l, meta)], []
        con = ("implicit", con_l) if style == "implicit" else ("dense", con_l)
        if lhs_side:
            return [("dense", free_l)], [con]
        return [con], [("dense", free_l)]

    lo, lp = specs(Lo, Lc, True)
    ro, rp = specs(Lp, Lc, False)
    return _mk(rng, lo, lp), _mk(rng, ro, rp)


def _narrow(t):
    from graphax.sparse.tensor import SparseTensor
    return SparseTensor(t.out_dims, t.primal_dims,
                        t.val.astype(jnp.bfloat16),
                        check_consistency=False)


def run(lhs, rhs, emit, demand):
    os.environ["GRAPHAX_PLANNER_DOT_GENERAL"] = emit
    os.environ["GRAPHAX_DEMAND_EMIT"] = "1" if demand else "0"
    tok = LOW._DEMAND_DENSE.set(bool(demand))
    set_approx_active(True)
    try:
        return _dense_mat(matmul(lhs, rhs))
    finally:
        set_approx_active(False)
        LOW._DEMAND_DENSE.reset(tok)


def main():
    prev64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    fails, n_ok, bitsame = [], 0, 0
    for demand in (False, True):
        for i in range(N):
            lhs, rhs = draw(i)
            want = _dense_mat(lhs) @ _dense_mat(rhs)
            try:
                a = run(lhs, rhs, "0", demand)
                b = run(lhs, rhs, "1", demand)
            except Exception as e:
                fails.append((demand, i, f"RAISE {type(e).__name__}: {e!s:.120}"))
                continue
            if a.shape != b.shape:
                fails.append((demand, i, f"SHAPE einsum {a.shape} dot {b.shape}"))
                continue
            if b.shape != want.shape:
                fails.append((demand, i, f"SHAPE dot {b.shape} oracle {want.shape}"))
                continue
            d_ab = float(np.abs(a - b).max())
            d_bo = float(np.abs(b - want).max())
            if d_ab > 1e-9 or d_bo > 1e-9:
                fails.append((demand, i, f"VALUE emit-diff {d_ab:.2e} "
                                         f"oracle-diff {d_bo:.2e}"))
                continue
            n_ok += 1
            bitsame += int(np.array_equal(a, b))
    jax.config.update("jax_enable_x64", prev64)

    # bf16 family (the Quant class): both operands narrow. The two emissions
    # are NOT expected to be bit-equal here -- dot_general asks for f32
    # product/accumulate (the tiled rule), einsum keeps bf16 -- so this checks
    # against the f32 oracle, and the dot form must be no worse.
    q_fail = []
    for i in range(N):
        lhs, rhs = draw(i)
        if lhs.val is None or rhs.val is None:
            continue
        lb = _narrow(lhs)
        rb = _narrow(rhs)
        want = _dense_mat(lhs) @ _dense_mat(rhs)
        try:
            a = run(lb, rb, "0", False).astype(np.float64)
            b = run(lb, rb, "1", False).astype(np.float64)
        except Exception as e:
            q_fail.append((i, f"RAISE {type(e).__name__}: {e!s:.120}"))
            continue
        scale = max(1e-12, float(np.abs(want).max()))
        ea = float(np.abs(a - want).max()) / scale
        eb = float(np.abs(b - want).max()) / scale
        if b.shape != want.shape or eb > 2e-2 or eb > 4 * max(ea, 1e-6):
            q_fail.append((i, f"einsum relerr {ea:.2e} dot relerr {eb:.2e}"))

    print(f"[t28a] structured families: {n_ok} ok, {len(fails)} bad "
          f"(of {2 * N}); bit-identical emissions: {bitsame}/{n_ok}")
    for f in fails[:15]:
        print("   demand=%s case %d: %s" % f)
    print(f"[t28a] bf16 (Quant) family: {len(q_fail)} bad of {N}")
    for f in q_fail[:10]:
        print("   case %d: %s" % f)
    stats = {k: v for k, v in LOW.LOWER_STATS.items() if k.startswith("emit:")}
    print("[t28a] emission census:", stats)
    return 1 if (fails or q_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
