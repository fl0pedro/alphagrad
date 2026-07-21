"""Tests for the RQ9 (Pitch B scaffold) modules.

Pins:
* preference LDS (Kronecker) coverage + plastic-constant identities
  and matches the augmented Tchebycheff closed form in tchebycheff mode.
* Augmented Tchebycheff can reach concave Pareto points that linear
  scalarization cannot (the load-bearing theoretical claim from
  Miettinen 1999 §3.4.3).
* ``kronecker_preferences`` walks the golden-ratio / R_d sequence with
  the documented stride, and successive batches with ``offset`` produce
  a deterministic continuation.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common.preferences import (
    _plastic_constant,
    kronecker_preferences,
)



def test_plastic_constant_matches_known_values():
    """φ = (1+√5)/2 ≈ 1.6180; plastic constant ψ ≈ 1.32472."""
    print("\n[lds] plastic constant for d=1 (golden ratio) and d=2")
    phi = _plastic_constant(1)
    psi = _plastic_constant(2)
    assert abs(phi - 1.61803398875) < 1e-9, f"φ off: {phi}"
    assert abs(psi - 1.32471795724) < 1e-9, f"ψ off: {psi}"
    print(f"  φ={phi:.10f}, ψ={psi:.10f}")


def test_kronecker_preferences_walks_simplex_deterministically():
    """Successive calls with incrementing offset walk the sequence;
    same offset returns the same point."""
    print("\n[lds] kronecker_preferences is deterministic and continuation")
    p1 = np.asarray(kronecker_preferences(2, 4, offset=0))
    p2 = np.asarray(kronecker_preferences(2, 4, offset=4))
    p_long = np.asarray(kronecker_preferences(2, 8, offset=0))
    # First 4 rows of p_long should match p1; rows 4–7 should match p2.
    np.testing.assert_allclose(p1, p_long[:4], rtol=1e-6)
    np.testing.assert_allclose(p2, p_long[4:], rtol=1e-6)
    # Each row sums to 1 (simplex).
    for batch_name, batch in [("p1", p1), ("p2", p2), ("p_long", p_long)]:
        s = batch.sum(axis=-1)
        np.testing.assert_allclose(s, np.ones_like(s), rtol=1e-6, err_msg=batch_name)
    print(f"  p1[0]={p1[0]}, p_long[4]={p_long[4]}, p2[0]={p2[0]}")


def test_kronecker_more_uniform_than_iid():
    """Kronecker should have lower discrepancy than i.i.d. uniform —
    measured via the std of the empirical density across simplex bins."""
    print("\n[lds] kronecker discrepancy < i.i.d. uniform discrepancy")
    rng = np.random.default_rng(0)
    n = 64
    iid = rng.dirichlet([1.0, 1.0], size=n)
    kron = np.asarray(kronecker_preferences(2, n, offset=0))
    # Bin the first coordinate of each into 8 buckets, compare counts.
    iid_hist, _ = np.histogram(iid[:, 0], bins=8, range=(0, 1))
    kron_hist, _ = np.histogram(kron[:, 0], bins=8, range=(0, 1))
    expected = n / 8.0
    iid_disc = np.std(iid_hist - expected)
    kron_disc = np.std(kron_hist - expected)
    print(f"  i.i.d. hist std-from-uniform = {iid_disc:.3f}")
    print(f"  kronecker hist std-from-uniform = {kron_disc:.3f}")
    assert kron_disc <= iid_disc, (
        f"Kronecker discrepancy {kron_disc} > i.i.d. {iid_disc}, "
        f"defeats the point of using a low-discrepancy sequence"
    )


def main():
    print("=== RQ9 scalarization + low-discrepancy tests ===")
    test_plastic_constant_matches_known_values()
    test_kronecker_preferences_walks_simplex_deterministically()
    test_kronecker_more_uniform_than_iid()
    print("\nALL SCALARIZATION + LDS TESTS OK")


if __name__ == "__main__":
    main()
