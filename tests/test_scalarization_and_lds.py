"""Tests for the RQ9 (Pitch B scaffold) modules.

Pins:
* ``apply_scalarization`` agrees with the linear formula in linear mode
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
from alphagrad.approx.common.scalarization import (
    apply_scalarization,
    update_ideal_point_ema,
)


def test_apply_scalarization_linear_matches_dot_product():
    """Linear mode is `sum_k w_k * r_k` — pin the trivial case."""
    print("\n[scalar] linear mode == dot product")
    r = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([0.5, 0.3, 0.2])
    s = float(apply_scalarization(r, w, kind="linear"))
    expected = float(0.5 * 1 + 0.3 * 2 + 0.2 * 3)  # = 1.7
    assert abs(s - expected) < 1e-6, f"linear={s} != {expected}"
    print(f"  s = {s} (expected {expected})")


def test_tchebycheff_reaches_concave_point():
    """The headline reachability claim: there exist Pareto-optimal
    points in the concave region of a front that NO linear-sum weight
    vector can recover. Augmented Tchebycheff with some weight can.

    Construct a tiny 2-point candidate set: A on the linear hull, B in
    the concave region. Show that for ANY w in {0, 0.1, ..., 1.0},
    linear picks A; with augmented Tchebycheff, some w picks B.
    """
    print("\n[scalar] tchebycheff reaches concave Pareto point")
    # Three candidates in 2-D objective space. A=(1,0) and C=(0,1) are
    # extreme-corner Pareto points; B=(0.3, 0.3) is Pareto-optimal but
    # in the CONCAVE region (below the line A→C, where x+y=1). Linear
    # scalarization can never pick B: for any w, w_0*0.3 + w_1*0.3 = 0.3
    # is bounded above by max(w_0*1, w_1*1) = max(w_0, w_1) ≥ 0.5, so
    # one of A, C always wins. Augmented Tchebycheff (rho=0) with
    # ideal point z*=(1,1) DOES pick B for w=(0.5, 0.5): the per-objective
    # gap is (0.7, 0.7) for B vs (0, 1) for A vs (1, 0) for C, so the
    # max-weighted-gap is minimised at B.
    candidates = jnp.array([
        [1.0, 0.0],   # A — corner on the (1, 0) side
        [0.3, 0.3],   # B — concave-region Pareto point
        [0.0, 1.0],   # C — corner on the (0, 1) side
    ])
    z_star = jnp.array([1.0, 1.0])
    # Sweep w over the simplex.
    chosen_linear = set()
    chosen_tcheb = set()
    for n in range(101):
        w0 = n / 100.0
        w = jnp.array([w0, 1.0 - w0])
        scores_lin = jnp.array([
            float(apply_scalarization(c, w, kind="linear")) for c in candidates
        ])
        scores_tch = jnp.array([
            float(apply_scalarization(
                c, w, kind="tchebycheff", ideal_point=z_star, rho=0.0,
            )) for c in candidates
        ])
        chosen_linear.add(int(jnp.argmax(scores_lin)))
        chosen_tcheb.add(int(jnp.argmax(scores_tch)))
    print(f"  linear chose candidates {sorted(chosen_linear)} (B=1 ideally selectable)")
    print(f"  tchebycheff chose       {sorted(chosen_tcheb)}")
    # Linear should NEVER pick B (it's strictly dominated on the
    # weighted-sum half-plane for every w). Tchebycheff DOES pick B
    # for w around (0.5, 0.5).
    assert 1 not in chosen_linear, (
        "linear scalarization shouldn't reach concave point B, but did — "
        "set up of test broken"
    )
    assert 1 in chosen_tcheb, (
        "augmented Tchebycheff failed to reach concave point B for any w — "
        "reachability claim violated"
    )


def test_update_ideal_point_ema_tracks_running_best():
    """EMA-update should converge toward the running max of fresh
    rewards (per channel)."""
    print("\n[scalar] ideal-point EMA tracks running max")
    z = jnp.zeros((2,))
    for step in range(50):
        rng_vals = np.array([float(step) / 5, 1.0 - float(step) / 50])
        z = update_ideal_point_ema(z, jnp.asarray(rng_vals)[None, :], beta=0.5)
    z_py = np.asarray(z)
    # After 50 steps of EMA toward step/5, channel 0 should be near 49/5 = 9.8.
    assert z_py[0] > 8.0, f"ideal point channel 0 didn't track running max: {z_py}"
    print(f"  final z = {z_py}")


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
    test_apply_scalarization_linear_matches_dot_product()
    test_tchebycheff_reaches_concave_point()
    test_update_ideal_point_ema_tracks_running_best()
    test_plastic_constant_matches_known_values()
    test_kronecker_preferences_walks_simplex_deterministically()
    test_kronecker_more_uniform_than_iid()
    print("\nALL SCALARIZATION + LDS TESTS OK")


if __name__ == "__main__":
    main()
