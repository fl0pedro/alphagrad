#!/usr/bin/env python3
"""Reward-geometry probe: is DESTROYING the computation profitable?

No training, no GPU, no RL — this evaluates the SCALARISATION ITSELF on
measured reward vectors taken from real runs. It answers, in closed form:

    "Under the current reward, does a policy that annihilates the Jacobian
     score better than one that computes it well?"

If yes, the collapse is a reward-design property and no amount of RL
tuning will fix it. Each variant below changes exactly ONE thing so the
responsible term is unambiguous.

Run:
    python tools/reward_geometry_probe.py
"""
from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX  # noqa: E402

LAT = REWARD_INDEX["latency_ns"]
MEM = REWARD_INDEX["peak_memory"]
COS = REWARD_INDEX["cosine_sim"]
FROB = REWARD_INDEX["frob_residual"]
MULS = REWARD_INDEX["muls_adds_fmas"]


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
# Plans observed in real runs (reward-vector conventions: costs NEGATED,
# cosine positive, frob NEGATED).
# ---------------------------------------------------------------------------
def plan(name, muls, lat_ns, peak_b, frob, cos=0.0):
    r = np.zeros(NUM_REWARDS, np.float64)
    r[MULS] = -muls
    r[LAT] = -lat_ns
    r[MEM] = -peak_b
    r[FROB] = -frob
    r[COS] = cos
    return name, r


PLANS = [
    # from job 56256 (3090) episode 1 — real work, quality still bad
    plan("real-work  (muls 6.4e11, frob 1.00)", 6.4e11, 1.31e7, 2.5e8, 1.00),
    # from job 56256 episode 20 — destroyed
    plan("destroyed  (muls 4.5e7,  frob 1.00)", 4.5e7, 3.19e5, 1.33e8, 1.00),
    # from job 56234 episode 58 — fully destroyed
    plan("annihilate (muls 1.6e4,  frob 1.00)", 1.6e4, 5.62e4, 1.32e8, 1.00),
    # hypothetical: same cost as real-work but a GOOD Jacobian
    plan("real+good  (muls 6.4e11, frob 0.20)", 6.4e11, 1.31e7, 2.5e8, 0.20),
    # hypothetical: the ideal — real work, exact Jacobian
    plan("real+exact (muls 6.4e11, frob 0.00)", 6.4e11, 1.31e7, 2.5e8, 0.00),
]


# ---------------------------------------------------------------------------
# Scalarisation variants — each differs from CURRENT in exactly one way.
# ---------------------------------------------------------------------------
def scal_current(r, w=(1.0, 1.0, 1.0)):
    """What the trainer does today: symlog every channel, weight, sum."""
    s = symlog(r)
    return w[0] * s[LAT] + w[1] * s[MEM] + w[2] * s[FROB]


def scal_lambda_frob(r, lf=30.0):
    """CURRENT but with --lambda-frob raised."""
    return scal_current(r, w=(1.0, 1.0, lf))


def scal_no_symlog_frob(r):
    """CURRENT but frob enters RAW (still bounded in [0,1])."""
    s = symlog(r)
    return s[LAT] + s[MEM] + r[FROB]


def scal_logfrob(r, eps=1e-3):
    """frob as -log(frob+eps): UNBOUNDED ABOVE as frob -> 0.

    reward contribution = -log(frob + eps):
        frob = 0    -> -log(eps)   = +6.9   (perfect Jacobian, big payoff)
        frob = 1    -> -log(1+eps) ~=  0    (destroyed, no payoff)
    So fidelity's dynamic range (~6.9) MATCHES the cost channels' (~6.4)
    instead of being capped at symlog(1)=0.693.
    """
    s = symlog(r)
    frob = -r[FROB]                      # stored negated -> positive residual
    return s[LAT] + s[MEM] - np.log(frob + eps)


def scal_mult_gate(r, tau=0.5, W=40.0):
    """Multiplicative gate: cost reward CONDITIONAL on fidelity."""
    fid = np.clip(1.0 + r[FROB], 0.0, 1.0)          # 1 - frob
    g = np.clip((fid - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)
    cost = -(symlog(-r[LAT]) + symlog(-r[MEM]))     # positive magnitudes
    return g * max(0.0, W - cost)


VARIANTS = [
    ("CURRENT (symlog all, 1/1/1)", scal_current),
    ("lambda_frob=30", scal_lambda_frob),
    ("frob raw (no symlog)", scal_no_symlog_frob),
    ("frob as -log(frob+eps)", scal_logfrob),
    ("multiplicative gate", scal_mult_gate),
]


def main():
    print("=" * 100)
    print("REWARD GEOMETRY PROBE — does destroying the computation pay?")
    print("=" * 100)
    print("\nPer-channel symlog contributions (CURRENT scalarisation):\n")
    print(f"{'plan':<42} {'sl(-lat)':>10} {'sl(-mem)':>10} "
          f"{'sl(-frob)':>10} {'TOTAL':>10}")
    for name, r in PLANS:
        s = symlog(r)
        print(f"{name:<42} {s[LAT]:>10.3f} {s[MEM]:>10.3f} "
              f"{s[FROB]:>10.3f} {scal_current(r):>10.3f}")

    print("\n" + "-" * 100)
    print("RANKING under each scalarisation (best first). "
          "A correct reward ranks 'real+exact' FIRST.")
    print("-" * 100)
    verdicts = []
    for vname, fn in VARIANTS:
        scored = sorted(((fn(r), n) for n, r in PLANS), reverse=True)
        best = scored[0][1]
        ok = best.startswith("real+exact")
        verdicts.append((vname, ok, best))
        print(f"\n{vname}")
        for val, n in scored:
            mark = "  <-- BEST" if n == best else ""
            print(f"    {val:>12.3f}  {n}{mark}")
        _verdict = "OK" if ok else "BROKEN — destruction outranks the ideal"
        print(f"    verdict: {_verdict}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for vname, ok, best in verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {vname:<32} best={best}")

    # The headline number: destruction gain vs the maximum quality can pay.
    _, r_real = PLANS[0]
    _, r_dead = PLANS[2]
    gain = scal_current(r_dead) - scal_current(r_real)
    frob_span = symlog(np.array([1.0]))[0]      # symlog(1) = max frob swing
    print(f"\n  destroying the computation buys : {gain:+.3f} symlog units")
    print(f"  perfect fidelity can ever pay    : {frob_span:+.3f} symlog units")
    if gain > frob_span:
        print(f"  => DESTRUCTION WINS by {gain / frob_span:.1f}x — "
              f"quality CANNOT outvote cost under CURRENT.")


if __name__ == "__main__":
    main()
