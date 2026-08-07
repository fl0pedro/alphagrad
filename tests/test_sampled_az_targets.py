"""Sampled-AZ target arithmetic (#93 / #95 / #76 closure).

These exercise the EXACT functions az_gumbel runs -- everything imports from
`alphagrad.approx.common.sampled_az`, the shared module the trainer calls,
not mirrored local copies (the historical drift mode of the old
test_gumbel_search.py helpers). No env, no GPU, no compile: the value net is
a stub dict of per-vertex Q.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common.sampled_az import (  # noqa: E402
    assert_same_depth, draw_weights, improved_policy, phase_plan,
    qscale_sigma, rho_from_beta_temp, weighted_q)


# ---------------------------------------------------------------- stub search
def _run_halving(logits, g, q_of, m):
    """gumbel_search's selection arithmetic on a stub value net: top-m by
    g + logits, then sequential halving over `phase_plan` with the weighted
    per-vertex Q -- the same imported helpers, the same keep/argsort lines.
    Returns ``(cands, winner)``."""
    order = np.argsort(-(logits + g))[:m]
    cands = [{"li": int(i), "qs": [], "g": float(g[i]),
              "logit": float(logits[i])} for i in order]
    plan = phase_plan(m)
    surv = list(cands)
    for _phase, (n_expect, n_draws, depth) in enumerate(plan):
        assert len(surv) == n_expect
        assert depth == 0          # (c): default mode never deepens
        for c in surv:
            for _ in range(n_draws):
                c["qs"].append(q_of(c["li"]))
        if len(surv) <= 1:
            break
        qbar = np.array([weighted_q(c["qs"]) for c in surv])
        sc = np.array([c["g"] + c["logit"] for c in surv]) + qscale_sigma(
            qbar, max_n=max(len(c["qs"]) for c in surv))
        keep_n = max(1, -(-len(surv) // 2))
        if keep_n >= len(surv):
            keep_n = len(surv) - 1
        keep = np.argsort(-sc)[:keep_n]
        surv = [surv[i] for i in keep]
        if len(surv) <= 1:
            break
    return cands, surv[0]


def _completed_target(logits, v_root, cands):
    """az_gumbel's completed-Q target off the stub search's candidates."""
    prior = np.exp(logits - logits.max())
    prior = prior / prior.sum()
    qv = {c["li"]: weighted_q(c["qs"]) for c in cands if c["qs"]}
    nv = {c["li"]: len(c["qs"]) for c in cands if c["qs"]}
    if qv:
        n_sum = float(sum(nv.values()))
        den = float(sum(prior[i] for i in qv)) or 1e-12
        num = float(sum(prior[i] * qv[i] for i in qv))
        v_mix = (v_root + n_sum * (num / den)) / (1.0 + n_sum)
    else:
        v_mix = v_root
    comp_q = np.full(len(logits), v_mix)
    for li, q in qv.items():
        comp_q[li] = q
    max_n = max([len(c["qs"]) for c in cands] + [1])
    return improved_policy(logits, comp_q, max_n=max_n)


# ------------------------------------------------------------------ (a)
def test_known_q_ranking_target_mass_on_argmax_and_winner_in_topm():
    """(a): with a stub value net whose Q ranking is known and noiseless, the
    improved-policy target puts its mass on the Q-argmax and the executed
    (winning) vertex is inside the Gumbel top-m."""
    n_legal, m = 6, 4
    logits = 0.01 * np.arange(n_legal)[::-1].astype(np.float64)  # near-flat
    g = np.zeros(n_legal)              # deterministic top-m = first m
    best = 2
    q_of = lambda li: 1.0 if li == best else 0.0

    cands, winner = _run_halving(logits, g, q_of, m)
    topm = {c["li"] for c in cands}
    assert best in topm
    assert winner["li"] == best        # executed action = Q argmax
    assert winner["li"] in topm        # ... and inside the top-m set

    pi = _completed_target(logits, v_root=0.0, cands=cands)
    assert abs(pi.sum() - 1.0) < 1e-12
    assert int(np.argmax(pi)) == best
    assert pi[best] > 0.9, pi          # mass, not just argmax


# ------------------------------------------------------------------ (b)
def test_weighted_q_reduces_to_plain_mean_when_sigmas_equal():
    """(b): w_k = rho * exp(sigma(q_k)); all sigma(q) equal => uniform w_hat
    => the weighted Q is the plain mean (and rho cancels entirely)."""
    qs = [0.7, 0.7, 0.7, 0.7]
    w, w_hat = draw_weights(qs)
    assert np.allclose(w_hat, 1.0 / len(qs))
    assert np.isclose(weighted_q(qs), np.mean(qs))
    # K == 1: sigma is identically 0, w_hat == [1], weighted == the value.
    assert np.isclose(weighted_q([0.31]), 0.31)
    # rho scales every w_k equally, so w_hat -- and the weighted Q -- cannot
    # depend on it.
    _, w_hat_rho = draw_weights(qs, rho=1.0)
    assert np.allclose(w_hat, w_hat_rho)


def test_weighted_q_upweights_higher_q():
    qs = [0.0, 1.0]
    _, w_hat = draw_weights(qs)
    assert w_hat[1] > w_hat[0]
    assert weighted_q(qs) > np.mean(qs)


# ------------------------------------------------------------------ (c)
def test_default_mode_all_target_depths_equal():
    """(c): with ALPHAGRAD_GAZ_DEEPEN unset every evaluation round is depth 0,
    so the depths entering one target are homogeneous by construction."""
    depths = [d for _n, _k, d in phase_plan(8)]
    assert depths == [0, 0, 0]
    assert_same_depth(depths, context="widen-mode target")  # must not raise


def test_deepen_mode_depths_mix_and_the_guard_catches_it():
    """(c'): the deepen schedule doubles depth per round -- mixing rounds in
    one target is exactly the #93 defect, and the guard must catch it, while
    the round-0-only choice passes."""
    plan = phase_plan(8, deepen=True, rollout_depth=3)
    depths = [d for _n, _k, d in plan]
    assert depths == [3, 6, 12]
    assert all(k == 1 for _n, k, _d in plan)     # deepen: one eval per round
    with pytest.raises(AssertionError, match="depth-mixed"):
        assert_same_depth(depths, context="deepen target")
    assert_same_depth([depths[0]] * 4)           # round-0 estimates only: OK


def test_phase_plan_widen_budget():
    """The halving budget widens 2**p draws per survivor: for m=8 the rounds
    are (8,1),(4,2),(2,4) -- 24 draws per decision, 7 for the winner."""
    plan = phase_plan(8)
    assert plan == [(8, 1, 0), (4, 2, 0), (2, 4, 0)]
    assert sum(n * k for n, k, _ in plan) == 24
    assert sum(k for _n, k, _ in plan) == 7
    assert phase_plan(1) == [(1, 1, 0)]


# ------------------------------------------------------------------ (d)
def test_rho_is_one_at_default_and_guard_fires_otherwise():
    """(d): rho = pi/beta == 1 while we sample from the head we train; any
    other ALPHAGRAD_GAZ_BETA_TEMP must hard-fail until the importance
    correction exists, so a proposal temperature cannot silently bias the
    weighted-Q target."""
    assert rho_from_beta_temp(1.0) == 1.0
    assert rho_from_beta_temp("1.0") == 1.0
    for bad in (0.5, 1.5, 0.0):
        with pytest.raises(AssertionError, match="BETA_TEMP"):
            rho_from_beta_temp(bad)


def test_improved_policy_is_a_distribution_and_monotone_in_q():
    logits = np.zeros(5)
    comp_q = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    pi = improved_policy(logits, comp_q, max_n=2)
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.all(np.diff(pi) > 0)     # higher completed-Q => more mass
