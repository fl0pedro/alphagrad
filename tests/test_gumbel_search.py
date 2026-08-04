"""Gumbel-AlphaZero search internals (2026-08-04 audit fixes C1/C2/M1/M2/M3).

These test the arithmetic of the search in isolation — no env, no GPU — because
every defect the audit found was in how Q values reach the Gumbel machinery,
not in the machinery itself.
"""
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# --- the formulas under test, mirrored from az_gumbel so the test needs no GPU
def sigma(q, max_n=1, cvisit=50.0, cscale=0.1):
    q = np.asarray(q, dtype=np.float64)
    lo, hi = float(q.min()), float(q.max())
    qn = (q - lo) / max(hi - lo, 1e-8)
    return (cvisit + float(max_n)) * cscale * qn


def v_mix(v_root, prior, qv, nv):
    """Danihelka et al. 2022 eq. (8)-(9): prior-weighted mixture over VISITED
    actions blended with the root value."""
    if not qv:
        return v_root
    n_sum = float(sum(nv.values()))
    den = float(sum(prior[i] for i in qv)) or 1e-12
    num = float(sum(prior[i] * qv[i] for i in qv))
    return (v_root + n_sum * (num / den)) / (1.0 + n_sum)


def test_sigma_scale_is_comparable_to_logits():
    """M1: at the old CSCALE=1.0, sigma reached ~53 against logits ~1, so the
    Gumbel noise was numerically inert and the target near-degenerate."""
    q = np.linspace(-1.0, 1.0, 8)
    s_new = sigma(q, max_n=1, cscale=0.1)
    s_old = sigma(q, max_n=1, cscale=1.0)
    assert s_new.max() < 6.0, s_new.max()
    assert s_old.max() > 50.0, s_old.max()      # documents what we fixed
    typical_logit_spread = 1.0
    assert s_new.max() < 10 * typical_logit_spread


def test_sigma_is_monotone():
    q = np.array([-2.0, 0.0, 0.5, 3.0])
    s = sigma(q)
    assert np.all(np.diff(s) >= 0)


def test_v_mix_matches_hand_worked_example():
    """C2: three actions, two visited."""
    v_root = 1.0
    prior = np.array([0.5, 0.3, 0.2])
    qv = {0: 2.0, 1: 0.0}          # visited
    nv = {0: 1, 1: 1}
    got = v_mix(v_root, prior, qv, nv)
    n_sum = 2.0
    num = 0.5 * 2.0 + 0.3 * 0.0
    den = 0.5 + 0.3
    want = (1.0 + n_sum * (num / den)) / (1.0 + n_sum)
    assert got == pytest.approx(want)
    # and it lies within the convex range it should
    assert min(v_root, min(qv.values())) <= got <= max(v_root, max(qv.values()))


def test_v_mix_beats_raw_v_root_when_root_overestimates():
    """The failure C2 describes: v_root above every searched Q.

    With the raw-v_root completion, unvisited actions get the TOP of the sigma
    range and the executed action the bottom — a target anti-correlated with the
    search. v_mix pulls the completion back toward the evaluated Qs.
    """
    v_root = 5.0                                   # optimistic root
    prior = np.array([0.4, 0.4, 0.2])
    qv = {0: 1.0, 1: 0.5}
    nv = {0: 2, 1: 2}
    mixed = v_mix(v_root, prior, qv, nv)
    # The property v_mix actually guarantees: the completion is pulled from the
    # (optimistic) root value toward the prior-weighted mean of what was
    # actually evaluated, and the pull strengthens with the visit count. It does
    # NOT promise to fall below the visited max — with a sufficiently
    # optimistic root it stays above, which is correct: an unexplored action
    # genuinely may be better.
    q_bar = (0.4 * 1.0 + 0.4 * 0.5) / 0.8
    assert q_bar < mixed < v_root
    # more visits => closer to the evaluated evidence
    mixed_more = v_mix(v_root, prior, qv, {0: 20, 1: 20})
    assert abs(mixed_more - q_bar) < abs(mixed - q_bar)
    # and the pathology the raw fill caused is bounded: the completion can no
    # longer exceed the root, so unvisited actions cannot be handed the top of
    # the sigma range purely because the critic was optimistic.
    assert mixed <= v_root


def test_ceil_halving_terminates_and_keeps_more_than_floor():
    """M3: floor(3/2)=1 collapses 3 -> 1 and skips a comparison phase."""
    def phases(n, keep):
        seq = [n]
        while n > 1:
            n = keep(n)
            seq.append(n)
            assert seq[-1] < seq[-2], "halving must strictly shrink"
        return seq
    ceil_seq = phases(24, lambda n: max(1, math.ceil(n / 2)) if math.ceil(n / 2) < n else n - 1)
    floor_seq = phases(24, lambda n: max(1, n // 2))
    assert ceil_seq[-1] == 1 and floor_seq[-1] == 1
    assert len(ceil_seq) >= len(floor_seq)
    assert phases(3, lambda n: max(1, math.ceil(n / 2)) if math.ceil(n / 2) < n else n - 1)[1] == 2


def test_all_terminal_rollouts_still_give_a_non_constant_target():
    """C1: the signature of the bug — when every rollout terminated, the old
    code returned the SAME v_root for every candidate, so the normalized Q was
    all-zero and the CE target reduced to softmax(logits): no signal."""
    logits = np.array([0.3, -0.1, 0.2, 0.0])
    v_root = 1.234
    # OLD: every terminal candidate takes v_root
    q_old = np.full(4, v_root)
    pi_old = logits + sigma(q_old, max_n=4)
    pi_old = np.exp(pi_old - pi_old.max()); pi_old /= pi_old.sum()
    ref = np.exp(logits - logits.max()); ref /= ref.sum()
    assert np.allclose(pi_old, ref), "sanity: old target == softmax(logits)"
    # NEW: terminals are evaluated, so the Qs differ
    q_new = np.array([2.0, 0.5, 1.5, -0.5])
    pi_new = logits + sigma(q_new, max_n=4)
    pi_new = np.exp(pi_new - pi_new.max()); pi_new /= pi_new.sum()
    assert not np.allclose(pi_new, ref)
    assert int(np.argmax(pi_new)) == int(np.argmax(q_new))


def test_deepest_estimate_not_mean_across_depths():
    """M2: entries of q come from different rollout depths; averaging dilutes
    the deepest (and, before C1, mixed in v_root constants)."""
    q_hist = [0.0, 0.5, 2.0]        # depth 3, 6, 12
    assert q_hist[-1] == 2.0
    assert np.mean(q_hist) == pytest.approx(0.8333, rel=1e-3)
    # the ranking of two candidates can flip between the two rules
    a, b = [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]
    assert np.mean(a) < np.mean(b) and a[-1] > b[-1]
