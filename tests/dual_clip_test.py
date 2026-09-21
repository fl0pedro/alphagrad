"""dsnn-dfw.95: DUAL-CLIP PPO ON THE POLICY SURROGATE (--dual-clip C).

WHAT WENT WRONG.  The face head's log-prob is a SUM over every live face
slot, 126-195 of them on the SHD recurrent target against 63 on NN256, so the
PPO importance ratio is a PRODUCT over that many categorical choices whose
logits move together through shared parameters.  Smoke job 67342 measured
max |log ratio| 9.6-11.2, that is a ratio of 1.5e4-7e4, from episode 0.
PPO's surrogate min(r A, clip(r) A) bounds the ratio only for A > 0.  For
A < 0 the term is r A with r unbounded, so a violating plan -- whose slots
chose OP_NONE 99 percent of the time -- pushed the none logit DOWN with
weight up to 2e4 while every feasible plan pushed it up with weight at most
1.2.  More violations, more unbounded pushes, more approximations, more
violations: the approximation runaway.

THE FIX (Ye et al. 2020, "Mastering Complex Control in MOBA Games with Deep
RL").  Cap the objective from below at c * A on the negative branch:

    objective = where(A < 0, maximum(standard, c * A), standard)

c > 1, so a healthy sample (r near 1) never reaches the cap.

WHAT IS PINNED HERE.  The four things a reader of a dual-clipped run has to
be able to trust:

  (a) OFF IS BIT-IDENTICAL, and the gate is a STATIC Python branch, so an
      off run traces the same jaxpr it traced before the flag existed.
  (b) The cap holds an exploded negative sample at exactly c * A.
  (c) A positive advantage is returned untouched, for any ratio and any c.
  (d) `ppo/dual_clip_frac` counts the samples the cap actually bit, over
      the LIVE samples only.

The expressions below are the loss's own lines (`_ppo_loss_fragment` mirrors
alphagrad.approx.ppo, the `clipping_objective` block).  A change to the loss
that this file does not see is a change the gate cannot certify.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

from types import SimpleNamespace                               # noqa: E402

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx.ppo import (                              # noqa: E402
    _dual_clip_active,
    _dual_clip_objective,
    _live_mean,
    dual_clip_coef,
    dual_clip_enabled,
    make_argparser,
)

_EPS = 0.2


def _args(c):
    return SimpleNamespace(dual_clip=c)


def _standard(ratio, norm_adv, eps=_EPS):
    """PPO's surrogate, exactly as the loss writes it."""
    return jnp.minimum(
        ratio * norm_adv,
        jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * norm_adv,
    )


def _ppo_loss_fragment(args, ratio, norm_adv, w_live, eps=_EPS):
    """The loss's `clipping_objective` block, gate and all.

    Returns (ppo_loss, dual_clip_frac).  Kept line for line the same as
    alphagrad.approx.ppo so the jaxpr comparison below means what it says.
    """
    clipping_objective = _standard(ratio, norm_adv, eps)
    if dual_clip_enabled(args):
        c = dual_clip_coef(args)
        dual_clip_frac = _live_mean(
            _dual_clip_active(
                clipping_objective, norm_adv, c
            ).astype(jnp.float32),
            w_live)
        clipping_objective = _dual_clip_objective(
            clipping_objective, norm_adv, c)
    else:
        dual_clip_frac = jnp.zeros((), jnp.float32)
    return _live_mean(-clipping_objective, w_live), dual_clip_frac


def _before_the_flag(ratio, norm_adv, w_live, eps=_EPS):
    """The loss as it stood before --dual-clip existed."""
    return _live_mean(-_standard(ratio, norm_adv, eps), w_live)


def _bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _batch(n=64, seed=0):
    rng = np.random.default_rng(seed)
    ratio = jnp.asarray(np.exp(rng.normal(0.0, 1.5, n)), jnp.float32)
    norm_adv = jnp.asarray(rng.normal(0.0, 1.0, n), jnp.float32)
    w_live = jnp.asarray(rng.integers(0, 2, n), jnp.float32)
    return ratio, norm_adv, w_live


# ------------------------------------------------------------------ the gate

def test_the_flag_is_off_by_default():
    p = make_argparser()
    assert p.get_default("dual_clip") == 0.0
    assert dual_clip_enabled(_args(0.0)) is False
    assert dual_clip_coef(_args(0.0)) == 0.0
    # An args object that predates the flag reads as off, never as a crash.
    assert dual_clip_enabled(SimpleNamespace()) is False


def test_the_help_text_names_the_ticket_and_the_rule():
    help_text = make_argparser().format_help()
    assert "--dual-clip" in help_text
    assert "dsnn-dfw.95" in help_text


@pytest.mark.parametrize("c", [1.0, 0.5, -3.0])
def test_a_coefficient_that_is_not_above_one_is_refused(c):
    with pytest.raises(SystemExit) as e:
        dual_clip_coef(_args(c))
    assert "--dual-clip" in str(e.value)


@pytest.mark.parametrize("c", [1.0001, 3.0, 5.0])
def test_a_coefficient_above_one_turns_the_cap_on(c):
    assert dual_clip_enabled(_args(c)) is True
    assert dual_clip_coef(_args(c)) == pytest.approx(c)


# ------------------------------------------------------- (a) off is identity

def test_off_is_bit_identical_to_the_loss_before_the_flag():
    ratio, norm_adv, w_live = _batch()
    loss, frac = _ppo_loss_fragment(_args(0.0), ratio, norm_adv, w_live)
    ref = _before_the_flag(ratio, norm_adv, w_live)
    assert _bits(loss) == _bits(ref)
    assert float(frac) == 0.0


def test_off_traces_the_same_jaxpr_as_the_loss_before_the_flag():
    """The gate is a STATIC Python branch, so `off` is not `a cap that never
    bites` -- it is the same graph, with no extra operation in it at all."""
    off = jax.make_jaxpr(
        lambda r, a, w: _ppo_loss_fragment(_args(0.0), r, a, w)[0])
    before = jax.make_jaxpr(_before_the_flag)
    ratio, norm_adv, w_live = _batch()
    assert str(off(ratio, norm_adv, w_live)) == \
        str(before(ratio, norm_adv, w_live))


def test_on_does_change_the_graph():
    """The counterpart of the test above: if the two jaxprs matched with the
    flag ON as well, the comparison would be pinning nothing."""
    ratio, norm_adv, w_live = _batch()
    on = jax.make_jaxpr(
        lambda r, a, w: _ppo_loss_fragment(_args(3.0), r, a, w)[0])
    before = jax.make_jaxpr(_before_the_flag)
    assert str(on(ratio, norm_adv, w_live)) != \
        str(before(ratio, norm_adv, w_live))


# ------------------------------------------- (b) an exploded negative sample

def test_an_exploded_negative_sample_is_held_at_c_times_the_advantage():
    c = 3.0
    ratio = jnp.asarray([100.0], jnp.float32)
    norm_adv = jnp.asarray([-0.7], jnp.float32)
    standard = _standard(ratio, norm_adv)
    # What the unclipped surrogate would have done: r * A, weight 100.
    assert float(standard[0]) == pytest.approx(-70.0, rel=1e-6)
    got = _dual_clip_objective(standard, norm_adv, c)
    want = jnp.asarray(c, jnp.float32) * norm_adv
    assert _bits(got)[0] == _bits(want)[0]
    assert float(got[0]) == pytest.approx(-2.1, rel=1e-6)
    assert bool(_dual_clip_active(standard, norm_adv, c)[0]) is True


def test_a_healthy_negative_sample_is_not_touched():
    """c = 3 sits well above a ratio near 1, so an ordinary sample keeps the
    standard surrogate bit for bit."""
    c = 3.0
    ratio = jnp.asarray([0.9, 1.0, 1.1, 2.9], jnp.float32)
    norm_adv = jnp.asarray([-1.0, -1.0, -1.0, -1.0], jnp.float32)
    standard = _standard(ratio, norm_adv)
    got = _dual_clip_objective(standard, norm_adv, c)
    assert (_bits(got) == _bits(standard)).all()
    assert not bool(jnp.any(_dual_clip_active(standard, norm_adv, c)))


# ------------------------------------------- (c) a positive advantage stands

def test_a_positive_advantage_is_never_touched():
    rng = np.random.default_rng(7)
    ratio = jnp.asarray(np.exp(rng.normal(0.0, 2.0, 256)), jnp.float32)
    norm_adv = jnp.asarray(np.abs(rng.normal(0.0, 1.0, 256)) + 1e-3,
                           jnp.float32)
    standard = _standard(ratio, norm_adv)
    for c in (1.5, 3.0, 5.0, 50.0):
        got = _dual_clip_objective(standard, norm_adv, c)
        assert (_bits(got) == _bits(standard)).all(), c
        assert not bool(jnp.any(_dual_clip_active(standard, norm_adv, c))), c


def test_a_zero_advantage_is_never_touched():
    """A == 0 is not the negative branch: every surrogate term is 0 there and
    the cap must not turn it into anything else."""
    ratio = jnp.asarray([1e-4, 1.0, 2e4], jnp.float32)
    norm_adv = jnp.zeros((3,), jnp.float32)
    standard = _standard(ratio, norm_adv)
    got = _dual_clip_objective(standard, norm_adv, 3.0)
    assert (_bits(got) == _bits(standard)).all()
    assert not bool(jnp.any(_dual_clip_active(standard, norm_adv, 3.0)))


# ------------------------------------------------------ (d) the frac metric

#: ratio, advantage, and whether c = 3 bites.  The cap is active exactly
#: where A < 0 and max(r, clip(r, 0.8, 1.2)) > c.
_FRAC_ROWS = (
    (100.0, -1.0, True),      # the runaway sample
    (5.0, -1.0, True),        # above the cap, still
    (1.0, -1.0, False),       # healthy
    (100.0, 1.0, False),      # exploded, but the advantage is positive
    (2.9, -1.0, False),       # just under the cap
    (50.0, -2.0, True),       # the cap scales with the advantage
)


def _frac_batch():
    ratio = jnp.asarray([r for r, _a, _b in _FRAC_ROWS], jnp.float32)
    norm_adv = jnp.asarray([a for _r, a, _b in _FRAC_ROWS], jnp.float32)
    want = [b for _r, _a, b in _FRAC_ROWS]
    return ratio, norm_adv, want


def test_the_active_mask_is_exactly_the_samples_the_cap_bit():
    ratio, norm_adv, want = _frac_batch()
    standard = _standard(ratio, norm_adv)
    got = np.asarray(_dual_clip_active(standard, norm_adv, 3.0))
    assert list(got) == want


def test_the_fraction_counts_every_sample_when_nothing_is_refused():
    ratio, norm_adv, want = _frac_batch()
    w_live = jnp.ones((len(want),), jnp.float32)
    _loss, frac = _ppo_loss_fragment(_args(3.0), ratio, norm_adv, w_live)
    assert float(frac) == pytest.approx(sum(want) / len(want))
    assert float(frac) == pytest.approx(0.5)


def test_the_fraction_excludes_a_refused_sample():
    """`_live_mean` weights by TrainBatch.live: a refused measurement is
    missing data and must not enter the number that is logged, even when its
    ratio is the largest in the batch."""
    ratio, norm_adv, want = _frac_batch()
    w_live = jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], jnp.float32)
    _loss, frac = _ppo_loss_fragment(_args(3.0), ratio, norm_adv, w_live)
    assert float(frac) == pytest.approx(2.0 / 5.0)


def test_the_fraction_is_zero_when_the_flag_is_off():
    ratio, norm_adv, want = _frac_batch()
    w_live = jnp.ones((len(want),), jnp.float32)
    _loss, frac = _ppo_loss_fragment(_args(0.0), ratio, norm_adv, w_live)
    assert float(frac) == 0.0


# ------------------------------------------------------- the runaway itself

def test_the_cap_bounds_the_gradient_weight_of_a_violating_plan():
    """The mechanism, stated as a number.  The surrogate is differentiated
    with respect to the LOG-probability, and d(ratio)/d(log-prob) = ratio, so
    an uncapped violating sample at r = 2e4 pushes with weight 2e4 while a
    feasible sample pushes with weight at most 1 + eps.  Past the cap the
    objective is the constant c * A and the weight is 0."""
    c = 3.0
    adv = jnp.asarray([-1.0], jnp.float32)
    log_ratio = jnp.asarray([np.log(2e4)], jnp.float32)

    def uncapped(lr):
        return _standard(jnp.exp(lr), adv).sum()

    def capped(lr):
        std = _standard(jnp.exp(lr), adv)
        return _dual_clip_objective(std, adv, c).sum()

    assert float(jax.grad(uncapped)(log_ratio)[0]) == pytest.approx(
        -2e4, rel=1e-3)
    assert float(jax.grad(capped)(log_ratio)[0]) == 0.0
    # and a feasible sample is untouched by the cap, weight 1 + eps at most
    feasible = jnp.asarray([np.log(1.5)], jnp.float32)
    pos = jnp.asarray([1.0], jnp.float32)

    def feasible_obj(lr):
        return _standard(jnp.exp(lr), pos).sum()

    assert float(jax.grad(feasible_obj)(feasible)[0]) == 0.0
