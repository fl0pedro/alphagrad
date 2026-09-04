"""FIDELITY (reward slot 8): the clip convention, the two channel tables, the
``--lambda-frob`` repoint, and FLAG-OFF BIT-IDENTITY.

Workstream A2. The owner's choice was "clipped relative-Frobenius TRAINED,
cosine LOGGED, gradient coverage a hard GUARD". What is pinned here:

1. THE SIGN AND CLIP CONVENTION of ``env.clipped_rel_frob``:

       fidelity = clip(1 - ||J_e - J_a||_F / ||J_e||_F, -1, +1)

   +1 iff the Jacobian is exact; EXACTLY 0 when the error equals the gradient
   norm -- which is where an all-zero Jacobian lands, the case the cosine
   cannot score because it is 0/0 there; -1 is the floor, reached at
   rel_frob >= 2 (e.g. a sign-flipped Jacobian of the same magnitude).

2. THE TWO CHANNEL TABLES AGREE. ``env.REWARD_NAMES`` is a strict PREFIX of
   ``common.reward_scaling.REWARD_NAMES``; slot 7 is ``grad_coverage`` in BOTH
   (RESERVED since 2026-09-03, never populated); slot 8 is ``fidelity``;
   slot 9 stays ``bkstep_acc``, because persisted PopArt / calibration state is
   keyed by INDEX and must never be renumbered.

3. ``--lambda-frob`` WEIGHTS THE FIDELITY CHANNEL, and its two 1.0 defaults are
   gone. Between `bcb61a1` and A2 the flag put a full-strength weight on the
   COVERAGE guard on every driver that defaulted it to 1.0.

4. FLAG-OFF BIT-IDENTITY. With ``--fidelity-weight 0``: no extra value head, no
   extra pytree leaf, the symlog exempt set is unchanged, the slot reads 0.0,
   and ``build_reward_weights`` puts nothing anywhere new.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
for _k in ("ALPHAGRAD_FIDELITY", "ALPHAGRAD_FIDELITY_WEIGHT",
           "ALPHAGRAD_COS_LOG_EVERY"):
    os.environ.pop(_k, None)

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.common import reward_scaling as rs        # noqa: E402
from alphagrad.approx.cpu_approx_pool import (                  # noqa: E402
    _SENTINEL_REWARD_VALUE, _sentinel_callback_output,
)

FSLOT = int(envmod.REWARD_INDEX["fidelity"])
GSLOT = int(envmod.REWARD_INDEX["grad_coverage"])


# ------------------------------------------------- 1. clip / sign convention
def test_exact_scores_exactly_one():
    assert envmod.clipped_rel_frob(0.0) == 1.0


def test_unit_relative_error_scores_exactly_zero():
    """The definitional midpoint: the error is the size of the gradient."""
    assert envmod.clipped_rel_frob(1.0) == 0.0


def test_floor_is_minus_one_and_binds_at_two():
    assert envmod.clipped_rel_frob(2.0) == -1.0
    assert envmod.clipped_rel_frob(2.0000001) == -1.0
    assert envmod.clipped_rel_frob(37.0) == -1.0
    assert envmod.clipped_rel_frob(1e30) == -1.0


def test_ceiling_binds_too():
    """rel_frob is never negative in exact arithmetic, but a float32
    accumulation of ||e-a||^2 can round to a hair below 0 on an exact plan;
    the clip must not let that report better than exact."""
    assert envmod.clipped_rel_frob(-1e-9) == 1.0


def test_non_finite_residual_takes_the_floor():
    """nan/inf comes from a blown-up plan or a broken comparison. Both are
    statements about the PLAN, so they score the floor -- the same rule
    `_quality_metrics` applies to a shape-mismatched pair."""
    assert envmod.clipped_rel_frob(float("nan")) == -1.0
    assert envmod.clipped_rel_frob(float("inf")) == -1.0


def test_monotone_in_the_residual():
    xs = [0.0, 0.1, 0.25, 0.5, 0.9, 1.0, 1.5, 2.0]
    vals = [envmod.clipped_rel_frob(x) for x in xs]
    assert vals == sorted(vals, reverse=True)


# --------------------------------------- the same convention, end to end
def _leaves(*arrays):
    return [jnp.asarray(a, dtype=jnp.float32) for a in arrays]


def test_identical_jacobian_is_fidelity_one():
    e = _leaves([[1.0, -2.0], [3.0, 0.5]], [7.0, -7.0])
    rf, cos = envmod._residual_scores(e, list(e), has_aux=False)
    assert rf == pytest.approx(0.0, abs=1e-6)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert envmod.clipped_rel_frob(rf) == pytest.approx(1.0, abs=1e-6)


def test_zeroed_gradient_is_fidelity_zero_where_cosine_is_undefined():
    """THE CASE THAT MOTIVATED THE CHOICE. ||J_e - 0||_F == ||J_e||_F, so
    rel_frob is exactly 1 and fidelity is exactly 0 -- a real, ordered score.
    The cosine is <J_e, 0> / (||J_e|| * 0): 0/0, and the eps-clamped formula
    only reports 0 there by accident of the epsilon."""
    e = _leaves([[1.0, -2.0], [3.0, 0.5]], [7.0, -7.0])
    a = _leaves([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0])
    rf, _cos = envmod._residual_scores(e, a, has_aux=False)
    assert rf == pytest.approx(1.0, abs=1e-6)
    assert envmod.clipped_rel_frob(rf) == pytest.approx(0.0, abs=1e-6)


def test_sign_flipped_jacobian_hits_the_floor():
    e = _leaves([[1.0, -2.0], [3.0, 0.5]])
    a = [-x for x in e]
    rf, cos = envmod._residual_scores(e, a, has_aux=False)
    assert rf == pytest.approx(2.0, abs=1e-6)
    assert cos == pytest.approx(-1.0, abs=1e-6)
    assert envmod.clipped_rel_frob(rf) == pytest.approx(-1.0, abs=1e-6)


def test_halved_jacobian_separates_fidelity_from_the_cosine():
    """The cosine is blind to magnitude; the residual is not. This is the whole
    reason one is trained and the other only logged."""
    e = _leaves([[1.0, -2.0], [3.0, 0.5]])
    a = [0.5 * x for x in e]
    rf, cos = envmod._residual_scores(e, a, has_aux=False)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert rf == pytest.approx(0.5, abs=1e-6)
    assert envmod.clipped_rel_frob(rf) == pytest.approx(0.5, abs=1e-6)


def test_broken_comparison_is_not_silently_perfect():
    e = _leaves([[1.0, 2.0]])
    a = _leaves([[1.0, 2.0]], [3.0])
    rf, cos = envmod._residual_scores(e, a, has_aux=False)
    assert not np.isfinite(rf) and not np.isfinite(cos)
    assert envmod.clipped_rel_frob(rf) == -1.0


def test_residual_matches_quality_metrics_on_the_same_pair():
    """`_residual_scores` exists only because the coverage path holds the two
    outputs somewhere else; it must be the SAME number."""
    e = _leaves([[1.0, -2.0], [3.0, 0.5]], [7.0, -7.0])
    a = _leaves([[0.9, -2.2], [2.7, 0.1]], [6.1, -8.0])
    rf, cos = envmod._residual_scores(e, a, has_aux=False)
    q_cos, q_rf = envmod._quality_metrics(e, a)
    assert rf == pytest.approx(float(q_rf), rel=1e-6)
    assert cos == pytest.approx(float(q_cos), rel=1e-6)


# ------------------------------------------------------ 2. the two tables
def test_env_table_is_a_prefix_of_the_scaling_table():
    assert rs.REWARD_NAMES[:envmod.NUM_REWARDS] == envmod.REWARD_NAMES
    # A7 STRENGTHENS THIS TO EQUALITY. env.py reserved index 9
    # (`bkstep_acc`, which it never populates) so that the sparsity
    # channel could be APPENDED at 10 in both tables rather than colliding
    # with bkstep_acc at 9 in one of them. With the tables identical, a
    # weight vector built from either lands on the same channel of the
    # other -- which is the property the prefix pin was approximating.
    assert rs.REWARD_NAMES == envmod.REWARD_NAMES
    for name in envmod.REWARD_NAMES:
        assert rs.REWARD_INDEX[name] == envmod.REWARD_INDEX[name], name


def test_slot_names_and_the_indices_that_must_not_move():
    # 9 until 2026-08-28; A7 APPENDED slot 10 (`sparsity`) and RESERVED
    # slot 9 for reward_scaling's `bkstep_acc` so the two tables are
    # index-identical. No existing index moved.
    assert envmod.NUM_REWARDS == 11
    assert envmod.REWARD_NAMES[9] == "bkstep_acc"
    assert envmod.REWARD_NAMES[10] == "sparsity"
    assert rs.REWARD_NAMES[10] == "sparsity"
    assert rs.SPARSITY_IDX == 10
    assert envmod.REWARD_NAMES[7] == "grad_coverage"
    assert envmod.REWARD_NAMES[8] == "fidelity"
    assert rs.REWARD_NAMES[7] == "grad_coverage"
    assert rs.REWARD_NAMES[8] == "fidelity"
    # Persisted PopArt / calibration state is index-keyed: 9 stays 9.
    assert rs.REWARD_NAMES[9] == "bkstep_acc"
    assert rs.BKSTEP_ACC_IDX == 9
    assert rs.FIDELITY_IDX == 8 == FSLOT
    assert rs.GRAD_COVERAGE_IDX == 7 == GSLOT


def test_historical_aliases_still_resolve_in_both_tables():
    for tbl in (envmod.REWARD_INDEX, rs.REWARD_INDEX):
        assert tbl["cosine_sim"] == 6
        assert tbl["frob_residual"] == 7
    assert rs.FROB_RESIDUAL_IDX == rs.GRAD_COVERAGE_IDX
    # xla_peak_memory owned index 8 and was documented as dead there; it now
    # aliases the ONE memory channel, which is what --mem-type already did.
    assert rs.REWARD_INDEX["xla_peak_memory"] == rs.REWARD_INDEX["peak_memory"]
    assert rs._MEM_TYPE_TO_REWARD["xla_peak_memory"] == "peak_memory"


def test_fidelity_is_a_quality_channel_not_a_cost_channel():
    assert FSLOT in rs._QUALITY_REWARD_INDICES
    assert FSLOT not in rs.COST_REWARD_INDICES
    assert FSLOT in rs.SPARSE_TERMINAL_INDICES
    # Bounded [-1, 1] => symlog would only discount its per-unit price.
    assert FSLOT in rs.NO_SYMLOG_REWARD_INDICES
    assert FSLOT in envmod.QUALITY_REWARD_INDICES


def test_sentinel_writes_the_clip_floor_not_minus_1e10():
    s = np.asarray(envmod._SENTINEL_BAD_REWARD)
    assert s.shape == (envmod.NUM_REWARDS,)
    assert s[FSLOT] == -1.0
    _t, _e, r = _sentinel_callback_output(
        4, envmod.NUM_REWARDS, int(envmod.REWARD_INDEX["cosine_sim"]),
        GSLOT, FSLOT)
    assert r[FSLOT] == -1.0
    # ...and without the new argument the wire is exactly what it was.
    _t, _e, r0 = _sentinel_callback_output(
        4, envmod.NUM_REWARDS, int(envmod.REWARD_INDEX["cosine_sim"]), GSLOT)
    assert r0[FSLOT] == _SENTINEL_REWARD_VALUE


# --------------------------------------------------- 3. --lambda-frob repoint
def _rs_args(**kw):
    d = dict(rewards=["cmp", "mem", "acc"], cmp_type="latency",
             mem_type="peak_memory", lambda_cmp=1.0, lambda_mem=1.0,
             lambda_acc=2.0, lambda_frob=0.0, measure_latency=True)
    d.update(kw)
    return SimpleNamespace(**d)


def test_lambda_frob_weights_fidelity_and_never_coverage():
    w = rs.build_reward_weights(_rs_args(lambda_frob=3.0))
    assert float(w[rs.FIDELITY_IDX]) == 3.0
    assert float(w[rs.GRAD_COVERAGE_IDX]) == 0.0


def test_lambda_frob_zero_weights_nothing():
    w = rs.build_reward_weights(_rs_args(lambda_frob=0.0))
    assert float(w[rs.FIDELITY_IDX]) == 0.0
    assert float(w[rs.GRAD_COVERAGE_IDX]) == 0.0


def _lambda_frob_default(parser):
    for a in parser._actions:
        if "--lambda-frob" in a.option_strings:
            return a.default
    pytest.fail("--lambda-frob not found")


def test_args_ppo_lambda_frob_default_is_zero():
    """This defaulted to 1.0 and, after `bcb61a1` renamed slot 7, put a
    full-strength weight on the COVERAGE guard on every run that used it."""
    from alphagrad.approx.args_ppo import add_ppo_args
    p = add_ppo_args(argparse.ArgumentParser())
    assert _lambda_frob_default(p) == 0.0


def test_mu0_args_lambda_frob_default_is_zero():
    from alphagrad.approx.mu0_args import make_argparser
    assert _lambda_frob_default(make_argparser()) == 0.0





# ------------------------------------------------ 4. flag-off bit-identity
def _args(**kw):
    d = dict(fidelity_weight=0.0, cos_log_every=0,
             symlog_channels="all", no_symlog=False, reward_mode="additive",
             rewards=["cmp", "mem", "acc"], lambda_cmp=1.0, lambda_mem=1.0,
             lambda_acc=16.0, cmp_type="latency", mem_type="peak_memory")
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(autouse=True)
def _restore_head_config():
    saved = (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
             ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR)
    yield
    (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
     ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR) = saved
    for k in ("ALPHAGRAD_FIDELITY", "ALPHAGRAD_FIDELITY_WEIGHT",
              "ALPHAGRAD_COS_LOG_EVERY"):
        os.environ.pop(k, None)
    envmod._COS_LOG_SEEN[0] = 0
    envmod.consume_fidelity_stats()


def test_flag_off_leaves_the_head_configuration_at_three():
    w, every = ppo.configure_fidelity(_args())
    assert (w, every) == (0.0, 0)
    assert ppo.NUM_VALUE_HEADS == 3
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality")
    assert ppo.VALUE_HEAD_ATTRS == (
        "value_head_flops", "value_head_mem", "value_head_cos")
    assert ppo.FIDELITY_HEAD not in ppo.VALUE_HEAD_ATTRS
    assert not envmod.fidelity_enabled()


def test_flag_on_appends_exactly_one_head_on_slot_8_and_is_idempotent():
    w, _ = ppo.configure_fidelity(_args(fidelity_weight=4.0))
    assert w == 4.0
    assert ppo.NUM_VALUE_HEADS == 4
    assert ppo.HEAD_NAMES[3] == "fidelity"
    assert ppo.HEAD_REWARD_INDICES[3] == FSLOT
    ppo.configure_fidelity(_args(fidelity_weight=4.0))
    assert ppo.NUM_VALUE_HEADS == 4
    assert envmod.fidelity_enabled()
    wts = ppo._build_head_weights(_args(fidelity_weight=4.0))
    assert [float(x) for x in wts] == [1.0, 1.0, 16.0, 4.0]


def test_symlog_exempt_set_is_unchanged_when_the_channel_is_off():
    for mode in ("all", "cost", "none"):
        for rmode in ("additive", "lagrangian"):
            a = _args(symlog_channels=mode, reward_mode=rmode)
            ppo.configure_fidelity(a)
            ppo.configure_symlog(a)
            assert FSLOT not in ppo._NO_SYMLOG_REWARD_INDICES, (mode, rmode)
            assert not bool(ppo._NO_SYMLOG_MASK_NP[FSLOT]), (mode, rmode)


def test_symlog_exempts_fidelity_when_the_channel_is_on():
    a = _args(symlog_channels="cost", fidelity_weight=1.0)
    ppo.configure_fidelity(a)
    ppo.configure_symlog(a)
    assert bool(ppo._NO_SYMLOG_MASK_NP[FSLOT])
    assert bool(ppo._NO_SYMLOG_MASK_NP[int(envmod.REWARD_INDEX["cosine_sim"])])


def test_display_weights_never_touch_slot_8():
    for a in (_args(), _args(fidelity_weight=4.0)):
        ppo.configure_fidelity(a)
        assert float(ppo._build_reward_weights(a)[FSLOT]) == 0.0


# ------------------------------------------------- the cosine log subsample
def test_cos_log_is_off_by_default():
    assert envmod.cos_log_every() == 0
    assert not envmod._cos_log_due()
    assert not envmod._cos_log_due()


def test_cos_log_stride_fires_on_every_nth_terminal_plan():
    os.environ["ALPHAGRAD_COS_LOG_EVERY"] = "4"
    envmod._COS_LOG_SEEN[0] = 0
    fired = [envmod._cos_log_due() for _ in range(12)]
    assert fired == [True, False, False, False] * 3
    assert sum(fired) == 3


def test_fidelity_stats_report_the_amortised_price():
    envmod.consume_fidelity_stats()
    envmod._FIDELITY_STATS["wall_s"] = 2.0
    envmod._record_fidelity(0.5, 0.5, 0.99)
    envmod._record_fidelity(-1.0, 9.0, None)
    out = envmod.consume_fidelity_stats()
    assert out["count"] == 2
    assert out["mean"] == pytest.approx(-0.25)
    assert out["min"] == -1.0 and out["max"] == 0.5
    assert out["clipped_low"] == 1
    assert out["cos_count"] == 1 and out["cos_mean"] == pytest.approx(0.99)
    assert out["wall_amortised_s"] == pytest.approx(1.0)
    # ...and the poller resets, like every other one in env.py.
    assert envmod.consume_fidelity_stats()["count"] == 0


def test_cos_only_sample_does_not_move_the_fidelity_counter():
    envmod.consume_fidelity_stats()
    envmod._record_fidelity(None, None, 0.42)
    out = envmod.consume_fidelity_stats()
    assert out["count"] == 0 and out["cos_count"] == 1
