"""SPARSITY (reward slot 10): the definition, the bounds, the identity case,
the two channel tables, the sentinel floor and FLAG-OFF INERTNESS.

Workstream A7. The owner's ask was "measure sparsity as a reward too, as a
ratio to the original". What is pinned here:

1. THE DEFINITION IS MEASURED FROM WHAT IS STORED, not from declared classes.
   graphax tallies ``val.size * dtype.itemsize`` at its one accumulated-Jacobian
   edge-store site, once per TRACE. The tally is inert unless armed, and an
   armed trace and a disarmed trace produce the same jaxpr.

2. THE IDENTITY CASE IS EXACT, NOT APPROXIMATE. Two traces of the same
   elimination with no approximation give byte-identical tallies, so
   ``stored(approx)/stored(exact) == 1.0`` and the channel reads EXACTLY 0.0.
   This is a property of the construction (both arms are the same order with
   nothing but the approximation kwargs dropped), not a tolerance.

3. THE BOUNDS AND THE SIGN CONVENTION of ``env.sparsity_channel``:
   +1 at ratio 0 (stored nothing -- the all-SKIP ceiling), 0 at ratio 1
   (stored what exact stores), -1 at ratio >= 2 (densified), and 0.0 --
   "not measured" -- for an undefined ratio, which is apparatus and not a
   verdict on the plan.

4. THE TWO CHANNEL TABLES ARE NOW IDENTICAL. env reserves index 9
   (``bkstep_acc``, which it never populates) so sparsity could be APPENDED at
   10 in both rather than colliding with bkstep_acc in one of them. No existing
   index moved.

5. NO GUARD PRECONDITION (owner ruling 2026-09-03, ticket dsnn-3qm.15). The gradient-coverage guard that
   used to be required for a non-zero ``--sparsity-weight`` was removed --
   no guard, never a reward gate. Sparsity is still maximised by DELETING
   computation and nothing refuses those plans; the channel stays default off.

6. FLAG-OFF INERTNESS: no extra value head, no extra pytree leaf, the symlog
   exempt set unchanged, the slot reads 0.0.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
for _k in ("ALPHAGRAD_SPARSITY", "ALPHAGRAD_SPARSITY_WEIGHT",
           "ALPHAGRAD_FIDELITY", "ALPHAGRAD_FIDELITY_WEIGHT"):
    os.environ.pop(_k, None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                            # noqa: E402
import alphagrad.approx.ppo as ppo                               # noqa: E402
from alphagrad.approx.common import reward_scaling as rs         # noqa: E402
from alphagrad.approx.cpu_approx_pool import (                   # noqa: E402
    _SENTINEL_REWARD_VALUE, _sentinel_callback_output,
)

SSLOT = int(envmod.REWARD_INDEX["sparsity"])


# ------------------------------------------------------- 1. the graphax tally
def _target(x, y):
    z = x * y
    w = jnp.sin(z)
    return jnp.sum(w * z + w)


_ARGS = (jnp.ones((4, 3)), jnp.ones((4, 3)) * 0.5)


def _tally(order="rev"):
    """Stored-byte totals for ONE abstract walk of the elimination."""
    from graphax import jacve
    from graphax.core import (arm_store_accounting, disarm_store_accounting,
                              reset_store_accounting, store_accounting_totals)
    fn = jacve(_target, order, argnums=(0, 1))
    reset_store_accounting()
    arm_store_accounting()
    try:
        jax.eval_shape(fn, *_ARGS)
    finally:
        disarm_store_accounting()
    return store_accounting_totals()


def test_the_tally_counts_something_and_counts_bytes():
    t = _tally()
    assert t["edges"] > 0, "no accumulated-Jacobian edge was stored at all"
    assert t["bytes"] > 0
    assert t["cells"] > 0
    # bytes are cells-or-fewer times an itemsize: a `val is None` edge stores
    # nothing but still HAS structure, so cells >= elems always.
    assert t["cells"] >= t["elems"]
    assert t["bytes"] % 1 == 0


def test_the_tally_is_inert_when_disarmed():
    """The channel is default-off. A disarmed trace must leave the counters
    exactly where it found them -- this is the flag-off contract at the
    library level, below anything EQ_DUMP can see."""
    from graphax import jacve
    from graphax.core import reset_store_accounting, store_accounting_totals
    reset_store_accounting()
    jax.eval_shape(jacve(_target, "rev", argnums=(0, 1)), *_ARGS)
    t = store_accounting_totals()
    assert t["bytes"] == 0 and t["edges"] == 0 and t["cells"] == 0


def test_the_tally_is_deterministic_so_the_identity_ratio_is_exactly_one():
    """THE IDENTITY CASE. Two walks of the SAME elimination with no
    approximation give byte-identical tallies, so the ratio is exactly 1.0 and
    the channel exactly 0.0 -- by construction, not within a tolerance."""
    a, b = _tally(), _tally()
    assert a == b
    ratio = a["bytes"] / b["bytes"]
    assert ratio == 1.0
    assert envmod.sparsity_channel(ratio) == 0.0


# --------------------------------------------------- 2. bounds / sign / edges
@pytest.mark.parametrize("ratio,expected", [
    (0.0, 1.0),      # stored nothing: the all-SKIP ceiling
    (0.25, 0.75),
    (1.0, 0.0),      # stored exactly what exact stores: the identity plan
    (1.5, -0.5),
    (2.0, -1.0),     # twice the exact arm: the clip floor
    (17.0, -1.0),    # unbounded densification still clips
])
def test_sparsity_channel_values(ratio, expected):
    assert envmod.sparsity_channel(ratio) == pytest.approx(expected)


def test_sparsity_channel_is_bounded_everywhere():
    for r in np.concatenate([np.linspace(0.0, 50.0, 501), [1e9]]):
        v = envmod.sparsity_channel(float(r))
        assert -1.0 <= v <= 1.0


def test_undefined_ratio_reads_not_measured_not_the_floor():
    """An exact arm that stored nothing is an UNDEFINED comparison, i.e.
    apparatus -- so the slot reads 0.0 ("not measured"), never the floor.
    Contrast `clipped_rel_frob`, where a non-finite residual really is a
    statement about the plan and takes the floor."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert envmod.sparsity_channel(bad) == 0.0


# ------------------------------------------------------ 3. the two tables
def test_the_two_channel_tables_are_identical():
    assert envmod.REWARD_NAMES == rs.REWARD_NAMES
    for name in envmod.REWARD_NAMES:
        assert rs.REWARD_INDEX[name] == envmod.REWARD_INDEX[name], name


def test_nothing_moved_and_sparsity_was_appended():
    assert SSLOT == 10
    assert rs.SPARSITY_IDX == 10
    assert envmod.NUM_REWARDS == 12
    assert envmod.REWARD_NAMES[11] == "mem_objective"
    # every pre-existing index is where it was
    assert envmod.REWARD_INDEX["quality"] == 6
    assert envmod.REWARD_INDEX["grad_coverage"] == 7
    assert envmod.REWARD_INDEX["fidelity"] == 8
    assert rs.BKSTEP_ACC_IDX == 9
    assert envmod.REWARD_NAMES[9] == "bkstep_acc"


def test_sparsity_is_a_bounded_quality_channel_not_a_cost():
    assert SSLOT in rs._QUALITY_REWARD_INDICES
    assert SSLOT not in rs.COST_REWARD_INDICES
    assert SSLOT in rs.SPARSE_TERMINAL_INDICES
    assert SSLOT in rs.NO_SYMLOG_REWARD_INDICES
    assert SSLOT in envmod.QUALITY_REWARD_INDICES
    # and it is NOT one of the six cost channels `_is_degen` keys on
    assert SSLOT not in envmod.COMPUTE_REWARD_INDICES


# ------------------------------------------------------------- 4. sentinels
def test_a_sentinelled_plan_takes_the_sparsity_floor_not_the_ceiling():
    """THE most important line in the channel. A sentinelled plan is
    overwhelmingly a plan that DELETED computation -- which is the
    sparsity ceiling. It must score the floor instead."""
    v = np.asarray(envmod._SENTINEL_BAD_REWARD)
    assert v.shape == (envmod.NUM_REWARDS,)
    assert v[SSLOT] == -1.0
    # the reserved slot must NOT be stamped with the cost sentinel
    assert v[envmod.REWARD_INDEX["bkstep_acc"]] == 0.0


def test_the_pool_sentinel_also_floors_it():
    _, _, r = _sentinel_callback_output(
        8, envmod.NUM_REWARDS, int(envmod.REWARD_INDEX["cosine_sim"]),
        int(envmod.REWARD_INDEX["frob_residual"]),
        int(envmod.REWARD_INDEX["fidelity"]), SSLOT)
    assert r[SSLOT] == -1.0
    assert r[0] == _SENTINEL_REWARD_VALUE


def test_the_pool_sentinel_stays_optional():
    """Callers that never pass the index (mu0/gfn workers) keep working."""
    _, _, r = _sentinel_callback_output(
        8, envmod.NUM_REWARDS, int(envmod.REWARD_INDEX["cosine_sim"]),
        int(envmod.REWARD_INDEX["frob_residual"]))
    assert r.shape == (envmod.NUM_REWARDS,)


# ------------------------- 5. no guard precondition (owner ruling 2026-09-03)
def _args(**kw):
    base = dict(sparsity_weight=0.0, sparsity_log=False,
                fidelity_weight=0.0, cos_log_every=0,
                reward_mode="additive", symlog_channels="all")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _restore_head_globals():
    saved = (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
             ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR)
    env_saved = {k: os.environ.get(k) for k in
                 ("ALPHAGRAD_SPARSITY", "ALPHAGRAD_SPARSITY_WEIGHT",
                  "ALPHAGRAD_FIDELITY_WEIGHT", "ALPHAGRAD_COS_LOG_EVERY")}
    yield
    (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
     ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR) = saved
    for k, v in env_saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_training_sparsity_is_accepted_without_any_guard():
    """owner ruling 2026-09-03, ticket dsnn-3qm.15: the gradient-coverage guard is gone and
    nothing gates a reward. A non-zero weight is accepted as-is; the
    hackability warning on env._SPARSITY_STATS is the only defence."""
    w = ppo.configure_sparsity(_args(sparsity_weight=1.0))
    assert w == 1.0
    assert ppo.SPARSITY_HEAD in ppo.VALUE_HEAD_ATTRS


def test_logging_sparsity_is_allowed():
    """--sparsity-log records the channel without training on it: nothing can
    be reward-hacked through a weight of 0."""
    w = ppo.configure_sparsity(_args(sparsity_log=True))
    assert w == 0.0
    assert os.environ["ALPHAGRAD_SPARSITY"] == "1"
    assert envmod.sparsity_enabled()
    assert ppo.SPARSITY_HEAD not in ppo.VALUE_HEAD_ATTRS


# --------------------------------------------------------- 6. flag-off / on
def test_flag_off_is_inert():
    w = ppo.configure_sparsity(_args())
    assert w == 0.0
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality")
    assert ppo.SPARSITY_HEAD not in ppo.VALUE_HEAD_ATTRS
    assert ppo.NUM_VALUE_HEADS == 3
    # explicitly OFF, so a respawned measure actor cannot inherit a stale "on"
    assert os.environ["ALPHAGRAD_SPARSITY"] == "0"
    assert not envmod.sparsity_enabled()
    # the symlog exempt set does not gain the index
    ppo.configure_symlog(_args())
    assert SSLOT not in ppo._NO_SYMLOG_REWARD_INDICES


def test_weight_appends_exactly_one_head_at_the_end():
    ppo.configure_sparsity(_args(sparsity_weight=2.5))
    assert ppo.HEAD_NAMES[-1] == "sparsity"
    assert ppo.HEAD_REWARD_INDICES[-1] == SSLOT
    assert ppo.VALUE_HEAD_ATTRS[-1] == ppo.SPARSITY_HEAD
    assert ppo.NUM_VALUE_HEADS == len(ppo.HEAD_REWARD_INDICES) == 4
    # IDEMPOTENT
    ppo.configure_sparsity(_args(sparsity_weight=2.5))
    assert ppo.NUM_VALUE_HEADS == 4


def test_the_head_order_is_deterministic_after_fidelity():
    ppo.configure_fidelity(_args(fidelity_weight=1.0))
    ppo.configure_sparsity(_args(sparsity_weight=1.0))
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality", "fidelity",
                              "sparsity")
    assert ppo.HEAD_REWARD_INDICES[-2:] == (
        int(envmod.REWARD_INDEX["fidelity"]), SSLOT)


def test_weight_exempts_it_from_symlog():
    ppo.configure_symlog(_args(sparsity_weight=1.0))
    assert SSLOT in ppo._NO_SYMLOG_REWARD_INDICES
    assert bool(ppo._NO_SYMLOG_MASK_NP[SSLOT])


def test_head_weights_carry_the_lambda():
    ppo.configure_sparsity(_args(sparsity_weight=3.0))
    a = _args(sparsity_weight=3.0, rewards=["cmp"], lambda_cmp=1.0,
              lambda_mem=0.0, lambda_acc=0.0)
    w = ppo._build_head_weights(a)
    assert w.shape == (ppo.NUM_VALUE_HEADS,)
    assert w[ppo.HEAD_NAMES.index("sparsity")] == np.float32(3.0)


def test_reward_scaling_lambda_lands_on_slot_ten():
    a = argparse.Namespace(rewards=[], cmp_type="flops", mem_type="peak_memory",
                           lambda_cmp=0.0, lambda_mem=0.0, lambda_acc=0.0,
                           lambda_frob=0.0, sparsity_weight=0.75,
                           measure_latency=False)
    w = rs.build_reward_weights(a)
    assert w.shape == (rs.NUM_REWARDS,)
    assert w[rs.SPARSITY_IDX] == np.float32(0.75)


def test_all_channels_mode_does_not_silently_weight_sparsity():
    """"all channels" must not mean "reward deleting computation". Sparsity is
    reachable only by naming it."""
    os.environ["ALPHAGRAD_SPARSITY"] = "1"
    try:
        a = argparse.Namespace(rewards=["all"], cmp_type="flops",
                               mem_type="peak_memory", lambda_cmp=0.0,
                               lambda_mem=0.0, lambda_acc=0.0, lambda_frob=0.0,
                               sparsity_weight=0.0, measure_latency=False)
        w = rs.build_reward_weights(a)
        assert w[rs.SPARSITY_IDX] == 0.0
    finally:
        os.environ["ALPHAGRAD_SPARSITY"] = "0"
