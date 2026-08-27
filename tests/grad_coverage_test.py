"""GRADIENT COVERAGE: the arithmetic, the wire, the guard, and FLAG-OFF
BIT-IDENTITY.

Context (docs/UNBIASED_PARETO_AND_MEASUREMENT.md sec 10). On the TLM target a
single skipped face removes 62-73% of the backward pass and zeroes the
gradient of most trainable parameters, and the 200-step Adam quality probe
prices that at 0.02% (0.9258 against an exact 0.9260) -- at EVERY horizon
tested, because it measures single-batch overfitting, which a small subset of
the parameters achieves alone. Coverage measures the freezing directly.

What is pinned here:

1. THE ARITHMETIC of ``env._grad_coverage``, including the epsilon policy and
   the one-slot ``channel`` encoding, and a REPLAY of the recorded forensics
   norms (``run_analysis/landscape/face_forensics.json``) so the shipped
   function must produce the same zeroed-leaf sets as the investigation did.
2. THE WIRE. Reward slot 7 is ``grad_coverage``; ``frob_residual`` is a
   back-compat alias for the same index (cpu_approx_pool's sentinel writer,
   alpha0's --lambda-frob and az_gumbel all address it by that name), the
   vector is still 8 wide, and the degenerate sentinel is unchanged -- so
   ``ppo.train_episode``'s ``_is_degen`` still recognises a rejected plan.
3. THE GUARD is OFF at the library level unless explicitly enabled, so a job
   launched before this commit cannot pick it up from a respawned measure
   actor, while ``ppo.py --reject-frozen-grads`` (default True) exports it.
4. FLAG-OFF BIT-IDENTITY. With ``--grad-coverage-weight 0``:
   * the value-head configuration is HEAD's exactly -- 3 heads on
     (latency_ns, peak_memory, quality);
   * the agent carries NO extra pytree leaf (``value_head_gcov is None``);
   * ``configure_symlog``'s exempt set is HEAD's in all three modes;
   * ``_build_reward_weights`` leaves slot 7 at zero.
   And with the flag ON, every PRE-EXISTING leaf of the agent is BITWISE
   unchanged -- the fourth head's key is folded in rather than taken from a
   widened ``jrand.split``, which is what would silently move every seeded
   run's randomness, flag off included.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
# The guard must never be inherited into a unit test.
os.environ.pop("ALPHAGRAD_REJECT_FROZEN_GRADS", None)
os.environ.pop("ALPHAGRAD_GRAD_COVERAGE", None)
os.environ.pop("ALPHAGRAD_GRAD_COVERAGE_WEIGHT", None)

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    COMPUTE_REWARD_INDICES, NUM_REWARDS, REWARD_INDEX, REWARD_NAMES,
    SENTINEL_COST, _SENTINEL_BAD_REWARD, _grad_coverage,
    grad_coverage_enabled, reject_frozen_grads,
)

GSLOT = int(REWARD_INDEX["grad_coverage"])
_HERE = os.path.dirname(os.path.abspath(__file__))
_FORENSICS = os.path.join(
    _HERE, "..", "..", "run_analysis", "landscape", "face_forensics.json")
_FORENSICS_ALT = "/Users/assmuth/dsnn/run_analysis/landscape/face_forensics.json"


# ---------------------------------------------------------------- the wire
def test_slot7_is_grad_coverage_and_frob_is_an_alias():
    assert NUM_REWARDS == 8
    assert REWARD_NAMES[7] == "grad_coverage"
    assert REWARD_INDEX["grad_coverage"] == 7
    # 100+ historical call sites address slot 7 by the old name.
    assert REWARD_INDEX["frob_residual"] == 7


def test_degenerate_sentinel_unchanged():
    """A rejected plan is emitted as EXACTLY this vector, and ppo's
    `_is_degen` (all six COST channels at the sentinel) must still fire."""
    s = np.asarray(_SENTINEL_BAD_REWARD)
    assert s.shape == (NUM_REWARDS,)
    assert np.all(s[list(COMPUTE_REWARD_INDICES)] <= SENTINEL_COST * 0.99)
    assert s[int(REWARD_INDEX["quality"])] == 0.0
    # Slot 7 == -1.0 == "-frac_leaves_zeroed with every leaf frozen", which is
    # exactly the right reading of a rejected plan under the new encoding.
    assert s[GSLOT] == -1.0


# ------------------------------------------------------------ the arithmetic
def test_identity_is_exactly_one_and_zero():
    e = [3.19, 105.4, 0.555, 64.79]
    cov = _grad_coverage(e, e)
    assert cov["min_leaf_ratio"] == 1.0
    assert cov["frac_zeroed"] == 0.0
    assert cov["channel"] == 1.0
    assert cov["n_uncounted"] == 0


def test_zeroed_leaves_are_counted_and_encoded():
    e = [1.0, 2.0, 4.0, 8.0]
    a = [0.0, 2.0, 0.0, 8.0]
    cov = _grad_coverage(a, e)
    assert cov["zeroed"] == [0, 2]
    assert cov["n_zeroed"] == 2
    assert cov["frac_zeroed"] == 0.5
    assert cov["min_leaf_ratio"] == 0.0
    # frac_zeroed > 0 => min_leaf_ratio == 0 BY CONSTRUCTION, which is what
    # makes the one-slot encoding lossless.
    assert cov["channel"] == -0.5


def test_partial_attenuation_is_the_min_ratio():
    e = [1.0, 2.0, 4.0]
    a = [1.0, 0.5, 4.0]
    cov = _grad_coverage(a, e)
    assert cov["frac_zeroed"] == 0.0
    assert cov["min_leaf_ratio"] == pytest.approx(0.25)
    assert cov["channel"] == pytest.approx(0.25)


def test_ratio_is_capped_at_one():
    """An INFLATED leaf is not a coverage problem; the channel stays in [0,1]
    so it needs no symlog and composes with --symlog-channels cost."""
    cov = _grad_coverage([10.0, 1.0], [1.0, 1.0])
    assert cov["min_leaf_ratio"] == 1.0
    assert cov["channel"] == 1.0


def test_epsilon_policy_excludes_leaves_with_no_exact_gradient():
    """A leaf the EXACT reference does not differentiate either cannot be
    'frozen' by an approximation, and dividing by it would be 0/0."""
    e = [1.0, 0.0, 2.0]
    a = [1.0, 0.0, 2.0]
    cov = _grad_coverage(a, e)
    assert cov["n_counted"] == 2
    assert cov["n_uncounted"] == 1
    assert cov["frac_zeroed"] == 0.0          # denominator is COUNTED leaves
    assert cov["min_leaf_ratio"] == 1.0
    # Relative epsilon: 1e-12 of the largest exact norm.
    cov2 = _grad_coverage([0.0, 1.0], [1e-9, 1e9])
    assert cov2["n_uncounted"] == 1
    assert cov2["frac_zeroed"] == 0.0


def test_all_leaves_uncounted_is_undefined_and_never_rejected():
    cov = _grad_coverage([0.0, 0.0], [0.0, 0.0])
    assert cov["defined"] is False
    assert cov["frac_zeroed"] == 0.0          # so the guard cannot fire
    assert cov["min_leaf_ratio"] == 1.0


def test_non_finite_gradient_counts_as_frozen():
    """A NaN gradient is not a gradient. `jnp.where` guards the forward pass
    only (feedback_jnp_where_gradient_trap) and a NaN leaf poisons the update
    exactly as a zero one starves it."""
    cov = _grad_coverage([float("nan"), 1.0], [1.0, 1.0])
    assert cov["zeroed"] == [0]
    assert cov["frac_zeroed"] == 0.5


def test_channel_decodes_to_both_numbers():
    """ppo.py's per-plan joint record decodes slot 7 as
    (max(c,0), max(-c,0)) == (min_leaf_ratio, frac_zeroed)."""
    for a, e in (([1.0, 2.0], [1.0, 2.0]),
                 ([0.5, 2.0], [1.0, 2.0]),
                 ([0.0, 2.0], [1.0, 2.0]),
                 ([0.0, 0.0], [1.0, 2.0])):
        cov = _grad_coverage(a, e)
        c = cov["channel"]
        assert max(c, 0.0) == pytest.approx(cov["min_leaf_ratio"])
        assert max(-c, 0.0) == pytest.approx(cov["frac_zeroed"])
        assert -1.0 <= c <= 1.0


# ------------------------------------------------- the forensics, replayed
def _forensics():
    for p in (os.path.normpath(_FORENSICS), _FORENSICS_ALT):
        if os.path.exists(p):
            return json.load(open(p))
    return None


def test_forensics_zeroed_sets_reproduce():
    """THE REGRESSION ANCHOR. The recorded per-leaf norms from
    `ls_face_forensics.py` (job 62415 / T1) fed through the SHIPPED
    `_grad_coverage` must give the same zeroed-leaf sets the investigation
    reported: k24/f0 11, k22/f0 12, k19/f0 14, k13/f1 15 of 16 leaves.
    If this ever disagrees, the implementation changed -- do not adjust the
    expectation."""
    fx = _forensics()
    if fx is None:
        pytest.skip("face_forensics.json not available in this checkout")
    expect = {"skip_k24f0": 11, "skip_k22f0": 12,
              "skip_k19f0": 14, "skip_k13f1": 15}
    e = fx["exact"]["norms"]
    assert _grad_coverage(e, e)["min_leaf_ratio"] == 1.0
    for name, n_zero in expect.items():
        cov = _grad_coverage(fx[name]["norms"], e)
        want = sorted(i for i, (na, ne) in enumerate(zip(fx[name]["norms"], e))
                      if ne > 0 and na == 0.0)
        assert sorted(cov["zeroed"]) == want, name
        assert cov["n_zeroed"] == n_zero, (name, cov["n_zeroed"], n_zero)
        assert cov["min_leaf_ratio"] == 0.0
        assert cov["channel"] < 0.0


# --------------------------------------------------------------- the guard
def test_guard_is_off_at_the_library_level_unless_asked():
    """DEFAULT OFF in env.py, DEFAULT ON at the ppo.py flag. A measure actor
    respawned inside a job that was launched before this commit sees no
    variable and must behave exactly as HEAD did."""
    for k in ("ALPHAGRAD_REJECT_FROZEN_GRADS", "ALPHAGRAD_GRAD_COVERAGE",
              "ALPHAGRAD_GRAD_COVERAGE_WEIGHT"):
        os.environ.pop(k, None)
    assert grad_coverage_enabled() is False
    assert reject_frozen_grads() is False
    os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] = "1"
    try:
        assert grad_coverage_enabled() is True
        assert reject_frozen_grads() is True
    finally:
        os.environ.pop("ALPHAGRAD_REJECT_FROZEN_GRADS", None)
    os.environ["ALPHAGRAD_GRAD_COVERAGE_WEIGHT"] = "0.5"
    try:
        assert grad_coverage_enabled() is True     # the channel turns it on
        assert reject_frozen_grads() is False      # but not the guard
    finally:
        os.environ.pop("ALPHAGRAD_GRAD_COVERAGE_WEIGHT", None)


def test_rejection_is_counted_not_dropped():
    """bug (c) of the docs: the res-slot plans were excluded from the gradient
    with NO counter anywhere, for a whole campaign. A rejection has its own
    poller and it must drain."""
    envmod.consume_frozen_grad_plan_count()
    envmod._record_frozen_grad_plan(
        _grad_coverage([0.0, 1.0], [1.0, 1.0]), list(range(4)))
    assert envmod.consume_frozen_grad_plan_count() == 1
    assert envmod.consume_frozen_grad_plan_count() == 0


# ------------------------------------------------ flag-off bit-identity (d)
def _args(**kw):
    d = dict(reject_frozen_grads=False, grad_coverage_weight=0.0,
             symlog_channels="all", no_symlog=False, reward_mode="additive",
             rewards=["cmp", "mem", "acc"], lambda_cmp=1.0, lambda_mem=1.0,
             lambda_acc=16.0, cmp_type="latency", mem_type="peak_memory")
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(autouse=True)
def _restore_head_config():
    """Every test here mutates ppo's module-level head configuration; put it
    back so test ORDER cannot decide an assertion."""
    saved = (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
             ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR)
    yield
    (ppo.HEAD_REWARD_INDICES, ppo.NUM_VALUE_HEADS, ppo.HEAD_NAMES,
     ppo.VALUE_HEAD_ATTRS, ppo._HEAD_REWARD_INDICES_ARR) = saved
    for k in ("ALPHAGRAD_REJECT_FROZEN_GRADS", "ALPHAGRAD_GRAD_COVERAGE",
              "ALPHAGRAD_GRAD_COVERAGE_WEIGHT"):
        os.environ.pop(k, None)


def test_flag_off_head_configuration_is_heads():
    guard, w = ppo.configure_grad_coverage(_args())
    assert (guard, w) == (False, 0.0)
    assert ppo.NUM_VALUE_HEADS == 3
    assert ppo.HEAD_NAMES == ("latency", "mem", "quality")
    assert ppo.HEAD_REWARD_INDICES == (
        REWARD_INDEX["latency_ns"], REWARD_INDEX["peak_memory"],
        REWARD_INDEX["cosine_sim"])
    assert ppo.VALUE_HEAD_ATTRS == (
        "value_head_flops", "value_head_mem", "value_head_cos")
    assert os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] == "0"


def test_flag_on_appends_a_fourth_head_on_slot_7():
    guard, w = ppo.configure_grad_coverage(
        _args(reject_frozen_grads=True, grad_coverage_weight=2.0))
    assert (guard, w) == (True, 2.0)
    assert ppo.NUM_VALUE_HEADS == 4
    assert ppo.HEAD_NAMES[3] == "grad_cov"
    assert ppo.HEAD_REWARD_INDICES[3] == GSLOT
    assert os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] == "1"
    wts = ppo._build_head_weights(
        _args(reject_frozen_grads=True, grad_coverage_weight=2.0))
    assert wts.shape == (4,)
    assert float(wts[3]) == 2.0
    # ... and the composition the docs state:
    #   lambda_cmp*symlog(lat) + lambda_mem*symlog(mem) + lambda_acc*q + W*cov
    assert [float(x) for x in wts] == [1.0, 1.0, 16.0, 2.0]


def test_display_weights_never_touch_slot_7():
    for a in (_args(), _args(grad_coverage_weight=2.0)):
        ppo.configure_grad_coverage(a)
        assert float(ppo._build_reward_weights(a)[GSLOT]) == 0.0


def test_symlog_exempt_set_is_unchanged_when_the_channel_is_off():
    for mode in ("all", "cost", "none"):
        for rmode in ("additive", "lagrangian"):
            a = _args(symlog_channels=mode, reward_mode=rmode)
            ppo.configure_grad_coverage(a)
            ppo.configure_symlog(a)
            got = set(np.nonzero(ppo._NO_SYMLOG_MASK_NP)[0].tolist())
            want = set()
            if mode == "cost" or (mode == "all" and rmode == "lagrangian"):
                want = {int(REWARD_INDEX["cosine_sim"])}
            assert got == want, (mode, rmode, got, want)


def test_symlog_cost_composes_with_the_coverage_channel():
    """--symlog-channels cost + --grad-coverage-weight: BOTH bounded channels
    stay raw, the two cost channels are still compressed."""
    a = _args(symlog_channels="cost", grad_coverage_weight=1.0)
    ppo.configure_grad_coverage(a)
    assert ppo.configure_symlog(a) == "cost"
    exempt = set(np.nonzero(ppo._NO_SYMLOG_MASK_NP)[0].tolist())
    assert exempt == {int(REWARD_INDEX["cosine_sim"]), GSLOT}
    r = np.zeros((1, 1, NUM_REWARDS), np.float32)
    r[..., REWARD_INDEX["latency_ns"]] = -140470.0
    r[..., REWARD_INDEX["cosine_sim"]] = 0.9260
    r[..., GSLOT] = 0.5325
    out = np.asarray(ppo._symlog_rewards(jnp.asarray(r)))
    assert out[0, 0, GSLOT] == pytest.approx(0.5325, abs=1e-6)
    assert out[0, 0, int(REWARD_INDEX["cosine_sim"])] == pytest.approx(
        0.9260, abs=1e-6)
    assert abs(out[0, 0, int(REWARD_INDEX["latency_ns"])]) < 20.0


def _tiny_agent_args():
    """The smallest agent `_build_agent` will make. Only the value heads and
    `pref_proj` matter here."""
    p = ppo.make_argparser()
    a = p.parse_args([])
    a.embd_dim = 16
    a.num_heads = 2
    a.num_layers = 1
    a.value_dims = "8"
    a.op_embd_dim = 4
    a.hidden_dim = 16
    a.vocab_size = 32
    a.no_approx_head = True
    return a


def _leaf_map(agent):
    """{path -> bytes} over the agent's ARRAY leaves."""
    import equinox as eqx
    arrays = eqx.filter(agent, eqx.is_array)
    paths, _ = jax.tree_util.tree_flatten_with_path(arrays)
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in paths}


def test_agent_flag_off_has_no_extra_leaf_and_flag_on_perturbs_nothing():
    import jax.random as jrand
    a_off = _tiny_agent_args()
    a_off.grad_coverage_weight = 0.0
    a_off.reject_frozen_grads = False
    ppo.configure_grad_coverage(a_off)
    key = jrand.PRNGKey(0)
    ag_off = ppo._build_agent(a_off, 6, 4, 2, key)
    assert ag_off.value_head_gcov is None
    off = _leaf_map(ag_off)

    a_on = _tiny_agent_args()
    a_on.grad_coverage_weight = 1.0
    a_on.reject_frozen_grads = True
    ppo.configure_grad_coverage(a_on)
    ag_on = ppo._build_agent(a_on, 6, 4, 2, key)
    assert ag_on.value_head_gcov is not None
    on = _leaf_map(ag_on)

    # THE BIT-IDENTITY CLAIM. Every leaf that exists in the flag-off agent
    # exists in the flag-on agent with the SAME BYTES -- except pref_proj,
    # whose INPUT width is NUM_VALUE_HEADS and legitimately grows by one
    # column. Nothing else may move: the fourth head's key is folded in, not
    # taken from a widened split.
    moved = []
    for k, v in off.items():
        if "pref_proj" in k:
            continue
        w = on.get(k)
        assert w is not None, f"leaf disappeared under the flag: {k}"
        if v.shape != w.shape or not np.array_equal(v, w):
            moved.append(k)
    assert not moved, moved
    # The ONLY new leaves are the fourth head's own.
    import equinox as eqx
    n_new = len(jax.tree_util.tree_leaves(
        eqx.filter(ag_on.value_head_gcov, eqx.is_array)))
    assert n_new > 0
    assert len(on) == len(off) + n_new
    assert {k for k in on if k not in off} == {
        k for k in on if "value_head_gcov" in k}


def test_value_vector_width_follows_the_head_count():
    import jax.random as jrand
    a = _tiny_agent_args()
    a.grad_coverage_weight = 0.0
    a.reject_frozen_grads = False
    ppo.configure_grad_coverage(a)
    ag = ppo._build_agent(a, 6, 4, 2, jrand.PRNGKey(0))
    assert ag.num_value_heads == 3
    a2 = _tiny_agent_args()
    a2.grad_coverage_weight = 1.0
    a2.reject_frozen_grads = True
    ppo.configure_grad_coverage(a2)
    ag2 = ppo._build_agent(a2, 6, 4, 2, jrand.PRNGKey(0))
    assert ag2.num_value_heads == 4
