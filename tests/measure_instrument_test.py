"""ONE INSTRUMENT for the candidate and its paired rev-exact reference.

Until 2026-09-04 this module (as tests/quality_gate_test.py) pinned the
additive quality gate: ALPHAGRAD_QUALITY_GATE_MIN, `_apply_quality_gate`,
`_exact_cost_reference`, the per-order floor and the clamp telemetry. The gate
was deleted by owner ruling (ticket dsnn-3qm.9; the quality floor is a reward
channel option, --quality-floor, not a clamp on the cost channels), and with
it every test that asserted a clamp, a floor cache, a fail-open reference or
the legacy 3x20 audit. What survives is the property the gate's fix taught:

THE REFERENCE AND THE PLAN SHARE THE INSTRUMENT. The gate's floor used to be
timed by a throughput protocol (3 laps of 20) while the plan it floored was
timed by the campaign loop at --latency-inner-reps 5, so the floor read
13-15% below an honest exact plan for the whole v57-v66 campaign. The paired
reference of ticket .9 is measured by `_campaign_measure_cost`, which is
`_time_one_rep` in the same loop the candidate runs, with the same (points,
reps, inner, warmup) and the same median -- these tests pin that, plus the
warmup and cold-start properties the instrument carries.
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest

from alphagrad.approx import env as env_mod


def _campaign_loop_reference(ex, eval_args_all, devices, inner, warmup, n_reps):
    """What ``_callback``'s measurement loop does, written out: warmup +
    n_points x n_reps calls to _time_one_rep, medianed. The reference must
    equal THIS, which is the whole point."""
    import jax
    lat, peak = [], []
    for eval_args in eval_args_all:
        for _ in range(warmup):
            jax.block_until_ready(ex(*eval_args))
        for _ in range(n_reps):
            _l, _p, _s, _o = env_mod._time_one_rep(ex, eval_args, devices, inner)
            lat.append(_l)
            peak.append(_p)
    return (float(env_mod._aggregate_samples(lat, want_top_quartile=True)),
            float(env_mod._aggregate_samples(peak, want_top_quartile=True)))


def test_reference_and_plan_share_one_instrument(monkeypatch):
    """Deterministic pin: with the per-rep timer faked to a function of the
    instrument parameters, the paired reference and the campaign loop
    produce the SAME number. If a future edit re-forks them (different
    inner-reps, different aggregation, a second timing protocol) this
    diverges."""
    seen = []

    def _fake_rep(ex, eval_args, devices, inner):
        seen.append(inner)
        # per-call latency falls as dispatch is amortised over `inner`
        # executions -- exactly the effect that made the two protocols differ
        return 1_000_000.0 / inner, 4.0e9, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    eval_args_all = [(i,) for i in range(5)]

    ref = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=5, warmup=0, n_reps=4)
    plan = _campaign_loop_reference(
        lambda *a: None, eval_args_all, [], inner=5, warmup=0, n_reps=4)

    assert ref == plan
    assert ref[0] == pytest.approx(200_000.0)     # 1e6 / 5
    assert len(seen) == 40                        # 5 points x 4 reps, twice
    assert set(seen) == {5}                       # the run's own inner-reps


def test_a_throughput_protocol_would_read_low(monkeypatch):
    """The historical bug, reproduced: timing one side at inner=20 (the
    legacy 3x20 protocol) while the other is timed at inner=5 under-reads
    it. Documents WHY the parameters must be shared, not merely the code."""
    def _fake_rep(ex, eval_args, devices, inner):
        return 1_000_000.0 / inner, 4.0e9, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    eval_args_all = [(0,)]
    plan = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=5, warmup=0, n_reps=4)[0]
    legacy = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=20, warmup=0, n_reps=4)[0]
    assert legacy < plan
    assert legacy / plan == pytest.approx(0.25)


def test_reference_matches_the_campaign_measurement_of_the_same_executable():
    """INTEGRATION (CPU, real compile): the number the paired reference is
    charged equals the campaign-path measurement of the same executable.
    Same executable, same eval args, same instrument -- so the only
    difference left is timer noise."""
    import jax
    from graphax import jacve

    rng = np.random.RandomState(0)
    W = jnp.asarray(rng.randn(32, 16).astype(np.float32))
    x = jnp.asarray(rng.randn(16).astype(np.float32))

    def loss(x, W):
        y = jnp.tanh(W @ x)
        return jnp.sum(y * y)

    ex = jax.jit(
        jacve(loss, "rev", argnums=(1,), has_aux=False,
              sparse_representation=False),
        keep_unused=True,
    ).lower(x, W).compile()

    devices = list(jax.local_devices())
    eval_args_all = [[x, W] for _ in range(3)]
    inner, warmup, n_reps = 5, 1, 4

    plan_lat, plan_peak = _campaign_loop_reference(
        ex, eval_args_all, devices, inner, warmup, n_reps)
    ref_lat, ref_peak = env_mod._campaign_measure_cost(
        ex, eval_args_all, devices, inner, warmup, n_reps)

    assert plan_lat > 0.0 and ref_lat > 0.0
    # Generous: a shared CPU box moves ~20% between back-to-back windows. The
    # bug this guards was a SYSTEMATIC 13-15% offset in one direction that
    # survived every run; a real re-fork of the protocol (inner 5 vs 20) shows
    # up as ~4x here, not as noise.
    assert ref_lat / plan_lat == pytest.approx(1.0, rel=0.5)
    assert ref_peak == pytest.approx(plan_peak, rel=0.5, abs=1.0)


def test_campaign_measure_cost_honours_warmup_and_reps(monkeypatch):
    """Instrument parameters are actually plumbed: warmup calls happen outside
    the timing window, and points x reps samples are taken."""
    import jax
    calls = {"warm": 0, "timed": 0}

    def _fake_rep(ex, eval_args, devices, inner):
        calls["timed"] += 1
        return 100.0, 1.0, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)

    def _ex(*_a):
        calls["warm"] += 1
        return None

    env_mod._campaign_measure_cost(
        _ex, [(0,), (1,), (2,)], [], inner=5, warmup=2, n_reps=4)
    assert calls["timed"] == 12               # 3 points x 4 reps
    assert calls["warm"] == 6                 # 3 points x 2 warmups


# ---------------------------------------------------------------------------
# COLD-START (2026-08-26). Without a warmup the FIRST timed sample of a plan is
# the FIRST EXECUTION of a freshly compiled executable, so first-touch lands
# inside a timed window. The campaign path's 5x4=20-sample median absorbs one
# cold sample almost completely, but a reference is a fresh compile the first
# time it is measured and any low-sample paired harness is fully exposed.
# ALPHAGRAD_MEASURE_WARMUP (default 1) closes it for both.
# ---------------------------------------------------------------------------


def test_warmup_defaults_to_one_and_is_overridable(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.delenv("ALPHAGRAD_MEASURE_WARMUP", raising=False)
    assert env_mod._resolve_warmup(NS()) == 1
    assert env_mod._resolve_warmup(NS(latency_warmup=0)) == 1
    # an explicit config value always wins
    assert env_mod._resolve_warmup(NS(latency_warmup=3)) == 3
    # ...and the old behaviour is one env var away
    monkeypatch.setenv("ALPHAGRAD_MEASURE_WARMUP", "0")
    assert env_mod._resolve_warmup(NS()) == 0
    assert env_mod._resolve_warmup(NS(latency_warmup=0)) == 0
    assert env_mod._resolve_warmup(NS(latency_warmup=3)) == 3


def test_first_timed_sample_is_not_the_first_execution(monkeypatch):
    """The property that matters, pinned directly: at least one execution of
    the executable happens BEFORE the first timed rep of every data point."""
    import jax
    log = []

    def _ex(*_a):
        log.append("exec")
        return None

    def _fake_rep(ex, eval_args, devices, inner):
        log.append("TIMED")
        return 100.0, 1.0, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)

    env_mod._campaign_measure_cost(
        _ex, [(0,), (1,)], [], inner=5, warmup=1, n_reps=4)

    assert log[0] == "exec"                    # warmup ran first
    assert log.count("TIMED") == 8             # 2 points x 4 reps
    first_timed = log.index("TIMED")
    assert "exec" in log[:first_timed]
    second_point = log.index("exec", first_timed)
    assert log[second_point + 1] == "TIMED"


def test_reference_inherits_the_warmup(monkeypatch):
    """The reference is a FRESH compile the first time it is measured, so it
    is the most cold-start-exposed measurement in the loop. Routing it
    through _campaign_measure_cost makes it inherit the warmup."""
    import jax
    execs = {"n": 0}

    def _ex(*_a):
        execs["n"] += 1
        return None

    monkeypatch.setattr(
        env_mod, "_time_one_rep",
        lambda *a, **k: (100.0, 1.0, "runtime_delta", None))
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)

    env_mod._campaign_measure_cost(
        _ex, [(0,)] * 5, [], inner=5, warmup=1, n_reps=4)
    assert execs["n"] == 5                     # one untimed warmup per point


def test_median_of_twenty_absorbs_one_cold_sample():
    """Quantified: the campaign path pools points x reps into ONE median, so
    a single inflated sample cannot move the reported value materially --
    which is why the 20-sample campaign numbers are not cold-contaminated even
    though the first execution was timed."""
    steady = [155.0 + 0.5 * ((i % 7) - 3) for i in range(20)]
    base = float(np.median(steady))
    for blow_up in (2, 5, 10):
        cold = list(steady)
        cold[0] = steady[0] * blow_up
        moved = float(np.median(cold))
        assert abs(moved - base) / base < 0.01     # under 1%, every time
    # The protection is the MEDIAN, not the sample count: the same twenty
    # samples averaged instead of medianed move ~45% on one cold reading, and
    # a single-sample harness moves by the full blow-up factor. That is the
    # difference between the campaign path (median, unexposed) and a paired
    # sweep script that averages a handful of trials (fully exposed).
    cold = list(steady)
    cold[0] = steady[0] * 10
    assert abs(float(np.mean(cold)) - float(np.mean(steady))) \
        / float(np.mean(steady)) > 0.4


def test_the_quality_gate_is_gone():
    """A SKIP IS A FAILURE, and so is a resurrection: the clamp path and its
    env var must not come back under any name."""
    for name in ("_apply_quality_gate", "_exact_cost_reference",
                 "_order_floor", "_measure_exec_cost", "_QUALITY_GATE_STATS",
                 "_COST_REF", "_ORDER_FLOOR_CACHE"):
        assert not hasattr(env_mod, name), name
    src = open(env_mod.__file__).read()
    # The name survives only in the block comment that records the deletion.
    assert src.count("ALPHAGRAD_QUALITY_GATE_MIN") == 1
    assert "ALPHAGRAD_QUALITY_GATE_MIN" not in os.environ
