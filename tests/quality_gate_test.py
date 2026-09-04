"""Pin the ADDITIVE quality gate (_apply_quality_gate / _exact_cost_reference).

Sign-trap regression: cost channels are PENALTIES, so the gate must FLOOR a
destroyed plan's costs at the exact-reverse reference (destruction pays what
exact pays), never scale them toward zero (which would reward destruction).
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest

# The gate's memory floor is a runtime watermark, so it only composes with
# the pre-.49 channel; under --mem-channel temp (the default since ticket
# dsnn-3qm.49) a clamp is a MemChannelFault (see tests/mem_channel_test.py).
os.environ["ALPHAGRAD_MEM_CHANNEL"] = "watermark"

from alphagrad.approx import env as env_mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    env_mod._COST_REF.clear()
    env_mod._QUALITY_GATE_STATS["clamps"] = 0
    yield
    env_mod._COST_REF.clear()


def _seed_ref(lat, mem):
    env_mod._COST_REF["ref"] = (lat, mem)


def test_gate_off_is_identity(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_QUALITY_GATE_MIN", raising=False)
    _seed_ref(1e9, 1e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_destroyed_plan_pays_reference(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert lat == 132e3 and mem == 4.0e9          # floored, not zeroed
    assert env_mod._QUALITY_GATE_STATS["clamps"] == 1


def test_good_plan_keeps_its_real_costs(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    # pulldown-class plan: genuinely fast AND high quality -> untouched
    lat, mem = env_mod._apply_quality_gate(
        124e3, 3.9e9, 0.99997, True, True, None, None)
    assert (lat, mem) == (124e3, 3.9e9)
    assert env_mod._QUALITY_GATE_STATS["clamps"] == 0


def test_slower_than_ref_stays_slower(monkeypatch):
    """max() only floors: a destroyed plan that is ALSO slow keeps its own
    worse cost -- the gate never improves anyone."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        400e3, 9.9e9, 0.0, True, True, None, None)
    assert (lat, mem) == (400e3, 9.9e9)


def test_unmeasured_quality_never_gates(monkeypatch):
    """Non-terminal steps / steps without a quality sample must pass through
    even under the env var -- quality 0.0 there means NOT MEASURED."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, False, False, None, None)
    assert (lat, mem) == (37e3, 1e6)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, False, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_reference_failure_fails_open(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    env_mod._COST_REF["ref"] = None                # reference build failed
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None)
    assert (lat, mem) == (37e3, 1e6)


def test_reference_measures_on_a_real_target():
    """End-to-end (CPU): the exact-rev reference builds and returns positive
    latency for a tiny scalar-loss target through the real compile path."""
    import jax
    from types import SimpleNamespace as NS
    W = jnp.asarray(np.random.RandomState(0).randn(8, 4).astype(np.float32))
    x = jnp.asarray(np.random.RandomState(1).randn(4).astype(np.float32))

    def loss(x, W):
        y = W @ x
        return jnp.sum(y * y)

    cfg = NS(target_fun=loss, argnums=(1,), has_aux=False, sparse=False)
    ref = env_mod._exact_cost_reference(cfg, [x, W])
    assert ref is not None
    lat, mem = ref
    assert lat > 0.0 and mem >= 0.0


def test_order_floor_preferred_over_rev_reference(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)                     # global rev reference
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None,
        order_floor_fn=lambda: (67e6, 9.0e9))   # this order, done exactly
    assert lat == 67e6 and mem == 9.0e9         # order floor wins


def test_order_floor_failure_falls_back_to_rev(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)

    def _boom():
        raise RuntimeError("compile died")

    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None, order_floor_fn=_boom)
    assert lat == 132e3 and mem == 4.0e9        # fell back, still floored


def test_oomed_order_floor_carries_alloc_into_mem_floor(monkeypatch):
    """destroy + un-compilable order must not earn the tiny rev floor: the
    failed allocation becomes the memory floor."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    _seed_ref(132e3, 4.0e9)

    def _oom():
        raise RuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
            "34.05GiB. [tf-allocator-allocation-error='']")

    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None, order_floor_fn=_oom)
    assert lat == 132e3
    assert mem == pytest.approx(34.05 * 2**30)


# ---------------------------------------------------------------------------
# ONE INSTRUMENT (2026-08-26). The gate's floor used to be timed by
# _measure_exec_cost (median of 3 laps of 20 back-to-back executions) while the
# plan it floors was timed by the campaign loop at --latency-inner-reps 5. The
# throughput protocol amortises per-call dispatch far better, so the floor read
# 13-15% BELOW what an honest exact plan was charged in the same run -- i.e.
# the mechanism that exists to make destruction cost-neutral was paying
# destruction a guaranteed -13% bonus, in every run of the v57-v66 campaign.
# These tests pin the floor and the plan to the SAME function with the SAME
# parameters.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_instrument_state():
    env_mod._ORDER_FLOOR_CACHE.clear()
    del env_mod._FLOOR_AUDIT_DONE[:]
    env_mod._QUALITY_GATE_STATS["unclamped"] = []
    env_mod._QUALITY_GATE_STATS["instrument"] = None
    yield
    env_mod._ORDER_FLOOR_CACHE.clear()
    del env_mod._FLOOR_AUDIT_DONE[:]
    env_mod._QUALITY_GATE_STATS["unclamped"] = []
    env_mod._QUALITY_GATE_STATS["instrument"] = None


def _campaign_loop_reference(ex, eval_args_all, devices, inner, warmup, n_reps):
    """What ``_callback``'s measurement loop does, written out: warmup +
    n_points x n_reps calls to _time_one_rep, medianed. The floor must equal
    THIS, which is the whole point of the fix."""
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


def test_floor_and_plan_share_one_instrument(monkeypatch):
    """Deterministic pin: with the per-rep timer faked to a function of the
    instrument parameters, the gate floor and the campaign loop produce the
    SAME number. If a future edit re-forks them (different inner-reps,
    different aggregation, a second timing protocol) this diverges."""
    seen = []

    def _fake_rep(ex, eval_args, devices, inner):
        seen.append(inner)
        # per-call latency falls as dispatch is amortised over `inner`
        # executions -- exactly the effect that made the two protocols differ
        return 1_000_000.0 / inner, 4.0e9, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    eval_args_all = [(i,) for i in range(5)]

    floor = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=5, warmup=0, n_reps=4)
    plan = _campaign_loop_reference(
        lambda *a: None, eval_args_all, [], inner=5, warmup=0, n_reps=4)

    assert floor == plan
    assert floor[0] == pytest.approx(200_000.0)   # 1e6 / 5
    assert len(seen) == 40                        # 5 points x 4 reps, twice
    assert set(seen) == {5}                       # the run's own inner-reps


def test_legacy_instrument_would_have_differed(monkeypatch):
    """The bug, reproduced: timing the floor at inner=20 (the legacy 3x20
    protocol) while the plan is timed at inner=5 under-reads the floor. This
    test documents WHY the parameters must be shared, not merely the code."""
    def _fake_rep(ex, eval_args, devices, inner):
        return 1_000_000.0 / inner, 4.0e9, "runtime_delta", None

    monkeypatch.setattr(env_mod, "_time_one_rep", _fake_rep)
    eval_args_all = [(0,)]
    plan = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=5, warmup=0, n_reps=4)[0]
    legacy = env_mod._campaign_measure_cost(
        object(), eval_args_all, [], inner=20, warmup=0, n_reps=4)[0]
    assert legacy < plan                       # the floor read LOW
    assert legacy / plan == pytest.approx(0.25)


def test_clamped_plan_never_calls_the_legacy_instrument(monkeypatch):
    """A gate clamp must not reach _measure_exec_cost. Blowing up if it does
    is the cheapest possible regression detector for a re-fork."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")

    def _forbidden(*_a, **_k):
        raise AssertionError(
            "the quality gate floor used the LEGACY 3x20 instrument")

    monkeypatch.setattr(env_mod, "_measure_exec_cost", _forbidden)
    lat, mem = env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None,
        order_floor_fn=lambda: (155e3, 4.0e9))
    assert lat == 155e3 and mem == 4.0e9


def test_gate_min_zero_disables_the_gate_entirely(monkeypatch):
    """Run R1 needs ALPHAGRAD_QUALITY_GATE_MIN=0 to be a true no-op: no floor
    measured, no compile triggered, both channels byte-identical."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0")

    def _boom(*_a, **_k):
        raise AssertionError("the gate measured a floor while disabled")

    for q in (0.0, -1.0, 0.9):
        lat, mem = env_mod._apply_quality_gate(
            37e3, 1e6, q, True, True, None, None,
            order_floor_fn=_boom, ref_measure_fn=_boom)
        assert (lat, mem) == (37e3, 1e6)
    assert env_mod._QUALITY_GATE_STATS["clamps"] == 0
    # ...and the empty string (an unset-but-exported env var) behaves the same
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "")
    assert env_mod._apply_quality_gate(
        37e3, 1e6, 0.0, True, True, None, None,
        order_floor_fn=_boom) == (37e3, 1e6)


def test_order_floor_is_measured_once_per_order(monkeypatch):
    """Per-(order, shapes, device) memoisation: a repeat offender order pays
    one compile + one measurement, not one per clamp."""
    calls = {"compile": 0, "measure": 0}

    def _compile():
        calls["compile"] += 1
        return "EX"

    def _measure(ex):
        calls["measure"] += 1
        return (155e3, 4.0e9)

    key = b"\x01\x02\x03\x04order-A"
    for _ in range(10):
        assert env_mod._order_floor(key, _compile, _measure) == (155e3, 4.0e9)
    assert calls == {"compile": 1, "measure": 1}
    # a DIFFERENT order is a different key -> measured on its own
    env_mod._order_floor(b"\x09order-B", _compile, _measure)
    assert calls == {"compile": 2, "measure": 2}


def test_floor_audit_runs_once_and_reports_the_ratio(capsys):
    """The one-shot audit re-times the first floor with the legacy protocol so
    every run records the size of the gap it used to pay."""
    env_mod._order_floor(
        b"order-A", lambda: "EX", lambda _ex: (155e3, 4.0e9),
        legacy_measure_fn=lambda _ex: (135e3, 3.95e9))
    out = capsys.readouterr().out
    assert "floor AUDIT" in out
    assert "0.871" in out                      # 135 / 155
    env_mod._order_floor(
        b"order-B", lambda: "EX", lambda _ex: (155e3, 4.0e9),
        legacy_measure_fn=lambda _ex: (135e3, 3.95e9))
    assert "floor AUDIT" not in capsys.readouterr().out


def test_clamp_telemetry_reports_floor_ratio_and_instrument(monkeypatch, capsys):
    """The clamp line must carry the floor, the instrument parameters and
    floor / median-unclamped -- the three numbers a future regression shows up
    in."""
    monkeypatch.setenv("ALPHAGRAD_QUALITY_GATE_MIN", "0.05")
    env_mod._QUALITY_GATE_STATS["instrument"] = {
        "points": 5, "reps": 4, "inner": 5, "warmup": 0}
    # honest plans the gate let through
    for _ in range(8):
        env_mod._apply_quality_gate(
            160e3, 4.0e9, 0.9, True, True, None, None)
    assert len(env_mod._QUALITY_GATE_STATS["unclamped"]) == 8
    capsys.readouterr()
    env_mod._apply_quality_gate(
        60e3, 1e6, 0.0, True, True, None, None,
        order_floor_fn=lambda: (160e3, 4.0e9))
    out = capsys.readouterr().out
    assert "floor/median-unclamped=1.000" in out
    assert "inner=5" in out and "points=5" in out and "reps=4" in out


def test_gate_floor_matches_the_campaign_measurement_of_the_same_order():
    """INTEGRATION (CPU, real compile): the floor a clamped plan is charged
    equals the campaign-path measurement of the exact plan for that same
    order. Same executable, same eval args, same instrument -- so the only
    difference left is timer noise."""
    import jax
    from types import SimpleNamespace as NS
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

    # what the campaign loop would print for this plan
    plan_lat, plan_peak = _campaign_loop_reference(
        ex, eval_args_all, devices, inner, warmup, n_reps)
    # what the gate charges a plan that destroyed this same order
    floor_lat, floor_peak = env_mod._order_floor(
        b"integration-order", lambda: ex,
        lambda _ex: env_mod._campaign_measure_cost(
            _ex, eval_args_all, devices, inner, warmup, n_reps))

    assert plan_lat > 0.0 and floor_lat > 0.0
    # Generous: a shared CPU box moves ~20% between back-to-back windows. The
    # bug this guards was a SYSTEMATIC 13-15% offset in one direction that
    # survived every run; a real re-fork of the protocol (inner 5 vs 20) shows
    # up as ~4x here, not as noise.
    assert floor_lat / plan_lat == pytest.approx(1.0, rel=0.5)
    assert floor_peak == pytest.approx(plan_peak, rel=0.5, abs=1.0)


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
# cold sample almost completely, but the gate floor is a fresh compile every
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
    # every point warms up before its own first timed rep
    first_timed = log.index("TIMED")
    assert "exec" in log[:first_timed]
    second_point = log.index("exec", first_timed)
    assert log[second_point + 1] == "TIMED"


def test_gate_floor_inherits_the_warmup(monkeypatch):
    """(d) The floor is a FRESH compile each time it is measured, so it is the
    most cold-start-exposed measurement in the loop. Routing it through
    _campaign_measure_cost makes it inherit the warmup automatically."""
    import jax
    execs = {"n": 0}

    def _ex(*_a):
        execs["n"] += 1
        return None

    monkeypatch.setattr(
        env_mod, "_time_one_rep",
        lambda *a, **k: (100.0, 1.0, "runtime_delta", None))
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)

    env_mod._order_floor(
        b"warm-order", lambda: _ex,
        lambda ex: env_mod._campaign_measure_cost(
            ex, [(0,)] * 5, [], inner=5, warmup=1, n_reps=4))
    assert execs["n"] == 5                     # one untimed warmup per point


def test_median_of_twenty_absorbs_one_cold_sample():
    """(b) Quantified: the campaign path pools points x reps into ONE median,
    so a single inflated sample cannot move the reported value materially --
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
