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
`_time_one_rep` in the same loop the candidate runs, over the same eval args
with the same (inner, warmup) and the same median -- these tests pin that,
plus the warmup and cold-start properties the instrument carries.

SINCE 2026-09-14 THE POINTS AND THE REPS ARE THE REFERENCE'S OWN (owner
ruling; `EnvConfig.ref_num_data_points` / `ref_reps_per_point`, defaults 5
and 32). That is not a second protocol: the per-window instrument is
unchanged and only the SAMPLE COUNT differs, because the two halves of the
pair are 150x apart in cost and the shared budget left the cheap half with a
tenth of a second of integration. The last two tests in this module pin the
fork end to end and pin that the reference's compile key is untouched by it.
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
    difference left is timer noise.

    STILL MEANINGFUL AFTER THE 2026-09-14 FORK. The counts are passed in
    here, not read off a config, so this compares the two CODE PATHS at one
    budget, which is the property it has always pinned. The fork changes how
    many samples the reference takes, not what one sample means, and the
    medians agree either way -- the test that the reference runs its OWN
    counts is `test_the_reference_runs_its_own_points_and_reps` below."""
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


# ==========================================================================
# THE REFERENCE'S OWN BUDGET (owner ruling 2026-09-14)
# ==========================================================================
#
# Until this ruling `_campaign_measure_cost` was handed the CANDIDATE's
# `num_data_points` x `reps_per_point`. The two halves of the pair are not
# the same size: on the transformer arm one candidate execution is 18.2 ms
# and one reference execution is 0.121 ms, so the shared 5 x 4 budget bought
# the candidate 18.3 s of integration and the reference 0.12 s. Measured over
# 128 repeats of one plan (job 65468) the candidate reading has a coefficient
# of variation of 0.56 percent and the reference 4.53 percent, and since the
# two are independent the paired log ratio scatters by 0.045 nats, essentially
# all of it the reference's.
#
# `EnvConfig.ref_num_data_points` / `ref_reps_per_point` (defaults 5 and 32)
# fork the POINTS and the REPS only. The inner reps, the warmup, the eval
# args and the median stay shared, so the tests above still describe one
# instrument.


def _one_instrument_toy_env(**kw):
    """A 16-wide scalar loss, measured on the CPU, terminal rewards only.

    Small on purpose: these tests count TIMED WINDOWS, so the cheapest graph
    that still has several eliminable vertices is the right one.
    """
    import jax
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    kw.setdefault("measure_latency", True)
    kw.setdefault("terminal_rewards_only", True)
    kw.setdefault("latency_inner_reps", 1)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy, **kw)


def _walk(env, order):
    """Run `order` to the end with no rule and no face action."""
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX, StepAction)

    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    faces = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
    skips = jnp.zeros((MAX_FACES,), jnp.int32)
    for v in order:
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules, faces, skips),
        ).state
    return state


@pytest.fixture
def _paired_log_cpu(monkeypatch):
    """The cost form and the cheap channels these two tests measure under."""
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "1")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    yield
    env_mod.consume_plan_records()


def test_the_reference_runs_its_own_points_and_reps(_paired_log_cpu,
                                                    monkeypatch):
    """END TO END: the reference takes `ref_points x ref_reps` timed windows
    and the candidate takes `points x reps`, in the SAME terminal callback.

    Counted the way `test_a_throughput_protocol_would_read_low` counts, by
    intercepting `_time_one_rep` -- but through the real callback, so this
    fails if the fork is added to `_campaign_measure_cost` and not actually
    wired at the call site. The candidate's order is the FORWARD one so its
    executable can never be the reference's rev-exact one.
    """
    seen: list[int] = []
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        seen.append(id(ex))
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)

    env = _one_instrument_toy_env(
        num_data_points=2, reps_per_point=2,
        ref_num_data_points=3, ref_reps_per_point=5)
    vs = sorted(int(v) for v in np.asarray(env.valid_vertices))
    assert len(vs) >= 2
    _walk(env, vs)                                   # forward order

    from collections import Counter
    counts = sorted(Counter(seen).values())
    # One executable took 3 x 5 = 15 windows (the reference) and every other
    # one took 2 x 2 = 4 (the candidate). Before the ruling every group was 4.
    assert counts.count(15) == 1, counts
    assert set(counts) == {4, 15}, counts


def test_the_reference_compile_key_does_not_depend_on_the_rep_counts(
        _paired_log_cpu, monkeypatch):
    """`paired_ref_key` is the reverse order, the sparse flag, the argument
    shapes and dtypes and the device -- and nothing else.

    If a future edit folds the budget into the key, every change of
    --ref-reps-per-point would recompile the reference (about 1.9 s) and,
    worse, two arms that measure the same executable would stop sharing the
    cross-actor compile cache. Pinned by running the same plan under two
    different reference budgets and comparing the key the cache was asked
    for.
    """
    from alphagrad.approx.common import compile_cache as _cc

    def _run(ref_reps):
        keys: list[bytes] = []
        real = _cc.cached_compile

        def _spy(cache_key, compile_fn):
            keys.append(bytes(cache_key))
            return real(cache_key, compile_fn)

        monkeypatch.setattr(_cc, "cached_compile", _spy)
        env = _one_instrument_toy_env(
            num_data_points=2, reps_per_point=2,
            ref_num_data_points=2, ref_reps_per_point=ref_reps)
        vs = sorted(int(v) for v in np.asarray(env.valid_vertices))
        _walk(env, vs)
        monkeypatch.setattr(_cc, "cached_compile", real)
        return [k for k in keys if k.startswith(b"paired-ref:")]

    a = _run(3)
    b = _run(17)
    assert a and b
    assert set(a) == set(b)


# ==========================================================================
# THE PER-PLAN TIME BUDGET (owner ruling 2026-09-14)
# ==========================================================================
#
# The candidate used to run a FIXED 5 points x 4 reps x 50 inner = 1005
# executions of the plan whatever the plan cost. On the transformer arm one
# execution is 18.2 ms, so a plan cost 18.3 s to measure and an episode of 16
# plans cost 293 s of its 405 s, to take twenty samples of a reading whose
# coefficient of variation is 0.56 percent.
#
# The counts now come from ONE warm-up execution's measured time: the window
# rule picks the executions per window, the budget picks the number of
# windows, and --num-data-points x --reps-per-point is the CAP on that
# number. The reference keeps its own window count and takes only its inner
# from the same rule, applied to its own execution time.


def test_the_window_rule_reads_the_owners_numbers():
    """clamp(ceil(window / t), 5, --latency-inner-reps), on the two programs
    the ruling names."""
    r = env_mod.resolve_measure_inner
    # The rev-exact reference on the Markowitz order: 121 us. A 50 ms window
    # would hold 413 executions, so it takes the ceiling of 50.
    assert r(121e-6, 0.05, 50) == 50
    # The candidate: 18.2 ms. The window holds 3, so it takes the floor of 5.
    assert r(18.2e-3, 0.05, 50) == 5
    # In between the rule is the arithmetic, not a clamp: a 2 ms program
    # fills a 50 ms window 25 times.
    assert r(2e-3, 0.05, 50) == 25
    # THE FLAG IS THE CEILING. landscape_map runs --latency-inner-reps 5 and
    # the legacy default is 1; both collapse the interval onto themselves, so
    # those callers measure exactly what they measured before the ruling.
    assert r(121e-6, 0.05, 5) == 5
    assert r(121e-6, 0.05, 1) == 1
    # A broken probe takes the ceiling, never an unbounded count.
    assert r(0.0, 0.05, 50) == 50
    assert r(float("nan"), 0.05, 50) == 50


def test_the_budget_picks_the_windows_and_the_flags_are_caps():
    """clamp(round(budget / (inner * t)), 1, num_data_points * reps)."""
    w = env_mod.resolve_measure_windows
    # The transformer candidate at inner 5: 5 x 18.2 ms = 91 ms a window, so
    # one second buys 11 of them -- against the old 20, and against 1005
    # executions rather than 55.
    assert w(18.2e-3, 5, 1.0, 20) == 11
    # A CHEAP plan is capped by --num-data-points x --reps-per-point, which
    # is what makes those two flags caps rather than counts.
    assert w(1e-5, 50, 1.0, 20) == 20
    # A SLOW plan gets ONE window and the owner accepts that: slow runs do
    # not matter, they are too large anyway.
    assert w(10.0, 5, 1.0, 20) == 1
    assert w(0.0, 5, 1.0, 20) == 1


def test_the_two_halves_interleave_instead_of_blocking():
    """A B A B, and proportionally when the counts differ.

    Two blocks put the whole reference measurement after the whole candidate
    measurement, so a clock or thermal excursion during a plan lands on one
    half of the ratio only. The ratio can cancel only the drift both halves
    saw.
    """
    il = env_mod.interleave_windows
    assert il(3, 3) == [0, 1, 0, 1, 0, 1]
    assert il(4, 0) == [0, 0, 0, 0]
    assert il(0, 3) == [1, 1, 1]
    # 11 candidate windows against 160 reference ones is the campaign shape.
    sched = il(11, 160)
    assert len(sched) == 171
    assert sched.count(0) == 11
    # NOT A BLOCK: the candidate's last window is nowhere near its first, and
    # the first reference window comes long before the last candidate one.
    first_c = sched.index(0)
    last_c = len(sched) - 1 - sched[::-1].index(0)
    assert last_c - first_c > 100, sched[:40]
    assert sched.index(1) < last_c


@pytest.fixture
def _fresh_dedupe():
    """No duplicate cache and no episode accounting carried between tests."""
    env_mod._PLAN_DEDUPE.clear()
    env_mod._MEASURE_EPISODE.update(
        {"key": None, "label": None, "n_plans": 0, "n_measured": 0,
         "secs": [], "cand_secs": [], "ref_secs": []})
    yield
    env_mod._PLAN_DEDUPE.clear()
    env_mod._MEASURE_EPISODE.update(
        {"key": None, "label": None, "n_plans": 0, "n_measured": 0,
         "secs": [], "cand_secs": [], "ref_secs": []})


def test_the_budget_bounds_the_windows_end_to_end(_paired_log_cpu,
                                                  _fresh_dedupe, monkeypatch):
    """A budget smaller than one window leaves exactly one window.

    End to end through the real callback, so this fails if the rule is
    implemented and not wired. The toy is microseconds per execution, so a
    budget of 1 nanosecond is "smaller than one window" for it.
    """
    seen: list[int] = []
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        seen.append(inner)
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)
    env = _one_instrument_toy_env(
        num_data_points=2, reps_per_point=2,
        ref_num_data_points=1, ref_reps_per_point=1,
        measure_budget_secs=1e-9)
    vs = sorted(int(v) for v in np.asarray(env.valid_vertices))
    _walk(env, vs)
    # One candidate window plus the reference's own one, which the budget
    # does NOT govern: its count is ref_points x ref_reps.
    assert len(seen) == 2, seen


def test_the_candidate_windows_are_spread_over_the_data_points(
        _paired_log_cpu, _fresh_dedupe, monkeypatch):
    """ROUND-ROBIN, not all on sample 0.

    The old loop ran every rep of point 0, then every rep of point 1. A plan
    that now earns three windows out of a cap of twenty would have spent all
    three on point 0 under that loop.
    """
    points: list[int] = []
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        points.append(float(np.asarray(eval_args[0]).reshape(-1)[0]))
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)
    env = _one_instrument_toy_env(num_data_points=3, reps_per_point=1)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _plan_arrays(len(order))
    # Three data points, each recognisable by its first element.
    samples = (jnp.asarray(
        np.stack([np.full(16, float(k), dtype=np.float32)
                  for k in (1.0, 2.0, 3.0)])),)
    env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order), specs,
        faces, skips, len(order), *samples)
    assert sorted(set(points)) == [1.0, 2.0, 3.0], points


def _plan_arrays(n):
    """An all-exact plan of `n` vertices: no rule, no face action."""
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


# ==========================================================================
# PER-EPISODE DEDUPLICATION (owner ruling 2026-09-14)
# ==========================================================================
#
# At the identity init all 16 terminal plans of an episode are the same
# program. Measuring it sixteen times costs sixteen seconds and learns
# nothing: the reward vector is a function of the plan and of the episode's
# eval samples, and neither changes between the duplicates.


def _measure_once(env, order, samples):
    """One terminal callback, returning (reward vector, windows timed)."""
    specs, faces, skips = _plan_arrays(len(order))
    return env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order), specs,
        faces, skips, len(order), *samples)


def test_a_repeated_plan_in_one_episode_is_measured_once(
        _paired_log_cpu, _fresh_dedupe, monkeypatch):
    """The second identical plan times NO windows and returns the first
    plan's reward vector, bit for bit."""
    n_windows = [0]
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        n_windows[0] += 1
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)
    env = _one_instrument_toy_env(num_data_points=2, reps_per_point=2)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    samples = (jnp.asarray(
        np.stack([np.full(16, 0.5, dtype=np.float32),
                  np.full(16, -0.5, dtype=np.float32)])),)

    *_a, r1 = _measure_once(env, order, samples)
    first = n_windows[0]
    assert first > 0
    *_b, r2 = _measure_once(env, order, samples)
    assert n_windows[0] == first, "the duplicate was measured again"
    assert np.array_equal(np.asarray(r1), np.asarray(r2))


def test_the_cache_never_crosses_an_episode(_paired_log_cpu, _fresh_dedupe,
                                            monkeypatch):
    """New samples are a new episode, and a new episode re-measures.

    The samples ARE the measurement's inputs, so a reading taken against last
    episode's samples is not this episode's reading.
    """
    n_windows = [0]
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        n_windows[0] += 1
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)
    env = _one_instrument_toy_env(num_data_points=1, reps_per_point=1)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))

    def _samples(v):
        return (jnp.asarray(np.full((1, 16), v, dtype=np.float32)),)

    _measure_once(env, order, _samples(0.25))
    first = n_windows[0]
    _measure_once(env, order, _samples(0.75))
    assert n_windows[0] > first, "a new episode's samples were served stale"


def test_the_duplicates_record_says_where_its_numbers_came_from(
        _paired_log_cpu, _fresh_dedupe, monkeypatch):
    """The duplicate IS a plan-log record: same rewards, `measured_from`
    naming the plan that was measured, and no timing fields of its own."""
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    env_mod.consume_plan_records()
    env = _one_instrument_toy_env(num_data_points=2, reps_per_point=2)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    samples = (jnp.asarray(
        np.stack([np.full(16, 0.5, dtype=np.float32),
                  np.full(16, -0.5, dtype=np.float32)])),)
    _measure_once(env, order, samples)
    _measure_once(env, order, samples)
    recs = env_mod.consume_plan_records()["records"]
    assert len(recs) == 2, recs
    a, b = recs
    assert a.get("measured_from") is None
    assert b.get("measured_from") == 0
    # THE COUNTS THE MEASURED PLAN RAN UNDER, on the measured record only.
    assert int(a["measure_inner"]) >= 1
    assert int(a["measure_windows"]) >= 1
    assert float(a["measure_secs"]) > 0.0
    assert int(a["ref_measure_windows"]) >= 1
    # NO TIMING FIELDS OF ITS OWN on the duplicate: nothing was timed for it.
    for k in ("measure_inner", "measure_windows", "measure_secs",
              "ref_measure_inner", "ref_measure_windows", "ref_measure_secs",
              "ref_latency_ns", "candidate_latency_ns"):
        assert b.get(k) is None, (k, b.get(k))
    # The rewards are the first plan's, which is the point of the cache.
    assert a["rewards"] == b["rewards"]


def test_the_dedupe_is_off_without_eval_samples(_paired_log_cpu,
                                                _fresh_dedupe, monkeypatch):
    """A cache that cannot be bounded to an episode is not kept.

    Every probe and every direct caller of `_callback` measures on the fixed
    `args`, with no samples and therefore no episode key. Those callers
    re-measure, which is what they exist to do.
    """
    n_windows = [0]
    real = env_mod._time_one_rep

    def _counting(ex, eval_args, devices, inner):
        n_windows[0] += 1
        return real(ex, eval_args, devices, inner)

    monkeypatch.setattr(env_mod, "_time_one_rep", _counting)
    env = _one_instrument_toy_env(num_data_points=1, reps_per_point=1)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _plan_arrays(len(order))
    for _ in range(2):
        env_mod._callback(env.config, env.args, env.consts,
                          jnp.asarray(order), specs, faces, skips, len(order))
    assert n_windows[0] >= 2, n_windows
