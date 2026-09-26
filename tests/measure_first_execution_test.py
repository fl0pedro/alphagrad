# The measurement order of the decision of 2026-09-26 (dsnn-19wc; CONTEXT.md "Paired ratio"): the
# quality execution goes first, on every plan and on the reference. It is timed and recorded as the
# cold reading, and it never sets the counts or enters the latency sample. The next execution is
# warm and sets the counts: past the 1 s budget it is the whole sample, otherwise the window rule
# gives inner and windows and it stays in the sample. A plan without a quality execution takes its
# cold reading on the timed inputs. The reference is timed once per process under the same rule.
from __future__ import annotations

import os
import time
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx.common import compile_cache as _cc        # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    MEASURE_INNER_MIN, REWARD_INDEX, resolve_measure_inner,
    resolve_measure_windows)

# Looked up, not imported, so the end-to-end tests below run on a tree without
# the rule and fail on their assertions rather than at collection.
warm_execution_counts = getattr(env_mod, "warm_execution_counts", None)

WINDOW, BUDGET, HI, CAP = 0.05, 1.0, 50, 20
POINTS, REPS = 5, 4
REF_POINTS, REF_REPS = 2, 3
LIMIT = 16_000_000_000
SWELL = 20_000_000_000
LAT = int(REWARD_INDEX["latency_ns"])
QUALITY = int(REWARD_INDEX["quality"])
PROBE = np.full(16, 0.375, dtype=np.float32)


def test_the_warm_execution_sets_the_counts_or_is_the_whole_sample():
    assert warm_execution_counts is not None, "env has no warm-execution rule"
    # Past the budget: one timed run.
    assert warm_execution_counts(1.5, WINDOW, BUDGET, HI, CAP) == (1, 1, True)
    assert warm_execution_counts(BUDGET + 1e-6, WINDOW, BUDGET, HI, CAP) == (
        1, 1, True)
    # Between 0.2 s and 1 s: the floor of 5, one window.
    for t in (0.2, 0.5, 1.0):
        assert warm_execution_counts(t, WINDOW, BUDGET, HI, CAP) == (
            MEASURE_INNER_MIN, 1, False)
    # Under 0.2 s: the window rule of 2026-09-14.
    t = 0.002
    inner = resolve_measure_inner(t, WINDOW, HI)
    assert warm_execution_counts(t, WINDOW, BUDGET, HI, CAP) == (
        inner, resolve_measure_windows(t, inner, BUDGET, CAP), False)
    assert warm_execution_counts(1e-6, WINDOW, BUDGET, HI, CAP) == (
        HI, CAP, False)
    # A broken reading is not a slow program: the ceiling and one window.
    assert warm_execution_counts(float("nan"), WINDOW, BUDGET, HI, CAP) == (
        HI, 1, False)


class _Clock:
    def __init__(self):
        self.now = 0.0


class _Analysis:
    def __init__(self, inner, extra):
        self._inner = inner
        self.temp_size_in_bytes = int(inner.temp_size_in_bytes) + int(extra)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Timed:
    # An executable whose executions advance the fake clock by `secs`, the
    # first one `cold` times as much. It keeps the first argument of its
    # first two calls.
    def __init__(self, exe, clock, secs, cold=1.0, swell=0):
        self._exe, self._clock = exe, clock
        self._secs, self._cold, self._swell = secs, cold, swell
        self.calls = 0
        self.args: list = []

    def __call__(self, *a, **k):
        self.calls += 1
        if len(self.args) < 2:
            self.args.append(np.asarray(a[0]))
        self._clock.now += self._secs * (self._cold if self.calls == 1 else 1.0)
        return self._exe(*a, **k)

    def memory_analysis(self):
        a = self._exe.memory_analysis()
        return _Analysis(a, self._swell) if self._swell else a

    def __getattr__(self, name):
        return getattr(self._exe, name)


class _Hook:
    def __init__(self, monkeypatch, clock, cand, ref):
        self.cand: list = []
        self.ref: list = []
        real = _cc.cached_compile

        def compile_(key, fn, *a, **kw):
            out = real(key, fn, *a, **kw)
            head = bytes(key).split(b":", 1)[0]
            if head == b"approx":
                out = _Timed(out, clock, *cand)
                self.cand.append(out)
            elif head == b"paired-ref":
                out = _Timed(out, clock, *ref)
                self.ref.append(out)
            return out

        monkeypatch.setattr(_cc, "cached_compile", compile_)

    def ref_calls(self) -> int:
        return sum(w.calls for w in self.ref)


def _clear_quality_caches():
    env_mod._COSINE_REF.clear()
    env_mod._PROBE_BATCH.clear()
    env_mod._PROBE_META.clear()


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    fake = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time)
                                    if not k.startswith("_")})
    fake.perf_counter = lambda: c.now
    monkeypatch.setattr(env_mod, "time", fake)
    return c


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "watermark")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setenv("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "1")
    for name in ("ALPHAGRAD_GRAD_COSINE_K", "ALPHAGRAD_REV_EXACT_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(120.0)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()
    _clear_quality_caches()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()
    _clear_quality_caches()


def _probe_gen(keys):
    return (jnp.asarray(PROBE),)


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    return VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, data_gen=_probe_gen, scalar_target=True,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=HI, num_data_points=POINTS, reps_per_point=REPS,
        ref_num_data_points=REF_POINTS, ref_reps_per_point=REF_REPS,
        measure_budget_secs=BUDGET, measure_window_secs=WINDOW)


def _orders(env):
    vs = sorted(int(v) for v in np.asarray(env.valid_vertices))
    return vs, vs[::-1]


def _samples(n):
    rows = [np.linspace(-1.0, 1.0, 16, dtype=np.float32) * (0.5 + 0.25 * k)
            for k in range(n)]
    return (jnp.asarray(np.stack(rows)),)


def _measure(env, order, samples):
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    n = len(order)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order),
        jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips), n,
        *samples)
    return env_mod.consume_plan_records()["records"][-1]


def _cold_then_warm(timed, samples):
    # The first call ran on the probe batch, the second on the timed inputs.
    assert len(timed.args) == 2, timed.args
    assert np.array_equal(timed.args[0], PROBE), "the first run was not the quality run"
    assert np.array_equal(timed.args[1], np.asarray(samples[0][0]))


def test_the_quality_execution_is_the_cold_reading_and_the_warm_execution_sets_the_counts(
        paired, clock, monkeypatch):
    # Both halves read 10x cold on their first execution: 20 ms and 5 ms.
    hook = _Hook(monkeypatch, clock, cand=(0.002, 10.0), ref=(0.0005, 10.0))
    env = _toy_env()
    fwd, _rev = _orders(env)
    samples = _samples(POINTS)
    rec = _measure(env, fwd, samples)
    # The candidate: counts from the warm 2 ms, not from the cold 20 ms.
    inner = resolve_measure_inner(0.002, WINDOW, HI)
    windows = resolve_measure_windows(0.002, inner, BUDGET, CAP)
    assert (inner, windows) == (25, 20)
    assert (rec["measure_inner"], rec["measure_windows"]) == (inner, windows)
    assert rec.get("measure_first_s") == pytest.approx(0.02)
    _cold_then_warm(hook.cand[-1], samples)
    assert hook.cand[-1].calls == 1 + 1 + inner * windows
    assert rec["ratio_log"]["latency"]["n"] == 1 + windows
    assert rec["candidate_latency_ns"] == pytest.approx(2e6)
    assert rec["measure_secs"] == pytest.approx(0.002 + inner * windows * 0.002)
    # The reference: its quality run is its cold reading, the warm run its inner.
    ref_inner = resolve_measure_inner(0.0005, WINDOW, HI)
    assert ref_inner == HI
    assert rec["ref_measure_inner"] == ref_inner
    assert rec["ref_measure_windows"] == REF_POINTS * REF_REPS
    assert rec.get("ref_measure_first_s") == pytest.approx(0.005)
    _cold_then_warm(hook.ref[-1], samples)
    assert hook.ref_calls() == 1 + 1 + ref_inner * REF_POINTS * REF_REPS
    assert rec["ref_latency_ns"] == pytest.approx(5e5)
    assert rec["ref_measure_secs"] == pytest.approx(
        0.0005 + ref_inner * REF_POINTS * REF_REPS * 0.0005)
    assert rec["ref_timing"] == "timed"
    assert rec["rewards"][LAT] == pytest.approx(-np.log(2e6 / 5e5))
    assert rec["rewards"][QUALITY] == pytest.approx(1.0, abs=1e-5)


def test_a_slow_plan_is_two_executions_and_its_sample_is_the_warm_one(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(1.2, 2.0), ref=(0.0005,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    samples = _samples(POINTS)
    rec = _measure(env, fwd, samples)
    _cold_then_warm(hook.cand[-1], samples)
    assert hook.cand[-1].calls == 2
    assert (rec["measure_inner"], rec["measure_windows"]) == (1, 1)
    assert rec.get("measure_first_s") == pytest.approx(2.4)
    assert rec["candidate_latency_ns"] == pytest.approx(1.2e9)
    assert rec["measure_secs"] == pytest.approx(1.2)
    assert rec["ratio_log"]["latency"]["n"] == 1


def test_a_plan_slow_cold_but_fast_warm_takes_the_window_rule_not_one_run(
        paired, clock, monkeypatch):
    # The cold reading is 1.2 s, past the budget; the warm one is 2 ms.
    hook = _Hook(monkeypatch, clock, cand=(0.002, 600.0), ref=(0.0005,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    samples = _samples(POINTS)
    rec = _measure(env, fwd, samples)
    inner = resolve_measure_inner(0.002, WINDOW, HI)
    windows = resolve_measure_windows(0.002, inner, BUDGET, CAP)
    assert (rec["measure_inner"], rec["measure_windows"]) == (inner, windows)
    assert (inner, windows) != (1, 1)
    assert rec.get("measure_first_s") == pytest.approx(1.2)
    assert hook.cand[-1].calls == 1 + 1 + inner * windows
    assert rec["ratio_log"]["latency"]["n"] == 1 + windows
    assert rec["candidate_latency_ns"] == pytest.approx(2e6)


def test_a_plan_between_keeps_the_floor_of_five_and_its_warm_reading(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.3,), ref=(0.0005,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    rec = _measure(env, fwd, _samples(POINTS))
    assert hook.cand[-1].calls == 1 + 1 + MEASURE_INNER_MIN
    assert (rec["measure_inner"], rec["measure_windows"]) == (
        MEASURE_INNER_MIN, 1)
    assert rec.get("measure_first_s") == pytest.approx(0.3)
    assert rec["ratio_log"]["latency"]["n"] == 2
    assert rec["candidate_latency_ns"] == pytest.approx(3e8)
    assert rec["measure_secs"] == pytest.approx(0.3 + 5 * 0.3)


def test_a_reference_slow_cold_but_fast_warm_takes_its_windows_once_per_process(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.002,), ref=(0.0005, 2400.0))
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(POINTS)
    rec_a = _measure(env, fwd, samples)
    _cold_then_warm(hook.ref[-1], samples)
    assert rec_a.get("ref_measure_first_s") == pytest.approx(1.2)
    assert (rec_a["ref_measure_inner"], rec_a["ref_measure_windows"]) == (
        HI, REF_POINTS * REF_REPS)
    assert hook.ref_calls() == 1 + 1 + HI * REF_POINTS * REF_REPS
    assert rec_a["ref_latency_ns"] == pytest.approx(5e5)
    assert rec_a["ref_timing"] == "timed"
    calls = hook.ref_calls()
    rec_b = _measure(env, rev, samples)
    assert hook.ref_calls() == calls, "the second plan timed the reference again"
    assert rec_b["ref_timing"] == "reused"
    for k in ("ref_measure_inner", "ref_measure_windows", "ref_measure_secs",
              "ref_measure_first_s", "ref_latency_ns"):
        assert rec_b[k] == rec_a[k], k


def test_a_slow_reference_is_two_executions_and_its_sample_is_the_warm_one(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.002,), ref=(1.2, 1.5))
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(POINTS)
    rec_a = _measure(env, fwd, samples)
    _cold_then_warm(hook.ref[-1], samples)
    assert hook.ref_calls() == 2
    assert (rec_a["ref_measure_inner"], rec_a["ref_measure_windows"]) == (1, 1)
    assert rec_a.get("ref_measure_first_s") == pytest.approx(1.8)
    assert rec_a["ref_latency_ns"] == pytest.approx(1.2e9)
    assert rec_a["ref_measure_secs"] == pytest.approx(1.2)
    rec_b = _measure(env, rev, samples)
    assert hook.ref_calls() == 2, "the second plan timed the reference again"
    assert rec_b["ref_timing"] == "reused"
    assert rec_b["ref_latency_ns"] == rec_a["ref_latency_ns"]


def test_a_refused_plan_times_the_reference_under_the_same_rule(
        paired, clock, monkeypatch):
    # No quality run: the reference's cold reading is one run on its inputs.
    hook = _Hook(monkeypatch, clock, cand=(0.002, 1.0, SWELL), ref=(1.2, 1.5))
    env = _toy_env()
    fwd, _rev = _orders(env)
    samples = _samples(POINTS)
    rec = _measure(env, fwd, samples)
    assert rec["refused"] == "gate"
    assert hook.cand[-1].calls == 0
    assert hook.ref_calls() == 2
    assert [np.array_equal(a, np.asarray(samples[0][0]))
            for a in hook.ref[-1].args] == [True, True]
    assert (rec["ref_measure_inner"], rec["ref_measure_windows"]) == (1, 1)
    assert rec.get("ref_measure_first_s") == pytest.approx(1.8)
    assert rec["ref_latency_ns"] == pytest.approx(1.2e9)
    assert rec["ref_timing"] == "timed"


def test_without_a_quality_execution_the_first_run_on_the_inputs_is_the_cold_reading(
        paired, clock, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    hook = _Hook(monkeypatch, clock, cand=(0.002, 10.0), ref=(0.0005, 10.0))
    env = _toy_env()
    fwd, _rev = _orders(env)
    samples = _samples(POINTS)
    rec = _measure(env, fwd, samples)
    inner = resolve_measure_inner(0.002, WINDOW, HI)
    windows = resolve_measure_windows(0.002, inner, BUDGET, CAP)
    assert (rec["measure_inner"], rec["measure_windows"]) == (inner, windows)
    assert rec.get("measure_first_s") == pytest.approx(0.02)
    assert hook.cand[-1].calls == 1 + 1 + inner * windows
    assert [np.array_equal(a, np.asarray(samples[0][0]))
            for a in hook.cand[-1].args] == [True, True]
    assert rec.get("ref_measure_first_s") == pytest.approx(0.005)
    assert rec["ref_measure_inner"] == HI
    assert hook.ref_calls() == 1 + 1 + HI * REF_POINTS * REF_REPS
