# The measurement protocol of the owner's rulings of 2026-09-25 (dsnn-dfw.221; CONTEXT.md
# "Paired ratio"): every execution is timed and none runs untimed first; the first execution
# sets the counts, is the whole sample past the 1 s budget and otherwise leaves the sample; the
# reference is timed once per process under the same rule; the record carries the counts.
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
first_execution_counts = getattr(env_mod, "first_execution_counts", None)

WINDOW, BUDGET, HI, CAP = 0.05, 1.0, 50, 20
POINTS, REPS = 5, 4
REF_POINTS, REF_REPS = 2, 3
LIMIT = 16_000_000_000
SWELL = 20_000_000_000
LAT = int(REWARD_INDEX["latency_ns"])


def test_the_first_execution_sets_the_counts_or_is_the_whole_sample():
    assert first_execution_counts is not None, "env has no first-execution rule"
    # Past the budget: one timed run.
    assert first_execution_counts(1.5, WINDOW, BUDGET, HI, CAP) == (1, 1, True)
    assert first_execution_counts(BUDGET + 1e-6, WINDOW, BUDGET, HI, CAP) == (
        1, 1, True)
    # Between 0.2 s and 1 s: the floor of 5, one window.
    for t in (0.2, 0.5, 1.0):
        assert first_execution_counts(t, WINDOW, BUDGET, HI, CAP) == (
            MEASURE_INNER_MIN, 1, False)
    # Under 0.2 s: the window rule of 2026-09-14.
    t = 0.002
    inner = resolve_measure_inner(t, WINDOW, HI)
    assert first_execution_counts(t, WINDOW, BUDGET, HI, CAP) == (
        inner, resolve_measure_windows(t, inner, BUDGET, CAP), False)
    assert first_execution_counts(1e-6, WINDOW, BUDGET, HI, CAP) == (
        HI, CAP, False)
    # A broken reading is not a slow program: the ceiling and one window.
    assert first_execution_counts(float("nan"), WINDOW, BUDGET, HI, CAP) == (
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
    # first one `cold` times as much.
    def __init__(self, exe, clock, secs, cold=1.0, swell=0):
        self._exe, self._clock = exe, clock
        self._secs, self._cold, self._swell = secs, cold, swell
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
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
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "watermark")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setenv("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "1")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(120.0)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    return VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
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


def _executions(rec, half=""):
    return 1 + rec[f"{half}measure_inner"] * rec[f"{half}measure_windows"]


def test_a_fast_plan_takes_the_windows_the_rule_gives_and_its_cold_first_execution_leaves_the_sample(
        paired, clock, monkeypatch):
    # The candidate's first execution reads 20 ms cold and 2 ms afterwards;
    # the reference's first 5 ms cold and 0.5 ms afterwards.
    hook = _Hook(monkeypatch, clock, cand=(0.002, 10.0), ref=(0.0005, 10.0))
    env = _toy_env()
    fwd, _rev = _orders(env)
    rec = _measure(env, fwd, _samples(POINTS))
    inner, windows = MEASURE_INNER_MIN, 10
    assert rec["measure_inner"] == inner
    assert rec["measure_windows"] == windows
    assert rec.get("measure_first_s") == pytest.approx(0.02)
    assert rec["candidate_latency_ns"] == pytest.approx(2e6)
    assert rec["measure_secs"] == pytest.approx(0.02 + inner * windows * 0.002)
    assert hook.cand[-1].calls == _executions(rec)
    ref_inner = resolve_measure_inner(0.005, WINDOW, HI)
    assert rec["ref_measure_inner"] == ref_inner == 10
    assert rec["ref_measure_windows"] == REF_POINTS * REF_REPS
    assert rec.get("ref_measure_first_s") == pytest.approx(0.005)
    assert rec["ref_latency_ns"] == pytest.approx(5e5)
    assert rec["ref_timing"] == "timed"
    assert hook.ref_calls() == _executions(rec, "ref_")
    assert rec["rewards"][LAT] == pytest.approx(-np.log(2e6 / 5e5))


def test_a_slow_plan_is_one_timed_run_even_when_that_run_is_cold(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.9, 2.0), ref=(0.0005,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    rec = _measure(env, fwd, _samples(POINTS))
    assert hook.cand[-1].calls == 1
    assert (rec["measure_inner"], rec["measure_windows"]) == (1, 1)
    assert rec.get("measure_first_s") == pytest.approx(1.8)
    assert rec["candidate_latency_ns"] == pytest.approx(1.8e9)
    assert rec["measure_secs"] == pytest.approx(1.8)
    assert rec["ratio_log"]["latency"]["n"] == 1


def test_a_plan_between_keeps_the_floor_of_five(paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.3,), ref=(0.0005,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    rec = _measure(env, fwd, _samples(POINTS))
    assert hook.cand[-1].calls == 1 + MEASURE_INNER_MIN
    assert (rec["measure_inner"], rec["measure_windows"]) == (
        MEASURE_INNER_MIN, 1)
    assert rec.get("measure_first_s") == pytest.approx(0.3)
    assert rec["candidate_latency_ns"] == pytest.approx(3e8)
    assert rec["measure_secs"] == pytest.approx(0.3 + 5 * 0.3)


def test_a_slow_reference_is_one_timed_run_reused_by_the_plans_after_it(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.002,), ref=(1.2,))
    env = _toy_env()
    fwd, rev = _orders(env)
    rec_a = _measure(env, fwd, _samples(POINTS))
    assert hook.ref_calls() == 1
    assert (rec_a["ref_measure_inner"], rec_a["ref_measure_windows"]) == (1, 1)
    assert rec_a.get("ref_measure_first_s") == pytest.approx(1.2)
    assert rec_a["ref_latency_ns"] == pytest.approx(1.2e9)
    assert rec_a["ref_measure_secs"] == pytest.approx(1.2)
    assert rec_a["ref_timing"] == "timed"
    rec_b = _measure(env, rev, _samples(POINTS))
    assert hook.ref_calls() == 1, "the second plan timed the reference again"
    assert rec_b["ref_timing"] == "reused"
    for k in ("ref_measure_inner", "ref_measure_windows", "ref_measure_secs",
              "ref_measure_first_s", "ref_latency_ns"):
        assert rec_b[k] == rec_a[k], k


def test_a_refused_plan_times_the_reference_under_the_same_rule(
        paired, clock, monkeypatch):
    hook = _Hook(monkeypatch, clock, cand=(0.002, 1.0, SWELL), ref=(1.2,))
    env = _toy_env()
    fwd, _rev = _orders(env)
    rec = _measure(env, fwd, _samples(POINTS))
    assert rec["refused"] == "gate"
    assert hook.cand[-1].calls == 0
    assert hook.ref_calls() == 1
    assert (rec["ref_measure_inner"], rec["ref_measure_windows"]) == (1, 1)
    assert rec.get("ref_measure_first_s") == pytest.approx(1.2)
    assert rec["ref_latency_ns"] == pytest.approx(1.2e9)
    assert rec["ref_timing"] == "timed"
